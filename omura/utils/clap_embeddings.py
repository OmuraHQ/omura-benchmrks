"""CLAP (Contrastive Language-Audio Pretraining) embeddings for audio search.

Used for the audio modality only: ``laion/larger_clap_general`` benchmarked
86.65% on ESC-50 (the only A/V model that cleared the 85% bar), so audio search
uses CLAP rather than the main image/text embedding model.

Produces a shared 512-d joint space for audio and text:
  * ``embed_audio(bytes, ext)`` — encode an audio clip  -> [512] float32, L2-normalized
  * ``embed_text(str)``         — encode a text query    -> [512] float32, L2-normalized

The model is loaded lazily on first use, on the process's visible CUDA device
(CPU fallback). Both functions return ``None`` on failure so callers can skip
gracefully.
"""

from __future__ import annotations

import io
import os
import threading
from typing import Optional

import numpy as np

CLAP_MODEL = os.getenv("OMURA_CLAP_MODEL", "laion/larger_clap_general")
CLAP_DIM = int(os.getenv("OMURA_CLAP_DIM", "512"))
CLAP_SR = 48000  # CLAP expects 48 kHz mono audio

_model = None
_processor = None
_device = None
_load_lock = threading.Lock()
_load_failed = False


def _ensure_loaded() -> bool:
    """Lazily load the CLAP model + processor. Returns True if usable."""
    global _model, _processor, _device, _load_failed
    if _model is not None:
        return True
    if _load_failed:
        return False
    with _load_lock:
        if _model is not None:
            return True
        if _load_failed:
            return False
        try:
            import torch
            from transformers import ClapModel, ClapProcessor

            _device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[CLAP] Loading {CLAP_MODEL} on {_device}…")
            model = ClapModel.from_pretrained(CLAP_MODEL)
            model.eval().to(_device)
            _processor = ClapProcessor.from_pretrained(CLAP_MODEL)
            _model = model
            print("[CLAP] Loaded.")
            return True
        except Exception as e:  # pragma: no cover - environment dependent
            print(f"[CLAP] Load failed: {e}")
            _load_failed = True
            return False


def _normalize(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32).flatten()
    n = np.linalg.norm(arr)
    if n > 0:
        arr = arr / n
    return arr.astype(np.float32, copy=False)


def _decode_ffmpeg(audio_data: bytes) -> Optional[np.ndarray]:
    """Decode any audio (m4a/aac/odd mp3/etc) via ffmpeg -> mono f32 48 kHz."""
    try:
        import subprocess
        import imageio_ffmpeg

        exe = imageio_ffmpeg.get_ffmpeg_exe()
        p = subprocess.run(
            [exe, "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
             "-f", "f32le", "-ac", "1", "-ar", str(CLAP_SR), "pipe:1"],
            input=audio_data, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120,
        )
        if p.returncode != 0 or not p.stdout:
            return None
        return np.frombuffer(p.stdout, dtype=np.float32).copy()
    except Exception as e:
        print(f"[CLAP] ffmpeg decode failed: {e}")
        return None


def _decode_audio(audio_data: bytes) -> Optional[np.ndarray]:
    """Decode arbitrary audio bytes to a mono float32 waveform at 48 kHz."""
    try:
        import soundfile as sf

        wav, sr = sf.read(io.BytesIO(audio_data), dtype="float32", always_2d=False)
    except Exception:
        # Fallback 1: librosa (handles some mp3 via audioread)
        try:
            import librosa

            wav, sr = librosa.load(io.BytesIO(audio_data), sr=CLAP_SR, mono=True)
            return np.asarray(wav, dtype=np.float32)
        except Exception:
            # Fallback 2: ffmpeg (m4a/aac and tolerant of odd encodings)
            return _decode_ffmpeg(audio_data)
    # Downmix to mono
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    # Resample to 48 kHz if needed
    if sr != CLAP_SR:
        try:
            import librosa

            wav = librosa.resample(np.asarray(wav, dtype=np.float32), orig_sr=sr, target_sr=CLAP_SR)
        except Exception as e:
            print(f"[CLAP] resample failed: {e}")
            return None
    return np.asarray(wav, dtype=np.float32)


def embed_audio(audio_data: bytes, ext: str = "", blob_id: str = "") -> Optional[np.ndarray]:
    """Encode an audio clip into the CLAP joint space. Returns [512] or None."""
    if not _ensure_loaded():
        return None
    wav = _decode_audio(audio_data)
    if wav is None or wav.size == 0:
        return None
    try:
        import torch

        inputs = _processor(audio=[wav], sampling_rate=CLAP_SR, return_tensors="pt")
        inputs = {k: v.to(_device) for k, v in inputs.items()}
        with torch.no_grad():
            out = _model.get_audio_features(**inputs)
        # transformers>=5 returns an output object whose pooler_output is the
        # projected 512-d joint embedding (== ClapOutput.audio_embeds).
        feats = out.pooler_output if hasattr(out, "pooler_output") else out
        return _normalize(feats[0].float().cpu().numpy())
    except Exception as e:
        print(f"[CLAP] embed_audio failed for {blob_id}: {e}")
        return None


def embed_text(text: str) -> Optional[np.ndarray]:
    """Encode a text query into the CLAP joint space. Returns [512] or None."""
    if not _ensure_loaded():
        return None
    text = (text or "").strip()
    if not text:
        return None
    try:
        import torch

        inputs = _processor(text=[text], return_tensors="pt", padding=True)
        inputs = {k: v.to(_device) for k, v in inputs.items()}
        with torch.no_grad():
            out = _model.get_text_features(**inputs)
        feats = out.pooler_output if hasattr(out, "pooler_output") else out
        return _normalize(feats[0].float().cpu().numpy())
    except Exception as e:
        print(f"[CLAP] embed_text failed: {e}")
        return None


__all__ = ["embed_audio", "embed_text", "CLAP_MODEL", "CLAP_DIM", "CLAP_SR"]
