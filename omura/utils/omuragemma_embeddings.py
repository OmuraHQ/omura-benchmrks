"""Local EmbeddingGemma 2 backend (`omuragemma`).

`google/embeddinggemma-2` is a 740M-parameter Apache-2.0 model that maps text,
images, audio and video into one 768-d space. It runs entirely on local GPUs,
so this backend replaces the separate CLAP / InternVideo2 encoders and keeps
all embeddings on-device.

Enable with::

    OMURA_EMBEDDING_BACKEND=omuragemma
    OMURA_EMBEDDING_DIM=768          # native; MRL also supports 512/256/128

Deployment needs ``sentence-transformers>=6.1`` and a recent Transformers build
(``transformers>=5.18``), because EmbeddingGemma 2 was added with the Gemma 4
architecture. Video is decoded locally with decord (PyAV fallback), so
``torchcodec`` is not required.
"""

from __future__ import annotations

import io
import os
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any, Optional

import numpy as np

from omura.utils.gpu_offload import IdleOffloader, guard

MODEL_NAME = os.getenv("OMURA_OMURAGEMMA_MODEL", "google/embeddinggemma-2")
EMBEDDING_DIM = int(os.getenv("OMURA_EMBEDDING_DIM", "768"))
DEVICE = os.getenv("OMURA_OMURAGEMMA_DEVICE", "").strip() or (
    "cuda" if __import__("torch").cuda.is_available() else "cpu"
)
BATCH_SIZE = max(1, int(os.getenv("OMURA_OMURAGEMMA_BATCH_SIZE", "8")))
MAX_FRAMES = max(1, int(os.getenv("OMURA_OMURAGEMMA_MAX_FRAMES", "32")))
MAX_AUDIO_SECONDS = float(os.getenv("OMURA_OMURAGEMMA_AUDIO_SECONDS", "300"))
AUDIO_SR = 16000
JPEG_QUALITY = int(os.getenv("OMURA_OMURAGEMMA_JPEG_QUALITY", "92"))

QUERY_PREFIX = os.getenv("OMURA_OMURAGEMMA_QUERY_PREFIX", "task: search result | query: ")
DOCUMENT_PREFIX = os.getenv("OMURA_OMURAGEMMA_DOCUMENT_PREFIX", "title: none | text: ")
MAX_TEXT_CHARS = int(os.getenv("OMURA_OMURAGEMMA_TEXT_MAX_CHARS", "24000"))

_DEBUG = os.getenv("OMURA_OMURAGEMMA_DEBUG", "false").strip().lower() == "true"

_model = None
_ready = False
_init_error: Optional[str] = None
_load_lock = threading.Lock()
_infer_lock = threading.Lock()


def _unload_model() -> None:
    """Drop the local model so the idle reaper can reclaim GPU memory."""
    global _model, _ready, _init_error

    with _load_lock:
        _model = None
        _ready = False
        _init_error = None
    print("[Omuragemma] released model on idle offload", flush=True)


# 5 minutes without inference -> free GPU 7; the next query reloads lazily.
_offload = IdleOffloader("omuragemma", _unload_model)


# ── lifecycle ─────────────────────────────────────────────────────────────────


def _load_model():
    global _model
    if _model is not None:
        return _model
    with _load_lock:
        if _model is not None:
            return _model
        import torch
        from sentence_transformers import SentenceTransformer

        model_kwargs = {}
        if str(DEVICE).startswith("cuda"):
            model_kwargs["torch_dtype"] = torch.bfloat16
        if _DEBUG:
            print(f"[Omuragemma] loading {MODEL_NAME} on {DEVICE}")
        _model = SentenceTransformer(
            MODEL_NAME,
            device=DEVICE,
            model_kwargs=model_kwargs,
        )
        _model.eval()
        return _model


def initialize_embedding_model(
    store_dir: Optional[str | Path] = None, preload: bool = False
) -> None:
    global _ready, _init_error
    if _ready:
        return
    with _load_lock:
        if _ready:
            return
        try:
            verify_store_compatibility(store_dir)
            _ready = True
            _init_error = None
            print(
                f"[Omuragemma] Backend ready: {MODEL_NAME} dim={EMBEDDING_DIM} "
                f"device={DEVICE}"
            )
        except Exception as exc:  # noqa: BLE001
            _init_error = str(exc)
            _ready = False
            raise

    # Load outside the lock: _load_model() takes the same (non-reentrant) lock.
    if preload:
        with _offload.busy():
            _load_model()


def ensure_model_loaded() -> None:
    global _ready
    if not _ready:
        initialize_embedding_model()
    with _offload.busy():
        _load_model()
    _ready = True


def is_model_ready() -> bool:
    return _ready or _model is not None


def last_error() -> Optional[str]:
    return _init_error


def verify_store_compatibility(store_dir: Optional[str | Path] = None) -> Path:
    from omura.utils.store_identity import verify_store

    return verify_store(
        store_dir,
        backend="omuragemma",
        model=MODEL_NAME,
        dim=EMBEDDING_DIM,
    )


# ── encoding helpers ──────────────────────────────────────────────────────────


def _normalize(vec: Any) -> np.ndarray:
    """L2-normalize along the last axis, truncating it to ``EMBEDDING_DIM``.

    Accepts a single vector ``(D,)`` or a batch ``(N, D)``. The leading batch
    axis is preserved: flattening here would silently collapse ``encode([...])``
    from ``(N, 768)`` to ``(768,)`` and lose every row but the first.
    """
    arr = np.asarray(vec, dtype=np.float32)
    if arr.size == 0:
        raise ValueError("empty embedding")
    if arr.ndim == 0:
        raise ValueError("embedding must have at least one dimension")
    if arr.ndim == 1:
        arr = arr.reshape(-1)

    last = int(arr.shape[-1])
    if last > EMBEDDING_DIM:
        # EmbeddingGemma 2 uses MRL; prefixes are valid truncations.
        arr = arr[..., :EMBEDDING_DIM]
    elif last < EMBEDDING_DIM:
        raise ValueError(
            f"EmbeddingGemma returned {last}-d vector, expected {EMBEDDING_DIM}"
        )
    if arr.ndim == 1:
        norm = float(np.linalg.norm(arr))
        if norm > 0:
            arr = arr / norm
        return arr.astype(np.float32, copy=False)
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return (arr / norms).astype(np.float32, copy=False)


@guard(_offload)
def _encode(inputs: Any) -> np.ndarray:
    model = _load_model()
    with _infer_lock:
        out = model.encode(
            inputs,
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
    return _normalize(out)


def prepare_query(text: str) -> str:
    return f"{QUERY_PREFIX}{text.strip()[:MAX_TEXT_CHARS]}"


def prepare_document(text: str) -> str:
    return f"{DOCUMENT_PREFIX}{text.strip()[:MAX_TEXT_CHARS]}"


def _image_from_bytes(data: bytes):
    from PIL import Image

    with Image.open(io.BytesIO(data)) as src:
        src.load()
        return src.convert("RGB")


def _decode_audio(data: bytes, ext: str) -> Optional[np.ndarray]:
    """Return mono float32 16 kHz audio, truncated to the model's budget."""
    max_samples = int(MAX_AUDIO_SECONDS * AUDIO_SR)

    try:
        import soundfile as sf

        audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
        if sr != AUDIO_SR:
            import librosa

            audio = librosa.resample(audio, orig_sr=sr, target_sr=AUDIO_SR)
        return np.ascontiguousarray(audio[:max_samples], dtype=np.float32)
    except Exception as exc:  # noqa: BLE001
        if _DEBUG:
            print(f"[Omuragemma] soundfile decode failed ({exc}); trying librosa")

    try:
        import librosa

        audio, _ = librosa.load(
            io.BytesIO(data), sr=AUDIO_SR, mono=True, duration=MAX_AUDIO_SECONDS
        )
        return np.ascontiguousarray(audio[:max_samples], dtype=np.float32)
    except Exception as exc:  # noqa: BLE001
        if _DEBUG:
            print(f"[Omuragemma] librosa decode failed ({ext}): {exc}")

    # Last resort: ffmpeg probes the actual container, so mislabeled `.bin` /
    # `.mp3` blobs still decode. This is the most robust path.
    try:
        proc = subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-i", "pipe:0",
                "-t", str(MAX_AUDIO_SECONDS),
                "-vn", "-ac", "1", "-ar", str(AUDIO_SR),
                "-f", "f32le", "pipe:1",
            ],
            input=data,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=180,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout:
            audio = np.frombuffer(proc.stdout, dtype=np.float32).copy()
            if audio.size:
                return np.ascontiguousarray(audio[:max_samples], dtype=np.float32)
        if _DEBUG:
            print(
                f"[Omuragemma] ffmpeg decode failed ({ext}): "
                f"{proc.stderr.decode('utf-8', 'replace')[:200]}"
            )
    except (OSError, subprocess.TimeoutExpired) as exc:  # noqa: BLE001
        print(f"[Omuragemma] ffmpeg audio decode error ({ext}): {exc}")
    return None


def _extract_video_frames(data: bytes, ext: str) -> Optional[list]:
    """Sample up to MAX_FRAMES frames uniformly across the whole video."""
    from PIL import Image

    suffix = "." + (ext or "mp4").lower().lstrip(".")
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        path = tmp.name

    try:
        try:
            import decord

            reader = decord.VideoReader(path, ctx=decord.cpu(0))
            total = len(reader)
            if total > 0:
                count = min(MAX_FRAMES, total)
                indices = np.linspace(0, total - 1, count).astype(int)
                batch = reader.get_batch(indices).asnumpy()
                return [Image.fromarray(frame) for frame in batch]
        except Exception as exc:  # noqa: BLE001
            if _DEBUG:
                print(f"[Omuragemma] decord failed ({exc}); trying PyAV")

        import av

        container = av.open(path)
        try:
            stream = container.streams.video[0]
            if container.duration:
                duration = float(container.duration / av.time_base)
                count = min(MAX_FRAMES, max(1, int(duration)))
                targets = np.linspace(0, max(duration - 0.1, 0), count)
                frames = []
                for target in targets:
                    container.seek(
                        int(target * av.time_base), stream=stream, backward=True
                    )
                    frame = next(container.decode(stream), None)
                    if frame is not None:
                        frames.append(frame.to_image())
                if frames:
                    return frames
            # No duration: decode and keep every k-th frame.
            frames = [frame.to_image() for frame in container.decode(stream)]
            if not frames:
                return None
            count = min(MAX_FRAMES, len(frames))
            indices = np.linspace(0, len(frames) - 1, count).astype(int)
            return [frames[int(i)] for i in indices]
        finally:
            container.close()
    except Exception as exc:  # noqa: BLE001
        print(f"[Omuragemma] video decode failed ({ext}): {exc}")
        return None
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


# ── public API (mirrors imagebind_embeddings) ─────────────────────────────────


def generate_text_embedding(
    text: str,
    is_document: bool = False,
    instruction: Optional[str] = None,
) -> Optional[np.ndarray]:
    if not text or not text.strip():
        return None
    try:
        payload = prepare_document(text) if is_document else prepare_query(text)
        return _encode(payload)
    except Exception as exc:  # noqa: BLE001
        print(f"[Omuragemma] text embedding error: {exc}")
        return None


def _encode_batch(payloads: list[Any], batch_size: Optional[int] = None) -> list[np.ndarray]:
    """Encode a batch in one call, preserving row order.

    ``model.encode`` returns ``(N, D)`` for a list input and ``(D,)`` for a
    single item, so the result is normalized (batch axis preserved) here. If a
    batch fails (e.g. one corrupt input), fall back to per-item encoding so a
    single bad row does not discard the whole batch.
    """
    if not payloads:
        return []
    try:
        vecs = _encode(payloads)
        if vecs.ndim == 1:
            vecs = vecs.reshape(1, -1)
        if vecs.shape[0] == len(payloads):
            return [np.asarray(v, dtype=np.float32) for v in vecs]
    except Exception as exc:  # noqa: BLE001
        if _DEBUG:
            print(f"[Omuragemma] batch encode failed ({exc}); retrying per item")

    out: list[np.ndarray] = []
    for payload in payloads:
        try:
            v = _encode(payload)
            if v.ndim > 1:
                v = v[0]
            out.append(np.asarray(v, dtype=np.float32))
        except Exception as exc:  # noqa: BLE001
            if _DEBUG:
                print(f"[Omuragemma] per-item encode failed: {exc}")
            out.append(np.zeros(EMBEDDING_DIM, dtype=np.float32))
    return out


def generate_text_embeddings(
    texts: list[str], is_document: bool = False
) -> list[Optional[np.ndarray]]:
    """Batch text embeddings. Returns one row per input (``None`` for blanks)."""
    payloads: list[str] = []
    positions: list[int] = []
    for i, text in enumerate(texts):
        if not text or not text.strip():
            continue
        payloads.append(prepare_document(text) if is_document else prepare_query(text))
        positions.append(i)
    out: list[Optional[np.ndarray]] = [None] * len(texts)
    if not payloads:
        return out
    try:
        rows = _encode_batch(payloads)
    except Exception as exc:  # noqa: BLE001
        print(f"[Omuragemma] text batch embedding error: {exc}")
        return out
    for pos, row in zip(positions, rows):
        # Treat an all-zero row as a failure rather than a real embedding.
        if np.any(row):
            out[pos] = row
    return out


def generate_image_embeddings(
    image_datas: list[bytes], blob_ids: Optional[list[Optional[str]]] = None
) -> list[Optional[np.ndarray]]:
    """Batch image embeddings (decode per item, encode as one batch)."""
    payloads: list[Any] = []
    positions: list[int] = []
    out: list[Optional[np.ndarray]] = [None] * len(image_datas)
    for i, data in enumerate(image_datas):
        if not data:
            continue
        try:
            payloads.append({"image": _image_from_bytes(data)})
            positions.append(i)
        except Exception as exc:  # noqa: BLE001
            blob_id = blob_ids[i] if blob_ids else i
            print(f"[Omuragemma] image decode error ({blob_id}): {exc}")
    if not payloads:
        return out
    try:
        rows = _encode_batch(payloads)
    except Exception as exc:  # noqa: BLE001
        print(f"[Omuragemma] image batch embedding error: {exc}")
        return out
    for pos, row in zip(positions, rows):
        if np.any(row):
            out[pos] = row
    return out


def generate_audio_embeddings(
    audio_datas: list[bytes],
    exts: Optional[list[str]] = None,
    blob_ids: Optional[list[str]] = None,
) -> list[Optional[np.ndarray]]:
    """Batch audio embeddings (decode per item, encode as one batch)."""
    payloads: list[Any] = []
    positions: list[int] = []
    out: list[Optional[np.ndarray]] = [None] * len(audio_datas)
    for i, data in enumerate(audio_datas):
        if not data:
            continue
        ext = (exts[i] if exts and i < len(exts) else "") or ""
        try:
            audio = _decode_audio(data, ext)
        except Exception as exc:  # noqa: BLE001
            audio = None
            print(f"[Omuragemma] audio decode error: {exc}")
        if audio is None:
            continue
        payloads.append({"audio": audio})
        positions.append(i)
    if not payloads:
        return out
    rows = _encode_batch(payloads)
    for pos, row in zip(positions, rows):
        if np.any(row):
            out[pos] = row
    return out


def generate_video_embeddings(
    video_datas: list[bytes],
    exts: Optional[list[str]] = None,
    blob_ids: Optional[list[str]] = None,
) -> list[Optional[np.ndarray]]:
    """Batch video embeddings (sample frames per item, encode as one batch)."""
    payloads: list[Any] = []
    positions: list[int] = []
    out: list[Optional[np.ndarray]] = [None] * len(video_datas)
    for i, data in enumerate(video_datas):
        if not data:
            continue
        ext = (exts[i] if exts and i < len(exts) else "") or ""
        try:
            frames = _extract_video_frames(data, ext)
        except Exception as exc:  # noqa: BLE001
            frames = None
            print(f"[Omuragemma] video decode error: {exc}")
        if not frames:
            continue
        payloads.append({"video": frames})
        positions.append(i)
    if not payloads:
        return out
    rows = _encode_batch(payloads)
    for pos, row in zip(positions, rows):
        if np.any(row):
            out[pos] = row
    return out


def generate_image_embedding(
    image_data: bytes,
    blob_id: Optional[str] = None,
    instruction: Optional[str] = None,
) -> Optional[np.ndarray]:
    if not image_data:
        return None
    try:
        return _encode({"image": _image_from_bytes(image_data)})
    except Exception as exc:  # noqa: BLE001
        print(f"[Omuragemma] image embedding error ({blob_id}): {exc}")
        return None


def generate_audio_embedding(
    audio_data: bytes, ext: str = "", blob_id: str = ""
) -> Optional[np.ndarray]:
    if not audio_data:
        return None
    try:
        audio = _decode_audio(audio_data, ext)
        if audio is None:
            return None
        return _encode({"audio": audio})
    except Exception as exc:  # noqa: BLE001
        print(f"[Omuragemma] audio embedding error ({blob_id}): {exc}")
        return None


def generate_video_embedding(
    video_data: bytes, ext: str = "", blob_id: str = ""
) -> Optional[np.ndarray]:
    if not video_data:
        return None
    try:
        frames = _extract_video_frames(video_data, ext)
        if not frames:
            return None
        return _encode({"video": frames})
    except Exception as exc:  # noqa: BLE001
        print(f"[Omuragemma] video embedding error ({blob_id}): {exc}")
        return None


def generate_multimodal_embedding(
    text: str, images: Optional[list] = None
) -> Optional[np.ndarray]:
    try:
        payload: dict[str, Any] = {}
        if text and text.strip():
            payload["text"] = text.strip()[:MAX_TEXT_CHARS]
        if images:
            payload["image"] = [
                img.convert("RGB") if hasattr(img, "convert") else img for img in images
            ]
        if not payload:
            return None
        return _encode(payload)
    except Exception as exc:  # noqa: BLE001
        print(f"[Omuragemma] multimodal embedding error: {exc}")
        return None


__all__ = [
    "MODEL_NAME",
    "EMBEDDING_DIM",
    "initialize_embedding_model",
    "ensure_model_loaded",
    "is_model_ready",
    "last_error",
    "verify_store_compatibility",
    "prepare_query",
    "prepare_document",
    "generate_text_embedding",
    "generate_text_embeddings",
    "generate_image_embedding",
    "generate_image_embeddings",
    "generate_audio_embedding",
    "generate_audio_embeddings",
    "generate_video_embedding",
    "generate_video_embeddings",
    "generate_multimodal_embedding",
]
