"""Benchmark embedding backend with pluggable model dispatch.

Default (``OMURA_EMBEDDING_BACKEND`` unset or any value other than
``omuragemma``): the SigLIP / ``omura_embed`` checkpoint loaded through
Transformers.

``OMURA_EMBEDDING_BACKEND=omuragemma``: dispatch to the repo-local
:mod:`omura.utils.omuragemma_embeddings` backend (EmbeddingGemma 2, 768-d,
text/image/audio/video). Batch helpers are exposed where the backend supports
them so benchmarks can embed a corpus in one model call instead of per item.
"""

from __future__ import annotations

import os
import sys
from io import BytesIO
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BACKEND = os.getenv("OMURA_EMBEDDING_BACKEND", "").strip().lower()
IS_OMURAGEMMA = BACKEND == "omuragemma"

_gemma = None


def _gemma_backend():
    global _gemma
    if _gemma is None:
        from omura.utils import omuragemma_embeddings as backend

        _gemma = backend
    return _gemma


if IS_OMURAGEMMA:
    MODEL_NAME = _gemma_backend().MODEL_NAME
    EMBEDDING_DIM = int(_gemma_backend().EMBEDDING_DIM)
else:
    # SigLIP/omura_embed path (import kept lazy for the gemma backend so a
    # benchmark run does not pull transformers just to dispatch).
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from transformers import AutoModel, AutoProcessor

    MODEL_NAME = os.getenv(
        "OMURA_EMBEDDING_MODEL", "immortaltatsu/omura_emebd"
    )
    EMBEDDING_DIM = int(os.getenv("OMURA_EMBEDDING_DIM", "0")) or None

    _MODEL = None
    _PROCESSOR = None
    _DEVICE = None


def _extract_embedding_tensor(out_or_feats, kind: str):
    if isinstance(out_or_feats, torch.Tensor):
        return out_or_feats
    attr_name = "text_embeds" if kind == "text" else "image_embeds"
    feats = getattr(out_or_feats, attr_name, None)
    if feats is None:
        feats = getattr(out_or_feats, "pooler_output", None)
    if feats is None and isinstance(out_or_feats, tuple) and len(out_or_feats) > 0:
        feats = out_or_feats[0]
    return feats


def _norm(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n > 0:
        return (v / n).astype(np.float32, copy=False)
    return v.astype(np.float32, copy=False)


# ── lifecycle ────────────────────────────────────────────────────────────────


def initialize_embedding_model() -> None:
    if IS_OMURAGEMMA:
        _gemma_backend().ensure_model_loaded()
        return
    global _MODEL, _PROCESSOR, _DEVICE
    if _MODEL is not None:
        return
    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if _DEVICE == "cuda" else torch.float32
    _MODEL = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True, dtype=dtype)
    _MODEL = _MODEL.to(_DEVICE)
    _MODEL.eval()
    _PROCESSOR = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)


# ── text ─────────────────────────────────────────────────────────────────────


def generate_text_embedding(text: str, is_document: bool = False) -> Optional[np.ndarray]:
    if IS_OMURAGEMMA:
        return _gemma_backend().generate_text_embedding(text, is_document=is_document)

    initialize_embedding_model()
    if not text or not text.strip():
        return None
    payload = text.strip()[:8000 if is_document else 4000]
    # SigLIP/SigLIP2 are trained with fixed-length text padding; padding=True breaks retrieval quality.
    inputs = _PROCESSOR(text=[payload], return_tensors="pt", padding="max_length")
    inputs = {k: v.to(_DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        out_or_feats = (
            _MODEL.get_text_features(**inputs)
            if hasattr(_MODEL, "get_text_features")
            else _MODEL(**inputs)
        )
        feats = _extract_embedding_tensor(out_or_feats, "text")
        if feats is None:
            return None
    feats = F.normalize(feats.float(), dim=-1).cpu().numpy().reshape(-1)
    return _norm(feats)


def generate_text_embeddings(
    texts: Sequence[str], is_document: bool = False
) -> list[Optional[np.ndarray]]:
    if IS_OMURAGEMMA:
        return _gemma_backend().generate_text_embeddings(
            list(texts), is_document=is_document
        )
    return [generate_text_embedding(t, is_document=is_document) for t in texts]


# ── images ───────────────────────────────────────────────────────────────────


def generate_image_embedding(image_data: bytes, blob_id: str | None = None) -> Optional[np.ndarray]:
    if IS_OMURAGEMMA:
        return _gemma_backend().generate_image_embedding(image_data, blob_id=blob_id)

    initialize_embedding_model()
    try:
        img = Image.open(BytesIO(image_data)).convert("RGB")
    except Exception:
        return None
    inputs = _PROCESSOR(images=[img], return_tensors="pt")
    inputs = {k: v.to(_DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        out_or_feats = (
            _MODEL.get_image_features(**inputs)
            if hasattr(_MODEL, "get_image_features")
            else _MODEL(**inputs)
        )
        feats = _extract_embedding_tensor(out_or_feats, "image")
        if feats is None:
            return None
    feats = F.normalize(feats.float(), dim=-1).cpu().numpy().reshape(-1)
    return _norm(feats)


def generate_image_embeddings(
    image_datas: Sequence[bytes], blob_ids: Sequence[Optional[str]] | None = None
) -> list[Optional[np.ndarray]]:
    if IS_OMURAGEMMA:
        return _gemma_backend().generate_image_embeddings(
            list(image_datas), blob_ids=list(blob_ids) if blob_ids else None
        )
    return [
        generate_image_embedding(data, blob_id=blob_ids[i] if blob_ids else None)
        for i, data in enumerate(image_datas)
    ]


# ── audio / video (omuragemma only) ──────────────────────────────────────────


def generate_audio_embedding(
    audio_data: bytes, ext: str = "", blob_id: str = ""
) -> Optional[np.ndarray]:
    if not IS_OMURAGEMMA:
        raise RuntimeError("audio embedding requires OMURA_EMBEDDING_BACKEND=omuragemma")
    return _gemma_backend().generate_audio_embedding(audio_data, ext=ext, blob_id=blob_id)


def generate_audio_embeddings(
    audio_datas: Sequence[bytes],
    exts: Sequence[str] | None = None,
    blob_ids: Sequence[str] | None = None,
) -> list[Optional[np.ndarray]]:
    if not IS_OMURAGEMMA:
        raise RuntimeError("audio embedding requires OMURA_EMBEDDING_BACKEND=omuragemma")
    return _gemma_backend().generate_audio_embeddings(
        list(audio_datas),
        exts=list(exts) if exts else None,
        blob_ids=list(blob_ids) if blob_ids else None,
    )


def generate_video_embedding(
    video_data: bytes, ext: str = "", blob_id: str = ""
) -> Optional[np.ndarray]:
    if not IS_OMURAGEMMA:
        raise RuntimeError("video embedding requires OMURA_EMBEDDING_BACKEND=omuragemma")
    return _gemma_backend().generate_video_embedding(video_data, ext=ext, blob_id=blob_id)


def generate_video_embeddings(
    video_datas: Sequence[bytes],
    exts: Sequence[str] | None = None,
    blob_ids: Sequence[str] | None = None,
) -> list[Optional[np.ndarray]]:
    if not IS_OMURAGEMMA:
        raise RuntimeError("video embedding requires OMURA_EMBEDDING_BACKEND=omuragemma")
    return _gemma_backend().generate_video_embeddings(
        list(video_datas),
        exts=list(exts) if exts else None,
        blob_ids=list(blob_ids) if blob_ids else None,
    )
