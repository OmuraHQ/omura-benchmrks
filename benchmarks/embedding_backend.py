from __future__ import annotations

import os
from io import BytesIO
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel, AutoProcessor

MODEL_NAME = os.getenv(
    "OMURA_EMBEDDING_MODEL", "immortaltatsu/omura_emebd"
)

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


def initialize_embedding_model() -> None:
    global _MODEL, _PROCESSOR, _DEVICE
    if _MODEL is not None:
        return
    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if _DEVICE == "cuda" else torch.float32
    _MODEL = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True, dtype=dtype)
    _MODEL = _MODEL.to(_DEVICE)
    _MODEL.eval()
    _PROCESSOR = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)


def generate_text_embedding(text: str, is_document: bool = False) -> Optional[np.ndarray]:
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


def generate_image_embedding(image_data: bytes, blob_id: str | None = None) -> Optional[np.ndarray]:
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
