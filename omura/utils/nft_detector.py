"""Zero-shot NFT/avatar detector via CLIP-style text-image alignment.

Reuses the existing Omura-Embed (CLIP-class) model that already produces our image
embeddings. We build two **concept anchor** vectors — averaged text embeddings of
prompts describing the NFT/profile-pic cluster vs. normal/photographic content —
and score each result image with a softmax over their dot products.

The result, ``nft_probability(image_emb) ∈ [0, 1]``, is used by the soft reranker
as an additive penalty signal. No second model, no per-image inference — every
score is a couple of dot products against the stored embedding the search
pipeline already loaded.

Tunables (env):
  OMURA_NFT_PROMPTS_NFT     | semicolon-separated NFT-cluster prompts
  OMURA_NFT_PROMPTS_NORMAL  | semicolon-separated normal-content prompts
  OMURA_NFT_TEMPERATURE     | softmax temperature (default 100.0 — matches CLIP)
"""

from __future__ import annotations

import os
import threading
from typing import List, Optional, Tuple

import numpy as np


_DEFAULT_NFT_PROMPTS = [
    "a cartoon NFT profile picture avatar",
    "a flat-color digital collectible character illustration",
    "a generative art pfp with a cat or hat or sunglasses",
    "a pixel-art NFT avatar with simple geometric shapes",
    "a 3d rendered toon character profile picture",
]
_DEFAULT_NORMAL_PROMPTS = [
    "a photograph of a natural scene",
    "a real photo of a person or animal",
    "a candid photograph with realistic lighting",
    "a landscape photograph",
    "a documentary photograph",
]


def _split_env_prompts(env_value: str) -> Optional[List[str]]:
    raw = (env_value or "").strip()
    if not raw:
        return None
    return [p.strip() for p in raw.split(";") if p.strip()]


NFT_PROMPTS = _split_env_prompts(os.getenv("OMURA_NFT_PROMPTS_NFT", "")) or _DEFAULT_NFT_PROMPTS
NORMAL_PROMPTS = _split_env_prompts(os.getenv("OMURA_NFT_PROMPTS_NORMAL", "")) or _DEFAULT_NORMAL_PROMPTS
TEMPERATURE = float(os.getenv("OMURA_NFT_TEMPERATURE", "100.0"))


# Lazy-built anchor vectors (one-time cost on first call)
_anchors_lock = threading.Lock()
_nft_anchor: Optional[np.ndarray] = None
_normal_anchor: Optional[np.ndarray] = None
_anchors_built: bool = False
_build_failed: bool = False


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def _build_anchors() -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Lazy-load anchors by averaging text embeddings of each prompt list.

    Uses ``omura.utils.imagebind_embeddings.generate_text_embedding`` which calls
    the active CLIP-class model. Returns (None, None) if the model isn't ready —
    callers should treat that as "skip NFT signal for this query".
    """
    global _nft_anchor, _normal_anchor, _anchors_built, _build_failed
    with _anchors_lock:
        if _anchors_built:
            return _nft_anchor, _normal_anchor
        if _build_failed:
            return None, None

        try:
            from omura.utils.imagebind_embeddings import (
                generate_text_embedding, is_model_ready
            )
            if not is_model_ready():
                return None, None

            nft_embs = []
            for p in NFT_PROMPTS:
                e = generate_text_embedding(p)
                if e is not None:
                    nft_embs.append(_unit(np.asarray(e, dtype=np.float32).flatten()))
            norm_embs = []
            for p in NORMAL_PROMPTS:
                e = generate_text_embedding(p)
                if e is not None:
                    norm_embs.append(_unit(np.asarray(e, dtype=np.float32).flatten()))
            if not nft_embs or not norm_embs:
                _build_failed = True
                return None, None

            _nft_anchor = _unit(np.mean(np.stack(nft_embs), axis=0))
            _normal_anchor = _unit(np.mean(np.stack(norm_embs), axis=0))
            _anchors_built = True
            print(
                f"[NFTDetector] anchors built — {len(nft_embs)} nft + {len(norm_embs)} normal prompts, "
                f"dim={_nft_anchor.shape[0]}",
                flush=True,
            )
            return _nft_anchor, _normal_anchor
        except Exception as exc:
            print(f"[NFTDetector] anchor build failed: {exc}", flush=True)
            _build_failed = True
            return None, None


def nft_probability(image_emb: np.ndarray) -> Optional[float]:
    """Return P(NFT | image) ∈ [0, 1], or None if anchors aren't ready.

    Softmax over (sim_to_nft_anchor, sim_to_normal_anchor) with temperature.
    Higher temp = sharper separation; CLIP's native logit scale is ~100.
    """
    nft, norm = _build_anchors()
    if nft is None or norm is None:
        return None
    img = np.asarray(image_emb, dtype=np.float32).flatten()
    n = float(np.linalg.norm(img))
    if n <= 0:
        return None
    img = img / n
    s_nft = float(img @ nft)
    s_norm = float(img @ norm)
    a, b = s_nft * TEMPERATURE, s_norm * TEMPERATURE
    m = max(a, b)
    e_nft = np.exp(a - m)
    e_norm = np.exp(b - m)
    return float(e_nft / (e_nft + e_norm))


def warm_up() -> bool:
    """Eagerly build anchors. Returns True if successful. Safe to call repeatedly."""
    nft, _ = _build_anchors()
    return nft is not None


def get_anchors() -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Expose anchors for callers that want to score a batch of embeddings directly."""
    return _build_anchors()


def reset_anchors() -> None:
    """Force the next call to rebuild — useful after model swap or prompt change."""
    global _nft_anchor, _normal_anchor, _anchors_built, _build_failed
    with _anchors_lock:
        _nft_anchor = None
        _normal_anchor = None
        _anchors_built = False
        _build_failed = False


__all__ = [
    "NFT_PROMPTS",
    "NORMAL_PROMPTS",
    "TEMPERATURE",
    "nft_probability",
    "warm_up",
    "get_anchors",
    "reset_anchors",
]
