"""Perceptual hashing for exact / near-exact duplicate detection in reverse-image search.

Embedding cosine finds *visually similar* images; a perceptual hash (dHash) confirms *the same
image* (re-encodes, crops, mild rescales) by Hamming distance on a compact 64-bit fingerprint.
Used to harden reverse search for NFT provenance / exact-duplicate verification — the embedding
retrieval narrows to candidates, the hash confirms which are true duplicates.

Pure PIL + numpy (both already dependencies); no extra packages.
"""
from __future__ import annotations

import io
from typing import Optional

import numpy as np


def _load_gray(image_bytes: bytes, size):
    """Decode → flatten any alpha over a white background → grayscale → resize.
    Compositing over white prevents transparent PNGs from collapsing to all-black
    (which would make every transparent image hash-collide)."""
    from PIL import Image
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode in ("RGBA", "LA", "PA") or "transparency" in img.info:
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img)
    return img.convert("L").resize(size, Image.Resampling.LANCZOS)

# Hamming-distance thresholds on the 64-bit hash.
EXACT_MAX = 0     # identical fingerprint = exact duplicate (re-encode/format change)
NEAR_MAX = 6      # ≤6 bits differ = near-duplicate (mild crop/scale/recompress)


def dhash(image_bytes: bytes, hash_size: int = 8) -> Optional[int]:
    """Difference hash: grayscale → resize to (hash_size+1, hash_size) → compare adjacent
    columns → (hash_size*hash_size)-bit integer. Robust to scale/compression, sensitive to
    content. Returns None if the bytes aren't a decodable image."""
    try:
        img = _load_gray(image_bytes, (hash_size + 1, hash_size))
    except Exception:
        return None
    px = np.asarray(img, dtype=np.int16)
    diff = px[:, 1:] > px[:, :-1]            # [hash_size, hash_size] bool
    bits = 0
    for b in diff.flatten():
        bits = (bits << 1) | int(b)
    return bits


def phash(image_bytes: bytes, hash_size: int = 8, highfreq: int = 4) -> Optional[int]:
    """DCT-based perceptual hash — more robust than dHash to gamma/brightness shifts."""
    try:
        import scipy.fftpack as fft  # optional; fall back to dhash if missing
        img = _load_gray(image_bytes, (hash_size * highfreq, hash_size * highfreq))
    except Exception:
        return None
    px = np.asarray(img, dtype=np.float32)
    d = fft.dct(fft.dct(px, axis=0), axis=1)[:hash_size, :hash_size]
    med = np.median(d[1:].flatten())
    bits = 0
    for b in (d > med).flatten():
        bits = (bits << 1) | int(b)
    return bits


def hamming(a: int, b: int) -> int:
    """Number of differing bits between two hashes."""
    return bin(a ^ b).count("1")


def classify(distance: Optional[int]) -> str:
    """Map a Hamming distance to a duplicate class."""
    if distance is None:
        return "unknown"
    if distance <= EXACT_MAX:
        return "exact_duplicate"
    if distance <= NEAR_MAX:
        return "near_duplicate"
    return "similar"
