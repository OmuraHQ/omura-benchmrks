"""Shared helpers for the Phase 3 retrieval benchmarks.

Every benchmark writes a JSON artifact with the same provenance fields
(``model``, ``dim``, ``protocol``, ``pool_size``, ``num_queries``, ``seed``,
``commit``, ``utc_date``) plus per-direction Recall@K.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def l2_normalize(v: np.ndarray) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float32)
    if arr.ndim == 1:
        n = float(np.linalg.norm(arr))
        return (arr / n).astype(np.float32) if n > 0 else arr
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return (arr / norms).astype(np.float32)


def stack_vectors(vecs: Sequence[Optional[np.ndarray]], dim: int) -> tuple[np.ndarray, list[int]]:
    """Stack non-None embeddings and return (matrix, kept_positions)."""
    kept, out = [], []
    for i, v in enumerate(vecs):
        if v is None:
            continue
        arr = np.asarray(v, dtype=np.float32).reshape(-1)
        if arr.size != dim:
            continue
        out.append(l2_normalize(arr))
        kept.append(i)
    if not out:
        return np.zeros((0, dim), dtype=np.float32), []
    return np.stack(out, axis=0), kept


def _rank_of_first_hit(sims: np.ndarray, gt: Sequence) -> np.ndarray:
    """1-based rank of the best positive per row (len(gt)+1 if none)."""
    ranks = np.empty(sims.shape[0], dtype=np.int32)
    for i in range(sims.shape[0]):
        positives = np.asarray(gt[i], dtype=np.int64).reshape(-1)
        positives = positives[(positives >= 0) & (positives < sims.shape[1])]
        if positives.size == 0:
            ranks[i] = sims.shape[1] + 1
            continue
        pos_sims = sims[i, positives]
        best = float(pos_sims.max())
        # Rank = number of candidates strictly above the best positive + 1.
        ranks[i] = int(np.sum(sims[i] > best)) + 1
    return ranks


def recall_at_k(sims: np.ndarray, gt: Sequence, ks: Sequence[int] = (1, 5, 10)) -> dict:
    """Recall@K from a query x candidate similarity matrix and per-query qrels."""
    ranks = _rank_of_first_hit(sims, gt)
    n = int(sims.shape[0])
    return {
        f"R@{k}": float(np.sum(ranks <= k)) / float(n) if n else 0.0 for k in ks
    }


def ranks_at_k(sims: np.ndarray, gt: Sequence) -> np.ndarray:
    return _rank_of_first_hit(sims, gt)


def embed_texts(
    embed_fn: Callable[[Sequence[str]], Sequence[Optional[np.ndarray]]],
    texts: Sequence[str],
    dim: int,
    batch_size: int = 64,
    desc: str = "Embedding text",
    is_document: bool = False,
) -> tuple[np.ndarray, list[int]]:
    all_vecs: list[Optional[np.ndarray]] = []
    for start in tqdm(range(0, len(texts), batch_size), desc=desc, unit="batch"):
        chunk = [str(t) for t in texts[start : start + batch_size]]
        if is_document:
            vecs = embed_fn(chunk, is_document=True)
        else:
            vecs = embed_fn(chunk)
        all_vecs.extend(vecs)
    return stack_vectors(all_vecs, dim)


def embed_images(
    embed_fn: Callable[[Sequence[bytes]], Sequence[Optional[np.ndarray]]],
    datas: Sequence[bytes],
    dim: int,
    batch_size: int = 32,
    desc: str = "Embedding images",
    blob_ids: Optional[Sequence[str]] = None,
) -> tuple[np.ndarray, list[int]]:
    all_vecs: list[Optional[np.ndarray]] = []
    for start in tqdm(range(0, len(datas), batch_size), desc=desc, unit="batch"):
        chunk = datas[start : start + batch_size]
        ids = blob_ids[start : start + batch_size] if blob_ids else None
        try:
            vecs = embed_fn(chunk, ids)
        except TypeError:
            vecs = embed_fn(chunk)
        all_vecs.extend(vecs)
    return stack_vectors(all_vecs, dim)


def write_result(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def env_snapshot() -> dict[str, str]:
    keys = (
        "OMURA_EMBEDDING_BACKEND",
        "OMURA_EMBEDDING_DIM",
        "OMURA_OMURAGEMMA_MODEL",
        "OMURA_VECTOR_STORE_DIR",
        "CUDA_VISIBLE_DEVICES",
    )
    return {k: os.getenv(k, "") for k in keys}
