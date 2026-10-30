"""Segment ("clip") level vector index for long audio/video.

EmbeddingGemma 2 accepts a limited number of video frames / audio seconds per
request. Indexing overlapping or tiled clips lets one long blob contribute many
vectors while still returning the parent blob to the caller.

The index is deliberately a *sidecar*: it does not touch ``VectorStore`` or the
blob catalog, so counters, NSFW scoring and image search are unaffected.

Layout::

    <dir>/clip_vectors.npy   float32 (N, dim)
    <dir>/clips.json         [{"blob_id", "kind", "clip_index",
                               "start_sec", "end_sec", ...}, ...]

The API loads ``clip_vectors.npy`` with ``mmap_mode="r"`` and scores it in
chunks, so RAM use stays bounded even for large indexes. The backfill script
runs in writable mode and appends in memory before an atomic save.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Optional

import numpy as np

VECTORS_FILE = "clip_vectors.npy"
RECORDS_FILE = "clips.json"
SCORE_CHUNK = int(os.getenv("OMURA_CLIP_SCORE_CHUNK", "65536"))


class ClipIndex:
    """Appendable, memory-mappable clip vector index."""

    def __init__(self, directory: str | Path, embedding_dim: int, writable: bool = False):
        self.directory = Path(directory)
        self.embedding_dim = int(embedding_dim)
        self.writable = writable
        self.vectors_path = self.directory / VECTORS_FILE
        self.records_path = self.directory / RECORDS_FILE

        self._vectors: Optional[np.ndarray] = None
        self._records: list[dict[str, Any]] = []
        self._added_vectors: list[np.ndarray] = []
        self._added_records: list[dict[str, Any]] = []
        self._mtime: Optional[tuple[int, int]] = None
        self._lock = threading.RLock()

    # ── persistence ───────────────────────────────────────────────────────────

    def _file_mtime(self) -> Optional[tuple[int, int]]:
        try:
            return (
                self.vectors_path.stat().st_mtime_ns,
                self.records_path.stat().st_mtime_ns,
            )
        except OSError:
            return None

    def load(self, force: bool = False) -> "ClipIndex":
        with self._lock:
            mtime = self._file_mtime()
            if mtime is None:
                self._vectors = None
                self._records = []
                self._mtime = None
                return self
            if not force and mtime == self._mtime:
                return self

            vectors = np.load(
                self.vectors_path, mmap_mode=None if self.writable else "r"
            )
            if vectors.ndim != 2 or vectors.shape[1] != self.embedding_dim:
                raise ValueError(
                    f"clip index {self.vectors_path} has shape {vectors.shape}, "
                    f"expected (N, {self.embedding_dim})"
                )
            records = json.loads(self.records_path.read_text(encoding="utf-8"))
            if len(records) != vectors.shape[0]:
                raise ValueError(
                    f"clip index mismatch: {vectors.shape[0]} vectors vs "
                    f"{len(records)} records"
                )
            self._vectors = vectors
            self._records = records
            self._mtime = mtime
            return self

    def reload_if_stale(self) -> "ClipIndex":
        if self._file_mtime() != self._mtime:
            return self.load(force=True)
        return self

    def save(self) -> None:
        if not self.writable:
            raise RuntimeError("clip index is read-only")
        with self._lock:
            matrix = self._stacked_vectors()
            records = self._all_records()
            self.directory.mkdir(parents=True, exist_ok=True)

            tmp_vectors = self.directory / "clip_vectors.tmp.npy"
            tmp_records = self.records_path.with_suffix(".json.tmp")
            # Pass a file handle: np.save() appends ".npy" to bare paths.
            with open(tmp_vectors, "wb") as fh:
                np.save(fh, matrix)
            tmp_records.write_text(
                json.dumps(records, ensure_ascii=False, separators=(",", ":")),
                encoding="utf-8",
            )
            os.replace(tmp_vectors, self.vectors_path)
            os.replace(tmp_records, self.records_path)

            self._vectors = matrix
            self._records = records
            self._added_vectors = []
            self._added_records = []
            self._mtime = self._file_mtime()

    # ── write path ────────────────────────────────────────────────────────────

    def add(
        self,
        *,
        blob_id: str,
        vector: np.ndarray,
        kind: str,
        start_sec: float,
        end_sec: float,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        if not self.writable:
            raise RuntimeError("clip index is read-only")
        vec = np.asarray(vector, dtype=np.float32).flatten()
        if vec.shape[0] != self.embedding_dim:
            raise ValueError(
                f"expected {self.embedding_dim}-d clip vector, got {vec.shape[0]}"
            )
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm

        record: dict[str, Any] = {
            "blob_id": blob_id,
            "kind": kind,
            "start_sec": round(float(start_sec), 3),
            "end_sec": round(float(end_sec), 3),
        }
        record.update(metadata or {})
        with self._lock:
            self._added_vectors.append(vec.astype(np.float32, copy=False))
            self._added_records.append(record)

    # ── read path ─────────────────────────────────────────────────────────────

    def _all_records(self) -> list[dict[str, Any]]:
        return self._records + self._added_records

    def _stacked_vectors(self) -> np.ndarray:
        if self._vectors is None or self._vectors.shape[0] == 0:
            base = None
        else:
            base = np.asarray(self._vectors, dtype=np.float32)
        if not self._added_vectors:
            if base is None:
                return np.zeros((0, self.embedding_dim), dtype=np.float32)
            return base
        added = np.vstack(self._added_vectors)
        if base is None:
            return added
        return np.vstack([base, added])

    def _matrix(self) -> Optional[np.ndarray]:
        # Keep the memmap when nothing was appended (API read path).
        if not self._added_vectors:
            return self._vectors
        return self._stacked_vectors()

    def size(self) -> int:
        return len(self._records) + len(self._added_records)

    def is_empty(self) -> bool:
        return self.size() == 0

    def has_parent(self, blob_id: str) -> bool:
        return any(rec.get("blob_id") == blob_id for rec in self._all_records())

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        *,
        kind: Optional[str] = None,
        parent_blob_id: Optional[str] = None,
        dedupe_parents: bool = True,
    ) -> list[dict[str, Any]]:
        """Cosine search over clips, returning the best clip per parent by default."""
        matrix = self._matrix()
        records = self._all_records()
        if matrix is None or matrix.shape[0] == 0 or not records:
            return []

        query = np.asarray(query_embedding, dtype=np.float32).flatten()
        if query.shape[0] != self.embedding_dim:
            raise ValueError(
                f"expected {self.embedding_dim}-d query, got {query.shape[0]}"
            )
        norm = float(np.linalg.norm(query))
        if norm > 0:
            query = query / norm

        scores = np.empty(matrix.shape[0], dtype=np.float32)
        for start in range(0, matrix.shape[0], SCORE_CHUNK):
            block = np.asarray(matrix[start : start + SCORE_CHUNK], dtype=np.float32)
            scores[start : start + len(block)] = block @ query

        if kind is not None or parent_blob_id is not None:
            mask = np.fromiter(
                (
                    (kind is None or rec.get("kind") == kind)
                    and (parent_blob_id is None or rec.get("blob_id") == parent_blob_id)
                    for rec in records
                ),
                dtype=bool,
                count=len(records),
            )
            scores = np.where(mask, scores, -np.inf)

        take = min(len(scores), max(int(top_k) * 4, int(top_k)))
        if take <= 0:
            return []
        top = np.argpartition(-scores, take - 1)[:take]
        top = top[np.argsort(-scores[top])]

        results: list[dict[str, Any]] = []
        seen: set[str] = set()
        for idx in top:
            score = float(scores[int(idx)])
            if not np.isfinite(score):
                continue
            rec = dict(records[int(idx)])
            parent = rec.get("blob_id")
            if dedupe_parents:
                if parent in seen:
                    continue
                seen.add(parent)
            rec["score"] = score
            results.append(rec)
            if len(results) >= int(top_k):
                break
        return results

    def stats(self) -> dict[str, Any]:
        records = self._all_records()
        kinds: dict[str, int] = {}
        parents: set[str] = set()
        for rec in records:
            kinds[rec.get("kind", "other")] = kinds.get(rec.get("kind", "other"), 0) + 1
            if rec.get("blob_id"):
                parents.add(rec["blob_id"])
        return {
            "clips": len(records),
            "parents": len(parents),
            "kinds": kinds,
            "directory": str(self.directory),
        }


__all__ = ["ClipIndex", "VECTORS_FILE", "RECORDS_FILE"]
