"""Multimodal indexer: images, video, audio, docs via Omni-Embed-Nemotron-3B + FAISS.

Two-phase operation:
  Backfill – iterate all active Walrus blobs; skip already-indexed.
  Listen   – after backfill, poll for new blobs every OMURA_LISTEN_INTERVAL_SECONDS.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path  # still needed for stats_path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests

from omura.parsers.file_detection import detect_file_type
from omura.parsers.multimodal import (
    is_supported_audio,
    is_supported_image,
    is_supported_video,
)
from omura.parsers.quilt import iter_quilt_patch_contents
from omura.utils.aggregator_pool import get_pool
from omura.utils.blockberry import get_current_epoch
from omura.utils.blob_discovery import iter_active_blob_entries
from omura.utils.imagebind_embeddings import (
    generate_audio_embedding,
    generate_image_embedding,
    generate_video_embedding,
    get_nsfw_embeddings,
    is_model_ready,
    is_nsfw_from_tag_score,
    nsfw_similarity_score_0_100,
)
from omura.utils.vector_store import VectorStore

# ── Configuration ──────────────────────────────────────────────────────────────

BATCH_SIZE = int(os.getenv("OMURA_INDEXER_BATCH_SIZE", "200"))
MAX_WORKERS = int(os.getenv("OMURA_INDEXER_WORKERS", "16"))
# Resume from cursor by default (faster restarts); set true to always full-scan
FULL_HISTORICAL_SCAN = (
    os.getenv("OMURA_FULL_HISTORICAL_SCAN", "false").lower() == "true"
)
# Skip already-indexed blobs by default (set true to re-embed every active blob)
FETCH_EVERY_ACTIVE_BLOB = (
    os.getenv("OMURA_FETCH_EVERY_ACTIVE_BLOB", "false").lower() == "true"
)
# Force a non-destructive refresh pass by resetting DB/cursor state only.
REINDEX_IN_PLACE = os.getenv("OMURA_REINDEX_IN_PLACE", "false").lower() == "true"

# Re-query the Walrus epoch at most this often
_EPOCH_CACHE_SEC = float(os.getenv("OMURA_INDEXER_EPOCH_CACHE_SEC", "300"))

# Listen mode
LISTEN_INTERVAL_SECONDS = int(os.getenv("OMURA_LISTEN_INTERVAL_SECONDS", "10"))
# Max pages per listen poll (100 blobs/page)
LISTEN_MAX_PAGES = int(os.getenv("OMURA_LISTEN_MAX_PAGES", "5"))
# Stop listen poll early after this many consecutive already-indexed blobs
LISTEN_EARLY_STOP_KNOWN = int(os.getenv("OMURA_LISTEN_EARLY_STOP_KNOWN", "20"))

DEFAULT_AGGREGATOR = "https://agrregator.omura.fun"
AGGREGATOR_URL = os.getenv("WALRUS_AGGREGATOR_URL", DEFAULT_AGGREGATOR).rstrip("/")

# ── HTTP session pool (thread-local, one session per worker thread) ─────────────

_thread_local = threading.local()


def _get_session() -> requests.Session:
    """Return a thread-local requests.Session with connection pooling + auto-retry."""
    if not hasattr(_thread_local, "session"):
        s = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=4,
            pool_maxsize=16,
            max_retries=requests.adapters.Retry(
                total=2,
                backoff_factor=0.5,
                status_forcelist=[502, 503, 504],
                allowed_methods=["GET"],
            ),
        )
        s.mount("https://", adapter)
        s.mount("http://", adapter)
        _thread_local.session = s
    return _thread_local.session


# ── Epoch cache ────────────────────────────────────────────────────────────────

_epoch_lock = threading.Lock()
_cached_walrus_epoch: Optional[int] = None
_cached_walrus_epoch_at: float = 0.0


def _get_walrus_epoch() -> int:
    global _cached_walrus_epoch, _cached_walrus_epoch_at
    with _epoch_lock:
        now = time.monotonic()
        if (
            _cached_walrus_epoch is not None
            and (now - _cached_walrus_epoch_at) < _EPOCH_CACHE_SEC
        ):
            return _cached_walrus_epoch
        try:
            _cached_walrus_epoch = get_current_epoch(silent=True)
        except Exception:
            if _cached_walrus_epoch is None:
                _cached_walrus_epoch = 0
        _cached_walrus_epoch_at = now
        return _cached_walrus_epoch


# ── IndexStats ─────────────────────────────────────────────────────────────────


@dataclass
class IndexStats:
    """Running counters for the indexer. Shared global; update via _update_stats()."""

    # Blobs successfully embedded and stored, by type
    indexed_image: int = 0
    indexed_video: int = 0
    indexed_audio: int = 0
    indexed_doc: int = 0
    indexed_quilt: int = 0  # quilt containers whose patches were dispatched

    # Not indexed
    skipped_already_indexed: int = 0
    skipped_expired: int = 0
    skipped_unsupported: int = 0
    failed: int = 0

    # Phase state
    backfill_complete: bool = False
    backfill_completed_at: Optional[str] = None
    last_listen_at: Optional[str] = None
    listening: bool = False

    # Coverage counters from blob_status.sqlite
    total_seen_blobs: int = 0
    active_seen_blobs: int = 0
    total_indexed_blobs: int = 0
    active_indexed_blobs: int = 0
    active_extension_distribution: Dict[str, int] = None

    @property
    def total_indexed(self) -> int:
        return (
            self.indexed_image
            + self.indexed_video
            + self.indexed_audio
            + self.indexed_doc
            + self.indexed_quilt
        )

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if d.get("active_extension_distribution") is None:
            d["active_extension_distribution"] = {}
        d["total_indexed"] = self.total_indexed
        return d

    def _add(self, other: "IndexStats") -> None:
        self.indexed_image += other.indexed_image
        self.indexed_video += other.indexed_video
        self.indexed_audio += other.indexed_audio
        self.indexed_doc += other.indexed_doc
        self.indexed_quilt += other.indexed_quilt
        self.skipped_already_indexed += other.skipped_already_indexed
        self.skipped_expired += other.skipped_expired
        self.skipped_unsupported += other.skipped_unsupported
        self.failed += other.failed

    def save(self, path: Path) -> None:
        try:
            path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        except OSError:
            pass

    @classmethod
    def load(cls, path: Path) -> "IndexStats":
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
            obj = cls()
            for k in (
                "indexed_image",
                "indexed_video",
                "indexed_audio",
                "indexed_doc",
                "indexed_quilt",
                "skipped_already_indexed",
                "skipped_expired",
                "skipped_unsupported",
                "failed",
                "backfill_complete",
                "backfill_completed_at",
                "last_listen_at",
                "total_seen_blobs",
                "active_seen_blobs",
                "total_indexed_blobs",
                "active_indexed_blobs",
                "active_extension_distribution",
            ):
                if k in d:
                    setattr(obj, k, d[k])
            if obj.active_extension_distribution is None:
                obj.active_extension_distribution = {}
            return obj
        except (OSError, json.JSONDecodeError, TypeError, AttributeError):
            return cls()


# Module-level shared stats (updated by indexer, read by API via get_indexer_stats())
_global_stats = IndexStats()
_stats_lock = threading.Lock()


def get_indexer_stats() -> Dict[str, Any]:
    """Return a snapshot of current IndexStats as a dict (safe to call from any thread)."""
    with _stats_lock:
        return _global_stats.to_dict()


def _update_stats(delta: IndexStats) -> None:
    with _stats_lock:
        _global_stats._add(delta)


def _set_stats_field(field: str, value: Any) -> None:
    with _stats_lock:
        setattr(_global_stats, field, value)


def _refresh_coverage_stats(status_db_path: Path, current_epoch: int) -> None:
    """Refresh total/active counters from blob_status.sqlite."""
    try:
        with sqlite3.connect(str(status_db_path), timeout=30) as conn:
            total_seen = conn.execute("SELECT COUNT(*) FROM blob_status").fetchone()[0]
            active_seen = conn.execute(
                """
                SELECT COUNT(*) FROM blob_status
                WHERE end_epoch IS NULL
                   OR CAST(end_epoch AS INTEGER) > ?
                """,
                (int(current_epoch),),
            ).fetchone()[0]
            total_indexed = conn.execute(
                "SELECT COUNT(*) FROM blob_status WHERE indexed=1"
            ).fetchone()[0]
            active_indexed = conn.execute(
                """
                SELECT COUNT(*) FROM blob_status
                WHERE indexed=1
                  AND (end_epoch IS NULL OR CAST(end_epoch AS INTEGER) > ?)
                """,
                (int(current_epoch),),
            ).fetchone()[0]
            ext_rows = conn.execute(
                """
                SELECT
                    LOWER(TRIM(COALESCE(extension, 'unknown'))) AS ext,
                    COUNT(*) AS c
                FROM blob_status
                WHERE end_epoch IS NULL
                   OR CAST(end_epoch AS INTEGER) > ?
                GROUP BY LOWER(TRIM(COALESCE(extension, 'unknown')))
                ORDER BY c DESC
                """,
                (int(current_epoch),),
            ).fetchall()
    except Exception:
        return

    ext_dist: Dict[str, int] = {}
    for ext, c in ext_rows or []:
        key = ext if ext else "unknown"
        ext_dist[str(key)] = int(c or 0)

    with _stats_lock:
        _global_stats.total_seen_blobs = int(total_seen or 0)
        _global_stats.active_seen_blobs = int(active_seen or 0)
        _global_stats.total_indexed_blobs = int(total_indexed or 0)
        _global_stats.active_indexed_blobs = int(active_indexed or 0)
        _global_stats.active_extension_distribution = ext_dist


def _ensure_status_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS blob_status (
                blob_id TEXT PRIMARY KEY,
                indexed INTEGER NOT NULL,
                status TEXT NOT NULL,
                reason TEXT,
                owner TEXT,
                start_epoch TEXT,
                end_epoch TEXT,
                size INTEGER,
                mime_type TEXT,
                extension TEXT,
                kind TEXT,
                first_seen_at TEXT NOT NULL,
                last_updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_blob_status_status ON blob_status(status)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_blob_status_indexed ON blob_status(indexed)"
        )
        cols = {row[1] for row in conn.execute("PRAGMA table_info(blob_status)")}
        if "is_quilt" not in cols:
            conn.execute(
                "ALTER TABLE blob_status ADD COLUMN is_quilt INTEGER NOT NULL DEFAULT 0"
            )
        conn.commit()


def _prepare_in_place_reindex(
    store: VectorStore, stats_path: Path, status_db_path: Path
) -> None:
    """Mark DB state for full refresh without deleting vector/index files."""
    try:
        with sqlite3.connect(str(status_db_path), timeout=30) as conn:
            conn.execute(
                """
                UPDATE blob_status
                SET indexed=0,
                    status='queued_reindex',
                    reason='in_place_reindex',
                    last_updated_at=?
                """,
                (_now_iso(),),
            )
            conn.commit()
    except Exception as e:
        print(f"[Indexer] In-place reindex DB prep failed: {e}")

    try:
        store.update_cursor(page=0, last_processed_blob_id=None)
    except Exception as e:
        print(f"[Indexer] In-place reindex cursor reset failed: {e}")

    try:
        with _stats_lock:
            _global_stats.backfill_complete = False
            _global_stats.backfill_completed_at = None
            _global_stats.save(stats_path)
    except Exception as e:
        print(f"[Indexer] In-place reindex stats reset failed: {e}")


def _upsert_blob_status(
    db_path: Path,
    *,
    blob_id: str,
    indexed: bool,
    status: str,
    reason: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    size: Optional[int] = None,
    mime_type: Optional[str] = None,
    extension: Optional[str] = None,
    kind: Optional[str] = None,
) -> None:
    md = metadata or {}
    now = _now_iso()
    owner = md.get("owner")
    start_epoch = md.get("start_epoch") or md.get("startEpoch")
    end_epoch = md.get("end_epoch") or md.get("endEpoch")
    is_quilt = bool(md.get("is_quilt", False)) or ("::" in blob_id) or (kind == "quilt")
    try:
        with sqlite3.connect(str(db_path), timeout=30) as conn:
            conn.execute(
                """
                INSERT INTO blob_status (
                    blob_id, indexed, status, reason, owner, start_epoch, end_epoch,
                    size, mime_type, extension, kind, is_quilt, first_seen_at, last_updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(blob_id) DO UPDATE SET
                    indexed=excluded.indexed,
                    status=excluded.status,
                    reason=excluded.reason,
                    owner=COALESCE(excluded.owner, blob_status.owner),
                    start_epoch=COALESCE(excluded.start_epoch, blob_status.start_epoch),
                    end_epoch=COALESCE(excluded.end_epoch, blob_status.end_epoch),
                    size=COALESCE(excluded.size, blob_status.size),
                    mime_type=COALESCE(excluded.mime_type, blob_status.mime_type),
                    extension=COALESCE(excluded.extension, blob_status.extension),
                    kind=COALESCE(excluded.kind, blob_status.kind),
                    is_quilt=COALESCE(excluded.is_quilt, blob_status.is_quilt),
                    last_updated_at=excluded.last_updated_at
                """,
                (
                    blob_id,
                    1 if indexed else 0,
                    status,
                    reason,
                    owner,
                    str(start_epoch) if start_epoch is not None else None,
                    str(end_epoch) if end_epoch is not None else None,
                    int(size) if size is not None else None,
                    mime_type,
                    extension,
                    kind,
                    1 if is_quilt else 0,
                    now,
                    now,
                ),
            )
            conn.commit()
    except Exception:
        # Keep indexing resilient even if status DB is temporarily unavailable.
        pass


# ── Fetch ──────────────────────────────────────────────────────────────────────


def fetch_blob_http(blob_id: str) -> Optional[bytes]:
    """Fetch blob bytes via the aggregator pool (load-balanced across upstreams)."""
    resp, _ = get_pool().get(
        f"/v1/blobs/{blob_id}", session=_get_session(), timeout=60
    )
    if resp is None or resp.status_code != 200:
        return None
    return resp.content


def fetch_quilt_patches_by_id(blob_id: str) -> List[Dict[str, str]]:
    """Fetch quilt patch descriptors via the aggregator pool.

    Endpoint (Walrus aggregator v1.48+): ``GET /v1/quilts/{quilt_id}/patches``.
    Returns ``[{"identifier": "...", "patch_id": "...", "tags": {...}}, ...]``.
    """
    resp, _ = get_pool().get(
        f"/v1/quilts/{blob_id}/patches", session=_get_session(), timeout=60
    )
    if resp is None or resp.status_code != 200:
        return []
    try:
        data = resp.json()
    except ValueError:
        return []
    if not isinstance(data, list):
        return []
    out: List[Dict[str, str]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        ident = item.get("identifier")
        patch_id = item.get("patch_id")
        if isinstance(ident, str) and isinstance(patch_id, str):
            out.append({"identifier": ident, "patch_id": patch_id})
    return out


def fetch_quilt_blob_by_identifier(quilt_id: str, identifier: str) -> Optional[bytes]:
    """Fetch one quilt patch content by identifier via the aggregator pool."""
    safe_identifier = requests.utils.quote(identifier, safe="")
    resp, _ = get_pool().get(
        f"/v1/blobs/by-quilt-id/{quilt_id}/{safe_identifier}",
        session=_get_session(), timeout=60,
    )
    if resp is None or resp.status_code != 200:
        return None
    return resp.content


# ── DB helpers (fast lookup + batch writes) ────────────────────────────────────


def _db_lookup_status(db_path: Path, blob_id: str) -> Optional[str]:
    """Return the current status string for a blob from blob_status, or None."""
    try:
        with sqlite3.connect(str(db_path), timeout=5) as conn:
            row = conn.execute(
                "SELECT status FROM blob_status WHERE blob_id=?", (blob_id,)
            ).fetchone()
            return row[0] if row else None
    except Exception:
        return None


def _collect_status_row(
    *,
    blob_id: str,
    indexed: bool,
    status: str,
    reason: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    size: Optional[int] = None,
    mime_type: Optional[str] = None,
    extension: Optional[str] = None,
    kind: Optional[str] = None,
) -> Tuple[Any, ...]:
    """Build an upsert parameter tuple (same logic as _upsert_blob_status, no commit)."""
    md = metadata or {}
    now = _now_iso()
    owner = md.get("owner")
    start_epoch = md.get("start_epoch") or md.get("startEpoch")
    end_epoch = md.get("end_epoch") or md.get("endEpoch")
    is_quilt = bool(md.get("is_quilt", False)) or ("::" in blob_id) or (kind == "quilt")
    return (
        blob_id,
        1 if indexed else 0,
        status,
        reason,
        owner,
        str(start_epoch) if start_epoch is not None else None,
        str(end_epoch) if end_epoch is not None else None,
        int(size) if size is not None else None,
        mime_type,
        extension,
        kind,
        1 if is_quilt else 0,
        now,
        now,
    )


_UPSERT_SQL = """
INSERT INTO blob_status (
    blob_id, indexed, status, reason, owner, start_epoch, end_epoch,
    size, mime_type, extension, kind, is_quilt, first_seen_at, last_updated_at
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(blob_id) DO UPDATE SET
    indexed=excluded.indexed,
    status=excluded.status,
    reason=excluded.reason,
    owner=COALESCE(excluded.owner, blob_status.owner),
    start_epoch=COALESCE(excluded.start_epoch, blob_status.start_epoch),
    end_epoch=COALESCE(excluded.end_epoch, blob_status.end_epoch),
    size=COALESCE(excluded.size, blob_status.size),
    mime_type=COALESCE(excluded.mime_type, blob_status.mime_type),
    extension=COALESCE(excluded.extension, blob_status.extension),
    kind=COALESCE(excluded.kind, blob_status.kind),
    is_quilt=COALESCE(excluded.is_quilt, blob_status.is_quilt),
    last_updated_at=excluded.last_updated_at
"""


def _flush_blob_status_batch(db_path: Path, rows: List[Tuple[Any, ...]]) -> None:
    """Write a batch of status rows in a single transaction (much faster than per-row commits)."""
    if not rows:
        return
    try:
        with sqlite3.connect(str(db_path), timeout=30) as conn:
            conn.executemany(_UPSERT_SQL, rows)
            conn.commit()
    except Exception as exc:
        print(f"[IndexerDB] Batch flush error (non-fatal): {exc}")


# ── Content indexing ───────────────────────────────────────────────────────────


def _index_content(
    blob_id: str,
    content: bytes,
    metadata: Dict[str, Any],
    store: VectorStore,
    *,
    parent_quilt_id: Optional[str] = None,
    quilt_identifier: Optional[str] = None,
    quilt_tags: Optional[Dict[str, str]] = None,
    _detected: Optional[Tuple[str, str, str]] = None,
) -> Optional[str]:
    """Detect type, embed, and store one payload. Returns kind string or None if skipped.

    Pass ``_detected=(mime, ext, kind)`` to skip redundant file-type detection.
    """
    if _detected is not None:
        mime_type, ext, kind = _detected
    else:
        mime_type, ext, kind = detect_file_type(content)
    size = len(content)

    # Dispatch by kind. Embedding functions return None when the active model
    # doesn't support that modality (e.g. Omura-Embed is image+text only).
    if kind == "image" and is_supported_image(ext):
        gen = "image"
    elif kind == "video" and is_supported_video(ext):
        gen = "video"
    elif kind == "audio" and is_supported_audio(ext):
        gen = "audio"
    else:
        return None

    embedding = None
    is_nsfw = False

    try:
        if gen == "image":
            embedding = generate_image_embedding(content, blob_id=blob_id)
        elif gen == "video":
            embedding = generate_video_embedding(content, ext, blob_id)
        elif gen == "audio":
            embedding = generate_audio_embedding(content, ext, blob_id)

        if embedding is not None:
            nsfw_vecs = get_nsfw_embeddings()
            if nsfw_vecs:
                tag_score = float(nsfw_similarity_score_0_100(embedding, nsfw_vecs))
                if is_nsfw_from_tag_score(tag_score):
                    is_nsfw = True
                    print(
                        f"{blob_id}: NSFW ({gen}) (tag_score={tag_score:.2f}/100, "
                        f"min>{os.getenv('OMURA_NSFW_TAG_SCORE_MIN', '85')})"
                    )

    except RuntimeError as e:
        err = str(e).lower()
        if "illegal" in err and "memory" in err:
            print("FATAL: CUDA Illegal Memory Access. Exiting to restart.")
            os._exit(1)
        if "cuda" in err or "illegal" in err or "memory" in err:
            print(f"{blob_id}: CUDA error: {e}")
        else:
            print(f"{blob_id}: Runtime error: {e}")
        return None
    except Exception as e:
        print(f"{blob_id}: Embedding failed ({gen}): {e}")
        return None

    if embedding is None:
        return None

    reserved = {
        "embedding",
        "blob_id",
        "mime_type",
        "size",
        "extension",
        "kind",
        "temp_file_path",
        "is_nsfw",
        "parent_quilt_id",
        "quilt_identifier",
        "quilt_tags_json",
    }
    extra = {k: v for k, v in metadata.items() if k not in reserved}
    if parent_quilt_id:
        extra["is_quilt"] = True
        extra["parent_quilt_id"] = parent_quilt_id
    if quilt_identifier:
        extra["quilt_identifier"] = quilt_identifier
    if quilt_tags:
        extra["quilt_tags_json"] = json.dumps(quilt_tags, sort_keys=True)

    store.add(
        embedding=embedding,
        blob_id=blob_id,
        mime_type=mime_type,
        size=size,
        extension=ext,
        kind=kind,
        is_nsfw=is_nsfw,
        **extra,
    )
    print(f"{blob_id}: indexed {gen} {mime_type} ({size // 1024}KB)")
    return gen


def _process_blob(
    blob_id: str,
    metadata: Dict[str, Any],
    store: VectorStore,
    walrus_epoch: int,
    status_db_path: Path,
) -> Tuple["IndexStats", List[Tuple[Any, ...]]]:
    """Fetch and index one blob.

    Returns ``(delta, status_rows)`` where ``status_rows`` is a list of upsert
    parameter tuples to be flushed in bulk by the caller via
    ``_flush_blob_status_batch``.  No DB commits happen inside this function.
    """
    delta = IndexStats()
    rows: List[Tuple[Any, ...]] = []

    # Fast path 1: already embedded in vector store
    if not FETCH_EVERY_ACTIVE_BLOB and store.get_embedding(blob_id) is not None:
        delta.skipped_already_indexed += 1
        rows.append(
            _collect_status_row(
                blob_id=blob_id,
                indexed=True,
                status="skipped_known",
                reason="already_indexed",
                metadata=metadata,
            )
        )
        return delta, rows

    # Fast path 2: already classified as unsupported in DB — skip network fetch
    if not FETCH_EVERY_ACTIVE_BLOB:
        db_status = _db_lookup_status(status_db_path, blob_id)
        if db_status == "skipped_unsupported":
            delta.skipped_unsupported += 1
            return delta, rows  # no row update needed

    # Skip expired
    end = metadata.get("end_epoch")
    if end is not None:
        try:
            if int(end) <= walrus_epoch:
                delta.skipped_expired += 1
                rows.append(
                    _collect_status_row(
                        blob_id=blob_id,
                        indexed=False,
                        status="skipped_expired",
                        reason="expired",
                        metadata=metadata,
                    )
                )
                return delta, rows
        except (TypeError, ValueError):
            pass

    try:
        content = fetch_blob_http(blob_id)
        if not content:
            delta.failed += 1
            rows.append(
                _collect_status_row(
                    blob_id=blob_id,
                    indexed=False,
                    status="failed",
                    reason="fetch_failed",
                    metadata=metadata,
                )
            )
            return delta, rows

        mime_type, ext, kind = detect_file_type(content)

        if kind == "quilt":
            patch_count = 0
            # Preferred: aggregator quilt patch API.
            patch_items = fetch_quilt_patches_by_id(blob_id)
            if patch_items:
                for item in patch_items:
                    ident = item["identifier"]
                    patch_id = item["patch_id"]
                    inner = fetch_quilt_blob_by_identifier(blob_id, ident)
                    if not inner:
                        inner = fetch_blob_http(patch_id)
                    if not inner:
                        delta.failed += 1
                        rows.append(
                            _collect_status_row(
                                blob_id=f"{blob_id}::{ident}",
                                indexed=False,
                                status="failed",
                                reason=f"patch_fetch_failed:{patch_id}",
                                metadata=metadata,
                            )
                        )
                        continue
                    inner_id = f"{blob_id}::{ident}"
                    # Detect once, pass to _index_content to avoid double work
                    inner_mime, inner_ext, inner_kind = detect_file_type(inner)
                    result = _index_content(
                        inner_id,
                        inner,
                        metadata,
                        store,
                        parent_quilt_id=blob_id,
                        quilt_identifier=ident,
                        quilt_tags={"patch_id": patch_id},
                        _detected=(inner_mime, inner_ext, inner_kind),
                    )
                    if result:
                        patch_count += 1
                        _inc_kind(delta, result)
                        rows.append(
                            _collect_status_row(
                                blob_id=inner_id,
                                indexed=True,
                                status="indexed",
                                metadata=metadata,
                                size=len(inner),
                                mime_type=inner_mime,
                                extension=inner_ext,
                                kind=inner_kind,
                            )
                        )
                    else:
                        delta.skipped_unsupported += 1
                        rows.append(
                            _collect_status_row(
                                blob_id=inner_id,
                                indexed=False,
                                status="skipped_unsupported",
                                reason="unsupported_or_empty",
                                metadata=metadata,
                                size=len(inner),
                                mime_type=inner_mime,
                                extension=inner_ext,
                                kind=inner_kind,
                            )
                        )
            else:
                # Fallback: parse quilt payload directly.
                for ident, tags, inner in iter_quilt_patch_contents(content):
                    inner_id = f"{blob_id}::{ident}"
                    inner_mime, inner_ext, inner_kind = detect_file_type(inner)
                    result = _index_content(
                        inner_id,
                        inner,
                        metadata,
                        store,
                        parent_quilt_id=blob_id,
                        quilt_identifier=ident,
                        quilt_tags=tags or None,
                        _detected=(inner_mime, inner_ext, inner_kind),
                    )
                    if result:
                        patch_count += 1
                        _inc_kind(delta, result)
                        rows.append(
                            _collect_status_row(
                                blob_id=inner_id,
                                indexed=True,
                                status="indexed",
                                metadata=metadata,
                                size=len(inner),
                                mime_type=inner_mime,
                                extension=inner_ext,
                                kind=inner_kind,
                            )
                        )
                    else:
                        delta.skipped_unsupported += 1
                        rows.append(
                            _collect_status_row(
                                blob_id=inner_id,
                                indexed=False,
                                status="skipped_unsupported",
                                reason="unsupported_or_empty",
                                metadata=metadata,
                                size=len(inner),
                                mime_type=inner_mime,
                                extension=inner_ext,
                                kind=inner_kind,
                            )
                        )
            if patch_count:
                delta.indexed_quilt += 1
                rows.append(
                    _collect_status_row(
                        blob_id=blob_id,
                        indexed=True,
                        status="indexed",
                        metadata=metadata,
                        size=len(content),
                        mime_type=mime_type,
                        extension=ext,
                        kind=kind,
                    )
                )
            else:
                rows.append(
                    _collect_status_row(
                        blob_id=blob_id,
                        indexed=False,
                        status="skipped_unsupported",
                        reason="quilt_no_supported_patches",
                        metadata=metadata,
                        size=len(content),
                        mime_type=mime_type,
                        extension=ext,
                        kind=kind,
                    )
                )
            return delta, rows

        result = _index_content(
            blob_id, content, metadata, store, _detected=(mime_type, ext, kind)
        )
        if result:
            _inc_kind(delta, result)
            rows.append(
                _collect_status_row(
                    blob_id=blob_id,
                    indexed=True,
                    status="indexed",
                    metadata=metadata,
                    size=len(content),
                    mime_type=mime_type,
                    extension=ext,
                    kind=kind,
                )
            )
        else:
            delta.skipped_unsupported += 1
            rows.append(
                _collect_status_row(
                    blob_id=blob_id,
                    indexed=False,
                    status="skipped_unsupported",
                    reason="unsupported_or_empty",
                    metadata=metadata,
                    size=len(content),
                    mime_type=mime_type,
                    extension=ext,
                    kind=kind,
                )
            )

    except Exception as e:
        print(f"{blob_id}: ERROR: {e}")
        delta.failed += 1
        rows.append(
            _collect_status_row(
                blob_id=blob_id,
                indexed=False,
                status="failed",
                reason=str(e)[:500],
                metadata=metadata,
            )
        )

    return delta, rows


def _inc_kind(delta: IndexStats, kind: str) -> None:
    if kind == "image":
        delta.indexed_image += 1
    elif kind == "video":
        delta.indexed_video += 1
    elif kind == "audio":
        delta.indexed_audio += 1
    elif kind == "doc":
        delta.indexed_doc += 1


def _wait_for_model(poll_interval: float = 30.0) -> None:
    """Block until the embedding model is loaded. Logs once per wait cycle."""
    import time as _t

    while not is_model_ready():
        print(
            f"[Indexer] Model not ready (GPU OOM?); waiting {poll_interval:.0f}s before retrying..."
        )
        _t.sleep(poll_interval)
        # Attempt to load — will be a no-op if still in backoff
        try:
            from omura.utils.imagebind_embeddings import ensure_model_loaded

            ensure_model_loaded()
        except Exception:
            pass


def process_batch(
    batch: List[Tuple[str, Dict[str, Any]]],
    store: VectorStore,
    walrus_epoch: int,
    status_db_path: Path,
) -> IndexStats:
    """Process a batch of blobs in parallel worker threads.

    DB status rows are collected from all workers and flushed in a single
    batch transaction for maximum write throughput.
    """
    _wait_for_model()  # don't fetch blobs if the GPU can't embed them yet
    batch_delta = IndexStats()
    all_rows: List[Tuple[Any, ...]] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(
                _process_blob, blob_id, meta, store, walrus_epoch, status_db_path
            ): blob_id
            for blob_id, meta in batch
        }
        for future in as_completed(futures):
            try:
                delta, rows = future.result()
                batch_delta._add(delta)
                all_rows.extend(rows)
            except Exception as e:
                print(f"{futures[future]}: worker exception: {e}")
                batch_delta.failed += 1
    # Single batch commit for all status rows from this batch
    _flush_blob_status_batch(status_db_path, all_rows)
    return batch_delta


# ── Backfill ───────────────────────────────────────────────────────────────────


def _backfill(
    store: VectorStore,
    stats_path: Path,
    status_db_path: Path,
    batch_size: int = BATCH_SIZE,
    max_batches: Optional[int] = None,
) -> None:
    """One complete pass through all active Walrus blobs, skipping already-indexed."""
    cursor = store.get_cursor()
    start_page = 0 if FULL_HISTORICAL_SCAN else cursor.get("last_page", 0)
    last_blob = None if FULL_HISTORICAL_SCAN else cursor.get("last_processed_blob_id")

    print(
        f"[Backfill] page={start_page} last_blob={last_blob} "
        f"full_scan={FULL_HISTORICAL_SCAN} fetch_all={FETCH_EVERY_ACTIVE_BLOB} workers={MAX_WORKERS}"
    )

    batch: List[Tuple[str, Dict[str, Any]]] = []
    batches_done = 0
    current_page = start_page
    skip_until = last_blob

    for page, blob_id, meta in iter_active_blob_entries(start_page=start_page):
        current_page = page

        # Resume past cursor position
        if skip_until:
            if blob_id == skip_until:
                skip_until = None
            continue

        batch.append((blob_id, meta))
        if len(batch) < batch_size:
            continue

        walrus_epoch = _get_walrus_epoch()
        print(
            f"[Backfill] Batch #{batches_done + 1} ({batch_size} blobs) page={current_page}"
        )
        delta = process_batch(batch, store, walrus_epoch, status_db_path)
        _update_stats(delta)
        _refresh_coverage_stats(status_db_path, walrus_epoch)
        batches_done += 1
        last_in_batch = batch[-1][0]
        batch = []

        store.update_cursor(page=current_page, last_processed_blob_id=last_in_batch)

        if batches_done % 5 == 0:
            store.save(create_backup=False)
            with _stats_lock:
                _global_stats.save(stats_path)
            _print_stats("[Backfill]")

        if max_batches is not None and batches_done >= max_batches:
            print("[Backfill] max_batches reached, stopping.")
            break

    # Trailing partial batch
    if batch:
        walrus_epoch = _get_walrus_epoch()
        print(f"[Backfill] Final batch ({len(batch)} blobs)")
        delta = process_batch(batch, store, walrus_epoch, status_db_path)
        _update_stats(delta)
        _refresh_coverage_stats(status_db_path, walrus_epoch)
        store.update_cursor(page=current_page, last_processed_blob_id=batch[-1][0])
        store.save(create_backup=False)

    with _stats_lock:
        _global_stats.backfill_complete = True
        _global_stats.backfill_completed_at = _now_iso()
        _global_stats.save(stats_path)

    print("[Backfill] Complete.")
    _print_stats("[Backfill]")


# ── Listen ─────────────────────────────────────────────────────────────────────


def _listen_once(store: VectorStore, stats_path: Path, status_db_path: Path) -> None:
    """Poll for new blobs (newest-first by timestamp), index anything not yet in store.

    New blobs are collected into a batch and processed together for maximum
    parallelism rather than sequentially one-by-one.
    """
    walrus_epoch = _get_walrus_epoch()
    seen_known = 0
    last_page = -1
    new_blobs: List[Tuple[str, Dict[str, Any]]] = []
    known_rows: List[Tuple[Any, ...]] = []

    # Always use Blockberry for listen — it supports TIMESTAMP sort; GraphQL does not.
    for page, blob_id, meta in iter_active_blob_entries(
        sort_by="TIMESTAMP", order_by="DESC", source="blockberry"
    ):
        if page != last_page:
            if page >= LISTEN_MAX_PAGES:
                break
            last_page = page

        if store.get_embedding(blob_id) is not None:
            seen_known += 1
            known_rows.append(
                _collect_status_row(
                    blob_id=blob_id,
                    indexed=True,
                    status="skipped_known",
                    reason="already_indexed",
                    metadata=meta,
                )
            )
            # Stop early: if we've seen many consecutive known blobs, we're past new ones
            if seen_known >= LISTEN_EARLY_STOP_KNOWN:
                break
            continue

        seen_known = 0  # reset streak on any new blob
        new_blobs.append((blob_id, meta))

    # Flush known-blob rows in one shot
    _flush_blob_status_batch(status_db_path, known_rows)

    if new_blobs:
        print(f"[Listen] {len(new_blobs)} new blob(s) to index")
        delta = process_batch(new_blobs, store, walrus_epoch, status_db_path)
        _update_stats(delta)
        if delta.total_indexed > 0:
            store.save(create_backup=False)
            _print_stats("[Listen]")

    _refresh_coverage_stats(status_db_path, walrus_epoch)

    with _stats_lock:
        _global_stats.last_listen_at = _now_iso()
        _global_stats.save(stats_path)


# ── Main loop ──────────────────────────────────────────────────────────────────


def _print_stats(prefix: str = "") -> None:
    with _stats_lock:
        s = _global_stats.to_dict()
    print(
        f"{prefix} indexed: image={s['indexed_image']} video={s['indexed_video']} "
        f"audio={s['indexed_audio']} doc={s['indexed_doc']} quilt={s['indexed_quilt']} "
        f"total={s['total_indexed']} | "
        f"skip: known={s['skipped_already_indexed']} expired={s['skipped_expired']} "
        f"unsupported={s['skipped_unsupported']} | failed={s['failed']} | "
        f"coverage: total_seen={s.get('total_seen_blobs', 0)} "
        f"active_seen={s.get('active_seen_blobs', 0)} "
        f"indexed_total={s.get('total_indexed_blobs', 0)} "
        f"indexed_active={s.get('active_indexed_blobs', 0)}"
    )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_indexer_with_stores(
    vector_stores: Dict[str, VectorStore],
    batch_size: int = BATCH_SIZE,
    max_batches: Optional[int] = None,
) -> None:
    """Entry point: backfill all active blobs, then listen continuously for new ones.

    Also spawns the QuiltExpander background thread which walks every ``kind='quilt'``
    row in the catalog and identifies inner patches (audio/video by filename, image by
    fetch + embed). Disable via ``OMURA_EXPAND_QUILTS=false``.
    """
    global _global_stats

    store = vector_stores.get("image") or next(iter(vector_stores.values()))
    stats_path = store.index_path.parent / "indexer_stats.json"
    status_db_path = store.index_path.parent / "blob_status.sqlite"
    _ensure_status_db(status_db_path)

    with _stats_lock:
        _global_stats = IndexStats.load(stats_path)

    # Background quilt expander: walks every quilt in catalog, records inner patches.
    try:
        from omura.indexers.quilt_expander import start_quilt_expander_thread
        from omura.utils.blob_catalog import CATALOG_DB_PATH

        start_quilt_expander_thread(CATALOG_DB_PATH, store)
    except Exception as e:
        print(f"[Indexer] QuiltExpander failed to start (non-fatal): {e}")

    # Real-time Sui event listener: detects new blobs the instant they're certified
    # on-chain (way faster than the cataloger's polling). Catalog row gets written
    # within ~2-5s of upload.
    try:
        from omura.indexers.sui_event_listener import start_sui_event_listener_thread
        from omura.utils.blob_catalog import CATALOG_DB_PATH as _CDB

        start_sui_event_listener_thread(_CDB)
    except Exception as e:
        print(f"[Indexer] SuiEventListener failed to start (non-fatal): {e}")

    if REINDEX_IN_PLACE:
        print(
            "[Indexer] OMURA_REINDEX_IN_PLACE=true -> preparing non-destructive reindex"
        )
        _prepare_in_place_reindex(store, stats_path, status_db_path)

    # By default, require a full backfill pass before listen mode.
    # Set OMURA_REQUIRE_FULL_BACKFILL=false to restore legacy behavior.
    require_full_backfill = (
        os.getenv("OMURA_REQUIRE_FULL_BACKFILL", "true").lower() == "true"
    )
    if REINDEX_IN_PLACE:
        require_full_backfill = True
        print("[Indexer] In-place reindex enabled: forcing backfill from page 0")
    if require_full_backfill:
        max_batches = None

    retry_count = 0
    max_retries = 5
    retry_delay = 30

    while True:
        try:
            # ── Backfill ───────────────────────────────────────────────────────
            with _stats_lock:
                already_done = _global_stats.backfill_complete

            if require_full_backfill or not already_done:
                print("[Indexer] Backfill: scanning all active Walrus blobs...")
                _backfill(
                    store,
                    stats_path,
                    status_db_path,
                    batch_size=batch_size,
                    max_batches=max_batches,
                )
            else:
                print(
                    f"[Indexer] Backfill already complete "
                    f"({_global_stats.backfill_completed_at}). Skipping to listen."
                )

            # ── Listen ─────────────────────────────────────────────────────────
            print(
                f"[Indexer] Listening for new blobs every {LISTEN_INTERVAL_SECONDS}s..."
            )
            _set_stats_field("listening", True)

            while True:
                time.sleep(LISTEN_INTERVAL_SECONDS)
                try:
                    _listen_once(store, stats_path, status_db_path)
                except Exception as e:
                    print(f"[Listen] Poll error (non-fatal): {e}")

        except Exception as e:
            retry_count += 1
            import traceback

            print(f"[Indexer] Error ({retry_count}/{max_retries}): {e}")
            traceback.print_exc()
            if retry_count >= max_retries:
                store.save(create_backup=True)
                time.sleep(retry_delay * 2)
                retry_count = 0
            else:
                time.sleep(retry_delay)


# Legacy alias
def run_indexer_with_store(vector_store: VectorStore, **kwargs) -> None:
    run_indexer_with_stores({"image": vector_store}, **kwargs)
