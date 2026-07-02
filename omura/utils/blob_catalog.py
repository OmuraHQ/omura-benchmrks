"""
Blob catalog: shared SQLite schema and helpers for the cataloger and vector indexer.

Tables
------
blobs        — every Walrus blob ever seen (type, status, indexing, NSFW, cluster)
index_queue  — images waiting to be embedded by the vector indexer
clusters     — semantic cluster metadata (rebuilt periodically)
"""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

CATALOG_DIR = Path(os.getenv("OMURA_DATA_DIR", "data"))
CATALOG_DB_PATH = CATALOG_DIR / "blob_catalog.sqlite"

_CREATE_BLOBS = """
CREATE TABLE IF NOT EXISTS blobs (
    blob_id         TEXT PRIMARY KEY,
    -- Discovery
    end_epoch       INTEGER,
    size            INTEGER,
    owner           TEXT,
    source          TEXT DEFAULT 'blockberry',
    -- File type (written by cataloger after partial fetch)
    mime_type       TEXT,
    extension       TEXT,
    kind            TEXT,
    actual_size     INTEGER,
    -- Lifecycle
    is_active       INTEGER DEFAULT 1,
    fetch_ok        INTEGER DEFAULT 0,
    status          TEXT DEFAULT 'discovered',
    -- Vector indexing (written by vector indexer)
    indexed         INTEGER DEFAULT 0,
    is_nsfw         INTEGER DEFAULT 0,
    nsfw_score      REAL,
    cluster_id      INTEGER,
    -- Text search
    caption         TEXT,
    -- Timestamps
    first_seen_at   TEXT NOT NULL,
    last_updated_at TEXT NOT NULL
)
"""

_CREATE_INDEX_QUEUE = """
CREATE TABLE IF NOT EXISTS index_queue (
    blob_id         TEXT PRIMARY KEY,
    queued_at       TEXT NOT NULL,
    attempts        INTEGER DEFAULT 0,
    last_attempt_at TEXT
)
"""

_CREATE_CLUSTERS = """
CREATE TABLE IF NOT EXISTS clusters (
    cluster_id        INTEGER PRIMARY KEY,
    label             TEXT,
    size              INTEGER DEFAULT 0,
    centroid_blob_id  TEXT,
    updated_at        TEXT
)
"""

_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_blobs_kind    ON blobs(kind)",
    "CREATE INDEX IF NOT EXISTS idx_blobs_status  ON blobs(status)",
    "CREATE INDEX IF NOT EXISTS idx_blobs_indexed ON blobs(indexed)",
    "CREATE INDEX IF NOT EXISTS idx_blobs_cluster ON blobs(cluster_id)",
    "CREATE INDEX IF NOT EXISTS idx_blobs_nsfw    ON blobs(is_nsfw)",
    "CREATE INDEX IF NOT EXISTS idx_blobs_active  ON blobs(is_active)",
]

_UPSERT_BLOB_SQL = """
INSERT INTO blobs (
    blob_id, end_epoch, size, owner, source,
    mime_type, extension, kind, actual_size,
    is_active, fetch_ok, status,
    first_seen_at, last_updated_at
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(blob_id) DO UPDATE SET
    end_epoch       = COALESCE(excluded.end_epoch,   blobs.end_epoch),
    size            = COALESCE(excluded.size,         blobs.size),
    owner           = COALESCE(excluded.owner,        blobs.owner),
    source          = COALESCE(excluded.source,       blobs.source),
    mime_type       = COALESCE(excluded.mime_type,    blobs.mime_type),
    extension       = COALESCE(excluded.extension,    blobs.extension),
    kind            = COALESCE(excluded.kind,         blobs.kind),
    actual_size     = COALESCE(excluded.actual_size,  blobs.actual_size),
    is_active       = excluded.is_active,
    fetch_ok        = excluded.fetch_ok,
    status          = excluded.status,
    last_updated_at = excluded.last_updated_at
"""

_ENQUEUE_SQL = """
INSERT OR IGNORE INTO index_queue (blob_id, queued_at) VALUES (?, ?)
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_catalog_db(db_path: Path = CATALOG_DB_PATH) -> None:
    """Create tables and indexes if they don't exist."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(db_path), timeout=30) as conn:
        conn.execute(_CREATE_BLOBS)
        conn.execute(_CREATE_INDEX_QUEUE)
        conn.execute(_CREATE_CLUSTERS)
        for idx_sql in _INDEXES:
            conn.execute(idx_sql)
        # Migrations
        try:
            conn.execute("ALTER TABLE blobs ADD COLUMN caption TEXT")
        except sqlite3.OperationalError:
            pass
            
        # Create FTS5 virtual table
        conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS blobs_fts USING fts5(blob_id UNINDEXED, caption)")
        
        # Drop old triggers to ensure we replace them with corrected ones
        conn.execute("DROP TRIGGER IF EXISTS blobs_after_insert")
        conn.execute("DROP TRIGGER IF EXISTS blobs_after_update")
        conn.execute("DROP TRIGGER IF EXISTS blobs_after_delete")

        # Setup Triggers for blobs_fts synchronization
        conn.execute(
            """
            CREATE TRIGGER IF NOT EXISTS blobs_after_insert AFTER INSERT ON blobs
            BEGIN
                DELETE FROM blobs_fts WHERE blob_id = new.blob_id;
                INSERT INTO blobs_fts(blob_id, caption) VALUES(new.blob_id, new.caption);
            END;
            """
        )
        conn.execute(
            """
            CREATE TRIGGER IF NOT EXISTS blobs_after_update AFTER UPDATE OF caption ON blobs
            BEGIN
                DELETE FROM blobs_fts WHERE blob_id = new.blob_id;
                INSERT INTO blobs_fts(blob_id, caption) VALUES(new.blob_id, new.caption);
            END;
            """
        )
        conn.execute(
            """
            CREATE TRIGGER IF NOT EXISTS blobs_after_delete AFTER DELETE ON blobs
            BEGIN
                DELETE FROM blobs_fts WHERE blob_id = old.blob_id;
            END;
            """
        )
        
        # Clean up existing duplicate rows in blobs_fts and repopulate cleanly
        conn.execute("DELETE FROM blobs_fts")
        conn.execute(
            """
            INSERT INTO blobs_fts(blob_id, caption)
            SELECT blob_id, caption FROM blobs
            WHERE caption IS NOT NULL AND is_active = 1
            """
        )
        conn.commit()


def build_blob_row(
    blob_id: str,
    metadata: Dict[str, Any],
    *,
    mime_type: Optional[str] = None,
    extension: Optional[str] = None,
    kind: Optional[str] = None,
    actual_size: Optional[int] = None,
    fetch_ok: bool = False,
    is_active: bool = True,
    status: str = "discovered",
    current_epoch: int = 0,
) -> Tuple:
    """Build an upsert parameter tuple for the blobs table."""
    end_epoch = metadata.get("end_epoch")
    if end_epoch is not None:
        try:
            end_epoch = int(end_epoch)
            if current_epoch and end_epoch <= current_epoch:
                is_active = False
                if status == "discovered":
                    status = "expired"
        except (TypeError, ValueError):
            end_epoch = None

    now = _now()
    return (
        blob_id,
        end_epoch,
        metadata.get("size"),
        metadata.get("owner"),
        metadata.get("source", "blockberry"),
        mime_type,
        extension,
        kind,
        actual_size,
        1 if is_active else 0,
        1 if fetch_ok else 0,
        status,
        now,
        now,
    )


def upsert_blobs_batch(rows: List[Tuple], db_path: Path = CATALOG_DB_PATH) -> None:
    """Batch-upsert blob rows into the catalog (single transaction)."""
    if not rows:
        return
    try:
        with sqlite3.connect(str(db_path), timeout=30) as conn:
            conn.executemany(_UPSERT_BLOB_SQL, rows)
            conn.commit()
    except Exception as exc:
        print(f"[BlobCatalog] upsert error (non-fatal): {exc}")


def enqueue_images(blob_ids: List[str], db_path: Path = CATALOG_DB_PATH) -> int:
    """Add image blob_ids to the index_queue. Returns count of newly queued."""
    if not blob_ids:
        return 0
    now = _now()
    rows = [(bid, now) for bid in blob_ids]
    try:
        with sqlite3.connect(str(db_path), timeout=30) as conn:
            conn.executemany(_ENQUEUE_SQL, rows)
            conn.commit()
            return conn.execute("SELECT changes()").fetchone()[0]
    except Exception as exc:
        print(f"[BlobCatalog] enqueue error (non-fatal): {exc}")
        return 0


def dequeue_blobs(limit: int = 64, db_path: Path = CATALOG_DB_PATH) -> List[str]:
    """Atomically claim up to `limit` blobs from the queue. Returns blob_ids."""
    try:
        with sqlite3.connect(str(db_path), timeout=30) as conn:
            rows = conn.execute(
                """
                SELECT blob_id FROM index_queue
                ORDER BY queued_at ASC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            if not rows:
                return []
            ids = [r[0] for r in rows]
            conn.execute(
                f"DELETE FROM index_queue WHERE blob_id IN ({','.join('?' * len(ids))})",
                ids,
            )
            conn.commit()
            return ids
    except Exception as exc:
        print(f"[BlobCatalog] dequeue error (non-fatal): {exc}")
        return []


def requeue_blobs(blob_ids: List[str], db_path: Path = CATALOG_DB_PATH) -> None:
    """Return blob_ids to the queue after a transient failure."""
    if not blob_ids:
        return
    now = _now()
    rows = [(bid, now) for bid in blob_ids]
    try:
        with sqlite3.connect(str(db_path), timeout=30) as conn:
            conn.executemany(
                "INSERT OR IGNORE INTO index_queue (blob_id, queued_at, attempts) VALUES (?, ?, 1)",
                rows,
            )
            conn.commit()
    except Exception as exc:
        print(f"[BlobCatalog] requeue error (non-fatal): {exc}")


def mark_indexed(
    blob_id: str,
    *,
    is_nsfw: bool = False,
    nsfw_score: float = 0.0,
    cluster_id: Optional[int] = None,
    caption: Optional[str] = None,
    db_path: Path = CATALOG_DB_PATH,
) -> None:
    try:
        with sqlite3.connect(str(db_path), timeout=30) as conn:
            conn.execute(
                """
                UPDATE blobs SET
                    indexed = 1,
                    status = 'indexed',
                    is_nsfw = ?,
                    nsfw_score = ?,
                    cluster_id = ?,
                    caption = ?,
                    last_updated_at = ?
                WHERE blob_id = ?
                """,
                (1 if is_nsfw else 0, nsfw_score, cluster_id, caption, _now(), blob_id),
            )
            conn.commit()
    except Exception as exc:
        print(f"[BlobCatalog] mark_indexed error (non-fatal): {exc}")


def mark_indexed_batch(
    records: List[Dict[str, Any]],
    db_path: Path = CATALOG_DB_PATH,
) -> None:
    """Batch update indexed status. Each record: {blob_id, is_nsfw, nsfw_score, cluster_id, caption}."""
    if not records:
        return
    now = _now()
    rows = [
        (
            1 if r.get("is_nsfw") else 0,
            float(r.get("nsfw_score", 0.0)),
            r.get("cluster_id"),
            r.get("caption"),
            now,
            r["blob_id"],
        )
        for r in records
    ]
    try:
        with sqlite3.connect(str(db_path), timeout=30) as conn:
            conn.executemany(
                """
                UPDATE blobs SET
                    indexed=1, status='indexed',
                    is_nsfw=?, nsfw_score=?, cluster_id=?,
                    caption=?, last_updated_at=?
                WHERE blob_id=?
                """,
                rows,
            )
            conn.commit()
    except Exception as exc:
        print(f"[BlobCatalog] mark_indexed_batch error (non-fatal): {exc}")


def mark_failed_batch(
    blob_ids: List[str],
    *,
    status: str = "embed_failed",
    db_path: Path = CATALOG_DB_PATH,
) -> None:
    """Mark blobs as permanently failed for indexing after retry budget is exhausted."""
    if not blob_ids:
        return
    now = _now()
    rows = [(status, now, bid) for bid in blob_ids]
    try:
        with sqlite3.connect(str(db_path), timeout=30) as conn:
            conn.executemany(
                """
                UPDATE blobs SET
                    indexed=0,
                    status=?,
                    last_updated_at=?
                WHERE blob_id=?
                """,
                rows,
            )
            conn.commit()
    except Exception as exc:
        print(f"[BlobCatalog] mark_failed_batch error (non-fatal): {exc}")


def set_nsfw(blob_id: str, is_nsfw: bool, db_path: Path = CATALOG_DB_PATH) -> None:
    """Manually override NSFW flag for a blob."""
    try:
        with sqlite3.connect(str(db_path), timeout=10) as conn:
            conn.execute(
                "UPDATE blobs SET is_nsfw=?, last_updated_at=? WHERE blob_id=?",
                (1 if is_nsfw else 0, _now(), blob_id),
            )
            conn.commit()
    except Exception as exc:
        print(f"[BlobCatalog] set_nsfw error: {exc}")


def get_blob_metadata(
    blob_id: str, db_path: Path = CATALOG_DB_PATH
) -> Optional[Dict[str, Any]]:
    """Fetch blob metadata from the catalog database."""
    try:
        with sqlite3.connect(str(db_path), timeout=10) as conn:
            row = conn.execute(
                """
                SELECT mime_type, extension, kind, size, is_active
                FROM blobs WHERE blob_id=?
                """,
                (blob_id,),
            ).fetchone()
            if row:
                return {
                    "mime_type": row[0],
                    "extension": row[1],
                    "kind": row[2],
                    "size": row[3],
                    "is_active": bool(row[4]),
                }
    except Exception as exc:
        print(f"[BlobCatalog] get_blob_metadata error: {exc}")
    return None


def get_stats(db_path: Path = CATALOG_DB_PATH) -> Dict[str, Any]:
    """Get catalog statistics."""
    try:
        with sqlite3.connect(str(db_path), timeout=10) as conn:
            total = conn.execute("SELECT COUNT(*) FROM blobs").fetchone()[0]
            indexed = conn.execute(
                "SELECT COUNT(*) FROM blobs WHERE indexed=1"
            ).fetchone()[0]
            nsfw = conn.execute(
                "SELECT COUNT(*) FROM blobs WHERE is_nsfw=1"
            ).fetchone()[0]
            queue = conn.execute("SELECT COUNT(*) FROM index_queue").fetchone()[0]
            return {
                "total": total,
                "indexed": indexed,
                "nsfw": nsfw,
                "queue": queue,
            }
    except Exception as exc:
        print(f"[BlobCatalog] get_stats error: {exc}")
        return {}


# Alias for compatibility
get_catalog_stats = get_stats


def get_clusters(db_path: Path = CATALOG_DB_PATH) -> List[Dict[str, Any]]:
    """Get all clusters with their counts."""
    try:
        with sqlite3.connect(str(db_path), timeout=10) as conn:
            rows = conn.execute(
                """
                SELECT cluster_id, COUNT(*) as count
                FROM blobs
                WHERE indexed=1 AND cluster_id IS NOT NULL
                GROUP BY cluster_id
                """
            ).fetchall()
            return [{"cluster_id": r[0], "count": r[1]} for r in rows]
    except Exception as exc:
        print(f"[BlobCatalog] get_clusters error: {exc}")
        return []


def update_cluster(
    cluster_id: int,
    label: Optional[str] = None,
    size: Optional[int] = None,
    centroid_blob_id: Optional[str] = None,
    db_path: Path = CATALOG_DB_PATH,
) -> None:
    """Upsert cluster metadata into the ``clusters`` table."""
    try:
        with sqlite3.connect(str(db_path), timeout=10) as conn:
            conn.execute(
                """
                INSERT INTO clusters (cluster_id, label, size, centroid_blob_id, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(cluster_id) DO UPDATE SET
                    label = COALESCE(excluded.label, clusters.label),
                    size = COALESCE(excluded.size, clusters.size),
                    centroid_blob_id = COALESCE(excluded.centroid_blob_id, clusters.centroid_blob_id),
                    updated_at = excluded.updated_at
                """,
                (cluster_id, label, size, centroid_blob_id, _now()),
            )
            conn.commit()
    except Exception as exc:
        print(f"[BlobCatalog] update_cluster error: {exc}")


__all__ = [
    "CATALOG_DB_PATH",
    "init_catalog_db",
    "add_blob",
    "add_blobs",
    "get_blob_metadata",
    "get_stats",
    "get_catalog_stats",  # Alias
    "get_clusters",
    "dequeue_blobs",
    "mark_indexed",
    "mark_indexed_batch",
    "mark_failed_batch",
    "set_nsfw",
    "requeue_blobs",
    "update_cluster",
    "get_index_queue_stats",
    "get_cluster_blobs",
]


def get_cluster_blobs(
    cluster_id: int,
    limit: int = 100,
    offset: int = 0,
    exclude_nsfw: bool = True,
    db_path: Path = CATALOG_DB_PATH,
) -> List[Dict[str, Any]]:
    nsfw_clause = "AND is_nsfw=0" if exclude_nsfw else ""
    try:
        with sqlite3.connect(str(db_path), timeout=10) as conn:
            rows = conn.execute(
                f"""
                SELECT blob_id, mime_type, nsfw_score
                FROM blobs
                WHERE cluster_id=? AND indexed=1 {nsfw_clause}
                ORDER BY nsfw_score DESC
                LIMIT ? OFFSET ?
                """,
                (cluster_id, limit, offset),
            ).fetchall()
        return [{"blob_id": r[0], "mime_type": r[1], "nsfw_score": r[2]} for r in rows]
    except Exception:
        return []


__all__ = [
    "CATALOG_DB_PATH",
    "init_catalog_db",
    "build_blob_row",
    "upsert_blobs_batch",
    "enqueue_images",
    "dequeue_blobs",
    "requeue_blobs",
    "mark_indexed",
    "mark_indexed_batch",
    "set_nsfw",
    "get_catalog_stats",
    "update_cluster",
    "get_clusters",
    "get_cluster_blobs",
]
