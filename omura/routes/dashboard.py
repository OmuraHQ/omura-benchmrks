"""
Dashboard API — semantic clusters, NSFW tagging, catalog stats.

Endpoints
---------
GET  /dashboard/stats                  — blob catalog overview
GET  /dashboard/distribution           — file type & kind distribution
GET  /dashboard/clusters               — list semantic clusters (label, size, representative image)
GET  /dashboard/clusters/{cluster_id}  — blobs in a specific cluster (paginated)
GET  /dashboard/nsfw                   — NSFW-flagged images (paginated)
PATCH /dashboard/blobs/{blob_id}/nsfw  — manually set NSFW flag
POST /dashboard/clusters/rebuild       — trigger async cluster rebuild
"""

from __future__ import annotations

import threading
import json
import sqlite3
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from pydantic import BaseModel

from omura.utils.blob_catalog import (
    CATALOG_DB_PATH,
    get_catalog_stats,
    get_cluster_blobs,
    get_clusters,
    set_nsfw,
)

router = APIRouter(prefix="/dashboard", tags=["dashboard"])

# Shared reference to the vector store (set by api.py on startup)
_shared_vector_store = None
_cluster_rebuild_lock = threading.Lock()
_cluster_rebuild_running = False


def set_shared_vector_store(store) -> None:
    global _shared_vector_store
    _shared_vector_store = store


def _catalog_db_path():
    if _shared_vector_store is not None and hasattr(_shared_vector_store, "catalog_db_path"):
        return _shared_vector_store.catalog_db_path
    return CATALOG_DB_PATH


# ── Response models ────────────────────────────────────────────────────────────


class ClusterInfo(BaseModel):
    cluster_id: int
    label: Optional[str]
    size: int
    centroid_blob_id: Optional[str]
    updated_at: Optional[str]


class BlobRef(BaseModel):
    blob_id: str
    mime_type: Optional[str]
    nsfw_score: Optional[float]


class NsfwPatch(BaseModel):
    is_nsfw: bool


class BlobCounts(BaseModel):
    total_blobs: int
    active_blobs: int
    expired_blobs: int


# ── Endpoints ──────────────────────────────────────────────────────────────────


@router.get(
    "/stats",
    summary="Blob catalog overview",
    description=(
        "Returns total blobs traversed, active/expired counts, file type distribution "
        "(kind + MIME), status distribution, semantic cluster distribution, "
        "indexing coverage, queue size, NSFW count, and vector store stats."
    ),
)
async def stats() -> Dict[str, Any]:
    result = get_catalog_stats(_catalog_db_path())

    # Merge vector store stats if available
    if _shared_vector_store is not None:
        try:
            result["vector_store"] = {
                "total_embeddings": _shared_vector_store.size(),
                "index_built": _shared_vector_store.index is not None,
            }
        except Exception:
            pass

    return result


@router.get(
    "/distribution",
    summary="File type distribution",
    description="Kind and MIME type breakdown across all catalogued blobs.",
)
async def distribution() -> Dict[str, Any]:
    s = get_catalog_stats(_catalog_db_path())
    return {
        "total_blobs_traversed": s.get("total_blobs_traversed", 0),
        "active_blobs": s.get("active_blobs", 0),
        "kind_distribution": s.get("kind_distribution", {}),
        "mime_type_distribution": s.get("mime_type_distribution", {}),
        "status_distribution": s.get("status_distribution", {}),
    }


@router.get(
    "/blob-counts",
    summary="Blob lifecycle counts",
    description="Returns total blobs plus active and expired counts from the catalog.",
    response_model=BlobCounts,
)
async def blob_counts() -> Dict[str, int]:
    try:
        with sqlite3.connect(str(_catalog_db_path()), timeout=10) as conn:
            total, active = conn.execute(
                """
                SELECT
                    COUNT(*) AS total_blobs,
                    COALESCE(SUM(CASE WHEN is_active=1 THEN 1 ELSE 0 END), 0) AS active_blobs
                FROM blobs
                """
            ).fetchone()

        expired = max(0, int(total) - int(active))
        return {
            "total_blobs": int(total),
            "active_blobs": int(active),
            "expired_blobs": expired,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get(
    "/clusters",
    summary="Semantic cluster list",
    description=(
        "Returns all semantic clusters sorted by size descending. "
        "Each cluster has a label (assigned from image-text similarity against semantic categories), "
        "size (number of indexed images), and a representative blob_id."
    ),
    response_model=List[ClusterInfo],
)
async def clusters() -> List[Dict[str, Any]]:
    return get_clusters(_catalog_db_path())


@router.get(
    "/clusters/{cluster_id}",
    summary="Blobs in a cluster",
    description="Returns paginated list of image blob_ids in a semantic cluster.",
    response_model=List[BlobRef],
)
async def cluster_blobs(
    cluster_id: int,
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    exclude_nsfw: bool = Query(default=True),
) -> List[Dict[str, Any]]:
    return get_cluster_blobs(
        cluster_id=cluster_id,
        limit=limit,
        offset=offset,
        exclude_nsfw=exclude_nsfw,
        db_path=_catalog_db_path(),
    )


@router.get(
    "/nsfw",
    summary="NSFW-flagged images",
    description="Returns paginated list of images flagged as NSFW (is_nsfw=1).",
    response_model=List[BlobRef],
)
async def nsfw_blobs(
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> List[Dict[str, Any]]:
    try:
        with sqlite3.connect(str(_catalog_db_path()), timeout=10) as conn:
            rows = conn.execute(
                """
                SELECT blob_id, mime_type, nsfw_score
                FROM blobs
                WHERE is_nsfw=1 AND indexed=1
                ORDER BY nsfw_score DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            ).fetchall()
        return [{"blob_id": r[0], "mime_type": r[1], "nsfw_score": r[2]} for r in rows]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.patch(
    "/blobs/{blob_id}/nsfw",
    summary="Set NSFW flag",
    description="Manually override the NSFW flag for a blob. Persists to blob_catalog.sqlite.",
)
async def patch_nsfw(blob_id: str, body: NsfwPatch) -> Dict[str, Any]:
    set_nsfw(blob_id, body.is_nsfw, db_path=_catalog_db_path())
    # Keep in-memory metadata consistent with SQLite for immediate search behavior.
    if _shared_vector_store is not None:
        raw = _shared_vector_store.metadata.get(blob_id)
        if raw is not None:
            try:
                meta = json.loads(raw) if isinstance(raw, str) else dict(raw)
                meta["is_nsfw"] = body.is_nsfw
                _shared_vector_store.metadata[blob_id] = json.dumps(meta)
            except Exception:
                pass
    return {"blob_id": blob_id, "is_nsfw": body.is_nsfw, "ok": True}


@router.post(
    "/clusters/rebuild",
    summary="Trigger cluster rebuild",
    description=(
        "Starts an asynchronous k-means cluster rebuild over all indexed embeddings. "
        "Only one rebuild runs at a time. Returns immediately."
    ),
)
async def rebuild_clusters(background_tasks: BackgroundTasks) -> Dict[str, Any]:
    global _cluster_rebuild_running

    if _cluster_rebuild_running:
        return {"status": "already_running", "ok": False}

    if _shared_vector_store is None:
        raise HTTPException(status_code=503, detail="Vector store not initialized")

    background_tasks.add_task(_do_rebuild)
    return {"status": "started", "ok": True}


def _do_rebuild() -> None:
    global _cluster_rebuild_running
    if not _cluster_rebuild_lock.acquire(blocking=False):
        return
    _cluster_rebuild_running = True
    try:
        from omura.indexers.vector_indexer import _rebuild_clusters
        _rebuild_clusters(_shared_vector_store, _catalog_db_path())
    except Exception as exc:
        print(f"[Dashboard] Cluster rebuild error: {exc}")
    finally:
        _cluster_rebuild_running = False
        _cluster_rebuild_lock.release()
