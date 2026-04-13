"""Search API endpoints for vector similarity search."""

from __future__ import annotations

import os
import threading
import sqlite3
import json
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from omura.utils.blockberry import MANUAL_EPOCH, get_current_epoch
from omura.utils.imagebind_embeddings import (
    get_nsfw_embeddings,
    generate_image_embedding,
    generate_text_embedding,
    is_nsfw_from_tag_score,
    nsfw_similarity_score_0_100,
    nsfw_tag_score_min,
)
from omura.utils.blob_catalog import CATALOG_DB_PATH
from omura.utils.vector_store import VectorStore

CLASSIFY_HIGH_SCORE = float(os.getenv("OMURA_CLASSIFY_HIGH_SCORE", "0.62"))
CLASSIFY_LOW_SCORE = float(os.getenv("OMURA_CLASSIFY_LOW_SCORE", "0.56"))
CLASSIFY_MARGIN_HIGH = float(os.getenv("OMURA_CLASSIFY_MARGIN_HIGH", "0.02"))
CLASSIFY_MARGIN_LOW = float(os.getenv("OMURA_CLASSIFY_MARGIN_LOW", "0.008"))
NSFW_SEMANTIC_SCORE_THRESHOLD = float(
    os.getenv("OMURA_NSFW_SEMANTIC_SCORE_THRESHOLD", "0.62")
)


def _normalize_kind(kind: Optional[str]) -> Optional[str]:
    """Normalize the kind field to a canonical form."""
    if not kind:
        return None
    k = kind.lower().strip()
    if k in ("image", "video", "audio", "doc", "quilt"):
        return k
    return None


def _sql_dashboard_counts(catalog_db_path: str, current_epoch: int) -> Dict[str, Any]:
    """Query blob_catalog.sqlite for dashboard counters."""
    conn = sqlite3.connect(catalog_db_path)
    cur = conn.cursor()

    # Total blobs (all time)
    cur.execute("SELECT COUNT(*) FROM blobs")
    total_blobs = cur.fetchone()[0]

    # Active blobs (end_epoch > current_epoch or end_epoch is NULL)
    cur.execute(
        """
        SELECT COUNT(*) FROM blobs
        WHERE end_epoch IS NULL OR CAST(end_epoch AS INTEGER) > ?
        """,
        (current_epoch,),
    )
    active_blobs = cur.fetchone()[0]

    # Modality counts (all time)
    cur.execute(
        """
        SELECT COALESCE(kind, 'unknown'), COUNT(*) FROM blobs GROUP BY kind
        """
    )
    counts_all = {row[0]: row[1] for row in cur.fetchall()}

    # Modality counts (active)
    cur.execute(
        """
        SELECT COALESCE(kind, 'unknown'), COUNT(*) FROM blobs
        WHERE end_epoch IS NULL OR CAST(end_epoch AS INTEGER) > ?
        GROUP BY kind
        """,
        (current_epoch,),
    )
    counts_active = {row[0]: row[1] for row in cur.fetchall()}

    conn.close()

    return {
        "total_blobs": total_blobs,
        "active_blobs": active_blobs,
        "counts_all": counts_all,
        "counts_active": counts_active,
    }


def _sql_indexed_counts(catalog_db_path: str, current_epoch: int) -> Dict[str, Any]:
    """Indexed counters from blob_catalog.sqlite for API/worker consistency."""
    conn = sqlite3.connect(catalog_db_path)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM blobs WHERE indexed=1")
    total_indexed = int(cur.fetchone()[0])
    cur.execute(
        """
        SELECT COUNT(*) FROM blobs
        WHERE indexed=1 AND (end_epoch IS NULL OR CAST(end_epoch AS INTEGER) > ?)
        """,
        (current_epoch,),
    )
    active_indexed = int(cur.fetchone()[0])
    cur.execute(
        """
        SELECT COALESCE(kind, 'other'), COUNT(*)
        FROM blobs
        WHERE indexed=1
        GROUP BY kind
        """
    )
    by_kind = {str(row[0]): int(row[1]) for row in cur.fetchall()}
    conn.close()
    return {
        "total_indexed": total_indexed,
        "active_indexed": active_indexed,
        "by_kind": by_kind,
    }


class SearchResponse(BaseModel):
    results: List[Dict[str, Any]]
    total: int


class SearchRequest(BaseModel):
    query: Optional[str] = None
    instruction: Optional[str] = None
    top_k: int = 10
    exclude_nsfw: bool = True


class MediaBlobCounterResponse(BaseModel):
    total_blobs: int
    active_blobs: int
    identified_image: int
    identified_video: int
    identified_audio: int
    modality_counts_all: Dict[str, int]
    modality_counts_active: Dict[str, int]


class RebuildNsfwResponse(BaseModel):
    scanned_images: int
    flagged_nsfw: int
    updated_rows: int
    threshold: float


class VectorStoreStats(BaseModel):
    total_embeddings: int
    index_built: bool
    status: str
    by_modality: Dict[str, int]


class ClassifierDashboardResponse(BaseModel):
    """Dashboard response for classifier bucket counts."""

    categories: Dict[str, int] = Field(
        default_factory=dict,
        description="Count per semantic category",
        examples=[
            {
                "meme": 120,
                "cat": 100,
                "dog": 88,
                "human": 900,
                "scenery_landscape": 1432,
            }
        ],
    )
    nsfw_count: int = Field(
        default=0, description="Total number of items flagged as NSFW"
    )


router = APIRouter(
    prefix="/search",
    tags=["search"],
    responses={
        503: {"description": "Service Unavailable - Vector store not initialized"},
        500: {"description": "Internal Server Error"},
    },
)

# Global shared vector store instance
_shared_vector_store: Optional[VectorStore] = None
_vector_store_lock = threading.Lock()
_shared_embeddings_mtime_ns: Optional[int] = None
_atlas_anchor_cache: Optional[tuple[str, list[str], np.ndarray]] = None
_projection_counts_cache: Optional[tuple[str, Dict[str, int], int]] = None


def _load_dashboard_projection_counts() -> Optional[tuple[Dict[str, int], int]]:
    """Load category_counts + total_images from a precomputed atlas projection JSON.

    When ``data/atlas/omura_emmbed_11_categories_projection.json`` (or
    ``OMURA_DASHBOARD_PROJECTION_PATH``) exists, dashboard classifier buckets match
    the offline nearest-anchor assignment without re-embedding.
    """
    global _projection_counts_cache
    raw = os.getenv(
        "OMURA_DASHBOARD_PROJECTION_PATH",
        "data/atlas/omura_emmbed_11_categories_projection.json",
    )
    if not (raw or "").strip():
        return None
    path = Path(raw.strip())
    if not path.exists():
        return None
    try:
        mtime = str(path.stat().st_mtime_ns)
        if (
            _projection_counts_cache is not None
            and _projection_counts_cache[0] == mtime
        ):
            d, t = _projection_counts_cache[1], _projection_counts_cache[2]
            return dict(d), int(t)
        obj = json.loads(path.read_text(encoding="utf-8"))
        summary = obj.get("summary") or {}
        cc = summary.get("category_counts") or {}
        if not cc:
            return None
        out = {str(k): int(v) for k, v in cc.items()}
        total = int(summary.get("total_images") or sum(out.values()))
        _projection_counts_cache = (mtime, out, total)
        return out, total
    except Exception as e:
        print(f"[Search] dashboard projection load failed: {e}")
        return None


def set_shared_vector_store(store: VectorStore) -> None:
    """Set the shared vector store instance (called from API startup)."""
    global _shared_vector_store, _shared_embeddings_mtime_ns
    with _vector_store_lock:
        _shared_vector_store = store
        embeddings_path = store.index_path.parent / "embeddings.npy"
        try:
            _shared_embeddings_mtime_ns = embeddings_path.stat().st_mtime_ns
        except Exception:
            _shared_embeddings_mtime_ns = None
        print(f"[Search] Shared vector store set with {len(store.metadata)} items")


def get_vector_store() -> VectorStore:
    """Get the shared vector store instance.

    The vector store is initialized at startup and kept in memory.
    It's shared between the API and indexer, so new embeddings are
    immediately visible without reloading.

    Returns:
        VectorStore instance
    """
    global _shared_vector_store, _shared_embeddings_mtime_ns

    with _vector_store_lock:
        if _shared_vector_store is None:
            # Lazy init fallback: keeps search endpoints functional even if startup wiring
            # hasn't run yet in this worker.
            try:
                store = VectorStore()
                store.load()
                _shared_vector_store = store
                embeddings_path = store.index_path.parent / "embeddings.npy"
                try:
                    _shared_embeddings_mtime_ns = embeddings_path.stat().st_mtime_ns
                except Exception:
                    _shared_embeddings_mtime_ns = None
                print(
                    f"[Search] Lazy-initialized vector store with {len(store.metadata)} items"
                )
            except Exception:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Vector store not initialized. The service may still be starting up.",
                )
        else:
            # Heal stale worker state: metadata may be present while vectors weren't loaded.
            # In that case searches return empty because embeddings_dict is empty.
            try:
                if (
                    len(_shared_vector_store.embeddings_dict) == 0
                    and _shared_vector_store.index_path.exists()
                ):
                    _shared_vector_store.load()
                    embeddings_path = _shared_vector_store.index_path.parent / "embeddings.npy"
                    try:
                        _shared_embeddings_mtime_ns = embeddings_path.stat().st_mtime_ns
                    except Exception:
                        _shared_embeddings_mtime_ns = None
                    print(
                        "[Search] Reloaded vector store in worker "
                        f"(embeddings={len(_shared_vector_store.embeddings_dict)})"
                    )
            except Exception:
                pass

            # If embeddings changed on disk (e.g. worker reindexed), reload in this API worker.
            try:
                embeddings_path = _shared_vector_store.index_path.parent / "embeddings.npy"
                current_mtime_ns = (
                    embeddings_path.stat().st_mtime_ns if embeddings_path.exists() else None
                )
                if (
                    current_mtime_ns is not None
                    and _shared_embeddings_mtime_ns is not None
                    and current_mtime_ns != _shared_embeddings_mtime_ns
                ):
                    _shared_vector_store.load()
                    _shared_embeddings_mtime_ns = current_mtime_ns
                    print(
                        "[Search] Detected updated embeddings on disk; reloaded store "
                        f"(embeddings={len(_shared_vector_store.embeddings_dict)})"
                    )
                elif current_mtime_ns is not None and _shared_embeddings_mtime_ns is None:
                    _shared_embeddings_mtime_ns = current_mtime_ns
            except Exception:
                pass
        return _shared_vector_store


# Max bytes for reverse-image upload (default 25 MiB)
_REVERSE_IMAGE_MAX_BYTES = int(
    os.getenv("OMURA_REVERSE_IMAGE_MAX_BYTES", str(25 * 1024 * 1024))
)


def _similar_images_from_embedding(
    store: VectorStore,
    qemb: np.ndarray,
    top_k: int,
    exclude_blob_id: Optional[str] = None,
    exclude_nsfw: bool = True,
) -> SearchResponse:
    """Run cosine retrieval using vector-store index (legacy-compatible path)."""
    qvec = np.asarray(qemb, dtype=np.float32).flatten()
    qnorm = float(np.linalg.norm(qvec))
    if qnorm <= 0:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Query embedding is invalid (zero norm); model may still be warming up.",
        )
    qvec = qvec / qnorm

    # Over-fetch then apply exclusions (nsfw/blob id) while preserving rank quality.
    overfetch_k = max(int(top_k) * 20, int(top_k))
    raw_results = store.search(qvec, top_k=overfetch_k, kind_filter="image")
    nsfw_vecs = get_nsfw_embeddings() or []

    results = []
    for meta, sim in raw_results:
        if not isinstance(meta, dict):
            continue
        blob_id = meta.get("blob_id")
        if not blob_id:
            continue
        if exclude_blob_id and blob_id == exclude_blob_id:
            continue

        # Result-time NSFW fallback: same 0–100 tag score as indexing (score > min, default 85).
        is_nsfw = bool(meta.get("is_nsfw", False))
        nsfw_tag_score = None
        if nsfw_vecs:
            emb = store.get_embedding(blob_id)
            if emb is not None:
                nsfw_tag_score = float(nsfw_similarity_score_0_100(emb, nsfw_vecs))
                if not is_nsfw and is_nsfw_from_tag_score(nsfw_tag_score):
                    is_nsfw = True

        if exclude_nsfw and is_nsfw:
            continue

        # Match legacy score: max(0, sim) * 1000, capped at 100 (0-100 scale).
        score = float(min(max(float(sim), 0.0) * 1000.0, 100.0))
        # Keep a distance-like field for compatibility (cosine distance proxy).
        dist = float(1.0 - float(sim))
        out = {"blob_id": blob_id, "score": score, "distance": dist, **meta}
        out["is_nsfw"] = is_nsfw
        if nsfw_tag_score is not None:
            out["nsfw_tag_score"] = nsfw_tag_score
        if is_nsfw and not out.get("categories"):
            out["categories"] = ["nsfw", "pornographic"]
            out["category_confidence"] = out.get("category_confidence") or "low"
        results.append(out)
        if len(results) >= int(top_k):
            break

    return SearchResponse(results=results, total=len(results))


@router.post(
    "",
    response_model=SearchResponse,
    summary="Search by text",
    description=(
        "Text-to-image search endpoint (JSON only).\n\n"
        "Request body example:\n"
        "`{ \"query\": \"cat\", \"top_k\": 10, \"exclude_nsfw\": true }`\n\n"
        "Use `/search/reverse-image` for image upload search."
    ),
    responses={
        200: {
            "description": "Search completed",
            "content": {
                "application/json": {
                    "examples": {
                        "text_search": {
                            "summary": "Text query result",
                            "value": {
                                "results": [
                                    {
                                        "blob_id": "example_blob_id",
                                        "score": 0.82,
                                        "mime_type": "image/png",
                                        "kind": "image",
                                        "is_nsfw": False,
                                    }
                                ],
                                "total": 1234,
                            },
                        }
                    }
                }
            },
        },
        400: {"description": "Invalid request payload"},
        503: {"description": "Vector store or embedding model not ready"},
    },
)
@router.post(
    "/",
    response_model=SearchResponse,
    include_in_schema=False,
)
async def search(
    payload: SearchRequest,
):
    """Search by text query (JSON body)."""
    store = get_vector_store()
    # Legacy-compatible normalization: trim + lowercase before embedding.
    q = (payload.query or "").strip().lower()
    if not q:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Field 'query' is required.",
        )

    qemb = generate_text_embedding(
        q,
        is_document=False,
        instruction=(payload.instruction or None),
    )
    if qemb is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Embedding model not ready for text search yet.",
        )

    return _similar_images_from_embedding(
        store,
        qemb,
        top_k=int(payload.top_k),
        exclude_nsfw=bool(payload.exclude_nsfw),
    )


@router.get(
    "/dashboard/media-counters",
    response_model=MediaBlobCounterResponse,
    summary="Media and blob counters",
    description=(
        "Returns total/active blob counts and modality distributions.\n"
        "Reads from `blob_catalog.sqlite` and falls back gracefully if in-memory state is not ready."
    ),
)
async def media_counters_dashboard() -> MediaBlobCounterResponse:
    store: Optional[VectorStore] = None
    try:
        store = get_vector_store()
    except HTTPException:
        store = None
    catalog_db_path = getattr(store, "catalog_db_path", CATALOG_DB_PATH)

    try:
        epoch = get_current_epoch(silent=True)
        current_epoch = MANUAL_EPOCH if epoch is None else int(epoch)
    except Exception:
        current_epoch = MANUAL_EPOCH

    if catalog_db_path.exists():
        try:
            stats = _sql_dashboard_counts(str(catalog_db_path), current_epoch)
            active = stats["counts_active"]  # type: ignore[index]
            return MediaBlobCounterResponse(
                total_blobs=int(stats["total_blobs"]),  # type: ignore[arg-type]
                active_blobs=int(stats["active_blobs"]),  # type: ignore[arg-type]
                identified_image=int(active.get("image", 0)),
                identified_video=int(active.get("video", 0)),
                identified_audio=int(active.get("audio", 0)),
                modality_counts_all=stats["counts_all"],  # type: ignore[arg-type]
                modality_counts_active=active,  # type: ignore[arg-type]
            )
        except Exception as e:
            print(
                f"[Search] SQL media counters failed, falling back to vector metadata: {e}"
            )

    # Fallback if catalog DB is unavailable.
    by_modality = store.counts_by_kind() if store is not None else {}
    return MediaBlobCounterResponse(
        total_blobs=store.size() if store is not None else 0,
        active_blobs=store.size() if store is not None else 0,
        identified_image=by_modality.get("image", 0),
        identified_video=by_modality.get("video", 0),
        identified_audio=by_modality.get("audio", 0),
        modality_counts_all=by_modality,
        modality_counts_active=by_modality,
    )


@router.get(
    "/dashboard/classifier-counts",
    response_model=ClassifierDashboardResponse,
    summary="Classifier category counts",
    description=(
        "Returns NSFW count and category counters.\n"
        "Prefers precomputed atlas projection summary (``OMURA_DASHBOARD_PROJECTION_PATH``) "
        "when present; otherwise nearest-anchor counts from live embeddings; "
        "else semantic metadata or indexed modality buckets."
    ),
)
async def classifier_counts_dashboard() -> ClassifierDashboardResponse:
    # DB is authoritative for dashboard counters (works even on API-only nodes).
    db_nsfw_count = 0
    try:
        with sqlite3.connect(str(CATALOG_DB_PATH), timeout=10) as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM blobs WHERE indexed=1 AND is_nsfw=1"
            ).fetchone()
            db_nsfw_count = int(row[0] if row else 0)
    except Exception:
        db_nsfw_count = 0

    # Keep in-memory count as secondary signal and use the larger value to avoid stale-underflow.
    mem_nsfw_count = 0
    semantic_counts: Dict[str, int] = {}
    store: Optional[VectorStore] = None
    try:
        store = get_vector_store()
        for _, raw in store.metadata.items():
            try:
                meta = json.loads(raw) if isinstance(raw, str) else dict(raw)
                if meta.get("is_nsfw"):
                    mem_nsfw_count += 1
                # Aggregate semantic categories if present.
                for cat in (meta.get("categories") or []):
                    if isinstance(cat, str) and cat.strip():
                        semantic_counts[cat] = semantic_counts.get(cat, 0) + 1
            except Exception:
                continue
    except HTTPException:
        pass

    # Preferred path: precomputed atlas projection (stable, matches offline analysis).
    computed_counts: Dict[str, int] = {}
    projection_total_images: Optional[int] = None
    proj = _load_dashboard_projection_counts()
    if proj is not None:
        computed_counts, projection_total_images = proj
        mem_nsfw_count = max(mem_nsfw_count, int(computed_counts.get("nsfw", 0)))

    # Else: compute dashboard buckets from current embeddings + atlas labels.
    if not computed_counts and store is not None and len(store.embeddings_dict) > 0:
        try:
            from omura.utils.imagebind_embeddings import generate_text_embedding

            atlas_path = Path(
                os.getenv(
                    "OMURA_DASHBOARD_ATLAS_PATH",
                    "data/atlas/omura_emmbed_11_categories_atlas.json",
                )
            )
            if not atlas_path.exists():
                raise RuntimeError(f"Atlas file not found: {atlas_path}")

            global _atlas_anchor_cache
            atlas_texts: list[str] = []
            atlas_embs: list[np.ndarray] = []
            atlas_mtime = str(atlas_path.stat().st_mtime_ns)
            if (
                _atlas_anchor_cache is not None
                and _atlas_anchor_cache[0] == atlas_mtime
            ):
                atlas_texts = list(_atlas_anchor_cache[1])
                C = np.asarray(_atlas_anchor_cache[2], dtype=np.float32)
            else:
                atlas_obj = json.loads(atlas_path.read_text(encoding="utf-8"))
                points = atlas_obj.get("points", [])
                for p in points:
                    txt = str(p.get("text") or "").strip()
                    if txt:
                        atlas_texts.append(txt)
                # Keep unique order
                seen = set()
                atlas_texts = [t for t in atlas_texts if not (t in seen or seen.add(t))]
                if not atlas_texts:
                    raise RuntimeError("Atlas points missing `text` labels.")

                for label in atlas_texts:
                    emb = generate_text_embedding(label, is_document=False)
                    if emb is None:
                        continue
                    v = np.asarray(emb, dtype=np.float32).flatten()
                    n = np.linalg.norm(v)
                    if n > 0:
                        v = v / n
                    atlas_embs.append(v)
                if len(atlas_embs) != len(atlas_texts):
                    raise RuntimeError("Failed to embed all atlas labels.")
                C = np.stack(atlas_embs, axis=0).astype(np.float32)
                _atlas_anchor_cache = (atlas_mtime, list(atlas_texts), C.copy())

            if len(atlas_texts) == int(C.shape[0]) and int(C.shape[0]) > 0:
                image_ids: List[str] = []
                image_vecs: List[np.ndarray] = []
                for blob_id, emb in store.embeddings_dict.items():
                    meta = store.get_blob_metadata(blob_id) or {}
                    if _normalize_kind(meta.get("kind")) != "image":
                        continue
                    vec = np.asarray(emb, dtype=np.float32).flatten()
                    n = np.linalg.norm(vec)
                    if n > 0:
                        vec = vec / n
                    image_ids.append(blob_id)
                    image_vecs.append(vec)

                if image_vecs:
                    X = np.stack(image_vecs, axis=0)
                    D = np.linalg.norm(X[:, None, :] - C[None, :, :], axis=2)

                    best_idx = D.argmin(axis=1)
                    # Projection-accurate category counts:
                    # one nearest class per image, no extra NSFW gate.
                    for cat in atlas_texts:
                        computed_counts[cat] = 0
                    for i in range(len(X)):
                        cat = atlas_texts[int(best_idx[i])]
                        computed_counts[cat] += 1
                    mem_nsfw_count = max(mem_nsfw_count, int(computed_counts.get("nsfw", 0)))
        except Exception as e:
            print(f"[Search] classifier dashboard compute fallback: {e}")

    # Fallback category buckets from DB indexed modality counts if semantic labels are absent.
    fallback_counts: Dict[str, int] = {}
    try:
        epoch = get_current_epoch(silent=True)
        current_epoch = MANUAL_EPOCH if epoch is None else int(epoch)
    except Exception:
        current_epoch = MANUAL_EPOCH
    try:
        if CATALOG_DB_PATH.exists():
            db_idx = _sql_indexed_counts(str(CATALOG_DB_PATH), current_epoch)
            by_kind = db_idx.get("by_kind", {})
            fallback_counts = {str(k): int(v) for k, v in by_kind.items()}
    except Exception:
        fallback_counts = {}

    # Dashboard contract: expose only requested buckets + "other".
    requested_buckets = ["nsfw", "screen_ui", "building", "art", "animal", "food"]

    if computed_counts:
        categories_out = {k: int(computed_counts.get(k, 0)) for k in requested_buckets}
        if projection_total_images is not None:
            used = sum(int(v) for v in categories_out.values())
            categories_out["other"] = max(0, int(projection_total_images - used))
        else:
            # Live NN path: approximate total from embedding rows (includes non-image rows).
            total_images = 0
            try:
                for _, emb in store.embeddings_dict.items() if store is not None else []:
                    total_images += 1
            except Exception:
                total_images = sum(int(v) for v in categories_out.values())
            used = sum(int(v) for v in categories_out.values())
            categories_out["other"] = max(0, int(total_images - used))
    elif semantic_counts:
        categories_out = {k: int(semantic_counts.get(k, 0)) for k in requested_buckets}
        used = sum(int(v) for v in categories_out.values())
        # If semantic metadata exists but does not fully cover images, keep residual in "other".
        total_images = int(fallback_counts.get("image", used))
        categories_out["other"] = max(0, total_images - used)
    else:
        # Fallback with no semantic metadata: map by modality and keep rest in "other".
        categories_out = {k: 0 for k in requested_buckets}
        total_images = int(fallback_counts.get("image", 0))
        categories_out["other"] = total_images

    return ClassifierDashboardResponse(
        categories=categories_out,
        nsfw_count=max(db_nsfw_count, mem_nsfw_count),
    )


@router.get(
    "/beta/classifier-counts",
    response_model=ClassifierDashboardResponse,
    summary="Beta: Semantic category counts",
    description="Returns counts for semantic categories (cat, dog, etc.) using zero-shot similarity matching.",
)
async def beta_classifier_counts_dashboard(
    category_threshold: float = Query(default=0.25, ge=0.0, le=1.0),
    use_metadata: bool = Query(
        default=True, description="Use stored metadata categories if available"
    ),
) -> ClassifierDashboardResponse:
    """Beta endpoint for semantic category counts."""
    from omura.utils.imagebind_embeddings import get_semantic_category_embeddings

    store = get_vector_store()

    # Define popular categories for the beta version
    categories = [
        "cat",
        "dog",
        "animal",
        "pet",
        "wildlife",
        "meme",
        "scenery",
        "human",
        "car",
        "food",
        "nsfw",
        "pornographic",
    ]

    nsfw_count = 0

    # If using metadata, just aggregate from stored categories
    if use_metadata:
        counts = {cat: 0 for cat in categories}
        for blob_id, raw in store.metadata.items():
            try:
                meta = json.loads(raw) if isinstance(raw, str) else dict(raw)
                if meta.get("kind") != "image":
                    continue
                # Count NSFW
                if meta.get("is_nsfw"):
                    nsfw_count += 1
                # Count categories from metadata
                cat_list = meta.get("categories", [])
                for cat in cat_list:
                    if cat in counts:
                        counts[cat] += 1
            except Exception:
                continue
        return ClassifierDashboardResponse(categories=counts, nsfw_count=nsfw_count)

    # Otherwise, compute on-the-fly using zero-shot classification
    category_embs = await run_in_threadpool(
        get_semantic_category_embeddings, categories
    )

    if not category_embs:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate semantic category embeddings.",
        )

    # Prepare category counts
    counts = {cat: 0 for cat in categories}
    counts["uncertain"] = 0
    for c in categories:
        counts[f"{c}:high"] = 0
        counts[f"{c}:low"] = 0

    # We also need the NSFW embeddings for the nsfw_count part of the response
    nsfw_vecs = get_nsfw_embeddings()
    if nsfw_vecs is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="NSFW prototypes unavailable.",
        )

    nsfw_label_idxs = [
        i for i, c in enumerate(categories) if c in ("nsfw", "pornographic")
    ]
    normal_label_idxs = [i for i, c in enumerate(categories) if i not in nsfw_label_idxs]

    # Iterate through all blobs to classify
    for blob_id, raw in store.metadata.items():
        try:
            meta = json.loads(raw) if isinstance(raw, str) else dict(raw)

            # Only process images for semantic classification
            if meta.get("kind") != "image":
                continue

            emb = store.get_embedding(blob_id)
            if emb is None:
                continue

            vec = np.asarray(emb, dtype=np.float32).flatten()
            n = np.linalg.norm(vec)
            if n > 0:
                vec = vec / n

            # 1) Hybrid NSFW: 0–100 prototype tag score (>85 by default) + semantic anchors
            tag_s = float(nsfw_similarity_score_0_100(vec, nsfw_vecs))
            tag_hit = bool(is_nsfw_from_tag_score(tag_s))

            all_scores = [float(np.dot(vec, cat_emb)) for cat_emb in category_embs]
            nsfw_sem_score = (
                max(all_scores[i] for i in nsfw_label_idxs) if nsfw_label_idxs else -1.0
            )
            sem_hit = bool(nsfw_sem_score >= NSFW_SEMANTIC_SCORE_THRESHOLD)
            is_nsfw = bool(tag_hit or sem_hit)
            if is_nsfw:
                nsfw_count += 1
                counts["nsfw"] += 1
                if tag_hit:
                    counts["nsfw:high"] += 1
                else:
                    counts["nsfw:low"] += 1

            # 2) Semantic category with confidence tiers (excluding NSFW labels)
            if not normal_label_idxs:
                counts["uncertain"] += 1
                continue

            ranked = sorted(
                ((i, all_scores[i]) for i in normal_label_idxs),
                key=lambda t: t[1],
                reverse=True,
            )
            best_i, best_s = ranked[0]
            second_s = ranked[1][1] if len(ranked) > 1 else -1.0
            margin = best_s - second_s
            best_label = categories[best_i]

            if best_s >= max(category_threshold, CLASSIFY_HIGH_SCORE) and margin >= CLASSIFY_MARGIN_HIGH:
                counts[best_label] += 1
                counts[f"{best_label}:high"] += 1
            elif best_s >= max(category_threshold, CLASSIFY_LOW_SCORE) and margin >= CLASSIFY_MARGIN_LOW:
                counts[best_label] += 1
                counts[f"{best_label}:low"] += 1
            else:
                counts["uncertain"] += 1

        except Exception:
            continue

    return ClassifierDashboardResponse(categories=counts, nsfw_count=nsfw_count)


@router.post(
    "/beta/rebuild-semantic-classifier",
    response_model=RebuildNsfwResponse,
    summary="Beta: Rebuild semantic classifier",
    description="Re-scores all indexed images against semantic prototypes and updates metadata.",
)
async def rebuild_semantic_classifier(
    threshold: float = Form(default=0.35, ge=-1.0, le=1.0),
) -> RebuildNsfwResponse:
    """Beta endpoint to rebuild semantic categories in metadata."""
    from omura.utils.imagebind_embeddings import get_semantic_category_embeddings

    store = get_vector_store()
    category_names = [
        "cat",
        "dog",
        "animal",
        "pet",
        "wildlife",
        "meme",
        "scenery",
        "human",
        "car",
        "food",
        "nsfw",
        "pornographic",
    ]
    cat_embs = await run_in_threadpool(get_semantic_category_embeddings, category_names)

    if not cat_embs:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Semantic prototypes unavailable.",
        )

    scanned = 0
    updated = 0
    nsfw_vecs = get_nsfw_embeddings() or []
    nsfw_label_idxs = [
        i for i, c in enumerate(category_names) if c in ("nsfw", "pornographic")
    ]
    normal_label_idxs = [i for i in range(len(category_names)) if i not in nsfw_label_idxs]

    for blob_id, raw in list(store.metadata.items()):
        try:
            meta = json.loads(raw) if isinstance(raw, str) else dict(raw)
        except Exception:
            continue

        if meta.get("kind") != "image":
            continue

        emb = store.get_embedding(blob_id)
        if emb is None:
            continue

        vec = np.asarray(emb, dtype=np.float32).flatten()
        n = np.linalg.norm(vec)
        if n > 0:
            vec = vec / n

        # Perform classification with confidence tiers and hybrid NSFW.
        all_scores = [float(np.dot(vec, cat_emb)) for cat_emb in cat_embs]
        nsfw_sem_score = (
            max(all_scores[i] for i in nsfw_label_idxs) if nsfw_label_idxs else -1.0
        )
        tag_s = float(nsfw_similarity_score_0_100(vec, nsfw_vecs))
        is_nsfw = bool(
            is_nsfw_from_tag_score(tag_s)
            or nsfw_sem_score >= NSFW_SEMANTIC_SCORE_THRESHOLD
        )

        matched_categories = []
        confidence = "uncertain"
        margin = 0.0
        if normal_label_idxs:
            ranked = sorted(
                ((i, all_scores[i]) for i in normal_label_idxs),
                key=lambda t: t[1],
                reverse=True,
            )
            best_i, best_s = ranked[0]
            second_s = ranked[1][1] if len(ranked) > 1 else -1.0
            margin = float(best_s - second_s)
            best_label = category_names[best_i]

            if best_s >= max(threshold, CLASSIFY_HIGH_SCORE) and margin >= CLASSIFY_MARGIN_HIGH:
                matched_categories.append(best_label)
                confidence = "high"
            elif best_s >= max(threshold, CLASSIFY_LOW_SCORE) and margin >= CLASSIFY_MARGIN_LOW:
                matched_categories.append(best_label)
                confidence = "low"

        if is_nsfw:
            matched_categories.extend(["nsfw", "pornographic"])
            if confidence == "uncertain":
                confidence = "low"

        # Check if metadata actually needs updating
        old_categories = meta.get("categories", [])
        old_conf = str(meta.get("category_confidence", ""))
        old_margin = float(meta.get("category_margin", -1.0))
        old_nsfw = bool(meta.get("is_nsfw", False))
        new_categories = sorted(set(matched_categories))
        new_nsfw = bool(is_nsfw)
        if (
            new_categories != sorted(old_categories)
            or old_conf != confidence
            or abs(old_margin - margin) > 1e-6
            or old_nsfw != new_nsfw
        ):
            meta["categories"] = new_categories
            meta["category_confidence"] = confidence
            meta["category_margin"] = margin
            meta["is_nsfw"] = new_nsfw
            store.metadata[blob_id] = json.dumps(meta)
            updated += 1

        scanned += 1

    if updated > 0:
        store.save(create_backup=False)

    return RebuildNsfwResponse(
        scanned_images=scanned,
        flagged_nsfw=0,
        updated_rows=updated,
        threshold=threshold,
    )


@router.post(
    "/admin/rebuild-nsfw",
    response_model=RebuildNsfwResponse,
    summary="Rebuild NSFW flags for indexed images",
    description=(
        "Re-scores all indexed image embeddings against NSFW text prototypes and updates "
        "stored metadata `is_nsfw` in-place."
    ),
)
async def rebuild_nsfw_classifier(
    score_min: Optional[float] = Form(
        default=None,
        ge=0.0,
        le=100.0,
        description="Exclusive minimum 0–100 NSFW tag score (default: OMURA_NSFW_TAG_SCORE_MIN or 85).",
    ),
) -> RebuildNsfwResponse:
    sm = float(score_min) if score_min is not None else nsfw_tag_score_min()
    store = get_vector_store()
    nsfw_vecs = get_nsfw_embeddings()
    if not nsfw_vecs:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="NSFW text prototypes unavailable (embedding model not ready).",
        )

    scanned = 0
    flagged = 0
    updated = 0

    for blob_id, raw in list(store.metadata.items()):
        try:
            meta = json.loads(raw) if isinstance(raw, str) else dict(raw)
        except Exception:
            continue

        if _normalize_kind(meta.get("kind")) != "image":
            continue

        emb = store.get_embedding(blob_id)
        if emb is None:
            continue

        score = float(nsfw_similarity_score_0_100(emb, nsfw_vecs))
        is_nsfw = bool(is_nsfw_from_tag_score(score, score_min=sm))

        scanned += 1
        if is_nsfw:
            flagged += 1

        old = bool(meta.get("is_nsfw", False))
        old_score = meta.get("nsfw_score")
        meta["nsfw_score"] = score
        meta["is_nsfw"] = is_nsfw
        if (
            old != is_nsfw
            or old_score is None
            or abs(float(old_score) - score) > 1e-5
        ):
            store.metadata[blob_id] = json.dumps(meta)
            updated += 1

    if updated > 0:
        # Persist metadata updates without forcing heavy backup each run.
        store.save(create_backup=False)

    return RebuildNsfwResponse(
        scanned_images=scanned,
        flagged_nsfw=flagged,
        updated_rows=updated,
        threshold=float(sm),
    )


@router.post(
    "/reverse-image",
    response_model=SearchResponse,
    summary="Reverse image search",
    description=(
        "Upload an image (`multipart/form-data`, field name: `file`) and return visually similar indexed images.\n"
        "Optional fields: `top_k` (default 10), `exclude_nsfw` (default true)."
    ),
    responses={
        200: {"description": "Reverse-image search completed"},
        400: {"description": "Could not generate embedding from uploaded image"},
        413: {"description": "Uploaded image exceeds max size"},
        503: {"description": "Embedding model or vector store not ready"},
    },
)
async def reverse_image_search(
    file: UploadFile = File(...),
    instruction: Optional[str] = Form(None),
    top_k: int = Form(10),
    exclude_nsfw: bool = Form(True),
) -> SearchResponse:
    store = get_vector_store()
    image_bytes = await file.read()
    if len(image_bytes) > _REVERSE_IMAGE_MAX_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Image too large. Max {_REVERSE_IMAGE_MAX_BYTES} bytes.",
        )
    qemb = await run_in_threadpool(
        generate_image_embedding, image_bytes, None, instruction
    )
    if qemb is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to generate embedding for image.",
        )
    return _similar_images_from_embedding(
        store, qemb, top_k=top_k, exclude_nsfw=exclude_nsfw
    )


@router.get(
    "/stats",
    response_model=VectorStoreStats,
    summary="Get vector store statistics",
    description="""
    Retrieve statistics about the vector store, including:
    - Total number of embeddings indexed
    - Items per modality (image, video, audio, doc)
    - Whether the index has been built
    - Current status

    Use this endpoint to check if the indexer has populated the vector store
    and if it's ready for searching.
    """,
    responses={
        200: {
            "description": "Statistics retrieved successfully",
            "content": {
                "application/json": {
                    "example": {
                        "total_embeddings": 1234,
                        "index_built": True,
                        "status": "ready",
                        "by_modality": {
                            "image": 1000,
                            "video": 150,
                            "audio": 50,
                            "doc": 34,
                            "other": 0,
                        },
                    }
                }
            },
        },
    },
)
async def get_stats() -> VectorStoreStats:
    """Get statistics about the vector store.

    Automatically reloads the vector store to show current statistics,
    including newly indexed embeddings.
    """
    try:
        store = get_vector_store()
        mem_total = store.size()
        mem_by_kind = store.counts_by_kind()
        mem_index_built = bool(store._is_built)

        try:
            epoch = get_current_epoch(silent=True)
            current_epoch = MANUAL_EPOCH if epoch is None else int(epoch)
        except Exception:
            current_epoch = MANUAL_EPOCH

        db_total = 0
        db_by_kind: Dict[str, int] = {}
        if CATALOG_DB_PATH.exists():
            try:
                db_stats = _sql_indexed_counts(str(CATALOG_DB_PATH), current_epoch)
                db_total = int(db_stats["total_indexed"])
                db_by_kind = dict(db_stats["by_kind"])
            except Exception:
                pass

        # Prefer whichever source has richer information, and merge modality maps.
        total = max(mem_total, db_total)
        merged_by_kind = dict(db_by_kind)
        for k, v in mem_by_kind.items():
            merged_by_kind[k] = max(int(v), int(merged_by_kind.get(k, 0)))

        return VectorStoreStats(
            total_embeddings=total,
            index_built=mem_index_built or total > 0,
            status="ready" if total > 0 else "indexing",
            by_modality=merged_by_kind,
        )
    except HTTPException:
        return VectorStoreStats(
            total_embeddings=0,
            index_built=False,
            status="not_initialized",
            by_modality={},
        )
    except Exception as e:
        print(f"[Search] Error getting stats: {e}")
        return VectorStoreStats(
            total_embeddings=0,
            index_built=False,
            status="error",
            by_modality={},
        )


@router.get(
    "/indexer/stats",
    summary="Indexer progress and type counters",
    description=(
        "Returns live counters from the background indexer: blobs indexed by type "
        "(image, video, audio, doc, quilt), skips, failures, backfill status, and "
        "whether the listen loop is active."
    ),
)
async def get_indexer_stats() -> dict:
    try:
        from omura.indexers.multimodal_indexer import get_indexer_stats

        stats = get_indexer_stats()

        try:
            epoch = get_current_epoch(silent=True)
            current_epoch = MANUAL_EPOCH if epoch is None else int(epoch)
        except Exception:
            current_epoch = MANUAL_EPOCH

        if CATALOG_DB_PATH.exists():
            try:
                db_dash = _sql_dashboard_counts(str(CATALOG_DB_PATH), current_epoch)
                db_idx = _sql_indexed_counts(str(CATALOG_DB_PATH), current_epoch)
                by_kind = db_idx.get("by_kind", {})

                # Reconcile to DB so API-only nodes still expose meaningful progress.
                stats["total_seen_blobs"] = int(db_dash.get("total_blobs", 0))
                stats["active_seen_blobs"] = int(db_dash.get("active_blobs", 0))
                stats["total_indexed_blobs"] = int(db_idx.get("total_indexed", 0))
                stats["active_indexed_blobs"] = int(db_idx.get("active_indexed", 0))
                stats["indexed_image"] = int(by_kind.get("image", 0))
                stats["indexed_video"] = int(by_kind.get("video", 0))
                stats["indexed_audio"] = int(by_kind.get("audio", 0))
                stats["indexed_doc"] = int(by_kind.get("doc", 0))
                stats["indexed_quilt"] = int(by_kind.get("quilt", 0))
                stats["total_indexed"] = int(db_idx.get("total_indexed", 0))
            except Exception:
                pass

        # Keep response minimal for dashboard cards (hide low-signal internals).
        allowed_keys = {
            "indexed_image",
            "indexed_video",
            "indexed_audio",
            "indexed_doc",
            "indexed_quilt",
            "total_indexed",
            "total_seen_blobs",
            "active_seen_blobs",
            "total_indexed_blobs",
            "active_indexed_blobs",
            "backfill_complete",
        }
        return {k: v for k, v in stats.items() if k in allowed_keys}
    except Exception as e:
        return {"error": str(e)}
