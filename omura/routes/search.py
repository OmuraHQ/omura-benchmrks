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
from omura.utils.blockberry import get_current_epoch
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

# Maximal Marginal Relevance reranking — suppresses near-duplicate / same-collection
# hits (NFT quilts in particular). λ=1.0 disables diversity (pure cosine); λ=0.0 is
# pure diversity. Off by default; enable with OMURA_RERANK_MMR=true.
RERANK_MMR_ENABLED = os.getenv("OMURA_RERANK_MMR", "false").lower() == "true"
RERANK_MMR_LAMBDA = float(os.getenv("OMURA_RERANK_MMR_LAMBDA", "0.85"))
# Extra additive penalty when a candidate shares parent_quilt_id with an already-selected
# result. Raise to spread different collections; 0 disables the quilt-level penalty.
RERANK_SAME_QUILT_PENALTY = float(os.getenv("OMURA_RERANK_SAME_QUILT_PENALTY", "0.15"))

# Cluster-based reranking: cluster candidate embeddings on-the-fly (MiniBatchKMeans),
# then return one representative per cluster in cluster-rank order. Cluster rank is the
# sum of member similarities to the query — clusters whose members are most relevant
# rank first. Solves NFT-collection over-representation: a 50-NFT collection becomes
# ONE cluster contributing ONE result, not 50 near-duplicates.
RERANK_CLUSTER_ENABLED = os.getenv("OMURA_RERANK_CLUSTER", "false").lower() == "true"
# n_clusters = top_k * RERANK_CLUSTER_K_MULT, clamped to len(candidates).
# >1.0 gives slack (some clusters yield ≥2 picks if needed to fill top_k).
RERANK_CLUSTER_K_MULT = float(os.getenv("OMURA_RERANK_CLUSTER_K_MULT", "2.0"))
# How many candidates to cluster over. Higher = better cluster quality but slower.
RERANK_CLUSTER_OVERFETCH_MULT = int(os.getenv("OMURA_RERANK_CLUSTER_OVERFETCH", "30"))

# Multi-signal soft reranker — conservative; preserves top hit, only adjusts lower ranks.
# Combines embedding similarity (primary) with content-quality signals.
RERANK_SOFT_ENABLED = os.getenv("OMURA_RERANK_SOFT", "true").lower() == "true"
# How strongly each signal moves a candidate. 0 disables the signal.
RERANK_W_SIM = float(os.getenv("OMURA_RERANK_W_SIM", "1.0"))         # similarity weight (always primary)
RERANK_W_NSFW_PENALTY = float(os.getenv("OMURA_RERANK_W_NSFW", "0.15"))  # penalty fraction per unit NSFW score
RERANK_W_DUPE_PENALTY = float(os.getenv("OMURA_RERANK_W_DUPE", "0.20"))  # penalty when same parent_quilt_id repeats
RERANK_W_TINY_PENALTY = float(os.getenv("OMURA_RERANK_W_TINY", "0.10"))  # penalty for sub-10KB images (likely thumbnails / broken)
RERANK_W_RECENT_BOOST = float(os.getenv("OMURA_RERANK_W_RECENT", "0.05"))  # boost for blobs catalogued in the last 24h
RERANK_PRESERVE_TOP_N = int(os.getenv("OMURA_RERANK_PRESERVE_TOP_N", "1"))  # never demote first N hits

RERANK_W_NFT_PENALTY = 0.0

# Hybrid (text+FTS/BM25) fusion. DISABLED by default: the auto-generated captions are
# noisy, so keyword-matching the caption pollutes results — a blob whose caption merely
# contains "cat" gets boosted to the top even when its image is not cat-like (true
# cosine ~0.05). With FTS off, both text and reverse-image search rank purely by
# embedding (visual-semantic) similarity, keeping them truly 1-to-1. Set
# OMURA_SEARCH_HYBRID_FTS=true to re-enable the caption-fusion path.
HYBRID_FTS_ENABLED = os.getenv("OMURA_SEARCH_HYBRID_FTS", "false").lower() == "true"
# When hybrid FTS is on, a caption match gets an effective similarity floor of
# FTS_SIM_TOP decaying by FTS_SIM_DECAY per FTS rank (×450 text multiplier → ~90).
FTS_SIM_TOP = float(os.getenv("OMURA_FTS_SIM_TOP", "0.20"))
FTS_SIM_DECAY = float(os.getenv("OMURA_FTS_SIM_DECAY", "0.005"))

# Greedy MMR pass after soft rerank — same-image-embedding similarity dedup.
# Solves "wizard cat shown 3 times" duplicates that no other signal catches.
RERANK_MMR_AFTER_SOFT = os.getenv("OMURA_RERANK_MMR_AFTER_SOFT", "true").lower() == "true"
RERANK_MMR_AFTER_LAMBDA = float(os.getenv("OMURA_RERANK_MMR_AFTER_LAMBDA", "0.7"))


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


class InVideoSearchRequest(BaseModel):
    blob_id: Optional[str] = None
    query: Optional[str] = None
    top_k: int = 5
    win_sec: float = 4.0
    stride_sec: float = 2.0


class InVideoSearchResponse(BaseModel):
    blob_id: str
    query: str
    duration: Optional[float] = None
    source: Optional[str] = None
    blob_url: Optional[str] = None
    segments: List[Dict[str, Any]]


class ReverseImageResponse(BaseModel):
    results: List[Dict[str, Any]]
    total: int
    query_phash: Optional[str] = None
    duplicates_found: int = 0
    exact_duplicate_blob_id: Optional[str] = None
    provenance: Optional[Dict[str, Any]] = None


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


# ── Modality-specific stores (audio = CLAP 512-d, video = omura-embed-video 768-d) ──
# These are independent of the main image/text store: different models, different
# dims, separate FAISS indexes. Search is exact cosine over a single-modality index,
# so no image-specific FTS / NFT / cluster reranking is applied.
_shared_audio_vector_store: Optional[VectorStore] = None
_shared_video_vector_store: Optional[VectorStore] = None


def set_shared_audio_vector_store(store: VectorStore) -> None:
    global _shared_audio_vector_store
    _shared_audio_vector_store = store
    print(f"[Search] Shared AUDIO (CLAP) store set with {len(store.metadata)} items")


def set_shared_video_vector_store(store: VectorStore) -> None:
    global _shared_video_vector_store
    _shared_video_vector_store = store
    print(f"[Search] Shared VIDEO store set with {len(store.metadata)} items")


# Serve-time precheck: drop (and self-heal) audio/video results whose blob 404s.
# Default OFF: the per-query 404 HEAD-probes add latency and block the event loop. The store is
# kept clean by the /blob 404-prune + periodic sweep, so the precheck is opt-in only.
_SEARCH_PRECHECK = os.getenv("OMURA_SEARCH_PRECHECK", "0") not in ("0", "false", "False", "")
_PRECHECK_TIMEOUT = float(os.getenv("OMURA_PRECHECK_TIMEOUT", "6"))


def _probe_blob_status(blob_id: str) -> int:
    """HEAD-probe a blob (plain or quilt-patch) via the aggregator pool. Returns the HTTP
    status (404 = gone), or -1 on network error (treated as 'keep', not broken)."""
    from omura.utils.aggregator_pool import get_pool
    import requests as _rq
    if "::" in blob_id:
        quilt, ident = blob_id.split("::", 1)
        path = f"/v1/blobs/by-quilt-id/{quilt}/{_rq.utils.quote(ident, safe='')}"
    else:
        path = f"/v1/blobs/{blob_id}"
    try:
        resp, _ = get_pool().head(path, timeout=_PRECHECK_TIMEOUT, allow_redirects=True)
        if resp is None:
            return -1
        if resp.status_code in (405, 501):  # some upstreams reject HEAD
            resp, _ = get_pool().get(path, headers={"Range": "bytes=0-0"},
                                     timeout=_PRECHECK_TIMEOUT, stream=True)
            if resp is not None:
                resp.close()
            return resp.status_code if resp is not None else -1
        return resp.status_code
    except Exception:
        return -1


def _prune_broken_modality_blobs(store: VectorStore, blob_ids: List[str]) -> None:
    """Remove 404'd blobs from a modality store and mark them expired in the catalog so they
    stop being served. Best-effort; runs in a background thread."""
    if not blob_ids:
        return
    try:
        with store._lock:
            removed = store.remove_blob_ids(blob_ids)
            if removed > 0:
                store.save(create_backup=False)
        with sqlite3.connect(str(CATALOG_DB_PATH), timeout=15) as conn:
            conn.executemany(
                "UPDATE blobs SET is_active = 0, status = 'expired' WHERE blob_id = ?",
                [(b,) for b in blob_ids],
            )
            conn.commit()
        print(f"[precheck] pruned {len(blob_ids)} broken blobs (removed {removed} from store)")
    except Exception as e:  # noqa: BLE001
        print(f"[precheck] prune failed: {e}")


def _modality_search(
    store: Optional[VectorStore],
    qemb: Optional[np.ndarray],
    top_k: int,
    kind: str,
    exclude_nsfw: bool,
) -> SearchResponse:
    """Exact cosine retrieval over a single-modality store, serialized like /search."""
    if store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"{kind} index not initialized (set OMURA_{kind.upper()}_BACKEND and build the index).",
        )
    if qemb is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"{kind} embedding model not ready yet.",
        )
    qvec = np.asarray(qemb, dtype=np.float32).flatten()
    n = float(np.linalg.norm(qvec))
    if n <= 0:
        raise HTTPException(status_code=503, detail="Query embedding has zero norm.")
    qvec = qvec / n

    raw = store.search(qvec, top_k=max(int(top_k) * 4, int(top_k)), kind_filter=None)
    # Build the candidate list (over-fetched ×4 so the precheck can drop broken ones).
    candidates: List[Dict[str, Any]] = []
    for meta, sim in raw:
        if not isinstance(meta, dict):
            continue
        if exclude_nsfw and meta.get("is_nsfw"):
            continue
        sim_f = float(sim)
        candidates.append({
            "blob_id": meta.get("blob_id"),
            # Cosine in CLAP/IV2 joint spaces is modest in magnitude; scale into a
            # readable 0-100 score (clamped) without manufacturing ties.
            "score": round(max(0.0, min(1.0, (sim_f + 1.0) / 2.0)) * 100.0, 1),
            "distance": round(1.0 - sim_f, 4),
            **meta,
        })

    # Precheck: probe candidate blobs and drop the ones the aggregator can't serve (404),
    # so the portal never receives a broken blob. Broken ids are pruned in the background.
    broken: List[str] = []
    if _SEARCH_PRECHECK and candidates:
        from concurrent.futures import ThreadPoolExecutor
        bids = [c["blob_id"] for c in candidates]
        with ThreadPoolExecutor(max_workers=min(16, len(bids))) as ex:
            statuses = dict(zip(bids, ex.map(_probe_blob_status, bids)))
        broken = [b for b, s in statuses.items() if s == 404]
        if broken:
            bset = set(broken)
            candidates = [c for c in candidates if c["blob_id"] not in bset]
            threading.Thread(
                target=_prune_broken_modality_blobs, args=(store, broken), daemon=True
            ).start()

    results = candidates[: int(top_k)]
    return SearchResponse(results=results, total=len(results))


# Max bytes for reverse-image upload (default 25 MiB)
_REVERSE_IMAGE_MAX_BYTES = int(
    os.getenv("OMURA_REVERSE_IMAGE_MAX_BYTES", str(25 * 1024 * 1024))
)


def _soft_rerank(
    candidates: List[Dict[str, Any]],
    top_k: int,
) -> List[Dict[str, Any]]:
    """Conservative multi-signal reranker + optional MMR dedup tail.

    Penalties (multiplicative on base similarity):
      - NSFW tag-score normalized [0,1] × RERANK_W_NSFW_PENALTY
      - **NFT/avatar P(NFT|image) × RERANK_W_NFT_PENALTY** (zero-shot CLIP softmax)
      - Repeated parent_quilt_id × RERANK_W_DUPE_PENALTY (per duplicate)
      - size < 10 KB × RERANK_W_TINY_PENALTY (thumbnail/broken)

    Boosts:
      - first_seen_at within last 24h × RERANK_W_RECENT_BOOST

    The top ``RERANK_PRESERVE_TOP_N`` similarity hits are *never* demoted. After the
    soft pass, if ``RERANK_MMR_AFTER_SOFT`` is on, a greedy MMR pass on the stored
    image embeddings deduplicates near-identical results (e.g. the same wizard-cat
    appearing 3× from a Quilt-Inscriptions style collection).

    Each result is annotated with ``rerank_score`` (0-100) and ``rerank_signals``.
    """
    from omura.utils.nft_detector import nft_probability
    import time as _t
    from datetime import datetime, timezone

    if not candidates:
        return []

    now_ts = _t.time()
    seen_quilts: Dict[str, int] = {}

    out: List[Dict[str, Any]] = []
    for c in candidates:
        # Use raw cosine similarity (_sim) when available; it's stored as [0,1] and
        # provides proper differentiation between results.  The legacy `score` field
        # is capped at 100 (score = min(cosine_sim * 1000, 100)) which collapses
        # almost all results to 1.0 and destroys the similarity ranking.
        raw_sim = c.get("_sim")
        if isinstance(raw_sim, (int, float)):
            sim_norm = max(float(raw_sim), 0.0)
        else:
            sim_norm = float(c.get("score", 0.0)) / 100.0
        signals = {"base_sim": round(sim_norm, 4)}

        # NSFW penalty
        nsfw_score = c.get("nsfw_tag_score")
        nsfw_penalty = 0.0
        if isinstance(nsfw_score, (int, float)):
            nsfw_penalty = (float(nsfw_score) / 100.0) * RERANK_W_NSFW_PENALTY
        elif c.get("is_nsfw"):
            nsfw_penalty = RERANK_W_NSFW_PENALTY
        signals["nsfw_penalty"] = round(nsfw_penalty, 3)

        # NFT/avatar penalty — zero-shot CLIP text-anchor softmax.
        # Requires the candidate to have its stored embedding attached as ``_embedding``;
        # the search loop already provides it when soft rerank is on.
        nft_penalty = 0.0
        nft_p: Optional[float] = None
        emb = c.get("_embedding")
        if isinstance(emb, np.ndarray) and RERANK_W_NFT_PENALTY > 0:
            try:
                nft_p = nft_probability(emb)
                if nft_p is not None:
                    nft_penalty = float(nft_p) * RERANK_W_NFT_PENALTY
            except Exception:
                nft_p = None
        if nft_p is not None:
            signals["nft_prob"] = round(nft_p, 3)
            c["nft_score"] = round(nft_p, 4)
        signals["nft_penalty"] = round(nft_penalty, 3)

        # Duplicate-quilt penalty (escalates per repeat)
        quilt = c.get("parent_quilt_id")
        dupe_penalty = 0.0
        if quilt:
            seen_count = seen_quilts.get(quilt, 0)
            dupe_penalty = RERANK_W_DUPE_PENALTY * seen_count  # 0 on first, full on second, etc.
            seen_quilts[quilt] = seen_count + 1
        signals["dupe_penalty"] = round(dupe_penalty, 3)

        # Size sanity (tiny = probably thumbnail or corrupt)
        size = c.get("size") or 0
        tiny_penalty = 0.0
        if isinstance(size, int) and 0 < size < 10240:
            tiny_penalty = RERANK_W_TINY_PENALTY
        signals["tiny_penalty"] = round(tiny_penalty, 3)

        # Recency boost
        recent_boost = 0.0
        first_seen = c.get("first_seen_at")
        if isinstance(first_seen, str):
            try:
                fs = datetime.fromisoformat(first_seen.replace("Z", "+00:00"))
                if fs.tzinfo is None:
                    fs = fs.replace(tzinfo=timezone.utc)
                age_h = (datetime.now(timezone.utc) - fs).total_seconds() / 3600.0
                if age_h < 24:
                    recent_boost = RERANK_W_RECENT_BOOST * (1.0 - age_h / 24.0)
            except (ValueError, TypeError):
                pass
        signals["recent_boost"] = round(recent_boost, 3)

        rerank = sim_norm * RERANK_W_SIM * (
            1.0 + recent_boost - nsfw_penalty - nft_penalty - dupe_penalty - tiny_penalty
        )
        c["rerank_score"] = round(rerank * 100.0, 2)
        c["rerank_signals"] = signals
        out.append(c)

    # Sort by rerank_score, but preserve the top-N from the original similarity order.
    if RERANK_PRESERVE_TOP_N > 0 and len(out) > RERANK_PRESERVE_TOP_N:
        preserved = out[: RERANK_PRESERVE_TOP_N]
        rest_sorted = sorted(out[RERANK_PRESERVE_TOP_N:], key=lambda x: x["rerank_score"], reverse=True)
        ranked = preserved + rest_sorted
    else:
        ranked = sorted(out, key=lambda x: x["rerank_score"], reverse=True)

    # Optional MMR dedup pass — greedy pick on rerank_score minus max-similarity-to-already-picked.
    if RERANK_MMR_AFTER_SOFT and len(ranked) > top_k:
        lam = RERANK_MMR_AFTER_LAMBDA
        selected: List[Dict[str, Any]] = []
        pool = list(ranked)
        while pool and len(selected) < int(top_k):
            if not selected:
                pick = pool[0]
            else:
                best, best_v = None, -1e9
                sel_embs = [s.get("_embedding") for s in selected if isinstance(s.get("_embedding"), np.ndarray)]
                for r in pool:
                    rs = float(r.get("rerank_score", 0.0)) / 100.0
                    re = r.get("_embedding")
                    if sel_embs and isinstance(re, np.ndarray):
                        sims = [float(np.dot(re, s)) for s in sel_embs]
                        max_sim = max(sims) if sims else 0.0
                    else:
                        max_sim = 0.0
                    v = lam * rs - (1.0 - lam) * max_sim
                    if v > best_v:
                        best_v, best = v, r
                pick = best if best is not None else pool[0]
            selected.append(pick)
            pool.remove(pick)
        final = selected
    else:
        final = ranked[: int(top_k)]

    # Strip internal helpers before returning (avoid leaking np arrays in the API response)
    for r in final:
        r.pop("_embedding", None)
        r.pop("_sim", None)
    return final


def _mmr_rerank(
    candidates: List[Dict[str, Any]],
    top_k: int,
    lam: float = RERANK_MMR_LAMBDA,
    same_quilt_penalty: float = RERANK_SAME_QUILT_PENALTY,
) -> List[Dict[str, Any]]:
    """Maximal Marginal Relevance reranking with same-quilt penalty.

    Each candidate must carry ``_embedding`` (np.ndarray) and ``_sim`` (float) keys; both
    are stripped from the returned dicts. Diversity term is the max cosine similarity
    against already-selected items, plus an additive bump when the candidate shares a
    ``parent_quilt_id`` with any selected item (this is what breaks up NFT-collection
    clustering, since NFTs in a collection are stored as patches of the same quilt).
    """
    if not candidates:
        return []
    if lam >= 1.0 - 1e-9:
        # No diversity wanted — just take the top-K by similarity
        out: List[Dict[str, Any]] = []
        for c in candidates[:top_k]:
            c.pop("_embedding", None)
            c.pop("_sim", None)
            out.append(c)
        return out

    selected: List[Dict[str, Any]] = []
    selected_embs: List[np.ndarray] = []
    selected_quilts: set = set()
    remaining = list(candidates)

    while remaining and len(selected) < top_k:
        best_idx = -1
        best_score = -float("inf")
        for i, c in enumerate(remaining):
            sim_q = float(c.get("_sim", 0.0))
            if selected_embs:
                emb = c.get("_embedding")
                if emb is not None and isinstance(emb, np.ndarray):
                    sims = np.dot(np.stack(selected_embs), emb)
                    div = float(np.max(sims))
                else:
                    div = 0.0
                quilt = c.get("parent_quilt_id")
                if quilt and quilt in selected_quilts:
                    div += same_quilt_penalty
            else:
                div = 0.0
            mmr = lam * sim_q - (1.0 - lam) * div
            if mmr > best_score:
                best_score = mmr
                best_idx = i
        if best_idx < 0:
            break
        picked = remaining.pop(best_idx)
        emb = picked.pop("_embedding", None)
        picked.pop("_sim", None)
        if isinstance(emb, np.ndarray):
            selected_embs.append(emb)
        q = picked.get("parent_quilt_id")
        if q:
            selected_quilts.add(q)
        selected.append(picked)
    return selected


def _cluster_rerank(
    candidates: List[Dict[str, Any]],
    top_k: int,
    k_mult: float = RERANK_CLUSTER_K_MULT,
) -> List[Dict[str, Any]]:
    """Cluster candidates by embedding, return one representative per cluster.

    Steps:
      1. KMeans over candidate embeddings with k = round(top_k * k_mult).
      2. Score each cluster as Σ similarities of its members → ``cluster_rank``.
      3. Pick the highest-sim member from each cluster, in cluster_rank order, until
         top_k is filled. If clusters are exhausted (fewer clusters than top_k), fill
         remaining slots from each cluster's 2nd-best, 3rd-best, etc.

    Annotates each returned dict with ``cluster_id`` (0-indexed, query-local) and
    ``cluster_rank`` (1-indexed, smaller = more relevant cluster).
    """
    if not candidates:
        return []
    if len(candidates) <= top_k:
        for i, c in enumerate(candidates):
            c.pop("_embedding", None)
            c.pop("_sim", None)
            c["cluster_id"] = i
            c["cluster_rank"] = i + 1
        return candidates

    try:
        from sklearn.cluster import MiniBatchKMeans
    except ImportError:
        # Gracefully degrade to pure top-k by similarity if sklearn missing.
        out: List[Dict[str, Any]] = []
        for c in candidates[:top_k]:
            c.pop("_embedding", None)
            c.pop("_sim", None)
            out.append(c)
        return out

    embs = []
    valid: List[Dict[str, Any]] = []
    for c in candidates:
        e = c.get("_embedding")
        if isinstance(e, np.ndarray):
            embs.append(e)
            valid.append(c)
    if len(valid) <= top_k:
        for i, c in enumerate(valid):
            c.pop("_embedding", None)
            c.pop("_sim", None)
            c["cluster_id"] = i
            c["cluster_rank"] = i + 1
        return valid

    n_clusters = max(2, min(int(round(top_k * k_mult)), len(valid)))
    X = np.vstack(embs).astype(np.float32)
    try:
        km = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=3,
            batch_size=min(256, len(valid)),
            max_iter=50,
        )
        labels = km.fit_predict(X)
    except Exception as exc:
        print(f"[Search] cluster rerank failed ({exc}); returning top-k by similarity")
        out = []
        for c in valid[:top_k]:
            c.pop("_embedding", None)
            c.pop("_sim", None)
            out.append(c)
        return out

    # Group candidates by cluster, each sorted by similarity desc.
    by_cluster: Dict[int, List[Dict[str, Any]]] = {}
    for c, lab in zip(valid, labels):
        by_cluster.setdefault(int(lab), []).append(c)
    for lst in by_cluster.values():
        lst.sort(key=lambda x: float(x.get("_sim", 0.0)), reverse=True)

    # Rank clusters by sum of member similarities (most relevant cluster first).
    cluster_scores = {
        cid: float(sum(float(x.get("_sim", 0.0)) for x in members))
        for cid, members in by_cluster.items()
    }
    ranked_clusters = sorted(cluster_scores.keys(), key=lambda c: cluster_scores[c], reverse=True)
    cluster_rank_map = {cid: i + 1 for i, cid in enumerate(ranked_clusters)}

    # Round-robin: take best member from each cluster in rank order, then 2nd-best, etc.
    selected: List[Dict[str, Any]] = []
    indices = {cid: 0 for cid in ranked_clusters}
    while len(selected) < top_k:
        progress = False
        for cid in ranked_clusters:
            if len(selected) >= top_k:
                break
            members = by_cluster[cid]
            idx = indices[cid]
            if idx >= len(members):
                continue
            picked = members[idx]
            indices[cid] = idx + 1
            picked.pop("_embedding", None)
            picked.pop("_sim", None)
            picked["cluster_id"] = cid
            picked["cluster_rank"] = cluster_rank_map[cid]
            selected.append(picked)
            progress = True
        if not progress:
            break

    return selected


def _similar_images_from_embedding(
    store: VectorStore,
    qemb: np.ndarray,
    top_k: int,
    exclude_blob_id: Optional[str] = None,
    exclude_nsfw: bool = True,
    text_query: Optional[str] = None,
    query_kind: str = "text",
) -> SearchResponse:
    """Run cosine retrieval using vector-store index (legacy-compatible path).

    ``query_kind`` ("text" or "image") only affects the similarity→score scale
    so cross-modal (text) and same-modal (reverse-image) queries both anchor a
    strong match near 100. The candidate set, ranking, NSFW handling and
    response shape are identical for both, keeping the two endpoints 1-to-1.
    """
    qvec = np.asarray(qemb, dtype=np.float32).flatten()
    qnorm = float(np.linalg.norm(qvec))
    if qnorm <= 0:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Query embedding is invalid (zero norm); model may still be warming up.",
        )
    qvec = qvec / qnorm

    # Over-fetch then apply exclusions (nsfw/blob id) while preserving rank quality.
    # Cluster rerank needs more candidates to form meaningful clusters.
    overfetch_mult = (
        max(20, RERANK_CLUSTER_OVERFETCH_MULT) if RERANK_CLUSTER_ENABLED else 20
    )
    overfetch_k = max(int(top_k) * overfetch_mult, int(top_k))
    raw_results = store.search(qvec, top_k=overfetch_k, kind_filter="image")
    nsfw_vecs = get_nsfw_embeddings() or []

    # 1. Gather vector candidate IDs
    vec_ids = []
    for meta, _ in raw_results:
        if isinstance(meta, dict) and meta.get("blob_id"):
            vec_ids.append(meta["blob_id"])

    # 2. Gather text candidate (FTS5 BM25) IDs — only when hybrid FTS is enabled.
    # Off by default so search ranks purely by embedding similarity (noisy captions
    # otherwise inflate keyword matches); see HYBRID_FTS_ENABLED.
    fts_results = []
    if HYBRID_FTS_ENABLED and text_query:
        cleaned_q = "".join(c if c.isalnum() or c.isspace() else " " for c in text_query).strip()
        if cleaned_q:
            tokens = [t for t in cleaned_q.split() if t]
            if tokens:
                match_expr = " AND ".join(tokens)
                try:
                    with sqlite3.connect(str(CATALOG_DB_PATH), timeout=10) as conn:
                        conn.row_factory = sqlite3.Row
                        cur = conn.execute(
                            """
                            SELECT b.*, bm25(blobs_fts) AS bm25_score
                            FROM blobs_fts f
                            JOIN blobs b ON f.blob_id = b.blob_id
                            WHERE blobs_fts MATCH ? AND b.is_active = 1 AND (b.kind = 'image' OR b.mime_type LIKE 'image/%')
                            ORDER BY bm25_score ASC
                            LIMIT ?
                            """,
                            (match_expr, overfetch_k)
                        )
                        fts_results = [dict(row) for row in cur.fetchall()]
                except Exception as e:
                    print(f"[Search] FTS5 search error: {e}")

    fts_ids = [r["blob_id"] for r in fts_results if r.get("blob_id")]

    # Map details and compute similarities for hybrid candidate list
    candidate_details = {}
    for meta, sim in raw_results:
        if isinstance(meta, dict) and meta.get("blob_id"):
            bid = meta["blob_id"]
            candidate_details[bid] = (dict(meta), float(sim))

    for r in fts_results:
        bid = r["blob_id"]
        if bid not in candidate_details:
            emb = store.get_embedding(bid)
            if emb is not None:
                sim = float(np.dot(qvec, emb.flatten() / np.linalg.norm(emb)))
            else:
                sim = 0.0
            meta_cleaned = {
                "blob_id": r["blob_id"],
                "mime_type": r["mime_type"],
                "extension": r["extension"],
                "kind": r["kind"],
                "size": r["size"],
                "is_nsfw": bool(r["is_nsfw"]),
                "first_seen_at": r["first_seen_at"],
                "owner": r.get("owner"),
                "caption": r.get("caption"),
            }
            candidate_details[bid] = (meta_cleaned, sim)
        else:
            meta, sim = candidate_details[bid]
            meta["caption"] = r.get("caption")

    # Combine rankings using RRF
    k = 60
    rrf_scores = {}
    all_bids = set(vec_ids + fts_ids)

    for idx, bid in enumerate(vec_ids):
        rrf_scores[bid] = rrf_scores.get(bid, 0.0) + 1.0 / (k + idx + 1)

    for idx, bid in enumerate(fts_ids):
        rrf_scores[bid] = rrf_scores.get(bid, 0.0) + 1.0 / (k + idx + 1)

    # FTS rank (best = 0) per blob, used to give genuine text matches a score floor.
    fts_rank = {bid: idx for idx, bid in enumerate(fts_ids)}

    sorted_bids = sorted(all_bids, key=lambda x: rrf_scores.get(x, 0.0), reverse=True)

    candidates: List[Dict[str, Any]] = []
    for bid in sorted_bids:
        if exclude_blob_id and bid == exclude_blob_id:
            continue

        meta, sim = candidate_details[bid]

        # Hybrid scoring: ``sim`` stays the true cosine similarity (used for vector
        # ranking + diversity). For the user-visible ``score`` we additionally floor
        # text/FTS hits by a text-relevance value (below), so a strong caption match
        # whose cross-modal cosine is naturally low doesn't show a near-zero score and
        # then drag every result beneath it down via the monotonic clamp — the old
        # "hybrid scores look random" symptom.

        emb = (
            store.get_embedding(bid)
            if (nsfw_vecs or RERANK_MMR_ENABLED or RERANK_CLUSTER_ENABLED or RERANK_SOFT_ENABLED)
            else None
        )

        # Result-time NSFW fallback: same 0–100 tag score as indexing (score > min, default 85).
        is_nsfw = bool(meta.get("is_nsfw", False))
        nsfw_tag_score = None
        if nsfw_vecs and emb is not None:
            nsfw_tag_score = float(nsfw_similarity_score_0_100(emb, nsfw_vecs))
            if not is_nsfw and is_nsfw_from_tag_score(nsfw_tag_score):
                is_nsfw = True

        if exclude_nsfw and is_nsfw:
            continue

        # Ranking similarity = cosine, floored by FTS text-relevance for caption hits
        # so vector and text matches live on one scale. This single value drives both
        # the reranker and the displayed score, so order and percentage never diverge.
        rank_sim = max(float(sim), 0.0)
        if bid in fts_rank:
            fts_sim = max(0.0, FTS_SIM_TOP - FTS_SIM_DECAY * fts_rank[bid])
            rank_sim = max(rank_sim, fts_sim)

        # Map similarity → 0-100 using a modality-aware multiplier (cross-modal text
        # vs same-modal image); see get_score_multiplier for why these differ.
        from omura.utils.imagebind_embeddings import get_score_multiplier
        multiplier = get_score_multiplier(query_kind)
        score = float(min(rank_sim * multiplier, 100.0))
        dist = float(1.0 - float(sim))
        out = {"blob_id": bid, "score": score, "distance": dist, **meta}
        out["is_nsfw"] = is_nsfw
        if nsfw_tag_score is not None:
            out["nsfw_tag_score"] = nsfw_tag_score
        if is_nsfw and not out.get("categories"):
            out["categories"] = ["nsfw", "pornographic"]
            out["category_confidence"] = out.get("category_confidence") or "low"

        if RERANK_MMR_ENABLED or RERANK_CLUSTER_ENABLED or RERANK_SOFT_ENABLED:
            # Normalize embeddings to unit norm; soft rerank uses them for NFT + MMR dedup.
            if isinstance(emb, np.ndarray):
                e = np.asarray(emb, dtype=np.float32).flatten()
                n = float(np.linalg.norm(e))
                if n > 0:
                    out["_embedding"] = e / n
            # Rank by the hybrid similarity (cosine floored by FTS relevance) so the
            # reranker's order matches the displayed score; the diversity/dedup terms
            # use ``_embedding`` (true cosine geometry), not this value.
            out["_sim"] = float(rank_sim)
        candidates.append(out)

    # Reranker priority: cluster (hard diversity) > MMR (greedy diversity) > soft
    # (multi-signal, top-N preserved). All default off except soft.
    if RERANK_CLUSTER_ENABLED:
        results = _cluster_rerank(candidates, top_k=int(top_k))
    elif RERANK_MMR_ENABLED:
        results = _mmr_rerank(candidates, top_k=int(top_k))
    elif RERANK_SOFT_ENABLED:
        results = _soft_rerank(candidates, top_k=int(top_k))
    else:
        results = candidates[: int(top_k)]
        # Rerankers strip these internally; the no-rerank path must too so the
        # response never leaks np arrays / raw sims (keeps both endpoints 1-to-1).
        for r in results:
            if isinstance(r, dict):
                r.pop("_embedding", None)
                r.pop("_sim", None)

    # The displayed score must track the order the user actually sees. RRF fusion and
    # the rerankers (cluster/MMR/soft) reorder results for relevance/diversity, so a
    # later item can carry a higher raw score than an earlier one. The old fix clamped
    # each score down to the running minimum — but that *manufactures ties*, collapsing
    # whole runs to one value ("why are so many cats at 74%?"). Instead, keep the real
    # set of computed score values and assign them in descending order to the final
    # ranked items: monotonic with the displayed order, no fake ties, magnitudes (and
    # thus a strong top match near 100) preserved, and lower ranks stay differentiated.
    score_items = [r for r in results if isinstance(r, dict)]
    sorted_scores = sorted(
        (float(r.get("score", 0.0) or 0.0) for r in score_items), reverse=True
    )
    for r, s in zip(score_items, sorted_scores):
        r["score"] = round(s, 1)

    # Caption enrichment (display only): the vector-store metadata has no caption, so
    # without the FTS join results would show blank captions. Fill them from the catalog
    # for the final top_k only — purely informational, never used for ranking. Applies to
    # both text and reverse-image, keeping the two responses 1-to-1.
    missing = [r["blob_id"] for r in score_items
               if r.get("blob_id") and not r.get("caption")]
    if missing and CATALOG_DB_PATH.exists():
        try:
            with sqlite3.connect(str(CATALOG_DB_PATH), timeout=10) as conn:
                ph = ",".join("?" * len(missing))
                cap_map = {
                    row[0]: row[1]
                    for row in conn.execute(
                        f"SELECT blob_id, caption FROM blobs WHERE blob_id IN ({ph})",
                        missing,
                    )
                }
            for r in score_items:
                if not r.get("caption") and cap_map.get(r["blob_id"]):
                    r["caption"] = cap_map[r["blob_id"]]
        except Exception as e:
            print(f"[Search] caption enrichment failed: {e}")

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
        text_query=q,
    )


@router.post(
    "/audio",
    response_model=SearchResponse,
    summary="Audio search (text → audio via CLAP)",
    description="Text-to-audio retrieval over the active-audio catalog using CLAP (laion/larger_clap_general).",
)
async def search_audio(payload: SearchRequest):
    """Search active audio blobs by a text query in the CLAP joint space."""
    q = (payload.query or "").strip()
    if not q:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Field 'query' is required.")
    from omura.utils import clap_embeddings as clap

    # Run the (synchronous) embed + FAISS search off the event loop so concurrent
    # requests and /health don't block on each other.
    def _run():
        qemb = clap.embed_text(q)
        return _modality_search(
            _shared_audio_vector_store, qemb, top_k=int(payload.top_k),
            kind="audio", exclude_nsfw=bool(payload.exclude_nsfw),
        )
    return await run_in_threadpool(_run)


@router.post(
    "/video",
    response_model=SearchResponse,
    summary="Video search (text → video via omura-embed-video)",
    description="Text-to-video retrieval over the active-video catalog using the finetuned InternVideo2 model.",
)
async def search_video(payload: SearchRequest):
    """Search active video blobs by a text query in the omura-embed-video space."""
    q = (payload.query or "").strip()
    if not q:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Field 'query' is required.")
    def _run():
        try:
            from omura.utils import video_embeddings as videoemb
            qemb = videoemb.embed_text(q)
        except Exception:
            qemb = None
        return _modality_search(
            _shared_video_vector_store, qemb, top_k=int(payload.top_k),
            kind="video", exclude_nsfw=bool(payload.exclude_nsfw),
        )
    return await run_in_threadpool(_run)


@router.post(
    "/video/in-video",
    response_model=InVideoSearchResponse,
    summary="Search inside a video (text → timestamp / seek-to-timestamp)",
    description=(
        "Temporal localization within a single video: given a `blob_id` and a text `query`, "
        "returns ranked time `segments` ([{start, end, score}] in seconds) marking where the "
        "query content appears, so the portal can seek the player to that moment. Uses the "
        "precomputed temporal index when available, else localizes on-demand."
    ),
)
async def search_in_video(payload: InVideoSearchRequest):
    blob_id = (payload.blob_id or "").strip()
    q = (payload.query or "").strip()
    if not blob_id or not q:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="Fields 'blob_id' and 'query' are required.")
    from omura.utils import video_embeddings as videoemb
    out = videoemb.search_in_video(
        blob_id, q, top_k=int(payload.top_k),
        win_sec=float(payload.win_sec), stride_sec=float(payload.stride_sec),
    )
    if out is None:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY,
                            detail="video temporal service unavailable")
    from urllib.parse import quote
    return InVideoSearchResponse(
        blob_id=blob_id, query=q,
        duration=out.get("duration"), source=out.get("source"),
        blob_url=f"/blob/{quote(blob_id, safe='')}",
        segments=out.get("segments", []),
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
def media_counters_dashboard() -> MediaBlobCounterResponse:
    store: Optional[VectorStore] = None
    try:
        store = get_vector_store()
    except HTTPException:
        store = None
    catalog_db_path = getattr(store, "catalog_db_path", CATALOG_DB_PATH)

    try:
        current_epoch = get_current_epoch(silent=True)
    except Exception:
        current_epoch = 0

    if catalog_db_path.exists():
        try:
            stats = _sql_dashboard_counts(str(catalog_db_path), current_epoch)
            active = stats["counts_active"]  # type: ignore[index]
            # Both modality_counts_all and modality_counts_active surface to the
            # dashboard; keep them on the same (active) denominator so the top
            # counters and the per-modality bars don't disagree. The all-vs-active
            # split is still queryable from counts_all if needed downstream.
            return MediaBlobCounterResponse(
                total_blobs=int(stats["total_blobs"]),  # type: ignore[arg-type]
                active_blobs=int(stats["active_blobs"]),  # type: ignore[arg-type]
                identified_image=int(active.get("image", 0)),
                identified_video=int(active.get("video", 0)),
                identified_audio=int(active.get("audio", 0)),
                modality_counts_all=active,  # type: ignore[arg-type]
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
    # DB is authoritative for NSFW counter baseline (works even on API-only nodes).
    db_nsfw_count = 0
    try:
        with sqlite3.connect(str(CATALOG_DB_PATH), timeout=10) as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM blobs WHERE indexed=1 AND is_nsfw=1"
            ).fetchone()
            db_nsfw_count = int(row[0] if row else 0)
    except Exception:
        db_nsfw_count = 0

    requested_buckets = ["nsfw", "screen_ui", "building", "art", "animal", "food"]
    categories_out: Dict[str, int] = {k: 0 for k in requested_buckets}
    categories_out["other"] = 0

    store: Optional[VectorStore] = None
    try:
        store = get_vector_store()
    except HTTPException:
        store = None

    live_nsfw_count = 0
    if store is not None and len(store.embeddings_dict) > 0:
        try:
            from omura.utils.imagebind_embeddings import get_semantic_category_embeddings

            semantic_labels = [
                "screen_ui",
                "building",
                "art",
                "animal",
                "food",
                "human",
                "nature",
                "vehicle",
                "document",
                "meme",
            ]
            semantic_embs = get_semantic_category_embeddings(semantic_labels)
            nsfw_vecs = get_nsfw_embeddings() or []

            for blob_id, raw in store.metadata.items():
                try:
                    meta = json.loads(raw) if isinstance(raw, str) else dict(raw)
                    if _normalize_kind(meta.get("kind")) != "image":
                        continue

                    emb = store.get_embedding(blob_id)
                    if emb is None:
                        continue

                    vec = np.asarray(emb, dtype=np.float32).flatten()
                    n = np.linalg.norm(vec)
                    if n > 0:
                        vec = vec / n

                    # Prioritize hybrid NSFW assignment.
                    tag_s = float(nsfw_similarity_score_0_100(vec, nsfw_vecs))
                    if is_nsfw_from_tag_score(tag_s):
                        categories_out["nsfw"] += 1
                        live_nsfw_count += 1
                        continue

                    if semantic_embs:
                        scores = [float(np.dot(vec, cat_emb)) for cat_emb in semantic_embs]
                        best_idx = int(np.argmax(np.asarray(scores, dtype=np.float32)))
                        best_label = semantic_labels[best_idx]
                        if best_label in categories_out:
                            categories_out[best_label] += 1
                        else:
                            categories_out["other"] += 1
                    else:
                        categories_out["other"] += 1
                except Exception:
                    continue
        except Exception as e:
            print(f"[Search] classifier dashboard live classification failed: {e}")

    if sum(int(v) for v in categories_out.values()) == 0:
        # Last-resort fallback with no live store data.
        try:
            current_epoch = get_current_epoch(silent=True)
        except Exception:
            current_epoch = 0
        try:
            if CATALOG_DB_PATH.exists():
                db_idx = _sql_indexed_counts(str(CATALOG_DB_PATH), current_epoch)
                categories_out["other"] = int(db_idx.get("by_kind", {}).get("image", 0))
        except Exception:
            pass

    return ClassifierDashboardResponse(
        categories=categories_out,
        nsfw_count=max(db_nsfw_count, live_nsfw_count),
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


def _fetch_blob_bytes(blob_id: str, timeout: int = 30) -> Optional[bytes]:
    """Fetch raw bytes of an indexed blob (plain or quilt-patch) via the aggregator pool."""
    from omura.utils.aggregator_pool import get_pool
    import requests as _rq
    if "::" in blob_id:
        quilt, ident = blob_id.split("::", 1)
        path = f"/v1/blobs/by-quilt-id/{quilt}/{_rq.utils.quote(ident, safe='')}"
    else:
        path = f"/v1/blobs/{blob_id}"
    try:
        resp, _ = get_pool().get(path, timeout=timeout)
        if resp is None or resp.status_code != 200:
            return None
        return resp.content
    except Exception:
        return None


def _harden_reverse_results(results: List[Dict[str, Any]], query_bytes: bytes,
                            verify_top: int = 12) -> Dict[str, Any]:
    """Confirm exact/near duplicates among the top embedding hits via perceptual hash, and
    attach provenance. Annotates each verified result with `phash_hamming`,
    `duplicate_class`, `is_exact_duplicate`. Returns a summary dict."""
    from concurrent.futures import ThreadPoolExecutor
    from omura.utils import perceptual_hash as ph

    qhash = ph.dhash(query_bytes)
    summary = {"query_phash": (f"{qhash:016x}" if qhash is not None else None),
               "duplicates_found": 0, "exact_duplicate_blob_id": None, "provenance": None}
    if qhash is None or not results:
        return summary

    head = results[:verify_top]

    def _verify(r):
        data = _fetch_blob_bytes(r.get("blob_id", ""))
        if not data:
            return
        h = ph.dhash(data)
        if h is None:
            return
        dist = ph.hamming(qhash, h)
        r["phash_hamming"] = dist
        r["duplicate_class"] = ph.classify(dist)
        r["is_exact_duplicate"] = dist <= ph.EXACT_MAX
        r["is_near_duplicate"] = dist <= ph.NEAR_MAX

    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(_verify, head))

    dups = [r for r in head if r.get("is_near_duplicate")]
    summary["duplicates_found"] = len(dups)
    exact = next((r for r in head if r.get("is_exact_duplicate")), None)
    top_dup = exact or (dups[0] if dups else None)
    if exact:
        summary["exact_duplicate_blob_id"] = exact.get("blob_id")
    if top_dup is not None:
        summary["provenance"] = {
            "blob_id": top_dup.get("blob_id"),
            "owner": top_dup.get("owner"),
            "parent_quilt_id": top_dup.get("parent_quilt_id"),
            "quilt_identifier": top_dup.get("quilt_identifier"),
            "duplicate_class": top_dup.get("duplicate_class"),
            "phash_hamming": top_dup.get("phash_hamming"),
        }
    return summary


@router.post(
    "/reverse-image",
    response_model=ReverseImageResponse,
    summary="Reverse image search (hardened: exact-duplicate + NFT provenance)",
    description=(
        "Upload an image (`multipart/form-data`, field name: `file`) and return visually similar "
        "indexed images. Top hits are confirmed as exact/near duplicates via perceptual hash and "
        "annotated with `duplicate_class`/`phash_hamming`; a `provenance` block (owner, parent "
        "quilt) is returned for the strongest duplicate. Fields: `top_k` (default 10), "
        "`exclude_nsfw` (default true), `verify_duplicates` (default true)."
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
    verify_duplicates: bool = Form(True),
) -> ReverseImageResponse:
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
    base = _similar_images_from_embedding(
        store, qemb, top_k=top_k, exclude_nsfw=exclude_nsfw, query_kind="image"
    )
    results = base.results if hasattr(base, "results") else base.get("results", [])
    summary = {"query_phash": None, "duplicates_found": 0,
               "exact_duplicate_blob_id": None, "provenance": None}
    if verify_duplicates:
        summary = await run_in_threadpool(_harden_reverse_results, results, image_bytes)
    return ReverseImageResponse(
        results=results, total=len(results),
        query_phash=summary["query_phash"],
        duplicates_found=summary["duplicates_found"],
        exact_duplicate_blob_id=summary["exact_duplicate_blob_id"],
        provenance=summary["provenance"],
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
            current_epoch = int(epoch) if epoch is not None else get_current_epoch()
        except Exception:
            current_epoch = get_current_epoch()

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
            current_epoch = int(epoch) if epoch is not None else get_current_epoch()
        except Exception:
            current_epoch = get_current_epoch()

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
