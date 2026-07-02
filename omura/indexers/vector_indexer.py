"""
Vector Indexer — Process 2

Responsibilities
----------------
1. Drain index_queue from blob_catalog.sqlite in batches
2. Fetch full blob bytes from the Walrus aggregator
3. Embed with the configured multimodal embedding model (default: Qwen3-VL-Embedding-8B)
4. Store in FAISS vector index
5. Score NSFW (0–100 tag score vs text prototypes; flag when score > OMURA_NSFW_TAG_SCORE_MIN, default 85)
6. Assign semantic cluster (k-means, rebuilt periodically)
7. Update blob record: indexed=1, nsfw_score, cluster_id
8. Never stop — queue is drained continuously; if empty, sleeps then retries

Run standalone:
  python -m omura.indexers.vector_indexer

Or started via the API server (OMURA_ENABLE_INDEXER=true).
"""

from __future__ import annotations

import os
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import requests
import requests.adapters

from omura.utils.aggregator_pool import get_pool
from omura.utils.blob_catalog import (
    CATALOG_DB_PATH,
    dequeue_blobs,
    get_blob_metadata,
    init_catalog_db,
    mark_failed_batch,
    mark_indexed_batch,
    requeue_blobs,
    update_cluster,
)
from omura.utils.imagebind_embeddings import (
    generate_image_embedding,
    generate_text_embedding,
    get_nsfw_embeddings,
    initialize_embedding_model,
    is_model_ready,
    is_nsfw_from_tag_score,
    nsfw_similarity_score_0_100,
)
from omura.utils.vector_store import VectorStore

# ── Configuration ──────────────────────────────────────────────────────────────

BATCH_SIZE = int(os.getenv("OMURA_INDEXER_BATCH_SIZE", "32"))
MAX_FETCH_WORKERS = int(os.getenv("OMURA_INDEXER_FETCH_WORKERS", "16"))
POLL_INTERVAL = int(
    os.getenv("OMURA_INDEXER_POLL_INTERVAL", "10")
)  # seconds when queue empty
CLUSTER_INTERVAL = int(os.getenv("OMURA_CLUSTER_INTERVAL_SECONDS", "3600"))  # 1 hour
N_CLUSTERS = int(os.getenv("OMURA_N_CLUSTERS", "30"))
MAX_RETRIES = int(os.getenv("OMURA_INDEXER_MAX_RETRIES", "3"))

DEFAULT_AGGREGATOR = "https://agrregator.omura.fun"
AGGREGATOR_URL = os.getenv("WALRUS_AGGREGATOR_URL", DEFAULT_AGGREGATOR).rstrip("/")

# Per-process retry budget for blobs that repeatedly fail embedding.
_failure_counts: Dict[str, int] = {}

# ── Thread-local HTTP session ──────────────────────────────────────────────────

_tls = threading.local()


def _session() -> requests.Session:
    if not hasattr(_tls, "s"):
        s = requests.Session()
        a = requests.adapters.HTTPAdapter(
            pool_connections=4,
            pool_maxsize=16,
            max_retries=requests.adapters.Retry(
                total=2,
                backoff_factor=0.5,
                status_forcelist=[502, 503, 504],
                allowed_methods=["GET"],
            ),
        )
        s.mount("https://", a)
        s.mount("http://", a)
        _tls.s = s
    return _tls.s


# ── Fetch ──────────────────────────────────────────────────────────────────────


def _fetch_blob(blob_id: str) -> Optional[bytes]:
    """Fetch blob bytes via the aggregator pool (load-balanced)."""
    resp, _ = get_pool().get(f"/v1/blobs/{blob_id}", session=_session(), timeout=60)
    if resp is None or resp.status_code != 200:
        return None
    return resp.content


# ── NSFW scoring ───────────────────────────────────────────────────────────────


def _nsfw_score(embedding: np.ndarray) -> float:
    """0–100 NSFW alignment score (same scale as ``/search``); higher = closer to NSFW prototypes."""
    nsfw_vecs = get_nsfw_embeddings()
    if not nsfw_vecs:
        return 0.0
    return float(nsfw_similarity_score_0_100(embedding, nsfw_vecs))


# ── Per-blob worker ────────────────────────────────────────────────────────────


def _index_one(blob_id: str) -> Optional[Dict[str, Any]]:
    """
    Fetch + embed one image blob.
    Returns a result dict or None on failure.
    """
    content = _fetch_blob(blob_id)
    if not content:
        return None

    try:
        embedding = generate_image_embedding(content, blob_id=blob_id)
    except RuntimeError as exc:
        err = str(exc).lower()
        if "illegal" in err and "memory" in err:
            print("[VectorIndexer] FATAL: CUDA Illegal Memory Access. Exiting to force restart.")
            os._exit(1)
        print(f"[VectorIndexer] {blob_id}: CUDA error: {exc}")
        return None
    except Exception as exc:
        print(f"[VectorIndexer] {blob_id}: embedding failed: {exc}")
        return None

    if embedding is None:
        return None

    # NSFW: prefer the VLM labeler (actually looks at the image); fall back to the
    # legacy embedding-cosine score when the VLM is disabled/unavailable.
    from omura.utils.nsfw_labeler import classify_nsfw

    vlm = classify_nsfw(content)
    if vlm is not None:
        score, is_nsfw, _label = vlm
    else:
        score = _nsfw_score(embedding)
        is_nsfw = is_nsfw_from_tag_score(score)

    # Generate image caption — check DB cache first (blob_id is content-addressed,
    # so a stored caption is permanent and never needs regeneration). Matches prod.
    import sqlite3
    from omura.utils.captioning import generate_caption
    caption: Optional[str] = None
    try:
        with sqlite3.connect(str(CATALOG_DB_PATH), timeout=10) as _conn:
            row = _conn.execute(
                "SELECT caption FROM blobs WHERE blob_id=? AND caption IS NOT NULL AND caption!=''",
                (blob_id,),
            ).fetchone()
            if row:
                caption = row[0]
    except Exception:
        pass
    if not caption:
        caption = generate_caption(content)

    return {
        "blob_id": blob_id,
        "embedding": embedding,
        "nsfw_score": score,
        "is_nsfw": is_nsfw,
        "caption": caption,
    }


# ── Clustering ─────────────────────────────────────────────────────────────────


def _rebuild_clusters(store: VectorStore, db_path: Path) -> None:
    """
    Run k-means over all indexed embeddings and save cluster assignments.
    Uses sklearn if available, otherwise skips (non-fatal).
    """
    try:
        from sklearn.cluster import MiniBatchKMeans
    except ImportError:
        print("[VectorIndexer] sklearn not available — skipping cluster rebuild")
        return

    with store._lock:
        # embeddings_dict is {blob_id: np.ndarray}; values are arrays, not dicts.
        items = (
            list(store.embeddings_dict.items())
            if hasattr(store, "embeddings_dict")
            else []
        )

    if len(items) < N_CLUSTERS * 2:
        print(
            f"[VectorIndexer] Not enough embeddings ({len(items)}) for {N_CLUSTERS} clusters, skipping"
        )
        return

    print(
        f"[VectorIndexer] Rebuilding {N_CLUSTERS} clusters over {len(items)} embeddings..."
    )
    blob_ids = [b for b, _ in items]
    X = np.vstack([np.asarray(e, dtype=np.float32).flatten() for _, e in items])

    km = MiniBatchKMeans(
        n_clusters=N_CLUSTERS, random_state=42, n_init=3, batch_size=2048
    )
    labels = km.fit_predict(X)

    # Find representative (closest to centroid) blob per cluster
    centroid_blobs = {}
    for cid in range(N_CLUSTERS):
        mask = labels == cid
        if not mask.any():
            continue
        members = X[mask]
        centroid = km.cluster_centers_[cid]
        dists = np.linalg.norm(members - centroid, axis=1)
        rep_idx = np.where(mask)[0][np.argmin(dists)]
        centroid_blobs[cid] = blob_ids[rep_idx]

    # Label clusters with semantic categories via text embeddings
    CATEGORIES = [
        "nature landscape outdoor scenery",
        "people portrait face",
        "urban city architecture building",
        "food cuisine restaurant",
        "animals pets wildlife",
        "art illustration design graphic",
        "technology devices electronics",
        "sports fitness activity",
        "vehicles cars transportation",
        "abstract texture pattern",
    ]
    cat_labels = _assign_cluster_labels(km.cluster_centers_, CATEGORIES)

    # Write to DB
    cluster_sizes = {cid: int((labels == cid).sum()) for cid in range(N_CLUSTERS)}
    for cid in range(N_CLUSTERS):
        update_cluster(
            cluster_id=cid,
            label=cat_labels.get(cid, f"cluster_{cid}"),
            size=cluster_sizes.get(cid, 0),
            centroid_blob_id=centroid_blobs.get(cid),
            db_path=db_path,
        )

    # Batch-update cluster_id on blobs
    import sqlite3

    rows = [(int(labels[i]), blob_ids[i]) for i in range(len(blob_ids))]
    try:
        with sqlite3.connect(str(db_path), timeout=30) as conn:
            conn.executemany("UPDATE blobs SET cluster_id=? WHERE blob_id=?", rows)
            conn.commit()
    except Exception as exc:
        print(f"[VectorIndexer] Cluster DB write error (non-fatal): {exc}")

    print(f"[VectorIndexer] Cluster rebuild complete. {N_CLUSTERS} clusters assigned.")


def _assign_cluster_labels(
    centroids: np.ndarray, categories: List[str]
) -> Dict[int, str]:
    """Assign each cluster the category label with highest cosine sim to its centroid."""
    try:
        cat_embs = []
        for cat in categories:
            emb = generate_text_embedding(cat, is_document=False)
            if emb is not None:
                cat_embs.append(np.asarray(emb, dtype=np.float32))
            else:
                cat_embs.append(None)

        labels: Dict[int, str] = {}
        for cid, centroid in enumerate(centroids):
            best_score = -1.0
            best_label = f"cluster_{cid}"
            for cat, cat_emb in zip(categories, cat_embs):
                if cat_emb is None:
                    continue
                score = float(
                    np.dot(centroid / (np.linalg.norm(centroid) + 1e-8), cat_emb)
                )
                if score > best_score:
                    best_score = score
                    best_label = cat.split()[0]  # first word as label
            labels[cid] = best_label
        return labels
    except Exception as exc:
        print(f"[VectorIndexer] Label assignment error (non-fatal): {exc}")
        return {}


# ── Model readiness ────────────────────────────────────────────────────────────


def _wait_for_model(poll: float = 30.0) -> None:
    while not is_model_ready():
        print(f"[VectorIndexer] Waiting for embedding model ({poll:.0f}s)...")
        time.sleep(poll)
        try:
            from omura.utils.imagebind_embeddings import ensure_model_loaded

            ensure_model_loaded()
        except Exception:
            pass


# ── Main batch loop ────────────────────────────────────────────────────────────


def _process_queue_batch(
    store: VectorStore,
    db_path: Path,
) -> int:
    """
    Drain one batch from the queue. Returns count of successfully indexed blobs.
    """
    _wait_for_model()

    blob_ids = dequeue_blobs(limit=BATCH_SIZE, db_path=db_path)
    if not blob_ids:
        return 0

    results: List[Dict[str, Any]] = []
    failed_ids: List[str] = []

    with ThreadPoolExecutor(max_workers=MAX_FETCH_WORKERS) as ex:
        futs = {ex.submit(_index_one, bid): bid for bid in blob_ids}
        for fut in as_completed(futs):
            bid = futs[fut]
            try:
                res = fut.result()
                if res is not None:
                    results.append(res)
                else:
                    failed_ids.append(bid)
            except Exception as exc:
                print(f"[VectorIndexer] Worker error ({bid}): {exc}")
                failed_ids.append(bid)

    # Add embeddings to vector store
    for res in results:
        try:
            store.add(
                embedding=res["embedding"],
                blob_id=res["blob_id"],
                mime_type="image",
                size=0,
                extension="",
                kind="image",
                is_nsfw=res["is_nsfw"],
            )
            _failure_counts.pop(res["blob_id"], None)
        except Exception as exc:
            print(f"[VectorIndexer] Store.add error ({res['blob_id']}): {exc}")
            failed_ids.append(res["blob_id"])
            results = [r for r in results if r["blob_id"] != res["blob_id"]]

    # Bulk update DB
    mark_indexed_batch(
        [
            {
                "blob_id": r["blob_id"],
                "is_nsfw": r["is_nsfw"],
                "nsfw_score": r["nsfw_score"],
                "caption": r.get("caption"),
            }
            for r in results
        ],
        db_path=db_path,
    )

    # Re-queue transient failures; permanently skip blobs beyond retry budget.
    retry_ids: List[str] = []
    permanent_fail_ids: List[str] = []
    for bid in failed_ids:
        cnt = _failure_counts.get(bid, 0) + 1
        _failure_counts[bid] = cnt
        if cnt >= MAX_RETRIES:
            permanent_fail_ids.append(bid)
            _failure_counts.pop(bid, None)
        else:
            retry_ids.append(bid)

    if retry_ids:
        requeue_blobs(retry_ids, db_path=db_path)
    if permanent_fail_ids:
        mark_failed_batch(permanent_fail_ids, status="embed_failed", db_path=db_path)
        print(
            f"[VectorIndexer] Permanently skipped {len(permanent_fail_ids)} blobs "
            f"after {MAX_RETRIES} retries"
        )

    indexed = len(results)
    if indexed:
        nsfw_count = sum(1 for r in results if r["is_nsfw"])
        print(
            f"[VectorIndexer] Indexed {indexed} images "
            f"({nsfw_count} NSFW) | {len(retry_ids)} retry | {len(permanent_fail_ids)} skipped"
        )
    return indexed


def run_vector_indexer(
    store: VectorStore,
    db_path: Path = CATALOG_DB_PATH,
) -> None:
    """
    Never-stopping vector indexer service.
    Drains index_queue continuously; sleeps POLL_INTERVAL when empty.
    Rebuilds semantic clusters every CLUSTER_INTERVAL seconds.
    """
    init_catalog_db(db_path)
    print(
        f"[VectorIndexer] Starting vector indexer that's crunching the embeddings. model=OMURA_EMBEDDING_MODEL "
        f"batch={BATCH_SIZE} fetch_workers={MAX_FETCH_WORKERS}"
    )

    initialize_embedding_model()

    last_save = time.monotonic()
    last_cluster = time.monotonic()
    save_interval = int(os.getenv("OMURA_SAVE_INTERVAL_SECONDS", "300"))
    total_indexed = 0

    while True:
        try:
            indexed = _process_queue_batch(store, db_path)
            total_indexed += indexed

            now = time.monotonic()

            if indexed == 0:
                time.sleep(POLL_INTERVAL)
            else:
                # Save periodically
                if now - last_save >= save_interval:
                    store.save(create_backup=False)
                    print(
                        f"[VectorIndexer] Saved index ({total_indexed} total indexed)"
                    )
                    last_save = now

            # Rebuild clusters periodically
            if (
                now - last_cluster >= CLUSTER_INTERVAL
                and total_indexed >= N_CLUSTERS * 2
            ):
                try:
                    _rebuild_clusters(store, db_path)
                except Exception as exc:
                    print(f"[VectorIndexer] Cluster rebuild error (non-fatal): {exc}")
                last_cluster = now

        except KeyboardInterrupt:
            print("[VectorIndexer] Interrupted. Saving index...")
            store.save(create_backup=True)
            break
        except Exception as exc:
            print(f"[VectorIndexer] Unexpected error, continuing in 30s: {exc}")
            traceback.print_exc()
            time.sleep(30)


if __name__ == "__main__":
    from omura.utils.vector_store import VectorStore

    _store = VectorStore()
    _store.load()
    run_vector_indexer(_store)
