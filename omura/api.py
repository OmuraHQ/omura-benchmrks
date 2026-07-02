"""FastAPI application setup."""

from __future__ import annotations

import asyncio
import os
import threading
import fcntl
import tempfile
import time
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from omura.routes import admin as admin_routes
from omura.routes import blobs, search
from omura.routes import dashboard as dashboard_routes
from omura.utils.blockberry import get_current_epoch
from omura.utils.aggregator_pool import get_pool
from omura.utils.imagebind_embeddings import initialize_embedding_model
from omura.utils.vector_store import VectorStore

app = FastAPI(
    title="Omura Search API",
    description="""
## Multimodal Search Engine for Walrus Protocol

Omura provides GPU-accelerated vector similarity search for Walrus protocol blobs using Qwen3-VL-Embedding-8B (default) and FAISS.

### Architecture

Two background processes run alongside the API:
- **Cataloger** (Process 1): Discovers all Walrus blobs, detects file types, populates `blob_catalog.sqlite`
- **Vector Indexer** (Process 2): Embeds content via the configured model (default: Qwen3-VL-Embedding-8B), stores in FAISS, scores NSFW, assigns semantic clusters

### Features

* **Multimodal Search**: Natural language and image-driven retrieval using a unified embedding model
* **Multilingual Retrieval**: Cross-lingual semantic retrieval
* **Semantic Clusters**: K-means clustering with automatic category labelling
* **NSFW Detection**: Automatic scoring + manual override via dashboard
* **Dashboard API**: `/dashboard/clusters`, `/dashboard/stats`, `/dashboard/nsfw`
* **Reverse Image Search**: Upload an image to find similar indexed blobs

### Quick Start

1. Start the API server: `python main.py` (indexer runs automatically in background)
2. Wait for images to be indexed (check `/search/stats`)
3. Search using text queries via `/search/` or upload an image via `/search/reverse-image`

### API Documentation

- **Swagger UI**: Available at `/docs`
- **ReDoc**: Available at `/redoc`
- **OpenAPI Schema**: Available at `/openapi.json`
    """,
    version="0.1.0",
    contact={
        "name": "Omura Project",
    },
    license_info={
        "name": "MIT",
    },
    tags_metadata=[
        {
            "name": "search",
            "description": "Vector similarity search operations. Search for images using text queries.",
        },
        {
            "name": "blobs",
            "description": "Walrus blob metadata and retrieval operations.",
        },
    ],
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*",
        "https://omura-frontend.vercel.app",
        "https://omura.fun",
        "https://www.omura.fun",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(search.router)
app.include_router(blobs.router)
app.include_router(dashboard_routes.router)
app.include_router(admin_routes.router)

# Background thread handles
_cataloger_thread: Optional[threading.Thread] = None
_indexer_thread: Optional[threading.Thread] = None
_bellyseal_thread: Optional[threading.Thread] = None
_walrus_indexer_thread: Optional[threading.Thread] = None
_save_task: Optional[asyncio.Task] = None
_expire_flush_task: Optional[asyncio.Task] = None

# Shared lock for all live indexers writing to the same VectorStore
_live_indexer_store_lock = threading.Lock()

# Global vector store (shared between API and vector indexer)
_shared_vector_store = None


def _start_daemon(target, name: str) -> threading.Thread:
    t = threading.Thread(target=target, name=name, daemon=True)
    t.start()
    return t


def _run_cataloger() -> None:
    """Process 1: blob discovery + type detection. Never stops."""
    from omura.cataloger import run_cataloger
    from omura.utils.blob_catalog import CATALOG_DB_PATH
    while True:
        try:
            run_cataloger(CATALOG_DB_PATH)
        except Exception as exc:
            import traceback
            print(f"[Cataloger] Crashed, restarting in 30s: {exc}")
            traceback.print_exc()
            time.sleep(30)


def _run_vector_indexer() -> None:
    """Process 2: queue → embed → FAISS. Never stops."""
    global _shared_vector_store
    from omura.indexers.vector_indexer import run_vector_indexer
    from omura.utils.blob_catalog import CATALOG_DB_PATH
    while True:
        try:
            run_vector_indexer(_shared_vector_store, CATALOG_DB_PATH)
        except Exception as exc:
            import traceback
            print(f"[VectorIndexer] Crashed, restarting in 30s: {exc}")
            traceback.print_exc()
            time.sleep(30)


async def periodic_save() -> None:
    """Periodic save task (like cron) that saves the vector store with backups.
    
    This task is designed to be crash-resistant and non-blocking:
    - Does NOT build the index (indexer handles that)
    - Only saves embeddings and metadata (lightweight operation)
    - Has extensive error handling to prevent crashes
    - Uses a longer interval to avoid conflicts with indexer
    """
    global _shared_vector_store
    # Save interval in seconds (default: 10 minutes - longer to avoid conflicts)
    save_interval = int(os.getenv("OMURA_SAVE_INTERVAL_SECONDS", "600"))
    # Expired blob prune interval in seconds (default: 1 hour)
    prune_interval = int(os.getenv("OMURA_PRUNE_INTERVAL_SECONDS", "3600"))
    last_prune_ts = 0.0
    
    while True:
        try:
            await asyncio.sleep(save_interval)
            
            if _shared_vector_store is None:
                print("[PeriodicSave] Vector store not initialized, skipping save")
                continue
            
            # Check if there's anything to save
            if _shared_vector_store.size() == 0:
                print("[PeriodicSave] No embeddings to save, skipping")
                continue

            now = time.monotonic()
            if now - last_prune_ts >= prune_interval:
                try:
                    current_epoch = get_current_epoch()

                    prune_summary = _shared_vector_store.prune_expired(
                        current_epoch=current_epoch, dry_run=False
                    )
                    print(
                        "[PeriodicSave] Prune summary: "
                        f"before={prune_summary['total_before']}, "
                        f"expired={prune_summary['expired_count']}, "
                        f"kept={prune_summary['kept_count']}, "
                        f"epoch={current_epoch}"
                    )
                    last_prune_ts = now
                except Exception as prune_error:
                    print(f"[PeriodicSave] Error during prune (non-fatal): {prune_error}")
                    import traceback

                    traceback.print_exc()
                    print("[PeriodicSave] Continuing without prune this cycle")
            
            print(f"[PeriodicSave] Starting save (size: {_shared_vector_store.size()} embeddings)...")
            
            try:
                # Only save embeddings and metadata - DO NOT build index here
                # The indexer handles index building, and we don't want to conflict
                # Save without building index (lightweight operation)
                _shared_vector_store.save(create_backup=True)
                print(f"[PeriodicSave] Successfully saved {_shared_vector_store.size()} embeddings")
                
            except Exception as save_error:
                # Log error but don't crash - this is critical for stability
                print(f"[PeriodicSave] Error during save (non-fatal): {save_error}")
                import traceback
                traceback.print_exc()
                print("[PeriodicSave] Will retry on next interval")
                # Continue to next iteration - don't break the loop
                
        except asyncio.CancelledError:
            # Graceful shutdown - try one final save
            print("[PeriodicSave] Shutdown requested, attempting final save...")
            if _shared_vector_store is not None and _shared_vector_store.size() > 0:
                try:
                    _shared_vector_store.save(create_backup=True)
                    print("[PeriodicSave] Final save completed")
                except Exception as e:
                    print(f"[PeriodicSave] Error in final save (non-fatal): {e}")
            break
            
        except Exception as e:
            # Catch-all for any unexpected errors - prevent task from crashing
            print(f"[PeriodicSave] Unexpected error (non-fatal): {e}")
            import traceback
            traceback.print_exc()
            print("[PeriodicSave] Task will continue after delay")
            # Wait a bit longer before retrying after an unexpected error
            try:
                await asyncio.sleep(120)  # Wait 2 minutes before retrying
            except asyncio.CancelledError:
                break

async def daily_expire_flush() -> None:
    """Daily cron: remove expired Walrus blobs from the vector store + catalog DB.

    Runs once at startup (after a short warm-up) and then every 24 h at midnight UTC.
    Fetches the current Walrus storage epoch, prunes any blob whose ``expiresAt``
    metadata is <= current epoch, marks them ``is_active=0`` in the catalog, then
    saves the updated index without creating a backup.

    Disable via ``OMURA_EXPIRE_FLUSH_ENABLED=false``.
    Schedule override via ``OMURA_EXPIRE_FLUSH_INTERVAL_SECS`` (default: 86400).
    """
    import sqlite3 as _sqlite3
    from omura.utils.blob_catalog import CATALOG_DB_PATH as _CATALOG_DB_PATH

    enabled = os.getenv("OMURA_EXPIRE_FLUSH_ENABLED", "true").lower() == "true"
    if not enabled:
        print("[ExpireFlush] disabled via OMURA_EXPIRE_FLUSH_ENABLED=false")
        return

    interval = int(os.getenv("OMURA_EXPIRE_FLUSH_INTERVAL_SECS", "86400"))  # 24 h

    # Initial warm-up delay so the store is fully loaded before first prune
    warmup = int(os.getenv("OMURA_EXPIRE_FLUSH_WARMUP_SECS", "300"))  # 5 min
    print(f"[ExpireFlush] Scheduled — first run in {warmup}s, then every {interval}s")

    try:
        await asyncio.sleep(warmup)
    except asyncio.CancelledError:
        return

    while True:
        try:
            global _shared_vector_store
            if _shared_vector_store is None or _shared_vector_store.size() == 0:
                print("[ExpireFlush] Store not ready, skipping this run")
            else:
                # 1. Get current Walrus epoch
                try:
                    current_epoch = get_current_epoch()
                except Exception as e:
                    print(f"[ExpireFlush] Could not fetch epoch: {e} — skipping")
                    current_epoch = None

                if current_epoch is not None:
                    print(f"[ExpireFlush] Running — epoch={current_epoch} store_size={_shared_vector_store.size()}")

                    # 2. Prune vector store
                    summary = _shared_vector_store.prune_expired(
                        current_epoch=current_epoch, dry_run=False
                    )
                    expired_ids = summary.get("expired_count", 0)
                    print(
                        f"[ExpireFlush] Pruned {expired_ids} expired blobs "
                        f"({summary.get('kept_count')} kept)"
                    )

                    # 3. Mark expired blobs inactive in catalog DB
                    if expired_ids > 0:
                        try:
                            with _sqlite3.connect(str(_CATALOG_DB_PATH), timeout=15) as conn:
                                conn.execute(
                                    """
                                    UPDATE blobs
                                    SET is_active = 0,
                                        status = 'expired',
                                        last_updated_at = datetime('now')
                                    WHERE CAST(end_epoch AS INTEGER) <= ?
                                      AND is_active = 1
                                    """,
                                    (current_epoch,),
                                )
                                affected = conn.execute("SELECT changes()").fetchone()[0]
                                conn.commit()
                            print(f"[ExpireFlush] Marked {affected} blobs inactive in catalog")
                        except Exception as db_err:
                            print(f"[ExpireFlush] Catalog update error (non-fatal): {db_err}")

                        # 4. Save updated index
                        try:
                            _shared_vector_store.save(create_backup=False)
                            print(f"[ExpireFlush] Index saved — new size: {_shared_vector_store.size()}")
                        except Exception as save_err:
                            print(f"[ExpireFlush] Save error (non-fatal): {save_err}")

        except asyncio.CancelledError:
            print("[ExpireFlush] Shutdown — exiting.")
            break
        except Exception as e:
            print(f"[ExpireFlush] Unexpected error (non-fatal): {e}")

        # Sleep until next run (default: 24 h)
        try:
            await asyncio.sleep(interval)
        except asyncio.CancelledError:
            break


def acquire_lock(lock_name: str) -> Optional[object]:
    """Acquire a file lock to ensure single execution across workers.

    OMURA_LOCK_NAMESPACE lets multiple independent instances (e.g. prod + a
    staging instance on another port) run their own background indexers on the
    same host without contending for the same /tmp lock. Empty (default) keeps
    the original ``omura_<name>.lock`` path so existing deployments are unchanged.
    """
    ns = os.getenv("OMURA_LOCK_NAMESPACE", "").strip()
    prefix = f"omura_{ns}_" if ns else "omura_"
    lock_file = os.path.join(tempfile.gettempdir(), f"{prefix}{lock_name}.lock")
    try:
        fp = open(lock_file, 'w')
        fcntl.lockf(fp, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return fp
    except IOError:
        return None


_leader_lock_fp = None
_saver_lock_fp = None


@app.on_event("startup")
async def startup_event() -> None:
    global _cataloger_thread, _indexer_thread, _save_task, _expire_flush_task
    global _shared_vector_store, _leader_lock_fp, _saver_lock_fp

    # ── Aggregator pool startup ping (background, non-blocking) ──
    def _ping_pool() -> None:
        try:
            get_pool().startup_ping()
        except Exception as exc:
            print(f"[API] Aggregator pool ping failed: {exc}")
    threading.Thread(target=_ping_pool, daemon=True, name="agg-pool-ping").start()

    # ── Vector store ──
    print("[API] Initializing vector store...")
    try:
        _shared_vector_store = VectorStore()
        _shared_vector_store.load()
        print(f"[API] Vector store: {_shared_vector_store.size()} embeddings")
        search.set_shared_vector_store(_shared_vector_store)
        dashboard_routes.set_shared_vector_store(_shared_vector_store)
    except Exception as e:
        import traceback
        print(f"[API] Vector store init failed: {e}")
        traceback.print_exc()

    # ── Audio store (CLAP, 512-d) — separate index for text→audio search ──
    if os.getenv("OMURA_AUDIO_BACKEND", "").strip().lower() == "clap":
        try:
            from pathlib import Path as _Path
            from omura.utils.clap_embeddings import CLAP_DIM
            audio_dir = _Path(os.getenv("OMURA_AUDIO_VECTOR_STORE_DIR", "data/vector_index_clap"))
            audio_store = VectorStore(index_path=audio_dir / "vector_index.faiss", embedding_dim=CLAP_DIM)
            audio_store.load()
            search.set_shared_audio_vector_store(audio_store)
            print(f"[API] Audio (CLAP) store: {audio_store.size()} embeddings")

            # Pre-warm the CLAP text encoder in the background so the first /search/audio
            # query doesn't pay a ~6s cold-start model load.
            def _warm_clap():
                try:
                    from omura.utils import clap_embeddings as _clap
                    _clap.embed_text("warmup")
                    print("[API] CLAP text encoder warmed up")
                except Exception as _e:
                    print(f"[API] CLAP warmup failed: {_e}")
            import threading as _t
            _t.Thread(target=_warm_clap, daemon=True).start()
        except Exception as e:
            print(f"[API] Audio store init failed: {e}")

    # ── Video store (omura-embed-video / InternVideo2, 768-d) — text→video search ──
    if os.getenv("OMURA_VIDEO_BACKEND", "").strip().lower() in ("internvideo2", "omura_embed_video"):
        try:
            from pathlib import Path as _Path
            video_dim = int(os.getenv("OMURA_VIDEO_EMBEDDING_DIM", "768"))
            video_dir = _Path(os.getenv("OMURA_VIDEO_VECTOR_STORE_DIR", "data/vector_index_iv2"))
            video_store = VectorStore(index_path=video_dir / "vector_index.faiss", embedding_dim=video_dim)
            video_store.load()
            search.set_shared_video_vector_store(video_store)
            print(f"[API] Video store: {video_store.size()} embeddings")
        except Exception as e:
            print(f"[API] Video store init failed: {e}")

    # ── Catalog DB ──
    from omura.utils.blob_catalog import init_catalog_db, CATALOG_DB_PATH
    init_catalog_db(CATALOG_DB_PATH)

    # ── Embedding model preload policy ──
    # API-only QoS nodes should not spend GPU/memory loading the embedder unless explicitly requested.
    preload_policy = os.getenv("OMURA_PRELOAD_EMBEDDING_MODEL", "auto").strip().lower()
    role = os.getenv("OMURA_INSTANCE_ROLE", "all").strip().lower()
    enable_indexer = os.getenv("OMURA_ENABLE_INDEXER", "true").lower() == "true"
    should_preload = (
        preload_policy == "true"
        or (preload_policy == "auto" and role != "api" and enable_indexer)
    )
    if should_preload:
        print("[API] Starting embedding model load in background...")
        threading.Thread(target=_load_model_bg, daemon=True, name="model-loader").start()
    else:
        print("[API] Embedding model preload skipped for API QoS mode")

    # ── Periodic save + daily expire flush (one worker only) ──
    _saver_lock_fp = acquire_lock("saver")
    if _saver_lock_fp:
        _save_task = asyncio.create_task(periodic_save())
        print("[API] Periodic save task started")
        _expire_flush_task = asyncio.create_task(daily_expire_flush())
        print("[API] Daily expire-flush task scheduled (first run in 5 min)")
    else:
        print("[API] Periodic save: another worker holds lock, skipping")

    # ── Background processes (one worker only) ──
    _leader_lock_fp = acquire_lock("leader")
    if _leader_lock_fp:
        if os.getenv("OMURA_ENABLE_CATALOGER", "true").lower() == "true":
            _cataloger_thread = _start_daemon(_run_cataloger, "cataloger")
            print("[API] Cataloger (Process 1) started")

        if os.getenv("OMURA_ENABLE_INDEXER", "true").lower() == "true":
            _indexer_thread = _start_daemon(_run_vector_indexer, "vector-indexer")
            print("[API] Vector Indexer (Process 2) started")

        if os.getenv("OMURA_BELLYSEAL_ENABLED", "true").lower() == "true":
            from omura.indexers.bellyseal_indexer import start_bellyseal_indexer_thread
            _bellyseal_thread = start_bellyseal_indexer_thread(
                _shared_vector_store, _live_indexer_store_lock
            )
            if _bellyseal_thread:
                print("[API] BellySeal live indexer started")

        if os.getenv("OMURA_WALRUS_INDEXER_ENABLED", "true").lower() == "true":
            from omura.indexers.walrus_blob_indexer import start_walrus_blob_indexer_thread
            _walrus_indexer_thread = start_walrus_blob_indexer_thread(
                _shared_vector_store, _live_indexer_store_lock
            )
            if _walrus_indexer_thread:
                print("[API] Walrus blob indexer started")
    else:
        print("[API] Background processes: another worker is leader, skipping")


def _load_model_bg() -> None:
    try:
        initialize_embedding_model()
        from omura.utils.imagebind_embeddings import MODEL_NAME
        print(f"[API] Embedding model ready: {MODEL_NAME}")
    except Exception as exc:
        print(f"[API] Embedding model load failed (search/indexing will degrade): {exc}")


@app.on_event("shutdown")
async def shutdown_event() -> None:
    global _save_task, _shared_vector_store

    if _save_task:
        _save_task.cancel()
        try:
            await _save_task
        except asyncio.CancelledError:
            pass

    if _shared_vector_store:
        try:
            _shared_vector_store.save(create_backup=True)
            print(f"[API] Final save: {_shared_vector_store.size()} embeddings")
        except Exception as e:
            print(f"[API] Final save error: {e}")


@app.get(
    "/",
    summary="API Information",
    description="Get basic information about the Omura Search API and available endpoints.",
    tags=["general"],
)
async def root() -> dict:
    """
    Root endpoint providing API information.

    Returns basic API metadata and links to available endpoints.
    """
    return {
        "name": "Omura Search API",
        "version": "0.1.0",
        "description": "Multimodal search engine for Walrus protocol blobs",
        "endpoints": {
            "search": {
                "url": "/search",
                "method": "POST",
                "description": "Unified text/image search (JSON or multipart form)",
            },
            "stats": {
                "url": "/search/stats",
                "method": "GET",
                "description": "Vector store statistics",
            },
            "reverse_image": {
                "url": "/search/reverse-image",
                "method": "POST",
                "description": "Reverse image search from uploaded file (multipart form `file`)",
            },
            "blob": {
                "url": "/blob/{blob_id}",
                "method": "GET",
                "description": "Proxy blob file content with correct MIME type (use in <img src> tags)",
            },
        },
        "docs": {
            "swagger": "/docs",
            "redoc": "/redoc",
            "openapi": "/openapi.json",
        },
    }


@app.get(
    "/health",
    summary="Health Check",
    description="Health check endpoint for monitoring and load balancers.",
    tags=["general"],
    responses={
        200: {
            "description": "Service is healthy",
            "content": {
                "application/json": {
                    "example": {"status": "ok", "service": "omura-api"}
                }
            },
        },
    },
)
async def health() -> dict:
    """
    Health check endpoint.

    Returns the health status of the API service.
    """
    return {"status": "ok", "service": "omura-api"}
