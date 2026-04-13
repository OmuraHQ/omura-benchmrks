"""FastAPI application setup."""

from __future__ import annotations

import asyncio
import os
import threading
import fcntl
import tempfile
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Indexer will be imported dynamically in run_indexer()
from omura.routes import blobs, search
from omura.utils.imagebind_embeddings import initialize_imagebind
from omura.utils.vector_store import VectorStore

app = FastAPI(
    title="Omura Search API",
    description="""
## Image Search Engine for Walrus Protocol

Omura provides GPU-accelerated vector similarity search for Walrus protocol blobs using SigLIP-2 embeddings and FAISS.

### Features

* **Text-to-Image Search**: Search images using natural language queries
* **GPU-Accelerated**: Uses FAISS for fast vector similarity search
* **SigLIP-2 Embeddings**: Powered by Google's SigLIP-2 ViT-B-16 model (768-dim)
* **NSFW Detection**: Automatic flagging using zero-shot classification with explicit terms
* **Real-time Indexing**: Background indexer continuously processes new blobs

### Quick Start

1. Start the API server: `python main.py` (indexer runs automatically in background)
2. Wait for images to be indexed (check `/search/stats`)
3. Search using text queries via `/search/`

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
app.include_router(blobs.router)  # Blob proxy endpoint at /blob/{blob_id}

# Daemon thread for running the indexer (isolated from API)
_indexer_thread: Optional[threading.Thread] = None
_save_task: Optional[asyncio.Task] = None

# Global vector store instance (shared between API and indexer)
_shared_vector_store = None


def run_indexer() -> None:
    """Run the indexer in a separate thread using the shared vector store.
    
    This function is designed to run in isolation - any exceptions are caught
    and logged, but never propagated to the API. The indexer will automatically
    restart if it crashes.
    """
    global _shared_vector_store
    import time
    
    # Get configuration from environment
    batch_size = int(os.getenv("OMURA_INDEXER_BATCH_SIZE", "100"))
    max_batches = os.getenv("OMURA_INDEXER_MAX_BATCHES")
    max_batches = int(max_batches) if max_batches else None
    
    restart_delay = 60  # Wait 60 seconds before restarting after a crash
    
    # Continuous loop with automatic restart on crash
    while True:
        try:
            print(f"[Indexer] Starting background indexer with shared vector store...")
            # Pass the shared vector store to the indexer
            from omura.indexers.multimodal_indexer import run_indexer_with_store
            run_indexer_with_store(
                _shared_vector_store,
                batch_size=batch_size,
                max_batches=max_batches,
            )
            # If indexer exits normally (shouldn't happen with continuous loop)
            print(f"[Indexer] Indexer exited normally, restarting in {restart_delay} seconds...")
            time.sleep(restart_delay)
        except KeyboardInterrupt:
            # Allow graceful shutdown on KeyboardInterrupt
            print("[Indexer] Received interrupt signal, shutting down...")
            break
        except Exception as e:
            # Catch ALL exceptions to prevent API crash
            print(f"[Indexer] CRITICAL: Indexer crashed with error: {e}")
            import traceback
            traceback.print_exc()
            print(f"[Indexer] Indexer will restart in {restart_delay} seconds...")
            print(f"[Indexer] API remains running - search endpoint is still available")
            time.sleep(restart_delay)
            # Continue loop to restart indexer


def start_indexer_daemon() -> None:
    """Start the indexer as a daemon thread.
    
    Daemon threads are automatically killed when the main program exits,
    and they don't prevent the program from shutting down. This ensures
    the indexer is fully isolated from the API.
    """
    global _indexer_thread
    if _indexer_thread is not None and _indexer_thread.is_alive():
        print("[API] Indexer thread already running")
        return
    
    _indexer_thread = threading.Thread(
        target=run_indexer,
        name="indexer-daemon",
        daemon=True,  # Daemon thread - won't block shutdown
    )
    _indexer_thread.start()
    print("[API] Indexer daemon thread started (isolated - won't block API shutdown)")


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


@app.on_event("startup")
async def startup_event() -> None:
    """Start the background indexer and periodic save task on application startup."""
    global _indexer_thread, _save_task, _shared_vector_store

    # Initialize SigLIP-2 model at startup
    print("[API] Initializing SigLIP-2 model...")
    try:
        initialize_imagebind()  # Function name kept for compatibility, but loads SigLIP-2
        print("[API] SigLIP-2 model loaded successfully")
    except Exception as e:
        print(f"[API] Warning: Failed to initialize SigLIP: {e}")
        print("[API] Continuing without model (search/indexing will fail)")

    # Initialize and load vector store (shared between API and indexer)
    print("[API] Initializing vector store...")
    try:
        vector_store_path = os.getenv("OMURA_VECTOR_STORE_PATH")
        _shared_vector_store = VectorStore()
        if vector_store_path:
            from pathlib import Path
            _shared_vector_store.index_path = Path(vector_store_path)
        
        # Load from saved index (boots from saved state in case of crash)
        _shared_vector_store.load()
        print(f"[API] Vector store loaded: {_shared_vector_store.size()} embeddings, index_built={_shared_vector_store._is_built}")
        
        # Share with search routes
        search.set_shared_vector_store(_shared_vector_store)
    except ImportError as e:
        print(f"[API] Warning: cuVS not available: {e}")
        print("[API] Vector store will not be available")
    except Exception as e:
        print(f"[API] Warning: Failed to initialize vector store: {e}")
        import traceback
        traceback.print_exc()

import fcntl
import tempfile

# ...

def acquire_lock(lock_name: str) -> Optional[object]:
    """Acquire a file lock to ensure single execution across workers."""
    lock_file = os.path.join(tempfile.gettempdir(), f"omura_{lock_name}.lock")
    try:
        fp = open(lock_file, 'w')
        fcntl.lockf(fp, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return fp
    except IOError:
        return None

# Global locks to keep file handles open
_indexer_lock_fp = None
_saver_lock_fp = None

# ...

@app.on_event("startup")
async def startup_event() -> None:
    """Start the background indexer and periodic save task on application startup."""
    global _indexer_thread, _save_task, _shared_vector_store, _indexer_lock_fp, _saver_lock_fp

    # ... (existing ImageBind and VectorStore init)

    # Start periodic save task (only in one worker)
    _saver_lock_fp = acquire_lock("saver")
    if _saver_lock_fp:
        _save_task = asyncio.create_task(periodic_save())
        print("[API] Periodic save task started (leader worker)")
    else:
        print("[API] Periodic save task skipped (another worker holds lock)")

    # Only start indexer if enabled via environment variable AND we can acquire lock
    if os.getenv("OMURA_ENABLE_INDEXER", "true").lower() == "true":
        _indexer_lock_fp = acquire_lock("indexer")
        if _indexer_lock_fp:
            # Start indexer as a daemon thread - fully isolated from API
            start_indexer_daemon()
            print("[API] Background indexer started (leader worker)")
        else:
            print("[API] Background indexer skipped (another worker holds lock)")
    else:
        print("[API] Background indexer disabled (set OMURA_ENABLE_INDEXER=true to enable)")


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """Clean up resources on application shutdown."""
    global _indexer_thread, _save_task, _shared_vector_store

    # Stop periodic save task (will do final save)
    if _save_task:
        print("[API] Stopping periodic save task...")
        _save_task.cancel()
        try:
            await _save_task
        except asyncio.CancelledError:
            pass

    # Final save of vector store
    if _shared_vector_store:
        print("[API] Performing final save of vector store...")
        try:
            _shared_vector_store.save(create_backup=True)
            print(f"[API] Final save completed: {_shared_vector_store.size()} embeddings")
        except Exception as e:
            print(f"[API] Error in final save: {e}")

    # Note: Daemon threads are automatically killed on shutdown
    # No need to explicitly stop them - they won't block shutdown
    if _indexer_thread and _indexer_thread.is_alive():
        print("[API] Indexer daemon thread will be automatically terminated on shutdown")


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
                "url": "/search/",
                "method": "POST",
                "description": "Text-to-image search",
            },
            "stats": {
                "url": "/search/stats",
                "method": "GET",
                "description": "Vector store statistics",
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
