"""Background worker service for cataloging + indexing.

Run with:
  OMURA_INSTANCE_ROLE=worker uv run main.py
"""

from __future__ import annotations

import os
import threading
import time

from omura.indexers.vector_indexer import run_vector_indexer
from omura.indexers.bellyseal_indexer import start_bellyseal_indexer_thread
from omura.indexers.walrus_blob_indexer import start_walrus_blob_indexer_thread
from omura.cataloger import run_cataloger
from omura.utils.blob_catalog import CATALOG_DB_PATH, init_catalog_db
from omura.utils.vector_store import VectorStore
from omura.utils.imagebind_embeddings import initialize_embedding_model


def _start_daemon(target, name: str) -> threading.Thread:
    t = threading.Thread(target=target, name=name, daemon=True)
    t.start()
    return t


def _run_cataloger_loop() -> None:
    while True:
        try:
            run_cataloger(CATALOG_DB_PATH)
        except Exception as exc:
            print(f"[Worker/Cataloger] Crashed, restarting in 30s: {exc}")
            time.sleep(30)


def _run_indexer_loop(store: VectorStore) -> None:
    while True:
        try:
            run_vector_indexer(store, CATALOG_DB_PATH)
        except Exception as exc:
            print(f"[Worker/Indexer] Crashed, restarting in 30s: {exc}")
            time.sleep(30)


def run_worker_service() -> None:
    """Run dedicated background ingestion service."""
    init_catalog_db(CATALOG_DB_PATH)

    # Prefer multi-GPU embedding pool on worker nodes unless explicitly overridden.
    if (
        "OMURA_EMBEDDING_DEVICES" not in os.environ
        and "OMURA_EMBEDDING_MAX_DEVICES" not in os.environ
    ):
        os.environ["OMURA_EMBEDDING_MAX_DEVICES"] = "0"  # 0 => all visible GPUs
        print("[Worker] OMURA_EMBEDDING_MAX_DEVICES not set; using all visible GPUs")

    print("[Worker] Initializing vector store...")
    store = VectorStore()
    store.load()
    print(f"[Worker] Vector store ready ({store.size()} embeddings)")

    # Shared lock so BellySeal indexer and vector indexer don't race on store writes.
    store_lock = threading.Lock()

    print("[Worker] Loading embedding model...")
    initialize_embedding_model()
    print("[Worker] Embedding model ready")

    threads: list[threading.Thread] = []
    if os.getenv("OMURA_ENABLE_CATALOGER", "true").lower() == "true":
        threads.append(_start_daemon(_run_cataloger_loop, "cataloger"))
        print("[Worker] Cataloger started")
    if os.getenv("OMURA_ENABLE_INDEXER", "true").lower() == "true":
        threads.append(_start_daemon(lambda: _run_indexer_loop(store), "vector-indexer"))
        print("[Worker] Vector indexer started")
    if os.getenv("OMURA_BELLYSEAL_ENABLED", "true").lower() == "true":
        t = start_bellyseal_indexer_thread(store, store_lock)
        if t:
            threads.append(t)
            print("[Worker] BellySeal live indexer started")
    if os.getenv("OMURA_WALRUS_INDEXER_ENABLED", "true").lower() == "true":
        t = start_walrus_blob_indexer_thread(store, store_lock)
        if t:
            threads.append(t)
            print("[Worker] Walrus blob indexer started")

    if not threads:
        print("[Worker] Nothing enabled (set OMURA_ENABLE_CATALOGER/INDEXER=true)")
        return

    # Keep process alive while daemon threads run.
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("[Worker] Shutting down...")
        try:
            store.save(create_backup=True)
        except Exception as exc:
            print(f"[Worker] Final save error: {exc}")

