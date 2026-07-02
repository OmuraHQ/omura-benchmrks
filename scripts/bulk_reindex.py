#!/usr/bin/env python3
"""
Bulk Image Reindexer for Omura

Downloads all image blobs cataloged in the SQL database, computes their embeddings using
the loaded embedding model, and rebuilds the FAISS vector index and embeddings.npy from scratch.
"""

import os
import sys
import sqlite3
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import torch
from tqdm import tqdm

# Add parent directory to path so we can import omura modules
sys.path.append(str(Path(__file__).resolve().parents[1]))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from omura.utils.vector_store import VectorStore
from omura.utils.imagebind_embeddings import (
    initialize_embedding_model,
    generate_image_embedding,
    is_model_ready
)
from omura.utils.aggregator_pool import get_pool

def fetch_and_embed(blob_id: str, aggregator: str) -> tuple[str, np.ndarray | None]:
    """Fetch blob bytes and generate embedding."""
    try:
        resp, _ = get_pool().get(f"/v1/blobs/{blob_id}", timeout=30)
        if resp is None or resp.status_code != 200:
            return blob_id, None
        
        emb = generate_image_embedding(resp.content, blob_id=blob_id)
        return blob_id, emb
    except Exception:
        return blob_id, None

def main():
    parser = argparse.ArgumentParser(description="Bulk download images and rebuild FAISS index.")
    parser.add_argument("--db", default="data/blob_catalog.sqlite", help="Path to SQLite catalog database")
    parser.add_argument("--aggregator", default="https://agrregator.omura.fun", help="Walrus aggregator URL")
    parser.add_argument("--workers", type=int, default=16, help="Number of concurrent download workers")
    parser.add_argument("--limit", type=int, default=None, help="Limit total rebuild count")
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"Error: Catalog database not found at {db_path}", file=sys.stderr)
        sys.exit(1)

    print("Initializing embedding models...")
    initialize_embedding_model()
    if not is_model_ready():
        print("Error: Embedding model is not loaded or ready.")
        sys.exit(1)

    print(f"Reading images from {db_path}...")
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    
    # Get current Walrus epoch to skip expired blobs
    from omura.utils.blockberry import get_current_epoch
    try:
        current_epoch = get_current_epoch()
        print(f"Current Walrus epoch: {current_epoch}")
    except Exception as e:
        print(f"Warning: Could not fetch Walrus epoch, fallback to skipping epoch filters: {e}")
        current_epoch = None

    query = (
        "SELECT blob_id, mime_type, size, extension, kind, is_nsfw "
        "FROM blobs "
        "WHERE (kind = 'image' OR mime_type LIKE 'image/%') AND is_active = 1"
    )
    if current_epoch is not None:
        query += f" AND (end_epoch IS NULL OR CAST(end_epoch AS INTEGER) > {current_epoch})"
        
    if args.limit:
        query += f" LIMIT {args.limit}"
        
    cur.execute(query)
    rows = cur.fetchall()
    conn.close()

    total = len(rows)
    if total == 0:
        print("No image blobs found in database.")
        return

    print(f"Starting bulk reindex of {total} images...")

    # Initialize a clean VectorStore
    store = VectorStore()
    
    # We clean up the old store lists and dicts
    store.embeddings_dict = {}
    store.embeddings_list = []
    store.position_to_blob_id = []
    store.metadata = {}

    success = 0
    failed = 0

    # Map details
    blob_details = {r[0]: {"mime_type": r[1], "size": r[2], "extension": r[3], "kind": r[4], "is_nsfw": r[5]} for r in rows}
    blob_ids = list(blob_details.keys())

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(fetch_and_embed, bid, args.aggregator.rstrip("/")): bid 
            for bid in blob_ids
        }
        
        with tqdm(total=total, desc="Reindexing") as pbar:
            for fut in as_completed(futures):
                bid = futures[fut]
                try:
                    blob_id, emb = fut.result()
                    if emb is not None:
                        # Normalize embedding vector
                        emb = np.asarray(emb, dtype=np.float32).flatten()
                        norm = np.linalg.norm(emb)
                        if norm > 0:
                            emb = emb / norm
                            
                        details = blob_details[blob_id]
                        # Add directly to store list & dict
                        store.add(
                            embedding=emb,
                            blob_id=blob_id,
                            mime_type=details["mime_type"] or "image",
                            size=details["size"] or 0,
                            extension=details["extension"] or "",
                            kind=details["kind"] or "image",
                            is_nsfw=bool(details["is_nsfw"])
                        )
                        success += 1
                    else:
                        failed += 1
                except Exception as e:
                    print(f"Failed to process {bid}: {e}")
                    failed += 1
                pbar.update(1)

    print("\nSaving new vector index...")
    # This automatically runs build_index() and saves all npy/faiss files cleanly
    store.build_index()
    store.save(create_backup=True)

    # Sync back SQLite status
    print("Updating SQL index database states...")
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    
    # Mark successfully indexed blobs as indexed
    indexed_bids = list(store.embeddings_dict.keys())
    if indexed_bids:
        cur.executemany(
            "UPDATE blobs SET indexed = 1, status = 'indexed' WHERE blob_id = ?",
            [(bid,) for bid in indexed_bids]
        )
    
    # Mark failed ones
    failed_bids = [bid for bid in blob_ids if bid not in store.embeddings_dict]
    if failed_bids:
        cur.executemany(
            "UPDATE blobs SET indexed = 0, status = 'embed_failed' WHERE blob_id = ?",
            [(bid,) for bid in failed_bids]
        )
        
    conn.commit()
    conn.close()

    print("\n--- Summary ---")
    print(f"Successfully Indexed: {success}")
    print(f"Failed/Missing:      {failed}")
    print(f"New index size:      {store.size()} embeddings")

if __name__ == "__main__":
    main()
