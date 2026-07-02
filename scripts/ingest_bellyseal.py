"""Ingest all BellySeal (bellyseal.com) ImageNFT blobs into the catalog + vector index.

Reads /tmp/bellyseal_blobs.json (produced by the scrape step), fetches each
walrus_blob_id via the aggregator pool, embeds it, and adds it to the live
FAISS store.  Also upserts a row into blob_catalog.sqlite so the blob appears
in the dashboard.

Usage:
  uv run python scripts/ingest_bellyseal.py
  uv run python scripts/ingest_bellyseal.py --blobs-json /path/to/bellyseal_blobs.json
  uv run python scripts/ingest_bellyseal.py --workers 32
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from omura.utils.aggregator_pool import get_pool  # noqa: E402
from omura.utils.blob_catalog import CATALOG_DB_PATH  # noqa: E402
from omura.utils.vector_store import VectorStore  # noqa: E402
from omura.parsers.file_detection import detect_file_type  # noqa: E402
from omura.parsers.multimodal import is_supported_image  # noqa: E402
from omura.utils.imagebind_embeddings import generate_image_embedding  # noqa: E402

DEFAULT_BLOBS_JSON = "/tmp/bellyseal_blobs.json"
DEFAULT_WORKERS = int(os.getenv("OMURA_INDEXER_WORKERS", "16"))
FETCH_TIMEOUT = 60.0

BELLYSEAL_PACKAGE = "0xcfc06380ac1526bffd2a29e6f4db5ec8a5dea9240164a77ff0e532456fed0706"

_tls = threading.local()


def _session():
    import requests, requests.adapters
    s = getattr(_tls, "s", None)
    if s is None:
        s = requests.Session()
        a = requests.adapters.HTTPAdapter(
            pool_connections=4, pool_maxsize=64,
            max_retries=requests.adapters.Retry(
                total=1, backoff_factor=0.3,
                status_forcelist=[502, 503, 504],
                allowed_methods=["GET"],
            ),
        )
        s.mount("https://", a)
        s.mount("http://", a)
        _tls.s = s
    return s


# ── Catalog helpers ─────────────────────────────────────────────────────────────

def _upsert_catalog(blob_id: str, meta: Dict) -> None:
    """Insert/update blob_catalog.sqlite with BellySeal NFT metadata."""
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    try:
        with sqlite3.connect(str(CATALOG_DB_PATH), timeout=15) as conn:
            conn.execute(
                """
                INSERT INTO blobs (
                    blob_id, kind, source, is_active, fetch_ok,
                    status, indexed, first_seen_at, last_updated_at
                ) VALUES (?, 'image', 'bellyseal', 1, 1,
                          'discovered', 0, ?, ?)
                ON CONFLICT(blob_id) DO UPDATE SET
                    kind        = 'image',
                    source      = 'bellyseal',
                    is_active   = 1,
                    fetch_ok    = 1,
                    last_updated_at = ?
                """,
                (blob_id, now, now, now),
            )
            conn.commit()
    except Exception as e:
        print(f"  [catalog] upsert failed for {blob_id}: {e}")


def _mark_indexed(blob_id: str, indexed: bool) -> None:
    try:
        with sqlite3.connect(str(CATALOG_DB_PATH), timeout=10) as conn:
            conn.execute(
                "UPDATE blobs SET indexed=?, status=?, last_updated_at=datetime('now') WHERE blob_id=?",
                (1 if indexed else 0, "indexed" if indexed else "embed_failed", blob_id),
            )
            conn.commit()
    except Exception:
        pass


def _already_in_catalog(blob_ids: List[str]) -> set:
    """Return the subset of blob_ids already in the catalog."""
    found = set()
    chunk = 500
    try:
        with sqlite3.connect(str(CATALOG_DB_PATH), timeout=15) as conn:
            for i in range(0, len(blob_ids), chunk):
                batch = blob_ids[i:i+chunk]
                ph = ",".join("?" * len(batch))
                rows = conn.execute(
                    f"SELECT blob_id FROM blobs WHERE blob_id IN ({ph})", batch
                ).fetchall()
                found.update(r[0] for r in rows)
    except Exception:
        pass
    return found


# ── Fetch + embed ────────────────────────────────────────────────────────────────

def _fetch(blob_id: str) -> Optional[bytes]:
    resp, _ = get_pool().get(
        f"/v1/blobs/{blob_id}", session=_session(), timeout=FETCH_TIMEOUT
    )
    if resp is None or resp.status_code != 200:
        return None
    return resp.content


def _process(
    blob_id: str,
    meta: Dict,
    store: VectorStore,
    store_lock: threading.Lock,
) -> str:
    if blob_id in store.metadata:
        return "skip_already_indexed"

    _upsert_catalog(blob_id, meta)

    content = _fetch(blob_id)
    if content is None:
        return "fetch_failed"

    mime, ext, kind = detect_file_type(content)
    if kind != "image" or not is_supported_image(ext):
        return f"unsupported:{kind}/{ext}"

    emb = generate_image_embedding(content, blob_id=blob_id)
    if emb is None:
        _mark_indexed(blob_id, False)
        return "embed_failed"

    with store_lock:
        store.add(
            embedding=emb,
            blob_id=blob_id,
            mime_type=mime,
            size=len(content),
            extension=ext,
            kind=kind,
            is_nsfw=False,
            source="bellyseal",
            nft_id=meta.get("nft_id", ""),
            generation_id=meta.get("generation_id", ""),
        )
    _mark_indexed(blob_id, True)
    return "indexed"


# ── Main ─────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blobs-json", default=DEFAULT_BLOBS_JSON)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--save-every", type=int, default=50)
    args = parser.parse_args()

    print("═" * 60)
    print("  BellySeal NFT ingestion")
    print("═" * 60)

    blobs_path = Path(args.blobs_json)
    if not blobs_path.exists():
        print(f"ERROR: {blobs_path} not found. Run the scrape step first.")
        return 1

    with open(blobs_path) as f:
        blobs: Dict[str, Dict] = json.load(f)

    print(f"  Loaded {len(blobs)} BellySeal blob IDs from {blobs_path}")

    # Filter already indexed
    all_ids = list(blobs.keys())
    print(f"  Checking catalog for existing entries …")
    in_catalog = _already_in_catalog(all_ids)
    print(f"  Already in catalog: {len(in_catalog)}")

    print(f"\nLoading vector store …")
    store = VectorStore()
    store.load()
    already_vs = len([b for b in all_ids if b in store.metadata])
    print(f"  Already in vector store: {already_vs}")
    print(f"  Store current size: {len(store.metadata)}")

    todo = [(bid, blobs[bid]) for bid in all_ids]
    print(f"\n── Ingesting {len(todo)} BellySeal blobs ({args.workers} workers) ──")

    counters: Dict[str, int] = {}
    newly_indexed = 0
    store_lock = threading.Lock()
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_process, bid, meta, store, store_lock): bid
                for bid, meta in todo}
        done = 0
        for fut in as_completed(futs):
            bid = futs[fut]
            try:
                outcome = fut.result()
            except Exception as e:
                outcome = f"error:{e}"
            counters[outcome] = counters.get(outcome, 0) + 1
            if outcome == "indexed":
                newly_indexed += 1
                if newly_indexed % args.save_every == 0:
                    with store_lock:
                        store.save(create_backup=False)
                    elapsed = time.time() - t0
                    print(f"  checkpoint: {newly_indexed} indexed "
                          f"({done+1}/{len(todo)}, {done/elapsed:.1f} blobs/s)")
            done += 1
            if done % 100 == 0:
                elapsed = time.time() - t0
                print(f"  {done}/{len(todo)}  indexed={newly_indexed}  "
                      f"({done/elapsed:.1f} blobs/s)")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Results: {dict(sorted(counters.items()))}")
    print(f"  Newly indexed: {newly_indexed}")

    print("\nSaving final index …")
    store.save(create_backup=False)
    print(f"Final store size: {len(store.metadata)} blobs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
