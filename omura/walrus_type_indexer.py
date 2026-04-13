"""Simple Walrus blob type indexer using magic bytes.

This is a first-step "service" for the indexer: it
  - reads blob IDs from the files produced by `get_blob_ids.py`
  - fetches each blob from Walrus mainnet
  - detects its file type via magic bytes
  - prints the result

Later we can plug this into a proper database/vector index.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import requests

from omura.parsers.file_detection import detect_file_type
from omura.utils.blockberry import MANUAL_EPOCH, get_current_epoch
from omura.utils.blob_discovery import iter_active_blob_entries


BATCH_SIZE = 100
MAX_WORKERS = int(os.getenv("OMURA_INDEXER_WORKERS", "8"))
_EPOCH_CACHE_SEC = float(os.getenv("OMURA_INDEXER_EPOCH_CACHE_SEC", "300"))
_cached_walrus_epoch: Optional[int] = None
_cached_walrus_epoch_at: float = 0.0

# Default to redundex mainnet read-only aggregator if no env override
DEFAULT_AGGREGATOR = "https://walrus-mainnet-aggregator.redundex.com"
AGGREGATOR_URL = os.getenv("WALRUS_AGGREGATOR_URL", DEFAULT_AGGREGATOR).rstrip("/")


def _walrus_epoch_for_index_batch() -> int:
    global _cached_walrus_epoch, _cached_walrus_epoch_at
    now = time.monotonic()
    if _cached_walrus_epoch is not None and (now - _cached_walrus_epoch_at) < _EPOCH_CACHE_SEC:
        return _cached_walrus_epoch
    e = get_current_epoch(silent=True)
    _cached_walrus_epoch = MANUAL_EPOCH if e is None else e
    _cached_walrus_epoch_at = now
    return _cached_walrus_epoch


def _blob_expired_vs_epoch(metadata: Dict[str, Any], current_epoch: int) -> bool:
    end = metadata.get("end_epoch")
    if end is None:
        return False
    try:
        return int(end) <= int(current_epoch)
    except (TypeError, ValueError):
        return False


def fetch_blob_http(blob_id: str) -> Optional[bytes]:
    """Fetch blob bytes from the configured HTTP aggregator.

    Uses the raw numeric blob ID form, which is what Blockberry returns and
    what the HTTP API at `/v1/blobs/{id}` expects.
    """
    url = f"{AGGREGATOR_URL}/v1/blobs/{blob_id}"
    try:
        resp = requests.get(url, timeout=60)
        if resp.status_code == 200:
            return resp.content
        # Treat 404/410/400 as not-found; log minimally
        print(f"{blob_id}: HTTP {resp.status_code} from aggregator")
        return None
    except requests.RequestException as e:
        print(f"{blob_id}: HTTP error fetching from aggregator: {e}")
        return None


def _process_blob(blob_id: str, metadata: Dict[str, Any], walrus_epoch: int) -> None:
    """Pipeline for a single blob: fetch → detect type → (future) index."""
    try:
        if _blob_expired_vs_epoch(metadata, walrus_epoch):
            print(
                f"{blob_id}: skip expired (end_epoch={metadata.get('end_epoch')} <= epoch {walrus_epoch})"
            )
            return

        content = fetch_blob_http(blob_id)
        if not content:
            print(f"{blob_id}: NOT_FOUND or empty content")
            return

        mime, ext, kind = detect_file_type(content)
        size = len(content)

        # Placeholder for future parsing/indexing. For now, just log.
        print(f"{blob_id}: {mime} ({ext}, {kind}), {size} bytes")
    except Exception as e:
        print(f"{blob_id}: ERROR in pipeline: {e}")


def process_batch(
    batch: List[Tuple[str, Dict[str, Any]]],
) -> None:
    """Process a batch of blobs in parallel worker threads."""
    walrus_epoch = _walrus_epoch_for_index_batch()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(_process_blob, blob_id, meta, walrus_epoch): blob_id
            for blob_id, meta in batch
        }
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"{futures[future]}: worker exception: {e}")


def main(batch_size: int = BATCH_SIZE, max_batches: Optional[int] = None) -> None:
    """End-to-end pipeline to index all active blobs (type detection + hooks).

    1. Discover active blobs via `iter_active_blob_entries()` (default: Sui GraphQL / on-chain `end_epoch`; see OMURA_BLOB_DISCOVERY).
    2. Accumulate them into batches of `batch_size`.
    3. For each batch:
         - fetch from HTTP aggregator
         - detect type via magic bytes
         - (future) parse and index content.
    """
    print(
        f"Starting Walrus indexing pipeline (discovery → fetch → detect) "
        f"using aggregator {AGGREGATOR_URL}"
    )

    current_batch: List[Tuple[str, Dict[str, Any]]] = []
    batches_processed = 0

    for _page, blob_id, meta in iter_active_blob_entries():
        current_batch.append((blob_id, meta))
        if len(current_batch) >= batch_size:
            print(f"\nProcessing batch #{batches_processed + 1} ({len(current_batch)} blobs)...")
            process_batch(current_batch)
            batches_processed += 1
            current_batch = []

            if max_batches is not None and batches_processed >= max_batches:
                print("Reached max_batches limit, stopping pipeline.")
                break

    # Process any trailing blobs < batch_size
    if current_batch and (max_batches is None or batches_processed < max_batches):
        print(f"\nProcessing final batch ({len(current_batch)} blobs)...")
        process_batch(current_batch)


if __name__ == "__main__":
    main()

