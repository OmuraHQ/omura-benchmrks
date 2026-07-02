"""
Blob Cataloger — Process 1

Responsibilities
----------------
1. Discover ALL Walrus blobs (active + expired) via Blockberry / GraphQL
2. Fetch only the first 8 KB of each blob (stream, no full download)
3. Detect file type via magic bytes
4. Record every blob in blob_catalog.sqlite with mime / ext / kind
5. Enqueue images for vector indexing (Process 2)
6. Never stop: historical backfill → continuous listen for new blobs

Run standalone:
  python -m omura.cataloger

Environment
-----------
OMURA_CATALOGER_BATCH_SIZE      int   default 200
OMURA_CATALOGER_WORKERS         int   default 16
OMURA_CATALOGER_LISTEN_INTERVAL int   default 60   (seconds)
OMURA_CATALOGER_LISTEN_PAGES    int   default 5
WALRUS_AGGREGATOR_URL           str   aggregator base URL
OMURA_BLOB_DISCOVERY            str   blockberry | graphql | sui_owned
"""

from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
import requests.adapters

from omura.parsers.file_detection import detect_file_type
from omura.parsers.multimodal import is_supported_image
from omura.utils.blob_catalog import (
    CATALOG_DB_PATH,
    build_blob_row,
    enqueue_images,
    init_catalog_db,
    upsert_blobs_batch,
)
from omura.utils.aggregator_pool import get_pool
from omura.utils.blob_discovery import iter_all_blob_entries
from omura.utils.blockberry import get_current_epoch

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None

# ── Configuration ──────────────────────────────────────────────────────────────

BATCH_SIZE = int(os.getenv("OMURA_CATALOGER_BATCH_SIZE", "200"))
MAX_WORKERS = int(os.getenv("OMURA_CATALOGER_WORKERS", "32"))
LISTEN_INTERVAL = int(os.getenv("OMURA_CATALOGER_LISTEN_INTERVAL", "10"))
LISTEN_MAX_PAGES = int(os.getenv("OMURA_CATALOGER_LISTEN_PAGES", "5"))

DEFAULT_AGGREGATOR = "https://agrregator.omura.fun"
AGGREGATOR_URL = os.getenv("WALRUS_AGGREGATOR_URL", DEFAULT_AGGREGATOR).rstrip("/")

# Number of bytes to stream for type detection (magic bytes only)
SNIFF_BYTES = 8192

# ── Per-thread HTTP session ────────────────────────────────────────────────────

_tls = threading.local()


def _session() -> requests.Session:
    if not hasattr(_tls, "s"):
        s = requests.Session()
        # Keep adapter pool aligned with worker count to reduce connection churn.
        pool_size = max(16, MAX_WORKERS * 2)
        a = requests.adapters.HTTPAdapter(
            pool_connections=pool_size,
            pool_maxsize=pool_size,
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


# ── Epoch cache ────────────────────────────────────────────────────────────────

_epoch_lock = threading.Lock()
_epoch_val: Optional[int] = None
_epoch_at: float = 0.0
_EPOCH_TTL = 300.0


def _current_epoch() -> int:
    global _epoch_val, _epoch_at
    with _epoch_lock:
        now = time.monotonic()
        if _epoch_val is not None and (now - _epoch_at) < _EPOCH_TTL:
            return _epoch_val
        try:
            _epoch_val = get_current_epoch(silent=True)
        except Exception:
            if _epoch_val is None:
                _epoch_val = 0
        _epoch_at = now
        return _epoch_val


# ── Blob processing ────────────────────────────────────────────────────────────


def _sniff_blob(blob_id: str) -> Tuple[Optional[bytes], Optional[int]]:
    """Stream the first SNIFF_BYTES from a blob via the aggregator pool.

    Load is spread across all upstreams; failed upstreams trip out automatically.
    """
    pool = get_pool()
    sess = _session()
    # Fast path: request just the first bytes needed for type sniffing.
    hdr = {"Range": f"bytes=0-{SNIFF_BYTES - 1}"}
    resp, _ = pool.get(
        f"/v1/blobs/{blob_id}", session=sess, timeout=20, headers=hdr
    )
    if resp is None:
        return None, None
    if resp.status_code in (200, 206):
        prefix = resp.content[:SNIFF_BYTES]
        content_range = resp.headers.get("content-range")
        content_length = resp.headers.get("content-length")
        actual_size = None
        if content_range and "/" in content_range:
            try:
                actual_size = int(content_range.rsplit("/", 1)[-1])
            except ValueError:
                actual_size = None
        if actual_size is None and content_length:
            try:
                actual_size = int(content_length)
            except ValueError:
                actual_size = None
        return prefix, actual_size

    # Fallback for gateways that ignore byte-range — pool retries across upstreams.
    resp, _ = pool.get(f"/v1/blobs/{blob_id}", session=sess, timeout=30, stream=True)
    if resp is None or resp.status_code != 200:
        return None, None
    content_length = resp.headers.get("content-length")
    actual_size = int(content_length) if content_length else None
    prefix = b""
    try:
        for chunk in resp.iter_content(SNIFF_BYTES):
            prefix += chunk
            if len(prefix) >= SNIFF_BYTES:
                break
    except requests.RequestException:
        resp.close()
        return None, None
    resp.close()
    return prefix[:SNIFF_BYTES], actual_size


def _process_one(
    blob_id: str,
    metadata: Dict[str, Any],
    current_epoch: int,
    is_active: bool = True,
) -> Tuple[Any, bool]:
    """
    Returns (db_row, should_index).
    should_index=True iff the blob is an image and should be queued for embedding.
    """
    # Use the is_active from iterator, but also check end_epoch
    end_epoch = metadata.get("end_epoch")
    if end_epoch is not None:
        try:
            if int(end_epoch) <= current_epoch:
                is_active = False
        except (TypeError, ValueError):
            pass

    if not is_active:
        row = build_blob_row(
            blob_id,
            metadata,
            is_active=False,
            status="expired",
            current_epoch=current_epoch,
        )
        return row, False

    prefix, actual_size = _sniff_blob(blob_id)
    if prefix is None:
        row = build_blob_row(
            blob_id,
            metadata,
            is_active=True,
            status="fetch_failed",
            current_epoch=current_epoch,
        )
        return row, False

    mime_type, extension, kind = detect_file_type(prefix)
    should_index = kind == "image" and is_supported_image(extension)

    row = build_blob_row(
        blob_id,
        metadata,
        mime_type=mime_type,
        extension=extension,
        kind=kind,
        actual_size=actual_size or len(prefix),
        fetch_ok=True,
        is_active=True,
        status="type_detected",
        current_epoch=current_epoch,
    )
    return row, should_index


def _process_batch(
    batch: List[Tuple[str, Dict[str, Any], bool]],
    db_path: Path,
    current_epoch: int,
    pbar=None,
) -> Tuple[int, int, int]:
    """Process a batch in parallel. Returns (total, images_queued, failed)."""
    db_rows: List[Any] = []
    image_ids: List[str] = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {
            ex.submit(_process_one, bid, meta, current_epoch, is_active): bid
            for bid, meta, is_active in batch
        }
        for fut in as_completed(futs):
            try:
                row, should_index = fut.result()
                db_rows.append(row)
                if should_index:
                    image_ids.append(futs[fut])
            except Exception as exc:
                print(f"[Cataloger] worker error ({futs[fut]}): {exc}")
            finally:
                # Real-time progress: advance as each blob is fully processed.
                if pbar is not None:
                    pbar.update(1)

    upsert_blobs_batch(db_rows, db_path=db_path)
    queued = enqueue_images(image_ids, db_path=db_path)
    failed = len(batch) - len(db_rows)
    return len(batch), queued, failed


# ── Cursor (resumable backfill) ────────────────────────────────────────────────

_CURSOR_FILE_NAME = "cataloger_cursor.txt"


def _cursor_path(db_path: Path) -> Path:
    return db_path.parent / _CURSOR_FILE_NAME


def _load_cursor(db_path: Path) -> int:
    """Load last completed page index (0 = start from beginning)."""
    p = _cursor_path(db_path)
    if p.is_file():
        try:
            return int(p.read_text().strip())
        except Exception:
            pass
    return 0


def _save_cursor(db_path: Path, page: int) -> None:
    try:
        _cursor_path(db_path).write_text(str(page))
    except Exception:
        pass


def _reset_cursor(db_path: Path) -> None:
    p = _cursor_path(db_path)
    if p.is_file():
        p.unlink(missing_ok=True)


# ── Backfill ───────────────────────────────────────────────────────────────────


def _backfill(db_path: Path) -> None:
    """
    Full historical pass: scan all active blobs from the cursor page to the end.
    Saves progress after every batch. Retries discovery errors indefinitely.
    """
    start_page = _load_cursor(db_path)
    print(f"[Cataloger] Backfill start (page={start_page})")

    batch: List[Tuple[str, Dict[str, Any]]] = []
    batches_done = 0
    current_page = start_page
    total_seen = 0
    total_queued = 0
    pbar = (
        tqdm(
            total=None,
            desc=f"[Cataloger] Backfill page={start_page}",
            unit="blob",
            dynamic_ncols=True,
            smoothing=0.05,
            mininterval=0.1,
        )
        if tqdm is not None
        else None
    )

    while True:  # retry discovery errors forever
        try:
            for page, blob_id, meta, is_active in iter_all_blob_entries(
                start_page=start_page,
                sort_by="TIMESTAMP",
                order_by="DESC",
            ):
                current_page = page
                epoch = _current_epoch()
                batch.append((blob_id, meta, is_active))

                if len(batch) < BATCH_SIZE:
                    continue

                seen, queued, _ = _process_batch(batch, db_path, epoch, pbar=pbar)
                total_seen += seen
                total_queued += queued
                batches_done += 1
                batch = []
                _save_cursor(db_path, current_page)
                if pbar is not None:
                    pbar.set_description(f"[Cataloger] Backfill page={current_page}")
                    pbar.set_postfix(
                        seen=total_seen,
                        queued=total_queued,
                        batch=batches_done,
                        refresh=False,
                    )

                # Log every batch for visibility
                print(
                    f"[Cataloger] Batch #{batches_done:04d} | "
                    f"page={current_page:03d} | "
                    f"seen={total_seen:06d} | "
                    f"queued={total_queued:05d}"
                )

                if batches_done % 100 == 0:
                    print(
                        f"[Cataloger] CHECKPOINT: batches={batches_done} seen={total_seen} queued={total_queued} page={current_page}"
                    )

            # Final partial batch
            if batch:
                epoch = _current_epoch()
                seen, queued, _ = _process_batch(batch, db_path, epoch, pbar=pbar)
                total_seen += seen
                total_queued += queued
                batch = []
                _save_cursor(db_path, current_page)
                if pbar is not None:
                    pbar.set_postfix(
                        seen=total_seen,
                        queued=total_queued,
                        batch=batches_done,
                        refresh=False,
                    )

            break  # discovery exhausted — backfill complete

        except KeyboardInterrupt:
            if pbar is not None:
                pbar.close()
            raise
        except Exception as exc:
            print(
                f"[Cataloger] Discovery error: {exc}. Retrying in 60s from page={current_page}..."
            )
            start_page = current_page  # resume from last good page
            time.sleep(60)

    if pbar is not None:
        pbar.close()
    _reset_cursor(db_path)
    print(
        f"[Cataloger] Backfill complete. total_seen={total_seen} queued_images={total_queued}"
    )


# ── Listen ─────────────────────────────────────────────────────────────────────


def _listen_once(db_path: Path) -> int:
    """Poll newest blobs from Blockberry (sorted by TIMESTAMP DESC). Returns count of new."""
    epoch = _current_epoch()
    new_batch: List[Tuple[str, Dict[str, Any], bool]] = []
    last_page = -1

    try:
        for page, blob_id, meta, is_active in iter_all_blob_entries(
            sort_by="TIMESTAMP", order_by="DESC", source="blockberry"
        ):
            if page != last_page:
                if page >= LISTEN_MAX_PAGES:
                    break
                last_page = page
            new_batch.append((blob_id, meta, is_active))
    except Exception as exc:
        print(f"[Cataloger] Listen discovery error (non-fatal): {exc}")

    if not new_batch:
        return 0

    seen, queued, _ = _process_batch(new_batch, db_path, epoch)
    if queued:
        print(f"[Cataloger] Listen: {seen} blobs, {queued} new images queued")
    return queued


# ── Main service loop ──────────────────────────────────────────────────────────

_backfill_done = False


def run_cataloger(db_path: Path = CATALOG_DB_PATH) -> None:
    """
    Never-stopping cataloger service.
    Phase 1: historical backfill (resumes from cursor on restart).
    Phase 2: continuous listen for new blobs every LISTEN_INTERVAL seconds.
    """
    global _backfill_done
    init_catalog_db(db_path)

    print(
        f"[Cataloger] Starting. DB={db_path} workers={MAX_WORKERS} batch={BATCH_SIZE}"
    )

    while True:
        try:
            if not _backfill_done:
                _backfill(db_path)
                _backfill_done = True
                print(
                    f"[Cataloger] Entering listen mode (poll every {LISTEN_INTERVAL}s)"
                )

            # Listen loop
            while True:
                time.sleep(LISTEN_INTERVAL)
                try:
                    _listen_once(db_path)
                except Exception as exc:
                    print(f"[Cataloger] Listen error (non-fatal): {exc}")

        except KeyboardInterrupt:
            print("[Cataloger] Interrupted, shutting down.")
            break
        except Exception as exc:
            import traceback

            print(f"[Cataloger] Unexpected error, restarting in 30s: {exc}")
            traceback.print_exc()
            time.sleep(30)


if __name__ == "__main__":
    run_cataloger()
