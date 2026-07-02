"""Re-detect file type for active blobs labelled video/audio by sniffing the first 8 KB
again with the fixed ISO BMFF brand+handler detection.

Targets ``kind IN ('video','audio','application')`` rows that were previously sniffed
when the classifier defaulted to ``video`` for any unknown ftyp brand. After the fix,
many M4A audio files (and other non-video ISO BMFF blobs) will reclassify correctly.

Usage:
  uv run python scripts/reclassify_av_kinds.py                # process all video kinds
  uv run python scripts/reclassify_av_kinds.py --limit 500    # cap
  uv run python scripts/reclassify_av_kinds.py --workers 6
  uv run python scripts/reclassify_av_kinds.py --dry-run
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Tuple

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from omura.parsers.file_detection import detect_file_type  # noqa: E402
from omura.utils.aggregator_pool import get_pool  # noqa: E402

CATALOG_DB = Path(os.getenv("OMURA_CATALOG_DB_PATH", "data/blob_catalog.sqlite"))
SNIFF_BYTES = 8192

_tls = threading.local()


def _session() -> requests.Session:
    s = getattr(_tls, "s", None)
    if s is None:
        s = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=4, pool_maxsize=32,
            max_retries=requests.adapters.Retry(
                total=1, backoff_factor=0.3,
                status_forcelist=[502, 503, 504],
                allowed_methods=["GET"],
            ),
        )
        s.mount("https://", adapter)
        s.mount("http://", adapter)
        _tls.s = s
    return s


def _sniff(blob_id: str) -> Optional[bytes]:
    """Fetch first 8KB via the aggregator pool (Range request)."""
    resp, _ = get_pool().get(
        f"/v1/blobs/{blob_id}",
        session=_session(),
        timeout=20,
        headers={"Range": f"bytes=0-{SNIFF_BYTES - 1}"},
    )
    if resp is None:
        return None
    if resp.status_code in (200, 206):
        return resp.content[:SNIFF_BYTES]
    return None


def _reclassify_one(blob_id: str) -> Tuple[str, Optional[str], Optional[str], Optional[str]]:
    """Returns (blob_id, mime, ext, kind) or (blob_id, None, None, None) on fetch failure."""
    data = _sniff(blob_id)
    if data is None:
        return blob_id, None, None, None
    mime, ext, kind = detect_file_type(data)
    return blob_id, mime, ext, kind


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kinds",
        default="video",
        help="comma-separated list of catalog kinds to re-detect (default: video)",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    targets = [k.strip() for k in args.kinds.split(",") if k.strip()]
    placeholders = ",".join("?" for _ in targets)

    with sqlite3.connect(str(CATALOG_DB), timeout=30) as conn:
        cur = conn.cursor()
        cur.execute(
            f"SELECT blob_id FROM blobs WHERE kind IN ({placeholders}) AND is_active=1 ORDER BY blob_id",
            tuple(targets),
        )
        ids = [r[0] for r in cur.fetchall()]
    if args.limit:
        ids = ids[: args.limit]
    print(f"To reclassify: {len(ids):,} blobs (kinds={targets})")
    if not ids:
        return 0

    counters = {"unchanged": 0, "audio": 0, "image": 0, "video": 0, "other": 0, "fetch_failed": 0}
    by_kind_after = {}
    start = time.time()
    updates: list[Tuple[str, str, str, str]] = []

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_reclassify_one, b): b for b in ids}
        done = 0
        for fut in as_completed(futs):
            blob_id, mime, ext, kind = fut.result()
            done += 1
            if kind is None:
                counters["fetch_failed"] += 1
            else:
                by_kind_after[kind] = by_kind_after.get(kind, 0) + 1
                if kind in targets:
                    counters["unchanged"] += 1
                else:
                    counters[kind if kind in counters else "other"] = counters.get(
                        kind if kind in counters else "other", 0
                    ) + 1
                    updates.append((mime, ext, kind, blob_id))
            if done % 200 == 0:
                el = time.time() - start
                rate = done / el if el else 0
                print(f"  {done}/{len(ids)} ({rate:.1f}/s)  reclassified_so_far={len(updates):,}")

    elapsed = time.time() - start
    print()
    print(f"Done in {elapsed:.1f}s.")
    print(f"  fetch failures:  {counters['fetch_failed']:,}")
    print(f"  unchanged (still video): {counters['unchanged']:,}")
    print(f"  changed to audio: {counters['audio']:,}")
    print(f"  changed to image: {counters['image']:,}")
    print(f"  changed to other: {counters['other']:,}")
    print(f"  result kind distribution:")
    for k, n in sorted(by_kind_after.items(), key=lambda x: -x[1]):
        print(f"    {k!r:18s} {n:,}")

    if updates and not args.dry_run:
        print(f"  applying {len(updates):,} updates to catalog DB...")
        with sqlite3.connect(str(CATALOG_DB), timeout=60) as conn:
            conn.executemany(
                "UPDATE blobs SET mime_type=?, extension=?, kind=?, last_updated_at=datetime('now') WHERE blob_id=?",
                updates,
            )
            conn.commit()
        print(f"  catalog updated.")
    elif args.dry_run:
        print("  (--dry-run: catalog not modified)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
