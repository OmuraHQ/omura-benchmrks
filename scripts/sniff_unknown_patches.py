"""Second-pass quilt-patch sniffer.

After ``extract_quilts.py`` runs (which catalogs patches by filename extension
only), many patches end up as ``kind='unknown'`` because their identifier is a
UUID or path with no file extension (Walrus Site shards, blockchain dumps with
no extension, etc.). The file type is still recoverable — fetch the first 8 KB
and run magic-byte detection.

This script:
  1. Pulls every ``kind='unknown'`` inner patch (blob_id LIKE ``%::%``) from
     ``blob_catalog.sqlite``.
  2. For each, fetches the first ``SNIFF_BYTES`` bytes via the aggregator pool.
  3. Runs ``detect_file_type`` and updates the catalog row with the real
     ``kind`` / ``mime`` / ``extension``.

Sharded files (``..._shard=N-M``): we sniff every shard independently. Magic
bytes typically live in shard 0; later shards may detect as ``binary`` /
``application``, which we still record (better than ``unknown``).

Usage:
  uv run python scripts/sniff_unknown_patches.py                 # all unknowns
  uv run python scripts/sniff_unknown_patches.py --limit 5000    # cap
  uv run python scripts/sniff_unknown_patches.py --workers 32    # default 32
  uv run python scripts/sniff_unknown_patches.py --kinds unknown,binary,application
  uv run python scripts/sniff_unknown_patches.py --dry-run

Env:
  WALRUS_AGGREGATOR_URLS    pool of upstreams (load-balanced)
  OMURA_CATALOG_DB_PATH     default data/blob_catalog.sqlite
  OMURA_PATCH_SNIFF_BYTES   first N bytes to fetch (default 8192)
  OMURA_PATCH_SNIFF_TIMEOUT default 30
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
SNIFF_BYTES = int(os.getenv("OMURA_PATCH_SNIFF_BYTES", "8192"))
TIMEOUT = float(os.getenv("OMURA_PATCH_SNIFF_TIMEOUT", "30"))

_tls = threading.local()
_DB_INIT = False
_DB_INIT_LOCK = threading.Lock()


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


def _ensure_pragmas() -> None:
    global _DB_INIT
    with _DB_INIT_LOCK:
        if _DB_INIT:
            return
        try:
            with sqlite3.connect(str(CATALOG_DB), timeout=30) as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
                conn.execute("PRAGMA temp_store=MEMORY")
                conn.execute("PRAGMA cache_size=-65536")
                conn.commit()
            _DB_INIT = True
        except Exception as exc:
            print(f"WAL init error: {exc}", file=sys.stderr)


def _parse_inner_id(inner_id: str) -> Tuple[str, str]:
    """Split composite id into (parent_quilt_id, identifier)."""
    if "::" not in inner_id:
        return ("", inner_id)
    parent, ident = inner_id.split("::", 1)
    return (parent, ident)


def _fetch_prefix(parent: str, ident: str) -> Optional[bytes]:
    """Range-GET the first SNIFF_BYTES of a quilt patch via the aggregator pool."""
    safe = requests.utils.quote(ident, safe="")
    resp, _ = get_pool().get(
        f"/v1/blobs/by-quilt-id/{parent}/{safe}",
        session=_session(),
        timeout=TIMEOUT,
        headers={"Range": f"bytes=0-{SNIFF_BYTES - 1}"},
    )
    if resp is None:
        return None
    if resp.status_code in (200, 206):
        return resp.content[: SNIFF_BYTES]
    return None


def _sniff_one(inner_id: str) -> Tuple[str, Optional[str], Optional[str], Optional[str]]:
    """Returns (inner_id, mime, ext, kind) or (..., None, None, None) on failure."""
    parent, ident = _parse_inner_id(inner_id)
    if not parent:
        return inner_id, None, None, None
    data = _fetch_prefix(parent, ident)
    if data is None or not data:
        return inner_id, None, None, None
    mime, ext, kind = detect_file_type(data)
    return inner_id, mime, ext, kind


def _list_unknowns(
    kinds: list, limit: Optional[int], shard_filter: str = "headers"
) -> list:
    """List unknown patches.

    shard_filter:
      - "headers"  (default): unsharded patches + ``_shard=0-N`` (only patches that
        contain the file header — detectable types). Middle/end shards are skipped
        because their first 8 KB never matches magic bytes for the real file type.
      - "all": every patch regardless of shard position.
      - "shard0-only": only ``_shard=0-N`` patches.
    """
    placeholders = ",".join("?" for _ in kinds)
    with sqlite3.connect(str(CATALOG_DB), timeout=30) as conn:
        cur = conn.cursor()
        cur.execute(
            f"""
            SELECT blob_id FROM blobs
            WHERE blob_id LIKE '%::%' AND is_active=1
              AND kind IN ({placeholders})
            ORDER BY blob_id
            """,
            tuple(kinds),
        )
        rows = cur.fetchall()
    ids = [r[0] for r in rows]
    if shard_filter == "headers":
        # Keep: unsharded (no _shard= suffix) OR _shard=0-N
        ids = [
            b for b in ids
            if "_shard=" not in b.split("::", 1)[1]
            or "_shard=0-" in b.split("::", 1)[1]
        ]
    elif shard_filter == "shard0-only":
        ids = [b for b in ids if "_shard=0-" in b.split("::", 1)[1]]
    # "all" passes everything
    return ids[:limit] if limit else ids


def _batch_update(rows: list) -> None:
    """Batch UPDATE catalog with new (mime, ext, kind, blob_id) tuples."""
    if not rows:
        return
    _ensure_pragmas()
    try:
        with sqlite3.connect(str(CATALOG_DB), timeout=60) as conn:
            conn.executemany(
                """
                UPDATE blobs SET
                    mime_type=?, extension=?, kind=?, fetch_ok=1,
                    last_updated_at=datetime('now')
                WHERE blob_id=?
                """,
                rows,
            )
            conn.commit()
    except Exception as exc:
        print(f"  batch update error ({len(rows)} rows): {exc}", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kinds",
        default="unknown",
        help="comma-separated catalog kinds to (re)sniff (default: unknown)",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--workers",
        type=int,
        default=32,
        help="parallel sniffs (default 32, assumes ample bandwidth)",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--commit-every",
        type=int,
        default=200,
        help="batch-commit catalog updates every N rows (default 200)",
    )
    parser.add_argument(
        "--shard-filter",
        choices=["headers", "all", "shard0-only"],
        default="headers",
        help=(
            "headers (default): unsharded + shard=0-N — only patches that contain the "
            "file header. all: every patch. shard0-only: only first shards."
        ),
    )
    args = parser.parse_args()

    kinds = [k.strip() for k in args.kinds.split(",") if k.strip()]
    pool = get_pool()
    print(f"Aggregator pool ({len(pool.upstreams)} upstreams):")
    for u in pool.upstreams:
        print(f"  - {u.url}")
    print(f"Catalog DB: {CATALOG_DB}")
    print(f"Target kinds: {kinds}")
    print(f"Sniff bytes: {SNIFF_BYTES}")
    print(f"Workers: {args.workers}")

    print(f"Shard filter: {args.shard_filter}")
    ids = _list_unknowns(kinds, args.limit, shard_filter=args.shard_filter)
    print(f"Patches to sniff: {len(ids):,}")
    if not ids:
        return 0

    start = time.time()
    counters = {
        "sniffed": 0, "updated": 0, "fetch_failed": 0,
        "by_new_kind": {},
    }
    pending_updates: list = []

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_sniff_one, b): b for b in ids}
        done = 0
        for fut in as_completed(futs):
            done += 1
            inner_id, mime, ext, kind = fut.result()
            counters["sniffed"] += 1
            if kind is None:
                counters["fetch_failed"] += 1
            else:
                counters["by_new_kind"][kind] = counters["by_new_kind"].get(kind, 0) + 1
                # Only update when the detection changed something useful.
                if kind not in kinds or ext:
                    pending_updates.append((mime, ext, kind, inner_id))
                    counters["updated"] += 1

            if len(pending_updates) >= args.commit_every:
                if not args.dry_run:
                    _batch_update(pending_updates)
                pending_updates = []

            if done % 50 == 0:
                elapsed = time.time() - start
                rate = done / elapsed if elapsed else 0
                top = ", ".join(
                    f"{k}={v}"
                    for k, v in sorted(
                        counters["by_new_kind"].items(), key=lambda x: -x[1]
                    )[:6]
                )
                print(
                    f"  {done:>6}/{len(ids)} ({rate:.1f}/s)  "
                    f"updated={counters['updated']:,}  "
                    f"fail={counters['fetch_failed']:,}  | {top}",
                    flush=True,
                )

    if pending_updates and not args.dry_run:
        _batch_update(pending_updates)

    elapsed = time.time() - start
    print()
    print(f"Done in {elapsed:.1f}s. ({len(ids) / elapsed:.2f} patches/s)")
    print(f"  sniffed:          {counters['sniffed']:,}")
    print(f"  updated:          {counters['updated']:,}")
    print(f"  fetch failed:     {counters['fetch_failed']:,}")
    print(f"  kinds discovered:")
    for k, n in sorted(counters["by_new_kind"].items(), key=lambda x: -x[1]):
        print(f"    {k!r:18s} {n:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
