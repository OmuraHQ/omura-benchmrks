"""Immediate catalog ingest for a list of blob_ids.

Used as a post-upload hook from the bash uploader: each time ``walrus store``
prints ``Blob ID: <id>`` lines, we feed those ids here to:
  1. Range-GET the first 8 KB via the aggregator pool
  2. Detect file type via magic bytes
  3. UPSERT into ``blob_catalog.sqlite`` with ``source='walrus_store_local'``

This closes the loop instantly — no waiting for Blockberry / cataloger-listen polling.

Usage:
  echo "BLOB_ID_1" | uv run python scripts/catalog_blob_ids.py
  uv run python scripts/catalog_blob_ids.py BLOB_ID_1 BLOB_ID_2 BLOB_ID_3
  uv run python scripts/catalog_blob_ids.py --file /tmp/new_blob_ids.txt
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from omura.parsers.file_detection import detect_file_type  # noqa: E402
from omura.utils.aggregator_pool import get_pool  # noqa: E402

CATALOG_DB = Path(os.getenv("OMURA_CATALOG_DB_PATH", "data/blob_catalog.sqlite"))
SNIFF_BYTES = 8192


def _sniff(blob_id: str) -> tuple[str, str, str, int]:
    """Returns (mime, ext, kind, size)."""
    resp, _ = get_pool().get(
        f"/v1/blobs/{blob_id}",
        timeout=15,
        headers={"Range": f"bytes=0-{SNIFF_BYTES - 1}"},
    )
    if resp is None or resp.status_code not in (200, 206):
        return "application/octet-stream", "", "unknown", 0
    data = resp.content[:SNIFF_BYTES]
    mime, ext, kind = detect_file_type(data)
    cl = resp.headers.get("content-range") or resp.headers.get("content-length", "0")
    size = 0
    if "/" in cl:
        try: size = int(cl.rsplit("/", 1)[-1])
        except ValueError: pass
    else:
        try: size = int(cl)
        except ValueError: pass
    return mime, ext, kind, size


def _upsert(blob_id: str, mime: str, ext: str, kind: str, size: int) -> None:
    try:
        with sqlite3.connect(str(CATALOG_DB), timeout=10) as conn:
            conn.execute(
                """
                INSERT INTO blobs (
                    blob_id, kind, mime_type, extension, size,
                    is_active, fetch_ok, status, indexed,
                    source, first_seen_at, last_updated_at
                ) VALUES (?, ?, ?, ?, ?, 1, ?, 'discovered', 0,
                          'walrus_store_local', datetime('now'), datetime('now'))
                ON CONFLICT(blob_id) DO UPDATE SET
                    kind=excluded.kind,
                    mime_type=excluded.mime_type,
                    extension=excluded.extension,
                    size=COALESCE(NULLIF(excluded.size,0), blobs.size),
                    source=excluded.source,
                    last_updated_at=datetime('now')
                """,
                (blob_id, kind, mime, ext, size, 1 if size > 0 else 0),
            )
            conn.commit()
    except Exception as exc:
        print(f"  [{blob_id[:18]}…] catalog write error: {exc}", file=sys.stderr)


def _process_one(blob_id: str) -> str:
    mime, ext, kind, size = _sniff(blob_id)
    _upsert(blob_id, mime, ext, kind, size)
    return f"  cat: {blob_id}  kind={kind:<8s} ext={ext or '∅':<6s} size={size}"


def _read_ids(args: argparse.Namespace) -> List[str]:
    ids: List[str] = []
    if args.file:
        for line in Path(args.file).read_text().splitlines():
            s = line.strip()
            if s: ids.append(s)
    if args.blob_ids:
        ids.extend(args.blob_ids)
    if not ids and not sys.stdin.isatty():
        for line in sys.stdin:
            s = line.strip()
            if s and "BLOB" not in s.upper() and len(s) == 43:
                ids.append(s)
    # dedupe preserving order
    seen = set()
    out = []
    for b in ids:
        if b not in seen:
            seen.add(b); out.append(b)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("blob_ids", nargs="*")
    p.add_argument("--file", help="newline-separated blob_id list")
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    ids = _read_ids(args)
    if not ids:
        print("no blob_ids on stdin or args", file=sys.stderr)
        return 1

    print(f"Cataloging {len(ids)} new blob(s)...", file=sys.stderr)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for fut in as_completed([ex.submit(_process_one, b) for b in ids]):
            print(fut.result(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
