#!/usr/bin/env python3
"""
Build a local SQLite of **active** Walrus blobs from both Sui GraphQL and Blockberry.

We keep a normalized schema that matches Blockberry's shape (camelCase) while also accepting
GraphQL-derived metadata (snake_case). The canonical identifier for downstream is
`blob_id_base64` (Walrus aggregator form / base64url), and we key by (blob_id_base64, source)
to allow comparing sources side-by-side.

Schema (single table, keyed by (blob_id_base64, source)):

    CREATE TABLE blobs (
        blob_id_base64   TEXT NOT NULL,     -- e.g. "1dEitcCn13Q5..."
        source           TEXT NOT NULL,     -- 'graphql' | 'blockberry'
        blob_id_u256_dec TEXT,              -- decimal string (Blockberry `blobId`)
        object_id        TEXT,              -- Sui object id (Blockberry `objectId`, GraphQL `address`)
        status           TEXT,              -- e.g. "Certified" (Blockberry)
        start_epoch      INTEGER,
        end_epoch        INTEGER,
        size             INTEGER,
        timestamp_ms     INTEGER,           -- Blockberry `timestamp`
        deletable        INTEGER,
        registered_epoch INTEGER,
        object_version   INTEGER,
        meta_json        TEXT,              -- raw metadata from backend (JSON)
        discovered_at    TEXT DEFAULT (datetime('now')),
        PRIMARY KEY (blob_id_base64, source)
    );

Usage examples (from repo root):

    uv run python scripts/build_blob_db.py --db walrus_blobs.sqlite
    uv run python scripts/build_blob_db.py --db walrus_blobs.sqlite --skip-graphql
    uv run python scripts/build_blob_db.py --db walrus_blobs.sqlite --skip-blockberry
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from omura.utils.blob_discovery import iter_active_blob_entries  # type: ignore
from omura.utils.blockberry import get_current_epoch  # type: ignore


def ensure_schema(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.executescript(
        """
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;

        CREATE TABLE IF NOT EXISTS blobs (
            blob_id_base64   TEXT NOT NULL,
            source           TEXT NOT NULL,
            blob_id_u256_dec TEXT,
            object_id        TEXT,
            status           TEXT,
            start_epoch      INTEGER,
            end_epoch        INTEGER,
            size             INTEGER,
            timestamp_ms     INTEGER,
            deletable        INTEGER,
            registered_epoch INTEGER,
            object_version   INTEGER,
            meta_json        TEXT,
            discovered_at    TEXT DEFAULT (datetime('now')),
            PRIMARY KEY (blob_id_base64, source)
        );

        CREATE INDEX IF NOT EXISTS idx_blobs_end_epoch
            ON blobs(end_epoch);
        CREATE INDEX IF NOT EXISTS idx_blobs_size
            ON blobs(size);
        """
    )
    # Best-effort migrations (older DBs).
    for stmt in [
        "ALTER TABLE blobs ADD COLUMN blob_id_u256_dec TEXT",
        "ALTER TABLE blobs ADD COLUMN object_id TEXT",
        "ALTER TABLE blobs ADD COLUMN status TEXT",
        "ALTER TABLE blobs ADD COLUMN start_epoch INTEGER",
        "ALTER TABLE blobs ADD COLUMN timestamp_ms INTEGER",
        # back-compat columns (if created with old schema)
        "ALTER TABLE blobs ADD COLUMN blob_id_base64 TEXT",
    ]:
        try:
            cur.execute(stmt)
        except sqlite3.OperationalError:
            pass
    conn.commit()


def current_walrus_epoch() -> int:
    ep = get_current_epoch(silent=True)
    return int(ep)


def _upsert_rows(
    conn: sqlite3.Connection,
    rows: Iterable[Tuple[str, str, Dict]],
) -> int:
    cur = conn.cursor()
    count = 0
    for agg_id, source, meta in rows:
        # Normalize Blockberry camelCase + GraphQL snake_case to one schema.
        # `agg_id` is already the aggregator/base64url id from `iter_active_blob_entries`.
        blob_id_base64 = agg_id

        blob_id_u256_dec = meta.get("blobId") or meta.get("blob_id")
        object_id = meta.get("objectId") or meta.get("object_id") or meta.get("sui_object_id")
        status = meta.get("status")

        start_epoch = meta.get("startEpoch") or meta.get("start_epoch")
        end_epoch = meta.get("endEpoch") or meta.get("end_epoch")
        timestamp_ms = meta.get("timestamp") or meta.get("timestamp_ms")

        try:
            end_epoch_i = int(end_epoch) if end_epoch is not None else None
        except (TypeError, ValueError):
            end_epoch_i = None

        size = meta.get("size")
        try:
            size_i = int(size) if size is not None else None
        except (TypeError, ValueError):
            size_i = None

        deletable = meta.get("deletable")
        deletable_i = int(bool(deletable)) if deletable is not None else None

        deletable = meta.get("deletable")
        deletable_i = int(bool(deletable)) if deletable is not None else None

        registered_epoch = meta.get("registeredEpoch") or meta.get("registered_epoch")
        try:
            registered_epoch_i = int(registered_epoch) if registered_epoch is not None else None
        except (TypeError, ValueError):
            registered_epoch_i = None

        object_version = meta.get("objectVersion") or meta.get("object_version")
        try:
            object_version_i = int(object_version) if object_version is not None else None
        except (TypeError, ValueError):
            object_version_i = None

        try:
            start_epoch_i = int(start_epoch) if start_epoch is not None else None
        except (TypeError, ValueError):
            start_epoch_i = None

        try:
            timestamp_ms_i = int(timestamp_ms) if timestamp_ms is not None else None
        except (TypeError, ValueError):
            timestamp_ms_i = None

        cur.execute(
            """
            INSERT INTO blobs (
                blob_id_base64, source,
                blob_id_u256_dec, object_id, status,
                start_epoch, end_epoch, size, timestamp_ms,
                deletable, registered_epoch, object_version,
                meta_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(blob_id_base64, source) DO UPDATE SET
                blob_id_u256_dec = excluded.blob_id_u256_dec,
                object_id        = excluded.object_id,
                status           = excluded.status,
                start_epoch      = excluded.start_epoch,
                end_epoch        = excluded.end_epoch,
                size             = excluded.size,
                timestamp_ms     = excluded.timestamp_ms,
                deletable        = excluded.deletable,
                registered_epoch = excluded.registered_epoch,
                object_version   = excluded.object_version,
                meta_json        = excluded.meta_json,
                discovered_at    = datetime('now')
            """,
            (
                blob_id_base64,
                source,
                str(blob_id_u256_dec) if blob_id_u256_dec is not None else None,
                object_id,
                status,
                start_epoch_i,
                end_epoch_i,
                size_i,
                timestamp_ms_i,
                deletable_i,
                registered_epoch_i,
                object_version_i,
                json.dumps(meta, separators=(",", ":"), sort_keys=True),
            ),
        )
        count += 1
    conn.commit()
    return count


def scrape_source(
    conn: sqlite3.Connection,
    source: str,
    epoch: int,
) -> None:
    assert source in ("graphql", "blockberry", "sui_owned")

    if source == "graphql":
        os.environ["OMURA_BLOB_DISCOVERY"] = "graphql"
    elif source == "blockberry":
        os.environ["OMURA_BLOB_DISCOVERY"] = "blockberry"
    else:
        os.environ["OMURA_BLOB_DISCOVERY"] = "sui_owned"

    print(f"== {source.upper()} active blobs (end_epoch > {epoch}) ==", file=sys.stderr)

    batch: list[Tuple[str, str, Dict]] = []
    total = 0
    page_seen = -1

    for page_idx, agg_id, meta in iter_active_blob_entries(current_epoch=epoch, source=source):
        if page_idx != page_seen:
            page_seen = page_idx
            print(f"  page {page_idx}", file=sys.stderr)

        batch.append((agg_id, source, meta))
        if len(batch) >= 1000:
            n = _upsert_rows(conn, batch)
            total += n
            batch.clear()

    if batch:
        n = _upsert_rows(conn, batch)
        total += n

    print(f"  {source}: upserted {total} rows into blobs", file=sys.stderr)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Scrape active Walrus blobs from Sui GraphQL and Blockberry into a SQLite DB.",
    )
    p.add_argument(
        "--db",
        required=True,
        help="Path to SQLite DB to create/update (e.g. walrus_blobs.sqlite).",
    )
    p.add_argument(
        "--epoch",
        type=int,
        help="Walrus epoch for active filter (default: auto via Sui RPC / Blockberry heuristics).",
    )
    p.add_argument(
        "--skip-graphql",
        action="store_true",
        help="Skip the Sui GraphQL crawl.",
    )
    p.add_argument(
        "--skip-blockberry",
        action="store_true",
        help="Skip the Blockberry API crawl.",
    )
    args = p.parse_args()

    db_path = Path(args.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    try:
        ensure_schema(conn)

        epoch = args.epoch if args.epoch is not None else current_walrus_epoch()
        print(f"# walrus_epoch={epoch}", file=sys.stderr)

        if not args.skip_graphql:
            scrape_source(conn, "graphql", epoch)
        if not args.skip_blockberry:
            scrape_source(conn, "blockberry", epoch)
    finally:
        conn.close()

    print(f"SQLite written at {db_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

