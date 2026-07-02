"""Real-time Walrus blob detection via Sui events.

The Walrus contract emits ``BlobCertified`` events the moment a blob is fully
stored. Polling Blockberry every 60s misses this window — by subscribing to
Sui's event stream we can detect new blobs within seconds of upload.

Approach:
  - Poll ``suix_queryEvents`` every 2-3s with an event-module filter on the
    Walrus contract. (Pure WebSocket subscription is fragile under reconnects;
    a tight HTTP poll with a cursor is simpler and still feels real-time.)
  - For each new ``BlobCertified`` event, extract the blob_id and trigger
    immediate ingest: fetch first 8 KB via the aggregator pool, detect file
    type, upsert into ``blob_catalog.sqlite`` with ``source='sui_event'``.

Runs as a daemon thread inside the indexer process. Disable via
``OMURA_REALTIME_SUI_EVENTS=false``.
"""

from __future__ import annotations

import base64
import json
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from omura.parsers.file_detection import detect_file_type
from omura.utils.aggregator_pool import get_pool


ENABLED = os.getenv("OMURA_REALTIME_SUI_EVENTS", "true").lower() == "true"
POLL_INTERVAL_SECS = float(os.getenv("OMURA_SUI_EVENT_POLL_SECS", "2"))
SUI_RPC_URL = os.getenv("SUI_RPC_URL", "https://fullnode.mainnet.sui.io:443").rstrip("/")
WALRUS_PACKAGE = os.getenv(
    "WALRUS_PACKAGE_ID",
    "0xfdc88f7d7cf30afab2f82e8380d11ee8f70efb90e863d1de8616fae1bb09ea77",
)
SNIFF_BYTES = int(os.getenv("OMURA_SNIFF_BYTES", "8192"))
CURSOR_FILE = Path(os.getenv("OMURA_SUI_EVENT_CURSOR", "data/.sui_event_cursor.json"))

# Walrus event names we care about — these contain blob_id fields.
INTERESTING_EVENTS = {"BlobCertified", "BlobRegistered"}


def _load_cursor() -> Optional[Dict[str, Any]]:
    try:
        if CURSOR_FILE.exists():
            return json.loads(CURSOR_FILE.read_text())
    except Exception:
        pass
    return None


def _save_cursor(cursor: Optional[Dict[str, Any]]) -> None:
    if not cursor:
        return
    try:
        CURSOR_FILE.parent.mkdir(parents=True, exist_ok=True)
        CURSOR_FILE.write_text(json.dumps(cursor))
    except Exception:
        pass


def _query_events(cursor: Optional[Dict[str, Any]], limit: int = 100) -> Tuple[List[dict], Optional[Dict[str, Any]]]:
    """Page through new events from the Walrus package via suix_queryEvents."""
    # Use the most precise filter we can — MoveModule for the events module within the package.
    event_filter = {
        "MoveEventModule": {
            "package": WALRUS_PACKAGE,
            "module": "events",
        }
    }
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "suix_queryEvents",
        "params": [event_filter, cursor, limit, False],  # descending=False -> oldest first
    }
    try:
        r = requests.post(SUI_RPC_URL, json=payload, timeout=30)
        if r.status_code != 200:
            return [], cursor
        body = r.json()
        result = body.get("result") or {}
        return result.get("data", []), result.get("nextCursor")
    except Exception:
        return [], cursor


def _blob_id_from_event(ev: dict) -> Optional[str]:
    """Walrus events embed the blob_id as parsedJson.blob_id (u256 number string).

    Convert the u256 to the URL-safe base64 form the aggregator expects
    (32 bytes big-endian → base64url, strip padding).
    """
    try:
        parsed = ev.get("parsedJson") or {}
        raw = parsed.get("blob_id")
        if raw is None:
            return None
        try:
            n = int(raw)
        except (TypeError, ValueError):
            return None
        b = n.to_bytes(32, "big")
        return base64.urlsafe_b64encode(b).rstrip(b"=").decode("ascii")
    except Exception:
        return None


def _record_realtime_blob(
    catalog_db: Path,
    blob_id: str,
    mime: str,
    ext: str,
    kind: str,
    size: int,
) -> None:
    """Upsert a real-time-detected blob into the catalog."""
    try:
        with sqlite3.connect(str(catalog_db), timeout=10) as conn:
            conn.execute(
                """
                INSERT INTO blobs (
                    blob_id, kind, mime_type, extension, size,
                    is_active, fetch_ok, status, indexed,
                    source, first_seen_at, last_updated_at
                ) VALUES (?, ?, ?, ?, ?, 1, 1, 'discovered', 0,
                          'sui_event', datetime('now'), datetime('now'))
                ON CONFLICT(blob_id) DO UPDATE SET
                    kind=excluded.kind,
                    mime_type=excluded.mime_type,
                    extension=excluded.extension,
                    size=COALESCE(NULLIF(excluded.size,0), blobs.size),
                    last_updated_at=datetime('now')
                """,
                (blob_id, kind, mime, ext, size),
            )
            conn.commit()
    except Exception as e:
        print(f"[SuiEventListener] catalog write error for {blob_id[:18]}…: {e}", flush=True)


def _sniff_and_catalog(catalog_db: Path, blob_id: str) -> Tuple[str, str]:
    """Range-GET first 8 KB via the aggregator pool, detect type, write to catalog."""
    pool = get_pool()
    resp, used = pool.get(
        f"/v1/blobs/{blob_id}",
        timeout=15,
        headers={"Range": f"bytes=0-{SNIFF_BYTES - 1}"},
    )
    if resp is None:
        return "fetch_failed", ""
    if resp.status_code not in (200, 206):
        return f"http_{resp.status_code}", ""
    data = resp.content[:SNIFF_BYTES]
    mime, ext, kind = detect_file_type(data)
    # Read declared content-length if available
    cl = resp.headers.get("content-range") or resp.headers.get("content-length")
    size = 0
    if cl and "/" in cl:
        try: size = int(cl.rsplit("/", 1)[-1])
        except ValueError: size = len(data)
    elif cl:
        try: size = int(cl)
        except ValueError: size = len(data)
    _record_realtime_blob(catalog_db, blob_id, mime, ext, kind, size)
    return kind, ext


def _event_loop(catalog_db: Path) -> None:
    """Continuously poll Sui events, sniff each new blob, record in catalog."""
    print(
        f"[SuiEventListener] Starting — RPC={SUI_RPC_URL} package={WALRUS_PACKAGE[:14]}… "
        f"poll={POLL_INTERVAL_SECS}s",
        flush=True,
    )
    cursor = _load_cursor()
    if cursor:
        print(f"[SuiEventListener] Resuming from cursor: {cursor}", flush=True)
    else:
        print("[SuiEventListener] No prior cursor; starting from latest", flush=True)
        # Burn one query to get the latest cursor so we don't replay history.
        _, cursor = _query_events(None, limit=1)
        _save_cursor(cursor)

    seen_in_session = 0
    while True:
        try:
            events, new_cursor = _query_events(cursor, limit=100)
            new_events = [e for e in events if e.get("type", "").rsplit("::", 1)[-1] in INTERESTING_EVENTS]
            for ev in new_events:
                etype = ev.get("type", "").rsplit("::", 1)[-1]
                blob_id = _blob_id_from_event(ev)
                if not blob_id:
                    continue
                # Real-time ingest
                kind, ext = _sniff_and_catalog(catalog_db, blob_id)
                seen_in_session += 1
                print(
                    f"[SuiEventListener] {etype:<18s} {blob_id} kind={kind:<10s} ext={ext or '∅'}",
                    flush=True,
                )
            if new_cursor and new_cursor != cursor:
                cursor = new_cursor
                _save_cursor(cursor)
            time.sleep(POLL_INTERVAL_SECS)
        except Exception as e:
            print(f"[SuiEventListener] loop error: {e}", flush=True)
            time.sleep(POLL_INTERVAL_SECS * 5)


def start_sui_event_listener_thread(catalog_db: Path) -> Optional[threading.Thread]:
    """Spawn the listener loop in a daemon thread. Returns the Thread (or None if disabled)."""
    if not ENABLED:
        print("[SuiEventListener] disabled via OMURA_REALTIME_SUI_EVENTS=false")
        return None
    t = threading.Thread(
        target=_event_loop, args=(catalog_db,),
        name="SuiEventListener", daemon=True,
    )
    t.start()
    return t
