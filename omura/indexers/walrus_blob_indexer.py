"""Real-time Walrus blob indexer — listens for BlobCertified events and
immediately fetches + embeds + indexes any image blobs into the vector store.

This is intentionally separate from ``sui_event_listener.py`` which only
catalogs blobs (sniff + DB row). This module goes further: full fetch → embed
→ FAISS add → save, so new Walrus images appear in search within seconds of
being certified on-chain.

Config (env):
  OMURA_WALRUS_INDEXER_ENABLED       enable/disable (default: true)
  OMURA_WALRUS_INDEXER_POLL_SECS     poll interval in seconds (default: 2)
  OMURA_WALRUS_INDEXER_CURSOR_FILE   cursor path (default: data/.walrus_indexer_cursor.json)
  OMURA_WALRUS_INDEXER_FETCH_TIMEOUT per-blob fetch timeout seconds (default: 60)
  WALRUS_PACKAGE_ID                  Walrus contract package address
  SUI_RPC_URL                        Sui fullnode RPC
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
from omura.parsers.multimodal import is_supported_image
from omura.utils.aggregator_pool import get_pool
from omura.utils.blob_catalog import CATALOG_DB_PATH
from omura.utils.imagebind_embeddings import generate_image_embedding
from omura.utils.vector_store import VectorStore

_LOG = "[WalrusBlobIndexer]"

ENABLED = os.getenv("OMURA_WALRUS_INDEXER_ENABLED", "true").lower() == "true"
POLL_INTERVAL = float(os.getenv("OMURA_WALRUS_INDEXER_POLL_SECS", "2"))
FETCH_TIMEOUT = float(os.getenv("OMURA_WALRUS_INDEXER_FETCH_TIMEOUT", "60"))
SUI_RPC_URL = os.getenv("SUI_RPC_URL", "https://fullnode.mainnet.sui.io:443").rstrip("/")

WALRUS_PACKAGE = os.getenv(
    "WALRUS_PACKAGE_ID",
    "0xfdc88f7d7cf30afab2f82e8380d11ee8f70efb90e863d1de8616fae1bb09ea77",
)

CURSOR_FILE = Path(
    os.getenv("OMURA_WALRUS_INDEXER_CURSOR_FILE", "data/.walrus_indexer_cursor.json")
)

# Only embed these event types — BlobCertified means the blob is fully stored.
EMBED_EVENTS = {"BlobCertified"}


# ── Cursor ───────────────────────────────────────────────────────────────────────

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


# ── Sui RPC ─────────────────────────────────────────────────────────────────────

def _query_events(
    cursor: Optional[Dict[str, Any]], limit: int = 100
) -> Tuple[List[Dict], Optional[Dict[str, Any]], bool]:
    payload = {
        "jsonrpc": "2.0", "id": 1,
        "method": "suix_queryEvents",
        "params": [
            {"MoveEventModule": {"package": WALRUS_PACKAGE, "module": "events"}},
            cursor, limit, False,  # oldest-first
        ],
    }
    try:
        r = requests.post(SUI_RPC_URL, json=payload, timeout=30)
        if r.status_code != 200:
            return [], cursor, False
        result = r.json().get("result") or {}
        return result.get("data", []), result.get("nextCursor"), result.get("hasNextPage", False)
    except Exception as e:
        print(f"{_LOG} RPC error: {e}", flush=True)
        return [], cursor, False


def _blob_id_from_event(ev: Dict) -> Optional[str]:
    """Convert u256 blob_id field → URL-safe base64 (no padding)."""
    try:
        raw = (ev.get("parsedJson") or {}).get("blob_id")
        if raw is None:
            return None
        b = int(raw).to_bytes(32, "big")
        return base64.urlsafe_b64encode(b).rstrip(b"=").decode("ascii")
    except Exception:
        return None


# ── Catalog helper ───────────────────────────────────────────────────────────────

def _upsert_catalog(blob_id: str, mime: str, ext: str, kind: str, size: int, indexed: bool) -> None:
    status = "indexed" if indexed else "discovered"
    try:
        with sqlite3.connect(str(CATALOG_DB_PATH), timeout=10) as conn:
            conn.execute(
                """
                INSERT INTO blobs (
                    blob_id, kind, mime_type, extension, size,
                    is_active, fetch_ok, status, indexed,
                    source, first_seen_at, last_updated_at
                ) VALUES (?, ?, ?, ?, ?, 1, 1, ?, ?, 'walrus_event', datetime('now'), datetime('now'))
                ON CONFLICT(blob_id) DO UPDATE SET
                    kind       = excluded.kind,
                    mime_type  = excluded.mime_type,
                    extension  = excluded.extension,
                    size       = COALESCE(NULLIF(excluded.size, 0), blobs.size),
                    fetch_ok   = 1,
                    indexed    = MAX(blobs.indexed, excluded.indexed),
                    status     = CASE WHEN blobs.indexed = 1 THEN blobs.status ELSE excluded.status END,
                    last_updated_at = datetime('now')
                """,
                (blob_id, kind, mime, ext, size, status, 1 if indexed else 0),
            )
            conn.commit()
    except Exception as e:
        print(f"{_LOG} catalog error {blob_id[:16]}…: {e}", flush=True)


# ── Core ingest ──────────────────────────────────────────────────────────────────

def _ingest_blob(
    blob_id: str,
    store: VectorStore,
    store_lock: threading.Lock,
) -> str:
    """Full pipeline: fetch → detect → embed → index. Returns outcome label."""

    # Skip if already in the vector store
    if blob_id in store.metadata:
        return "already_indexed"

    # Full fetch via load-balanced aggregator pool
    resp, _ = get_pool().get(f"/v1/blobs/{blob_id}", timeout=FETCH_TIMEOUT)
    if resp is None or resp.status_code != 200:
        code = resp.status_code if resp else "no_resp"
        _upsert_catalog(blob_id, "", "", "unknown", 0, False)
        return f"fetch_failed:{code}"

    content = resp.content
    mime, ext, kind = detect_file_type(content)
    size = len(content)

    # Only embed supported images
    if kind != "image" or not is_supported_image(ext):
        _upsert_catalog(blob_id, mime, ext, kind, size, False)
        return f"skip:{kind}/{ext}"

    emb = generate_image_embedding(content, blob_id=blob_id)
    if emb is None:
        _upsert_catalog(blob_id, mime, ext, kind, size, False)
        return "embed_failed"

    from omura.utils.nsfw_labeler import classify_nsfw
    vlm = classify_nsfw(content)
    if vlm is not None:
        nsfw_score, is_nsfw, _label = vlm
    else:
        from omura.utils.imagebind_embeddings import (
            get_nsfw_embeddings, is_nsfw_from_tag_score, nsfw_similarity_score_0_100,
        )
        nsfw_vecs = get_nsfw_embeddings()
        nsfw_score = nsfw_similarity_score_0_100(emb, nsfw_vecs) if nsfw_vecs else 0.0
        is_nsfw = is_nsfw_from_tag_score(nsfw_score)

    with store_lock:
        store.add(
            embedding=emb,
            blob_id=blob_id,
            mime_type=mime,
            size=size,
            extension=ext,
            kind=kind,
            is_nsfw=is_nsfw,
            nsfw_score=nsfw_score,
            source="walrus_event",
        )
        # Save immediately so the API hot-reloads within the same poll cycle
        store.save(create_backup=False)

    _upsert_catalog(blob_id, mime, ext, kind, size, True)
    return "indexed"


# ── Event loop ───────────────────────────────────────────────────────────────────

def _event_loop(store: VectorStore, store_lock: threading.Lock) -> None:
    print(
        f"{_LOG} Starting — Walrus package={WALRUS_PACKAGE[:18]}… "
        f"poll={POLL_INTERVAL}s RPC={SUI_RPC_URL}",
        flush=True,
    )

    cursor = _load_cursor()
    if cursor:
        print(f"{_LOG} Resuming from saved cursor.", flush=True)
    else:
        print(f"{_LOG} No prior cursor — starting from tip (new blobs only).", flush=True)
        _, cursor, _ = _query_events(None, limit=1)
        _save_cursor(cursor)

    session_indexed = 0
    session_seen = 0

    while True:
        try:
            events, new_cursor, has_next = _query_events(cursor, limit=100)

            certified = [
                e for e in events
                if e.get("type", "").rsplit("::", 1)[-1] in EMBED_EVENTS
            ]

            for ev in certified:
                blob_id = _blob_id_from_event(ev)
                if not blob_id:
                    continue

                session_seen += 1
                outcome = _ingest_blob(blob_id, store, store_lock)

                if outcome == "indexed":
                    session_indexed += 1

                # Only log non-trivial outcomes to keep logs clean
                if not outcome.startswith("skip:") and outcome != "already_indexed":
                    print(
                        f"{_LOG} {outcome:<30s} {blob_id[:24]}…  "
                        f"total_indexed={session_indexed}",
                        flush=True,
                    )

            if new_cursor and new_cursor != cursor:
                cursor = new_cursor
                _save_cursor(cursor)

            # Drain pages without sleeping; sleep only when caught up
            if not has_next:
                time.sleep(POLL_INTERVAL)

        except Exception as e:
            print(f"{_LOG} loop error: {e}", flush=True)
            time.sleep(POLL_INTERVAL * 5)


# ── Public API ───────────────────────────────────────────────────────────────────

def start_walrus_blob_indexer_thread(
    store: VectorStore,
    store_lock: threading.Lock,
) -> Optional[threading.Thread]:
    """Spawn the Walrus blob indexer as a daemon thread.

    Returns the Thread, or None if disabled via OMURA_WALRUS_INDEXER_ENABLED=false.
    """
    if not ENABLED:
        print(f"{_LOG} disabled (OMURA_WALRUS_INDEXER_ENABLED=false)", flush=True)
        return None
    t = threading.Thread(
        target=_event_loop,
        args=(store, store_lock),
        name="WalrusBlobIndexer",
        daemon=True,
    )
    t.start()
    print(f"{_LOG} daemon thread started.", flush=True)
    return t
