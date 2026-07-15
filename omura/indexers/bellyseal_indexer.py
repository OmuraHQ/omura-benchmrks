"""Real-time indexer for BellySeal (bellyseal.com) ImageNFT mint events.

Polls ``suix_queryEvents`` every few seconds for new ``ImageNFTMinted`` events
from the BellySeal package, fetches each Walrus blob via the aggregator pool,
embeds it with the active embedding model, and adds it to the shared vector
store — all without restarting the API.

The cursor is persisted to disk so restarts pick up exactly where they left off.

Config (env):
  OMURA_BELLYSEAL_ENABLED         enable/disable (default: true)
  OMURA_BELLYSEAL_POLL_SECS       poll interval in seconds (default: 3)
  OMURA_BELLYSEAL_PACKAGE         BellySeal Move package address
  OMURA_BELLYSEAL_CURSOR_FILE     path for cursor persistence
  SUI_RPC_URL                     Sui fullnode RPC (default: mainnet)
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from omura.parsers.file_detection import detect_file_type
from omura.parsers.multimodal import is_supported_image
from omura.utils.aggregator_pool import get_pool
from omura.utils.blob_catalog import CATALOG_DB_PATH
from omura.utils.imagebind_embeddings import generate_image_embedding
from omura.utils.vector_store import VectorStore

import requests

ENABLED = os.getenv("OMURA_BELLYSEAL_ENABLED", "true").lower() == "true"
POLL_INTERVAL = float(os.getenv("OMURA_BELLYSEAL_POLL_SECS", "3"))
SUI_RPC_URL = os.getenv("SUI_RPC_URL", "https://fullnode.mainnet.sui.io:443").rstrip("/")

BELLYSEAL_PACKAGE = os.getenv(
    "OMURA_BELLYSEAL_PACKAGE",
    "0xcfc06380ac1526bffd2a29e6f4db5ec8a5dea9240164a77ff0e532456fed0706",
)
MINT_EVENT_TYPE = f"{BELLYSEAL_PACKAGE}::image_nft::ImageNFTMinted"

CURSOR_FILE = Path(
    os.getenv("OMURA_BELLYSEAL_CURSOR_FILE", "data/.bellyseal_event_cursor.json")
)

FETCH_TIMEOUT = float(os.getenv("OMURA_BELLYSEAL_FETCH_TIMEOUT", "60"))

_LOG = "[BellySealIndexer]"


# ── Cursor persistence ──────────────────────────────────────────────────────────

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

def _query_mint_events(
    cursor: Optional[Dict[str, Any]], limit: int = 100
) -> Tuple[List[Dict], Optional[Dict[str, Any]], bool]:
    """Query ImageNFTMinted events, oldest-first from cursor.

    Returns (events, next_cursor, has_next_page).
    """
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "suix_queryEvents",
        "params": [{"MoveEventType": MINT_EVENT_TYPE}, cursor, limit, False],
    }
    try:
        r = requests.post(SUI_RPC_URL, json=payload, timeout=30)
        if r.status_code != 200:
            return [], cursor, False
        body = r.json()
        result = body.get("result") or {}
        return result.get("data", []), result.get("nextCursor"), result.get("hasNextPage", False)
    except Exception as e:
        print(f"{_LOG} RPC error: {e}", flush=True)
        return [], cursor, False


# ── Catalog helpers ─────────────────────────────────────────────────────────────

def _upsert_catalog(
    blob_id: str,
    mime: str,
    ext: str,
    kind: str,
    size: int,
    nft_id: str,
    generation_id: str,
    creator: str,
    prompt: str,
) -> None:
    now_sql = "datetime('now')"
    try:
        with sqlite3.connect(str(CATALOG_DB_PATH), timeout=10) as conn:
            conn.execute(
                f"""
                INSERT INTO blobs (
                    blob_id, kind, mime_type, extension, size,
                    is_active, fetch_ok, status, indexed,
                    source, first_seen_at, last_updated_at
                ) VALUES (?, ?, ?, ?, ?, 1, 1, 'discovered', 0,
                          'bellyseal', {now_sql}, {now_sql})
                ON CONFLICT(blob_id) DO UPDATE SET
                    kind            = excluded.kind,
                    mime_type       = excluded.mime_type,
                    extension       = excluded.extension,
                    size            = COALESCE(NULLIF(excluded.size,0), blobs.size),
                    source          = 'bellyseal',
                    last_updated_at = {now_sql}
                """,
                (blob_id, kind, mime, ext, size),
            )
            conn.commit()
    except Exception as e:
        print(f"{_LOG} catalog write error {blob_id[:16]}…: {e}", flush=True)


def _mark_indexed(blob_id: str, indexed: bool) -> None:
    status = "indexed" if indexed else "embed_failed"
    try:
        with sqlite3.connect(str(CATALOG_DB_PATH), timeout=10) as conn:
            conn.execute(
                "UPDATE blobs SET indexed=?, status=?, fetch_ok=1, last_updated_at=datetime('now') WHERE blob_id=?",
                (1 if indexed else 0, status, blob_id),
            )
            conn.commit()
    except Exception:
        pass


# ── Ingest one NFT ──────────────────────────────────────────────────────────────

def _ingest_nft(
    blob_id: str,
    nft_id: str,
    generation_id: str,
    creator: str,
    prompt: str,
    store: VectorStore,
    store_lock: threading.Lock,
) -> str:
    """Fetch → detect → embed → add to store. Returns outcome string."""
    if blob_id in store.metadata:
        return "already_indexed"

    resp, _ = get_pool().get(
        f"/v1/blobs/{blob_id}", timeout=FETCH_TIMEOUT
    )
    if resp is None or resp.status_code != 200:
        _upsert_catalog(blob_id, "", "", "image", 0, nft_id, generation_id, creator, prompt)
        return f"fetch_failed:{resp.status_code if resp else 'no_resp'}"

    content = resp.content
    mime, ext, kind = detect_file_type(content)

    _upsert_catalog(blob_id, mime, ext, kind, len(content), nft_id, generation_id, creator, prompt)

    if kind != "image" or not is_supported_image(ext):
        return f"unsupported:{kind}/{ext}"

    emb = generate_image_embedding(content, blob_id=blob_id)
    if emb is None:
        _mark_indexed(blob_id, False)
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
            size=len(content),
            extension=ext,
            kind=kind,
            is_nsfw=is_nsfw,
            nsfw_score=nsfw_score,
            source="bellyseal",
            nft_id=nft_id,
            generation_id=generation_id,
        )

    _mark_indexed(blob_id, True)
    return "indexed"


# ── Event loop ──────────────────────────────────────────────────────────────────

def _event_loop(store: VectorStore, store_lock: threading.Lock) -> None:
    print(
        f"{_LOG} Starting — package={BELLYSEAL_PACKAGE[:18]}… "
        f"poll={POLL_INTERVAL}s RPC={SUI_RPC_URL}",
        flush=True,
    )

    cursor = _load_cursor()
    if cursor:
        print(f"{_LOG} Resuming from saved cursor.", flush=True)
    else:
        print(f"{_LOG} No prior cursor — starting from latest event.", flush=True)
        # Burn one query to get current tip so we don't replay history.
        _, cursor, _ = _query_mint_events(None, limit=1)
        _save_cursor(cursor)

    session_indexed = 0
    session_seen = 0

    while True:
        try:
            events, new_cursor, has_next = _query_mint_events(cursor, limit=100)

            for ev in events:
                pj = ev.get("parsedJson") or {}
                blob_id     = (pj.get("walrus_blob_id") or "").strip()
                nft_id      = pj.get("nft_id", "")
                generation_id = pj.get("generation_id", "")
                creator     = pj.get("creator", "")
                prompt      = pj.get("prompt", "")

                session_seen += 1

                if not blob_id:
                    # Mint event fired but blob not yet attached (pending generation)
                    print(
                        f"{_LOG} NFT {nft_id[:16]}… minted, blob pending — skipping",
                        flush=True,
                    )
                    continue

                outcome = _ingest_nft(
                    blob_id, nft_id, generation_id, creator, prompt,
                    store, store_lock,
                )
                if outcome == "indexed":
                    session_indexed += 1
                    # Save immediately after each new NFT so API sees it right away
                    with store_lock:
                        store.save(create_backup=False)

                print(
                    f"{_LOG} {outcome:<22s} blob={blob_id[:20]}… "
                    f"gen={generation_id}  total_indexed={session_indexed}",
                    flush=True,
                )

            if new_cursor and new_cursor != cursor:
                cursor = new_cursor
                _save_cursor(cursor)

            # If there are more pages, drain them without sleeping
            if not has_next:
                time.sleep(POLL_INTERVAL)

        except Exception as e:
            print(f"{_LOG} loop error: {e}", flush=True)
            time.sleep(POLL_INTERVAL * 5)


# ── Public API ──────────────────────────────────────────────────────────────────

def start_bellyseal_indexer_thread(
    store: VectorStore,
    store_lock: threading.Lock,
) -> Optional[threading.Thread]:
    """Spawn the BellySeal event loop as a daemon thread.

    Returns the Thread, or None if disabled via OMURA_BELLYSEAL_ENABLED=false.
    """
    if not ENABLED:
        print(f"{_LOG} disabled (OMURA_BELLYSEAL_ENABLED=false)", flush=True)
        return None

    t = threading.Thread(
        target=_event_loop,
        args=(store, store_lock),
        name="BellySealIndexer",
        daemon=True,
    )
    t.start()
    print(f"{_LOG} daemon thread started.", flush=True)
    return t
