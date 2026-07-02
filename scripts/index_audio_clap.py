#!/usr/bin/env python
"""Backfill the CLAP audio vector index from the catalog of active audio blobs.

CLAP (laion/larger_clap_general) is the audio search model (86.65% ESC-50). This
builds a separate 512-d index at OMURA_AUDIO_VECTOR_STORE_DIR so audio search is
independent of the main image/text store.

  PYTHONPATH=. CUDA_VISIBLE_DEVICES=7 .venv/bin/python scripts/index_audio_clap.py \
      [--limit N] [--workers 8] [--save-every 100] [--dry-run]

Idempotent: blob_ids already in the CLAP store are skipped (unless --reindex).
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from omura.utils import clap_embeddings as clap
from omura.utils.vector_store import VectorStore

CATALOG_DB = os.getenv("OMURA_CATALOG_DB_PATH", "data/blob_catalog.sqlite")
AUDIO_STORE_DIR = Path(os.getenv("OMURA_AUDIO_VECTOR_STORE_DIR", "data/vector_index_clap"))

# Fetch directly from known-good aggregators rather than the in-process pool: the
# pool trips a node after a few consecutive failures and, with a single pinned
# upstream, cascades into mass fetch_failed under concurrent multi-MB requests.
AGGREGATORS = [
    a.strip().rstrip("/")
    for a in os.getenv(
        "OMURA_FETCH_AGGREGATORS",
        "https://agrregator.omura.fun,https://aggregator.walrus-mainnet.walrus.space",
    ).split(",")
    if a.strip()
]
MAX_FETCH_BYTES = int(os.getenv("OMURA_MAX_AUDIO_BYTES", str(200 * 1024 * 1024)))  # skip absurd blobs

_tls = threading.local()


def _session() -> requests.Session:
    s = getattr(_tls, "s", None)
    if s is None:
        s = requests.Session()
        _tls.s = s
    return s


def _fetch_patch(blob_id: str, timeout: int) -> bytes | None:
    """blob_id = 'quiltId::identifier' -> resolve patch_id, fetch patch bytes."""
    quilt_id, ident = blob_id.split("::", 1)
    for agg in AGGREGATORS:
        try:
            r = _session().get(f"{agg}/v1/quilts/{quilt_id}/patches", timeout=timeout)
            if r.status_code != 200:
                continue
            pid = next((p.get("patch_id") for p in r.json() if p.get("identifier") == ident), None)
            if not pid:
                continue
            pr = _session().get(f"{agg}/v1/blobs/by-quilt-patch-id/{pid}", timeout=timeout)
            if pr.status_code == 200 and pr.content:
                return pr.content
        except Exception:
            continue
    return None


def robust_fetch(blob_id: str, timeout: int = 60) -> bytes | None:
    if "::" in blob_id:
        return _fetch_patch(blob_id, timeout)
    for agg in AGGREGATORS:
        for attempt in range(2):
            try:
                r = _session().get(f"{agg}/v1/blobs/{blob_id}", timeout=timeout)
                if r.status_code == 200 and r.content:
                    return r.content
            except Exception:
                pass
    return None


def load_audio_store() -> VectorStore:
    store = VectorStore(
        index_path=AUDIO_STORE_DIR / "vector_index.faiss",
        embedding_dim=clap.CLAP_DIM,
    )
    try:
        store.load()
    except Exception as e:
        print(f"[audio-index] fresh store ({e})")
    return store


def fetch_active_audio(limit: int | None) -> list[dict]:
    conn = sqlite3.connect(CATALOG_DB)
    conn.row_factory = sqlite3.Row
    q = (
        "SELECT blob_id, extension, mime_type, size, end_epoch, owner, is_nsfw "
        "FROM blobs WHERE kind='audio' AND is_active=1"
    )
    if limit:
        q += f" LIMIT {int(limit)}"
    rows = [dict(r) for r in conn.execute(q).fetchall()]
    conn.close()
    return rows


def embed_one(row: dict):
    blob_id = row["blob_id"]
    try:
        if int(row.get("size") or 0) > MAX_FETCH_BYTES:
            return blob_id, None, "too_large"
    except Exception:
        pass
    data = robust_fetch(blob_id)
    if not data:
        return blob_id, None, "fetch_failed"
    if len(data) > MAX_FETCH_BYTES:
        return blob_id, None, "too_large"
    emb = clap.embed_audio(data, row.get("extension") or "", blob_id)
    if emb is None:
        return blob_id, None, "embed_failed"
    return blob_id, emb, "ok"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--reindex", action="store_true", help="re-embed blobs already in the store")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not clap._ensure_loaded():
        print("[audio-index] CLAP model unavailable; aborting", file=sys.stderr)
        return 1

    store = load_audio_store()
    existing = set(store.metadata.keys())
    rows = fetch_active_audio(args.limit)
    todo = [r for r in rows if args.reindex or r["blob_id"] not in existing]
    print(f"[audio-index] active audio={len(rows)} already={len(existing)} to_process={len(todo)}")
    if args.dry_run or not todo:
        return 0

    ok = fetch_fail = embed_fail = 0
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(embed_one, r): r for r in todo}
        for fut in as_completed(futs):
            row = futs[fut]
            blob_id, emb, status = fut.result()
            done += 1
            if status == "ok":
                store.add(
                    embedding=emb,
                    blob_id=blob_id,
                    mime_type=row.get("mime_type") or "audio",
                    size=int(row.get("size") or 0),
                    extension=row.get("extension"),
                    kind="audio",
                    is_nsfw=bool(row.get("is_nsfw")),
                    end_epoch=row.get("end_epoch"),
                    owner=row.get("owner"),
                )
                ok += 1
            elif status == "fetch_failed":
                fetch_fail += 1
            else:
                embed_fail += 1
            if done % 25 == 0:
                print(f"[audio-index] {done}/{len(todo)} ok={ok} fetch_fail={fetch_fail} embed_fail={embed_fail}")
            if ok and ok % args.save_every == 0:
                store.save(create_backup=False)
                print(f"[audio-index] checkpoint saved ({ok} embedded)")

    store.save(create_backup=False)
    print(f"[audio-index] DONE ok={ok} fetch_fail={fetch_fail} embed_fail={embed_fail} store_size={store.size()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
