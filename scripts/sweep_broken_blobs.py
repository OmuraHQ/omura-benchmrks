"""Probe every blob_id in the vector store; remove those the aggregator can't serve.

A blob is considered broken when the aggregator returns 404 for ``/v1/blobs/<id>``.
Other status codes (400 malformed id, 451 blocked, 503 transient, 200 OK) are
preserved — only 404 (definitively gone) is removed.

Fetches are load-balanced across the full aggregator pool (health-aware round-robin)
via ``WALRUS_AGGREGATOR_URLS`` / ``WALRUS_AGGREGATOR_URL``.  If neither env var is
set, the built-in pool of public aggregators is used automatically.

Usage:
  uv run python scripts/sweep_broken_blobs.py            # check + remove + save
  uv run python scripts/sweep_broken_blobs.py --dry-run  # only report
  uv run python scripts/sweep_broken_blobs.py --workers 32

Env:
  WALRUS_AGGREGATOR_URLS   comma-separated pool (default: built-in list)
  WALRUS_AGGREGATOR_URL    single fallback (back-compat with existing code)
  OMURA_SWEEP_404_MIN      a blob must 404 this many times before removal (default 2)
  OMURA_SWEEP_TIMEOUT      per-request timeout seconds (default 10)
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import sqlite3  # noqa: E402

from omura.utils.aggregator_pool import get_pool  # noqa: E402
from omura.utils.blob_catalog import CATALOG_DB_PATH  # noqa: E402
from omura.utils.vector_store import VectorStore  # noqa: E402

TIMEOUT = float(os.getenv("OMURA_SWEEP_TIMEOUT", "10"))
MIN_404_HITS = int(os.getenv("OMURA_SWEEP_404_MIN", "2"))

_tls = threading.local()


def _session():
    import requests
    import requests.adapters

    s = getattr(_tls, "s", None)
    if s is None:
        s = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=4,
            pool_maxsize=64,
            max_retries=requests.adapters.Retry(
                total=1,
                backoff_factor=0.3,
                status_forcelist=[502, 503, 504],
                allowed_methods=["HEAD", "GET"],
            ),
        )
        s.mount("https://", adapter)
        s.mount("http://", adapter)
        _tls.s = s
    return s


def _blob_path(blob_id: str) -> str:
    """Aggregator path for a blob — plain or quilt-patch ('<quilt>::<identifier>')."""
    if "::" in blob_id:
        import requests as _rq
        quilt, ident = blob_id.split("::", 1)
        return f"/v1/blobs/by-quilt-id/{quilt}/{_rq.utils.quote(ident, safe='')}"
    return f"/v1/blobs/{blob_id}"


def _probe(blob_id: str) -> Tuple[str, int]:
    """HEAD probe via aggregator pool; fall back to GET range 0-0 on 405/501.

    Returns (blob_id, http_status_code).  -1 means a network error.
    """
    path = _blob_path(blob_id)
    resp, used_url = get_pool().head(
        path,
        session=_session(),
        timeout=TIMEOUT,
        allow_redirects=True,
    )
    if resp is None:
        return blob_id, -1
    if resp.status_code in (405, 501):
        resp, _ = get_pool().get(
            path,
            session=_session(),
            headers={"Range": "bytes=0-0"},
            timeout=TIMEOUT,
            stream=True,
        )
        if resp is not None:
            resp.close()
        if resp is None:
            return blob_id, -1
    return blob_id, resp.status_code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="don't modify the store")
    parser.add_argument("--workers", type=int, default=16, help="concurrent probes")
    parser.add_argument(
        "--passes",
        type=int,
        default=MIN_404_HITS,
        help="confirm 404 across this many passes before removal",
    )
    parser.add_argument(
        "--store",
        choices=["default", "audio", "video"],
        default="default",
        help="which modality store to sweep (audio=CLAP 512-d, video=IV2 768-d)",
    )
    args = parser.parse_args()

    pool = get_pool()
    print(f"Aggregator pool upstreams ({len(pool.upstreams)}):")
    for u in pool.upstreams:
        print(f"  {u.url}")
    print(f"\nLoading vector store… (store={args.store})")
    if args.store == "audio":
        d = Path(os.getenv("OMURA_AUDIO_VECTOR_STORE_DIR", "data/vector_index_clap"))
        store = VectorStore(index_path=d / "vector_index.faiss",
                            embedding_dim=int(os.getenv("OMURA_AUDIO_EMBEDDING_DIM", "512")))
    elif args.store == "video":
        d = Path(os.getenv("OMURA_VIDEO_VECTOR_STORE_DIR", "data/vector_index_iv2"))
        store = VectorStore(index_path=d / "vector_index.faiss",
                            embedding_dim=int(os.getenv("OMURA_VIDEO_EMBEDDING_DIM", "768")))
    else:
        store = VectorStore()
    store.load()
    all_ids = list(store.metadata.keys())
    total = len(all_ids)
    print(f"Loaded {total} blob ids.")

    if total == 0:
        print("Nothing to sweep.")
        return 0

    candidates = set(all_ids)
    final_404: set = set()

    for pass_n in range(1, args.passes + 1):
        if not candidates:
            break
        print(
            f"\n=== Pass {pass_n}/{args.passes}: probing {len(candidates)} blobs "
            f"({args.workers} workers) ==="
        )
        status_counts: Dict[int, int] = {}
        confirmed_404: List[str] = []

        start = time.time()
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(_probe, b) for b in candidates]
            done = 0
            for fut in as_completed(futs):
                bid, code = fut.result()
                status_counts[code] = status_counts.get(code, 0) + 1
                if code == 404:
                    confirmed_404.append(bid)
                done += 1
                if done % 200 == 0:
                    elapsed = time.time() - start
                    rate = done / elapsed if elapsed > 0 else 0
                    print(f"  {done}/{len(futs)}  ({rate:.1f} req/s)")
        elapsed = time.time() - start
        print(f"  pass complete in {elapsed:.1f}s")
        print(f"  status distribution: {dict(sorted(status_counts.items()))}")

        candidates = set(confirmed_404)
        if pass_n == args.passes:
            final_404 = candidates

    print(f"\nFinal: {len(final_404)} blobs returned 404 in all {args.passes} passes.")
    if args.dry_run:
        print("--dry-run: not modifying store.")
        for b in list(final_404)[:10]:
            print(f"  would remove: {b}")
        if len(final_404) > 10:
            print(f"  … and {len(final_404) - 10} more")
        return 0

    if not final_404:
        print("Nothing to remove.")
        return 0

    # Persist blacklist alongside the store so the running server's indexer
    # (which may still hold the old blobs in memory) also drops them on its
    # next periodic save — preventing the "broken blobs come back" race.
    blacklist_path = store.index_path.parent / "deleted_blobs.json"
    try:
        import json as _json
        existing: set = set()
        if blacklist_path.exists():
            with open(blacklist_path) as _f:
                existing = set(_json.load(_f))
        existing.update(final_404)
        with open(blacklist_path, "w") as _f:
            _json.dump(sorted(existing), _f)
        print(f"Blacklist written: {blacklist_path} ({len(existing)} total entries)")
    except Exception as e:
        print(f"Warning: could not write sweep blacklist: {e}")

    # Mark them deleted in the catalog too so the FTS/text-search path (which filters
    # on is_active = 1) stops surfacing them, and drop their full-text rows.
    try:
        with sqlite3.connect(str(CATALOG_DB_PATH), timeout=30) as conn:
            conn.executemany(
                "UPDATE blobs SET is_active = 0, status = 'expired' WHERE blob_id = ?",
                [(b,) for b in final_404],
            )
            conn.executemany(
                "DELETE FROM blobs_fts WHERE blob_id = ?",
                [(b,) for b in final_404],
            )
            conn.commit()
        print(f"Marked {len(final_404)} blobs is_active=0/expired in catalog.")
    except Exception as e:
        print(f"Warning: could not mark swept blobs in catalog: {e}")

    print(f"Removing {len(final_404)} blobs from vector store…")
    removed = store.remove_blob_ids(list(final_404))
    print(f"Removed: {removed}. Saving…")
    store.save()
    print(f"Done. Store now has {len(store.metadata)} blobs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
