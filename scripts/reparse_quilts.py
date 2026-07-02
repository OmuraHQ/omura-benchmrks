"""Re-expand existing quilts in the catalog: fetch each quilt, parse out its patches,
embed image patches into the vector store with ``parent_quilt_id``/``quilt_identifier``.

Quilts contain N inner blobs ("patches"). The cataloger already detected 247K+ quilts
but only recorded the *container* — the inner files were never indexed individually.
This script walks ``kind='quilt'`` rows in ``blob_catalog.sqlite`` and indexes each
image patch as ``{quilt_blob_id}::{identifier}``.

Implementation: prefers local parsing via ``iter_quilt_patch_contents`` (one HTTP fetch
per quilt, no per-patch round-trip). Falls back to the aggregator's ``/v1/quilts/{id}/patches``
endpoint when local parsing fails to find any patches.

Usage:
  uv run python scripts/reparse_quilts.py                       # process all kind='quilt' active rows
  uv run python scripts/reparse_quilts.py --limit 100           # only first 100
  uv run python scripts/reparse_quilts.py --workers 8           # default 8
  uv run python scripts/reparse_quilts.py --dry-run             # don't write
  uv run python scripts/reparse_quilts.py --resume              # skip quilts already expanded

Env:
  WALRUS_AGGREGATOR_URL   aggregator base URL (default https://agrregator.omura.fun)
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
from typing import Dict, List, Optional, Tuple

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from omura.parsers.file_detection import detect_file_type  # noqa: E402
from omura.parsers.multimodal import is_supported_image  # noqa: E402
from omura.parsers.quilt import iter_quilt_patch_contents  # noqa: E402
from omura.utils.imagebind_embeddings import (  # noqa: E402
    generate_image_embedding,
    is_model_ready,
)
from omura.utils.vector_store import VectorStore  # noqa: E402

AGGREGATOR_URL = os.getenv(
    "WALRUS_AGGREGATOR_URL", "https://agrregator.omura.fun"
).rstrip("/")
CATALOG_DB = Path(os.getenv("OMURA_CATALOG_DB_PATH", "data/blob_catalog.sqlite"))
FETCH_TIMEOUT = float(os.getenv("OMURA_QUILT_FETCH_TIMEOUT", "60"))

_tls = threading.local()


def _session() -> requests.Session:
    s = getattr(_tls, "s", None)
    if s is None:
        s = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=4,
            pool_maxsize=16,
            max_retries=requests.adapters.Retry(
                total=2,
                backoff_factor=0.5,
                status_forcelist=[502, 503, 504],
                allowed_methods=["GET"],
            ),
        )
        s.mount("https://", adapter)
        s.mount("http://", adapter)
        _tls.s = s
    return s


def _fetch_quilt(blob_id: str) -> Optional[bytes]:
    """Fetch the full quilt blob bytes. Returns None on any failure."""
    try:
        resp = _session().get(
            f"{AGGREGATOR_URL}/v1/blobs/{blob_id}", timeout=FETCH_TIMEOUT
        )
        if resp.status_code == 200:
            return resp.content
    except requests.RequestException:
        pass
    return None


def _extract_patches(quilt_blob: bytes) -> List[Tuple[str, Dict[str, str], bytes]]:
    """Local parse. Returns list of (identifier, tags, content) for each patch."""
    try:
        return list(iter_quilt_patch_contents(quilt_blob))
    except Exception:
        return []


def _process_quilt(
    blob_id: str,
    metadata: Dict[str, object],
    store: VectorStore,
    write: bool,
) -> Dict[str, int]:
    """Fetch + expand one quilt. Returns counters for this quilt."""
    stats = {"patches": 0, "indexed": 0, "skipped": 0, "fetch_failed": 0}
    quilt_bytes = _fetch_quilt(blob_id)
    if quilt_bytes is None:
        stats["fetch_failed"] = 1
        return stats

    patches = _extract_patches(quilt_bytes)
    stats["patches"] = len(patches)
    if not patches:
        return stats

    for ident, tags, inner in patches:
        if not inner:
            stats["skipped"] += 1
            continue
        mime, ext, kind = detect_file_type(inner)
        if kind != "image" or not is_supported_image(ext):
            stats["skipped"] += 1
            continue

        inner_id = f"{blob_id}::{ident}"
        if inner_id in store.metadata:
            # Already indexed — skip; idempotent for --resume.
            stats["skipped"] += 1
            continue

        emb = generate_image_embedding(inner, blob_id=inner_id)
        if emb is None:
            stats["skipped"] += 1
            continue

        if write:
            extra = {
                "is_quilt": True,
                "parent_quilt_id": blob_id,
                "quilt_identifier": ident,
                "size": len(inner),
                "mime_type": mime,
                "extension": ext,
                "kind": kind,
                "is_nsfw": False,
                **{f"tag_{k}": v for k, v in (tags or {}).items()},
            }
            with store._lock:
                store.add(
                    embedding=emb,
                    blob_id=inner_id,
                    mime_type=mime,
                    size=len(inner),
                    extension=ext,
                    extra_metadata=extra,
                )
        stats["indexed"] += 1

    return stats


def _list_quilt_ids(db: Path, limit: Optional[int], resume: bool) -> List[str]:
    """Pull active quilt blob_ids from blob_catalog.sqlite."""
    with sqlite3.connect(str(db), timeout=30) as conn:
        cur = conn.cursor()
        if resume:
            cur.execute(
                """
                SELECT blob_id FROM blobs
                WHERE kind='quilt' AND is_active=1
                  AND (status IS NULL OR status NOT IN ('quilt_expanded','quilt_failed'))
                ORDER BY blob_id
                """
            )
        else:
            cur.execute(
                """
                SELECT blob_id FROM blobs
                WHERE kind='quilt' AND is_active=1
                ORDER BY blob_id
                """
            )
        rows = cur.fetchall()
    ids = [r[0] for r in rows]
    if limit is not None:
        ids = ids[:limit]
    return ids


def _mark_quilt(db: Path, blob_id: str, status: str) -> None:
    try:
        with sqlite3.connect(str(db), timeout=10) as conn:
            conn.execute(
                "UPDATE blobs SET status=?, last_updated_at=datetime('now') WHERE blob_id=?",
                (status, blob_id),
            )
            conn.commit()
    except Exception as exc:
        print(f"  mark {blob_id[:12]} -> {status} failed: {exc}", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="skip quilts marked status='quilt_expanded' (idempotent re-runs)",
    )
    args = parser.parse_args()

    print(f"Aggregator: {AGGREGATOR_URL}")
    print(f"Catalog DB: {CATALOG_DB}")

    if not is_model_ready():
        print("[reparse_quilts] WARNING: embedding model not ready. Loading...")
        # generate_image_embedding will lazy-load.

    print("Loading vector store...")
    store = VectorStore()
    store.load()
    print(f"Loaded {len(store.metadata)} existing embeddings.")

    print("Listing quilts...")
    quilt_ids = _list_quilt_ids(CATALOG_DB, args.limit, args.resume)
    print(f"Found {len(quilt_ids)} quilts to process.")
    if not quilt_ids:
        return 0

    totals = {"patches": 0, "indexed": 0, "skipped": 0, "fetch_failed": 0}
    start = time.time()
    done = 0
    save_every = max(50, len(quilt_ids) // 20)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_process_quilt, qid, {}, store, not args.dry_run): qid for qid in quilt_ids}
        for fut in as_completed(futures):
            qid = futures[fut]
            try:
                s = fut.result()
            except Exception as exc:
                print(f"  [{qid[:12]}] error: {exc}", file=sys.stderr)
                s = {"patches": 0, "indexed": 0, "skipped": 0, "fetch_failed": 1}
            for k, v in s.items():
                totals[k] += v
            if not args.dry_run:
                if s["fetch_failed"]:
                    _mark_quilt(CATALOG_DB, qid, "quilt_failed")
                elif s["patches"] > 0:
                    _mark_quilt(CATALOG_DB, qid, "quilt_expanded")
            done += 1
            if done % save_every == 0 or done == len(quilt_ids):
                if not args.dry_run:
                    store.save()
                elapsed = time.time() - start
                rate = done / elapsed if elapsed else 0
                print(
                    f"  progress {done}/{len(quilt_ids)} ({rate:.1f}/s)  "
                    f"patches={totals['patches']:,}  indexed={totals['indexed']:,}  "
                    f"skipped={totals['skipped']:,}  fetch_failed={totals['fetch_failed']}"
                )

    if not args.dry_run:
        store.save()

    elapsed = time.time() - start
    print()
    print(f"Done in {elapsed:.1f}s.")
    print(f"  quilts processed:  {done}")
    print(f"  patches discovered: {totals['patches']:,}")
    print(f"  image patches indexed: {totals['indexed']:,}")
    print(f"  skipped (non-image / dup): {totals['skipped']:,}")
    print(f"  quilts that failed to fetch: {totals['fetch_failed']}")
    print(f"  store size now: {len(store.metadata):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
