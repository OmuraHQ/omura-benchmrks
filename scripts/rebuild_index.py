"""Rebuild the FAISS search index from all active identified image blobs.

Queries blob_catalog.sqlite for every blob with:
  is_active=1 AND kind='image'

Fetches each via the aggregator pool (load-balanced, health-aware round-robin),
generates an embedding, and builds a fresh FAISS index.  Existing index is
backed-up then replaced atomically on success.

After the rebuild a sweep pass probes every blob now in the new index and
removes any that 404 across all passes (default 2) — also via the pool.

Usage:
  uv run python scripts/rebuild_index.py
  uv run python scripts/rebuild_index.py --workers 32 --sweep-passes 2
  uv run python scripts/rebuild_index.py --no-sweep   # skip the sweep phase

Env:
  WALRUS_AGGREGATOR_URLS   comma-separated pool (default: built-in list)
  WALRUS_AGGREGATOR_URL    single fallback
  OMURA_SWEEP_TIMEOUT      per-request timeout for sweep probes (default 10)
  OMURA_SWEEP_404_MIN      sweep confirmation passes (default 2)
  OMURA_INDEXER_WORKERS    default worker count if --workers not given
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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from omura.utils.aggregator_pool import get_pool  # noqa: E402
from omura.utils.blob_catalog import CATALOG_DB_PATH  # noqa: E402
from omura.utils.vector_store import VectorStore  # noqa: E402
from omura.parsers.file_detection import detect_file_type  # noqa: E402
from omura.parsers.multimodal import is_supported_image  # noqa: E402
from omura.utils.imagebind_embeddings import generate_image_embedding  # noqa: E402

SWEEP_TIMEOUT = float(os.getenv("OMURA_SWEEP_TIMEOUT", "10"))
SWEEP_404_MIN = int(os.getenv("OMURA_SWEEP_404_MIN", "2"))
DEFAULT_WORKERS = int(os.getenv("OMURA_INDEXER_WORKERS", "16"))

_tls = threading.local()


def _session():
    import requests

    s = getattr(_tls, "s", None)
    if s is None:
        import requests.adapters

        s = requests.Session()
        a = requests.adapters.HTTPAdapter(
            pool_connections=4,
            pool_maxsize=64,
            max_retries=requests.adapters.Retry(
                total=1,
                backoff_factor=0.3,
                status_forcelist=[502, 503, 504],
                allowed_methods=["HEAD", "GET"],
            ),
        )
        s.mount("https://", a)
        s.mount("http://", a)
        _tls.s = s
    return s


# ── DB helpers ─────────────────────────────────────────────────────────────────


def load_active_image_blobs() -> List[Tuple[str, Optional[str], Optional[int]]]:
    """Return (blob_id, extension, size) for every active identified image blob."""
    with sqlite3.connect(str(CATALOG_DB_PATH), timeout=30) as conn:
        rows = conn.execute(
            """
            SELECT blob_id, extension, size
            FROM blobs
            WHERE is_active = 1
              AND kind = 'image'
            ORDER BY blob_id
            """
        ).fetchall()
    return [(r[0], r[1], r[2]) for r in rows]


def _has_caption(blob_id: str) -> bool:
    try:
        with sqlite3.connect(str(CATALOG_DB_PATH), timeout=10) as conn:
            row = conn.execute(
                "SELECT caption FROM blobs WHERE blob_id = ?", (blob_id,)
            ).fetchone()
        return bool(row and row[0] and str(row[0]).strip())
    except Exception:
        return False


def _set_nsfw(blob_id: str, score: float, is_nsfw: bool) -> None:
    try:
        with sqlite3.connect(str(CATALOG_DB_PATH), timeout=10) as conn:
            conn.execute(
                "UPDATE blobs SET nsfw_score = ?, is_nsfw = ?, last_updated_at = datetime('now') WHERE blob_id = ?",
                (float(score), 1 if is_nsfw else 0, blob_id),
            )
            conn.commit()
    except Exception as e:
        print(f"  nsfw write failed for {blob_id[:12]}: {e}")


def _set_caption(blob_id: str, caption: str) -> None:
    if not caption:
        return
    try:
        with sqlite3.connect(str(CATALOG_DB_PATH), timeout=10) as conn:
            conn.execute(
                "UPDATE blobs SET caption = ?, last_updated_at = datetime('now') WHERE blob_id = ?",
                (caption, blob_id),
            )
            conn.commit()
    except Exception as e:
        print(f"  caption write failed for {blob_id[:12]}: {e}")


def _mark_catalog_indexed(blob_id: str, indexed: bool, status: str) -> None:
    try:
        with sqlite3.connect(str(CATALOG_DB_PATH), timeout=10) as conn:
            conn.execute(
                """
                UPDATE blobs
                   SET indexed = ?,
                       status  = ?,
                       last_updated_at = datetime('now')
                 WHERE blob_id = ?
                """,
                (1 if indexed else 0, status, blob_id),
            )
            conn.commit()
    except Exception:
        pass


# ── Fetch ──────────────────────────────────────────────────────────────────────


def _fetch(blob_id: str, timeout: float = 60.0) -> Optional[bytes]:
    resp, _ = get_pool().get(
        f"/v1/blobs/{blob_id}", session=_session(), timeout=timeout
    )
    if resp is None or resp.status_code != 200:
        return None
    return resp.content


# ── Sweep helpers ──────────────────────────────────────────────────────────────


def _probe(blob_id: str) -> Tuple[str, int]:
    """HEAD probe via pool; fall back to GET range 0-0. Returns (id, status_code)."""
    resp, _ = get_pool().head(
        f"/v1/blobs/{blob_id}",
        session=_session(),
        timeout=SWEEP_TIMEOUT,
        allow_redirects=True,
    )
    if resp is None:
        return blob_id, -1
    if resp.status_code in (405, 501):
        resp, _ = get_pool().get(
            f"/v1/blobs/{blob_id}",
            session=_session(),
            headers={"Range": "bytes=0-0"},
            timeout=SWEEP_TIMEOUT,
            stream=True,
        )
        if resp is not None:
            resp.close()
        if resp is None:
            return blob_id, -1
    return blob_id, resp.status_code


def sweep(store: VectorStore, passes: int, workers: int) -> int:
    """Remove from *store* all blobs that 404 across *passes* consecutive checks.

    Returns number of blobs removed.
    """
    candidates = set(store.metadata.keys())
    total = len(candidates)
    if not candidates:
        print("  sweep: nothing to probe.")
        return 0

    print(f"\n── Sweep phase: {total} blobs, {passes} pass(es), {workers} workers ──")
    for pass_n in range(1, passes + 1):
        if not candidates:
            break
        confirmed_404: List[str] = []
        status_counts: Dict[int, int] = {}
        start = time.time()
        print(f"  Pass {pass_n}/{passes}: probing {len(candidates)} blobs …")
        with ThreadPoolExecutor(max_workers=workers) as ex:
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
                    print(f"    {done}/{len(futs)}  ({done/elapsed:.1f} req/s)")
        elapsed = time.time() - start
        print(f"  pass done in {elapsed:.1f}s  statuses={dict(sorted(status_counts.items()))}")
        candidates = set(confirmed_404)

    final_404 = candidates
    if not final_404:
        print("  sweep: no broken blobs found.")
        return 0

    print(f"  sweep: removing {len(final_404)} confirmed-404 blobs …")
    removed = store.remove_blob_ids(list(final_404))
    print(f"  sweep: removed {removed}.")
    return removed


# ── Worker ─────────────────────────────────────────────────────────────────────


def _process_one(
    blob_id: str,
    extension: Optional[str],
    size: Optional[int],
    store: VectorStore,
    store_lock: threading.Lock,
    caption: bool = False,
    nsfw: bool = False,
) -> str:
    """Fetch → detect → (caption/nsfw) → embed → add to store. Returns outcome string."""
    if blob_id in store.metadata:
        return "skip_already"

    content = _fetch(blob_id)
    if content is None:
        return "fetch_failed"

    mime, ext, kind = detect_file_type(content)
    if kind != "image" or not is_supported_image(ext):
        return f"unsupported:{kind}/{ext}"

    emb = generate_image_embedding(content, blob_id=blob_id)
    if emb is None:
        return "embed_failed"

    # NSFW label (Gemma 4 VL) — written to the catalog and carried into the index metadata.
    is_nsfw_val = False
    if nsfw:
        try:
            from omura.utils.nsfw_labeler import classify_nsfw

            res = classify_nsfw(content)
            if res is not None:
                score, is_nsfw_val, _label = res
                _set_nsfw(blob_id, score, is_nsfw_val)
        except Exception as e:
            print(f"  nsfw failed for {blob_id[:12]}: {e}")

    # Smart caption (Gemma 4 VL via captioning.py) — written to the catalog so search
    # display + any caption-based retrieval use the upgraded captions. Skip blobs that
    # already have a caption (captions persist across index wipes), so a re-run only
    # captions the missing ones instead of redoing the whole (slow) reasoning pass.
    if caption and not _has_caption(blob_id):
        try:
            from omura.utils.captioning import generate_caption

            _set_caption(blob_id, generate_caption(content))
        except Exception as e:
            print(f"  caption failed for {blob_id[:12]}: {e}")

    with store_lock:
        store.add(
            embedding=emb,
            blob_id=blob_id,
            mime_type=mime,
            size=len(content),
            extension=ext,
            kind=kind,
            is_nsfw=is_nsfw_val,
        )
    return "indexed"


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument(
        "--sweep-passes",
        type=int,
        default=SWEEP_404_MIN,
        help="confirmation passes for the post-rebuild sweep",
    )
    parser.add_argument(
        "--no-sweep", action="store_true", help="skip the sweep phase"
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=100,
        help="checkpoint-save the index every N newly indexed blobs",
    )
    parser.add_argument(
        "--caption",
        action="store_true",
        help="also (re)generate a smart caption per blob via captioning.py and write it to the catalog",
    )
    parser.add_argument(
        "--nsfw",
        action="store_true",
        help="also label NSFW per blob via the Gemma 4 VL labeler and write nsfw_score/is_nsfw to the catalog",
    )
    args = parser.parse_args()

    print("═" * 60)
    print("  Omura index rebuild")
    print("═" * 60)
    print(f"  workers      : {args.workers}")
    print(f"  sweep passes : {'disabled' if args.no_sweep else args.sweep_passes}")
    print(f"  save every   : {args.save_every}")
    print()

    # ── Load blob list ──────────────────────────────────────────────────────────
    print("Loading active image blobs from catalog …")
    blobs = load_active_image_blobs()
    print(f"  Found {len(blobs)} active image blobs.")
    if not blobs:
        print("Nothing to index.")
        return 0

    # ── Prepare fresh store ─────────────────────────────────────────────────────
    print("\nPreparing fresh vector store …")
    store = VectorStore()

    # Wipe stale live-index files so the new index loads clean on next start.
    from omura.utils.vector_store import VECTOR_STORE_DIR

    for fname in ("vector_index.faiss", "metadata.json", "cursor.json",
                  "embeddings.npy", "blob_id_mapping.npy"):
        stale = VECTOR_STORE_DIR / fname
        if stale.exists():
            stale.unlink()
            print(f"  removed stale {fname}")

    # Start fresh — no old data carried over
    print("  Starting with empty index.")

    store_lock = threading.Lock()

    # ── Embed all blobs ─────────────────────────────────────────────────────────
    print(f"\n── Embed phase: {len(blobs)} blobs, {args.workers} workers ──")
    counters: Dict[str, int] = {}
    newly_indexed = 0
    t_start = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(_process_one, bid, ext, sz, store, store_lock, args.caption, args.nsfw): bid
            for bid, ext, sz in blobs
        }
        done = 0
        for fut in as_completed(futs):
            bid = futs[fut]
            try:
                outcome = fut.result()
            except Exception as e:
                outcome = f"error:{e}"
            counters[outcome] = counters.get(outcome, 0) + 1
            if outcome == "indexed":
                newly_indexed += 1
                _mark_catalog_indexed(bid, True, "indexed")
                # Periodic checkpoint save
                if newly_indexed % args.save_every == 0:
                    with store_lock:
                        store.save(create_backup=False)
                    print(
                        f"  checkpoint saved at {newly_indexed} indexed "
                        f"({done+1}/{len(blobs)} done)"
                    )
            elif outcome.startswith("fetch_failed") or outcome.startswith("embed_failed"):
                _mark_catalog_indexed(bid, False, outcome.split(":")[0])
            done += 1
            if done % 200 == 0:
                elapsed = time.time() - t_start
                rate = done / elapsed if elapsed else 0
                print(f"  {done}/{len(blobs)}  ({rate:.1f} blobs/s)  indexed={newly_indexed}")

    elapsed = time.time() - t_start
    print(f"\nEmbed phase done in {elapsed:.1f}s")
    print(f"  Results: {dict(sorted(counters.items()))}")
    print(f"  Store size: {len(store.metadata)}")

    # ── Sweep phase ─────────────────────────────────────────────────────────────
    removed = 0
    if not args.no_sweep and len(store.metadata) > 0:
        removed = sweep(store, passes=args.sweep_passes, workers=args.workers)

    # ── Final save ──────────────────────────────────────────────────────────────
    print("\nSaving index …")
    store.save(create_backup=False)
    print(f"Done. Final store size: {len(store.metadata)} blobs  (removed by sweep: {removed})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
