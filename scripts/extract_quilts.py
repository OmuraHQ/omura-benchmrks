"""Standalone quilt extraction + indexer.

For every active ``kind='quilt'`` row in ``blob_catalog.sqlite``:
  1. List its inner patches via the documented Walrus aggregator endpoint
     ``/v1/quilts/{quilt_id}/patches`` (load-balanced across the aggregator pool).
  2. For patches whose identifier filename ends in an **audio** / **video** extension,
     record a catalog row directly (no fetch — the aggregator's per-patch fetch is
     prohibitively slow for non-image media and we don't have an audio embedder by
     default anyway).
  3. For other patches, fetch the bytes via ``/v1/blobs/by-quilt-id/{quilt_id}/{ident}``,
     run file-type detection, and:
       - image patches → embed into the vector store + catalog as ``kind=image``
       - audio patches (via byte sniff or filename rescue) → catalog only
       - everything else → catalog as ``kind=<detected>``
  4. Mark the quilt's row ``status='quilt_expanded'`` (or ``quilt_failed`` if the patch
     list endpoint failed). Idempotent — re-running with ``--resume`` skips finished quilts.

Coordination with the running indexer: the indexer keeps the vector store in memory
and writes it back periodically; if you run this script while the indexer is up, its
save will overwrite your additions. **Stop the indexer first**, or use the in-process
admin endpoint (``POST /admin/reparse-quilts``) instead.

Usage:
  uv run python scripts/extract_quilts.py                       # process all unfinished
  uv run python scripts/extract_quilts.py --limit 100           # cap
  uv run python scripts/extract_quilts.py --workers 8           # parallel quilts
  uv run python scripts/extract_quilts.py --kinds-fastpath audio
  uv run python scripts/extract_quilts.py --no-fetch            # name-only catalog (super fast)
  uv run python scripts/extract_quilts.py --dry-run             # don't write

Env:
  WALRUS_AGGREGATOR_URLS    comma-separated pool of aggregator base URLs
  WALRUS_AGGREGATOR_URL     single fallback (back-compat)
  OMURA_CATALOG_DB_PATH     default data/blob_catalog.sqlite
  OMURA_QUILT_FETCH_TIMEOUT default 300
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
from typing import Any, Dict, List, Optional, Tuple

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from omura.parsers.file_detection import detect_file_type  # noqa: E402
from omura.parsers.multimodal import (  # noqa: E402
    is_supported_audio,
    is_supported_image,
)
from omura.utils.aggregator_pool import get_pool  # noqa: E402
from omura.utils.imagebind_embeddings import (  # noqa: E402
    generate_image_embedding,
    is_model_ready,
)
from omura.utils.vector_store import VectorStore  # noqa: E402

CATALOG_DB = Path(os.getenv("OMURA_CATALOG_DB_PATH", "data/blob_catalog.sqlite"))
FETCH_TIMEOUT = float(os.getenv("OMURA_QUILT_FETCH_TIMEOUT", "300"))

_AUDIO_EXTS = {
    "mp3", "wav", "wave", "flac", "ogg", "oga", "opus",
    "m4a", "m4b", "aac", "aif", "aiff", "wma", "amr",
}
_VIDEO_EXTS = {
    "mp4", "m4v", "mov", "webm", "mkv", "avi", "flv", "wmv", "mpg", "mpeg",
}
_IMAGE_EXTS = {
    "png", "jpg", "jpeg", "webp", "gif", "bmp", "tif", "tiff",
    "heic", "heif", "avif", "ico", "svg",
}
_DOC_EXTS = {"pdf", "doc", "docx", "rtf", "odt", "epub", "md", "mobi"}
_TEXT_EXTS = {"txt", "json", "yaml", "yml", "xml", "html", "htm", "csv", "tsv", "log"}
_ARCHIVE_EXTS = {"zip", "tar", "gz", "tgz", "rar", "7z", "bz2", "xz", "zst"}
_CODE_EXTS = {"py", "js", "ts", "rs", "go", "java", "cpp", "c", "h", "rb", "php", "sh"}
_DATA_EXTS = {"parquet", "arrow", "feather", "orc", "avro", "msgpack", "pb", "proto"}


def _kind_from_ext(ext: str) -> Optional[str]:
    """Best-effort kind from a filename extension. None when unrecognized."""
    if not ext:
        return None
    if ext in _AUDIO_EXTS: return "audio"
    if ext in _VIDEO_EXTS: return "video"
    if ext in _IMAGE_EXTS: return "image"
    if ext in _DOC_EXTS: return "doc"
    if ext in _TEXT_EXTS: return "text"
    if ext in _ARCHIVE_EXTS: return "archive"
    if ext in _CODE_EXTS: return "code"
    if ext in _DATA_EXTS: return "data"
    return None

_tls = threading.local()


def _session() -> requests.Session:
    s = getattr(_tls, "s", None)
    if s is None:
        s = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=4,
            pool_maxsize=16,
            max_retries=requests.adapters.Retry(
                total=1,
                backoff_factor=0.3,
                status_forcelist=[502, 503, 504],
                allowed_methods=["GET"],
            ),
        )
        s.mount("https://", adapter)
        s.mount("http://", adapter)
        _tls.s = s
    return s


def _ext_from_identifier(identifier: str) -> str:
    """Extract file extension from the BASENAME of a patch identifier.

    Identifiers can be path-like (``xrp_ledger.raw.transactions/2024-01-01/part_4``).
    A naive ``rsplit(".", 1)[-1]`` would yield ``raw.transactions/.../part_4`` which is
    nonsense. We take the segment after the last ``/`` first, then the final ``.``
    suffix of that segment — and only if it's short enough to plausibly be an
    extension (≤8 ASCII alphanumeric chars, the longest standard extension is 6).
    """
    basename = identifier.rsplit("/", 1)[-1]
    if "." not in basename:
        return ""
    ext = basename.rsplit(".", 1)[-1].lower().strip()
    if not ext or len(ext) > 8 or not ext.isalnum():
        return ""
    return ext


def _list_patches(quilt_id: str) -> List[Dict[str, Any]]:
    """GET /v1/quilts/{quilt_id}/patches via the pool. Returns [] on failure."""
    resp, _ = get_pool().get(
        f"/v1/quilts/{quilt_id}/patches", session=_session(), timeout=FETCH_TIMEOUT
    )
    if resp is None or resp.status_code != 200:
        return []
    try:
        data = resp.json()
    except ValueError:
        return []
    if not isinstance(data, list):
        return []
    out: List[Dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        ident = item.get("identifier")
        patch_id = item.get("patch_id")
        if isinstance(ident, str) and patch_id is not None:
            tags = item.get("tags") if isinstance(item.get("tags"), dict) else {}
            out.append({"identifier": ident, "patch_id": str(patch_id), "tags": tags})
    return out


def _fetch_patch(quilt_id: str, identifier: str) -> Optional[bytes]:
    safe = requests.utils.quote(identifier, safe="")
    resp, _ = get_pool().get(
        f"/v1/blobs/by-quilt-id/{quilt_id}/{safe}",
        session=_session(),
        timeout=FETCH_TIMEOUT,
    )
    if resp is None or resp.status_code != 200:
        return None
    return resp.content


_DB_INIT_LOCK = threading.Lock()
_DB_INITIALIZED = False


def _ensure_db_pragmas() -> None:
    """One-shot: switch SQLite to WAL mode so concurrent readers/writers don't block."""
    global _DB_INITIALIZED
    with _DB_INIT_LOCK:
        if _DB_INITIALIZED:
            return
        try:
            with sqlite3.connect(str(CATALOG_DB), timeout=30) as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")  # faster, still durable
                conn.execute("PRAGMA temp_store=MEMORY")
                conn.execute("PRAGMA cache_size=-65536")  # 64 MiB page cache
                conn.commit()
            _DB_INITIALIZED = True
        except Exception as exc:
            print(f"  WAL init error: {exc}", file=sys.stderr)


def _record_patch_in_catalog(
    inner_id: str, size: int, mime: str, ext: str, kind: str, indexed: bool
) -> None:
    """Legacy single-row write — kept for the slow image-fetch path."""
    _ensure_db_pragmas()
    try:
        with sqlite3.connect(str(CATALOG_DB), timeout=10) as conn:
            conn.execute(
                """
                INSERT INTO blobs (
                    blob_id, kind, mime_type, extension, size,
                    is_active, fetch_ok, status, indexed,
                    first_seen_at, last_updated_at
                ) VALUES (?, ?, ?, ?, ?, 1, ?, ?, ?, datetime('now'), datetime('now'))
                ON CONFLICT(blob_id) DO UPDATE SET
                    kind=excluded.kind,
                    mime_type=excluded.mime_type,
                    extension=excluded.extension,
                    size=excluded.size,
                    status=excluded.status,
                    indexed=excluded.indexed,
                    last_updated_at=datetime('now')
                """,
                (
                    inner_id, kind, mime, ext, size,
                    1 if size > 0 else 0,
                    "indexed" if indexed else "discovered",
                    1 if indexed else 0,
                ),
            )
            conn.commit()
    except Exception as exc:
        print(f"  [{inner_id[:18]}…] catalog write error: {exc}", file=sys.stderr)


def _batch_record_patches(rows: List[Tuple[str, int, str, str, str, bool]]) -> None:
    """Insert MANY catalog rows in a single transaction. Much faster than per-row writes."""
    if not rows:
        return
    _ensure_db_pragmas()
    params = [
        (
            inner_id, kind, mime, ext, size,
            1 if size > 0 else 0,
            "indexed" if indexed else "discovered",
            1 if indexed else 0,
        )
        for (inner_id, size, mime, ext, kind, indexed) in rows
    ]
    try:
        with sqlite3.connect(str(CATALOG_DB), timeout=30) as conn:
            conn.executemany(
                """
                INSERT INTO blobs (
                    blob_id, kind, mime_type, extension, size,
                    is_active, fetch_ok, status, indexed,
                    first_seen_at, last_updated_at
                ) VALUES (?, ?, ?, ?, ?, 1, ?, ?, ?, datetime('now'), datetime('now'))
                ON CONFLICT(blob_id) DO UPDATE SET
                    kind=excluded.kind,
                    mime_type=excluded.mime_type,
                    extension=excluded.extension,
                    size=COALESCE(NULLIF(excluded.size,0), blobs.size),
                    status=excluded.status,
                    indexed=excluded.indexed,
                    last_updated_at=datetime('now')
                """,
                params,
            )
            conn.commit()
    except Exception as exc:
        print(f"  batch catalog write error ({len(rows)} rows): {exc}", file=sys.stderr)


def _mark_quilt_status(blob_id: str, status: str) -> None:
    try:
        with sqlite3.connect(str(CATALOG_DB), timeout=10) as conn:
            conn.execute(
                "UPDATE blobs SET status=?, last_updated_at=datetime('now') WHERE blob_id=?",
                (status, blob_id),
            )
            conn.commit()
    except Exception:
        pass


def _kind_from_tags(tags: Dict[str, Any]) -> Optional[str]:
    """Map ``tags.content-type`` / ``content_type`` / ``mime`` → top-level kind."""
    if not isinstance(tags, dict):
        return None
    for key in ("content-type", "content_type", "Content-Type", "mime", "mimeType"):
        v = tags.get(key)
        if isinstance(v, str) and "/" in v:
            top = v.split("/", 1)[0].strip().lower()
            if top in ("image", "audio", "video", "application", "text"):
                return top
    return None


def _process_quilt(
    quilt_id: str,
    store: VectorStore,
    *,
    write: bool,
    no_fetch: bool,
    fastpath_kinds: set,
    skip_embed: bool,
) -> Dict[str, int]:
    """Extract + classify patches for one quilt. Returns counters.

    Improvements:
      - Image extension fast-path: catalog ``kind=image`` from filename without fetching
      - Tag-based classification: use ``tags.content-type`` from the patch listing
      - Image fetches happen within a per-quilt thread pool when many images are present
    """
    counters = {
        "patches": 0,
        "audio_discovered": 0,
        "video_discovered": 0,
        "image_discovered": 0,
        "image_indexed": 0,
        "non_media_catalogued": 0,
        "fetch_failed": 0,
        "skipped": 0,
    }

    patches = _list_patches(quilt_id)
    if not patches:
        counters["fetch_failed"] = 1
        return counters

    counters["patches"] = len(patches)
    image_jobs: List[Tuple[str, str, str]] = []
    # Buffer all catalog rows for this quilt and write them in one transaction
    # at the end. ~10x faster than per-patch commits under contention.
    batch_rows: List[Tuple[str, int, str, str, str, bool]] = []

    for item in patches:
        ident = item["identifier"]
        inner_id = f"{quilt_id}::{ident}"
        if inner_id in store.metadata:
            counters["skipped"] += 1
            continue

        name_ext = _ext_from_identifier(ident)
        tag_kind = _kind_from_tags(item.get("tags") or {})

        def _log_patch(kind: str, mime: str) -> None:
            short = ident if len(ident) <= 36 else ident[:33] + "…"
            print(
                f"    · {short:<36s}  ext={name_ext or '∅':<7s} kind={kind:<10s} mime={mime}",
                flush=True,
            )

        # Fast-path: audio by ext OR tag
        if (name_ext in _AUDIO_EXTS or tag_kind == "audio") and "audio" in fastpath_kinds:
            counters["audio_discovered"] += 1
            mime = f"audio/{name_ext}" if name_ext else "audio/unknown"
            if write:
                batch_rows.append((inner_id, 0, mime, name_ext or "", "audio", False))
            _log_patch("audio", mime)
            continue
        if (name_ext in _VIDEO_EXTS or tag_kind == "video") and "video" in fastpath_kinds:
            counters["video_discovered"] += 1
            mime = f"video/{name_ext}" if name_ext else "video/unknown"
            if write:
                batch_rows.append((inner_id, 0, mime, name_ext or "", "video", False))
            _log_patch("video", mime)
            continue
        if name_ext in _IMAGE_EXTS or tag_kind == "image":
            counters["image_discovered"] += 1
            mime = f"image/{name_ext}" if name_ext else "image/unknown"
            if write:
                batch_rows.append((inner_id, 0, mime, name_ext or "", "image", False))
            _log_patch("image", mime)
            if not no_fetch:
                image_jobs.append((quilt_id, ident, inner_id))
            continue

        # Catch-all
        guessed_kind = tag_kind or _kind_from_ext(name_ext) or "unknown"
        mime = f"{guessed_kind}/{name_ext}" if name_ext else f"{guessed_kind}/unknown"
        if write:
            batch_rows.append((inner_id, 0, mime, name_ext or "", guessed_kind, False))
        _log_patch(guessed_kind, mime)
        counters.setdefault("catalogued_by_ext", 0)
        counters["catalogued_by_ext"] += 1

        if no_fetch:
            counters["skipped"] += 1
            continue

        # Slow path: fetch + detect for unknown-extension patches
        inner = _fetch_patch(quilt_id, ident)
        if not inner:
            counters["skipped"] += 1
            continue
        mime, ext, kind = detect_file_type(inner)
        if kind == "image" and is_supported_image(ext):
            # detected as image after fetch — queue for embed and catalog
            counters["image_discovered"] += 1
            if write:
                _record_patch_in_catalog(
                    inner_id, size=len(inner), mime=mime, ext=ext,
                    kind="image", indexed=False,
                )
            image_jobs.append((quilt_id, ident, inner_id))
        elif kind == "audio" or (name_ext in _AUDIO_EXTS):
            counters["audio_discovered"] += 1
            if write:
                _record_patch_in_catalog(
                    inner_id, size=len(inner),
                    mime=mime if kind == "audio" else f"audio/{name_ext}",
                    ext=ext or name_ext, kind="audio", indexed=False,
                )
        else:
            if write:
                _record_patch_in_catalog(
                    inner_id, size=len(inner), mime=mime, ext=ext,
                    kind=kind, indexed=False,
                )
            counters["non_media_catalogued"] += 1

    # Flush the catalog batch in a single transaction (one commit per quilt instead of one per patch).
    if batch_rows and write:
        _batch_record_patches(batch_rows)

    # Parallel fetch + embed of queued image patches within this quilt
    if image_jobs and not skip_embed and write:
        max_workers = min(4, len(image_jobs))
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            def _do_image_embed(args):
                qid, ident, inner_id = args
                inner = _fetch_patch(qid, ident)
                if not inner:
                    return None
                mime, ext, kind = detect_file_type(inner)
                if kind != "image" or not is_supported_image(ext):
                    return None
                emb = generate_image_embedding(inner, blob_id=inner_id)
                if emb is None:
                    return None
                return (qid, ident, inner_id, inner, mime, ext, emb)

            for fut in as_completed([ex.submit(_do_image_embed, j) for j in image_jobs]):
                res = fut.result()
                if res is None:
                    continue
                qid, ident, inner_id, inner, mime, ext, emb = res
                _record_patch_in_catalog(
                    inner_id, size=len(inner), mime=mime, ext=ext,
                    kind="image", indexed=True,
                )
                extra = {
                    "is_quilt": True,
                    "parent_quilt_id": qid,
                    "quilt_identifier": ident,
                    "size": len(inner),
                    "mime_type": mime,
                    "extension": ext,
                    "kind": "image",
                    "is_nsfw": False,
                }
                with store._lock:
                    store.add(
                        embedding=emb, blob_id=inner_id,
                        mime_type=mime, size=len(inner),
                        extension=ext, extra_metadata=extra,
                    )
                counters["image_indexed"] += 1

    return counters


def _list_quilts(limit: Optional[int], resume: bool) -> List[str]:
    with sqlite3.connect(str(CATALOG_DB), timeout=30) as conn:
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
                "SELECT blob_id FROM blobs WHERE kind='quilt' AND is_active=1 ORDER BY blob_id"
            )
        rows = cur.fetchall()
    ids = [r[0] for r in rows]
    return ids[:limit] if limit else ids


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--workers",
        type=int,
        default=64,
        help="parallel quilts (default 64 — assumes ample bandwidth + cores)",
    )
    parser.add_argument(
        "--kinds-fastpath",
        default="audio,video",
        help="comma-separated kinds to fast-path by filename (default: audio,video)",
    )
    parser.add_argument(
        "--no-fetch",
        action="store_true",
        help="Skip patch byte fetch — only catalog patches with audio/video/image extensions in their filename. Super fast.",
    )
    parser.add_argument(
        "--skip-embed",
        action="store_true",
        help="Don't generate image embeddings (catalog images only, leave them un-embedded).",
    )
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    fastpath = {k.strip() for k in args.kinds_fastpath.split(",") if k.strip()}

    pool = get_pool()
    print(f"Aggregator pool ({len(pool.upstreams)} upstreams):")
    for u in pool.upstreams:
        print(f"  - {u.url}")
    print(f"Catalog DB: {CATALOG_DB}")
    print(f"Fast-path kinds: {fastpath}")
    print(f"No-fetch mode: {args.no_fetch}")
    print(f"Skip embed: {args.skip_embed}")

    if not args.skip_embed and not is_model_ready():
        print(
            "[extract_quilts] WARNING: embedding model not ready. "
            "Image patches will skip embedding for this run."
        )

    print("Loading vector store...")
    store = VectorStore()
    store.load()
    print(f"  loaded: {len(store.metadata):,} existing entries")

    quilt_ids = _list_quilts(args.limit, args.resume)
    print(f"Quilts to process: {len(quilt_ids):,}")
    if not quilt_ids:
        return 0

    totals = {
        "patches": 0,
        "audio_discovered": 0,
        "video_discovered": 0,
        "image_discovered": 0,
        "image_indexed": 0,
        "non_media_catalogued": 0,
        "fetch_failed": 0,
        "skipped": 0,
    }
    start = time.time()
    # Save the vector store every ~5% of the run, but print PROGRESS far more often
    # so long runs don't look dead. (Old code coupled the two — for 247K quilts the
    # progress line came every ~12K quilts which is hours of silence.)
    save_every = max(500, len(quilt_ids) // 20)
    print_every = 25
    done = 0

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(
                _process_quilt, qid, store,
                write=not args.dry_run,
                no_fetch=args.no_fetch,
                fastpath_kinds=fastpath,
                skip_embed=args.skip_embed,
            ): qid
            for qid in quilt_ids
        }
        for fut in as_completed(futs):
            qid = futs[fut]
            try:
                c = fut.result()
            except Exception as exc:
                print(f"  [{qid[:14]}…] worker error: {exc}", file=sys.stderr)
                c = {"fetch_failed": 1}
            for k, v in c.items():
                totals[k] = totals.get(k, 0) + v
            if not args.dry_run:
                if c.get("fetch_failed"):
                    _mark_quilt_status(qid, "quilt_failed")
                elif c.get("patches", 0) > 0:
                    _mark_quilt_status(qid, "quilt_expanded")
            done += 1

            # Per-quilt line: result of this single quilt + cumulative tallies.
            elapsed = time.time() - start
            rate = done / elapsed if elapsed else 0
            this_summary = (
                f"p={c.get('patches', 0)} "
                f"a={c.get('audio_discovered', 0)} "
                f"v={c.get('video_discovered', 0)} "
                f"i={c.get('image_discovered', 0)} "
                f"ix={c.get('image_indexed', 0)} "
                f"fail={c.get('fetch_failed', 0)}"
            )
            print(
                f"  [{done:>6}/{len(quilt_ids)}] ({rate:.1f}/s) {qid[:20]}…  "
                f"{this_summary}  | total audio={totals['audio_discovered']:,} "
                f"video={totals['video_discovered']:,} "
                f"img_disc={totals['image_discovered']:,} "
                f"img_ix={totals['image_indexed']:,} "
                f"fail={totals['fetch_failed']:,}",
                flush=True,
            )

            if done % save_every == 0 or done == len(quilt_ids):
                if not args.dry_run:
                    store.save()

    if not args.dry_run:
        store.save()
    elapsed = time.time() - start
    print()
    print(f"Done in {elapsed:.1f}s. ({done / elapsed:.2f} quilts/s)")
    print(f"  quilts processed:     {done:,}")
    print(f"  patches discovered:   {totals['patches']:,}")
    print(f"  audio discovered:     {totals['audio_discovered']:,}")
    print(f"  video discovered:     {totals['video_discovered']:,}")
    print(f"  image discovered:     {totals['image_discovered']:,}")
    print(f"  image patches indexed: {totals['image_indexed']:,}")
    print(f"  other catalogued:     {totals['non_media_catalogued']:,}")
    print(f"  patch fetches skipped: {totals['skipped']:,}")
    print(f"  quilts failed to fetch list: {totals['fetch_failed']:,}")
    print(f"  store size now:       {len(store.metadata):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
