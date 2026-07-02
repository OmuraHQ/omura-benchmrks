"""Background quilt expansion loop for the indexer.

Continuously walks ``kind='quilt'`` rows in ``blob_catalog.sqlite``, calls the Walrus
aggregator's documented patches endpoint, and records each inner patch in the catalog.
Audio/video patches are identified by filename extension (no per-patch fetch — fast).
Image patches are fetched + embedded into the shared vector store.

Runs as a daemon thread inside the indexer process so the shared in-memory vector store
isn't clobbered by an external script's save.

Env:
  OMURA_EXPAND_QUILTS              true|false (default true)
  OMURA_QUILT_EXPANSION_BATCH      quilts per pass (default 30)
  OMURA_QUILT_EXPANSION_INTERVAL   seconds between passes (default 60)
  OMURA_QUILT_EXPANSION_WORKERS    parallel quilt list calls per pass (default 4)
  OMURA_QUILT_EXPANSION_FETCH_IMAGES  true|false (default true) — fetch image patch bytes
"""

from __future__ import annotations

import os
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from omura.parsers.file_detection import detect_file_type
from omura.parsers.multimodal import is_supported_image
from omura.utils.aggregator_pool import get_pool
from omura.utils.imagebind_embeddings import generate_image_embedding, is_model_ready
from omura.utils.vector_store import VectorStore


ENABLED = os.getenv("OMURA_EXPAND_QUILTS", "true").lower() == "true"
BATCH = int(os.getenv("OMURA_QUILT_EXPANSION_BATCH", "30"))
INTERVAL_SECS = int(os.getenv("OMURA_QUILT_EXPANSION_INTERVAL", "60"))
WORKERS = int(os.getenv("OMURA_QUILT_EXPANSION_WORKERS", "4"))
PATCH_WORKERS = int(os.getenv("OMURA_QUILT_PATCH_WORKERS", "4"))  # parallelism INSIDE one quilt
FETCH_IMAGES = os.getenv("OMURA_QUILT_EXPANSION_FETCH_IMAGES", "true").lower() == "true"
LIST_TIMEOUT = float(os.getenv("OMURA_QUILT_LIST_TIMEOUT", "120"))
FETCH_TIMEOUT = float(os.getenv("OMURA_QUILT_FETCH_TIMEOUT", "180"))

_AUDIO_EXTS = {
    "mp3", "wav", "wave", "flac", "ogg", "oga", "opus",
    "m4a", "m4b", "aac", "aif", "aiff", "wma", "amr",
}
_VIDEO_EXTS = {
    "mp4", "m4v", "mov", "webm", "mkv", "avi", "flv", "wmv", "mpg", "mpeg",
}
_IMAGE_EXTS = {
    "png", "jpg", "jpeg", "webp", "gif", "bmp",
    "heic", "heif", "avif",
    # tif/tiff excluded: not browser-renderable, too large for embedding
}
_DOC_EXTS = {"pdf", "doc", "docx", "rtf", "odt", "epub", "md", "mobi"}
_TEXT_EXTS = {"txt", "json", "yaml", "yml", "xml", "html", "htm", "csv", "tsv", "log"}
_ARCHIVE_EXTS = {"zip", "tar", "gz", "tgz", "rar", "7z", "bz2", "xz", "zst"}
_CODE_EXTS = {"py", "js", "ts", "rs", "go", "java", "cpp", "c", "h", "rb", "php", "sh"}
_DATA_EXTS = {"parquet", "arrow", "feather", "orc", "avro", "msgpack", "pb", "proto"}


def _kind_from_ext(ext: str) -> Optional[str]:
    if not ext: return None
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
            pool_connections=4, pool_maxsize=16,
            max_retries=requests.adapters.Retry(
                total=1, backoff_factor=0.3,
                status_forcelist=[502, 503, 504],
                allowed_methods=["GET"],
            ),
        )
        s.mount("https://", adapter)
        s.mount("http://", adapter)
        _tls.s = s
    return s


def _ext(identifier: str) -> str:
    """Extension from a patch identifier's BASENAME only (handle path-like idents)."""
    basename = identifier.rsplit("/", 1)[-1]
    if "." not in basename:
        return ""
    ext = basename.rsplit(".", 1)[-1].lower().strip()
    if not ext or len(ext) > 8 or not ext.isalnum():
        return ""
    return ext


def _list_patches(quilt_id: str) -> List[Dict[str, Any]]:
    resp, _ = get_pool().get(
        f"/v1/quilts/{quilt_id}/patches", session=_session(), timeout=LIST_TIMEOUT
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
        if isinstance(item, dict):
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
        session=_session(), timeout=FETCH_TIMEOUT,
    )
    if resp is None or resp.status_code != 200:
        return None
    return resp.content


def _record_patch_in_catalog(
    db_path: Path,
    inner_id: str,
    size: int,
    mime: str,
    ext: str,
    kind: str,
    indexed: bool,
) -> None:
    try:
        with sqlite3.connect(str(db_path), timeout=10) as conn:
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
                    size=COALESCE(NULLIF(excluded.size,0), blobs.size),
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
    except Exception:
        pass


def _mark_quilt_status(db_path: Path, blob_id: str, status: str) -> None:
    try:
        with sqlite3.connect(str(db_path), timeout=10) as conn:
            conn.execute(
                "UPDATE blobs SET status=?, last_updated_at=datetime('now') WHERE blob_id=?",
                (status, blob_id),
            )
            conn.commit()
    except Exception:
        pass


def _list_pending_quilts(db_path: Path, limit: int) -> List[str]:
    with sqlite3.connect(str(db_path), timeout=30) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT blob_id FROM blobs
            WHERE kind='quilt' AND is_active=1
              AND (status IS NULL OR status NOT IN ('quilt_expanded','quilt_failed','expired'))
            ORDER BY blob_id
            LIMIT ?
            """,
            (limit,),
        )
        return [r[0] for r in cur.fetchall()]


def _kind_from_tags(tags: Dict[str, Any]) -> Optional[str]:
    """Walrus aggregator returns a ``tags`` dict per patch. Surface the standard
    ``content-type`` / ``content_type`` / ``mime`` fields and map back to our kinds."""
    if not isinstance(tags, dict):
        return None
    for key in ("content-type", "content_type", "Content-Type", "mime", "mimeType"):
        v = tags.get(key)
        if isinstance(v, str) and "/" in v:
            top = v.split("/", 1)[0].strip().lower()
            if top in ("image", "audio", "video", "application", "text"):
                return top
    return None


def _process_image_patch(
    quilt_id: str,
    ident: str,
    inner_id: str,
    catalog_db: Path,
    store: VectorStore,
) -> str:
    """Fetch + embed one image patch. Returns 'indexed' / 'cataloged' / 'failed'."""
    inner = _fetch_patch(quilt_id, ident)
    if not inner:
        return "failed"
    mime, det_ext, kind = detect_file_type(inner)
    if kind != "image" or not is_supported_image(det_ext):
        _record_patch_in_catalog(
            catalog_db, inner_id, len(inner), mime, det_ext, kind, False
        )
        return "cataloged"
    emb = None
    if is_model_ready():
        try:
            emb = generate_image_embedding(inner, blob_id=inner_id)
        except Exception:
            emb = None
    _record_patch_in_catalog(
        catalog_db, inner_id, len(inner), mime, det_ext, "image",
        indexed=emb is not None,
    )
    if emb is None:
        return "cataloged"
    extra = {
        "is_quilt": True,
        "parent_quilt_id": quilt_id,
        "quilt_identifier": ident,
        "size": len(inner),
        "mime_type": mime,
        "extension": det_ext,
        "kind": "image",
        "is_nsfw": False,
    }
    with store._lock:
        store.add(
            embedding=emb, blob_id=inner_id,
            mime_type=mime, size=len(inner),
            extension=det_ext, extra_metadata=extra,
        )
    return "indexed"


def _process_one_quilt(
    quilt_id: str,
    catalog_db: Path,
    store: VectorStore,
    fetch_images: bool,
) -> Dict[str, int]:
    """Expand one quilt into the catalog (+ vector store for images).

    Fast paths (no per-patch fetch):
      - filename extension in _AUDIO_EXTS/_VIDEO_EXTS/_IMAGE_EXTS → catalog by name
      - tags.content_type from the aggregator's patch listing → use that as kind

    Image patches go through a parallel sub-thread-pool for fetch + embed so a quilt
    with N images doesn't serialize the fetch — important for image-heavy quilts.
    """
    counters = {
        "patches": 0,
        "audio_discovered": 0,
        "video_discovered": 0,
        "image_discovered": 0,
        "image_indexed": 0,
        "other_catalogued": 0,
        "fetch_failed": 0,
    }
    patches = _list_patches(quilt_id)
    if not patches:
        counters["fetch_failed"] = 1
        return counters

    counters["patches"] = len(patches)
    image_jobs: List[tuple] = []  # patches that need fetch + embed
    seen_inner_ids: set = set()

    for item in patches:
        ident = item["identifier"]
        inner_id = f"{quilt_id}::{ident}"
        if inner_id in seen_inner_ids or inner_id in store.metadata:
            continue
        seen_inner_ids.add(inner_id)
        ext = _ext(ident)
        tag_kind = _kind_from_tags(item.get("tags") or {})

        # Fast path: audio by filename extension OR tag content-type
        if ext in _AUDIO_EXTS or tag_kind == "audio":
            counters["audio_discovered"] += 1
            _record_patch_in_catalog(
                catalog_db, inner_id, 0,
                f"audio/{ext}" if ext else "audio/unknown",
                ext or "", "audio", False,
            )
            continue
        # Fast path: video by filename extension OR tag content-type
        if ext in _VIDEO_EXTS or tag_kind == "video":
            counters["video_discovered"] += 1
            _record_patch_in_catalog(
                catalog_db, inner_id, 0,
                f"video/{ext}" if ext else "video/unknown",
                ext or "", "video", False,
            )
            continue
        # Image: catalog from filename immediately; also queue for fetch+embed.
        if ext in _IMAGE_EXTS or tag_kind == "image":
            counters["image_discovered"] += 1
            # Record an immediate kind=image catalog row so the dashboard reflects it
            # even before the (slower) fetch+embed completes.
            _record_patch_in_catalog(
                catalog_db, inner_id, 0,
                f"image/{ext}" if ext else "image/unknown",
                ext or "", "image", False,
            )
            if fetch_images:
                image_jobs.append((quilt_id, ident, inner_id))
            continue

        # Catch-all: every other patch gets a catalog row with its inferred kind
        # (from the filename extension) or "unknown" when there's no extension hint.
        guessed_kind = tag_kind or _kind_from_ext(ext) or "unknown"
        _record_patch_in_catalog(
            catalog_db, inner_id, 0,
            f"{guessed_kind}/{ext}" if ext else f"{guessed_kind}/unknown",
            ext or "", guessed_kind, False,
        )
        counters["other_catalogued"] += 1

    # Parallel image fetch + embed within this quilt
    if image_jobs and fetch_images:
        max_workers = min(PATCH_WORKERS, len(image_jobs))
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [
                ex.submit(_process_image_patch, q, i, iid, catalog_db, store)
                for (q, i, iid) in image_jobs
            ]
            for fut in as_completed(futs):
                try:
                    r = fut.result()
                except Exception:
                    r = "failed"
                if r == "indexed":
                    counters["image_indexed"] += 1
                elif r == "cataloged":
                    counters["other_catalogued"] += 1

    return counters


def _expansion_loop(catalog_db: Path, store: VectorStore) -> None:
    """Runs forever, processing batches of quilts."""
    print(
        f"[QuiltExpander] Starting loop: batch={BATCH} interval={INTERVAL_SECS}s "
        f"workers={WORKERS} fetch_images={FETCH_IMAGES}"
    )
    while True:
        try:
            quilt_ids = _list_pending_quilts(catalog_db, BATCH)
            if not quilt_ids:
                time.sleep(INTERVAL_SECS * 5)
                continue

            t0 = time.time()
            totals = {
                "patches": 0, "audio_discovered": 0, "video_discovered": 0,
                "image_discovered": 0, "image_indexed": 0,
                "other_catalogued": 0, "fetch_failed": 0,
            }
            done_in_batch = 0
            with ThreadPoolExecutor(max_workers=WORKERS) as ex:
                futs = {
                    ex.submit(_process_one_quilt, qid, catalog_db, store, FETCH_IMAGES): qid
                    for qid in quilt_ids
                }
                for fut in as_completed(futs):
                    qid = futs[fut]
                    try:
                        c = fut.result()
                    except Exception as e:
                        print(f"[QuiltExpander] {qid[:14]}… worker error: {e}", flush=True)
                        c = {"fetch_failed": 1}
                    for k, v in c.items():
                        totals[k] = totals.get(k, 0) + v
                    if c.get("fetch_failed"):
                        _mark_quilt_status(catalog_db, qid, "quilt_failed")
                    elif c.get("patches", 0) > 0:
                        _mark_quilt_status(catalog_db, qid, "quilt_expanded")
                    done_in_batch += 1
                    print(
                        f"[QuiltExpander] {qid[:20]}…  p={c.get('patches',0)} "
                        f"a={c.get('audio_discovered',0)} v={c.get('video_discovered',0)} "
                        f"i={c.get('image_discovered',0)} ix={c.get('image_indexed',0)} "
                        f"fail={c.get('fetch_failed',0)}",
                        flush=True,
                    )
            elapsed = time.time() - t0
            print(
                f"[QuiltExpander] BATCH DONE {done_in_batch}/{len(quilt_ids)} in {elapsed:.1f}s "
                f"audio={totals['audio_discovered']} video={totals['video_discovered']} "
                f"img_disc={totals['image_discovered']} img_ix={totals['image_indexed']} "
                f"other={totals['other_catalogued']} fail={totals['fetch_failed']}",
                flush=True,
            )

            # Save store after each batch (image_indexed may have added rows).
            if totals["image_indexed"] > 0:
                try:
                    store.save()
                except Exception as e:
                    print(f"[QuiltExpander] store save error: {e}", flush=True)

            time.sleep(INTERVAL_SECS)
        except Exception as e:
            print(f"[QuiltExpander] loop error: {e}", flush=True)
            time.sleep(INTERVAL_SECS * 2)


def start_quilt_expander_thread(
    catalog_db: Path, store: VectorStore
) -> Optional[threading.Thread]:
    """Spawn the expansion loop in a daemon thread. Returns the Thread (or None if disabled)."""
    if not ENABLED:
        print("[QuiltExpander] disabled via OMURA_EXPAND_QUILTS=false")
        return None
    t = threading.Thread(
        target=_expansion_loop, args=(catalog_db, store),
        name="QuiltExpander", daemon=True,
    )
    t.start()
    return t
