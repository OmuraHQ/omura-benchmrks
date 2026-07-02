"""Admin endpoints for maintenance tasks that need to run in-process.

These routes mutate shared in-memory state (vector store, caches) and must execute
inside the running API server — running them out-of-process via a separate script
races with the indexer's periodic save and gets clobbered.

Auth: gated by ``OMURA_ADMIN_TOKEN`` (header ``X-Admin-Token``). If unset, the
endpoint refuses all requests rather than running unauthenticated.
"""

from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

import requests
from fastapi import APIRouter, Header, HTTPException, status
from pydantic import BaseModel, Field

from omura.parsers.file_detection import detect_file_type
from omura.parsers.multimodal import is_supported_audio, is_supported_image
from omura.utils.aggregator_pool import get_pool
from omura.utils.blob_catalog import CATALOG_DB_PATH
from omura.utils.imagebind_embeddings import (
    generate_audio_embedding,
    generate_image_embedding,
    is_model_ready,
)
from omura.utils.vector_store import VectorStore
from omura.routes import search as search_module

import sqlite3

router = APIRouter(prefix="/admin", tags=["admin"])

AGGREGATOR_URL = os.getenv(
    "WALRUS_AGGREGATOR_URL", "https://agrregator.omura.fun"
).rstrip("/")
SWEEP_TIMEOUT = float(os.getenv("OMURA_SWEEP_TIMEOUT", "8"))
SWEEP_404_MIN = int(os.getenv("OMURA_SWEEP_404_MIN", "2"))

_tls = threading.local()


def _session() -> requests.Session:
    s = getattr(_tls, "s", None)
    if s is None:
        s = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=4,
            pool_maxsize=32,
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


def _probe(blob_id: str) -> int:
    """HEAD probe via the aggregator pool (round-robin across upstreams)."""
    pool = get_pool()
    resp, _ = pool.head(
        f"/v1/blobs/{blob_id}",
        session=_session(),
        timeout=SWEEP_TIMEOUT,
        allow_redirects=True,
    )
    if resp is None:
        return -1
    if resp.status_code in (405, 501):
        resp, _ = pool.get(
            f"/v1/blobs/{blob_id}",
            session=_session(),
            headers={"Range": "bytes=0-0"},
            timeout=SWEEP_TIMEOUT,
            stream=True,
        )
        if resp is not None:
            resp.close()
        if resp is None:
            return -1
    return resp.status_code


class SweepRequest(BaseModel):
    workers: int = Field(default=16, ge=1, le=64)
    passes: int = Field(default=SWEEP_404_MIN, ge=1, le=5)
    dry_run: bool = False


class SweepResponse(BaseModel):
    total_probed: int
    confirmed_404: int
    removed: int
    remaining: int
    duration_seconds: float
    passes_run: int


def _check_admin(token: Optional[str]) -> None:
    expected = os.getenv("OMURA_ADMIN_TOKEN", "").strip()
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="admin endpoints disabled (OMURA_ADMIN_TOKEN not set)",
        )
    if not token or token != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid admin token",
        )


@router.post(
    "/sweep-broken-blobs",
    response_model=SweepResponse,
    summary="Remove blobs that 404 from the aggregator (in-process; safe vs indexer)",
)
def sweep_broken_blobs(
    req: SweepRequest,
    x_admin_token: Optional[str] = Header(default=None),
) -> SweepResponse:
    _check_admin(x_admin_token)

    store = search_module._shared_vector_store
    if store is None:
        store = VectorStore()
        store.load()

    start = time.time()
    candidates = set(store.metadata.keys())
    total_probed = len(candidates)

    passes_run = 0
    for pass_n in range(1, req.passes + 1):
        if not candidates:
            break
        passes_run = pass_n
        confirmed: List[str] = []
        with ThreadPoolExecutor(max_workers=req.workers) as ex:
            futs = {ex.submit(_probe, b): b for b in candidates}
            for fut in as_completed(futs):
                bid = futs[fut]
                code = fut.result()
                if code == 404:
                    confirmed.append(bid)
        candidates = set(confirmed)

    final_404 = candidates
    removed = 0
    if not req.dry_run and final_404:
        # Persist the blacklist BEFORE modifying the in-memory store so that the
        # indexer (which may be running in another worker process) also respects
        # these deletions on its next periodic save.
        blacklist_path = store.index_path.parent / "deleted_blobs.json"
        try:
            existing: set = set()
            if blacklist_path.exists():
                import json as _json
                with open(blacklist_path) as _f:
                    existing = set(_json.load(_f))
            existing.update(final_404)
            import json as _json
            with open(blacklist_path, "w") as _f:
                _json.dump(sorted(existing), _f)
        except Exception as _bl_err:
            print(f"[Admin] Warning: could not write sweep blacklist: {_bl_err}")

        # Mark the swept blobs deleted in the catalog so the FTS/text-search path
        # (which filters on is_active = 1) also stops surfacing them, and drop their
        # full-text rows. Without this, vector search is cleaned but text queries can
        # still return the now-gone blobs.
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
        except Exception as _db_err:
            print(f"[Admin] Warning: could not mark swept blobs in catalog: {_db_err}")

        with store._lock:
            removed = store.remove_blob_ids(list(final_404))
        store.save()

    duration = time.time() - start
    return SweepResponse(
        total_probed=total_probed,
        confirmed_404=len(final_404),
        removed=removed,
        remaining=len(store.metadata),
        duration_seconds=duration,
        passes_run=passes_run,
    )


# ── Quilt re-parse ─────────────────────────────────────────────────────────────


class ReparseQuiltsRequest(BaseModel):
    limit: Optional[int] = Field(default=None, ge=1)
    workers: int = Field(default=8, ge=1, le=32)
    resume: bool = Field(
        default=True,
        description="Skip quilts already marked status='quilt_expanded'",
    )
    dry_run: bool = False


class ReparseQuiltsResponse(BaseModel):
    quilts_processed: int
    patches_discovered: int
    image_patches_indexed: int
    audio_patches_discovered: int
    audio_patches_indexed: int
    skipped: int
    fetch_failed: int
    store_size: int
    duration_seconds: float


def _list_quilt_patches_api(blob_id: str, timeout: float = 300.0) -> List[Dict[str, str]]:
    """Pool-based: GET /v1/quilts/{quilt_id}/patches -> [{identifier, patch_id, tags?}].
    Tries each upstream in turn — public aggregators are typically faster than the local one."""
    resp, _ = get_pool().get(
        f"/v1/quilts/{blob_id}/patches", session=_session(), timeout=timeout
    )
    if resp is None or resp.status_code != 200:
        return []
    try:
        data = resp.json()
    except ValueError:
        return []
    if not isinstance(data, list):
        return []
    out: List[Dict[str, str]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        ident = item.get("identifier")
        patch_id = item.get("patch_id")
        if isinstance(ident, str) and isinstance(patch_id, (str, dict)):
            tags_field = item.get("tags") if isinstance(item.get("tags"), dict) else {}
            out.append(
                {"identifier": ident, "patch_id": str(patch_id), "tags": tags_field}  # type: ignore
            )
    return out


def _fetch_patch_by_identifier(
    blob_id: str, identifier: str, timeout: float = 300.0
) -> Optional[bytes]:
    """Pool-based: GET /v1/blobs/by-quilt-id/{quilt_id}/{identifier}."""
    safe = requests.utils.quote(identifier, safe="")
    resp, _ = get_pool().get(
        f"/v1/blobs/by-quilt-id/{blob_id}/{safe}", session=_session(), timeout=timeout
    )
    if resp is None or resp.status_code != 200:
        return None
    return resp.content


def _list_quilt_ids_in_db(limit: Optional[int], resume: bool) -> List[str]:
    with sqlite3.connect(str(CATALOG_DB_PATH), timeout=30) as conn:
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


def _mark_quilt_status(blob_id: str, status: str) -> None:
    try:
        with sqlite3.connect(str(CATALOG_DB_PATH), timeout=10) as conn:
            conn.execute(
                "UPDATE blobs SET status=?, last_updated_at=datetime('now') WHERE blob_id=?",
                (status, blob_id),
            )
            conn.commit()
    except Exception:
        pass


def _record_inner_patch_in_catalog(
    inner_id: str,
    parent_quilt_id: str,
    identifier: str,
    size: int,
    mime: str,
    ext: str,
    kind: str,
    indexed: bool,
) -> None:
    """Insert a row into blob_catalog.sqlite for each discovered patch.

    Lets the dashboard's audio/video/image counters reflect the inner patches even when
    they aren't (yet) embedded in the vector store.
    """
    try:
        with sqlite3.connect(str(CATALOG_DB_PATH), timeout=10) as conn:
            conn.execute(
                """
                INSERT INTO blobs (
                    blob_id, kind, mime_type, extension, size,
                    is_active, fetch_ok, status, indexed,
                    first_seen_at, last_updated_at
                ) VALUES (?, ?, ?, ?, ?, 1, 1, ?, ?, datetime('now'), datetime('now'))
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
                    inner_id,
                    kind,
                    mime,
                    ext,
                    size,
                    "indexed" if indexed else "discovered",
                    1 if indexed else 0,
                ),
            )
            conn.commit()
    except Exception:
        pass


_AUDIO_EXTS = {
    "mp3", "wav", "wave", "flac", "ogg", "oga", "opus",
    "m4a", "m4b", "aac", "aif", "aiff", "wma", "amr",
}
_IMAGE_EXTS = {
    "png", "jpg", "jpeg", "webp", "gif", "bmp",
    "heic", "heif", "avif", "ico", "svg",
    # tif/tiff excluded: not browser-renderable, too large for embedding
}
_VIDEO_EXTS = {
    "mp4", "m4v", "mov", "webm", "mkv", "avi", "flv", "wmv", "mpg", "mpeg",
}
_AUDIO_MIME_HINTS = ("audio/",)


def _ext_from_identifier(identifier: str) -> str:
    """Lower-cased extension from a patch identifier, or '' if absent."""
    if "." not in identifier:
        return ""
    return identifier.rsplit(".", 1)[-1].lower().strip()


def _maybe_reclassify_audio(
    identifier: str, mime: str, ext: str, kind: str
) -> Tuple[str, str, str]:
    """If magic-byte detection labelled the patch ``application``/``binary``, salvage it
    via the filename extension. Many audio formats (M4A, AAC, OPUS, WebM-audio) have weak
    or container-ambiguous magic bytes and end up generic.

    Returns possibly-updated ``(mime, ext, kind)``. Image / video kinds are left alone.
    """
    if kind in ("audio", "image", "video"):
        return mime, ext, kind
    # Pull extension from identifier if present (case-insensitive).
    name_ext = ""
    if "." in identifier:
        name_ext = identifier.rsplit(".", 1)[-1].lower().strip()
    if name_ext in _AUDIO_EXTS:
        # Reclassify as audio; preserve mime if it already looks audio-y.
        new_mime = mime if any(mime.startswith(h) for h in _AUDIO_MIME_HINTS) else f"audio/{name_ext}"
        return new_mime, name_ext, "audio"
    return mime, ext, kind


def _expand_one_quilt(
    blob_id: str,
    store: VectorStore,
    write: bool,
) -> Dict[str, int]:
    """Use the documented aggregator endpoints to enumerate + fetch patches.

    1. ``/v1/quilts/{quilt_id}/patches`` → list of {identifier, patch_id, tags?}
    2. ``/v1/blobs/by-quilt-id/{quilt_id}/{identifier}`` → patch bytes

    Discovers ALL patches and records them in the catalog as ``{quilt_id}::{ident}``.
    Embeds images directly. Embeds audio when the active model supports it (the current
    Omura-Embed model is image-only and returns None for audio — those patches are still
    catalogued as ``discovered`` for visibility, just not added to the vector store).
    A filename-extension rescue catches audio files whose magic bytes are ambiguous
    (M4A, AAC, OPUS, etc.) and would otherwise land in ``application``.
    """
    s = {
        "patches": 0,
        "indexed": 0,
        "audio_discovered": 0,
        "audio_indexed": 0,
        "skipped": 0,
        "fetch_failed": 0,
    }
    patch_items = _list_quilt_patches_api(blob_id)
    if not patch_items:
        s["fetch_failed"] = 1
        return s

    s["patches"] = len(patch_items)
    for item in patch_items:
        ident = item["identifier"]
        tags = item.get("tags") or {}
        inner_id = f"{blob_id}::{ident}"
        if inner_id in store.metadata:
            s["skipped"] += 1
            continue

        # Fast-path: if the patch identifier already declares an audio extension,
        # skip the expensive fetch and catalog it directly. Audio similarity search
        # would need a multimodal model anyway, so we only need to surface it.
        name_ext = _ext_from_identifier(ident)
        if name_ext in _AUDIO_EXTS:
            s["audio_discovered"] += 1
            if write:
                _record_inner_patch_in_catalog(
                    inner_id,
                    blob_id,
                    ident,
                    size=0,  # unknown without fetch
                    mime=f"audio/{name_ext}",
                    ext=name_ext,
                    kind="audio",
                    indexed=False,
                )
            s["skipped"] += 1  # not embedded, counted as discovery only
            continue

        # Fast-path for clearly-video patches: catalog without fetch.
        if name_ext in _VIDEO_EXTS:
            if write:
                _record_inner_patch_in_catalog(
                    inner_id, blob_id, ident,
                    size=0, mime=f"video/{name_ext}", ext=name_ext, kind="video",
                    indexed=False,
                )
            s["skipped"] += 1
            continue

        inner = _fetch_patch_by_identifier(blob_id, ident)
        if not inner:
            s["skipped"] += 1
            continue
        mime, ext, kind = detect_file_type(inner)
        # Rescue: filename-extension audio detection when magic bytes were ambiguous.
        mime, ext, kind = _maybe_reclassify_audio(ident, mime, ext, kind)

        is_image = kind == "image" and is_supported_image(ext)
        # For the audio rescue path, ``is_supported_audio`` may reject if the
        # parsers/multimodal extension list is narrow; accept any audio kind.
        is_audio = kind == "audio" and (is_supported_audio(ext) or ext in _AUDIO_EXTS)

        if not (is_image or is_audio):
            # Still catalog non-image/non-audio so dashboard counts reflect reality.
            if write:
                _record_inner_patch_in_catalog(
                    inner_id, blob_id, ident, len(inner), mime, ext, kind, indexed=False
                )
            s["skipped"] += 1
            continue

        if is_audio:
            s["audio_discovered"] += 1

        emb = None
        if is_image:
            emb = generate_image_embedding(inner, blob_id=inner_id)
        else:  # audio
            emb = generate_audio_embedding(inner, ext, inner_id)

        # Catalog every discovered media patch regardless of embedding success.
        if write:
            _record_inner_patch_in_catalog(
                inner_id, blob_id, ident, len(inner), mime, ext, kind,
                indexed=emb is not None,
            )

        if emb is None:
            # No embedding (e.g. audio with image-only model) — discovered but not embedded.
            if is_audio:
                pass  # already counted in audio_discovered
            s["skipped"] += 1
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
                **{f"tag_{k}": v for k, v in (tags or {}).items() if isinstance(v, str)},
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
        if is_image:
            s["indexed"] += 1
        else:
            s["audio_indexed"] += 1
    return s


@router.get(
    "/aggregator-pool",
    summary="Aggregator pool health snapshot (ranked by speed)",
)
def aggregator_pool_health(
    x_admin_token: Optional[str] = Header(default=None),
):
    """Return per-upstream stats sorted by score (fastest first)."""
    _check_admin(x_admin_token)
    return {"upstreams": get_pool().health_snapshot()}


@router.post(
    "/reparse-quilts",
    response_model=ReparseQuiltsResponse,
    summary="Expand quilts and index inner image patches (in-process, safe vs indexer)",
)
def reparse_quilts(
    req: ReparseQuiltsRequest,
    x_admin_token: Optional[str] = Header(default=None),
) -> ReparseQuiltsResponse:
    _check_admin(x_admin_token)

    # Discovery does NOT require the embedding model. We still catalog every patch
    # (so dashboard counters reflect inner audio/video/image), and skip the embedding
    # step gracefully when the model isn't ready. A warning is logged instead.
    if not is_model_ready():
        print(
            "[admin] reparse-quilts: embedding model not ready — patches will be "
            "catalogued without embeddings",
            flush=True,
        )

    store = search_module._shared_vector_store
    if store is None:
        store = VectorStore()
        store.load()

    quilt_ids = _list_quilt_ids_in_db(req.limit, req.resume)
    totals = {
        "patches": 0,
        "indexed": 0,
        "audio_discovered": 0,
        "audio_indexed": 0,
        "skipped": 0,
        "fetch_failed": 0,
    }
    start = time.time()

    if not quilt_ids:
        return ReparseQuiltsResponse(
            quilts_processed=0,
            patches_discovered=0,
            image_patches_indexed=0,
            audio_patches_discovered=0,
            audio_patches_indexed=0,
            skipped=0,
            fetch_failed=0,
            store_size=len(store.metadata),
            duration_seconds=0.0,
        )

    with ThreadPoolExecutor(max_workers=req.workers) as ex:
        futs = {ex.submit(_expand_one_quilt, qid, store, not req.dry_run): qid for qid in quilt_ids}
        for fut in as_completed(futs):
            qid = futs[fut]
            try:
                s = fut.result()
            except Exception:
                s = {
                    "patches": 0,
                    "indexed": 0,
                    "audio_discovered": 0,
                    "audio_indexed": 0,
                    "skipped": 0,
                    "fetch_failed": 1,
                }
            for k, v in s.items():
                totals[k] += v
            if not req.dry_run:
                if s["fetch_failed"]:
                    _mark_quilt_status(qid, "quilt_failed")
                elif s["patches"] > 0:
                    _mark_quilt_status(qid, "quilt_expanded")

    if not req.dry_run:
        store.save()

    return ReparseQuiltsResponse(
        quilts_processed=len(quilt_ids),
        patches_discovered=totals["patches"],
        image_patches_indexed=totals["indexed"],
        audio_patches_discovered=totals["audio_discovered"],
        audio_patches_indexed=totals["audio_indexed"],
        skipped=totals["skipped"],
        fetch_failed=totals["fetch_failed"],
        store_size=len(store.metadata),
        duration_seconds=time.time() - start,
    )


class PurgeBlobRequest(BaseModel):
    blob_id: str = Field(..., description="Blob ID to mark expired and remove from index")


class PurgeBlobResponse(BaseModel):
    blob_id: str
    removed_from_faiss: bool
    marked_expired_in_db: bool
    store_size: int


@router.post(
    "/purge-blob",
    response_model=PurgeBlobResponse,
    summary="Immediately expire a broken blob — removes it from FAISS and marks it expired in DB",
)
def purge_blob(
    req: PurgeBlobRequest,
    x_admin_token: Optional[str] = Header(default=None),
) -> PurgeBlobResponse:
    _check_admin(x_admin_token)

    blob_id = req.blob_id.strip()
    if not blob_id:
        raise HTTPException(status_code=400, detail="blob_id is required")

    store = search_module._shared_vector_store
    if store is None:
        store = VectorStore()
        store.load()

    # Mark expired in DB and remove from FTS
    marked_db = False
    try:
        with sqlite3.connect(str(CATALOG_DB_PATH), timeout=15) as conn:
            conn.execute(
                "UPDATE blobs SET is_active = 0, status = 'expired' WHERE blob_id = ?",
                (blob_id,),
            )
            conn.execute("DELETE FROM blobs_fts WHERE blob_id = ?", (blob_id,))
            conn.commit()
        marked_db = True
    except Exception as e:
        print(f"[admin/purge-blob] DB update failed for {blob_id}: {e}")

    # Remove from live FAISS index
    removed_faiss = False
    try:
        with store._lock:
            n = store.remove_blob_ids([blob_id])
            if n > 0:
                store.save(create_backup=False)
                removed_faiss = True
                print(f"[admin/purge-blob] Removed {blob_id} from FAISS and saved.")
            else:
                print(f"[admin/purge-blob] {blob_id} not found in FAISS (already gone).")
    except Exception as e:
        print(f"[admin/purge-blob] FAISS removal failed for {blob_id}: {e}")

    return PurgeBlobResponse(
        blob_id=blob_id,
        removed_from_faiss=removed_faiss,
        marked_expired_in_db=marked_db,
        store_size=len(store.metadata),
    )
