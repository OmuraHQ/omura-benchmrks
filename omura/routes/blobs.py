"""Blob proxy endpoint for universal file access."""

from __future__ import annotations

import io
import os

import requests
from fastapi import APIRouter, HTTPException, Path, Request, status
from fastapi.responses import Response, StreamingResponse

from omura.parsers.file_detection import detect_file_type
from omura.utils.aggregator_pool import get_pool

router = APIRouter(
    prefix="/blob",
    tags=["blobs"],
    responses={
        404: {"description": "Blob not found"},
    },
)

# Default aggregator URL (kept for back-compat; pool takes precedence)
DEFAULT_AGGREGATOR = os.getenv(
    "WALRUS_AGGREGATOR_URL", "https://agrregator.omura.fun"
).rstrip("/")

# Blobs larger than this are streamed without full buffering
_STREAM_THRESHOLD_BYTES = 8 * 1024 * 1024   # 8 MB
# Bytes read for magic-byte MIME detection
_SNIFF_BYTES = 512


def _catalog_mime(blob_id: str):
    """Authoritative (mime_type, extension, kind) for a blob from the catalog, or None.
    Used so the proxy serves the correct Content-Type even on Range responses (where
    body-sniffing only sees the requested byte range)."""
    import sqlite3
    from omura.utils.blob_catalog import CATALOG_DB_PATH
    try:
        with sqlite3.connect(str(CATALOG_DB_PATH), timeout=5) as conn:
            row = conn.execute(
                "SELECT mime_type, extension, kind FROM blobs WHERE blob_id = ?", (blob_id,)
            ).fetchone()
    except Exception:
        return None
    if not row:
        return None
    mime, ext, kind = row
    if not mime or not str(mime).strip():
        return None
    return str(mime), (str(ext) if ext else ""), (str(kind) if kind else "")


def _mark_and_prune_broken_blob(blob_id: str) -> None:
    from omura.routes.search import get_vector_store
    import sqlite3
    from omura.utils.blob_catalog import CATALOG_DB_PATH
    try:
        store = get_vector_store()
        with sqlite3.connect(str(CATALOG_DB_PATH), timeout=15) as conn:
            conn.execute("UPDATE blobs SET is_active = 0, status = 'expired' WHERE blob_id = ?", (blob_id,))
            conn.commit()

        with store._lock:
            removed = store.remove_blob_ids([blob_id])
            if removed > 0:
                store.save(create_backup=False)
                print(f"[BlobProxy] Marked broken blob {blob_id} as expired and rebuilt/saved FAISS.")
            else:
                print(f"[BlobProxy] Marked broken blob {blob_id} as expired in DB (was not in FAISS).")
    except Exception as err:
        print(f"[BlobProxy] Failed to prune broken blob {blob_id}: {err}")


@router.get(
    "/{blob_id}",
    summary="Proxy blob file content (universal file type support)",
    description="""
    Universal proxy endpoint for Walrus blob content with automatic MIME type detection.

    Supports all file types (images, PDFs, videos, archives, text, etc.).
    Large blobs (>8 MB) are streamed without buffering to avoid timeouts.
    MIME type is detected from the first 512 magic bytes.
    """,
)
def proxy_blob(
    request: Request,
    blob_id: str = Path(
        ...,
        description="Walrus blob ID. For quilt patches use '<quiltId>::<identifier>'.",
        examples=["DFBxFKxkkJ23oNMgn_baUcs5HVce_FEGrqJqlruEpRo"],
    ),
) -> Response:
    """Proxy blob content from Walrus with streaming + HTTP Range support.

    Handles quilt-patch ids ('<quiltId>::<identifier>') via the by-quilt-id route, and
    forwards a client `Range` header so video/audio players can seek (the aggregator
    answers 206 for plain blobs; quilt patches are small and buffered whole client-side).
    """
    # -- Resolve the upstream path: plain blob vs. quilt patch ------------------
    if "::" in blob_id:
        quilt_id, identifier = blob_id.split("::", 1)
        safe_ident = requests.utils.quote(identifier, safe="")
        upstream_path = f"/v1/blobs/by-quilt-id/{quilt_id}/{safe_ident}"
    else:
        upstream_path = f"/v1/blobs/{blob_id}"

    # -- Forward a Range header if the client sent one (seek support) -----------
    fwd_headers = {}
    range_header = request.headers.get("range") if request is not None else None
    if range_header:
        fwd_headers["Range"] = range_header

    try:
        response, used_url = get_pool().get(
            upstream_path, timeout=60, stream=True, headers=fwd_headers or None
        )
        if response is None:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="all aggregator upstreams failed",
            )
        if response.status_code == 404:
            _mark_and_prune_broken_blob(blob_id)
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Blob {blob_id} not found in aggregator",
            )
        if response.status_code not in (200, 206):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Aggregator returned status {response.status_code}",
            )

        is_partial = response.status_code == 206

        # -- Resolve MIME ------------------------------------------------------
        # Prefer the catalog's authoritative mime/extension (keyed by blob_id). Body-sniffing
        # is unreliable on Range/206 responses (only the requested bytes are present, so a tiny
        # range yields text/plain and breaks <audio>/<video> playback). Fall back to sniffing
        # the first bytes only when the blob isn't catalogued.
        header_chunk = response.raw.read(_SNIFF_BYTES)
        cat = _catalog_mime(blob_id)
        if cat is not None:
            mime_type, extension, kind = cat
        else:
            mime_type, extension, kind = detect_file_type(header_chunk)

        content_length = response.headers.get("content-length")
        blob_size = int(content_length) if content_length and content_length.isdigit() else None

        headers = {
            "X-Blob-ID": blob_id,
            "X-File-Kind": kind,
            "Cache-Control": "public, max-age=3600",
            # Advertise range support so browsers attempt seek on video/audio.
            "Accept-Ranges": "bytes",
        }
        # Inline-render media the browser can play/show; download the rest.
        if kind in ("image", "pdf", "video", "audio") or mime_type.startswith(("text/", "video/", "audio/")):
            headers["Content-Disposition"] = f'inline; filename="{blob_id[:20]}.{extension}"'
        else:
            headers["Content-Disposition"] = f'attachment; filename="{blob_id[:20]}.{extension}"'

        if content_length:
            headers["Content-Length"] = content_length
        if is_partial and response.headers.get("content-range"):
            headers["Content-Range"] = response.headers["content-range"]

        out_status = 206 if is_partial else 200
        is_large = blob_size is not None and blob_size > _STREAM_THRESHOLD_BYTES

        if is_large:
            # -- Stream large blobs without buffering --------------------------
            chunk_size = 256 * 1024  # 256 KB chunks

            def _stream():
                yield header_chunk
                for chunk in response.raw.stream(chunk_size, decode_content=False):
                    yield chunk

            return StreamingResponse(
                _stream(),
                media_type=mime_type,
                headers=headers,
                status_code=out_status,
            )
        else:
            # -- Small blobs: buffer fully -------------------------------------
            rest = response.raw.read(decode_content=False)
            content = header_chunk + rest
            return Response(
                content=content,
                media_type=mime_type,
                headers=headers,
                status_code=out_status,
            )

    except HTTPException:
        raise
    except requests.RequestException as e:
        # Only prune on genuine 404-style failures, not large-blob read aborts
        if "404" in str(e) or "not found" in str(e).lower():
            _mark_and_prune_broken_blob(blob_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching blob from aggregator: {e}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {e}",
        )
