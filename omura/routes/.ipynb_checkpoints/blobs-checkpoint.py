"""Blob proxy endpoint for universal file access."""

from __future__ import annotations

import os

import requests
from fastapi import APIRouter, HTTPException, Path, status
from fastapi.responses import Response

from omura.parsers.file_detection import detect_file_type

router = APIRouter(
    prefix="/blob",
    tags=["blobs"],
    responses={
        404: {"description": "Blob not found"},
    },
)

# Default aggregator URL
DEFAULT_AGGREGATOR = os.getenv(
    "WALRUS_AGGREGATOR_URL", "https://walrus-mainnet-aggregator.redundex.com"
).rstrip("/")


@router.get(
    "/{blob_id}",
    summary="Proxy blob file content (universal file type support)",
    description="""
    Universal proxy endpoint for Walrus blob content with automatic MIME type detection.
    
    This endpoint works for **all file types** (images, PDFs, videos, archives, text, etc.):
    
    **For images in HTML:**
    ```html
    <img src="/blob/DFBxFKxkkJ23oNMgn_baUcs5HVce_FEGrqJqlruEpRo" />
    ```
    
    **For PDFs:**
    ```html
    <iframe src="/blob/ABC123..." width="100%" height="600px"></iframe>
    ```
    
    **For direct file access:**
    ```html
    <a href="/blob/ABC123..." download>Download File</a>
    ```
    
    **How it works:**
    1. Fetches the blob content from the Walrus aggregator
    2. Detects the file type using magic bytes (supports: PNG, JPEG, GIF, WebP, BMP, TIFF, PDF, MP4, ZIP, text, and more)
    3. Returns the file with the correct Content-Type header
    
    The endpoint automatically detects file types using magic bytes and sets appropriate MIME types
    for proper browser handling. Works as a transparent proxy for any blob content.
    """,
    responses={
        200: {
            "description": "Blob content retrieved successfully with correct MIME type",
            "content": {
                "image/png": {},
                "image/jpeg": {},
                "image/gif": {},
                "image/webp": {},
                "image/bmp": {},
                "image/tiff": {},
                "application/pdf": {},
                "video/mp4": {},
                "application/zip": {},
                "text/plain": {},
                "application/octet-stream": {},
            },
        },
        404: {"description": "Blob not found"},
        500: {"description": "Error fetching blob from aggregator"},
    },
)
async def proxy_blob(
    blob_id: str = Path(
        ...,
        description="Base64-encoded Walrus blob ID",
        example="DFBxFKxkkJ23oNMgn_baUcs5HVce_FEGrqJqlruEpRo",
    )
) -> Response:
    """
    Proxy blob content from Walrus aggregator with universal file type detection.
    
    This endpoint automatically detects the file type using magic bytes and returns
    the content with the appropriate Content-Type header. Works for all file types:
    images (PNG, JPEG, GIF, WebP, BMP, TIFF), PDFs, videos (MP4), archives (ZIP),
    text files, and any other binary content.
    
    Perfect for:
    - Direct image embedding in HTML (`<img src>`)
    - PDF viewing in iframes
    - File downloads
    - Any other blob content access
    """
    # Fetch blob from aggregator
    url = f"{DEFAULT_AGGREGATOR}/v1/blobs/{blob_id}"
    try:
        response = requests.get(url, timeout=60, stream=True)
        if response.status_code == 404:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Blob {blob_id} not found in aggregator",
            )
        if response.status_code != 200:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Aggregator returned status {response.status_code}",
            )
        
        # Read content
        content = response.content
        
        # Detect file type (universal detection for all file types)
        mime_type, extension, kind = detect_file_type(content)
        
        # Build headers with appropriate Content-Disposition based on file type
        headers = {
            "X-Blob-ID": blob_id,
            "X-File-Kind": kind,
            "Cache-Control": "public, max-age=3600",
        }
        
        # For images, PDFs, and text, use inline; for others, allow browser to decide
        if kind in ("image", "pdf") or mime_type.startswith("text/"):
            headers["Content-Disposition"] = f'inline; filename="{blob_id[:20]}.{extension}"'
        else:
            # For downloads or other types, let browser handle it
            headers["Content-Disposition"] = f'attachment; filename="{blob_id[:20]}.{extension}"'
        
        return Response(
            content=content,
            media_type=mime_type,
            headers=headers,
        )
    except HTTPException:
        raise
    except requests.RequestException as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching blob from aggregator: {e}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {e}",
        )
