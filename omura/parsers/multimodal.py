"""Parsers for image, video, and audio content used by Omni-Embed indexing."""

from __future__ import annotations

import io
import os
from pathlib import Path

from PIL import Image

TEMP_DIR = Path(os.getenv("OMURA_TEMP_BLOB_DIR", "data/temp_blobs"))
TEMP_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED_IMAGE = frozenset({"png", "jpg", "jpeg", "gif", "webp", "bmp"})  # tiff/tif excluded: not browser-renderable
SUPPORTED_VIDEO = frozenset({"mp4", "webm", "mov", "avi", "mkv", "ts", "m4v", "3gp", "flv", "ogv", "mpg", "mpeg"})
SUPPORTED_AUDIO = frozenset({"mp3", "wav", "ogg", "flac", "m4a", "aac", "opus"})
SUPPORTED_DOC = frozenset({"pdf", "txt", "md", "markdown"})

def parse_image(data):
    if not data:
        raise ValueError("Empty image data")
    return Image.open(io.BytesIO(data)).convert("RGB")

def _temp_path(blob_id, ext, kind):
    sub = TEMP_DIR / kind
    sub.mkdir(parents=True, exist_ok=True)
    safe = blob_id.replace("/", "_").replace("\\", "_")[:80]
    return sub / ("temp_" + safe + "." + ext)

def parse_video(data, ext, blob_id):
    e = ext.lower().strip()
    if e not in SUPPORTED_VIDEO:
        raise ValueError("Unsupported video extension: " + ext)
    path = _temp_path(blob_id, e, "video")
    path.write_bytes(data)
    return path

def parse_audio(data, ext, blob_id):
    e = ext.lower().strip()
    if e not in SUPPORTED_AUDIO:
        raise ValueError("Unsupported audio extension: " + ext)
    path = _temp_path(blob_id, e, "audio")
    path.write_bytes(data)
    return path

def parse_pdf_content(data: bytes, max_images: int = 5):
    """Extracts text and up to `max_images` (sorted by size) from a PDF."""
    import io
    from pypdf import PdfReader
    try:
        reader = PdfReader(io.BytesIO(data))
        text_parts = []
        all_images = []
        
        for page in reader.pages:
            # Text
            extracted = page.extract_text()
            if extracted:
                text_parts.append(extracted)
            
            # Images
            try:
                for img_file in page.images:
                    # pypdf >= 3.0.0
                    pil_img = img_file.image
                    # Filter very small icons/lines
                    if pil_img.width > 50 and pil_img.height > 50:
                        all_images.append(pil_img)
            except Exception:
                pass

        full_text = "\n".join(text_parts)
        
        # Sort images by area (largest first) to capture main content like charts
        all_images.sort(key=lambda x: x.width * x.height, reverse=True)
        return full_text, all_images[:max_images]

    except Exception as e:
        print(f"Warning: PDF parse error: {e}")
        return "", []

def parse_plain_text(data: bytes) -> str:
    # Limit to 10MB
    limit = 10 * 1024 * 1024
    if len(data) > limit:
        data = data[:limit]
    return data.decode("utf-8", errors="replace")

def is_supported_image(ext):
    return ext.lower().strip() in SUPPORTED_IMAGE

def is_supported_video(ext):
    return ext.lower().strip() in SUPPORTED_VIDEO

def is_supported_audio(ext):
    return ext.lower().strip() in SUPPORTED_AUDIO

def is_supported_doc(ext):
    return ext.lower().strip() in SUPPORTED_DOC

__all__ = ["parse_image", "parse_video", "parse_audio", "parse_pdf_content", "parse_plain_text",
           "is_supported_image", "is_supported_video", "is_supported_audio", "is_supported_doc",
           "SUPPORTED_IMAGE", "SUPPORTED_VIDEO", "SUPPORTED_AUDIO", "SUPPORTED_DOC", "TEMP_DIR"]
