"""Magic-bytes based file type detection for Walrus blobs.

Walrus/aggregator responses often omit ``Content-Type``; we classify from raw bytes.

Primary: **libmagic** via ``python-magic`` on a prefix of the blob (same as ``file(1)``).
Fallback: structural sniff (ISO BMFF / EBML / RIFF / …) when libmagic is missing or
returns only a generic type (e.g. ``application/octet-stream``).
"""

from __future__ import annotations

import struct
from typing import Optional, Tuple

try:
    import magic
except ImportError:
    magic = None

from omura.parsers.quilt import sniff_walrus_quilt_v1

# Bytes passed to libmagic and to structural fallback (libmagic docs suggest ≥2048)
_SNIFF_WINDOW = 8192

# Treat these as "unknown" from libmagic and try structural sniff
_LIBMAGIC_GENERIC_MIMES = frozenset(
    {
        "application/octet-stream",
        "application/octetstream",
        "application/x-empty",
        "inode/x-empty",
    }
)

# ISO BMFF major brands (first ftyp brand, 4 ASCII bytes)
_FTYP_AUDIO_BRANDS = frozenset(
    {
        b"M4A ",
        b"M4B ",
        b"f4a ",
        b"F4A ",
    }
)
_FTYP_IMAGE_BRANDS = frozenset(
    {
        b"heic",
        b"heix",
        b"heif",
        b"mif1",
        b"msf1",
        b"avif",
        b"avis",
        b"AVIF",
    }
)

def _mime_to_ext(mime: str) -> str:
    """Map MIME type to file extension.
    
    Extracts extension from common MIME types. For unknown types,
    attempts to extract from the MIME type string itself.
    """
    # Common MIME type to extension mapping
    mime_ext_map = {
        # Images
        "image/png": "png",
        "image/jpeg": "jpg",
        "image/jpg": "jpg",
        "image/gif": "gif",
        "image/webp": "webp",
        "image/bmp": "bmp",
        "image/tiff": "tiff",
        "image/tif": "tiff",
        "image/svg+xml": "svg",
        "image/x-icon": "ico",
        "image/vnd.microsoft.icon": "ico",
        # Documents
        "application/pdf": "pdf",
        "application/msword": "doc",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
        "application/vnd.ms-excel": "xls",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
        "application/vnd.ms-powerpoint": "ppt",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation": "pptx",
        # Archives
        "application/zip": "zip",
        "application/x-zip-compressed": "zip",
        "application/x-tar": "tar",
        "application/gzip": "gz",
        "application/x-gzip": "gz",
        "application/x-bzip2": "bz2",
        "application/x-7z-compressed": "7z",
        "application/x-rar-compressed": "rar",
        # Walrus protocol
        "application/x-walrus-quilt": "quilt",
        # Videos
        "video/mp4": "mp4",
        "video/webm": "webm",
        "video/quicktime": "mov",
        "video/x-msvideo": "avi",
        "video/x-matroska": "mkv",
        # Audio
        "audio/mpeg": "mp3",
        "audio/mp3": "mp3",
        "audio/wav": "wav",
        "audio/x-wav": "wav",
        "audio/ogg": "ogg",
        "audio/flac": "flac",
        # Text
        "text/plain": "txt",
        "text/html": "html",
        "text/css": "css",
        "text/javascript": "js",
        "text/x-python": "py",
        "application/json": "json",
        "application/xml": "xml",
        "text/xml": "xml",
    }
    
    # Check direct mapping first
    if mime in mime_ext_map:
        return mime_ext_map[mime]
    
    # Try to extract extension from MIME type (e.g., "image/x-png" -> "png")
    # This handles variations and less common MIME types
    parts = mime.split("/")
    if len(parts) == 2:
        subtype = parts[1]
        # Remove common prefixes/suffixes
        subtype = subtype.replace("x-", "").replace("vnd.", "")
        # Remove common suffixes
        for suffix in ["+xml", "+json", "+zip"]:
            if subtype.endswith(suffix):
                subtype = subtype[:-len(suffix)]
        # Use subtype as extension if it looks reasonable
        if len(subtype) <= 5 and subtype.isalnum():
            return subtype
    
    # Fallback
    return "bin"


def _is_probable_fourcc(b: bytes) -> bool:
    if len(b) != 4:
        return False
    return all(32 <= x < 127 for x in b)


def _parse_ftyp_major_brand(data: bytes) -> Optional[bytes]:
    """Return ISO BMFF major brand (4 bytes) if an ``ftyp`` box is found in the sniff window."""
    limit = min(len(data), _SNIFF_WINDOW)
    if limit < 12:
        return None
    window = memoryview(data)[:limit]

    # Common case: one top-level box, ``ftyp`` at offset 4
    if window[4:8].tobytes() == b"ftyp" and _is_probable_fourcc(window[8:12].tobytes()):
        return window[8:12].tobytes()

    pos = 0
    while pos + 8 <= limit:
        try:
            box_size = struct.unpack_from(">I", data, pos)[0]
        except struct.error:
            break
        box_type = data[pos + 4 : pos + 8]
        if box_type == b"ftyp" and pos + 12 <= limit:
            major = data[pos + 8 : pos + 12]
            if _is_probable_fourcc(major):
                return major
        if box_size < 8 or box_size > 1_000_000:
            pos += 1
            continue
        if pos + box_size > limit:
            break
        pos += box_size

    # Last resort: search for ftyp (some writers omit predictable box layout in edge cases)
    idx = data.find(b"ftyp", 0, limit)
    if idx >= 4 and idx + 8 <= limit:
        major = data[idx + 4 : idx + 8]
        if _is_probable_fourcc(major):
            return major
    return None


def _classify_ftyp_brand(major: bytes) -> Tuple[str, str, str]:
    if major in _FTYP_IMAGE_BRANDS:
        ext = "heic" if major.lower().startswith(b"hei") or major in (b"mif1", b"msf1") else "avif"
        return f"image/{ext}", ext, "image"
    if major in _FTYP_AUDIO_BRANDS:
        return "audio/mp4", "m4a", "audio"
    # Default: video container (mp4 / mov / etc.)
    return "video/mp4", "mp4", "video"


def _enhanced_magic_sniff(data: bytes) -> Optional[Tuple[str, str, str]]:
    """Strong magic-byte / container sniff. Returns None if inconclusive."""
    if not data:
        return None

    n = len(data)
    b = data

    # Walrus quilt (batch blob): v1 + LE u32 + BCS QuiltIndexV1; must run before 0x01-ish heuristics
    q = sniff_walrus_quilt_v1(b)
    if q is not None:
        return q

    # --- Audio: codecs / containers (before RIFF / ISO) ---
    if n >= 4 and b[:4] == b"fLaC":
        return "audio/flac", "flac", "audio"
    if n >= 2 and b[0] == 0xFF and (b[1] & 0xF6) in (0xF0, 0xF8):
        return "audio/aac", "aac", "audio"
    if n >= 3 and b[:3] == b"ID3":
        return "audio/mpeg", "mp3", "audio"
    if n >= 2 and b[0] == 0xFF and (b[1] & 0xE0) == 0xE0:
        return "audio/mpeg", "mp3", "audio"

    # Ogg: Theora vs typical audio
    if n >= 4 and b[:4] == b"OggS":
        head = b[: min(n, 512)]
        if b"theora" in head or b"dirac" in head:
            return "video/ogg", "ogv", "video"
        return "audio/ogg", "ogg", "audio"

    # --- EBML (WebM / Matroska) ---
    if n >= 4 and b[:4] == b"\x1a\x45\xdf\xa3":
        head = b[: min(n, 2048)]
        if b"webm" in head:
            return "video/webm", "webm", "video"
        return "video/x-matroska", "mkv", "video"

    # --- RIFF ---
    if n >= 12 and b[:4] == b"RIFF":
        fourcc = b[8:12]
        if fourcc == b"WAVE":
            return "audio/wav", "wav", "audio"
        if fourcc == b"AVI ":
            return "video/x-msvideo", "avi", "video"
        if fourcc == b"WEBP":
            return "image/webp", "webp", "image"
        if fourcc in (b"QLCM", b"AMV "):
            return "video/x-msvideo", "avi", "video"

    # --- MPEG program stream / elementary ---
    if n >= 4 and b[:3] == b"\x00\x00\x01" and b[3] in (0xBA, 0xB3, 0xBB):
        return "video/mpeg", "mpg", "video"

    # MPEG transport stream (sync byte 0x47 every 188 bytes)
    if n >= 564:
        if b[0] == 0x47 and b[188] == 0x47 and b[376] == 0x47:
            return "video/mp2t", "ts", "video"

    # Flash Video
    if n >= 4 and b[:3] == b"FLV" and b[3] == 0x01:
        return "video/x-flv", "flv", "video"

    # ISO BMFF (MP4 / MOV / M4A / HEIC / AVIF …)
    major = _parse_ftyp_major_brand(b)
    if major is not None:
        return _classify_ftyp_brand(major)

    # --- Still images & PDF (fast paths) ---
    if b.startswith(b"%PDF-"):
        return "application/pdf", "pdf", "pdf"
    if b.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png", "png", "image"
    if b.startswith(b"\xff\xd8\xff"):
        return "image/jpeg", "jpg", "image"
    if b.startswith(b"GIF87a") or b.startswith(b"GIF89a"):
        return "image/gif", "gif", "image"
    if b.startswith(b"BM"):
        return "image/bmp", "bmp", "image"
    if b.startswith(b"II*\x00") or b.startswith(b"MM\x00*"):
        return "image/tiff", "tiff", "image"

    # 7z
    if n >= 6 and b[:6] == b"7z\xbc\xaf\x27\x1c":
        return "application/x-7z-compressed", "7z", "archive"

    # gzip
    if n >= 2 and b[:2] == b"\x1f\x8b":
        return "application/gzip", "gz", "archive"

    # ZIP
    if n >= 4 and b.startswith(b"PK\x03\x04"):
        return "application/zip", "zip", "archive"

    return None


def _kind_from_mime(mime_type: str) -> str:
    if mime_type == "application/x-walrus-quilt":
        return "quilt"
    if mime_type.startswith("image/"):
        return "image"
    if mime_type.startswith("video/"):
        return "video"
    if mime_type.startswith("audio/"):
        return "audio"
    if mime_type.startswith("text/"):
        return "text"
    if mime_type == "application/pdf":
        return "pdf"
    if mime_type in ("application/zip", "application/x-zip-compressed"):
        return "archive"
    if mime_type.startswith("application/"):
        return "application"
    return "binary"


def _libmagic_detect(data: bytes) -> Optional[Tuple[str, str, str]]:
    """Run libmagic on a buffer prefix. Returns None if unavailable or result is too generic."""
    if magic is None:
        return None
    try:
        buf_len = min(len(data), max(2048, _SNIFF_WINDOW))
        buffer = data[:buf_len]
        mime_type = magic.from_buffer(buffer, mime=True)
        if not mime_type:
            return None
        mime_type = mime_type.strip()
        if mime_type.lower() in _LIBMAGIC_GENERIC_MIMES:
            return None
        extension = _mime_to_ext(mime_type)
        return mime_type, extension, _kind_from_mime(mime_type)
    except Exception:
        return None


def detect_file_type(data: bytes) -> Tuple[str, str, str]:
    """Detect file type from blob bytes (no HTTP Content-Type required).

    Order: libmagic on a prefix, then structural sniff, then tiny text heuristic.

    Returns:
        (mime_type, extension, kind_string)
    """
    if not data:
        return "application/octet-stream", "bin", "empty"

    lm = _libmagic_detect(data)
    if lm is not None:
        return lm

    enhanced = _enhanced_magic_sniff(data)
    if enhanced is not None:
        return enhanced

    header = data[:16]
    if b"\x00" not in header:
        try:
            header.decode("utf-8")
            return "text/plain", "txt", "text"
        except UnicodeDecodeError:
            pass

    return "application/octet-stream", "bin", "binary"


__all__ = ["detect_file_type"]
