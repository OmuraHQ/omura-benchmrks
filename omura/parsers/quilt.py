"""Walrus quilt v1: parse index, infer matrix layout, extract inner blobs (crawl patches).

Matches Mysten TypeScript ``encodeQuilt`` / ``QuiltReader`` layout (RS2 grid).
Aggregator returns decoded blob bytes (no HTTP Content-Type); detection uses magic + layout.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple

# --- grid geometry (RS2), from Mysten ``getSourceSymbols`` ---


def _max_faulty_nodes(n_shards: int) -> int:
    return (n_shards - 1) // 3


def _decoding_safety_limit(n_shards: int) -> int:
    return 0  # RS2


def get_source_symbols_rs2(n_shards: int) -> Tuple[int, int]:
    """Returns (primary_symbols / n_rows, secondary_symbols / n_cols)."""
    safety = _decoding_safety_limit(n_shards)
    mf = _max_faulty_nodes(n_shards)
    min_correct = n_shards - mf
    primary = min_correct - mf - safety
    secondary = min_correct - safety
    return primary, secondary


# --- BCS helpers (same as file_detection quilt sniff) ---

_MAX_QUILT_PATCHES = 666
_MAX_ID_LEN = (1 << 16) - 1
_MAX_TAG_ENTRIES = 4096
QUILT_INDEX_VERSION = 1
HAS_TAGS_FLAG = 1 << 0


def _read_uleb128(data: bytes, pos: int) -> Optional[Tuple[int, int]]:
    value = 0
    shift = 0
    p = pos
    while p < len(data):
        b = data[p]
        p += 1
        value |= (b & 0x7F) << shift
        if (b & 0x80) == 0:
            return value, p
        shift += 7
        if shift > 63:
            return None
    return None


def _bcs_utf8_string(data: bytes, pos: int) -> Optional[Tuple[str, int]]:
    r = _read_uleb128(data, pos)
    if r is None:
        return None
    ln, pos = r
    if ln > _MAX_ID_LEN or pos + ln > len(data):
        return None
    raw = data[pos : pos + ln]
    try:
        s = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None
    return s, pos + ln


def _parse_bcs_map_string_string(data: bytes, pos: int) -> Optional[Tuple[Dict[str, str], int]]:
    r = _read_uleb128(data, pos)
    if r is None:
        return None
    n, pos = r
    if n > _MAX_TAG_ENTRIES:
        return None
    out: Dict[str, str] = {}
    for _ in range(n):
        kr = _bcs_utf8_string(data, pos)
        if kr is None:
            return None
        k, pos = kr
        vr = _bcs_utf8_string(data, pos)
        if vr is None:
            return None
        v, pos = vr
        out[k] = v
    return out, pos


def parse_quilt_index_body(body: bytes) -> Optional[List[Tuple[int, str, Dict[str, str]]]]:
    """QuiltIndexV1: vector of (end_column: u16, identifier, tags map)."""
    r = _read_uleb128(body, 0)
    if r is None:
        return None
    n_patches, pos = r
    if n_patches > _MAX_QUILT_PATCHES:
        return None
    patches: List[Tuple[int, str, Dict[str, str]]] = []
    for _ in range(n_patches):
        if pos + 2 > len(body):
            return None
        end_col = struct.unpack_from("<H", body, pos)[0]
        pos += 2
        ir = _bcs_utf8_string(body, pos)
        if ir is None:
            return None
        ident, pos = ir
        tr = _parse_bcs_map_string_string(body, pos)
        if tr is None:
            return None
        tags, pos = tr
        patches.append((end_col, ident, tags))
    if pos != len(body):
        return None
    return patches


def parse_quilt_linear_prefix(blob: bytes) -> Optional[Tuple[int, bytes, List[Tuple[int, str, Dict[str, str]]]]]:
    """
    If ``blob`` starts with a valid v1 quilt index, return
    (index_total_len, full_index_bytes, patches_meta).
    """
    if len(blob) < 6 or blob[0] != QUILT_INDEX_VERSION:
        return None
    body_len = struct.unpack_from("<I", blob, 1)[0]
    if body_len < 1 or body_len > 50_000_000:
        return None
    if len(blob) < 5 + body_len:
        return None
    body = blob[5 : 5 + body_len]
    patches = parse_quilt_index_body(body)
    if patches is None:
        return None
    index_total = 5 + body_len
    return index_total, blob[:index_total], patches


def quilt_read_linear(
    blob: bytes,
    start_column: int,
    total_bytes: int,
    offset_within_patch: int,
    symbol_size: int,
    n_rows: int,
    n_cols: int,
) -> bytes:
    """Port of Mysten ``#readBytesFromBlob`` (matrix column-major symbol fill)."""
    if total_bytes <= 0:
        return b""

    row_size = symbol_size * n_cols
    if row_size <= 0 or len(blob) % row_size != 0:
        return b""

    n_rows_total = len(blob) // row_size
    result = bytearray(total_bytes)
    bytes_read = 0

    symbols_to_skip = offset_within_patch // symbol_size
    remaining_offset = offset_within_patch % symbol_size
    current_col = start_column + symbols_to_skip // n_rows_total
    current_row = symbols_to_skip % n_rows_total

    while bytes_read < total_bytes:
        base_index = current_row * row_size + current_col * symbol_size
        start_index = base_index + remaining_offset
        end_bound = min(
            base_index + symbol_size,
            start_index + total_bytes - bytes_read,
            len(blob),
        )
        size = end_bound - start_index
        if start_index >= len(blob) or size <= 0:
            break
        result[bytes_read : bytes_read + size] = blob[start_index : start_index + size]
        bytes_read += size
        remaining_offset = 0
        current_row = (current_row + 1) % n_rows_total
        if current_row == 0:
            current_col += 1

    return bytes(result[:bytes_read])


def infer_quilt_layout(blob: bytes) -> Optional[Tuple[int, int, int, int]]:
    """
    Brute-force RS2 shard counts so matrix read of the index matches the linear prefix.

    Returns (symbol_size, n_rows, n_cols, index_total_len) or None.
    """
    parsed = parse_quilt_linear_prefix(blob)
    if parsed is None:
        return None
    index_total_len, linear_index, _patches = parsed

    blob_len = len(blob)
    for n_shards in range(10, min(2000, blob_len + 1)):
        n_rows, n_cols = get_source_symbols_rs2(n_shards)
        cells = n_rows * n_cols
        if cells <= 0 or blob_len % cells != 0:
            continue
        symbol_size = blob_len // cells
        if symbol_size < 2 or symbol_size > 65535:
            continue
        row_size = symbol_size * n_cols
        if blob_len % row_size != 0:
            continue
        rec = quilt_read_linear(blob, 0, index_total_len, 0, symbol_size, n_rows, n_cols)
        if rec == linear_index:
            return symbol_size, n_rows, n_cols, index_total_len

    return None


def parse_quilt_patch_blob(raw: bytes) -> Optional[Tuple[str, Dict[str, str], bytes]]:
    """Parse one patch stream: ``QuiltPatchBlobHeader`` + id + tags + inner file bytes.

    NOTE: This local parser does NOT fully match Mysten's wire format and yields
    truncated identifiers / unaligned content for some quilts. Prefer the aggregator
    endpoints ``/v1/quilts/{id}/patches`` and ``/v1/blobs/by-quilt-id/{id}/{ident}``
    (or ``by-quilt-patch-id/{patch_id}``) for production use.
    """
    if len(raw) < 6:
        return None
    if raw[0] != 1:
        return None
    payload_len = struct.unpack_from("<I", raw, 1)[0]
    mask = raw[5]
    pos = 6
    end = 6 + payload_len
    if end > len(raw):
        return None
    if pos + 2 > end:
        return None
    id_len = struct.unpack_from("<H", raw, pos)[0]
    pos += 2
    if pos + id_len > end:
        return None
    identifier = raw[pos : pos + id_len].decode("utf-8")
    pos += id_len
    tags: Dict[str, str] = {}
    if mask & HAS_TAGS_FLAG:
        if pos + 2 > end:
            return None
        tags_len = struct.unpack_from("<H", raw, pos)[0]
        pos += 2
        if pos + tags_len > end:
            return None
        tslice = raw[pos : pos + tags_len]
        tr = _parse_bcs_map_string_string(tslice, 0)
        if tr is None:
            return None
        tags, consumed = tr
        if consumed != len(tslice):
            return None
        pos += tags_len
    content = raw[pos:end]
    return identifier, tags, content


@dataclass
class QuiltLayout:
    symbol_size: int
    n_rows: int
    n_cols: int
    index_total_len: int
    column_size: int
    index_columns: int
    patches: List[Tuple[int, str, Dict[str, str]]]  # end_col, identifier, tags


def resolve_layout(blob: bytes) -> Optional[QuiltLayout]:
    parsed = parse_quilt_linear_prefix(blob)
    if parsed is None:
        return None
    index_total_len, _linear, patch_meta = parsed
    inf = infer_quilt_layout(blob)
    if inf is None:
        return None
    symbol_size, n_rows, n_cols, idx_len = inf
    assert idx_len == index_total_len
    column_size = symbol_size * n_rows
    index_columns = (index_total_len + column_size - 1) // column_size
    return QuiltLayout(
        symbol_size=symbol_size,
        n_rows=n_rows,
        n_cols=n_cols,
        index_total_len=index_total_len,
        column_size=column_size,
        index_columns=index_columns,
        patches=patch_meta,
    )


def iter_quilt_patch_contents(blob: bytes) -> Iterator[Tuple[str, Dict[str, str], bytes]]:
    """Yield ``(identifier, tags, inner_bytes)`` for each patch in a quilt blob."""
    layout = resolve_layout(blob)
    if layout is None:
        return

    start_col = layout.index_columns
    for end_col, identifier, _index_tags in layout.patches:
        if end_col < start_col:
            return
        n_cols_span = end_col - start_col
        raw_len = n_cols_span * layout.column_size
        raw = quilt_read_linear(
            blob,
            start_col,
            raw_len,
            0,
            layout.symbol_size,
            layout.n_rows,
            layout.n_cols,
        )
        parsed = parse_quilt_patch_blob(raw)
        if parsed is None:
            return
        ident, tmap, content = parsed
        if ident != identifier:
            return
        yield identifier, tmap, content
        start_col = end_col


def sniff_walrus_quilt_v1(data: bytes) -> Optional[Tuple[str, str, str]]:
    """For ``detect_file_type``: MIME / ext / kind if linear quilt index is valid at offset 0."""
    if parse_quilt_linear_prefix(data) is None:
        return None
    return "application/x-walrus-quilt", "quilt", "quilt"


__all__ = [
    "iter_quilt_patch_contents",
    "resolve_layout",
    "sniff_walrus_quilt_v1",
    "parse_quilt_linear_prefix",
    "QUILT_INDEX_VERSION",
]
