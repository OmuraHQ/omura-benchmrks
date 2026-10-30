"""Aggregator blob fetching helpers for offline backfill scripts.

Kept intentionally dependency-free (requests only) and separate from the
in-process aggregator pool: backfills hammer public aggregators concurrently and
should not trip the live pool's failure counters.
"""

from __future__ import annotations

import os
import threading
from typing import Optional

import requests

AGGREGATORS = [
    a.strip().rstrip("/")
    for a in os.getenv(
        "OMURA_FETCH_AGGREGATORS",
        "https://agrregator.omura.fun,https://aggregator.walrus-mainnet.walrus.space",
    ).split(",")
    if a.strip()
]

MAX_FETCH_BYTES = int(
    os.getenv("OMURA_MAX_MEDIA_BYTES", str(200 * 1024 * 1024))
)
FETCH_TIMEOUT = float(os.getenv("OMURA_FETCH_TIMEOUT", "60"))
FETCH_ATTEMPTS = max(1, int(os.getenv("OMURA_FETCH_ATTEMPTS", "1")))

_tls = threading.local()


def _session() -> requests.Session:
    session = getattr(_tls, "session", None)
    if session is None:
        session = requests.Session()
        _tls.session = session
    return session


def _fetch_patch(blob_id: str, timeout: int) -> Optional[bytes]:
    """Resolve ``quiltId::identifier`` to a patch id, then fetch its bytes."""
    quilt_id, ident = blob_id.split("::", 1)
    for agg in AGGREGATORS:
        try:
            resp = _session().get(f"{agg}/v1/quilts/{quilt_id}/patches", timeout=timeout)
            if resp.status_code != 200:
                continue
            patch_id = next(
                (p.get("patch_id") for p in resp.json() if p.get("identifier") == ident),
                None,
            )
            if not patch_id:
                continue
            patch = _session().get(
                f"{agg}/v1/blobs/by-quilt-patch-id/{patch_id}", timeout=timeout
            )
            if patch.status_code == 200 and patch.content:
                return patch.content
        except Exception:  # noqa: BLE001 - try the next aggregator
            continue
    return None


def _resolve_patch_id(quilt_id: str, ident: str, timeout: float) -> Optional[str]:
    for agg in AGGREGATORS:
        try:
            resp = _session().get(f"{agg}/v1/quilts/{quilt_id}/patches", timeout=timeout)
            if resp.status_code != 200:
                continue
            patch_id = next(
                (p.get("patch_id") for p in resp.json() if p.get("identifier") == ident),
                None,
            )
            if patch_id:
                return patch_id
        except Exception:  # noqa: BLE001
            continue
    return None


def _stream_prefix(url: str, max_bytes: int, timeout: float) -> Optional[bytes]:
    try:
        resp = _session().get(
            url,
            timeout=timeout,
            stream=True,
            headers={"Range": f"bytes=0-{max_bytes - 1}"},
        )
        if resp.status_code not in (200, 206):
            resp.close()
            return None
        buf = bytearray()
        for chunk in resp.iter_content(1 << 20):
            if not chunk:
                break
            buf.extend(chunk)
            if len(buf) >= max_bytes:
                break
        resp.close()
        return bytes(buf[:max_bytes]) if buf else None
    except Exception:  # noqa: BLE001
        return None


def fetch_prefix(
    blob_id: str, max_bytes: int, timeout: float | None = None
) -> Optional[bytes]:
    """Fetch at most ``max_bytes`` of a blob (including quilt patches).

    Used for very large videos: many MP4s are faststart, so a prefix still
    decodes the opening frames. If the server ignores Range (200), the stream is
    closed after the cap instead of downloading the whole blob.
    """
    timeout = FETCH_TIMEOUT if timeout is None else timeout
    if "::" in blob_id:
        quilt_id, ident = blob_id.split("::", 1)
        patch_id = _resolve_patch_id(quilt_id, ident, timeout)
        if not patch_id:
            return None
        for agg in AGGREGATORS:
            data = _stream_prefix(
                f"{agg}/v1/blobs/by-quilt-patch-id/{patch_id}", max_bytes, timeout
            )
            if data:
                return data
        return None

    for agg in AGGREGATORS:
        data = _stream_prefix(f"{agg}/v1/blobs/{blob_id}", max_bytes, timeout)
        if data:
            return data
    return None


def robust_fetch(blob_id: str, timeout: float | None = None) -> Optional[bytes]:
    timeout = FETCH_TIMEOUT if timeout is None else timeout
    if "::" in blob_id:
        return _fetch_patch(blob_id, timeout)
    for agg in AGGREGATORS:
        for _ in range(FETCH_ATTEMPTS):
            try:
                resp = _session().get(f"{agg}/v1/blobs/{blob_id}", timeout=timeout)
                if resp.status_code == 200 and resp.content:
                    return resp.content
            except Exception:  # noqa: BLE001
                pass
    return None


__all__ = ["AGGREGATORS", "MAX_FETCH_BYTES", "robust_fetch", "fetch_prefix"]
