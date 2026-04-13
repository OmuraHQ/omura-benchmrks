"""Walrus blob id conversions: on-chain ``blob_id`` (u256) vs HTTP aggregator base64url id."""

from __future__ import annotations

import base64
import os
from typing import Any, Dict, Optional, Tuple

import requests


def uint256_blob_id_to_aggregator_string(blob_id: int | str) -> str:
    """Map Walrus ``Blob.blob_id`` (integer / decimal string) to the id used in ``GET /v1/blobs/{id}``.

    On-chain encoding is **little-endian** 32-byte u256, then URL-safe Base64 without padding.
    """
    n = int(blob_id)
    raw = n.to_bytes(32, "little")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def sui_get_object(object_id: str, *, rpc_url: str | None = None, timeout: float = 30.0) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """``sui_getObject`` with ``showContent``; returns ``(parsed_result_data_dict, error)``."""
    url = (rpc_url or os.getenv("SUI_RPC_URL", "https://fullnode.mainnet.sui.io:443")).strip().rstrip("/")
    if not url.startswith("http"):
        url = "https://" + url
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "sui_getObject",
        "params": [object_id.strip(), {"showContent": True}],
    }
    try:
        resp = requests.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=timeout,
        )
        if resp.status_code != 200:
            return None, f"http_{resp.status_code}"
        body = resp.json()
        if not isinstance(body, dict) or body.get("error"):
            return None, "rpc_error"
        result = body.get("result")
        if not isinstance(result, dict):
            return None, "no_result"
        data = result.get("data")
        if not isinstance(data, dict):
            return None, "object_not_found"
        return data, None
    except requests.RequestException as e:
        return None, f"request:{e!s}"


def aggregator_blob_id_from_sui_blob_object(
    sui_object_id: str, *, rpc_url: str | None = None, timeout: float = 30.0
) -> Tuple[Optional[str], Dict[str, Any], Optional[str]]:
    """Resolve a Sui ``walrus::blob::Blob`` object id to the HTTP aggregator blob id + chain metadata.

    Returns ``(aggregator_blob_id, chain_info, error)``. ``chain_info`` always includes useful fields when found.
    """
    data, err = sui_get_object(sui_object_id, rpc_url=rpc_url, timeout=timeout)
    if err:
        return None, {}, err
    content = data.get("content") or {}
    fields = content.get("fields") or {}
    type_str = str(data.get("type") or content.get("type") or "")
    raw_id = fields.get("blob_id")
    if raw_id is None or not isinstance(fields, dict):
        return None, {}, "missing_blob_id_field"

    agg = uint256_blob_id_to_aggregator_string(raw_id)
    storage = fields.get("storage") or {}
    storage_fields = storage.get("fields") if isinstance(storage, dict) else {}

    chain_info: Dict[str, Any] = {
        "sui_object_id": sui_object_id.strip(),
        "blob_id_uint256": str(raw_id),
        "aggregator_blob_id": agg,
        "type": type_str or None,
        "size": fields.get("size"),
        "registered_epoch": fields.get("registered_epoch"),
        "certified_epoch": fields.get("certified_epoch"),
        "storage_end_epoch": storage_fields.get("end_epoch") if isinstance(storage_fields, dict) else None,
        "storage_start_epoch": storage_fields.get("start_epoch") if isinstance(storage_fields, dict) else None,
    }
    return agg, chain_info, None


__all__ = [
    "aggregator_blob_id_from_sui_blob_object",
    "sui_get_object",
    "uint256_blob_id_to_aggregator_string",
]
