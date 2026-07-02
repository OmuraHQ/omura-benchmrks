"""Blockberry API utilities for discovering Walrus blob IDs.

This module centralizes the logic that was previously in `get_blob_ids.py`:
- infer current Walrus epoch
- page through /walrus-mainnet/v1/blobs
- collect all and active blob IDs with metadata
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import requests
from tqdm import tqdm


BLOCKBERRY_API_URL = "https://api.blockberry.one/walrus-mainnet/v1/blobs"


def blockberry_api_key() -> str:
    """API key from environment (set BLOCKBERRY_API_KEY)."""
    return (os.getenv("BLOCKBERRY_API_KEY") or "").strip()


DEFAULT_PAGE_SIZE = 100
DEFAULT_TIMEOUT = 30


class WalrusEpochError(RuntimeError):
    """Raised when the current Walrus epoch cannot be resolved from any source."""


def _item_size_int(item: Dict[str, Any]) -> int:
    s = item.get("size") or item.get("fileSize")
    if s is None:
        return 0
    try:
        return int(s)
    except (TypeError, ValueError):
        return 0


def _blob_list_sort_by_size(
    items: List[Dict[str, Any]], *, descending: bool
) -> List[Dict[str, Any]]:
    """Sort API items by declared size. Unknown/null size is treated as 0 (last when descending)."""

    return sorted(items, key=_item_size_int, reverse=descending)


@dataclass
class FetchState:
    """Mutable state while paging through results."""

    all_blob_ids: Set[str]
    active_blob_ids: Set[str]
    blob_metadata: Dict[str, Dict[str, Any]]
    total_blobs: Optional[int] = None
    pbar: Optional[tqdm] = None


def get_headers() -> Dict[str, str]:
    """Shared headers for Blockberry requests."""
    return {"accept": "*/*", "x-api-key": blockberry_api_key()}


def _parse_json_body(
    response: requests.Response,
) -> Tuple[Optional[Any], Optional[str]]:
    """Parse Blockberry JSON body. Returns ``(data, None)`` or ``(None, short error)``."""
    raw = (response.text or "").strip()
    if not raw:
        return None, f"empty body (HTTP {response.status_code})"
    try:
        return json.loads(raw), None
    except json.JSONDecodeError:
        preview = raw[:200].replace("\n", " ")
        return None, f"not JSON (HTTP {response.status_code}): {preview!r}"


def get_blob_details_by_id(
    blob_id: str, timeout: float = DEFAULT_TIMEOUT
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """GET ``/v1/blobs/{id}`` (Blockberry ``getBlobById``; id is usually ``blobIdBase64``).

    Returns ``(details_dict, None)`` on success, or ``(None, error_tag)`` with tags such as
    ``no_api_key``, ``not_found``, ``http_<status>``, or ``invalid_response``.
    """
    if not blockberry_api_key():
        return None, "no_api_key"
    url = f"{BLOCKBERRY_API_URL.rstrip('/')}/{blob_id}"
    response = requests.get(url, headers=get_headers(), timeout=timeout)
    if response.status_code == 404:
        return None, "not_found"
    if response.status_code != 200:
        return None, f"http_{response.status_code}"
    data, jerr = _parse_json_body(response)
    if jerr or not isinstance(data, dict):
        return None, "invalid_response"
    return data, None


def is_blob_expired_for_epoch(
    details: Dict[str, Any], current_epoch: int
) -> Optional[bool]:
    """True if ``endEpoch <= current_epoch`` (same active rule as :func:`iter_active_blobs`)."""
    end = details.get("endEpoch") or details.get("end_epoch")
    if end is None:
        return None
    try:
        end_i = int(end)
    except (TypeError, ValueError):
        return None
    return end_i <= current_epoch


def fetch_recent_items(page_size: int = DEFAULT_PAGE_SIZE) -> List[Dict[str, Any]]:
    """Fetch the most recent blobs to infer the current epoch."""
    params = {"page": 0, "size": page_size, "orderBy": "DESC", "sortBy": "TIMESTAMP"}
    response = requests.get(
        BLOCKBERRY_API_URL,
        params=params,
        headers=get_headers(),
        timeout=DEFAULT_TIMEOUT,
    )
    if response.status_code != 200:
        return []
    data, jerr = _parse_json_body(response)
    if jerr:
        print(f"  Blockberry: {jerr}", file=sys.stderr)
        if not blockberry_api_key():
            print(
                "  Hint: set BLOCKBERRY_API_KEY (empty or HTML bodies often mean missing key).",
                file=sys.stderr,
            )
        return []
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        items = (
            data.get("content")
            or data.get("data")
            or data.get("items")
            or data.get("blobs")
        )
    else:
        items = None
    return items or []


def _epoch_from_sui_dynamic_field_response(body: Dict[str, Any]) -> Optional[int]:
    """Parse ``epoch`` from ``suix_getDynamicFieldObject`` JSON (``result.data.content.fields.value.fields.epoch``)."""
    if not isinstance(body, dict) or body.get("error"):
        return None
    result = body.get("result")
    if not isinstance(result, dict):
        return None
    data = result.get("data")
    if not isinstance(data, dict):
        return None
    content = data.get("content") or {}
    fields = content.get("fields") or {}
    value = fields.get("value")
    if not isinstance(value, dict):
        return None
    inner = value.get("fields") or {}
    ep = inner.get("epoch")
    if ep is None:
        return None
    try:
        return int(ep)
    except (TypeError, ValueError):
        return None


def _sui_rpc_post(url: str, payload: Dict[str, Any], timeout: float) -> Optional[Dict[str, Any]]:
    """POST a single JSON-RPC call; return parsed body or None on any failure."""
    resp = requests.post(
        url,
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=timeout,
    )
    if resp.status_code != 200:
        return None
    body, jerr = _parse_json_body(resp)
    if jerr or not isinstance(body, dict):
        return None
    return body


def get_walrus_epoch_from_sui_rpc(
    *, silent: bool = False, timeout: float = DEFAULT_TIMEOUT
) -> Optional[int]:
    """Read current Walrus epoch from Sui mainnet.

    Walrus stores its state as a dynamic field on the ``staking`` object keyed by the
    contract version. We:
      1. Fetch the staking object to read its current ``version`` field.
      2. Look up the dynamic field keyed by that version (type ``u64``).
      3. Extract ``StakingInnerVN.epoch``.

    This is robust across Walrus contract upgrades (v1, v2, v3, ...) — no hardcoded
    field key. Set ``WALRUS_SUI_DYNAMIC_FIELD_U64`` to override version discovery for
    testing; otherwise it is read live.

    Env: ``SUI_RPC_URL``, ``WALRUS_SUI_SYSTEM_STATE_PARENT``.
    """
    rpc_url = (
        os.getenv("SUI_RPC_URL", "https://fullnode.mainnet.sui.io:443")
        .strip()
        .rstrip("/")
    )
    if not rpc_url.startswith("http"):
        rpc_url = "https://" + rpc_url
    parent = os.getenv(
        "WALRUS_SUI_SYSTEM_STATE_PARENT",
        "0x10b9d30c28448939ce6c4d6c6e0ffce4a7f8a4ada8248bdad09ef8b70e4a3904",
    ).strip()

    try:
        version_override = os.getenv("WALRUS_SUI_DYNAMIC_FIELD_U64", "").strip()
        if version_override:
            field_u64 = version_override
        else:
            body = _sui_rpc_post(
                rpc_url,
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "sui_getObject",
                    "params": [parent, {"showContent": True}],
                },
                timeout,
            )
            if body is None:
                if not silent:
                    print("  Sui RPC: failed to fetch staking object", file=sys.stderr)
                return None
            try:
                field_u64 = str(
                    body["result"]["data"]["content"]["fields"]["version"]
                )
            except (KeyError, TypeError) as e:
                if not silent:
                    print(
                        f"  Sui RPC: staking object missing version field: {e}",
                        file=sys.stderr,
                    )
                return None

        body = _sui_rpc_post(
            rpc_url,
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "suix_getDynamicFieldObject",
                "params": [parent, {"type": "u64", "value": field_u64}],
            },
            timeout,
        )
        if body is None:
            return None
        return _epoch_from_sui_dynamic_field_response(body)
    except requests.RequestException as e:
        if not silent:
            print(f"  Sui RPC request failed: {e}", file=sys.stderr)
        return None
    except Exception as e:
        if not silent:
            print(f"  Sui RPC error: {e}", file=sys.stderr)
        return None


def _infer_walrus_epoch_from_blockberry(*, silent: bool = False) -> Optional[int]:
    """Infer current Walrus epoch from recent Blockberry blob listings."""
    try:
        if not silent:
            print("  Querying recent blobs to infer Walrus epoch...")
        items = fetch_recent_items()
        if not items:
            if not silent:
                print("  Could not infer epoch from blob data")
            return None

        start_epochs: List[int] = []
        end_epochs: List[int] = []

        for item in items:
            start_epoch = (
                item.get("startEpoch")
                or item.get("start_epoch")
                or item.get("registeredEpoch")
            )
            if start_epoch:
                try:
                    start_epochs.append(int(start_epoch))
                except (ValueError, TypeError):
                    pass

            end_epoch = (
                item.get("endEpoch")
                or item.get("end_epoch")
                or item.get("expiresAtEpoch")
            )
            if end_epoch:
                try:
                    end_epochs.append(int(end_epoch))
                except (ValueError, TypeError):
                    pass

        if start_epochs:
            from collections import Counter

            max_start_epoch = max(start_epochs)
            min_start_epoch = min(start_epochs)
            start_counts = Counter(start_epochs)
            most_common_start, most_common_count = start_counts.most_common(1)[0]
            current_epoch = most_common_start

            if not silent:
                print(f"  Detected Walrus epoch: {current_epoch}")
                print(
                    f"    (startEpoch range: {min_start_epoch}-{max_start_epoch}, most common: {most_common_start} x{most_common_count})"
                )
                if end_epochs:
                    print(f"    (endEpoch range: {min(end_epochs)}-{max(end_epochs)})")

            return current_epoch

        if not silent:
            print("  Could not infer epoch from blob data")
        return None
    except Exception as e:
        if not silent:
            print(f"  Warning: Could not get current epoch: {e}")
        return None


def get_current_epoch(*, silent: bool = False) -> int:
    """Resolve current Walrus epoch from Sui RPC, then Blockberry heuristics.

    Raises ``WalrusEpochError`` if no source produces an epoch. Callers that need
    a sentinel value should catch this explicitly — there is no implicit fallback.
    """
    source = os.getenv("OMURA_WALRUS_EPOCH_SOURCE", "auto").strip().lower()
    if source not in ("auto", "sui", "blockberry"):
        source = "auto"

    try_sui = source in ("auto", "sui")
    try_bb = source in ("auto", "blockberry")

    if try_sui:
        if not silent:
            print("  Querying Walrus epoch from Sui mainnet RPC...")
        ep = get_walrus_epoch_from_sui_rpc(silent=silent)
        if ep is not None:
            if not silent:
                print(f"  Walrus epoch (Sui): {ep}")
            return ep
        if source == "sui":
            raise WalrusEpochError("Sui RPC did not return a Walrus epoch.")
        if not silent:
            print("  Sui RPC failed or empty; falling back to Blockberry...")

    if try_bb:
        ep = _infer_walrus_epoch_from_blockberry(silent=silent)
        if ep is not None:
            return ep
    raise WalrusEpochError(
        "Could not resolve current Walrus epoch from Sui RPC or Blockberry."
    )


def get_all_blob_ids() -> Tuple[List[str], List[str], Dict[str, Any]]:
    """Get all blob IDs from Walrus mainnet and filter for active ones.

    Returns:
        (all_blob_ids, active_blob_ids, blob_metadata)
    """
    print(f"Connecting to Blockberry API: {BLOCKBERRY_API_URL}")
    print("Fetching ALL active Walrus blob IDs...\n")

    current_epoch = get_current_epoch()
    print("Using Walrus epoch for filtering...")
    print(f"Current Walrus epoch: {current_epoch}")
    print(f"Filtering for blobs with end_epoch > {current_epoch} (active blobs only)\n")

    state = FetchState(all_blob_ids=set(), active_blob_ids=set(), blob_metadata={})
    page = 0
    page_size = DEFAULT_PAGE_SIZE
    headers = get_headers()

    while True:
        try:
            # For the main scan, sort by blob size so we see largest blobs first.
            params = {
                "page": page,
                "size": page_size,
                "orderBy": "DESC",
                "sortBy": "SIZE",
            }

            if state.pbar:
                state.pbar.set_description(f"Fetching page {page}")
            else:
                print(f"Fetching page {page}...", end="\r")

            response = requests.get(
                BLOCKBERRY_API_URL,
                params=params,
                headers=headers,
                timeout=DEFAULT_TIMEOUT,
            )

            if response.status_code == 429:
                print(f"\nRate limited (HTTP 429) at page {page}, waiting 5 seconds...")
                time.sleep(5)
                continue
            elif response.status_code != 200:
                print(f"\nError: HTTP {response.status_code}")
                print(f"Response: {response.text[:200]}")
                if response.status_code == 400:
                    break
                print("Waiting 2 seconds before retry...")
                time.sleep(2)
                continue

            data, jerr = _parse_json_body(response)
            if jerr:
                print(f"\nBlockberry: {jerr}", file=sys.stderr)
                if not blockberry_api_key():
                    print(
                        "  Hint: set BLOCKBERRY_API_KEY (empty or HTML bodies often mean missing key).",
                        file=sys.stderr,
                    )
                break

            items = None
            if isinstance(data, dict):
                items = (
                    data.get("data")
                    or data.get("items")
                    or data.get("blobs")
                    or data.get("content")
                )
                if isinstance(data, dict) and "list" in data:
                    items = data["list"]

                if state.total_blobs is None:
                    state.total_blobs = (
                        data.get("total")
                        or data.get("totalElements")
                        or data.get("count")
                    )
                    if state.total_blobs:
                        total_blobs_int = int(state.total_blobs)
                        print(f"\nTotal blobs available: {total_blobs_int:,}")
                        state.pbar = tqdm(
                            total=total_blobs_int,
                            desc="Fetching blobs",
                            unit="blob",
                            unit_scale=True,
                            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
                        )
                        state.pbar.set_postfix({"active": 0, "expired": 0})
            elif isinstance(data, list):
                items = data

            if not items:
                print(f"\nNo more items found at page {page}")
                break

            page_blob_ids: Set[str] = set()
            page_active_blob_ids: Set[str] = set()
            page_expired_count = 0

            for item in items:
                end_epoch = (
                    item.get("endEpoch")
                    or item.get("end_epoch")
                    or item.get("expiresAtEpoch")
                )
                if end_epoch is None:
                    page_expired_count += 1
                    continue
                try:
                    end_epoch_int = int(end_epoch)
                except (ValueError, TypeError):
                    page_expired_count += 1
                    continue
                if end_epoch_int <= current_epoch:
                    page_expired_count += 1
                    continue

                # Prefer the canonical Walrus base64 blob identifier when available.
                # Blockberry returns both a numeric `blobId` and a base64 `blobIdBase64`;
                # the HTTP aggregator expects the base64 form.
                blob_id = (
                    item.get("blobIdBase64")
                    or item.get("blob_id_base64")
                    or item.get("blobId")
                    or item.get("blob_id")
                    or item.get("id")
                    or item.get("identifier")
                    or item.get("hash")
                )
                if blob_id:
                    state.all_blob_ids.add(blob_id)
                    page_blob_ids.add(blob_id)

                    size = item.get("size") or item.get("fileSize")
                    owner = item.get("owner") or item.get("ownerAddress")
                    deletable = item.get("deletable") or item.get("isDeletable")

                    state.blob_metadata[blob_id] = {
                        "end_epoch": end_epoch,
                        "size": size,
                        "owner": owner,
                        "deletable": deletable,
                        "timestamp": item.get("timestamp") or item.get("createdAt"),
                    }

                    state.active_blob_ids.add(blob_id)
                    page_active_blob_ids.add(blob_id)

            if state.pbar:
                state.pbar.update(len(items))
                state.pbar.set_postfix(
                    {
                        "active": len(state.active_blob_ids),
                        "expired": len(state.all_blob_ids) - len(state.active_blob_ids),
                        "page": page,
                    }
                )

            if len(page_blob_ids) == 0 and len(items) > 0:
                page += 1
                if page % 10 == 0:
                    time.sleep(0.5)
                continue

            if len(page_blob_ids) > 0:
                if state.pbar:
                    tqdm.write(
                        f"✓ Page {page}: Found {len(page_blob_ids)} active blobs!"
                    )
                else:
                    print(
                        f"\nPage {page}: Found {len(page_blob_ids)} active blobs "
                        f"(skipped {page_expired_count} expired)"
                    )

            if len(items) < page_size:
                print(
                    f"\nReached last page (API returned {len(items)} items, expected {page_size})"
                )
                break

            if state.total_blobs and len(state.all_blob_ids) >= state.total_blobs:
                print(f"\nFetched all {state.total_blobs} blobs")
                break

            page += 1

            if page % 10 == 0:
                time.sleep(0.5)

            if page > 200000:
                print(f"\nSafety limit reached at page {page}")
                break

        except requests.exceptions.RequestException as e:
            if state.pbar:
                state.pbar.close()
            print(f"\nRequest error: {e}")
            break
        except Exception as e:
            if state.pbar:
                state.pbar.close()
            print(f"\nError: {e}")
            import traceback

            traceback.print_exc()
            break

    if state.pbar:
        state.pbar.close()

    def sort_active_key(blob_id: str) -> int:
        metadata = state.blob_metadata.get(blob_id, {})
        end_epoch = metadata.get("end_epoch")
        try:
            return -int(end_epoch) if end_epoch else 0
        except (ValueError, TypeError):
            return 0

    def sort_all_key(blob_id: str) -> Tuple[bool, int]:
        is_active = blob_id in state.active_blob_ids
        metadata = state.blob_metadata.get(blob_id, {})
        end_epoch = metadata.get("end_epoch")
        try:
            end_epoch_val = -int(end_epoch) if end_epoch else 0
        except (ValueError, TypeError):
            end_epoch_val = 0
        return (not is_active, end_epoch_val)

    all_unique_blob_ids = sorted(list(state.all_blob_ids), key=sort_all_key)
    active_unique_blob_ids = sorted(list(state.active_blob_ids), key=sort_active_key)

    print("\n\n=== FINAL SUMMARY ===")
    print(f"Current Walrus epoch (manual): {current_epoch}")
    print(f"Total blob IDs found: {len(all_unique_blob_ids)}")
    print(f"Active blob IDs: {len(active_unique_blob_ids)}")
    print(f"Expired blob IDs: {len(all_unique_blob_ids) - len(active_unique_blob_ids)}")

    # Persist outputs (kept for compatibility with original script)
    if all_unique_blob_ids:
        all_output_file = "blob_ids.txt"
        with open(all_output_file, "w") as f:
            for blob_id in all_unique_blob_ids:
                f.write(f"{blob_id}\n")

        active_output_file = "blob_ids_active.txt"
        with open(active_output_file, "w") as f:
            for blob_id in active_unique_blob_ids:
                f.write(f"{blob_id}\n")

        json_file = "blob_ids.json"
        with open(json_file, "w") as f:
            json.dump(
                {
                    "current_epoch": current_epoch,
                    "timestamp": datetime.now().isoformat(),
                    "all_blobs": {
                        "blob_ids": all_unique_blob_ids,
                        "total": len(all_unique_blob_ids),
                    },
                    "active_blobs": {
                        "blob_ids": active_unique_blob_ids,
                        "total": len(active_unique_blob_ids),
                    },
                    "expired_count": len(all_unique_blob_ids)
                    - len(active_unique_blob_ids),
                    "metadata": state.blob_metadata,
                    "source": "Blockberry API (api.blockberry.one)",
                    "api_url": BLOCKBERRY_API_URL,
                },
                f,
                indent=2,
            )

        print("\nFiles saved:")
        print(f"  - {all_output_file}: All {len(all_unique_blob_ids)} blob IDs")
        print(
            f"  - {active_output_file}: {len(active_unique_blob_ids)} active blob IDs"
        )
        print(f"  - {json_file}: Complete data with metadata")

    return all_unique_blob_ids, active_unique_blob_ids, state.blob_metadata


def iter_active_blobs(
    current_epoch: Optional[int] = None,
    start_page: int = 0,
    *,
    sort_by: str = "SIZE",
    order_by: str = "DESC",
) -> Iterable[Tuple[int, str, Dict[str, Any]]]:
    """Streaming generator that yields (page, blob_id, metadata) for active blobs.

    This reuses much of the paging logic from `get_all_blob_ids`, but instead of
    materializing everything, it yields each active blob as soon as it's seen.

    Default ``sort_by=SIZE`` / ``order_by=DESC`` matches Blockberry largest-first ordering
    (better for spotting large media). Each page is also re-sorted client-side by size so
    null/unknown sizes do not jump ahead of large blobs.

    Args:
        current_epoch: Walrus epoch for filtering active blobs. If None, resolved via ``get_current_epoch()``.
        start_page: Page number to start from (for resuming indexing).
        sort_by: Blockberry ``sortBy`` param (e.g. ``SIZE``, ``TIMESTAMP``).
        order_by: Blockberry ``orderBy`` param (``ASC`` or ``DESC``).

    Yields:
        (page_number, blob_id, metadata) tuples
    """
    if current_epoch is None:
        current_epoch = get_current_epoch()

    print(f"Streaming active Walrus blobs (epoch > {current_epoch}) from Blockberry...")
    if not blockberry_api_key():
        print(
            "\nBlockberry: BLOCKBERRY_API_KEY is not set. Export it or add it to .env — "
            "the API often returns an empty or non-JSON body without a valid key.\n",
            file=sys.stderr,
        )
        return
    if start_page > 0:
        print(f"Resuming from page {start_page}...")

    page = start_page
    page_size = DEFAULT_PAGE_SIZE
    headers = get_headers()

    total_blobs: Optional[int] = None
    pbar: Optional[tqdm] = None

    while True:
        try:
            params = {
                "page": page,
                "size": page_size,
                "orderBy": order_by,
                "sortBy": sort_by,
            }

            if pbar is not None:
                pbar.set_description(f"Page {page}")
            else:
                print(f"Streaming page {page}...", end="\r")

            response = requests.get(
                BLOCKBERRY_API_URL,
                params=params,
                headers=headers,
                timeout=DEFAULT_TIMEOUT,
            )

            if response.status_code == 429:
                print(f"\nRate limited (HTTP 429) at page {page}, waiting 5 seconds...")
                time.sleep(5)
                continue
            elif response.status_code != 200:
                print(f"\nError: HTTP {response.status_code}")
                print(f"Response: {response.text[:200]}")
                if response.status_code == 400:
                    break
                print("Waiting 2 seconds before retry...")
                time.sleep(2)
                continue

            data, jerr = _parse_json_body(response)
            if jerr:
                print(f"\nBlockberry: {jerr}", file=sys.stderr)
                if not blockberry_api_key():
                    print(
                        "  Hint: set BLOCKBERRY_API_KEY (empty or HTML bodies often mean missing key).",
                        file=sys.stderr,
                    )
                break

            items = None
            if isinstance(data, dict):
                items = (
                    data.get("data")
                    or data.get("items")
                    or data.get("blobs")
                    or data.get("content")
                )
                if isinstance(data, dict) and "list" in data:
                    items = data["list"]

                if total_blobs is None:
                    total_blobs = (
                        data.get("total")
                        or data.get("totalElements")
                        or data.get("count")
                    )
                    if total_blobs:
                        total_blobs_int = int(total_blobs)
                        # API "total" is listing rows, not "still active" for current_epoch; most
                        # rows per page are skipped below. Progress counts only yielded active blobs.
                        print(
                            f"\nBlockberry ~{total_blobs_int:,} listing rows (includes inactive); "
                            f"streaming active only (end_epoch > current epoch)."
                        )
                        pbar = tqdm(
                            total=None,
                            desc="Active blobs",
                            unit="blob",
                            unit_scale=True,
                            bar_format="{l_bar}{bar}| {n_fmt} [{elapsed}, {rate_fmt}]",
                        )
            elif isinstance(data, list):
                items = data

            if not items:
                print(f"\nNo more items found at page {page}")
                break

            if sort_by.upper() == "SIZE":
                items = _blob_list_sort_by_size(
                    items, descending=order_by.upper() == "DESC"
                )

            emitted = 0

            for item in items:
                end_epoch = (
                    item.get("endEpoch")
                    or item.get("end_epoch")
                    or item.get("expiresAtEpoch")
                )
                # Strict: missing or unparsable end_epoch is not treated as active (avoids stale listings).
                if end_epoch is None:
                    continue
                try:
                    end_epoch_int = int(end_epoch)
                except (ValueError, TypeError):
                    continue
                if end_epoch_int <= current_epoch:
                    continue

                # Prefer the canonical Walrus base64 blob identifier when available.
                blob_id = (
                    item.get("blobIdBase64")
                    or item.get("blob_id_base64")
                    or item.get("blobId")
                    or item.get("blob_id")
                    or item.get("id")
                    or item.get("identifier")
                    or item.get("hash")
                )
                if not blob_id:
                    continue

                size = item.get("size") or item.get("fileSize")
                owner = item.get("owner") or item.get("ownerAddress")
                deletable = item.get("deletable") or item.get("isDeletable")

                metadata = {
                    "end_epoch": end_epoch,
                    "size": size,
                    "owner": owner,
                    "deletable": deletable,
                    "timestamp": item.get("timestamp") or item.get("createdAt"),
                }

                emitted += 1
                # Log active blob discovery
                if pbar is not None:
                    pbar.set_postfix_str(
                        f"active +{emitted} this page / {len(items)} rows",
                        refresh=False,
                    )
                yield page, blob_id, metadata

            if pbar is not None:
                pbar.update(emitted)

            if len(items) < page_size:
                print(
                    f"\nReached last page while streaming (API returned {len(items)} items, expected {page_size})"
                )
                break

            page += 1
            if page % 10 == 0:
                time.sleep(0.5)

            if page > 200000:
                print(f"\nSafety limit reached at page {page}")
                break

        except requests.exceptions.RequestException as e:
            if pbar is not None:
                pbar.close()
            print(f"\nRequest error while streaming: {e}")
            break
        except Exception as e:
            if pbar is not None:
                pbar.close()
            print(f"\nError while streaming: {e}")
            import traceback

            traceback.print_exc()
            break

    if pbar is not None:
        pbar.close()


__all__ = [
    "WalrusEpochError",
    "blockberry_api_key",
    "get_blob_details_by_id",
    "get_current_epoch",
    "get_all_blob_ids",
    "get_walrus_epoch_from_sui_rpc",
    "is_blob_expired_for_epoch",
    "iter_active_blobs",
]
