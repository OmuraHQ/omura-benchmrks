"""Discover active Walrus blobs without relying on Blockberry.

Supported sources:

- **graphql** (default when ``OMURA_BLOB_DISCOVERY`` is unset): Sui GraphQL
  ``objects(filter: { type: Walrus blob::Blob })`` on ``SUI_GRAPHQL_URL`` (default
  ``https://graphql.mainnet.sui.io/graphql``). Public Sui ``ObjectFilter`` only supports
  ``type``, ``owner``, ``ownerKind``, ``objectIds`` — **not** Move fields like
  ``storage.end_epoch``. Inactive / expired blobs **cannot** be excluded in the GraphQL
  filter; Omura drops them in :func:`_graphql_node_to_active_blob` (``end_epoch > epoch``).
  No API key. Can be **very large** (full chain crawl). For raw ``curl`` output, pipe
  through ``scripts/filter_graphql_blobs_active.py --epoch N``.

- **blockberry** (``OMURA_BLOB_DISCOVERY=blockberry``): existing
  :func:`omura.utils.blockberry.iter_active_blobs` — fast, sorted pages; requires
  ``BLOCKBERRY_API_KEY``. Listings can disagree with chain state; use GraphQL when you need
  **on-chain** ``end_epoch`` only.

- **sui_owned**: ``suix_getOwnedObjects`` for one address — only blobs **owned** by that
  wallet; set ``SUI_BLOB_OWNER_ADDRESS``.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any, Dict, Iterable, Iterator, Optional, Tuple

import requests

from omura.utils.blockberry import iter_active_blobs as _iter_blockberry
from omura.utils.blockberry import get_current_epoch
from omura.utils.walrus_ids import uint256_blob_id_to_aggregator_string

DEFAULT_GRAPHQL_URL = "https://graphql.mainnet.sui.io/graphql"
DEFAULT_WALRUS_BLOB_TYPE = (
    "0xfdc88f7d7cf30afab2f82e8380d11ee8f70efb90e863d1de8616fae1bb09ea77::blob::Blob"
)


def _graphql_page_size() -> int:
    raw = (os.getenv("OMURA_GRAPHQL_PAGE_SIZE") or "50").strip()
    try:
        n = int(raw)
    except ValueError:
        n = 50
    return max(1, min(200, n))


def _graphql_sleep_sec() -> float:
    raw = (os.getenv("OMURA_GRAPHQL_SLEEP_SEC") or "0.08").strip()
    try:
        s = float(raw)
    except ValueError:
        s = 0.08
    return max(0.0, s)


# Sui ObjectFilter has no predicate on Move contents (see docs: ObjectFilter).
# "Inactive" Walrus blobs are skipped in _graphql_node_to_active_blob, not in this query.
_OBJECTS_QUERY = """
# Walrus Blob objects by Move type only. Expired blobs are still returned by the server.
query WalrusBlobObjectsByType($first: Int!, $after: String, $type: String!) {
  objects(first: $first, after: $after, filter: { type: $type }) {
    nodes {
      address
      version
      asMoveObject {
        contents {
          json
        }
      }
    }
    pageInfo {
      hasNextPage
      endCursor
    }
  }
}
"""


def _discovery_source() -> str:
    s = (os.getenv("OMURA_BLOB_DISCOVERY") or "blockberry").strip().lower()
    if s in ("graphql", "gql", "sui_graphql"):
        return "graphql"
    if s in ("blockberry", "berry", "api"):
        return "blockberry"
    if s in ("sui_owned", "owned", "get_owned"):
        return "sui_owned"
    return "blockberry"


def get_blob_discovery_source() -> str:
    """Current backend from ``OMURA_BLOB_DISCOVERY`` (``blockberry`` | ``graphql`` | ``sui_owned``)."""
    return _discovery_source()


def _rpc_url() -> str:
    u = (
        (os.getenv("SUI_RPC_URL") or "https://fullnode.mainnet.sui.io:443")
        .strip()
        .rstrip("/")
    )
    return u if u.startswith("http") else f"https://{u}"


def _graphql_url() -> str:
    return (os.getenv("SUI_GRAPHQL_URL") or DEFAULT_GRAPHQL_URL).strip().rstrip("/")


def _walrus_blob_type() -> str:
    return (os.getenv("WALRUS_BLOB_MOVE_TYPE") or DEFAULT_WALRUS_BLOB_TYPE).strip()


def _graphql_node_to_active_blob(
    node: Any, current_epoch: int
) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Return ``(aggregator_blob_id, meta)`` if this node is an **active** Walrus ``Blob``; else None.

    Public Sui GraphQL has **no** ``filter`` field for ``storage.end_epoch``; dropping
    expired / inactive blobs happens here (``end_epoch`` present and ``> current_epoch``).
    """
    mov = (node or {}).get("asMoveObject") or {}
    contents = (mov.get("contents") or {}).get("json")
    if not isinstance(contents, dict):
        return None
    raw_blob_id = contents.get("blob_id")
    if raw_blob_id is None:
        return None
    storage = contents.get("storage") or {}
    if isinstance(storage, dict):
        storage = storage.get("fields") or storage
    end_epoch = None
    if isinstance(storage, dict):
        end_epoch = storage.get("end_epoch")
    try:
        end_i = int(end_epoch) if end_epoch is not None else None
    except (TypeError, ValueError):
        end_i = None
    if end_i is None or end_i <= current_epoch:
        return None

    try:
        agg = uint256_blob_id_to_aggregator_string(raw_blob_id)
    except (TypeError, ValueError, OverflowError):
        return None

    meta: Dict[str, Any] = {
        "end_epoch": end_epoch,
        "size": contents.get("size"),
        "deletable": contents.get("deletable"),
        "sui_object_id": contents.get("id") or (node or {}).get("address"),
        "registered_epoch": contents.get("registered_epoch"),
        "object_version": (node or {}).get("version"),
        "source": "sui_graphql",
    }
    return agg, meta


def iter_active_blobs_graphql(
    current_epoch: int,
    *,
    page_size: Optional[int] = None,
    sleep_sec: Optional[float] = None,
    max_pages: Optional[int] = None,
) -> Iterator[Tuple[int, str, Dict[str, Any]]]:
    """Yield ``(page_index, aggregator_blob_id, metadata)`` from Sui GraphQL blob objects.

    Only yields blobs whose on-chain ``storage.end_epoch`` is present and strictly greater
    than ``current_epoch``. Rows without a parseable ``end_epoch`` are **skipped** (strict
    active rule; avoids treating missing expiry as active).
    """
    url = _graphql_url()
    type_str = _walrus_blob_type()
    ps = page_size if page_size is not None else _graphql_page_size()
    sl = sleep_sec if sleep_sec is not None else _graphql_sleep_sec()
    cursor: Optional[str] = None
    page_idx = 0
    pages_done = 0
    total_yielded = 0

    while True:
        if max_pages is not None and pages_done >= max_pages:
            break
        payload = {
            "query": _OBJECTS_QUERY,
            "variables": {"first": ps, "after": cursor, "type": type_str},
        }
        try:
            resp = requests.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=120,
            )
            if resp.status_code != 200:
                print(
                    f"\nGraphQL HTTP {resp.status_code}: {resp.text[:300]}",
                    file=sys.stderr,
                )
                break
            body = resp.json()
        except requests.RequestException as e:
            print(f"\nGraphQL request failed: {e}", file=sys.stderr)
            break

        if body.get("errors"):
            print(f"\nGraphQL errors: {body['errors'][:3]}", file=sys.stderr)
            break

        data = (body.get("data") or {}).get("objects") or {}
        nodes = data.get("nodes") or []
        pinfo = data.get("pageInfo") or {}
        active_this_page = 0

        for node in nodes:
            out = _graphql_node_to_active_blob(node, current_epoch)
            if out is None:
                continue
            agg, meta = out
            active_this_page += 1
            total_yielded += 1
            yield page_idx, agg, meta

        print(
            f"  GraphQL page {page_idx}: objects={len(nodes)} active_this_page={active_this_page} "
            f"active_total={total_yielded}",
            file=sys.stderr,
            flush=True,
        )

        pages_done += 1
        page_idx += 1
        if not pinfo.get("hasNextPage"):
            break
        cursor = pinfo.get("endCursor")
        if not cursor:
            break
        if sl > 0:
            time.sleep(sl)


def iter_active_blobs_sui_owned(
    owner_address: str,
    current_epoch: int,
    *,
    page_size: int = 50,
    sleep_sec: float = 0.1,
) -> Iterator[Tuple[int, str, Dict[str, Any]]]:
    """Yield active blobs for **one** Sui address via ``suix_getOwnedObjects`` (Walrus ``Blob``)."""
    rpc = _rpc_url()
    struct_type = _walrus_blob_type()
    cursor = None
    page_idx = 0

    while True:
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "suix_getOwnedObjects",
            "params": [
                owner_address.strip(),
                {
                    "filter": {"StructType": struct_type},
                    "options": {"showContent": True},
                },
                cursor,
                page_size,
            ],
        }
        try:
            resp = requests.post(
                rpc,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=60,
            )
            body = resp.json()
        except requests.RequestException as e:
            print(f"\nsuix_getOwnedObjects failed: {e}", file=sys.stderr)
            break
        except ValueError as e:
            print(f"\nsuix_getOwnedObjects response not JSON: {e}", file=sys.stderr)
            break

        if body.get("error"):
            print(f"\nRPC error: {body['error']}", file=sys.stderr)
            break

        result = body.get("result") or {}
        items = result.get("data") or []
        for item in items:
            data = (item or {}).get("data") or {}
            content = data.get("content") or {}
            fields = content.get("fields") or {}
            raw_blob_id = fields.get("blob_id")
            if raw_blob_id is None:
                continue
            storage = fields.get("storage") or {}
            sf = storage.get("fields") if isinstance(storage, dict) else {}
            if not isinstance(sf, dict):
                sf = {}
            end_epoch = sf.get("end_epoch")
            try:
                end_i = int(end_epoch) if end_epoch is not None else None
            except (TypeError, ValueError):
                end_i = None
            if end_i is None or end_i <= current_epoch:
                continue
            try:
                agg = uint256_blob_id_to_aggregator_string(raw_blob_id)
            except (TypeError, ValueError, OverflowError):
                continue
            meta = {
                "end_epoch": end_epoch,
                "size": fields.get("size"),
                "deletable": fields.get("deletable"),
                "sui_object_id": data.get("objectId"),
                "source": "sui_owned",
            }
            yield page_idx, agg, meta

        if not result.get("hasNextPage"):
            break
        cursor = result.get("nextCursor")
        if not cursor:
            break
        page_idx += 1
        if sleep_sec > 0:
            time.sleep(sleep_sec)


def iter_active_blob_entries(
    current_epoch: Optional[int] = None,
    start_page: int = 0,
    *,
    sort_by: str = "SIZE",
    order_by: str = "DESC",
    source: Optional[str] = None,
    graphql_page_size: Optional[int] = None,
) -> Iterable[Tuple[int, str, Dict[str, Any]]]:
    """Unified iterator matching :func:`iter_active_blobs` signature (page, id, metadata).

    Source is ``OMURA_BLOB_DISCOVERY`` (``blockberry`` | ``graphql`` | ``sui_owned``) or
    the ``source`` argument.
    """
    src = (source or _discovery_source()).strip().lower()
    if src in ("graphql", "gql", "sui_graphql"):
        src = "graphql"
    elif src in ("sui_owned", "owned"):
        src = "sui_owned"
    else:
        src = "blockberry"

    if current_epoch is None:
        from omura.utils.blockberry import MANUAL_EPOCH

        current_epoch = get_current_epoch(silent=True)
        if current_epoch is None:
            current_epoch = MANUAL_EPOCH

    if src == "blockberry":
        yield from _iter_blockberry(
            current_epoch=current_epoch,
            start_page=start_page,
            sort_by=sort_by,
            order_by=order_by,
        )
        return

    if src == "graphql":
        if start_page > 0:
            print(
                "GraphQL discovery does not resume Blockberry page numbers; "
                "ignoring start_page>0 (consider clearing vector cursor for a clean run).",
                file=sys.stderr,
            )
        print(
            f"Streaming active Walrus blobs via Sui GraphQL (end_epoch > {current_epoch})…",
            file=sys.stderr,
        )
        print(
            f"  GraphQL: {_graphql_url()}  page_size={graphql_page_size or _graphql_page_size()}",
            file=sys.stderr,
        )
        yield from iter_active_blobs_graphql(current_epoch, page_size=graphql_page_size)
        return

    if src == "sui_owned":
        owner = (os.getenv("SUI_BLOB_OWNER_ADDRESS") or "").strip()
        if not owner:
            print(
                "OMURA_BLOB_DISCOVERY=sui_owned requires SUI_BLOB_OWNER_ADDRESS",
                file=sys.stderr,
            )
            return
        print(
            f"Streaming Walrus blobs for owner {owner[:16]}… (end_epoch > {current_epoch})",
            file=sys.stderr,
        )
        yield from iter_active_blobs_sui_owned(owner, current_epoch)
        return


def iter_all_blobs_blockberry(
    start_page: int = 0,
    *,
    sort_by: str = "SIZE",
    order_by: str = "DESC",
) -> Iterable[Tuple[int, str, Dict[str, Any], bool]]:
    """Stream ALL Walrus blobs (active + expired) from Blockberry.

    Returns:
        (page, blob_id, metadata, is_active) tuples
    """
    from omura.utils.blockberry import (
        MANUAL_EPOCH,
        get_headers,
        DEFAULT_PAGE_SIZE,
        _parse_json_body,
        _blob_list_sort_by_size,
    )
    from tqdm import tqdm

    import time
    import requests

    print(f"Streaming ALL Walrus blobs (active + expired) from Blockberry...")

    page = start_page
    page_size = DEFAULT_PAGE_SIZE
    headers = get_headers()

    total_active = 0
    total_expired = 0
    pbar = tqdm(
        desc="All blobs",
        unit="blob",
        unit_scale=True,
        bar_format="{l_bar}{bar}| {n_fmt} [{elapsed}, {rate_fmt}]",
    )

    while True:
        try:
            params = {
                "page": page,
                "size": page_size,
                "orderBy": order_by,
                "sortBy": sort_by,
            }

            response = requests.get(
                "https://api.blockberry.one/walrus-mainnet/v1/blobs",
                params=params,
                headers=headers,
                timeout=30,
            )

            if response.status_code != 200:
                print(f"Error: HTTP {response.status_code}")
                if response.status_code == 400:
                    break
                time.sleep(2)
                continue

            data, jerr = _parse_json_body(response)
            if jerr:
                print(f"Blockberry: {jerr}")
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
            elif isinstance(data, list):
                items = data

            if not items:
                print(f"No more items at page {page}")
                break

            items = _blob_list_sort_by_size(
                items, descending=order_by.upper() == "DESC"
            )

            for item in items:
                blob_id = (
                    item.get("blobIdBase64")
                    or item.get("blob_id_base64")
                    or item.get("blobId")
                    or item.get("blob_id")
                    or item.get("id")
                )
                if not blob_id:
                    continue

                end_epoch = item.get("endEpoch") or item.get("end_epoch")
                is_active = end_epoch is not None and int(end_epoch) > MANUAL_EPOCH

                if is_active:
                    total_active += 1
                else:
                    total_expired += 1

                metadata = {
                    "end_epoch": end_epoch,
                    "size": item.get("size") or item.get("fileSize"),
                    "owner": item.get("owner") or item.get("ownerAddress"),
                    "deletable": item.get("deletable") or item.get("isDeletable"),
                    "timestamp": item.get("timestamp") or item.get("createdAt"),
                }

                pbar.set_postfix_str(
                    f"active={total_active} expired={total_expired}", refresh=False
                )
                yield page, blob_id, metadata, is_active

            if len(items) < page_size:
                print(f"Reached last page at page {page}")
                pbar.close()
                print(
                    f"Done! Total: {total_active + total_expired} blobs (active={total_active}, expired={total_expired})"
                )
                break

            page += 1
            if page % 10 == 0:
                time.sleep(0.5)

            if page > 50000:
                print(f"Safety limit reached at page {page}")
                pbar.close()
                break

        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}")
            break
        except Exception as e:
            print(f"Error: {e}")
            break


def iter_all_blob_entries(
    start_page: int = 0,
    sort_by: str = "SIZE",
    order_by: str = "DESC",
    source: Optional[str] = None,
) -> Iterable[Tuple[int, str, Dict[str, Any], bool]]:
    """Stream ALL Walrus blobs (active + expired).

    Args:
        start_page: Page number to start from
        sort_by: Sort field (SIZE, TIMESTAMP, etc.)
        order_by: ASC or DESC

    Yields:
        (page, blob_id, metadata, is_active) tuples
    """
    src = (source or os.getenv("OMURA_BLOB_DISCOVERY", "blockberry")).strip().lower()
    if src in ("graphql", "gql", "sui_graphql"):
        src = "graphql"
    elif src in ("sui_owned", "owned"):
        src = "sui_owned"
    else:
        src = "blockberry"

    if src == "blockberry":
        yield from iter_all_blobs_blockberry(
            start_page=start_page,
            sort_by=sort_by,
            order_by=order_by,
        )
        return

    # Default to active-only for other sources
    print(
        f"Warning: Source '{src}' does not support all blobs; using active-only stream"
    )
    for page, blob_id, metadata in iter_active_blob_entries(
        start_page=start_page,
        sort_by=sort_by,
        order_by=order_by,
        source=src,
    ):
        yield page, blob_id, metadata, True


__all__ = [
    "get_blob_discovery_source",
    "iter_active_blob_entries",
    "iter_active_blobs_graphql",
    "iter_active_blobs_sui_owned",
    "iter_all_blob_entries",
    "iter_all_blobs_blockberry",
]
