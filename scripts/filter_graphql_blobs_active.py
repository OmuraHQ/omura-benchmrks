#!/usr/bin/env python3
"""Filter a Sui GraphQL ``objects`` JSON response to **active** Walrus blobs only.

Active rule (same as Omura / Blockberry): ``storage.end_epoch > current_epoch``.

GraphQL cannot express this in the query filter — you still query ``type: ...::blob::Blob``,
then pipe the response through this script.

Examples:
  EPOCH=27 curl -sS ... | uv run python scripts/filter_graphql_blobs_active.py --epoch 27
  uv run python scripts/filter_graphql_blobs_active.py --epoch 27 < response.json
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List


def _end_epoch(node: Dict[str, Any]) -> int | None:
    j = (node.get("asMoveObject") or {}).get("contents") or {}
    js = j.get("json")
    if not isinstance(js, dict):
        return None
    st = js.get("storage")
    if not isinstance(st, dict):
        return None
    ee = st.get("end_epoch")
    if ee is None:
        return None
    try:
        return int(ee)
    except (TypeError, ValueError):
        return None


def main() -> int:
    p = argparse.ArgumentParser(description="Keep only Blob nodes with end_epoch > epoch.")
    p.add_argument(
        "--epoch",
        type=int,
        required=True,
        help="Current Walrus epoch; keep blobs with storage.end_epoch > this value.",
    )
    p.add_argument(
        "--compact",
        action="store_true",
        help="One JSON object per line: address, blob_id (uint), size, end_epoch, sui_object_id.",
    )
    args = p.parse_args()

    data = json.load(sys.stdin)
    objects = (((data.get("data") or {}).get("objects")) or {})
    nodes = objects.get("nodes")
    if not isinstance(nodes, list):
        print("Invalid GraphQL shape: expected data.objects.nodes[]", file=sys.stderr)
        return 2

    active: List[Dict[str, Any]] = []
    for n in nodes:
        if not isinstance(n, dict):
            continue
        ee = _end_epoch(n)
        if ee is None:
            continue
        if ee > args.epoch:
            active.append(n)

    if args.compact:
        for n in active:
            j = (n.get("asMoveObject") or {}).get("contents") or {}
            js = j.get("json") if isinstance(j.get("json"), dict) else {}
            print(
                json.dumps(
                    {
                        "sui_object_id": n.get("address") or js.get("id"),
                        "blob_id_uint256": js.get("blob_id"),
                        "size": js.get("size"),
                        "end_epoch": (js.get("storage") or {}).get("end_epoch")
                        if isinstance(js.get("storage"), dict)
                        else None,
                    }
                )
            )
        return 0

    objects["nodes"] = active
    print(json.dumps(data, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
