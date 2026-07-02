#!/usr/bin/env python3
"""Stream active Walrus blob ids (HTTP aggregator form) using configured discovery backend.

Examples:
  uv run python scripts/list_active_blob_ids.py | head
  uv run python scripts/list_active_blob_ids.py --discovery blockberry --max 100
  # One JSON object with the first active blob id (good for smoke tests):
  uv run python scripts/list_active_blob_ids.py --discovery graphql --first --json
  uv run python scripts/list_active_blob_ids.py --discovery graphql --max 20 --with-size
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from omura.utils.blob_discovery import get_blob_discovery_source, iter_active_blob_entries
from omura.utils.blockberry import get_current_epoch


def main() -> int:
    p = argparse.ArgumentParser(description="Print active Walrus blob ids (one per line).")
    p.add_argument(
        "--discovery",
        choices=("blockberry", "graphql", "sui_owned"),
        default=None,
        help="Overrides OMURA_BLOB_DISCOVERY.",
    )
    p.add_argument("--epoch", type=int, default=None, help="Walrus epoch for active filter (default: auto).")
    p.add_argument("--max", type=int, default=None, help="Stop after N blob ids.")
    p.add_argument(
        "--first",
        action="store_true",
        help="Only return the first active blob (useful with --json).",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Print JSON to stdout (with --first: one object with blob_id; with --max: blob_ids array).",
    )
    p.add_argument(
        "--with-size",
        action="store_true",
        help="Plain text: tab-separated blob_id and size (bytes, as on-chain string). JSON: include size in output.",
    )
    args = p.parse_args()

    if args.json and not args.first and args.max is None:
        print("--json requires --first (single blob) or --max N (array)", file=sys.stderr)
        return 2

    if args.discovery:
        os.environ["OMURA_BLOB_DISCOVERY"] = args.discovery

    epoch = args.epoch
    if epoch is None:
        epoch = get_current_epoch(silent=True)
    if epoch is None:
        epoch = get_current_epoch()

    if not args.json:
        print(f"# discovery={get_blob_discovery_source()} epoch={epoch}", file=sys.stderr)

    collected: list[str] = []
    metas: list[dict] = []
    n = 0
    limit = 1 if args.first else args.max

    for _page, blob_id, meta in iter_active_blob_entries(current_epoch=epoch):
        if args.json:
            collected.append(blob_id)
            metas.append(meta)
        elif args.with_size:
            sz = meta.get("size")
            print(f"{blob_id}\t{sz if sz is not None else ''}")
        else:
            print(blob_id)
        n += 1
        if args.first:
            break
        if args.max is not None and n >= args.max:
            break

    if args.json:
        src = get_blob_discovery_source()
        if args.first:
            out = {
                "ok": bool(collected),
                "discovery": src,
                "epoch": epoch,
                "blob_id": collected[0] if collected else None,
                "metadata": metas[0] if metas else None,
            }
            if metas:
                m0 = metas[0]
                out["size"] = m0.get("size")
            if not collected:
                out["error"] = "no_active_blob_found"
            print(json.dumps(out, indent=2))
            return 0 if collected else 1

        items = []
        for bid, m in zip(collected, metas):
            items.append(
                {
                    "blob_id": bid,
                    "size": m.get("size"),
                    "end_epoch": m.get("end_epoch"),
                    "sui_object_id": m.get("sui_object_id"),
                }
            )

        payload = {
            "ok": True,
            "discovery": src,
            "epoch": epoch,
            "count": len(collected),
            "blob_ids": collected,
            "items": items,
        }
        print(json.dumps(payload, indent=2))
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
