"""Prune expired Walrus blobs from persisted vector store files.

Usage:
  uv run python scripts/prune_expired_blobs.py
  uv run python scripts/prune_expired_blobs.py --epoch 22
  uv run python scripts/prune_expired_blobs.py --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from omura.utils.blockberry import get_current_epoch
from omura.utils.vector_store import VECTOR_STORE_DIR, VectorStore


def _resolve_epoch(explicit_epoch: int | None) -> int:
    if explicit_epoch is not None:
        return explicit_epoch

    return get_current_epoch()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prune expired blobs from vector index metadata/embeddings."
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=None,
        help="Current Walrus epoch. If omitted, auto-detect via Blockberry.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be pruned without writing changes.",
    )
    parser.add_argument(
        "--store-dir",
        type=str,
        default=None,
        help=f"Vector store directory (default: {VECTOR_STORE_DIR}).",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Disable backup creation before saving.",
    )
    args = parser.parse_args()

    if args.store_dir:
        store_dir = Path(args.store_dir).expanduser()
        os.environ["OMURA_VECTOR_STORE_DIR"] = str(store_dir)
        print(f"Using custom vector store dir: {store_dir}")

    current_epoch = _resolve_epoch(args.epoch)
    print(f"Pruning with current_epoch={current_epoch}")

    store = VectorStore()
    store.load()

    if store.size() == 0:
        print("Vector store is empty. Nothing to prune.")
        return 0

    summary = store.prune_expired(current_epoch=current_epoch, dry_run=args.dry_run)
    print(
        "Prune summary: "
        f"before={summary['total_before']}, "
        f"expired={summary['expired_count']}, "
        f"kept={summary['kept_count']}"
    )

    if args.dry_run:
        print("Dry run mode: no files were modified.")
        return 0

    if summary["expired_count"] == 0:
        print("No expired blobs found. No changes written.")
        return 0

    store.save(create_backup=not args.no_backup)
    print("Expired blobs pruned and vector store saved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
