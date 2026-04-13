"""Remove Omura runtime state: vector index, temp blobs, SQLite DB, lock files.

Usage:
    uv run python scripts/clean_state.py             # dry-run (safe default)
    uv run python scripts/clean_state.py --yes       # actually delete
    uv run python scripts/clean_state.py --yes --index   # index only
    uv run python scripts/clean_state.py --yes --temp    # temp blobs only
    uv run python scripts/clean_state.py --yes --db      # sqlite db only
    uv run python scripts/clean_state.py --yes --locks   # /tmp lock files only

Without --index/--temp/--db/--locks the script cleans everything.
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _remove(path: Path, dry: bool) -> None:
    if not path.exists():
        return
    tag = "[dry-run]" if dry else "[delete]"
    if path.is_dir():
        print(f"  {tag} {path}/")
        if not dry:
            shutil.rmtree(path)
    else:
        print(f"  {tag} {path}")
        if not dry:
            path.unlink()


def clean_index(dry: bool) -> None:
    """Remove the persisted vector index (embeddings, FAISS/cuVS, metadata, cursor, backups)."""
    print("\n-- Vector index --")
    base = Path(os.getenv("OMURA_VECTOR_STORE_DIR", str(REPO_ROOT / "data" / "vector_index")))
    version = os.getenv("OMURA_VECTOR_INDEX_VERSION", "omni").strip()
    store_dir = (base.parent / f"{base.name}_{version}") if version else base

    # Remove all known files inside the store dir but keep the dir itself
    for name in (
        "vector_index.faiss",
        "vector_index.cagra",
        "embeddings.npy",
        "blob_id_mapping.npy",
        "metadata.json",
        "cursor.json",
        "indexer_stats.json",
    ):
        _remove(store_dir / name, dry)

    backups = store_dir / "backups"
    if backups.is_dir():
        for entry in sorted(backups.iterdir()):
            _remove(entry, dry)

    # Also clean unversioned store if it differs
    if store_dir != base and base.exists():
        for name in ("vector_index.faiss", "embeddings.npy", "blob_id_mapping.npy",
                     "metadata.json", "cursor.json"):
            _remove(base / name, dry)


def clean_temp(dry: bool) -> None:
    """Remove temporary blob files written during video/audio embedding."""
    print("\n-- Temp blobs --")
    temp_dir = Path(os.getenv("OMURA_TEMP_BLOB_DIR", str(REPO_ROOT / "data" / "temp_blobs")))
    if not temp_dir.exists():
        print("  (nothing)")
        return
    for sub in temp_dir.iterdir():
        _remove(sub, dry)


def clean_db(dry: bool) -> None:
    """Remove the Walrus SQLite blob database and WAL side-car files."""
    print("\n-- SQLite DB --")
    for pattern in ("walrus_blobs.sqlite", "walrus_blobs.sqlite-shm", "walrus_blobs.sqlite-wal"):
        for hit in glob.glob(str(REPO_ROOT / pattern)):
            _remove(Path(hit), dry)
    # blob_ids.json (legacy snapshot)
    _remove(REPO_ROOT / "blob_ids.json", dry)


def clean_locks(dry: bool) -> None:
    """Remove Omura fcntl lock files from the system temp directory."""
    print("\n-- Lock files --")
    tmp = Path(tempfile.gettempdir())
    found = list(tmp.glob("omura_*.lock"))
    if not found:
        print("  (nothing)")
    for f in found:
        _remove(f, dry)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--yes", action="store_true", help="Actually delete (default is dry-run)")
    ap.add_argument("--index", action="store_true", help="Clean vector index only")
    ap.add_argument("--temp", action="store_true", help="Clean temp blobs only")
    ap.add_argument("--db", action="store_true", help="Clean SQLite DB only")
    ap.add_argument("--locks", action="store_true", help="Clean lock files only")
    args = ap.parse_args()

    dry = not args.yes
    selective = args.index or args.temp or args.db or args.locks

    if dry:
        print("DRY-RUN mode — pass --yes to actually delete")

    if not selective or args.index:
        clean_index(dry)
    if not selective or args.temp:
        clean_temp(dry)
    if not selective or args.db:
        clean_db(dry)
    if not selective or args.locks:
        clean_locks(dry)

    print()
    if dry:
        print("Nothing deleted. Re-run with --yes to apply.")
    else:
        print("Done.")


if __name__ == "__main__":
    main()
