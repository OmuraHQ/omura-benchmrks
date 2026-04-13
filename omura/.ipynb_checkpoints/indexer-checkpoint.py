"""Minimal indexer service that fetches Walrus blobs and prints detected types.

This is NOT the full Omura architecture, just a small, testable starting point:
1. Load a list of blob IDs (from blob_ids_active.txt or blob_ids.json)
2. Fetch blob content via walrus-python
3. Detect file type via magic bytes
4. Print results
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, List

from omura.parsers.file_detection import detect_file_type
from omura.walrus_client import SimpleWalrusClient


DATA_DIR = Path(os.getenv("OMURA_DATA_DIR", "."))  # default to cwd


def _load_blob_ids_from_txt(path: Path, limit: int | None = 20) -> List[str]:
    if not path.is_file():
        return []
    ids: List[str] = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ids.append(line)
            if limit is not None and len(ids) >= limit:
                break
    return ids


def _load_blob_ids_from_json(path: Path, limit: int | None = 20) -> List[str]:
    if not path.is_file():
        return []
    with path.open("r") as f:
        data = json.load(f)
    # Prefer active blobs if available
    active_ids = (
        data.get("active_blobs", {})
        .get("blob_ids", [])
    )
    all_ids = data.get("all_blobs", {}).get("blob_ids", [])
    ids: List[str] = active_ids or all_ids or []
    if limit is not None:
        ids = ids[:limit]
    return ids


def load_sample_blob_ids(limit: int = 20) -> List[str]:
    """Load a small sample of blob IDs from existing outputs."""
    txt_path = DATA_DIR / "blob_ids_active.txt"
    json_path = DATA_DIR / "blob_ids.json"

    ids = _load_blob_ids_from_txt(txt_path, limit=limit)
    if ids:
        return ids

    ids = _load_blob_ids_from_json(json_path, limit=limit)
    if ids:
        return ids

    raise FileNotFoundError(
        "No blob_ids_active.txt or blob_ids.json found. "
        "Run get_blob_ids.py first to generate blob IDs."
    )


def inspect_blobs(blob_ids: Iterable[str]) -> None:
    """Fetch each blob and print detected file type."""
    client = SimpleWalrusClient()

    for idx, blob_id in enumerate(blob_ids, start=1):
        print(f"\n[{idx}] Blob ID: {blob_id}")
        content = client.fetch_blob(blob_id)
        if content is None:
            print("  - Status: NOT FOUND or error from Walrus")
            continue

        mime, ext, kind = detect_file_type(content)
        size = len(content)
        print(f"  - Size: {size} bytes")
        print(f"  - Detected kind: {kind}")
        print(f"  - MIME type: {mime}")
        print(f"  - Suggested extension: .{ext}" if ext else "  - Suggested extension: (none)")


def main() -> None:
    """Entry point: load sample IDs and print their file types."""
    try:
        blob_ids = load_sample_blob_ids(limit=20)
    except FileNotFoundError as e:
        print(str(e))
        return

    print(f"Loaded {len(blob_ids)} blob IDs for inspection.")
    inspect_blobs(blob_ids)


if __name__ == "__main__":
    main()

