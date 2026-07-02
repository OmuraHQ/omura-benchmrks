#!/usr/bin/env python3
"""
Bulk Image Downloader for Omura

Fetches images cataloged in blob_catalog.sqlite, downloads them from the Walrus 
aggregator pool (or fallbacks), and saves them to a local directory.
"""

import os
import sqlite3
import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import requests
from tqdm import tqdm

DEFAULT_AGGREGATOR = "https://agrregator.omura.fun"

def fetch_image(blob_id: str, aggregator: str, output_dir: Path, timeout: int = 30) -> bool:
    """Download a single image blob and save it with its correct extension if possible."""
    url = f"{aggregator}/v1/blobs/{blob_id}"
    try:
        resp = requests.get(url, timeout=timeout)
        if resp.status_code != 200:
            return False
        
        # Sniff content type or fall back to .bin
        content_type = resp.headers.get("Content-Type", "").lower()
        ext = ".bin"
        if "image/png" in content_type:
            ext = ".png"
        elif "image/jpeg" in content_type or "image/jpg" in content_type:
            ext = ".jpg"
        elif "image/webp" in content_type:
            ext = ".webp"
        elif "image/gif" in content_type:
            ext = ".gif"
            
        out_path = output_dir / f"{blob_id}{ext}"
        out_path.write_bytes(resp.content)
        return True
    except Exception:
        return False

def main():
    parser = argparse.ArgumentParser(description="Bulk download images indexed in the catalog database.")
    parser.add_argument("--db", default="data/blob_catalog.sqlite", help="Path to SQLite catalog database")
    parser.add_argument("--out", default="downloaded_images", help="Output directory to save images")
    parser.add_argument("--aggregator", default=DEFAULT_AGGREGATOR, help="Walrus aggregator URL")
    parser.add_argument("--workers", type=int, default=16, help="Number of concurrent download workers")
    parser.add_argument("--limit", type=int, default=None, help="Limit total downloads")
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"Error: Catalog database not found at {db_path}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading images from {db_path}...")
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    
    # Query all blobs that are images
    query = "SELECT blob_id FROM blobs WHERE kind = 'image' OR mime_type LIKE 'image/%'"
    if args.limit:
        query += f" LIMIT {args.limit}"
        
    cur.execute(query)
    blob_ids = [r[0] for r in cur.fetchall()]
    conn.close()

    total = len(blob_ids)
    if total == 0:
        print("No image blobs found in database.")
        return

    print(f"Starting bulk download of {total} images into '{out_dir}' with {args.workers} workers...")
    
    success = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(fetch_image, bid, args.aggregator.rstrip("/"), out_dir): bid 
            for bid in blob_ids
        }
        
        with tqdm(total=total, desc="Downloading") as pbar:
            for fut in as_completed(futures):
                bid = futures[fut]
                try:
                    res = fut.result()
                    if res:
                        success += 1
                    else:
                        failed += 1
                except Exception:
                    failed += 1
                pbar.update(1)

    print("\n--- Summary ---")
    print(f"Successfully downloaded: {success}")
    print(f"Failed downloads:        {failed}")
    print(f"Saved directory:         {out_dir.absolute()}")

if __name__ == "__main__":
    main()
