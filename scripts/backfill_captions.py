#!/usr/bin/env python3
import os
import sys
import sqlite3
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Add parent directory to path so we can import omura modules
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
load_dotenv()

from omura.utils.captioning import generate_caption
from omura.utils.aggregator_pool import get_pool
from omura.utils.blob_catalog import CATALOG_DB_PATH

BLOB_CACHE_DIR = os.getenv("OMURA_BLOB_CACHE_DIR", "data/blob_cache")

def process_blob(blob_id: str) -> tuple[str, str | None]:
    """Fetch blob bytes (disk-cached — blob content is immutable) and generate a caption."""
    try:
        data, used_url = get_pool().get_blob_cached(f"/v1/blobs/{blob_id}", BLOB_CACHE_DIR, timeout=30)
        if data is None:
            print(f"Fetch failed for {blob_id}: all aggregators exhausted")
            return blob_id, None

        caption = generate_caption(data)
        return blob_id, caption
    except Exception as e:
        print(f"Error fetching/captioning {blob_id}: {e}")
        return blob_id, None

def main():
    parser = argparse.ArgumentParser(description="Backfill captions for cataloged images.")
    parser.add_argument("--db", default="data/blob_catalog.sqlite", help="Path to SQLite catalog database")
    parser.add_argument("--workers", type=int, default=8, help="Number of concurrent downloader workers")
    parser.add_argument("--force", action="store_true", help="Re-caption even if already captioned")
    parser.add_argument("--caption-like", default=None,
                         help="Only recaption rows whose current caption matches this SQL LIKE pattern "
                              "(e.g. '%%cat%%'). Implies --force for the matched rows.")
    parser.add_argument("--blob-ids-file", default=None,
                         help="Only recaption blob_ids listed in this file (one per line). "
                              "Implies --force for the listed rows.")
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"Error: Database not found at {db_path}", file=sys.stderr)
        sys.exit(1)

    # Make sure schema is migrated to contain caption column
    from omura.utils.blob_catalog import init_catalog_db
    init_catalog_db(db_path)

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    
    query = "SELECT blob_id FROM blobs WHERE (kind = 'image' OR mime_type LIKE 'image/%') AND is_active = 1"
    params: list[str] = []
    if args.blob_ids_file:
        wanted = [l.strip() for l in Path(args.blob_ids_file).read_text().splitlines() if l.strip()]
        query += f" AND blob_id IN ({','.join('?' for _ in wanted)})"
        params.extend(wanted)
    elif args.caption_like:
        query += " AND caption LIKE ?"
        params.append(args.caption_like)
    elif not args.force:
        query += " AND (caption IS NULL OR caption = '')"

    cur.execute(query, params)
    rows = cur.fetchall()
    conn.close()

    blob_ids = [r[0] for r in rows]
    total = len(blob_ids)
    if total == 0:
        print("No uncaptioned images found. Use --force to rewrite them.")
        return

    print("Pinging aggregators to warm routing scores...")
    get_pool().startup_ping()

    print(f"Starting captioning for {total} images...")

    success = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_blob, bid): bid for bid in blob_ids}
        
        with tqdm(total=total, desc="Captioning") as pbar:
            for fut in as_completed(futures):
                bid = futures[fut]
                try:
                    blob_id, caption = fut.result()
                    if caption:
                        # Save back to sqlite immediately to prevent loss
                        conn = sqlite3.connect(str(db_path), timeout=30)
                        cur = conn.cursor()
                        cur.execute("UPDATE blobs SET caption = ? WHERE blob_id = ?", (caption, blob_id))
                        conn.commit()
                        conn.close()
                        success += 1
                    else:
                        failed += 1
                except Exception as e:
                    print(f"Failed to process {bid}: {e}")
                    failed += 1
                pbar.update(1)

    print("\n--- Summary ---")
    print(f"Successfully Captioned: {success}")
    print(f"Failed/Missing:         {failed}")

if __name__ == "__main__":
    main()
