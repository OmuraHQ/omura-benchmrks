"""Download ~N random Creative-Commons / public-domain audio files from
the Internet Archive and zip them.

Sources (all CC0 / public domain / CC-BY):
  - collection:opensource_audio       (mixed open-license audio)
  - collection:librivoxaudio          (LibriVox audiobook chapters)
  - collection:georgeblood            (78rpm digitization — vintage music)

Usage:
  uv run python scripts/fetch_random_audios.py --count 500 --out /tmp/audios.zip
  uv run python scripts/fetch_random_audios.py --count 100 --max-size-mb 5 --workers 24

The script:
  1. Queries IA advancedsearch for random audio items.
  2. For each item, hits the item details API to find the first .mp3 / .ogg / .flac.
  3. Downloads in parallel (size-capped per file so the run doesn't go gigabyte-crazy).
  4. Zips everything (STORED, since audio is already compressed).

If a download fails or the file isn't audio, it's skipped and another item is tried.
"""

from __future__ import annotations

import argparse
import os
import random
import re
import sys
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

import requests

IA_SEARCH = "https://archive.org/advancedsearch.php"
IA_METADATA = "https://archive.org/metadata"
IA_DOWNLOAD = "https://archive.org/download"

# Each query targets a different open collection; we mix them.
COLLECTIONS = [
    "opensource_audio",
    "librivoxaudio",
    "georgeblood",
    "audio_podcast",
    "audio_music",
]
AUDIO_EXTS = ("mp3", "ogg", "flac", "wav", "m4a")
DEFAULT_HEADERS = {"User-Agent": "omura-research/1.0 (audio testset builder)"}


_tls = threading.local()


def _session() -> requests.Session:
    s = getattr(_tls, "s", None)
    if s is None:
        s = requests.Session()
        s.headers.update(DEFAULT_HEADERS)
        a = requests.adapters.HTTPAdapter(
            pool_connections=4, pool_maxsize=32,
            max_retries=requests.adapters.Retry(
                total=2, backoff_factor=0.5, status_forcelist=[502, 503, 504],
                allowed_methods=["GET"],
            ),
        )
        s.mount("https://", a)
        s.mount("http://", a)
        _tls.s = s
    return s


def _ia_item_ids(collection: str, rows: int = 200, page: int = 1) -> List[str]:
    """List item identifiers from an IA collection.

    IA's advancedsearch wants ``q`` as a plain-ASCII Lucene-style query — using
    ``+`` for AND is fine, but the encoded form (``%2B``) breaks the parser, and
    using ``sort[]=random`` is invalid. We omit sort and shuffle in Python instead.
    """
    q = f"collection:({collection}) AND mediatype:audio"
    params = {
        "q": q,
        "fl[]": "identifier",
        "rows": str(rows),
        "page": str(page),
        "output": "json",
    }
    try:
        r = _session().get(IA_SEARCH, params=params, timeout=30)
        if r.status_code != 200:
            return []
        docs = r.json().get("response", {}).get("docs", [])
        return [d["identifier"] for d in docs if "identifier" in d]
    except Exception:
        return []


def _pick_audio_file(item_id: str, max_size_bytes: int) -> Optional[Tuple[str, int]]:
    """Return (download_url, size_bytes) of the first audio file under the size cap."""
    try:
        r = _session().get(f"{IA_METADATA}/{item_id}", timeout=30)
        if r.status_code != 200:
            return None
        meta = r.json()
        for f in meta.get("files", []):
            name = f.get("name", "")
            ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
            if ext not in AUDIO_EXTS:
                continue
            try:
                size = int(f.get("size", "0"))
            except (TypeError, ValueError):
                size = 0
            if 0 < size <= max_size_bytes:
                return f"{IA_DOWNLOAD}/{item_id}/{name}", size
        return None
    except Exception:
        return None


def _safe_filename(item_id: str, url: str) -> str:
    """Filename-safe ID derived from item + URL basename."""
    base = url.rsplit("/", 1)[-1]
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", base)
    return f"{item_id[:40]}__{base}"[:200]


def _download_one(url: str, item_id: str, out_dir: Path, max_bytes: int) -> Optional[Path]:
    """Stream a single audio file to out_dir."""
    fname = _safe_filename(item_id, url)
    path = out_dir / fname
    try:
        with _session().get(url, stream=True, timeout=60) as r:
            if r.status_code != 200:
                return None
            ctype = r.headers.get("content-type", "").lower()
            if "audio" not in ctype and "octet-stream" not in ctype:
                return None
            total = 0
            with open(path, "wb") as fh:
                for chunk in r.iter_content(64 * 1024):
                    fh.write(chunk)
                    total += len(chunk)
                    if total > max_bytes:
                        # Truncate at the cap; truncated audio still has its header,
                        # which is all we need for type detection / partial playback.
                        break
        if total < 1024:  # too tiny to be real audio
            path.unlink(missing_ok=True)
            return None
        return path
    except Exception:
        path.unlink(missing_ok=True)
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=500, help="number of audio files")
    parser.add_argument("--out", type=Path, default=Path("/tmp/walrus_test_audios.zip"))
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument(
        "--max-size-mb", type=float, default=10.0,
        help="per-file size cap (default 10 MB)",
    )
    parser.add_argument(
        "--scratch", type=Path, default=Path("/tmp/walrus_audio_scratch"),
        help="temp dir to download into before zipping",
    )
    args = parser.parse_args()

    max_bytes = int(args.max_size_mb * 1024 * 1024)
    args.scratch.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Target: {args.count} audio files into {args.out}")
    print(f"Per-file cap: {args.max_size_mb:.1f} MB | Workers: {args.workers}")

    # --- gather candidate item identifiers from each collection ---
    all_items: List[Tuple[str, str]] = []  # (collection, item_id)
    per_collection = max(args.count * 3 // len(COLLECTIONS), 100)
    print(f"Querying {len(COLLECTIONS)} collections ({per_collection} items each)...")
    for col in COLLECTIONS:
        page = 1
        gathered = 0
        while gathered < per_collection and page <= 5:
            ids = _ia_item_ids(col, rows=min(200, per_collection - gathered), page=page)
            if not ids:
                break
            for i in ids:
                all_items.append((col, i))
            gathered += len(ids)
            page += 1
            time.sleep(0.2)
        print(f"  {col:<25s} {gathered:>5} items")
    random.shuffle(all_items)
    print(f"Total candidates: {len(all_items):,}")

    # --- pick first audio file from each item, then download ---
    downloaded: List[Path] = []
    skipped = 0
    fetch_failed = 0
    t0 = time.time()

    def _try_one(col_item: Tuple[str, str]) -> Optional[Path]:
        col, item = col_item
        picked = _pick_audio_file(item, max_bytes)
        if picked is None:
            return None
        url, _size = picked
        return _download_one(url, item, args.scratch, max_bytes)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_try_one, ci): ci for ci in all_items}
        for fut in as_completed(futs):
            if len(downloaded) >= args.count:
                break
            try:
                p = fut.result()
            except Exception:
                p = None
            if p is None:
                fetch_failed += 1
                continue
            downloaded.append(p)
            if len(downloaded) % 10 == 0:
                rate = len(downloaded) / max(time.time() - t0, 0.1)
                print(
                    f"  downloaded {len(downloaded):>4}/{args.count} "
                    f"({rate:.1f}/s, failed={fetch_failed})",
                    flush=True,
                )

    print()
    print(f"Downloaded {len(downloaded)} files in {time.time() - t0:.1f}s")
    if not downloaded:
        print("Nothing downloaded — exiting without zip.")
        return 1

    # --- zip them up ---
    print(f"Zipping into {args.out}...")
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_STORED) as zf:
        for p in downloaded:
            zf.write(p, arcname=p.name)
    zip_size = args.out.stat().st_size
    print(f"Done. Zip size: {zip_size / (1024*1024):.1f} MB  ({len(downloaded)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
