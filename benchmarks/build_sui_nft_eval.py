"""Build a frozen Sui-native image retrieval eval set from Omura's own catalog.

The catalog (``data/blob_catalog.sqlite``) is opened read-only. We take the
images that are *already in the production EmbeddingGemma snapshot* (so their
Walrus blobs were fetchable when the snapshot was built), require a usable
caption/metadata string, deduplicate by caption and then by image SHA-256, and
fetch the image bytes through the configured Walrus aggregator.

Output (default ``data/sui_nft_eval``)::

    manifest.json          frozen eval manifest (blob ids, captions, sha256)
    images/<blob_id>.jpg   fetched bytes (the manifest is authoritative)

The manifest records ``collection`` / ``project`` / ``sui_object_id`` as null
when the catalog has no such columns; Walrus provenance is the ``blob_id`` plus
the catalog ``source`` field (``blockberry`` / ``walrus_event``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = REPO_ROOT / "data" / "blob_catalog.sqlite"
DEFAULT_STORE_META = REPO_ROOT / "data" / "vector_index_omuragemma" / "metadata.json"
DEFAULT_OUT = REPO_ROOT / "data" / "sui_nft_eval"


def _dotenv_values(path: Path) -> dict[str, str]:
    """Read the two Walrus keys from .env without importing python-dotenv."""
    out: dict[str, str] = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if key in {"WALRUS_AGGREGATOR_URL", "WALRUS_AGGREGATOR_URLS"}:
                out[key] = value.strip().strip('"').strip("'")
    except OSError:
        pass
    return out


def _aggregators() -> list[str]:
    urls: list[str] = []
    file_vals = _dotenv_values(REPO_ROOT / ".env")
    for key in ("WALRUS_AGGREGATOR_URL", "WALRUS_AGGREGATOR_URLS"):
        raw = os.getenv(key) or file_vals.get(key, "")
        for part in raw.split(","):
            part = part.strip().rstrip("/")
            if part and part not in urls:
                urls.append(part)
    if not urls:
        urls = ["https://aggregator.walrus-testnet.walrus.space"]
    return urls


def load_candidates(catalog: Path, store_meta: Path) -> list[dict[str, Any]]:
    meta = json.loads(store_meta.read_text(encoding="utf-8"))
    store_images = {
        blob_id
        for blob_id, raw in meta.items()
        if json.loads(raw).get("kind") == "image"
    }
    conn = sqlite3.connect(f"file:{catalog}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            """
            SELECT blob_id, caption, size, mime_type, extension, source,
                   first_seen_at, last_updated_at
            FROM blobs
            WHERE caption IS NOT NULL
              AND LENGTH(TRIM(caption)) > 10
              AND kind = 'image'
            """
        ).fetchall()
    finally:
        conn.close()

    seen_caption: set[str] = set()
    out: list[dict[str, Any]] = []
    for blob_id, caption, size, mime, ext, source, first_seen, last_seen in rows:
        if blob_id not in store_images:
            continue
        cap = " ".join(str(caption).split())
        key = cap.lower()
        if not cap or key in seen_caption:
            continue
        seen_caption.add(key)
        out.append(
            {
                "blob_id": blob_id,
                "walrus_blob_id": blob_id,
                "caption": cap,
                "captions": [cap],
                "collection": None,
                "project": None,
                "sui_object_id": None,
                "catalog_source": source,
                "mime_type": mime,
                "extension": ext,
                "catalog_size": int(size or 0),
                "first_seen_at": first_seen,
                "last_updated_at": last_seen,
            }
        )
    return out


def fetch_one(blob_id: str, timeout: float) -> tuple[bytes | None, int | None, str]:
    import urllib.error
    import urllib.request

    last_err = "no aggregator"
    for base in _aggregators():
        url = f"{base}/v1/blobs/{quote(blob_id)}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "omura-benchmark/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read()
                if resp.status == 200 and data:
                    return data, resp.status, url
                last_err = f"HTTP {resp.status} from {base}"
        except urllib.error.HTTPError as exc:
            last_err = f"HTTP {exc.code} from {base}"
        except Exception as exc:  # noqa: BLE001
            last_err = f"{type(exc).__name__}: {exc} ({base})"
    return None, None, last_err


def main() -> None:
    ap = argparse.ArgumentParser(description="Build Sui-native NFT retrieval eval set")
    ap.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    ap.add_argument("--store-metadata", type=Path, default=DEFAULT_STORE_META)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--target", type=int, default=1000)
    ap.add_argument("--max-candidates", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--timeout", type=float, default=40.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--retry-failed", action="store_true")
    args = ap.parse_args()

    candidates = load_candidates(args.catalog, args.store_metadata)
    rng = random.Random(args.seed)
    rng.shuffle(candidates)
    candidates = candidates[: args.max_candidates]
    print(f"[SuiEval] catalog candidates after caption dedupe: {len(candidates)}")

    images_dir = args.out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "manifest.json"

    existing: dict[str, dict[str, Any]] = {}
    if manifest_path.exists():
        try:
            prev = json.loads(manifest_path.read_text(encoding="utf-8"))
            existing = {e["blob_id"]: e for e in prev.get("images", [])}
            print(f"[SuiEval] resuming from {len(existing)} manifest entries")
        except Exception:  # noqa: BLE001
            existing = {}

    entries: list[dict[str, Any]] = list(existing.values())
    have = {e["blob_id"] for e in entries}
    failures: list[dict[str, Any]] = []
    content_hashes: dict[str, str] = {e["sha256"]: e["blob_id"] for e in entries}

    todo = [c for c in candidates if c["blob_id"] not in have]
    if args.target > 0:
        todo = todo[: max(0, (args.target - len(entries)) * 2)]
    print(f"[SuiEval] fetching up to {len(todo)} candidates with {args.workers} workers")

    started = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(fetch_one, c["blob_id"], args.timeout): c for c in todo
        }
        done = 0
        for fut in as_completed(futures):
            cand = futures[fut]
            done += 1
            data, status, detail = fut.result()
            if data is None:
                failures.append({"blob_id": cand["blob_id"], "error": detail})
                continue
            sha = hashlib.sha256(data).hexdigest()
            if sha in content_hashes:
                failures.append(
                    {
                        "blob_id": cand["blob_id"],
                        "error": f"duplicate image content of {content_hashes[sha]}",
                    }
                )
                continue
            suffix = "." + (cand.get("extension") or "jpg").lstrip(".")
            path = images_dir / f"{cand['blob_id']}{suffix}"
            path.write_bytes(data)
            content_hashes[sha] = cand["blob_id"]
            entry = dict(cand)
            entry.update(
                {
                    "sha256": sha,
                    "file": str(path.relative_to(args.out_dir)),
                    "bytes": len(data),
                    "fetched_via": detail,
                }
            )
            entries.append(entry)
            have.add(cand["blob_id"])
            if done % 50 == 0:
                print(
                    f"[SuiEval] {done}/{len(todo)} fetched, kept={len(entries)}, "
                    f"failures={len(failures)}",
                    flush=True,
                )
            if args.target and len(entries) >= args.target:
                for other in futures:
                    other.cancel()
                break

    entries.sort(key=lambda e: e["blob_id"])
    payload = {
        "name": "omura_sui_nft_eval",
        "description": (
            "Curated Sui/Walrus image retrieval set drawn from Omura's own "
            "production blob catalog and EmbeddingGemma snapshot."
        ),
        "source_catalog": str(args.catalog),
        "source_store_metadata": str(args.store_metadata),
        "selection_rules": [
            "blob present in the production omuragemma image snapshot",
            "catalog kind = image and caption/metadata length > 10 chars",
            "dedupe by normalized caption",
            "dedupe by sha256 of fetched bytes",
            "deterministic shuffle with seed before fetching",
        ],
        "provenance_notes": (
            "The catalog has no collection/project/sui_object_id columns, so those "
            "fields are null. Walrus provenance is blob_id + catalog_source "
            "(blockberry / walrus_event)."
        ),
        "seed": args.seed,
        "target": args.target,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "num_images": len(entries),
        "num_failures": len(failures),
        "failures": failures,
        "images": entries,
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    elapsed = time.time() - started
    print(
        f"[SuiEval] kept {len(entries)} images, {len(failures)} failures, "
        f"{elapsed:.1f}s -> {manifest_path}"
    )
    if len(entries) < args.target:
        print(f"[SuiEval][Warning] target {args.target} not reached")


if __name__ == "__main__":
    main()
