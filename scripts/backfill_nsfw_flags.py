#!/usr/bin/env python3
"""
Re-score indexed image embeddings against NSFW text prototypes and persist flags.

Uses the same 0–100 tag score as ``/search`` (``min(max(cosine, 0) * 1000, 100)``).
NSFW is tagged when that score is **strictly greater** than the minimum (default **85**).

Updates:
  - Vector store ``metadata.json`` (``is_nsfw``, ``nsfw_score`` as 0–100 tag score)
  - ``blob_catalog.sqlite`` via :meth:`omura.utils.vector_store.VectorStore.save` sync

Environment:
  - ``OMURA_NSFW_TAG_SCORE_MIN`` — exclusive floor (default ``85``); flag when ``score >`` this.

Examples::

    PYTHONPATH=. uv run python scripts/backfill_nsfw_flags.py --dry-run
    PYTHONPATH=. uv run python scripts/backfill_nsfw_flags.py
    OMURA_NSFW_TAG_SCORE_MIN=80 PYTHONPATH=. uv run python scripts/backfill_nsfw_flags.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _normalize_kind(kind: str | None) -> str | None:
    if not kind:
        return None
    k = kind.lower().strip()
    if k in ("image", "video", "audio", "doc", "quilt"):
        return k
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print counts only; do not write metadata or SQLite.",
    )
    parser.add_argument(
        "--score-min",
        type=float,
        default=None,
        help="Exclusive minimum 0–100 tag score to mark NSFW (default: OMURA_NSFW_TAG_SCORE_MIN or 85).",
    )
    parser.add_argument(
        "--add-categories",
        action="store_true",
        help="When flagging NSFW, set categories to nsfw/pornographic if missing.",
    )
    args = parser.parse_args()

    from omura.utils.blob_catalog import init_catalog_db
    from omura.utils.imagebind_embeddings import (
        get_nsfw_embeddings,
        is_nsfw_from_tag_score,
        MODEL_NAME,
        nsfw_similarity_score_0_100,
        nsfw_tag_score_min,
    )
    from omura.utils.vector_store import VectorStore

    init_catalog_db()

    sm = float(args.score_min) if args.score_min is not None else nsfw_tag_score_min()
    print(
        f"[backfill_nsfw] model={MODEL_NAME!r} score_min={sm:.2f} (flag when score > {sm:.2f}) "
        f"dry_run={args.dry_run}"
    )

    nsfw_vecs = get_nsfw_embeddings()
    if not nsfw_vecs:
        print("[backfill_nsfw] ERROR: NSFW prototypes empty (is the embedding model loaded?)", file=sys.stderr)
        return 1

    store = VectorStore()
    store.load()

    scanned = 0
    flagged = 0
    flips = 0
    score_writes = 0

    for blob_id, raw in list(store.metadata.items()):
        try:
            meta = json.loads(raw) if isinstance(raw, str) else dict(raw)
        except Exception:
            continue

        if _normalize_kind(meta.get("kind")) != "image":
            continue

        emb = store.get_embedding(blob_id)
        if emb is None:
            continue

        tag_score = float(nsfw_similarity_score_0_100(emb, nsfw_vecs))
        is_nsfw = bool(is_nsfw_from_tag_score(tag_score, score_min=sm))

        scanned += 1
        if is_nsfw:
            flagged += 1

        old = bool(meta.get("is_nsfw", False))
        old_score = meta.get("nsfw_score")
        meta["nsfw_score"] = tag_score
        meta["is_nsfw"] = is_nsfw
        if args.add_categories and is_nsfw:
            cats = list(meta.get("categories") or [])
            for c in ("nsfw", "pornographic"):
                if c not in cats:
                    cats.append(c)
            meta["categories"] = cats

        if old != is_nsfw:
            flips += 1
        if old_score is None or abs(float(old_score) - tag_score) > 1e-5:
            score_writes += 1

        if not args.dry_run:
            store.metadata[blob_id] = json.dumps(meta)

    print(
        f"[backfill_nsfw] images_scanned={scanned} flagged_nsfw={flagged} "
        f"is_nsfw_flips={flips} nsfw_score_refreshes={score_writes}"
    )

    if args.dry_run:
        return 0

    if scanned > 0:
        store.save(create_backup=False)
        print("[backfill_nsfw] Saved metadata + synced SQLite.")
    else:
        print("[backfill_nsfw] No image rows to persist.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
