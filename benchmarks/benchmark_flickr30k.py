"""Flickr30k text<->image retrieval benchmark (EmbeddingGemma 2 ready).

Standard protocol
-----------------
The official Flickr30k test split has 1,000 images with 5 human captions each.
All 5,000 captions are queries; the pool is the 1,000 test images (global
retrieval, no sampled negatives).

``--pool 5000`` keeps the same 1,000 test queries but adds 4,000 deterministic
train-split images as distractors (a harder, explicitly non-standard 5K pool).

Usage
-----
    OMURA_EMBEDDING_BACKEND=omuragemma OMURA_EMBEDDING_DIM=768 \
    CUDA_VISIBLE_DEVICES=4 PYTHONPATH=.:benchmarks \
    .venv/bin/python benchmarks/benchmark_flickr30k.py --pool 1000
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))

import embedding_backend as eb
from bench_common import (
    REPO_ROOT,
    embed_images,
    embed_texts,
    git_commit,
    recall_at_k,
    utc_now,
    write_result,
)

DEFAULT_ROOT = REPO_ROOT / "data" / "flickr30k"
ANN_CSV = DEFAULT_ROOT / "flickr_annotations_30k.csv"
ZIP_PATH = DEFAULT_ROOT / "flickr30k-images.zip"
IMAGES_DIR = DEFAULT_ROOT / "images"


@dataclass
class FlickrItem:
    img_id: int
    filename: str
    split: str
    captions: List[str]


def ensure_images(root: Path) -> Path:
    """Return the directory that directly contains the jpgs."""
    candidates = [
        root / "images" / "flickr30k-images",
        root / "images",
        root / "flickr30k-images",
    ]
    for c in candidates:
        if c.is_dir() and any(c.glob("*.jpg")):
            return c

    part = root / "flickr30k-images.zip.part"
    if part.exists() and not ZIP_PATH.exists():
        part.replace(ZIP_PATH)
    if not ZIP_PATH.exists():
        raise FileNotFoundError(
            f"Missing {ZIP_PATH}. Download from "
            "https://huggingface.co/datasets/nlphuji/flickr30k/resolve/main/flickr30k-images.zip"
        )
    print(f"[Flickr30k] Extracting {ZIP_PATH} -> {root}")
    with zipfile.ZipFile(ZIP_PATH) as zf:
        zf.extractall(root)
    for c in candidates:
        if c.is_dir() and any(c.glob("*.jpg")):
            return c
    raise RuntimeError(f"Extracted {ZIP_PATH} but found no *.jpg under {root}")


def load_items(csv_path: Path) -> List[FlickrItem]:
    import csv

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Missing {csv_path}. Download from "
            "https://huggingface.co/datasets/nlphuji/flickr30k/resolve/main/flickr_annotations_30k.csv"
        )
    items: List[FlickrItem] = []
    with csv_path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                raw = json.loads(row["raw"])
            except Exception:
                continue
            captions = [str(c).strip() for c in raw if str(c).strip()]
            if not captions:
                continue
            items.append(
                FlickrItem(
                    img_id=int(row["img_id"]),
                    filename=str(row["filename"]),
                    split=str(row["split"]).strip().lower(),
                    captions=captions,
                )
            )
    items.sort(key=lambda x: x.img_id)
    return items


def main() -> None:
    ap = argparse.ArgumentParser(description="Flickr30k retrieval (Recall@K)")
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--pool", type=int, default=1000, choices=[1000, 5000])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="Smoke-test: cap test queries/images.")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out-json", type=Path, default=REPO_ROOT / "benchmarks" / "results" / "flickr30k_gemma_1k.json")
    args = ap.parse_args()

    images_dir = ensure_images(args.root)
    items = load_items(args.root / "flickr_annotations_30k.csv")
    test_items = [it for it in items if it.split == "test"]
    train_items = [it for it in items if it.split == "train"]
    if len(test_items) != 1000:
        print(f"[Flickr30k][Warning] expected 1000 test images, found {len(test_items)}")
    if args.limit:
        test_items = test_items[: args.limit]

    rng = np.random.default_rng(args.seed)
    distractors: List[FlickrItem] = []
    if args.pool > len(test_items):
        need = args.pool - len(test_items)
        need = min(need, len(train_items))
        idx = rng.choice(len(train_items), size=need, replace=False)
        distractors = [train_items[i] for i in sorted(int(x) for x in idx)]
    pool_items = test_items + distractors

    print(f"[Flickr30k] images dir: {images_dir}")
    print(f"[Flickr30k] test queries : {len(test_items)} images / {sum(len(i.captions) for i in test_items)} captions")
    print(f"[Flickr30k] pool         : {len(pool_items)} images")
    print(f"[Flickr30k] model        : {eb.MODEL_NAME} (backend={'omuragemma' if eb.IS_OMURAGEMMA else 'siglip'})")

    eb.initialize_embedding_model()
    dim = int(eb.EMBEDDING_DIM or 768)

    image_datas: List[bytes] = []
    kept_pool: List[FlickrItem] = []
    for it in pool_items:
        p = images_dir / it.filename
        if not p.exists():
            continue
        try:
            image_datas.append(p.read_bytes())
            kept_pool.append(it)
        except OSError:
            continue
    pool_vecs, kept_pos = embed_images(
        eb.generate_image_embeddings,
        image_datas,
        dim,
        batch_size=args.batch_size,
        desc="Embedding Flickr30k images",
    )
    kept_pool = [kept_pool[i] for i in kept_pos]
    test_ids = {it.img_id for it in test_items}
    test_pool_idx = {it.img_id: j for j, it in enumerate(kept_pool) if it.img_id in test_ids}
    missing_queries = len(test_items) - len(test_pool_idx)
    if missing_queries:
        print(f"[Flickr30k][Warning] {missing_queries} test images failed to embed and are excluded")

    # Text queries: all 5 captions of every test image that has a vector.
    query_texts: List[str] = []
    gt_img: List[np.ndarray] = []
    for it in test_items:
        j = test_pool_idx.get(it.img_id)
        if j is None:
            continue
        for cap in it.captions:
            query_texts.append(cap)
            gt_img.append(np.asarray([j], dtype=np.int64))
    text_vecs, kept_text = embed_texts(
        eb.generate_text_embeddings,
        query_texts,
        dim,
        batch_size=max(32, args.batch_size * 2),
        desc="Embedding Flickr30k captions",
    )
    gt_img = [gt_img[i] for i in kept_text]

    sims_t2i = text_vecs @ pool_vecs.T
    t2i = recall_at_k(sims_t2i, gt_img, ks=(1, 5, 10))

    # Image->text: candidate set = all captions of all pool images.
    cand_texts: List[str] = []
    gt_ranges: List[np.ndarray] = []
    cursor = 0
    for it in kept_pool:
        caps = it.captions if it.captions else [it.filename]
        cand_texts.extend(caps)
        gt_ranges.append(np.arange(cursor, cursor + len(caps), dtype=np.int64))
        cursor += len(caps)
    cand_vecs, kept_cand = embed_texts(
        eb.generate_text_embeddings,
        cand_texts,
        dim,
        batch_size=max(32, args.batch_size * 2),
        desc="Embedding Flickr30k candidate captions",
        is_document=True,
    )
    # Translate candidate row positions if any failed (rare).
    if len(kept_cand) != len(cand_texts):
        remap = {old: new for new, old in enumerate(kept_cand)}
        gt_ranges = [
            np.asarray([remap[int(x)] for x in arr if int(x) in remap], dtype=np.int64)
            for arr in gt_ranges
        ]

    query_img_rows = [
        test_pool_idx[it.img_id] for it in test_items if it.img_id in test_pool_idx
    ]
    sims_i2t = pool_vecs[query_img_rows] @ cand_vecs.T
    i2t = recall_at_k(
        sims_i2t, [gt_ranges[r] for r in query_img_rows], ks=(1, 5, 10)
    )

    protocol = (
        "flickr30k_standard_1k_test"
        if args.pool == len(test_items)
        else f"flickr30k_1k_test_queries_pool_{len(pool_items)}_with_train_distractors"
    )
    payload = {
        "model": eb.MODEL_NAME,
        "dim": dim,
        "backend": "omuragemma" if eb.IS_OMURAGEMMA else "siglip",
        "protocol": protocol,
        "dataset": "Flickr30k (nlphuji/flickr30k)",
        "split": "test",
        "seed": args.seed,
        "commit": git_commit(),
        "utc_date": utc_now(),
        "pool_size": int(pool_vecs.shape[0]),
        "pool_size_requested": int(args.pool),
        "num_queries": int(text_vecs.shape[0]),
        "num_images": int(pool_vecs.shape[0]),
        "num_caption_candidates": int(cand_vecs.shape[0]),
        "caption_count": int(sum(len(it.captions) for it in kept_pool)),
        "captions_per_image": 5,
        "text_to_image": t2i,
        "image_to_text": i2t,
        "notes": (
            "Global retrieval over the full pool; no sampled negatives. "
            "All 5 test captions per image are queries. i2t candidate set is "
            "every caption belonging to every pool image."
        ),
        "env": {
            "OMURA_EMBEDDING_BACKEND": __import__("os").getenv("OMURA_EMBEDDING_BACKEND", ""),
            "CUDA_VISIBLE_DEVICES": __import__("os").getenv("CUDA_VISIBLE_DEVICES", ""),
        },
    }
    write_result(args.out_json, payload)
    print(json.dumps(payload, indent=2))
    print(f"[Flickr30k] Wrote: {args.out_json}")


if __name__ == "__main__":
    main()
