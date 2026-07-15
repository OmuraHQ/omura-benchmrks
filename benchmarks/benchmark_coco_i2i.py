"""Benchmark Image-to-Image (i2i) retrieval accuracy on transformed COCO images.

This script measures Recall@K when retrieving original images using cropped, resized, 
and compressed query versions of the same images.

It leverages the same COCO dataset download/parsing utilities as benchmark_coco_retrieval.py.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

# Add parent directory to path so we can import embedding_backend and benchmark_coco_retrieval
sys.path.append(str(Path(__file__).resolve().parent))

from embedding_backend import (
    generate_image_embedding,
    initialize_embedding_model,
    MODEL_NAME,
)

from benchmark_coco_retrieval import (
    ensure_coco_data,
    ensure_karpathy_split_file,
    load_coco_items,
    load_karpathy_items,
    sample_items,
    l2_normalize,
    ImageItem,
    DEFAULT_COCO_ROOT,
)


def transform_image(image_path: Path) -> bytes:
    """Apply crop (5% border), resize (90%), and compression (JPEG Q60) to simulate query variation."""
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        w, h = img.size
        # Crop 5% from borders
        crop_box = (int(w * 0.05), int(h * 0.05), int(w * 0.95), int(h * 0.95))
        cropped = img.crop(crop_box)
        # Resize to 90%
        resized = cropped.resize((max(16, int(w * 0.9)), max(16, int(h * 0.9))))
        # Save as JPEG with quality 60
        buf = io.BytesIO()
        resized.save(buf, format="JPEG", quality=60)
        return buf.getvalue()


def embed_original_and_queries(
    items: List[ImageItem], 
    images_dir: Path
) -> Tuple[np.ndarray, np.ndarray, List[ImageItem]]:
    """Generate embeddings for original images (candidates) and transformed versions (queries)."""
    orig_vecs: List[np.ndarray] = []
    query_vecs: List[np.ndarray] = []
    kept: List[ImageItem] = []
    
    for item in tqdm(items, desc="Embedding original & query images", unit="img"):
        image_path = images_dir / item.file_name
        if not image_path.exists():
            continue
        try:
            # 1. Original image
            orig_data = image_path.read_bytes()
            orig_emb = generate_image_embedding(orig_data, blob_id=f"coco_orig_{item.image_id}")
            if orig_emb is None:
                continue
                
            # 2. Transformed query image
            query_data = transform_image(image_path)
            query_emb = generate_image_embedding(query_data, blob_id=f"coco_query_{item.image_id}")
            if query_emb is None:
                continue
                
            orig_vecs.append(l2_normalize(np.asarray(orig_emb, dtype=np.float32).flatten()))
            query_vecs.append(l2_normalize(np.asarray(query_emb, dtype=np.float32).flatten()))
            kept.append(item)
        except Exception as e:
            print(f"[Warning] Failed processing {item.file_name}: {e}")
            continue
            
    if not orig_vecs:
        raise RuntimeError("No image embeddings generated.")
        
    return np.stack(orig_vecs, axis=0), np.stack(query_vecs, axis=0), kept


def recall_at_k(query_vecs: np.ndarray, candidate_vecs: np.ndarray, k: int) -> float:
    """Calculate Recall@K where each query matches exactly its corresponding candidate index."""
    sims = query_vecs @ candidate_vecs.T  # Shape: [num_queries, num_candidates]
    # Argpartition sorts ascending, so negate similarities to get top-K descending
    topk = np.argpartition(-sims, kth=min(k - 1, sims.shape[1] - 1), axis=1)[:, :k]
    hits = 0
    for i in range(topk.shape[0]):
        # The correct match is the same index i
        if i in set(int(x) for x in topk[i].tolist()):
            hits += 1
    return float(hits) / float(topk.shape[0])


def main() -> None:
    parser = argparse.ArgumentParser(description="COCO Image-to-Image (i2i) retrieval benchmark")
    parser.add_argument("--captions-json", type=Path, required=False)
    parser.add_argument("--images-dir", type=Path, required=False)
    parser.add_argument(
        "--split-file",
        type=Path,
        required=False,
        help="Karpathy split JSON file (e.g., dataset_coco.json).",
    )
    parser.add_argument(
        "--download-karpathy-split",
        action="store_true",
        help="Auto-download Karpathy dataset_coco.json if --split-file is missing.",
    )
    parser.add_argument(
        "--karpathy-split",
        type=str,
        default="test",
        choices=["train", "val", "test", "restval"],
    )
    parser.add_argument(
        "--no-download-coco",
        action="store_true",
        help="Do not download COCO; fail if captions/images are missing.",
    )
    parser.add_argument(
        "--coco-root",
        type=Path,
        default=DEFAULT_COCO_ROOT,
    )
    parser.add_argument(
        "--coco-split",
        type=str,
        default="val2014",
    )
    parser.add_argument("--num-images", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("benchmarks/results/coco_i2i_repro.json"),
    )
    args = parser.parse_args()

    # Load dataset items
    if args.split_file is not None:
        if args.images_dir is None:
            raise SystemExit("--split-file requires --images-dir to be set.")
        captions_json = ensure_karpathy_split_file(
            args.split_file, bool(args.download_karpathy_split)
        )
        images_dir = args.images_dir
        items_all = load_karpathy_items(args.split_file, args.karpathy_split)
    else:
        captions_json, images_dir = ensure_coco_data(
            coco_root=args.coco_root,
            split=args.coco_split,
            captions_json=args.captions_json,
            images_dir=args.images_dir,
            download=not bool(args.no_download_coco),
        )
        items_all = load_coco_items(captions_json)
        
    if not items_all:
        raise SystemExit("No valid image items found.")
        
    items = sample_items(items_all, args.num_images, args.seed)

    print(f"[i2i Benchmark] Model: {MODEL_NAME}")
    print(f"[i2i Benchmark] Images selected: {len(items)}")
    
    initialize_embedding_model()
    
    orig_vecs, query_vecs, kept_items = embed_original_and_queries(items, images_dir)
    print(f"[i2i Benchmark] Embedded images: {orig_vecs.shape[0]}")
    
    r1 = recall_at_k(query_vecs, orig_vecs, 1)
    r5 = recall_at_k(query_vecs, orig_vecs, 5)
    r10 = recall_at_k(query_vecs, orig_vecs, 10)
    
    out = {
        "model": MODEL_NAME,
        "eval_mode": "image_to_image_transformed",
        "num_images": int(orig_vecs.shape[0]),
        "transformations": {
            "border_crop": "5%",
            "resize": "90%",
            "jpeg_quality": 60
        },
        "image_to_image": {
            "R@1": r1,
            "R@5": r5,
            "R@10": r10
        }
    }
    
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    
    print("\n[i2i Benchmark] Results:")
    print(json.dumps(out, indent=2))
    print(f"\n[i2i Benchmark] Wrote results to: {args.out_json}")


if __name__ == "__main__":
    main()
