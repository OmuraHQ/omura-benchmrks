"""Compare pure-visual vs image+caption hybrid image-to-image retrieval.

Uses MS COCO val2014. Candidate pool = original images. Queries = hard-transformed
versions of the same images (10% border crop, 80% resize, JPEG Q40).

Measures Recall@K for:
  - pure visual query (transformed image embedding only)
  - hybrid query     (transformed image embedding blended with the image's caption
                      text embedding, weight 0.2 by default)

This isolates the effect of text descriptions in image-to-image retrieval.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageFilter
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent))

from embedding_backend import (
    generate_image_embedding,
    generate_text_embedding,
    initialize_embedding_model,
    MODEL_NAME,
)
from benchmark_coco_retrieval import (
    ensure_coco_data,
    load_coco_items,
    sample_items,
    l2_normalize,
    ImageItem,
    DEFAULT_COCO_ROOT,
)


def transform_image(
    image_path: Path,
    crop_frac: float,
    resize_frac: float,
    jpeg_quality: int,
    blur_sigma: float = 0.0,
    color_jitter: float = 0.0,
    seed: int = 0,
) -> bytes:
    """Apply crop, resize, blur, color jitter, and compression to simulate query variation."""
    rng = np.random.default_rng(seed)
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        w, h = img.size
        cx = crop_frac / 2.0
        crop_box = (int(w * cx), int(h * cx), int(w * (1.0 - cx)), int(h * (1.0 - cx)))
        cropped = img.crop(crop_box)
        resized = cropped.resize((max(16, int(w * resize_frac)), max(16, int(h * resize_frac))))

        if blur_sigma > 0:
            resized = resized.filter(ImageFilter.GaussianBlur(radius=blur_sigma))

        if color_jitter > 0:
            arr = np.asarray(resized, dtype=np.float32) / 255.0
            arr *= float(rng.uniform(1.0 - color_jitter, 1.0 + color_jitter))
            mean = 0.5
            contrast = float(rng.uniform(1.0 - color_jitter, 1.0 + color_jitter))
            arr = (arr - mean) * contrast + mean
            gray = np.mean(arr, axis=2, keepdims=True)
            saturation = float(rng.uniform(1.0 - color_jitter, 1.0 + color_jitter))
            arr = gray + (arr - gray) * saturation
            arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
            resized = Image.fromarray(arr)

        buf = io.BytesIO()
        resized.save(buf, format="JPEG", quality=jpeg_quality)
        return buf.getvalue()


def embed_image(data: bytes, blob_id: str) -> np.ndarray | None:
    emb = generate_image_embedding(data, blob_id=blob_id)
    if emb is None:
        return None
    return l2_normalize(np.asarray(emb, dtype=np.float32).flatten())


def embed_text(text: str) -> np.ndarray | None:
    emb = generate_text_embedding(text, is_document=False)
    if emb is None:
        return None
    return l2_normalize(np.asarray(emb, dtype=np.float32).flatten())


def blend_vectors(visual: np.ndarray, text: np.ndarray, weight: float) -> np.ndarray:
    blended = (1.0 - weight) * visual + weight * text
    n = float(np.linalg.norm(blended))
    return (blended / n).astype(np.float32, copy=False) if n > 0 else visual


def recall_at_k(query_vecs: np.ndarray, candidate_vecs: np.ndarray, k: int) -> float:
    sims = query_vecs @ candidate_vecs.T
    topk = np.argpartition(-sims, kth=min(k - 1, sims.shape[1] - 1), axis=1)[:, :k]
    hits = 0
    for i in range(topk.shape[0]):
        if i in set(int(x) for x in topk[i].tolist()):
            hits += 1
    return float(hits) / float(topk.shape[0])


def build_candidate_and_query_vectors(
    items: List[ImageItem],
    images_dir: Path,
    caption_weight: float,
    crop_frac: float,
    resize_frac: float,
    jpeg_quality: int,
    blur_sigma: float,
    color_jitter: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[ImageItem]]:
    cand_vecs: List[np.ndarray] = []
    visual_vecs: List[np.ndarray] = []
    hybrid_vecs: List[np.ndarray] = []
    kept: List[ImageItem] = []

    rng = np.random.default_rng(seed)
    for idx, item in enumerate(tqdm(items, desc="Embedding candidates & queries", unit="img")):
        image_path = images_dir / item.file_name
        if not image_path.exists():
            continue

        try:
            orig_data = image_path.read_bytes()
            orig_emb = embed_image(orig_data, f"coco_orig_{item.image_id}")
            if orig_emb is None:
                continue

            query_data = transform_image(
                image_path,
                crop_frac,
                resize_frac,
                jpeg_quality,
                blur_sigma=blur_sigma,
                color_jitter=color_jitter,
                seed=int(rng.integers(0, 2**31)),
            )
            visual_emb = embed_image(query_data, f"coco_query_{item.image_id}")
            if visual_emb is None:
                continue

            caption = item.captions[0] if item.captions else ""
            if caption:
                text_emb = embed_text(caption)
            else:
                text_emb = None

            hybrid_emb = blend_vectors(visual_emb, text_emb, caption_weight) if text_emb is not None else visual_emb

            cand_vecs.append(orig_emb)
            visual_vecs.append(visual_emb)
            hybrid_vecs.append(hybrid_emb)
            kept.append(item)
        except Exception as e:
            print(f"[Warning] Failed processing {item.file_name}: {e}")
            continue

    if not cand_vecs:
        raise RuntimeError("No embeddings generated.")

    return (
        np.stack(cand_vecs, axis=0),
        np.stack(visual_vecs, axis=0),
        np.stack(hybrid_vecs, axis=0),
        kept,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="COCO i2i: pure visual vs image+caption hybrid")
    parser.add_argument("--num-images", type=int, default=1000)
    parser.add_argument("--caption-weight", type=float, default=0.2)
    parser.add_argument("--crop-frac", type=float, default=0.25, help="Border crop fraction (e.g. 0.20 = 20% total border)")
    parser.add_argument("--resize-frac", type=float, default=0.50, help="Resize scale (e.g. 0.70 = 70% of original)")
    parser.add_argument("--jpeg-quality", type=int, default=15, help="JPEG compression quality")
    parser.add_argument("--blur-sigma", type=float, default=1.5, help="Gaussian blur radius")
    parser.add_argument("--color-jitter", type=float, default=0.3, help="Brightness/contrast/saturation jitter scale")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--coco-root", type=Path, default=DEFAULT_COCO_ROOT)
    parser.add_argument("--out-json", type=Path, default=Path("benchmarks/results/coco_i2i_hybrid.json"))
    args = parser.parse_args()

    captions_json, images_dir = ensure_coco_data(
        coco_root=args.coco_root,
        split="val2014",
        captions_json=None,
        images_dir=None,
        download=True,
    )
    items_all = load_coco_items(captions_json)
    items = sample_items(items_all, args.num_images, args.seed)

    transform_label = (
        f"{args.crop_frac*100:.0f}% crop + {args.resize_frac*100:.0f}% resize + "
        f"JPEG Q{args.jpeg_quality} + blur σ{args.blur_sigma} + color-jitter {args.color_jitter}"
    )
    print(f"[Hybrid i2i] Model: {MODEL_NAME}")
    print(f"[Hybrid i2i] Images selected: {len(items)}")
    print(f"[Hybrid i2i] Transform: {transform_label}")
    print(f"[Hybrid i2i] Caption weight: {args.caption_weight}")

    initialize_embedding_model()

    cand, visual, hybrid, kept = build_candidate_and_query_vectors(
        items, images_dir, args.caption_weight,
        args.crop_frac, args.resize_frac, args.jpeg_quality,
        args.blur_sigma, args.color_jitter, args.seed,
    )
    print(f"[Hybrid i2i] Kept images: {len(kept)}")

    metrics = {}
    for name, qvecs in (("pure_visual", visual), ("image_caption_hybrid", hybrid)):
        metrics[name] = {
            "R@1": recall_at_k(qvecs, cand, 1),
            "R@5": recall_at_k(qvecs, cand, 5),
            "R@10": recall_at_k(qvecs, cand, 10),
        }

    out = {
        "model": MODEL_NAME,
        "dataset": "MS COCO val2014",
        "num_images": len(kept),
        "transform": transform_label,
        "caption_weight": args.caption_weight,
        "caption_source": "first COCO caption per image",
        "results": metrics,
        "improvement": {
            "R@1_pp": round((metrics["image_caption_hybrid"]["R@1"] - metrics["pure_visual"]["R@1"]) * 100, 2),
            "R@5_pp": round((metrics["image_caption_hybrid"]["R@5"] - metrics["pure_visual"]["R@5"]) * 100, 2),
            "R@10_pp": round((metrics["image_caption_hybrid"]["R@10"] - metrics["pure_visual"]["R@10"]) * 100, 2),
        },
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("\n[Hybrid i2i] Results:")
    print(json.dumps(out, indent=2))
    print(f"\n[Hybrid i2i] Wrote results to: {args.out_json}")


if __name__ == "__main__":
    main()
