"""NFT1000 image-to-image retrieval: pure visual vs image+caption hybrid.

Uses a single NFT collection (e.g. ALIENFRENS). Candidate pool = original NFT
images. Queries = transformed versions of the same images (crop + resize + compress).

Measures Recall@K and mAP for:
  - pure visual query (transformed image embedding only)
  - hybrid query     (transformed image embedding blended with the NFT caption
                      text embedding, weight 0.2 by default)

This isolates the effect of text descriptions (NFT metadata captions) in
image-to-image retrieval for highly-similar PFP-style NFTs.
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
from benchmark_coco_retrieval import l2_normalize


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
        img = img.convert("RGBA")
        # Composite on white to avoid transparent edges after crop/resize
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img).convert("RGB")
        w, h = img.size
        cx = crop_frac / 2.0
        crop_box = (int(w * cx), int(h * cx), int(w * (1.0 - cx)), int(h * (1.0 - cx)))
        cropped = img.crop(crop_box)
        resized = cropped.resize((max(16, int(w * resize_frac)), max(16, int(h * resize_frac))))

        if blur_sigma > 0:
            resized = resized.filter(ImageFilter.GaussianBlur(radius=blur_sigma))

        if color_jitter > 0:
            # Random brightness/contrast/saturation shifts in [-jitter, +jitter]
            arr = np.asarray(resized, dtype=np.float32) / 255.0
            # Brightness
            arr *= float(rng.uniform(1.0 - color_jitter, 1.0 + color_jitter))
            # Contrast: move around mean gray
            mean = 0.5
            contrast = float(rng.uniform(1.0 - color_jitter, 1.0 + color_jitter))
            arr = (arr - mean) * contrast + mean
            # Saturation
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


def mean_average_precision(query_vecs: np.ndarray, candidate_vecs: np.ndarray) -> float:
    """mAP where each query has exactly one relevant candidate at the same index."""
    sims = query_vecs @ candidate_vecs.T
    # ranks[i, j] = position of candidate j in descending similarity for query i
    ranks = np.argsort(-sims, axis=1)
    aps = []
    for i in range(ranks.shape[0]):
        rank_pos = np.where(ranks[i] == i)[0]
        if len(rank_pos) == 0:
            aps.append(0.0)
        else:
            pos = int(rank_pos[0]) + 1  # 1-based rank
            aps.append(1.0 / pos)
    return float(np.mean(aps))


def load_pairs(img_dir: Path, caption_dir: Path, num_samples: int, seed: int) -> List[Tuple[Path, Path, str]]:
    """Return list of (image_path, caption_path, token_id) pairs."""
    img_files = sorted(img_dir.glob("*.png"))
    rng = np.random.default_rng(seed)
    if num_samples < len(img_files):
        img_files = [img_files[i] for i in rng.choice(len(img_files), num_samples, replace=False)]
    pairs = []
    for img_path in img_files:
        token_id = img_path.stem
        cap_path = caption_dir / f"{token_id}.txt"
        if cap_path.exists():
            pairs.append((img_path, cap_path, token_id))
    return pairs


def build_vectors(
    pairs: List[Tuple[Path, Path, str]],
    caption_weight: float,
    crop_frac: float,
    resize_frac: float,
    jpeg_quality: int,
    blur_sigma: float,
    color_jitter: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    cand_vecs: List[np.ndarray] = []
    visual_vecs: List[np.ndarray] = []
    hybrid_vecs: List[np.ndarray] = []
    kept_ids: List[str] = []

    rng = np.random.default_rng(seed)
    for idx, (img_path, cap_path, token_id) in enumerate(tqdm(pairs, desc="Embedding NFTs & queries", unit="img")):
        try:
            orig_data = img_path.read_bytes()
            orig_emb = embed_image(orig_data, f"nft_orig_{token_id}")
            if orig_emb is None:
                continue

            query_data = transform_image(
                img_path,
                crop_frac,
                resize_frac,
                jpeg_quality,
                blur_sigma=blur_sigma,
                color_jitter=color_jitter,
                seed=int(rng.integers(0, 2**31)),
            )
            visual_emb = embed_image(query_data, f"nft_query_{token_id}")
            if visual_emb is None:
                continue

            caption = cap_path.read_text(encoding="utf-8").strip()
            text_emb = embed_text(caption) if caption else None
            hybrid_emb = blend_vectors(visual_emb, text_emb, caption_weight) if text_emb is not None else visual_emb

            cand_vecs.append(orig_emb)
            visual_vecs.append(visual_emb)
            hybrid_vecs.append(hybrid_emb)
            kept_ids.append(token_id)
        except Exception as e:
            print(f"[Warning] Failed processing {img_path.name}: {e}")
            continue

    if not cand_vecs:
        raise RuntimeError("No embeddings generated.")

    return (
        np.stack(cand_vecs, axis=0),
        np.stack(visual_vecs, axis=0),
        np.stack(hybrid_vecs, axis=0),
        kept_ids,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="NFT1000 i2i: pure visual vs image+caption hybrid")
    parser.add_argument("--img-dir", type=Path, required=True)
    parser.add_argument("--caption-dir", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=2000)
    parser.add_argument("--caption-weight", type=float, default=0.2)
    parser.add_argument("--crop-frac", type=float, default=0.25)
    parser.add_argument("--resize-frac", type=float, default=0.50)
    parser.add_argument("--jpeg-quality", type=int, default=15)
    parser.add_argument("--blur-sigma", type=float, default=1.5)
    parser.add_argument("--color-jitter", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-json", type=Path, default=Path("benchmarks/results/nft1000_i2i_hybrid.json"))
    args = parser.parse_args()

    pairs = load_pairs(args.img_dir, args.caption_dir, args.num_samples, args.seed)
    transform_label = (
        f"{args.crop_frac*100:.0f}% crop + {args.resize_frac*100:.0f}% resize + "
        f"JPEG Q{args.jpeg_quality} + blur σ{args.blur_sigma} + color-jitter {args.color_jitter}"
    )

    print(f"[NFT1000 i2i] Model: {MODEL_NAME}")
    print(f"[NFT1000 i2i] Collection: {args.img_dir}")
    print(f"[NFT1000 i2i] Sampled images with captions: {len(pairs)}")
    print(f"[NFT1000 i2i] Transform: {transform_label}")
    print(f"[NFT1000 i2i] Caption weight: {args.caption_weight}")

    initialize_embedding_model()

    cand, visual, hybrid, kept_ids = build_vectors(
        pairs,
        args.caption_weight,
        args.crop_frac,
        args.resize_frac,
        args.jpeg_quality,
        args.blur_sigma,
        args.color_jitter,
        args.seed,
    )
    print(f"[NFT1000 i2i] Kept images: {len(kept_ids)}")

    metrics = {}
    for name, qvecs in (("pure_visual", visual), ("image_caption_hybrid", hybrid)):
        metrics[name] = {
            "R@1": recall_at_k(qvecs, cand, 1),
            "R@5": recall_at_k(qvecs, cand, 5),
            "R@10": recall_at_k(qvecs, cand, 10),
            "mAP": mean_average_precision(qvecs, cand),
        }

    out = {
        "model": MODEL_NAME,
        "dataset": "NFT1000",
        "collection": str(args.img_dir.name),
        "num_images": len(kept_ids),
        "transform": transform_label,
        "caption_weight": args.caption_weight,
        "caption_source": "NFT1000 generated caption",
        "results": metrics,
        "improvement": {
            "R@1_pp": round((metrics["image_caption_hybrid"]["R@1"] - metrics["pure_visual"]["R@1"]) * 100, 2),
            "R@5_pp": round((metrics["image_caption_hybrid"]["R@5"] - metrics["pure_visual"]["R@5"]) * 100, 2),
            "R@10_pp": round((metrics["image_caption_hybrid"]["R@10"] - metrics["pure_visual"]["R@10"]) * 100, 2),
            "mAP_pp": round((metrics["image_caption_hybrid"]["mAP"] - metrics["pure_visual"]["mAP"]) * 100, 2),
        },
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("\n[NFT1000 i2i] Results:")
    print(json.dumps(out, indent=2))
    print(f"\n[NFT1000 i2i] Wrote results to: {args.out_json}")


if __name__ == "__main__":
    main()
