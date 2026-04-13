from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from benchmark_coco_retrieval import ensure_coco_data, load_coco_items, sample_items
from embedding_backend import MODEL_NAME, generate_image_embedding, initialize_embedding_model

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COCO_ROOT = REPO_ROOT / "data" / "coco"


def _pca_2d(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    x = x - x.mean(axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(x, full_matrices=False)
    return (u[:, :2] * s[:2]).astype(np.float32, copy=False)


def _l2(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n > 0:
        return (v / n).astype(np.float32, copy=False)
    return v.astype(np.float32, copy=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate COCO embedding activation atlas chart")
    parser.add_argument("--coco-root", type=Path, default=DEFAULT_COCO_ROOT)
    parser.add_argument("--coco-split", type=str, default="val2014")
    parser.add_argument(
        "--no-download-coco",
        action="store_true",
        help="Do not download COCO; fail if captions/images are missing.",
    )
    parser.add_argument("--captions-json", type=Path, required=False)
    parser.add_argument("--images-dir", type=Path, required=False)
    parser.add_argument("--num-images", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out-png",
        type=Path,
        default=Path("benchmarks/results/coco_activation_atlas.png"),
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("benchmarks/results/coco_activation_atlas_stats.json"),
    )
    args = parser.parse_args()

    captions_json, images_dir = ensure_coco_data(
        coco_root=args.coco_root,
        split=args.coco_split,
        captions_json=args.captions_json,
        images_dir=args.images_dir,
        download=not bool(args.no_download_coco),
    )
    items = sample_items(load_coco_items(captions_json), args.num_images, args.seed)

    initialize_embedding_model()

    vecs: List[np.ndarray] = []
    kept = 0
    for i, item in enumerate(items, start=1):
        p = images_dir / item.file_name
        if not p.exists():
            continue
        try:
            data = p.read_bytes()
        except Exception:
            continue
        emb = generate_image_embedding(data, blob_id=f"atlas_{item.image_id}")
        if emb is None:
            continue
        vecs.append(_l2(np.asarray(emb, dtype=np.float32).flatten()))
        kept += 1
        if i % 100 == 0:
            print(f"[Atlas] Embedded images: {kept}/{len(items)}")

    if not vecs:
        raise SystemExit("No embeddings generated for atlas.")

    arr = np.stack(vecs, axis=0)
    p2 = _pca_2d(arr)
    x = p2[:, 0]
    y = p2[:, 1]

    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 8))
    hb = plt.hexbin(x, y, gridsize=45, cmap="viridis", mincnt=1)
    plt.colorbar(hb, label="Embedding density")
    plt.title(f"COCO Activation Atlas (PCA-2D) - {MODEL_NAME}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(args.out_png, dpi=180)
    plt.close()

    stats: Dict[str, object] = {
        "model": MODEL_NAME,
        "num_points": int(arr.shape[0]),
        "embedding_dim": int(arr.shape[1]),
        "pc1_mean": float(np.mean(x)),
        "pc1_std": float(np.std(x)),
        "pc2_mean": float(np.mean(y)),
        "pc2_std": float(np.std(y)),
        "png": str(args.out_png),
    }
    args.out_json.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    print(f"[Atlas] Wrote chart: {args.out_png}")
    print(f"[Atlas] Wrote stats: {args.out_json}")


if __name__ == "__main__":
    main()
