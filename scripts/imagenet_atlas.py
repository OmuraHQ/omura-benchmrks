"""Build a 2D embedding atlas for ImageNet 1K class labels.

Outputs:
  - JSON with per-class embedding + 2D coordinates
  - CSV with id/label/x/y
  - Optional PNG scatter plot when matplotlib is available
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from omura.utils.imagebind_embeddings import generate_text_embedding, initialize_embedding_model


ROW_RE = re.compile(r"^\|\s*(\d+)\s*\|\s*(.*?)\s*\|$")


def parse_imagenet_markdown(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        m = ROW_RE.match(line.strip())
        if not m:
            continue
        class_id = int(m.group(1))
        class_name = m.group(2).strip()
        if class_name.lower() in {"class name", "---"}:
            continue
        if class_id < 0 or class_id > 999:
            continue
        primary_label = class_name.split(",")[0].strip()
        rows.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "label": primary_label,
                "prompt": f"a photo of {primary_label}",
            }
        )
    rows.sort(key=lambda x: x["class_id"])
    return rows


def pca_2d(x: np.ndarray) -> np.ndarray:
    if x.shape[0] == 1:
        return np.array([[0.0, 0.0]], dtype=np.float32)
    centered = x - x.mean(axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(centered, full_matrices=False)
    coords = u[:, :2] * s[:2]
    if coords.shape[1] < 2:
        pad = np.zeros((coords.shape[0], 2 - coords.shape[1]), dtype=coords.dtype)
        coords = np.concatenate([coords, pad], axis=1)
    return coords.astype(np.float32)


def maybe_write_plot(points: list[dict[str, Any]], out_png: Path) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    x = [p["x"] for p in points]
    y = [p["y"] for p in points]
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111)
    ax.scatter(x, y, s=7, alpha=0.65)
    ax.set_title("ImageNet 1K Text Atlas (PCA 2D)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ImageNet 1K embedding atlas")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("/root/.cursor/projects/workspace-proj-omura/uploads/IMAGENET-0.md"),
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("data/atlas/imagenet_1k_atlas.json"),
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("data/atlas/imagenet_1k_atlas.csv"),
    )
    parser.add_argument(
        "--out-png",
        type=Path,
        default=Path("data/atlas/imagenet_1k_atlas.png"),
    )
    args = parser.parse_args()

    classes = parse_imagenet_markdown(args.input)
    if len(classes) != 1000:
        raise SystemExit(f"Expected 1000 classes, got {len(classes)} from {args.input}")

    print("[ImageNetAtlas] Initializing embedding model...")
    initialize_embedding_model()

    vecs: list[np.ndarray] = []
    points: list[dict[str, Any]] = []
    for row in classes:
        emb = generate_text_embedding(row["prompt"], is_document=False)
        if emb is None:
            continue
        v = np.asarray(emb, dtype=np.float32).flatten()
        vecs.append(v)
        points.append(dict(row))

    if len(vecs) < 2:
        raise SystemExit("Too few class embeddings generated.")

    mat = np.stack(vecs, axis=0)
    coords = pca_2d(mat)

    for idx, p in enumerate(points):
        p["x"] = float(coords[idx, 0])
        p["y"] = float(coords[idx, 1])
        p["embedding_dim"] = int(mat.shape[1])

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps({"points": points}, indent=2), encoding="utf-8")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class_id", "label", "class_name", "x", "y"])
        for p in points:
            writer.writerow([p["class_id"], p["label"], p["class_name"], p["x"], p["y"]])

    plotted = maybe_write_plot(points, args.out_png)
    print(f"[ImageNetAtlas] Classes embedded: {len(points)}")
    print(f"[ImageNetAtlas] JSON: {args.out_json}")
    print(f"[ImageNetAtlas] CSV: {args.out_csv}")
    if plotted:
        print(f"[ImageNetAtlas] PNG: {args.out_png}")
    else:
        print("[ImageNetAtlas] PNG skipped (matplotlib unavailable)")


if __name__ == "__main__":
    main()
