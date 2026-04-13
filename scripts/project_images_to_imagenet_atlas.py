"""Project indexed image embeddings onto the ImageNet 1K atlas.

Features:
- Nearest-neighbor assignment to ImageNet class anchors.
- NSFW exception bias: force-mark top-N most NSFW-like blobs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from omura.utils.imagebind_embeddings import (
    generate_text_embedding,
    get_nsfw_embeddings,
    initialize_embedding_model,
)
from omura.utils.vector_store import VectorStore


def _normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).flatten()
    n = np.linalg.norm(v)
    if n > 0:
        v = v / n
    return v


def _load_atlas(path: Path) -> list[dict[str, Any]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    points = obj.get("points", [])
    if not points:
        raise RuntimeError(f"No atlas points found in {path}")
    return points


def _build_anchor_matrix(points: list[dict[str, Any]]) -> tuple[np.ndarray, list[dict[str, Any]]]:
    vecs: list[np.ndarray] = []
    kept_points: list[dict[str, Any]] = []
    for p in points:
        prompt = p.get("prompt") or f"a photo of {p.get('label', '')}".strip()
        emb = generate_text_embedding(prompt, is_document=False)
        if emb is None:
            continue
        vecs.append(_normalize(np.asarray(emb, dtype=np.float32)))
        kept_points.append(p)
    if len(vecs) < 2:
        raise RuntimeError("Too few atlas anchors could be embedded.")
    return np.stack(vecs, axis=0), kept_points


def main() -> None:
    parser = argparse.ArgumentParser(description="Project image embeddings onto ImageNet atlas")
    parser.add_argument(
        "--atlas-json",
        type=Path,
        default=Path("data/atlas/imagenet_1k_atlas.json"),
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("data/atlas/image_projection_on_imagenet_1k.json"),
    )
    parser.add_argument(
        "--nsfw-bias-count",
        type=int,
        default=58,
        help="Force-mark top-N most NSFW-like blobs as nsfw_exception.",
    )
    parser.add_argument(
        "--include-non-images",
        action="store_true",
        help="Include non-image entries from vector store (default: images only).",
    )
    parser.add_argument(
        "--border-margin-threshold",
        type=float,
        default=0.01,
        help="If (second_distance - nearest_distance) <= threshold, mark as border-zone.",
    )
    args = parser.parse_args()

    print("[ProjectAtlas] Loading vector store...")
    store = VectorStore()
    store.load()

    blob_ids: list[str] = []
    image_vecs: list[np.ndarray] = []
    for blob_id, emb in store.embeddings_dict.items():
        meta = store.get_blob_metadata(blob_id) or {}
        if not args.include_non_images and meta.get("kind") != "image":
            continue
        image_vecs.append(_normalize(np.asarray(emb, dtype=np.float32)))
        blob_ids.append(blob_id)

    if len(image_vecs) < 2:
        raise RuntimeError("Too few image embeddings to project.")

    print(f"[ProjectAtlas] Candidate blobs: {len(blob_ids)}")
    X = np.stack(image_vecs, axis=0)  # [N, D], normalized

    print("[ProjectAtlas] Loading atlas + embedding class anchors...")
    atlas_points = _load_atlas(args.atlas_json)
    initialize_embedding_model()
    A, anchor_points = _build_anchor_matrix(atlas_points)  # [M, D]
    print(f"[ProjectAtlas] Anchors embedded: {A.shape[0]}")

    # Distance matrix (L2 over normalized vectors)
    # smaller = closer
    dists = np.linalg.norm(X[:, None, :] - A[None, :, :], axis=2)
    nn_sorted = np.argsort(dists, axis=1)
    nn_idx = nn_sorted[:, 0]
    nn2_idx = nn_sorted[:, 1]
    nn_dist = dists[np.arange(len(blob_ids)), nn_idx]
    nn2_dist = dists[np.arange(len(blob_ids)), nn2_idx]
    nn_margin = nn2_dist - nn_dist

    # NSFW exception bias
    nsfw_proto = get_nsfw_embeddings() or []
    forced_nsfw_indices: set[int] = set()
    nsfw_scores = None
    if nsfw_proto:
        P = np.stack([_normalize(p) for p in nsfw_proto], axis=0)
        nsfw_d = np.linalg.norm(X[:, None, :] - P[None, :, :], axis=2).min(axis=1)
        nsfw_scores = nsfw_d
        k = max(0, min(args.nsfw_bias_count, len(blob_ids)))
        if k > 0:
            # smallest distance -> most nsfw-like
            top_idx = np.argpartition(nsfw_d, k - 1)[:k]
            forced_nsfw_indices = set(int(i) for i in top_idx.tolist())
        print(f"[ProjectAtlas] NSFW bias forced count: {len(forced_nsfw_indices)}")
    else:
        print("[ProjectAtlas] NSFW prototypes unavailable; skipping nsfw bias")

    rows: list[dict[str, Any]] = []
    for i, blob_id in enumerate(blob_ids):
        p = anchor_points[int(nn_idx[i])]
        p2 = anchor_points[int(nn2_idx[i])]
        forced = i in forced_nsfw_indices
        is_border_zone = bool(nn_margin[i] <= args.border_margin_threshold)
        rows.append(
            {
                "blob_id": blob_id,
                "projected_x": float(p["x"]),
                "projected_y": float(p["y"]),
                "nearest_class_id": int(p["class_id"]),
                "nearest_label": p["label"],
                "nearest_class_name": p["class_name"],
                "nearest_distance": float(nn_dist[i]),
                "second_class_id": int(p2["class_id"]),
                "second_label": p2["label"],
                "second_distance": float(nn2_dist[i]),
                "decision_margin": float(nn_margin[i]),
                "is_border_zone": is_border_zone,
                "nsfw_distance": float(nsfw_scores[i]) if nsfw_scores is not None else None,
                "is_nsfw_exception": forced,
                "assigned_label": "nsfw_exception" if forced else p["label"],
            }
        )

    rows.sort(key=lambda r: (not r["is_nsfw_exception"], r["nearest_distance"]))
    class_counts: dict[str, int] = {}
    border_count = 0
    for r in rows:
        label = str(r["assigned_label"])
        class_counts[label] = class_counts.get(label, 0) + 1
        if r["is_border_zone"]:
            border_count += 1

    margin_arr = np.asarray([r["decision_margin"] for r in rows], dtype=np.float32)
    dist_arr = np.asarray([r["nearest_distance"] for r in rows], dtype=np.float32)
    out = {
        "summary": {
            "total_projected": len(rows),
            "atlas_anchor_count": int(A.shape[0]),
            "nsfw_bias_count": int(len(forced_nsfw_indices)),
            "nsfw_bias_requested": int(args.nsfw_bias_count),
            "border_margin_threshold": float(args.border_margin_threshold),
            "border_zone_count": int(border_count),
            "border_zone_ratio": float(border_count / len(rows)) if rows else 0.0,
            "nearest_distance_p50": float(np.percentile(dist_arr, 50)),
            "nearest_distance_p90": float(np.percentile(dist_arr, 90)),
            "decision_margin_p50": float(np.percentile(margin_arr, 50)),
            "decision_margin_p90": float(np.percentile(margin_arr, 90)),
            "assigned_label_counts": class_counts,
        },
        "rows": rows,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[ProjectAtlas] Wrote projection: {args.out_json}")
    print(f"[ProjectAtlas] Projected rows: {len(rows)}")


if __name__ == "__main__":
    main()
