"""Quick benchmark for Omura embedding models.

This script measures:
1) model init time
2) text embedding latency / throughput
3) optional image embedding latency / throughput
4) tiny text-retrieval quality sanity check (Recall@1)

Usage examples:
  OMURA_EMBEDDING_MODEL="<user>/omura_emmbed" uv run python benchmarks/benchmark_omura_emmbed.py
  uv run python benchmarks/benchmark_omura_emmbed.py --model "<user>/omura_emmbed" --image-dir data/samples
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from embedding_backend import (
    MODEL_NAME,
    generate_image_embedding,
    generate_text_embedding,
    initialize_embedding_model,
)


TEXT_PAIRS = [
    ("a red sports car on a road", "photo of a red sports car"),
    ("a dog playing in a park", "a puppy running on grass"),
    ("city skyline at sunset", "sunset over downtown buildings"),
    ("fresh sushi on a plate", "close-up of Japanese sushi"),
    ("snowy mountain landscape", "mountains covered in snow"),
]

NEGATIVE_TEXTS = [
    "quantum computing research paper",
    "database transaction rollback logs",
    "server rack in a datacenter",
    "medical x-ray image report",
    "circuit board schematic diagram",
]


def l2_normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n > 0:
        return (v / n).astype(np.float32, copy=False)
    return v.astype(np.float32, copy=False)


def benchmark_text(rounds: int) -> dict:
    samples = [p[0] for p in TEXT_PAIRS] + [p[1] for p in TEXT_PAIRS] + NEGATIVE_TEXTS
    vecs = []
    started = time.perf_counter()
    calls = 0
    for _ in range(rounds):
        for text in samples:
            emb = generate_text_embedding(text, is_document=False)
            if emb is None:
                continue
            vecs.append(l2_normalize(np.asarray(emb, dtype=np.float32).flatten()))
            calls += 1
    elapsed = max(1e-9, time.perf_counter() - started)
    ms_per_item = (elapsed / max(1, calls)) * 1000.0
    return {
        "calls": calls,
        "elapsed_seconds": elapsed,
        "items_per_second": calls / elapsed,
        "avg_ms_per_item": ms_per_item,
        "vectors_collected": len(vecs),
    }


def benchmark_retrieval() -> dict:
    queries = []
    docs = []
    gt = []

    for i, (q, d) in enumerate(TEXT_PAIRS):
        queries.append(q)
        docs.append(d)
        gt.append(i)
    docs.extend(NEGATIVE_TEXTS)

    qvecs = []
    for q in queries:
        emb = generate_text_embedding(q, is_document=False)
        if emb is None:
            return {"error": "query embedding failed"}
        qvecs.append(l2_normalize(np.asarray(emb, dtype=np.float32).flatten()))

    dvecs = []
    for d in docs:
        emb = generate_text_embedding(d, is_document=True)
        if emb is None:
            return {"error": "document embedding failed"}
        dvecs.append(l2_normalize(np.asarray(emb, dtype=np.float32).flatten()))

    qarr = np.stack(qvecs, axis=0)
    darr = np.stack(dvecs, axis=0)
    sims = qarr @ darr.T

    def recall_at_k(k: int) -> float:
        k = max(1, min(k, sims.shape[1]))
        topk = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
        hits = 0
        for i in range(topk.shape[0]):
            if int(gt[i]) in set(int(x) for x in topk[i].tolist()):
                hits += 1
        return float(hits) / float(topk.shape[0])

    return {
        "queries": len(queries),
        "docs": len(docs),
        "recall_at_1": recall_at_k(1),
        "recall_at_5": recall_at_k(5),
        "recall_at_10": recall_at_k(10),
    }


def benchmark_images(image_dir: Path | None, max_images: int) -> dict | None:
    if image_dir is None:
        return None
    if not image_dir.exists() or not image_dir.is_dir():
        return {"error": f"image_dir not found: {image_dir}"}

    paths = []
    for p in sorted(image_dir.iterdir()):
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
            paths.append(p)
        if len(paths) >= max_images:
            break

    if not paths:
        return {"error": "no image files found"}

    started = time.perf_counter()
    calls = 0
    for p in paths:
        try:
            data = p.read_bytes()
        except Exception:
            continue
        emb = generate_image_embedding(data, blob_id=f"bench_{p.name}")
        if emb is None:
            continue
        calls += 1
    elapsed = max(1e-9, time.perf_counter() - started)
    return {
        "calls": calls,
        "elapsed_seconds": elapsed,
        "items_per_second": calls / elapsed if calls > 0 else 0.0,
        "avg_ms_per_item": (elapsed / max(1, calls)) * 1000.0,
        "images_considered": len(paths),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Omura embedding model")
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="Optional model id override (same as OMURA_EMBEDDING_MODEL).",
    )
    parser.add_argument("--rounds", type=int, default=20, help="Rounds for text benchmark.")
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=None,
        help="Optional directory of images for image benchmark.",
    )
    parser.add_argument("--max-images", type=int, default=100)
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("benchmarks/results/omura_emmbed_benchmark.json"),
    )
    args = parser.parse_args()

    if args.model.strip():
        os.environ["OMURA_EMBEDDING_MODEL"] = args.model.strip()

    t0 = time.perf_counter()
    initialize_embedding_model()
    init_s = time.perf_counter() - t0

    result = {
        "model": os.getenv("OMURA_EMBEDDING_MODEL", MODEL_NAME),
        "init_seconds": init_s,
        "text_benchmark": benchmark_text(rounds=max(1, args.rounds)),
        "retrieval_sanity": benchmark_retrieval(),
    }

    image_metrics = benchmark_images(args.image_dir, max_images=max(1, args.max_images))
    if image_metrics is not None:
        result["image_benchmark"] = image_metrics

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps(result, indent=2))
    print(f"[Benchmark] Wrote: {args.out_json}")


if __name__ == "__main__":
    main()
