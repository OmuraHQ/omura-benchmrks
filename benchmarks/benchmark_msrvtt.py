"""MSR-VTT 1K-A text->video retrieval with EmbeddingGemma 2.

Uses the local InternVideo2 evaluation copy of the 1K-A split (1,000 unique
videos, one caption each) and the 10,000 local MSR-VTT videos. EmbeddingGemma
samples up to 32 frames uniformly across each clip.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent))

import embedding_backend as eb
from bench_common import (
    REPO_ROOT,
    embed_texts,
    git_commit,
    ranks_at_k,
    recall_at_k,
    utc_now,
    write_result,
)

DEFAULT_ANNO = (
    REPO_ROOT / "benchmarks" / "eval" / "internvideo2" / "data" / "anno" / "msrvtt_test_1k.json"
)
DEFAULT_VIDEOS = (
    REPO_ROOT / "benchmarks" / "eval" / "internvideo2" / "data" / "msrvtt_videos" / "video"
)


def embed_videos_batched(
    datas: List[bytes],
    exts: List[str],
    ids: List[str],
    dim: int,
    batch_size: int,
) -> tuple[np.ndarray, List[int], List[int]]:
    vecs: List[np.ndarray | None] = []
    for start in tqdm(range(0, len(datas), batch_size), desc="Embedding videos", unit="batch"):
        chunk = datas[start : start + batch_size]
        vecs.extend(
            eb.generate_video_embeddings(
                chunk,
                exts=exts[start : start + batch_size],
                blob_ids=ids[start : start + batch_size],
            )
        )
    rows, kept = [], []
    for i, v in enumerate(vecs):
        if v is None:
            continue
        row = np.asarray(v, dtype=np.float32).reshape(-1)
        if row.size != dim:
            continue
        n = float(np.linalg.norm(row))
        if n > 0:
            row = row / n
        kept.append(row)
        rows.append(i)
    return (np.stack(kept, axis=0) if kept else np.zeros((0, dim), np.float32)), rows, rows


def main() -> None:
    ap = argparse.ArgumentParser(description="MSR-VTT 1K-A text->video retrieval")
    ap.add_argument("--anno", type=Path, default=DEFAULT_ANNO)
    ap.add_argument("--videos-dir", type=Path, default=DEFAULT_VIDEOS)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument(
        "--out-json",
        type=Path,
        default=REPO_ROOT / "benchmarks" / "results" / "msrvtt_gemma.json",
    )
    args = ap.parse_args()

    anno: List[Dict[str, Any]] = json.loads(args.anno.read_text(encoding="utf-8"))
    if args.limit:
        anno = anno[: args.limit]
    print(f"[MSRVTT] annotations: {len(anno)}  videos dir: {args.videos_dir}")
    if not args.videos_dir.exists():
        payload = {
            "model": eb.MODEL_NAME,
            "dim": int(eb.EMBEDDING_DIM or 768),
            "protocol": "msrvtt_1k_a_text_to_video",
            "pool_size": 0,
            "num_queries": 0,
            "seed": args.seed,
            "commit": git_commit(),
            "utc_date": utc_now(),
            "status": "no_data",
            "blocker": f"videos dir missing: {args.videos_dir}",
        }
        write_result(args.out_json, payload)
        print(json.dumps(payload, indent=2))
        return

    datas, exts, ids, rows, missing = [], [], [], [], []
    for i, row in enumerate(anno):
        p = args.videos_dir / row["video"]
        if not p.exists():
            missing.append(str(p))
            continue
        datas.append(p.read_bytes())
        exts.append(p.suffix.lstrip("."))
        ids.append(str(row["video_id"]))
        rows.append(i)
    print(f"[MSRVTT] videos found: {len(datas)}/{len(anno)}  missing={len(missing)}")
    print(f"[MSRVTT] model: {eb.MODEL_NAME} dim={eb.EMBEDDING_DIM}")

    eb.initialize_embedding_model()
    dim = int(eb.EMBEDDING_DIM or 768)
    video_vecs, kept_rows, _ = embed_videos_batched(
        datas, exts, ids, dim, args.batch_size
    )
    kept_rows = [rows[i] for i in kept_rows]
    kept_anno = [anno[i] for i in kept_rows]

    texts = [str(a["caption"]).strip() for a in kept_anno]
    text_vecs, kept_text = embed_texts(
        eb.generate_text_embeddings,
        texts,
        dim,
        batch_size=64,
        desc="Embedding MSR-VTT captions",
    )
    # Query i belongs to video row i; if a text or video failed, drop both sides.
    valid = [pos for pos in kept_text if pos < video_vecs.shape[0]]
    if len(valid) != len(kept_text):
        row_mask = [j for j, pos in enumerate(kept_text) if pos < video_vecs.shape[0]]
        text_vecs = text_vecs[row_mask]
    gt = [np.asarray([i], dtype=np.int64) for i in valid]
    sims = text_vecs @ video_vecs.T
    t2v = recall_at_k(sims, gt)
    ranks = ranks_at_k(sims, gt)

    payload = {
        "model": eb.MODEL_NAME,
        "dim": dim,
        "backend": "omuragemma" if eb.IS_OMURAGEMMA else "siglip",
        "protocol": "msrvtt_1k_a_text_to_video",
        "dataset": "MSR-VTT 1K-A (InternVideo2 local anno)",
        "anno": str(args.anno),
        "videos_dir": str(args.videos_dir),
        "seed": args.seed,
        "commit": git_commit(),
        "utc_date": utc_now(),
        "pool_size": int(video_vecs.shape[0]),
        "num_queries": int(text_vecs.shape[0]),
        "missing_videos": len(missing),
        "text_to_video": t2v,
        "frames_per_video": int(__import__("os").getenv("OMURA_OMURAGEMMA_MAX_FRAMES", "32")),
        "per_query_ranks": [
            {"video_id": kept_anno[i]["video_id"], "rank": int(r)}
            for i, r in zip(valid, ranks)
        ],
        "notes": (
            "Full 1K-A pool, global retrieval, one caption per video. "
            "Frames are sampled uniformly across the whole clip (max 32)."
        ),
    }
    write_result(args.out_json, payload)
    print(json.dumps({k: payload[k] for k in ("text_to_video", "pool_size", "num_queries")}, indent=2))
    print(f"[MSRVTT] Wrote: {args.out_json}")


if __name__ == "__main__":
    main()
