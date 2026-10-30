"""ActivityNet Captions text->video retrieval with EmbeddingGemma 2.

ActivityNet videos are large; this harness accepts an already-downloaded video
directory and a captions JSON (official ``activitynet_val.json`` or the
``ActivityNet-dense-cap.json`` 1-fps mirror). It evaluates a deterministic
subset pool (default 500 videos) so a CPU/GPU run stays inside a sane budget.

If the video files are not available the script still writes an artifact with
``"status": "no_data"`` and the exact blocker — it never fabricates a metric.
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
    recall_at_k,
    utc_now,
    write_result,
)

DEFAULT_ROOT = REPO_ROOT / "data" / "activitynet"
DEFAULT_ANNO = DEFAULT_ROOT / "ActivityNet-dense-cap.json"
DEFAULT_VIDEOS = DEFAULT_ROOT / "videos"


def load_annotations(path: Path) -> List[Dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    items: List[Dict[str, Any]] = []

    if isinstance(raw, list):
        # ActivityNet-dense-cap.json style.
        for row in raw:
            vid = str(row.get("image_id") or row.get("video_id") or "").strip()
            caps = []
            if row.get("pure_cap"):
                caps = [c.strip() for c in str(row["pure_cap"]).split(" , ") if c.strip()]
            if not caps:
                continue
            items.append({"video_id": vid, "captions": caps, "duration": row.get("duration")})
    elif isinstance(raw, dict):
        # Official activitynet_{val,test}.json style: {vid: {sentences, duration}}.
        for vid, row in raw.items():
            caps = [str(s).strip() for s in (row.get("sentences") or []) if str(s).strip()]
            if not caps:
                continue
            items.append(
                {
                    "video_id": str(vid).strip(),
                    "captions": caps,
                    "duration": row.get("duration"),
                }
            )
    else:
        raise ValueError(f"Unrecognised annotations structure in {path}")

    # One representative sentence per video keeps the query/video ratio at 1:1.
    for it in items:
        it["caption"] = it["captions"][0]
    return items


def resolve_video(videos_dir: Path, vid: str) -> Path | None:
    stem = vid[:-4] if vid.lower().endswith(".mp4") else vid
    for name in (f"{stem}.mp4", f"{stem}.mkv", f"{stem}.webm", f"{vid}"):
        p = videos_dir / name
        if p.exists():
            return p
    return None


def embed_videos_batched(
    datas: List[bytes], exts: List[str], ids: List[str], dim: int, batch_size: int
) -> tuple[np.ndarray, List[int]]:
    vecs: List[np.ndarray | None] = []
    for start in tqdm(range(0, len(datas), batch_size), desc="Embedding ActivityNet videos", unit="batch"):
        vecs.extend(
            eb.generate_video_embeddings(
                datas[start : start + batch_size],
                exts=exts[start : start + batch_size],
                blob_ids=ids[start : start + batch_size],
            )
        )
    kept, rows = [], []
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
    return (np.stack(kept, axis=0) if kept else np.zeros((0, dim), np.float32)), rows


def main() -> None:
    ap = argparse.ArgumentParser(description="ActivityNet Captions text->video retrieval")
    ap.add_argument("--annotations", type=Path, default=DEFAULT_ANNO)
    ap.add_argument("--videos-dir", type=Path, default=DEFAULT_VIDEOS)
    ap.add_argument("--num-videos", type=int, default=500, help="Pool size (deterministic subset).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument(
        "--out-json",
        type=Path,
        default=REPO_ROOT / "benchmarks" / "results" / "activitynet_gemma.json",
    )
    args = ap.parse_args()

    base = {
        "model": eb.MODEL_NAME,
        "dim": int(eb.EMBEDDING_DIM or 768),
        "protocol": "activitynet_captions_text_to_video_subset",
        "dataset": "ActivityNet Captions",
        "annotations": str(args.annotations),
        "videos_dir": str(args.videos_dir),
        "seed": args.seed,
        "commit": git_commit(),
        "utc_date": utc_now(),
        "pool_size": 0,
        "num_queries": 0,
    }

    if not args.annotations.exists():
        base.update(
            status="no_data",
            blocker=(
                f"annotations missing: {args.annotations}. Download "
                "ActivityNet-dense-cap.json from "
                "https://huggingface.co/datasets/rrustlee/ActivityNet1fps"
            ),
        )
        write_result(args.out_json, base)
        print(json.dumps(base, indent=2))
        return

    items = load_annotations(args.annotations)
    rng = np.random.default_rng(args.seed)
    if args.num_videos and args.num_videos < len(items):
        idx = rng.choice(len(items), size=args.num_videos, replace=False)
        items = [items[i] for i in sorted(int(x) for x in idx)]
    if args.limit:
        items = items[: args.limit]

    available = []
    for it in items:
        p = resolve_video(args.videos_dir, it["video_id"])
        if p is not None:
            available.append((it, p))
    print(f"[ActivityNet] candidates={len(items)} videos_found={len(available)}")

    if not available:
        base.update(
            status="no_data",
            blocker=(
                "No ActivityNet video files found under "
                f"{args.videos_dir}. The official mirrors are large "
                "(rrustlee/ActivityNet1fps tar.gz ~8.5 GB; friedrichor tar ~42 GB); "
                "download a mirror or point --videos-dir at an existing copy."
            ),
        )
        write_result(args.out_json, base)
        print(json.dumps(base, indent=2))
        return

    print(f"[ActivityNet] model: {eb.MODEL_NAME} dim={eb.EMBEDDING_DIM}")
    eb.initialize_embedding_model()
    dim = int(eb.EMBEDDING_DIM or 768)

    datas = [p.read_bytes() for _, p in available]
    exts = [p.suffix.lstrip(".") for _, p in available]
    ids = [it["video_id"] for it, _ in available]
    video_vecs, kept = embed_videos_batched(datas, exts, ids, dim, args.batch_size)
    kept_items = [available[i][0] for i in kept]

    texts = [it["caption"] for it in kept_items]
    text_vecs, kept_text = embed_texts(
        eb.generate_text_embeddings, texts, dim, batch_size=64, desc="Embedding captions"
    )
    valid = [pos for pos in kept_text if pos < video_vecs.shape[0]]
    if len(valid) != len(kept_text):
        text_vecs = text_vecs[[j for j, p in enumerate(kept_text) if p < video_vecs.shape[0]]]
    sims = text_vecs @ video_vecs.T
    gt = [np.asarray([i], dtype=np.int64) for i in valid]
    t2v = recall_at_k(sims, gt)

    base.update(
        status="ok",
        pool_size=int(video_vecs.shape[0]),
        num_queries=int(text_vecs.shape[0]),
        num_videos_requested=int(args.num_videos),
        videos_found=len(available),
        text_to_video=t2v,
        frames_per_video=int(__import__("os").getenv("OMURA_OMURAGEMMA_MAX_FRAMES", "32")),
        notes=(
            "Deterministic subset pool; one representative caption per video. "
            "Global retrieval, no sampled negatives."
        ),
    )
    write_result(args.out_json, base)
    print(json.dumps(base, indent=2))
    print(f"[ActivityNet] Wrote: {args.out_json}")


if __name__ == "__main__":
    main()
