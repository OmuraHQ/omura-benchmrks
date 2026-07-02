#!/usr/bin/env python
"""MSR-VTT 1K-A zero-shot text->video retrieval with nvidia/omni-embed-nemotron-3b.

Protocol:
  - 1000 video/caption pairs (friedrichor/MSR-VTT msrvtt_test_1k.json, the JSFUSION 1K-A split).
  - Embed 1000 videos (encode_document) + 1000 captions (encode_query).
  - For each caption, rank all 1000 videos by cosine; report R@1/R@5/R@10 (text->video).
  - Frames sampled at fps=2, min/max pixels per the model card defaults.
"""
import json, time, argparse, os
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

MODEL = "nvidia/omni-embed-nemotron-3b"
DATA = "data/msrvtt"


def recall_at_k(sims, k):
    # sims[i,j] = cosine(caption_i, video_j). Ground truth = diagonal (caption i <-> video i).
    n = sims.shape[0]
    ranks = np.zeros(n, dtype=np.int64)
    for i in range(n):
        order = np.argsort(-sims[i])
        ranks[i] = int(np.where(order == i)[0][0])
    return {f"R@{kk}": float((ranks < kk).mean()) * 100.0 for kk in k}, ranks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--vbatch", type=int, default=4, help="video encode batch")
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--out", default="results/omni_nemotron_msrvtt.json")
    args = ap.parse_args()

    t0 = time.time()
    data = json.load(open(os.path.join(DATA, "msrvtt_test_1k.json")))
    if args.limit:
        data = data[: args.limit]
    n = len(data)
    captions = [d["caption"] for d in data]
    vpaths = [os.path.join(DATA, "video", d["video"]) for d in data]
    for p in vpaths:
        assert os.path.exists(p), f"missing video {p}"
    print(f"pairs={n}")

    model = SentenceTransformer(
        MODEL, trust_remote_code=True,
        model_kwargs={"attn_implementation": "sdpa", "torch_dtype": torch.bfloat16},
        device="cuda",
    )
    model[0].processing_kwargs.update({
        "video": {"min_pixels": 32 * 14 * 14, "max_pixels": 64 * 28 * 28,
                  "do_sample_frames": True, "fps": args.fps},
    })

    # caption embeddings
    cap_emb = model.encode_query(captions, convert_to_numpy=True, batch_size=32,
                                 show_progress_bar=False)
    cap_emb = np.asarray(cap_emb, dtype=np.float32)

    # video embeddings (batched, robust to per-video failures)
    vid_emb = np.zeros((n, cap_emb.shape[1]), dtype=np.float32)
    failed = []
    B = args.vbatch
    for start in range(0, n, B):
        end = min(start + B, n)
        docs = [{"video": vpaths[i]} for i in range(start, end)]
        try:
            emb = model.encode_document(docs, convert_to_numpy=True, show_progress_bar=False)
            vid_emb[start:end] = np.asarray(emb, dtype=np.float32)
        except Exception as e:
            # fall back to per-item to isolate the broken video
            for i in range(start, end):
                try:
                    emb = model.encode_document([{"video": vpaths[i]}],
                                                convert_to_numpy=True, show_progress_bar=False)
                    vid_emb[i] = np.asarray(emb, dtype=np.float32)[0]
                except Exception as e2:
                    failed.append((i, str(e2)[:120]))
        if start % (B * 20) == 0:
            print(f"  videos {end}/{n} failed={len(failed)}", flush=True)

    sims = cap_emb @ vid_emb.T  # cosine (both normalized)
    metrics, ranks = recall_at_k(sims, [1, 5, 10])
    median_rank = float(np.median(ranks) + 1)
    mean_rank = float(ranks.mean() + 1)

    elapsed = time.time() - t0
    result = {
        "model": MODEL,
        "checkpoint": "nvidia/omni-embed-nemotron-3b (SentenceTransformer, mean pool, 2048-d, bidirectional Qwen2.5-Omni-3B Thinker)",
        "task": "MSR-VTT 1K-A zero-shot text->video retrieval",
        "split": "msrvtt_test_1k.json (JSFUSION 1K-A, 1000 video/caption pairs)",
        "protocol": "embed 1000 videos (encode_document) + 1000 captions (encode_query); rank videos by cosine per caption; R@K text->video",
        "num_samples": n,
        "video_fps_sampled": args.fps,
        "video_min_pixels": 32 * 14 * 14,
        "video_max_pixels": 64 * 28 * 28,
        "metrics": metrics,
        "median_rank": median_rank,
        "mean_rank": mean_rank,
        "num_failed_videos": len(failed),
        "failed": failed[:20],
        "target_threshold_R@10": 85.0,
        "pass": metrics["R@10"] >= 85.0,
        "attn_implementation": "sdpa",
        "dtype": "bfloat16",
        "elapsed_sec": round(elapsed, 1),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != "failed"}, indent=2))


if __name__ == "__main__":
    main()
