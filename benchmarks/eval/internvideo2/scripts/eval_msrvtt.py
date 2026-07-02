"""Zero-shot text->video retrieval on MSR-VTT 1K-A test split (1000 pairs)."""
import os, sys, json, time, argparse
import numpy as np
import torch
import cv2

sys.path.insert(0, os.path.dirname(__file__))
import iv2_common as C


# video reading uses the official decord+BICUBIC pipeline in iv2_common.read_video_official


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="msrvtt_test_1k.json")
    ap.add_argument("--video_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    model, tok, cfg = C.load_model()
    nf = cfg.num_frames_test
    print(f"[msrvtt] model ready in {time.time()-t0:.1f}s, num_frames_test={nf}")

    data = json.load(open(args.json))
    if args.limit:
        data = data[: args.limit]
    n = len(data)
    print(f"[msrvtt] {n} video/caption pairs")

    # ---- embed videos ----
    vfeats = []
    captions = []
    bad = 0
    t = time.time()
    for i, item in enumerate(data):
        vp = os.path.join(args.video_dir, item["video"])
        ft = C.read_video_official(vp, num_frames=nf, image_res=224, sample="middle")
        if ft is None:
            bad += 1
            vfeats.append(torch.zeros(1, 768))
            captions.append(item["caption"])
            continue
        vf = C.get_video_feat(model, ft)  # [1,768]
        vfeats.append(vf.cpu())
        captions.append(item["caption"])
        if (i + 1) % 100 == 0:
            print(f"  video {i+1}/{n}  ({(time.time()-t)/(i+1):.2f}s/clip)")
    vfeats = torch.cat(vfeats, 0)  # [N,768]
    print(f"[msrvtt] videos embedded in {time.time()-t:.1f}s, unreadable={bad}")

    # ---- embed captions (batched) ----
    t = time.time()
    tfeats = []
    bs = 64
    for i in range(0, n, bs):
        tf = C.get_text_feat(model, tok, captions[i : i + bs], max_txt_l=cfg.max_txt_l)
        tfeats.append(tf.cpu())
    tfeats = torch.cat(tfeats, 0)  # [N,768]
    print(f"[msrvtt] captions embedded in {time.time()-t:.1f}s")

    # ---- text->video retrieval ----
    sim = tfeats @ vfeats.T  # [N_text, N_video]; diagonal is the correct pair
    ranks = np.zeros(n, dtype=np.int64)
    for i in range(n):
        order = torch.argsort(sim[i], descending=True)
        ranks[i] = (order == i).nonzero(as_tuple=True)[0].item()

    def recall(k):
        return float((ranks < k).mean() * 100.0)

    metrics = {
        "R@1": round(recall(1), 2),
        "R@5": round(recall(5), 2),
        "R@10": round(recall(10), 2),
        "MedianR": float(np.median(ranks) + 1),
        "MeanR": round(float(ranks.mean() + 1), 2),
    }
    print("[msrvtt] T2V:", metrics)

    result = {
        "task": "MSR-VTT 1K-A zero-shot text->video retrieval",
        "model": "InternVideo2-Stage2-6B (with audio encoder), naive-attn fp/bf16",
        "checkpoint": "OpenGVLab/InternVideo2-Stage2_6B-224p-f4 :: internvideo2-s2_6b-224p-f4_with_audio_encoder.pt",
        "protocol": {
            "split": "MSR-VTT 1K-A (friedrichor/MSR-VTT :: msrvtt_test_1k.json)",
            "num_pairs": n,
            "num_frames": nf,
            "frame_sample": "uniform/middle (frames2tensor)",
            "resolution": 224,
            "direction": "text->video",
            "metric": "cosine over contrastive embeddings (vision_proj / text_proj)",
        },
        "unreadable_videos": bad,
        "metrics_t2v": metrics,
        "target_R@10": 85,
        "pass_R@10": metrics["R@10"] >= 85,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=2)
    print("[msrvtt] wrote", args.out)


if __name__ == "__main__":
    main()
