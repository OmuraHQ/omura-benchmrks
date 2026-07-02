"""Standardized verification of in-video temporal localization on Charades-STA.

Zero-shot moment retrieval: for each (video, sentence, gt_window), slide multi-scale windows
over the video, embed each with omura-embed-video, score vs the sentence, take the top-1
window, and measure IoU with the ground-truth moment. Reports the standard Charades-STA
metric: R@1 IoU@{0.3,0.5,0.7} + mIoU.

  IV2_CKPT=<ckpt> OMURA_VIDEO_HEADS=<heads> CUDA_VISIBLE_DEVICES=6 \
    .venv-iv2/bin/python scripts/eval_charades_sta.py \
      --videos-dir data/charades/Charades_v1_480 \
      --anno data/charades/charades_sta_test.txt \
      --max-videos 300 --out results/charades_sta.json
"""
import os, sys, json, time, argparse
from collections import defaultdict
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import iv2_temporal as T


def multiscale_windows(duration, scales=(4.0, 8.0, 16.0)):
    wins = []
    for w in scales:
        if w >= duration:
            wins.append((0.0, round(float(duration), 3)))
            continue
        t = 0.0
        while t < duration:
            wins.append((round(t, 3), round(min(duration, t + w), 3)))
            if t + w >= duration:
                break
            t += w / 2.0
    # dedup
    return sorted(set(wins))


def iou(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def parse_anno(path):
    by = defaultdict(list)
    for line in open(path):
        line = line.strip()
        if not line or "##" not in line:
            continue
        head, sent = line.split("##", 1)
        parts = head.split()
        if len(parts) < 3:
            continue
        vid, s, e = parts[0], float(parts[1]), float(parts[2])
        by[vid].append((s, e, sent.strip()))
    return by


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos-dir", default="data/charades/Charades_v1_480")
    ap.add_argument("--anno", default="data/charades/charades_sta_test.txt")
    ap.add_argument("--out", default="results/charades_sta.json")
    ap.add_argument("--max-videos", type=int, default=0)
    ap.add_argument("--num-frames", type=int, default=4)
    args = ap.parse_args()

    by = parse_anno(args.anno)
    vids = sorted(by.keys())
    if args.max_videos:
        vids = vids[:args.max_videos]

    ious = []
    per_thr = {0.3: 0, 0.5: 0, 0.7: 0}
    q_total = 0
    used_videos = 0
    missing = 0
    t0 = time.time()

    for vi, vid in enumerate(vids):
        path = os.path.join(args.videos_dir, vid + ".mp4")
        if not os.path.exists(path):
            missing += 1
            continue
        dur, fps, vlen = T.C.video_duration(path)
        if not dur:
            missing += 1
            continue
        wins = multiscale_windows(dur)
        feats, kept = T.embed_windows(path, wins, num_frames=args.num_frames)
        if feats is None:
            missing += 1
            continue
        used_videos += 1
        # score every sentence for this video against the shared window bank
        for (gs, ge, sent) in by[vid]:
            qemb = T.M.embed_text([sent])[0]
            sims = feats @ np.asarray(qemb, dtype=np.float32)
            top = int(np.argmax(sims))
            pred = kept[top]
            j = iou(pred, (gs, ge))
            ious.append(j); q_total += 1
            for thr in per_thr:
                if j >= thr:
                    per_thr[thr] += 1
        if used_videos % 25 == 0:
            print(f"  videos={used_videos} queries={q_total} "
                  f"mIoU={np.mean(ious):.3f} R@1IoU0.5={per_thr[0.5]/max(q_total,1):.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    result = {
        "task": "Charades-STA test — zero-shot natural-language moment retrieval (R@1)",
        "model": "omura-embed-video (finetuned InternVideo2-6B), naive-attn bf16",
        "protocol": {
            "windows": "multi-scale sliding (4/8/16s, 50% overlap), top-1 by cosine",
            "num_frames_per_window": args.num_frames,
            "videos_evaluated": used_videos, "videos_missing": missing,
            "num_queries": q_total,
        },
        "num_queries": q_total,
        "mIoU": round(float(np.mean(ious)) if ious else 0.0, 4),
        "R@1_IoU": {f"{thr}": round(per_thr[thr] / max(q_total, 1), 4) for thr in per_thr},
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
