"""Verify in-video temporal localization on OUR Walrus catalog (production-representative).

Catalog clips are short single-scene videos. To get ground-truth timestamps we build
synthetic multi-scene videos: concatenate K real catalog clips into one mp4 and record each
clip's [start,end] span. Then for each clip's caption we localize it inside the concatenated
video and check whether the top-1 predicted window lands on the correct span (IoU + hit-rate).

  IV2_CKPT=<ckpt> OMURA_VIDEO_HEADS=<heads> CUDA_VISIBLE_DEVICES=6 \
    .venv-iv2/bin/python scripts/eval_walrus_temporal.py --num-videos 60 --clips-per 4 \
      --out results/walrus_temporal.json
"""
import os, sys, re, json, time, argparse, tempfile, subprocess, signal
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import requests

sys.path.insert(0, os.path.dirname(__file__))
import iv2_temporal as T


class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout()


signal.signal(signal.SIGALRM, _alarm)

FFMPEG = "/usr/bin/ffmpeg"
FFPROBE = "/usr/bin/ffprobe"
AGG = [a.strip().rstrip("/") for a in os.getenv(
    "OMURA_FETCH_AGGREGATORS",
    "https://agrregator.omura.fun,https://aggregator.walrus-mainnet.walrus.space").split(",") if a.strip()]
_DATE_HASH = re.compile(r"^\d{6,8}-\d{4,6}-[0-9a-fA-F]{16,}-")


def caption_from_id(blob_id):
    ident = blob_id.split("::", 1)[1] if "::" in blob_id else blob_id
    ident = re.sub(r"\.(mp4|mov|webm|mkv)$", "", ident, flags=re.I)
    ident = _DATE_HASH.sub("", ident)
    return re.sub(r"[-_]+", " ", ident).strip()


def fetch(blob_id):
    paths = []
    if "::" in blob_id:
        q, ident = blob_id.split("::", 1)
        paths.append(f"/v1/blobs/by-quilt-id/{q}/{requests.utils.quote(ident, safe='')}")
    else:
        paths.append(f"/v1/blobs/{blob_id}")
    for agg in AGG:
        for p in paths:
            try:
                r = requests.get(f"{agg}{p}", timeout=120)
                if r.status_code == 200 and r.content:
                    return r.content
            except Exception:
                continue
    return None


def reencode(src, dst):
    """Normalize to a common codec/size/fps so concat works; return duration or None."""
    try:
        subprocess.run([FFMPEG, "-y", "-i", src, "-vf", "scale=320:240,fps=12",
                        "-c:v", "libx264", "-an", "-pix_fmt", "yuv420p", dst],
                       check=True, capture_output=True, timeout=120)
        out = subprocess.run([FFPROBE, "-v", "error", "-show_entries", "format=duration",
                              "-of", "csv=p=0", dst], capture_output=True, text=True, timeout=30)
        return float(out.stdout.strip())
    except Exception:
        return None


def iou(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def _dump(out, built, clips_per, q_total, win_sec, stride_sec, ious, hits, per_thr, partial=False):
    result = {
        "task": "In-video temporal localization on Omura/Walrus catalog (synthetic multi-scene)",
        "model": "omura-embed-video (finetuned InternVideo2-6B)",
        "partial": partial,
        "protocol": {
            "synthetic_videos": built, "clips_per_video": clips_per,
            "num_queries": q_total, "win_sec": win_sec, "stride_sec": stride_sec,
            "ground_truth": "each query is a constituent clip's caption; GT span = that clip's [start,end] in the concatenation",
            "metric": "top-1 predicted window vs GT span",
        },
        "num_queries": q_total,
        "mIoU": round(float(np.mean(ious)) if ious else 0.0, 4),
        "correct_clip_hit_rate": round(hits / max(q_total, 1), 4),
        "R@1_IoU": {f"{thr}": round(per_thr[thr] / max(q_total, 1), 4) for thr in per_thr},
    }
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(result, open(out, "w"), indent=2)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids-npz", default="data/cache/video_embeds.npz")
    ap.add_argument("--out", default="results/walrus_temporal.json")
    ap.add_argument("--num-videos", type=int, default=60)
    ap.add_argument("--clips-per", type=int, default=4)
    ap.add_argument("--win-sec", type=float, default=2.0)
    ap.add_argument("--stride-sec", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ids = [str(x) for x in np.load(args.ids_npz, allow_pickle=True)["ids"]]
    rng = np.random.default_rng(args.seed)
    rng.shuffle(ids)
    pool = iter(ids)

    ious, hits, q_total = [], 0, 0
    per_thr = {0.3: 0, 0.5: 0, 0.7: 0}
    built = 0
    exhausted = False
    t0 = time.time()

    while built < args.num_videos and not exhausted:
        clips = []
        # gather clips-per decodable clips
        with tempfile.TemporaryDirectory() as td:
            norm_paths, spans, caps = [], [], []
            cum = 0.0
            for _ in range(args.clips_per * 2):
                if len(norm_paths) >= args.clips_per:
                    break
                try:
                    bid = next(pool)
                except StopIteration:
                    exhausted = True
                    break
                try:
                    signal.alarm(45)  # hard cap per clip: skip hanging fetch/reencode
                    data = fetch(bid)
                    if not data:
                        continue
                    raw = os.path.join(td, "raw.mp4"); open(raw, "wb").write(data)
                    npth = os.path.join(td, f"n{len(norm_paths)}.mp4")
                    dur = reencode(raw, npth)
                except _Timeout:
                    continue
                finally:
                    signal.alarm(0)
                if not dur or dur < 1.0:
                    continue
                norm_paths.append(npth); caps.append(caption_from_id(bid))
                spans.append((round(cum, 3), round(cum + dur, 3))); cum += dur
            if len(norm_paths) < 2:
                continue
            # concat
            listf = os.path.join(td, "list.txt")
            open(listf, "w").write("".join(f"file '{p}'\n" for p in norm_paths))
            cat = os.path.join(td, "cat.mp4")
            try:
                subprocess.run([FFMPEG, "-y", "-f", "concat", "-safe", "0", "-i", listf,
                                "-c", "copy", cat], check=True, capture_output=True, timeout=120)
            except Exception:
                continue
            built += 1
            # one query per constituent clip
            for span, cap in zip(spans, caps):
                if not cap:
                    continue
                res = T.localize_file(cat, cap, win_sec=args.win_sec, stride_sec=args.stride_sec, top_k=1)
                segs = res.get("segments", [])
                if not segs:
                    continue
                pred = (segs[0]["start"], segs[0]["end"])
                j = iou(pred, span)
                ious.append(j); q_total += 1
                mid = (pred[0] + pred[1]) / 2
                if span[0] <= mid <= span[1]:
                    hits += 1
                for thr in per_thr:
                    if j >= thr:
                        per_thr[thr] += 1
        if built % 10 == 0:
            print(f"  built={built}/{args.num_videos} queries={q_total} "
                  f"mIoU={np.mean(ious) if ious else 0:.3f} hit={hits/max(q_total,1):.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
            _dump(args.out, built, args.clips_per, q_total, args.win_sec, args.stride_sec,
                  ious, hits, per_thr, partial=True)

    result = _dump(args.out, built, args.clips_per, q_total, args.win_sec, args.stride_sec,
                   ious, hits, per_thr, partial=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
