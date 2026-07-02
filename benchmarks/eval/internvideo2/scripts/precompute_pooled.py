"""Precompute & cache frozen InternVideo2 vision `pooled` features (pre vision_proj)
for MSR-VTT train(9k)+test(1k). This is the expensive 6B forward; caching it lets the
finetune iterate cheaply over just the projection heads + text encoder.

  IV2_CKPT=<ckpt> CUDA_VISIBLE_DEVICES=7 .venv-iv2/bin/python scripts/precompute_pooled.py \
      --anno data/anno/msrvtt_train_9k.json --video_dir data/msrvtt_videos/video \
      --out data/cache/pooled_train.npz
"""
import os, sys, json, time, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import iv2_common as C


@torch.no_grad()
def encode_pooled(model, frames):  # frames [B,T,C,H,W]
    frames = frames.to(model._iv2_dtype)
    _, pooled = model.encode_vision(frames, test=True)
    if pooled.dim() == 3:
        pooled = pooled.squeeze(1)
    return pooled.float().cpu().numpy()  # [B,768]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anno", required=True)
    ap.add_argument("--video_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    model, tok, cfg = C.load_model()
    nf = cfg.num_frames_test
    data = json.load(open(args.anno))
    if args.limit:
        data = data[: args.limit]
    print(f"[precompute] {len(data)} videos, num_frames={nf}, batch={args.batch}")

    ids, feats = [], []
    buf_ids, buf_frames = [], []
    bad = 0
    t = time.time()

    def flush():
        nonlocal buf_ids, buf_frames
        if not buf_frames:
            return
        batch = torch.cat(buf_frames, 0)  # [B,T,C,H,W]
        pooled = encode_pooled(model, batch)
        for vid, p in zip(buf_ids, pooled):
            ids.append(vid)
            feats.append(p)
        buf_ids, buf_frames = [], []

    for i, item in enumerate(data):
        vid = item["video_id"]
        vp = os.path.join(args.video_dir, item["video"])
        ft = C.read_video_official(vp, num_frames=nf, image_res=224, sample="middle")
        if ft is None:
            bad += 1
            continue
        buf_ids.append(vid)
        buf_frames.append(ft)
        if len(buf_frames) >= args.batch:
            flush()
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(data)} ({(time.time()-t)/(i+1):.2f}s/clip) bad={bad}", flush=True)
    flush()

    feats = np.stack(feats).astype(np.float32)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, ids=np.array(ids), feats=feats)
    print(f"[precompute] wrote {args.out}: {feats.shape}, unreadable={bad}, {time.time()-t:.1f}s")


if __name__ == "__main__":
    main()
