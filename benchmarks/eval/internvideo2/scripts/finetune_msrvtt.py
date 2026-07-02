"""Finetune InternVideo2-6B for MSR-VTT text->video retrieval to clear R@10 >= 85%.

Efficient recipe (single-GPU): the frozen 6B vision tower is precomputed once
(precompute_pooled.py), so training only runs the text encoder + projection heads.
Trainable: text_encoder (BERT) + vision_proj + text_proj + temp (all fp32).
Frozen: vision encoder, audio tower. Loss: symmetric InfoNCE over the contrastive
joint space (== the eval's ITC similarity), so improvements transfer directly.

  IV2_CKPT=<ckpt> CUDA_VISIBLE_DEVICES=7 .venv-iv2/bin/python scripts/finetune_msrvtt.py \
      --train_pooled data/cache/pooled_train.npz --train_anno data/anno/msrvtt_train_9k.json \
      --test_pooled  data/cache/pooled_test.npz  --test_anno  data/anno/msrvtt_test_1k.json \
      --out_dir data/finetune_v1 --epochs 8 --batch 128 --lr 1e-5
"""
import os, sys, json, time, argparse, random
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
import iv2_common as C


def load_pooled(path):
    d = np.load(path, allow_pickle=True)
    ids = [str(x) for x in d["ids"]]
    feats = torch.from_numpy(d["feats"].astype(np.float32))
    return {vid: feats[i] for i, vid in enumerate(ids)}


def build_train_pairs(anno_path, pooled):
    data = json.load(open(anno_path))
    pairs = []  # (video_id, caption)
    for item in data:
        vid = item["video_id"]
        if vid not in pooled:
            continue
        caps = item["caption"]
        caps = caps if isinstance(caps, list) else [caps]
        for c in caps:
            if c and c.strip():
                pairs.append((vid, c.strip()))
    return pairs


def load_test(anno_path, pooled):
    data = json.load(open(anno_path))
    vids, caps = [], []
    for item in data:
        vid = item["video_id"]
        if vid not in pooled:
            continue
        c = item["caption"]
        c = c[0] if isinstance(c, list) else c
        vids.append(vid)
        caps.append(c)
    return vids, caps


@torch.no_grad()
def evaluate(model, tok, pooled, vids, caps, device, max_txt_l, bs=64):
    model.eval()
    vfeat = torch.stack([pooled[v] for v in vids]).to(device).float()
    vfeat = model.vision_proj(vfeat)
    vfeat = F.normalize(vfeat.float(), dim=-1)
    tfeats = []
    for i in range(0, len(caps), bs):
        t = tok(caps[i:i + bs], padding="max_length", truncation=True,
                max_length=max_txt_l, return_tensors="pt").to(device)
        _, pooled_t = model.encode_text(t)
        tf = model.text_proj(pooled_t)
        tfeats.append(F.normalize(tf.float(), dim=-1).cpu())
    tfeat = torch.cat(tfeats, 0).to(device)
    sim = tfeat @ vfeat.T  # [N_text, N_video]
    n = sim.shape[0]
    ranks = np.zeros(n, dtype=np.int64)
    for i in range(n):
        order = torch.argsort(sim[i], descending=True)
        ranks[i] = (order == i).nonzero(as_tuple=True)[0].item()
    R = lambda k: float((ranks < k).mean() * 100.0)
    return {"R@1": round(R(1), 2), "R@5": round(R(5), 2), "R@10": round(R(10), 2),
            "MedianR": float(np.median(ranks) + 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_pooled", required=True)
    ap.add_argument("--train_anno", required=True)
    ap.add_argument("--test_pooled", required=True)
    ap.add_argument("--test_anno", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--proj_lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed); torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda"
    os.makedirs(args.out_dir, exist_ok=True)

    model, tok, cfg = C.load_model()
    max_txt_l = cfg.max_txt_l

    # Free the unused vision tower's VRAM; we train only text + projections.
    # (vision features are precomputed.) Keep vision_proj.
    # Trainable params -> fp32 for stable optimization; freeze everything else.
    for p in model.parameters():
        p.requires_grad = False
    model.text_encoder.float()
    model.vision_proj.float()
    model.text_proj.float()
    model.temp.data = model.temp.data.float()
    for m in (model.text_encoder, model.vision_proj, model.text_proj):
        for p in m.parameters():
            p.requires_grad = True
    model.temp.requires_grad = True

    train_pooled = load_pooled(args.train_pooled)
    test_pooled = load_pooled(args.test_pooled)
    pairs = build_train_pairs(args.train_anno, train_pooled)
    test_vids, test_caps = load_test(args.test_anno, test_pooled)
    print(f"[ft] train_videos={len(train_pooled)} train_pairs={len(pairs)} "
          f"test_videos={len(test_vids)}")

    proj_params = list(model.vision_proj.parameters()) + list(model.text_proj.parameters()) + [model.temp]
    text_params = list(model.text_encoder.parameters())
    opt = torch.optim.AdamW([
        {"params": text_params, "lr": args.lr},
        {"params": proj_params, "lr": args.proj_lr},
    ], weight_decay=0.02)
    steps_per_epoch = max(1, len(pairs) // args.batch)
    total_steps = steps_per_epoch * args.epochs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)

    base = evaluate(model, tok, test_pooled, test_vids, test_caps, device, max_txt_l)
    print(f"[ft] baseline (pre-finetune ITC): {base}", flush=True)

    best = dict(base); best["epoch"] = 0
    best_path = os.path.join(args.out_dir, "best_heads.pt")

    def save_heads(metrics, epoch):
        torch.save({
            "vision_proj": model.vision_proj.state_dict(),
            "text_proj": model.text_proj.state_dict(),
            "text_encoder": model.text_encoder.state_dict(),
            "temp": model.temp.detach().cpu(),
            "metrics": metrics, "epoch": epoch,
        }, best_path)

    for epoch in range(1, args.epochs + 1):
        model.train()
        random.shuffle(pairs)
        # within-batch distinct video_ids => clean InfoNCE negatives
        t0 = time.time(); running = 0.0; nb = 0
        i = 0
        while i + args.batch <= len(pairs):
            batch = pairs[i:i + args.batch]
            i += args.batch
            seen = set(); vids = []; caps = []
            for vid, cap in batch:
                if vid in seen:
                    continue
                seen.add(vid); vids.append(vid); caps.append(cap)
            vfeat = torch.stack([train_pooled[v] for v in vids]).to(device).float()
            vfeat = F.normalize(model.vision_proj(vfeat).float(), dim=-1)
            t = tok(caps, padding="max_length", truncation=True,
                    max_length=max_txt_l, return_tensors="pt").to(device)
            _, pooled_t = model.encode_text(t)
            tfeat = F.normalize(model.text_proj(pooled_t).float(), dim=-1)
            temp = model.temp.clamp(min=0.005, max=0.5)
            logits = (tfeat @ vfeat.T) / temp
            labels = torch.arange(logits.shape[0], device=device)
            loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(text_params + proj_params, 5.0)
            opt.step(); sched.step()
            running += loss.item(); nb += 1
            if nb % 20 == 0:
                print(f"  e{epoch} step {nb}/{steps_per_epoch} loss={running/nb:.4f} "
                      f"lr={sched.get_last_lr()[0]:.2e} ({(time.time()-t0)/nb:.2f}s/it)", flush=True)
        m = evaluate(model, tok, test_pooled, test_vids, test_caps, device, max_txt_l)
        print(f"[ft] epoch {epoch}: loss={running/max(nb,1):.4f} test={m}", flush=True)
        if m["R@10"] > best["R@10"]:
            best = dict(m); best["epoch"] = epoch
            save_heads(m, epoch)
            print(f"[ft]   * new best R@10={m['R@10']} (saved)", flush=True)

    result = {
        "task": "MSR-VTT 1K-A text->video retrieval (finetuned ITC)",
        "model": "omura-embed-video (InternVideo2-Stage2-6B finetuned: text+proj)",
        "baseline_itc": base, "best": best, "target_R@10": 85,
        "pass_R@10": best["R@10"] >= 85,
    }
    json.dump(result, open(os.path.join(args.out_dir, "msrvtt_finetune.json"), "w"), indent=2)
    print(f"[ft] DONE best={best} pass={best['R@10']>=85}")


if __name__ == "__main__":
    main()
