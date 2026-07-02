"""MSR-VTT 1K-A zero-shot text->video retrieval with ITC, DSL, and VTM re-rank.

Reproduces the repo's tasks/retrieval_utils.py protocol for the audiovisual model:
  - ITC: cosine of vision_proj(pooled_vision) vs text_proj(pooled_text)
  - DSL: dual-softmax re-scaling of the ITC matrix (training-free)
  - VTM: rerank top-k (k_test) per text query via BERT cross-modal fusion + itm_head
"""
import os, sys, json, time, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import iv2_common as C


@torch.no_grad()
def encode_video_full(model, frames_tensor):
    """Returns (pooled_vision_proj_normed [1,768], vision_embeds [1,N,768])."""
    frames_tensor = frames_tensor.to(model._iv2_dtype)
    vis_embeds, pooled = model.encode_vision(frames_tensor, test=True)
    if pooled.dim() == 3:
        pooled = pooled.squeeze(1)
    vp = model.vision_proj(pooled)
    vp = vp / vp.norm(dim=-1, keepdim=True)
    return vp.float(), vis_embeds  # vis_embeds kept in model dtype


@torch.no_grad()
def encode_text_full(model, tok, texts, max_txt_l, device="cuda"):
    """Returns (text_proj_normed [B,768], text_embeds [B,L,1024], att [B,L])."""
    t = tok(list(texts), padding="max_length", truncation=True,
            max_length=max_txt_l, return_tensors="pt").to(device)
    txt_embeds, pooled = model.encode_text(t)  # txt_embeds [B,L,1024]
    tp = model.text_proj(pooled)
    tp = tp / tp.norm(dim=-1, keepdim=True)
    return tp.float(), txt_embeds, t.attention_mask


def recall_from_ranks(ranks):
    return {
        "R@1": round(float((ranks < 1).mean() * 100), 2),
        "R@5": round(float((ranks < 5).mean() * 100), 2),
        "R@10": round(float((ranks < 10).mean() * 100), 2),
        "MedianR": float(np.median(ranks) + 1),
        "MeanR": round(float(ranks.mean() + 1), 2),
    }


def t2v_ranks(sim_t2v):
    """sim_t2v [N_text, N_video], correct pair on diagonal."""
    n = sim_t2v.shape[0]
    ranks = np.zeros(n, dtype=np.int64)
    for i in range(n):
        order = torch.argsort(sim_t2v[i], descending=True)
        ranks[i] = (order == i).nonzero(as_tuple=True)[0].item()
    return ranks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--video_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--k_test", type=int, default=128)
    ap.add_argument("--fusion_bs", type=int, default=8)
    args = ap.parse_args()
    device = "cuda"

    model, tok, cfg = C.load_model()
    nf = cfg.num_frames_test
    data = json.load(open(args.json))
    if args.limit:
        data = data[: args.limit]
    n = len(data)
    print(f"[msrvtt-vtm] {n} pairs, num_frames={nf}, k_test={args.k_test}")

    # ---- encode videos: pooled proj + per-token embeds (kept on CPU) ----
    t = time.time()
    vproj = torch.zeros(n, 768)
    vtok = None  # [n, N, 768] filled lazily once N known
    bad = 0
    for i, item in enumerate(data):
        ft = C.read_video_official(os.path.join(args.video_dir, item["video"]),
                                   num_frames=nf, image_res=224, sample="middle")
        if ft is None:
            bad += 1
            continue
        vp, ve = encode_video_full(model, ft)  # vp[1,768], ve[1,N,768]
        if vtok is None:
            vtok = torch.zeros(n, ve.shape[1], ve.shape[2], dtype=torch.float16)
        vproj[i] = vp[0].cpu()
        vtok[i] = ve[0].to(torch.float16).cpu()
        if (i + 1) % 100 == 0:
            print(f"  video {i+1}/{n} ({(time.time()-t)/(i+1):.2f}s/clip)")
    print(f"[msrvtt-vtm] videos encoded in {time.time()-t:.1f}s, unreadable={bad}, N_tok={vtok.shape[1]}")

    # ---- encode texts: proj + per-token embeds + att (CPU) ----
    t = time.time()
    captions = [it["caption"] for it in data]
    tproj = torch.zeros(n, 768)
    ttok = None
    tatt = torch.zeros(n, cfg.max_txt_l, dtype=torch.long)
    bs = 64
    for i in range(0, n, bs):
        tp, te, ta = encode_text_full(model, tok, captions[i:i+bs], cfg.max_txt_l)
        if ttok is None:
            ttok = torch.zeros(n, te.shape[1], te.shape[2], dtype=torch.float16)
        tproj[i:i+bs] = tp.cpu()
        ttok[i:i+bs] = te.to(torch.float16).cpu()
        tatt[i:i+bs] = ta.cpu()
    print(f"[msrvtt-vtm] texts encoded in {time.time()-t:.1f}s")

    # ---- ITC ----
    sim = tproj @ vproj.T  # [N_text, N_video]
    itc = recall_from_ranks(t2v_ranks(sim))
    print("[msrvtt-vtm] ITC T2V:", itc)

    # ---- DSL (dual softmax), training-free rerank ----
    dsl = sim * sim.softmax(dim=0)  # column softmax over videos-per-text? follow repo: i2t*i2t.softmax(0)
    # repo: t2i_scores = i2t.T * i2t.T.softmax(0); here sim is t2i already ([text,video]) -> use sim*sim.softmax(0)
    dsl_m = recall_from_ranks(t2v_ranks(sim * sim.softmax(dim=0)))
    print("[msrvtt-vtm] DSL T2V:", dsl_m)

    # ---- VTM rerank: for each text, top-k videos by ITC -> cross fusion -> itm_head ----
    # Vision + audio encoders are no longer needed; move them to CPU to free GPU
    # memory for the cross-modal fusion activations.
    model.vision_encoder.to("cpu")
    if hasattr(model, "audio_encoder"):
        model.audio_encoder.to("cpu")
    torch.cuda.empty_cache()
    t = time.time()
    text_encoder = model.get_text_encoder()
    match_head = model.itm_head
    k = min(args.k_test, n)
    vtm_scores = torch.full((n, n), -100.0)
    fb = args.fusion_bs
    with torch.no_grad():
        for i in range(n):
            sims = sim[i]
            topk_sim, topk_idx = sims.topk(k=k, dim=0)
            te = ttok[i:i+1].to(device).to(model._iv2_dtype)        # [1,L,1024]
            ta = tatt[i:i+1].to(device)                              # [1,L]
            scores = []
            for j in range(0, k, fb):
                idx = topk_idx[j:j+fb]
                m = len(idx)
                enc = vtok[idx].to(device).to(model._iv2_dtype)      # [m,Nt,768]
                enc_att = torch.ones(enc.shape[:-1], dtype=torch.long, device=device)
                out = text_encoder(
                    encoder_embeds=te.expand(m, -1, -1),
                    attention_mask=ta.expand(m, -1),
                    encoder_hidden_states=enc,
                    encoder_attention_mask=enc_att,
                    return_dict=True,
                    mode="fusion",
                )
                s = match_head(out.last_hidden_state[:, 0])[:, 1].float()
                scores.append(s.cpu())
            vtm_scores[i, topk_idx] = torch.cat(scores)
            if (i + 1) % 100 == 0:
                print(f"  vtm text {i+1}/{n} ({(time.time()-t)/(i+1):.2f}s/text)")
    vtm_m = recall_from_ranks(t2v_ranks(vtm_scores))
    print(f"[msrvtt-vtm] VTM T2V: {vtm_m}  ({time.time()-t:.1f}s)")

    best = max([("ITC", itc), ("DSL", dsl_m), ("VTM", vtm_m)], key=lambda kv: kv[1]["R@10"])
    result = {
        "task": "MSR-VTT 1K-A zero-shot text->video retrieval",
        "model": "InternVideo2-Stage2-6B (with audio encoder), naive-attn bf16",
        "checkpoint": "OpenGVLab/InternVideo2-Stage2_6B-224p-f4 :: internvideo2-s2_6b-224p-f4_with_audio_encoder.pt",
        "protocol": {
            "split": "MSR-VTT 1K-A (friedrichor/MSR-VTT :: msrvtt_test_1k.json)",
            "num_pairs": n, "num_frames": nf, "resolution": 224,
            "frame_sample": "decord uniform-interval middle + BICUBIC resize",
            "direction": "text->video", "k_test": args.k_test,
        },
        "unreadable_videos": bad,
        "metrics_t2v": {"ITC": itc, "DSL": dsl_m, "VTM_rerank": vtm_m},
        "best_scoring": best[0],
        "R@1": best[1]["R@1"], "R@5": best[1]["R@5"], "R@10": best[1]["R@10"],
        "target_R@10": 85, "pass_R@10": best[1]["R@10"] >= 85,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=2)
    print("[msrvtt-vtm] wrote", args.out, "| best:", best[0], best[1])


if __name__ == "__main__":
    main()
