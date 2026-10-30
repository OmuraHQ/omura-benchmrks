#!/usr/bin/env python
"""Train the Gemma-token cross-attention reranker on AudioCaps.

Protocol
--------
* Frozen stage-1: the AudioCaps projection adapter (``audiocaps_gemma_adapter.pt``)
  over frozen EmbeddingGemma 2 pooled embeddings.  Its top-100 candidates are
  the reranking pool and its scores are the interpolation prior.
* Gemma towers stay frozen.  Only the cross-attention head is trained
  (text token states as queries, audio-projector token states as K/V).
* Train/dev split is the adapter's 90/10 split (seed 42).  Hard negatives are
  mined with the stage-1 adapter from the train split only.  Dev is used for
  early stopping and for tuning the interpolation weight ``alpha``.
* Test is never touched here.

Usage::

    CUDA_VISIBLE_DEVICES=3 .venv/bin/python benchmarks/train_crossattn.py
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from bench_common import REPO_ROOT, git_commit, utc_now, write_result
from crossattn_common import (
    CROSS_CACHE,
    HEAD_PATH,
    RerankerConfig,
    CrossAttnScorer,
    DeviceTokens,
    load_adapter,
    load_token_cache,
    mix_scores,
    first_hit_ranks,
    pad_batch,
    project,
    resolve_alpha,
    save_head,
    score_pairs,
    score_pairs_train,
    zscore_rows,
)

AUDIOCAPS_TEST = REPO_ROOT / "data" / "audiocaps"
ADAPTER_CACHE = REPO_ROOT / "data" / "audiocaps_gemma_cache"
HARD_NEGS = CROSS_CACHE / "audiocaps_hard_negatives.npz"
TRAIN_LOG = REPO_ROOT / "benchmarks" / "results" / "audiocaps_crossattn_train.json"


# ── stage-1 adapter embeddings + hard negatives ─────────────────────────────


def adapter_embeddings() -> Dict[str, np.ndarray]:
    ckpt, audio_head, text_head = load_adapter()
    print(f"[train] adapter best_epoch={ckpt.get('best_epoch')} hidden={ckpt.get('arch', {}).get('hidden')}")
    out = {
        "audio": project(np.load(ADAPTER_CACHE / "train_audio.npy"), audio_head),
        "text_query": project(np.load(ADAPTER_CACHE / "train_text_query.npy"), text_head),
        "text_doc": project(np.load(ADAPTER_CACHE / "train_text_doc.npy"), text_head),
    }
    return out


def _topk_rows(query: np.ndarray, query_ids: np.ndarray, cand_emb: np.ndarray, cand_ids: np.ndarray, k: int, exclude_self: bool) -> np.ndarray:
    n = len(query)
    out = np.empty((n, min(k, len(cand_emb))), dtype=np.int32)
    c = np.asarray(cand_emb, dtype=np.float32)
    ids = np.asarray(cand_ids, dtype=np.int64)
    qids = np.asarray(query_ids, dtype=np.int64)
    for start in range(0, n, 1024):
        q = np.asarray(query[start : start + 1024], dtype=np.float32)
        sims = q @ c.T
        for r in range(len(q)):
            gi = int(qids[start + r])
            if exclude_self:
                loc = np.where(ids == gi)[0]
                if len(loc):
                    sims[r, loc[0]] = -np.inf
        idx = np.argpartition(-sims, kth=out.shape[1] - 1, axis=1)[:, : out.shape[1]]
        out[start : start + len(q)] = ids[idx]
    return out


def mine_hard_negatives(emb: Dict[str, np.ndarray], train_idx: np.ndarray, k: int = 100) -> Dict[str, np.ndarray]:
    if HARD_NEGS.exists():
        z = np.load(HARD_NEGS)
        if int(z["k"]) == k and int(z["audio_negs"].shape[0]) == len(emb["audio"]):
            print(f"[train] reuse hard negatives {HARD_NEGS.name}")
            return {"audio_negs": z["audio_negs"], "text_negs": z["text_negs"]}
    t0 = time.time()
    audio_negs = _topk_rows(emb["text_query"][train_idx], train_idx, emb["audio"][train_idx], train_idx, k, exclude_self=True)
    text_negs = _topk_rows(emb["audio"][train_idx], train_idx, emb["text_doc"][train_idx], train_idx, k, exclude_self=True)
    n_total = len(emb["audio"])
    full_a = np.zeros((n_total, audio_negs.shape[1]), dtype=np.int32)
    full_t = np.zeros((n_total, text_negs.shape[1]), dtype=np.int32)
    full_a[train_idx] = audio_negs
    full_t[train_idx] = text_negs
    np.savez_compressed(HARD_NEGS, audio_negs=full_a, text_negs=full_t, k=k, pool=len(train_idx))
    print(f"[train] mined hard negatives in {time.time() - t0:.0f}s")
    return {"audio_negs": full_a, "text_negs": full_t}


# ── training batch ──────────────────────────────────────────────────────────


def make_pairs(
    batch: np.ndarray,
    audio_negs: np.ndarray,
    text_negs: np.ndarray,
    h_audio: int,
    h_text: int,
    rng: np.random.Generator,
    neg_top: int = 50,
    mode: str = "random",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Tuple[int, int], Tuple[int, int]]:
    """Group A: query texts vs candidate audios; Group B: doc texts vs audios.

    ``random`` = in-batch positives + random hard negatives (old recipe).
    ``listwise`` = the clip's own stage-1 top-N negatives plus the positive,
    i.e. the exact reranking task (label 0 = first column).
    """
    b = len(batch)
    pa_a: List[int] = []
    pa_t: List[int] = []
    pb_a: List[int] = []
    pb_t: List[int] = []
    if mode == "listwise":
        for i in batch:
            negs = audio_negs[i][: max(h_audio, 100)]
            sel = rng.choice(negs, size=min(h_audio, len(negs)), replace=False)
            auds = np.concatenate([[int(i)], sel]).astype(np.int64)
            pa_a.extend(auds.tolist())
            pa_t.extend([int(i)] * len(auds))
        for j in batch:
            negs = text_negs[j][: max(h_text, 100)]
            sel = rng.choice(negs, size=min(h_text, len(negs)), replace=False)
            docs = np.concatenate([[int(j)], sel]).astype(np.int64)
            pb_t.extend(docs.tolist())
            pb_a.extend([int(j)] * len(docs))
        return (
            np.asarray(pa_t, dtype=np.int64),
            np.asarray(pa_a, dtype=np.int64),
            np.asarray(pb_t, dtype=np.int64),
            np.asarray(pb_a, dtype=np.int64),
            (b, 1 + h_audio),
            (b, 1 + h_text),
        )
    for i in batch:
        negs = audio_negs[i][:neg_top]
        sel = rng.choice(negs, size=min(h_audio, len(negs)), replace=False)
        auds = np.concatenate([batch, sel]).astype(np.int64)
        pa_a.extend(auds.tolist())
        pa_t.extend([int(i)] * len(auds))
    for j in batch:
        negs = text_negs[j][:neg_top]
        sel = rng.choice(negs, size=min(h_text, len(negs)), replace=False)
        docs = np.concatenate([batch, sel]).astype(np.int64)
        pb_t.extend(docs.tolist())
        pb_a.extend([int(j)] * len(docs))
    return (
        np.asarray(pa_t, dtype=np.int64),
        np.asarray(pa_a, dtype=np.int64),
        np.asarray(pb_t, dtype=np.int64),
        np.asarray(pb_a, dtype=np.int64),
        (b, b + h_audio),
        (b, b + h_text),
    )


def evaluate_dev(
    scorer: CrossAttnScorer,
    tok: Dict[str, Any],
    audio_dev,
    dev_idx: np.ndarray,
    stage1_audio: np.ndarray,
    stage1_textq: np.ndarray,
    alpha: float = 0.5,
    max_queries: int | None = None,
    pool_k: int = 100,
) -> Dict[str, Any]:
    """Rerank top-``pool_k`` of the dev-split pool; returns metrics + scores."""
    qidx = dev_idx if max_queries is None else dev_idx[:max_queries]
    audio_pool = dev_idx
    sims = stage1_textq[qidx] @ stage1_audio[audio_pool].T  # (Q, P)
    k = min(pool_k, sims.shape[1])
    cand = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
    pair_t, pair_a = [], []
    for r in range(len(qidx)):
        pair_t.extend([int(qidx[r])] * k)
        pair_a.extend(audio_pool[cand[r]].tolist())
    pair_t = np.asarray(pair_t, dtype=np.int64)
    pair_a = np.asarray(pair_a, dtype=np.int64)
    cross = score_pairs(
        scorer,
        tok["text_query"], np.asarray(tok["text_query_lengths"]),
        audio_dev,
        pair_t, pair_a, chunk=512,
    ).reshape(len(qidx), k)
    # positive column: the query's own clip inside the dev pool
    pool_pos = {int(v): p for p, v in enumerate(audio_pool)}
    pos_col = np.array([pool_pos[int(i)] for i in qidx], dtype=np.int64)
    pos_hit = np.full(len(qidx), k, dtype=np.int64)
    for r in range(len(qidx)):
        hit = np.where(cand[r] == pos_col[r])[0]
        if len(hit):
            pos_hit[r] = int(hit[0])
    s1 = np.take_along_axis(sims, cand, axis=1)

    def _ranks(alpha: float) -> Tuple[np.ndarray, np.ndarray]:
        mixed = mix_scores(s1, cross, alpha)
        ranks = np.full(len(qidx), k + 1, dtype=np.int32)
        base = np.full(len(qidx), k + 1, dtype=np.int32)
        for r in range(len(qidx)):
            hit = np.where(cand[r] == pos_col[r])[0]
            if len(hit) == 0:
                continue
            p = hit[0]
            ranks[r] = int(np.sum(mixed[r] > mixed[r, p])) + 1
            base[r] = int(np.sum(s1[r] > s1[r, p])) + 1
        return ranks, base

    ranks, base_ranks = _ranks(alpha)
    ranks_head, _ = _ranks(0.0)
    metrics = {
        "stage1_top_pool": {f"R@{t}": float(np.mean(base_ranks <= t)) for t in (1, 5, 10)},
        "reranked": {f"R@{t}": float(np.mean(ranks <= t)) for t in (1, 5, 10)},
        "head_only": {f"R@{t}": float(np.mean(ranks_head <= t)) for t in (1, 5, 10)},
    }
    return {
        "metrics": metrics,
        "stage1": s1,
        "cross": cross,
        "cand": cand,
        "qidx": qidx,
        "audio_pool": audio_pool,
        "pos_col": pos_col,
        "pos_hit": pos_hit,
        "alpha": alpha,
        "k": k,
    }


def tune_alpha_on_dev(dev: Dict[str, Any], grid: np.ndarray | None = None) -> Tuple[float, List[Dict[str, float]]]:
    grid = list(grid if grid is not None else np.round(np.arange(0.0, 1.0001, 0.1), 2))
    s1, cross, pos_hit, k = dev["stage1"], dev["cross"], dev["pos_hit"], dev["k"]
    rows: List[Dict[str, float]] = []
    best = (-1.0, 1.0)
    for a in grid:
        mixed = mix_scores(s1, cross, float(a))
        ranks = np.full(len(pos_hit), k + 1, dtype=np.int32)
        for r, p in enumerate(pos_hit):
            if p >= k:
                continue
            ranks[r] = int(np.sum(mixed[r] > mixed[r, p])) + 1
        r10 = float(np.mean(ranks <= 10))
        rows.append({"alpha": float(a), "R@1": float(np.mean(ranks <= 1)), "R@10": r10})
        if r10 > best[0] + 1e-9:
            best = (r10, float(a))
    return best[1], rows


# ── main ────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--h-audio", type=int, default=12)
    ap.add_argument("--h-text", type=int, default=4)
    ap.add_argument("--pair-chunk", type=int, default=256)
    ap.add_argument("--dev-queries", type=int, default=1500)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--token-view", choices=("projector", "lm"), default="projector")
    ap.add_argument("--neg-top", type=int, default=50)
    ap.add_argument("--neg-mode", choices=("random", "listwise"), default="random")
    ap.add_argument("--audio-stride", type=int, default=1)
    ap.add_argument("--out", type=Path, default=HEAD_PATH)
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = "cuda"

    tok = load_token_cache("audiocaps_train")
    # Text states are random-accessed every step; keep the 2 x 5 GB caches in
    # host RAM instead of faulting pages from the on-disk memmap.
    tok["text_query"] = np.array(tok["text_query"])
    tok["text_doc"] = np.array(tok["text_doc"])
    if args.token_view == "lm":
        from build_crossattn_lm_cache import LM_CACHE

        lm_path = LM_CACHE / "audiocaps_train"
        tok["audio"] = np.load(lm_path / "audio_lm_tok.npy", mmap_mode="r")
        tok["audio_lengths"] = json.loads((lm_path / "meta.json").read_text())["audio_lm_lengths"]
        audio_in = 768
        print(f"[train] using LM-contextualised audio tokens {tok['audio'].shape}")
    else:
        audio_in = 512
    train_idx = np.asarray(tok["train_idx"], dtype=np.int64)
    dev_idx = np.asarray(tok["dev_idx"], dtype=np.int64)
    print(f"[train] train={len(train_idx)} dev={len(dev_idx)} audio_tok={tok['audio'].shape} text_tok={tok['text_query'].shape}")
    audio_dev = DeviceTokens(tok["audio"], tok["audio_lengths"], device=device, pool_stride=args.audio_stride)
    print(f"[train] audio tokens resident on GPU {audio_dev.tok.shape} {audio_dev.tok.dtype}")

    emb = adapter_embeddings()
    print("[train] adapter embeddings ready")
    negs = mine_hard_negatives(emb, train_idx)

    # validate dev pool baseline once
    dev_sims = emb["text_query"][dev_idx] @ emb["audio"][dev_idx].T
    dev_base_ranks = first_hit_ranks(dev_sims, [np.asarray([i]) for i in range(len(dev_idx))])
    dev_base = {f"R@{k}": float(np.mean(dev_base_ranks <= k)) for k in (1, 5, 10)}
    print("[train] dev pool baseline", dev_base)

    cfg = RerankerConfig(layers=args.layers, heads=args.heads, dropout=args.dropout, audio_in=audio_in)
    scorer = CrossAttnScorer(cfg).to(device)
    opt = torch.optim.AdamW(scorer.net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = max(1, len(train_idx) // args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    warmup = max(1, int(0.05 * total_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total_steps)))
    )

    best = (-1.0, None, 0)
    history: List[Dict[str, Any]] = []
    t_start = time.time()
    step = 0
    for epoch in range(1, args.epochs + 1):
        scorer.train()
        perm = rng.permutation(train_idx)
        losses = []
        t_ep = time.time()
        for si in range(steps_per_epoch):
            if args.max_steps and step >= args.max_steps:
                break
            batch = perm[si * args.batch_size : (si + 1) * args.batch_size]
            if len(batch) < 4:
                continue
            pt_a, pa_a, pt_b, pa_b, shp_a, shp_b = make_pairs(
                batch, negs["audio_negs"], negs["text_negs"], args.h_audio, args.h_text, rng
            )
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                sa = score_pairs_train(
                    scorer,
                    tok["text_query"], np.asarray(tok["text_query_lengths"]),
                    audio_dev, pt_a, pa_a, chunk=args.pair_chunk,
                )
                sb = score_pairs_train(
                    scorer,
                    tok["text_doc"], np.asarray(tok["text_doc_lengths"]),
                    audio_dev, pt_b, pa_b, chunk=args.pair_chunk,
                )
                la = sa.float().reshape(shp_a) * scorer.net.logit_scale.exp().clamp(max=100.0)
                lb = sb.float().reshape(shp_b) * scorer.net.logit_scale.exp().clamp(max=100.0)
                if args.neg_mode == "listwise":
                    target = torch.zeros(shp_a[0], dtype=torch.long, device=la.device)
                else:
                    target = torch.arange(shp_a[0], device=la.device)
                loss = 0.5 * (F.cross_entropy(la, target) + F.cross_entropy(lb, target))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(scorer.net.parameters(), 1.0)
            opt.step()
            sched.step()
            losses.append(float(loss.detach()))
            step += 1
            if step % 100 == 0:
                print(
                    f"[train] ep{epoch} step{step} loss={np.mean(losses[-100:]):.4f} "
                    f"scale={float(scorer.net.logit_scale.exp()):.1f} "
                    f"{time.time() - t_ep:.0f}s",
                    flush=True,
                )
        if args.max_steps and step >= args.max_steps:
            print("[train] max steps reached")
            break

        scorer.eval()
        ev = evaluate_dev(
            scorer, tok, audio_dev, dev_idx, emb["audio"], emb["text_query"],
            alpha=0.5, max_queries=args.dev_queries,
        )
        r10 = ev["metrics"]["reranked"]["R@10"]
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)) if losses else None,
                "dev_reranked": ev["metrics"]["reranked"],
                "dev_head_only": ev["metrics"]["head_only"],
                "dev_stage1_top_pool": ev["metrics"]["stage1_top_pool"],
                "seconds": time.time() - t_ep,
            }
        )
        print(
            f"[train] epoch {epoch}: loss={np.mean(losses):.4f} "
            f"dev rerank R@10={r10:.4f} (head-only {ev['metrics']['head_only']['R@10']:.4f}, "
            f"stage1 {ev['metrics']['stage1_top_pool']['R@10']:.4f}) "
            f"{time.time() - t_ep:.0f}s",
            flush=True,
        )
        if r10 > best[0]:
            best = (r10, {k: v.cpu().clone() for k, v in scorer.state_dict().items()}, epoch)
            # Periodically persist the current best head so a long run can be
            # stopped early without losing the checkpoint (alpha finalised below).
            save_head(
                args.out,
                scorer,
                {
                    "stage": "cross_attention_adapter",
                    "arch": cfg.to_dict(),
                    "token_view": args.token_view,
                    "audio_token_stride": args.audio_stride,
                    "alpha": 0.5,
                    "train_size": int(len(train_idx)),
                    "dev_size": int(len(dev_idx)),
                    "seed": args.seed,
                    "commit": git_commit(),
                    "utc_date": utc_now(),
                    "best_epoch": epoch,
                    "history": history,
                    "normalization": {"note": "interim checkpoint; alpha not yet dev-tuned"},
                },
            )

    if best[1] is None:
        # dry-run / max-steps mode: evaluate the current head
        best = (-1.0, {k: v.cpu().clone() for k, v in scorer.state_dict().items()}, 0)
    scorer.load_state_dict(best[1])
    scorer.eval()
    print(f"[train] best epoch {best[2]} dev-monitor R@10={best[0]:.4f}")

    # Full dev pass: early-stop checkpoint, alpha tuned on dev only.
    dev_full = evaluate_dev(scorer, tok, audio_dev, dev_idx, emb["audio"], emb["text_query"], alpha=0.5, max_queries=None)
    alpha, grid = tune_alpha_on_dev(dev_full)
    dev_full_a = evaluate_dev(scorer, tok, audio_dev, dev_idx, emb["audio"], emb["text_query"], alpha=alpha, max_queries=None)
    print(f"[train] dev alpha={alpha} grid={grid}")
    print(f"[train] dev reranked {dev_full_a['metrics']['reranked']}")

    payload = {
        "stage": "cross_attention_adapter",
        "model": "google/embeddinggemma-2 (frozen) + audiocaps adapter (stage-1) + cross-attn head",
        "arch": cfg.to_dict(),
        "token_view": args.token_view,
        "neg_mode": args.neg_mode,
        "audio_token_stride": args.audio_stride,
        "alpha": float(alpha),
        "alpha_grid": grid,
        "normalization": {
            "stage1": "per-query z-score over the candidate pool",
            "cross_attn": "per-query z-score over the candidate pool",
            "mix": "alpha * z(stage1) + (1-alpha) * z(cross_attn)",
            "dev_stage1_mean": float(dev_full["stage1"].mean()),
            "dev_stage1_std": float(dev_full["stage1"].std()),
            "dev_cross_mean": float(dev_full["cross"].mean()),
            "dev_cross_std": float(dev_full["cross"].std()),
        },
        "candidate_pool": 100,
        "train_size": int(len(train_idx)),
        "dev_size": int(len(dev_idx)),
        "seed": args.seed,
        "commit": git_commit(),
        "utc_date": utc_now(),
        "best_epoch": best[2],
        "history": history,
        "dev_pool_baseline": dev_base,
        "dev_pool_reranked": dev_full_a["metrics"]["reranked"],
        "dev_pool_stage1_at_k": dev_full_a["metrics"]["stage1_top_pool"],
        "stage1_adapter": "data/audiocaps_gemma_adapter.pt",
        "hard_negatives": {
            "source": "stage-1 adapter top-100 over the 40660 train split",
            "audio_per_query": args.h_audio,
            "text_per_clip": args.h_text,
            "sampled_from_top": 50,
        },
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "dropout": args.dropout,
            "loss": "symmetric InfoNCE (query-text rows + doc-text rows), learnable temperature",
            "dev_monitor_queries": args.dev_queries,
        },
        "notes": (
            "Gemma audio tower + audio projector and the text language model are frozen; "
            "only the cross-attention head is trained. Audio tokens are 25 Hz 512-d "
            "embed_audio outputs; text tokens are 768-d language-model states. "
            "Test never used for training or model selection."
        ),
    }
    save_head(args.out, scorer, payload)
    write_result(TRAIN_LOG, {k: v for k, v in payload.items() if k != "history"} | {"history": history})
    print(f"[train] saved {args.out} ({time.time() - t_start:.0f}s total)")


if __name__ == "__main__":
    main()
