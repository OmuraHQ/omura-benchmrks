#!/usr/bin/env python
"""Rerank AudioCaps / Clotho with the Gemma-token cross-attention head.

Stage-1 is the frozen AudioCaps projection adapter over frozen
EmbeddingGemma 2 pooled vectors (the published baseline).  The reranker scores
(text token states -> audio projector token states) and is interpolated with
the stage-1 cosine after a per-query z-score over the candidate pool.

No tuning happens here: ``alpha`` and the head come from
``data/audiocaps_crossattn.pt`` (``alpha`` selected on AudioCaps dev only).

Usage::

    CUDA_VISIBLE_DEVICES=3 .venv/bin/python benchmarks/benchmark_crossattn_rerank.py --dataset audiocaps
    CUDA_VISIBLE_DEVICES=3 .venv/bin/python benchmarks/benchmark_crossattn_rerank.py --dataset clotho
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from bench_common import REPO_ROOT, git_commit, utc_now, write_result
from crossattn_common import (
    ADAPTER_CACHE,
    CLOTHO_CACHE,
    DOCUMENT_PREFIX,
    HEAD_PATH,
    QUERY_PREFIX,
    DeviceTokens,
    load_adapter,
    load_head,
    load_token_cache,
    mix_scores,
    project,
    score_pairs,
    zscore_rows,
)

RESULTS = REPO_ROOT / "benchmarks" / "results"
ADAPTER_BASELINE = RESULTS / "audiocaps_gemma_finetuned.json"
CLOTHO_BASELINE = RESULTS / "clotho_gemma_adapter.json"
CLOTHO_FROZEN = RESULTS / "clotho_gemma.json"
TEST_MANIFEST = REPO_ROOT / "data" / "audiocaps" / "manifest.json"


def _recall(ranks: np.ndarray, ks: Sequence[int] = (1, 5, 10)) -> Dict[str, float]:
    return {f"R@{k}": float(np.mean(ranks <= k)) for k in ks}


def _attach_audio_view(tok: Dict[str, Any], split: str, view: str) -> None:
    if view == "projector":
        return
    from build_crossattn_lm_cache import LM_CACHE

    meta = json.loads((LM_CACHE / split / "meta.json").read_text())
    tok["audio"] = np.load(LM_CACHE / split / "audio_lm_tok.npy", mmap_mode="r")
    tok["audio_lengths"] = meta["audio_lm_lengths"]


def _cand_ranks(final: np.ndarray, cand: np.ndarray, positives: List[np.ndarray], k: int) -> np.ndarray:
    ranks = np.full(len(cand), k + 1, dtype=np.int32)
    for r in range(len(cand)):
        hit = np.intersect1d(cand[r], positives[r], assume_unique=False)
        if len(hit) == 0:
            continue
        cols = np.array([np.where(cand[r] == h)[0][0] for h in hit])
        best = float(final[r, cols].max())
        ranks[r] = int(np.sum(final[r] > best)) + 1
    return ranks


def _pair_scores(
    scorer,
    tok: Dict[str, Any],
    audio_dev,
    stage1: np.ndarray,
    k: int,
    text_key: str,
    audio_is_candidate: bool,
    positives: List[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Rerank the top-``k`` stage-1 candidates for every query row."""
    cand = np.argpartition(-stage1, kth=min(k, stage1.shape[1]) - 1, axis=1)[:, :k]
    q = len(cand)
    pair_t = np.empty(q * k, dtype=np.int64)
    pair_a = np.empty(q * k, dtype=np.int64)
    if audio_is_candidate:
        # text query row -> candidate audio column
        for r in range(q):
            pair_t[r * k : (r + 1) * k] = r
            pair_a[r * k : (r + 1) * k] = cand[r]
    else:
        # audio query row -> candidate text column; audio index is the row
        for r in range(q):
            pair_a[r * k : (r + 1) * k] = r
            pair_t[r * k : (r + 1) * k] = cand[r]
    cross = score_pairs(
        scorer,
        tok[text_key], np.asarray(tok["text_query_lengths"] if text_key == "text_query" else tok["text_doc_lengths"]),
        audio_dev,
        pair_t, pair_a, chunk=512,
    ).reshape(q, k)
    s1 = np.take_along_axis(stage1, cand, axis=1)
    return s1, cross


def _leakage_audiocaps() -> Dict[str, Any]:
    train_meta = json.loads((ADAPTER_CACHE / "train_meta.json").read_text())
    manifest = json.loads(TEST_MANIFEST.read_text())
    test_keys = {(c["youtube_id"], int(c["start_time"])) for c in manifest["clips"]}
    test_ids = {int(c["audiocap_id"]) for c in manifest["clips"]}
    test_yt = {c["youtube_id"] for c in manifest["clips"]}
    train_keys = {(str(k[0]), int(k[1])) for k in train_meta["keys"]}
    train_ids = {int(x) for x in train_meta["ids"]}
    train_yt = {str(k[0]) for k in train_meta["keys"]}
    return {
        "train_clips": len(train_meta["ids"]),
        "test_clips": len(manifest["clips"]),
        "train_test_clip_disjoint": len(train_keys & test_keys) == 0,
        "train_test_audiocap_id_disjoint": len(train_ids & test_ids) == 0,
        "train_test_youtube_id_disjoint": len(train_yt & test_yt) == 0,
        "overlap_clip_keys": len(train_keys & test_keys),
        "overlap_audiocap_ids": len(train_ids & test_ids),
        "overlap_youtube_ids": len(train_yt & test_yt),
        "test_captions_used_as_training_documents": False,
        "crossattn_train_source": "AudioCaps train split only",
        "note": "Reranker trains on AudioCaps train; AudioCaps test is queried once. Clotho is never used for training or alpha selection.",
    }


def _leakage_clotho() -> Dict[str, Any]:
    return {
        "clotho_used_for_training": False,
        "clotho_used_for_alpha_selection": False,
        "crossattn_train_source": "AudioCaps train split only",
        "alpha_source": "AudioCaps dev split (seed 42)",
        "note": "Clotho v2.1 test is a held-out cross-dataset evaluation; the head and alpha are applied frozen.",
    }


# ── AudioCaps ───────────────────────────────────────────────────────────────


def run_audiocaps(scorer, ckpt, args) -> Dict[str, Any]:
    tok = load_token_cache("audiocaps_test")
    tok["text_query"] = np.array(tok["text_query"])
    tok["text_doc"] = np.array(tok["text_doc"])
    _attach_audio_view(tok, "audiocaps_test", args.token_view)
    audio_dev = DeviceTokens(tok["audio"], tok["audio_lengths"], device=args.device, pool_stride=args.audio_stride)
    z = np.load(ADAPTER_CACHE / "test_emb.npz")
    _, audio_head, text_head = load_adapter()
    audio_p = project(z["audio"], audio_head)
    tq_p = project(z["text_query"], text_head)
    td_p = project(z["text_doc"], text_head)
    clip_index = np.asarray(z["clip_index"], dtype=np.int64)
    alpha = float(ckpt["alpha"])

    manifest = json.loads(TEST_MANIFEST.read_text())
    n_clips = len(manifest["clips"])
    n_q = len(clip_index)

    # stage-1 baselines
    sims_t2a = tq_p @ audio_p.T
    sims_a2t = audio_p @ td_p.T
    t2a_pos = [np.asarray([int(c)], dtype=np.int64) for c in clip_index]
    caps_by_clip: List[List[int]] = [[] for _ in range(n_clips)]
    for j, c in enumerate(clip_index):
        caps_by_clip[int(c)].append(j)
    a2t_pos = [np.asarray(v, dtype=np.int64) for v in caps_by_clip]

    out: Dict[str, Any] = {
        "model": "google/embeddinggemma-2 (frozen towers) + audiocaps adapter (stage-1) + cross-attn reranker",
        "dim": 768,
        "backend": "omuragemma",
        "protocol": "audiocaps_test_rerank_stage1_topk_t2a_and_a2t",
        "dataset": "AudioCaps test (OpenSound/AudioCaps)",
        "pool_size": int(n_clips),
        "num_queries": int(n_q),
        "num_audio_queries": int(n_clips),
        "num_caption_candidates": int(len(td_p)),
        "train_size": int(ckpt["train_size"]),
        "dev_size": int(ckpt["dev_size"]),
        "seed": 42,
        "commit": git_commit(),
        "utc_date": utc_now(),
        "alpha": alpha,
        "normalization": ckpt["normalization"],
        "arch": ckpt["arch"],
        "token_view": ckpt.get("token_view", "projector"),
        "audio_token_stride": ckpt.get("audio_token_stride", 1),
        "crossattn_head": str(HEAD_PATH.relative_to(REPO_ROOT)),
        "best_epoch": ckpt["best_epoch"],
        "leakage_check": _leakage_audiocaps(),
        "frozen_adapter_baseline": {
            "file": str(ADAPTER_BASELINE.relative_to(REPO_ROOT)),
            "text_to_audio": json.loads(ADAPTER_BASELINE.read_text())["text_to_audio"],
            "audio_to_text": json.loads(ADAPTER_BASELINE.read_text())["audio_to_text"],
        },
        "stage1_full_pool": {
            "text_to_audio": _recall(np.array([int(np.sum(sims_t2a[i] > sims_t2a[i, p[0]])) + 1 for i, p in enumerate(t2a_pos)])),
            "audio_to_text": _recall(np.array([int(np.sum(sims_a2t[i] > sims_a2t[i, p].max())) + 1 for i, p in enumerate(a2t_pos)])),
        },
        "runs": {},
        "per_query_ranks": {},
    }

    for k in args.pools:
        cand_t = np.argpartition(-sims_t2a, kth=k - 1, axis=1)[:, :k]
        s1, cross = _pair_scores(scorer, tok, audio_dev, sims_t2a, k, "text_query", True, t2a_pos)
        final = mix_scores(s1, cross, alpha)
        ranks = _cand_ranks(final, cand_t, t2a_pos, k)
        ceiling = float(np.mean([np.isin(t2a_pos[i], cand_t[i]).any() for i in range(n_q)]))
        cand_a = np.argpartition(-sims_a2t, kth=k - 1, axis=1)[:, :k]
        s1_a, cross_a = _pair_scores(scorer, tok, audio_dev, sims_a2t, k, "text_doc", False, a2t_pos)
        final_a = mix_scores(s1_a, cross_a, alpha)
        ranks_a = _cand_ranks(final_a, cand_a, a2t_pos, k)
        out["runs"][f"top{k}"] = {
            "candidate_pool": k,
            "text_to_audio": _recall(ranks),
            "audio_to_text": _recall(ranks_a),
            "text_to_audio_pool_ceiling": ceiling,
            "text_to_audio_stage1_at_k": _recall(_cand_ranks(s1, cand_t, t2a_pos, k)),
            "audio_to_text_stage1_at_k": _recall(_cand_ranks(s1_a, cand_a, a2t_pos, k)),
        }
        out["per_query_ranks"][f"top{k}"] = {
            "text_to_audio": [
                {"audiocap_id": int(manifest["clips"][int(i)]["audiocap_id"]), "rank": int(r)}
                for i, r in zip(clip_index, ranks)
            ],
            "audio_to_text": [
                {"audiocap_id": int(c["audiocap_id"]), "rank": int(r)}
                for c, r in zip(manifest["clips"], ranks_a)
            ],
        }
        print(f"[rerank] audiocaps top{k} t2a={out['runs'][f'top{k}']['text_to_audio']} a2t={out['runs'][f'top{k}']['audio_to_text']}", flush=True)
    return out


# ── Clotho ──────────────────────────────────────────────────────────────────


def run_clotho(scorer, ckpt, args) -> Dict[str, Any]:
    tok = load_token_cache("clotho_test")
    tok["text_query"] = np.array(tok["text_query"])
    tok["text_doc"] = np.array(tok["text_doc"])
    _attach_audio_view(tok, "clotho_test", args.token_view)
    audio_dev = DeviceTokens(tok["audio"], tok["audio_lengths"], device=args.device, pool_stride=args.audio_stride)
    z = np.load(CLOTHO_CACHE / "gemma_frozen.npz")
    _, audio_head, text_head = load_adapter()
    audio_p = project(z["audio"], audio_head)
    tq_p = project(z["text_query"], text_head)
    td_p = project(z["text_doc"], text_head)
    clip_index = np.asarray(tok["clip_index"], dtype=np.int64)
    n_clips = audio_p.shape[0]
    n_q = tq_p.shape[0]
    alpha = float(ckpt["alpha"])

    sims_t2a = tq_p @ audio_p.T
    sims_a2t = audio_p @ td_p.T
    t2a_pos = [np.asarray([int(c)], dtype=np.int64) for c in clip_index]
    caps_by_clip: List[List[int]] = [[] for _ in range(n_clips)]
    for j, c in enumerate(clip_index):
        caps_by_clip[int(c)].append(j)
    a2t_pos = [np.asarray(v, dtype=np.int64) for v in caps_by_clip]

    out: Dict[str, Any] = {
        "model": "google/embeddinggemma-2 (frozen towers) + audiocaps adapter (stage-1) + cross-attn reranker",
        "dim": 768,
        "backend": "omuragemma",
        "protocol": "clotho_v2.1_test_rerank_stage1_top100_t2a_and_a2t",
        "dataset": "Clotho v2.1 test (mteb/clotho_t2a_v2)",
        "pool_size": int(n_clips),
        "text_pool_size": int(n_q),
        "num_queries": int(n_q),
        "num_audio_queries": int(n_clips),
        "caption_count": int(n_q),
        "caption_count_per_clip": 5,
        "train_size": int(ckpt["train_size"]),
        "dev_size": int(ckpt["dev_size"]),
        "seed": 42,
        "commit": git_commit(),
        "utc_date": utc_now(),
        "alpha": alpha,
        "normalization": ckpt["normalization"],
        "arch": ckpt["arch"],
        "token_view": ckpt.get("token_view", "projector"),
        "audio_token_stride": ckpt.get("audio_token_stride", 1),
        "crossattn_head": str(HEAD_PATH.relative_to(REPO_ROOT)),
        "best_epoch": ckpt["best_epoch"],
        "leakage_check": _leakage_clotho(),
        "frozen_adapter_baseline": {
            "file": str(CLOTHO_BASELINE.relative_to(REPO_ROOT)),
            "text_to_audio": json.loads(CLOTHO_BASELINE.read_text())["text_to_audio"],
            "audio_to_text": json.loads(CLOTHO_BASELINE.read_text())["audio_to_text"],
        },
        "frozen_no_adapter_baseline": {
            "file": str(CLOTHO_FROZEN.relative_to(REPO_ROOT)),
            "text_to_audio": json.loads(CLOTHO_FROZEN.read_text())["text_to_audio"],
            "audio_to_text": json.loads(CLOTHO_FROZEN.read_text())["audio_to_text"],
        },
        "runs": {},
        "per_query_ranks": {},
    }
    for k in args.pools:
        cand = np.argpartition(-sims_t2a, kth=k - 1, axis=1)[:, :k]
        s1, cross = _pair_scores(scorer, tok, audio_dev, sims_t2a, k, "text_query", True, t2a_pos)
        final = mix_scores(s1, cross, alpha)
        ranks = _cand_ranks(final, cand, t2a_pos, k)
        ceiling = float(np.mean([np.isin(t2a_pos[i], cand[i]).any() for i in range(n_q)]))
        cand_a = np.argpartition(-sims_a2t, kth=k - 1, axis=1)[:, :k]
        s1_a, cross_a = _pair_scores(scorer, tok, audio_dev, sims_a2t, k, "text_doc", False, a2t_pos)
        final_a = mix_scores(s1_a, cross_a, alpha)
        ranks_a = _cand_ranks(final_a, cand_a, a2t_pos, k)
        out["runs"][f"top{k}"] = {
            "candidate_pool": k,
            "text_to_audio": _recall(ranks),
            "audio_to_text": _recall(ranks_a),
            "text_to_audio_pool_ceiling": ceiling,
            "text_to_audio_stage1_at_k": _recall(_cand_ranks(s1, cand, t2a_pos, k)),
            "audio_to_text_stage1_at_k": _recall(_cand_ranks(s1_a, cand_a, a2t_pos, k)),
        }
        out["per_query_ranks"][f"top{k}"] = {
            "text_to_audio": [{"query_id": str(tok["query_ids"][i]), "rank": int(r)} for i, r in enumerate(ranks)],
            "audio_to_text": [{"clip_id": str(tok["ids"][i]), "rank": int(r)} for i, r in enumerate(ranks_a)],
        }
        print(f"[rerank] clotho top{k} t2a={out['runs'][f'top{k}']['text_to_audio']} a2t={out['runs'][f'top{k}']['audio_to_text']}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=("audiocaps", "clotho"), required=True)
    ap.add_argument("--pools", type=int, nargs="+", default=None)
    ap.add_argument("--head", type=Path, default=HEAD_PATH)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--token-view", choices=("projector", "lm"), default="projector")
    ap.add_argument("--audio-stride", type=int, default=1)
    args = ap.parse_args()
    args.pools = args.pools or ([100, 200] if args.dataset == "audiocaps" else [100])
    out_path = args.out or (
        RESULTS / f"audiocaps_gemma_crossattn.json"
        if args.dataset == "audiocaps"
        else RESULTS / f"clotho_gemma_crossattn.json"
    )
    scorer, ckpt = load_head(args.head)
    args.token_view = ckpt.get("token_view", args.token_view)
    args.audio_stride = int(ckpt.get("audio_token_stride", args.audio_stride))
    payload = run_audiocaps(scorer, ckpt, args) if args.dataset == "audiocaps" else run_clotho(scorer, ckpt, args)
    write_result(out_path, payload)
    print(f"[rerank] wrote {out_path}")


if __name__ == "__main__":
    main()
