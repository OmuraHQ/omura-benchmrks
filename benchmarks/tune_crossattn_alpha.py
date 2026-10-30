#!/usr/bin/env python
"""Re-tune the interpolation weight alpha on AudioCaps dev for a saved head.

Used when a training run is stopped early (interim checkpoint carries
``alpha=0.5``) or to refresh dev statistics without retraining.  Test is never
read here.

Usage::

    CUDA_VISIBLE_DEVICES=3 .venv/bin/python benchmarks/tune_crossattn_alpha.py \
        --head data/audiocaps_crossattn_lm.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bench_common import git_commit, utc_now, write_result
from crossattn_common import (
    ADAPTER_CACHE,
    DeviceTokens,
    load_head,
    load_token_cache,
    save_head,
)
from train_crossattn import adapter_embeddings, evaluate_dev, tune_alpha_on_dev


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--head", type=Path, required=True)
    ap.add_argument("--audio-stride", type=int, default=None)
    args = ap.parse_args()

    scorer, ckpt = load_head(args.head)
    stride = int(ckpt.get("audio_token_stride", 1)) if args.audio_stride is None else args.audio_stride
    tok = load_token_cache("audiocaps_train")
    tok["text_query"] = np.array(tok["text_query"])
    tok["text_doc"] = np.array(tok["text_doc"])
    if ckpt.get("token_view", "projector") == "lm":
        from build_crossattn_lm_cache import LM_CACHE

        meta = json.loads((LM_CACHE / "audiocaps_train" / "meta.json").read_text())
        tok["audio"] = np.load(LM_CACHE / "audiocaps_train" / "audio_lm_tok.npy", mmap_mode="r")
        tok["audio_lengths"] = meta["audio_lm_lengths"]
    audio = DeviceTokens(tok["audio"], tok["audio_lengths"], pool_stride=stride)
    dev_idx = np.asarray(tok["dev_idx"], dtype=np.int64)
    emb = adapter_embeddings()
    dev = evaluate_dev(scorer, tok, audio, dev_idx, emb["audio"], emb["text_query"], alpha=0.5, max_queries=None)
    alpha, grid = tune_alpha_on_dev(dev)
    final = evaluate_dev(scorer, tok, audio, dev_idx, emb["audio"], emb["text_query"], alpha=alpha, max_queries=None)
    print(f"[tune] alpha={alpha} dev reranked={final['metrics']['reranked']} head_only={final['metrics']['head_only']}")
    ckpt = dict(ckpt)
    ckpt.update(
        {
            "alpha": float(alpha),
            "alpha_grid": grid,
            "dev_pool_reranked": final["metrics"]["reranked"],
            "dev_pool_head_only": final["metrics"]["head_only"],
            "dev_pool_stage1_at_k": final["metrics"]["stage1_top_pool"],
            "dev_monitor_complete": True,
            "commit": git_commit(),
            "utc_date": utc_now(),
            "normalization": {
                "stage1": "per-query z-score over the candidate pool",
                "cross_attn": "per-query z-score over the candidate pool",
                "mix": "alpha * z(stage1) + (1-alpha) * z(cross_attn)",
                "dev_stage1_mean": float(dev["stage1"].mean()),
                "dev_stage1_std": float(dev["stage1"].std()),
                "dev_cross_mean": float(dev["cross"].mean()),
                "dev_cross_std": float(dev["cross"].std()),
            },
        }
    )
    payload = {k: v for k, v in ckpt.items() if k not in ("cfg", "state_dict")}
    save_head(args.head, scorer, payload)
    write_result(Path(str(args.head).replace(".pt", "_dev.json")), payload)
    print(f"[tune] wrote {args.head}")


if __name__ == "__main__":
    main()
