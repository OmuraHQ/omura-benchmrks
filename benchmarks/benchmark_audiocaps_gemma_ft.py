#!/usr/bin/env python
"""Evaluate a fine-tuned AudioCaps audio side on the frozen 883-clip test pool.

The test embeddings are the *frozen* Gemma embeddings cached by
``finetune_audiocaps_gemma.py``.  A checkpoint then supplies the projection
heads / LoRA / unfrozen layers.  Test captions are only ever queries or
retrieval candidates, never training documents.

Usage::

    CUDA_VISIBLE_DEVICES=4 .venv/bin/python benchmarks/benchmark_audiocaps_gemma_ft.py \
        --checkpoint data/audiocaps_gemma_adapter.pt \
        --out benchmarks/results/audiocaps_gemma_finetuned.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))

from bench_common import REPO_ROOT, git_commit, utc_now, write_result  # noqa: E402
from finetune_audiocaps_gemma import (  # noqa: E402
    CACHE,
    DIM,
    ProjHead,
    decode_audio_bytes,
    encode_audio_model,
    load_finetuned_model,
    load_test_manifest,
)

RESULTS = REPO_ROOT / "benchmarks" / "results"


def _project(x: np.ndarray, state: Dict[str, Any], hidden: int = DIM, arch: str = "residual_mlp", device: str = "cuda") -> np.ndarray:
    import torch

    if state is None:
        arr = np.asarray(x, dtype=np.float32)
        arr = arr / np.maximum(np.linalg.norm(arr, axis=1, keepdims=True), 1e-12)
        return arr
    head = ProjHead(dim=DIM, hidden=hidden, arch=arch)
    head.load_state_dict({k: v.to("cpu") for k, v in state.items()})
    head.to(device).eval()
    with torch.no_grad():
        out = torch.nn.functional.normalize(
            head(torch.from_numpy(np.asarray(x, dtype=np.float32)).to(device)), dim=-1
        )
    return out.float().cpu().numpy()


def _load_test_cache(name: str = "test_emb.npz") -> Dict[str, Any]:
    z = np.load(CACHE / name)
    return {
        "audio": z["audio"],
        "text_query": z["text_query"],
        "text_doc": z["text_doc"],
        "clip_index": z["clip_index"],
        "ids": z["ids"],
    }


def _first_hit_ranks(sims: np.ndarray, gt: List[np.ndarray]) -> np.ndarray:
    ranks = np.empty(sims.shape[0], dtype=np.int32)
    for i in range(sims.shape[0]):
        pos = gt[i]
        best = float(sims[i, pos].max())
        ranks[i] = int(np.sum(sims[i] > best)) + 1
    return ranks


def leakage_check(train_meta: Dict[str, Any], manifest: Dict[str, Any]) -> Dict[str, Any]:
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
        "test_captions_used_as_documents": False,
        "note": (
            "Train split is OpenSound/AudioCaps train; test is OpenSound/AudioCaps "
            "test. Disjointness is asserted on (youtube_id, start_time), audiocap_id "
            "and youtube_id."
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, default=REPO_ROOT / "data" / "audiocaps_gemma_adapter.pt")
    ap.add_argument("--out", type=Path, default=RESULTS / "audiocaps_gemma_finetuned.json")
    ap.add_argument("--stage-label", default="adapter")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-emb", default="test_emb.npz")
    ap.add_argument("--model-label", default=None)
    args = ap.parse_args()

    import torch

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    stage = ckpt.get("stage", args.stage_label)
    cache = _load_test_cache(args.test_emb)
    manifest = load_test_manifest()
    train_meta = json.loads((CACHE / "train_meta.json").read_text())

    audio = cache["audio"]
    tq = cache["text_query"]
    td = cache["text_doc"]
    clip_index = cache["clip_index"]
    ids = cache["ids"]
    if not (len(audio) == len(manifest["clips"]) == len(ids)):
        raise RuntimeError("test audio cache does not match manifest")
    if not np.array_equal(ids, np.asarray([int(c["audiocap_id"]) for c in manifest["clips"]])):
        raise RuntimeError("test cache id order mismatch")

    if stage == "adapter":
        hidden = int(ckpt.get("arch", {}).get("hidden", DIM))
        arch = ckpt.get("arch", {}).get("kind", "residual_mlp")
        audio_p = _project(audio, ckpt["audio_head"], hidden, arch)
        tq_p = _project(tq, ckpt["text_head"], hidden, arch)
        td_p = _project(td, ckpt["text_head"], hidden, arch)
        model_name = ckpt.get("model", "google/embeddinggemma-2+audiocaps-adapter")
        recipe = (
            "audio = L2(f_audio(Gemma_a.encode(16k mono))); "
            "text = L2(f_text(Gemma.encode(QUERY_PREFIX+caption))) for t2a; "
            "documents use DOCUMENT_PREFIX and the same f_text. "
            "f_* = x + fc2(GELU(fc1(LayerNorm(x)))) with zero-init fc2."
        )
    elif stage in ("lora", "full"):
        model, lora_ckpt = load_finetuned_model(args.checkpoint)
        model.eval()
        test_arrays = []
        for c in manifest["clips"]:
            blob = (REPO_ROOT / "data" / "audiocaps" / c["file"]).read_bytes()
            test_arrays.append(decode_audio_bytes(blob))
        audio_raw = encode_audio_model(model, test_arrays, batch_size=32, train=False)
        audio_raw = audio_raw.float().cpu().numpy()
        ad = ckpt.get("adapter_state")
        if ad:
            hidden = int(ad["hidden"])
            arch = ad.get("arch", "residual_mlp")
            audio_p = _project(audio_raw, ad["audio_head"], hidden, arch)
            tq_p = _project(tq, ad["text_head"], hidden, arch)
            td_p = _project(td, ad["text_head"], hidden, arch)
            recipe = (
                f"audio = L2(f_audio(Gemma_a_{stage}.encode(16k mono))); "
                "text = L2(f_text(frozen Gemma.encode(QUERY_PREFIX+caption))) / "
                "DOCUMENT_PREFIX for a2t. f_* are the frozen residual heads from the "
                "adapter stage; the audio tower carries the fine-tuned weights."
            )
        else:
            audio_p = audio_raw / np.maximum(np.linalg.norm(audio_raw, axis=1, keepdims=True), 1e-12)
            tq_p = tq / np.maximum(np.linalg.norm(tq, axis=1, keepdims=True), 1e-12)
            td_p = td / np.maximum(np.linalg.norm(td, axis=1, keepdims=True), 1e-12)
            recipe = (
                "audio = L2(Gemma_a_finetuned.encode(16k mono)) where Gemma_a carries "
                f"{stage} weights on the audio tower/embed_audio and the text backbone is "
                "frozen; text = L2(frozen Gemma.encode(QUERY_PREFIX+caption)) for t2a and "
                "DOCUMENT_PREFIX for a2t documents. Same 768-d Gemma space, no extra head."
            )
        model_name = ckpt.get("model", "google/embeddinggemma-2+" + stage)
    else:
        raise NotImplementedError(f"unknown stage {stage!r}")

    # t2a: every caption is a query against the 883 audio clips.
    t2a_gt = [np.asarray([int(i)], dtype=np.int64) for i in clip_index]
    sims_t2a = tq_p @ audio_p.T
    t2a_ranks = _first_hit_ranks(sims_t2a, t2a_gt)

    # a2t: each clip against all 4308 caption documents (its own 5 are positive).
    caps_by_clip: List[List[int]] = [[] for _ in range(len(audio))]
    for j, i in enumerate(clip_index):
        caps_by_clip[int(i)].append(j)
    a2t_gt = [np.asarray(v, dtype=np.int64) for v in caps_by_clip]
    sims_a2t = audio_p @ td_p.T
    a2t_ranks = _first_hit_ranks(sims_a2t, a2t_gt)

    def _recall(ranks: np.ndarray) -> Dict[str, float]:
        return {f"R@{k}": float(np.mean(ranks <= k)) for k in (1, 5, 10)}

    t2a = _recall(t2a_ranks)
    a2t = _recall(a2t_ranks)
    leak = leakage_check(train_meta, manifest)
    if not (
        leak["train_test_clip_disjoint"]
        and leak["train_test_audiocap_id_disjoint"]
        and leak["train_test_youtube_id_disjoint"]
    ):
        raise RuntimeError(f"leakage check failed: {leak}")

    hparams = ckpt.get("hyperparams", {})
    if args.model_label:
        model_name = args.model_label
    payload = {
        "model": model_name,
        "dim": int(DIM),
        "backend": "omuragemma",
        "stage": stage,
        "protocol": "audiocaps_test_full_pool_t2a_and_a2t",
        "dataset": "AudioCaps test (OpenSound/AudioCaps)",
        "pool_size": int(audio_p.shape[0]),
        "num_queries": int(tq_p.shape[0]),
        "num_caption_candidates": int(td_p.shape[0]),
        "caption_count": int(len(td_p)),
        "train_size": int(ckpt.get("train_size", 0)),
        "dev_size": int(ckpt.get("dev_size", 0)),
        "seed": int(ckpt.get("seed", args.seed)),
        "commit": git_commit(),
        "utc_date": utc_now(),
        "text_to_audio": t2a,
        "audio_to_text": a2t,
        "per_query_ranks": {
            "text_to_audio": [
                {"audiocap_id": int(manifest["clips"][int(i)]["audiocap_id"]), "rank": int(r)}
                for i, r in zip(clip_index, t2a_ranks)
            ],
            "audio_to_text": [
                {"audiocap_id": int(c["audiocap_id"]), "rank": int(r)}
                for c, r in zip(manifest["clips"], a2t_ranks)
            ],
        },
        "checkpoint": str(args.checkpoint.resolve().relative_to(REPO_ROOT)).replace("\\", "/"),
        "best_epoch": ckpt.get("best_epoch"),
        "hyperparameters": hparams,
        "inference_recipe": recipe,
        "leakage_check": leak,
        "dev_pool_metrics": ckpt.get("dev_pool"),
        "dev_full_metrics": ckpt.get("dev_full"),
        "notes": (
            "Full 883-clip test pool, 4308 caption queries; no sampled negatives. "
            "Test evaluated once with the final dev-selected checkpoint."
        ),
    }
    write_result(args.out, payload)
    print(json.dumps({k: payload[k] for k in ("model", "text_to_audio", "audio_to_text", "pool_size", "num_queries", "train_size", "dev_size")}, indent=2))
    print(f"[ft-bench] wrote {args.out}")


if __name__ == "__main__":
    main()