#!/usr/bin/env python
"""Multi-model rerank pipeline for AudioCaps text->audio retrieval (fallback).

Stage-1 candidate pools come from the AudioCaps projection adapter over frozen
EmbeddingGemma 2 pooled embeddings (``data/audiocaps_gemma_adapter.pt``).  A
second, independently trained CLAP scorer (``laion/larger_clap_general`` +
``data/audiocaps_adapter.pt``, CSLS k=10) reranks each pool.  Gemma and CLAP
scores are z-scored per query over the candidate set and combined as

    score = (1 - w) * z(gemma) + w * z(clap)

The single scalar ``w`` per direction is selected on the AudioCaps validation
split only; AudioCaps test and Clotho are evaluated once with those frozen
weights.  This is a labelled fallback artifact: it deliberately does not touch
``audiocaps_gemma_*.json``, ``clotho_gemma*.json`` or
``scripts/check_conditions.py``.

Usage::

    CUDA_VISIBLE_DEVICES=5 .venv/bin/python benchmarks/rerank_audiocaps_pipeline.py

Prerequisites (cached by the helper scripts):
    data/audiocaps_val/gemma_val_emb.npz   (benchmarks/cache_val_gemma.py)
    data/clotho/cache/clap_frozen.npz      (benchmarks/cache_clotho_clap.py)
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_common import REPO_ROOT, git_commit, utc_now, write_result  # noqa: E402

RESULTS = REPO_ROOT / "benchmarks" / "results"
AUDIOCAPS = REPO_ROOT / "data" / "audiocaps"
AUDIOCAPS_VAL = REPO_ROOT / "data" / "audiocaps_val"
GEMMA_CACHE = REPO_ROOT / "data" / "audiocaps_gemma_cache"
CLAP_CACHE = REPO_ROOT / "data" / "audiocaps_clap_cache"
CLOTHO = REPO_ROOT / "data" / "clotho"
CLOTHO_CACHE = CLOTHO / "cache"
SEED = 42
CSLS_K = 10
POOLS = ("gemma_top100", "gemma_top200", "union_gemma100_clap100")


# ── score helpers ────────────────────────────────────────────────────────────


def l2(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    return arr / np.maximum(np.linalg.norm(arr, axis=-1, keepdims=True), 1e-12)


def project_gemma(x: np.ndarray, state: Dict[str, Any], hidden: int) -> np.ndarray:
    import torch
    import torch.nn.functional as F
    from finetune_audiocaps_gemma import ProjHead

    head = ProjHead(dim=768, hidden=hidden)
    head.load_state_dict({k: v.to("cpu") for k, v in state.items()})
    head.eval()
    with torch.no_grad():
        out = F.normalize(head(torch.from_numpy(np.asarray(x, dtype=np.float32))), dim=-1)
    return out.float().cpu().numpy()


def apply_clap_adapter(audio: np.ndarray, text: np.ndarray):
    import torch
    from finetune_audiocaps_adapter import DualAdapter

    ckpt = torch.load(REPO_ROOT / "data" / "audiocaps_adapter.pt", map_location="cpu",
                      weights_only=False)
    cfg = ckpt["config"]
    adapter = DualAdapter(hidden=cfg["hidden"], dropout=cfg["dropout"],
                          mode=cfg.get("adapter_mode", "dual"),
                          linear=cfg.get("linear", False))
    adapter.load_state_dict(ckpt["state_dict"])
    adapter.eval()
    with torch.no_grad():
        a = adapter.audio(torch.from_numpy(np.asarray(audio, dtype=np.float32))).numpy()
        t = adapter.text(torch.from_numpy(np.asarray(text, dtype=np.float32))).numpy()
    return l2(a), l2(t)


def csls(sims: np.ndarray, k: int = CSLS_K) -> np.ndarray:
    if k <= 0:
        return sims
    k = min(k, sims.shape[0], sims.shape[1])
    rq = np.sort(sims, axis=1)[:, -k:].mean(axis=1)
    rd = np.sort(sims, axis=0)[-k:, :].mean(axis=0)
    return 2.0 * sims - rq[:, None] - rd[None, :]


def topk_pool(sims: np.ndarray, k: int) -> List[np.ndarray]:
    k = min(k, sims.shape[1])
    return [np.argpartition(-sims[i], k - 1)[:k] for i in range(sims.shape[0])]


def union_pool(a: np.ndarray, b: np.ndarray, k: int) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    for i in range(a.shape[0]):
        pa = np.argpartition(-a[i], k - 1)[:k]
        pb = np.argpartition(-b[i], k - 1)[:k]
        out.append(np.unique(np.concatenate([pa, pb])))
    return out


def zscore_over_pool(sims: np.ndarray, pool: Sequence[np.ndarray]) -> np.ndarray:
    out = np.zeros_like(sims, dtype=np.float32)
    for i, cand in enumerate(pool):
        v = sims[i, cand]
        out[i, cand] = (v - v.mean()) / (v.std() + 1e-9)
    return out


def fuse(sims_a: np.ndarray, sims_b: np.ndarray, pool: Sequence[np.ndarray],
         w_b: float) -> np.ndarray:
    return (1.0 - w_b) * zscore_over_pool(sims_a, pool) + w_b * zscore_over_pool(sims_b, pool)


def pool_recall(pool: Sequence[np.ndarray], positives: Sequence[np.ndarray]) -> float:
    hits = 0
    for cand, pos in zip(pool, positives):
        if np.intersect1d(cand, pos).size:
            hits += 1
    return float(hits) / float(len(pool)) if pool else 0.0


def ranks_in_pool(sims: np.ndarray, pool: Sequence[np.ndarray],
                  positives: Sequence[np.ndarray]) -> np.ndarray:
    """1-based first-hit rank inside each pool; -1 when no positive is present."""
    ranks = np.full(len(pool), -1, dtype=np.int32)
    for i, cand in enumerate(pool):
        where = {int(c): j for j, c in enumerate(cand)}
        cols = [where[int(p)] for p in positives[i] if int(p) in where]
        if not cols:
            continue
        best = float(sims[i, cand[cols]].max())
        ranks[i] = int(np.sum(sims[i, cand] > best)) + 1
    return ranks


def recall_metrics(ranks: np.ndarray) -> Dict[str, float]:
    hit = ranks > 0
    n = len(ranks)
    return {f"R@{k}": float(np.sum(hit & (ranks <= k))) / n for k in (1, 5, 10)}


def eval_pipeline(sims: np.ndarray, pool: Sequence[np.ndarray],
                  positives: Sequence[np.ndarray]) -> Tuple[Dict[str, float], np.ndarray]:
    ranks = ranks_in_pool(sims, pool, positives)
    return recall_metrics(ranks), ranks


# ── data builders ────────────────────────────────────────────────────────────


def load_gemma_adapter():
    import torch

    ckpt = torch.load(REPO_ROOT / "data" / "audiocaps_gemma_adapter.pt",
                      map_location="cpu", weights_only=False)
    hidden = int(ckpt.get("arch", {}).get("hidden", 768))
    return ckpt, hidden


def build_audiocaps_test(ckpt, hidden) -> Dict[str, Any]:
    manifest = json.loads((AUDIOCAPS / "manifest.json").read_text())
    clips = manifest["clips"]
    z = np.load(GEMMA_CACHE / "test_emb.npz")
    audio = project_gemma(z["audio"], ckpt["audio_head"], hidden)
    tq = project_gemma(z["text_query"], ckpt["text_head"], hidden)
    td = project_gemma(z["text_doc"], ckpt["text_head"], hidden)
    clip_index = np.asarray(z["clip_index"], dtype=np.int64)

    c_audio = np.load(CLAP_CACHE / "clap_audio_883.npz")["vecs"].astype(np.float32)
    c_text = np.load(CLAP_CACHE / "clap_text_4308.npz")["vecs"].astype(np.float32)
    c_audio, c_text = apply_clap_adapter(c_audio, c_text)

    caps_by_clip: List[List[int]] = [[] for _ in range(len(clips))]
    for j, c in enumerate(clip_index):
        caps_by_clip[int(c)].append(j)

    return {
        "audio": audio, "tq": tq, "td": td,
        "c_audio": c_audio, "c_text": c_text, "clip_index": clip_index,
        "clips": clips,
        "sims": {
            "t2a": {"gemma": tq @ audio.T, "clap": c_text @ c_audio.T},
            "a2t": {"gemma": audio @ td.T, "clap": c_audio @ c_text.T},
        },
        "gt": {
            "t2a": [np.asarray([int(c)], dtype=np.int64) for c in clip_index],
            "a2t": [np.asarray(v, dtype=np.int64) for v in caps_by_clip],
        },
    }


def build_audiocaps_dev(ckpt, hidden) -> Dict[str, Any]:
    z = np.load(AUDIOCAPS_VAL / "gemma_val_emb.npz", allow_pickle=True)
    audio = project_gemma(z["audio"], ckpt["audio_head"], hidden)
    tq = project_gemma(z["text_query"], ckpt["text_head"], hidden)
    td = project_gemma(z["text_doc"], ckpt["text_head"], hidden)
    keys = [str(k) for k in z["keys"].tolist()]
    text_clip = np.asarray(z["text_clip"], dtype=np.int64)

    # Assemble the exact CLAP validation index used by the adapter run.
    key_to_idx: Dict[str, int] = {}
    keys2: List[str] = []
    audio_parts: List[np.ndarray] = []
    text_parts: List[np.ndarray] = []
    tc2: List[int] = []
    for shard in sorted(glob.glob(str(AUDIOCAPS_VAL / "features" / "*.npz"))):
        q = np.load(shard, allow_pickle=True)
        sk = [str(k) for k in q["keys"].tolist()]
        for i, k in enumerate(sk):
            if k not in key_to_idx:
                key_to_idx[k] = len(keys2)
                keys2.append(k)
                audio_parts.append(q["audio"][i:i + 1])
        text_parts.append(q["text"])
        tc2.extend(key_to_idx[sk[int(c)]] for c in q["text_clip"].tolist())
    if keys2 != keys:
        raise RuntimeError("validation Gemma/CLAP key order mismatch")
    c_audio = np.concatenate(audio_parts).astype(np.float32)
    c_text = np.concatenate(text_parts).astype(np.float32)
    c_audio, c_text = apply_clap_adapter(c_audio, c_text)
    if not np.array_equal(np.asarray(tc2, dtype=np.int64), text_clip):
        raise RuntimeError("validation text_clip mismatch")

    caps_by_clip: List[List[int]] = [[] for _ in range(len(keys))]
    for j, c in enumerate(text_clip):
        caps_by_clip[int(c)].append(j)

    return {
        "audio": audio, "tq": tq, "td": td,
        "c_audio": c_audio, "c_text": c_text,
        "keys": keys, "text_clip": text_clip,
        "sims": {
            "t2a": {"gemma": tq @ audio.T, "clap": c_text @ c_audio.T},
            "a2t": {"gemma": audio @ td.T, "clap": c_audio @ c_text.T},
        },
        "gt": {
            "t2a": [np.asarray([int(c)], dtype=np.int64) for c in text_clip],
            "a2t": [np.asarray(v, dtype=np.int64) for v in caps_by_clip],
        },
    }


def _reindex(ids: Sequence[str], source_ids: Sequence[str], arr: np.ndarray) -> np.ndarray:
    where = {str(k): i for i, k in enumerate(source_ids)}
    return np.asarray([arr[where[str(k)]] for k in ids], dtype=np.float32)


def build_clotho(ckpt, hidden) -> Dict[str, Any]:
    manifest = json.loads((CLOTHO / "mteb_clotho" / "manifest.json").read_text())
    clips = manifest["clips"]
    queries = manifest["queries"]
    g = np.load(CLOTHO_CACHE / "gemma_frozen.npz", allow_pickle=True)
    c = np.load(CLOTHO_CACHE / "clap_frozen.npz", allow_pickle=True)

    clip_ids = [str(x) for x in g["clip_ids"].tolist()]
    query_ids = [str(x) for x in g["query_ids"].tolist()]
    doc_ids = [str(x) for x in g["doc_ids"].tolist()]
    if clip_ids != [str(x) for x in c["clip_ids"].tolist()]:
        raise RuntimeError("Clotho clip order mismatch")
    if query_ids != [str(x) for x in c["query_ids"].tolist()]:
        raise RuntimeError("Clotho query order mismatch")

    audio = project_gemma(g["audio"], ckpt["audio_head"], hidden)
    tq = project_gemma(g["text_query"], ckpt["text_head"], hidden)
    td = project_gemma(g["text_doc"], ckpt["text_head"], hidden)
    c_audio, c_text = apply_clap_adapter(c["audio"], c["text_query"])
    c_text_doc = _reindex(doc_ids, [str(x) for x in c["query_ids"].tolist()],
                          c["text_query"])

    clip_to_idx = {c_id: i for i, c_id in enumerate(clip_ids)}
    query_clip = {q["id"]: q["clip_id"] for q in queries}
    t2a_pos = [np.asarray([clip_to_idx[query_clip[q]]], dtype=np.int64) for q in query_ids]
    caps_by_clip: List[List[int]] = [[] for _ in range(len(clip_ids))]
    for j, d in enumerate(doc_ids):
        caps_by_clip[clip_to_idx[query_clip[d]]].append(j)
    a2t_pos = [np.asarray(v, dtype=np.int64) for v in caps_by_clip]

    return {
        "audio": audio, "tq": tq, "td": td,
        "c_audio": c_audio, "c_text": c_text, "c_text_doc": c_text_doc,
        "clip_ids": clip_ids, "query_ids": query_ids, "doc_ids": doc_ids,
        "sims": {
            "t2a": {"gemma": tq @ audio.T, "clap": c_text @ c_audio.T},
            "a2t": {"gemma": audio @ td.T, "clap": c_audio @ c_text_doc.T},
        },
        "gt": {"t2a": t2a_pos, "a2t": a2t_pos},
    }


# ── candidate pools / evaluation ─────────────────────────────────────────────


def make_pools(sims_g: np.ndarray, sims_c: np.ndarray) -> Dict[str, List[np.ndarray]]:
    c_csls = csls(sims_c, CSLS_K)
    return {
        "gemma_top100": topk_pool(sims_g, 100),
        "gemma_top200": topk_pool(sims_g, 200),
        "union_gemma100_clap100": union_pool(sims_g, c_csls, 100),
    }


def evaluate_all(data: Dict[str, Any], weights: Dict[str, float]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for direction in ("t2a", "a2t"):
        s_g = data["sims"][direction]["gemma"]
        s_c = csls(data["sims"][direction]["clap"], CSLS_K)
        gt = data["gt"][direction]
        pools = make_pools(s_g, data["sims"][direction]["clap"])
        block: Dict[str, Any] = {}
        for name in POOLS:
            pool = pools[name]
            fused_sims = fuse(s_g, s_c, pool, weights[direction])
            gem_metrics, gem_ranks = eval_pipeline(s_g, pool, gt)
            clap_metrics, clap_ranks = eval_pipeline(s_c, pool, gt)
            fused_metrics, fused_ranks = eval_pipeline(fused_sims, pool, gt)
            sizes = np.asarray([len(p) for p in pool])
            block[name] = {
                "pool_recall": pool_recall(pool, gt),
                "pool_size_mean": float(sizes.mean()),
                "pool_size_max": int(sizes.max()),
                "gemma_only": gem_metrics,
                "clap_rerank": clap_metrics,
                "fused": fused_metrics,
                "_ranks": {
                    "gemma_only": gem_ranks,
                    "clap_rerank": clap_ranks,
                    "fused": fused_ranks,
                },
            }
        out[direction] = block
    return out


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def strip_ranks(block: Dict[str, Any]) -> Dict[str, Any]:
    return {
        name: {k: v for k, v in entry.items() if k != "_ranks"}
        for name, entry in block.items()
    }


def tune_weight(data: Dict[str, Any], direction: str, pool_name: str = "union_gemma100_clap100"):
    s_g = data["sims"][direction]["gemma"]
    s_c = csls(data["sims"][direction]["clap"], CSLS_K)
    gt = data["gt"][direction]
    pool = make_pools(s_g, data["sims"][direction]["clap"])[pool_name]
    best = None
    grid = []
    for w in np.round(np.arange(0.50, 1.0001, 0.01), 2):
        metrics, _ = eval_pipeline(fuse(s_g, s_c, pool, float(w)), pool, gt)
        key = (metrics["R@10"], metrics["R@5"], metrics["R@1"])
        grid.append({"w_clap": float(w), **metrics})
        if best is None or key > best[0]:
            best = (key, float(w), metrics)
    return {"direction": direction, "pool": pool_name, "w_clap": best[1],
            "metrics": best[2], "grid": grid}


def per_query_ranks(data: Dict[str, Any], ranks: Dict[str, np.ndarray],
                    direction: str, pipeline: str, dataset: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if dataset == "audiocaps":
        if direction == "t2a":
            for j, c in enumerate(data["clip_index"]):
                rows.append({
                    "query_index": j,
                    "audiocap_id": int(data["clips"][int(c)]["audiocap_id"]),
                    "rank": int(ranks[pipeline][j]),
                })
        else:
            for i, clip in enumerate(data["clips"]):
                rows.append({
                    "audiocap_id": int(clip["audiocap_id"]),
                    "rank": int(ranks[pipeline][i]),
                })
    else:
        if direction == "t2a":
            for j, qid in enumerate(data["query_ids"]):
                rows.append({
                    "query_id": qid,
                    "clip_id": data["clip_ids"][int(data["gt"]["t2a"][j][0])],
                    "rank": int(ranks[pipeline][j]),
                })
        else:
            for i, cid in enumerate(data["clip_ids"]):
                rows.append({"clip_id": cid, "rank": int(ranks[pipeline][i])})
    return rows


# ── leakage ──────────────────────────────────────────────────────────────────


def audiocaps_leakage() -> Dict[str, Any]:
    train_meta = json.loads((GEMMA_CACHE / "train_meta.json").read_text())
    test_manifest = json.loads((AUDIOCAPS / "manifest.json").read_text())
    test_keys = {(c["youtube_id"], int(c["start_time"])) for c in test_manifest["clips"]}
    test_ids = {int(c["audiocap_id"]) for c in test_manifest["clips"]}
    test_yt = {c["youtube_id"] for c in test_manifest["clips"]}
    train_keys = {(str(k[0]), int(k[1])) for k in train_meta["keys"]}
    train_ids = {int(x) for x in train_meta["ids"]}
    train_yt = {str(k[0]) for k in train_meta["keys"]}

    # Validation split vs test split.
    val_keys = set()
    val_ids = set()
    for shard in sorted(glob.glob(str(AUDIOCAPS_VAL / "features" / "*.npz"))):
        q = np.load(shard, allow_pickle=True)
        val_keys.update(str(k) for k in q["keys"].tolist())
    for path in sorted((AUDIOCAPS_VAL / "parquet").glob("validation-*.parquet")):
        import pyarrow.parquet as pq
        for row in pq.read_table(path, columns=["audiocap_id", "youtube_id", "start_time"]).to_pylist():
            val_ids.add(int(row["audiocap_id"]))
    val_key_tuples = {(k.split("|")[0], int(k.split("|")[1])) for k in val_keys}
    return {
        "train_clips": len(train_meta["ids"]),
        "test_clips": len(test_manifest["clips"]),
        "dev_clips": len(val_keys),
        "train_test_clip_disjoint": len(train_keys & test_keys) == 0,
        "train_test_audiocap_id_disjoint": len(train_ids & test_ids) == 0,
        "train_test_youtube_id_disjoint": len(train_yt & test_yt) == 0,
        "dev_test_clip_disjoint": len(val_key_tuples & test_keys) == 0,
        "dev_test_audiocap_id_disjoint": len(val_ids & test_ids) == 0,
        "overlap_train_test_clip_keys": len(train_keys & test_keys),
        "overlap_train_test_audiocap_ids": len(train_ids & test_ids),
        "overlap_train_test_youtube_ids": len(train_yt & test_yt),
        "overlap_dev_test_clip_keys": len(val_key_tuples & test_keys),
        "overlap_dev_test_audiocap_ids": len(val_ids & test_ids),
        "test_captions_used_as_training_documents": False,
        "note": (
            "Train is OpenSound/AudioCaps train; dev is the official validation "
            "split (445 clips / 2179 captions); test is the 883-clip test pool. "
            "Disjointness is asserted on (youtube_id, start_time) and audiocap_id."
        ),
    }


def clotho_leakage() -> Dict[str, Any]:
    return {
        "clotho_used_for_training": False,
        "clotho_used_for_weight_selection": False,
        "clotho_used_for_csls_selection": False,
        "clotho_used_for_pool_selection": False,
        "audiocaps_weights_applied_frozen": True,
        "note": (
            "Clotho v2.1 test is only ever an evaluation set. Fusion weights and "
            "CSLS k were selected on the AudioCaps validation split and applied "
            "unchanged."
        ),
    }


# ── main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-audiocaps", type=Path,
                    default=RESULTS / "audiocaps_pipeline_rerank.json")
    ap.add_argument("--out-clotho", type=Path,
                    default=RESULTS / "clotho_pipeline_rerank.json")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    np.random.seed(args.seed)
    ckpt, hidden = load_gemma_adapter()
    print(f"[rerank] gemma adapter hidden={hidden}")

    dev = build_audiocaps_dev(ckpt, hidden)
    test = build_audiocaps_test(ckpt, hidden)
    clotho = build_clotho(ckpt, hidden)

    print("[rerank] tuning fusion weights on AudioCaps validation union pool")
    tune_t2a = tune_weight(dev, "t2a")
    tune_a2t = tune_weight(dev, "a2t")
    weights = {"t2a": tune_t2a["w_clap"], "a2t": tune_a2t["w_clap"]}
    print(f"[rerank] frozen weights: {weights}")

    dev_eval = evaluate_all(dev, weights)
    test_eval = evaluate_all(test, weights)
    clotho_eval = evaluate_all(clotho, weights)

    leak = audiocaps_leakage()
    if not (leak["train_test_clip_disjoint"] and leak["train_test_audiocap_id_disjoint"]
            and leak["dev_test_clip_disjoint"] and leak["dev_test_audiocap_id_disjoint"]):
        raise RuntimeError(f"AudioCaps leakage check failed: {leak}")

    common = {
        "pipeline": "gemma_adapter_stage1_clap_adapter_rerank",
        "evidence_tier": "fallback_pipeline",
        "components": [
            {
                "name": "gemma_audiocaps_adapter",
                "model": "google/embeddinggemma-2 + audiocaps residual adapter",
                "dim": 768,
                "role": "stage-1 candidate generator and fusion component",
                "checkpoint": "data/audiocaps_gemma_adapter.pt",
                "weights": "gemma",
            },
            {
                "name": "clap_audiocaps_adapter",
                "model": "laion/larger_clap_general + audiocaps dual adapter",
                "dim": 512,
                "role": "reranker and fusion component",
                "checkpoint": "data/audiocaps_adapter.pt",
                "weights": "clap",
            },
        ],
        "vast": {
            "status": "unavailable",
            "official_repo": "https://github.com/CASIA-IVA-Lab/VAST",
            "license": "MIT",
            "release": "pretrain_vast.tar.gz (3.1 GB, Google Drive), pretraining weights only",
            "audio_caps_finetuned_checkpoint_released": False,
            "reason": (
                "The official VAST release contains pretraining weights for "
                "fine-tuning, not the AudioCaps-finetuned checkpoint behind the "
                "reported 82.9% t2a R@10. Running it also needs BEATs, "
                "bert-base-uncased and EVA01-CLIP weights that are not present "
                "locally. VAST is therefore not used as a frozen reranker."
            ),
        },
        "seed": args.seed,
        "commit": git_commit(),
        "utc_date": utc_now(),
        "weights": weights,
        "fusion": {
            "normalization": "per-query z-score over the candidate pool for each model",
            "formula": "score = (1 - w_clap) * z(gemma) + w_clap * z(clap)",
            "csls_k": CSLS_K,
            "csls_applied_to": "CLAP only",
            "csls_selected_on": "AudioCaps validation split (inherited from audiocaps_finetuned.json)",
            "weights": weights,
            "weight_selection": {
                "t2a": {k: v for k, v in tune_t2a.items() if k != "grid"},
                "a2t": {k: v for k, v in tune_a2t.items() if k != "grid"},
                "grid_step": 0.01,
                "selected_on": "AudioCaps validation split only",
            },
        },
        "candidate_pools": {
            "gemma_top100": {"k": 100, "description": "top-100 by Gemma+adapter"},
            "gemma_top200": {"k": 200, "description": "top-200 by Gemma+adapter"},
            "union_gemma100_clap100": {
                "k": 100,
                "description": "union of Gemma top-100 and CLAP(CSLS) top-100",
            },
        },
        "protocol": {
            "t2a": "each of 4308 AudioCaps test captions queries all 883 clips",
            "a2t": "each of 883 AudioCaps test clips queries all 4308 caption documents",
            "no_test_tuning": True,
            "test_evaluated_once_with_frozen_dev_weights": True,
        },
    }

    audiocaps_payload = {
        **common,
        "dataset": "AudioCaps test (OpenSound/AudioCaps; 883 clips / 4308 captions)",
        "dev_dataset": "AudioCaps validation (445 clips / 2179 captions)",
        "pool_size": 883,
        "num_queries": 4308,
        "num_caption_candidates": 4308,
        "dev": {d: strip_ranks(dev_eval[d]) for d in ("t2a", "a2t")},
        "test": {d: strip_ranks(test_eval[d]) for d in ("t2a", "a2t")},
        "per_query_ranks": {
            direction: {
                pool: {
                    pipeline: per_query_ranks(test, test_eval[direction][pool]["_ranks"],
                                              direction, pipeline, "audiocaps")
                    for pipeline in ("gemma_only", "clap_rerank", "fused")
                }
                for pool in POOLS
            }
            for direction in ("t2a", "a2t")
        },
        "leakage_check": leak,
        "notes": [
            "Fallback evidence tier; canonical Gemma-only artifacts are untouched.",
            "CLAP dev numbers are optimistic because the CLAP adapter and CSLS k "
            "were early-stopped/selected on the same AudioCaps validation split; "
            "fusion weights are also selected there. Test is untouched.",
            "Fused test metrics use weights frozen on dev; no test-set tuning.",
        ],
    }
    write_result(args.out_audiocaps, _jsonable(audiocaps_payload))
    print(f"[rerank] wrote {args.out_audiocaps}")

    clotho_payload = {
        **common,
        "dataset": "Clotho v2.1 test (mteb/clotho_t2a_v2; 1045 clips / 5225 captions)",
        "pool_size": 1045,
        "num_queries": 5225,
        "num_caption_candidates": 5225,
        "frozen_config": {
            "weights": weights,
            "csls_k": CSLS_K,
            "source": "AudioCaps validation split; no Clotho tuning",
        },
        "protocol": {
            "t2a": "each of 5225 Clotho test captions queries all 1045 clips",
            "a2t": "each of 1045 Clotho test clips queries all 5225 caption documents",
            "no_clotho_tuning": True,
            "weights_frozen_from_audiocaps_dev": True,
        },
        "test": {d: strip_ranks(clotho_eval[d]) for d in ("t2a", "a2t")},
        "per_query_ranks": {
            direction: {
                pool: {
                    pipeline: per_query_ranks(clotho, clotho_eval[direction][pool]["_ranks"],
                                              direction, pipeline, "clotho")
                    for pipeline in ("gemma_only", "clap_rerank", "fused")
                }
                for pool in POOLS
            }
            for direction in ("t2a", "a2t")
        },
        "leakage_check": clotho_leakage(),
        "notes": [
            "Clotho is a held-out cross-dataset evaluation; no weight, CSLS or "
            "pool selection used Clotho.",
            "The Gemma adapter was trained on AudioCaps only; Clotho remains OOD.",
        ],
    }
    write_result(args.out_clotho, _jsonable(clotho_payload))
    print(f"[rerank] wrote {args.out_clotho}")

    for direction in ("t2a", "a2t"):
        print(f"[test] {direction}")
        for pool in POOLS:
            e = test_eval[direction][pool]
            print(f"  {pool}: recall={e['pool_recall']:.4f} "
                  f"gemma R@10={e['gemma_only']['R@10']:.4f} "
                  f"clap R@10={e['clap_rerank']['R@10']:.4f} "
                  f"fused R@10={e['fused']['R@10']:.4f}")
    print("[clotho] t2a R@10 by pool")
    for pool in POOLS:
        e = clotho_eval["t2a"][pool]
        print(f"  {pool}: recall={e['pool_recall']:.4f} "
              f"gemma={e['gemma_only']['R@10']:.4f} "
              f"clap={e['clap_rerank']['R@10']:.4f} "
              f"fused={e['fused']['R@10']:.4f}")


if __name__ == "__main__":
    main()
