#!/usr/bin/env python
"""Cache frozen EmbeddingGemma 2 token states for the cross-attention reranker.

Outputs (under ``data/audiocaps_crossattn_cache/<split>/``):

* ``audio_tok.npy``       float16 [N, T_a, 512]  = embed_audio(audio_tower(...))
* ``text_query_tok.npy``  float16 [N, T_t, 768]  = language_model(QUERY_PREFIX + text)
* ``text_doc_tok.npy``    float16 [N, T_t, 768]  = language_model(DOCUMENT_PREFIX + text)
* ``meta.json``           lengths, ids, captions, clip order

The Gemma towers are frozen; only the 512-d audio projector output and the
768-d text token states are stored.  This is the only step that runs the base
model, so training and reranking never re-encode audio.

Usage::

    CUDA_VISIBLE_DEVICES=3 .venv/bin/python benchmarks/build_crossattn_cache.py --splits all
"""

from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from bench_common import REPO_ROOT
from crossattn_common import (
    AUDIO_SR,
    CLOTHO_ROOT,
    CROSS_CACHE,
    DOCUMENT_PREFIX,
    QUERY_PREFIX,
    extract_audio_tokens,
    extract_text_tokens,
    load_gemma,
)

AUDIOCAPS_TRAIN = REPO_ROOT / "data" / "audiocaps_train"
AUDIOCAPS_TEST = REPO_ROOT / "data" / "audiocaps"
ADAPTER_CACHE = REPO_ROOT / "data" / "audiocaps_gemma_cache"


# ── audio io ─────────────────────────────────────────────────────────────────


def decode_bytes(blob: bytes) -> np.ndarray:
    import soundfile as sf

    audio, sr = sf.read(io.BytesIO(blob), dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    if sr != AUDIO_SR:
        import soxr

        mono = soxr.resample(mono, sr, AUDIO_SR)
    return np.ascontiguousarray(mono, dtype=np.float32)


def decode_file(path: Path) -> np.ndarray:
    import soundfile as sf

    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    if sr != AUDIO_SR:
        import soxr

        mono = soxr.resample(mono, sr, AUDIO_SR)
    return np.ascontiguousarray(mono, dtype=np.float32)


# ── split builders ───────────────────────────────────────────────────────────


def _write_split(name: str, audio: np.ndarray, a_len, tq: np.ndarray, tq_len, td: np.ndarray, td_len, meta: Dict[str, Any]) -> None:
    out = CROSS_CACHE / name
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "audio_tok.npy", audio)
    np.save(out / "text_query_tok.npy", tq)
    np.save(out / "text_doc_tok.npy", td)
    meta = dict(meta)
    meta.update(
        {
            "audio_tokens": int(audio.shape[1]),
            "text_tokens": int(tq.shape[1]),
            "audio_lengths": a_len.astype(int).tolist(),
            "text_query_lengths": tq_len.astype(int).tolist(),
            "text_doc_lengths": td_len.astype(int).tolist(),
            "audio_dim": int(audio.shape[2]),
            "text_dim": int(tq.shape[2]),
            "token_source": "gemma.embed_audio(gemma.audio_tower(x)) / gemma.language_model(text)",
            "model": "google/embeddinggemma-2",
        }
    )
    (out / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    print(f"[cross-cache] wrote {name}: audio={audio.shape} text={tq.shape}", flush=True)


def build_audiocaps_train(model) -> None:
    name = "audiocaps_train"
    if (CROSS_CACHE / name / "meta.json").exists():
        print(f"[cross-cache] {name} exists, skip")
        return
    meta = json.loads((ADAPTER_CACHE / "train_meta.json").read_text())
    n = int(meta["n_clips"])
    slot = int(meta["samples_per_clip"])
    mem = np.memmap(ADAPTER_CACHE / "train_audio_i16.bin", dtype=np.int16, mode="r")

    def get(i: int) -> np.ndarray:
        ln = int(meta["lengths"][i])
        return (mem[i * slot : i * slot + ln].astype(np.float32) / 32767.0).copy()

    t0 = time.time()
    audio, a_len, _ = extract_audio_tokens(model, [get(i) for i in range(n)], batch_size=32)
    print(f"[cross-cache] train audio {time.time() - t0:.0f}s", flush=True)

    captions = list(meta["captions"])
    tq, tq_len, _ = extract_text_tokens(model, [QUERY_PREFIX + c for c in captions], batch_size=256)
    td, td_len, _ = extract_text_tokens(model, [DOCUMENT_PREFIX + c for c in captions], batch_size=256)

    rng = np.random.default_rng(42)
    perm = rng.permutation(n)
    n_dev = max(1, int(round(n * 0.1)))
    dev = np.sort(perm[:n_dev]).tolist()
    train = np.sort(perm[n_dev:]).tolist()
    _write_split(
        name,
        audio,
        a_len,
        tq,
        tq_len,
        td,
        td_len,
        {
            "split": "audiocaps_train",
            "ids": list(map(int, meta["ids"])),
            "keys": meta["keys"],
            "captions": captions,
            "audio_seconds": [int(x) / AUDIO_SR for x in meta["lengths"]],
            "train_idx": train,
            "dev_idx": dev,
            "split_rule": "np.random.default_rng(42).permutation(45178), first 10% dev (matches adapter)",
        },
    )


def build_audiocaps_test(model) -> None:
    name = "audiocaps_test"
    if (CROSS_CACHE / name / "meta.json").exists():
        print(f"[cross-cache] {name} exists, skip")
        return
    manifest = json.loads((AUDIOCAPS_TEST / "manifest.json").read_text())
    clips = manifest["clips"]
    arrs = [decode_file(REPO_ROOT / "data" / "audiocaps" / c["file"]) for c in clips]
    audio, a_len, _ = extract_audio_tokens(model, arrs, batch_size=32)
    captions: List[str] = []
    clip_index: List[int] = []
    for i, c in enumerate(clips):
        for cap in c["captions"]:
            captions.append(cap)
            clip_index.append(i)
    tq, tq_len, _ = extract_text_tokens(model, [QUERY_PREFIX + c for c in captions], batch_size=256)
    td, td_len, _ = extract_text_tokens(model, [DOCUMENT_PREFIX + c for c in captions], batch_size=256)
    _write_split(
        name,
        audio,
        a_len,
        tq,
        tq_len,
        td,
        td_len,
        {
            "split": "audiocaps_test",
            "ids": [int(c["audiocap_id"]) for c in clips],
            "clip_index": clip_index,
            "captions": captions,
            "audio_files": [c["file"] for c in clips],
            "audio_seconds": [len(a) / AUDIO_SR for a in arrs],
        },
    )


def clotho_queries() -> tuple[List[str], List[str]]:
    import pyarrow.parquet as pq

    m = json.loads((CLOTHO_ROOT / "mteb_clotho" / "manifest.json").read_text())
    wanted = [qid for c in m["clips"] for qid in c["captions"]]
    table = pq.read_table(CLOTHO_ROOT / "mteb_clotho" / "queries" / "test-00000-of-00001.parquet")
    texts = {row["id"]: row["text"] for row in table.to_pylist()}
    missing = [q for q in wanted if q not in texts]
    if missing:
        raise RuntimeError(f"missing {len(missing)} clotho queries")
    return wanted, [texts[q] for q in wanted]


def build_clotho_test(model) -> None:
    name = "clotho_test"
    if (CROSS_CACHE / name / "meta.json").exists():
        print(f"[cross-cache] {name} exists, skip")
        return
    m = json.loads((CLOTHO_ROOT / "mteb_clotho" / "manifest.json").read_text())
    clips = m["clips"]
    arrs = [decode_file(CLOTHO_ROOT / "mteb_clotho" / c["file"]) for c in clips]
    print(f"[cross-cache] clotho audio seconds max={max(len(a) for a in arrs) / AUDIO_SR:.1f}", flush=True)
    audio, a_len, _ = extract_audio_tokens(model, arrs, batch_size=16)
    qids, texts = clotho_queries()
    tq, tq_len, _ = extract_text_tokens(model, [QUERY_PREFIX + t for t in texts], batch_size=256)
    td, td_len, _ = extract_text_tokens(model, [DOCUMENT_PREFIX + t for t in texts], batch_size=256)
    _write_split(
        name,
        audio,
        a_len,
        tq,
        tq_len,
        td,
        td_len,
        {
            "split": "clotho_test",
            "ids": [c["id"] for c in clips],
            "query_ids": qids,
            "texts": texts,
            "clip_index": [i for i, c in enumerate(clips) for _ in c["captions"]],
            "audio_files": [c["file"] for c in clips],
            "audio_seconds": [len(a) / AUDIO_SR for a in arrs],
            "note": "Clotho v2.1 test, mteb/clotho_t2a_v2; never used for training or selection.",
        },
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits", default="all")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    CROSS_CACHE.mkdir(parents=True, exist_ok=True)
    model = load_gemma(args.device)
    todo = ["audiocaps_train", "audiocaps_test", "clotho_test"] if args.splits == "all" else args.splits.split(",")
    for split in todo:
        if split == "audiocaps_train":
            build_audiocaps_train(model)
        elif split == "audiocaps_test":
            build_audiocaps_test(model)
        elif split == "clotho_test":
            build_clotho_test(model)
        else:
            raise SystemExit(f"unknown split {split}")


if __name__ == "__main__":
    main()
