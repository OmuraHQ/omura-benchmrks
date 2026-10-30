#!/usr/bin/env python
"""Cache language-model-contextualised Gemma audio tokens for the reranker.

Complements ``build_crossattn_cache.py``: instead of the pre-LM
``embed_audio(audio_tower(x))`` projector output, this stores the hidden state
of the *language model* at each audio placeholder token (768-d), i.e. the same
space as the text token states.

Output: ``data/audiocaps_crossattn_cache_lm/<split>/audio_lm_tok.npy`` plus a
``meta.json`` that reuses the split metadata from the base cache.

Usage::

    CUDA_VISIBLE_DEVICES=5 .venv/bin/python benchmarks/build_crossattn_lm_cache.py --splits all
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List, Sequence

import numpy as np

from bench_common import REPO_ROOT
from crossattn_common import AUDIO_SR, CROSS_CACHE, load_gemma
from build_crossattn_cache import decode_file, clotho_queries

LM_CACHE = REPO_ROOT / "data" / "audiocaps_crossattn_cache_lm"
AUDIO_TOKEN_ID = 258881


def extract_audio_lm_tokens(model, arrays: Sequence[np.ndarray], batch_size: int = 32):
    import torch

    trans = model[0]
    root = trans.auto_model
    rows = []
    lens = []
    max_tok = 0
    t0 = time.time()
    with torch.no_grad():
        for start in range(0, len(arrays), batch_size):
            chunk = [{"audio": a} for a in arrays[start : start + batch_size]]
            batch = trans.preprocess(chunk)
            batch.pop("modality", None)
            batch = {k: (v.to("cuda") if torch.is_tensor(v) else v) for k, v in batch.items()}
            out = root(**batch, return_dict=True)
            hidden = out.last_hidden_state.float().cpu().numpy()
            ids = batch["input_ids"].cpu().numpy()
            for r in range(len(chunk)):
                pos = np.where(ids[r] == AUDIO_TOKEN_ID)[0]
                rows.append(hidden[r, pos].astype(np.float16))
                lens.append(len(pos))
                max_tok = max(max_tok, len(pos))
            if start % (batch_size * 40) == 0:
                rate = (start + len(chunk)) / max(1e-9, time.time() - t0)
                print(
                    f"[lm-cache] {start + len(chunk)}/{len(arrays)} max_tok={max_tok} "
                    f"{rate:.1f}/s eta {(len(arrays) - start - len(chunk)) / max(rate, 1e-9) / 60:.1f} min",
                    flush=True,
                )
    out = np.zeros((len(arrays), max_tok, 768), dtype=np.float16)
    for i, r in enumerate(rows):
        out[i, : r.shape[0]] = r
    return out, np.asarray(lens, dtype=np.int32)


def _load_audio(split: str) -> List[np.ndarray]:
    if split == "audiocaps_train":
        meta = json.loads((CROSS_CACHE / split / "meta.json").read_text())
        from crossattn_common import ADAPTER_CACHE

        slot = 160000
        mem = np.memmap(ADAPTER_CACHE / "train_audio_i16.bin", dtype=np.int16, mode="r")
        lens = meta["audio_seconds"]
        out = []
        for i, sec in enumerate(lens):
            n = int(round(sec * AUDIO_SR))
            out.append((mem[i * slot : i * slot + n].astype(np.float32) / 32767.0).copy())
        return out
    if split == "audiocaps_test":
        m = json.loads((REPO_ROOT / "data" / "audiocaps" / "manifest.json").read_text())
        return [decode_file(REPO_ROOT / "data" / "audiocaps" / c["file"]) for c in m["clips"]]
    if split == "clotho_test":
        m = json.loads((REPO_ROOT / "data" / "clotho" / "mteb_clotho" / "manifest.json").read_text())
        return [decode_file(REPO_ROOT / "data" / "clotho" / "mteb_clotho" / c["file"]) for c in m["clips"]]
    raise SystemExit(split)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits", default="all")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()
    splits = ["audiocaps_train", "audiocaps_test", "clotho_test"] if args.splits == "all" else args.splits.split(",")
    model = load_gemma("cuda")
    for split in splits:
        out_dir = LM_CACHE / split
        if (out_dir / "meta.json").exists():
            print(f"[lm-cache] {split} exists, skip")
            continue
        arrays = _load_audio(split)
        print(f"[lm-cache] {split}: {len(arrays)} clips", flush=True)
        toks, lens = extract_audio_lm_tokens(model, arrays, batch_size=args.batch_size)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "audio_lm_tok.npy", toks)
        base = json.loads((CROSS_CACHE / split / "meta.json").read_text())
        base.update(
            {
                "audio_lm_tokens": int(toks.shape[1]),
                "audio_lm_dim": 768,
                "audio_lm_lengths": lens.astype(int).tolist(),
                "audio_lm_source": "language_model last_hidden_state at <|audio|> placeholder positions (audio-only forward)",
            }
        )
        (out_dir / "meta.json").write_text(json.dumps(base), encoding="utf-8")
        print(f"[lm-cache] wrote {split}: {toks.shape}", flush=True)


if __name__ == "__main__":
    main()
