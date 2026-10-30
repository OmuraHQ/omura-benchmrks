#!/usr/bin/env python
"""Cache frozen CLAP features for the Clotho v2.1 test split."""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
from typing import List
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

from finetune_audiocaps_adapter import (  # noqa: E402
    CLAP_MODEL, CLAP_SR, _decode_wav, embed_texts_clap, embed_wavs, load_clap,
)

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "data" / "clotho"
OUT = ROOT / "cache" / "clap_frozen.npz"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    manifest = json.loads((ROOT / "mteb_clotho" / "manifest.json").read_text())
    clips = manifest["clips"]
    queries = manifest["queries"]
    print(f"[clotho-clap] {len(clips)} clips / {len(queries)} captions")

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, processor = load_clap(device)

    wavs = []
    for i, clip in enumerate(clips):
        blob = (ROOT / "mteb_clotho" / clip["file"]).read_bytes()
        wav = _decode_wav(blob)
        if wav is None:
            raise RuntimeError(f"decode failed: {clip['id']}")
        wavs.append(wav)
    audio = embed_wavs(model, processor, device, wavs, batch_size=16)
    texts = [q["text"] for q in queries]
    text = embed_texts_clap(model, processor, device, texts, batch_size=256)
    np.savez_compressed(
        args.out,
        audio=audio.astype(np.float32),
        text_query=text.astype(np.float32),
        text_doc=text.astype(np.float32),
        clip_ids=np.array([c["id"] for c in clips], dtype=object),
        query_ids=np.array([q["id"] for q in queries], dtype=object),
    )
    print(f"[clotho-clap] wrote {args.out} audio={audio.shape} text={text.shape}")


if __name__ == "__main__":
    main()
