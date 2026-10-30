#!/usr/bin/env python
"""Cache AudioCaps-validation Gemma(+nothing) embeddings for rerank tuning.

Builds the exact test recipe (audio embedding for each clip; QUERY_PREFIX for
t2a captions; DOCUMENT_PREFIX for a2t caption documents) on the official
AudioCaps validation split, using the same ids/captions captured by
``data/audiocaps_val/features/*.npz`` (CLAP feature cache).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("OMURA_EMBEDDING_BACKEND", "omuragemma")
os.environ.setdefault(
    "OMURA_VECTOR_STORE_DIR",
    str(Path(__file__).resolve().parents[1] / "data" / "audiocaps_val" / "vector_store_scratch"),
)

from benchmarks import embedding_backend as eb  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
VAL = REPO / "data" / "audiocaps_val"
OUT = VAL / "gemma_val_emb.npz"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    shards = sorted(glob.glob(str(VAL / "features" / "*.npz")))
    keys: List[str] = []
    caps: List[str] = []
    text_clip: List[int] = []
    key_to_idx: Dict[str, int] = {}
    for shard in shards:
        z = np.load(shard, allow_pickle=True)
        sk = [str(k) for k in z["keys"].tolist()]
        for k in sk:
            if k not in key_to_idx:
                key_to_idx[k] = len(keys)
                keys.append(k)
        caps.extend([str(c) for c in z["caps"].tolist()])
        text_clip.extend(key_to_idx[sk[int(c)]] for c in z["text_clip"].tolist())
    print(f"[val] {len(keys)} clips / {len(caps)} captions from {len(shards)} shards")

    # Collect audio bytes per clip key from the validation parquet shards.
    import pyarrow.parquet as pq
    by_key: Dict[str, bytes] = {}
    for path in sorted((VAL / "parquet").glob("validation-*.parquet")):
        for row in pq.read_table(path).to_pylist():
            audio = row.get("audio") or {}
            blob = audio.get("bytes")
            if not blob:
                continue
            key = f"{row.get('youtube_id')}|{int(row.get('start_time') or 0)}"
            by_key.setdefault(key, blob)
    missing = [k for k in keys if k not in by_key]
    if missing:
        raise RuntimeError(f"{len(missing)} validation clips missing audio bytes")
    datas = [by_key[k] for k in keys]
    exts = ["flac"] * len(datas)

    eb.initialize_embedding_model()
    dim = int(eb.EMBEDDING_DIM or 768)
    print(f"[val] gemma dim={dim}")

    audio: List[np.ndarray] = []
    for start in range(0, len(datas), 8):
        chunk = datas[start:start + 8]
        vecs = eb.generate_audio_embeddings(chunk, exts=exts[start:start + 8],
                                            blob_ids=keys[start:start + 8])
        audio.extend(vecs)
    audio = np.stack([np.asarray(v, np.float32) for v in audio]).astype(np.float32)
    audio /= np.maximum(np.linalg.norm(audio, axis=1, keepdims=True), 1e-12)

    tq: List[np.ndarray] = []
    for start in range(0, len(caps), 64):
        vecs = eb.generate_text_embeddings(caps[start:start + 64], is_document=False)
        tq.extend(vecs)
    tq = np.stack([np.asarray(v, np.float32) for v in tq]).astype(np.float32)
    tq /= np.maximum(np.linalg.norm(tq, axis=1, keepdims=True), 1e-12)

    td: List[np.ndarray] = []
    for start in range(0, len(caps), 64):
        vecs = eb.generate_text_embeddings(caps[start:start + 64], is_document=True)
        td.extend(vecs)
    td = np.stack([np.asarray(v, np.float32) for v in td]).astype(np.float32)
    td /= np.maximum(np.linalg.norm(td, axis=1, keepdims=True), 1e-12)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        audio=audio,
        text_query=tq,
        text_doc=td,
        keys=np.array(keys, dtype=object),
        caps=np.array(caps, dtype=object),
        text_clip=np.array(text_clip, dtype=np.int32),
    )
    print(f"[val] wrote {args.out} audio={audio.shape} tq={tq.shape} td={td.shape}")
    (args.out.with_suffix(".json")).write_text(json.dumps({
        "split": "AudioCaps validation",
        "clips": len(keys), "captions": len(caps),
        "model": eb.MODEL_NAME, "dim": dim,
    }, indent=2))


if __name__ == "__main__":
    main()
