"""AudioCaps text<->audio retrieval with the dedicated CLAP audio model.

This is the Phase-2 audio baseline that replaces the unified EmbeddingGemma 2
audio path (35.19% text->audio R@10) with ``laion/larger_clap_general``.

Protocol is kept identical to ``benchmark_audiocaps.py``:

* candidate pool = all unique AudioCaps test clips in ``data/audiocaps``
  (883 clips for the public OpenSound mirror, 4,308 captions);
* text->audio query = each caption, positive = its own clip;
* audio->text query = each clip, positives = that clip's five test captions
  among all 4,308 caption documents (no test caption of another clip is ever
  used as a positive, and no train metadata is indexed);
* no sampling, no rerank, full pool.

Embeddings are cached as ``.npz`` so the supervised-adapter step can reuse the
exact same frozen CLAP features. The optional ``immortaltatsu/omura-embed-audio``
head is force-disabled here so the baseline is pure CLAP.

Run:
    CUDA_VISIBLE_DEVICES=5 OMURA_CLAP_ADAPTER=false \
        python benchmarks/benchmark_audiocaps_clap.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent))

from bench_common import (  # noqa: E402
    REPO_ROOT,
    git_commit,
    l2_normalize,
    ranks_at_k,
    recall_at_k,
    utc_now,
    write_result,
)

DEFAULT_ROOT = REPO_ROOT / "data" / "audiocaps"
DEFAULT_CACHE = REPO_ROOT / "data" / "audiocaps_clap_cache"


def load_manifest(root: Path) -> Dict:
    path = root / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing; run benchmarks/benchmark_audiocaps.py once to build it"
        )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("version") != 2:
        raise RuntimeError(f"unexpected manifest version: {manifest.get('version')}")
    return manifest


def _load_cache(path: Path, expected: int, dim: int) -> np.ndarray | None:
    if not path.exists():
        return None
    try:
        arr = np.load(path)["vecs"].astype(np.float32)
    except Exception:  # noqa: BLE001
        return None
    if arr.shape != (expected, dim):
        return None
    return arr


def _save_cache(path: Path, vecs: np.ndarray, meta: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, vecs=vecs.astype(np.float32), meta=json.dumps(meta))


def embed_audio_clips(clips: List[Dict], root: Path, dim: int) -> tuple[np.ndarray, List[int]]:
    from omura.utils import clap_embeddings as clap

    vecs: List[np.ndarray | None] = []
    for clip in tqdm(clips, desc="CLAP audio", unit="clip"):
        path = root / clip["file"]
        try:
            data = path.read_bytes()
        except OSError:
            vecs.append(None)
            continue
        vecs.append(clap.embed_audio(data, ext=clip.get("ext") or path.suffix.lstrip("."),
                                     blob_id=str(clip["audiocap_id"])))
    kept = [i for i, v in enumerate(vecs) if v is not None and np.asarray(v).size == dim]
    if not kept:
        return np.zeros((0, dim), np.float32), []
    return np.stack([l2_normalize(vecs[i]) for i in kept]).astype(np.float32), kept


def embed_captions(captions: List[str], dim: int) -> tuple[np.ndarray, List[int]]:
    from omura.utils import clap_embeddings as clap

    vecs: List[np.ndarray | None] = []
    for text in tqdm(captions, desc="CLAP text", unit="caption"):
        vecs.append(clap.embed_text(text))
    kept = [i for i, v in enumerate(vecs) if v is not None and np.asarray(v).size == dim]
    if not kept:
        return np.zeros((0, dim), np.float32), []
    return np.stack([l2_normalize(vecs[i]) for i in kept]).astype(np.float32), kept


def caption_queries(clips: List[Dict], clip_rows: List[int]) -> tuple[List[str], List[np.ndarray]]:
    texts: List[str] = []
    gt: List[np.ndarray] = []
    for a_idx, row in enumerate(clip_rows):
        for cap in clips[row]["captions"]:
            texts.append(cap)
            gt.append(np.asarray([a_idx], dtype=np.int64))
    return texts, gt


def main() -> None:
    ap = argparse.ArgumentParser(description="AudioCaps CLAP retrieval benchmark")
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--out-json", type=Path,
                    default=REPO_ROOT / "benchmarks" / "results" / "audiocaps_clap.json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="Smoke-test cap on clips.")
    ap.add_argument("--no-cache", action="store_true", help="Ignore cached CLAP vectors.")
    args = ap.parse_args()

    # Pure-CLAP baseline: the optional fine-tuned head must stay off.
    os.environ["OMURA_CLAP_ADAPTER"] = "false"

    from omura.utils import clap_embeddings as clap

    manifest = load_manifest(args.root)
    clips = manifest["clips"]
    if args.limit:
        clips = clips[: args.limit]

    all_captions: List[str] = [c for clip in clips for c in clip["captions"]]
    dim = int(clap.CLAP_DIM)
    print(f"[CLAP] model={clap.CLAP_MODEL} dim={dim} clips={len(clips)} "
          f"captions={len(all_captions)}")

    audio_cache = args.cache_dir / f"clap_audio_{len(clips)}.npz"
    text_cache = args.cache_dir / f"clap_text_{len(all_captions)}.npz"

    audio_vecs = None if args.no_cache else _load_cache(audio_cache, len(clips), dim)
    if audio_vecs is None:
        audio_vecs, kept_audio = embed_audio_clips(clips, args.root, dim)
        if len(kept_audio) != len(clips):
            print(f"[CLAP] WARNING: kept {len(kept_audio)}/{len(clips)} audio clips")
        _save_cache(audio_cache, audio_vecs, {"model": clap.CLAP_MODEL, "dim": dim,
                                              "clips": len(clips)})
    else:
        kept_audio = list(range(len(clips)))
        print(f"[CLAP] audio cache hit: {audio_cache}")

    text_vecs = None if args.no_cache else _load_cache(text_cache, len(all_captions), dim)
    if text_vecs is None:
        text_vecs, kept_text = embed_captions(all_captions, dim)
        if len(kept_text) != len(all_captions):
            print(f"[CLAP] WARNING: kept {len(kept_text)}/{len(all_captions)} captions")
        _save_cache(text_cache, text_vecs, {"model": clap.CLAP_MODEL, "dim": dim,
                                            "captions": len(all_captions)})
    else:
        kept_text = list(range(len(all_captions)))
        print(f"[CLAP] text cache hit: {text_cache}")

    kept_clips = [clips[i] for i in kept_audio]
    query_texts, t2a_gt = caption_queries(clips, kept_audio)
    # Map caption positions -> kept rows if any caption failed.
    if len(kept_text) != len(all_captions):
        remap = {old: new for new, old in enumerate(kept_text)}
        query_texts = [t for i, t in enumerate(query_texts) if i in remap]
        t2a_gt = [g for i, g in enumerate(t2a_gt) if i in remap]

    sims_t2a = text_vecs @ audio_vecs.T
    t2a = recall_at_k(sims_t2a, t2a_gt)
    t2a_ranks = ranks_at_k(sims_t2a, t2a_gt)

    # a2t: clip -> its own five captions among all caption documents.
    text_remap = {old: new for new, old in enumerate(kept_text)}
    a2t_gt: List[np.ndarray] = []
    a2t_audio_rows: List[int] = []
    cursor = 0
    for a_idx, clip_row in enumerate(kept_audio):
        caps = clips[clip_row]["captions"]
        positives = np.asarray(
            [text_remap[i] for i in range(cursor, cursor + len(caps)) if i in text_remap],
            dtype=np.int64,
        )
        cursor += len(caps)
        if positives.size:
            a2t_gt.append(positives)
            a2t_audio_rows.append(a_idx)
    sims_a2t = audio_vecs[a2t_audio_rows] @ text_vecs.T
    a2t = recall_at_k(sims_a2t, a2t_gt)
    a2t_ranks = ranks_at_k(sims_a2t, a2t_gt)

    payload = {
        "model": clap.CLAP_MODEL,
        "dim": dim,
        "backend": "clap",
        "adapter": "none (pure frozen CLAP)",
        "protocol": "audiocaps_test_full_pool_t2a_and_a2t",
        "dataset": "AudioCaps test (OpenSound/AudioCaps, 883 unique clips)",
        "seed": args.seed,
        "commit": git_commit(),
        "utc_date": utc_now(),
        "pool_size": int(audio_vecs.shape[0]),
        "num_images": int(audio_vecs.shape[0]),
        "num_queries": int(text_vecs.shape[0]),
        "num_caption_candidates": int(text_vecs.shape[0]),
        "caption_count": int(sum(len(c["captions"]) for c in kept_clips)),
        "text_to_audio": t2a,
        "audio_to_text": a2t,
        "per_query_ranks": {
            "text_to_audio": [
                {"audiocap_id": kept_clips[int(g[0])]["audiocap_id"], "rank": int(r)}
                for g, r in zip(t2a_gt, t2a_ranks)
            ],
            "audio_to_text": [
                {"audiocap_id": kept_clips[i]["audiocap_id"], "rank": int(r)}
                for i, r in zip(a2t_audio_rows, a2t_ranks)
            ],
        },
        "cache": {
            "audio": str(audio_cache),
            "text": str(text_cache),
        },
        "notes": (
            "Full 883-clip test pool, no sampled negatives, no rerank. Audio "
            "documents are audio-only CLAP embeddings; caption documents are the "
            "AudioCaps test captions only (5 per clip). Same ids/captions as "
            "benchmarks/results/audiocaps_gemma.json."
        ),
    }
    write_result(args.out_json, payload)
    print(json.dumps({k: payload[k] for k in
                      ("model", "text_to_audio", "audio_to_text", "pool_size", "num_queries")},
                     indent=2))
    print(f"[CLAP] Wrote: {args.out_json}")


if __name__ == "__main__":
    main()
