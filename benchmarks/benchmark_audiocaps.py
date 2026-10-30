"""AudioCaps text<->audio retrieval with EmbeddingGemma 2.

Data source: the public ``OpenSound/AudioCaps`` parquet shards (each row has
the 10 s clip bytes + one caption). Shards are cached under
``data/audiocaps/parquet`` and decoded into ``data/audiocaps/audio``.

Protocol: full AudioCaps test pool (all unique clips). Every caption is a
text->audio query; every clip is an audio->text query whose qrels are that
clip's five captions.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent))

import embedding_backend as eb
from bench_common import (
    REPO_ROOT,
    embed_texts,
    git_commit,
    ranks_at_k,
    recall_at_k,
    utc_now,
    write_result,
)

DEFAULT_ROOT = REPO_ROOT / "data" / "audiocaps"
SHARD_TEMPLATE = (
    "https://huggingface.co/datasets/OpenSound/AudioCaps/resolve/main/"
    "data/test-{idx:05d}-of-00041.parquet"
)


def ensure_parquet(root: Path, download: bool) -> List[Path]:
    pdir = root / "parquet"
    pdir.mkdir(parents=True, exist_ok=True)
    files = [
        f
        for f in sorted(pdir.glob("test-*-of-00041.parquet"))
        if f.stat().st_size > 1000
    ]
    if len(files) >= 41:
        return files
    if not download:
        raise FileNotFoundError(f"Missing AudioCaps parquet shards under {pdir}")
    from concurrent.futures import ThreadPoolExecutor

    def _fetch(idx: int) -> None:
        dst = pdir / f"test-{idx:05d}-of-00041.parquet"
        if dst.exists() and dst.stat().st_size > 1000:
            return
        url = SHARD_TEMPLATE.format(idx=idx)
        tmp = dst.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        if tmp.stat().st_size < 1000:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"shard {idx} download failed (too small)")
        tmp.replace(dst)

    print("[AudioCaps] downloading missing shards")
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(_fetch, range(41)))
    files = [
        f
        for f in sorted(pdir.glob("test-*-of-00041.parquet"))
        if f.stat().st_size > 1000
    ]
    if len(files) < 41:
        raise RuntimeError(f"only {len(files)}/41 AudioCaps shards available")
    return files


def build_manifest(root: Path, files: List[Path], force: bool = False) -> Path:
    import pyarrow.parquet as pq

    manifest_path = root / "manifest.json"
    audio_dir = root / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists() and not force:
        data = json.loads(manifest_path.read_text())
        if (
            data.get("version") == 2
            and data.get("clips")
            and all((root / c["file"]).exists() for c in data["clips"])
        ):
            return manifest_path

    # AudioCaps stores 5 caption rows per clip with distinct audiocap_ids but the
    # same (youtube_id, start_time). Group by the clip key, not by caption id.
    by_clip: Dict[tuple, Dict[str, Any]] = {}
    for path in tqdm(files, desc="Reading AudioCaps shards"):
        table = pq.read_table(path)
        for row in table.to_pylist():
            cap_id = int(row["audiocap_id"])
            key = (str(row.get("youtube_id")), int(row.get("start_time") or 0))
            caption = " ".join(str(row["caption"]).split())
            audio = row.get("audio") or {}
            blob = audio.get("bytes")
            if not blob:
                continue
            entry = by_clip.setdefault(
                key,
                {
                    "audiocap_id": cap_id,
                    "youtube_id": key[0],
                    "start_time": key[1],
                    "captions": [],
                    "audio_bytes": blob,
                    "audio_path": audio.get("path") or "",
                    "audio_length": int(row.get("audio_length") or 0),
                },
            )
            if caption and caption not in entry["captions"]:
                entry["captions"].append(caption)

    clips: List[Dict[str, Any]] = []
    for key in sorted(by_clip):
        entry = by_clip[key]
        ext = Path(entry["audio_path"]).suffix.lstrip(".") or "flac"
        dst = audio_dir / f"{entry['audiocap_id']}.{ext}"
        if not dst.exists():
            dst.write_bytes(entry["audio_bytes"])
        clips.append(
            {
                "audiocap_id": entry["audiocap_id"],
                "youtube_id": entry["youtube_id"],
                "start_time": entry["start_time"],
                "captions": entry["captions"],
                "file": str(dst.relative_to(root)),
                "ext": ext,
                "bytes": len(entry["audio_bytes"]),
                "audio_length": entry["audio_length"],
            }
        )

    manifest = {
        "version": 2,
        "name": "audiocaps_test",
        "source": "OpenSound/AudioCaps (HuggingFace) test shards",
        "created_utc": utc_now(),
        "num_clips": len(clips),
        "num_captions": sum(len(c["captions"]) for c in clips),
        "clips": clips,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[AudioCaps] manifest: {len(clips)} clips, {manifest['num_captions']} captions")
    return manifest_path


def embed_audio_batched(
    datas: List[bytes], exts: List[str], ids: List[str], dim: int, batch_size: int
) -> tuple[np.ndarray, List[int]]:
    out: List[np.ndarray | None] = []
    for start in tqdm(range(0, len(datas), batch_size), desc="Embedding audio", unit="batch"):
        chunk = datas[start : start + batch_size]
        vecs = eb.generate_audio_embeddings(
            chunk, exts=exts[start : start + batch_size], blob_ids=ids[start : start + batch_size]
        )
        out.extend(vecs)
    kept, rows = [], []
    for i, v in enumerate(out):
        if v is None:
            continue
        row = np.asarray(v, dtype=np.float32).reshape(-1)
        if row.size != dim:
            continue
        n = float(np.linalg.norm(row))
        if n > 0:
            row = row / n
        kept.append(row)
        rows.append(i)
    return (np.stack(kept, axis=0) if kept else np.zeros((0, dim), np.float32)), rows


def main() -> None:
    ap = argparse.ArgumentParser(description="AudioCaps retrieval benchmark")
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--force-manifest", action="store_true")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="Smoke-test cap on clips.")
    ap.add_argument(
        "--out-json",
        type=Path,
        default=REPO_ROOT / "benchmarks" / "results" / "audiocaps_gemma.json",
    )
    args = ap.parse_args()

    out_json = args.out_json
    try:
        files = ensure_parquet(args.root, download=not args.no_download)
        manifest_path = build_manifest(args.root, files, force=args.force_manifest)
        manifest = json.loads(manifest_path.read_text())
    except Exception as exc:  # noqa: BLE001
        payload = {
            "model": eb.MODEL_NAME,
            "dim": int(eb.EMBEDDING_DIM or 768),
            "protocol": "audiocaps_test_full_pool_t2a_and_a2t",
            "dataset": "AudioCaps test",
            "pool_size": 0,
            "num_queries": 0,
            "seed": args.seed,
            "commit": git_commit(),
            "utc_date": utc_now(),
            "status": "no_data",
            "blocker": f"{type(exc).__name__}: {exc}",
        }
        write_result(out_json, payload)
        print(json.dumps(payload, indent=2))
        return

    clips = manifest["clips"]
    if args.limit:
        clips = clips[: args.limit]
    print(f"[AudioCaps] clips: {len(clips)}  captions: {sum(len(c['captions']) for c in clips)}")
    print(f"[AudioCaps] model: {eb.MODEL_NAME} dim={eb.EMBEDDING_DIM}")

    eb.initialize_embedding_model()
    dim = int(eb.EMBEDDING_DIM or 768)

    datas, exts, ids, clip_rows = [], [], [], []
    for i, c in enumerate(clips):
        p = args.root / c["file"]
        try:
            datas.append(p.read_bytes())
            exts.append(c.get("ext") or p.suffix.lstrip("."))
            ids.append(str(c["audiocap_id"]))
            clip_rows.append(i)
        except OSError:
            continue

    audio_vecs, kept_audio = embed_audio_batched(datas, exts, ids, dim, args.batch_size)
    clip_rows = [clip_rows[i] for i in kept_audio]
    kept_clips = [clips[i] for i in clip_rows]

    # t2a: every caption is a query.
    query_texts: List[str] = []
    t2a_gt: List[np.ndarray] = []
    for a_idx, clip in enumerate(kept_clips):
        for cap in clip["captions"]:
            query_texts.append(cap)
            t2a_gt.append(np.asarray([a_idx], dtype=np.int64))
    text_vecs, kept_text = embed_texts(
        eb.generate_text_embeddings,
        query_texts,
        dim,
        batch_size=max(32, args.batch_size * 4),
        desc="Embedding AudioCaps captions",
    )
    t2a_gt = [t2a_gt[i] for i in kept_text]

    sims_t2a = text_vecs @ audio_vecs.T
    t2a = recall_at_k(sims_t2a, t2a_gt)
    t2a_ranks = ranks_at_k(sims_t2a, t2a_gt)
    t2a_qids = [
        {"audiocap_id": kept_clips[int(g[0])]["audiocap_id"], "rank": int(r)}
        for g, r in zip(t2a_gt, t2a_ranks)
    ]

    # a2t: each clip retrieves its own captions among all captions.
    cand_texts: List[str] = []
    a2t_gt: List[np.ndarray] = []
    cursor = 0
    for clip in kept_clips:
        caps = clip["captions"]
        cand_texts.extend(caps)
        a2t_gt.append(np.arange(cursor, cursor + len(caps), dtype=np.int64))
        cursor += len(caps)
    cand_vecs, kept_cand = embed_texts(
        eb.generate_text_embeddings,
        cand_texts,
        dim,
        batch_size=max(32, args.batch_size * 4),
        desc="Embedding AudioCaps caption candidates",
        is_document=True,
    )
    if len(kept_cand) != len(cand_texts):
        remap = {old: new for new, old in enumerate(kept_cand)}
        a2t_gt = [
            np.asarray([remap[int(x)] for x in arr if int(x) in remap], dtype=np.int64)
            for arr in a2t_gt
        ]
    sims_a2t = audio_vecs @ cand_vecs.T
    a2t = recall_at_k(sims_a2t, a2t_gt)
    a2t_ranks = ranks_at_k(sims_a2t, a2t_gt)

    payload = {
        "model": eb.MODEL_NAME,
        "dim": dim,
        "backend": "omuragemma" if eb.IS_OMURAGEMMA else "siglip",
        "protocol": "audiocaps_test_full_pool_t2a_and_a2t",
        "dataset": "AudioCaps test (OpenSound/AudioCaps)",
        "seed": args.seed,
        "commit": git_commit(),
        "utc_date": utc_now(),
        "pool_size": int(audio_vecs.shape[0]),
        "num_images": int(audio_vecs.shape[0]),
        "num_queries": int(text_vecs.shape[0]),
        "num_caption_candidates": int(cand_vecs.shape[0]),
        "caption_count": int(sum(len(c["captions"]) for c in kept_clips)),
        "text_to_audio": t2a,
        "audio_to_text": a2t,
        "audio_seconds_cap": float(
            __import__("os").getenv("OMURA_OMURAGEMMA_AUDIO_SECONDS", "300")
        ),
        "per_query_ranks": {
            "text_to_audio": t2a_qids,
            "audio_to_text": [
                {"audiocap_id": c["audiocap_id"], "rank": int(r)}
                for c, r in zip(kept_clips, a2t_ranks)
            ],
        },
        "notes": (
            "Full test pool; no sampled negatives. 5 captions per clip, so "
            "audio->text R@1 is bounded by 5/num_caption_candidates chance."
        ),
    }
    write_result(out_json, payload)
    print(json.dumps({k: payload[k] for k in ("text_to_audio", "audio_to_text", "pool_size", "num_queries")}, indent=2))
    print(f"[AudioCaps] Wrote: {out_json}")


if __name__ == "__main__":
    main()
