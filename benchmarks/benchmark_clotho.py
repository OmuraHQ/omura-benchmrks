#!/usr/bin/env python
"""Clotho v2.1 test retrieval with EmbeddingGemma 2 (cross-dataset / OOD).

Clotho is the held-out audio dataset required by ``docs/GENERALIZATION_PROTOCOL.md``:
it is never used for training or model selection.  The AudioCaps adapter is
trained only on AudioCaps, so evaluating it here measures whether the gain
generalises beyond the dataset it was fitted on.

Data source: ``mteb/clotho_t2a_v2`` (HuggingFace), the canonical MTEB audio
retrieval packaging of the Clotho v2.1 **test** split: 1045 clips and 5225
captions with a full qrels matrix.  Cached under ``data/clotho/``.

Protocol (nothing is shrunk, nothing is tuned):

* text -> audio: each of the 5225 captions is a query against all 1045 clips.
* audio -> text: each of the 1045 clips is a query against all 5225 caption
  documents (its own 5 captions are the positives).
* The model card prefixes only: ``prepare_query`` for queries,
  ``prepare_document`` for caption documents.  No per-dataset prompt tuning.
* Random baseline is the exact chance recall for ``m`` positives in ``N``
  candidates: ``1 - C(N-m, k) / C(N, k)``.

Usage::

    CUDA_VISIBLE_DEVICES=3 .venv/bin/python benchmarks/benchmark_clotho.py

    # with a *final* AudioCaps adapter (frozen base + projection heads)
    CUDA_VISIBLE_DEVICES=3 .venv/bin/python benchmarks/benchmark_clotho.py \
        --adapter data/audiocaps_gemma_adapter.pt \
        --out benchmarks/results/clotho_gemma_adapter.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent))

# This benchmark must never touch the live store (``data/vector_index``): the
# backend writes an ownership marker on init. Pin a scratch store inside
# ``data/clotho/`` unless the caller explicitly chose one.
os.environ.setdefault("OMURA_EMBEDDING_BACKEND", "omuragemma")
os.environ.setdefault(
    "OMURA_VECTOR_STORE_DIR",
    str(Path(__file__).resolve().parents[1] / "data" / "clotho" / "vector_store_scratch"),
)

from bench_common import (  # noqa: E402
    REPO_ROOT,
    embed_texts,
    git_commit,
    ranks_at_k,
    recall_at_k,
    utc_now,
    write_result,
)

DATASET = "mteb/clotho_t2a_v2"
DEFAULT_ROOT = REPO_ROOT / "data" / "clotho"
DEFAULT_DATA = DEFAULT_ROOT / "mteb_clotho"
CACHE_DIR = DEFAULT_ROOT / "cache"
RESULTS = REPO_ROOT / "benchmarks" / "results"
ADAPTER_MANIFEST_DIR = REPO_ROOT / "data" / "audiocaps_gemma_cache"
ADAPTER_MIN_STABLE_SECONDS = 120.0


# ── data ─────────────────────────────────────────────────────────────────────


def ensure_dataset(root: Path, download: bool) -> Path:
    """Return a directory holding corpus/, queries/ and qrels/ parquet files."""
    if (root / "queries").is_dir() and (root / "qrels").is_dir() and (root / "corpus").is_dir():
        return root
    # The corpus parquets may have been pruned after the audio was extracted;
    # manifest.json + audio/ are then enough to evaluate.
    if (root / "manifest.json").exists() and (root / "audio").is_dir():
        return root
    if not download:
        raise FileNotFoundError(f"missing Clotho parquets under {root}")
    from huggingface_hub import snapshot_download

    root.parent.mkdir(parents=True, exist_ok=True)
    print(f"[Clotho] downloading {DATASET} -> {root}")
    snapshot_download(
        DATASET,
        repo_type="dataset",
        allow_patterns=["queries/*", "qrels/*", "corpus/*"],
        local_dir=str(root),
        max_workers=8,
    )
    return root


def _audio_struct(blob: Any) -> tuple[Optional[bytes], str]:
    if isinstance(blob, dict):
        return blob.get("bytes"), str(blob.get("path") or "")
    if isinstance(blob, (bytes, bytearray)):
        return bytes(blob), ""
    return None, ""


def build_manifest(root: Path, force: bool = False) -> Path:
    """Materialise the corpus audio and the clip -> caption-id mapping.

    Writes ``<root>/manifest.json`` with ``clips`` (id, file, ext) and the
    ``queries`` list (id, text) in qrels order.  Audio is written once to
    ``<root>/audio/`` so re-runs do not re-download.
    """
    manifest_path = root / "manifest.json"
    audio_dir = root / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists() and not force:
        data = json.loads(manifest_path.read_text())
        if data.get("version") == 1 and data.get("clips"):
            if all((root / c["file"]).exists() for c in data["clips"]):
                return manifest_path

    import pyarrow.parquet as pq

    corpus_files = sorted((root / "corpus").glob("*.parquet"))
    if force and not corpus_files:
        raise RuntimeError(
            "cannot rebuild the manifest without the corpus parquets; re-run "
            "without --force-manifest or delete data/clotho/mteb_clotho/manifest.json"
        )

    qrels: List[tuple[str, str]] = []
    for path in sorted((root / "qrels").glob("*.parquet")):
        table = pq.read_table(path)
        qrels.extend(
            (str(q), str(c))
            for q, c in zip(
                table.column("query-id").to_pylist(),
                table.column("corpus-id").to_pylist(),
            )
        )
    if not qrels:
        raise RuntimeError(f"no qrels rows under {root / 'qrels'}")

    captions: Dict[str, str] = {}
    for path in sorted((root / "queries").glob("*.parquet")):
        table = pq.read_table(path)
        for qid, text in zip(table.column("id").to_pylist(), table.column("text").to_pylist()):
            captions[str(qid)] = " ".join(str(text).split())

    clips: List[Dict[str, Any]] = []
    for path in tqdm(sorted((root / "corpus").glob("*.parquet")), desc="Writing Clotho audio"):
        table = pq.read_table(path)
        has_bytes = "audio" in table.column_names
        ids = table.column("id").to_pylist()
        blobs = table.column("audio").to_pylist() if has_bytes else [None] * len(ids)
        for cid, blob in zip(ids, blobs):
            cid = str(cid)
            dst = audio_dir / f"{cid}.wav"
            if not dst.exists():
                data, src_path = _audio_struct(blob)
                if not data:
                    if src_path and Path(src_path).exists():
                        data = Path(src_path).read_bytes()
                    else:
                        continue
                dst.write_bytes(data)
            ext = dst.suffix.lstrip(".") or "wav"
            clips.append({"id": cid, "file": str(dst.relative_to(root)), "ext": ext})

    by_clip: Dict[str, List[str]] = {c["id"]: [] for c in clips}
    queries: List[Dict[str, str]] = []
    for qid, cid in qrels:
        if qid not in captions:
            continue
        if cid not in by_clip:
            continue
        queries.append({"id": qid, "clip_id": cid, "text": captions[qid]})
        by_clip[cid].append(qid)
    for clip in clips:
        clip["captions"] = by_clip[clip["id"]]

    if not clips or not queries:
        raise RuntimeError("Clotho manifest build produced no rows")

    manifest = {
        "version": 1,
        "name": "clotho_v2.1_test",
        "source": f"{DATASET} (HuggingFace), MTEB audio retrieval packaging",
        "created_utc": utc_now(),
        "num_clips": len(clips),
        "num_captions": len(queries),
        "num_qrels": len(qrels),
        "clips": clips,
        "queries": queries,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        f"[Clotho] manifest: {len(clips)} clips, {len(queries)} captions, "
        f"{len(qrels)} qrels"
    )
    return manifest_path


# ── embedding ────────────────────────────────────────────────────────────────


def _fingerprint(backend: Any, manifest: Dict[str, Any]) -> Dict[str, Any]:
    gemma = backend._gemma_backend()
    pool = [c["id"] for c in manifest["clips"]] + [q["id"] for q in manifest["queries"]]
    return {
        "model": backend.MODEL_NAME,
        "dim": int(backend.EMBEDDING_DIM),
        "query_prefix": gemma.QUERY_PREFIX,
        "document_prefix": gemma.DOCUMENT_PREFIX,
        "audio_seconds_cap": float(gemma.MAX_AUDIO_SECONDS),
        "dataset": DATASET,
        "pool_hash": hashlib.sha1("\n".join(pool).encode()).hexdigest(),
    }


def embed_clotho(
    backend: Any,
    root: Path,
    manifest: Dict[str, Any],
    batch_size: int,
    force: bool = False,
) -> Dict[str, np.ndarray]:
    """Frozen Gemma embeddings for clips + query captions + document captions.

    Cached in ``data/clotho/cache/gemma_frozen.npz`` keyed by a provenance
    fingerprint, so the adapter run can reuse the (expensive) frozen pass.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / "gemma_frozen.npz"
    fp = _fingerprint(backend, manifest)
    dim = int(backend.EMBEDDING_DIM)
    if cache_path.exists() and not force:
        with np.load(cache_path, allow_pickle=False) as z:
            if json.loads(str(z["fingerprint"])) == fp:
                return {
                    "audio": z["audio"],
                    "text_query": z["text_query"],
                    "text_doc": z["text_doc"],
                }

    clips = manifest["clips"]
    queries = manifest["queries"]

    audio_vecs, kept = [], []
    for start in tqdm(range(0, len(clips), batch_size), desc="Embedding Clotho audio", unit="batch"):
        chunk = clips[start : start + batch_size]
        datas, exts, ids = [], [], []
        for c in chunk:
            try:
                datas.append((root / c["file"]).read_bytes())
            except OSError:
                continue
            exts.append(c["ext"])
            ids.append(c["id"])
        if not datas:
            continue
        vecs = backend.generate_audio_embeddings(datas, exts=exts, blob_ids=ids)
        for c, v in zip(chunk, vecs):
            if v is None:
                continue
            row = np.asarray(v, dtype=np.float32).reshape(-1)
            if row.size != dim:
                continue
            audio_vecs.append(row)
            kept.append(c["id"])
    if len(kept) != len(clips):
        print(f"[Clotho] WARNING: embedded {len(kept)}/{len(clips)} clips")
    audio = (np.stack(audio_vecs) if audio_vecs else np.zeros((0, dim), np.float32)).astype(np.float32)

    query_texts = [q["text"] for q in queries]
    text_query, kept_q = embed_texts(
        backend.generate_text_embeddings,
        query_texts,
        dim,
        batch_size=max(32, batch_size * 4),
        desc="Embedding Clotho query captions",
    )
    text_doc, kept_d = embed_texts(
        backend.generate_text_embeddings,
        query_texts,
        dim,
        batch_size=max(32, batch_size * 4),
        desc="Embedding Clotho document captions",
        is_document=True,
    )

    np.savez_compressed(
        cache_path,
        audio=audio,
        text_query=text_query,
        text_doc=text_doc,
        clip_ids=np.asarray(kept),
        query_ids=np.asarray([queries[i]["id"] for i in kept_q]),
        doc_ids=np.asarray([queries[i]["id"] for i in kept_d]),
        fingerprint=np.asarray(json.dumps(fp, sort_keys=True)),
    )
    print(f"[Clotho] cached frozen embeddings -> {cache_path}")
    return {"audio": audio, "text_query": text_query, "text_doc": text_doc}


# ── adapter ──────────────────────────────────────────────────────────────────


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _companion_manifest(path: Path) -> tuple[Optional[Path], bool]:
    """(manifest, is_byte_identical_sibling) for ``path``.

    The training workstream promotes a variant by copying it to the canonical
    name; the summary keeps the variant's name. Byte equality makes the
    promotion provable, so a renamed-but-identical checkpoint still gates in.
    """
    own = ADAPTER_MANIFEST_DIR / f"{path.stem}_dev.json"
    if own.exists():
        return own, False
    digest = _sha256(path)
    for sib in sorted(path.parent.glob("*.pt")):
        if sib == path:
            continue
        cand = ADAPTER_MANIFEST_DIR / f"{sib.stem}_dev.json"
        if cand.exists() and _sha256(sib) == digest:
            return cand, True
    return None, False


def adapter_gate(path: Path) -> tuple[bool, str]:
    """Admit an adapter only if it is stable and its companion manifest exists.

    Guards against reading a checkpoint that another workstream is still
    writing (``torch.save`` is not atomic).
    """
    if not path.exists():
        return False, f"adapter not found: {path}"
    age = time.time() - path.stat().st_mtime
    if age < ADAPTER_MIN_STABLE_SECONDS:
        return False, (
            f"adapter modified {age:.0f}s ago (< {ADAPTER_MIN_STABLE_SECONDS:.0f}s); "
            "training may still be writing it"
        )
    manifest, is_copy = _companion_manifest(path)
    if manifest is None:
        return False, (
            "companion manifest missing (no manifest for this checkpoint and no "
            "byte-identical sibling with one)"
        )
    try:
        meta = json.loads(manifest.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"companion manifest unreadable ({exc})"
    if not is_copy and manifest.stat().st_mtime < path.stat().st_mtime - 1.0:
        return False, "companion manifest predates the adapter weights"
    if not {"train_size", "dev_size", "best_epoch"} <= set(meta):
        return False, "companion manifest is not a completed training summary"
    origin = "byte-identical copy of" if is_copy else "manifest"
    return True, (
        f"final adapter ({origin} {manifest.stem.replace('_dev', '')}, "
        f"best_epoch={meta.get('best_epoch')})"
    )


def project(x: np.ndarray, state: Dict[str, Any], hidden: int) -> np.ndarray:
    """Residual MLP head from ``finetune_audiocaps_gemma.ProjHead``."""
    import torch
    from torch import nn

    dim = int(x.shape[-1])

    class _Head(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.ln = nn.LayerNorm(dim)
            self.fc1 = nn.Linear(dim, hidden)
            self.act = nn.GELU()
            self.fc2 = nn.Linear(hidden, dim)

        def forward(self, t):
            return t + self.fc2(self.act(self.fc1(self.ln(t))))

    head = _Head()
    head.load_state_dict({k: v.to("cpu") for k, v in state.items()})
    head.eval()
    with torch.no_grad():
        out = head(torch.from_numpy(np.asarray(x, dtype=np.float32)))
        out = torch.nn.functional.normalize(out, dim=-1)
    return out.numpy().astype(np.float32)


def load_adapter(path: Path) -> Dict[str, Any]:
    import torch

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if ckpt.get("stage") != "adapter":
        raise RuntimeError(f"unsupported adapter stage: {ckpt.get('stage')!r}")
    return ckpt


# ── metrics ──────────────────────────────────────────────────────────────────


def chance_recall(pool: int, positives: int, k: int) -> float:
    """Exact chance R@k for ``positives`` relevant items in a pool of ``pool``."""
    if pool <= 0 or positives <= 0:
        return 0.0
    k = min(k, pool)
    if pool - positives < k:
        return 1.0
    return 1.0 - (math.comb(pool - positives, k) / math.comb(pool, k))


def main() -> None:
    ap = argparse.ArgumentParser(description="Clotho v2.1 test retrieval (OOD)")
    ap.add_argument("--root", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--adapter", type=Path, default=None)
    ap.add_argument(
        "--extra-adapter",
        type=Path,
        action="append",
        default=None,
        help="Sibling checkpoint scored on the same pool (reported, never the headline).",
    )
    ap.add_argument("--out", type=Path, default=RESULTS / "clotho_gemma.json")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--force-manifest", action="store_true")
    ap.add_argument("--force-embed", action="store_true", help="Ignore the frozen embedding cache.")
    ap.add_argument("--limit", type=int, default=0, help="Smoke test: first N clips.")
    args = ap.parse_args()

    np.random.seed(args.seed)

    root = ensure_dataset(args.root, download=not args.no_download)
    manifest_path = build_manifest(root, force=args.force_manifest)
    manifest = json.loads(manifest_path.read_text())
    clips = manifest["clips"]
    queries = manifest["queries"]
    if args.limit:
        clips = clips[: args.limit]
        keep = {c["id"] for c in clips}
        queries = [q for q in queries if q["clip_id"] in keep]
        manifest = dict(manifest, clips=clips, queries=queries)

    import embedding_backend as eb

    gemma = eb._gemma_backend()
    model_name = eb.MODEL_NAME
    dim = int(eb.EMBEDDING_DIM or 768)

    adapter_meta: Dict[str, Any] = {}
    requested = args.adapter
    selected_adapter: Optional[Path] = None
    gate_reason = "no adapter requested"
    if requested is not None:
        ok, gate_reason = adapter_gate(requested)
        if ok:
            selected_adapter = requested
        else:
            print(f"[Clotho] adapter rejected: {gate_reason}")

    print(f"[Clotho] dataset : {DATASET} (test split)")
    print(f"[Clotho] clips   : {len(clips)}  captions: {len(queries)}")
    print(f"[Clotho] model   : {model_name} dim={dim}")
    print(f"[Clotho] adapter : {selected_adapter or 'frozen baseline'}")

    eb.initialize_embedding_model()
    emb = embed_clotho(eb, root, manifest, args.batch_size, force=args.force_embed)

    # Re-derive index maps from the cached ids so a partial embed cannot
    # silently misalign qrels.
    with np.load(CACHE_DIR / "gemma_frozen.npz", allow_pickle=False) as z:
        clip_ids = [str(x) for x in z["clip_ids"]]
        query_ids = [str(x) for x in z["query_ids"]]
        doc_ids = [str(x) for x in z["doc_ids"]]
    clip_pos = {cid: i for i, cid in enumerate(clip_ids)}
    query_pos = {qid: i for i, qid in enumerate(query_ids)}
    doc_pos = {qid: i for i, qid in enumerate(doc_ids)}

    audio = emb["audio"]
    text_query = emb["text_query"]
    text_doc = emb["text_doc"]

    # ── qrels, built once; every variant is scored on the identical pool ────
    t2a_q, t2a_gt = [], []
    for q in queries:
        qi = query_pos.get(q["id"])
        ci = clip_pos.get(q["clip_id"])
        if qi is None or ci is None:
            continue
        t2a_q.append(qi)
        t2a_gt.append(np.asarray([ci], dtype=np.int64))

    caps_by_clip: Dict[str, List[int]] = {}
    for q in queries:
        di = doc_pos.get(q["id"])
        if di is None:
            continue
        caps_by_clip.setdefault(q["clip_id"], []).append(di)
    a2t_rows, a2t_gt = [], []
    for cid in clip_ids:
        if cid not in caps_by_clip:
            continue
        a2t_rows.append(clip_pos[cid])
        a2t_gt.append(np.asarray(sorted(caps_by_clip[cid]), dtype=np.int64))

    def score(aud: np.ndarray, tq: np.ndarray, td: np.ndarray) -> Dict[str, Any]:
        sims_t2a = tq[t2a_q] @ aud.T
        t2a_ranks = ranks_at_k(sims_t2a, t2a_gt)
        sims_a2t = aud[a2t_rows] @ td.T
        a2t_ranks = ranks_at_k(sims_a2t, a2t_gt)
        return {
            "text_to_audio": recall_at_k(sims_t2a, t2a_gt),
            "audio_to_text": recall_at_k(sims_a2t, a2t_gt),
            "text_to_audio_ranks": t2a_ranks,
            "audio_to_text_ranks": a2t_ranks,
        }

    def apply_adapter(path: Path) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Load a gated adapter and return (projected features, metadata)."""
        path = path.resolve()
        ckpt = load_adapter(path)
        hidden = int((ckpt.get("arch") or {}).get("hidden") or dim)
        feats = {
            "audio": project(audio, ckpt["audio_head"], hidden),
            "text_query": project(text_query, ckpt["text_head"], hidden),
            "text_doc": project(text_doc, ckpt["text_head"], hidden),
        }
        meta = {
            "checkpoint": str(path.relative_to(REPO_ROOT)).replace("\\", "/"),
            "sha256": _sha256(path),
            "model": ckpt.get("model"),
            "arch": ckpt.get("arch"),
            "best_epoch": ckpt.get("best_epoch"),
            "train_size": ckpt.get("train_size"),
            "dev_size": ckpt.get("dev_size"),
            "seed": ckpt.get("seed"),
            "commit": ckpt.get("commit"),
            "utc_date": ckpt.get("utc_date"),
            "query_prefix": ckpt.get("query_prefix"),
            "document_prefix": ckpt.get("document_prefix"),
            "hyperparameters": ckpt.get("hyperparams"),
            "dev_pool_metrics": ckpt.get("dev_pool"),
            "dev_full_metrics": ckpt.get("dev_full"),
        }
        return feats, meta

    scores = score(audio, text_query, text_doc)
    t2a = scores["text_to_audio"]
    a2t = scores["audio_to_text"]
    t2a_ranks = scores["text_to_audio_ranks"]
    a2t_ranks = scores["audio_to_text_ranks"]

    if selected_adapter is not None:
        feats, adapter_meta = apply_adapter(selected_adapter)
        adapter_meta["gate"] = gate_reason
        audio, text_query, text_doc = (
            feats["audio"],
            feats["text_query"],
            feats["text_doc"],
        )
        model_name = adapter_meta.get("model") or model_name + "+audiocaps-adapter"
        scores = score(audio, text_query, text_doc)
        t2a = scores["text_to_audio"]
        a2t = scores["audio_to_text"]
        t2a_ranks = scores["text_to_audio_ranks"]
        a2t_ranks = scores["audio_to_text_ranks"]

    # Extra checkpoints (e.g. sibling adapter variants) scored on the same pool
    # so the comparison is like-for-like. Never part of the headline metrics.
    variant_reports: List[Dict[str, Any]] = []
    for extra in args.extra_adapter or []:
        ok, why = adapter_gate(extra)
        if not ok:
            variant_reports.append(
                {"checkpoint": str(extra.resolve().relative_to(REPO_ROOT)), "gate": why, "scored": False}
            )
            continue
        feats, meta = apply_adapter(extra)
        s = score(feats["audio"], feats["text_query"], feats["text_doc"])
        meta.update(
            {
                "gate": why,
                "scored": True,
                "text_to_audio": s["text_to_audio"],
                "audio_to_text": s["audio_to_text"],
            }
        )
        variant_reports.append(meta)

    n_clips = int(audio.shape[0])
    n_docs = int(text_doc.shape[0])
    t2a_chance = {f"R@{k}": chance_recall(n_clips, 1, k) for k in (1, 5, 10)}
    # a2t: 5 positive captions per clip out of n_docs.
    caps_per_clip = len(a2t_gt[0]) if a2t_gt else 5
    a2t_chance = {f"R@{k}": chance_recall(n_docs, caps_per_clip, k) for k in (1, 5, 10)}

    payload: Dict[str, Any] = {
        "model": model_name,
        "dim": int(dim),
        "backend": "omuragemma",
        "protocol": "clotho_v2.1_test_full_pool_t2a_and_a2t",
        "dataset": "Clotho v2.1 test (mteb/clotho_t2a_v2)",
        "pool_size": n_clips,
        "num_queries": int(len(t2a_q)),
        "caption_count": int(n_docs),
        "text_pool_size": n_docs,
        "seed": int(args.seed),
        "commit": git_commit(),
        "utc_date": utc_now(),
        "text_to_audio": t2a,
        "audio_to_text": a2t,
        "random_baseline": {
            "text_to_audio": t2a_chance,
            "audio_to_text": a2t_chance,
            "formula": "1 - C(N-m, k) / C(N, k) with m=1 (t2a) / m=5 (a2t)",
        },
        "num_audio_queries": int(len(a2t_rows)),
        "caption_count_per_clip": int(caps_per_clip),
        "per_query_ranks": {
            "text_to_audio": [],
            "audio_to_text": [],
        },
        "inference_recipe": (
            "audio = L2(f_audio(Gemma_a.encode(16k mono))); "
            "text query = L2(f_text(Gemma.encode(QUERY_PREFIX+caption))); "
            "text document = L2(f_text(Gemma.encode(DOCUMENT_PREFIX+caption))); "
            "f_* = identity for the frozen baseline"
        ),
        "provenance": {
            "dataset_id": DATASET,
            "dataset_source": manifest["source"],
            "num_qrels": int(manifest["num_qrels"]),
            "audio_seconds_cap": float(gemma.MAX_AUDIO_SECONDS),
            "query_prefix": gemma.QUERY_PREFIX,
            "document_prefix": gemma.DOCUMENT_PREFIX,
            "benchmark": "text<->audio retrieval",
            "held_out": True,
            "trained_on_this_dataset": False,
        },
        "notes": (
            "Clotho v2.1 test split is never used for training or model selection. "
            "Full 1045-clip pool for text->audio and all 5225 captions for "
            "audio->text; no sampled negatives. Chance R@1 for audio->text is "
            "5/5225, so compare against random_baseline, not against zero."
        ),
    }
    if adapter_meta:
        payload["adapter"] = adapter_meta
    if variant_reports:
        payload["adapter_variants"] = variant_reports

    # Per-query ranks (compact: one entry per query, ids inline).
    payload["per_query_ranks"] = {
        "text_to_audio": [
            {"query_id": queries[i]["id"], "clip_id": clips[int(g[0])]["id"], "rank": int(r)}
            for i, g, r in zip(t2a_q, t2a_gt, t2a_ranks)
        ],
        "audio_to_text": [
            {"clip_id": clip_ids[row], "rank": int(r)}
            for row, r in zip(a2t_rows, a2t_ranks)
        ],
    }

    write_result(args.out, payload)
    summary = {
        "model": model_name,
        "pool_size": n_clips,
        "caption_count": n_docs,
        "num_queries": int(len(t2a_q)),
        "text_to_audio": t2a,
        "audio_to_text": a2t,
        "random_baseline": payload["random_baseline"],
    }
    print(json.dumps(summary, indent=2))
    print(f"[Clotho] wrote {args.out}")
    if requested is not None and selected_adapter is None:
        print(f"[Clotho] adapter skipped: {gate_reason}")


if __name__ == "__main__":
    main()
