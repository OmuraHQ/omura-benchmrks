#!/usr/bin/env python
"""Fine-tune the audio side of EmbeddingGemma 2 on AudioCaps.

Three stages, each cheaper to abandon than the next:

1. ``adapter``  - frozen Gemma embeddings + residual projection heads trained
   with symmetric InfoNCE (this file, ``--stage adapter``).
2. ``lora``     - LoRA on the audio encoder + unfrozen ``embed_audio``.
3. ``full``     - unfreeze the last N audio-encoder layers.

Everything is cached under ``data/audiocaps_gemma_cache`` so the live API's
paths (``data/vector_index*``, ``.env``) are never touched.  The test split is
only read by :mod:`benchmarks.benchmark_audiocaps_gemma_ft`; this script uses
train/dev exclusively for model selection.

Usage::

    CUDA_VISIBLE_DEVICES=4 .venv/bin/python benchmarks/finetune_audiocaps_gemma.py \
        --stage features
    CUDA_VISIBLE_DEVICES=4 .venv/bin/python benchmarks/finetune_audiocaps_gemma.py \
        --stage adapter
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))

from bench_common import REPO_ROOT, git_commit, utc_now

MODEL_NAME = "google/embeddinggemma-2"
DIM = 768
AUDIO_SR = 16000
MAX_AUDIO_SECONDS = 300.0
QUERY_PREFIX = "task: search result | query: "
DOCUMENT_PREFIX = "title: none | text: "
SEED = 42

TEST_ROOT = REPO_ROOT / "data" / "audiocaps"
TRAIN_ROOT = REPO_ROOT / "data" / "audiocaps_train"
CACHE = REPO_ROOT / "data" / "audiocaps_gemma_cache"
ADAPTER_PATH = REPO_ROOT / "data" / "audiocaps_gemma_adapter.pt"
LORA_PATH = REPO_ROOT / "data" / "audiocaps_gemma_lora.pt"
FULL_PATH = REPO_ROOT / "data" / "audiocaps_gemma_full.pt"


# ── audio io ─────────────────────────────────────────────────────────────────


def decode_audio_bytes(blob: bytes) -> np.ndarray:
    """Decode any container to mono float32 16 kHz, capped at the model budget."""
    import soundfile as sf

    max_samples = int(MAX_AUDIO_SECONDS * AUDIO_SR)
    audio, sr = sf.read(io.BytesIO(blob), dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    if sr != AUDIO_SR:
        import soxr

        mono = soxr.resample(mono, sr, AUDIO_SR)
    return np.ascontiguousarray(mono[:max_samples], dtype=np.float32)


def load_test_manifest() -> Dict[str, Any]:
    manifest = json.loads((TEST_ROOT / "manifest.json").read_text())
    return manifest


# ── train meta + waveform memmap ─────────────────────────────────────────────


def build_train_waveforms(force: bool = False, shard_limit: int = 0) -> Dict[str, Any]:
    """One pass over the AudioCaps train parquets -> int16 16 kHz memmap."""
    import pyarrow.parquet as pq

    meta_path = CACHE / "train_meta.json"
    npy_path = CACHE / "train_audio_i16.bin"
    if not force and meta_path.exists() and npy_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("complete") and meta.get("n_clips") == len(meta["ids"]):
            print(f"[features] reuse train waveforms: {meta['n_clips']} clips")
            return meta

    CACHE.mkdir(parents=True, exist_ok=True)
    shards = sorted((TRAIN_ROOT / "parquet").glob("train-*.parquet"))
    if len(shards) != 412:
        raise RuntimeError(f"expected 412 train shards, found {len(shards)}")
    if shard_limit:
        shards = shards[:shard_limit]

    ids: List[int] = []
    keys: List[List[Any]] = []
    captions: List[str] = []
    lengths: List[int] = []
    target_len = int(10.0 * AUDIO_SR)

    # First pass: decode and either write directly at a growable offset in a
    # flat binary file, then memmap it.  Fixed 10 s slots keep indexing trivial;
    # true lengths are stored so training slices silence back off.
    raw_path = CACHE / "train_audio_i16.raw"
    offset = 0
    t0 = time.time()
    with open(raw_path, "wb") as fh:
        for si, shard in enumerate(shards):
            table = pq.read_table(shard, columns=["audiocap_id", "youtube_id", "start_time", "caption", "audio"])
            rows = table.to_pylist()
            for row in rows:
                blob = (row.get("audio") or {}).get("bytes")
                if not blob:
                    continue
                try:
                    audio = decode_audio_bytes(blob)
                except Exception as exc:  # noqa: BLE001
                    print(f"[features] decode failed {row.get('audiocap_id')}: {exc}")
                    continue
                n = len(audio)
                if n == 0:
                    continue
                ids.append(int(row["audiocap_id"]))
                keys.append([str(row["youtube_id"]), int(row.get("start_time") or 0)])
                captions.append(" ".join(str(row["caption"]).split()))
                lengths.append(n)
                pcm = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)
                if n >= target_len:
                    fh.write(pcm[:target_len].tobytes())
                else:
                    fh.write(pcm.tobytes())
                    fh.write(np.zeros(target_len - n, dtype=np.int16).tobytes())
                offset += target_len
            if si % 20 == 0 or si == len(shards) - 1:
                print(
                    f"[features] shard {si + 1}/{len(shards)} clips={len(ids)} "
                    f"{time.time() - t0:.0f}s",
                    flush=True,
                )

    size = offset * 2
    npy_path.unlink(missing_ok=True)
    os.replace(raw_path, npy_path)
    if npy_path.stat().st_size != offset * 2:
        raise RuntimeError(f"memmap size {npy_path.stat().st_size} != {offset * 2}")

    meta = {
        "n_clips": len(ids),
        "ids": ids,
        "keys": keys,
        "captions": captions,
        "lengths": lengths,
        "samples_per_clip": target_len,
        "bytes": size,
        "complete": True,
        "created_utc": utc_now(),
    }
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    print(f"[features] wrote {npy_path} ({size / 1e9:.1f} GB), {len(ids)} clips")
    return meta


_MEM: np.ndarray | None = None


def load_train_waveform(meta: Dict[str, Any], i: int) -> np.ndarray:
    global _MEM
    n = int(meta["lengths"][i])
    slot = int(meta["samples_per_clip"])
    if _MEM is None:
        _MEM = np.memmap(CACHE / "train_audio_i16.bin", dtype=np.int16, mode="r")
    pcm = _MEM[i * slot : i * slot + n]
    return (pcm.astype(np.float32) / 32767.0).copy()


# ── embedding extraction ─────────────────────────────────────────────────────


def _load_model():
    import torch
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        MODEL_NAME,
        device=os.getenv("OMURA_OMURAGEMMA_DEVICE", "cuda"),
        model_kwargs={"torch_dtype": torch.bfloat16},
    )
    model.eval()
    return model


def encode_audio_batches(model, arrays: Sequence[np.ndarray], batch_size: int = 32) -> np.ndarray:
    import torch

    out: List[np.ndarray] = []
    t0 = time.time()
    for start in range(0, len(arrays), batch_size):
        chunk = [{"audio": a} for a in arrays[start : start + batch_size]]
        with torch.no_grad():
            vecs = model.encode(
                chunk,
                batch_size=len(chunk),
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        out.append(np.asarray(vecs, dtype=np.float32))
        if start % (batch_size * 20) == 0:
            done = start + len(chunk)
            rate = done / max(1e-9, time.time() - t0)
            print(
                f"[features] audio {done}/{len(arrays)} ({rate:.1f}/s, "
                f"eta {(len(arrays) - done) / max(rate, 1e-9) / 60:.1f} min)",
                flush=True,
            )
    return np.concatenate(out, axis=0)


def encode_texts(model, texts: Sequence[str], batch_size: int = 128) -> np.ndarray:
    import torch

    out: List[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        chunk = list(texts[start : start + batch_size])
        with torch.no_grad():
            vecs = model.encode(
                chunk,
                batch_size=len(chunk),
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        out.append(np.asarray(vecs, dtype=np.float32))
    return np.concatenate(out, axis=0)


def build_features(force: bool = False, shard_limit: int = 0) -> None:
    """Cache frozen Gemma embeddings for train audio/text and test audio/text."""
    CACHE.mkdir(parents=True, exist_ok=True)
    meta = build_train_waveforms(force=force, shard_limit=shard_limit)

    need_train = force or not all(
        (CACHE / name).exists()
        for name in (
            "train_audio.npy",
            "train_text_query.npy",
            "train_text_doc.npy",
        )
    )
    test_npz = CACHE / "test_emb.npz"
    need_test = force or not test_npz.exists()

    if not need_train and not need_test:
        print("[features] all caches present")
        return

    model = _load_model()

    if need_train:
        n = len(meta["ids"])
        t0 = time.time()
        audio_arrays = [load_train_waveform(meta, i) for i in range(n)]
        # Keep decoded waveforms in RAM (≈ 14 GB int16); free after encoding.
        audio_emb = encode_audio_batches(model, audio_arrays, batch_size=32)
        np.save(CACHE / "train_audio.npy", audio_emb)
        del audio_arrays
        print(f"[features] train audio embeddings {audio_emb.shape}", flush=True)

        caps = list(meta["captions"])
        text_q = encode_texts(model, [QUERY_PREFIX + c for c in caps])
        np.save(CACHE / "train_text_query.npy", text_q)
        text_d = encode_texts(model, [DOCUMENT_PREFIX + c for c in caps])
        np.save(CACHE / "train_text_doc.npy", text_d)
        print(
            f"[features] train text embeddings {text_q.shape} {text_d.shape} "
            f"({time.time() - t0:.0f}s)",
            flush=True,
        )

    if need_test:
        manifest = load_test_manifest()
        clips = manifest["clips"]
        test_arrays = []
        for c in clips:
            import soundfile as sf

            audio, sr = sf.read(TEST_ROOT / c["file"], dtype="float32", always_2d=True)
            mono = audio.mean(axis=1)
            if sr != AUDIO_SR:
                import soxr

                mono = soxr.resample(mono, sr, AUDIO_SR)
            test_arrays.append(np.ascontiguousarray(mono, dtype=np.float32))
        test_audio = encode_audio_batches(model, test_arrays, batch_size=32)
        caps: List[str] = []
        clip_index: List[int] = []
        for i, c in enumerate(clips):
            for cap in c["captions"]:
                caps.append(cap)
                clip_index.append(i)
        test_q = encode_texts(model, [QUERY_PREFIX + c for c in caps], batch_size=128)
        test_d = encode_texts(model, [DOCUMENT_PREFIX + c for c in caps], batch_size=128)
        np.savez(
            test_npz,
            audio=test_audio,
            text_query=test_q,
            text_doc=test_d,
            clip_index=np.asarray(clip_index, dtype=np.int64),
            ids=np.asarray([c["audiocap_id"] for c in clips], dtype=np.int64),
        )
        print(
            f"[features] test embeddings audio={test_audio.shape} "
            f"text={test_q.shape}",
            flush=True,
        )

    print("[features] done")


# ── stage 1: residual projection heads + symmetric InfoNCE ───────────────────


def _split_indices(n: int, seed: int = SEED, dev_frac: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_dev = max(1, int(round(n * dev_frac)))
    dev = np.sort(perm[:n_dev])
    train = np.sort(perm[n_dev:])
    return train, dev


class ProjHead:
    """LayerNorm projection head; architecture selectable for dev sweeps.

    ``residual_mlp``: x + fc2(GELU(fc1(LN(x)))) with zero-init fc2 (near identity).
    ``mlp``:          fc2(GELU(fc1(LN(x)))) (full re-projection).
    ``linear``:       fc1(LN(x)) (one affine map).
    """

    def __init__(
        self,
        dim: int = DIM,
        hidden: int | None = None,
        dropout: float = 0.0,
        arch: str = "residual_mlp",
    ):
        import torch
        from torch import nn

        hidden = hidden or dim

        class _Head(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.arch = arch
                self.ln = nn.LayerNorm(dim)
                self.fc1 = nn.Linear(dim, hidden)
                self.act = nn.GELU()
                self.fc2 = nn.Linear(hidden, dim)
                if arch == "residual_mlp":
                    nn.init.zeros_(self.fc2.weight)
                    nn.init.zeros_(self.fc2.bias)
                self.drop = nn.Dropout(dropout) if dropout else nn.Identity()

            def forward(self, x):
                z = self.ln(x)
                if self.arch == "linear":
                    return self.fc1(z)
                if self.arch == "mlp":
                    return self.fc2(self.drop(self.act(self.fc1(z))))
                return x + self.drop(self.fc2(self.act(self.fc1(z))))

        self.module = _Head()

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, sd):
        self.module.load_state_dict(sd)

    def to(self, device):
        self.module.to(device)
        return self

    def train(self):
        self.module.train()

    def eval(self):
        self.module.eval()

    def __call__(self, x):
        return self.module(x)


def _contrastive_multi_positive(audio, text, logit_scale):
    """Symmetric InfoNCE where each row has >=1 positive text column."""
    import torch
    import torch.nn.functional as F

    a = F.normalize(audio.float(), dim=-1)
    t = F.normalize(text.float(), dim=-1)
    logits = (a @ t.T) * logit_scale
    # Text rows are interleaved [q0, d0, q1, d1, ...].  Multi-positive
    # InfoNCE: logsumexp over all candidates minus logsumexp over positives.
    pos_q = logits[:, 0::2]
    pos_d = logits[:, 1::2]
    pos_terms = torch.logsumexp(torch.stack([pos_q, pos_d], dim=0), dim=0)
    loss_a = -(pos_terms - torch.logsumexp(logits, dim=1)).mean()

    logits_t = logits.T
    rows = torch.arange(logits_t.shape[0], device=logits.device)
    pos_t = logits_t[rows, rows // 2]
    loss_t = -(pos_t - torch.logsumexp(logits_t, dim=1)).mean()
    return 0.5 * (loss_a + loss_t)


def _eval_dev(
    audio_emb: np.ndarray,
    text_q: np.ndarray,
    text_d: np.ndarray,
    head_a,
    head_t,
) -> Dict[str, float]:
    import torch

    with torch.no_grad():
        a = torch.from_numpy(audio_emb).cuda()
        tq = torch.from_numpy(text_q).cuda()
        td = torch.from_numpy(text_d).cuda()
        a = torch.nn.functional.normalize(head_a(a), dim=-1).float().cpu().numpy()
        if head_t is None:
            tq = torch.nn.functional.normalize(tq, dim=-1).float().cpu().numpy()
            td = torch.nn.functional.normalize(td, dim=-1).float().cpu().numpy()
        else:
            tq = torch.nn.functional.normalize(head_t(tq), dim=-1).float().cpu().numpy()
            td = torch.nn.functional.normalize(head_t(td), dim=-1).float().cpu().numpy()

    sims = tq @ a.T
    ranks = _first_hit_ranks(sims)
    t2a = {f"R@{k}": float(np.mean(ranks <= k)) for k in (1, 5, 10)}

    sims2 = a @ td.T
    ranks2 = _first_hit_ranks(sims2)
    a2t = {f"R@{k}": float(np.mean(ranks2 <= k)) for k in (1, 5, 10)}
    return {"text_to_audio": t2a, "audio_to_text": a2t}


def _first_hit_ranks(sims: np.ndarray) -> np.ndarray:
    """Rank of the diagonal positive against every column."""
    diag = np.diag(sims)
    return (sims > diag[:, None]).sum(axis=1) + 1


def train_adapter(args: argparse.Namespace) -> None:
    import torch

    meta = json.loads((CACHE / "train_meta.json").read_text())
    n = len(meta["ids"])
    train_idx, dev_idx = _split_indices(n, seed=args.seed, dev_frac=args.dev_frac)
    print(f"[adapter] train={len(train_idx)} dev={len(dev_idx)}")

    audio_cache = Path(args.audio_features) if args.audio_features else (CACHE / "train_audio.npy")
    a_all = np.load(audio_cache)
    tq_all = np.load(CACHE / "train_text_query.npy")
    td_all = np.load(CACHE / "train_text_doc.npy")
    if not (len(a_all) == len(tq_all) == len(td_all) == n):
        raise RuntimeError("cache length mismatch")

    a_tr, tq_tr, td_tr = a_all[train_idx], tq_all[train_idx], td_all[train_idx]
    a_dev, tq_dev, td_dev = a_all[dev_idx], tq_all[dev_idx], td_all[dev_idx]

    # A dev pool matched to the test pool size (883 clips) makes dev R@10
    # directly comparable to the test number.
    rng = np.random.default_rng(args.seed + 1)
    pool_size = min(args.dev_pool, len(dev_idx))
    pool = np.sort(rng.choice(len(dev_idx), size=pool_size, replace=False))
    a_pool, tq_pool, td_pool = a_dev[pool], tq_dev[pool], td_dev[pool]

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    shared = args.shared_head
    head_a = ProjHead(dim=DIM, hidden=args.hidden, dropout=args.dropout, arch=args.arch).to("cuda")
    if shared:
        head_t = head_a
    elif args.text_head == "identity":
        head_t = None
    else:
        head_t = ProjHead(dim=DIM, hidden=args.hidden, dropout=args.dropout, arch=args.arch).to("cuda")

    params = list(head_a.module.parameters())
    if not shared and head_t is not None:
        params += list(head_t.module.parameters())
    logit_scale = torch.nn.Parameter(torch.tensor(math.log(1.0 / 0.07), device="cuda"))
    params.append(logit_scale)

    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = max(1, len(train_idx) // args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=total_steps, pct_start=0.1
    )

    A = torch.from_numpy(a_tr)
    Tq = torch.from_numpy(tq_tr)
    Td = torch.from_numpy(td_tr)
    n_train = len(train_idx)
    bs = args.batch_size

    best = None
    best_epoch = -1
    history: List[Dict[str, Any]] = []
    patience = 0

    for epoch in range(args.epochs):
        head_a.train()
        if head_t is not None:
            head_t.train()
        perm = torch.randperm(n_train)
        losses = []
        t0 = time.time()
        for step in range(steps_per_epoch):
            idx = perm[step * bs : (step + 1) * bs]
            if len(idx) < 2:
                continue
            a = A[idx].cuda(non_blocking=True)
            tq = Tq[idx].cuda(non_blocking=True)
            td = Td[idx].cuda(non_blocking=True)
            # Interleave query/document text rows so each audio has 2 positives.
            text = torch.stack([tq, td], dim=1).reshape(-1, DIM)
            scale = logit_scale.exp().clamp(1.0, 100.0)
            loss = _contrastive_multi_positive(
                head_a(a), text if head_t is None else head_t(text), scale
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            losses.append(float(loss.detach()))

        head_a.eval()
        if head_t is not None:
            head_t.eval()
        dev = _eval_dev(a_pool, tq_pool, td_pool, head_a, head_t)
        dev_full = _eval_dev(a_dev, tq_dev, td_dev, head_a, head_t)
        score = dev["text_to_audio"]["R@10"]
        row = {
            "epoch": epoch + 1,
            "loss": float(np.mean(losses)) if losses else None,
            "dev_pool": dev,
            "dev_full": dev_full,
            "temp": float((1.0 / logit_scale.exp().clamp(1.0, 100.0)).detach().cpu()),
            "seconds": round(time.time() - t0, 1),
        }
        history.append(row)
        print(
            f"[adapter] epoch {epoch + 1}/{args.epochs} loss={row['loss']:.4f} "
            f"dev883 t2a R@10={score:.4f} a2t R@10={dev['audio_to_text']['R@10']:.4f} "
            f"dev_full t2a R@10={dev_full['text_to_audio']['R@10']:.4f} "
            f"temp={row['temp']:.5f} ({row['seconds']}s)",
            flush=True,
        )

        if best is None or score > best["score"] + 1e-6:
            best = {
                "score": score,
                "epoch": epoch + 1,
                "audio_head": {k: v.detach().cpu().clone() for k, v in head_a.state_dict().items()},
                "text_head": (
                    {k: v.detach().cpu().clone() for k, v in head_t.state_dict().items()}
                    if head_t is not None else None
                ),
                "logit_scale": logit_scale.detach().cpu().clone(),
                "dev_pool": dev,
                "dev_full": dev_full,
            }
            best_epoch = epoch + 1
            patience = 0
        else:
            patience += 1
            if patience >= args.patience:
                print(f"[adapter] early stop at epoch {epoch + 1} (best {best_epoch})")
                break

    ckpt = {
        "stage": "adapter",
        "model": MODEL_NAME + "+audiocaps-adapter",
        "dim": DIM,
        "arch": {
            "kind": args.arch,
            "text_head": args.text_head,
            "hidden": args.hidden,
            "shared": shared,
            "dropout": args.dropout,
            "norm": "LayerNorm",
            "act": "GELU",
            "init": "zero_fc2_residual_identity",
        },
        "query_prefix": QUERY_PREFIX,
        "document_prefix": DOCUMENT_PREFIX,
        "audio_head": best["audio_head"],
        "text_head": best["text_head"],
        "logit_scale": float(best["logit_scale"]),
        "best_epoch": best_epoch,
        "dev_pool": best["dev_pool"],
        "dev_full": best["dev_full"],
        "history": history,
        "hyperparams": {
            "lr": args.lr,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "weight_decay": args.weight_decay,
            "dev_frac": args.dev_frac,
            "dev_pool": args.dev_pool,
            "loss": "symmetric_multi_positive_info_nce",
            "learnable_temperature": True,
        },
        "train_size": int(len(train_idx)),
        "dev_size": int(len(dev_idx)),
        "seed": args.seed,
        "commit": git_commit(),
        "utc_date": utc_now(),
    }
    out_path = args.out or ADAPTER_PATH
    torch.save(ckpt, out_path)
    print(f"[adapter] saved {out_path} (best epoch {best_epoch})")
    summary = {k: v for k, v in ckpt.items() if k not in ("audio_head", "text_head")}
    summary["logit_scale"] = float(best["logit_scale"])
    dev_path = CACHE / (out_path.stem + "_dev.json")
    dev_path.write_text(json.dumps(summary, indent=2))



# ── stages 2/3: LoRA and partial unfreeze on the audio encoder ───────────────


class LoRALinear:
    """Rank-r LoRA over a frozen ``Gemma4ClippableLinear`` / ``nn.Linear``."""

    def __new__(cls, base, rank: int = 16, alpha: int = 32):
        import torch
        from torch import nn

        in_features, out_features = _in_out_features(base)

        class _LoRA(nn.Module):
            def __init__(self):
                super().__init__()
                self.base = base
                for p in self.base.parameters():
                    p.requires_grad_(False)
                self.A = nn.Linear(in_features, rank, bias=False)
                self.B = nn.Linear(rank, out_features, bias=False)
                nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
                nn.init.zeros_(self.B.weight)
                self.scaling = float(alpha) / float(rank)

            def forward(self, x):
                # LoRA math in fp32 for stable bf16 finetuning.
                delta = self.B(self.A(x.float())).to(x.dtype)
                return self.base(x) + delta * self.scaling

        mod = _LoRA()
        mod.to(next(base.parameters()).device)
        return mod


def _in_out_features(module) -> Tuple[int, int]:
    from torch import nn

    if hasattr(module, "linear"):
        return module.linear.in_features, module.linear.out_features
    if isinstance(module, nn.Linear):
        return module.in_features, module.out_features
    raise TypeError(f"unsupported LoRA target {type(module)}")


LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "post", "ffw_layer_1", "ffw_layer_2")


def add_lora(model, rank: int = 16, alpha: int = 32) -> Dict[str, Any]:
    import torch

    added: Dict[str, Any] = {}
    tower = model[0].auto_model.audio_tower
    for name, module in list(tower.named_modules()):
        if name.endswith(tuple(f".{t}" for t in LORA_TARGETS)):
            parent_name, _, attr = name.rpartition(".")
            parent = tower.get_submodule(parent_name) if parent_name else tower
            lora = LoRALinear(module, rank=rank, alpha=alpha)
            setattr(parent, attr, lora)
            added[name] = lora
            for p in lora.parameters():
                p.requires_grad_(True)
    return added


def _set_trainable(model, mode: str, layers: int = 6) -> List[str]:
    """Freeze everything, then re-enable the requested audio-side parameters."""
    import torch

    for p in model.parameters():
        p.requires_grad_(False)
    am = model[0].auto_model
    trainable: List[str] = []
    if mode == "lora":
        for name, p in model.named_parameters():
            if ".A." in name or ".B." in name:
                p.requires_grad_(True)
        trainable.append("lora_A/B")
    if mode in ("lora", "full"):
        for p in am.embed_audio.parameters():
            p.requires_grad_(True)
            trainable.append("embed_audio")
        for p in am.audio_tower.output_proj.parameters():
            p.requires_grad_(True)
            trainable.append("audio_tower.output_proj")
    if mode == "full":
        n = int(layers)
        for layer in am.audio_tower.layers[-n:]:
            for p in layer.parameters():
                p.requires_grad_(True)
        norm = getattr(am.audio_tower, "norm", None)
        if norm is not None:
            for p in norm.parameters():
                p.requires_grad_(True)
        trainable.append(f"audio_tower.layers[-{n}:]")
    return trainable


def encode_audio_model(model, arrays: Sequence[np.ndarray], batch_size: int = 32, train: bool = False):
    """Full Gemma audio pipeline (audio tower -> LLM -> pool -> normalize)."""
    import torch

    tf = model[0]
    device = next(model.parameters()).device
    out: List[Any] = []
    for start in range(0, len(arrays), batch_size):
        chunk = [{"audio": a} for a in arrays[start : start + batch_size]]
        feats = tf.preprocess(chunk)
        feats = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in feats.items()}
        if train:
            o = tf.forward(feats)
            o = model[1](o)
            o = model[2](o)
            out.append(o["sentence_embedding"])
        else:
            with torch.no_grad():
                o = tf.forward(feats)
                o = model[1](o)
                o = model[2](o)
            out.append(o["sentence_embedding"])
    return torch.cat(out, dim=0)


def _dev_audio_metrics(
    model,
    meta: Dict[str, Any],
    dev_pool_idx: np.ndarray,
    tq_pool: np.ndarray,
    td_pool: np.ndarray,
    batch_size: int = 32,
    head_a=None,
    head_t=None,
) -> Dict[str, Any]:
    import torch

    arrays = [load_train_waveform(meta, int(i)) for i in dev_pool_idx]
    emb = encode_audio_model(model, arrays, batch_size=batch_size, train=False)
    with torch.no_grad():
        a_t = emb.float()
        tq_t = torch.from_numpy(tq_pool).cuda().float()
        td_t = torch.from_numpy(td_pool).cuda().float()
        if head_a is not None:
            a_t = head_a(a_t)
        if head_t is not None:
            tq_t = head_t(tq_t)
            td_t = head_t(td_t)
        a = torch.nn.functional.normalize(a_t, dim=-1).cpu().numpy()
        tq = torch.nn.functional.normalize(tq_t, dim=-1).cpu().numpy()
        td = torch.nn.functional.normalize(td_t, dim=-1).cpu().numpy()
    sims = tq @ a.T
    ranks = _first_hit_ranks(sims)
    t2a = {f"R@{k}": float(np.mean(ranks <= k)) for k in (1, 5, 10)}
    sims2 = a @ td.T
    ranks2 = _first_hit_ranks(sims2)
    a2t = {f"R@{k}": float(np.mean(ranks2 <= k)) for k in (1, 5, 10)}
    return {"text_to_audio": t2a, "audio_to_text": a2t}


def train_audio_encoder(args: argparse.Namespace) -> None:
    import torch

    meta = json.loads((CACHE / "train_meta.json").read_text())
    n = len(meta["ids"])
    train_idx, dev_idx = _split_indices(n, seed=args.seed, dev_frac=args.dev_frac)
    if args.limit:
        train_idx = train_idx[: args.limit]
    rng = np.random.default_rng(args.seed + 1)
    pool_size = min(args.dev_pool, len(dev_idx))
    pool_rel = np.sort(rng.choice(len(dev_idx), size=pool_size, replace=False))
    pool_idx = dev_idx[pool_rel]

    tq_all = np.load(CACHE / "train_text_query.npy")
    td_all = np.load(CACHE / "train_text_doc.npy")
    tq_pool, td_pool = tq_all[pool_idx], td_all[pool_idx]

    head_a = head_t = None
    if args.adapter_ckpt is not None:
        akt = torch.load(args.adapter_ckpt, map_location="cpu", weights_only=False)
        hidden = int(akt.get("arch", {}).get("hidden", DIM))
        head_a = ProjHead(dim=DIM, hidden=hidden).to("cuda")
        head_a.load_state_dict({k: v.float() for k, v in akt["audio_head"].items()})
        head_t = ProjHead(dim=DIM, hidden=hidden).to("cuda")
        head_t.load_state_dict({k: v.float() for k, v in akt["text_head"].items()})
        head_a.eval()
        if head_t is not None:
            head_t.eval()
        for p in list(head_a.module.parameters()) + list(head_t.module.parameters()):
            p.requires_grad_(False)
        if args.train_audio_head:
            for p in head_a.module.parameters():
                p.requires_grad_(True)
        with torch.no_grad():
            tq_all = torch.nn.functional.normalize(
                head_t(torch.from_numpy(tq_all).cuda().float()), dim=-1
            ).cpu().numpy()
            td_all = torch.nn.functional.normalize(
                head_t(torch.from_numpy(td_all).cuda().float()), dim=-1
            ).cpu().numpy()
            tq_pool = torch.nn.functional.normalize(
                head_t(torch.from_numpy(tq_pool).cuda().float()), dim=-1
            ).cpu().numpy()
            td_pool = torch.nn.functional.normalize(
                head_t(torch.from_numpy(td_pool).cuda().float()), dim=-1
            ).cpu().numpy()
        print(f"[{args.stage}] adapter-target space from {args.adapter_ckpt}")

    model = _load_model()
    device = next(model.parameters()).device
    if args.drop_vision:
        am = model[0].auto_model
        am.vision_tower = None
        am.embed_vision = None
        import torch as _torch

        _torch.cuda.empty_cache()
        print(f"[{args.stage}] dropped unused vision tower")
    if args.grad_checkpoint:
        try:
            model[0].auto_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            print(f"[{args.stage}] gradient checkpointing enabled")
        except Exception as exc:  # noqa: BLE001
            print(f"[{args.stage}] gradient checkpointing unavailable: {exc}")

    if args.stage == "lora":
        lora_mods = add_lora(model, rank=args.rank, alpha=args.alpha)
        print(f"[{args.stage}] added LoRA to {len(lora_mods)} modules (rank {args.rank})")
        trainable = _set_trainable(model, "lora")
    else:
        trainable = _set_trainable(model, "full", layers=args.layers)
        print(f"[{args.stage}] unfroze {trainable}")

    params = [p for p in model.parameters() if p.requires_grad]
    if head_a is not None:
        params += [p for p in head_a.module.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in params)
    print(f"[{args.stage}] trainable params: {n_params/1e6:.2f}M over {len(params)} tensors")

    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    steps_per_epoch = max(1, len(train_idx) // args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    warmup = max(1, int(0.05 * total_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: min(1.0, (s + 1) / warmup)
        * max(0.0, 1.0 - max(0, s - warmup) / max(1, total_steps - warmup)),
    )

    A_PATH = LORA_PATH if args.stage == "lora" else FULL_PATH
    out_path = args.out or A_PATH
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    history: List[Dict[str, Any]] = []
    best = None
    for epoch in range(args.epochs):
        model.train()
        for p in model.parameters():
            if not p.requires_grad:
                p.requires_grad_(False)
        perm = np.random.permutation(len(train_idx))
        losses = []
        t0 = time.time()
        for step in range(steps_per_epoch):
            sel = train_idx[perm[step * args.batch_size : (step + 1) * args.batch_size]]
            arrays = [load_train_waveform(meta, int(i)) for i in sel]
            tq = torch.from_numpy(tq_all[sel]).to(device)
            td = torch.from_numpy(td_all[sel]).to(device)
            text = torch.stack([tq, td], dim=1).reshape(-1, DIM)
            emb = encode_audio_model(model, arrays, batch_size=len(arrays), train=True)
            if head_a is not None:
                emb = head_a(emb.float())
            scale = args.logit_scale
            loss = _contrastive_multi_positive(emb, text, scale)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            losses.append(float(loss.detach()))
            if step % 10 == 0:
                rate = (step + 1) / max(1e-9, time.time() - t0)
                print(
                    f"[{args.stage}] epoch {epoch+1} step {step+1}/{steps_per_epoch} "
                    f"loss={losses[-1]:.4f} {rate:.2f} steps/s",
                    flush=True,
                )

        model.eval()
        dev = _dev_audio_metrics(
            model, meta, pool_idx, tq_pool, td_pool, batch_size=32,
            head_a=head_a, head_t=head_t,
        )
        row = {
            "epoch": epoch + 1,
            "loss": float(np.mean(losses)),
            "dev_pool": dev,
            "seconds": round(time.time() - t0, 1),
        }
        history.append(row)
        score = dev["text_to_audio"]["R@10"]
        print(
            f"[{args.stage}] epoch {epoch+1} loss={row['loss']:.4f} "
            f"dev883 t2a R@10={score:.4f} a2t R@10={dev['audio_to_text']['R@10']:.4f} "
            f"({row['seconds']}s)",
            flush=True,
        )
        if best is None or score > best["score"]:
            best = {
                "score": score,
                "epoch": epoch + 1,
                "dev_pool": dev,
            }
            if args.stage == "lora":
                best["lora_state"] = {
                    name: {k: v.detach().cpu().clone() for k, v in mod.state_dict().items()}
                    for name, mod in lora_mods.items()
                }
                best["embed_audio"] = {
                    k: v.detach().cpu().clone() for k, v in model[0].auto_model.embed_audio.state_dict().items()
                }
            else:
                best["state"] = {
                    name: p.detach().cpu().clone()
                    for name, p in model.named_parameters()
                    if p.requires_grad
                }

    ckpt = {
        "stage": args.stage,
        "model": MODEL_NAME + ("+audiocaps-lora" if args.stage == "lora" else "+audiocaps-full"),
        "dim": DIM,
        "trainable": trainable,
        "best_epoch": best["epoch"],
        "dev_pool": best["dev_pool"],
        "history": history,
        "hyperparams": {
            "lr": args.lr,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "limit": args.limit,
            "rank": args.rank,
            "alpha": args.alpha,
            "layers": args.layers,
            "logit_scale": args.logit_scale,
            "loss": "symmetric_multi_positive_info_nce",
            "dev_frac": args.dev_frac,
            "dev_pool": args.dev_pool,
        },
        "train_size": int(len(train_idx)),
        "dev_size": int(len(dev_idx)),
        "seed": args.seed,
        "commit": git_commit(),
        "utc_date": utc_now(),
    }
    if args.adapter_ckpt is not None:
        ckpt["adapter_state"] = {
            "hidden": int(akt.get("arch", {}).get("hidden", DIM)),
            "audio_head": (
                {k: v.detach().cpu().clone() for k, v in head_a.state_dict().items()}
                if args.train_audio_head else akt["audio_head"]
            ),
            "text_head": akt["text_head"],
        }
        ckpt["adapter_source"] = str(args.adapter_ckpt)
    if args.stage == "lora":
        ckpt["rank"] = args.rank
        ckpt["alpha"] = args.alpha
        ckpt["lora_state"] = best["lora_state"]
        ckpt["embed_audio"] = best["embed_audio"]
    else:
        ckpt["state"] = best["state"]
    torch.save(ckpt, out_path)
    print(f"[{args.stage}] saved {out_path} (best epoch {best['epoch']})")


def load_finetuned_model(checkpoint: Path, device: str = "cuda"):
    """Rebuild the frozen base + checkpointed audio-side weights."""
    import torch

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    stage = ckpt["stage"]
    model = _load_model()
    if stage == "lora":
        mods = add_lora(model, rank=ckpt["rank"], alpha=ckpt["alpha"])
        for name, sd in ckpt["lora_state"].items():
            mods[name].load_state_dict(sd)
        model[0].auto_model.embed_audio.load_state_dict(ckpt["embed_audio"])
    elif stage == "full":
        missing = []
        state = ckpt["state"]
        named = dict(model.named_parameters())
        for name, tensor in state.items():
            if name not in named:
                missing.append(name)
                continue
            named[name].data.copy_(tensor.to(named[name].dtype))
        if missing:
            raise RuntimeError(f"missing full-tune params: {missing[:5]}")
    else:
        raise ValueError(f"not an audio-encoder checkpoint: {stage}")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, ckpt



def extract_audio_features(checkpoint: Path, out_name: str, batch_size: int = 32) -> None:
    """Re-encode train/test audio through a LoRA/full checkpoint (text untouched)."""
    import torch

    meta = json.loads((CACHE / "train_meta.json").read_text())
    model, ckpt = load_finetuned_model(checkpoint)
    model.eval()
    n = len(meta["ids"])

    out = np.zeros((n, DIM), dtype=np.float32)
    t0 = time.time()
    for start in range(0, n, batch_size):
        idx = range(start, min(start + batch_size, n))
        arrays = [load_train_waveform(meta, int(i)) for i in idx]
        emb = encode_audio_model(model, arrays, batch_size=len(arrays), train=False)
        out[start : start + len(arrays)] = emb.float().cpu().numpy()
        if start % (batch_size * 20) == 0:
            done = start + len(arrays)
            rate = done / max(1e-9, time.time() - t0)
            print(
                f"[extract] train audio {done}/{n} ({rate:.1f}/s, eta {(n-done)/max(rate,1e-9)/60:.1f} min)",
                flush=True,
            )
    np.save(CACHE / f"train_audio_{out_name}.npy", out)

    # Test audio re-encoded; frozen text vectors are reused.
    z = np.load(CACHE / "test_emb.npz")
    manifest = load_test_manifest()
    test_arrays = [decode_audio_bytes((TEST_ROOT / c["file"]).read_bytes()) for c in manifest["clips"]]
    test_audio = encode_audio_model(model, test_arrays, batch_size=batch_size, train=False)
    test_audio = test_audio.float().cpu().numpy()
    np.savez(
        CACHE / f"test_emb_{out_name}.npz",
        audio=test_audio,
        text_query=z["text_query"],
        text_doc=z["text_doc"],
        clip_index=z["clip_index"],
        ids=z["ids"],
    )
    print(f"[extract] saved train_audio_{out_name}.npy and test_emb_{out_name}.npz")


# ── CLI ──────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    global CACHE
    ap.add_argument("--stage", choices=["features", "adapter", "lora", "full", "extract"], default="adapter")
    ap.add_argument("--cache-dir", type=Path, default=None)
    ap.add_argument("--shard-limit", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--dev-frac", type=float, default=0.1)
    ap.add_argument("--dev-pool", type=int, default=883)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--hidden", type=int, default=DIM)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--shared-head", action="store_true")
    ap.add_argument("--arch", choices=["residual_mlp", "mlp", "linear"], default="residual_mlp")
    ap.add_argument("--text-head", choices=["train", "identity"], default="train")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--logit-scale", type=float, default=14.2857)
    ap.add_argument("--grad-checkpoint", action="store_true")
    ap.add_argument("--drop-vision", action="store_true")
    ap.add_argument("--audio-features", type=Path, default=None)
    ap.add_argument("--from-checkpoint", type=Path, default=None)
    ap.add_argument("--feature-name", default="lora")
    ap.add_argument("--adapter-ckpt", type=Path, default=None)
    ap.add_argument("--train-audio-head", action="store_true")
    args = ap.parse_args()
    if args.cache_dir is not None:
        CACHE = args.cache_dir

    if args.stage == "features":
        build_features(force=args.force, shard_limit=args.shard_limit)
    elif args.stage == "adapter":
        train_adapter(args)
    elif args.stage == "extract":
        if args.from_checkpoint is None:
            raise SystemExit("--from-checkpoint is required for --stage extract")
        extract_audio_features(args.from_checkpoint, args.feature_name, batch_size=32)
    else:
        train_audio_encoder(args)


if __name__ == "__main__":
    main()
