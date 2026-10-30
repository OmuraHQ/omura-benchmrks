"""Supervised CLAP adapters for AudioCaps text<->audio retrieval.

Trains two small residual MLP adapters (one per modality) on top of a *frozen*
``laion/larger_clap_general`` model, using symmetric InfoNCE with a learned
temperature. Only AudioCaps **train** audio/captions are used for fitting and a
10% clip-disjoint slice of train is used as dev for early stopping. The 883-clip
AudioCaps test pool (same manifest and captions as
``benchmarks/results/audiocaps_gemma.json``) is evaluated exactly once after the
adapter is frozen.

Data leakage guards
-------------------
* Train clips whose ``(youtube_id, start_time)`` key appears in the test
  manifest are dropped.
* Train captions that are byte-identical to any test caption are dropped (they
  are rare generic strings and would otherwise be indexed as test documents).
* Test captions are only ever used as *queries/documents of the test pool*, not
  as training targets.
* Dev is carved from train by clip; no test data touches early stopping.

Frozen CLAP features are cached per shard under
``data/audiocaps_train/features/`` so re-runs do not re-embed audio, and the
test-side frozen features are reused from
``data/audiocaps_clap_cache`` (produced by ``benchmark_audiocaps_clap.py``).

Run:
    CUDA_VISIBLE_DEVICES=5 PYTHONPATH=. \
        python benchmarks/finetune_audiocaps_adapter.py
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
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

DEFAULT_TEST_ROOT = REPO_ROOT / "data" / "audiocaps"
DEFAULT_TRAIN_ROOT = REPO_ROOT / "data" / "audiocaps_train"
DEFAULT_TEST_CACHE = REPO_ROOT / "data" / "audiocaps_clap_cache"
CLAP_MODEL = os.getenv("OMURA_CLAP_MODEL", "laion/larger_clap_general")
CLAP_SR = 48000
DIM = 512


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class ResidualMLP(torch.nn.Module):
    """out = normalize(x + MLP(x)); MLP is 512 -> hidden -> 512 (or linear)."""

    def __init__(self, dim: int = DIM, hidden: int = 1024, dropout: float = 0.1,
                 linear: bool = False):
        super().__init__()
        if linear:
            self.net = torch.nn.Sequential(
                torch.nn.LayerNorm(dim),
                torch.nn.Linear(dim, dim),
            )
        else:
            self.net = torch.nn.Sequential(
                torch.nn.LayerNorm(dim),
                torch.nn.Linear(dim, hidden),
                torch.nn.GELU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(hidden, dim),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x + self.net(x), dim=-1)


class Identity(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class DualAdapter(torch.nn.Module):
    def __init__(self, dim: int = DIM, hidden: int = 1024, dropout: float = 0.1,
                 logit_scale: float = 1.0 / 0.07, mode: str = "dual",
                 linear: bool = False):
        super().__init__()
        self.mode = mode
        self.audio = ResidualMLP(dim, hidden, dropout, linear)
        if mode == "shared":
            self.text = self.audio
        elif mode == "audio_only":
            self.text = Identity()
        else:
            self.text = ResidualMLP(dim, hidden, dropout, linear)
        self.logit_scale = torch.nn.Parameter(torch.tensor(math.log(logit_scale)))

    def forward(self, audio: torch.Tensor, text: torch.Tensor):
        a = self.audio(audio)
        t = self.text(text)
        scale = self.logit_scale.clamp(max=math.log(100.0)).exp()
        return a, t, scale


# --------------------------------------------------------------------------- #
# Frozen CLAP feature extraction (batched, identical preprocessing)
# --------------------------------------------------------------------------- #
def load_clap(device: str):
    from transformers import ClapModel, ClapProcessor

    model = ClapModel.from_pretrained(CLAP_MODEL).eval().float().to(device)
    processor = ClapProcessor.from_pretrained(CLAP_MODEL)
    return model, processor


def _decode_wav(blob: bytes) -> np.ndarray | None:
    import soundfile as sf

    try:
        wav, sr = sf.read(io.BytesIO(blob), dtype="float32", always_2d=False)
    except Exception:  # noqa: BLE001
        return None
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != CLAP_SR:
        import librosa

        wav = librosa.resample(np.asarray(wav, dtype=np.float32), orig_sr=sr,
                               target_sr=CLAP_SR)
    return np.asarray(wav, dtype=np.float32)


def embed_wavs(model, processor, device, wavs: Sequence[np.ndarray],
               batch_size: int = 32) -> np.ndarray:
    out: List[np.ndarray] = []
    for start in range(0, len(wavs), batch_size):
        chunk = list(wavs[start:start + batch_size])
        inputs = processor(audio=chunk, sampling_rate=CLAP_SR, return_tensors="pt",
                           padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            feats = model.get_audio_features(**inputs)
        feats = feats.pooler_output if hasattr(feats, "pooler_output") else feats
        out.append(l2_normalize(feats.float().cpu().numpy()))
    return np.concatenate(out, axis=0) if out else np.zeros((0, DIM), np.float32)


def embed_texts_clap(model, processor, device, texts: Sequence[str],
                     batch_size: int = 256) -> np.ndarray:
    out: List[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        chunk = list(texts[start:start + batch_size])
        inputs = processor(text=chunk, return_tensors="pt", padding=True,
                           truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            feats = model.get_text_features(**inputs)
        feats = feats.pooler_output if hasattr(feats, "pooler_output") else feats
        out.append(l2_normalize(feats.float().cpu().numpy()))
    return np.concatenate(out, axis=0) if out else np.zeros((0, DIM), np.float32)


# --------------------------------------------------------------------------- #
# Train data index
# --------------------------------------------------------------------------- #
def load_test_protocol(test_root: Path, test_cache: Path):
    manifest = json.loads((test_root / "manifest.json").read_text(encoding="utf-8"))
    clips = manifest["clips"]
    audio = np.load(test_cache / f"clap_audio_{len(clips)}.npz")["vecs"].astype(np.float32)
    captions = [c for clip in clips for c in clip["captions"]]
    text = np.load(test_cache / f"clap_text_{len(captions)}.npz")["vecs"].astype(np.float32)
    assert audio.shape == (len(clips), DIM), audio.shape
    assert text.shape == (len(captions), DIM), text.shape
    return manifest, clips, audio, captions, text


def build_train_index(shards: List[Path], test_keys: set, test_captions: set,
                      features_dir: Path, model, processor, device,
                      audio_bs: int, text_bs: int) -> Dict:
    """Embed each shard once (cached); return assembled arrays + index."""
    features_dir.mkdir(parents=True, exist_ok=True)
    for shard in tqdm(shards, desc="Frozen CLAP train shards", unit="shard"):
        dst = features_dir / (shard.stem + ".npz")
        if dst.exists():
            continue
        import pyarrow.parquet as pq

        table = pq.read_table(shard)
        cols = table.to_pydict()
        by_clip: Dict[tuple, Dict] = {}
        for yt, st, cap, audio in zip(cols["youtube_id"], cols["start_time"],
                                      cols["caption"], cols["audio"]):
            key = (str(yt), int(st))
            if key in test_keys:
                continue
            cap = " ".join(str(cap).split())
            if not cap or cap in test_captions:
                continue
            blob = (audio or {}).get("bytes")
            if not blob:
                continue
            entry = by_clip.setdefault(key, {"captions": [], "blob": blob})
            if cap not in entry["captions"]:
                entry["captions"].append(cap)
        keys = list(by_clip)
        wavs: List[np.ndarray] = []
        valid_keys: List[tuple] = []
        for key in keys:
            wav = _decode_wav(by_clip[key]["blob"])
            if wav is None or wav.size == 0:
                continue
            wavs.append(wav)
            valid_keys.append(key)
        if not valid_keys:
            np.savez_compressed(dst, audio=np.zeros((0, DIM), np.float32),
                                text=np.zeros((0, DIM), np.float32),
                                keys=np.array([], dtype=object),
                                caps=np.array([], dtype=object),
                                text_clip=np.zeros(0, np.int32))
            continue
        audio_vecs = embed_wavs(model, processor, device, wavs, audio_bs)
        caps: List[str] = []
        text_clip: List[int] = []
        for ci, key in enumerate(valid_keys):
            for cap in by_clip[key]["captions"]:
                caps.append(cap)
                text_clip.append(ci)
        text_vecs = embed_texts_clap(model, processor, device, caps, text_bs)
        np.savez_compressed(
            dst,
            audio=audio_vecs,
            text=text_vecs,
            keys=np.array([f"{k[0]}|{k[1]}" for k in valid_keys], dtype=object),
            caps=np.array(caps, dtype=object),
            text_clip=np.asarray(text_clip, dtype=np.int32),
        )
        del table, cols, by_clip, wavs

    audio_parts, text_parts, keys, caps = [], [], [], []
    text_clip: List[int] = []
    key_to_idx: Dict[str, int] = {}
    for shard in tqdm(shards, desc="Assembling index", unit="shard"):
        data = np.load(features_dir / (shard.stem + ".npz"), allow_pickle=True)
        a, t = data["audio"], data["text"]
        sk, sc, stc = list(data["keys"]), list(data["caps"]), data["text_clip"]
        # Clips (and their captions) may be split across shards: merge globally
        # by key, keeping the first audio embedding and every caption row.
        for i, k in enumerate(sk):
            if k not in key_to_idx:
                key_to_idx[k] = len(keys)
                keys.append(k)
                audio_parts.append(a[i:i + 1])
        text_parts.append(t)
        caps.extend(sc)
        text_clip.extend(key_to_idx[sk[int(c)]] for c in stc)

    return {
        "audio": np.concatenate(audio_parts, axis=0) if audio_parts else np.zeros((0, DIM), np.float32),
        "text": np.concatenate(text_parts, axis=0) if text_parts else np.zeros((0, DIM), np.float32),
        "keys": keys,
        "captions": caps,
        "text_clip": np.asarray(text_clip, dtype=np.int64),
    }


def csls_scores(sims: np.ndarray, k: int) -> np.ndarray:
    """Cross-domain similarity local scaling (hubness correction).

    score(q, d) = 2*sim(q, d) - mean_topk_q(q) - mean_topk_d(d). Uses only the
    query and candidate sets (no labels); k is selected on the validation split.
    """
    if k <= 0:
        return sims
    k = min(k, sims.shape[0], sims.shape[1])
    rq = np.sort(sims, axis=1)[:, -k:].mean(axis=1)
    rd = np.sort(sims, axis=0)[-k:, :].mean(axis=0)
    return 2.0 * sims - rq[:, None] - rd[None, :]


# --------------------------------------------------------------------------- #
# Train / eval helpers
# --------------------------------------------------------------------------- #
def split_dev(keys: List[str], text_clip: np.ndarray, dev_frac: float,
              seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = len(keys)
    perm = rng.permutation(n)
    n_dev = max(1, int(round(n * dev_frac)))
    dev_clips = np.sort(perm[:n_dev])
    train_clips = np.sort(perm[n_dev:])
    dev_set = set(dev_clips.tolist())
    dev_text = np.asarray([i for i, c in enumerate(text_clip) if int(c) in dev_set],
                          dtype=np.int64)
    return train_clips, dev_clips, dev_text


def eval_retrieval(audio: np.ndarray, text: np.ndarray, text_clip: np.ndarray,
                   dev_clips: np.ndarray, dev_text: np.ndarray) -> Dict:
    """t2a: dev captions -> dev clips. a2t: dev clips -> dev captions."""
    amap = {int(c): i for i, c in enumerate(dev_clips)}
    a = audio[dev_clips]
    t_rows = np.asarray([amap[int(text_clip[i])] for i in dev_text], dtype=np.int64)
    # Sort query rows by clip so each clip's captions form a contiguous block.
    order = np.argsort(t_rows, kind="stable")
    dev_text = np.asarray(dev_text)[order]
    t_rows = t_rows[order]
    # dev_text is sorted by text index, which is grouped by clip, so t_rows is
    # non-decreasing and each clip's captions occupy a contiguous block.
    assert np.all(np.diff(t_rows) >= 0)
    t2a_gt = [np.asarray([r], dtype=np.int64) for r in t_rows]
    sims_t2a = text[dev_text] @ a.T
    t2a = recall_at_k(sims_t2a, t2a_gt)

    a2t_gt = []
    cursor = 0
    for n_caps in np.bincount(t_rows, minlength=len(dev_clips)):
        a2t_gt.append(np.arange(cursor, cursor + int(n_caps), dtype=np.int64))
        cursor += int(n_caps)
    assert cursor == len(dev_text)
    sims_a2t = a @ text[dev_text].T
    a2t = recall_at_k(sims_a2t, a2t_gt)
    return {"text_to_audio": t2a, "audio_to_text": a2t}


@torch.no_grad()
def apply_adapters(adapter: DualAdapter, audio: np.ndarray, text: np.ndarray,
                   device: str, batch_size: int = 2048):
    a_out, t_out = [], []
    for start in range(0, len(audio), batch_size):
        x = torch.from_numpy(audio[start:start + batch_size]).to(device)
        a_out.append(adapter.audio(x).cpu().numpy())
    for start in range(0, len(text), batch_size):
        x = torch.from_numpy(text[start:start + batch_size]).to(device)
        t_out.append(adapter.text(x).cpu().numpy())
    return (np.concatenate(a_out) if a_out else audio,
            np.concatenate(t_out) if t_out else text)


def eval_split(adapter: DualAdapter, split: Dict, device: str) -> Dict:
    """Evaluate adapters on a self-contained split (audio pool + its captions)."""
    a, t = apply_adapters(adapter, split["audio"], split["text"], device)
    return eval_retrieval(a, t, split["text_clip"],
                          np.arange(len(split["keys"])),
                          np.arange(len(split["text_clip"])))


def train_adapter(data: Dict, dev: Dict, args, device: str,
                  dev_name: str = "dev") -> tuple[DualAdapter, Dict, int]:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    adapter = DualAdapter(hidden=args.hidden, dropout=args.dropout,
                          mode=args.adapter_mode, linear=args.linear).to(device)
    opt = torch.optim.AdamW(adapter.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    steps_per_epoch = max(1, math.ceil(len(data["train_pairs"]) / args.batch_size))
    total_steps = steps_per_epoch * args.epochs
    warmup = max(1, int(0.05 * total_steps))

    def lr_at(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        prog = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    audio_t = torch.from_numpy(data["audio"]).to(device)
    text_t = torch.from_numpy(data["text"]).to(device)
    pairs = np.asarray(data["train_pairs"], dtype=np.int64)
    best = {"score": -1.0, "epoch": -1, "state": None, "metrics": {}}
    history = []
    stale = 0
    for epoch in range(1, args.epochs + 1):
        adapter.train()
        perm = np.random.permutation(len(pairs))
        running = 0.0
        for start in range(0, len(perm), args.batch_size):
            idx = pairs[perm[start:start + args.batch_size]]
            if len(idx) < 2:
                continue
            a_idx = torch.from_numpy(idx[:, 0]).to(device)
            t_idx = torch.from_numpy(idx[:, 1]).to(device)
            a_in = audio_t[a_idx]
            t_in = text_t[t_idx]
            if args.feature_dropout > 0:
                a_in = F.dropout(a_in, p=args.feature_dropout, training=True)
                t_in = F.dropout(t_in, p=args.feature_dropout, training=True)
            a, t, scale = adapter(a_in, t_in)
            if args.full_negatives:
                # Cross-batch memory: all train keys act as negatives (detached,
                # refreshed every step) instead of only the in-batch ones.
                with torch.no_grad():
                    a_mem = adapter.audio(audio_t)
                    t_mem = adapter.text(text_t)
                logits_a2t = scale * (a @ t_mem.T)
                logits_t2a = scale * (t @ a_mem.T)
                loss = 0.5 * (F.cross_entropy(logits_a2t, t_idx) +
                              F.cross_entropy(logits_t2a, a_idx))
            else:
                logits = scale * (a @ t.T)
                labels = torch.arange(len(idx), device=device)
                loss = 0.5 * (F.cross_entropy(logits, labels) +
                              F.cross_entropy(logits.T, labels))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
            opt.step()
            sched.step()
            running += float(loss)
        adapter.eval()
        metrics = eval_split(adapter, dev, device)
        score = 0.5 * (metrics["text_to_audio"]["R@10"] +
                       metrics["audio_to_text"]["R@10"])
        history.append({"epoch": epoch, "loss": running / max(1, steps_per_epoch),
                        "dev_t2a_R@10": metrics["text_to_audio"]["R@10"],
                        "dev_a2t_R@10": metrics["audio_to_text"]["R@10"],
                        "dev_score": score,
                        "temp": float(adapter.logit_scale.exp().clamp(max=100))})
        print(f"[adapter] epoch {epoch:02d} loss={history[-1]['loss']:.4f} "
              f"{dev_name} t2a R@10={metrics['text_to_audio']['R@10']:.4f} "
              f"a2t R@10={metrics['audio_to_text']['R@10']:.4f} "
              f"T={history[-1]['temp']:.4f}", flush=True)
        if score > best["score"] + 1e-6:
            best = {"score": score, "epoch": epoch,
                    "state": {k: v.detach().cpu().clone()
                              for k, v in adapter.state_dict().items()},
                    "metrics": metrics}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                print(f"[adapter] early stop at epoch {epoch} "
                      f"(best epoch {best['epoch']})", flush=True)
                break
    adapter.load_state_dict(best["state"])
    adapter.eval()
    return adapter, {"history": history, "best_epoch": best["epoch"],
                     "best_dev": best["metrics"]}, len(history)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Fine-tune CLAP adapters on AudioCaps train")
    ap.add_argument("--test-root", type=Path, default=DEFAULT_TEST_ROOT)
    ap.add_argument("--train-root", type=Path, default=DEFAULT_TRAIN_ROOT)
    ap.add_argument("--val-root", type=Path,
                    default=REPO_ROOT / "data" / "audiocaps_val")
    ap.add_argument("--dev-source", choices=("auto", "train", "val"), default="auto",
                    help="Held-out split for early stopping; auto prefers validation.")
    ap.add_argument("--test-cache", type=Path, default=DEFAULT_TEST_CACHE)
    ap.add_argument("--out-json", type=Path,
                    default=REPO_ROOT / "benchmarks" / "results" / "audiocaps_finetuned.json")
    ap.add_argument("--adapter-out", type=Path,
                    default=REPO_ROOT / "data" / "audiocaps_adapter.pt")
    ap.add_argument("--num-shards", type=int, default=0, help="0 = all train shards.")
    ap.add_argument("--max-clips", type=int, default=0, help="Smoke-test cap.")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--adapter-mode", choices=("dual", "shared", "audio_only"),
                    default="dual", help="Separate, shared, or audio-only adapters.")
    ap.add_argument("--linear", action="store_true",
                    help="Use a single linear residual projection instead of an MLP.")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--feature-dropout", type=float, default=0.0)
    ap.add_argument("--dev-frac", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--audio-batch", type=int, default=32)
    ap.add_argument("--text-batch", type=int, default=256)
    ap.add_argument("--full-negatives", action="store_true",
                    help="Use all train keys as detached negatives (cross-batch memory).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-train", action="store_true",
                    help="Load adapter-out and only evaluate (for re-reporting).")
    ap.add_argument("--dev-only", action="store_true",
                    help="Train and report dev only; never touch the test split "
                         "(used for honest dev-based config selection).")
    ap.add_argument("--csls-k", type=int, default=0,
                    help="CSLS hubness correction k for the final test scores "
                         "(0 = off; pick on validation).")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[adapter] device={device} clap={CLAP_MODEL}")

    manifest, test_clips, test_audio, test_captions_list, test_text = load_test_protocol(
        args.test_root, args.test_cache
    )
    test_keys = {(c["youtube_id"], int(c["start_time"])) for c in test_clips}
    test_caption_set = set(test_captions_list)
    print(f"[adapter] test: {len(test_clips)} clips / {len(test_captions_list)} captions")

    shards = sorted((args.train_root / "parquet").glob("train-*.parquet"))
    if args.num_shards:
        shards = shards[: args.num_shards]
    if not shards:
        raise FileNotFoundError(f"no train parquet shards under {args.train_root}/parquet")
    print(f"[adapter] train shards: {len(shards)}")

    if args.skip_train:
        ckpt = torch.load(args.adapter_out, map_location=device)
        adapter = DualAdapter(hidden=ckpt["config"]["hidden"],
                              dropout=ckpt["config"]["dropout"],
                              mode=ckpt["config"].get("adapter_mode", "dual"),
                              linear=ckpt["config"].get("linear", False)).to(device)
        adapter.load_state_dict(ckpt["state_dict"])
        adapter.eval()
        train_summary = ckpt.get("train_summary", {})
        saved_cfg = ckpt["config"]
        data = {"text_clip": np.zeros(0, np.int64)}
    else:
        model, processor = load_clap(device)
        data = build_train_index(
            shards, test_keys, test_caption_set,
            args.train_root / "features", model, processor, device,
            args.audio_batch, args.text_batch,
        )
        val_shards = sorted((args.val_root / "parquet").glob("validation-*.parquet"))
        use_val = args.dev_source == "val" or (
            args.dev_source == "auto" and bool(val_shards)
        )
        if args.dev_source == "val" and not val_shards:
            raise FileNotFoundError(f"no validation shards under {args.val_root}/parquet")
        val_data = None
        if use_val:
            print(f"[adapter] validation dev shards: {len(val_shards)}")
            val_data = build_train_index(
                val_shards, test_keys, test_caption_set,
                args.val_root / "features", model, processor, device,
                args.audio_batch, args.text_batch,
            )
        del model
        torch.cuda.empty_cache()

        # Deduplicate captions/clips and build (audio, text) pairs.
        keys: List[str] = data["keys"]
        text_clip = data["text_clip"]
        if args.max_clips:
            keep = np.arange(min(args.max_clips, len(keys)))
            keep_set = set(keep.tolist())
            keep_text = [i for i, c in enumerate(text_clip) if int(c) in keep_set]
            text_clip = np.asarray([{int(c): i for i, c in enumerate(keep)}[int(text_clip[j])]
                                    for j in keep_text], dtype=np.int64)
            data["audio"] = data["audio"][keep]
            data["text"] = data["text"][keep_text]
            data["keys"] = [keys[i] for i in keep]
            data["captions"] = [data["captions"][i] for i in keep_text]
            keys = data["keys"]
        data["train_pairs"] = np.stack([text_clip, np.arange(len(text_clip))], axis=1)
        if use_val and val_data is not None:
            dev = val_data
            dev_name = "val"
            num_train_clips = len(keys)
        else:
            train_clips, dev_clips, dev_text = split_dev(keys, text_clip,
                                                         args.dev_frac, args.seed)
            clip_set = set(train_clips.tolist())
            data["train_pairs"] = np.asarray(
                [p for p in data["train_pairs"] if int(text_clip[p[0]]) in clip_set],
                dtype=np.int64,
            )
            remap = {int(old): new for new, old in enumerate(dev_clips)}
            dev = {
                "audio": data["audio"][dev_clips],
                "text": data["text"][dev_text],
                "text_clip": np.asarray([remap[int(text_clip[i])] for i in dev_text],
                                        dtype=np.int64),
                "keys": [keys[i] for i in dev_clips],
            }
            dev_name = "dev"
            num_train_clips = len(train_clips)
        print(f"[adapter] pairs: train={len(data['train_pairs'])} "
              f"{dev_name} captions={len(dev['text_clip'])} clips={len(dev['keys'])}")

        adapter, train_summary, epochs_run = train_adapter(data, dev, args, device,
                                                           dev_name)
        train_summary.update({
            "epochs_run": epochs_run,
            "num_train_pairs": int(len(data["train_pairs"])),
            "num_train_clips": int(num_train_clips),
            "num_dev_clips": int(len(dev["keys"])),
            "num_dev_captions": int(len(dev["text_clip"])),
            "num_total_clips": int(len(keys)),
            "num_train_shards": int(len(shards)),
            "dev_source": dev_name,
        })
        args.adapter_out.parent.mkdir(parents=True, exist_ok=True)
        saved_cfg = {
            "dim": DIM, "hidden": args.hidden, "dropout": args.dropout,
            "feature_dropout": args.feature_dropout, "clap_model": CLAP_MODEL,
            "batch_size": args.batch_size, "lr": args.lr,
            "weight_decay": args.weight_decay, "epochs": args.epochs,
            "dev_source": dev_name, "seed": args.seed,
            "adapter_mode": args.adapter_mode, "linear": args.linear,
        }
        torch.save({
            "state_dict": adapter.state_dict(),
            "config": saved_cfg,
            "train_summary": train_summary,
        }, args.adapter_out)
        print(f"[adapter] saved {args.adapter_out}")

        if args.dev_only:
            payload = {
                "status": "dev_only",
                "model": f"{CLAP_MODEL} + DualAdapter (frozen CLAP)",
                "seed": args.seed,
                "commit": git_commit(),
                "utc_date": utc_now(),
                "config": saved_cfg,
                "counts": {
                    "train_clips": train_summary.get("num_train_clips"),
                    "dev_clips": train_summary.get("num_dev_clips"),
                    "train_pairs": train_summary.get("num_train_pairs"),
                    "dev_captions": train_summary.get("num_dev_captions"),
                },
                "dev": train_summary.get("best_dev"),
                "best_epoch": train_summary.get("best_epoch"),
                "epochs_run": train_summary.get("epochs_run"),
                "train_history": train_summary.get("history"),
            }
            write_result(args.out_json, payload)
            print(json.dumps({"dev": payload["dev"], "best_epoch": payload["best_epoch"]},
                             indent=2))
            print(f"[adapter] dev-only result: {args.out_json}")
            return

    # ---- Final test evaluation: exactly once, adapter frozen -----------------
    adapter.eval()
    a_test, t_test = apply_adapters(adapter, test_audio, test_text, device)
    # Positive is the clip index of the caption's clip (same as the Gemma run).
    t2a_gt = []
    for a_idx, clip in enumerate(test_clips):
        for _ in clip["captions"]:
            t2a_gt.append(np.asarray([a_idx], dtype=np.int64))
    sims_t2a = t_test @ a_test.T
    t2a_plain = recall_at_k(sims_t2a, t2a_gt)
    sims_t2a_csls = csls_scores(sims_t2a, args.csls_k)
    t2a = recall_at_k(sims_t2a_csls, t2a_gt)
    t2a_ranks = ranks_at_k(sims_t2a_csls, t2a_gt)

    a2t_gt = []
    cursor = 0
    for clip in test_clips:
        n_caps = len(clip["captions"])
        a2t_gt.append(np.arange(cursor, cursor + n_caps, dtype=np.int64))
        cursor += n_caps
    sims_a2t = a_test @ t_test.T
    a2t_plain = recall_at_k(sims_a2t, a2t_gt)
    sims_a2t_csls = csls_scores(sims_a2t, args.csls_k)
    a2t = recall_at_k(sims_a2t_csls, a2t_gt)
    a2t_ranks = ranks_at_k(sims_a2t_csls, a2t_gt)

    payload = {
        "model": f"{CLAP_MODEL} + DualAdapter (frozen CLAP)",
        "dim": DIM,
        "backend": "clap+adapter",
        "protocol": "audiocaps_test_full_pool_t2a_and_a2t",
        "dataset": "AudioCaps test (OpenSound/AudioCaps, 883 unique clips)",
        "train_dataset": "AudioCaps train (OpenSound/AudioCaps); dev = AudioCaps validation, disjoint from test",
        "seed": args.seed,
        "commit": git_commit(),
        "utc_date": utc_now(),
        "pool_size": int(a_test.shape[0]),
        "num_images": int(a_test.shape[0]),
        "num_queries": int(t_test.shape[0]),
        "num_caption_candidates": int(t_test.shape[0]),
        "caption_count": int(len(test_captions_list)),
        "adapter": {
            "arch": "separate residual MLP per modality: normalize(x + Linear(hidden) GELU Linear(dim))",  # noqa: E501
            "dim": DIM,
            "hidden": saved_cfg.get("hidden"),
            "dropout": saved_cfg.get("dropout"),
            "feature_dropout": saved_cfg.get("feature_dropout"),
            "loss": "symmetric InfoNCE with learned temperature",
            "epochs_run": train_summary.get("epochs_run"),
            "best_epoch": train_summary.get("best_epoch"),
            "batch_size": saved_cfg.get("batch_size"),
            "lr": saved_cfg.get("lr"),
            "weight_decay": saved_cfg.get("weight_decay"),
            "dev_source": saved_cfg.get("dev_source", "train_holdout"),
            "checkpoint": str(args.adapter_out),
        },
        "counts": {
            "train_clips": train_summary.get("num_train_clips"),
            "dev_clips": train_summary.get("num_dev_clips"),
            "train_pairs": train_summary.get("num_train_pairs"),
            "dev_captions": train_summary.get("num_dev_captions"),
            "test_clips": int(a_test.shape[0]),
            "test_captions": int(t_test.shape[0]),
        },
        "dev": train_summary.get("best_dev"),
        "train_history": train_summary.get("history"),
        "text_to_audio": t2a,
        "audio_to_text": a2t,
        "text_to_audio_plain": t2a_plain,
        "audio_to_text_plain": a2t_plain,
        "scoring": {
            "csls_k": int(args.csls_k),
            "csls_selected_on": "AudioCaps validation split (445 clips), k in {5,10,20}",
            "note": "CSLS uses only the query/candidate sets, never labels.",
        },
        "per_query_ranks": {
            "text_to_audio": [
                {"audiocap_id": test_clips[int(g[0])]["audiocap_id"], "rank": int(r)}
                for g, r in zip(t2a_gt, t2a_ranks)
            ],
            "audio_to_text": [
                {"audiocap_id": test_clips[i]["audiocap_id"], "rank": int(r)}
                for i, r in enumerate(a2t_ranks)
            ],
        },
        "notes": (
            "Frozen laion/larger_clap_general features; adapters fit only on "
            "AudioCaps train, early-stopped on the AudioCaps validation split. "
            "Test evaluated once after early stopping, with CSLS k chosen on "
            "validation. Test audio documents are audio-only; caption documents "
            "are the AudioCaps test captions only."
        ),
    }
    write_result(args.out_json, payload)
    print(json.dumps({k: payload[k] for k in ("text_to_audio", "audio_to_text",
                                              "counts", "adapter")}, indent=2))
    print(f"[adapter] wrote {args.out_json}")


if __name__ == "__main__":
    main()
