#!/usr/bin/env python
"""Shared pieces for the Gemma-native cross-attention audio/text reranker.

The base ``google/embeddinggemma-2`` model is a single multimodal encoder:

* ``auto_model.audio_tower`` (Gemma4AudioModel) turns 16 kHz audio into a
  **bidirectional** sequence of 1536-d frames (25 tokens / second).
* ``auto_model.embed_audio`` is Gemma's own audio projector, mapping those
  frames into the language-model space (512-d for this checkpoint).
* ``auto_model.language_model`` turns text into 768-d token states (the pooled
  768-d sentence vector is the mean of these).

Nothing here is fine-tuned: the reranker consumes frozen Gemma token states and
trains only the cross-attention relevance head on top.
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))

from bench_common import REPO_ROOT, l2_normalize  # noqa: E402

MODEL_NAME = "google/embeddinggemma-2"
QUERY_PREFIX = "task: search result | query: "
DOCUMENT_PREFIX = "title: none | text: "
AUDIO_SR = 16000
AUDIO_TOKENS_PER_SEC = 25  # 2x conv stride on 100 mel frames/s
DIM = 768
AUDIO_IN = 512  # embed_audio output dim
TEXT_IN = 768  # language_model last_hidden_state dim

CROSS_CACHE = REPO_ROOT / "data" / "audiocaps_crossattn_cache"
ADAPTER_CACHE = REPO_ROOT / "data" / "audiocaps_gemma_cache"
ADAPTER_PATH = REPO_ROOT / "data" / "audiocaps_gemma_adapter.pt"
CLOTHO_ROOT = REPO_ROOT / "data" / "clotho"
CLOTHO_CACHE = CLOTHO_ROOT / "cache"
HEAD_PATH = REPO_ROOT / "data" / "audiocaps_crossattn.pt"


# ── model / token extraction ─────────────────────────────────────────────────


def load_gemma(device: str = "cuda"):
    import torch
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        MODEL_NAME,
        device=device,
        model_kwargs={"torch_dtype": torch.bfloat16},
    )
    model.eval()
    return model


def extract_audio_tokens(model, arrays: Sequence[np.ndarray], batch_size: int = 32):
    """Frozen Gemma audio tokens = embed_audio(audio_tower(audio)).

    Returns ``(tokens, lengths, hidden)`` with ``tokens`` float16 ``[N, T, 512]``
    already zero-padded to the longest clip in the whole call.
    """
    import torch

    trans = model[0]
    root = trans.auto_model
    feats: List[np.ndarray] = []
    lens: List[int] = []
    max_tok = 0
    with torch.no_grad():
        for start in range(0, len(arrays), batch_size):
            chunk = [{"audio": a} for a in arrays[start : start + batch_size]]
            batch = trans.preprocess(chunk)
            batch.pop("modality", None)
            batch = {k: (v.to("cuda") if torch.is_tensor(v) else v) for k, v in batch.items()}
            audio = root.audio_tower(
                batch["input_features"], batch.get("input_features_mask"), return_dict=True
            )
            proj = root.embed_audio(inputs_embeds=audio.last_hidden_state)
            mask = audio.attention_mask
            lengths = mask.sum(-1).tolist() if mask is not None else [proj.shape[1]] * proj.shape[0]
            for row, n in enumerate(lengths):
                n = int(n)
                feats.append(proj[row, :n].float().cpu().numpy().astype(np.float16))
                lens.append(n)
                max_tok = max(max_tok, n)
            if start % (batch_size * 100) == 0:
                print(
                    f"[cross-cache] audio {start + len(chunk)}/{len(arrays)} "
                    f"max_tok={max_tok}",
                    flush=True,
                )
    out = np.zeros((len(arrays), max_tok, proj.shape[-1]), dtype=np.float16)
    for i, f in enumerate(feats):
        out[i, : f.shape[0]] = f
    return out, np.asarray(lens, dtype=np.int32), int(proj.shape[-1])


def extract_text_tokens(model, texts: Sequence[str], batch_size: int = 256):
    """Frozen Gemma text token states (768-d) after the language model, padded."""
    import torch

    trans = model[0]
    root = trans.auto_model
    rows: List[np.ndarray] = []
    lens: List[int] = []
    max_len = 0
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            chunk = list(texts[start : start + batch_size])
            batch = trans.preprocess(chunk)
            batch.pop("modality", None)
            batch = {k: (v.to("cuda") if torch.is_tensor(v) else v) for k, v in batch.items()}
            out = root(**batch, return_dict=True)
            hidden = out.last_hidden_state.float().cpu().numpy()
            mask = batch["attention_mask"].cpu().numpy()
            for row, m in enumerate(mask):
                n = int(m.sum())
                rows.append(hidden[row, :n].astype(np.float16))
                lens.append(n)
                max_len = max(max_len, n)
            if start % (batch_size * 20) == 0:
                print(f"[cross-cache] text {start + len(chunk)}/{len(texts)} max_len={max_len}", flush=True)
    out = np.zeros((len(texts), max_len, hidden.shape[-1]), dtype=np.float16)
    for i, r in enumerate(rows):
        out[i, : r.shape[0]] = r
    return out, np.asarray(lens, dtype=np.int32), int(hidden.shape[-1])


# ── cache loading ────────────────────────────────────────────────────────────


def load_token_cache(name: str) -> Dict[str, Any]:
    """Load one cached token split, e.g. ``audiocaps_train`` or ``clotho_test``."""
    base = CROSS_CACHE / name
    meta = json.loads((base / "meta.json").read_text())
    audio = np.load(base / "audio_tok.npy", mmap_mode="r")
    text_q = np.load(base / "text_query_tok.npy", mmap_mode="r")
    text_d = np.load(base / "text_doc_tok.npy", mmap_mode="r")
    meta = dict(meta)
    meta.update({"audio": audio, "text_query": text_q, "text_doc": text_d})
    return meta


def pad_batch(arrays: np.ndarray, lengths: np.ndarray, idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Gather a padded batch + boolean mask from a padded token cache.

    Keeps the cache dtype (float16) so the host->device copy is half the size;
    callers cast to float32 on the GPU.
    """
    tok = np.asarray(arrays[idx])
    lens = np.asarray(lengths[idx], dtype=np.int64)
    mask = np.arange(tok.shape[1])[None, :] < lens[:, None]
    return tok, mask


class DeviceTokens:
    """A token cache resident on the GPU, with fast masked batch gathering."""

    def __init__(
        self,
        tokens: np.ndarray,
        lengths: Sequence[int],
        device: str = "cuda",
        pool_stride: int = 1,
    ):
        import torch

        self.device = device
        lens = torch.as_tensor(np.asarray(lengths, dtype=np.int64), device=device)
        if pool_stride > 1:
            arr = np.asarray(tokens)
            n, t, d = arr.shape
            t2 = (t // pool_stride) * pool_stride
            pooled = np.empty((n, t2 // pool_stride, d), dtype=np.float16)
            for start in range(0, n, 512):
                chunk = np.asarray(arr[start : start + 512, :t2], dtype=np.float32)
                chunk = chunk.reshape(chunk.shape[0], t2 // pool_stride, pool_stride, d).mean(2)
                pooled[start : start + chunk.shape[0]] = chunk.astype(np.float16)
            tok = torch.from_numpy(pooled).to(device)
            lens = torch.clamp((lens + pool_stride - 1) // pool_stride, max=t2 // pool_stride)
        else:
            tok = torch.from_numpy(np.ascontiguousarray(tokens)).to(device)
        self.tok = tok
        self.lens = lens
        self.T = int(self.tok.shape[1])
        self.pool_stride = pool_stride

    def batch(self, idx: np.ndarray):
        import torch

        it = torch.as_tensor(np.asarray(idx, dtype=np.int64), device=self.device)
        out = self.tok.index_select(0, it).float()
        mask = torch.arange(self.T, device=self.device)[None, :] < self.lens[it, None]
        return out, mask


# ── frozen stage-1 adapter ───────────────────────────────────────────────────


class ProjHead:
    """Exact copy of the audio-adapter projection head (read-only import)."""

    def __init__(self, dim: int = DIM, hidden: int = 2048):
        import torch
        from torch import nn

        class _Head(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.ln = nn.LayerNorm(dim)
                self.fc1 = nn.Linear(dim, hidden)
                self.act = nn.GELU()
                self.fc2 = nn.Linear(hidden, dim)

            def forward(self, x):
                return x + self.fc2(self.act(self.fc1(self.ln(x))))

        self.module = _Head()

    def load(self, sd):
        self.module.load_state_dict({k: v.to("cpu") for k, v in sd.items()})
        return self

    def eval(self):
        self.module.eval()
        return self


def load_adapter(path: Path = ADAPTER_PATH):
    import torch

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    hidden = int(ckpt.get("arch", {}).get("hidden", 2048))
    audio = ProjHead(hidden=hidden).load(ckpt["audio_head"])
    text = ProjHead(hidden=hidden).load(ckpt["text_head"])
    return ckpt, audio, text


def project(embed: np.ndarray, head: ProjHead, device: str = "cuda", batch: int = 8192) -> np.ndarray:
    import torch

    head.module.to(device)
    out = []
    with torch.no_grad():
        for start in range(0, len(embed), batch):
            x = torch.from_numpy(np.asarray(embed[start : start + batch], dtype=np.float32)).to(device)
            out.append(torch.nn.functional.normalize(head.module(x), dim=-1).float().cpu().numpy())
    return np.concatenate(out, axis=0)


# ── cross-attention head ─────────────────────────────────────────────────────


@dataclass
class RerankerConfig:
    d: int = 768
    audio_in: int = AUDIO_IN
    text_in: int = TEXT_IN
    layers: int = 3
    heads: int = 8
    ffn: int = 2048
    dropout: float = 0.1
    pooling: str = "attn+mean"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CrossAttnScorer:
    """Text tokens = queries, audio tokens = keys/values, bidirectional-free.

    Each pre-LN block does masked cross-attention (text -> audio) plus a
    residual FFN.  Audio tokens are never updated, so their per-layer K/V can
    be computed once per clip and reused across every text scored against it.
    A masked attention-pool + mean-pool of the text tokens feeds a scalar head.
    """

    def __init__(self, cfg: RerankerConfig):
        import torch
        from torch import nn

        self.cfg = cfg
        d, h = cfg.d, cfg.heads
        if d % h:
            raise ValueError("d must be divisible by heads")

        class _Block(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.ln_q = nn.LayerNorm(d)
                self.ln_kv = nn.LayerNorm(d)
                self.q = nn.Linear(d, d)
                self.k = nn.Linear(d, d)
                self.v = nn.Linear(d, d)
                self.o = nn.Linear(d, d)
                self.ln_ff = nn.LayerNorm(d)
                self.ff1 = nn.Linear(d, cfg.ffn)
                self.ff2 = nn.Linear(cfg.ffn, d)
                self.drop = nn.Dropout(cfg.dropout)

            def kv(self, audio_h, audio_mask):
                k = self.k(self.ln_kv(audio_h)).view(*audio_h.shape[:2], h, d // h).transpose(1, 2)
                v = self.v(self.ln_kv(audio_h)).view(*audio_h.shape[:2], h, d // h).transpose(1, 2)
                return k, v

            def forward(self, text, audio_h, audio_mask, kv):
                import torch.nn.functional as F

                q = self.q(self.ln_q(text))
                bh, tl, _ = q.shape
                q = q.view(bh, tl, h, d // h).transpose(1, 2)
                k, v = kv
                attn = (q @ k.transpose(-1, -2)) / math.sqrt(d // h)
                attn = attn.masked_fill(~audio_mask[:, None, None, :], float("-inf"))
                attn = attn.softmax(dim=-1)
                out = (attn @ v).transpose(1, 2).reshape(bh, tl, d)
                text = text + self.drop(self.o(out))
                ff = self.ff2(self.drop(F.gelu(self.ff1(self.ln_ff(text)))))
                return text + self.drop(ff)

        class _Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.audio_proj = nn.Linear(cfg.audio_in, d)
                self.text_proj = nn.Linear(cfg.text_in, d)
                self.ln_audio_in = nn.LayerNorm(d)
                self.ln_text_in = nn.LayerNorm(d)
                self.blocks = nn.ModuleList([_Block() for _ in range(cfg.layers)])
                self.ln_out = nn.LayerNorm(d)
                self.pool_w = nn.Linear(d, 1)
                self.head_ln = nn.LayerNorm(2 * d + 1)
                self.head = nn.Sequential(
                    nn.Linear(2 * d + 1, 512), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(512, 1)
                )
                self.logit_scale = nn.Parameter(torch.tensor(math.log(20.0)))

            def encode_audio(self, audio, audio_mask):
                a = self.ln_audio_in(self.audio_proj(audio))
                kvs = [blk.kv(a, audio_mask) for blk in self.blocks]
                denom = audio_mask.sum(-1, keepdim=True).clamp_min(1)
                a_mean = (a * audio_mask.unsqueeze(-1)).sum(1) / denom
                return kvs, a_mean

            def score_prepared(self, text, text_mask, audio_mask, kvs, a_mean):
                import torch.nn.functional as F

                x = self.ln_text_in(self.text_proj(text))
                for blk, kv in zip(self.blocks, kvs):
                    x = blk(x, None, audio_mask, kv)
                x = self.ln_out(x)
                s = self.pool_w(x).squeeze(-1).masked_fill(~text_mask, float("-inf"))
                w = s.softmax(-1)
                pooled = (w.unsqueeze(-1) * x).sum(1)
                denom = text_mask.sum(-1, keepdim=True).clamp_min(1)
                mean = (x * text_mask.unsqueeze(-1)).sum(1) / denom
                pooled_n = F.normalize(pooled, dim=-1)
                a_n = F.normalize(a_mean, dim=-1)
                cos = (pooled_n * a_n).sum(-1, keepdim=True)
                feat = self.head_ln(torch.cat([pooled, mean, cos], dim=-1))
                return self.head(feat).squeeze(-1)

            def score(self, text, text_mask, audio, audio_mask):
                kvs, a_mean = self.encode_audio(audio, audio_mask)
                return self.score_prepared(text, text_mask, audio_mask, kvs, a_mean)

        self.net = _Net()

    # convenience -------------------------------------------------------
    def to(self, device):
        self.net.to(device)
        return self

    def train(self):
        self.net.train()
        return self

    def eval(self):
        self.net.eval()
        return self

    def state_dict(self):
        return self.net.state_dict()

    def load_state_dict(self, sd):
        self.net.load_state_dict(sd)
        return self


def save_head(path: Path, scorer: CrossAttnScorer, payload: Dict[str, Any]) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "cfg": scorer.cfg.to_dict(),
            "state_dict": {k: v.cpu() for k, v in scorer.state_dict().items()},
            **payload,
        },
        path,
    )


def load_head(path: Path = HEAD_PATH, device: str = "cuda") -> Tuple[CrossAttnScorer, Dict[str, Any]]:
    import torch

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    scorer = CrossAttnScorer(RerankerConfig(**ckpt["cfg"]))
    scorer.load_state_dict(ckpt["state_dict"])
    scorer.to(device).eval()
    return scorer, ckpt


# ── reranking helpers ────────────────────────────────────────────────────────


def zscore_rows(x: np.ndarray) -> np.ndarray:
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True)
    sd[sd < 1e-6] = 1.0
    return (x - mu) / sd


def mix_scores(stage1: np.ndarray, cross: np.ndarray, alpha: float) -> np.ndarray:
    """Per-query z-normalised interpolation; alpha=1 -> pure stage-1."""
    return alpha * zscore_rows(stage1) + (1.0 - alpha) * zscore_rows(cross)


def topk_candidates(stage1: np.ndarray, k: int) -> np.ndarray:
    k = min(k, stage1.shape[1])
    return np.argpartition(-stage1, kth=k - 1, axis=1)[:, :k]


def first_hit_ranks(scores: np.ndarray, positives: Sequence[np.ndarray]) -> np.ndarray:
    ranks = np.empty(scores.shape[0], dtype=np.int32)
    for i in range(scores.shape[0]):
        pos = np.asarray(positives[i], dtype=np.int64).reshape(-1)
        best = float(scores[i, pos].max())
        ranks[i] = int(np.sum(scores[i] > best)) + 1
    return ranks


def _text_batch(text_tokens, text_lengths, idx, device):
    import torch

    tok, mask = pad_batch(text_tokens, text_lengths, idx)
    return torch.from_numpy(tok).to(device).float(), torch.from_numpy(mask).to(device)


def score_pairs(
    scorer: CrossAttnScorer,
    text_tokens: np.ndarray,
    text_lengths: np.ndarray,
    audio: "DeviceTokens",
    pair_text: np.ndarray,
    pair_audio: np.ndarray,
    chunk: int = 512,
) -> np.ndarray:
    """Score ``len(pair_text)`` (text, audio) pairs with the frozen head.

    Pairs are grouped by audio so each clip's per-layer K/V is computed once
    per chunk; the chunk size bounds activation memory for 250-token audio.
    """
    import torch

    n = len(pair_text)
    out = np.empty(n, dtype=np.float32)
    with torch.no_grad():
        order = np.argsort(pair_audio, kind="stable")
        for start in range(0, n, chunk):
            sel = order[start : start + chunk]
            uniq, inv = np.unique(pair_audio[sel], return_inverse=True)
            a_tok, a_mask = audio.batch(uniq)
            kvs, a_mean = scorer.net.encode_audio(a_tok, a_mask)
            t_tok, t_mask = _text_batch(text_tokens, text_lengths, pair_text[sel], audio.device)
            a_idx = torch.from_numpy(inv).to(audio.device)
            sub_kvs = [(k[a_idx], v[a_idx]) for k, v in kvs]
            s = scorer.net.score_prepared(t_tok, t_mask, a_mask[a_idx], sub_kvs, a_mean[a_idx])
            out[sel] = s.float().cpu().numpy()
    return out


def score_pairs_train(
    scorer: CrossAttnScorer,
    text_tokens: np.ndarray,
    text_lengths: np.ndarray,
    audio: "DeviceTokens",
    pair_text: np.ndarray,
    pair_audio: np.ndarray,
    chunk: int = 512,
):
    """Differentiable counterpart of :func:`score_pairs`; returns a tensor."""
    import torch

    n = len(pair_text)
    order = np.argsort(pair_audio, kind="stable")
    pieces = []
    for start in range(0, n, chunk):
        sel = order[start : start + chunk]
        uniq, inv = np.unique(pair_audio[sel], return_inverse=True)
        a_tok, a_mask = audio.batch(uniq)
        kvs, a_mean = scorer.net.encode_audio(a_tok, a_mask)
        t_tok, t_mask = _text_batch(text_tokens, text_lengths, pair_text[sel], audio.device)
        a_idx = torch.from_numpy(inv).to(audio.device)
        sub_kvs = [(k[a_idx], v[a_idx]) for k, v in kvs]
        pieces.append(
            scorer.net.score_prepared(t_tok, t_mask, a_mask[a_idx], sub_kvs, a_mean[a_idx])
        )
    ordered = torch.cat(pieces)
    out = torch.empty(n, device=ordered.device, dtype=ordered.dtype)
    out[torch.from_numpy(order).to(ordered.device)] = ordered
    return out


def resolve_alpha(
    stage1: np.ndarray,
    cross: np.ndarray,
    positives: Sequence[np.ndarray],
    grid: Optional[Sequence[float]] = None,
) -> Tuple[float, List[Dict[str, float]]]:
    grid = list(grid if grid is not None else np.round(np.arange(0.0, 1.0001, 0.1), 2))
    rows = []
    best = (-1.0, 1.0)
    for a in grid:
        ranks = first_hit_ranks(mix_scores(stage1, cross, float(a)), positives)
        r10 = float(np.mean(ranks <= 10))
        r1 = float(np.mean(ranks <= 1))
        rows.append({"alpha": float(a), "R@1": r1, "R@10": r10})
        if r10 > best[0] + 1e-9:
            best = (r10, float(a))
    return best[1], rows
