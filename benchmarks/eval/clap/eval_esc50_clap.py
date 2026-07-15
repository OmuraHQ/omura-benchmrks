"""Zero-shot ESC-50 accuracy for CLAP (native transformers ClapModel).

Protocol (matches the InternVideo2/Omni-Embed-Nemotron eval scripts):
  - dataset: ashraq/esc50 (2000 clips, 50 classes), cached in HF datasets cache
  - text prompt: "a sound of a {class}" (underscores -> spaces); also a small sweep
  - embed audio (resampled to 48 kHz) + class-text, argmax cosine, accuracy over all clips
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import ClapModel, ClapProcessor

PROMPTS = {
    "a_sound_of_a": "a sound of a {}",
    "a_recording_of_a": "a recording of a {}",
    "this_is_a": "this is a sound of {}",
}


def l2(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


class ClapAdapterHead(torch.nn.Module):
    """Small residual linear adapter: out = normalize(x + W x)."""

    def __init__(self, dim: int = 512):
        super().__init__()
        self.proj = torch.nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        import torch.nn.functional as F
        return F.normalize(x + self.proj(x), dim=-1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="laion/larger_clap_general")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--adapter", action="store_true", help="load and use the immortaltatsu/omura-embed-audio adapter")
    ap.add_argument("--fold", type=int, default=0, help="filter dataset by fold (1-5)")
    ap.add_argument("--out", type=Path, default=Path("results/clap_esc50.json"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[CLAP] loading {args.model} on {device}")
    model = ClapModel.from_pretrained(args.model).to(device).eval()
    processor = ClapProcessor.from_pretrained(args.model)
    target_sr = processor.feature_extractor.sampling_rate  # 48000 for CLAP

    adapter_head = None
    if args.adapter:
        try:
            from huggingface_hub import hf_hub_download
            print("[CLAP] Loading adapter head from HF (immortaltatsu/omura-embed-audio)...")
            path = hf_hub_download(repo_id="immortaltatsu/omura-embed-audio", filename="omura_clap_head.pt")
            ckpt = torch.load(path, map_location=device)
            adapter_head = ClapAdapterHead(dim=model.config.projection_dim).to(device)
            adapter_head.load_state_dict(ckpt["state_dict"])
            adapter_head.eval()
            print("[CLAP] Adapter loaded.")
        except Exception as e:
            print(f"[CLAP] Failed to load adapter: {e}")
            raise SystemExit(1)

    ds = load_dataset("ashraq/esc50", split="train")
    if args.fold:
        ds = ds.filter(lambda x: int(x["fold"]) == args.fold)
        print(f"[CLAP] Filtered to fold {args.fold}")
        
    # Canonical class list (sorted by ESC-50 target id for stability)
    cls_by_target = {}
    for ex in ds:
        cls_by_target[int(ex["target"])] = ex["category"]
    class_ids = sorted(cls_by_target)
    classes = [cls_by_target[i] for i in class_ids]
    target_to_idx = {t: i for i, t in enumerate(class_ids)}
    print(f"[CLAP] {len(ds)} clips, {len(classes)} classes, target_sr={target_sr}")

    import librosa

    # --- audio embeddings ---
    audio_vecs = np.zeros((len(ds), model.config.projection_dim), dtype=np.float32)
    labels = np.zeros(len(ds), dtype=np.int64)
    buf, idxs = [], []

    def flush(buf, idxs):
        if not buf:
            return
        inp = processor(audios=buf, sampling_rate=target_sr, return_tensors="pt").to(device)
        with torch.no_grad():
            feats = model.get_audio_features(**inp)
            if adapter_head is not None:
                feats = adapter_head(feats)
            feats = feats.cpu().numpy()
        for j, gi in enumerate(idxs):
            audio_vecs[gi] = feats[j]

    for i, ex in enumerate(ds):
        a = ex["audio"]
        arr = np.asarray(a["array"], dtype=np.float32)
        sr = int(a["sampling_rate"])
        if sr != target_sr:
            arr = librosa.resample(arr, orig_sr=sr, target_sr=target_sr)
        buf.append(arr)
        idxs.append(i)
        labels[i] = target_to_idx[int(ex["target"])]
        if len(buf) >= args.batch_size:
            flush(buf, idxs)
            buf, idxs = [], []
            if (i + 1) % 256 == 0:
                print(f"[CLAP] embedded {i+1}/{len(ds)} clips")
    flush(buf, idxs)
    audio_vecs = l2(audio_vecs)

    # --- per-prompt text embeddings + accuracy ---
    results = {}
    best = None
    for name, tmpl in PROMPTS.items():
        texts = [tmpl.format(c.replace("_", " ")) for c in classes]
        tin = processor(text=texts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            tvec = l2(model.get_text_features(**tin).cpu().numpy())
        preds = (audio_vecs @ tvec.T).argmax(axis=1)
        acc = float((preds == labels).mean())
        results[name] = {"prompt": tmpl, "accuracy": acc}
        print(f"[CLAP] prompt={name:18s} acc={acc*100:.2f}%")
        if best is None or acc > results[best]["accuracy"]:
            best = name

    out = {
        "model": args.model,
        "dataset": "ashraq/esc50",
        "task": f"zero-shot audio classification{' (adapter)' if args.adapter else ''}",
        "fold_filtered": args.fold or None,
        "num_clips": len(ds),
        "num_classes": len(classes),
        "target_sr": target_sr,
        "threshold": 0.85,
        "best_prompt": best,
        "best_accuracy": results[best]["accuracy"],
        "pass": results[best]["accuracy"] >= 0.85,
        "per_prompt": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\n[CLAP] BEST {best}: {results[best]['accuracy']*100:.2f}%  "
          f"({'PASS' if out['pass'] else 'FAIL'} vs 85%)")
    print(f"[CLAP] wrote {args.out}")


if __name__ == "__main__":
    main()
