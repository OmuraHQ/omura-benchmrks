#!/usr/bin/env python
"""ESC-50 zero-shot audio classification with nvidia/omni-embed-nemotron-3b.

Protocol:
  - 2000 clips (ESC-50, ashraq/esc50, single split of 2000).
  - 50 class text prompts: "a sound of a {class}" (underscores -> spaces).
  - Embed each clip's audio (encode_document), embed 50 class prompts (encode_query).
  - argmax cosine similarity over class text embeddings; accuracy over all clips.
"""
import json, time, argparse
import numpy as np
import torch
from datasets import load_dataset, Audio
from sentence_transformers import SentenceTransformer

MODEL = "nvidia/omni-embed-nemotron-3b"
TARGET_SR = 16000  # Qwen2.5-Omni audio (Whisper) feature extractor requires 16 kHz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="limit clips (0=all)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", default="results/omni_nemotron_esc50.json")
    args = ap.parse_args()

    t0 = time.time()
    full = load_dataset("ashraq/esc50", split="train")
    # Build the canonical target(0-49) -> category map from the FULL dataset so the
    # class/prompt list is always complete even when --limit subsets the clips.
    id2cat = {}
    for t, c in zip(full["target"], full["category"]):
        id2cat.setdefault(t, c)
    classes = [id2cat[i] for i in range(len(id2cat))]
    num_classes = len(classes)
    assert num_classes == 50, f"expected 50 classes, got {num_classes}"

    ds = full
    if args.limit:
        ds = ds.select(range(args.limit))
    ds = ds.cast_column("audio", Audio(sampling_rate=TARGET_SR))
    n = len(ds)
    prompts = ["a sound of a " + c.replace("_", " ") for c in classes]
    print(f"clips={n} classes={num_classes}")

    model = SentenceTransformer(
        MODEL, trust_remote_code=True,
        model_kwargs={"attn_implementation": "sdpa", "torch_dtype": torch.bfloat16},
        device="cuda",
    )
    model[0].processing_kwargs.update({"audio": {"max_length": 2048000}})

    # class text embeddings
    class_emb = model.encode_query(prompts, convert_to_numpy=True, show_progress_bar=False)
    class_emb = np.asarray(class_emb, dtype=np.float32)

    correct = 0
    targets = np.array([ds[i]["target"] for i in range(n)])
    preds = np.zeros(n, dtype=np.int64)
    B = args.batch
    for start in range(0, n, B):
        end = min(start + B, n)
        docs = []
        for i in range(start, end):
            a = ds[i]["audio"]
            docs.append({"audio": {"array": np.asarray(a["array"], dtype=np.float32),
                                   "sampling_rate": a["sampling_rate"]}})
        emb = model.encode_document(docs, convert_to_numpy=True, show_progress_bar=False)
        emb = np.asarray(emb, dtype=np.float32)
        sims = emb @ class_emb.T  # both L2-normalized -> cosine
        p = sims.argmax(axis=1)
        preds[start:end] = p
        correct += int((p == targets[start:end]).sum())
        if start % (B * 20) == 0:
            print(f"  {end}/{n} running_acc={correct/end:.4f}", flush=True)

    acc = correct / n
    elapsed = time.time() - t0
    result = {
        "model": MODEL,
        "checkpoint": "nvidia/omni-embed-nemotron-3b (SentenceTransformer, mean pool, 2048-d, bidirectional Qwen2.5-Omni-3B Thinker)",
        "task": "ESC-50 zero-shot audio classification",
        "protocol": "embed audio (encode_document); 50 text prompts 'a sound of a {class}' (encode_query); argmax cosine; accuracy",
        "prompt_template": "a sound of a {class}",
        "audio_sampling_rate": TARGET_SR,
        "num_samples": n,
        "num_classes": num_classes,
        "accuracy": acc,
        "correct": correct,
        "target_threshold": 0.85,
        "pass": acc >= 0.85,
        "attn_implementation": "sdpa",
        "dtype": "bfloat16",
        "elapsed_sec": round(elapsed, 1),
    }
    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
