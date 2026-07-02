"""ESC-50 zero-shot: compares several prompt templates (audio embedded once, reused)."""
import os, sys, json, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import iv2_common as C

TEMPLATES = [
    "a sound of a {}",
    "this is a sound of {}",
    "the sound of {}",
    "{}",
    "a recording of a {}",
]


def main():
    from datasets import load_dataset
    import torchaudio
    ds = load_dataset("ashraq/esc50", split="train")
    n = len(ds)
    cats = {}
    for ex in ds:
        cats[int(ex["target"])] = ex["category"]
    class_ids = sorted(cats.keys())
    class_names = [cats[i] for i in class_ids]
    id2pos = {cid: p for p, cid in enumerate(class_ids)}

    model, tok, cfg = C.load_model()

    # embed all audio once
    t = time.time()
    afeats, gts = [], []
    buf, bgt = [], []

    def flush():
        if not buf:
            return
        af = C.get_audio_feat(model, buf).cpu()
        afeats.append(af)
        gts.extend(bgt)
        buf.clear(); bgt.clear()

    for i, ex in enumerate(ds):
        a = ex["audio"]
        w = np.asarray(a["array"], dtype=np.float32)
        sr = int(a["sampling_rate"])
        if sr != 16000:
            w = torchaudio.functional.resample(torch.as_tensor(w), sr, 16000)
        buf.append(torch.as_tensor(w, dtype=torch.float32))
        bgt.append(id2pos[int(ex["target"])])
        if len(buf) >= 16:
            flush()
    flush()
    A = torch.cat(afeats, 0)  # [N,768]
    gts = np.array(gts)
    print(f"[esc50-prompts] audio embedded in {time.time()-t:.1f}s, N={A.shape[0]}")

    results = {}
    for tmpl in TEMPLATES:
        prompts = [tmpl.format(name.replace("_", " ")) for name in class_names]
        T = C.get_text_feat(model, tok, prompts, max_txt_l=cfg.max_txt_l).cpu()
        pred = (A @ T.T).argmax(dim=-1).numpy()
        acc = float((pred == gts).mean() * 100.0)
        results[tmpl] = round(acc, 2)
        print(f"  {tmpl!r:30s} -> {acc:.2f}%")

    best = max(results, key=results.get)
    out = {
        "task": "ESC-50 zero-shot audio classification (prompt sweep)",
        "checkpoint": "InternVideo2-Stage2-6B (with BEATs audio encoder)",
        "num_clips": n, "num_classes": len(class_names),
        "accuracy_by_prompt": results,
        "best_prompt": best, "best_accuracy": results[best],
        "target_accuracy": 85, "pass": results[best] >= 85,
    }
    op = os.path.join(os.path.dirname(__file__), "..", "results", "internvideo2_esc50_promptsweep.json")
    json.dump(out, open(op, "w"), indent=2)
    print("wrote", op)


if __name__ == "__main__":
    main()
