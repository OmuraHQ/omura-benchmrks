"""Zero-shot audio classification on ESC-50 (2000 clips, 50 classes)."""
import os, sys, json, time, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import iv2_common as C


def resample(wav, sr_in, sr_out=16000):
    if sr_in == sr_out:
        return wav
    import torchaudio
    return torchaudio.functional.resample(
        torch.as_tensor(wav, dtype=torch.float32), sr_in, sr_out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--prompt", default="a sound of a {}")
    args = ap.parse_args()

    from datasets import load_dataset
    ds = load_dataset("ashraq/esc50", split="train")
    if args.limit:
        ds = ds.select(range(args.limit))
    n = len(ds)

    # class list (index by 'target' label id)
    cats = {}
    for ex in ds:
        cats[int(ex["target"])] = ex["category"]
    class_ids = sorted(cats.keys())
    class_names = [cats[i] for i in class_ids]
    id2pos = {cid: p for p, cid in enumerate(class_ids)}
    prompts = [args.prompt.format(name.replace("_", " ")) for name in class_names]
    print(f"[esc50] {n} clips, {len(class_names)} classes")
    print("  sample prompts:", prompts[:3])

    t0 = time.time()
    model, tok, cfg = C.load_model()
    print(f"[esc50] model ready in {time.time()-t0:.1f}s")

    # text embeddings for the 50 class prompts
    tfeats = C.get_text_feat(model, tok, prompts, max_txt_l=cfg.max_txt_l).cpu()  # [50,768]

    # audio embeddings
    t = time.time()
    correct = 0
    bs = 16
    preds, gts = [], []
    buf_wav, buf_gt = [], []

    def flush():
        nonlocal correct
        if not buf_wav:
            return
        af = C.get_audio_feat(model, buf_wav).cpu()  # [B,768]
        sims = af @ tfeats.T  # [B,50]
        p = sims.argmax(dim=-1)
        for j, gt in enumerate(buf_gt):
            pred_pos = int(p[j])
            preds.append(pred_pos)
            gts.append(id2pos[gt])
            if pred_pos == id2pos[gt]:
                correct += 1
        buf_wav.clear(); buf_gt.clear()

    for i, ex in enumerate(ds):
        a = ex["audio"]
        wav = np.asarray(a["array"], dtype=np.float32)
        sr = int(a["sampling_rate"])
        w = resample(wav, sr, 16000)
        w = torch.as_tensor(w, dtype=torch.float32)
        buf_wav.append(w)
        buf_gt.append(int(ex["target"]))
        if len(buf_wav) >= bs:
            flush()
        if (i + 1) % 200 == 0:
            print(f"  clip {i+1}/{n}  acc_so_far={correct/(i+1)*100:.1f}%  ({(time.time()-t)/(i+1):.2f}s/clip)")
    flush()

    acc = correct / n * 100.0
    print(f"[esc50] accuracy = {acc:.2f}%  ({correct}/{n})")

    result = {
        "task": "ESC-50 zero-shot audio classification",
        "model": "InternVideo2-Stage2-6B (with BEATs audio encoder), naive-attn bf16",
        "checkpoint": "OpenGVLab/InternVideo2-Stage2_6B-224p-f4 :: internvideo2-s2_6b-224p-f4_with_audio_encoder.pt",
        "protocol": {
            "dataset": "ashraq/esc50 (2000 clips, 50 classes)",
            "num_clips": n,
            "num_classes": len(class_names),
            "prompt_template": args.prompt,
            "audio": "resampled to 16kHz, pad/trunc to 10s, BEATs 128-mel fbank",
            "metric": "argmax cosine(audio_proj, text_proj)",
        },
        "accuracy": round(acc, 2),
        "correct": correct,
        "target_accuracy": 85,
        "pass": acc >= 85,
        "note": "Checkpoint was trained with audio-video-text (avtc/avtm) objectives; "
                "direct audio-text contrastive (atc) was disabled (weight 0) during pretraining, "
                "so zero-shot audio->text alignment may be weaker than vision->text.",
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=2)
    print("[esc50] wrote", args.out)


if __name__ == "__main__":
    main()
