"""Publish the finetuned 'omura embed video' model to the Hugging Face Hub.

Uploads the finetuned heads + inference code + config + eval results + a generated
model card. Requires an HF write token (HF_TOKEN env or --token).

  HF_TOKEN=hf_xxx .venv-iv2/bin/python scripts/publish_omura_embed_video.py \
      --heads data/finetune_v1/best_heads.pt --results data/finetune_v1/msrvtt_finetune.json \
      --repo immortaltatsu/omura_embed_video
"""
import os, sys, json, argparse, tempfile, shutil


CARD = """---
license: mit
library_name: pytorch
pipeline_tag: feature-extraction
tags:
  - omura
  - video-retrieval
  - text-to-video
  - internvideo2
  - multimodal-embeddings
---

# Omura Embed Video

Text↔video embedding model for the Omura / Walrus search engine. A parameter-efficient
finetune of **InternVideo2-Stage2-6B** (`OpenGVLab/InternVideo2-Stage2_6B-224p-f4`) on
MSR-VTT, adapting the joint contrastive space for text→video retrieval.

## Results (MSR-VTT 1K-A, text→video, ITC)

| Metric | Base (zero-shot ITC) | Omura Embed Video |
|---|---|---|
| R@1  | {R1_base} | **{R1_ft}** |
| R@5  | {R5_base} | **{R5_ft}** |
| R@10 | {R10_base} | **{R10_ft}** |

Target R@10 ≥ 85: **{PASS}**.

## What was finetuned

Frozen: the 6B vision tower and BEATs audio tower. Trained: the BERT text encoder,
the vision/text projection heads, and the contrastive temperature, with symmetric
InfoNCE over MSR-VTT train (9k) captions. The frozen vision features were precomputed
once, so training runs on a single GPU.

## Files

- `best_heads.pt` — finetuned `text_encoder` / `vision_proj` / `text_proj` / `temp`.
- `iv2_finetuned.py`, `iv2_common.py`, `iv2_6b_av_config.py` — inference code.
- `msrvtt_finetune.json` — full eval result.

## Usage

Load the base InternVideo2-6B checkpoint, then apply these heads (see `iv2_finetuned.py`):

```python
import iv2_finetuned as M           # from this repo
M.load(heads_path="best_heads.pt")  # set IV2_CKPT to the base 6B checkpoint
tvec = M.embed_text("a person riding a bike")          # [768], L2-normalized
vvec = M.embed_video_frames(frames_tensor)             # [B,768]
# cosine(tvec, vvec) ranks videos for the query.
```

Base checkpoint: `OpenGVLab/InternVideo2-Stage2_6B-224p-f4 :: internvideo2-s2_6b-224p-f4_with_audio_encoder.pt`.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--repo", default=os.getenv("OMURA_VIDEO_HF_REPO", "immortaltatsu/omura_embed_video"))
    ap.add_argument("--token", default=os.getenv("HF_TOKEN", ""))
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    if not args.token:
        print("ERROR: no HF token (set HF_TOKEN or --token).", file=sys.stderr)
        return 2

    from huggingface_hub import HfApi

    res = json.load(open(args.results))
    base = res.get("baseline_itc", {})
    best = res.get("best", {})
    card = CARD.format(
        R1_base=base.get("R@1", "?"), R5_base=base.get("R@5", "?"), R10_base=base.get("R@10", "?"),
        R1_ft=best.get("R@1", "?"), R5_ft=best.get("R@5", "?"), R10_ft=best.get("R@10", "?"),
        PASS=res.get("pass_R@10", False),
    )

    here = os.path.dirname(__file__)
    staging = tempfile.mkdtemp(prefix="omura_embed_video_")
    shutil.copy(args.heads, os.path.join(staging, "best_heads.pt"))
    shutil.copy(args.results, os.path.join(staging, "msrvtt_finetune.json"))
    for f in ("iv2_finetuned.py", "iv2_common.py", "iv2_6b_av_config.py"):
        shutil.copy(os.path.join(here, f), os.path.join(staging, f))
    open(os.path.join(staging, "README.md"), "w").write(card)

    api = HfApi(token=args.token)
    api.create_repo(args.repo, repo_type="model", exist_ok=True, private=args.private)
    api.upload_folder(folder_path=staging, repo_id=args.repo, repo_type="model")
    print(f"[publish] uploaded to https://huggingface.co/{args.repo}")
    print(f"[publish] R@10 {best.get('R@10')} (base {base.get('R@10')}), pass={res.get('pass_R@10')}")


if __name__ == "__main__":
    raise SystemExit(main())
