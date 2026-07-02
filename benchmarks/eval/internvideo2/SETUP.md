# InternVideo2 grant-eval — reproducible setup

All work isolated under this directory. GPUs 0,1 only (prefix every torch command
with `CUDA_VISIBLE_DEVICES=0,1`). Nothing was added to the main project's uv env.

## Model variant used
**InternVideo2-Stage2-6B (with BEATs audio encoder)** — single combined checkpoint
`OpenGVLab/InternVideo2-Stage2_6B-224p-f4 :: internvideo2-s2_6b-224p-f4_with_audio_encoder.pt`
(28 GB, **ungated**, downloadable anonymously).

Why 6B instead of 1B: the 1B video-text checkpoints (`InternVideo2-Stage2_1B-224p-f4`,
`InternVideo2-CLIP-1B-224p-f8`) are **gated** (HTTP 401 without an accepted-terms HF token,
which this environment does not have). The 6B repo is ungated AND the only public checkpoint
that bundles the audio (BEATs) tower, so it is the single model that can do BOTH MSR-VTT
(video-text) and ESC-50 (audio-text). Published 6B MSR-VTT T2V is even stronger than 1B
(R@1 55.9 vs 51.9). 6B in bf16 ≈ 13 GB, fits comfortably on one A100-40GB.

Flash-attention / fused kernels are **disabled** (`use_flash_attn=use_fused_mlp=use_fused_rmsnorm=False`)
so the backbone runs its built-in `_naive_attn` path — identical math, no flash_attn/apex/deepspeed
build needed. A tiny `flash_attn` stub package is placed in the venv so the repo's eager
`import flash_attn` (only used by unused CLIP-teacher code) succeeds; it raises if ever called
(it never is).

## Venv + install
```bash
cd benchmarks/eval/internvideo2
uv venv .venv-iv2 --python 3.10
export VIRTUAL_ENV=$PWD/.venv-iv2

uv pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
   --index-url https://download.pytorch.org/whl/cu121
uv pip install transformers==4.28.1 "tokenizers<0.14" timm==0.5.4 einops \
   decord opencv-python-headless==4.8.0.76 librosa==0.10.1 soundfile==0.12.1 \
   "datasets>=2.18,<3" pandas pyarrow easydict pyyaml termcolor scipy ftfy regex tqdm \
   "huggingface_hub>=0.23"
uv pip install peft==0.5.0 "accelerate<0.30" open_clip_torch
uv pip install numpy==1.24.4          # cv2 0.4.8 needs numpy 1.x (open_clip pulled np2)
uv pip install "setuptools<81"        # librosa audio decode needs pkg_resources

# flash_attn stub (kernels disabled, never called):
#   site-packages/flash_attn/{__init__,flash_attn_interface,bert_padding}.py
#   site-packages/flash_attn/modules/mlp.py  (FusedMLP)
#   site-packages/flash_attn/ops/rms_norm.py (DropoutAddRMSNorm)
```
(av==11.0.0 from requirements.txt was dropped — needs system ffmpeg dev libs to build and
is not imported on the code paths used here; decord handles video reading.)

## Repo + checkpoint + datasets
```bash
git clone --depth 1 https://github.com/OpenGVLab/InternVideo.git repo
# checkpoint (HF cache, shared):
python -c "from huggingface_hub import hf_hub_download as d; d('OpenGVLab/InternVideo2-Stage2_6B-224p-f4','internvideo2-s2_6b-224p-f4_with_audio_encoder.pt')"
# MSR-VTT 1K-A (videos + 1k test json):
python -c "from huggingface_hub import hf_hub_download as d; [d('friedrichor/MSR-VTT',f,repo_type='dataset') for f in ['MSRVTT_Videos.zip','msrvtt_test_1k.json']]"
unzip -q MSRVTT_Videos.zip -d data/msrvtt_videos
# ESC-50: datasets.load_dataset('ashraq/esc50') (auto-cached)

# BEATs cfg+weights are extracted from the combined checkpoint into
# data/beats_from_combined.pth (BEATs_iter3+ config; 0 missing / 0 unexpected keys).
```

## Run
```bash
export IV2_CKPT=<.../internvideo2-s2_6b-224p-f4_with_audio_encoder.pt>
CUDA_VISIBLE_DEVICES=0,1 .venv-iv2/bin/python scripts/eval_msrvtt.py \
   --json <.../msrvtt_test_1k.json> --video_dir data/msrvtt_videos/video \
   --out results/internvideo2_msrvtt.json
CUDA_VISIBLE_DEVICES=0,1 .venv-iv2/bin/python scripts/eval_esc50.py \
   --out results/internvideo2_esc50.json
```
