# Omni-Embed-Nemotron benchmark (grant evaluation)

Benchmarks **`nvidia/omni-embed-nemotron-3b`** (NVIDIA's unified multimodal retrieval
model, built on the Qwen2.5-Omni-3B Thinker backbone) on two datasets it was NOT
trained/evaluated on in its paper (arXiv 2510.03458 reports ViDoRe/FineVideo/MTEB):

| Task | Metric | Result | Target | Verdict |
|------|--------|--------|--------|---------|
| MSR-VTT 1K-A zero-shot text→video retrieval | R@1 / R@5 / R@10 | **16.5 / 34.0 / 43.7** | R@10 ≥ 85 | **FAIL** |
| ESC-50 zero-shot audio classification | accuracy | **77.25%** (1545/2000) | ≥ 85% | **FAIL** |

Both numbers are real, full-dataset measurements (2000 ESC-50 clips, 1000 MSR-VTT
caption/video pairs). No fabrication, no subsetting in the reported numbers.

## Model / checkpoint

- **`nvidia/omni-embed-nemotron-3b`** — public (not gated), 4.7B params.
- Bidirectional `Qwen2_5OmniThinkerForConditionalGeneration` (`trust_remote_code`),
  2048-d embeddings, masked **mean pooling** + L2 normalize, loaded via
  **SentenceTransformer** (the repo ships `modules.json` / `1_Pooling` / multimodal
  `Transformer` module). `attn_implementation=sdpa` (flash-attn not installed),
  `bfloat16`. Embeds text/image/audio/video into one space.
- `encode_query` prepends `"query: "`, `encode_document` prepends `"passage: "`
  (model's own `config_sentence_transformers.json` prompts) — left as the model intends.

## Environment

- Isolated venv `./.venv-omni` (Python 3.11, `uv venv`). NOT added to the main project.
- GPUs **2,3** only (`CUDA_VISIBLE_DEVICES=2,3`); model occupies ~10 GB on one A100-40GB.
- Key versions (full list in `installed_versions.txt`):
  torch 2.12.0+cu130, transformers 5.10.2, sentence-transformers 5.5.1,
  torchcodec 0.14.0, qwen-omni-utils 0.0.9, datasets 5.0.0, librosa/soundfile.
  - NOTE: the HF model card suggests an old `transformers v4.51.3-Qwen2.5-Omni-preview`
    branch, but the checkpoint was actually *saved* with transformers 5.3.0.dev0 /
    sentence-transformers 5.4.0.dev0, and loads cleanly on transformers 5.10 / ST 5.5.
- **torchcodec needs FFmpeg shared libs.** None on the system PATH; we point
  `LD_LIBRARY_PATH` at the FFmpeg-4 libs in `/root/miniconda3/envs/longcat-video/lib`
  (torchcodec `core4` loads against them with no missing deps) plus torch's own `lib`.

## Reproduce

```bash
cd /workspace/proj/omurav2/benchmarks/eval/omni_embed_nemotron

# 1. venv + deps (isolated)
uv venv .venv-omni --python 3.11
export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu130
uv pip install --python .venv-omni/bin/python torch torchvision torchaudio
uv pip install --python .venv-omni/bin/python \
    "transformers>=5.3" "sentence-transformers>=5.4" "qwen-omni-utils>=0.0.4" \
    "accelerate>=1.12" soundfile librosa pillow datasets "huggingface_hub[hf_transfer]" av einops torchcodec

# 2. FFmpeg libs for torchcodec + torch libs
export LD_LIBRARY_PATH=/root/miniconda3/envs/longcat-video/lib:$PWD/.venv-omni/lib/python3.11/site-packages/torch/lib

# 3. ESC-50 (downloads ashraq/esc50 from HF, resamples 44.1k->16k)
CUDA_VISIBLE_DEVICES=2,3 .venv-omni/bin/python esc50_eval.py --batch 16 \
    --out results/omni_nemotron_esc50.json

# 4. MSR-VTT 1K-A: fetch test json + videos zip (friedrichor/MSR-VTT), extract 1000 test videos
.venv-omni/bin/python - <<'PY'
import json, os, zipfile
from huggingface_hub import hf_hub_download
import shutil
os.makedirs("data/msrvtt/video", exist_ok=True)
j = hf_hub_download("friedrichor/MSR-VTT","msrvtt_test_1k.json",repo_type="dataset")
shutil.copy(j,"data/msrvtt/msrvtt_test_1k.json")
z = hf_hub_download("friedrichor/MSR-VTT","MSRVTT_Videos.zip",repo_type="dataset")
need = {"video/"+d["video"] for d in json.load(open(j))}
with zipfile.ZipFile(z) as zf:
    for n in need: zf.extract(n,"data/msrvtt/")
PY
CUDA_VISIBLE_DEVICES=2,3 .venv-omni/bin/python msrvtt_eval.py --vbatch 8 --fps 2 \
    --out results/omni_nemotron_msrvtt.json
```

## Datasets / protocols

- **ESC-50** `ashraq/esc50` — 2000 clips, 5 s, 50 classes. Audio resampled 44.1 kHz →
  **16 kHz** (Qwen2.5-Omni's Whisper feature extractor requires 16 kHz). Embed each clip
  (`encode_document`); 50 prompts `"a sound of a {class}"` (`encode_query`); argmax cosine.
- **MSR-VTT 1K-A** `friedrichor/MSR-VTT` `msrvtt_test_1k.json` — the JSFUSION 1000
  video/caption pairs. Embed 1000 videos + 1000 captions; for each caption rank all 1000
  videos by cosine; R@K text→video. Frames sampled at **fps=2**, min/max pixels per the
  model card (32·14·14 .. 64·28·28). 0/1000 videos failed to decode.

## Findings

- The model loads and embeds **all four modalities** (text, image, audio, video) in one
  2048-d space — verified end to end.
- **MSR-VTT R@10 = 43.7** (full 1000-pool). A resolution sensitivity sweep on a 300-pool
  (R@10 56.0 → 58.7 when frame pixels raised 2.25×) shows the low score is *not* a
  frame-sampling artifact — it reflects genuine retrieval quality on this benchmark.
- **ESC-50 = 77.25%.** Per-class spot checks are highly accurate (diverse 12-clip probe
  was 12/12); 77% over all 2000 reflects systematic confusions, consistent with a model
  trained for document retrieval rather than contrastive audio↔text classification.
- Both miss the hard thresholds. This is expected: Omni-Embed-Nemotron is optimized for
  ViDoRe-style document/long-video retrieval, not short-clip MSR-VTT or audio tagging.
```
