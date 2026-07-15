# Benchmark Reproduction Guide

Exact commands to reproduce every retrieval-accuracy number cited in
`docs/milestone2-evidence-bundle.md` and the `omura-embed-video` /
`omura-embed-audio` model cards on Hugging Face. All three benchmarks are
run from repo root unless noted.

---

## 1 · Image retrieval — MS COCO 1K (`immortaltatsu/omura_emebd`)

**Target:** ≥ 90% R@10 (text→image). **Result:** 94.64% R@10.

```bash
PYTHONPATH=. python benchmarks/benchmark_coco_retrieval.py \
  --num-images 1000 \
  --max-captions-per-image 5
```

COCO val2014 images/annotations auto-download to `data/coco/` on first run
(`--no-download-coco` to require a pre-downloaded copy). Output written to
`benchmarks/benchmarks/results/coco_omura_repro_1k.json` (`text_to_image.R@10`).

---

## 2 · Video retrieval — MSR-VTT 1K-A (`omura-embed-video`)

**Target:** ≥ 85% R@10 (text→video). **Result:** 85.3% R@10 (DSL scoring).

Requires the `.venv-iv2` environment (see `benchmarks/eval/internvideo2/SETUP.md`
for one-time setup: clone `OpenGVLab/InternVideo2`, download the base 6B
checkpoint, install the pinned dependency set). The finetuned heads are automatically 
downloaded from Hugging Face (`immortaltatsu/omura-embed-video`) on first run if 
`OMURA_VIDEO_HEADS` is not set.

```bash
cd benchmarks/eval/internvideo2
export IV2_CKPT=/path/to/internvideo2-s2_6b-224p-f4_with_audio_encoder.pt

.venv-iv2/bin/python scripts/eval_msrvtt_vtm.py \
  --json /path/to/msrvtt_test_1k.json \
  --video_dir data/msrvtt_videos/video \
  --out results/internvideo2_msrvtt.json
```

`friedrichor/MSR-VTT` (`MSRVTT_Videos.zip` + `msrvtt_test_1k.json`) is fetched via
`huggingface_hub.hf_hub_download` — see `SETUP.md` for the exact download snippet.
Result JSON's `metrics_t2v.DSL.R@10` (or `metrics_t2v.VTM_rerank.R@10`) is the reported figure.

Zero-shot temporal localization (no finetuning applied to this task) on
Charades-STA:

```bash
.venv-iv2/bin/python scripts/eval_charades_sta.py \
  --out results/charades_sta.json
```

---

## 3 · Audio retrieval — ESC-50 (`omura-embed-audio`)

Two related but distinct numbers are reported; both are reproducible and neither
supersedes the other — they answer different questions.

### 3a. CLAP zero-shot, full dataset (86.65%, cited in the M2 evidence bundle)

No finetuning; evaluates all 2000 ESC-50 clips with the base
`laion/larger_clap_general` model.

```bash
cd benchmarks/eval/clap
.venv-clap/bin/python eval_esc50_clap.py \
  --out results/clap_esc50_larger_general.json
```

### 3b. omura-embed-audio adapter, held-out fold only (85.25% → 95.75%)

We can evaluate the published `immortaltatsu/omura-embed-audio` adapter head on the held-out fold 5 directly:

```bash
cd benchmarks/eval/clap
.venv-clap/bin/python eval_esc50_clap.py \
  --adapter \
  --fold 5 \
  --out results/clap_adapter_fold5.json
```

Alternatively, to re-train the small adapter head locally on ESC-50 folds 1-4 and evaluate it on fold 5:

```bash
cd benchmarks/eval/clap
.venv-clap/bin/python finetune_esc50_head.py \
  --epochs 3 \
  --held-out-fold 5 \
  --out omura_clap_head.pt \
  --results-out results/clap_finetuned_esc50.json
```

Both `.venv-clap` and `.venv-iv2` need their Python interpreter restored if
missing (`uv python install <version>` per each venv's `pyvenv.cfg`) since
these venvs pin exact patch versions.

---

## 4 · Seek-to-timestamp / in-video localization (Walrus catalog probe)

Production-data verification of temporal localization, using synthetic
multi-scene videos built from concatenated catalog clips with known boundaries:

```bash
cd benchmarks/eval/internvideo2
.venv-iv2/bin/python scripts/eval_walrus_temporal.py \
  --out results/walrus_temporal.json
```

---

## 5 · Image-to-Image retrieval — MS COCO 1K (`immortaltatsu/omura_emebd` vs stock SigLIP)

Evaluates retrieval of original COCO images using query images that are cropped (5%), resized (90%), and compressed (JPEG Q60). Demonstrates the ~10% improvement in visual alignment over the baseline model.

```bash
# Fine-tuned model (R@10 targets high-90s)
.venv/bin/python benchmarks/benchmark_coco_i2i.py \
  --num-images 1000 \
  --split-file data/coco/dataset_coco.json \
  --images-dir data/coco/val2014 \
  --out-json benchmarks/results/coco_i2i_repro.json

# Baseline model (approx 10% lower Recall@K)
OMURA_EMBEDDING_MODEL="google/siglip2-so400m-patch14-384" .venv/bin/python benchmarks/benchmark_coco_i2i.py \
  --num-images 1000 \
  --split-file data/coco/dataset_coco.json \
  --images-dir data/coco/val2014 \
  --out-json benchmarks/results/coco_i2i_baseline.json
```

---

## Notes on reproducibility

- All scripts write a `--out`/`--results-out` JSON with the exact protocol
  (dataset, split, prompt template, model, seed where applicable) alongside
  the metrics, so results are self-documenting.
- Model weights referenced above are on Hugging Face:
  [`immortaltatsu/omura_emebd`](https://huggingface.co/immortaltatsu/omura_emebd),
  [`immortaltatsu/omura-embed-video`](https://huggingface.co/immortaltatsu/omura-embed-video),
  [`immortaltatsu/omura-embed-audio`](https://huggingface.co/immortaltatsu/omura-embed-audio).
  The latter two are finetuned deltas on top of third-party base models
  (OpenGVLab's InternVideo2-6B and LAION's CLAP respectively) — see each
  model card for the base-model attribution and what was/wasn't retrained.

