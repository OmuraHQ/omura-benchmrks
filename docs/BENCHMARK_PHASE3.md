# Phase 3 retrieval benchmarks — measured results

Local backend: `google/embeddinggemma-2` (740M, 768-d unified space) through
`omura/utils/omuragemma_embeddings.py`, dispatched from
`benchmarks/embedding_backend.py` when `OMURA_EMBEDDING_BACKEND=omuragemma`.

| Field | Value |
|---|---|
| Model | `google/embeddinggemma-2` |
| Dimension | 768 (native, L2-normalized) |
| Runtime | `sentence-transformers` 6.2.0, `transformers` 5.20.2, bf16 |
| GPU | `CUDA_VISIBLE_DEVICES=4` and `5` (the live service keeps 6/7) |
| Commit | `172abd60a3c1ca41b1e1639dd935f03faff6012f` |
| Run date | 2026-10-30 (UTC) |
| Seed | 42 for every subset/sampling decision |

All numbers are **global retrieval over the full stated pool** (no sampled
negatives). Every artifact records `model`, `dim`, `protocol`, `pool_size`,
`num_queries`, `seed`, `commit`, `utc_date`, per-direction Recall@K, and — for
the Sui/AudioCaps/MSR-VTT sets — per-query ranks.

Text prefixes follow the model card: retrieval queries use
`task: search result | query: ...` and text corpus items use
`title: none | text: ...`. Image/audio/video inputs carry no text prefix. The
image→text direction therefore encodes each candidate caption with the document
prefix; the text→image direction encodes the same strings as queries.

## Headline table

| # | Condition | Direction | Pool | Queries | R@1 | R@5 | **R@10** | Met? | Artifact |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| 3 | Flickr30k 1K (official test split) | text→image | 1,000 | 5,000 | 86.28% | 97.48% | **98.70%** | ✅ ≥95% | `benchmarks/results/flickr30k_gemma_1k.json` |
| 3 | Flickr30k 1K (official test split) | image→text | 1,000 | 1,000 | 94.20% | 99.70% | **100.00%** | ✅ | same |
| 3 | Flickr30k 5K pool (1K test + 4K train distractors) | text→image | 5,000 | 5,000 | 73.78% | 91.62% | **95.06%** | ✅ ≥95% (harder, non-standard pool) | `benchmarks/results/flickr30k_gemma_5k.json` |
| 3 | Flickr30k 5K pool | image→text | 5,000 | 1,000 | 85.70% | 97.60% | **99.10%** | ✅ | same |
| 3 | Sui-native catalog set (Omura production) | text→image | 1,000 | 1,000 | 87.40% | 98.10% | **99.10%** | ✅ ≥95% | `benchmarks/results/sui_nft_gemma.json` |
| 3 | Sui-native catalog set (transformed query) | image→image | 1,000 | 1,000 | 91.90% | 97.70% | **98.70%** | ✅ | same |
| 4 | AudioCaps test (public audio mirror) | text→audio | 883 | 4,308 | 7.82% | 24.49% | **35.19%** | ❌ ≥90% | `benchmarks/results/audiocaps_gemma.json` |
| 4 | AudioCaps test (public audio mirror) | audio→text | 883 | 883 | 6.80% | 21.97% | **32.84%** | ❌ | same |
| 4 | ActivityNet Captions (1,000-video subset) | text→video | 1,000 | 1,000 | 93.10% | 98.20% | **98.70%** | ✅ ≥90% | `benchmarks/results/activitynet_gemma.json` |
| 4 | MSR-VTT 1K-A | text→video | 1,000 | 1,000 | 52.40% | 74.90% | **83.30%** | ❌ ≥90% | `benchmarks/results/msrvtt_gemma.json` |

## Protocols and qrels

### Flickr30k (`benchmark_flickr30k.py`)

* Source: `nlphuji/flickr30k` (`flickr30k-images.zip` + `flickr_annotations_30k.csv`),
  cached under `data/flickr30k/`.
* Official test split: 1,000 images × 5 captions. All 5 captions are text→image
  queries; qrels = the one source image.
* Text→image: query × 1,000 candidate images.
* Image→text: each test image × every caption of every pool image (5,000
  candidate captions at pool 1,000). A hit is any of that image's 5 captions.
* `--pool 5000`: the same 1,000 test images **plus 4,000 deterministic train
  images as distractors**. This is a deliberately harder, non-standard pool and
  is labelled as such in the artifact
  (`flickr30k_1k_test_queries_pool_5000_with_train_distractors`).
* Failure handling: images that fail to decode are dropped and counted; the
  result reports the kept pool size.

### Sui-native set (`build_sui_nft_eval.py` + `benchmark_sui_nft.py`)

* Source: `data/blob_catalog.sqlite` opened read-only
  (`file:...?mode=ro`), intersected with the production EmbeddingGemma image
  snapshot (`data/vector_index_omuragemma/metadata.json`, 4,669 images) so every
  chosen blob was fetchable when that snapshot was built.
* Filters: `kind='image'`, catalog caption length > 10 chars, dedupe by
  normalized caption (4,669 → 4,666 exact-unique), deterministic shuffle, then
  dedupe again by SHA-256 of fetched bytes.
* Fetch: `WALRUS_AGGREGATOR_URL` / `WALRUS_AGGREGATOR_URLS` from `.env`
  (local aggregator first, public fallback), 24 workers, 40 s timeout.
* Result: **1,000 images, 0 fetch failures**, frozen manifest at
  `data/sui_nft_eval/manifest.json`. Each entry records `blob_id`,
  `walrus_blob_id`, `caption(s)`, `sha256`, `catalog_source`, and explicit
  `null` for `collection` / `project` / `sui_object_id` — the catalog has no
  such columns. Provenance is therefore `blob_id` + `catalog_source`
  (`blockberry` / `walrus_event`), not a separate Sui object id.
* Text→image: one catalog caption per image, global 1,000-image pool.
* Image→image: query = original image with 5% border crop, 90% resize, JPEG
  quality 60; candidate = original 1,000 images. Images larger than 2,048 px on
  the long side are downscaled before the transform (some on-chain images are
  45 MP and otherwise stall Pillow on allocation).
* Per-query ranks for both directions are in the artifact.

### AudioCaps (`benchmark_audiocaps.py`)

* Source: `OpenSound/AudioCaps` test shards (41 parquet files, 3.57 GB) on
  HuggingFace, cached under `data/audiocaps/parquet`, audio extracted to
  `data/audiocaps/audio/`.
* The mirror's 4,411 caption rows group by `(youtube_id, start_time)` into
  **883 unique 10 s clips with 4,308 unique captions** (mean 4.88 captions per
  clip). The official AudioCaps test split has 975 clips / 4,875 captions; the
  public audio-bearing mirror is missing 92 clips. Pool size is reported as 883,
  never padded or synthesized.
* Text→audio: all 4,308 captions are queries over all 883 clips.
* Audio→text: each clip is a query over all 4,308 captions; qrels = that clip's
  captions.
* One clip per item, mono 16 kHz, capped at 300 s (clips are ~10 s, so no
  truncation in practice).
* Text handling follows the model card: query captions use the `SearchQuery`
  prefix (`task: search result | query: ...`), corpus captions use the document
  prefix (`title: none | text: ...`). Audio/video/image inputs carry no prefix.

### ActivityNet Captions (`benchmark_activitynet.py`)

* Source: `rrustlee/ActivityNet1fps` on HuggingFace
  (`ActivityNet.tar.gz`, 8.49 GB; 1-fps re-encode) plus
  `ActivityNet-dense-cap.json`. Extracted to `data/activitynet/ActivityNet/`
  (8,215 videos; all 4,036 annotation ids resolve).
* Protocol: deterministic 1,000-video subset (`--num-videos 1000`, seed 42),
  one representative caption per video (the first segment caption), global
  text→video retrieval. This is a subset of the 1-fps mirror, not the official
  full val split, and the artifact says so.
* If the video directory is absent the harness writes
  `{"status": "no_data", "blocker": ...}` rather than a fabricated metric. It
  did not need to do so here.

### MSR-VTT (`benchmark_msrvtt.py`)

* Source: existing local InternVideo2 copy —
  `benchmarks/eval/internvideo2/data/anno/msrvtt_test_1k.json` and
  `benchmarks/eval/internvideo2/data/msrvtt_videos/video/` (10,000 videos).
* 1K-A: 1,000 unique videos, one caption each; all 1,000 captions are queries
  over all 1,000 videos.
* Video embedding: up to 32 frames sampled uniformly across each clip
  (`OMURA_OMURAGEMMA_MAX_FRAMES=32`).
* No hybrid rerank was applied. The pure number is reported below the 90%
  threshold rather than tuned against it.

## Gaps (thresholds not reached)

| Gap | Measured | Threshold | Why it was not closed |
|---|---:|---:|---|
| AudioCaps text→audio R@10 | **35.19%** | 90% | The local EmbeddingGemma 2 audio encoder is far weaker than its image/text encoders on fine-grained environmental-sound captions. The harness is correct (same caption is retrieved far above chance: 35% vs 1.1% random at pool 883), but the model's audio retrieval quality is the blocker. No threshold-driven caption/audio tuning was applied. |
| AudioCaps audio→text R@10 | **32.84%** | 90% | Same model limitation, measured in the reverse direction. |
| MSR-VTT text→video R@10 | **83.30%** | 90% | 32-frame uniform sampling, pure video embedding, no rerank. A 58-frame inference was not run because selecting it after seeing the 32-frame result would be threshold fitting. |
| AudioCaps pool completeness | 883 clips | 975 official | The public mirror with actual audio (`OpenSound/AudioCaps`) has 883 unique clips; the missing 92 clips' audio is not in that mirror. Downloading `confit/audiocaps` test.zip (4 GB) or the gated `HuggingFaceM4/AudioCaps` is the next step; neither was required to report an honest full-pool number for the available data. |
| Sui collection / Sui object id | `null` | — | The production catalog has no `collection`, `project`, or `sui_object_id` columns. Walrus provenance (`blob_id` + `catalog_source`) is recorded. Adding object ids needs a catalog migration, not a benchmark change. |

## Reproduction

```bash
export OMURA_EMBEDDING_BACKEND=omuragemma
export OMURA_EMBEDDING_DIM=768
export OMURA_VECTOR_STORE_DIR=data/vector_index_omuragemma
export CUDA_VISIBLE_DEVICES=4
export PYTHONPATH=.:benchmarks

# 1) Flickr30k (downloads/extracts under data/flickr30k)
.venv/bin/python benchmarks/benchmark_flickr30k.py --pool 1000
.venv/bin/python benchmarks/benchmark_flickr30k.py --pool 5000

# 2) Sui-native set
.venv/bin/python benchmarks/build_sui_nft_eval.py --target 1000
.venv/bin/python benchmarks/benchmark_sui_nft.py

# 3) AudioCaps (downloads shards under data/audiocaps)
.venv/bin/python benchmarks/benchmark_audiocaps.py

# 4) ActivityNet (after extracting the mirror under data/activitynet)
.venv/bin/python benchmarks/benchmark_activitynet.py \
    --videos-dir data/activitynet/ActivityNet --num-videos 1000

# 5) MSR-VTT 1K-A
.venv/bin/python benchmarks/benchmark_msrvtt.py
```

Unit tests (no GPU/model download) cover the batch-normalization bug fixed in
`_normalize` and the batch text path:

```bash
.venv/bin/python -m pytest tests/test_omuragemma_embeddings.py -q
```

## Backend changes that made these runs possible

* `_normalize()` now preserves the leading batch axis. Before the fix,
  `model.encode([...N texts...])` returned `(768,)` instead of `(N, 768)`,
  silently discarding all but the first row.
* New batch helpers: `generate_text_embeddings`, `generate_image_embeddings`,
  `generate_audio_embeddings`, `generate_video_embeddings`, used by all
  benchmark scripts. `benchmarks/embedding_backend.py` dispatches to
  `omura.utils.omuragemma_embeddings` for all four modalities when
  `OMURA_EMBEDDING_BACKEND=omuragemma`; the SigLIP path remains the default.

## Audio side fine-tune: EmbeddingGemma 2 on AudioCaps

The frozen model's AudioCaps text→audio R@10 is **35.19%** (row 4 above).
Two audio-only fine-tuning stages were run to close the ≥90% gap while keeping
the 768-d Gemma family space (no CLAP component). The best honest test result is
**70.36% t2a R@10** (LoRA + adapter), i.e. **+35.17 points over frozen** but
**below the 90% target**.

### Data and leakage

* **Train:** `data/audiocaps_train/` — `OpenSound/AudioCaps` train split, 412
  parquet shards, **45,178 unique clips**, one caption per clip, 24 kHz stereo
  decoded to 16 kHz mono. A local int16 memmap
  (`data/audiocaps_gemma_cache/train_audio_i16.bin`, 14 GB) avoids re-reading
  the 37 GB parquet corpus each epoch.
* **Test:** the same **883 clips / 4,308 captions** as
  `benchmarks/results/audiocaps_gemma.json`.
* **Split (seed 42):** 40,660 train / 4,518 dev clips (10% dev). All model
  selection used a dev pool of 883 clips (same size as the test pool) plus the
  full 4,518-clip dev pool. Test was evaluated once per stage with the final
  dev-selected checkpoint.
* **Leakage:** asserted **0** overlap on `(youtube_id, start_time)`,
  `audiocap_id`, and `youtube_id`; test captions are never training documents.
  Every JSON carries a `leakage_check` block with the counts.

### Architecture and hyperparameters

The module tree was inspected at runtime. Audio path:
`Gemma4AudioModel` (12 layers, 1024 hidden, `output_proj` 1024→1536) →
`embed_audio` (RMSNorm + 1536→512) → soft tokens in the frozen text model
(24 layers) → mean pooling → 768-d L2-normalised embedding. 10 s audio is
25 tokens/s = 250 soft tokens.

**Stage 1 — frozen backbone + residual projection heads**
(`data/audiocaps_gemma_adapter.pt`)

* Features: frozen `model.encode({"audio": 16k mono})` and
  `model.encode("task: search result | query: " + caption)`; also the
  `"title: none | text: "` document variant for a2t.
* Head: `x + fc2(GELU(fc1(LayerNorm(x))))`, 768→2048→768, zero-init `fc2`
  (near-identity). Separate audio and text heads (shared was worse on dev).
* Loss: symmetric multi-positive InfoNCE; the 2 text prefixes per caption are
  both positives, learnable logit scale. AdamW lr 1e-3, weight decay 0.01,
  batch 256, ≤30 epochs, early stop on dev-pool t2a R@10 (best epoch 7).
* Dev-pool t2a R@10 **62.06%**.

**Stage 2 — LoRA + audio-side projector** (`data/audiocaps_gemma_lora.pt`,
then adapter refit `data/audiocaps_gemma_adapter_lora_h768_lr3e-4.pt`)

* LoRA rank 16 / alpha 32 on `q_proj`, `k_proj`, `v_proj`, `post`,
  `ffw_layer_1`, `ffw_layer_2` of all 12 audio-encoder layers (96 modules,
  7.87M params), plus unfrozen `embed_audio` and `audio_tower.output_proj`.
  Text backbone and vision tower frozen.
* InfoNCE against the frozen query/document text embeddings, lr 1e-4, bf16,
  2 epochs × 24,000 train clips, batch 8, AdamW. Dev-pool t2a R@10 **54.81%**
  in the native space; a residual adapter refit on the LoRA features (768→768,
  lr 3e-4, wd 0.01) lifts it to **67.27%**.
* The batch is 8, not ≥256: GPU 4 was shared with another 25–30 GB job during
  the run, leaving ~10 GB; batch 16 OOM'd even with the unused vision tower
  dropped. The stage-1 adapter (stage 2's second phase) does use batch 256.

**Stage 3 — last 6 audio layers full fine-tune**
(`data/audiocaps_gemma_full.pt`)

* Unfroze `audio_tower.layers[-6:]`, `output_proj`, `embed_audio`, and the
  stage-1 audio head; text frozen. Optimised symmetric InfoNCE in the frozen
  stage-1 adapter target space, lr 5e-5, bf16, 1 epoch × 24,000 clips, batch 8.
* Dev-pool t2a R@10 **51.98%** — it regressed versus stage 2, so it is reported
  but not promoted. The one allowed test run gives 64.02%.

### Results (full 883-clip test pool, 4,308 caption queries)

| Stage | t2a R@1 | t2a R@5 | **t2a R@10** | a2t R@1 | a2t R@5 | **a2t R@10** | Artifact |
|---|---:|---:|---:|---:|---:|---:|---|
| Frozen `google/embeddinggemma-2` | 7.82% | 24.49% | **35.19%** | 6.80% | 21.97% | **32.84%** | `audiocaps_gemma.json` |
| 1. Adapter (frozen backbone) | 17.76% | 46.77% | **62.05%** | 21.06% | 45.19% | **59.00%** | `audiocaps_gemma_finetuned.json` |
| 2. LoRA + adapter | 23.35% | 55.76% | **70.36%** | 26.73% | 57.87% | **71.12%** | `audiocaps_gemma_finetuned_lora.json` |
| 3. Last-6-layer full tune + adapter | 19.50% | 49.33% | **64.02%** | 21.18% | 49.15% | **63.53%** | `audiocaps_gemma_finetuned_full.json` |

`train_size` / `dev_size` are 40,660 / 4,518 for stages 1–2 and 24,000 / 4,518
for the stage-3 epoch; the LoRA backbone itself saw 24,000 clips.

### Why 90% was not reached

The frozen audio embeddings are strongly anisotropic (mean off-diagonal cosine
0.761, std 0.074) with only a weak audio-text margin (diagonal 0.694 vs
off-diagonal 0.627). A projection head can recover a lot of that margin
(35→62%), and LoRA adds acoustic adaptation (→70%), but fine-tuning the audio
side with an 8-clip contrastive batch under a ~10 GB budget did not produce the
fine-grained environmental-sound discrimination that a CLAP-style encoder
already has. Full unfreezing without more epochs/negatives regressed. The CLAP
adapter workstream remains the higher-ceiling path; this Gemma-only path is
reported honestly below the 90% threshold rather than tuned on test.

### Exact inference recipe

```python
# audio document (16 kHz mono float32)
a = gemma.encode({"audio": wav16k_mono}, normalize_embeddings=True)
a = l2(audio_head(a))                 # stage-1/2 residual head; no head for stage 3-only
# text query
t = gemma.encode("task: search result | query: " + caption, normalize_embeddings=True)
t = l2(text_head(t))
# text document for a2t
d = gemma.encode("title: none | text: " + caption, normalize_embeddings=True)
d = l2(text_head(d))
```

The LoRA/full weights live on the audio tower + `embed_audio`; the text backbone
and vision tower are unchanged, so both sides remain in the same 768-d
EmbeddingGemma 2 space. Checkpoints store the exact head architecture
(`residual_mlp`, hidden) with the weights.

### Reproduction

```bash
export CUDA_VISIBLE_DEVICES=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 0) cache frozen train/test embeddings + 16 kHz waveform memmap
.venv/bin/python benchmarks/finetune_audiocaps_gemma.py --stage features

# 1) projection adapter (best dev config)
.venv/bin/python benchmarks/finetune_audiocaps_gemma.py --stage adapter \
  --hidden 2048 --lr 3e-4 --dropout 0.1 --weight-decay 0.01 --epochs 30 \
  --out data/audiocaps_gemma_adapter.pt
.venv/bin/python benchmarks/benchmark_audiocaps_gemma_ft.py \
  --checkpoint data/audiocaps_gemma_adapter.pt \
  --out benchmarks/results/audiocaps_gemma_finetuned.json

# 2) LoRA + adapter
.venv/bin/python benchmarks/finetune_audiocaps_gemma.py --stage lora \
  --limit 24000 --epochs 2 --batch-size 8 --lr 1e-4 --dev-pool 883 \
  --drop-vision --out data/audiocaps_gemma_lora.pt
.venv/bin/python benchmarks/finetune_audiocaps_gemma.py --stage extract \
  --from-checkpoint data/audiocaps_gemma_lora.pt --feature-name lora
.venv/bin/python benchmarks/finetune_audiocaps_gemma.py --stage adapter \
  --audio-features data/audiocaps_gemma_cache/train_audio_lora.npy \
  --hidden 768 --lr 3e-4 --weight-decay 0.01 --epochs 25 \
  --out data/audiocaps_gemma_adapter_lora_h768_lr3e-4.pt
.venv/bin/python benchmarks/benchmark_audiocaps_gemma_ft.py \
  --checkpoint data/audiocaps_gemma_adapter_lora_h768_lr3e-4.pt \
  --test-emb test_emb_lora.npz \
  --model-label "google/embeddinggemma-2+audiocaps-lora+adapter" \
  --out benchmarks/results/audiocaps_gemma_finetuned_lora.json

# 3) last-6-layer full tune
.venv/bin/python benchmarks/finetune_audiocaps_gemma.py --stage full --layers 6 \
  --limit 24000 --epochs 1 --batch-size 8 --lr 5e-5 --dev-pool 883 \
  --drop-vision --adapter-ckpt data/audiocaps_gemma_adapter.pt \
  --train-audio-head --out data/audiocaps_gemma_full.pt
.venv/bin/python benchmarks/benchmark_audiocaps_gemma_ft.py \
  --checkpoint data/audiocaps_gemma_full.pt \
  --model-label "google/embeddinggemma-2+audiocaps-full" \
  --out benchmarks/results/audiocaps_gemma_finetuned_full.json
```
