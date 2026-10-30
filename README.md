# Omura Benchmarks — reproduction mirror

Standalone harnesses, result artifacts and protocols for Omura's retrieval
benchmarks. Every number in the evidence bundle can be regenerated from this
repository.

- **Backend / product repo:** <https://github.com/OmuraHQ/omura-backend>
- **Model family:** `google/embeddinggemma-2` (768-d, text + image + video;
  audio via the Gemma audio adapter)

## Phase 3 results (as measured)

| Dataset | Direction | R@10 | Artifact |
|---|---|---:|---|
| Flickr30k 1K test | text→image | **98.70%** | `benchmarks/results/flickr30k_gemma_1k.json` |
| Flickr30k 5K pool (headline) | text→image | **95.06%** | `benchmarks/results/flickr30k_gemma_5k.json` |
| Sui-native curated set | text→image | **99.10%** | `benchmarks/results/sui_nft_gemma.json` |
| COCO 1K (cross-check) | text→image | **96.80%** | `benchmarks/results/coco_gemma_1k.json` |
| ActivityNet Captions (1K videos) | text→video | **98.70%** | `benchmarks/results/activitynet_gemma.json` |
| MSR-VTT 1K-A | text→video | 83.30% | `benchmarks/results/msrvtt_gemma.json` |
| AudioCaps (883 clips) | text→audio | 62.05% (Gemma+adapter) | `benchmarks/results/audiocaps_gemma_finetuned.json` |
| Clotho (held-out) | text→audio | 31.12% (Gemma+adapter) | `benchmarks/results/clotho_gemma_adapter.json` |
| AudioCaps — CLAP reference | text→audio | 86.10% | `benchmarks/results/audiocaps_finetuned.json` |

Audio is the open gap: the Gemma audio adapter generalizes (Clotho 15.73% →
31.12%) but is below the 90% target. The CLAP run is a labelled non-canonical
reference, not part of the Gemma-only evidence.

## Quick start

```bash
uv sync
export OMURA_EMBEDDING_BACKEND=omuragemma
export OMURA_EMBEDDING_DIM=768
export PYTHONPATH=.:benchmarks
export CUDA_VISIBLE_DEVICES=4          # pick a free GPU

# image
.venv/bin/python benchmarks/benchmark_flickr30k.py --pool 5000
.venv/bin/python benchmarks/build_sui_nft_eval.py --target 1000
.venv/bin/python benchmarks/benchmark_sui_nft.py
.venv/bin/python benchmarks/benchmark_coco_retrieval.py --num-images 1000 --include-image-to-text

# video
.venv/bin/python benchmarks/benchmark_activitynet.py \
    --videos-dir data/activitynet/ActivityNet --num-videos 1000
.venv/bin/python benchmarks/benchmark_msrvtt.py

# audio
.venv/bin/python benchmarks/benchmark_audiocaps.py
.venv/bin/python benchmarks/benchmark_clotho.py
```

Datasets download into `data/` on first run (gitignored). Exact per-artifact
commands: [`docs/BENCHMARK_PHASE3.md`](docs/BENCHMARK_PHASE3.md).

## Protocol

All claims follow [`docs/GENERALIZATION_PROTOCOL.md`](docs/GENERALIZATION_PROTOCOL.md):

- No test-set tuning; dev split for selection, test evaluated once.
- Hardest documented pool is the headline (Flickr30k 5K, not 1K).
- One held-out dataset per modality (Sui-native/COCO, ActivityNet/MSR-VTT,
  Clotho).
- No caption leakage; `leakage_check` is recorded in the artifacts.
- Adapter weights, train/dev sizes and per-query ranks ship with the results.

## Layout

```
benchmarks/
  bench_common.py                 # recall@k, result writers
  embedding_backend.py            # SigLIP / omuragemma dispatch
  benchmark_*.py                  # dataset harnesses
  finetune_audiocaps_*.py         # audio adapters / LoRA
  train_crossattn.py              # Gemma cross-attention reranker
  rerank_audiocaps_pipeline.py    # CLAP/VAST fusion failsafe
  results/                        # measured artifacts (JSON)
docs/
  BENCHMARK_PHASE3.md             # numbers, protocols, gaps
  BENCHMARK_REPRODUCTION.md       # historical M1/M2 reproduction
  GENERALIZATION_PROTOCOL.md
  GENERALIZATION_CLOTHO.md
  PIPELINE_RERANK.md
  milestone3-evidence.md          # full Phase 3 evidence bundle
```

## Notes

- The `omura/utils/` subset here is the minimum needed by the harnesses; the
  product repo is canonical for the API.
- Result JSONs include per-query ranks, so metrics can be re-derived without
  re-running the models.
- CLAP results are retained as a reference; they do not count toward the
  Gemma-only Phase 3 claim.
