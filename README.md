# Benchmarks

This folder contains standalone benchmarking scripts and output artifacts.
Dependencies are defined in `benchmarks/pyproject.toml`.

## Scripts

- `benchmark_omura_emmbed.py`: startup time, text/image throughput, and retrieval sanity benchmark.
- `benchmark_coco_retrieval.py`: COCO text-image retrieval benchmark (Recall@K).

## Quick start

```bash
cd benchmarks
uv run benchmark_omura_emmbed.py --rounds 5
```

```bash
cd benchmarks
uv run benchmark_coco_retrieval.py --num-images 1000
```

COCO val2014 annotations and images are downloaded under `data/coco/` when missing (use `--no-download-coco` to require pre-downloaded data).

Paper-aligned Karpathy split protocol:

```bash
cd benchmarks
uv run benchmark_coco_retrieval.py --protocol siglip2_paper_val --split-file ../data/coco/dataset_coco.json --download-karpathy-split --images-dir ../data/coco/val2014 --num-images 1000
```

Note: `siglip2_paper_val` and `siglip2_paper_test` ignore `--num-images` and always use the full Karpathy split.

### Sanity check and pass/fail gate

`embedding_backend.py` uses **`padding="max_length"`** for SigLIP/SigLIP2 text (required for correct retrieval; `padding=True` breaks rankings).

- `--sanity-check N`: sample `N` captions and print mean cosine vs the GT image vs a random image (gap should be clearly positive).
- `--gate-r10 0.85 --gate-metric i2t`: exit with status 1 if **image→text** R@10 is below 0.85 (requires `--include-image-to-text`). On COCO 5k global, **i2t** R@10 in the high 80s is typical for strong encoders; **t2i** R@10 is usually **lower** than i2t, so do not use the same threshold for both unless you mean to.

Example:

```bash
cd benchmarks
uv run benchmark_coco_retrieval.py --protocol siglip2_paper_val \
  --split-file ../data/coco/dataset_coco.json --download-karpathy-split \
  --images-dir ../data/coco/val2014 --include-image-to-text \
  --sanity-check 512 --gate-r10 0.85 --gate-metric i2t \
  --out-json benchmarks/results/karpathy_val_full.json
```

## Results

Default outputs are written under `benchmarks/results/`.
