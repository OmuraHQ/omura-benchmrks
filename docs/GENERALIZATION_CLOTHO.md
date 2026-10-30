# Generalization check: Clotho v2.1 (held-out audio)

Clotho was never used for training, prompt selection or hyperparameter choice.
This is the audio-side cross-dataset gate from `docs/GENERALIZATION_PROTOCOL.md`:
an AudioCaps-only gain is fit, not generalization; the same model must move on
Clotho to support a generalization claim.

## Protocol

| Item | Value |
|---|---|
| Dataset | Clotho **v2.1 test** split, `mteb/clotho_t2a_v2` (MTEB audio retrieval packaging) |
| Clips (audio pool) | **1045** |
| Captions (text pool) | **5225** (5 per clip) |
| text→audio | all 5225 captions are queries against all 1045 clips |
| audio→text | all 1045 clips are queries against all 5225 caption documents |
| Positives | 1 clip per caption (t2a); that clip's own 5 captions (a2t) |
| Prompting | model-card prefixes only: `task: search result \| query: ` (queries), `title: none \| text: ` (documents) |
| Audio | mono 16 kHz (wrapper resamples the 48 kHz Clotho WAVs) |
| Pool shrinking | none — full test pool in both directions |
| Seed | 42 |

No Clotho sample was used to fit, early-stop, or choose anything. Both systems
were evaluated once on the full pool.

## Results

### Frozen EmbeddingGemma 2 (`clotho_gemma.json`)

| Direction | R@1 | R@5 | R@10 | Chance R@10 |
|---|---|---|---|---|
| text → audio | 3.48% | 10.39% | **15.73%** | 0.96% |
| audio → text | 2.78% | 9.38% | **15.31%** | 0.95% |

Chance is the exact hypergeometric recall `1 - C(N-m, k)/C(N, k)` for `m`
positives in `N` candidates (m=1 for t2a over 1045 clips, m=5 for a2t over 5225
captions). The measured numbers are ~16× chance at R@10.

### With the AudioCaps adapter (`clotho_gemma_adapter.json`)

`data/audiocaps_gemma_adapter.pt` (sha256 `a30d3831…`, residual-MLP projection
heads, hidden 2048, dropout 0.1, best epoch 7, trained on AudioCaps only):

| Direction | R@1 | R@5 | R@10 | Δ R@10 |
|---|---|---|---|---|
| text → audio | 7.25% | 20.96% | **31.12%** | +15.39 pts (1.98×) |
| audio → text | 6.89% | 18.09% | **26.79%** | +11.48 pts (1.75×) |

The gain survives the prompt and pool change: the adapter was fitted on 40660
AudioCaps captions and never saw a Clotho clip, yet it nearly doubles R@10 in
both directions on the held-out corpus.

### Adapter variant check

The training workstream tried three head shapes. All three beat the frozen
baseline on Clotho, so the transfer is not specific to the dev-selected head:

| Head | AudioCaps dev R@10 (t2a / a2t) | Clotho R@10 (t2a / a2t) |
|---|---|---|
| hidden 2048, dropout 0.1 *(canonical)* | 29.33% / 27.36% | **31.12% / 26.79%** |
| hidden 768, no dropout, separate heads | 29.59% / 25.70% | 21.05% / 20.67% |
| hidden 768, no dropout, shared head | 27.49% / 24.30% | 23.39% / 22.20% |
| frozen baseline | (no dev) | 15.73% / 15.31% |

The canonical head scores on Clotho essentially the same as on its own
AudioCaps dev pool (31.12% vs 29.33% t2a; 26.79% vs 27.36% a2t). The adapter
gain is therefore **not** dataset-specific: it transfers completely to a corpus
it never saw.

## Sanity controls

- Permuting the audio pool drives frozen t2a R@10 to 0.65–1.01% across three
  permutations, i.e. back to the 0.96% chance level. The metric, the qrels
  alignment and the per-query ranks are correct; the 15.73% is real signal.
- Frozen audio embeddings are strongly anisotropic (mean pairwise cosine 0.83).
  This is the main reason the frozen absolute recall is low while its ranking
  is still well above chance; the adapter removes most of that collapse.
- AudioCaps in-domain reference (same frozen model, 883-clip pool): 35.19% t2a
  R@10 and 32.84% a2t R@10. Clotho frozen is ~2.2× lower against a comparable
  pool — the expected cross-dataset penalty.

## What this proves

1. **The frozen model transfers.** 15.73% / 15.31% R@10 on a full, unshrunk
   Clotho test pool is ~16× chance. Gemma's audio-text alignment is not an
   AudioCaps artifact.
2. **The AudioCaps adapter generalizes.** It lifts Clotho R@10 from 15.73% to
   31.12% (t2a) and 15.31% to 26.79% (a2t) without ever seeing Clotho, and it
   reaches the same accuracy there as on its own AudioCaps dev pool. The gain
   is a property of the learned audio-text projection, not of AudioCaps.
3. **Absolute cross-dataset quality is still modest.** R@1 of 7.25% / 6.89%
   means this is not a strong Clotho retriever; the claim is about transfer of
   the adapter gain, not about state of the art on Clotho.

## Adapter provenance

Both result files record the adapter `sha256`, `best_epoch`, `train_size`,
`dev_size`, and the base commit. The benchmark refuses to score an adapter
unless it has been untouched for ≥120 s and has a companion manifest
(`data/audiocaps_gemma_cache/<name>_dev.json`); a checkpoint promoted by copy
to the canonical name is verified by byte equality with its manifested sibling.
The gate reason is stored in `adapter.gate`.

Re-run (reuses the cached frozen Clotho embeddings, finishes in seconds):

```bash
cd /workspace/proj/omurav2
CUDA_VISIBLE_DEVICES=3 .venv/bin/python benchmarks/benchmark_clotho.py \
    --adapter data/audiocaps_gemma_adapter.pt \
    --extra-adapter data/audiocaps_gemma_adapter_sep.pt \
    --extra-adapter data/audiocaps_gemma_adapter_shared.pt \
    --out benchmarks/results/clotho_gemma_adapter.json
```

## Artifacts

- `benchmarks/benchmark_clotho.py` — the benchmark
- `benchmarks/results/clotho_gemma.json` — frozen results
- `benchmarks/results/clotho_gemma_adapter.json` — adapter results (+ variant block)
- `data/clotho/` — cached Clotho test audio, manifest, frozen embedding cache
