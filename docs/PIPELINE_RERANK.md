# Multi-model rerank fallback (AudioCaps + Clotho)

**Evidence tier: `fallback_pipeline`.** This is a labelled fallback evaluation,
not the canonical Gemma-only evidence. It does not modify
`benchmarks/results/audiocaps_gemma_*.json`,
`benchmarks/results/clotho_gemma*.json`, or `scripts/check_conditions.py`.

## What was tested

Stage-1 pools come from the existing AudioCaps projection adapter over frozen
EmbeddingGemma 2 (`data/audiocaps_gemma_adapter.pt`, 768-d). A second,
independently trained scorer — frozen `laion/larger_clap_general` plus the
AudioCaps dual adapter (`data/audiocaps_adapter.pt`, 512-d, CSLS k=10) — reranks
each pool. Scores are z-scored per query over the candidate set and fused:

```
score = (1 - w_clap) * z(gemma) + w_clap * z(clap)
```

The scalar `w_clap` was selected on the **AudioCaps validation split only**
(445 clips / 2,179 captions), once per direction, using the union pool. The
frozen values are `w_clap = 0.84` for t2a and `0.80` for a2t. AudioCaps test and
Clotho were then evaluated with those weights; no test-set tuning was done.

## Candidate-pool ceilings (AudioCaps test)

| Pool | mean size | t2a pool recall | a2t pool recall |
|---|---:|---:|---:|
| Gemma top-100 | 100 | 0.9559 | 0.9536 |
| Gemma top-200 | 200 | 0.9851 | 0.9807 |
| Gemma top-100 ∪ CLAP top-100 | 151 | 0.9956 | 0.9977 |

The top-100 ceiling equals the published Gemma R@100 (95.59%). No reranker can
exceed the pool ceiling.

## AudioCaps test (883 clips / 4,308 captions)

t2a R@10:

| Pipeline | Gemma top-100 | Gemma top-200 | Union top-100 |
|---|---:|---:|---:|
| Gemma only | 0.6205 | 0.6205 | 0.6205 |
| CLAP rerank | 0.8498 | 0.8589 | 0.8610 |
| Fused (dev weights) | **0.8570** | **0.8691** | **0.8714** |

a2t R@10:

| Pipeline | Gemma top-100 | Gemma top-200 | Union top-100 |
|---|---:|---:|---:|
| Gemma only | 0.5900 | 0.5900 | 0.5900 |
| CLAP rerank | 0.8879 | 0.9026 | 0.9117 |
| Fused (dev weights) | 0.8788 | 0.9037 | **0.9173** |

Full R@1/5/10 values are in `benchmarks/results/audiocaps_pipeline_rerank.json`
under `test`. Representative fused t2a at union: R@1 0.3918, R@5 0.7539,
R@10 0.8714. The dev table is in the same artifact under `dev`; dev t2a fused
R@10 at union is 0.9307, so the test drop is a genuine domain gap, not a
protocol change.

## Clotho v2.1 test (1,045 clips / 5,225 captions)

Frozen AudioCaps weights and CSLS k were applied unchanged. No Clotho data was
used for training, CSLS selection, weight selection, or pool selection.

t2a R@10:

| Pipeline | Gemma top-100 | Gemma top-200 | Union top-100 |
|---|---:|---:|---:|
| Gemma only | 0.3112 | 0.3112 | 0.3112 |
| CLAP rerank | 0.5158 | 0.5370 | 0.5359 |
| Fused (frozen) | **0.5336** | **0.5636** | **0.5657** |

a2t R@10: Gemma 0.2679, fused 0.4622 (top-100), 0.4928 (top-200), 0.5225
(union). Clotho pool ceilings are 0.7847 / 0.8988 / 0.9533 for t2a and
0.7301 / 0.8574 / 0.9445 for a2t.

## VAST status

VAST was **not available as a frozen AudioCaps reranker**. The official release
is `CASIA-IVA-Lab/VAST` (MIT). Its published checkpoint
(`pretrain_vast.tar.gz`, 3.1 GB, Google Drive) contains pretraining weights for
fine-tuning, not the AudioCaps-finetuned checkpoint that produced the reported
82.9% t2a R@10. Running it also requires BEATs, `bert-base-uncased`, and
EVA01-CLIP weights that are not present locally. The pretraining archive was
downloaded and inspected but not integrated. This artifact is therefore the
required **CLAP-only** pipeline.

## Honest conclusion

**The fallback does not reach the 90% t2a R@10 target.** The best AudioCaps t2a
result is **87.14%** (fused, union pool), up from CLAP-alone 86.10% and
Gemma-alone 62.05%. The limit is the second scorer: CLAP's own full-pool t2a
R@10 is 86.10%, and reranking can only recover a little of CLAP's top-10 misses
using the weaker Gemma signal. The union pool ceiling (99.56%) confirms the
bottleneck is ranking quality, not candidate recall.

The pipeline does generalize in the useful direction on Clotho: fused t2a R@10
is 53.36% at top-100 versus 31.12% for Gemma alone, and 56.57% on the union pool.

## Caveat on dev

The CLAP adapter and CSLS k were early-stopped/selected on the same AudioCaps
validation split used to tune fusion weights, so dev metrics are optimistic for
the CLAP component. AudioCaps test and Clotho remain untouched by selection.

## Reproduce

```bash
# GPU 5 is the benchmark GPU; the live service uses 6-7.
CUDA_VISIBLE_DEVICES=5 OMURA_EMBEDDING_BACKEND=omuragemma \
  .venv/bin/python benchmarks/cache_val_gemma.py
CUDA_VISIBLE_DEVICES=5 .venv/bin/python benchmarks/cache_clotho_clap.py
CUDA_VISIBLE_DEVICES=5 .venv/bin/python benchmarks/rerank_audiocaps_pipeline.py
```

Artifacts:
- `benchmarks/results/audiocaps_pipeline_rerank.json`
- `benchmarks/results/clotho_pipeline_rerank.json`

Per-query ranks use `-1` when the ground truth is absent from the candidate
pool. Everything is tagged `"evidence_tier": "fallback_pipeline"`.
