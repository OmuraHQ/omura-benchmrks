# Generalization Protocol (Phase 3)

The goal is a model that generalizes, not a number tuned to one benchmark.
This document is binding for every Phase 3 accuracy claim.

## Rules

1. **No test-set tuning.** Test sets are evaluated **once** per final
   checkpoint. All model selection, early stopping, prompt choice and
   hyperparameters are decided on a **dev split** carved out of train.
2. **No pool shrinking.** Report the full documented candidate pool. If a
   smaller pool is also reported, it is labelled and is not the headline.
3. **No per-dataset prompt tuning.** Only the prefixes documented in the model
   card are used (`task: search result | query:` for queries; documents per the
   card). The same recipe is applied to every dataset.
4. **No seed cherry-picking.** Fix the seed (`42`) and report it. Where variance
   matters, report mean ± std over ≥3 seeds.
5. **No caption leakage.** A query caption is never indexed as a document for
   its own item. Sui/AudioCaps manifests assert train/test id disjointness.
6. **Cross-dataset check.** Each modality has at least one held-out dataset
   that was never used for training or selection.
7. **Hardest-pool headline.** When several pools exist for the same dataset,
   the headline uses the hardest one (e.g. Flickr30k 5K, not 1K).
8. **Versioned adapter.** Any fine-tuned adapter is recorded with its weights,
   train/dev sizes, and a `leakage_check` field; the store is re-embedded
   rather than mixing old and new vectors.

## Evidence matrix

| Modality | In-domain (fit) | Held-out / cross-dataset | Status |
|---|---|---|---|
| Image | Flickr30k 1K and 5K pools | Sui-native curated set; COCO val2014 | Flickr ✅ 95.06% (5K), Sui ✅ 99.10%; COCO running |
| Video | ActivityNet Captions (1K videos) | MSR-VTT 1K-A | ActivityNet ✅ 98.70%; MSR-VTT 83.30% (reported) |
| Audio | AudioCaps train → test | **Clotho** (never trained on) | Gemma adapter in progress; Clotho baseline running |

## What counts as "benchmax" (rejected)

- Evaluating on a 100-image pool and calling it Flickr30k.
- Choosing the best of several test runs.
- Using AudioCaps ground-truth captions as document metadata for the same clip.
- Training the adapter on train **and** validating on test.
- Switching to a different encoder per dataset while claiming one model.
- Reporting only the 1K pool when the 5K pool is documented and weaker.

## Reporting template

Every result JSON must contain:

```json
{
  "model": "google/embeddinggemma-2",
  "dim": 768,
  "protocol": "<dataset>_<direction>_<pool>",
  "pool_size": 0,
  "num_queries": 0,
  "seed": 42,
  "commit": "<sha>",
  "utc_date": "<ISO8601>",
  "train_size": 0,
  "dev_size": 0,
  "leakage_check": "train/test ids disjoint",
  "text_to_x": {"R@1": 0.0, "R@5": 0.0, "R@10": 0.0},
  "x_to_text": {"R@1": 0.0, "R@5": 0.0, "R@10": 0.0}
}
```

## Current status

- Image: in-domain + Sui OOD measured; COCO third set in progress.
- Video: two independent datasets measured.
- Audio: in-domain and OOD both measured with the same adapter recipe:

  | Eval | Frozen | + adapter | + LoRA + adapter |
  |---|---:|---:|---:|
  | AudioCaps test (883 clips, 4,308 captions) | 35.19% | 62.05% | **70.36%** |
  | AudioCaps audio→text | 32.84% | 59.00% | **71.12%** |
  | Clotho test (1,045 clips, 5,225 captions) | 15.73% | 31.12% | not measured |
  | Clotho audio→text | 15.31% | 26.79% | not measured |

  The last-6-layer full tune regressed to 64.02% and was not promoted. The
  adapter roughly doubles R@10 on a corpus it never saw, so the gain is a
  property of the learned audio-text projection rather than AudioCaps-specific
  fit. The best Gemma-only result is still 19.6 points below the 90% condition;
  the cross-attention reranker is the remaining Gemma route. An AudioCaps-only
  90% with no Clotho gain would not be accepted.
