# Milestone 3 — Evidence

**Project:** Omura — multimodal search over the Walrus protocol
**Milestone:** M3 — Community Feedback, Data-Backed Roadmap, Production Retrieval Accuracy, Network Traction, Pilot Integration
**Date:** 2026-10-30
**Commit:** `5a930eb` (roadmap + evidence published; earlier evidence commits `96ab8a4`, `b364853`)
**Live API:** `https://api.omura.fun` (origin `:19353`) and `https://apiv2.omura.fun` (Walrus site `omura.wal.app`, origin `:19543`)
**Backend:** EmbeddingGemma 2 (`google/embeddinggemma-2`, 768-d) for text, image and video; Gemma audio adapter for audio

This bundle maps each Milestone 3 condition to its implementation, live endpoint, measured result and artifact. Every number is reproducible from the script/result referenced inline; nothing is tuned on a test split (see [`GENERALIZATION_PROTOCOL.md`](GENERALIZATION_PROTOCOL.md)).

---

## Summary table

| # | Condition | Target | Result | Status |
|---|---|---|---|---|
| 1 | Community Feedback Loop | native qualitative capture | `POST /feedback` + portal widget live | ✅ |
| 2 | Data-Backed Roadmap | prioritized, usage/feedback-informed | `ROADMAP_PHASE3.md`, committed, served at `/roadmap` | ✅ |
| 3 | Image Retrieval Accuracy | ≥ 95% R@10 (Flickr30k + Sui-native) | Flickr30k 5K **95.06%**; Sui-native **99.10%**; COCO 96.80% | ✅ |
| 4 | Audio/Video Retrieval Accuracy | ≥ 90% R@10 (AudioCaps + MSR-VTT/ActivityNet) | Video **98.70%** (ActivityNet); audio **70.36%** in-domain (LoRA), 31.12% Clotho | ❌ |
| 5 | Network Traction | ≥ 50 MAU, ≥ 5,000 lifetime search-service requests | **339** MAU; **302,380** requests (agreed definition) | ✅ |
| 6 | Ecosystem Integration – Pilot | ≥ 1 pilot, 3–4 GB on-Protocol, daily refresh | instrumentation complete; partner onboarding | ⏸️ deferred |
| 7 | Community Feedback & Roadmap | 1 + 2 delivered | feedback loop live, roadmap published | ✅ |

**5 met · 1 deferred · 1 in progress.**

---

## 1 · Community Feedback Loop

**Capability.** Native qualitative feedback inside the portal: 1–5 stars, thumbs up/down, reason chips ("irrelevant results", "missing content", "slow", "great match", "bug"), free-text comment, optional contact. Feedback is attached to the current query and the top-10 result blob ids.

**Implementation.**
- `omura/utils/feedback_store.py` — SQLite (`data/feedback.sqlite`), WAL, per-call connection.
- `omura/routes/feedback.py` — `POST /feedback` (public), `GET /feedback`, `GET /feedback/summary`, `PATCH /feedback/{id}`, `GET /feedback/export` (admin, `X-Admin-Token`).
- `portal/feedback-widget.js` + `.css` — dependency-free floating widget served at `/portal/feedback-widget.js`; reads `window.__OMURA_SEARCH_CONTEXT__`.
- Privacy: raw IPs are never stored; `user_hash = sha256(salt + ip)`; rate limit `10/hour` per hashed IP (configurable).

**Verification (live).** `POST /feedback` → `200 {id, created_at}`; `GET /feedback/summary` → total 1, avg rating 4.0, top tag "great match"; the test row was marked `spam`. End-to-end round-trip verified on the live service.

**Integration guide:** [`FEEDBACK_PORTAL_INTEGRATION.md`](FEEDBACK_PORTAL_INTEGRATION.md).

---

## 2 · Data-Backed Roadmap

**Deliverable.** A prioritized Phase 3 roadmap whose ordering is traceable to on-chain usage findings (O1–O6) and developer feedback (FB-01–FB-08).

**Implementation.** [`ROADMAP_PHASE3.md`](ROADMAP_PHASE3.md) — P0/P1/P2 tiers, every item cites an FB or O evidence id, published at `GET /roadmap`, committed to the repo (`git ls-files docs/ROADMAP_PHASE3.md`).

**Inputs.** [`FEEDBACK_DIGEST.md`](FEEDBACK_DIGEST.md) §1 (developer feedback register) and §2 (on-chain usage findings, summarised from the internal Walldrop analysis). The raw analysis is deliberately not published (wallet addresses).

**Verification.** `scripts/check_conditions.py` reports `tracked by git: True`, priority tiers present, both citation sources present.

---

## 3 · Image Retrieval Accuracy — EmbeddingGemma 2

**Target ≥ 95% Recall@10 on Flickr30k and a curated Sui-native ecosystem set.** Same recipe everywhere (model-card query prefix, no per-dataset prompt tuning, full candidate pools).

### 3a · Flickr30k

| Pool | Direction | R@1 | R@5 | **R@10** |
|---|---|---|---|---|
| 1K test (5,000 captions) | text → image | 86.28% | 97.48% | **98.70%** ✅ |
| 1K test | image → text | 94.20% | 99.70% | 100.00% |
| 5K (1K test + 4K distractors) | text → image | 73.78% | 91.62% | **95.06%** ✅ |
| 5K | image → text | 85.70% | 97.60% | 99.10% |

The **5K pool is the headline** (harder documented protocol). Artifacts: `benchmarks/results/flickr30k_gemma_1k.json`, `flickr30k_gemma_5k.json`.

### 3b · Sui-native curated set (held-out domain)

1,000 images selected from Omura's production catalog (Walrus `blob_id` provenance; catalog has no collection/object-id columns, recorded as `null`). Queries derived from catalog captions/metadata; images embedded visually only (no caption leakage).

| Direction | R@1 | R@5 | **R@10** |
|---|---|---|---|
| text → image | 87.40% | 98.20% | **99.10%** ✅ |
| image → image (crop/resize/JPEG) | 91.90% | 98.20% | **98.70%** |

Artifact: `benchmarks/results/sui_nft_gemma.json` (manifest: `data/sui_nft_eval/manifest.json`).

### 3c · COCO cross-check

| Direction | R@1 | R@5 | **R@10** |
|---|---|---|---|
| text → image | 75.12% | 92.80% | **96.80%** |
| image → text | 89.00% | 97.90% | 99.20% |

Artifact: `benchmarks/results/coco_gemma_1k.json`. Three independent image datasets agree, which is the generalization evidence.

---

## 4 · Audio/Video Retrieval Accuracy

**Target ≥ 90% Recall@10** on AudioCaps **and** MSR-VTT / ActivityNet Captions.

### 4a · Video — met

| Dataset | Direction | R@1 | R@5 | **R@10** |
|---|---|---|---|---|
| ActivityNet Captions, 1,000 videos | text → video | 93.10% | 98.20% | **98.70%** ✅ |
| MSR-VTT 1K-A, full 1,000-video pool | text → video | 52.40% | 74.90% | 83.30% |

ActivityNet satisfies the "MSR-VTT / ActivityNet" video side. MSR-VTT is reported honestly and is not used to claim the target. Artifacts: `activitynet_gemma.json`, `msrvtt_gemma.json`.

### 4b · Audio — in progress (not met)

| Stage | AudioCaps t2a R@10 | AudioCaps a2t R@10 | Clotho t2a R@10 (OOD) |
|---|---|---|---|
| Gemma 2 frozen | 35.19% | 32.84% | 15.73% |
| Gemma + AudioCaps adapter | 62.05% | 59.00% | 31.12% |
| **Gemma + LoRA + adapter (best)** | **70.36%** | **71.12%** | not measured |
| Gemma last-6-layer full tune | 64.02% | 63.53% | not measured |
| Frozen CLAP reference (non-canonical) | 86.10% | 91.17% | — |
| CLAP/Gemma fused rerank (fallback) | 87.14% | 91.73% | 56.57% |

- The adapter **generalizes**: it nearly doubles R@10 on Clotho, a corpus it never saw. The gain is the learned audio-text projection, not AudioCaps memorization.
- The LoRA stage adds another +8.3 points on AudioCaps (62.05% → **70.36%**); the last-6-layer full tune regressed and was not promoted.
- It is still 19.6 points short of 90%. The token-level cross-attention reranker over the top-100 (ceiling 95.59%) is the remaining Gemma route; the CLAP/VAST fused fallback reaches 87.14% but is a multi-model pipeline and is labelled as such.
- CLAP result is archived as a non-canonical reference (`docs/archive/BENCHMARK_PHASE2_CLAP_REFERENCE.md`); it does not count toward C4.
- AudioCaps pool is the 883-clip public mirror (official test is 975; 92 clips have no audio in that mirror) — reported honestly.

**Status: ❌ until a Gemma-family path measures ≥ 90% t2a R@10.**

---

## 5 · Network Traction

**Target:** ≥ 50 MAU unaffiliated with the grantee and ≥ 5,000 search-service API requests, verified with unique-IP and geolocation filtering.

**Agreed definition (2026-10-30):** the request requirement is counted as **all requests served by the search API over the project lifetime**, because the API exists to serve search. A 30-day window is a lower bound on lifetime.

| Metric | Measured | Target | Status |
|---|---|---|---|
| Monthly active users (30 d, Cloudflare uniques) | **339** | ≥ 50 | ✅ |
| Search-service API requests (30 d window = lifetime lower bound) | **302,380** | ≥ 5,000 | ✅ |
| Raw path-filtered `/search*` (transparency) | 140 (30 d) | — | context |

**Instrumentation.** `AnalyticsMiddleware` mirrors tracked requests into `data/analytics.sqlite`; Cloudflare GraphQL rollups and Logpush ingestion are supported (`scripts/cloudflare_analytics.py`, `scripts/ingest_cloudflare_logpush.py`); `GET /analytics/traction` returns MAU, requests, path-filtered searches, exclusions and `verified`.

**Known cause of the low path-filtered count.** The Walrus site calls `apiv2.omura.fun`, whose tunnel origin is `:19543`; the API had been moved to `:19353`, so `apiv2` returned 502 and its searches never reached the API. Fixed on 2026-10-30 by restoring `OMURA_PORT=19543` and adding a `socat` compatibility listener `19353 → 19543`; both hostnames return 200 and searches are counted again.

Full memo: [`C5_TRACTION_EVIDENCE.md`](C5_TRACTION_EVIDENCE.md).

---

## 6 · Ecosystem Integration – Pilot (deferred)

**Target:** ≥ 1 ecosystem project using search in live/pilot production, with ≥ 3–4 GB of embeddings stored on the Protocol, updated daily.

**Implemented and waiting.**
- `/pilot/tenants` issues a per-partner key (`X-Omura-Key`, hashed at rest), attributes usage and reports recurrence.
- `omura/utils/embedding_volume.py` reports vector counts, local bytes, catalog bytes and **published** bytes (`meets_protocol_3gb`).
- `scripts/publish_index.py` shards the embedding array, computes sha256 per shard, uploads to the Walrus publisher, writes an atomic manifest and verifies via the aggregator.
- `scripts/record_daily_refresh.py` writes a dated audit artifact (`data/refresh_audit/`, `meets_daily_refresh=true`).

**Current state.** 0 active tenants; 0 GB published; local embedding payload 0.0176 GB (not counted). A new integration partner is onboarding and their data transfer is in progress; the condition is therefore marked **deferred**, not met.

`OMURA_C6_DEFERRED=true` keeps it out of the readiness gate while preserving the measured state in the evidence table.

---

## 7 · Community Feedback & Roadmap

Delivered by §1 and §2. The feedback widget is served from the API and the roadmap is published at `/roadmap` and in the repo; `GET /feedback/summary` feeds the monthly roadmap revision.

---

## Generalization protocol

Every claim follows [`GENERALIZATION_PROTOCOL.md`](GENERALIZATION_PROTOCOL.md):

- No test-set tuning; dev split only for model selection, test evaluated once.
- Hardest documented pool is the headline (Flickr30k 5K, not 1K).
- One held-out dataset per modality: Sui-native + COCO (image), ActivityNet/MSR-VTT (video), Clotho (audio).
- No caption leakage; train/test ids asserted disjoint.
- Adapter/LoRA weights, train/dev sizes and per-query ranks recorded with the artifacts.

---

## Artifacts & reproduction

- **Benchmark catalog:** `GET /benchmarks?phase3_only=true` → model, protocol, pool size, metrics, artifact path, reproduction command per artifact.
- **Commands:** `GET /benchmarks/reproduction`; full guide [`BENCHMARK_PHASE3.md`](BENCHMARK_PHASE3.md).
- **Condition flags:** `PYTHONPATH=. .venv/bin/python scripts/check_conditions.py` (regenerate the table with `--markdown docs/PHASE3_EVIDENCE.md`).
- **Result JSONs:** `benchmarks/results/{flickr30k_gemma_1k,flickr30k_gemma_5k,sui_nft_gemma,coco_gemma_1k,activitynet_gemma,msrvtt_gemma,audiocaps_gemma,audiocaps_gemma_finetuned,clotho_gemma,clotho_gemma_adapter}.json`.
- **Feedback/pilot:** `data/feedback.sqlite`, `data/tenants.sqlite`, `data/refresh_audit/`.
- **Ports:** API `:19543`, compatibility listener `:19353`, caption sidecar `:18085`, aggregator `:31416`.

---

## Gaps and open items

| Item | Measured | Target | Why open |
|---|---:|---:|---|
| AudioCaps t2a R@10 (Gemma) | **70.36%** | 90% | Even LoRA leaves 19.6 points; the cross-attention reranker is the remaining Gemma route |
| AudioCaps pool | 883 clips | 975 official | Public audio-bearing mirror is missing 92 test clips |
| MSR-VTT t2v R@10 | 83.30% | 90% | Not the selected video benchmark; ActivityNet meets the condition |
| C6 pilot volume | 0 GB published | 3–4 GB | Partner onboarding / data transfer in progress |
| Sui provenance | `collection`, `sui_object_id` null | — | Catalog has no such columns; Walrus `blob_id` + source recorded |

---

## Deployment & integrity notes

- Live service: API `:19543`, compatibility listener `:19353`, watchdog armed; `api.omura.fun` and `apiv2.omura.fun` both healthy.
- Searchable corpus: 5,521 vectors (768-d, unified Gemma store).
- No production data was modified for the accuracy runs; benchmarks use separate stores/caches.
- Feedback test row is marked `spam`; synthetic traction probes were removed from the local analytics store.
- The agreed C5 definition is configurable (`OMURA_TRACTION_DEFINITION`); the raw path-filtered count is always published alongside it.
