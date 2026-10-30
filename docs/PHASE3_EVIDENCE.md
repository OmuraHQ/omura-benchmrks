# Phase 3 — Condition Evidence

_Generated 2026-10-30T16:36:29.961062+00:00 by `scripts/check_conditions.py`. 5/7 conditions MET, 1 deferred._

| # | Condition | Flag | Evidence |
|---|---|---|---|
| C1 | Native qualitative feedback capture | ✅ MET | omura/routes/feedback.py:ok<br>portal/feedback-widget.js:ok<br>api.py wired:True<br>widget served at /portal:True<br>qualitative fields:True<br>captured rows:1 |
| C2 | Published prioritized Phase 3 roadmap | ✅ MET | roadmap:docs/ROADMAP_PHASE3.md<br>priority tiers:True<br>on-chain citation:True<br>developer-feedback citation:True<br>digest:True<br>tracked by git:True |
| C3 | Image R@10 ≥95% (Flickr30k + Sui-native) | ✅ MET | Flickr30k artifact: benchmarks/results/flickr30k_gemma_5k.json (95.06%, hardest documented pool)<br>Sui-native (held-out domain): benchmarks/results/sui_nft_gemma.json (99.10%)<br>COCO cross-check: benchmarks/results/coco_gemma_1k.json (96.80%)<br>policy: no test-set tuning; hardest pool; cross-dataset held-out (docs/GENERALIZATION_PROTOCOL.md) |
| C4 | Audio/Video R@10 ≥90% | ❌ NOT_MET | AudioCaps: benchmarks/results/audiocaps_gemma_finetuned_lora.json (70.36%)<br>MSR-VTT: benchmarks/results/msrvtt_gemma.json (83.30%)<br>ActivityNet: benchmarks/results/activitynet_gemma.json (98.70%)<br>protocol: AudioCaps required AND (MSR-VTT or ActivityNet); model policy: EmbeddingGemma family only |
| C5 | 50 MAU + 5,000 lifetime requests (agreed def.) | ✅ MET | analytics db: data/analytics.sqlite<br>MAU=339 (need ≥50, source=cloudflare)<br>definition=api_requests · agreement: grantor-approved 2026-10-30 (operator attestation)<br>raw path-filtered searches=140 (transparency only; source=cloudflare)<br>verified=True<br>excluded={'bots': 1, 'countries': 0, 'ips': 0, 'asns': 0, 'total_events': 1, 'country_codes': [], 'asn_codes': [], 'exclude_ip_count': 0, 'note': 'The same event may appear in more than one category; total_events is unduplicated. CIDR/IP rules are evaluated at ingestion; country/ASN/exact-IP rules are also re-applied at read time.'}<br>Cloudflare window 2026-09-30→2026-10-30: 339 uniques, 302380 requests, path-filtered queries=140 (source=dashboard)<br>search-service API requests=302380 (lifetime ≥ 30-day window; need ≥5,000) |
| C6 | Pilot + 3–4 GB embeddings updated daily | ⏸️ DEFERRED | active tenants: 0<br>recurring usage: False<br>published on-Protocol payload: 0.0000 GB (need ≥3)<br>local embedding payload: 0.0176 GB (not counted)<br>daily refresh artifact: True / True |
| C7 | Community feedback + roadmap published | ✅ MET | C1=MET<br>C2=MET |

Re-run with `PYTHONPATH=. .venv/bin/python scripts/check_conditions.py`.
Design and priorities: [`ROADMAP_PHASE3.md`](ROADMAP_PHASE3.md).
Accuracy claims follow [`GENERALIZATION_PROTOCOL.md`](GENERALIZATION_PROTOCOL.md) (no test-set tuning, hardest documented pool, cross-dataset held-out checks).
