# Omura Search — Frontend Integration Prompt

> Hand this whole document to your frontend developer or a coding agent. It is written as a
> self-contained build prompt: it specifies the live API contract, the UI surfaces to build,
> and the Milestone-2 features (audio search, video search, reverse / NFT-provenance search,
> and Seek-to-Timestamp). Copy it verbatim into your frontend repo as the integration spec.

---

## Build prompt (paste this to the agent)

You are building the search frontend for **Omura**, a multimodal search engine over files stored
on the Walrus protocol. The backend is a FastAPI service. Build a responsive web UI (React +
TypeScript + fetch; no SDK required — it's plain REST/JSON) that exposes **four search modalities**
and renders results as a media grid with inline preview/playback. Follow the API contract below
exactly. Do not invent fields. Treat `score` as a 0–100 relevance number (higher = better) and sort
descending (the API already returns results sorted).

### Environment

```
OMURA_API_BASE = https://<host>:19543      # v2 staging instance (image+audio+video+reverse)
                                            # prod is :19353 with the same contract
```
All endpoints are CORS-friendly JSON. No auth header today (add a bearer token slot you can fill later).

---

## 1. Endpoints

### 1.1 Text → media search  `POST /search`
Searches the image/text index (SigLIP-2 `omura_emebd`, COCO R@10 = 94.6%).

Request (JSON):
```json
{ "query": "a red sports car at sunset", "top_k": 24, "exclude_nsfw": true, "instruction": null }
```
- `query` (string, required), `top_k` (int, default 10), `exclude_nsfw` (bool, default true),
  `instruction` (string|null, optional task hint for the embedding model).

Response (`SearchResponse`):
```json
{ "total": 24, "results": [ SearchResult, ... ] }
```

`SearchResult`:
| field | type | notes |
|---|---|---|
| `blob_id` | string | Walrus blob id. For quilt patches it's `"<quiltId>::<identifier>"`. |
| `score` | number | 0–100 relevance. Primary sort key (desc). |
| `distance` | number | raw cosine distance (debug only). |
| `size` | int | bytes (0 if unknown for patches). |
| `mime_type` | string | e.g. `image/png`, `audio/mpeg`, `video/mp4`. |
| `extension` | string | e.g. `png`, `mp3`, `mp4`. |
| `kind` | string | `image` \| `audio` \| `video` \| `text` \| ... |
| `is_nsfw` | bool | render a blur/▢ overlay when true. |
| `is_quilt` | bool | true if the blob lives inside a quilt container. |
| `parent_quilt_id` | string\|null | the container blob id. |
| `quilt_identifier` | string\|null | filename inside the quilt. |
| `owner` | string\|null | on-chain owner address. |
| `expiresAt` | any\|null | expiry epoch if known. |

### 1.2 Audio semantic search  `POST /search/audio`
Natural-language → audio retrieval (CLAP `larger_clap_general`, ESC-50 = 86.7%). Finds spoken
content, music, and **environmental sound clips** ("dog barking", "rain on a window", "techno beat").
Same request/response shape as `/search` (`SearchResponse`). Results are `kind: "audio"`.

### 1.3 Video search  `POST /search/video`
Natural-language → video retrieval (`omura-embed-video` = finetuned InternVideo2-6B, MSR-VTT R@10 = 85.3%).
Same request/response shape. Results are `kind: "video"`, usually quilt patches
(`blob_id = "<quiltId>::<clip>.mp4"`).

### 1.4 Reverse image / NFT-provenance / duplicate search  `POST /search/reverse-image`
Upload an image, get visually-similar indexed media. **Hardened** for **NFT provenance
verification** and **exact-duplicate detection**: embedding retrieval narrows to candidates, then
a perceptual hash (dHash) confirms which are true duplicates by Hamming distance — robust to
re-encode/format/scale, not fooled by merely-similar art.

Request: `multipart/form-data`
- `file` (binary, required) — the query image (≤ 25 MiB)
- `top_k` (int, default 10), `exclude_nsfw` (bool, default true), `instruction` (optional),
  `verify_duplicates` (bool, default true — runs the perceptual-hash confirmation)

Response (`ReverseImageResponse`):
```json
{ "results": [ { ...SearchResult,
                 "phash_hamming": 0, "duplicate_class": "exact_duplicate",
                 "is_exact_duplicate": true, "is_near_duplicate": true } ],
  "total": 10,
  "query_phash": "02000b2323250110",
  "duplicates_found": 1,
  "exact_duplicate_blob_id": "Kxztx...UIA",
  "provenance": { "blob_id": "Kxztx...UIA", "owner": "0x..", "parent_quilt_id": "..",
                  "quilt_identifier": "..", "duplicate_class": "exact_duplicate", "phash_hamming": 0 } }
```
- `duplicate_class` ∈ `exact_duplicate` (ham 0) · `near_duplicate` (ham ≤ 6) · `similar`.
- If `exact_duplicate_blob_id` is set, show "Exact match found" and render the `provenance`
  block (owner = original on-chain holder, `parent_quilt_id` = source collection) for mint tracing.
- `phash_hamming` is per-result; sort/badge by `duplicate_class`.

---

## 2. Rendering media — the blob proxy  `GET /blob/{blob_id}`
Stream/serve any blob's bytes with the correct `Content-Type`. Use it for thumbnails, `<img>`,
`<audio>`, and `<video>` sources.

```
<img  src="${OMURA_API_BASE}/blob/${encodeURIComponent(blob_id)}" />
<audio controls src="${OMURA_API_BASE}/blob/${encodeURIComponent(blob_id)}"></audio>
<video controls src="${OMURA_API_BASE}/blob/${encodeURIComponent(blob_id)}"></video>
```
> `blob_id` for quilt patches contains `::` and the filename — **URL-encode it**.

---

## 3. UI surfaces to build

1. **Unified search bar** with a modality toggle: `All / Image / Audio / Video`. Route to
   `/search`, `/search/audio`, or `/search/video` accordingly.
2. **Results grid**: images as thumbnails; audio as a card with a waveform/▶ and inline `<audio>`;
   video as a poster with ▶ that opens an inline `<video>` player. Show `score` as a relevance
   chip, `kind`/`extension` as a tag, and a blur overlay when `is_nsfw`.
3. **Reverse-image panel**: drag-and-drop / file picker → `POST /search/reverse-image`. Banner
   "Exact duplicate" when top `score ≥ 99`; show provenance (`owner`, `parent_quilt_id`).
4. **Detail drawer**: full media preview + metadata (blob_id, owner, size, quilt parent, expiry).
5. **NSFW toggle** wired to `exclude_nsfw` (default on).

---

## 4. Seek-to-Timestamp (temporal navigation) — Milestone 2

Goal: a user searches a phrase, and for a matching video we deep-link the player to the **moment**
the content appears, not just the file. The player must seek to `t` seconds and start playing there.

### 4.1 Player behavior (build now)
- Render `<video>` with `src = ${OMURA_API_BASE}/blob/${encodeURIComponent(blob_id)}#t=${start}`
  and also set `videoEl.currentTime = start` on `loadedmetadata` as a fallback.
- A result that carries a timestamp shows a **"Seek to 0:42"** button and a timeline marker.
- Support a deep-link route `/{blob_id}?t=42` that opens the player pre-seeked.

### 4.2 API contract — `POST /search/video/in-video`  (LIVE)
Search *inside* one video and get ranked timestamps. This is the seek-to-timestamp backend.

Request (JSON):
```json
{ "blob_id": "<quiltId>::clip.mp4", "query": "a fluffy puppy",
  "top_k": 5, "win_sec": 4.0, "stride_sec": 2.0 }
```
Response:
```json
{ "blob_id": "<quiltId>::clip.mp4", "query": "a fluffy puppy",
  "duration": 5.06, "source": "precomputed|on_demand",
  "blob_url": "/blob/<url-encoded-blob_id>",
  "segments": [ { "start": 2.0, "end": 5.06, "score": 70.6, "cosine": 0.41 } ] }
```
- `segments` are sorted by `score` (desc). `start`/`end` are **seconds**. Render one timeline
  marker + a "Seek to mm:ss" button per segment.
- `source` is `precomputed` (instant, catalog videos) or `on_demand` (computed live, ~1-3s).
- Use `blob_url` as the `<video>` src; on click, set `videoEl.currentTime = segment.start`
  (or load src `${blob_url}#t=${start}`).

Typical flow: user runs `/search/video` → picks a result → the detail view calls
`/search/video/in-video` with that `blob_id` + the same (or a refined) query → render markers →
click seeks the player.

### 4.3 Seeking — `/blob` now supports HTTP Range (LIVE)
`GET /blob/{blob_id}` advertises `Accept-Ranges: bytes` and answers `206 Partial Content` with
`Content-Range` when the browser sends a `Range` header — so native `<video>` scrubbing/seek and
`#t=` work for both plain blobs and quilt-patch videos (`::`). No extra work needed on the
frontend beyond a standard `<video controls>`.

---

## 5. Implementation notes
- All POST bodies are JSON except reverse-image (multipart). Always send `Content-Type: application/json`
  for the JSON endpoints.
- Debounce the search box (~250 ms); cancel in-flight requests on new input.
- `score` is already 0–100; render as `Math.round(score)%` or a 5-dot scale.
- Handle `kind` you don't recognize by rendering a generic file card with a download link to `/blob/...`.
- Empty `results` → friendly "no matches" state. Network/500 → retry affordance.
- NSFW: when `is_nsfw && !showNsfw`, blur the media and require a click-to-reveal.

---

## 6. Quick smoke tests (curl)
```bash
curl -s -XPOST $BASE/search       -H 'Content-Type: application/json' -d '{"query":"red car","top_k":5}'
curl -s -XPOST $BASE/search/audio -H 'Content-Type: application/json' -d '{"query":"dog barking","top_k":5}'
curl -s -XPOST $BASE/search/video -H 'Content-Type: application/json' -d '{"query":"person riding a bike","top_k":5}'
curl -s -XPOST $BASE/search/reverse-image -F file=@query.jpg -F top_k=5
curl -s "$BASE/blob/<url-encoded-blob_id>" -o out.bin
```
