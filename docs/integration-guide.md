# Omura — Integration Guide (all features)

Single REST/JSON API. No SDK, no auth header (add a bearer slot for later). CORS is `*`.

**Base URL**
```
OMURA_API_BASE = http://100.117.12.3:19543          # v2 (Tailscale host "berryserver")
                 http://berryserver.tail9e5025.ts.net:19543
```
Health check: `GET /health` → `{"status":"ok"}`.

Capabilities: text→image, audio search, video search, **search-inside-a-video (seek-to-timestamp)**,
hardened reverse-image (exact-dup + provenance), and a blob proxy with HTTP Range for media playback.

---

## 0. Common result schema

Every search result is a JSON object. `score` is **0–100** relevance (higher = better); results are
pre-sorted descending.

| field | type | notes |
|---|---|---|
| `blob_id` | string | Walrus id. Quilt patches look like `"<quiltId>::<file>.ext"` (contains `::`). |
| `score` | number | 0–100 relevance — primary sort/display. |
| `kind` | string | `image` \| `audio` \| `video` \| `text` \| … |
| `mime_type`, `extension` | string | e.g. `audio/mpeg`/`mp3`, `video/mp4`/`mp4`. |
| `caption` | string\|null | VLM caption (image results). |
| `is_nsfw` | bool | blur/gate when true. |
| `is_quilt`, `parent_quilt_id`, `quilt_identifier` | — | quilt/collection membership. |
| `owner`, `expiresAt`, `size` | — | on-chain owner, expiry, bytes. |
| `distance`, `rerank_score`, `rerank_signals`, `nsfw_tag_score` | — | debug/ranking internals (optional). |

---

## 1. Text → image search  `POST /search`
SigLIP-2 `omura_emebd` (COCO R@10 94.6%) + BM25 hybrid FTS over captions.

```json
// request
{ "query": "a red sports car at sunset", "top_k": 24, "exclude_nsfw": true, "instruction": null }
// response
{ "total": 24, "results": [ {…result…}, … ] }
```
- `query` (required), `top_k` (default 10), `exclude_nsfw` (default true), `instruction` (optional hint).

## 2. Audio semantic search  `POST /search/audio`
CLAP (ESC-50 86.7%). Speech, music, environmental sounds ("dog barking", "rain", "techno beat").
Same body/response as `/search`. Results are `kind:"audio"`.

## 3. Video search  `POST /search/video`
`omura-embed-video` (finetuned InternVideo2-6B, MSR-VTT R@10 85.3%). Same body/response.
Results are `kind:"video"`, usually quilt patches.

> Audio & video search run a **liveness precheck** (drops 404'd blobs) → allow a **≥30 s** client
> timeout. Disable server-side with `OMURA_SEARCH_PRECHECK=0` if you prefer raw speed.

---

## 4. Search inside a video — Seek to Timestamp  `POST /search/video/in-video`
Find *where* a query appears within one video and seek the player there.

```json
// request
{ "blob_id": "<quiltId>::clip.mp4", "query": "a building", "top_k": 5,
  "win_sec": 4.0, "stride_sec": 2.0 }
// response
{ "blob_id": "...", "query": "a building", "duration": 5.06,
  "source": "precomputed|on_demand",
  "blob_url": "/blob/<url-encoded-blob_id>",
  "segments": [ { "start": 2.0, "end": 5.06, "score": 70.6, "cosine": 0.41 } ] }
```
- `segments` sorted by `score`; `start`/`end` in **seconds**. Render one marker + "Seek to mm:ss"
  per segment; on click set `videoEl.currentTime = segment.start`.
- `source`: `precomputed` (instant) or `on_demand` (~1–3 s).
- Typical flow: `/search/video` → user picks a result → call this with that `blob_id` → markers.

## 5. Hardened reverse-image  `POST /search/reverse-image`  (`multipart/form-data`)
Embedding retrieval + perceptual-hash duplicate confirmation + provenance.

Fields: `file` (required, ≤25 MiB), `top_k` (10), `exclude_nsfw` (true), `verify_duplicates` (true).
```json
{ "results": [ { …result…, "phash_hamming": 0, "duplicate_class": "exact_duplicate",
                 "is_exact_duplicate": true, "is_near_duplicate": true } ],
  "total": 10, "query_phash": "02000b2323250110",
  "duplicates_found": 1, "exact_duplicate_blob_id": "Kxztx…",
  "provenance": { "blob_id":"…", "owner":"0x…", "parent_quilt_id":"…",
                  "quilt_identifier":"…", "duplicate_class":"exact_duplicate", "phash_hamming":0 } }
```
- `duplicate_class`: `exact_duplicate` (ham 0) · `near_duplicate` (≤6) · `similar`.
- If `exact_duplicate_blob_id` set → show "Exact match found" + the `provenance` block (NFT mint trace).

---

## 6. Serving / playing media — blob proxy  `GET /blob/{blob_id}`
Streams bytes with correct `Content-Type` (from catalog), supports **HTTP Range (206)** so
`<audio>`/`<video>` scrub & seek work — including quilt-patch media.

```js
const url = `${OMURA_API_BASE}/blob/${encodeURIComponent(blob_id)}`; // blob_id may contain "::" → encode!
<img   src={url} />
<audio controls src={url} />
<video controls src={url} />                       // seeking works (Range honored)
<video controls src={`${url}#t=42`} />             // deep-link to 42s
```

---

## 7. Minimal React example
```tsx
const BASE = "http://100.117.12.3:19543";

async function search(modality: "all"|"audio"|"video", query: string, top_k = 24) {
  const path = modality === "audio" ? "/search/audio"
             : modality === "video" ? "/search/video" : "/search";
  const r = await fetch(BASE + path, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, top_k, exclude_nsfw: true }),
    signal: AbortSignal.timeout(35000),            // precheck needs headroom
  });
  return (await r.json()).results;
}

async function seekPoints(blob_id: string, query: string) {
  const r = await fetch(BASE + "/search/video/in-video", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ blob_id, query, top_k: 5 }),
  });
  return (await r.json()).segments;                // [{start,end,score}]
}

async function reverseImage(file: File) {
  const fd = new FormData(); fd.append("file", file); fd.append("top_k", "10");
  const r = await fetch(BASE + "/search/reverse-image", { method: "POST", body: fd });
  return r.json();                                 // {results, exact_duplicate_blob_id, provenance, …}
}

const blobUrl = (id: string) => `${BASE}/blob/${encodeURIComponent(id)}`;
```

```tsx
// Seek-to-timestamp player
function Player({ blob_id, segments }) {
  const ref = useRef<HTMLVideoElement>(null);
  const seek = (t: number) => { if (ref.current) { ref.current.currentTime = t; ref.current.play(); } };
  return (<>
    <video ref={ref} controls src={blobUrl(blob_id)} style={{ width: "100%" }} />
    {segments.map((s, i) => (
      <button key={i} onClick={() => seek(s.start)}>
        Seek to {Math.floor(s.start/60)}:{String(Math.floor(s.start%60)).padStart(2,"0")} ({Math.round(s.score)}%)
      </button>
    ))}
  </>);
}
```

---

## 8. curl quick-reference
```bash
B=http://100.117.12.3:19543
curl -s $B/health
curl -s -XPOST $B/search          -H 'Content-Type: application/json' -d '{"query":"red car","top_k":5}'
curl -s -XPOST $B/search/audio    -H 'Content-Type: application/json' -d '{"query":"rain","top_k":5}'
curl -s -XPOST $B/search/video    -H 'Content-Type: application/json' -d '{"query":"a city","top_k":5}'
curl -s -XPOST $B/search/video/in-video -H 'Content-Type: application/json' \
     -d '{"blob_id":"<quilt>::clip.mp4","query":"a building","top_k":5}'
curl -s -XPOST $B/search/reverse-image  -F file=@q.jpg -F top_k=5
curl -s "$B/blob/$(python -c 'import urllib.parse;print(urllib.parse.quote("<blob_id>",safe=""))')" -o out.bin
```

---

## 9. Gotchas
- **URL-encode `blob_id`** for `/blob` — quilt patches contain `::` and the filename.
- **Timeouts:** audio/video search ≥30 s (precheck); in-video on-demand ≤90 s for long clips.
- **NSFW:** keep `exclude_nsfw:true` default; when a result has `is_nsfw`, blur + click-to-reveal.
- **Scores** are already 0–100 — render as `Math.round(score)` or a 5-dot scale; don't re-normalize.
- **Debounce** the search box (~250 ms) and cancel in-flight requests.
- **Sidecar dependency:** `/search/video` and `/search/video/in-video` require the internal video
  service (`:19560`, localhost only — do NOT tunnel it). If it's down they 502/"model not ready".
```
```
