"""Minimal HTTP microservice exposing the finetuned 'omura embed video' model.

The main Omura app (transformers 5.x venv) can't load InternVideo2 (needs the repo +
transformers 4.28), so it calls this service (running in .venv-iv2, model in memory) for:
  - text embedding (/embed_text)               -> /search/video query embedding
  - in-video temporal localization (/search_in_video) -> seek-to-timestamp

  IV2_CKPT=<ckpt> OMURA_VIDEO_HEADS=<best_heads.pt> CUDA_VISIBLE_DEVICES=7 \
      .venv-iv2/bin/python scripts/iv2_video_service.py --port 19560

Endpoints:
  GET  /health                       -> {"ok": true, "dim": 768, "temporal_blobs": N}
  POST /embed_text {"text": "..."}    -> {"vec": [...768...]}
  POST /search_in_video {"blob_id": "...", "query": "...", "top_k": 5,
                         "win_sec": 4.0, "stride_sec": 2.0}
       -> {"blob_id","duration","source":"precomputed|on_demand","segments":[{start,end,score}]}
"""
import os, sys, json, argparse, tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import requests

sys.path.insert(0, os.path.dirname(__file__))
import iv2_finetuned as M
import iv2_temporal as T

# Reuse the main app's smart-routing aggregator pool (12 nodes, health-aware, short
# per-attempt timeouts) instead of a hardcoded 2-URL sequential fetch — the omura.utils
# package has no heavy deps (just `requests`), so it's safe to import cross-venv.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from omura.utils.aggregator_pool import get_pool  # noqa: E402

MAX_BYTES = int(os.getenv("OMURA_MAX_VIDEO_BYTES", str(80 * 1024 * 1024)))
TEMPORAL_NPZ = os.getenv("OMURA_VIDEO_TEMPORAL_NPZ",
                         os.path.join(os.path.dirname(__file__), "..", "data", "cache", "video_temporal.npz"))
BLOB_CACHE_DIR = os.getenv("OMURA_BLOB_CACHE_DIR", os.path.join(_REPO_ROOT, "data", "blob_cache"))

# Precomputed segment index: blob_id -> {"feats": [n,768], "meta": [{start,end}...]}
_PRECOMP = {}


def _load_precomputed():
    if not os.path.exists(TEMPORAL_NPZ):
        print(f"[iv2-service] no precomputed temporal index at {TEMPORAL_NPZ}", flush=True)
        return
    d = np.load(TEMPORAL_NPZ, allow_pickle=True)
    sb, ss, se, sf = d["seg_blob"], d["seg_start"], d["seg_end"], d["seg_feat"]
    by = {}
    for i in range(len(sb)):
        bid = str(sb[i])
        by.setdefault(bid, []).append(i)
    for bid, idxs in by.items():
        _PRECOMP[bid] = {
            "feats": sf[idxs].astype(np.float32),
            "meta": [{"start": float(ss[j]), "end": float(se[j])} for j in idxs],
        }
    print(f"[iv2-service] loaded precomputed temporal index: {len(_PRECOMP)} videos, {len(sb)} segments", flush=True)


def _fetch_video_bytes(blob_id):
    """Fetch a video blob (plain or quilt-patch '<quilt>::<identifier>') to bytes.
    Disk-cached (data/blob_cache) — video content is immutable once fetched, so repeated
    in-video searches on the same clip skip the network entirely after the first hit."""
    if "::" in blob_id:
        quilt, ident = blob_id.split("::", 1)
        safe = requests.utils.quote(ident, safe="")
        path = f"/v1/blobs/by-quilt-id/{quilt}/{safe}"
    else:
        path = f"/v1/blobs/{blob_id}"
    # (connect_timeout, read_timeout): fail over fast from dead/unreachable upstreams,
    # but tolerate slow-but-alive transfers once bytes are flowing (per-chunk, not total).
    data, _used_url = get_pool().get_blob_cached(path, BLOB_CACHE_DIR, timeout=(5, 60))
    if data is None or len(data) > MAX_BYTES:
        return None
    return data


def _search_in_video(blob_id, query, top_k=5, win_sec=4.0, stride_sec=2.0):
    # Fast path: precomputed segments for this blob.
    if blob_id in _PRECOMP:
        pc = _PRECOMP[blob_id]
        res = T.score_segments(pc["feats"], query=query, segments_meta=pc["meta"], top_k=top_k)
        return {"blob_id": blob_id, "duration": pc["meta"][-1]["end"] if pc["meta"] else None,
                "source": "precomputed", "segments": res["segments"]}
    # On-demand: fetch + decode + slide + embed.
    data = _fetch_video_bytes(blob_id)
    if data is None:
        return {"blob_id": blob_id, "error": "fetch_failed", "segments": []}
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.write(data); tmp.close()
    try:
        res = T.localize_file(tmp.name, query, win_sec=win_sec, stride_sec=stride_sec, top_k=top_k)
    finally:
        os.unlink(tmp.name)
    return {"blob_id": blob_id, "duration": res.get("duration"),
            "source": "on_demand", "segments": res.get("segments", [])}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self):
        if self.path == "/health":
            self._send(200, {"ok": True, "dim": 768, "temporal_blobs": len(_PRECOMP)})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        try:
            if self.path == "/embed_text":
                text = (self._body().get("text") or "").strip()
                if not text:
                    self._send(400, {"error": "text required"}); return
                vec = M.embed_text([text])[0].astype("float32").tolist()
                self._send(200, {"vec": vec})
            elif self.path == "/search_in_video":
                p = self._body()
                blob_id = (p.get("blob_id") or "").strip()
                query = (p.get("query") or "").strip()
                if not blob_id or not query:
                    self._send(400, {"error": "blob_id and query required"}); return
                out = _search_in_video(
                    blob_id, query,
                    top_k=int(p.get("top_k", 5)),
                    win_sec=float(p.get("win_sec", 4.0)),
                    stride_sec=float(p.get("stride_sec", 2.0)),
                )
                self._send(200, out)
            else:
                self._send(404, {"error": "not found"})
        except Exception as e:  # noqa: BLE001
            self._send(500, {"error": str(e)})

    def log_message(self, *a):  # quiet
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=int(os.getenv("IV2_SERVICE_PORT", "19560")))
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    print("[iv2-service] loading finetuned model…", flush=True)
    M.load()
    _load_precomputed()
    get_pool().startup_ping()
    print(f"[iv2-service] ready on {args.host}:{args.port}", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
