"""Moondream2 image-captioning microservice.

Omura's main app runs transformers 5.5 (for omura_emebd), where Moondream's remote code
produces garbage. So Moondream runs here in its own venv (.venv-caption, transformers 4.52)
and the app captions images by POSTing bytes to this service — same sidecar pattern as the
video model. Replaces the weak BLIP captioner (which mislabeled the blue Walrus mascot as
"a cat").

Run (GPU with ~5GB free):
  CUDA_VISIBLE_DEVICES=6 .venv-caption/bin/python scripts/caption_service.py --port 18081

Endpoints:
  GET  /health                      -> {"ok": true, "model": "moondream2"}
  POST /caption  (raw image bytes)  -> {"caption": "..."}
     optional query ?length=short|normal  (default normal)
  POST /query  (raw image bytes)    -> {"answer": "..."}
     required query ?question=...  (VQA; used by the NSFW labeler)
"""
import os, sys, json, argparse, threading
from io import BytesIO
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import torch
from PIL import Image
from transformers import AutoModelForCausalLM

MODEL_ID = os.getenv("OMURA_MOONDREAM_MODEL", "vikhyatk/moondream2")
REVISION = os.getenv("OMURA_MOONDREAM_REVISION", "2025-06-21")
DEFAULT_LENGTH = os.getenv("OMURA_CAPTION_LENGTH", "normal")
MAX_DIM = int(os.getenv("OMURA_CAPTION_MAX_DIM", "1024"))

_model = None
# ThreadingHTTPServer runs each request in its own thread, but the model instance
# is shared and its .caption() call is not thread-safe (concurrent calls corrupt
# each other's generation state, producing captions from a different request).
# Serialize inference; concurrency at the HTTP layer still overlaps image decode/resize.
_infer_lock = threading.Lock()


def load():
    global _model
    if _model is None:
        print(f"[caption-service] loading {MODEL_ID}@{REVISION}…", flush=True)
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, revision=REVISION, trust_remote_code=True, torch_dtype=torch.float16
        ).to("cuda").eval()
        print("[caption-service] ready", flush=True)
    return _model


def caption_bytes(image_bytes: bytes, length: str = DEFAULT_LENGTH) -> str:
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    if max(img.size) > MAX_DIM:
        s = MAX_DIM / max(img.size)
        img = img.resize((int(img.width * s), int(img.height * s)))
    if length not in ("short", "normal"):
        length = "normal"
    with _infer_lock:
        out = load().caption(img, length=length)["caption"]
    return " ".join((out or "").split())


def query_bytes(image_bytes: bytes, question: str) -> str:
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    if max(img.size) > MAX_DIM:
        s = MAX_DIM / max(img.size)
        img = img.resize((int(img.width * s), int(img.height * s)))
    with _infer_lock:
        out = load().query(img, question)["answer"]
    return " ".join((out or "").split())


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if urlparse(self.path).path == "/health":
            self._send(200, {"ok": True, "model": "moondream2", "revision": REVISION})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        u = urlparse(self.path)
        if u.path not in ("/caption", "/query"):
            self._send(404, {"error": "not found"}); return
        try:
            n = int(self.headers.get("Content-Length", 0))
            data = self.rfile.read(n)
            if not data:
                self._send(400, {"error": "empty body"}); return
            if u.path == "/caption":
                length = (parse_qs(u.query).get("length", [DEFAULT_LENGTH])[0])
                self._send(200, {"caption": caption_bytes(data, length)})
            else:
                question = (parse_qs(u.query).get("question", [""])[0])
                if not question:
                    self._send(400, {"error": "missing question"}); return
                self._send(200, {"answer": query_bytes(data, question)})
        except Exception as e:  # noqa: BLE001
            self._send(500, {"error": str(e)})

    def log_message(self, *a):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=int(os.getenv("OMURA_CAPTION_PORT", "18081")))
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    load()
    print(f"[caption-service] serving on {args.host}:{args.port}", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
