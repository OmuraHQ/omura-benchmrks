"""Image captioning for the Omura index.

Default backend is a smart vision-language model (Gemma 4 26B-A4B, a reasoning VLM)
served by a persistent llama.cpp `llama-server` with its `mmproj` vision projector.
It is prompted with the Walrus/NFT context so captions are accurate and aware that the
corpus is full of NFTs, PFP avatars, generative/pixel art, memes and collectibles — not
just generic photos. The model reasons internally (returned in `reasoning_content`); we
keep only the final `content` caption. The legacy BLIP-base captioner is the fallback
used when the server is unreachable.

Run the server (once), e.g. on GPU 7:
  CUDA_VISIBLE_DEVICES=7 llama-server \\
    -m gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf --mmproj mmproj-BF16.gguf \\
    -ngl 99 --jinja --host 127.0.0.1 --port 18080 -c 8192

Backends (env OMURA_CAPTION_BACKEND):
  "moondream"     (default) — Moondream2 sidecar (scripts/caption_service.py, .venv-caption).
                  Accurate + lightweight (1.8B); replaces the weak BLIP captioner that
                  mislabeled the blue Walrus mascot as "a cat".
  "llama_server"  — Gemma-4-26B VLM via llama.cpp /v1/chat/completions (heavier).

Config (env):
  OMURA_CAPTION_SERVER_URL  moondream: http://127.0.0.1:18081/caption (default)
                            llama_server: http://127.0.0.1:18080/v1/chat/completions
  OMURA_CAPTION_MAX_TOKENS  llama_server generation budget incl. reasoning (default 1600)
  OMURA_CAPTION_TIMEOUT     per-request seconds (default 180)
  OMURA_CAPTION_PROMPT      override the llama_server instruction prompt

There is intentionally NO BLIP fallback: a wrong caption ("a cat" for a walrus) is worse
than none. If the caption backend is unreachable, generate_caption returns "".
"""

import base64
import json
import os
import threading
import urllib.error
import urllib.request
from io import BytesIO

from PIL import Image

_CAPTION_BACKEND = os.getenv("OMURA_CAPTION_BACKEND", "moondream").strip().lower()
_DEFAULT_SERVER = (
    "http://127.0.0.1:18081/caption" if _CAPTION_BACKEND == "moondream"
    else "http://127.0.0.1:18080/v1/chat/completions"
)
_CAPTION_SERVER_URL = os.getenv("OMURA_CAPTION_SERVER_URL", _DEFAULT_SERVER).strip()
_CAPTION_MAX_TOKENS = int(os.getenv("OMURA_CAPTION_MAX_TOKENS", "1600"))
_CAPTION_TIMEOUT = float(os.getenv("OMURA_CAPTION_TIMEOUT", "180"))

# Short prompt on purpose: a verbose prompt makes the reasoning model over-analyze (and
# can make it mistake the prompt text for on-image text). The NFT/Walrus context is kept
# minimal so it reasons briefly and returns a literal one-line caption.
_DEFAULT_PROMPT = (
    "Caption this image in one short, literal sentence: the main subject and the art "
    "style (it may be an NFT, avatar, pixel/generative art, meme, logo, or photo). "
    "If it is a cat, say cat."
)
_CAPTION_PROMPT = os.getenv("OMURA_CAPTION_PROMPT", _DEFAULT_PROMPT)

_server_failed = False

_CAPTION_MAX_DIM = int(os.getenv("OMURA_CAPTION_MAX_DIM", "1024"))


def _caption_moondream(image_bytes: bytes) -> str:
    """POST raw image bytes to the Moondream sidecar -> caption string."""
    jpeg = _to_jpeg(image_bytes)
    req = urllib.request.Request(
        _CAPTION_SERVER_URL, data=jpeg,
        headers={"Content-Type": "application/octet-stream"},
    )
    with urllib.request.urlopen(req, timeout=_CAPTION_TIMEOUT) as resp:
        out = json.load(resp)
    return " ".join((out.get("caption") or "").split())


def _to_jpeg(image_bytes: bytes) -> bytes:
    """Normalize any input to a clean RGB JPEG.

    llama.cpp's mtmd image loader rejects some Walrus formats ("failed to decode image
    bytes" → HTTP 400). Re-encoding through PIL fixes that and bounds the size so image
    tokens / latency stay reasonable.
    """
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    if max(img.size) > _CAPTION_MAX_DIM:
        scale = _CAPTION_MAX_DIM / max(img.size)
        img = img.resize((int(img.width * scale), int(img.height * scale)))
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def _caption_server(image_bytes: bytes) -> str:
    jpeg = _to_jpeg(image_bytes)
    b64 = base64.b64encode(jpeg).decode()
    data_url = f"data:image/jpeg;base64,{b64}"
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": _CAPTION_PROMPT},
                ],
            }
        ],
        "max_tokens": _CAPTION_MAX_TOKENS,
        "temperature": 0.0,
        "stream": False,
    }
    req = urllib.request.Request(
        _CAPTION_SERVER_URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=_CAPTION_TIMEOUT) as resp:
        out = json.load(resp)
    msg = out["choices"][0]["message"]
    # The reasoning VLM returns its thinking in `reasoning_content`; the final caption is
    # in `content`. Fall back to channel-marker parsing if a build inlines both.
    caption = (msg.get("content") or "").strip()
    if not caption:
        raw = msg.get("reasoning_content") or ""
        if "<channel|>" in raw:
            caption = raw.rsplit("<channel|>", 1)[1].strip()
    return " ".join(caption.split())


def generate_caption(image_bytes: bytes) -> str:
    """Caption an image via the configured backend. Returns "" on failure — there is NO
    BLIP fallback (a confidently-wrong caption is worse than an empty one)."""
    if not image_bytes:
        return ""
    backend = _caption_moondream if _CAPTION_BACKEND == "moondream" else _caption_server
    try:
        return backend(image_bytes)
    except urllib.error.HTTPError as e:
        print(f"[Captioning] caption server rejected one image (HTTP {e.code}); skipping.")
        return ""
    except urllib.error.URLError as e:
        print(f"[Captioning] caption server unreachable ({e}); skipping (no fallback).")
        return ""
    except Exception as e:  # noqa: BLE001
        print(f"[Captioning] caption error for one image ({e}); skipping.")
        return ""
