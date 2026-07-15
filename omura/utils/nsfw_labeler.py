"""VLM-based NSFW labeler — the safety counterpart to captioning.py.

Uses the same persistent Moondream2 sidecar (scripts/caption_service.py, VQA via its
`/query` endpoint) to actually *look* at the image and classify its safety, instead of
the crude image-embedding cosine to NSFW text prototypes. Returns a graded 0-100 score
on the same scale the rest of the pipeline already uses (`nsfw_score`, `is_nsfw`), so
it's a drop-in upgrade.

Previously this targeted a standalone Gemma-4-VL `llama-server` (OpenAI chat-completions
protocol) that no sidecar ever actually started — every call 404/500'd and silently fell
back to the embedding-cosine method for every single image. Moondream's `/query` endpoint
(same already-loaded model as captioning, zero extra GPU memory) replaces that.

Config (env):
  OMURA_NSFW_BACKEND        "vlm" (default) | "embedding"  (embedding = legacy cosine)
  OMURA_NSFW_SERVER_URL     default: derived from OMURA_CAPTION_SERVER_URL's host/port
                            (Moondream sidecar), e.g. http://127.0.0.1:18085/query
  OMURA_NSFW_THRESHOLD      score ≥ this ⇒ is_nsfw (default 80)
"""

import json
import os
import urllib.error
import urllib.parse
import urllib.request

# Reuse the captioner's image normalization (fixes llama.cpp "failed to decode" 400s).
from omura.utils.captioning import _to_jpeg

NSFW_BACKEND = os.getenv("OMURA_NSFW_BACKEND", "vlm").strip().lower()
_CAPTION_URL = os.getenv("OMURA_CAPTION_SERVER_URL", "http://127.0.0.1:18085/caption").strip()
_DEFAULT_NSFW_URL = _CAPTION_URL.rsplit("/caption", 1)[0] + "/query"
_SERVER_URL = os.getenv("OMURA_NSFW_SERVER_URL", _DEFAULT_NSFW_URL).strip()
_TIMEOUT = float(os.getenv("OMURA_NSFW_TIMEOUT", "60"))
NSFW_THRESHOLD = float(os.getenv("OMURA_NSFW_THRESHOLD", "80"))

# Graded labels → 0-100 score. The model picks exactly one word.
_LABEL_SCORE = {
    "SAFE": 0.0,
    "SUGGESTIVE": 45.0,   # swimwear/lingerie/mild — not explicit
    "EXPLICIT": 98.0,     # nudity / sexual / pornographic
    "GORE": 92.0,         # graphic violence / gore
}
_PROMPT = (
    "You are a strict content-safety classifier. Look at the image and reply with ONLY "
    "one word, the single best label:\n"
    "SAFE = no nudity, sexual or graphic content.\n"
    "SUGGESTIVE = mildly suggestive (swimwear, lingerie, partial), not explicit.\n"
    "EXPLICIT = nudity, sexual or pornographic content.\n"
    "GORE = graphic violence, blood or gore.\n"
    "Answer with exactly one of: SAFE, SUGGESTIVE, EXPLICIT, GORE."
)

# Latches True if the server is unreachable, so we stop hammering it and fall back.
_server_down = False


def _vlm_label(image_bytes: bytes) -> str:
    jpeg = _to_jpeg(image_bytes)
    q = urllib.parse.quote(_PROMPT)
    req = urllib.request.Request(
        f"{_SERVER_URL}?question={q}", data=jpeg,
        headers={"Content-Type": "application/octet-stream"},
    )
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
        out = json.load(resp)
    text = (out.get("answer") or "").strip().upper()
    for label in ("EXPLICIT", "GORE", "SUGGESTIVE", "SAFE"):
        if label in text:
            return label
    return "SAFE"


def classify_nsfw(image_bytes: bytes):
    """Return ``(score_0_100, is_nsfw, label)`` or ``None`` if the VLM is unavailable.

    Returning None lets the caller fall back to the legacy embedding-cosine score.
    """
    global _server_down
    if NSFW_BACKEND != "vlm" or _server_down or not image_bytes:
        return None
    try:
        label = _vlm_label(image_bytes)
        score = _LABEL_SCORE.get(label, 0.0)
        return score, score >= NSFW_THRESHOLD, label
    except urllib.error.HTTPError as e:
        print(f"[NSFW] VLM rejected one image (HTTP {e.code}); skipping VLM for it.")
        return None
    except urllib.error.URLError as e:
        print(f"[NSFW] VLM server unreachable ({e}); falling back to embedding score.")
        _server_down = True
        return None
    except Exception as e:
        print(f"[NSFW] VLM classify error ({e}).")
        return None
