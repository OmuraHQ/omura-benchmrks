"""VLM-based NSFW labeler — the safety counterpart to captioning.py.

Uses the same persistent Gemma 4 VL `llama-server` (mmproj vision) to actually *look*
at the image and classify its safety, instead of the crude image-embedding cosine to
NSFW text prototypes. Returns a graded 0-100 score on the same scale the rest of the
pipeline already uses (`nsfw_score`, `is_nsfw`), so it's a drop-in upgrade.

Config (env):
  OMURA_NSFW_BACKEND        "vlm" (default) | "embedding"  (embedding = legacy cosine)
  OMURA_NSFW_SERVER_URL     default: same as captioner (OMURA_CAPTION_SERVER_URL or :18080)
  OMURA_NSFW_MAX_TOKENS     generation budget incl. reasoning (default 1200)
  OMURA_NSFW_THRESHOLD      score ≥ this ⇒ is_nsfw (default 80)
"""

import json
import os
import urllib.error
import urllib.request

# Reuse the captioner's image normalization (fixes llama.cpp "failed to decode" 400s).
from omura.utils.captioning import _to_jpeg
import base64

NSFW_BACKEND = os.getenv("OMURA_NSFW_BACKEND", "vlm").strip().lower()
_SERVER_URL = os.getenv(
    "OMURA_NSFW_SERVER_URL",
    os.getenv("OMURA_CAPTION_SERVER_URL", "http://127.0.0.1:18080/v1/chat/completions"),
).strip()
_MAX_TOKENS = int(os.getenv("OMURA_NSFW_MAX_TOKENS", "1200"))
_TIMEOUT = float(os.getenv("OMURA_NSFW_TIMEOUT", "180"))
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
    b64 = base64.b64encode(_to_jpeg(image_bytes)).decode()
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": _PROMPT},
                ],
            }
        ],
        "max_tokens": _MAX_TOKENS,
        "temperature": 0.0,
        "stream": False,
    }
    req = urllib.request.Request(
        _SERVER_URL, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
        out = json.load(resp)
    msg = out["choices"][0]["message"]
    # Parse ONLY the final answer (`content`) — the reasoning text echoes the prompt's
    # label definitions ("EXPLICIT = nudity…"), so scanning it would match every image.
    text = (msg.get("content") or "").strip().upper()
    if not text:
        # Token budget exhausted before the final word: use the tail of the reasoning,
        # where the model states its conclusion (not the definitions near the top).
        text = (msg.get("reasoning_content") or "")[-160:].upper()
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
