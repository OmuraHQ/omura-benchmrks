"""Text embedding for video search via the 'omura embed video' (finetuned InternVideo2)
microservice. InternVideo2 can't load in this venv (needs transformers 4.28 + the repo),
so we call the sidecar service (scripts/iv2_video_service.py, run in .venv-iv2).

Configure with OMURA_VIDEO_SERVICE_URL (default http://127.0.0.1:19560).
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
import requests

SERVICE_URL = os.getenv("OMURA_VIDEO_SERVICE_URL", "http://127.0.0.1:19560").rstrip("/")
_session = requests.Session()


def embed_text(text: str, timeout: int = 30) -> Optional[np.ndarray]:
    """Embed a text query into the omura-embed-video joint space ([768], L2-normalized)."""
    text = (text or "").strip()
    if not text:
        return None
    try:
        r = _session.post(f"{SERVICE_URL}/embed_text", json={"text": text}, timeout=timeout)
        if r.status_code != 200:
            print(f"[video-emb] service HTTP {r.status_code}")
            return None
        vec = np.asarray(r.json().get("vec"), dtype=np.float32)
        return vec if vec.size else None
    except Exception as e:  # noqa: BLE001
        print(f"[video-emb] service call failed: {e}")
        return None


def search_in_video(blob_id: str, query: str, top_k: int = 5,
                    win_sec: float = 4.0, stride_sec: float = 2.0,
                    timeout: int = 120) -> Optional[dict]:
    """Localize a text query within a single video (seek-to-timestamp). Returns the sidecar
    payload {blob_id, duration, source, segments:[{start,end,score}...]} or None on failure."""
    blob_id = (blob_id or "").strip()
    query = (query or "").strip()
    if not blob_id or not query:
        return None
    try:
        r = _session.post(
            f"{SERVICE_URL}/search_in_video",
            json={"blob_id": blob_id, "query": query, "top_k": top_k,
                  "win_sec": win_sec, "stride_sec": stride_sec},
            timeout=timeout,
        )
        if r.status_code != 200:
            print(f"[video-emb] in-video service HTTP {r.status_code}")
            return None
        return r.json()
    except Exception as e:  # noqa: BLE001
        print(f"[video-emb] in-video call failed: {e}")
        return None


__all__ = ["embed_text", "search_in_video"]
