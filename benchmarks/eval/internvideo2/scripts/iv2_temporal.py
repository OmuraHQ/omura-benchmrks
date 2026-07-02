"""In-video temporal localization ('search inside a video' / seek-to-timestamp).

Given a local video file and a text query, slide a window across the clip, embed each
window with omura-embed-video (finetuned InternVideo2-6B), score against the query text
embedding, and return ranked time segments [{start, end, score}].

Reused by:
  - iv2_video_service.py  (live /search_in_video endpoint, on-demand)
  - index_video_temporal.py (precompute segment vectors for the catalog)
  - eval_charades_sta.py / eval_walrus_temporal.py (verification)

Runs in .venv-iv2 (transformers 4.28 + repo). Model is loaded lazily via iv2_finetuned.
"""
import os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import iv2_common as C
import iv2_finetuned as M


def slide_windows(duration, win_sec=4.0, stride_sec=2.0, min_windows=1):
    """Return a list of (start, end) windows covering [0, duration]."""
    if duration is None or duration <= 0:
        return []
    if duration <= win_sec:
        return [(0.0, float(duration))]
    wins = []
    t = 0.0
    while t < duration:
        s = t
        e = min(duration, t + win_sec)
        wins.append((round(s, 3), round(e, 3)))
        if e >= duration:
            break
        t += stride_sec
    return wins


@torch.no_grad()
def embed_windows(video_path, windows, num_frames=4, batch=16, device="cuda", _vr=None):
    """Embed each (start,end) window of a video. Returns (feats[N,768], kept_windows).
    Windows that fail to decode are dropped (and removed from kept_windows).
    Pass `_vr` to reuse an already-opened decord VideoReader (avoids re-parsing the file)."""
    vr = _vr
    if vr is None:
        from decord import VideoReader
        try:
            vr = VideoReader(video_path, num_threads=1)
        except Exception:
            return None, []
    feats, kept = [], []
    buf, buf_w = [], []

    def flush():
        if not buf:
            return
        vecs = M.embed_video_frames(torch.cat(buf, 0), device=device)  # [b,768]
        feats.append(vecs)
        kept.extend(buf_w)

    for (s, e) in windows:
        ft = C.read_video_window(video_path, s, e, num_frames=num_frames, device=device, _vr=vr)
        if ft is None:
            continue
        buf.append(ft); buf_w.append((s, e))
        if len(buf) >= batch:
            flush(); buf, buf_w = [], []
    flush()
    if not feats:
        return None, []
    return np.concatenate(feats, 0), kept


@torch.no_grad()
def localize_file(video_path, query, win_sec=4.0, stride_sec=2.0, num_frames=4,
                  top_k=5, batch=16, device="cuda", qemb=None):
    """Localize a text query within a local video file.
    Returns {"duration": float, "segments": [{"start","end","score"}...]} sorted desc by score.
    `qemb` (a 768-d np vector) may be passed to skip text embedding (e.g. batched eval)."""
    from decord import VideoReader
    try:
        vr = VideoReader(video_path, num_threads=1)
        vlen = len(vr)
        fps = float(vr.get_avg_fps()) or 25.0
        duration = vlen / fps if vlen else None
    except Exception:
        duration, vr = None, None
    if duration is None:
        return {"duration": None, "segments": []}
    windows = slide_windows(duration, win_sec=win_sec, stride_sec=stride_sec)
    feats, kept = embed_windows(video_path, windows, num_frames=num_frames, batch=batch, device=device, _vr=vr)
    if feats is None:
        return {"duration": duration, "segments": []}
    if qemb is None:
        qemb = M.embed_text([query], device=device)[0]
    qemb = np.asarray(qemb, dtype=np.float32)
    sims = feats @ qemb  # cosine (both L2-normalized)
    order = np.argsort(-sims)
    segs = [{"start": float(kept[i][0]), "end": float(kept[i][1]),
             "score": round(float((sims[i] + 1.0) / 2.0 * 100.0), 2),
             "cosine": round(float(sims[i]), 4)} for i in order[:top_k]]
    return {"duration": round(float(duration), 3), "segments": segs}


@torch.no_grad()
def score_segments(seg_feats, query=None, qemb=None, segments_meta=None, top_k=5, device="cuda"):
    """Score precomputed segment features [N,768] against a query. segments_meta is a list of
    {"start","end"} parallel to seg_feats. Returns the same shape as localize_file's output."""
    seg_feats = np.asarray(seg_feats, dtype=np.float32)
    if qemb is None:
        qemb = M.embed_text([query], device=device)[0]
    qemb = np.asarray(qemb, dtype=np.float32)
    sims = seg_feats @ qemb
    order = np.argsort(-sims)
    segs = [{"start": float(segments_meta[i]["start"]), "end": float(segments_meta[i]["end"]),
             "score": round(float((sims[i] + 1.0) / 2.0 * 100.0), 2),
             "cosine": round(float(sims[i]), 4)} for i in order[:top_k]]
    return {"segments": segs}
