"""Precompute the temporal segment index for in-video search (seek-to-timestamp).

For each catalog video (ids taken from the whole-video embed npz), fetch the mp4, slice it
into sliding windows, embed each window with omura-embed-video, and store per-segment vectors.
Output npz (loaded by iv2_video_service.py):  seg_blob[], seg_start[], seg_end[], seg_feat[N,768].

  IV2_CKPT=<ckpt> OMURA_VIDEO_HEADS=<best_heads.pt> CUDA_VISIBLE_DEVICES=6 \
    .venv-iv2/bin/python scripts/index_video_temporal.py \
      --ids-npz data/cache/video_embeds.npz --out data/cache/video_temporal.npz \
      --win-sec 4 --stride-sec 2 --limit 4204
"""
import os, sys, time, argparse, tempfile, signal
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import requests

sys.path.insert(0, os.path.dirname(__file__))
import iv2_temporal as T


class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout()


signal.signal(signal.SIGALRM, _alarm)

AGG = [a.strip().rstrip("/") for a in os.getenv(
    "OMURA_FETCH_AGGREGATORS",
    "https://agrregator.omura.fun,https://aggregator.walrus-mainnet.walrus.space").split(",") if a.strip()]
MAX_BYTES = int(os.getenv("OMURA_MAX_VIDEO_BYTES", str(80 * 1024 * 1024)))
_tls = __import__("threading").local()


def sess():
    s = getattr(_tls, "s", None)
    if s is None:
        s = requests.Session(); _tls.s = s
    return s


def fetch_bytes(blob_id):
    paths = []
    if "::" in blob_id:
        quilt, ident = blob_id.split("::", 1)
        paths.append(f"/v1/blobs/by-quilt-id/{quilt}/{requests.utils.quote(ident, safe='')}")
    else:
        paths.append(f"/v1/blobs/{blob_id}")
    for agg in AGG:
        for p in paths:
            try:
                r = sess().get(f"{agg}{p}", timeout=120, stream=True)
                if r.status_code != 200:
                    continue
                buf = bytearray()
                for ch in r.iter_content(1 << 20):
                    buf += ch
                    if len(buf) > MAX_BYTES:
                        return None
                if buf:
                    return bytes(buf)
            except Exception:
                continue
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids-npz", default="data/cache/video_embeds.npz")
    ap.add_argument("--out", default="data/cache/video_temporal.npz")
    ap.add_argument("--win-sec", type=float, default=4.0)
    ap.add_argument("--stride-sec", type=float, default=2.0)
    ap.add_argument("--num-frames", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--fetch-workers", type=int, default=12)
    args = ap.parse_args()

    ids = [str(x) for x in np.load(args.ids_npz, allow_pickle=True)["ids"]]
    if args.limit:
        ids = ids[:args.limit]

    seg_blob, seg_start, seg_end, seg_feat = [], [], [], []
    done = set()
    if os.path.exists(args.out):
        d = np.load(args.out, allow_pickle=True)
        seg_blob = [str(x) for x in d["seg_blob"]]
        seg_start = [float(x) for x in d["seg_start"]]
        seg_end = [float(x) for x in d["seg_end"]]
        seg_feat = [v for v in d["seg_feat"]]
        done = set(seg_blob)
    todo = [b for b in ids if b not in done]
    print(f"[vt] videos={len(ids)} already={len(done)} todo={len(todo)}", flush=True)

    def save():
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        np.savez(args.out,
                 seg_blob=np.array(seg_blob), seg_start=np.array(seg_start, dtype=np.float32),
                 seg_end=np.array(seg_end, dtype=np.float32),
                 seg_feat=np.stack(seg_feat) if seg_feat else np.zeros((0, 768), np.float32))

    ok = fail = 0
    t0 = time.time()
    # Fetch in parallel, embed serially on the GPU.
    with ThreadPoolExecutor(max_workers=args.fetch_workers) as ex:
        for blob_id, data in zip(todo, ex.map(fetch_bytes, todo)):
            if data is None:
                fail += 1
            else:
                tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
                tmp.write(data); tmp.close()
                try:
                    signal.alarm(30)  # hard cap per video: skip clips that hang decode
                    dur, fps, vlen = T.C.video_duration(tmp.name)
                    if dur:
                        wins = T.slide_windows(dur, win_sec=args.win_sec, stride_sec=args.stride_sec)
                        feats, kept = T.embed_windows(tmp.name, wins, num_frames=args.num_frames)
                        if feats is not None:
                            for (s, e), v in zip(kept, feats):
                                seg_blob.append(blob_id); seg_start.append(s); seg_end.append(e)
                                seg_feat.append(v.astype(np.float32))
                            ok += 1
                        else:
                            fail += 1
                    else:
                        fail += 1
                except _Timeout:
                    fail += 1
                finally:
                    signal.alarm(0)
                    os.unlink(tmp.name)
            tot = ok + fail
            if tot % 50 == 0:
                print(f"  {tot}/{len(todo)} ok={ok} fail={fail} segs={len(seg_blob)} "
                      f"({(time.time()-t0)/max(tot,1):.2f}s/vid)", flush=True)
                save()
    save()
    print(f"[vt] DONE ok={ok} fail={fail} videos_in_npz={len(set(seg_blob))} segments={len(seg_blob)}")


if __name__ == "__main__":
    main()
