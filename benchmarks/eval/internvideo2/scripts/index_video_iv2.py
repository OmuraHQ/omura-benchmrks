"""Embed active VIDEO blobs with the finetuned 'omura embed video' model and write an
npz (ids + 768-d feats + metadata). A separate step (build_video_store.py, run in the
main venv) loads this npz into the FAISS video index — keeps faiss out of .venv-iv2.

  IV2_CKPT=<ckpt> OMURA_VIDEO_HEADS=<best_heads.pt> CUDA_VISIBLE_DEVICES=7 \
      .venv-iv2/bin/python scripts/index_video_iv2.py --limit 2000 --out data/cache/video_embeds.npz
"""
import os, sys, json, time, argparse, sqlite3, tempfile
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import numpy as np
import requests

sys.path.insert(0, os.path.dirname(__file__))
import iv2_common as C
import iv2_finetuned as M

CATALOG_DB = os.getenv("OMURA_CATALOG_DB_PATH",
                       "/workspace/proj/omurav2/data/blob_catalog.sqlite")
AGGREGATORS = [a.strip().rstrip("/") for a in os.getenv(
    "OMURA_FETCH_AGGREGATORS",
    "https://agrregator.omura.fun,https://aggregator.walrus-mainnet.walrus.space",
).split(",") if a.strip()]
MAX_BYTES = int(os.getenv("OMURA_MAX_VIDEO_BYTES", str(120 * 1024 * 1024)))


def fetch(blob_id, timeout=90):
    for agg in AGGREGATORS:
        try:
            r = requests.get(f"{agg}/v1/blobs/{blob_id}", timeout=timeout, stream=True)
            if r.status_code != 200:
                continue
            cl = int(r.headers.get("Content-Length") or 0)
            if cl and cl > MAX_BYTES:
                return None
            buf = bytearray()
            for chunk in r.iter_content(1 << 20):
                buf += chunk
                if len(buf) > MAX_BYTES:
                    return None
            if buf:
                return bytes(buf)
        except Exception:
            continue
    return None


def active_videos(limit, skip_ids):
    c = sqlite3.connect(CATALOG_DB); c.row_factory = sqlite3.Row
    rows = [dict(r) for r in c.execute(
        "SELECT blob_id, extension, mime_type, size, end_epoch, owner, is_nsfw "
        "FROM blobs WHERE kind='video' AND is_active=1").fetchall()]
    c.close()
    rows = [r for r in rows if r["blob_id"] not in skip_ids]
    return rows[:limit] if limit else rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/cache/video_embeds.npz")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--save-every", type=int, default=200)
    args = ap.parse_args()

    model, tok, cfg = M.load()
    nf = cfg.num_frames_test

    # resume from existing npz
    ids, feats, metas = [], [], []
    if os.path.exists(args.out):
        d = np.load(args.out, allow_pickle=True)
        ids = [str(x) for x in d["ids"]]
        feats = [v for v in d["feats"]]
        metas = list(d["metas"]) if "metas" in d else [{}] * len(ids)
        print(f"[video-index] resume: {len(ids)} already embedded")
    done_ids = set(ids)

    rows = active_videos(args.limit, done_ids)
    print(f"[video-index] to_process={len(rows)} (num_frames={nf}, batch={args.batch})")

    ok = fetch_fail = decode_fail = 0
    buf_frames, buf_rows = [], []

    def flush():
        nonlocal buf_frames, buf_rows, ok
        if not buf_frames:
            return
        import torch
        batch = torch.cat(buf_frames, 0)
        vecs = M.embed_video_frames(batch)
        for row, v in zip(buf_rows, vecs):
            ids.append(row["blob_id"]); feats.append(v.astype(np.float32))
            metas.append({"mime_type": row.get("mime_type") or "video",
                          "extension": row.get("extension"), "kind": "video",
                          "size": int(row.get("size") or 0),
                          "is_nsfw": bool(row.get("is_nsfw")),
                          "end_epoch": row.get("end_epoch"), "owner": row.get("owner")})
            ok += 1
        buf_frames, buf_rows = [], []

    def save():
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        np.savez(args.out, ids=np.array(ids), feats=np.stack(feats),
                 metas=np.array(metas, dtype=object))

    def fetch_decode(row):
        """Network + decord decode (runs in worker threads). Returns (row, frames|None, status)."""
        data = fetch(row["blob_id"])
        if not data:
            return row, None, "fetch_fail"
        tmp = tempfile.NamedTemporaryFile(suffix="." + (row.get("extension") or "mp4"), delete=False)
        try:
            tmp.write(data); tmp.flush(); tmp.close()
            ft = C.read_video_official(tmp.name, num_frames=nf, image_res=224, sample="middle", device="cpu")
        except Exception:
            ft = None
        finally:
            try: os.unlink(tmp.name)
            except Exception: pass
        return row, ft, ("ok" if ft is not None else "decode_fail")

    # Producer threads fetch+decode in parallel; main thread batches frames onto the GPU.
    import torch
    t = time.time(); seen = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for row, ft, status in ex.map(fetch_decode, rows):
            seen += 1
            if status == "fetch_fail":
                fetch_fail += 1
            elif status == "decode_fail" or ft is None:
                decode_fail += 1
            else:
                buf_frames.append(ft.to("cuda")); buf_rows.append(row)
                if len(buf_frames) >= args.batch:
                    flush()
                    if ok % args.save_every < args.batch:
                        save()
            if seen % 50 == 0:
                print(f"  {seen}/{len(rows)} ok={ok} fetch_fail={fetch_fail} decode_fail={decode_fail} "
                      f"({(time.time()-t)/seen:.2f}s/blob)", flush=True)
    flush(); save()
    print(f"[video-index] DONE ok={ok} fetch_fail={fetch_fail} decode_fail={decode_fail} total={len(ids)}")


if __name__ == "__main__":
    main()
