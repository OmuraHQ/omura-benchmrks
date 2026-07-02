"""Mine real videos from quilts: the 'video'-kind quilt containers are collections of
~60-80 short .mp4 clips each. Expand quilts, fetch each mp4 patch, decode + embed with
omura-embed-video, write to the video_embeds npz until --target videos are collected.

  IV2_CKPT=<ckpt> OMURA_VIDEO_HEADS=<best_heads.pt> CUDA_VISIBLE_DEVICES=7 \
      .venv-iv2/bin/python scripts/index_video_quilts.py --target 1200 --out data/cache/video_embeds.npz
"""
import os, sys, time, argparse, sqlite3, tempfile
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import requests

sys.path.insert(0, os.path.dirname(__file__))
import iv2_common as C
import iv2_finetuned as M

DB = os.getenv("OMURA_CATALOG_DB_PATH", "/workspace/proj/omurav2/data/blob_catalog.sqlite")
AGG = os.getenv("OMURA_FETCH_AGGREGATORS", "https://aggregator.walrus-mainnet.walrus.space").split(",")[0].strip().rstrip("/")
MAX_BYTES = int(os.getenv("OMURA_MAX_VIDEO_BYTES", str(60 * 1024 * 1024)))
_tls = __import__("threading").local()


def sess():
    s = getattr(_tls, "s", None)
    if s is None:
        s = requests.Session(); _tls.s = s
    return s


def list_mp4_patches(quilt_id):
    try:
        r = sess().get(f"{AGG}/v1/quilts/{quilt_id}/patches", timeout=30)
        if r.status_code != 200:
            return []
        return [(quilt_id, p["patch_id"], p["identifier"])
                for p in r.json() if p.get("identifier", "").lower().endswith((".mp4", ".mov", ".webm", ".mkv"))]
    except Exception:
        return []


def fetch_decode(item, nf):
    quilt_id, pid, ident = item
    try:
        r = sess().get(f"{AGG}/v1/blobs/by-quilt-patch-id/{pid}", timeout=120, stream=True)
        if r.status_code != 200:
            return item, None, "fetch_fail"
        buf = bytearray()
        for ch in r.iter_content(1 << 20):
            buf += ch
            if len(buf) > MAX_BYTES:
                return item, None, "too_large"
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp.write(buf); tmp.close()
        try:
            ft = C.read_video_official(tmp.name, num_frames=nf, image_res=224, sample="middle", device="cpu")
        finally:
            os.unlink(tmp.name)
        return item, (ft, len(buf)), ("ok" if ft is not None else "decode_fail")
    except Exception:
        return item, None, "err"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/cache/video_embeds.npz")
    ap.add_argument("--target", type=int, default=1200)
    ap.add_argument("--max-quilts", type=int, default=60)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    model, tok, cfg = M.load()
    nf = cfg.num_frames_test

    ids, feats, metas = [], [], []
    if os.path.exists(args.out):
        d = np.load(args.out, allow_pickle=True)
        ids = [str(x) for x in d["ids"]]; feats = [v for v in d["feats"]]
        metas = list(d["metas"]) if "metas" in d else [{}] * len(ids)
    done = set(ids)
    print(f"[vq] starting with {len(ids)} already indexed; target +{args.target}")

    c = sqlite3.connect(DB)
    quilts = [r[0] for r in c.execute(
        "SELECT blob_id FROM blobs WHERE kind='video' AND is_active=1 AND blob_id NOT LIKE '%::%' "
        "ORDER BY RANDOM() LIMIT ?", (args.max_quilts,)).fetchall()]
    c.close()

    # Build work-list of mp4 patches (skip already-indexed) until enough.
    need = args.target
    work = []
    for q in quilts:
        for (qid, pid, ident) in list_mp4_patches(q):
            bid = f"{qid}::{ident}"
            if bid not in done:
                work.append((qid, pid, ident))
        if len(work) >= need * 1.25:
            break
    print(f"[vq] collected {len(work)} mp4 patches from quilts", flush=True)

    import torch
    ok = fail = 0
    buf_frames, buf_items = [], []
    t = time.time()

    def flush():
        nonlocal buf_frames, buf_items, ok
        if not buf_frames:
            return
        vecs = M.embed_video_frames(torch.cat(buf_frames, 0))
        for (qid, pid, ident), v in zip(buf_items, vecs):
            ids.append(f"{qid}::{ident}"); feats.append(v.astype(np.float32))
            metas.append({"mime_type": "video/mp4", "extension": "mp4", "kind": "video",
                          "size": 0, "is_nsfw": False, "caption": ident})
            ok += 1
        buf_frames, buf_items = [], []

    def save():
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        np.savez(args.out, ids=np.array(ids), feats=np.stack(feats), metas=np.array(metas, dtype=object))

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for item, res, status in ex.map(lambda it: fetch_decode(it, nf), work):
            if status == "ok":
                ft, _ = res
                buf_frames.append(ft.to("cuda")); buf_items.append(item)
                if len(buf_frames) >= args.batch:
                    flush()
                    if ok % 100 < args.batch:
                        save()
            else:
                fail += 1
            tot = ok + fail
            if tot and tot % 50 == 0:
                print(f"  {tot}/{len(work)} ok={ok} fail={fail} ({(time.time()-t)/tot:.2f}s/patch)", flush=True)
            if ok >= args.target:
                break
    flush(); save()
    print(f"[vq] DONE ok={ok} fail={fail} total_in_npz={len(ids)}")


if __name__ == "__main__":
    main()
