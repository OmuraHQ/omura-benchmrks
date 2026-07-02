"""Index genuine VIDEO quilt-patches (blob_id 'quiltId::identifier') with the finetuned
omura-embed-video model. Fetches each patch via the aggregator quilt-patch endpoint
(the only reliable way to get the real video bytes — the top-level quilt blobs are
mislabeled containers). Writes/append to the video_embeds npz.

  IV2_CKPT=<ckpt> OMURA_VIDEO_HEADS=<best_heads.pt> CUDA_VISIBLE_DEVICES=7 \
      .venv-iv2/bin/python scripts/index_video_patches.py --out data/cache/video_embeds.npz
"""
import os, sys, json, time, argparse, sqlite3, tempfile
import numpy as np
import requests

sys.path.insert(0, os.path.dirname(__file__))
import iv2_common as C
import iv2_finetuned as M

CATALOG_DB = os.getenv("OMURA_CATALOG_DB_PATH", "/workspace/proj/omurav2/data/blob_catalog.sqlite")
AGG = os.getenv("OMURA_FETCH_AGGREGATORS",
                "https://aggregator.walrus-mainnet.walrus.space").split(",")[0].strip().rstrip("/")
MAX_BYTES = int(os.getenv("OMURA_MAX_VIDEO_BYTES", str(3 * 1024 * 1024 * 1024)))


def resolve_patch_id(quilt_id, identifier):
    try:
        r = requests.get(f"{AGG}/v1/quilts/{quilt_id}/patches", timeout=60)
        if r.status_code != 200:
            return None
        for p in r.json():
            if p.get("identifier") == identifier:
                return p.get("patch_id")
    except Exception:
        return None
    return None


def fetch_patch(patch_id, timeout=600):
    try:
        r = requests.get(f"{AGG}/v1/blobs/by-quilt-patch-id/{patch_id}", timeout=timeout, stream=True)
        if r.status_code != 200:
            return None
        buf = bytearray()
        for chunk in r.iter_content(1 << 20):
            buf += chunk
            if len(buf) > MAX_BYTES:
                return None
        return bytes(buf) if buf else None
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/cache/video_embeds.npz")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    model, tok, cfg = M.load()
    nf = cfg.num_frames_test

    c = sqlite3.connect(CATALOG_DB); c.row_factory = sqlite3.Row
    rows = [dict(r) for r in c.execute(
        "SELECT blob_id, extension, mime_type, size, end_epoch, owner, is_nsfw "
        "FROM blobs WHERE kind='video' AND is_active=1 AND blob_id LIKE '%::%'").fetchall()]
    c.close()
    if args.limit:
        rows = rows[: args.limit]

    ids, feats, metas = [], [], []
    if os.path.exists(args.out):
        d = np.load(args.out, allow_pickle=True)
        ids = [str(x) for x in d["ids"]]; feats = [v for v in d["feats"]]
        metas = list(d["metas"]) if "metas" in d else [{}] * len(ids)
    done = set(ids)
    rows = [r for r in rows if r["blob_id"] not in done]
    print(f"[video-patch] to_process={len(rows)} (already={len(done)})")

    ok = fail = 0
    for r in rows:
        bid = r["blob_id"]; quilt_id, ident = bid.split("::", 1)
        t0 = time.time()
        pid = resolve_patch_id(quilt_id, ident)
        data = fetch_patch(pid) if pid else None
        if not data:
            fail += 1; print(f"  FAIL fetch {bid[:60]}"); continue
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        try:
            tmp.write(data); tmp.flush(); tmp.close()
            ft = C.read_video_official(tmp.name, num_frames=nf, image_res=224, sample="middle", device="cpu")
        except Exception:
            ft = None
        finally:
            try: os.unlink(tmp.name)
            except Exception: pass
        if ft is None:
            fail += 1; print(f"  FAIL decode {bid[:60]}"); continue
        import torch
        vec = M.embed_video_frames(ft.to("cuda"))[0].astype(np.float32)
        ids.append(bid); feats.append(vec)
        metas.append({"mime_type": r.get("mime_type") or "video/mp4",
                      "extension": r.get("extension") or "mp4", "kind": "video",
                      "size": len(data), "is_nsfw": bool(r.get("is_nsfw")),
                      "end_epoch": r.get("end_epoch"), "owner": r.get("owner"),
                      "caption": ident})
        ok += 1
        print(f"  OK {ident[:50]} ({len(data)//(1024*1024)}MB, {time.time()-t0:.0f}s)", flush=True)
        np.savez(args.out, ids=np.array(ids), feats=np.stack(feats), metas=np.array(metas, dtype=object))
    print(f"[video-patch] DONE ok={ok} fail={fail} total_in_npz={len(ids)}")


if __name__ == "__main__":
    main()
