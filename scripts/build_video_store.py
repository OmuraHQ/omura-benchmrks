#!/usr/bin/env python
"""Load video embeddings (npz produced by index_video_iv2.py in .venv-iv2) into the
768-d FAISS video store that the API serves for /search/video. Runs in the main venv.

  PYTHONPATH=. .venv/bin/python scripts/build_video_store.py \
      --npz benchmarks/eval/internvideo2/data/cache/video_embeds.npz
"""
import argparse, os
from pathlib import Path
import numpy as np

from omura.utils.vector_store import VectorStore

VIDEO_DIR = Path(os.getenv("OMURA_VIDEO_VECTOR_STORE_DIR", "data/vector_index_iv2"))
VIDEO_DIM = int(os.getenv("OMURA_VIDEO_EMBEDDING_DIM", "768"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    ids = [str(x) for x in d["ids"]]
    feats = d["feats"].astype(np.float32)
    metas = list(d["metas"]) if "metas" in d else [{}] * len(ids)
    print(f"[build-video-store] {len(ids)} embeddings, dim={feats.shape[-1]}")

    store = VectorStore(index_path=VIDEO_DIR / "vector_index.faiss", embedding_dim=VIDEO_DIM)
    try:
        store.load()
    except Exception:
        pass
    added = 0
    for vid, vec, m in zip(ids, feats, metas):
        m = m if isinstance(m, dict) else {}
        store.add(embedding=vec, blob_id=vid,
                  mime_type=m.get("mime_type") or "video",
                  size=int(m.get("size") or 0),
                  extension=m.get("extension"), kind="video",
                  is_nsfw=bool(m.get("is_nsfw")),
                  end_epoch=m.get("end_epoch"), owner=m.get("owner"))
        added += 1
    store.save(create_backup=False)
    print(f"[build-video-store] DONE added={added} store_size={store.size()} dir={VIDEO_DIR}")


if __name__ == "__main__":
    main()
