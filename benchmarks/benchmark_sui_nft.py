"""Sui-native NFT retrieval benchmark on the frozen Omura eval manifest.

Directions
----------
* text -> image : catalog caption vs all manifest images (global pool).
* image -> image: transformed query image (5% crop, 90% resize, JPEG q60) vs
  the original images.

Both report R@1 / R@5 / R@10 and the per-query rank, so the R@10 condition is
auditable from the artifact.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent))

import embedding_backend as eb
from bench_common import (
    REPO_ROOT,
    embed_images,
    embed_texts,
    git_commit,
    ranks_at_k,
    recall_at_k,
    utc_now,
    write_result,
)

DEFAULT_MANIFEST = REPO_ROOT / "data" / "sui_nft_eval" / "manifest.json"


def transform_image(data: bytes, quality: int = 60) -> bytes | None:
    try:
        with Image.open(io.BytesIO(data)) as src:
            img = src.convert("RGBA")
            # Some on-chain images are enormous (decompression-bomb shaped).
            # Cap the working resolution before allocating an RGBA canvas.
            max_side = max(img.size)
            if max_side > 2048:
                scale = 2048.0 / max_side
                img = img.resize(
                    (max(1, int(img.width * scale)), max(1, int(img.height * scale)))
                )
            bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(bg, img).convert("RGB")
            w, h = img.size
            cx = 0.05 / 2.0
            crop = img.crop((int(w * cx), int(h * cx), int(w * (1 - cx)), int(h * (1 - cx))))
            resized = crop.resize((max(16, int(w * 0.9)), max(16, int(h * 0.9))))
            buf = io.BytesIO()
            resized.save(buf, format="JPEG", quality=quality)
            return buf.getvalue()
    except Exception:  # noqa: BLE001
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Sui NFT retrieval benchmark (gemma)")
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="Smoke-test cap on pool size.")
    ap.add_argument(
        "--out-json",
        type=Path,
        default=REPO_ROOT / "benchmarks" / "results" / "sui_nft_gemma.json",
    )
    args = ap.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    entries = manifest["images"]
    if args.limit:
        entries = entries[: args.limit]
    base = args.manifest.parent

    datas: List[bytes] = []
    kept: List[dict] = []
    unreadable: List[dict] = []
    for e in entries:
        p = base / e["file"]
        try:
            datas.append(p.read_bytes())
            kept.append(e)
        except OSError as exc:
            unreadable.append({"blob_id": e["blob_id"], "error": str(exc)})
    print(f"[SuiNFT] manifest images: {len(entries)}  readable: {len(kept)}")
    print(f"[SuiNFT] model: {eb.MODEL_NAME} dim={eb.EMBEDDING_DIM}")

    eb.initialize_embedding_model()
    dim = int(eb.EMBEDDING_DIM or 768)

    image_vecs, kept_pos = embed_images(
        eb.generate_image_embeddings,
        datas,
        dim,
        batch_size=args.batch_size,
        desc="Embedding Sui NFT images",
        blob_ids=[e["blob_id"] for e in kept],
    )
    kept = [kept[i] for i in kept_pos]
    blob_ids = [e["blob_id"] for e in kept]

    captions = [e["caption"] for e in kept]
    text_vecs, text_pos = embed_texts(
        eb.generate_text_embeddings,
        captions,
        dim,
        batch_size=max(32, args.batch_size * 2),
        desc="Embedding Sui NFT captions",
    )
    t2i_query_ids = [blob_ids[i] for i in text_pos]
    t2i_gt = [np.asarray([i], dtype=np.int64) for i in text_pos]
    sims_t2i = text_vecs @ image_vecs.T
    t2i = recall_at_k(sims_t2i, t2i_gt)
    t2i_ranks = ranks_at_k(sims_t2i, t2i_gt)

    queries: List[bytes] = []
    q_ids: List[str] = []
    q_rows: List[int] = []
    for row, data in enumerate(datas):
        t = transform_image(data)
        if t is None:
            continue
        queries.append(t)
        q_ids.append(blob_ids[row])
        q_rows.append(row)
    q_vecs, q_pos = embed_images(
        eb.generate_image_embeddings,
        queries,
        dim,
        batch_size=args.batch_size,
        desc="Embedding Sui NFT query transforms",
        blob_ids=q_ids,
    )
    q_rows = [q_rows[i] for i in q_pos]
    q_ids = [q_ids[i] for i in q_pos]
    i2i_gt = [np.asarray([r], dtype=np.int64) for r in q_rows]
    sims_i2i = q_vecs @ image_vecs.T
    i2i = recall_at_k(sims_i2i, i2i_gt)
    i2i_ranks = ranks_at_k(sims_i2i, i2i_gt)

    payload = {
        "model": eb.MODEL_NAME,
        "dim": dim,
        "backend": "omuragemma" if eb.IS_OMURAGEMMA else "siglip",
        "protocol": "sui_nft_manifest_global_t2i_and_i2i",
        "dataset": "Omura Sui/Walrus production catalog (frozen manifest)",
        "manifest": str(args.manifest),
        "seed": args.seed,
        "commit": git_commit(),
        "utc_date": utc_now(),
        "pool_size": int(image_vecs.shape[0]),
        "num_images": int(image_vecs.shape[0]),
        "num_queries": int(text_vecs.shape[0]),
        "num_i2i_queries": int(q_vecs.shape[0]),
        "text_to_image": t2i,
        "image_to_image": i2i,
        "i2i_transform": {"border_crop": "5%", "resize": "90%", "jpeg_quality": 60},
        "unreadable": unreadable,
        "manifest_failures": manifest.get("num_failures", 0),
        "per_query_ranks": {
            "text_to_image": [
                {"blob_id": b, "rank": int(r)} for b, r in zip(t2i_query_ids, t2i_ranks)
            ],
            "image_to_image": [
                {"blob_id": b, "rank": int(r)} for b, r in zip(q_ids, i2i_ranks)
            ],
        },
        "notes": (
            "Every image has one catalog caption, so t2i has one query per image. "
            "Global pool: all manifest images. No sampled negatives."
        ),
    }
    write_result(args.out_json, payload)
    print(json.dumps({k: payload[k] for k in ("text_to_image", "image_to_image", "pool_size", "num_queries")}, indent=2))
    print(f"[SuiNFT] Wrote: {args.out_json}")


if __name__ == "__main__":
    main()
