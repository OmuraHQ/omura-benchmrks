"""Benchmark text-image retrieval on COCO-style captions.

Computes Recall@K metrics for:
- text -> image (required)
- image -> text (optional)

Expected captions JSON format: standard COCO captions file, e.g.:
  annotations/captions_val2014.json

This script samples a deterministic image subset (default 1000 images), embeds
each image and caption using the current Omura embedding backend, then evaluates
retrieval using cosine similarity on normalized vectors.
"""

from __future__ import annotations

import argparse
import json
import sys
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

from embedding_backend import (
    generate_image_embedding,
    generate_text_embedding,
    initialize_embedding_model,
    MODEL_NAME,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COCO_ROOT = REPO_ROOT / "data" / "coco"
DEFAULT_KARPATHY_ARCHIVE_URL = (
    "https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip"
)


@dataclass
class ImageItem:
    image_id: int
    file_name: str
    captions: List[str]


def _download_file(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    print(f"[Benchmark] Downloading: {url}")
    with urllib.request.urlopen(url) as resp, tmp.open("wb") as f:
        shutil.copyfileobj(resp, f)
    tmp.replace(dst)


def _extract_zip(zip_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Benchmark] Extracting: {zip_path.name}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)


def ensure_coco_data(
    coco_root: Path,
    split: str,
    captions_json: Path | None,
    images_dir: Path | None,
    download: bool,
) -> tuple[Path, Path]:
    """Resolve or optionally fetch COCO captions/images.

    Returns:
        (captions_json_path, images_dir_path)
    """
    if captions_json is not None and images_dir is not None:
        return captions_json, images_dir

    split = split.lower()
    if split not in {"val2014", "train2014"}:
        raise ValueError("--coco-split must be val2014 or train2014")

    ann_path = coco_root / "annotations" / f"captions_{split}.json"
    img_dir = coco_root / split

    if ann_path.exists() and img_dir.exists():
        return ann_path, img_dir

    if not download:
        raise FileNotFoundError(
            "COCO files not found. Provide --captions-json/--images-dir "
            "or pass --download-coco with --coco-root."
        )

    coco_root.mkdir(parents=True, exist_ok=True)
    ann_zip = coco_root / "annotations_trainval2014.zip"
    img_zip = coco_root / f"{split}.zip"
    ann_url = "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
    img_url = f"http://images.cocodataset.org/zips/{split}.zip"

    if not ann_zip.exists():
        _download_file(ann_url, ann_zip)
    if not img_zip.exists():
        _download_file(img_url, img_zip)

    if not (coco_root / "annotations").exists():
        _extract_zip(ann_zip, coco_root)
    if not img_dir.exists():
        _extract_zip(img_zip, coco_root)

    if not ann_path.exists() or not img_dir.exists():
        raise RuntimeError(
            "COCO download/extract completed but expected files are missing: "
            f"{ann_path} and/or {img_dir}"
        )
    return ann_path, img_dir


def ensure_karpathy_split_file(split_file: Path, download: bool) -> Path:
    if split_file.exists():
        txt = split_file.read_text(encoding="utf-8", errors="ignore")
        # Some mirrors serve a Git LFS pointer file instead of JSON payload.
        if not txt.startswith("version https://git-lfs.github.com/spec/v1"):
            return split_file
        if not download:
            raise RuntimeError(
                f"Karpathy split file at {split_file} is a Git LFS pointer, not JSON. "
                "Pass --download-karpathy-split to fetch the real file."
            )
    if not download:
        raise FileNotFoundError(
            f"Karpathy split file not found: {split_file}. "
            "Provide --split-file or pass --download-karpathy-split."
        )
    split_file.parent.mkdir(parents=True, exist_ok=True)
    archive_path = split_file.parent / "caption_datasets.zip"
    _download_file(DEFAULT_KARPATHY_ARCHIVE_URL, archive_path)
    with zipfile.ZipFile(archive_path, "r") as zf:
        member = "dataset_coco.json"
        if member not in zf.namelist():
            raise RuntimeError(
                f"{archive_path} does not contain {member}; found: {zf.namelist()[:5]}"
            )
        with zf.open(member, "r") as src, split_file.open("wb") as dst:
            shutil.copyfileobj(src, dst)
    if not split_file.exists():
        raise RuntimeError(f"Download completed but missing file: {split_file}")
    return split_file


def load_coco_items(captions_json: Path) -> List[ImageItem]:
    data = json.loads(captions_json.read_text(encoding="utf-8"))
    images = data.get("images", [])
    annotations = data.get("annotations", [])

    id_to_file: Dict[int, str] = {}
    for im in images:
        try:
            image_id = int(im["id"])
            file_name = str(im["file_name"])
            id_to_file[image_id] = file_name
        except Exception:
            continue

    caps: Dict[int, List[str]] = {}
    for ann in annotations:
        try:
            image_id = int(ann["image_id"])
            caption = str(ann["caption"]).strip()
        except Exception:
            continue
        if not caption:
            continue
        caps.setdefault(image_id, []).append(caption)

    items: List[ImageItem] = []
    for image_id, file_name in id_to_file.items():
        c = caps.get(image_id, [])
        if not c:
            continue
        items.append(ImageItem(image_id=image_id, file_name=file_name, captions=c))

    items.sort(key=lambda x: x.image_id)
    return items


def load_karpathy_items(split_file: Path, split_name: str) -> List[ImageItem]:
    """Load COCO Karpathy split JSON.

    Expected common structure:
    {
      "images": [
        {
          "filename": "COCO_val2014_000000123456.jpg",
          "cocoid": 123456,
          "split": "test",
          "sentences": [{"raw": "a caption"}, ...]
        },
        ...
      ]
    }
    """
    data = json.loads(split_file.read_text(encoding="utf-8"))
    images = data.get("images", [])
    wanted = split_name.strip().lower()
    valid_splits = {"train", "val", "test", "restval"}
    if wanted not in valid_splits:
        raise ValueError(f"--karpathy-split must be one of {sorted(valid_splits)}")

    items: List[ImageItem] = []
    fallback_id = 0
    for im in images:
        try:
            im_split = str(im.get("split", "")).strip().lower()
            if im_split != wanted:
                continue
            file_name = str(im.get("filename", "")).strip()
            if not file_name:
                continue
            image_id = int(im.get("cocoid", im.get("imgid", fallback_id)))
            fallback_id += 1
            sents = im.get("sentences", [])
            captions: List[str] = []
            for s in sents:
                raw = str(s.get("raw", "")).strip()
                if raw:
                    captions.append(raw)
            if not captions:
                continue
            items.append(ImageItem(image_id=image_id, file_name=file_name, captions=captions))
        except Exception:
            continue

    items.sort(key=lambda x: x.image_id)
    return items


def sample_items(items: List[ImageItem], n: int, seed: int) -> List[ImageItem]:
    if n <= 0 or n >= len(items):
        return items
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(items), size=n, replace=False)
    idx_sorted = sorted(int(i) for i in idx.tolist())
    return [items[i] for i in idx_sorted]


def l2_normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n > 0:
        return (v / n).astype(np.float32, copy=False)
    return v.astype(np.float32, copy=False)


def embed_images(items: List[ImageItem], images_dir: Path) -> Tuple[np.ndarray, List[ImageItem]]:
    vecs: List[np.ndarray] = []
    kept: List[ImageItem] = []
    for item in tqdm(items, desc="Embedding images", unit="img"):
        image_path = images_dir / item.file_name
        if not image_path.exists():
            continue
        try:
            b = image_path.read_bytes()
        except Exception:
            continue
        emb = generate_image_embedding(b, blob_id=f"coco_{item.image_id}")
        if emb is None:
            continue
        v = l2_normalize(np.asarray(emb, dtype=np.float32).flatten())
        vecs.append(v)
        kept.append(item)
    if not vecs:
        raise RuntimeError("No image embeddings generated.")
    return np.stack(vecs, axis=0), kept


def _normalize_caption_text(text: str, mode: str) -> str:
    t = text.strip()
    if mode == "lower":
        return t.lower()
    return t


def embed_captions(
    items: List[ImageItem], text_normalization: str = "none"
) -> Tuple[np.ndarray, np.ndarray]:
    txt_vecs: List[np.ndarray] = []
    gt_img_index: List[int] = []
    total_caps = sum(len(it.captions) for it in items)
    pbar = tqdm(total=total_caps, desc="Embedding captions", unit="cap")
    for img_idx, item in enumerate(items):
        for cap in item.captions:
            cap_norm = _normalize_caption_text(cap, text_normalization)
            emb = generate_text_embedding(cap_norm, is_document=False)
            pbar.update(1)
            if emb is None:
                continue
            v = l2_normalize(np.asarray(emb, dtype=np.float32).flatten())
            txt_vecs.append(v)
            gt_img_index.append(img_idx)
    pbar.close()
    if not txt_vecs:
        raise RuntimeError("No text embeddings generated.")
    return np.stack(txt_vecs, axis=0), np.asarray(gt_img_index, dtype=np.int32)


def recall_at_k_text_to_image(
    text_vecs: np.ndarray,
    image_vecs: np.ndarray,
    gt_img_index: np.ndarray,
    k: int,
) -> float:
    sims = text_vecs @ image_vecs.T
    topk = np.argpartition(-sims, kth=min(k - 1, sims.shape[1] - 1), axis=1)[:, :k]
    hits = 0
    for i in range(topk.shape[0]):
        if int(gt_img_index[i]) in set(int(x) for x in topk[i].tolist()):
            hits += 1
    return float(hits) / float(topk.shape[0])


def recall_at_k_text_to_image_local_negatives(
    text_vecs: np.ndarray,
    image_vecs: np.ndarray,
    gt_img_index: np.ndarray,
    k: int,
    candidate_pool_size: int = 10,
    seed: int = 42,
) -> float:
    """Recall@K with per-query local candidate pools.

    For each query, we evaluate against:
    - 1 positive image (ground truth)
    - (candidate_pool_size - 1) sampled negatives

    This protocol is easier than full-corpus retrieval and often yields
    significantly higher Recall@K.
    """
    n_img = int(image_vecs.shape[0])
    pool = max(2, min(int(candidate_pool_size), n_img))
    kk = max(1, min(int(k), pool))
    rng = np.random.default_rng(seed)
    hits = 0
    for i in range(text_vecs.shape[0]):
        pos = int(gt_img_index[i])
        all_idx = np.arange(n_img, dtype=np.int32)
        neg = all_idx[all_idx != pos]
        need = pool - 1
        if need >= neg.shape[0]:
            cand = np.concatenate([np.asarray([pos], dtype=np.int32), neg], axis=0)
        else:
            sampled = rng.choice(neg, size=need, replace=False)
            cand = np.concatenate([np.asarray([pos], dtype=np.int32), sampled], axis=0)
        q = text_vecs[i]
        sims = image_vecs[cand] @ q
        topk_local = np.argpartition(-sims, kth=kk - 1)[:kk]
        topk_global_ids = cand[topk_local]
        if pos in set(int(x) for x in topk_global_ids.tolist()):
            hits += 1
    return float(hits) / float(text_vecs.shape[0])


def alignment_sanity_check(
    image_vecs: np.ndarray,
    text_vecs: np.ndarray,
    gt_img_index: np.ndarray,
    n_samples: int,
    seed: int,
) -> Dict[str, float | int]:
    """Mean cosine(text, gt_image) vs random wrong image — catches tokenizer/padding bugs."""
    rng = np.random.default_rng(seed)
    n_text = int(text_vecs.shape[0])
    n_img = int(image_vecs.shape[0])
    n_draw = max(1, min(int(n_samples), n_text))
    idx = rng.choice(n_text, size=n_draw, replace=False)
    pos_sims: List[float] = []
    neg_sims: List[float] = []
    for ti in idx:
        ti = int(ti)
        gt = int(gt_img_index[ti])
        pos_sims.append(float(text_vecs[ti] @ image_vecs[gt]))
        neg_j = int(rng.integers(0, n_img))
        if neg_j == gt:
            neg_j = (neg_j + 1) % n_img
        neg_sims.append(float(text_vecs[ti] @ image_vecs[neg_j]))
    mp = float(np.mean(pos_sims))
    mn = float(np.mean(neg_sims))
    return {
        "n_samples": n_draw,
        "mean_pos_cosine": mp,
        "mean_neg_cosine": mn,
        "mean_gap_pos_minus_neg": float(mp - mn),
    }


def recall_at_k_image_to_text(
    image_vecs: np.ndarray,
    text_vecs: np.ndarray,
    gt_text_ranges: List[Tuple[int, int]],
    k: int,
) -> float:
    sims = image_vecs @ text_vecs.T
    topk = np.argpartition(-sims, kth=min(k - 1, sims.shape[1] - 1), axis=1)[:, :k]
    hits = 0
    for i in range(topk.shape[0]):
        lo, hi = gt_text_ranges[i]
        ok = False
        for t_idx in topk[i].tolist():
            ti = int(t_idx)
            if lo <= ti < hi:
                ok = True
                break
        if ok:
            hits += 1
    return float(hits) / float(topk.shape[0])


def main() -> None:
    parser = argparse.ArgumentParser(description="COCO retrieval benchmark (Recall@K)")
    parser.add_argument("--captions-json", type=Path, required=False)
    parser.add_argument("--images-dir", type=Path, required=False)
    parser.add_argument(
        "--split-file",
        type=Path,
        required=False,
        help="Karpathy split JSON file (e.g., dataset_coco.json).",
    )
    parser.add_argument(
        "--download-karpathy-split",
        action="store_true",
        help="Auto-download Karpathy dataset_coco.json if --split-file is missing.",
    )
    parser.add_argument(
        "--karpathy-split",
        type=str,
        default="test",
        choices=["train", "val", "test", "restval"],
        help="Subset inside Karpathy split file (default: test).",
    )
    parser.add_argument(
        "--protocol",
        type=str,
        default="default",
        choices=["default", "siglip2_paper_test", "siglip2_paper_val"],
        help=(
            "Benchmark protocol preset. "
            "siglip2_paper_test/val lock Karpathy split + global retrieval settings."
        ),
    )
    parser.add_argument(
        "--download-coco",
        action="store_true",
        help="Download COCO 2014 annotations/images if local paths are missing.",
    )
    parser.add_argument(
        "--coco-root",
        type=Path,
        default=DEFAULT_COCO_ROOT,
        help="COCO root for auto-download/extract (default: <repo>/data/coco).",
    )
    parser.add_argument(
        "--coco-split",
        type=str,
        default="val2014",
        help="COCO split for auto-fetch: val2014 or train2014 (default: val2014).",
    )
    parser.add_argument("--num-images", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-captions-per-image", type=int, default=5)
    parser.add_argument(
        "--text-normalization",
        type=str,
        default="none",
        choices=["none", "lower"],
        help="Caption normalization before encoding (default: none).",
    )
    parser.add_argument(
        "--eval-mode",
        type=str,
        default="global",
        choices=["global", "local_negatives"],
        help="global: full-corpus retrieval; local_negatives: per-query sampled negatives.",
    )
    parser.add_argument(
        "--candidate-pool-size",
        type=int,
        default=10,
        help="Candidate pool size for local_negatives mode (default: 10).",
    )
    parser.add_argument("--include-image-to-text", action="store_true")
    parser.add_argument(
        "--sanity-check",
        type=int,
        default=0,
        help="If >0, sample this many text queries and report mean pos/neg cosine vs GT/random image.",
    )
    parser.add_argument(
        "--gate-r10",
        type=float,
        default=None,
        help="If set, exit with status 1 when R@10 for --gate-metric is below this threshold (e.g. 0.85).",
    )
    parser.add_argument(
        "--gate-metric",
        type=str,
        default="i2t",
        choices=["t2i", "i2t"],
        help="Which R@10 to compare against --gate-r10 (default: i2t — image-to-text).",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("benchmarks/results/coco_retrieval.json"),
    )
    args = parser.parse_args()
    warnings: List[str] = []

    strict_paper_protocol = args.protocol in {"siglip2_paper_test", "siglip2_paper_val"}
    if strict_paper_protocol:
        if args.split_file is None:
            raise SystemExit(
                "--protocol siglip2_paper_* requires --split-file (Karpathy JSON)."
            )
        args.karpathy_split = "test" if args.protocol.endswith("_test") else "val"
        args.eval_mode = "global"
        args.max_captions_per_image = 5
        args.candidate_pool_size = 1000

    if args.split_file is not None:
        if args.images_dir is None:
            raise SystemExit("--split-file requires --images-dir to be set.")
        captions_json = ensure_karpathy_split_file(
            args.split_file, bool(args.download_karpathy_split)
        )
        images_dir = args.images_dir
        print(f"[Benchmark] Split file: {captions_json}")
        print(f"[Benchmark] Images: {images_dir}")
        items_all = load_karpathy_items(args.split_file, args.karpathy_split)
        print(f"[Benchmark] Karpathy split: {args.karpathy_split}")
    else:
        captions_json, images_dir = ensure_coco_data(
            coco_root=args.coco_root,
            split=args.coco_split,
            captions_json=args.captions_json,
            images_dir=args.images_dir,
            download=bool(args.download_coco),
        )
        print(f"[Benchmark] Captions: {captions_json}")
        print(f"[Benchmark] Images: {images_dir}")
        items_all = load_coco_items(captions_json)
    if not items_all:
        raise SystemExit("No valid image/caption items found.")
    if strict_paper_protocol:
        items = items_all
    else:
        items = sample_items(items_all, args.num_images, args.seed)

    if args.max_captions_per_image > 0:
        for it in items:
            it.captions = it.captions[: args.max_captions_per_image]

    print(f"[Benchmark] Model: {MODEL_NAME}")
    print(f"[Benchmark] Items selected: {len(items)}")
    initialize_embedding_model()

    image_vecs, kept_items = embed_images(items, images_dir)
    if len(kept_items) != len(items):
        print(f"[Benchmark] Kept images after embedding: {len(kept_items)}/{len(items)}")
    items = kept_items

    text_vecs, gt_img_index = embed_captions(items, text_normalization=args.text_normalization)
    print(f"[Benchmark] Embedded captions: {text_vecs.shape[0]}")

    if args.eval_mode == "local_negatives" and int(args.candidate_pool_size) <= 10:
        msg = (
            "local_negatives with candidate_pool_size<=10 makes R@10 trivial "
            "(can saturate at ~1.0). Prefer pool>=50 for meaningful R@10."
        )
        warnings.append(msg)
        print(f"[Benchmark][Warning] {msg}")

    if args.eval_mode == "local_negatives":
        r1 = recall_at_k_text_to_image_local_negatives(
            text_vecs,
            image_vecs,
            gt_img_index,
            1,
            candidate_pool_size=args.candidate_pool_size,
            seed=args.seed,
        )
        r5 = recall_at_k_text_to_image_local_negatives(
            text_vecs,
            image_vecs,
            gt_img_index,
            5,
            candidate_pool_size=args.candidate_pool_size,
            seed=args.seed,
        )
        r10 = recall_at_k_text_to_image_local_negatives(
            text_vecs,
            image_vecs,
            gt_img_index,
            10,
            candidate_pool_size=args.candidate_pool_size,
            seed=args.seed,
        )
    else:
        r1 = recall_at_k_text_to_image(text_vecs, image_vecs, gt_img_index, 1)
        r5 = recall_at_k_text_to_image(text_vecs, image_vecs, gt_img_index, 5)
        r10 = recall_at_k_text_to_image(text_vecs, image_vecs, gt_img_index, 10)

    out = {
        "model": MODEL_NAME,
        "protocol": args.protocol,
        "eval_mode": args.eval_mode,
        "candidate_pool_size": int(args.candidate_pool_size),
        "num_images": int(image_vecs.shape[0]),
        "num_captions": int(text_vecs.shape[0]),
        "text_normalization": args.text_normalization,
        "text_to_image": {"R@1": r1, "R@5": r5, "R@10": r10},
    }
    out["metric_context"] = {
        "dataset": "MS COCO",
        "split_type": "karpathy" if args.split_file is not None else "coco_annotations",
        "split_name": args.karpathy_split if args.split_file is not None else args.coco_split,
        "retrieval_mode": args.eval_mode,
        "candidate_universe": (
            "full_split_global"
            if args.eval_mode == "global"
            else f"local_negatives_pool_{int(args.candidate_pool_size)}"
        ),
        "protocol_locked": bool(strict_paper_protocol),
        "num_images_selected": int(len(items)),
        "captions_per_image_limit": int(args.max_captions_per_image),
        "text_normalization": args.text_normalization,
    }

    if args.include_image_to_text:
        ranges: List[Tuple[int, int]] = []
        cursor = 0
        for it in items:
            c = len(it.captions)
            ranges.append((cursor, cursor + c))
            cursor += c
        ir1 = recall_at_k_image_to_text(image_vecs, text_vecs, ranges, 1)
        ir5 = recall_at_k_image_to_text(image_vecs, text_vecs, ranges, 5)
        ir10 = recall_at_k_image_to_text(image_vecs, text_vecs, ranges, 10)
        out["image_to_text"] = {"R@1": ir1, "R@5": ir5, "R@10": ir10}

    if args.sanity_check > 0:
        sanity = alignment_sanity_check(
            image_vecs,
            text_vecs,
            gt_img_index,
            args.sanity_check,
            args.seed,
        )
        out["sanity_check"] = sanity
        print("[Benchmark] Sanity (text↔image cosine)")
        print(json.dumps(sanity, indent=2))
        if float(sanity["mean_gap_pos_minus_neg"]) < 0.0:
            w = (
                "sanity_check: mean_pos_cosine <= mean_neg_cosine — check "
                "embedding_backend text padding (SigLIP/SigLIP2 need padding=max_length)."
            )
            warnings.append(w)
            print(f"[Benchmark][Warning] {w}")

    if args.gate_r10 is not None:
        if args.gate_metric == "t2i":
            gated = float(r10)
        else:
            if not args.include_image_to_text:
                raise SystemExit("--gate-metric i2t requires --include-image-to-text")
            gated = float(out["image_to_text"]["R@10"])
        if gated < float(args.gate_r10):
            print(
                f"[Benchmark] GATE FAIL: {args.gate_metric} R@10={gated:.6f} < {args.gate_r10}",
                file=sys.stderr,
            )
            sys.exit(1)

    if warnings:
        out["warnings"] = warnings

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("[Benchmark] Results")
    print(json.dumps(out, indent=2))
    print(f"[Benchmark] Wrote: {args.out_json}")


if __name__ == "__main__":
    main()
