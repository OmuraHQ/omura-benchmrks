"""Build a 2D text atlas from embedding vectors.

Usage examples:
  uv run python scripts/text_atlas.py --text cat --text dog --text tiger
  uv run python scripts/text_atlas.py --texts-file scripts/atlas_terms.txt
  uv run python scripts/text_atlas.py --texts-file scripts/atlas_terms.txt --out data/text_atlas.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np

from omura.utils.imagebind_embeddings import generate_text_embedding, initialize_embedding_model


def _read_lines(path: Path) -> List[str]:
    out: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            out.append(s)
    return out


def _pca_2d(x: np.ndarray) -> np.ndarray:
    """Simple PCA projection to 2D using SVD."""
    if x.shape[0] == 1:
        return np.array([[0.0, 0.0]], dtype=np.float32)
    centered = x - x.mean(axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(centered, full_matrices=False)
    coords = u[:, :2] * s[:2]
    if coords.shape[1] < 2:
        pad = np.zeros((coords.shape[0], 2 - coords.shape[1]), dtype=coords.dtype)
        coords = np.concatenate([coords, pad], axis=1)
    return coords.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a 2D text atlas from model embeddings.")
    parser.add_argument("--text", action="append", default=[], help="Text term (repeatable).")
    parser.add_argument(
        "--texts-file",
        type=str,
        default="",
        help="Path to newline-separated text terms (comments start with #).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="data/text_atlas.json",
        help="Output JSON file path.",
    )
    args = parser.parse_args()

    terms: List[str] = list(args.text or [])
    if args.texts_file:
        terms.extend(_read_lines(Path(args.texts_file)))
    terms = [t.strip() for t in terms if t and t.strip()]
    # Preserve order but deduplicate.
    seen = set()
    deduped: List[str] = []
    for t in terms:
        if t in seen:
            continue
        seen.add(t)
        deduped.append(t)
    terms = deduped

    if len(terms) < 2:
        raise SystemExit("Need at least 2 text terms (--text or --texts-file).")

    print(f"[TextAtlas] Loading embedding model and embedding {len(terms)} terms...")
    initialize_embedding_model()

    kept_terms: List[str] = []
    vectors: List[np.ndarray] = []
    for term in terms:
        emb = generate_text_embedding(term, is_document=False)
        if emb is None:
            print(f"[TextAtlas] Skipping term (no embedding): {term}")
            continue
        kept_terms.append(term)
        vectors.append(np.asarray(emb, dtype=np.float32))

    if len(vectors) < 2:
        raise SystemExit("Too few embeddings produced for atlas.")

    mat = np.stack(vectors, axis=0)
    coords = _pca_2d(mat)

    points = []
    for i, term in enumerate(kept_terms):
        points.append(
            {
                "text": term,
                "x": float(coords[i, 0]),
                "y": float(coords[i, 1]),
            }
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"points": points}, indent=2), encoding="utf-8")
    print(f"[TextAtlas] Wrote {len(points)} points to {out_path}")


if __name__ == "__main__":
    main()

