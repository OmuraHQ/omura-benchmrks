"""Diagnostic checks for image embedding diversity.

Run:
  uv run python tests/test_embedding_diversity.py
"""

from __future__ import annotations

import numpy as np
import requests

from omura.utils.imagebind_embeddings import generate_image_embedding
from omura.utils.vector_store import VectorStore


def _sample_image_rows_from_store(store: VectorStore, n: int = 4):
    blob_ids = []
    vectors = []
    for blob_id in store.position_to_blob_id:
        meta = store.get_blob_metadata(blob_id)
        if not meta or meta.get("kind") != "image":
            continue
        vec = store.get_embedding(blob_id)
        if vec is None:
            continue
        blob_ids.append(blob_id)
        vectors.append(np.asarray(vec, dtype=np.float32).flatten())
        if len(blob_ids) >= n:
            break
    return blob_ids, vectors


def _unique_rows(vectors: list[np.ndarray]) -> int:
    mat = np.vstack(vectors)
    mat_r = np.round(mat, 6)
    return int(np.unique(mat_r, axis=0).shape[0])


def _fresh_embeddings_for_blob_ids(blob_ids: list[str]) -> list[np.ndarray]:
    out = []
    for blob_id in blob_ids:
        url = f"https://walrus-mainnet-aggregator.redundex.com/v1/blobs/{blob_id}"
        resp = requests.get(url, timeout=60)
        if resp.status_code != 200:
            continue
        emb = generate_image_embedding(resp.content, blob_id=blob_id)
        if emb is None:
            continue
        out.append(np.asarray(emb, dtype=np.float32).flatten())
    return out


def main() -> None:
    store = VectorStore()
    store.load()

    image_blob_ids, image_vectors = _sample_image_rows_from_store(store, n=4)

    if len(image_blob_ids) < 4:
        raise SystemExit(
            f"FAIL: need at least 4 image embeddings, found {len(image_blob_ids)}"
        )

    unique_stored = _unique_rows(image_vectors)

    print("blob_ids:")
    for bid in image_blob_ids:
        print(f" - {bid}")
    print(f"stored_unique_rows={unique_stored} / sampled={len(image_blob_ids)}")

    fresh_vectors = _fresh_embeddings_for_blob_ids(image_blob_ids)
    if len(fresh_vectors) < 2:
        raise SystemExit("FAIL: could not generate enough fresh embeddings for comparison.")

    unique_fresh = _unique_rows(fresh_vectors)
    print(f"fresh_unique_rows={unique_fresh} / generated={len(fresh_vectors)}")

    if unique_fresh < 2:
        raise SystemExit("FAIL: encoder still collapses distinct images to identical vectors.")

    if unique_stored < 2 and unique_fresh >= 2:
        raise SystemExit(
            "FAIL: stored index vectors are collapsed, but fresh encoder output is diverse. "
            "Reindex required."
        )

    print("PASS: stored and fresh image embeddings are diverse.")


if __name__ == "__main__":
    main()

