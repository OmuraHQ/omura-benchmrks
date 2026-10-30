"""Store ownership marker shared by hosted and local embedding backends.

Different embedding models can share a dimension (EmbeddingGemma 2 and the
current omura-embed are both 768-d) while being semantically incompatible.
Loading one index with another model's vectors silently produces garbage
results, so every backend claims its store with a small marker file.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional


def verify_store(
    directory: Optional[str | Path],
    *,
    backend: str,
    model: str,
    dim: int,
    env_var: str = "OMURA_VECTOR_STORE_DIR",
) -> Path:
    path = Path(directory or os.getenv(env_var, "data/vector_index"))
    marker = path / "embedding_backend.json"
    expected = {"backend": backend, "model": model, "dim": int(dim)}

    if marker.exists():
        try:
            found = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            found = {}
        if (
            found.get("backend") != backend
            or found.get("model") != model
            or found.get("dim") != int(dim)
        ):
            raise RuntimeError(
                f"Embedding store {path} belongs to {found or 'an unknown backend'}, "
                f"not {expected}. Point {env_var} at a fresh directory."
            )
        return path

    for name in ("vector_index.faiss", "vector_index.cagra"):
        index_file = path / name
        if index_file.exists() and index_file.stat().st_size > 0:
            raise RuntimeError(
                f"Embedding store {path} already contains an index from another "
                f"embedding space. Refusing to mix vectors; set {env_var} to a "
                "fresh directory."
            )

    path.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")
    return path


__all__ = ["verify_store"]
