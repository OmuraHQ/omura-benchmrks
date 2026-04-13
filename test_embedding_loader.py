#!/usr/bin/env python3
"""Quick smoke test for Qwen3-VL embedding model loader."""

import sys
import numpy as np


def test_model_load():
    print("=" * 60)
    print("Testing Qwen3-VL Embedding Model Loader")
    print("=" * 60)

    try:
        from omura.utils.imagebind_embeddings import (
            initialize_embedding_model,
            is_model_ready,
            generate_text_embedding,
        )

        print("[OK] Imported embedding utilities")
    except ImportError as e:
        print(f"[FAIL] Import error: {e}")
        return False

    print("\n[1] Initializing model...")
    try:
        initialize_embedding_model()
        print("[OK] Model initialization called")
    except Exception as e:
        print(f"[FAIL] Model initialization error: {e}")
        return False

    if not is_model_ready():
        print("[WARN] Model not ready yet (may still loading)")
        return False

    print("[OK] Model is ready")

    print("\n[2] Testing text embedding...")
    test_text = "This is a test sentence for embedding."
    try:
        emb = generate_text_embedding(test_text, is_document=False)
        if emb is None:
            print("[FAIL] Got None embedding")
            return False

        print(f"[OK] Embedding shape: {emb.shape}")
        print(f"[OK] Embedding dtype: {emb.dtype}")
        print(f"[OK] Embedding norm: {np.linalg.norm(emb):.4f}")

        if abs(np.linalg.norm(emb) - 1.0) > 0.01:
            print("[WARN] Embedding not properly normalized")

    except Exception as e:
        print(f"[FAIL] Text embedding error: {e}")
        import traceback

        traceback.print_exc()
        return False

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_model_load()
    sys.exit(0 if success else 1)
