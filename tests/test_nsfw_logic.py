import os
import numpy as np
import torch
from omura.utils.imagebind_embeddings import (
    get_nsfw_embeddings,
    generate_text_embedding,
    generate_image_embedding,
)


def test_nsfw():
    print("Testing NSFW detection logic...")

    # 1. Test prototype embeddings generation
    nsfw_vecs = get_nsfw_embeddings()
    if nsfw_vecs is None:
        print("❌ FAILED: Could not get NSFW prototype embeddings.")
        return
    print(f"✅ SUCCESS: Got {len(nsfw_vecs)} NSFW prototype embeddings.")

    # 2. Test similarity calculation
    # We'll use a dummy "nsfw" text and see if its embedding has high similarity with prototypes
    nsfw_text = "explicit nudity, nude body, sexual content, pornography"
    print(f"Testing text: '{nsfw_text}'")
    emb = generate_text_embedding(nsfw_text, is_document=False)
    if emb is None:
        print("❌ FAILED: Could not generate embedding for test text.")
        return

    emb = np.asarray(emb, dtype=np.float32).flatten()
    emb = emb / np.linalg.norm(emb)

    scores = [float(np.dot(emb, vec)) for vec in nsfw_vecs]
    best_score = max(scores)
    print(f"Best similarity score: {best_score:.4f}")

    if best_score < 0.5:
        print(
            "⚠️ WARNING: Best similarity score is surprisingly low for a direct match."
        )
    else:
        print("✅ SUCCESS: High similarity found for direct match.")

    # 3. Test a clearly non-nsfw text
    safe_text = "a beautiful sunset over the mountains"
    print(f"\nTesting text: '{safe_text}'")
    emb_safe = generate_text_embedding(safe_text, is_document=False)
    if emb_safe is None:
        print("❌ FAILED: Could not generate embedding for safe text.")
        return

    emb_safe = np.asarray(emb_safe, dtype=np.float32).flatten()
    emb_safe = emb_safe / np.linalg.norm(emb_safe)

    scores_safe = [float(np.dot(emb_safe, vec)) for vec in nsfw_vecs]
    best_score_safe = max(scores_safe)
    print(f"Best similarity score (safe text): {best_score_safe:.4f}")

    if best_score_safe > 0.4:
        print("⚠️ WARNING: Safe text has high similarity to NSFW prototypes!")
    else:
        print("✅ SUCCESS: Safe text has low similarity.")


if __name__ == "__main__":
    test_nsfw()
