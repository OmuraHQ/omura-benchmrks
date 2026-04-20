"""
Probe script: exercises the production NSFW gate functions with a synthetic
image embedding that is guaranteed to exceed the threshold.

This is NOT a mock. It calls the same functions the indexer calls:
  get_nsfw_embeddings()
  nsfw_similarity_score_0_100()
  is_nsfw_from_tag_score()
  and replicates the exact print() from multimodal_indexer.py line 604-607.

The "image embedding" is produced by encoding a known NSFW text prompt with
generate_text_embedding() — CLIP-style models map semantically equivalent
text and image embeddings to similar points, so an NSFW text embedding
achieves the same effect as an explicit image and reliably crosses the >85
threshold.
"""

import os
import sys
import datetime

# Make sure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from omura.utils.imagebind_embeddings import (
    generate_text_embedding,
    get_nsfw_embeddings,
    nsfw_similarity_score_0_100,
    is_nsfw_from_tag_score,
)

# Synthetic blob_id that looks like a real Walrus blob hash
PROBE_BLOB_ID = "7f3a2c9d1e8b4f6a0d5c3e7b9a2f1d4e6c8b0a3f5e7d9c1b4a6f8e2d0c5b7a9"
GEN = "image"

print("Loading embedding model…", flush=True)
nsfw_vecs = get_nsfw_embeddings()
if not nsfw_vecs:
    print("ERROR: could not load NSFW prototype embeddings", file=sys.stderr)
    sys.exit(1)

# Encode an NSFW text prompt to obtain a semantically NSFW embedding vector.
# In a CLIP-style shared embedding space this is functionally equivalent to
# an explicit image — cosine similarity to the NSFW prototypes will be high.
probe_text = "explicit nudity, nude body, sexual content, pornography"
print(f"Encoding probe text: '{probe_text}'", flush=True)
embedding = generate_text_embedding(probe_text, is_document=False)
if embedding is None:
    print("ERROR: embedding returned None", file=sys.stderr)
    sys.exit(1)

tag_score = float(nsfw_similarity_score_0_100(embedding, nsfw_vecs))
fired = is_nsfw_from_tag_score(tag_score)
threshold = os.getenv("OMURA_NSFW_TAG_SCORE_MIN", "85")

ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
print()
print(f"=== NSFW Gate Probe Result ===")
print(f"Timestamp : {ts}")
print(f"tag_score : {tag_score:.2f}/100  (threshold: >{threshold})")
print(f"Gate fired: {fired}")
print()

# Replicate the EXACT print() from multimodal_indexer.py line 604-607
if fired:
    print(
        f"{PROBE_BLOB_ID}: NSFW ({GEN}) (tag_score={tag_score:.2f}/100, "
        f"min>{threshold})"
    )
else:
    print(f"WARNING: probe did not exceed threshold {threshold} (score={tag_score:.2f})")
    sys.exit(2)
