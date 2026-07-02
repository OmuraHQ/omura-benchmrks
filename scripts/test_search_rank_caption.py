#!/usr/bin/env python3
import os
import sys
import json
import httpx
from PIL import Image
from io import BytesIO
from pathlib import Path

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
load_dotenv()

# We will use Salesforce/blip-image-captioning-base for lightweight captioning
from transformers import BlipProcessor, BlipForConditionalGeneration

def main():
    print("Initializing BLIP image captioning model...")
    try:
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        # Use GPU if available
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        print(f"Model loaded on {device}.")
    except Exception as e:
        print(f"Error loading captioning model: {e}")
        sys.exit(1)

    # Search local endpoint
    api_url = "http://localhost:19353/search/"
    aggregator_url = os.getenv("WALRUS_AGGREGATOR_URL", "https://agrregator.omura.fun").rstrip("/")

    query = "cat"
    print(f"\nQuerying search API: {api_url} for '{query}'...")
    try:
        payload = {
            "query": query,
            "top_k": 5,
            "exclude_nsfw": True
        }
        resp = httpx.post(api_url, json=payload, timeout=30.0)
        resp.raise_for_status()
        search_data = resp.json()
    except Exception as e:
        print(f"Error querying search API: {e}")
        sys.exit(1)

    results = search_data.get("results", [])
    print(f"Found {len(results)} search results.")

    if not results:
        print("No results returned from search. Make sure index contains relevant images.")
        sys.exit(1)

    print("\n--- Verifying Search Rank with Captioning ---")
    correct_count = 0
    for idx, item in enumerate(results, 1):
        blob_id = item["blob_id"]
        score = item["score"]
        # Fetch image bytes
        blob_url = f"{aggregator_url}/v1/blobs/{blob_id}"
        print(f"\n[Rank {idx}] Blob ID: {blob_id} (Score: {score:.2f})")
        print(f"Fetching from Walrus aggregator: {blob_url}")
        
        try:
            img_resp = httpx.get(blob_url, timeout=20.0)
            if img_resp.status_code != 200:
                print(f"❌ Failed to fetch blob: HTTP {img_resp.status_code}")
                continue
            
            # Load as Image
            image = Image.open(BytesIO(img_resp.content)).convert("RGB")
            
            # Generate caption
            inputs = processor(images=image, return_tensors="pt").to(device)
            out = model.generate(**inputs)
            caption = processor.decode(out[0], skip_special_tokens=True)
            print(f"Generated Caption: \"{caption}\"")
            
            # Validate if caption mentions query terms
            keywords = ["cat", "feline", "kitten", "kitty", "tabby"]
            matched = any(kw in caption.lower() for kw in keywords)
            if matched:
                print("✅ VALIDATION PASSED: Image caption matches query terms.")
                correct_count += 1
            else:
                print("⚠️ VALIDATION WARNING: Caption does not explicitly mention keywords.")
                
        except Exception as e:
            print(f"❌ Error processing result: {e}")

    print("\n--- Rank Validation Summary ---")
    print(f"Successfully processed and validated: {correct_count}/{len(results)} top results.")

if __name__ == "__main__":
    main()
