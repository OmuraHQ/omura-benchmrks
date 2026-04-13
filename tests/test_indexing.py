import os
import requests
import io
import torch
import numpy as np
from omura.parsers.file_detection import detect_file_type
from omura.parsers.multimodal import (
    is_supported_image, is_supported_video, is_supported_audio, is_supported_doc,
    parse_pdf_content, parse_plain_text
)
from omura.utils.imagebind_embeddings import (
    generate_image_embedding,
    generate_video_embedding,
    generate_audio_embedding,
    generate_text_embedding,
    generate_multimodal_embedding
)

# Found web URLs for testing
SAMPLES = {
    "image": "https://upload.wikimedia.org/wikipedia/commons/7/70/Example.png",
    "video": "https://download.samplelib.com/mp4/sample-5s.mp4",
    "audio": "https://filesamples.com/samples/audio/mp3/sample3.mp3",
    "pdf": "https://sample-files.com/downloads/documents/pdf/sample-report.pdf",
    "text": "https://filesamples.com/samples/document/txt/sample3.txt"
}

def run_test():
    print(f"Testing embedding generation on {len(SAMPLES)} modalities...")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Count: {torch.cuda.device_count()}")
    
    for kind, url in SAMPLES.items():
        print(f"\n--- Testing {kind.upper()} ---")
        print(f"Fetching {url}...")
        
        try:
            # Use a browser-like user agent to avoid 403s
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            resp = requests.get(url, headers=headers, timeout=30)
            
            if resp.status_code != 200:
                print(f"Skipping {kind}: Failed to fetch (status {resp.status_code})")
                continue
                
            content = resp.content
            print(f"Downloaded {len(content)} bytes.")
            
            # 1. Detection
            mime, ext, detected_kind = detect_file_type(content)
            print(f"Detected: mime={mime}, ext={ext}, kind={detected_kind}")
            
            # 2. Embedding Generation
            emb = None
            if kind == "image" or detected_kind == "image":
                emb = generate_image_embedding(content)
            elif kind == "video" or detected_kind == "video":
                # Fallback ext if detection fails slightly
                ext_to_use = ext if ext else "mp4"
                emb = generate_video_embedding(content, ext_to_use, "test_video")
            elif kind == "audio" or detected_kind == "audio":
                ext_to_use = ext if ext else "mp3"
                emb = generate_audio_embedding(content, ext_to_use, "test_audio")
            elif kind == "pdf" or (detected_kind == "pdf" and ext == "pdf"):
                text, images = parse_pdf_content(content)
                print(f"Extracted PDF: {len(text)} chars, {len(images)} images")
                if text:
                    print(f"Snippet: {text[:100]}...")
                
                emb = generate_multimodal_embedding(text, images=images)
            elif kind == "text" or detected_kind == "text":
                text = parse_plain_text(content)
                print(f"Extracted Plain Text length: {len(text)} chars")
                if text:
                    print(f"Snippet: {text[:100]}...")
                    emb = generate_text_embedding(text[:1000], is_document=True)
            
            # 3. Verification
            if emb is not None:
                print(f"✅ SUCCESS: Generated embedding shape {emb.shape}")
                print(f"   First 5 values: {emb[:5]}")
            else:
                print(f"❌ FAILED: No embedding generated for {kind}")
                
        except Exception as e:
            print(f"❌ ERROR processing {kind}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    run_test()
