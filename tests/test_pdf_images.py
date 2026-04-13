import io
import requests
from pypdf import PdfReader
from PIL import Image

url = "https://sample-files.com/downloads/documents/pdf/sample-report.pdf"

def test_pdf_extraction():
    print(f"Fetching {url}...")
    headers = {'User-Agent': 'Mozilla/5.0'}
    resp = requests.get(url, headers=headers)
    data = io.BytesIO(resp.content)
    
    reader = PdfReader(data)
    print(f"PDF has {len(reader.pages)} pages.")
    
    total_text = 0
    total_images = 0
    
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        total_text += len(text)
        
        # pypdf image extraction
        images = page.images
        print(f"Page {i+1}: {len(text)} chars, {len(images)} images")
        
        for img_file_obj in images:
            try:
                # img_file_obj is a File object with .data, .name, .image (PIL)
                # In newer pypdf, .image property returns PIL image
                img = img_file_obj.image
                print(f"  - Image: {img.format} {img.size} mode={img.mode}")
                total_images += 1
            except Exception as e:
                print(f"  - Failed to process image: {e}")

    print(f"\nSummary: {total_text} chars, {total_images} images found.")

if __name__ == "__main__":
    test_pdf_extraction()
