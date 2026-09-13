import os
from paddleocr import PaddleOCR
import json

def process_gst_pdfs():
    # Initialize PaddleOCR
    # lang='en' for English
    # use_textline_orientation=True to automatically detect and rotate image text
    ocr = PaddleOCR(use_textline_orientation=True, lang='en') 

    data_dir = 'gst'
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(data_dir):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(data_dir, filename)
            print(f"Processing {pdf_path}...")
            
            # Run OCR on the PDF.
            try:
                # use predict instead of ocr to avoid deprecation warnings
                result = list(ocr.predict(pdf_path))
            except Exception as e:
                print(f"Error predicting {pdf_path}: {e}")
                continue
            
            # Format the output and Save
            for idx, page in enumerate(result):
                output_filename = f"{os.path.splitext(filename)[0]}_page_{idx+1}.json"
                output_path = os.path.join(output_dir, output_filename)
                
                # In PaddleX 3.0, OCRResult objects have save_to_json method
                if hasattr(page, 'save_to_json'):
                    page.save_to_json(output_path)
                    print(f"Saved results to {output_path}")
                else:
                    # Fallback: write a .txt diagnostic so the output dir never
                    # gets a malformed JSON file (str(page) is not valid JSON).
                    txt_path = output_path.replace('.json', '_raw.txt')
                    with open(txt_path, 'w', encoding='utf-8') as f:
                        f.write(str(page))
                    print(f"Saved raw diagnostic text to {txt_path} (no save_to_json available)")

if __name__ == '__main__':
    process_gst_pdfs()
