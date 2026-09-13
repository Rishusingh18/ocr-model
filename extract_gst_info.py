import os
import json
import glob
from pymongo import MongoClient

def extract_gst_info(rec_texts):
    data = {}
    
    for i, text in enumerate(rec_texts):
        # Clean text
        text = text.strip()
        
        if "Registration Number" in text or "GSTIN" in text:
            parts = text.split(":")
            if len(parts) > 1:
                data["GSTIN"] = parts[1].strip()
            elif i + 1 < len(rec_texts):
                data["GSTIN"] = rec_texts[i+1].strip()
                
        elif "Legal Name" in text:
            if i + 1 < len(rec_texts):
                data["Legal Name"] = rec_texts[i+1].strip()
                
        elif "Trade Name" in text:
            if i + 1 < len(rec_texts):
                data["Trade Name"] = rec_texts[i+1].strip()
                
        elif "Constitution of Business" in text:
            if i + 1 < len(rec_texts):
                data["Constitution of Business"] = rec_texts[i+1].strip()
                
        elif "Address of Principal Place" in text:
            address_lines = []
            j = i + 1
            # Capture until we hit the next section (which usually starts with "6." or "Date of Liability")
            while j < len(rec_texts):
                line = rec_texts[j].strip()
                if line.startswith("6.") or "Date of Liability" in line:
                    break
                # Only append non-empty valid address fields
                if line and not line.startswith("Business") and line != "Address":
                    address_lines.append(line)
                j += 1
            data["Address"] = ", ".join(address_lines)
            
        elif "Type of Registration" in text:
            if i + 1 < len(rec_texts):
                data["Type of Registration"] = rec_texts[i+1].strip()
                
        elif "Date of issue of Certificate" in text:
            if i + 1 < len(rec_texts):
                data["Date of Issue"] = rec_texts[i+1].strip()

    return data

def process_and_store():
    json_files = glob.glob(os.path.join('output', '*_page_1.json'))
    structured_data_list = []
    
    print(f"Found {len(json_files)} GST Certificate page 1 JSONs.")
    
    for jpath in json_files:
        with open(jpath, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
            
        rec_texts = raw_data.get("rec_texts", raw_data.get("rec_text", []))
        if not rec_texts:
            print(f"Warning: No 'rec_texts' found in {jpath}")
            continue
            
        structured_data = extract_gst_info(rec_texts)
        structured_data["source_file"] = os.path.basename(jpath)

        # Validate that a non-empty GSTIN was extracted before persisting
        if not structured_data.get("GSTIN"):
            print(f"Warning: Skipping {jpath} — no GSTIN found in OCR result: {structured_data}")
            continue

        structured_data_list.append(structured_data)
        print(f"Extracted data for {structured_data.get('GSTIN', 'Unknown')}")
        
    # Guard: do not overwrite existing output when nothing was successfully extracted
    if not json_files:
        print("No input JSON files found — skipping output write.")
        return
    if not structured_data_list:
        print("No valid records extracted (all missing GSTIN) — preserving existing output file.")
        return

    # Save structured JSON
    output_json = 'structured_gst_data.json'
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(structured_data_list, f, indent=4, ensure_ascii=False)
    print(f"\nSaved structured JSON to {output_json}")
    
    # Store to MongoDB
    print("\nAttempting to connect to local MongoDB...")
    try:
        client = MongoClient('mongodb://localhost:27017/', serverSelectionTimeoutMS=5000)
        # Verify connection
        client.server_info()
        print("Connected successfully!")
        
        db = client['gst_database']
        collection = db['gst_certificates']
        
        if structured_data_list:
            # We use insert_many for new documents
            # (In a production environment, you might use update_one with upsert=True based on GSTIN)
            result = collection.insert_many(structured_data_list)
            print(f"Successfully inserted {len(result.inserted_ids)} records into MongoDB 'gst_database.gst_certificates'.")
            
    except Exception as e:
        print("\nCould not connect to MongoDB. Is MongoDB running on your machine? (localhost:27017)")
        print(f"Error details: {e}")
        print("\nYour data is safely stored in 'structured_gst_data.json' and you can import it manually using MongoDB Compass.")

if __name__ == '__main__':
    process_and_store()
