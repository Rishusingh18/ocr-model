import json
import os

json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output', 'AA0904261362093_RC18042026_page_1.json')
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

print("Keys:", list(data.keys()))
if 'rec_texts' in data:
    for i, text in enumerate(data['rec_texts']):
        print(f"{i}: {text}")
elif 'rec_text' in data:
    for i, text in enumerate(data['rec_text']):
        print(f"{i}: {text}")
