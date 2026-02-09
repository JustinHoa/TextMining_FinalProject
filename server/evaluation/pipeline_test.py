from datasets import load_from_disk

import requests

import json
from tqdm import tqdm

ds = load_from_disk("../data/vifactcheck-normalized")

URL = "http://localhost:8000/check"
OUTPUT_PATH = "results.json"

def test_pipeline(claim):
    payload = {"claim": claim}
    try:
        response = requests.post(URL, json=payload)
        if response.status_code == 200:
            return {
                "status": "success",
                "response": response.json()
            }
        else:
            return {
                "status": "failed",
                "http_status": response.status_code,
                "error": response.text
            }
    except requests.exceptions.ConnectionError:
        return {
            "status": "error",
            "error": "Could not connect to server"
        }
    except requests.exceptions.Timeout:
        return {
            "status": "error",
            "error": "Request timeout"
        }

results = []
for idx, sample in enumerate(tqdm(ds.select(range(200)))):  # 200 samples
    claim = sample["Statement"]
    output = test_pipeline(claim)
    results.append({
        "id": idx,
        "output": output,
        **output
    })

# Export to JSON
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"Saved to {OUTPUT_PATH}")