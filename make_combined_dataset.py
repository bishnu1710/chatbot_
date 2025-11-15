"""
make_combined_dataset.py
------------------------
Builds a new combined_dataset.json by merging clean text from:
- data/intents.json
- any CSV files in data/ (like counsel_chat.csv)
- any JSONL or JSON files (optional)
Removes duplicates and empty entries.
"""

import json
import pandas as pd
from pathlib import Path
import re

DATA_DIR = Path("data")
OUTPUT_PATH = DATA_DIR / "combined_dataset_clean.json"

def normalize_text(t: str) -> str:
    """Basic text cleanup."""
    t = str(t).replace("\n", " ").replace("\r", " ")
    t = re.sub(r"\s+", " ", t).strip()
    return t

def load_intents():
    out = []
    p = DATA_DIR / "intents.json"
    if not p.exists():
        return out
    try:
        obj = json.load(open(p, "r", encoding="utf-8"))
        for intent in obj.get("intents", []):
            tag = intent.get("tag", "")
            for ptn in intent.get("patterns", []):
                out.append(normalize_text(ptn))
            for rsp in intent.get("responses", []):
                out.append(normalize_text(rsp))
    except Exception as e:
        print("[warn] could not load intents:", e)
    return out

def load_csv_texts():
    out = []
    for csvfile in DATA_DIR.glob("*.csv"):
        try:
            df = pd.read_csv(csvfile, dtype=str, nrows=5000)
            text_cols = [c for c in df.columns if df[c].dtype == object][:3]
            for _, row in df.iterrows():
                joined = " ".join(
                    [str(row[c]) for c in text_cols if isinstance(row[c], str)]
                )
                joined = normalize_text(joined)
                if len(joined) > 10:
                    out.append(joined)
            print(f"[+] loaded {len(out)} from {csvfile.name}")
        except Exception as e:
            print(f"[warn] skip {csvfile}: {e}")
    return out

def build_combined():
    texts = []
    texts += load_intents()
    texts += load_csv_texts()

    # remove duplicates
    uniq = list(dict.fromkeys(t for t in texts if t and len(t.split()) > 3))
    print(f"[info] {len(uniq)} unique text entries retained (from {len(texts)} total).")

    records = [{"id": f"t{i}", "text": t} for i, t in enumerate(uniq)]
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"[done] Clean dataset saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    build_combined()
