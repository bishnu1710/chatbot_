"""
build_index.py (improved)
-------------------------
Builds:
- models/faiss.index            (SBERT embeddings saved as FAISS index)
- models/meta.json             (texts, ids, metas)
- models/tfidf_vectorizer.pkl  (sklearn TfidfVectorizer)
- models/tfidf_matrix.npz      (sparse TF-IDF matrix saved)

Features / changes vs original:
- prefers combined_dataset_clean.json if present (falls back to combined_dataset.json)
- deduplicates text items and filters short/empty texts
- better error messages and debug prints
- command-line args for data/model paths and chunk sizes
- saves TF-IDF artifacts for lexical retrieval
"""

import argparse
import json
import os
import re
import sys
import traceback
from pathlib import Path
from typing import List, Dict, Any, Optional
import pickle

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
from tqdm.auto import tqdm

# -------------------- Defaults (can be overridden via CLI) --------------------
DEFAULT_DATA_DIR = Path("data")
DEFAULT_MODEL_DIR = Path("models")
DEFAULT_SBERT = os.environ.get("SBERT_MODEL", "all-mpnet-base-v2")
DEFAULT_CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 800))
DEFAULT_CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", 200))
DEFAULT_BATCH = int(os.environ.get("BATCH_SIZE", 64))
DEFAULT_TOP_K = int(os.environ.get("TOP_K", 4))

# -------------------- Helpers --------------------
def load_json_safe(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to parse JSON {path}: {e}")
        # print a short head for debug
        try:
            text = path.read_text(encoding="utf-8", errors="replace")[:1000]
            print("[DEBUG] file head:", text.replace("\n", " ")[:400])
        except Exception:
            pass
        return None

def simple_chunk_text(text: str, chunk_size=DEFAULT_CHUNK_SIZE, overlap=DEFAULT_CHUNK_OVERLAP) -> List[str]:
    if not text or not isinstance(text, str):
        return []
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end >= len(text):
            chunks.append(text[start:].strip())
            break
        window = text[start:end]
        # try split on sentence boundary
        last_punct = max(window.rfind("."), window.rfind("!"), window.rfind("?"), window.rfind("\n"))
        if last_punct > int(chunk_size * 0.5):
            cut = start + last_punct + 1
        else:
            cut = end
        chunk = text[start:cut].strip()
        chunks.append(chunk)
        start = max(cut - overlap, 0)
    return chunks

def gather_corpus(data_dir: Path) -> List[Dict[str, Any]]:
    """
    Load text documents from:
      - intents.json (patterns + responses)
      - combined_dataset_clean.json (preferred) or combined_dataset.json
      - CSV / parquet heuristics (best-effort)
    Returns list of {"id":..., "text":..., "source":..., "meta":...}
    """
    docs: List[Dict[str, Any]] = []
    intents_path = data_dir / "intents.json"
    comb_clean = data_dir / "combined_dataset_clean.json"
    comb_raw = data_dir / "combined_dataset.json"

    # 1) intents.json
    intents = load_json_safe(intents_path)
    if intents and isinstance(intents, dict):
        for intent in intents.get("intents", []):
            tag = intent.get("tag", "unknown")
            patterns = intent.get("patterns", []) or []
            responses = intent.get("responses", []) or []
            for i, p in enumerate(patterns):
                if isinstance(p, str) and p.strip():
                    docs.append({"id": f"intent::{tag}::pattern::{i}", "text": p.strip(), "source": "intents.json", "meta": {"tag": tag, "responses": responses}})
            for j, r in enumerate(responses):
                if isinstance(r, str) and r.strip():
                    docs.append({"id": f"intent::{tag}::response::{j}", "text": r.strip(), "source": "intents.json", "meta": {"tag": tag}})

    # 2) combined_dataset_clean.json preferred, else combined_dataset.json
    comb_source = None
    if comb_clean.exists():
        comb_source = comb_clean
    elif comb_raw.exists():
        comb_source = comb_raw

    if comb_source:
        comb = load_json_safe(comb_source)
        if comb is not None:
            if isinstance(comb, list):
                for i, rec in enumerate(comb):
                    txt = ""
                    if isinstance(rec, dict):
                        # heuristics for common text fields
                        for key in ("text", "content", "Context", "context", "body", "response", "reply"):
                            if key in rec and isinstance(rec[key], str) and len(rec[key].strip()) > 10:
                                txt = rec[key].strip()
                                break
                        if not txt:
                            # fallback: concat string values
                            txt = " ".join([v for v in rec.values() if isinstance(v, str)])[:4000].strip()
                    else:
                        txt = str(rec)[:4000].strip()
                    if txt:
                        docs.append({"id": f"combined::{i}", "text": txt, "source": comb_source.name, "meta": rec if isinstance(rec, dict) else {}})
            elif isinstance(comb, dict):
                text_parts = [v for v in comb.values() if isinstance(v, str) and len(v.strip()) > 20]
                joined = "\n\n".join(text_parts) if text_parts else json.dumps(comb)[:20000]
                docs.append({"id": "combined::full", "text": joined, "source": comb_source.name, "meta": {}})
        else:
            print(f"[WARN] combined source {comb_source} exists but couldn't be read - skipping.")
    else:
        print("[INFO] No combined_dataset file found (clean or raw).")

    # 3) CSV / Parquet heuristics (optional)
    csv_candidates = list(data_dir.glob("*.csv"))
    parquet_candidates = list(data_dir.glob("*.parquet")) + list(data_dir.glob("*.parq"))
    # load CSVs
    for p in csv_candidates + parquet_candidates:
        try:
            import pandas as pd
            if p.suffix.lower() == ".csv":
                df = pd.read_csv(p, dtype=str, nrows=5000)
            else:
                df = pd.read_parquet(p)
            # identify text-like columns (object dtype)
            text_cols = [c for c in df.columns if df[c].dtype == object][:3]
            if not text_cols:
                text_cols = df.columns[:1].tolist()
            for i, row in df.iterrows():
                texts = []
                for c in text_cols:
                    v = row.get(c)
                    if isinstance(v, str) and v.strip():
                        texts.append(v.strip())
                joined = " ".join(texts)[:4000].strip()
                if joined:
                    docs.append({"id": f"{p.name}::{i}", "text": joined, "source": p.name, "meta": {}})
            print(f"[INFO] Loaded {len(df)} rows from {p.name} (used cols: {text_cols})")
        except Exception as e:
            print(f"[WARN] Could not load {p.name}: {e}")

    return docs

def dedupe_and_filter(docs: List[Dict[str,Any]], min_words: int = 3) -> List[Dict[str,Any]]:
    """Remove duplicates (by normalized text) and very short texts."""
    seen = {}
    cleaned = []
    for doc in docs:
        text = doc.get("text", "")
        if not isinstance(text, str):
            continue
        # normalize
        norm = re.sub(r"\s+", " ", text).strip()
        # drop very short
        if len(norm.split()) < min_words:
            continue
        if norm in seen:
            # keep first occurrence (update meta if needed)
            continue
        seen[norm] = True
        doc["text"] = norm
        cleaned.append(doc)
    return cleaned

# -------------------- Main --------------------
def main(args):
    DATA_DIR = Path(args.data_dir)
    MODEL_DIR = Path(args.model_dir)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    FAISS_PATH = MODEL_DIR / "faiss.index"
    META_PATH = MODEL_DIR / "meta.json"
    TFIDF_VEC_PATH = MODEL_DIR / "tfidf_vectorizer.pkl"
    TFIDF_MATRIX_PATH = MODEL_DIR / "tfidf_matrix.npz"

    SBERT_MODEL = args.sbert_model
    CHUNK_SIZE = args.chunk_size
    CHUNK_OVERLAP = args.chunk_overlap
    BATCH_SIZE = args.batch_size

    print(f"[build] data_dir={DATA_DIR}, model_dir={MODEL_DIR}")
    print(f"[build] sbert_model={SBERT_MODEL}, chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}, batch={BATCH_SIZE}")

    # gather
    raw_docs = gather_corpus(DATA_DIR)
    print(f"[build] loaded {len(raw_docs)} raw documents (pre-filter)")

    # dedupe/filter
    raw_docs = dedupe_and_filter(raw_docs, min_words=3)
    print(f"[build] {len(raw_docs)} documents after dedup & short-text filtering")

    # chunk
    chunked = []
    for doc in raw_docs:
        chs = simple_chunk_text(doc["text"], chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
        for i, c in enumerate(chs):
            chunked.append({"id": f"{doc['id']}::chunk{i}", "text": c, "source": doc.get("source", ""), "meta": doc.get("meta", {})})
    print(f"[build] produced {len(chunked)} chunks")

    if not chunked:
        raise SystemExit("[build] No chunks produced - check your data files")

    # dedupe chunks (identical chunks)
    texts = []
    ids = []
    metas = []
    seen_texts = set()
    for c in chunked:
        t = c["text"].strip()
        if t in seen_texts:
            continue
        seen_texts.add(t)
        texts.append(t)
        ids.append(c["id"])
        metas.append(c.get("meta", {}))
    print(f"[build] {len(texts)} unique chunks after chunk deduplication")

    # ---- TF-IDF build (lexical retrieval) ----
    print("[build] building TF-IDF matrix...")
    vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=50000)
    tfidf_matrix = vectorizer.fit_transform(texts)  # sparse
    sparse.save_npz(TFIDF_MATRIX_PATH, tfidf_matrix)
    with open(TFIDF_VEC_PATH, "wb") as f:
        pickle.dump(vectorizer, f)
    print(f"[build] TF-IDF saved to {TFIDF_MATRIX_PATH}, vectorizer to {TFIDF_VEC_PATH}")

    # ---- SBERT embeddings & FAISS ----
    print(f"[build] loading SBERT model: {SBERT_MODEL}")
    try:
        embedder = SentenceTransformer(SBERT_MODEL)
    except Exception:
        print("[ERROR] Failed to load SentenceTransformer model. Traceback:")
        traceback.print_exc()
        sys.exit(1)
    embed_dim = embedder.get_sentence_embedding_dimension()
    print("[build] embed dim:", embed_dim)

    # encode in batches with tqdm
    all_embs = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="encoding"):
        batch = texts[i:i+BATCH_SIZE]
        # encode
        emb = embedder.encode(batch, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True)
        all_embs.append(emb)
    emb_matrix = np.vstack(all_embs).astype("float32")
    print("[build] embeddings shape:", emb_matrix.shape)

    # faiss - using inner product on normalized vectors equals cosine similarity
    index = faiss.IndexFlatIP(emb_matrix.shape[1])
    index.add(emb_matrix)
    faiss.write_index(index, str(FAISS_PATH))
    print("[build] FAISS saved to", FAISS_PATH)

    # save meta (texts, ids, metas)
    meta = {"ids": ids, "texts": texts, "metas": metas, "embed_dim": emb_matrix.shape[1], "sbert_model": SBERT_MODEL}
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("[build] metadata saved to", META_PATH)
    print("[build] done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build SBERT+FAISS index + TF-IDF for RAG")
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--model-dir", type=str, default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--sbert-model", type=str, default=DEFAULT_SBERT)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    args = parser.parse_args()
    main(args)
