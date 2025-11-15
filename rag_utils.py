"""
rag_utils.py
------------
Hybrid retrieval (TF-IDF + SBERT) + prompt building + Gemini call.
"""

import os
import json
import pickle
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv
load_dotenv()

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse

# gemini
GENAI_AVAILABLE = False
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except Exception:
    GENAI_AVAILABLE = False

from crisis_handler import check_and_handle

# Paths & envs
MODEL_DIR = "models"
FAISS_PATH = os.path.join(MODEL_DIR, "faiss.index")
META_PATH = os.path.join(MODEL_DIR, "meta.json")
TFIDF_VEC_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
TFIDF_MATRIX_PATH = os.path.join(MODEL_DIR, "tfidf_matrix.npz")

SBERT_MODEL = os.environ.get("SBERT_MODEL", "all-mpnet-base-v2")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

TOP_K = int(os.environ.get("TOP_K", 4))
TFIDF_TOP_K = int(os.environ.get("TFIDF_TOP_K", 4))

# lazy singletons
_index = None
_meta = None
_embedder = None
_tfidf_vectorizer = None
_tfidf_matrix = None

def _load_meta_index():
    global _index, _meta, _embedder, _tfidf_vectorizer, _tfidf_matrix
    if not os.path.exists(META_PATH):
        raise FileNotFoundError(f"Meta file not found: {META_PATH}. Run build_index.py first.")
    if not os.path.exists(FAISS_PATH):
        raise FileNotFoundError(f"FAISS index not found: {FAISS_PATH}. Run build_index.py first.")
    # the rest as before...
    if _meta is None:
        with open(META_PATH, "r", encoding="utf-8") as f:
            _meta = json.load(f)
    if _index is None:
        _index = faiss.read_index(FAISS_PATH)
    if _embedder is None:
        _embedder = SentenceTransformer(SBERT_MODEL)
    if _tfidf_vectorizer is None:
        with open(TFIDF_VEC_PATH, "rb") as f:
            _tfidf_vectorizer = pickle.load(f)
    if _tfidf_matrix is None:
        _tfidf_matrix = sparse.load_npz(TFIDF_MATRIX_PATH)
    return _index, _meta, _embedder, _tfidf_vectorizer, _tfidf_matrix


def retrieve_sbert(query: str, k: int = TOP_K) -> List[Dict[str, Any]]:
    index, meta, embedder, _, _ = _load_meta_index()
    q_emb = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    sims, idxs = index.search(q_emb, k)
    sims = sims[0].tolist(); idxs = idxs[0].tolist()
    results = []
    for idx, score in zip(idxs, sims):
        if idx < 0 or idx >= len(meta["texts"]): continue
        results.append({"id": meta["ids"][idx], "text": meta["texts"][idx], "meta": meta.get("metas", [{}]*len(meta["texts"]))[idx], "score": float(score), "source": "sbert"})
    return results


def retrieve_tfidf(query: str, k: int = TFIDF_TOP_K) -> List[Dict[str, Any]]:
    _, meta, _, vectorizer, tfidf_matrix = _load_meta_index()
    q_vec = vectorizer.transform([query])  # sparse
    # compute cosine similarity (sparse * dense)
    sims = (tfidf_matrix @ q_vec.T).toarray().ravel()
    # length normalisation (cosine) with vectorizer output is already tfidf; best-effort:
    # find top k
    top_idxs = np.argsort(-sims)[:k]
    results = []
    for idx in top_idxs:
        score = float(sims[idx])
        if score <= 0:
            continue
        results.append({"id": meta["ids"][idx], "text": meta["texts"][idx], "meta": meta.get("metas", [{}]*len(meta["texts"]))[idx], "score": score, "source": "tfidf"})
    return results

def _normalize_scores(list_of_scores):
    arr = np.array(list_of_scores, dtype=float)
    if arr.max() == arr.min():
        return [float(x) for x in arr]
    arr = (arr - arr.min()) / (arr.max() - arr.min())
    return arr.tolist()


def hybrid_retrieve(query: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
    """
    Combine SBERT (semantic) + TF-IDF (lexical) results.
    Strategy: take union of top results, sort by normalized score (mix).
    """
    sbert_res = retrieve_sbert(query, k=top_k)
    tfidf_res = retrieve_tfidf(query, k=top_k)
    # merge by id, prefer higher normalized score
    candidates = {}
    for r in sbert_res:
        candidates[r["id"]] = r.copy()
    for r in tfidf_res:
        if r["id"] in candidates:
            # average scores (simple fusion)
            candidates[r["id"]]["score"] = (candidates[r["id"]]["score"] + r["score"]) / 2.0
            # keep source list
            candidates[r["id"]]["source"] = f"{candidates[r['id']]['source']},tfidf"
        else:
            candidates[r["id"]] = r.copy()
    merged = list(candidates.values())
    # sort by score desc
    merged.sort(key=lambda x: x["score"], reverse=True)
    return merged[:top_k]


def build_prompt(question: str, retrieved: List[Dict[str, Any]], history: Optional[List[Dict[str, str]]] = None) -> str:
    ctx = "\n\n".join([f"[{i+1}] (score={r['score']:.3f})\n{r['text']}" for i, r in enumerate(retrieved)])
    hist = ""
    if history:
        pairs = []
        for turn in history[-6:]:
            role = turn.get("role", "user"); txt = turn.get("text", "")
            pairs.append(f"{role.upper()}: {txt}")
        hist = "\n".join(pairs)
    prompt = f"""You are Sahaara, an empathetic mental health conversational assistant. Use ONLY the retrieved snippets as evidence. Do not invent facts.

=== Retrieved snippets ===
{ctx or '(none)'}

=== Recent conversation ===
{hist or '(none)'}

=== User question ===
{question}

Guidelines:
- Be empathetic and concise (2-5 short paragraphs).
- Don't give medical diagnosis. Encourage seeking professional help when appropriate.
- If the user is in crisis, recommend emergency services and helplines.
- If retrieved snippets conflict, note that.
- Cite the snippet numbers if you reference them (e.g. [1], [2]).
"""
    return prompt


# paste this into rag_utils.py replacing the old call_gemini()
def call_gemini(prompt: str, max_output_tokens: int = 512, temperature: float = 0.2, debug_save: bool = True) -> str:
    """
    Robust Gemini call with debug dumping and safe extraction.
    """
    if not GENAI_AVAILABLE:
        raise RuntimeError("google-generativeai not installed; pip install google-generativeai")
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY missing. Set it in your environment or .env")

    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(GEMINI_MODEL)

    def _extract_text(resp):
        debug = {"type": str(type(resp))}
        try:
            if hasattr(resp, "text") and resp.text:
                debug["source"] = "resp.text"
                return str(resp.text).strip(), debug
        except Exception as e:
            debug["resp_text_error"] = repr(e)
        # try dict-like extraction
        try:
            rdict = resp if isinstance(resp, dict) else getattr(resp, "__dict__", None) or {}
            debug["repr_keys"] = list(rdict.keys()) if isinstance(rdict, dict) else None
            # candidates -> content
            if isinstance(rdict, dict) and "candidates" in rdict and rdict["candidates"]:
                cand0 = rdict["candidates"][0]
                if isinstance(cand0, dict):
                    for key in ("content", "message", "output", "text"):
                        if key in cand0 and cand0[key]:
                            return str(cand0[key]).strip(), debug
                else:
                    return str(cand0).strip(), debug
            # outputs list heuristic
            if isinstance(rdict, dict) and "outputs" in rdict and rdict["outputs"]:
                out0 = rdict["outputs"][0]
                if isinstance(out0, dict):
                    for key in ("content", "text"):
                        if key in out0 and out0[key]:
                            return str(out0[key]).strip(), debug
            # fallback
            return str(resp).strip(), debug
        except Exception as e:
            debug["extract_error"] = repr(e)
            return "", debug

    import time, json, os, traceback
    os.makedirs(MODEL_DIR, exist_ok=True)
    for attempt in range(2):
        try:
            response = model.generate_content(
                prompt,
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": max_output_tokens,
                }
            )
            text, debug = _extract_text(response)
            # write debug line
            if debug_save:
                try:
                    dump = {"attempt": attempt, "debug": debug}
                    dump["repr"] = repr(response)[:3000]
                    with open(os.path.join(MODEL_DIR, "gemini_debug.json"), "a", encoding="utf-8") as fh:
                        fh.write(json.dumps(dump, ensure_ascii=False) + "\n")
                except Exception:
                    pass
            if text and text.strip():
                return text
            # retry small backoff
            if attempt == 0:
                time.sleep(0.6)
                continue
            return "[Gemini returned no text. See models/gemini_debug.json for raw response.]"
        except Exception as e:
            # log and retry once
            trace = traceback.format_exc()
            try:
                with open(os.path.join(MODEL_DIR, "gemini_debug.json"), "a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"error": repr(e), "trace": trace[:5000]}, ensure_ascii=False) + "\n")
            except Exception:
                pass
            if attempt == 0:
                time.sleep(0.8)
                continue
            return f"[Gemini API error] {e}"

def generate_response(user_query: str, history: Optional[List[Dict[str, str]]] = None, user_meta: Optional[Dict[str, Any]] = None, top_k: int = TOP_K) -> Dict[str, Any]:
    # 1) crisis check
    country_code = (user_meta or {}).get("country_code")
    crisis = check_and_handle(user_query, user_meta=user_meta, country_code=country_code)
    if crisis.get("is_crisis"):
        return {"answer": crisis["response"], "source": "crisis_handler", "retrieved": [], "llm_called": False, "crisis": True, "meta": crisis}

    # 2) hybrid retrieve
    retrieved = hybrid_retrieve(user_query, top_k)
    prompt = build_prompt(user_query, retrieved, history=history)

    # 3) call LLM
    try:
        answer = call_gemini(prompt)
        return {"answer": answer, "source": "llm", "retrieved": retrieved, "llm_called": True, "crisis": False}
    except Exception as e:
        return {"answer": f"[LLM call failed] {e}", "source": "error", "retrieved": retrieved, "llm_called": False, "crisis": False}
