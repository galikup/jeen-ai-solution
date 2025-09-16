#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, argparse, textwrap
import psycopg2, numpy as np

import os, sys, argparse, textwrap
import psycopg2, numpy as np
from dotenv import load_dotenv
import google.generativeai as genai

# ---- init ----
# ---- init ----
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
POSTGRES_URL   = os.getenv("POSTGRES_URL")
assert GEMINI_API_KEY, "Missing GEMINI_API_KEY in .env"
assert POSTGRES_URL,   "Missing POSTGRES_URL in .env"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
POSTGRES_URL   = os.getenv("POSTGRES_URL")
assert GEMINI_API_KEY, "Missing GEMINI_API_KEY in .env"
assert POSTGRES_URL,   "Missing POSTGRES_URL in .env"

genai.configure(api_key=GEMINI_API_KEY)


EMBED_MODEL = "models/text-embedding-004"
GEN_MODEL   = "models/gemini-2.0-flash"
EMBED_DIM   = 768  # text-embedding-004
EMBED_DIM   = 768  # text-embedding-004

# ---- embeddings ----
# ---- embeddings ----
def embed(text: str) -> np.ndarray:
    resp = genai.embed_content(model=EMBED_MODEL, content=text)
    # Supports several possible formats:
    if isinstance(resp, dict) and "embedding" in resp:
        vec = resp["embedding"]
        vec = resp["embedding"]
    elif isinstance(resp, dict) and "embeddings" in resp:
        vec = resp["embeddings"][0]["values"]
    elif isinstance(resp, list):
        vec = resp[0]["embedding"]["values"]
        vec = resp["embeddings"][0]["values"]
    elif isinstance(resp, list):
        vec = resp[0]["embedding"]["values"]
    else:
        raise ValueError(f"Unexpected embedding response: {type(resp)} -> {resp}")
    v = np.asarray(vec, dtype=np.float64).ravel()
    v = np.asarray(vec, dtype=np.float64).ravel()
    if v.shape[0] != EMBED_DIM:
        raise ValueError(f"Bad embedding length: {v.shape[0]} (expected {EMBED_DIM})")
    return v

def cosine(a, b) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0

# ---- DB ----
# ---- DB ----
def fetch_chunks():
    sql = "SELECT id, filename, split_strategy, chunk_text, embedding FROM chunks"
    with psycopg2.connect(POSTGRES_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()
    out = []
    for id_, fn, strat, text, emb in rows:
        vec = np.asarray(emb, dtype=np.float64).ravel()
        if vec.shape[0] != EMBED_DIM:
            continue
        out.append({
            "id": id_,
            "filename": fn,
            "split_strategy": strat,
            "chunk_text": text,
            "embedding": vec
            "embedding": vec
        })
    return out

# ---- retrieval ----
def top_k(query: str, k: int = 5, max_per_file: int = 2):
    q = embed(query)
    rows = fetch_chunks()
def top_k(query: str, k: int = 5, max_per_file: int = 2):
    q = embed(query)
    rows = fetch_chunks()
    for r in rows:
        r["score"] = cosine(q, r["embedding"])
    rows.sort(key=lambda x: x["score"], reverse=True)

    # Supports several possible formats:
    seen = set()
    per_file_count = {}
    results = []
    for r in rows:
        key = (r["filename"], r["chunk_text"].strip())
        if key in seen:
            continue
        if per_file_count.get(r["filename"], 0) >= max_per_file:
            continue
        seen.add(key)
        per_file_count[r["filename"]] = per_file_count.get(r["filename"], 0) + 1
        results.append(r)
        if len(results) >= k:
            break
    return results

def llm_answer(query: str, contexts: list[dict]) -> str:
    ctx = "\n\n---\n\n".join(textwrap.shorten(c["chunk_text"].replace("\n", " "), width=800)
                              for c in contexts)
    prompt = f"""You are a concise assistant. Answer ONLY from the CONTEXT below.
If information is missing, say so briefly.

CONTEXT:
{ctx}

QUESTION:
{query}

Return 4-6 short bullet points, actionable and clear."""
    model = genai.GenerativeModel(GEN_MODEL)
    resp = model.generate_content(prompt)
    return (resp.text or "").strip()

# ---- CLI ----
def main():
    ap = argparse.ArgumentParser(description="Semantic search over indexed chunks")
    ap.add_argument("--k", type=int, default=5, help="number of results to show")
    ap.add_argument("--max-per-file", type=int, default=2, help="limit results per single file")
    ap.add_argument("--answer", action="store_true", help="ask Gemini to synthesize a short answer")
    args = ap.parse_args()

    try:
        query = input("Enter your query: ")
    except KeyboardInterrupt:
        print("\nAborted.")
        sys.exit(1)

    if not query.strip():
        print("Empty query. Bye.")
        return

    hits = top_k(query, k=args.k, max_per_file=args.max_per_file)
    if not hits:
        print("No data found. Did you index files?")
        return

    print("\n=== TOP RESULTS ===")
    for i, h in enumerate(hits, 1):
        snippet = textwrap.shorten(h["chunk_text"].replace("\n", " "), width=140)
        print(f"[{i}] {h['filename']} | {h['split_strategy']} | score={h['score']:.4f}")
        print(snippet)
        print("-" * 70)

    if args.answer:
        print("\n=== SYNTHESIZED ANSWER ===\n")
        try:
            print(llm_answer(query, hits))
        except Exception as e:
            print(f"(LLM error) {e}")

if __name__ == "__main__":
    main()
        print(snippet)
        print("-" * 70)

    if args.answer:
        print("\n=== SYNTHESIZED ANSWER ===\n")
        try:
            print(llm_answer(query, hits))
        except Exception as e:
            print(f"(LLM error) {e}")

if __name__ == "__main__":
    main()
