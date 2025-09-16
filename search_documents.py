#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, psycopg2, numpy as np
import textwrap
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
GEMINI_API_KEY=os.getenv("GEMINI_API_KEY")
POSTGRES_URL=os.getenv("POSTGRES_URL")
assert GEMINI_API_KEY and POSTGRES_URL

genai.configure(api_key=GEMINI_API_KEY)
# ---- constants ----
EMBED_MODEL = "models/text-embedding-004"
GEN_MODEL   = "models/gemini-2.0-flash"
EMBED_DIM   = 768  # expected length for text-embedding-004

# ---- embedding ----
def embed(text: str) -> np.ndarray:
    resp = genai.embed_content(model=EMBED_MODEL, content=text)

    # Handle possible SDK response shapes:
    if isinstance(resp, dict) and "embedding" in resp:
        vec = resp["embedding"]                 # list[float]
    elif isinstance(resp, dict) and "embeddings" in resp:
        vec = resp["embeddings"][0]["values"]              # list[float]
    elif isinstance(resp, list):                           # list of dicts
        vec = resp[0]["embedding"]["values"]               # list[float]
    else:
        raise ValueError(f"Unexpected embedding response: {type(resp)} -> {resp}")

    v = np.asarray(vec, dtype=np.float64).ravel()          # (768,)
    if v.shape[0] != EMBED_DIM:
        raise ValueError(f"Bad embedding length: {v.shape[0]} (expected {EMBED_DIM})")
    return v

# ---- cosine ----
def cosine(a, b) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.shape[0] != b.shape[0]:
        return float("-inf")  # skip mismatched rows gracefully
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0

# ---- DB fetch ----
def fetch_chunks():
    sql = "SELECT id, filename, split_strategy, chunk_text, embedding FROM chunks"
    with psycopg2.connect(POSTGRES_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()

    out = []
    for id_, fn, strat, text, emb in rows:
        vec = np.asarray(emb, dtype=np.float64).ravel()
        # filter out corrupt rows (wrong dimension)
        if vec.shape[0] != EMBED_DIM:
            continue
        out.append({
            "id": id_,
            "filename": fn,
            "split_strategy": strat,
            "chunk_text": text,
            "embedding": vec,  # already 1-D
        })
    return out

# ---- retrieval ----
def top_k(query: str, k: int = 5):
    q = embed(query)                 # (768,)
    rows = fetch_chunks()            # each embedding (768,)
    for r in rows:
        r["score"] = cosine(q, r["embedding"])
    rows.sort(key=lambda x: x["score"], reverse=True)
    rows = get_unique_dicts(rows)
    return rows[:k]

def get_unique_dicts(rows):
    seen = set()
    unique_dicts = []
    for d in rows:
        if d["filename"] not in seen:
            unique_dicts.append(d)
            seen.add(d["filename"])
    return unique_dicts

if __name__=="__main__":
    query = input("Enter your query: ")

    hits=top_k(query, k=5)
    print("\n=== TOP 5 ===")
    for i,h in enumerate(hits,1):
        print(f"[{i}] {h['filename']} | {h['split_strategy']} | score={h['score']:.4f}")
        print(textwrap.shorten(h["chunk_text"].replace("\n"," "), width=200))
        print("-"*70)