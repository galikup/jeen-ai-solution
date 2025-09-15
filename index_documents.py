#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, re, psycopg2
from dotenv import load_dotenv
import google.generativeai as genai
from pypdf import PdfReader
from docx import Document

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
POSTGRES_URL   = os.getenv("POSTGRES_URL")
assert GEMINI_API_KEY, "Missing GEMINI_API_KEY in .env"
assert POSTGRES_URL, "Missing POSTGRES_URL in .env"

genai.configure(api_key=GEMINI_API_KEY)
EMBED_MODEL = "models/text-embedding-004"

def read_text(path:str)->str:
    ext = os.path.splitext(path)[1].lower()
    if ext==".pdf":
        r=PdfReader(path)
        return "\n".join((p.extract_text() or "") for p in r.pages)
    if ext==".docx":
        d=Document(path)
        return "\n".join(p.text for p in d.paragraphs)
    raise ValueError("Use PDF/DOCX")

def clean(t:str)->str:
    t=t.replace("\x00"," ")
    t=re.sub(r"[ \t]+"," ",t)
    t=re.sub(r"\n{3,}","\n\n",t)
    return t.strip()

def split_paragraphs(t:str,max_len=1000):
    paras=[p.strip() for p in re.split(r"\n\s*\n", t) if p.strip()]
    out=[]; cur=""
    for p in paras:
        if len(cur)+len(p)+2<=max_len: cur=(cur+"\n\n"+p).strip()
        else: 
            if cur: out.append(cur)
            cur=p
    if cur: out.append(cur)
    return out

def embed_batch(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []

    embs = []
    for t in texts:
        resp = genai.embed_content(model=EMBED_MODEL, content=t)
        # Response format: {"embedding": {"values": [...]}}
        embs.append(normalize_embeddings(resp))
    return embs

def normalize_embeddings(resp):
    if isinstance(resp, dict):
        if "embedding" in resp:  # single
            return [resp["embedding"]]
        if "embeddings" in resp:  # batch
            return [e["values"] for e in resp["embeddings"]]
    elif isinstance(resp, list):  # SDK quirks
        return [item["embedding"]["values"] for item in resp]
    raise ValueError(f"Unexpected embedding response format: {resp}")

def ensure_table():
    ddl="""
    CREATE TABLE IF NOT EXISTS chunks (
      id SERIAL PRIMARY KEY,
      filename TEXT,
      split_strategy TEXT,
      chunk_text TEXT,
      embedding FLOAT8[],
      created_at TIMESTAMP DEFAULT NOW()
    );"""
    with psycopg2.connect(POSTGRES_URL) as conn:
        with conn.cursor() as cur: cur.execute(ddl); conn.commit()

def insert_rows(filename, strategy, chunks, embs):
    sql="INSERT INTO chunks(filename,split_strategy,chunk_text,embedding) VALUES(%s,%s,%s,%s)"
    rows=[(filename, strategy, c, e) for c,e in zip(chunks,embs)]
    with psycopg2.connect(POSTGRES_URL) as conn:
        with conn.cursor() as cur: cur.executemany(sql, rows); conn.commit()

def index_file(path:str):
    print(f"[INFO] reading {path}")
    text = clean(read_text(path))
    chunks = split_paragraphs(text, max_len=1000)
    print(f"[INFO] {len(chunks)} chunks")
    ensure_table()
    B=32
    for i in range(0,len(chunks),B):
        batch=chunks[i:i+B]
        embs=embed_batch(batch)
        insert_rows(os.path.basename(path), "paragraph_based", batch, embs)
        print(f"[OK] stored {i+len(batch)}/{len(chunks)}")
    print("[DONE] indexing complete")

if __name__ == "__main__":
    # path = input("Enter PDF/DOCX path: ")   # read string from terminal
    path_base = "test_med_files/doc_"
    path_suffix = ".pdf"
    for i in range(1,21):
        filename = f"{path_base}{i:02d}{path_suffix}"
        index_file(filename)
