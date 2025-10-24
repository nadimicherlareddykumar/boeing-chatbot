# app/indexer.py
# app/indexer.py
# Chunking -> embedding -> Chroma indexing and search (local persistent)
# Requires chromadb

from typing import List, Dict, Any
import os

import chromadb
from chromadb.config import Settings

from .embeddings import embed_texts

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "boeing_insight")

_client: chromadb.Client | None = None

def _get_client() -> chromadb.Client:
    global _client
    if _client is None:
        os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
        _client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=CHROMA_PERSIST_DIR))
    return _client

def _chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return chunks

def add_documents(docs: List[Dict[str, Any]], persist: bool = True):
    """
    docs: list of {'id':..., 'text':..., 'meta':{...}}
    Splits text into chunks and adds to Chroma.
    """
    client = _get_client()
    coll = client.get_or_create_collection(name=COLLECTION_NAME)

    ids = []
    documents = []
    metadatas = []
    for doc in docs:
        base_id = doc.get("id") or doc.get("name") or str(hash(doc.get("text","")))
        meta = doc.get("meta", {})
        text = doc.get("text", "")
        chunks = _chunk_text(text)
        for i, chunk in enumerate(chunks):
            chunk_id = f"{base_id}__{i}"
            ids.append(chunk_id)
            documents.append(chunk)
            m = dict(meta)
            m.update({"source_id": base_id, "chunk_index": i})
            metadatas.append(m)
    if not documents:
        return {"added": 0}
    embeddings = embed_texts(documents)
    coll.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
    if persist:
        client.persist()
    return {"added": len(documents)}

def search(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    client = _get_client()
    coll = client.get_or_create_collection(name=COLLECTION_NAME)
    q_embed = embed_texts([query])[0]
    res = coll.query(query_embeddings=[q_embed], n_results=top_k, include=["documents", "metadatas", "distances"])
    results = []
    if res and "documents" in res and len(res["documents"]) > 0:
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res.get("distances", [[]])[0]
        for doc, meta, dist in zip(docs, metas, dists):
            results.append({"text": doc, "meta": meta, "distance": float(dist)})
    return results
