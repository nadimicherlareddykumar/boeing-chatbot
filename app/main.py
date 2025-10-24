# app/main.py
# app/main.py
# FastAPI app exposing /ingest and /chat endpoints.
# /ingest: POST {"drive_folder_id": "..."} will download files from Drive and index them.
# /chat: POST {"message":"..."} returns top-K contexts and — if OPENAI_API_KEY is set — a generated reply.

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import logging
from typing import List, Optional

from .gdrive import download_folder
from .indexer import add_documents, search

import pathlib

app = FastAPI(title="Boeing Insight Chatbot (GoogleDrive + Free Embeddings)")

logger = logging.getLogger("uvicorn")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class IngestRequest(BaseModel):
    drive_folder_id: str

class ChatRequest(BaseModel):
    message: str
    top_k: Optional[int] = 5

class ChatResponse(BaseModel):
    reply: str
    sources: List[dict] = []

@app.post("/ingest")
def ingest(req: IngestRequest):
    if not req.drive_folder_id:
        raise HTTPException(status_code=400, detail="drive_folder_id required")
    out_dir = os.getenv("DRIVE_DOWNLOAD_DIR", "data/drive")
    downloaded = download_folder(req.drive_folder_id, out_dir=out_dir)
    # Read downloaded files and prepare docs list for indexer
    docs = []
    for d in downloaded:
        try:
            text = pathlib.Path(d["path"]).read_text(encoding="utf-8", errors="ignore")
        except Exception:
            logger.exception("failed reading %s", d.get("path"))
            text = ""
        docs.append({"id": d["id"], "text": text, "meta": {"name": d.get("name"), "mimeType": d.get("mimeType"), "path": d.get("path")}})
    res = add_documents(docs)
    return {"ok": True, "downloaded": len(downloaded), "indexed_chunks": res.get("added", 0)}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.message or req.message.strip() == "":
        raise HTTPException(status_code=400, detail="Empty message")
    top_k = req.top_k or 5
    hits = search(req.message, top_k=top_k)
    # Build context
    contexts = []
    sources = []
    for h in hits:
        contexts.append(h["text"])
        m = h.get("meta", {})
        sources.append({"source_id": m.get("source_id"), "chunk_index": m.get("chunk_index"), "meta": m})
    context_text = "\n\n---\n\n".join(contexts) if contexts else "No context found."

    # If OPENAI_API_KEY is configured, call OpenAI to produce a final answer; otherwise return the contexts.
    if OPENAI_API_KEY:
        try:
            import openai
            openai.api_key = OPENAI_API_KEY
            system = "You are Boeing Insight assistant. Use the provided context and cite sources. If unsure, say you don't know."
            prompt = f"{system}\n\nContext:\n{context_text}\n\nUser question: {req.message}\n\nProvide a concise answer and include source citations in the format [source_id:chunk_index]."
            resp = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role":"system","content":system},{"role":"user","content":prompt}], max_tokens=512, temperature=0.0)
            reply = resp["choices"][0]["message"]["content"].strip()
        except Exception:
            logger.exception("OpenAI call failed")
            reply = f"(generation failed) Here is retrieved context:\n\n{context_text}"
    else:
        # No generation available: return retrieved context as "reply"
        reply = f"Retrieved context (no generation configured):\n\n{context_text}"

    return ChatResponse(reply=reply, sources=sources)