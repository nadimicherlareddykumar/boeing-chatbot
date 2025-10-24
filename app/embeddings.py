# app/embeddings.py
# sentence-transformers wrapper for fast free embeddings

from typing import List
import os
from sentence_transformers import SentenceTransformer

_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
_model: SentenceTransformer | None = None

def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)
    return _model

def embed_texts(texts: List[str]) -> List[List[float]]:
    model = get_model()
    return model.encode(texts, show_progress_bar=False, convert_to_numpy=True).tolist()
