import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("all-MiniLM-L6-v2")

def build_store(chunks):
    emb = embedder.encode(chunks).astype("float32")
    index = faiss.IndexFlatL2(emb.shape[1])
    index.add(emb)
    return index, chunks


def retrieve(query, index, chunks, k=5):
    q = embedder.encode([query]).astype("float32")
    _, ids = index.search(q, k)
    return [chunks[i] for i in ids[0]]
