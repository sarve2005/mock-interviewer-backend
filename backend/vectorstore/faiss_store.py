import faiss
import numpy as np
from fastembed import TextEmbedding

# Lazy load embedder
embedder = None

def get_embedder():
    global embedder
    if embedder is None:
        print("Loading FastEmbed model...")
        # BAAI/bge-small-en-v1.5 is a good default, or use "sentence-transformers/all-MiniLM-L6-v2"
        # FastEmbed defaults to "BAAI/bge-small-en-v1.5"
        embedder = TextEmbedding()
    return embedder

def build_store(chunks):
    # FastEmbed 'embed' returns a generator of np.ndarray
    model = get_embedder()
    embeddings_generator = model.embed(chunks)
    emb = np.array(list(embeddings_generator)).astype("float32")
    
    index = faiss.IndexFlatL2(emb.shape[1])
    index.add(emb)
    return index, chunks


def retrieve(query, index, chunks, k=5):
    model = get_embedder()
    # model.embed returns generator
    q_gen = model.embed([query])
    q = np.array(list(q_gen)).astype("float32")
    
    _, ids = index.search(q, k)
    return [chunks[i] for i in ids[0]]
