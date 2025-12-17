import os
import faiss
import pickle
import numpy as np

# =========================
# CONFIG
# =========================
FAISS_DIR = "data/faiss"
INDEX_PATH = os.path.join(FAISS_DIR, "index.faiss")
META_PATH = os.path.join(FAISS_DIR, "meta.pkl")

# 🔑 EMBEDDING DIMENSION (text-embedding-3-large)
DIM = 3072

os.makedirs(FAISS_DIR, exist_ok=True)

# =========================
# CORE FUNCTIONS
# =========================
def save_index(index, meta):
    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "wb") as f:
        pickle.dump(meta, f)

def load_index():
    if not os.path.exists(INDEX_PATH) or not os.path.exists(META_PATH):
        index = faiss.IndexFlatL2(DIM)
        meta = []
        save_index(index, meta)
        return index, meta

    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)

    return index, meta

def add_vectors(vectors, metadata):
    index, meta = load_index()
    index.add(np.array(vectors).astype("float32"))
    meta.extend(metadata)
    save_index(index, meta)

# =========================
# RESET INDEX (IMPORTANT)
# =========================
def reset_index():
    """
    Completely deletes the existing vector DB
    and creates a fresh empty FAISS index.
    """
    if os.path.exists(INDEX_PATH):
        os.remove(INDEX_PATH)
    if os.path.exists(META_PATH):
        os.remove(META_PATH)

    index = faiss.IndexFlatL2(DIM)
    meta = []
    save_index(index, meta)

    return index, meta

# =========================
# SEARCH
# =========================
def search(query_vector, top_k=5):
    index, meta = load_index()
    D, I = index.search(
        np.array([query_vector]).astype("float32"),
        top_k
    )
    return [meta[i] for i in I[0]]
