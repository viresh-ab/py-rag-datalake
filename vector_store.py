import os
import pickle
import faiss
import numpy as np

# =========================
# PATHS
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FAISS_DIR = os.path.join(BASE_DIR, "data", "faiss")

INDEX_PATH = os.path.join(FAISS_DIR, "index.faiss")
META_PATH = os.path.join(FAISS_DIR, "metadata.pkl")

EMBEDDING_DIM = 3072  # text-embedding-3-large

os.makedirs(FAISS_DIR, exist_ok=True)

# =========================
# RESET INDEX
# =========================
def reset_index():
    index = faiss.IndexFlatL2(EMBEDDING_DIM)
    metadata = []

    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "wb") as f:
        pickle.dump(metadata, f)

# =========================
# ADD VECTORS (INGESTION)
# =========================
def add_vectors(vectors, metadatas):
    """
    vectors: List[List[float]]
    metadatas: List[dict]  -> must contain 'text' and 'source'
    """

    if not os.path.exists(INDEX_PATH):
        reset_index()

    index = faiss.read_index(INDEX_PATH)

    with open(META_PATH, "rb") as f:
        metadata = pickle.load(f)

    vecs = np.array(vectors).astype("float32")
    index.add(vecs)
    metadata.extend(metadatas)

    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "wb") as f:
        pickle.dump(metadata, f)

# =========================
# LOAD INDEX
# =========================
def load_index():
    if not os.path.exists(INDEX_PATH) or not os.path.exists(META_PATH):
        raise FileNotFoundError("Vector index not found")

    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "rb") as f:
        metadata = pickle.load(f)

    return index, metadata

# =========================
# SEARCH (RAG)
# =========================
def search(query_vector, top_k=5):
    index, metadata = load_index()

    query_vector = np.array([query_vector]).astype("float32")
    distances, indices = index.search(query_vector, top_k)

    results = []
    for score, idx in zip(distances[0], indices[0]):
        item = metadata[idx]
        item["score"] = float(score)
        results.append(item)

    return results
