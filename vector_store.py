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

# =========================
# LOAD INDEX + METADATA
# =========================
if not os.path.exists(INDEX_PATH):
    raise RuntimeError("❌ FAISS index not found. Run ingest.py first.")

index = faiss.read_index(INDEX_PATH)

with open(META_PATH, "rb") as f:
    metadata = pickle.load(f)

# =========================
# SEARCH
# =========================
def search(query_vector, top_k=5):
    query_vector = np.array([query_vector]).astype("float32")

    distances, indices = index.search(query_vector, top_k)

    results = []
    for score, idx in zip(distances[0], indices[0]):
        item = metadata[idx]
        item["score"] = float(score)
        results.append(item)

    return results
