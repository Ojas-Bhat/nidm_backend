import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Create FAISS index (768 is embedding size for MiniLM)
dimension = 384
index = faiss.IndexFlatL2(dimension)

# Store mapping of ids to texts
doc_map = {}

def add_document(doc_id: str, text: str):
    """Add a document into FAISS index"""
    embedding = model.encode([text])
    index.add(np.array(embedding, dtype="float32"))
    doc_map[doc_id] = text

def search(query: str, top_k: int = 5):
    """Search documents in FAISS index"""
    embedding = model.encode([query])
    embedding = np.array(embedding, dtype="float32")
    distances, indices = index.search(embedding, top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        if idx != -1 and idx < len(doc_map):
            doc_id = list(doc_map.keys())[idx]
            results.append({"id": doc_id, "text": doc_map[doc_id], "score": float(distances[0][i])})
    return results
