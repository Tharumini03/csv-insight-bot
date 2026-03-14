import os
import json
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"


def load_embedding_model():
    return SentenceTransformer(MODEL_NAME)


def build_faiss_index(file_id: str, chunks_path: str):
    """
    Build FAISS index from chunks.json and save it.
    """
    output_dir = f"app/storage/outputs/{file_id}/faiss_store"
    os.makedirs(output_dir, exist_ok=True)

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    texts = [chunk["text"] for chunk in chunks]

    model = load_embedding_model()
    embeddings = model.encode(texts, convert_to_numpy=True)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype("float32"))

    faiss.write_index(index, os.path.join(output_dir, "index.faiss"))

    with open(os.path.join(output_dir, "chunks.pkl"), "wb") as f:
        pickle.dump(chunks, f)

    return output_dir


def search_faiss(file_id: str, query: str, top_k: int = 3):
    """
    Search the FAISS index using the query embedding.
    """
    store_dir = f"app/storage/outputs/{file_id}/faiss_store"

    index_path = os.path.join(store_dir, "index.faiss")
    chunks_path = os.path.join(store_dir, "chunks.pkl")

    index = faiss.read_index(index_path)

    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)

    model = load_embedding_model()
    query_embedding = model.encode([query], convert_to_numpy=True).astype("float32")

    distances, indices = index.search(query_embedding, top_k)

    results = []
    for rank, idx in enumerate(indices[0]):
        if idx == -1:
            continue
        results.append({
            "chunk_id": chunks[idx]["chunk_id"],
            "text": chunks[idx]["text"],
            "score": float(distances[0][rank])
        })

    return results
