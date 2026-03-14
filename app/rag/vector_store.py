import os
import json
import pickle
import faiss
import numpy as np
import requests

OLLAMA_BASE_URL = "http://127.0.0.1:11434"
OLLAMA_EMBED_MODEL = "embeddinggemma"


def get_ollama_embeddings(texts):
    """
    Generate embeddings using Ollama's /api/embed endpoint.
    """
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/embed",
        json={
            "model": OLLAMA_EMBED_MODEL,
            "input": texts
        },
        timeout=120
    )
    response.raise_for_status()

    data = response.json()
    return np.array(data["embeddings"], dtype="float32")


def build_faiss_index(file_id: str, chunks_path: str):
    """
    Build FAISS index from chunks.json and save it.
    """
    output_dir = f"app/storage/outputs/{file_id}/faiss_store"
    os.makedirs(output_dir, exist_ok=True)

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    texts = [chunk["text"] for chunk in chunks]
    embeddings = get_ollama_embeddings(texts)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

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

    query_embedding = get_ollama_embeddings([query])

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
