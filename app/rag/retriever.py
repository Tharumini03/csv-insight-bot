import json
import re


def tokenize(text: str):
    """
    Simple tokenizer:
    lowercase + words only
    """
    return re.findall(r"\b\w+\b", text.lower())


def score_chunk(query: str, chunk_text: str):
    """
    Very simple keyword overlap scoring.
    """
    query_words = set(tokenize(query))
    chunk_words = set(tokenize(chunk_text))

    return len(query_words.intersection(chunk_words))


def retrieve_top_chunks(file_id: str, query: str, top_k: int = 3):
    """
    Load chunks.json and return the top matching chunks for the query.
    """
    chunks_path = f"app/storage/outputs/{file_id}/chunks.json"

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    scored = []
    for chunk in chunks:
        score = score_chunk(query, chunk["text"])
        scored.append({
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"],
            "score": score
        })

    scored.sort(key=lambda x: x["score"], reverse=True)

    return scored[:top_k]
