from app.rag.vector_store import search_faiss
from app.rag.llm_answerer import generate_grounded_answer


def answer_question(file_id: str, question: str):
    """
    Retrieve relevant chunks, generate a natural answer, and return source chunks.
    """
    top_chunks = search_faiss(file_id, question, top_k=3)

    if not top_chunks:
        return {
            "answer": "I could not find relevant information in the knowledge base.",
            "sources": []
        }

    answer = generate_grounded_answer(question, top_chunks)

    # Keep only clean source fields
    clean_sources = []
    for chunk in top_chunks:
        clean_sources.append({
            "chunk_id": chunk["chunk_id"],
            "score": chunk["score"],
            "text": chunk["text"]
        })

    return {
        "answer": answer,
        "sources": clean_sources
    }
