from app.rag.vector_store import search_faiss
from app.rag.llm_answerer import generate_grounded_answer


def answer_question(file_id: str, question: str, history=None):
    if history is None:
        history = []

    top_chunks = search_faiss(file_id, question, top_k=3)

    if not top_chunks:
        return {
            "answer": "I could not find relevant information in the knowledge base.",
            "sources": []
        }

    answer = generate_grounded_answer(question, top_chunks, history)

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
