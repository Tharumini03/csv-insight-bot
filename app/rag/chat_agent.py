from app.rag.vector_store import search_faiss
from app.rag.llm_answerer import generate_grounded_answer


def answer_question(file_id: str, question: str):
    """
    Retrieve the most relevant chunks using FAISS,
    then ask the local Ollama model to answer naturally.
    """
    top_chunks = search_faiss(file_id, question, top_k=3)

    if not top_chunks:
        return {
            "answer": "I could not find relevant information in the knowledge base.",
            "sources": []
        }

    answer = generate_grounded_answer(question, top_chunks)

    return {
        "answer": answer,
        "sources": top_chunks
    }
