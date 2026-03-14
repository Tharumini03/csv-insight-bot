from app.rag.vector_store import search_faiss


def answer_question(file_id: str, question: str):
    """
    Retrieve the most relevant chunks using FAISS and build a simple answer.
    """
    top_chunks = search_faiss(file_id, question, top_k=3)

    if not top_chunks:
        return {
            "answer": "I could not find relevant information in the knowledge base.",
            "sources": []
        }

    combined_text = "\n\n".join([c["text"] for c in top_chunks])

    answer = (
        "Here is the most relevant information I found:\n\n"
        f"{combined_text}"
    )

    return {
        "answer": answer,
        "sources": top_chunks
    }
