import requests

OLLAMA_BASE_URL = "http://127.0.0.1:11434"
OLLAMA_CHAT_MODEL = "gemma3"


def generate_grounded_answer(question: str, retrieved_chunks: list):
    """
    Use a local Ollama model to answer naturally,
    but only from the retrieved context.
    """
    if not retrieved_chunks:
        return "I could not find relevant information in the knowledge base."

    context = "\n\n".join(
        [f"[Chunk {c['chunk_id']}]\n{c['text']}" for c in retrieved_chunks]
    )

    system_prompt = """
You are a helpful data analysis assistant.

Answer the user's question ONLY using the retrieved context.
Do not invent facts.
If the answer is not clearly present in the context, say:
"I could not find that in the retrieved analysis results."

Write clearly and naturally for a beginner.
Summarize the answer instead of copying raw chunk text whenever possible.
"""

    payload = {
        "model": OLLAMA_CHAT_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Question:\n{question}\n\nRetrieved context:\n{context}"
            }
        ],
        "stream": False
    }

    response = requests.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload, timeout=120)
    response.raise_for_status()

    data = response.json()

    return data["message"]["content"]
