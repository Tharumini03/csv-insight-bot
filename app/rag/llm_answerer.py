import os
import requests

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_CHAT_MODEL = os.environ.get("OLLAMA_CHAT_MODEL", "gemma3")


def generate_grounded_answer(question: str, retrieved_chunks: list, history=None):
    if history is None:
        history = []

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

Use conversation history when the user refers to earlier questions like:
"that", "it", "this result", "why", "explain more".

Write clearly and naturally for a beginner.
Summarize instead of copying raw chunk text whenever possible.

At the end of the answer, add a short citation line like:
Sources: [Chunk 1], [Chunk 3]
Use only chunk IDs that were actually provided.
"""

    messages = [{"role": "system", "content": system_prompt}]

    for item in history[-6:]:
        if item["role"] in ["user", "assistant"]:
            messages.append({
                "role": item["role"],
                "content": item["content"]
            })

    messages.append({
        "role": "user",
        "content": f"Question:\n{question}\n\nRetrieved context:\n{context}"
    })

    payload = {
        "model": OLLAMA_CHAT_MODEL,
        "messages": messages,
        "stream": False
    }

    try:
        response = requests.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        return data["message"]["content"]
    except Exception as e:
        return f"Chat is currently unavailable (Ollama not reachable): {str(e)}"
