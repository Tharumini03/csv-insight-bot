import os
import json


def split_text_into_chunks(text: str, chunk_size: int = 400):
    """
    Split text into smaller chunks by paragraph first.
    If a paragraph is too long, split by character length.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []

    for para in paragraphs:
        if len(para) <= chunk_size:
            chunks.append(para)
        else:
            for i in range(0, len(para), chunk_size):
                chunks.append(para[i:i + chunk_size])

    return chunks


def build_chunks_file(file_id: str, knowledge_path: str):
    """
    Read knowledge.txt, split into chunks, and save as chunks.json
    """
    output_dir = f"app/storage/outputs/{file_id}"
    os.makedirs(output_dir, exist_ok=True)

    with open(knowledge_path, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = split_text_into_chunks(text)

    chunk_data = []
    for i, chunk in enumerate(chunks):
        chunk_data.append({
            "chunk_id": i,
            "text": chunk
        })

    chunks_path = f"{output_dir}/chunks.json"
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunk_data, f, indent=2)

    return chunks_path
