import json
import math
import os
from pathlib import Path

import numpy as np
import requests


OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")
EMBEDDING_MODEL = os.environ.get("OLLAMA_EMBEDDING_MODEL", "bge-m3")
REQUEST_TIMEOUT_SECONDS = float(os.environ.get("OLLAMA_TIMEOUT_SECONDS", "60"))
if not math.isfinite(REQUEST_TIMEOUT_SECONDS) or REQUEST_TIMEOUT_SECONDS <= 0:
    raise ValueError("OLLAMA_TIMEOUT_SECONDS must be a finite positive number.")


def create_embedding(text_list):
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/embed",
            json={"model": EMBEDDING_MODEL, "input": text_list},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        embeddings = response.json().get("embeddings")
    except (requests.RequestException, ValueError) as error:
        raise RuntimeError("Could not obtain embeddings from Ollama.") from error
    if not isinstance(embeddings, list) or len(embeddings) != len(text_list):
        raise RuntimeError("Ollama returned an invalid embeddings response.")
    return embeddings


json_files = sorted(Path("jsons").glob("*.json"))
if not json_files:
    raise FileNotFoundError("No transcript JSON files were found in the jsons directory.")

records = []
all_embeddings = []
chunk_id = 0

for json_file in json_files:
    with json_file.open(encoding="utf-8") as transcript_file:
        content = json.load(transcript_file)
    chunks = content.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        raise ValueError(f"{json_file} does not contain a non-empty 'chunks' list.")

    embeddings = create_embedding([chunk["text"] for chunk in chunks])
    for chunk, embedding in zip(chunks, embeddings):
        records.append(
            {
                "chunk_id": chunk_id,
                "number": chunk["number"],
                "title": chunk["title"],
                "start": chunk["start"],
                "end": chunk["end"],
                "text": chunk["text"],
            }
        )
        all_embeddings.append(embedding)
        chunk_id += 1

embeddings_array = np.asarray(all_embeddings, dtype=np.float32)
if embeddings_array.ndim != 2 or embeddings_array.shape[0] != len(records):
    raise RuntimeError("Ollama returned embeddings with an invalid shape.")

np.savez_compressed(
    "embeddings.npz",
    embeddings=embeddings_array,
    metadata=np.asarray(json.dumps(records, ensure_ascii=False)),
)
print(f"Saved {len(records)} chunks to embeddings.npz.")
