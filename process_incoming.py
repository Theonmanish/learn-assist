import json
import math
import os
from pathlib import Path

import numpy as np
import requests
from sklearn.metrics.pairwise import cosine_similarity


OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")
EMBEDDING_MODEL = os.environ.get("OLLAMA_EMBEDDING_MODEL", "bge-m3")
GENERATION_MODEL = os.environ.get("OLLAMA_GENERATION_MODEL", "llama3.2")
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


def inference(prompt):
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": GENERATION_MODEL, "prompt": prompt, "stream": False},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        result = response.json()
    except (requests.RequestException, ValueError) as error:
        raise RuntimeError("Could not obtain a response from Ollama.") from error
    if not isinstance(result.get("response"), str):
        raise RuntimeError("Ollama returned an invalid generation response.")
    return result


artifact_path = Path("embeddings.npz")
if not artifact_path.is_file():
    raise FileNotFoundError(
        "embeddings.npz was not found. Run preprocess_json.py first."
    )

with np.load(artifact_path, allow_pickle=False) as artifact:
    if "embeddings" not in artifact or "metadata" not in artifact:
        raise ValueError("embeddings.npz is missing required arrays.")
    stored_embeddings = np.asarray(artifact["embeddings"], dtype=np.float32)
    metadata = json.loads(str(artifact["metadata"].item()))

if stored_embeddings.ndim != 2 or not isinstance(metadata, list):
    raise ValueError("embeddings.npz has an invalid schema.")
if stored_embeddings.shape[0] != len(metadata) or not metadata:
    raise ValueError("embeddings.npz contains mismatched or empty data.")

incoming_query = input("Ask a Question: ")
question_embedding = create_embedding([incoming_query])[0]
similarities = cosine_similarity(stored_embeddings, [question_embedding]).flatten()
top_results = 5
max_indx = similarities.argsort()[::-1][0:top_results]
selected_chunks = [metadata[index] for index in max_indx]

prompt = f"""This is a course for basic Linux commands. Here are video subtitles and
chunks containing the video title, video number, start time, end time, and text:

{json.dumps(selected_chunks, ensure_ascii=False)}
----------------------------

"{incoming_query}"
User asked this question related to the video chunks. Answer naturally and explain
where the content is taught, including the video and timestamp. If the question is
unrelated, explain that you can only answer questions related to the course.
"""

with open("prompt.txt", "w", encoding="utf-8") as prompt_file:
    prompt_file.write(prompt)

response = inference(prompt)["response"]
print(response)

with open("response.txt", "w", encoding="utf-8") as response_file:
    response_file.write(response)
