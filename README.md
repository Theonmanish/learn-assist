# RAG Video Timestamp Identifier

This project answers questions about tutorial videos by retrieving relevant
transcript chunks and returning the video and timestamps where the topic is
taught. It is designed for locally owned or otherwise authorized tutorial
material.

## How it works

1. `video_to_mp3.py` converts source videos in `videos/` to audio using FFmpeg.
2. `mp3_to_json.py` transcribes audio with Whisper and stores timestamped
   segments in `jsons/`.
3. `merge_chuns.py` optionally combines adjacent transcript segments into
   larger chunks in `new_jsons/`.
4. `preprocess_json.py` sends transcript text to a local Ollama embedding model
   and writes numeric embeddings plus transcript metadata to `embeddings.npz`.
5. `process_incoming.py` embeds a question, retrieves the five most similar
   chunks, and asks a local Ollama generation model to explain the relevant
   video and timestamp.

The `.npz` artifact is used instead of Joblib/pickle because NumPy loading with
`allow_pickle=False` does not deserialize executable Python objects. The
artifact is still generated data and should only be used with transcript data
you trust.

## Technology stack

- Python 3
- OpenAI Whisper for speech-to-text
- Ollama for local embeddings and text generation
- NumPy and scikit-learn for storage and cosine similarity
- Requests for HTTP calls to Ollama
- FFmpeg for video-to-audio conversion

## Prerequisites

- Python 3.9 or newer (the code has not been tested across every Python
  release).
- FFmpeg installed and available on `PATH`.
- Ollama installed and running locally.
- Ollama models named `bge-m3` and `llama3.2`, unless overridden through
  environment variables.
- Sufficient disk space and compute resources for the Whisper model. The
  `large-v2` model can require substantial download time, memory, and storage.

Whisper models and Ollama models are downloaded from their respective
ecosystems. Review their licenses and costs/terms before use. This project
does not call a hosted LLM provider by default.

## Installation

Create and activate a virtual environment, then install Python dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install FFmpeg separately and confirm `ffmpeg -version` works. Install Ollama,
start it, and pull the required models:

```powershell
ollama pull bge-m3
ollama pull llama3.2
```

Copy `.env.example` to `.env` as a configuration template if you need to
change the Ollama URL, model names, or request timeout. The scripts read
environment variables from the process; they do not automatically parse `.env`.
In PowerShell, set values for the current session, for example:

```powershell
$env:OLLAMA_URL = "http://localhost:11434"
$env:OLLAMA_TIMEOUT_SECONDS = "60"
```

`.env` is intentionally ignored by Git and must contain only local
configuration, not committed credentials.

## Usage

Put your own source videos in `videos/`. The current filename parser expects a
name containing a tutorial number after `#`, a title in square brackets, and a
title section separated by ` -`, for example:

```text
Course #1 [metadata] - Linux basics - lesson.mp4
```

Run the pipeline from the project directory:

```powershell
python video_to_mp3.py
python mp3_to_json.py
python preprocess_json.py
python process_incoming.py
```

The final command prompts for a question. `prompt.txt` and `response.txt` are
local runtime outputs and are ignored because they can contain private
transcript or question data.

If transcripts have already been produced, start at `preprocess_json.py`.
Existing `embeddings.joblib` files are not loaded by this version. Regenerate
the artifact with `preprocess_json.py` to create `embeddings.npz`.

## Configuration

The optional variables in `.env.example` are read from the process environment:

- `OLLAMA_URL`: Ollama base URL; defaults to `http://localhost:11434`.
- `OLLAMA_EMBEDDING_MODEL`: embedding model; defaults to `bge-m3`.
- `OLLAMA_GENERATION_MODEL`: generation model; defaults to `llama3.2`.
- `OLLAMA_TIMEOUT_SECONDS`: finite HTTP timeout; defaults to `60`.

The scripts do not require API keys for the default local Ollama setup.

## Limitations and troubleshooting

- The pipeline is a command-line workflow, not a hosted web service.
- Retrieval uses cosine similarity over the five nearest transcript chunks; it
  does not guarantee that the answer is correct.
- The prompt includes retrieved transcript text. Treat transcript content as
  untrusted data and review generated answers.
- Whisper transcription quality depends on audio quality and language settings.
- The scripts currently expect the filename convention described above.
- If Ollama requests fail, confirm that Ollama is running, the models are
  pulled, and `OLLAMA_URL` is reachable.
- `embeddings.npz` must be regenerated whenever the transcript corpus or
  embedding model changes.

Future improvements could include stronger input schemas, configurable paths,
automated tests, a safer structured prompt format, and a user interface.



