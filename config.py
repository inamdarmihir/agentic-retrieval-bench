"""Settings shared by the index builder, the tools and the benchmark runner."""

import os
from pathlib import Path

ROOT = Path(__file__).parent
CORPUS_PATH = ROOT / "data" / "corpus.jsonl"
QUESTIONS_PATH = ROOT / "data" / "questions.jsonl"
RESULTS_DIR = ROOT / "results"

# Override with `QDRANT_URL=http://host:6333` to use a non-default server.
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "financebench_pages"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

OLLAMA_MODEL = "qwen2.5:3b-instruct"
MAX_TURNS = 4
