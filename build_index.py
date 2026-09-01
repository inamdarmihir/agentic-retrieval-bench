"""Embed every page in the corpus with FastEmbed and upsert into a real
Qdrant collection. Run once after extract_corpus.py.
"""

import json
import time
from pathlib import Path

from fastembed import TextEmbedding
from qdrant_client import QdrantClient, models

from tools import COLLECTION_NAME, CORPUS_PATH, EMBED_MODEL

QDRANT_URL = "http://localhost:6333"
BATCH_SIZE = 64


def build() -> None:
    with open(CORPUS_PATH) as f:
        corpus = [json.loads(line) for line in f]

    # Skip pages with no extractable text (scanned images, blank pages) -
    # nothing for either tool to find on them either way.
    corpus = [row for row in corpus if row["text"].strip()]
    print(f"{len(corpus)} non-empty pages to index")

    # threads=8: this repo isn't measuring embedding throughput, so unlike
    # some of this project's other benchmarks, ONNX runtime is deliberately
    # thread-tuned here rather than left at its slow single-threaded default.
    embedder = TextEmbedding(model_name=EMBED_MODEL, threads=8)
    client = QdrantClient(url=QDRANT_URL)

    sample_vec = next(embedder.embed(["dimension probe"]))
    dim = len(sample_vec)

    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
    )

    t0 = time.time()
    point_id = 0
    for i in range(0, len(corpus), BATCH_SIZE):
        batch = corpus[i : i + BATCH_SIZE]
        texts = [row["text"] for row in batch]
        vectors = list(embedder.embed(texts))
        points = [
            models.PointStruct(
                id=point_id + j,
                vector=vectors[j].tolist(),
                payload={"doc_name": row["doc_name"], "page_num": row["page_num"], "text": row["text"]},
            )
            for j, row in enumerate(batch)
        ]
        client.upsert(collection_name=COLLECTION_NAME, points=points)
        point_id += len(batch)
        if (i // BATCH_SIZE) % 10 == 0:
            print(f"  indexed {point_id}/{len(corpus)}")

    elapsed = time.time() - t0
    print(f"\nIndexed {point_id} pages in {elapsed:.1f}s ({point_id/elapsed:.1f} pages/sec)")
    info = client.get_collection(COLLECTION_NAME)
    print(f"Collection points_count: {info.points_count}")


if __name__ == "__main__":
    build()
