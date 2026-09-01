"""The two retrieval tools the agent chooses between: keyword search and
vector search. Same corpus, same page-level granularity, same result shape
(doc_name, page_num, text, a tool-specific score), so the only variable
between conditions is the retrieval mechanism itself.
"""

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from fastembed import TextEmbedding
from qdrant_client import QdrantClient

CORPUS_PATH = Path(__file__).parent / "data" / "corpus.jsonl"
COLLECTION_NAME = "financebench_pages"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"


@dataclass
class SearchHit:
    doc_name: str
    page_num: int
    text: str
    score: float


def load_corpus() -> list[dict]:
    with open(CORPUS_PATH) as f:
        return [json.loads(line) for line in f]


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class KeywordSearchTool:
    """Lexical search over the raw extracted page text, the same category
    of mechanism rga/pdfgrep give an agent: no embeddings, no vector index,
    exact-term matching against the corpus as plain text. Scored with plain
    TF-IDF (term frequency in the page x inverse document frequency across
    the corpus) rather than raw term counts, so a rare, distinguishing term
    like a ticker ("3M") outweighs common financial-filing boilerplate
    ("capital", "amount") that would otherwise dominate a raw-count match.
    Implemented in pure Python rather than shelling out to ripgrep so this
    repo has no system-binary dependency to reproduce.
    """

    name = "keyword_search"
    description = (
        "Search the SEC filing corpus for a keyword or short phrase. Returns the "
        "pages with the most matches. Use exact terms likely to appear verbatim "
        "in a financial statement (e.g. 'capital expenditures', 'net income')."
    )

    def __init__(self, corpus: list[dict]):
        self.corpus = corpus
        self._tokens_per_page = [_tokenize(row["text"]) for row in corpus]
        df: Counter = Counter()
        for tokens in self._tokens_per_page:
            df.update(set(tokens))
        n_docs = len(corpus)
        self._idf = {term: math.log(n_docs / count) + 1.0 for term, count in df.items()}

    def search(self, query: str, k: int = 5) -> list[SearchHit]:
        terms = _tokenize(query)
        if not terms:
            return []
        scored = []
        for row, tokens in zip(self.corpus, self._tokens_per_page):
            if not tokens:
                continue
            tf = Counter(tokens)
            score = sum(tf.get(t, 0) * self._idf.get(t, 0.0) for t in terms)
            if score > 0:
                scored.append((score, row))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            SearchHit(doc_name=row["doc_name"], page_num=row["page_num"], text=row["text"], score=float(score))
            for score, row in scored[:k]
        ]


class VectorSearchTool:
    """Semantic search over the same corpus, embedded once with FastEmbed's
    bge-small-en-v1.5 and indexed in a real Qdrant collection.
    """

    name = "vector_search"
    description = (
        "Semantically search the SEC filing corpus for information relevant to a "
        "natural-language question. Returns the most relevant pages by meaning, "
        "not exact wording."
    )

    def __init__(self, client: QdrantClient, embedder: TextEmbedding):
        self.client = client
        self.embedder = embedder

    def search(self, query: str, k: int = 5) -> list[SearchHit]:
        vector = next(self.embedder.embed([query]))
        results = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=vector.tolist(),
            limit=k,
        ).points
        return [
            SearchHit(
                doc_name=r.payload["doc_name"],
                page_num=r.payload["page_num"],
                text=r.payload["text"],
                score=r.score,
            )
            for r in results
        ]


def format_hits(hits: list[SearchHit], max_chars_per_hit: int = 1200) -> str:
    if not hits:
        return "No matches found."
    parts = []
    for h in hits:
        text = h.text[:max_chars_per_hit]
        parts.append(f"[{h.doc_name}, page {h.page_num}]\n{text}")
    return "\n\n---\n\n".join(parts)
