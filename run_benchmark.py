"""Run the full benchmark: a stratified sample of FinanceBench questions,
each answered twice by the same ReAct agent loop, once with keyword_search
bound as the only tool, once with vector_search. Writes one JSON line per
run to results/raw_results.jsonl as it goes, so a partial run is never lost.
"""

import argparse
import json
import random
import time
from pathlib import Path

from fastembed import TextEmbedding
from qdrant_client import QdrantClient

from agent import run_agent
from evaluate import is_correct
from tools import KeywordSearchTool, VectorSearchTool, load_corpus

QUESTIONS_PATH = Path(__file__).parent / "data" / "questions.jsonl"
RESULTS_PATH = Path(__file__).parent / "results" / "raw_results.jsonl"
QDRANT_URL = "http://localhost:6333"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
SEED = 42


def stratified_sample(questions: list[dict], n_per_type: int) -> list[dict]:
    by_type: dict[str, list[dict]] = {}
    for q in questions:
        by_type.setdefault(q["question_type"], []).append(q)
    rng = random.Random(SEED)
    sample = []
    for qtype, rows in sorted(by_type.items()):
        rng.shuffle(rows)
        sample.extend(rows[:n_per_type])
    rng.shuffle(sample)
    return sample


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-per-type", type=int, default=15, help="questions per question_type (3 types)")
    parser.add_argument("--resume", action="store_true", help="skip (question_id, condition) pairs already in results")
    args = parser.parse_args()

    with open(QUESTIONS_PATH) as f:
        questions = [json.loads(line) for line in f]
    sample = stratified_sample(questions, args.n_per_type)
    print(f"Sampled {len(sample)} questions ({args.n_per_type} per question_type)")

    corpus = load_corpus()
    keyword_tool = KeywordSearchTool(corpus)

    embedder = TextEmbedding(model_name=EMBED_MODEL)
    client = QdrantClient(url=QDRANT_URL)
    vector_tool = VectorSearchTool(client, embedder)

    RESULTS_PATH.parent.mkdir(exist_ok=True)
    done = set()
    if args.resume and RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            for line in f:
                r = json.loads(line)
                done.add((r["financebench_id"], r["condition"]))
        print(f"Resuming: {len(done)} runs already recorded")

    mode = "a" if args.resume else "w"
    with open(RESULTS_PATH, mode) as out:
        for i, q in enumerate(sample):
            gold_doc = q["doc_name"]
            gold_page = q["evidence"][0]["evidence_page_num"]

            for condition, tool in [("keyword", keyword_tool), ("vector", vector_tool)]:
                if (q["financebench_id"], condition) in done:
                    continue
                t0 = time.time()
                result = run_agent(q["question"], condition, tool, gold_doc, gold_page)
                correct = is_correct(result.final_answer, q["answer"])
                record = {
                    "financebench_id": q["financebench_id"],
                    "question_type": q["question_type"],
                    "doc_name": gold_doc,
                    "condition": condition,
                    "question": q["question"],
                    "reference_answer": q["answer"],
                    "final_answer": result.final_answer,
                    "correct": correct,
                    "hit_evidence": result.hit_evidence,
                    "turns_used": result.turns_used,
                    "n_tool_calls": len(result.tool_calls),
                    "wall_clock_s": result.wall_clock_s,
                    "tool_calls": result.tool_calls,
                }
                out.write(json.dumps(record) + "\n")
                out.flush()
                elapsed = time.time() - t0
                print(
                    f"[{i+1}/{len(sample)}] {q['financebench_id']} {condition:7s} "
                    f"correct={correct} evidence={result.hit_evidence} "
                    f"turns={result.turns_used} {elapsed:.1f}s"
                )

    print(f"\nDone. Results in {RESULTS_PATH}")


if __name__ == "__main__":
    main()
