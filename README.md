# 🔎 Agentic Retrieval Bench: Does a Small Local Agent Need a Vector Database?

> **The same Qwen 2.5 3B agent, the same financial filings, two search tools: keyword TF-IDF and Qdrant dense search. Raw traces for all 90 runs are committed.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Last commit](https://img.shields.io/github/last-commit/inamdarmihir/agentic-retrieval-bench)](https://github.com/inamdarmihir/agentic-retrieval-bench/commits/main)

---

## 🚀 What Is This?

This benchmark gives the same Qwen 2.5 3B agent two search tools, one at a time: lexical TF-IDF search and dense search in Qdrant. Both search the same financial filings. The questions, agent loop and scoring stay the same.

In the committed run, vector search found the reference evidence page for 18 of 45 questions. Keyword search found it for one. Final answers improved much less. Finding the page and reading the right number from it are different problems.

[Results](#-results) · [Quickstart](#-quick-start) · [Method](#-method) · [Limitations](#-limitations) · [Article](https://mihirinamdar.substack.com/p/does-a-small-agent-need-a-vector)

---

## 🚀 What You Can Do Here

- Inspect every search call and final answer from 90 agent runs.
- Swap retrieval tools without changing the agent harness.
- Run a six-call smoke test before downloading the full filing corpus.
- Separate evidence retrieval from answer correctness instead of treating them as one score.

---

## 📊 Results

45 FinanceBench questions, 15 from each question category. Each question was run once with each tool.

| Tool | Evidence recall | Answer correctness heuristic | Median run time |
| --- | ---: | ---: | ---: |
| Keyword TF-IDF | 2.2% (1/45) | 22.2% (10/45) | 11.7 s |
| Qdrant dense search | 40.0% (18/45) | 33.3% (15/45) | 13.8 s |

Source: [`results/summary.json`](results/summary.json), aggregated from [`results/raw_results.jsonl`](results/raw_results.jsonl). [`RESULTS.md`](RESULTS.md) explains the fields and scoring.

Evidence recall checks whether any search returned the gold document and page, allowing a one-page offset. Answer correctness is **not a human or LLM judgment**: a numeric answer passes if any predicted number matches any reference number within 2% relative tolerance. Non-numeric answers use substring matching.

**The answer grader has known errors.** The companion article reports a manual audit: keyword 7/45 (15.6%), vector 14/45 (31.1%), instead of the heuristic's 10/45 and 15/45. It shows years being counted as matching answers and unit normalization rejecting correct values. Those manual judgments are reported in the article, not stored as a separate scored artifact in this checkout. Treat the table's answer column as the original script output, not validated answer accuracy.

For metrics-generated questions, vector search found evidence 60% of the time. The article's manual review accepted two answers out of 15. Better retrieval did not remove the answer-extraction bottleneck. These results do not establish a general winner between lexical and vector search.

---

## ⚡ Quick Start

Requirements: Python, Docker and [Ollama](https://ollama.com). Dependencies are pinned in [`requirements.txt`](requirements.txt). First use downloads the model and dataset.

```bash
git clone https://github.com/inamdarmihir/agentic-retrieval-bench.git
cd agentic-retrieval-bench
make install
source .venv/bin/activate
ollama pull qwen2.5:3b-instruct
# Make sure Ollama is running before the smoke test.
make smoke
```

The smoke test starts a `qdrant-smoke` Docker container on port 6333, downloads three source PDFs, and runs three questions with both tools. It writes separate `results/smoke_*` files. It checks the pipeline, **not** the result table above.

### Full experiment

Use a disposable Qdrant instance on `localhost:6333`. If the smoke container is already running, it can serve this run too. Otherwise:

```bash
docker run -d --name qdrant-retrieval-bench -p 127.0.0.1:6333:6333 qdrant/qdrant
```

With the virtual environment active and Ollama running:

```bash
make download
make extract
make index
make benchmark
make analyze
```

`make benchmark` uses `--resume` to skip recorded question/tool pairs. Move the existing `results/raw_results.jsonl` aside for a fresh experiment rather than resuming the committed run. Preserve the original if you want to compare results.

**Data warning:** `build_index.py` deletes and rebuilds its named collection, `financebench_pages`. Use a disposable server, not a production instance. The Docker image in this repo is not pinned to a server version, so record the version you use.

### Check the aggregation without running the agent

```bash
python3 analyze_results.py --out /tmp/summary_check.json
diff /tmp/summary_check.json results/summary.json
```

---

## 🔬 Method

```text
FinanceBench questions + source PDFs
                 |
          page-text corpus
                 |
    same agent, same question, one tool
           /                 \
       TF-IDF              Qdrant dense
           \                 /
     search traces + final answers
                 |
       evidence and answer scoring
```

| Part | Implementation |
| --- | --- |
| Data | FinanceBench's 150-question public sample; experiment uses a seeded, stratified 45-question subset |
| Corpus | 84 financial filings; page-level text extracted with pypdf |
| Agent | Ollama `qwen2.5:3b-instruct`, up to four ReAct turns, first tool call per turn |
| Keyword tool | Pure-Python TF-IDF over page text, not BM25 or a shell search tool |
| Vector tool | FastEmbed `BAAI/bge-small-en-v1.5`, cosine search in Qdrant |
| Returned context | Up to 1,200 characters per hit |
| Sampling | Seed 42 in [`sampling.py`](sampling.py); generation itself is not seeded |

The original README records Ollama digest `357c53fb659c` (Q4_K_M). Check your installed model with `ollama show qwen2.5:3b-instruct`; a moving model tag is not a guarantee of the same weights.

---

## 🗂️ Project Map

| File | Purpose |
| --- | --- |
| `download_data.py`, `extract_corpus.py` | Download filings and extract pages |
| `build_index.py`, `tools.py` | Build the vector index and implement both tools |
| `agent.py`, `run_benchmark.py` | Agent loop and resumable paired runs |
| `evaluate.py`, `analyze_results.py` | Score and aggregate outputs |
| `results/`, `RESULTS.md` | Committed traces, summary and reading guide |

---

## ⚠️ Limitations

- One model, one corpus, 45 questions and one run per condition. There are no repeated-trial confidence intervals.
- A matching year can pass a wrong answer. Numeric units and valid paraphrases can also be scored incorrectly. The article's manual review is a different evaluation, not a fix already applied to the code.
- The small model's generation is stochastic, so identical sampling does not imply identical transcripts.
- TF-IDF is one lexical baseline. This is not a comparison against tuned BM25 or hybrid search.
- Serial local timings are not production-load latency measurements. The experiment does not establish multi-hop performance.

---

## 🤝 Contributing

Useful extensions include repeated trials, a stronger lexical baseline and manual auditing of the answer heuristic. Keep the same questions and corpus across conditions, and commit raw outputs alongside any new result table.

---

## 📚 Sources and License

Data: [FinanceBench](https://github.com/patronus-ai/financebench), CC BY-NC 4.0. Downloaded PDFs and extracted corpus are not committed.

Code: [MIT](LICENSE).
