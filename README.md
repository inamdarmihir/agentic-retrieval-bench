<p align="center">
  <a href="https://github.com/inamdarmihir/agentic-retrieval-bench">
    <img src="docs/assets/banner.svg" width="800px" alt="Agentic Retrieval Bench: does a small local agent need a vector database?">
  </a>
</p>

<p align="center">
  <a href="#results">Results</a>
  ·
  <a href="#quickstart">Quickstart</a>
  ·
  <a href="#method">Method</a>
  ·
  <a href="#limitations">Limitations</a>
  ·
  <a href="https://mihirinamdar.substack.com/p/does-a-small-agent-need-a-vector">Article</a>
</p>

<p align="center">
  <a href="https://github.com/inamdarmihir/agentic-retrieval-bench/actions/workflows/ci.yml">
    <img src="https://github.com/inamdarmihir/agentic-retrieval-bench/actions/workflows/ci.yml/badge.svg" alt="CI">
  </a>
  <a href="requirements.txt">
    <img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python 3.10+">
  </a>
  <a href="https://qdrant.tech">
    <img src="https://img.shields.io/badge/vector%20db-Qdrant-dc244c.svg" alt="Qdrant">
  </a>
  <a href="https://ollama.com">
    <img src="https://img.shields.io/badge/LLM-Ollama%20%7C%20Qwen%202.5%203B-black.svg" alt="Ollama, Qwen 2.5 3B">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT">
  </a>
  <a href="https://github.com/inamdarmihir/agentic-retrieval-bench/commits/main">
    <img src="https://img.shields.io/github/last-commit/inamdarmihir/agentic-retrieval-bench" alt="Last commit">
  </a>
</p>

# Agentic Retrieval Bench

The same Qwen 2.5 3B agent gets two search tools, one at a time: keyword TF-IDF and dense search in Qdrant. Both search the same financial filings. The questions, agent loop and scoring stay identical, so retrieval is the only variable.

**In the committed run, vector search found the reference evidence page for 18 of 45 questions. Keyword search found it for one. Final answers improved much less.** Finding the page and reading the right number from it are different problems.

| | |
| --- | --- |
| **Controlled** | One agent harness, one corpus, one question set. Only the retrieval tool changes. |
| **Auditable** | Raw traces for all 90 runs (45 questions x 2 tools) are committed. |
| **Two metrics** | Evidence recall and answer correctness are scored separately, not blended. |
| **Cheap to try** | A six-run smoke test checks the pipeline before the full corpus download. |

## Contents

- [Results](#results)
- [Quickstart](#quickstart)
- [Method](#method)
- [Project layout](#project-layout)
- [Limitations](#limitations)
- [Development](#development)
- [Sources and license](#sources-and-license)

## Results

45 FinanceBench questions, 15 from each question category. Each question was run once with each tool.

| Tool | Evidence recall | Answer correctness heuristic | Median run time |
| --- | ---: | ---: | ---: |
| Keyword TF-IDF | 2.2% (1/45) | 22.2% (10/45) | 11.7 s |
| Qdrant dense search | 40.0% (18/45) | 33.3% (15/45) | 13.8 s |

Source: [`results/summary.json`](results/summary.json), aggregated from [`results/raw_results.jsonl`](results/raw_results.jsonl). [`RESULTS.md`](RESULTS.md) explains the fields and scoring.

Evidence recall checks whether any search returned the gold document and page, allowing a one-page offset. Answer correctness is **not a human or LLM judgment**: a numeric answer passes if any predicted number matches any reference number within 2% relative tolerance. Non-numeric answers use substring matching.

> [!WARNING]
> **The answer grader has known errors.** The companion article reports a manual audit: keyword 7/45 (15.6%), vector 14/45 (31.1%), instead of the heuristic's 10/45 and 15/45. It shows years being counted as matching answers and unit normalization rejecting correct values. Those manual judgments are reported in the article, not stored as a separate scored artifact in this checkout. Treat the table's answer column as the original script output, not validated answer accuracy.

For metrics-generated questions, vector search found evidence 60% of the time. The article's manual review accepted two answers out of 15. Better retrieval did not remove the answer-extraction bottleneck. These results do not establish a general winner between lexical and vector search.

## Quickstart

Requires Python 3.10+, Docker and [Ollama](https://ollama.com). Dependencies are pinned in [`requirements.txt`](requirements.txt). First use downloads the model and dataset.

```bash
git clone https://github.com/inamdarmihir/agentic-retrieval-bench.git
cd agentic-retrieval-bench
make install
source .venv/bin/activate
ollama pull qwen2.5:3b-instruct
# Make sure Ollama is running before the smoke test.
make smoke
```

The smoke test starts a `qdrant-smoke` Docker container on port 6333, downloads three source PDFs (about 3 MB), and runs three questions with both tools. It writes separate `results/smoke_*` files. It checks the pipeline, **not** the result table above.

Run `make help` to list every target.

### Full experiment

Use a disposable Qdrant instance on `localhost:6333` (override with `QDRANT_URL`). If the smoke container is already running, it can serve this run too. Otherwise:

```bash
docker run -d --name qdrant-retrieval-bench -p 127.0.0.1:6333:6333 qdrant/qdrant
```

With the virtual environment active and Ollama running:

```bash
make download    # 150 questions + 84 PDFs, about 140 MB
make extract     # page-level text with pypdf
make index       # embed and index into Qdrant
make benchmark   # 45 questions x 2 tools = 90 agent runs
make analyze     # results/raw_results.jsonl -> results/summary.json
```

`make benchmark` uses `--resume` to skip recorded question/tool pairs. Move the existing `results/raw_results.jsonl` aside for a fresh experiment rather than resuming the committed run, and preserve the original if you want to compare results.

> [!WARNING]
> `build_index.py` deletes and rebuilds its collection, `financebench_pages`. Use a disposable server, not a production instance. The Docker image is not pinned to a server version, so record the version you use.

### Check the aggregation without running the agent

```bash
make test
```

This re-aggregates the committed traces and asserts that the result equals the committed `results/summary.json`, alongside unit tests for the scorer, sampler and keyword tool. No Ollama or Docker needed.

## Method

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
| Data | FinanceBench's 150-question public sample; the experiment uses a seeded, stratified 45-question subset |
| Corpus | 84 financial filings; page-level text extracted with pypdf |
| Agent | Ollama `qwen2.5:3b-instruct`, up to four ReAct turns, first tool call per turn |
| Keyword tool | Pure-Python TF-IDF over page text, not BM25 or a shell search tool |
| Vector tool | FastEmbed `BAAI/bge-small-en-v1.5`, cosine search in Qdrant |
| Returned context | Up to 1,200 characters per hit |
| Sampling | Seed 42 in [`sampling.py`](sampling.py); generation itself is not seeded |

The original run recorded Ollama digest `357c53fb659c` (Q4_K_M). Check your installed model with `ollama show qwen2.5:3b-instruct`; a moving model tag is not a guarantee of the same weights.

Settings (Qdrant URL, collection name, models, turn limit) live in [`config.py`](config.py).

## Project layout

| Path | Purpose |
| --- | --- |
| `config.py` | Shared settings: paths, Qdrant URL, collection, model names |
| `download_data.py`, `extract_corpus.py` | Download filings and extract pages |
| `build_index.py`, `tools.py` | Build the vector index and implement both search tools |
| `agent.py`, `run_benchmark.py` | Agent loop and resumable paired runs |
| `evaluate.py`, `analyze_results.py` | Score and aggregate outputs |
| `tests/` | Offline unit tests and the trace-to-summary round trip |
| `results/`, `RESULTS.md` | Committed traces, summary and reading guide |

## Limitations

- One model, one corpus, 45 questions and one run per condition. There are no repeated-trial confidence intervals.
- A matching year can pass a wrong answer. Numeric units and valid paraphrases can also be scored incorrectly. The article's manual review is a different evaluation, not a fix already applied to the code.
- The small model's generation is stochastic, so identical sampling does not imply identical transcripts.
- TF-IDF is one lexical baseline. This is not a comparison against tuned BM25 or hybrid search.
- Serial local timings are not production-load latency measurements. The experiment does not establish multi-hop performance.

## Development

```bash
make install     # venv + pinned dependencies + pytest
make test        # offline tests
```

Useful extensions: repeated trials, a stronger lexical baseline and manual auditing of the answer heuristic. Keep the same questions and corpus across conditions, and commit raw outputs alongside any new result table.

## Sources and license

Data: [FinanceBench](https://github.com/patronus-ai/financebench), CC BY-NC 4.0. Downloaded PDFs and extracted corpus are not committed.

Code: [MIT](LICENSE).
