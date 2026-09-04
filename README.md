# agentic-retrieval-bench

Does a local ReAct agent answering real financial-filing questions
([FinanceBench](https://github.com/patronus-ai/financebench)) actually need
a vector database, or does plain keyword search over the same documents get
it most of the way there? The current literature disagrees with itself on
this: a February 2026 Amazon Science paper found a keyword-tool-use agent
reaches 88-94.5% of RAG's context recall and answer correctness with no
vector store involved, a June 2026 follow-up says no prior work has
actually isolated retrieval strategy from agent-harness design, and a third
paper claims vector-based agentic RAG wins outright on FinanceBench
specifically. This repo runs that comparison directly: the same local
agent, the same real SEC filings, the same questions, with only the
retrieval tool swapped between keyword search and Qdrant vector search, as
one more measurement into that open disagreement — not a resolution of it.

**Contents**: [Why this exists](#why-this-exists) ·
[Method, at a glance](#method-at-a-glance) · [Results](#results) ·
[What this shows](#what-this-shows) ·
[What this does NOT show](#what-this-does-not-show) ·
[Quickstart](#quickstart) · [Smoke test](#smoke-test) ·
[Limitations](#limitations) · [Method, in full](#method-in-full)

## Why this exists

Every repo elsewhere in this project benchmarks Qdrant's own internals
(payload indexing cost, write-pattern cost, quantization tradeoffs). None
of them ask the more basic question an agent-building team actually faces
first: does adding a vector database to an agent's tool belt beat giving it
a keyword-search tool over the same documents? That question is contested
in the current literature, not settled:

- [Amazon Science, AAAI 2026](https://arxiv.org/abs/2602.23368): a
  keyword-tool-use agent reaches 94.5% of RAG's faithfulness, 88.0% of its
  context recall, and 91.5% of its answer correctness, and beats RAG
  outright on FinanceBench answer correctness (30.40% vs 24.24%).
- [arXiv:2605.15184](https://arxiv.org/abs/2605.15184), a June 2026
  follow-up, states plainly that no prior work has systematically
  separated retrieval strategy from agent-architecture effects.
- [arXiv:2605.05538](https://arxiv.org/abs/2605.05538) claims the opposite
  on FinanceBench specifically: vector-based agentic RAG at 92% answer
  correctness, beating both plain RAG and the keyword-tool baseline.

Rather than cite whichever side sounds more favorable to Qdrant, this repo
runs its own measurement into that open disagreement, on real data, with
both tools implemented in the same harness so the only variable is
retrieval mechanism.

## Method, at a glance

Everything here is pinned or stated exactly so a stranger can reproduce it
without asking the author anything:

| | |
|---|---|
| **Data** | [FinanceBench](https://github.com/patronus-ai/financebench) open-source sample, CC BY-NC 4.0. 150 question/answer pairs over real SEC 10-Ks/10-Qs, each with a gold `(doc_name, evidence_page_num)` |
| **Sample** | Stratified, 15 questions from each of FinanceBench's 3 `question_type` categories = **45 questions**, `random.Random(seed=42)`, see [`sampling.py`](sampling.py) |
| **Runs** | Each of the 45 questions run once per condition (keyword, vector) = **90 agent runs total** |
| **Agent** | Local ReAct loop ([`agent.py`](agent.py)), **≤4 turns, 1 tool call taken per turn** (max 4 search calls per run) |
| **Model** | [`qwen2.5:3b-instruct`](https://ollama.com/library/qwen2.5:3b-instruct) via Ollama — pinned to digest `357c53fb659c`, Q4_K_M quantization, 3.09B params, 1.9GB. Verify with `ollama show qwen2.5:3b-instruct` after pulling |
| **Tools** | `keyword_search`: pure-Python TF-IDF over raw page text, no embeddings. `vector_search`: FastEmbed `bge-small-en-v1.5` embeddings in a real Qdrant collection, cosine similarity |
| **Metrics** | **Evidence recall** — did any tool call return the gold page (±1)? A mechanical page-number check. **Answer correctness** — a **numeric-tolerance heuristic** (`evaluate.py`), not an LLM-judged score: extract numbers from the agent's answer and the reference, match within 2% relative tolerance, substring fallback for non-numeric answers. See [RESULTS.md](RESULTS.md) for exactly how these two metrics diverge and which one to trust for what. |

Full method detail (why TF-IDF, why this model size, corpus verification)
is in [Method, in full](#method-in-full) further down; the table above is
everything you need to read the results honestly.

## Results

Full numbers: [`results/summary.json`](results/summary.json). Raw
per-question data: [`results/raw_results.jsonl`](results/raw_results.jsonl).
See [RESULTS.md](RESULTS.md) for the file schemas and how to recompute or
re-slice these numbers yourself. 45 questions (15 per FinanceBench
`question_type`), each run once per condition, 90 agent runs total,
`qwen2.5:3b-instruct`.

**Overall, by condition:**

| Condition | Evidence recall | Answer correct | Avg tool calls | Avg turns | p50 latency |
|---|---|---|---|---|---|
| `keyword_search` | 2.2% (1/45) | 22.2% (10/45) | 0.93 | 1.93 | 11.7s |
| `vector_search` | 40.0% (18/45) | 33.3% (15/45) | 1.02 | 2.02 | 13.8s |

**By question type:**

| Question type | Condition | Evidence recall | Answer correct |
|---|---|---|---|
| domain-relevant | keyword | 0.0% | 53.3% |
| domain-relevant | vector | 33.3% | 60.0% |
| metrics-generated | keyword | 0.0% | 0.0% |
| metrics-generated | vector | 60.0% | 6.7% |
| novel-generated | keyword | 6.7% | 13.3% |
| novel-generated | vector | 26.7% | 33.3% |

## What this shows

- Vector search finds the actual gold evidence page **18x more often** than
  keyword search in this run (40.0% vs 2.2% evidence recall), a larger gap
  than either paper in the opening paragraph reports for FinanceBench
  specifically. That gap traces to a mechanism visible directly in the
  data, not a black box: FinanceBench questions are written in analyst
  language ("capital expenditures"), while the filings use GAAP line-item
  phrasing ("purchases of property, plant and equipment"). Confirmed
  directly during development on one 3M capex question: the terms
  "capital" and "expenditure" have zero term-frequency on the actual gold
  page. TF-IDF over exact tokens has no way to bridge that gap; embedding
  similarity does.
- Answer correctness moves the same direction but by a much smaller margin
  (33.3% vs 22.2%), and the by-type breakdown shows the overall number
  undersells the retrieval gap and oversells the answer-quality gap — see
  the per-type breakdown below.
- **domain-relevant**: `keyword_search` never once returned the correct
  evidence page (0/15), yet the agent still got 53.3% of final answers
  right anyway — these are the FinanceBench questions closest to common
  financial knowledge, which a 3B model appears to partly know unaided. For
  this question type, `answer_correct` is not a trustworthy signal of the
  retrieval tool doing real work; `evidence_recall` is.
- **metrics-generated**: `vector_search` had its best evidence recall of
  any cell (60%) but its worst answer correctness (6.7%), worse than
  keyword's 0% on the same type — the agent found the right page and still
  got the number wrong. That's a distinct failure mode from a retrieval
  failure: extracting one precise number out of a dense financial table,
  truncated to 1200 characters of page text, is genuinely hard for a 3B
  model, and better retrieval alone does not fix it.
- **novel-generated** is the one type where the result matches the "why
  this exists" framing directly: vector search beats keyword on both
  evidence recall (26.7% vs 6.7%) and answer correctness (33.3% vs 13.3%).
- Vector search also costs ~2.1s more p50 latency per run (13.8s vs
  11.7s): an embedding call plus a network round trip to Qdrant, against an
  in-process Python dict scan for keyword search.

The honest read: for this agent size, this corpus, and this tool
implementation, vector search is structurally better at finding the right
page, largely because of the vocabulary-mismatch mechanism above, but that
improvement does not translate cleanly into better final answers, because
answer quality is gated by a second, separate bottleneck — whether a small
local model can correctly read a number back out of what it retrieves.

## What this does NOT show

- **Not a resolution of the contested literature cited above.** This is
  one more data point run into that disagreement, on one corpus, with one
  model and one tool implementation per side — not a refutation of any of
  the three cited papers, none of which used this exact harness.
- **Not a statement about vector search "winning" in general.** It wins on
  evidence recall here; it does not cleanly win on answer correctness
  (worst cell in the whole table is `vector_search` on
  `metrics-generated`). See [RESULTS.md](RESULTS.md) for why those two
  metrics diverge.
- **Not a frontier-model result.** A 3B, 4-bit-quantized local model was
  chosen deliberately for zero-cost, no-API-key reproduction. A larger
  model's tool-use and number-extraction ability could shrink or widen the
  gap between conditions in either direction.
- **Not a production-scale or concurrent-load latency comparison.** The
  ~2.1s p50 latency gap is measured serially, on a local single-node
  Qdrant container, with no query concurrency.
- **Not a multi-hop retrieval test.** The agent can re-search within its
  4-turn budget, but nothing in the 45-question sample requires assembling
  evidence from multiple distinct pages.
- **Not an LLM-judged answer-quality score.** `answer_correct` is a
  numeric-tolerance heuristic (see above and [RESULTS.md](RESULTS.md)); a
  correct free-text answer with no clean matching number in it can be
  scored wrong.

## Quickstart

```bash
git clone https://github.com/inamdarmihir/agentic-retrieval-bench
cd agentic-retrieval-bench
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # pinned versions, see requirements.txt

python3 download_data.py               # fetches 150 questions + 84 PDFs (~140MB)
python3 extract_corpus.py              # extracts page-level text with pypdf (~35-40MB output)
python3 build_index.py                 # embeds + indexes into a local Qdrant collection
python3 run_benchmark.py               # runs the full 45-question sample through both conditions (90 runs)
python3 analyze_results.py             # aggregates results/raw_results.jsonl into results/summary.json
```

Or with `make` (see `make help` for every target):

```bash
make install && make download && make extract && make index && make benchmark && make analyze
```

**Prerequisites and what to expect:**

- **Docker**, for a local Qdrant container: `docker run -d -p 6333:6333 qdrant/qdrant` (image ~200MB pulled once).
- **[Ollama](https://ollama.com)** with the model pulled and pinned:
  `ollama pull qwen2.5:3b-instruct` (~1.9GB; verify what you got with
  `ollama show qwen2.5:3b-instruct`, digest `357c53fb659c`). No API key,
  no cost, runs on CPU.
- **Disk**: budget roughly **2.5GB free** — ~140MB for the FinanceBench
  PDFs (`data/pdfs/`, gitignored, not redistributed here per its CC BY-NC
  4.0 license), ~35-40MB for the extracted page-text corpus
  (`data/corpus.jsonl`, gitignored), ~200MB for the Qdrant Docker image,
  and ~1.9GB for the Ollama model weights (this last one is shared across
  any other project using the same model — it isn't duplicated per repo).
- **Time**: the full run is 90 sequential agent calls against a local CPU
  model, each call ~10-15s — expect roughly 15-25 minutes for
  `run_benchmark.py` alone, more on slower hardware. `run_benchmark.py
  --resume` skips `(question, condition)` pairs already recorded in the
  output file, so an interrupted run is never lost.

## Smoke test

Don't want to commit to the full ~140MB download and 90-run experiment just
to check the pipeline works? Run the smoke test instead: **3 questions (1
per `question_type`, guaranteed to be a subset of the committed 45-question
sample, same seed), 6 agent runs total.**

```bash
make smoke
```

This downloads only the 3 PDFs the smoke sample needs (`download_data.py
--smoke`, a few MB instead of ~140MB), builds a tiny 3-document corpus and
Qdrant index from just those, and runs `run_benchmark.py --n-per-type 1
--out results/smoke_results.jsonl` followed by `analyze_results.py --in
results/smoke_results.jsonl --out results/smoke_summary.json`. It never
touches the committed `results/raw_results.jsonl` or `results/summary.json`
— those stay exactly as committed regardless of how many times you run the
smoke test.

Equivalent manual steps, if you'd rather not use `make` or already have
Qdrant running elsewhere:

```bash
python3 download_data.py --smoke
python3 extract_corpus.py
python3 build_index.py
python3 run_benchmark.py --n-per-type 1 --out results/smoke_results.jsonl
python3 analyze_results.py --in results/smoke_results.jsonl --out results/smoke_summary.json
```

**What the smoke test proves, and what it doesn't:** it proves the
data-download → extraction → indexing → agent-loop → scoring pipeline runs
end to end with both tools, against real data, on your machine. It does
**not** reproduce the numbers in the Results section above — 3 questions
against a 3-document, ~300-page corpus is not a statistically meaningful
sample, and a smaller corpus changes both tools' retrieval difficulty. Only
`run_benchmark.py` with the default `--n-per-type 15` against the full
84-document corpus reproduces the committed results.

## Limitations

- **Answer-correctness is a numeric-tolerance heuristic, not an LLM
  judge.** Running a second LLM as judge would introduce that model's own
  error into the measurement; the tradeoff is that some correct free-text
  answers with no clean number in them may be scored wrong. Evidence
  recall (a page-match check, not a judgment call) is the more reliable of
  the two metrics for that reason — see [RESULTS.md](RESULTS.md) for the
  full exact algorithm and worked failure modes.
- **A 3B, 4-bit-quantized local model, not a frontier model.** This tests
  whether the *retrieval choice* matters for an agent this size on this
  hardware; a larger model's tool-use reasoning could change the gap
  between conditions in either direction.
- **45 of 150 questions, not all of them**, for tractability against a
  local model with no batching. Stratified by `question_type` to avoid
  skewing toward one question style. Seed is fixed (42) and recorded in
  [`sampling.py`](sampling.py) so the exact sample is reproducible, not
  just its size.
- **Single-hop retrieval**: the agent has 4 turns and can re-search, but
  this doesn't test genuinely multi-hop questions requiring evidence
  assembled from multiple distinct pages.
- **Keyword search is TF-IDF over already-extracted page text, not a
  literal `rg`/`pdfgrep` subprocess.** Functionally the same category of
  tool (lexical, no embeddings), chosen for zero system-binary
  dependencies; a real ripgrep-backed tool could behave somewhat
  differently on multi-word queries.
- **Single run per (question, condition), not repeated trials.** The local
  model is called with default (non-zero) sampling settings and no fixed
  generation seed is threaded through Ollama, so re-running the same
  question/condition pair is not guaranteed to reproduce the identical
  agent transcript, only the same *tool* and the same *question sample*.
  The committed `results/raw_results.jsonl` is the actual transcript this
  README's numbers come from, not a statistic averaged over multiple
  trials.

## Method, in full

**Data**: [FinanceBench](https://github.com/patronus-ai/financebench)
open-source sample (CC BY-NC 4.0), the same benchmark the Amazon Science
paper above reports FinanceBench-specific numbers on. 150 real
question/answer pairs over real SEC filings (10-Ks, 10-Qs), each with a
gold source document and evidence page. `download_data.py` fetches the 150
questions plus the 84 real PDF filings they reference (not vendored in this
repo, ~140MB) directly from the source repo. Corpus: 12,013 extracted pages
across those 84 filings, verified page-number-exact against FinanceBench's
own `evidence_page_num` field on a 30-question sample before any indexing
happened.

**Agent**: a local ReAct loop (`agent.py`) running
[`qwen2.5:3b-instruct`](https://ollama.com/library/qwen2.5:3b-instruct) via
Ollama, 4-bit quantized (Ollama's default `Q4_K_M`, digest `357c53fb659c`,
1.9GB), chosen deliberately small so this reproduces on ordinary hardware
with no API key and no cost. The agent gets exactly one tool per run and up
to 4 turns to call it and answer; the harness takes only the first tool
call per turn even if the model emits several (`agent.py`'s
"single-action ReAct step").

**Two tools, same corpus, same page-level granularity:**

- **`keyword_search`** — lexical TF-IDF search over the raw extracted page
  text (term frequency x inverse document frequency across the 12,013-page
  corpus). No embeddings, no vector index; the same category of mechanism
  `rga`/`pdfgrep` give an agent in the source papers. Implemented in pure
  Python (`tools.py`), no ripgrep binary dependency.
- **`vector_search`** — the same corpus, embedded once with FastEmbed
  `bge-small-en-v1.5` (the same embedder used elsewhere in this project),
  indexed in a real Qdrant collection, queried by cosine similarity.

**Metrics, per run:**

- **Evidence recall** — did any tool call in the run return the gold
  `(doc_name, page_num)` (±1 page tolerance)?
- **Answer correctness** — a numeric-tolerant heuristic (`evaluate.py`):
  extract numbers from the agent's final answer and the reference answer,
  match within 2% relative tolerance; non-numeric references fall back to
  substring containment. Documented as a heuristic, not an LLM-judged
  score — see Limitations and [RESULTS.md](RESULTS.md).
- **Tool calls, turns, wall-clock latency** per run.

Sample: a stratified 45-question sample (15 from each of FinanceBench's
three `question_type` categories: metrics-generated, domain-relevant,
novel-generated), each run once per condition (90 agent runs total), fixed
seed 42 (`sampling.py`) for reproducibility.

## Adapting this to your own data and tools

- **Different questions/documents**: replace `data/questions.jsonl` (needs
  `financebench_id`, `question_type`, `doc_name`, `question`, `answer`,
  `evidence[0].evidence_page_num`) and point `data/pdfs/` at your own PDF
  set, or rewrite `extract_corpus.py`'s input if your source isn't PDFs.
- **Different local model**: change `MODEL` in `agent.py` to any
  Ollama-served model that supports tool calling (`llama3.1`, `mistral`,
  larger `qwen2.5` sizes). No other code changes needed.
- **A third retrieval tool**: implement a class with a `.search(query, k)`
  method returning a list of `SearchHit(doc_name, page_num, text, score)`
  in `tools.py`, then add a `(condition_name, tool_instance)` pair to the
  loop in `run_benchmark.py`.

## What's not included, and why

- **The FinanceBench PDFs and extracted corpus.** ~140MB, CC BY-NC 4.0,
  fetched fresh by `download_data.py`, not redistributed here.
- **The Qdrant collection / vector index.** Regenerable in one command from
  the corpus; keeping it out of git keeps the repo small.

## Citation

If you use this repo's method or results, a link back is appreciated. Cite
FinanceBench separately if you use the dataset: ["FinanceBench: A New
Benchmark for Financial Question Answering",
2023](https://arxiv.org/abs/2311.11944). Cite the papers this repo tests
between if you reference their claims:
[arXiv:2602.23368](https://arxiv.org/abs/2602.23368),
[arXiv:2605.15184](https://arxiv.org/abs/2605.15184),
[arXiv:2605.05538](https://arxiv.org/abs/2605.05538).

## License

MIT for the code in this repository. See [LICENSE](LICENSE). FinanceBench
is separately licensed (CC BY-NC 4.0, non-commercial) and is not
redistributed here.
