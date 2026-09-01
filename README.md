# qdrant-agentic-retrieval-bench

Does an agent actually need a vector database, or does keyword search over
the raw documents get it most of the way there? A February 2026 Amazon
Science paper found a ReAct agent using plain keyword tools (`rga`,
`pdfgrep`) reaches 88-94.5% of RAG's context recall and answer correctness
across several benchmarks, no vector store involved, and a June 2026
follow-up states the literature has never systematically isolated
retrieval strategy from agent-harness design. This repo runs that
comparison directly: the same local ReAct agent, the same real financial
filings, the same questions, with only the retrieval tool swapped between
a keyword-search implementation and Qdrant vector search.

## Why this exists

Every repo elsewhere in this project benchmarks Qdrant's own internals
(payload indexing cost, write-pattern cost, quantization tradeoffs). None
of them ask the more basic question an agent-building team actually faces
first: does adding a vector database to an agent's tool belt beat giving
it a keyword-search tool over the same documents? That question is
contested in the current literature, not settled:

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

## Method

**Data**: [FinanceBench](https://github.com/patronus-ai/financebench)
open-source sample (CC BY-NC 4.0), the same benchmark the Amazon Science
paper above reports FinanceBench-specific numbers on. 150 real
question/answer pairs over real SEC filings (10-Ks, 10-Qs), each with a
gold source document and evidence page. `download_data.py` fetches the
150 questions plus the 84 real PDF filings they reference (not vendored
in this repo, ~140MB) directly from the source repo. Corpus: 12,013
extracted pages across those 84 filings, verified page-number-exact
against FinanceBench's own `evidence_page_num` field on a 30-question
sample before any indexing happened.

**Agent**: a local ReAct loop (`agent.py`) running
[`qwen2.5:3b-instruct`](https://ollama.com/library/qwen2.5) via Ollama,
4-bit quantized (Ollama's default `Q4_K_M`), chosen deliberately small so
this reproduces on ordinary hardware with no API key and no cost. The
agent gets exactly one tool per run and up to 4 turns to call it and
answer.

**Two tools, same corpus, same page-level granularity:**

- **`keyword_search`** — lexical TF-IDF search over the raw extracted
  page text (term frequency x inverse document frequency across the
  12,013-page corpus). No embeddings, no vector index; the same category
  of mechanism `rga`/`pdfgrep` give an agent in the source papers.
  Implemented in pure Python (`tools.py`), no ripgrep binary dependency.
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
  score, see Limitations.
- **Tool calls, turns, wall-clock latency** per run.

Sample: a stratified 45-question sample (15 from each of FinanceBench's
three `question_type` categories: metrics-generated, domain-relevant,
novel-generated), each run once per condition (90 agent runs total),
fixed seed for reproducibility.

## Results

Full numbers: [`results/summary.json`](results/summary.json). Raw per-question
data: [`results/raw_results.jsonl`](results/raw_results.jsonl). 45 questions
(15 per FinanceBench `question_type`), each run once per condition, 90 agent
runs total, `qwen2.5:3b-instruct`.

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

## What this actually shows

Vector search finds the actual gold evidence page 18x more often than keyword
search does in this run (40.0% vs 2.2% evidence recall), a larger gap than
either paper in "Why this exists" reports for FinanceBench specifically. That
gap traces to a mechanism visible directly in the data, not a black box:
FinanceBench questions are written in analyst language ("capital
expenditures"), while the filings themselves use GAAP line-item phrasing
("purchases of property, plant and equipment"). Confirmed directly during
development on one 3M capex question: the terms "capital" and "expenditure"
have zero term-frequency on the actual gold page. TF-IDF over exact tokens
has no way to bridge that gap; embedding similarity does.

Answer correctness moves the same direction but by a much smaller margin
(33.3% vs 22.2%), and the by-type breakdown shows why the overall number
undersells the retrieval gap and oversells the answer-quality gap:

- **domain-relevant**: `keyword_search` never once returned the correct
  evidence page (0/15), yet the agent still got 53.3% of final answers right
  anyway. These are the FinanceBench questions closest to common financial
  knowledge; a 3B model appears to already know a fair number of them
  unaided, independent of what its tool call actually surfaced. For this
  question type specifically, `answer_correct` is not a trustworthy signal
  of the retrieval tool doing real work, `evidence_recall` is.
- **metrics-generated**: `vector_search` had its best evidence recall of any
  cell (60%) but its worst answer correctness (6.7%), worse than keyword's
  0% on the same type. The agent found the right page and still got the
  number wrong. That is a distinct failure mode from a retrieval failure:
  extracting one precise number out of a dense financial table, truncated to
  1200 characters of page text, is a genuinely hard task for a 3B model that
  better retrieval alone does not fix.
- **novel-generated** is the one type where the result matches the "why this
  exists" framing directly: vector search beats keyword on both evidence
  recall (26.7% vs 6.7%) and answer correctness (33.3% vs 13.3%).

The honest read: for this agent size, this corpus, and this tool
implementation, vector search is structurally better at finding the right
page, largely for the reason named above (vocabulary mismatch), but that
improvement does not translate cleanly into better final answers, because
answer quality is gated by a second, separate bottleneck, whether a small
local model can correctly read a number back out of what it retrieves. A
benchmark that only measured answer correctness would have badly understated
the retrieval-mechanism gap on domain-relevant questions, and badly
overstated it on metrics-generated ones. That is the argument for reporting
evidence recall as its own metric rather than folding everything into one
pass/fail number. Vector search also costs about 2.1s more median latency
per run (13.8s vs 11.7s): the embedding call plus a network round trip to
Qdrant, against an in-process Python dict scan for keyword search.

## Limitations

- **Answer-correctness is a numeric-tolerance heuristic, not an LLM
  judge.** Running a second LLM as judge would introduce that model's own
  error into the measurement; the tradeoff is that some correct
  free-text answers with no clean number in them may be scored wrong.
  Evidence recall (a page-match check, not a judgment call) is the more
  reliable of the two metrics for that reason.
- **A 3B, 4-bit-quantized local model, not a frontier model.** This tests
  whether the *retrieval choice* matters for an agent this size on this
  hardware; a larger model's tool-use reasoning could change the gap
  between conditions in either direction.
- **45 of 150 questions, not all of them**, for tractability against a
  local model with no batching. Stratified by question_type to avoid
  skewing toward one question style.
- **Single-hop retrieval**: the agent has 4 turns and can re-search, but
  this doesn't test genuinely multi-hop questions requiring evidence
  assembled from multiple distinct pages.
- **Keyword search is TF-IDF over already-extracted page text, not a
  literal `rg`/`pdfgrep` subprocess.** Functionally the same category of
  tool (lexical, no embeddings), chosen for zero system-binary
  dependencies; a real ripgrep-backed tool could behave somewhat
  differently on multi-word queries.

## Reproducing this

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python3 download_data.py     # fetches real FinanceBench questions + 84 PDFs (~140MB)
python3 extract_corpus.py    # extracts page-level text with pypdf
python3 build_index.py       # embeds + indexes into a local Qdrant collection
python3 run_benchmark.py     # runs the stratified sample through both conditions
python3 analyze_results.py   # aggregates results/raw_results.jsonl into results/summary.json
```

Requires Docker (a local Qdrant container) and [Ollama](https://ollama.com)
with `qwen2.5:3b-instruct` pulled (`ollama pull qwen2.5:3b-instruct`,
~2GB). `run_benchmark.py --resume` skips (question, condition) pairs
already recorded, safe to re-run after an interruption.

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
- **The Qdrant collection / vector index.** Regenerable in one command
  from the corpus; keeping it out of git keeps the repo small.

## Citation

If you use this repo's method or results, a link back is appreciated.
Cite FinanceBench separately if you use the dataset:
["FinanceBench: A New Benchmark for Financial Question Answering", 2023](https://arxiv.org/abs/2311.11944).
Cite the papers
this repo tests between if you reference their claims:
[arXiv:2602.23368](https://arxiv.org/abs/2602.23368),
[arXiv:2605.15184](https://arxiv.org/abs/2605.15184),
[arXiv:2605.05538](https://arxiv.org/abs/2605.05538).

## License

MIT for the code in this repository. See [LICENSE](LICENSE). FinanceBench
is separately licensed (CC BY-NC 4.0, non-commercial) and is not
redistributed here.
