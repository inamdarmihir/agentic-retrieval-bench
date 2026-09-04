.PHONY: install download extract index benchmark analyze smoke clean help

help:
	@echo "agentic-retrieval-bench"
	@echo ""
	@echo "  make install    Install pinned Python dependencies"
	@echo "  make smoke      ~3-question smoke test (~2-3MB download, a few minutes)"
	@echo "  make download   Fetch all 150 questions + 84 PDFs (~140MB, full run)"
	@echo "  make extract    Extract page text from downloaded PDFs"
	@echo "  make index      Embed + index the corpus into Qdrant"
	@echo "  make benchmark  Run the full 45-question / 90-run benchmark"
	@echo "  make analyze    Aggregate results/raw_results.jsonl -> results/summary.json"
	@echo "  make clean      Remove downloaded PDFs, extracted corpus, and smoke results"
	@echo ""
	@echo "Requires Docker (for Qdrant) and Ollama with qwen2.5:3b-instruct pulled."
	@echo "See README.md 'Quickstart' and 'Smoke test' sections before running."

install:
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt

download:
	python3 download_data.py

extract:
	python3 extract_corpus.py

index:
	python3 build_index.py

benchmark:
	python3 run_benchmark.py --resume

analyze:
	python3 analyze_results.py

# Smoke test: 1 question per question_type (3 total, a subset of the
# committed 45-question sample), 6 agent runs instead of 90. Writes to
# results/smoke_results.jsonl and results/smoke_summary.json so it never
# touches the committed results/raw_results.jsonl or results/summary.json.
# See README.md "Smoke test" for expected output and what it does/doesn't prove.
smoke:
	@echo "== 1/5: fetching Qdrant (docker) and confirming Ollama model =="
	@command -v docker >/dev/null || (echo "Docker not found. Install Docker and rerun." && exit 1)
	@docker ps >/dev/null 2>&1 || (echo "Docker daemon not reachable. Start Docker and rerun." && exit 1)
	@docker start qdrant-smoke >/dev/null 2>&1 || docker run -d --name qdrant-smoke -p 6333:6333 qdrant/qdrant >/dev/null
	@echo "== 2/5: downloading smoke sample PDFs (~3MB) =="
	python3 download_data.py --smoke
	@echo "== 3/5: extracting page text =="
	python3 extract_corpus.py
	@echo "== 4/5: embedding + indexing into Qdrant =="
	python3 build_index.py
	@echo "== 5/5: running 3 questions x 2 conditions (6 agent runs) =="
	python3 run_benchmark.py --n-per-type 1 --out results/smoke_results.jsonl
	python3 analyze_results.py --in results/smoke_results.jsonl --out results/smoke_summary.json
	@echo ""
	@echo "Smoke test done. See results/smoke_summary.json (not the committed results/summary.json)."
	@echo "3 questions is not a statistically meaningful sample; this only proves the pipeline runs end to end."

clean:
	rm -rf data/pdfs data/corpus.jsonl results/smoke_results.jsonl results/smoke_summary.json
	docker rm -f qdrant-smoke >/dev/null 2>&1 || true
