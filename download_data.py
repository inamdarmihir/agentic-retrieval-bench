"""Fetch the real FinanceBench open-source data this benchmark runs against.

Pulls the 150-question open-source sample (CC BY-NC 4.0) and, by default,
all 84 SEC filing PDFs those questions reference, directly from Patronus
AI's financebench repo. Not vendored in this repo (the full PDF set is
~140MB); this script reproduces data/questions.jsonl and data/pdfs/ from
scratch.

Pass --smoke to fetch only the 3 PDFs needed for the smoke sample (see
sampling.py / `make smoke`), a few MB instead of ~140MB, so trying the
pipeline end-to-end doesn't require the full download.
"""

import argparse
import json
import subprocess
from pathlib import Path

from sampling import stratified_sample

DATA_DIR = Path(__file__).parent / "data"
PDF_DIR = DATA_DIR / "pdfs"
QUESTIONS_URL = "https://raw.githubusercontent.com/patronus-ai/financebench/main/data/financebench_open_source.jsonl"
PDF_URL_TMPL = "https://raw.githubusercontent.com/patronus-ai/financebench/main/pdfs/{doc}.pdf"

SMOKE_N_PER_TYPE = 1  # 1 question per question_type = 3 total; keep in sync with `make smoke`


def download_questions() -> list[dict]:
    DATA_DIR.mkdir(exist_ok=True)
    dest = DATA_DIR / "questions.jsonl"
    subprocess.run(["curl", "-sS", "--retry", "5", "-o", str(dest), QUESTIONS_URL], check=True)
    with open(dest) as f:
        rows = [json.loads(line) for line in f]
    print(f"{len(rows)} questions -> {dest}")
    return rows


def download_pdfs(doc_names: list[str]) -> None:
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    doc_names = sorted(set(doc_names))
    print(f"{len(doc_names)} unique source documents to fetch")
    for doc in doc_names:
        dest = PDF_DIR / f"{doc}.pdf"
        if dest.exists() and dest.stat().st_size > 10_000:
            continue
        url = PDF_URL_TMPL.format(doc=doc)
        subprocess.run(["curl", "-sS", "--retry", "5", "--retry-delay", "2", "-o", str(dest), url], check=True)
        print(f"  {doc}.pdf ({dest.stat().st_size:,} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Only fetch the PDFs referenced by the smoke sample "
            f"({SMOKE_N_PER_TYPE} per question_type, 3 total) instead of all 84 filings. "
            "A few MB instead of ~140MB."
        ),
    )
    args = parser.parse_args()

    rows = download_questions()
    if args.smoke:
        sample = stratified_sample(rows, n_per_type=SMOKE_N_PER_TYPE)
        doc_names = [q["doc_name"] for q in sample]
        print(f"--smoke: fetching PDFs for {len(doc_names)} question(s): {', '.join(doc_names)}")
    else:
        doc_names = [row["doc_name"] for row in rows]
    download_pdfs(doc_names)
    print("\nDone. Next: python extract_corpus.py && python build_index.py")


if __name__ == "__main__":
    main()
