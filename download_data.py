"""Fetch the real FinanceBench open-source data this benchmark runs against.

Pulls the 150-question open-source sample (CC BY-NC 4.0) and the specific
84 SEC filing PDFs those questions reference, directly from Patronus AI's
financebench repo. Not vendored in this repo (the PDFs alone are ~140MB);
this script reproduces data/questions.jsonl and data/pdfs/ from scratch.
"""

import json
import subprocess
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
PDF_DIR = DATA_DIR / "pdfs"
QUESTIONS_URL = "https://raw.githubusercontent.com/patronus-ai/financebench/main/data/financebench_open_source.jsonl"
PDF_URL_TMPL = "https://raw.githubusercontent.com/patronus-ai/financebench/main/pdfs/{doc}.pdf"


def download_questions() -> list[dict]:
    DATA_DIR.mkdir(exist_ok=True)
    dest = DATA_DIR / "questions.jsonl"
    subprocess.run(["curl", "-sS", "--retry", "5", "-o", str(dest), QUESTIONS_URL], check=True)
    with open(dest) as f:
        rows = [json.loads(line) for line in f]
    print(f"{len(rows)} questions -> {dest}")
    return rows


def download_pdfs(rows: list[dict]) -> None:
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    doc_names = sorted({row["doc_name"] for row in rows})
    print(f"{len(doc_names)} unique source documents to fetch")
    for doc in doc_names:
        dest = PDF_DIR / f"{doc}.pdf"
        if dest.exists() and dest.stat().st_size > 10_000:
            continue
        url = PDF_URL_TMPL.format(doc=doc)
        subprocess.run(["curl", "-sS", "--retry", "5", "--retry-delay", "2", "-o", str(dest), url], check=True)
        print(f"  {doc}.pdf ({dest.stat().st_size:,} bytes)")


if __name__ == "__main__":
    rows = download_questions()
    download_pdfs(rows)
    print("\nDone. Next: python extract_corpus.py && python build_index.py")
