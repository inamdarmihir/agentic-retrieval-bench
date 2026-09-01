"""Extract page-level text from the FinanceBench PDF corpus.

Reads every PDF in data/pdfs/, extracts text page by page with pypdf, and
writes one JSON object per page to data/corpus.jsonl:

    {"doc_name": "3M_2018_10K", "page_num": 42, "text": "..."}

page_num is 0-indexed, matching FinanceBench's own evidence_page_num field,
so downstream evaluation can compare directly without an off-by-one fixup.
"""

import json
from pathlib import Path

import pypdf

PDF_DIR = Path(__file__).parent / "data" / "pdfs"
OUT_PATH = Path(__file__).parent / "data" / "corpus.jsonl"


def extract_all() -> None:
    pdf_paths = sorted(PDF_DIR.glob("*.pdf"))
    if not pdf_paths:
        raise SystemExit(f"No PDFs found in {PDF_DIR}. Run download_data.py first.")

    total_pages = 0
    empty_pages = 0
    with open(OUT_PATH, "w") as out:
        for pdf_path in pdf_paths:
            doc_name = pdf_path.stem
            reader = pypdf.PdfReader(pdf_path)
            if reader.is_encrypted:
                reader.decrypt("")
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                text = text.strip()
                if not text:
                    empty_pages += 1
                out.write(json.dumps({"doc_name": doc_name, "page_num": page_num, "text": text}) + "\n")
                total_pages += 1
            print(f"{doc_name}: {len(reader.pages)} pages")

    print(f"\nWrote {total_pages} pages ({empty_pages} empty) from {len(pdf_paths)} documents to {OUT_PATH}")


if __name__ == "__main__":
    extract_all()
