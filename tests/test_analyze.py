import json
from pathlib import Path

from analyze_results import summarize

RESULTS = Path(__file__).parent.parent / "results"


def test_committed_summary_matches_raw_results(tmp_path):
    """Re-aggregating the committed traces must reproduce the committed summary."""
    out = tmp_path / "summary.json"
    summarize(RESULTS / "raw_results.jsonl", out)
    assert json.loads(out.read_text()) == json.loads((RESULTS / "summary.json").read_text())


def test_raw_results_has_90_paired_runs():
    rows = [json.loads(line) for line in (RESULTS / "raw_results.jsonl").read_text().splitlines()]
    assert len(rows) == 90
    assert {r["condition"] for r in rows} == {"keyword", "vector"}
