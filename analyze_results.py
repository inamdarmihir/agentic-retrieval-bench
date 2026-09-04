"""Aggregate a raw_results.jsonl file into the summary table reported in
the README: evidence recall, answer correctness, tool calls, and latency,
by condition and by question_type.

Defaults to the committed results/raw_results.jsonl -> results/summary.json
pair. Pass --in/--out to point at a different file (e.g. results from
`make smoke`) without touching the committed results.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

DEFAULT_IN = Path(__file__).parent / "results" / "raw_results.jsonl"
DEFAULT_OUT = Path(__file__).parent / "results" / "summary.json"


def summarize(in_path: Path = DEFAULT_IN, out_path: Path = DEFAULT_OUT) -> None:
    with open(in_path) as f:
        rows = [json.loads(line) for line in f]

    def agg(rows: list[dict]) -> dict:
        n = len(rows)
        if n == 0:
            return {}
        return {
            "n": n,
            "evidence_recall": sum(r["hit_evidence"] for r in rows) / n,
            "answer_correct": sum(r["correct"] for r in rows) / n,
            "avg_tool_calls": sum(r["n_tool_calls"] for r in rows) / n,
            "avg_turns": sum(r["turns_used"] for r in rows) / n,
            "avg_wall_clock_s": sum(r["wall_clock_s"] for r in rows) / n,
            "p50_wall_clock_s": sorted(r["wall_clock_s"] for r in rows)[n // 2],
        }

    print("=== Overall, by condition ===")
    by_condition = defaultdict(list)
    for r in rows:
        by_condition[r["condition"]].append(r)
    for cond in sorted(by_condition):
        print(f"\n{cond}:")
        for k, v in agg(by_condition[cond]).items():
            print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\n\n=== By question_type x condition ===")
    by_type_cond = defaultdict(list)
    for r in rows:
        by_type_cond[(r["question_type"], r["condition"])].append(r)
    for (qtype, cond) in sorted(by_type_cond):
        stats = agg(by_type_cond[(qtype, cond)])
        print(
            f"{qtype:20s} {cond:8s} n={stats['n']:3d} "
            f"evidence_recall={stats['evidence_recall']:.2f} "
            f"answer_correct={stats['answer_correct']:.2f} "
            f"avg_tool_calls={stats['avg_tool_calls']:.1f} "
            f"p50_latency={stats['p50_wall_clock_s']:.1f}s"
        )

    summary = {
        "overall": {cond: agg(rs) for cond, rs in by_condition.items()},
        "by_type": {f"{qtype}__{cond}": agg(rs) for (qtype, cond), rs in by_type_cond.items()},
    }
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="in_path", type=Path, default=DEFAULT_IN, help="raw results JSONL to read")
    parser.add_argument("--out", dest="out_path", type=Path, default=DEFAULT_OUT, help="summary JSON to write")
    args = parser.parse_args()
    summarize(args.in_path, args.out_path)


if __name__ == "__main__":
    main()
