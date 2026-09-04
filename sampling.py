"""Stratified sampling shared by run_benchmark.py (the full 90-run
experiment) and download_data.py (the smoke path's minimal PDF fetch).

Sharing this function is what guarantees the smoke sample is a strict
subset of the full sample: for a fixed SEED, asking for n_per_type=1
returns exactly the first question of each question_type that
n_per_type=15 would also return, because random.Random(seed).shuffle()
on the same per-type list consumes the same random state regardless of
how many items you slice off the front afterwards.
"""

import random

SEED = 42


def stratified_sample(questions: list[dict], n_per_type: int, seed: int = SEED) -> list[dict]:
    by_type: dict[str, list[dict]] = {}
    for q in questions:
        by_type.setdefault(q["question_type"], []).append(q)
    rng = random.Random(seed)
    sample = []
    for qtype, rows in sorted(by_type.items()):
        rng.shuffle(rows)
        sample.extend(rows[:n_per_type])
    rng.shuffle(sample)
    return sample
