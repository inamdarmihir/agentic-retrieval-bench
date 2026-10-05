from sampling import stratified_sample


def make_questions():
    return [{"financebench_id": f"{t}-{i}", "question_type": t}
            for t in ("a", "b", "c") for i in range(20)]


def test_n_per_type_is_respected():
    sample = stratified_sample(make_questions(), 5)
    assert len(sample) == 15
    for t in ("a", "b", "c"):
        assert sum(q["question_type"] == t for q in sample) == 5


def test_deterministic_for_a_fixed_seed():
    assert stratified_sample(make_questions(), 5) == stratified_sample(make_questions(), 5)


def test_smoke_sample_is_a_subset_of_the_full_sample():
    ids = lambda s: {q["financebench_id"] for q in s}
    assert ids(stratified_sample(make_questions(), 1)) <= ids(stratified_sample(make_questions(), 15))
