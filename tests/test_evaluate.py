from evaluate import _extract_numbers, is_correct


def test_extracts_scaled_and_percent_numbers():
    assert _extract_numbers("$2.5 billion") == [2.5e9]
    assert _extract_numbers("1,577 million") == [1.577e9]
    assert _extract_numbers("63.4%") == [63.4]


def test_numeric_match_within_tolerance():
    assert is_correct("Capex was $1,577 million", "$1577.00 million")
    assert is_correct("about 100.5", "100", rel_tol=0.02)
    assert not is_correct("about 120", "100", rel_tol=0.02)


def test_missing_number_is_incorrect():
    assert not is_correct("I could not find it", "$1577.00")


def test_non_numeric_reference_uses_substring_match():
    assert is_correct("The company is Netflix, Inc.", "netflix")
    assert not is_correct("Hulu", "netflix")


def test_known_heuristic_weakness_year_can_match():
    # Documented in the README: a matching year can pass a wrong answer.
    assert is_correct("In 2018 it was unclear", "2018")
