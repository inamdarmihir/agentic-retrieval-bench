from tools import KeywordSearchTool, SearchHit, format_hits

CORPUS = [
    {"doc_name": "A_2020_10K", "page_num": 1, "text": "capital expenditures were 5 million"},
    {"doc_name": "B_2020_10K", "page_num": 2, "text": "the amount of capital raised was large"},
    {"doc_name": "C_2020_10K", "page_num": 3, "text": "unrelated boilerplate text"},
]


def test_rare_terms_outrank_common_ones():
    hits = KeywordSearchTool(CORPUS).search("capital expenditures")
    assert [h.doc_name for h in hits][0] == "A_2020_10K"


def test_no_match_and_empty_query_return_nothing():
    tool = KeywordSearchTool(CORPUS)
    assert tool.search("zzzz") == []
    assert tool.search("   ") == []


def test_k_limits_results():
    assert len(KeywordSearchTool(CORPUS).search("capital", k=1)) == 1


def test_format_hits_truncates_and_labels():
    text = format_hits([SearchHit("A_2020_10K", 7, "x" * 5000, 1.0)], max_chars_per_hit=10)
    assert text.startswith("[A_2020_10K, page 7]")
    assert text.endswith("x" * 10)
    assert format_hits([]) == "No matches found."
