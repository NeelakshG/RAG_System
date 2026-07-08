from src.retriever import reciprocal_rank_fusion, rerank


def test_reciprocal_rank_fusion_rewards_chunks_in_both_lists():
    dense_results = ["a::0", "b::0", "c::0"]
    sparse_results = ["b::0", "d::0"]

    fused = reciprocal_rank_fusion(dense_results, sparse_results, k=60)

    assert fused[0] == "b::0"                              # in both lists -- wins overall
    assert set(fused) == {"a::0", "b::0", "c::0", "d::0"}   # union of both, nothing dropped


def test_reciprocal_rank_fusion_single_list_still_ranks():
    fused = reciprocal_rank_fusion(["x::0", "y::0"], [], k=60)

    assert fused == ["x::0", "y::0"]


def test_reciprocal_rank_fusion_empty_inputs_returns_empty():
    assert reciprocal_rank_fusion([], [], k=60) == []


class StubReranker:
    """Deterministic offline reranker: score = count of query words that
    also appear in the text (case-insensitive) -- no real model needed."""

    def score(self, query, texts):
        query_words = set(query.lower().split())
        return [float(len(query_words & set(t.lower().split()))) for t in texts]


def test_rerank_picks_highest_scoring_candidate_first():
    candidates = [
        ("a::0", "ERR_2043 troubleshooting guide"),
        ("b::0", "unrelated cat facts"),
        ("c::0", "some retry configuration info"),
    ]
    query = "ERR_2043 troubleshooting"

    result = rerank(query, candidates, StubReranker(), top_n=2)

    assert result[0] == "a::0"   # shares both query words -- clear winner
    assert len(result) == 2


def test_rerank_respects_top_n():
    candidates = [("a::0", "alpha"), ("b::0", "beta"), ("c::0", "gamma")]

    result = rerank("alpha beta gamma", candidates, StubReranker(), top_n=1)

    assert len(result) == 1
