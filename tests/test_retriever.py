from src.retriever import reciprocal_rank_fusion, rerank, retrieve


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

    assert result[0][0] == "a::0"   # shares both query words -- clear winner
    assert len(result) == 2


def test_rerank_respects_top_n():
    candidates = [("a::0", "alpha"), ("b::0", "beta"), ("c::0", "gamma")]

    result = rerank("alpha beta gamma", candidates, StubReranker(), top_n=1)

    assert len(result) == 1


class StubDenseIndex:
    _TEXTS = {"a::0": "alpha text", "b::0": "beta text", "c::0": "gamma text", "d::0": "delta text"}

    def query(self, query_embedding, k=10):
        return ["a::0", "b::0", "c::0"][:k]

    def get_texts(self, chunk_ids):
        return {cid: self._TEXTS[cid] for cid in chunk_ids}


class StubSparseIndex:
    def query(self, query_tokens, k=10):
        return ["b::0", "d::0"][:k]


class StubQueryEmbedder:
    def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]


def test_retrieve_hybrid_runs_full_funnel():
    result = retrieve(
        "some query", StubDenseIndex(), StubSparseIndex(),
        StubQueryEmbedder(), StubReranker(),
        use_hybrid=True,
    )

    result_ids = [chunk_id for chunk_id, score in result]
    assert "b::0" in result_ids   # found by both dense and sparse -- should survive to the end


def test_retrieve_dense_only_skips_sparse():
    result = retrieve(
        "some query", StubDenseIndex(), StubSparseIndex(),
        StubQueryEmbedder(), StubReranker(),
        use_hybrid=False,
    )

    result_ids = {chunk_id for chunk_id, score in result}
    assert result_ids.issubset({"a::0", "b::0", "c::0"})   # never touches sparse's d::0


class StubFilterableDenseIndex(StubDenseIndex):
    """Records whether source_names reached query() -- StubDenseIndex itself
    has no source_names param, so retrieve() must not pass the kwarg unless
    a caller actually asked for a filtered search."""

    last_source_names = "not called"

    def query(self, query_embedding, k=10, source_names=None):
        type(self).last_source_names = source_names
        return super().query(query_embedding, k)


class StubFilterableSparseIndex(StubSparseIndex):
    last_source_names = "not called"

    def query(self, query_tokens, k=10, source_names=None):
        type(self).last_source_names = source_names
        return super().query(query_tokens, k)


def test_retrieve_without_source_names_never_passes_the_kwarg():
    """Default ('all documents') path must keep working against index
    implementations that don't know about source_names at all."""
    retrieve(
        "some query", StubDenseIndex(), StubSparseIndex(),
        StubQueryEmbedder(), StubReranker(),
    )  # no source_names -- would raise TypeError if retrieve() always passed the kwarg


def test_retrieve_forwards_source_names_to_both_indexes():
    dense = StubFilterableDenseIndex()
    sparse = StubFilterableSparseIndex()

    retrieve(
        "some query", dense, sparse,
        StubQueryEmbedder(), StubReranker(),
        source_names=["a.md"],
    )

    assert StubFilterableDenseIndex.last_source_names == ["a.md"]
    assert StubFilterableSparseIndex.last_source_names == ["a.md"]
