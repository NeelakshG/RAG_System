from collections import defaultdict

from src.tokenizer import tokenize


def reciprocal_rank_fusion(
    dense_results: list[str],
    sparse_results: list[str],
    k: int = 60,
) -> list[str]:
    """Merge two ranked chunk_id lists (dense + sparse) into one combined
    ranking, best first, via Reciprocal Rank Fusion: score(chunk) = sum of
    1/(k + rank) across every list the chunk appears in. A chunk found by
    BOTH methods accumulates both contributions and outranks a chunk found
    by only one -- this is what makes hybrid search reward agreement
    between dense (meaning) and sparse (exact keyword) retrieval.
    """
    scores = defaultdict(float)  # any chunk_id not seen yet defaults to 0.0

    for rank, chunk_id in enumerate(dense_results, start=1):  # start=1: RRF ranks are 1-indexed
        scores[chunk_id] += 1 / (k + rank)

    for rank, chunk_id in enumerate(sparse_results, start=1):
        scores[chunk_id] += 1 / (k + rank)  # += so a chunk in both lists sums both contributions

    ranked_pairs = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
    return [chunk_id for chunk_id, score in ranked_pairs]


class CrossEncoderReranker:
    """Real reranker: wraps sentence-transformers' CrossEncoder
    (cross-encoder/ms-marco-MiniLM-L-6-v2). Exposes .score(query, texts)
    -> list[float] -- the same interface any stub reranker fakes in tests.
    Lazy import keeps this the only place allowed to import CrossEncoder
    directly.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model_name)

    def score(self, query: str, texts: list[str]) -> list[float]:
        pairs = [(query, text) for text in texts]
        return self._model.predict(pairs).tolist()


def rerank(
    query: str,
    candidates: list[tuple[str, str]],
    reranker,
    top_n: int = 5,
) -> list[str]:
    """Score each (chunk_id, chunk_text) candidate against the query using
    the cross-encoder, return the top_n chunk_ids sorted by score, best first.
    """
    chunk_ids = [chunk_id for chunk_id, text in candidates]
    texts = [text for chunk_id, text in candidates]

    scores = reranker.score(query, texts)

    ranked_pairs = sorted(zip(chunk_ids, scores), key=lambda pair: pair[1], reverse=True)
    top = ranked_pairs[:top_n]
    return [chunk_id for chunk_id, score in top]


def retrieve(
    query: str,
    dense_index,
    sparse_index,
    embedder,
    reranker,
    config=None,
    use_hybrid: bool = True,
) -> list[str]:
    """Run the full retrieval funnel for one query: embed + tokenize ->
    dense/sparse search -> RRF fusion (skipped if use_hybrid=False) ->
    fetch candidate text -> cross-encoder rerank. Returns the final
    chunk_ids, best first.
    """
    dense_k = getattr(config, "dense_k", 10)
    sparse_k = getattr(config, "sparse_k", 10)
    rrf_k = getattr(config, "rrf_k", 60)
    fusion_candidates = getattr(config, "fusion_candidates", 20)
    final_k = getattr(config, "final_k", 5)

    query_embedding = embedder.embed([query])[0]
    query_tokens = tokenize(query)

    dense_results = dense_index.query(query_embedding, k=dense_k)

    if use_hybrid:
        sparse_results = sparse_index.query(query_tokens, k=sparse_k)
        fused = reciprocal_rank_fusion(dense_results, sparse_results, k=rrf_k)
    else:
        fused = dense_results

    top_fused = fused[:fusion_candidates]

    texts = dense_index.get_texts(top_fused)
    candidates = [(chunk_id, texts[chunk_id]) for chunk_id in top_fused]

    return rerank(query, candidates, reranker, top_n=final_k)
