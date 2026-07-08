import math

from src.llm import SENTENCE_SPLIT_PATTERN, CitationCheck, extract_claims


def retrieval_confidence(retrieval_results: list[tuple[str, float]]) -> float:
    """Squash the top reranked chunk's cross-encoder score into (0, 1) via
    sigmoid. The cross-encoder's raw score is an unbounded logit -- higher
    means a stronger match -- so sigmoid gives a comparable, bounded signal
    for how confident retrieval was about its best chunk.
    """
    if not retrieval_results:
        return 0.0
    top_score = retrieval_results[0][1]
    return 1 / (1 + math.exp(-top_score))


def citation_coverage(citation_checks: list[CitationCheck]) -> float:
    """Fraction of citations that verification confirmed were actually
    supported by their chunk. No citations at all means nothing was
    verified -- treated as zero coverage, not undefined.
    """
    if not citation_checks:
        return 0.0
    supported = sum(1 for check in citation_checks if check.supported)
    return supported / len(citation_checks)


def completeness(answer: str) -> float:
    """Fraction of the answer's sentences that carry at least one citation.
    A high score means most of the answer is grounded in cited evidence
    rather than asserted without a source.
    """
    sentences = [s for s in SENTENCE_SPLIT_PATTERN.split(answer.strip()) if s]
    if not sentences:
        return 0.0
    cited_sentences = extract_claims(answer)
    return len(cited_sentences) / len(sentences)


def compute_confidence(
    answer: str,
    retrieval_results: list[tuple[str, float]],
    citation_checks: list[CitationCheck],
    retrieval_weight: float = 1 / 3,
    citation_weight: float = 1 / 3,
    completeness_weight: float = 1 / 3,
) -> float:
    """Composite confidence score for a generated answer: a weighted blend
    of how confident retrieval was, how many citations verification actually
    confirmed, and how much of the answer is grounded at all.
    """
    return (
        retrieval_weight * retrieval_confidence(retrieval_results)
        + citation_weight * citation_coverage(citation_checks)
        + completeness_weight * completeness(answer)
    )
