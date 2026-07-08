import math
from dataclasses import dataclass

from src.llm import (
    NO_ANSWER_SENTINEL,
    SENTENCE_SPLIT_PATTERN,
    CitationCheck,
    OllamaClient,
    extract_claims,
    verify_citations,
)
from src.models import Chunk


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
    """Composite confidence score for a generated answer: a weighted
    GEOMETRIC mean of how confident retrieval was, how many citations
    verification actually confirmed, and how much of the answer is
    grounded at all. Geometric mean (not arithmetic) means any one weak
    component drags the whole score down -- strong retrieval can't mask
    a hallucinated citation the way an average would let it.
    """
    r = retrieval_confidence(retrieval_results)
    c = citation_coverage(citation_checks)
    p = completeness(answer)
    return (r ** retrieval_weight) * (c ** citation_weight) * (p ** completeness_weight)


@dataclass
class AnswerResult:
    text: str
    confidence: float
    fallback_triggered: bool


def answer_with_confidence(
    query: str,
    chunks: list[Chunk],
    retrieval_results: list[tuple[str, float]],
    client: OllamaClient,
    threshold: float = 0.5,
) -> AnswerResult:
    """Generate a grounded answer, verify its citations, and score its
    confidence. Falls back to NO_ANSWER_SENTINEL when confidence is below
    `threshold`, keeping the real computed score so callers can see how
    close it was. Short-circuits (no fallback, no wasted LLM calls) when
    retrieval found nothing, or when the model already said it doesn't know.
    """
    if not chunks:
        return AnswerResult(text=NO_ANSWER_SENTINEL, confidence=0.0, fallback_triggered=False)

    text = client.answer(query, chunks)

    if text == NO_ANSWER_SENTINEL:
        return AnswerResult(text=text, confidence=0.0, fallback_triggered=False)

    checks = verify_citations(text, chunks, client)
    confidence = compute_confidence(text, retrieval_results, checks)

    if confidence < threshold:
        return AnswerResult(text=NO_ANSWER_SENTINEL, confidence=confidence, fallback_triggered=True)

    return AnswerResult(text=text, confidence=confidence, fallback_triggered=False)
