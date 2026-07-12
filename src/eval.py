import json
from dataclasses import asdict, dataclass
from pathlib import Path

from src.confidence import citation_coverage, compute_confidence
from src.llm import NO_ANSWER_SENTINEL, SENTENCE_SPLIT_PATTERN, OllamaClient, verify_citations
from src.models import Chunk
from src.retriever import retrieve


@dataclass
class GoldenQuestion:
    id: str
    question: str
    category: str  # "lookup" | "multi-hop" | "no-answer" | "ambiguous"
    expected_answer: str
    expected_source_docs: list[str]


def load_golden_questions(path: str) -> list[GoldenQuestion]:
    with open(path) as f:
        data = json.load(f)
    return [GoldenQuestion(**entry) for entry in data]


def retrieval_relevance(question: GoldenQuestion, retrieved_chunks: list[Chunk]) -> float | None:
    """Fraction of expected_source_docs that appear among the retrieved
    chunks' source docs. None (not zero) when the question has no expected
    source docs at all -- e.g. no-answer questions, where relevance isn't
    a meaningful concept.
    """
    if not question.expected_source_docs:
        return None
    retrieved_docs = {chunk.source_name for chunk in retrieved_chunks}
    hits = sum(1 for doc in question.expected_source_docs if doc in retrieved_docs)
    return hits / len(question.expected_source_docs)


FAITHFULNESS_INSTRUCTIONS = (
    "You are checking whether a piece of context supports a claim. "
    "Answer only YES or NO -- no explanation.\n\n"
)


def build_faithfulness_prompt(sentence: str, context: str) -> str:
    return (
        f"{FAITHFULNESS_INSTRUCTIONS}"
        f"Context: {context}\n"
        f"Claim: {sentence}\n"
        "Does the context support the claim?"
    )


def judge_faithfulness(answer: str, chunks: list[Chunk], client: OllamaClient) -> float | None:
    """Fraction of the answer's sentences -- cited or not -- that the FULL
    retrieved context supports. Unlike citation_coverage (which only checks
    a claim against the specific chunk it cited), this catches claims that
    slipped in uncited or cited the wrong chunk. None when there's no answer
    to check.
    """
    if answer == NO_ANSWER_SENTINEL:
        return None
    sentences = [s for s in SENTENCE_SPLIT_PATTERN.split(answer.strip()) if s]
    if not sentences:
        return None

    context = "\n".join(chunk.text for chunk in chunks)
    supported = 0
    for sentence in sentences:
        response = client.generate(build_faithfulness_prompt(sentence, context))
        if response.strip().upper().startswith("YES"):
            supported += 1
    return supported / len(sentences)


CORRECTNESS_INSTRUCTIONS = (
    "You are grading whether a generated answer correctly answers a question, "
    "compared to a known-correct reference answer. Minor wording differences are "
    "fine as long as the key facts match. Answer only YES or NO -- no explanation.\n\n"
)


def build_correctness_prompt(question: str, expected_answer: str, actual_answer: str) -> str:
    return (
        f"{CORRECTNESS_INSTRUCTIONS}"
        f"Question: {question}\n"
        f"Reference answer: {expected_answer}\n"
        f"Generated answer: {actual_answer}\n"
        "Does the generated answer convey the same key facts as the reference answer?"
    )


AMBIGUITY_INSTRUCTIONS = (
    "You are grading whether a generated answer correctly recognizes that a "
    "question is ambiguous or underspecified. A correct answer asks for "
    "clarification, or explicitly names multiple valid interpretations, rather "
    "than confidently committing to a single interpretation as if it were the "
    "only one. Answer only YES or NO -- no explanation.\n\n"
)


def build_ambiguity_prompt(question: str, actual_answer: str) -> str:
    return (
        f"{AMBIGUITY_INSTRUCTIONS}"
        f"Question: {question}\n"
        f"Generated answer: {actual_answer}\n"
        "Does the generated answer recognize the ambiguity, rather than "
        "confidently picking one reading?"
    )


def judge_correctness(question: GoldenQuestion, answer: str, client: OllamaClient) -> bool:
    """Category-aware correctness check.

    no-answer: correct iff the system said it doesn't know -- deterministic,
    no LLM call needed.
    ambiguous: correct iff the system flagged the ambiguity (fallback counts
    as flagging it too) rather than confidently picking one interpretation.
    lookup / multi-hop: LLM judge compares against expected_answer.
    """
    if question.category == "no-answer":
        return answer == NO_ANSWER_SENTINEL

    if question.category == "ambiguous":
        if answer == NO_ANSWER_SENTINEL:
            return True
        response = client.generate(build_ambiguity_prompt(question.question, answer))
        return response.strip().upper().startswith("YES")

    response = client.generate(
        build_correctness_prompt(question.question, question.expected_answer, answer)
    )
    return response.strip().upper().startswith("YES")


@dataclass
class QuestionResult:
    id: str
    category: str
    strategy: str
    answer: str
    fallback_triggered: bool
    confidence: float
    retrieved_chunk_ids: list[str]
    retrieval_relevance: float | None
    faithfulness: float | None
    citation_accuracy: float | None
    correctness: bool


def run_question(
    question: GoldenQuestion,
    strategy: str,
    dense_index,
    sparse_index,
    embedder,
    reranker,
    client: OllamaClient,
    config=None,
) -> QuestionResult:
    """Run one golden question through the full pipeline (retrieve ->
    generate -> confidence-gated fallback) for one chunking strategy, then
    score all four eval metrics against the result.
    """
    threshold = getattr(config, "confidence_threshold", 0.5)

    retrieval_results = retrieve(question.question, dense_index, sparse_index, embedder, reranker, config)
    chunk_ids = [chunk_id for chunk_id, _ in retrieval_results]
    chunks = dense_index.get_chunks(chunk_ids)
    relevance = retrieval_relevance(question, chunks)

    checks = []
    confidence = 0.0
    fallback_triggered = False

    if not chunks:
        answer = NO_ANSWER_SENTINEL
    else:
        answer = client.answer(question.question, chunks)
        if answer != NO_ANSWER_SENTINEL:
            checks = verify_citations(answer, chunks, client)
            confidence = compute_confidence(answer, retrieval_results, checks)
            if confidence < threshold:
                fallback_triggered = True
                answer = NO_ANSWER_SENTINEL

    if answer == NO_ANSWER_SENTINEL:
        faithfulness = None
        citation_accuracy = None
    else:
        faithfulness = judge_faithfulness(answer, chunks, client)
        citation_accuracy = citation_coverage(checks)

    correctness = judge_correctness(question, answer, client)

    return QuestionResult(
        id=question.id,
        category=question.category,
        strategy=strategy,
        answer=answer,
        fallback_triggered=fallback_triggered,
        confidence=confidence,
        retrieved_chunk_ids=chunk_ids,
        retrieval_relevance=relevance,
        faithfulness=faithfulness,
        citation_accuracy=citation_accuracy,
        correctness=correctness,
    )


def _load_completed(path: Path) -> dict[str, QuestionResult]:
    if not path.exists():
        return {}
    completed = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            completed[data["id"]] = QuestionResult(**data)
    return completed


def run_eval(
    questions: list[GoldenQuestion],
    strategy: str,
    dense_index,
    sparse_index,
    embedder,
    reranker,
    client: OllamaClient,
    config=None,
    output_path: str | None = None,
) -> list[QuestionResult]:
    """Run every golden question through run_question for one chunking
    strategy, persisting incrementally to output_path (JSONL) so a crash
    mid-run doesn't waste completed work -- already-scored question ids are
    skipped on resume.
    """
    if output_path is None:
        output_path = f"data/eval/{strategy}.jsonl"
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    completed = _load_completed(path)
    results = list(completed.values())

    with open(path, "a") as f:
        for question in questions:
            if question.id in completed:
                continue
            result = run_question(question, strategy, dense_index, sparse_index, embedder, reranker, client, config)
            f.write(json.dumps(asdict(result)) + "\n")
            f.flush()
            results.append(result)

    return results


def _mean(values: list[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return sum(present) / len(present) if present else None


def _summarize(results: list[QuestionResult]) -> dict:
    return {
        "n": len(results),
        "correctness": _mean([1.0 if r.correctness else 0.0 for r in results]),
        "faithfulness": _mean([r.faithfulness for r in results]),
        "retrieval_relevance": _mean([r.retrieval_relevance for r in results]),
        "citation_accuracy": _mean([r.citation_accuracy for r in results]),
        "fallback_rate": _mean([1.0 if r.fallback_triggered else 0.0 for r in results]),
    }


def aggregate(results: list[QuestionResult]) -> dict:
    """Overall + per-category means for each metric. A metric that's None
    (not applicable) on a given question is excluded from that mean, not
    treated as zero.
    """
    categories = sorted({r.category for r in results})
    return {
        "overall": _summarize(results),
        "by_category": {
            category: _summarize([r for r in results if r.category == category])
            for category in categories
        },
    }
