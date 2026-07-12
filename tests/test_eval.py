import json

from src.eval import (
    GoldenQuestion,
    QuestionResult,
    aggregate,
    judge_correctness,
    judge_faithfulness,
    load_golden_questions,
    retrieval_relevance,
)
from src.llm import NO_ANSWER_SENTINEL
from src.models import Chunk


def test_load_golden_questions(tmp_path):
    data = [
        {
            "id": "lookup_001",
            "question": "What is the default value of MAX_RETRY_COUNT?",
            "category": "lookup",
            "expected_answer": "3",
            "expected_source_docs": ["config_guide.txt"],
        }
    ]
    path = tmp_path / "golden_qa.json"
    path.write_text(json.dumps(data))

    questions = load_golden_questions(str(path))

    assert len(questions) == 1
    assert questions[0].id == "lookup_001"
    assert questions[0].category == "lookup"
    assert questions[0].expected_source_docs == ["config_guide.txt"]


def _make_question(category="lookup", expected_source_docs=None) -> GoldenQuestion:
    return GoldenQuestion(
        id="q1",
        question="What is X?",
        category=category,
        expected_answer="X is 3.",
        expected_source_docs=expected_source_docs if expected_source_docs is not None else ["a.md"],
    )


def _make_chunk(source_name="a.md") -> Chunk:
    return Chunk(
        chunk_id=f"{source_name}::0",
        doc_id=source_name,
        source_name=source_name,
        chunk_index=0,
        text="some chunk text",
        section_heading=None,
        chunking_strategy="fixed",
        char_count=15,
        token_count=3,
        start_char=0,
        end_char=15,
    )


class StubClient:
    """Returns a fixed response and records how many times it was called."""

    def __init__(self, response="YES"):
        self.response = response
        self.calls = 0

    def generate(self, prompt: str) -> str:
        self.calls += 1
        return self.response


class ExplodingClient:
    """Fails the test if the LLM is ever called -- used to prove a code path
    is deterministic and doesn't need a judge call.
    """

    def generate(self, prompt: str) -> str:
        raise AssertionError("LLM should not be called for this case")


# --- retrieval_relevance ---

def test_retrieval_relevance_no_expected_docs_is_none():
    question = _make_question(expected_source_docs=[])
    assert retrieval_relevance(question, [_make_chunk()]) is None


def test_retrieval_relevance_full_match():
    question = _make_question(expected_source_docs=["a.md", "b.md"])
    chunks = [_make_chunk("a.md"), _make_chunk("b.md")]
    assert retrieval_relevance(question, chunks) == 1.0


def test_retrieval_relevance_partial_match():
    question = _make_question(expected_source_docs=["a.md", "b.md"])
    chunks = [_make_chunk("a.md"), _make_chunk("c.md")]
    assert retrieval_relevance(question, chunks) == 0.5


def test_retrieval_relevance_no_match():
    question = _make_question(expected_source_docs=["a.md"])
    chunks = [_make_chunk("c.md")]
    assert retrieval_relevance(question, chunks) == 0.0


# --- judge_faithfulness ---

def test_judge_faithfulness_no_answer_sentinel_is_none():
    assert judge_faithfulness(NO_ANSWER_SENTINEL, [_make_chunk()], ExplodingClient()) is None


def test_judge_faithfulness_all_supported():
    client = StubClient(response="YES")
    score = judge_faithfulness("X is 3. Y is 4.", [_make_chunk()], client)
    assert score == 1.0
    assert client.calls == 2


def test_judge_faithfulness_none_supported():
    client = StubClient(response="NO")
    score = judge_faithfulness("X is 3.", [_make_chunk()], client)
    assert score == 0.0


# --- judge_correctness ---

def test_judge_correctness_no_answer_correct_is_deterministic():
    question = _make_question(category="no-answer")
    assert judge_correctness(question, NO_ANSWER_SENTINEL, ExplodingClient()) is True


def test_judge_correctness_no_answer_incorrect_is_deterministic():
    question = _make_question(category="no-answer")
    assert judge_correctness(question, "X is 3.", ExplodingClient()) is False


def test_judge_correctness_ambiguous_fallback_is_deterministic():
    question = _make_question(category="ambiguous")
    assert judge_correctness(question, NO_ANSWER_SENTINEL, ExplodingClient()) is True


def test_judge_correctness_ambiguous_flags_ambiguity():
    question = _make_question(category="ambiguous")
    client = StubClient(response="YES")
    assert judge_correctness(question, "It depends on what you mean.", client) is True


def test_judge_correctness_ambiguous_picks_one_reading():
    question = _make_question(category="ambiguous")
    client = StubClient(response="NO")
    assert judge_correctness(question, "X is definitely 3.", client) is False


def test_judge_correctness_lookup_uses_judge():
    question = _make_question(category="lookup")
    assert judge_correctness(question, "X is 3.", StubClient(response="YES")) is True
    assert judge_correctness(question, "X is 5.", StubClient(response="NO")) is False


# --- aggregate ---

def _make_result(category, correctness, faithfulness=None, relevance=None, citation_accuracy=None, fallback=False):
    return QuestionResult(
        id="q",
        category=category,
        strategy="fixed",
        answer="some answer",
        fallback_triggered=fallback,
        confidence=0.8,
        retrieved_chunk_ids=["a::0"],
        retrieval_relevance=relevance,
        faithfulness=faithfulness,
        citation_accuracy=citation_accuracy,
        correctness=correctness,
    )


def test_aggregate_overall_and_by_category_means():
    results = [
        _make_result("lookup", correctness=True, faithfulness=1.0, relevance=1.0, citation_accuracy=1.0),
        _make_result("lookup", correctness=False, faithfulness=0.0, relevance=0.5, citation_accuracy=0.0),
        _make_result("no-answer", correctness=True, fallback=True),
    ]

    summary = aggregate(results)

    assert summary["overall"]["n"] == 3
    assert summary["overall"]["correctness"] == 2 / 3
    assert summary["by_category"]["lookup"]["correctness"] == 0.5
    assert summary["by_category"]["lookup"]["faithfulness"] == 0.5
    assert summary["by_category"]["no-answer"]["correctness"] == 1.0


def test_aggregate_excludes_none_from_mean_instead_of_zero_filling():
    results = [
        _make_result("no-answer", correctness=True, faithfulness=None, relevance=None, citation_accuracy=None),
    ]

    summary = aggregate(results)

    assert summary["overall"]["faithfulness"] is None
    assert summary["overall"]["retrieval_relevance"] is None
    assert summary["overall"]["citation_accuracy"] is None
