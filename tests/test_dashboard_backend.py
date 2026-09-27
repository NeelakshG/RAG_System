import json

import pytest

from dashboard.backend import BackendError, EmbeddedBackend, cache_key, load_seed_answers
from src.llm import LLMRateLimitedError, LLMUnavailableError


class StubService:
    def __init__(self, raises=None):
        self.calls = []
        self.raises = raises

    def ask(self, question, use_hybrid=True, source_names=None):
        self.calls.append(question)
        if self.raises:
            raise self.raises
        return {"answer": f"answer to {question}"}

    def list_documents(self):
        return [{"source_name": "a.md", "chunk_count": 1, "total_tokens": 7}]


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_cache_key_ignores_case_and_whitespace():
    assert cache_key("  What does  ERR_2043 mean? ", True, None) == cache_key("what does err_2043 mean?", True, None)
    assert cache_key("q", True, None) != cache_key("q", False, None)
    assert cache_key("q", True, ["b.md", "a.md"]) == cache_key("q", True, ["a.md", "b.md"])


def test_repeat_questions_are_served_from_cache():
    service = StubService()
    backend = EmbeddedBackend(service)

    first = backend.ask("What is X?", True, None)
    second = backend.ask("what is x?", True, None)

    assert first == second
    assert service.calls == ["What is X?"]
    assert backend.is_cached("WHAT IS X?", True, None)


def test_global_rate_limit_blocks_new_questions_but_not_cached_ones():
    clock = FakeClock()
    service = StubService()
    backend = EmbeddedBackend(service, max_new_questions_per_minute=2, clock=clock)

    backend.ask("q1", True, None)
    backend.ask("q2", True, None)
    with pytest.raises(BackendError, match="try again in a minute"):
        backend.ask("q3", True, None)
    assert backend.ask("q1", True, None) == {"answer": "answer to q1"}  # cached: still works

    clock.now = 60.0  # window slides
    backend.ask("q3", True, None)
    assert service.calls == ["q1", "q2", "q3"]


def test_rate_limited_llm_gives_friendly_message_and_is_not_cached():
    service = StubService(raises=LLMRateLimitedError("429"))
    backend = EmbeddedBackend(service)

    with pytest.raises(BackendError, match="free usage limit"):
        backend.ask("q", True, None)
    assert not backend.is_cached("q", True, None)


def test_unavailable_llm_gives_friendly_message():
    backend = EmbeddedBackend(StubService(raises=LLMUnavailableError("down")))

    with pytest.raises(BackendError, match="unavailable"):
        backend.ask("q", True, None)


def test_uploads_are_disabled():
    backend = EmbeddedBackend(StubService())

    assert backend.uploads_enabled is False
    with pytest.raises(BackendError):
        backend.upload([("a.md", b"x")])


def test_cache_is_bounded():
    backend = EmbeddedBackend(StubService(), max_new_questions_per_minute=10_000)
    backend.MAX_CACHED_ANSWERS = 3

    for i in range(5):
        backend.ask(f"q{i}", True, None)

    assert not backend.is_cached("q0", True, None)
    assert backend.is_cached("q4", True, None)


def test_seed_answers_answer_examples_without_calling_the_llm(tmp_path):
    path = tmp_path / "demo_answers.json"
    path.write_text(json.dumps([{"question": "What is X?", "use_hybrid": True, "result": {"answer": "recorded"}}]))
    service = StubService(raises=LLMRateLimitedError("quota spent"))
    backend = EmbeddedBackend(service, seed_answers=load_seed_answers(path))

    assert backend.ask("what is x?", True, None) == {"answer": "recorded"}
    assert service.calls == []


def test_missing_seed_file_is_fine(tmp_path):
    assert load_seed_answers(tmp_path / "nope.json") == {}
