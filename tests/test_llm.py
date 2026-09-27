from src.llm import extract_claims


def test_extract_claims_trailing_citation_merges_into_sentence():
    claims = extract_claims("ERR_2043 means the upload was dropped. [1]")
    assert len(claims) == 1
    claim, citation_numbers = claims[0]
    assert claim == "ERR_2043 means the upload was dropped. [1]"
    assert citation_numbers == [1]


def test_extract_claims_inline_citation_stays_attached():
    claims = extract_claims("The default is 3 [1]. The endpoint is /v1/health [2].")
    assert len(claims) == 2
    assert claims[0] == ("The default is 3 [1].", [1])
    assert claims[1] == ("The endpoint is /v1/health [2].", [2])


def test_extract_claims_sentence_without_citation_is_dropped():
    claims = extract_claims("This has no citation. This one does [1].")
    assert len(claims) == 1
    assert claims[0] == ("This one does [1].", [1])


def test_extract_claims_no_citations_returns_empty():
    assert extract_claims("No citations here at all.") == []


# --- OllamaClient: retries + readiness ---

import pytest
import requests

import src.llm
from src.llm import LLMUnavailableError, OllamaClient


class _Response:
    def __init__(self, status_code=200, body=None):
        self.status_code = status_code
        self._body = body or {}

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}")


def _fake_post(outcomes):
    """requests.post stand-in that plays back outcomes in order: an
    exception instance is raised, anything else is returned."""
    calls = []

    def post(*args, **kwargs):
        outcome = outcomes[len(calls)]
        calls.append(kwargs)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    return post, calls


def test_generate_retries_transient_failures_then_succeeds(monkeypatch):
    post, calls = _fake_post([
        requests.ConnectionError("refused"),
        _Response(503),
        _Response(200, {"response": "hello"}),
    ])
    monkeypatch.setattr(src.llm.requests, "post", post)
    client = OllamaClient(retries=2, backoff_s=0, timeout=7)

    assert client.generate("hi") == "hello"
    assert len(calls) == 3
    assert all(c["timeout"] == 7 for c in calls)


def test_generate_gives_up_after_retries(monkeypatch):
    post, calls = _fake_post([requests.Timeout("slow")] * 3)
    monkeypatch.setattr(src.llm.requests, "post", post)
    client = OllamaClient(retries=2, backoff_s=0)

    with pytest.raises(LLMUnavailableError):
        client.generate("hi")
    assert len(calls) == 3


def test_generate_does_not_retry_client_errors(monkeypatch):
    post, calls = _fake_post([_Response(404, {"error": "model not found"})])
    monkeypatch.setattr(src.llm.requests, "post", post)
    client = OllamaClient(retries=2, backoff_s=0)

    with pytest.raises(LLMUnavailableError):
        client.generate("hi")
    assert len(calls) == 1


@pytest.mark.parametrize(
    "model, pulled, expected_ok",
    [
        ("llama3.1", ["llama3.1:latest"], True),
        ("llama3.1", ["llama3.1"], True),
        ("llama3.1:8b", ["llama3.1:8b"], True),
        ("llama3.1", ["llama3.2:3b"], False),
    ],
)
def test_readiness_checks_model_is_pulled(monkeypatch, model, pulled, expected_ok):
    body = {"models": [{"name": name} for name in pulled]}
    monkeypatch.setattr(src.llm.requests, "get", lambda *a, **kw: _Response(200, body))

    ok, _ = OllamaClient(model=model).readiness()

    assert ok is expected_ok


def test_readiness_reports_unreachable(monkeypatch):
    def get(*args, **kwargs):
        raise requests.ConnectionError("refused")

    monkeypatch.setattr(src.llm.requests, "get", get)

    ok, reason = OllamaClient().readiness()

    assert ok is False
    assert "unreachable" in reason


# --- GroqClient + provider selection ---

from config import Config
from src.llm import GroqClient, LLMRateLimitedError, make_llm_client


class _GroqResponse(_Response):
    def __init__(self, status_code=200, body=None, headers=None):
        super().__init__(status_code, body)
        self.headers = headers or {}


def _chat(content):
    return _GroqResponse(200, {"choices": [{"message": {"content": content}}]})


def test_groq_generate_sends_chat_request_with_bearer_key(monkeypatch):
    post, calls = _fake_post([_chat("hello")])
    monkeypatch.setattr(src.llm.requests, "post", post)

    assert GroqClient(api_key="k", model="m").generate("hi") == "hello"
    assert calls[0]["headers"] == {"Authorization": "Bearer k"}
    assert calls[0]["json"]["model"] == "m"
    assert calls[0]["json"]["messages"] == [{"role": "user", "content": "hi"}]


def test_groq_waits_out_short_retry_after(monkeypatch):
    post, calls = _fake_post([_GroqResponse(429, headers={"retry-after": "2"}), _chat("ok")])
    monkeypatch.setattr(src.llm.requests, "post", post)
    slept = []
    monkeypatch.setattr(src.llm.time, "sleep", slept.append)

    assert GroqClient(api_key="k").generate("hi") == "ok"
    assert slept == [2.0]


def test_groq_long_retry_after_fails_fast_as_rate_limited(monkeypatch):
    post, calls = _fake_post([_GroqResponse(429, headers={"retry-after": "3600"})])
    monkeypatch.setattr(src.llm.requests, "post", post)

    with pytest.raises(LLMRateLimitedError):
        GroqClient(api_key="k").generate("hi")
    assert len(calls) == 1


def test_groq_repeated_429s_end_as_rate_limited(monkeypatch):
    post, calls = _fake_post([_GroqResponse(429, headers={"retry-after": "0"})] * 3)
    monkeypatch.setattr(src.llm.requests, "post", post)

    with pytest.raises(LLMRateLimitedError):
        GroqClient(api_key="k", retries=2).generate("hi")
    assert len(calls) == 3


def test_groq_bad_key_is_not_retried(monkeypatch):
    post, calls = _fake_post([_GroqResponse(401)])
    monkeypatch.setattr(src.llm.requests, "post", post)

    with pytest.raises(LLMUnavailableError) as excinfo:
        GroqClient(api_key="secret-key").generate("hi")
    assert len(calls) == 1
    assert "secret-key" not in str(excinfo.value)


@pytest.mark.parametrize(
    "status, body, expected_ok",
    [
        (200, {"data": [{"id": "llama-3.1-8b-instant"}]}, True),
        (200, {"data": [{"id": "some-other-model"}]}, False),
        (401, {}, False),
    ],
)
def test_groq_readiness(monkeypatch, status, body, expected_ok):
    monkeypatch.setattr(src.llm.requests, "get", lambda *a, **kw: _GroqResponse(status, body))

    ok, _ = GroqClient(api_key="k").readiness()

    assert ok is expected_ok


def test_make_llm_client_defaults_to_ollama():
    assert isinstance(make_llm_client(Config(_env_file=None)), OllamaClient)


def test_make_llm_client_groq():
    client = make_llm_client(Config(_env_file=None, llm_provider="groq", groq_api_key="k", groq_model="m"))

    assert isinstance(client, GroqClient)
    assert client.model == "m"


def test_make_llm_client_groq_without_key_fails_clearly():
    with pytest.raises(ValueError, match="RAG_GROQ_API_KEY"):
        make_llm_client(Config(_env_file=None, llm_provider="groq"))


def test_extract_claims_bare_citation_is_not_a_claim():
    # Regression: llama3.1 once answered just "[1]", which was verified as a
    # "supported claim" and scored ~100% confidence.
    assert extract_claims("[1]") == []
    assert extract_claims("[1] [2]") == []
