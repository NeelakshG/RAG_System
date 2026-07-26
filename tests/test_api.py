from fastapi.testclient import TestClient

from api import app
from config import Config
from src.llm import NO_ANSWER_SENTINEL
from src.models import Chunk


class StubDenseIndex:
    _CHUNKS = {
        "a.md::0": dict(
            chunk_id="a.md::0", doc_id="a.md", source_name="a.md", chunk_index=0,
            text="ERR_2043 means the upload was dropped.", section_heading=None,
            chunking_strategy="fixed", char_count=39, token_count=7,
            start_char=0, end_char=39,
        ),
    }

    def query(self, query_embedding, k=10):
        return list(self._CHUNKS)[:k]

    def get_texts(self, chunk_ids):
        return {cid: self._CHUNKS[cid]["text"] for cid in chunk_ids}

    def get_chunks(self, chunk_ids):
        return [Chunk(**self._CHUNKS[cid]) for cid in chunk_ids if cid in self._CHUNKS]


class StubSparseIndex:
    def query(self, query_tokens, k=10):
        return []


class StubEmbedder:
    def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]


class StubReranker:
    def score(self, query, texts):
        return [1.0 for _ in texts]


class StubClient:
    """Fakes OllamaClient: .answer() returns a fixed response, .generate()
    (used by citation verification) returns a fixed YES/NO."""

    def __init__(self, answer_text="ERR_2043 means the upload was dropped. [1]", verify_response="YES"):
        self.answer_text = answer_text
        self.verify_response = verify_response

    def answer(self, query, chunks):
        return self.answer_text

    def generate(self, prompt):
        return self.verify_response


def _set_stub_state(answer_text=None, verify_response="YES"):
    app.state.config = Config()
    app.state.embedder = StubEmbedder()
    app.state.reranker = StubReranker()
    app.state.dense_index = StubDenseIndex()
    app.state.sparse_index = StubSparseIndex()
    kwargs = {"verify_response": verify_response}
    if answer_text is not None:
        kwargs["answer_text"] = answer_text
    app.state.client = StubClient(**kwargs)


def test_ask_returns_answer_with_citations():
    _set_stub_state()
    client = TestClient(app)

    response = client.post("/v1/ask", json={"query": "what does ERR_2043 mean?"})

    assert response.status_code == 200
    body = response.json()
    assert body["citations"] == [
        {
            "number": 1,
            "chunk_id": "a.md::0",
            "source_name": "a.md",
            "text": "ERR_2043 means the upload was dropped.",
        }
    ]
    assert body["fallback_triggered"] is False
    assert "[1]" in body["answer"]


def test_ask_blank_query_is_rejected():
    _set_stub_state()
    client = TestClient(app)

    response = client.post("/v1/ask", json={"query": "   "})

    assert response.status_code == 422


def test_ask_no_answer_short_circuits():
    _set_stub_state(answer_text=NO_ANSWER_SENTINEL)
    client = TestClient(app)

    response = client.post("/v1/ask", json={"query": "what does ERR_9999 mean?"})

    body = response.json()
    assert body["answer"] == NO_ANSWER_SENTINEL
    assert body["fallback_triggered"] is False
    assert body["confidence"] == 0.0


def test_ask_low_confidence_falls_back():
    _set_stub_state(verify_response="NO")
    client = TestClient(app)

    response = client.post("/v1/ask", json={"query": "what does ERR_2043 mean?"})

    body = response.json()
    assert body["answer"] == NO_ANSWER_SENTINEL
    assert body["fallback_triggered"] is True
    assert body["confidence"] == 0.0
