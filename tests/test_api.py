import json
import logging
import threading

import pytest
from fastapi.testclient import TestClient

import api.service
from api.main import app
from api.observability import JsonFormatter, logger as api_logger
from api.security import safe_filename
from api.service import IngestInProgressError, RAGService
from config import Config
from src.llm import NO_ANSWER_SENTINEL, LLMUnavailableError
from src.models import Chunk

API_KEY = "test-key"


def _chunk_fields(chunk_id: str, text: str) -> dict:
    return dict(
        chunk_id=chunk_id, doc_id=chunk_id.split("::")[0], source_name=chunk_id.split("::")[0],
        chunk_index=0, text=text, section_heading=None, chunking_strategy="fixed",
        char_count=len(text), token_count=7, start_char=0, end_char=len(text),
    )


class StubDenseIndex:
    def __init__(self, chunks: dict | None = None):
        if chunks is None:
            chunks = {"a.md::0": _chunk_fields("a.md::0", "ERR_2043 means the upload was dropped.")}
        self._chunks = chunks

    def query(self, query_embedding, k=10, source_names=None):
        return list(self._chunks)[:k]

    def get_texts(self, chunk_ids):
        return {cid: self._chunks[cid]["text"] for cid in chunk_ids}

    def get_chunks(self, chunk_ids):
        return [Chunk(**self._chunks[cid]) for cid in chunk_ids if cid in self._chunks]

    def count(self):
        return len(self._chunks)

    def list_documents(self):
        return [{"source_name": c["source_name"], "chunk_count": 1, "total_tokens": 7} for c in self._chunks.values()]


class StubSparseIndex:
    def query(self, query_tokens, k=10, source_names=None):
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

    def __init__(self, answer_text="ERR_2043 means the upload was dropped. [1]", verify_response="YES",
                 ready=(True, "ok"), raises=None):
        self.answer_text = answer_text
        self.verify_response = verify_response
        self.ready = ready
        self.raises = raises

    def answer(self, query, chunks):
        if self.raises:
            raise self.raises
        return self.answer_text

    def generate(self, prompt):
        return self.verify_response

    def readiness(self):
        return self.ready


def _make_service(tmp_path=None, api_key=None, **client_kwargs) -> RAGService:
    service = RAGService.__new__(RAGService)  # skip real model/index loading
    corpus_dir = str(tmp_path / "corpus") if tmp_path else "data/corpus"
    # _env_file=None: a developer's local .env (e.g. with RAG_API_KEY) must not leak into tests.
    service.config = Config(_env_file=None, api_key=api_key, corpus_dir=corpus_dir, max_upload_mb=1)
    service.embedder = StubEmbedder()
    service.reranker = StubReranker()
    service.indexes = (StubDenseIndex(), StubSparseIndex())
    service.client = StubClient(**client_kwargs)
    service._ingest_lock = threading.Lock()
    return service


def _set_stub_state(answer_text=None, verify_response="YES", **kwargs) -> RAGService:
    client_kwargs = {"verify_response": verify_response}
    if answer_text is not None:
        client_kwargs["answer_text"] = answer_text
    service = _make_service(**kwargs, **client_kwargs)
    app.state.service = service
    return service


def _stub_rebuild(service, stats=None):
    calls = []

    def rebuild():
        calls.append(True)
        return stats or {"indexed": 3, "deduped": 1}

    service._rebuild = rebuild
    return calls


# --- /v1/ask ---

def test_ask_returns_answer_with_citations():
    _set_stub_state()
    client = TestClient(app)

    response = client.post("/v1/ask", json={"question": "what does ERR_2043 mean?"})

    assert response.status_code == 200
    body = response.json()
    assert [(c["citation_number"], c["chunk_id"], c["supported"]) for c in body["citations"]] == [
        (1, "a.md::0", True)
    ]
    assert [c["chunk_id"] for c in body["chunks"]] == ["a.md::0"]
    assert body["fallback_triggered"] is False
    assert "[1]" in body["answer"]


def test_ask_blank_question_is_rejected():
    _set_stub_state()
    client = TestClient(app)

    response = client.post("/v1/ask", json={"question": "   "})

    assert response.status_code == 422


def test_ask_overlong_question_is_rejected():
    _set_stub_state()
    client = TestClient(app)

    response = client.post("/v1/ask", json={"question": "x" * 1001})

    assert response.status_code == 422


def test_ask_no_answer_short_circuits():
    _set_stub_state(answer_text=NO_ANSWER_SENTINEL)
    client = TestClient(app)

    response = client.post("/v1/ask", json={"question": "what does ERR_9999 mean?"})

    body = response.json()
    assert body["answer"] == NO_ANSWER_SENTINEL
    assert body["fallback_triggered"] is False
    assert body["confidence"]["composite"] == 0.0


def test_ask_low_confidence_falls_back():
    _set_stub_state(verify_response="NO")
    client = TestClient(app)

    response = client.post("/v1/ask", json={"question": "what does ERR_2043 mean?"})

    body = response.json()
    assert body["answer"] == NO_ANSWER_SENTINEL
    assert body["fallback_triggered"] is True
    assert body["confidence"]["composite"] == 0.0


def test_ask_llm_down_returns_503_not_500():
    _set_stub_state(raises=LLMUnavailableError("connection refused"))
    client = TestClient(app)

    response = client.post("/v1/ask", json={"question": "what does ERR_2043 mean?"})

    assert response.status_code == 503
    assert response.json() == {"detail": "LLM unavailable"}


# --- auth ---

def test_write_endpoints_require_api_key(tmp_path):
    service = _set_stub_state(tmp_path=tmp_path, api_key=API_KEY)
    calls = _stub_rebuild(service)
    client = TestClient(app)

    assert client.post("/v1/ingest").status_code == 401
    assert client.post("/v1/ingest", headers={"X-API-Key": "wrong"}).status_code == 401
    upload = client.post("/v1/documents", files=[("files", ("a.md", b"# hi"))])
    assert upload.status_code == 401
    assert calls == []
    assert not (tmp_path / "corpus").exists()

    ok = client.post("/v1/ingest", headers={"X-API-Key": API_KEY})
    assert ok.status_code == 200
    assert ok.json() == {"indexed": 3, "deduped": 1}


def test_read_endpoints_stay_open_with_api_key_set():
    _set_stub_state(api_key=API_KEY)
    client = TestClient(app)

    assert client.post("/v1/ask", json={"question": "what does ERR_2043 mean?"}).status_code == 200
    assert client.get("/v1/documents").status_code == 200


def test_ingest_ignores_any_client_supplied_path(tmp_path):
    service = _set_stub_state(tmp_path=tmp_path)
    _stub_rebuild(service)
    client = TestClient(app)

    response = client.post("/v1/ingest", json={"corpus_dir": "C:/Windows"})

    assert response.status_code == 200
    assert service.config.corpus_dir == str(tmp_path / "corpus")


# --- uploads ---

def test_upload_writes_into_corpus_and_reindexes(tmp_path):
    service = _set_stub_state(tmp_path=tmp_path)
    calls = _stub_rebuild(service)
    client = TestClient(app)

    response = client.post(
        "/v1/documents", files=[("files", ("guide.md", b"# Guide")), ("files", ("NOTES.TXT", b"notes"))]
    )

    assert response.status_code == 200
    assert response.json() == {"indexed": 3, "deduped": 1, "uploaded": ["guide.md", "NOTES.txt"]}
    assert (tmp_path / "corpus" / "guide.md").read_bytes() == b"# Guide"
    assert (tmp_path / "corpus" / "NOTES.txt").read_bytes() == b"notes"
    assert calls == [True]


def test_upload_path_traversal_is_contained(tmp_path):
    service = _set_stub_state(tmp_path=tmp_path)
    _stub_rebuild(service)
    client = TestClient(app)

    response = client.post("/v1/documents", files=[("files", ("../../evil.md", b"x"))])

    assert response.status_code == 200
    assert response.json()["uploaded"] == ["evil.md"]
    assert (tmp_path / "corpus" / "evil.md").exists()
    assert not (tmp_path.parent / "evil.md").exists()


def test_upload_rejects_unsupported_type(tmp_path):
    service = _set_stub_state(tmp_path=tmp_path)
    _stub_rebuild(service)
    client = TestClient(app)

    response = client.post("/v1/documents", files=[("files", ("data.csv", b"a,b"))])

    assert response.status_code == 415


def test_upload_rejects_oversized_file(tmp_path):
    service = _set_stub_state(tmp_path=tmp_path)  # max_upload_mb=1
    _stub_rebuild(service)
    client = TestClient(app)

    response = client.post("/v1/documents", files=[("files", ("big.txt", b"x" * (1024 * 1024 + 1)))])

    assert response.status_code == 413


def test_upload_with_one_bad_file_writes_nothing(tmp_path):
    service = _set_stub_state(tmp_path=tmp_path)
    calls = _stub_rebuild(service)
    client = TestClient(app)

    response = client.post(
        "/v1/documents", files=[("files", ("good.md", b"# ok")), ("files", ("bad.exe", b"MZ"))]
    )

    assert response.status_code == 415
    assert not (tmp_path / "corpus").exists()
    assert calls == []


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("notes.md", "notes.md"),
        ("../../etc/passwd.md", "passwd.md"),
        ("..\\..\\evil.md", "evil.md"),
        ("C:evil.md", "evil.md"),
        ("/abs/path/x.txt", "x.txt"),
        ("..", None),
        (".env", None),
        ("", None),
        (None, None),
        ("bad\x00name.md", None),
    ],
)
def test_safe_filename(raw, expected):
    assert safe_filename(raw) == expected


# --- concurrency ---

def test_concurrent_ingest_returns_409(tmp_path):
    service = _set_stub_state(tmp_path=tmp_path)
    _stub_rebuild(service)
    client = TestClient(app)

    service._ingest_lock.acquire()  # simulate an ingest already running
    try:
        response = client.post("/v1/ingest")
    finally:
        service._ingest_lock.release()

    assert response.status_code == 409


def test_ask_keeps_its_index_snapshot_across_a_reindex(monkeypatch, tmp_path):
    """An ingest that finishes mid-ask must not change which index that ask reads."""
    service = _make_service(tmp_path=tmp_path)
    old_dense = StubDenseIndex({"old.md::0": _chunk_fields("old.md::0", "old text")})
    new_dense = StubDenseIndex({"new.md::0": _chunk_fields("new.md::0", "new text")})
    service.indexes = (old_dense, StubSparseIndex())

    in_rerank, release = threading.Event(), threading.Event()
    original_get_texts = old_dense.get_texts

    def blocking_get_texts(chunk_ids):
        in_rerank.set()
        release.wait(timeout=5)
        return original_get_texts(chunk_ids)

    old_dense.get_texts = blocking_get_texts
    monkeypatch.setattr(api.service, "ingest_and_publish", lambda *a: {"indexed": 1, "deduped": 0})
    service._load_indexes = lambda: (new_dense, StubSparseIndex())

    results = {}
    worker = threading.Thread(target=lambda: results.update(service.ask("q")))
    worker.start()
    assert in_rerank.wait(timeout=5)
    service.ingest()  # swaps to the new index while the ask is mid-flight
    release.set()
    worker.join(timeout=5)

    assert [c["chunk_id"] for c in results["chunks"]] == ["old.md::0"]
    assert service.indexes[0] is new_dense  # later requests see the new index


# --- health ---

def test_healthz_always_ok():
    _set_stub_state(ready=(False, "Ollama unreachable"))
    client = TestClient(app)

    assert client.get("/healthz").json() == {"status": "ok"}


def test_readyz_ready():
    _set_stub_state()
    client = TestClient(app)

    response = client.get("/readyz")

    assert response.status_code == 200
    assert response.json() == {"ready": True, "checks": {"llm": "ok", "index": "1 chunks"}}


def test_readyz_not_ready_when_llm_down():
    _set_stub_state(ready=(False, "Ollama unreachable"))
    client = TestClient(app)

    response = client.get("/readyz")

    assert response.status_code == 503
    assert response.json()["checks"]["llm"] == "Ollama unreachable"


def test_readyz_not_ready_when_index_empty():
    service = _set_stub_state()
    service.indexes = (StubDenseIndex({}), StubSparseIndex())
    client = TestClient(app)

    response = client.get("/readyz")

    assert response.status_code == 503
    assert response.json()["checks"]["index"].startswith("empty")


# --- request logging ---

class _Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines: list[str] = []
        self.setFormatter(JsonFormatter())

    def emit(self, record):
        self.lines.append(self.format(record))


@pytest.fixture
def captured_logs():
    handler = _Capture()
    previous_level = api_logger.level
    api_logger.addHandler(handler)
    api_logger.setLevel(logging.INFO)
    yield handler.lines
    api_logger.removeHandler(handler)
    api_logger.setLevel(previous_level)


def test_ask_logs_one_json_line_with_stage_timings(captured_logs):
    _set_stub_state()
    client = TestClient(app)

    response = client.post(
        "/v1/ask", json={"question": "what does ERR_2043 mean?"}, headers={"X-Request-ID": "req-123"}
    )

    assert response.headers["X-Request-ID"] == "req-123"
    assert len(captured_logs) == 1
    line = json.loads(captured_logs[0])
    assert line["request_id"] == "req-123"
    assert line["route"] == "/v1/ask"
    assert line["status"] == 200
    for field in ["total_ms", "dense_ms", "sparse_ms", "fuse_ms", "rerank_ms", "generate_ms", "verify_ms",
                  "confidence", "fallback_triggered"]:
        assert field in line, field


def test_request_id_generated_when_absent(captured_logs):
    _set_stub_state()
    client = TestClient(app)

    response = client.get("/healthz")

    request_id = response.headers["X-Request-ID"]
    assert len(request_id) == 32
    assert json.loads(captured_logs[0])["request_id"] == request_id


def test_ask_llm_rate_limited_returns_429():
    from src.llm import LLMRateLimitedError

    _set_stub_state(raises=LLMRateLimitedError("quota"))
    client = TestClient(app)

    response = client.post("/v1/ask", json={"question": "what does ERR_2043 mean?"})

    assert response.status_code == 429
