"""Where the dashboard gets its answers from.

- ApiBackend: calls the FastAPI service over HTTP (local / Docker setup).
- EmbeddedBackend: runs the pipeline inside the Streamlit process, for
  Streamlit Community Cloud where there's no separate API. This is the
  public demo, so it carries the guards that keep a free LLM quota alive:
  a shared answer cache, a global rate limit, and no uploads.

Both expose the same ask / documents / upload methods and raise
BackendError with a message that's safe to show visitors.
"""

import json
import threading
import time
from collections import OrderedDict, deque
from pathlib import Path

import requests

from src.llm import LLMRateLimitedError, LLMUnavailableError

# Shown as one-click buttons in the public demo. Picked to show off each
# behaviour: exact-identifier lookup (hybrid search), multi-hop across two
# docs, and a question the corpus can't answer ("I don't know").
EXAMPLE_QUESTIONS = [
    "What does ERR_2043 indicate?",
    "What error and HTTP status code correspond to an oversized document?",
    "What is the default value of MAX_RETRY_COUNT?",
    "Who owns the document intake service?",
]

DEMO_ANSWERS_PATH = Path(__file__).resolve().parent / "demo_answers.json"


class BackendError(Exception):
    """Something went wrong; str(error) is safe to show to visitors."""


def cache_key(question: str, use_hybrid: bool, source_names: list[str] | None) -> str:
    """Case- and whitespace-insensitive, so trivial re-phrasings of the same
    question hit the cache instead of the LLM."""
    normalized = " ".join(question.lower().split())
    sources = ",".join(sorted(source_names or []))
    return f"{normalized}|{'hybrid' if use_hybrid else 'dense'}|{sources}"


class ApiBackend:
    uploads_enabled = True

    def __init__(self, base_url: str, api_key: str | None = None):
        self.base_url = base_url
        self._headers = {"X-API-Key": api_key} if api_key else {}

    @staticmethod
    def _error(error: requests.RequestException) -> BackendError:
        """Prefer the API's own `detail` message over requests' generic one."""
        response = getattr(error, "response", None)
        if response is not None:
            try:
                return BackendError(f"{response.status_code}: {response.json()['detail']}")
            except (ValueError, KeyError, TypeError):
                pass
        return BackendError(str(error))

    def is_cached(self, question: str, use_hybrid: bool, source_names: list[str] | None) -> bool:
        return False

    def ask(self, question: str, use_hybrid: bool, source_names: list[str] | None) -> dict:
        payload = {"question": question, "use_hybrid": use_hybrid}
        if source_names:
            payload["source_names"] = source_names
        try:
            response = requests.post(f"{self.base_url}/v1/ask", json=payload, timeout=180)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise self._error(e) from e

    def documents(self) -> list[dict]:
        try:
            response = requests.get(f"{self.base_url}/v1/documents", timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise self._error(e) from e

    def upload(self, files: list[tuple[str, bytes]]) -> dict:
        try:
            response = requests.post(
                f"{self.base_url}/v1/documents",
                files=[("files", (name, content)) for name, content in files],
                headers=self._headers,
                timeout=300,
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise self._error(e) from e


class EmbeddedBackend:
    """Runs RAGService in-process with public-demo guards.

    One instance is shared by every visitor (Streamlit caches it per
    process), so the cache and the rate limit are global -- that's the
    point: they protect the one free LLM quota everyone shares.
    """

    uploads_enabled = False
    MAX_CACHED_ANSWERS = 500

    def __init__(self, service, max_new_questions_per_minute: int = 4, seed_answers: dict | None = None,
                 clock=time.monotonic):
        self.service = service
        self.max_new_questions_per_minute = max_new_questions_per_minute
        self._clock = clock
        self._lock = threading.Lock()
        self._recent_llm_calls: deque[float] = deque()
        self._cache: OrderedDict[str, dict] = OrderedDict(seed_answers or {})

    def is_cached(self, question: str, use_hybrid: bool, source_names: list[str] | None) -> bool:
        with self._lock:
            return cache_key(question, use_hybrid, source_names) in self._cache

    def _reserve_llm_slot(self) -> None:
        now = self._clock()
        with self._lock:
            while self._recent_llm_calls and now - self._recent_llm_calls[0] >= 60:
                self._recent_llm_calls.popleft()
            if len(self._recent_llm_calls) >= self.max_new_questions_per_minute:
                raise BackendError(
                    "The demo is getting a lot of questions right now. Please try again in a minute, "
                    "or pick one of the example questions."
                )
            self._recent_llm_calls.append(now)

    def ask(self, question: str, use_hybrid: bool, source_names: list[str] | None) -> dict:
        key = cache_key(question, use_hybrid, source_names)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]

        self._reserve_llm_slot()
        try:
            result = self.service.ask(question, use_hybrid=use_hybrid, source_names=source_names)
        except LLMRateLimitedError as e:
            raise BackendError(
                "The demo has reached its free usage limit for now. Please try again in a few minutes, "
                "or pick one of the example questions."
            ) from e
        except LLMUnavailableError as e:
            raise BackendError("The language model is unavailable right now. Please try again shortly.") from e

        with self._lock:  # only successful answers are cached; errors retry next time
            self._cache[key] = result
            while len(self._cache) > self.MAX_CACHED_ANSWERS:
                self._cache.popitem(last=False)
        return result

    def documents(self) -> list[dict]:
        return self.service.list_documents()

    def upload(self, files: list[tuple[str, bytes]]) -> dict:
        raise BackendError("Uploads are turned off in the public demo.")


def load_seed_answers(path: Path = DEMO_ANSWERS_PATH) -> dict:
    """Pre-recorded answers (scripts/record_demo_answers.py), keyed like the
    cache, so the example questions work even when the LLM quota is spent."""
    try:
        records = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    return {cache_key(r["question"], r["use_hybrid"], None): r["result"] for r in records}


def build_embedded_service(repo_root: Path):
    """Create the RAGService for the hosted demo. The container's disk starts
    empty (data/ isn't in git), so on first boot generate the synthetic
    corpus and index it -- a few seconds for this corpus size."""
    from api.service import RAGService  # deferred: loads the embedding + reranker models
    from config import Config
    from scripts.make_corpus import make_corpus

    data_dir = repo_root / "data"
    config = Config(data_dir=str(data_dir), corpus_dir=str(data_dir / "corpus"))
    corpus_dir = Path(config.corpus_dir)
    if not corpus_dir.exists() or not any(corpus_dir.iterdir()):
        make_corpus(corpus_dir)

    service = RAGService(config)
    if service.indexes[0].count() == 0:
        service.ingest()
    return service
