import threading
from contextlib import contextmanager
from pathlib import Path

from config import Config
from ingest import SentenceTransformerEmbedder, ingest_and_publish
from src.confidence import citation_coverage, completeness, compute_confidence, retrieval_confidence
from src.indexer import DenseIndex, SparseIndex
from src.llm import NO_ANSWER_SENTINEL, make_llm_client, verify_citations
from src.retriever import CrossEncoderReranker, retrieve, stage_timer


class IngestInProgressError(RuntimeError):
    """Another ingest is already rebuilding the index. The API maps this to a 409."""


class RAGService:
    """Owns the expensive pipeline objects -- embedder, reranker, LLM client,
    both indexes -- built once and reused across requests. One instance
    serves one chunking strategy per process (see config.py: strategy drives
    the index paths, so switching strategies means a new instance).

    Concurrency: `indexes` is a (dense, sparse) tuple that ask() reads ONCE
    per request and ingest() replaces in a single assignment, so a query can
    never mix a dense index from one build with a sparse index from another.
    Builds go to a fresh version directory (never the live one) and at most
    one runs at a time.
    """

    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        self.embedder = SentenceTransformerEmbedder(self.config.embedding_model)
        self.reranker = CrossEncoderReranker(self.config.reranker_model)
        self.client = make_llm_client(self.config)  # Ollama locally, Groq for the hosted demo
        self._ingest_lock = threading.Lock()
        self.indexes = self._load_indexes()

    def _load_indexes(self) -> tuple[DenseIndex, SparseIndex]:
        """Load both indexes for the live version of this strategy. On a
        fresh clone, before anything has been ingested, the BM25 pickle
        won't exist yet -- start with an empty SparseIndex rather than
        crashing, so the API can come up and POST /v1/ingest is how you
        populate it.
        """
        dense_index = DenseIndex(
            persist_dir=self.config.chroma_persist_dir, collection_name=self.config.collection_name
        )
        try:
            sparse_index = SparseIndex.load(self.config.bm25_path)
        except FileNotFoundError:
            sparse_index = SparseIndex()
        return dense_index, sparse_index

    def ask(
        self,
        question: str,
        use_hybrid: bool = True,
        source_names: list[str] | None = None,
        timings: dict | None = None,
    ) -> dict:
        """timings, when given, is filled with per-stage ms (dense, sparse,
        fuse, rerank, generate, verify) for request logging."""
        dense_index, sparse_index = self.indexes  # one consistent snapshot for this whole request

        retrieval_results = retrieve(
            question,
            dense_index,
            sparse_index,
            self.embedder,
            self.reranker,
            self.config,
            use_hybrid=use_hybrid,
            source_names=source_names,
            timings=timings,
        )
        chunk_ids = [chunk_id for chunk_id, _ in retrieval_results]
        chunks = dense_index.get_chunks(chunk_ids)
        scores = dict(retrieval_results)

        answer = NO_ANSWER_SENTINEL
        checks = []
        fallback_triggered = False

        if chunks:
            with stage_timer(timings, "generate_ms"):
                answer = self.client.answer(question, chunks)
            if answer != NO_ANSWER_SENTINEL:
                with stage_timer(timings, "verify_ms"):
                    checks = verify_citations(answer, chunks, self.client)
                composite = compute_confidence(answer, retrieval_results, checks)
                if composite < self.config.confidence_threshold:
                    fallback_triggered = True
                    answer = NO_ANSWER_SENTINEL

        has_answer = answer != NO_ANSWER_SENTINEL
        breakdown = {
            "retrieval": retrieval_confidence(retrieval_results),
            "citation": citation_coverage(checks) if has_answer else 0.0,
            "completeness": completeness(answer) if has_answer else 0.0,
            "composite": compute_confidence(answer, retrieval_results, checks) if has_answer else 0.0,
        }

        return {
            "answer": answer,
            "fallback_triggered": fallback_triggered,
            "confidence": breakdown,
            "chunks": [
                {
                    "chunk_id": c.chunk_id,
                    "source_name": c.source_name,
                    "section_heading": c.section_heading,
                    "text": c.text,
                    "score": scores.get(c.chunk_id, 0.0),
                }
                for c in chunks
            ],
            "citations": [
                {
                    "claim": check.claim,
                    "citation_number": check.citation_number,
                    "chunk_id": check.chunk.chunk_id if check.chunk else None,
                    "supported": check.supported,
                }
                for check in checks
            ],
        }

    @contextmanager
    def _exclusive_ingest(self):
        if not self._ingest_lock.acquire(blocking=False):
            raise IngestInProgressError("an ingest is already running; retry when it finishes")
        try:
            yield
        finally:
            self._ingest_lock.release()

    def _rebuild(self) -> dict:
        stats = ingest_and_publish(Path(self.config.corpus_dir), self.config, self.embedder)
        self.indexes = self._load_indexes()  # single assignment = atomic swap for readers
        return stats

    def ingest(self) -> dict:
        """Re-index everything under config.corpus_dir. The path is fixed by
        config -- never taken from a request -- so callers can't point the
        server at arbitrary folders."""
        with self._exclusive_ingest():
            return self._rebuild()

    def add_documents(self, files: list[tuple[str, bytes]]) -> dict:
        """Write already-validated (safe_name, content) pairs into the corpus
        dir, then re-index. Held under the ingest lock so a concurrent
        upload can't land files mid-build."""
        with self._exclusive_ingest():
            corpus_dir = Path(self.config.corpus_dir)
            corpus_dir.mkdir(parents=True, exist_ok=True)
            for name, content in files:
                (corpus_dir / name).write_bytes(content)
            return self._rebuild()

    def readiness(self) -> dict:
        """Can this instance actually answer questions right now?"""
        llm_ok, llm_status = self.client.readiness()
        dense_index, _ = self.indexes
        chunk_count = dense_index.count()
        index_status = f"{chunk_count} chunks" if chunk_count else "empty -- POST /v1/ingest"
        return {
            "ready": llm_ok and chunk_count > 0,
            "checks": {"llm": llm_status, "index": index_status},
        }

    def list_documents(self) -> list[dict]:
        dense_index, _ = self.indexes
        return dense_index.list_documents()
