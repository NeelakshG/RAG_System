from pathlib import Path

from config import Config
from ingest import SentenceTransformerEmbedder, chunk_corpus, load_corpus
from src.confidence import citation_coverage, completeness, compute_confidence, retrieval_confidence
from src.indexer import DenseIndex, SparseIndex, build_indexes
from src.llm import NO_ANSWER_SENTINEL, OllamaClient, verify_citations
from src.retriever import CrossEncoderReranker, retrieve


class RAGService:
    """Owns the expensive pipeline objects -- embedder, reranker, LLM client,
    both indexes -- built once and reused across requests. One instance
    serves one chunking strategy per process (see config.py: strategy drives
    the index paths, so switching strategies means a new instance).
    """

    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        self.embedder = SentenceTransformerEmbedder(self.config.embedding_model)
        self.reranker = CrossEncoderReranker(self.config.reranker_model)
        self.client = OllamaClient(model=self.config.llm_model, host=self.config.ollama_host)
        self._load_indexes()

    def _load_indexes(self) -> None:
        """Load both indexes for this strategy. On a fresh clone, before
        anything has been ingested, the BM25 pickle won't exist yet -- start
        with an empty SparseIndex rather than crashing, so the API can come
        up and POST /v1/ingest is how you populate it.
        """
        self.dense_index = DenseIndex(
            persist_dir=self.config.chroma_persist_dir, collection_name=self.config.collection_name
        )
        try:
            self.sparse_index = SparseIndex.load(self.config.bm25_path)
        except FileNotFoundError:
            self.sparse_index = SparseIndex()

    def ask(self, question: str, use_hybrid: bool = True) -> dict:
        retrieval_results = retrieve(
            question,
            self.dense_index,
            self.sparse_index,
            self.embedder,
            self.reranker,
            self.config,
            use_hybrid=use_hybrid,
        )
        chunk_ids = [chunk_id for chunk_id, _ in retrieval_results]
        chunks = self.dense_index.get_chunks(chunk_ids)
        scores = dict(retrieval_results)

        answer = NO_ANSWER_SENTINEL
        checks = []
        fallback_triggered = False

        if chunks:
            answer = self.client.answer(question, chunks)
            if answer != NO_ANSWER_SENTINEL:
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

    def ingest(self, corpus_dir: str) -> dict:
        docs = load_corpus(Path(corpus_dir))
        chunks = chunk_corpus(docs, self.config.strategy, self.config, self.embedder)
        stats = build_indexes(chunks, self.embedder, self.config)
        self._load_indexes()  # this process's indexes must reflect what was just built
        return stats

    def list_documents(self) -> list[dict]:
        return self.dense_index.list_documents()
