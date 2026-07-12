from dataclasses import dataclass


@dataclass
class Config:
    """Central, config-driven settings for the whole pipeline.

    `strategy` drives the dense/sparse index paths so the three chunking
    strategies (fixed/recursive/semantic) each get their own index and can
    never silently drift onto each other's data during the Phase 4 comparison.
    """

    strategy: str = "recursive"

    # models
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    llm_model: str = "llama3.1"
    ollama_host: str = "http://localhost:11434"

    # chunking
    chunk_size: int = 256
    chunk_overlap: int = 32
    max_tokens: int = 256
    semantic_percentile: float = 95

    # indexing
    dedup_threshold: float = 0.95

    # retrieval
    dense_k: int = 10
    sparse_k: int = 10
    rrf_k: int = 60
    fusion_candidates: int = 20
    final_k: int = 5

    # generation
    confidence_threshold: float = 0.5

    @property
    def chroma_persist_dir(self) -> str:
        return f"data/chroma_{self.strategy}"

    @property
    def collection_name(self) -> str:
        return f"chunks_{self.strategy}"

    @property
    def bm25_path(self) -> str:
        return f"data/bm25_{self.strategy}.pkl"
