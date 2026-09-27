from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    """Central, config-driven settings for the whole pipeline.

    Every field can be overridden from the environment (or a `.env` file)
    with a `RAG_` prefix, e.g. `RAG_STRATEGY=semantic`, `RAG_FINAL_K=3`,
    `RAG_OLLAMA_HOST=http://ollama:11434`.

    `strategy` drives the index paths so the three chunking strategies
    (fixed/recursive/semantic) each get their own index and can never
    silently drift onto each other's data during the Phase 4 comparison.
    """

    model_config = SettingsConfigDict(env_prefix="RAG_", env_file=".env", extra="ignore")

    strategy: Literal["fixed", "recursive", "semantic"] = "recursive"

    # models
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    llm_provider: Literal["ollama", "groq"] = "ollama"  # groq = hosted, for the public demo
    llm_model: str = "llama3.1"
    ollama_host: str = "http://localhost:11434"
    ollama_timeout_s: float = 60.0  # applies to whichever provider is in use
    ollama_retries: int = 2
    groq_api_key: SecretStr | None = None
    groq_model: str = "openai/gpt-oss-20b"  # Groq retired free-tier Llama 3.x in Aug 2026

    # chunking
    chunk_size: int = 256
    chunk_overlap: int = 32
    max_tokens: int = 256
    semantic_percentile: float = 95

    # indexing
    dedup_threshold: float = 0.95
    index_versions_to_keep: int = 2

    # retrieval
    dense_k: int = 10
    sparse_k: int = 10
    rrf_k: int = 60
    fusion_candidates: int = 20
    final_k: int = 5

    # generation
    confidence_threshold: float = 0.5

    # storage
    data_dir: str = "data"
    corpus_dir: str = "data/corpus"

    # API
    api_key: SecretStr | None = None  # unset = no auth (local dev only)
    max_upload_mb: int = 10

    # public demo (dashboard running the pipeline in-process)
    demo_new_questions_per_minute: int = 2  # across ALL visitors; Groq free tier is ~8K tokens/min
    demo_questions_per_session: int = 5  # per visitor; cached answers don't count

    @property
    def index_versions_dir(self) -> Path:
        """Every build lands in its own subdirectory here; see src/indexer.py."""
        return Path(self.data_dir) / "indexes" / self.strategy

    @property
    def current_index_pointer(self) -> Path:
        """File holding the name of the live index version for this strategy."""
        return Path(self.data_dir) / f"CURRENT_{self.strategy}"

    def current_index_dir(self) -> Path | None:
        """The live index version, or None before the first versioned build."""
        try:
            name = self.current_index_pointer.read_text().strip()
        except FileNotFoundError:
            return None
        return self.index_versions_dir / name if name else None

    @property
    def chroma_persist_dir(self) -> str:
        current = self.current_index_dir()
        if current is not None:
            return str(current / "chroma")
        return str(Path(self.data_dir) / f"chroma_{self.strategy}")  # pre-versioning layout

    @property
    def collection_name(self) -> str:
        return f"chunks_{self.strategy}"

    @property
    def bm25_path(self) -> str:
        current = self.current_index_dir()
        if current is not None:
            return str(current / "bm25.pkl")
        return str(Path(self.data_dir) / f"bm25_{self.strategy}.pkl")  # pre-versioning layout
