from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from pydantic import BaseModel, field_validator

from config import Config
from ingest import SentenceTransformerEmbedder
from src.confidence import answer_with_confidence
from src.indexer import DenseIndex, SparseIndex
from src.llm import OllamaClient
from src.retriever import CrossEncoderReranker, retrieve


class AskRequest(BaseModel):
    query: str
    use_hybrid: bool = True

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("query must not be blank")
        return stripped


class Citation(BaseModel):
    number: int
    chunk_id: str
    source_name: str
    text: str


class AskResponse(BaseModel):
    answer: str
    confidence: float
    fallback_triggered: bool
    citations: list[Citation]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build every heavy, request-independent object once at startup and
    stash it on app.state -- embedding/reranker models load from disk, so
    constructing them per-request would reload both on every question.
    """
    config = Config()
    app.state.config = config
    app.state.embedder = SentenceTransformerEmbedder(config.embedding_model)
    app.state.reranker = CrossEncoderReranker(config.reranker_model)
    app.state.dense_index = DenseIndex(config.chroma_persist_dir, config.collection_name)
    app.state.sparse_index = SparseIndex.load(config.bm25_path)
    app.state.client = OllamaClient(model=config.llm_model, host=config.ollama_host)
    yield


app = FastAPI(lifespan=lifespan)


@app.post("/v1/ask", response_model=AskResponse)
def ask(request: AskRequest, req: Request) -> AskResponse:
    """Sync def, not async: every downstream call (embedding, reranking,
    the Ollama HTTP request) is blocking I/O, so FastAPI's threadpool is
    what actually gives this concurrency -- async def would just block
    the event loop on each call instead.
    """
    state = req.app.state

    results = retrieve(
        request.query,
        state.dense_index,
        state.sparse_index,
        state.embedder,
        state.reranker,
        state.config,
        use_hybrid=request.use_hybrid,
    )

    chunk_ids = [chunk_id for chunk_id, _ in results]
    chunks = state.dense_index.get_chunks(chunk_ids)

    result = answer_with_confidence(
        request.query,
        chunks,
        results,
        state.client,
        threshold=state.config.confidence_threshold,
    )

    citations = [
        Citation(number=i, chunk_id=chunk.chunk_id, source_name=chunk.source_name, text=chunk.text)
        for i, chunk in enumerate(chunks, start=1)
    ]

    return AskResponse(
        answer=result.text,
        confidence=result.confidence,
        fallback_triggered=result.fallback_triggered,
        citations=citations,
    )
