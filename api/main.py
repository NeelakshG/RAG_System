import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.schemas import (
    AskRequest,
    AskResponse,
    DocumentOut,
    IngestRequest,
    IngestResponse,
)
from api.service import RAGService
from config import Config


@asynccontextmanager
async def lifespan(app: FastAPI):
    strategy = os.environ.get("RAG_STRATEGY", "recursive")
    config = Config(strategy=strategy)
    if "OLLAMA_HOST" in os.environ:
        config.ollama_host = os.environ["OLLAMA_HOST"]
    app.state.service = RAGService(config)
    yield


app = FastAPI(title="RAG Pipeline API", lifespan=lifespan)


@app.post("/v1/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    result = app.state.service.ask(
        request.question, use_hybrid=request.use_hybrid, source_names=request.source_names
    )
    return AskResponse(**result)


@app.post("/v1/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest) -> IngestResponse:
    stats = app.state.service.ingest(request.corpus_dir)
    return IngestResponse(**stats)


@app.get("/v1/documents", response_model=list[DocumentOut])
def documents() -> list[DocumentOut]:
    return [DocumentOut(**d) for d in app.state.service.list_documents()]
