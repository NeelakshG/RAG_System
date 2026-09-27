import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from api.observability import configure_logging, end_request, log_fields, logger, start_request
from api.schemas import (
    AskRequest,
    AskResponse,
    DocumentOut,
    IngestResponse,
    ReadinessOut,
    UploadResponse,
)
from api.security import require_api_key, safe_filename
from api.service import IngestInProgressError, RAGService
from config import Config
from ingest import SUPPORTED_EXTENSIONS
from src.llm import LLMRateLimitedError, LLMUnavailableError


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    app.state.service = RAGService(Config())  # every setting comes from RAG_* env vars / .env
    yield


app = FastAPI(title="RAG Pipeline API", lifespan=lifespan)


@app.middleware("http")
async def request_logging(request: Request, call_next):
    """One JSON log line per request, tagged with a request ID that's echoed
    back in X-Request-ID (taken from the caller's header when present, so
    the dashboard and API logs can be correlated)."""
    request_id = request.headers.get("X-Request-ID", "")[:64] or uuid.uuid4().hex
    fields, token = start_request()
    start = time.perf_counter()
    status = 500  # stays 500 if the handler raises something unhandled
    try:
        response = await call_next(request)
        status = response.status_code
        response.headers["X-Request-ID"] = request_id
        return response
    finally:
        logger.info(
            "request",
            extra={
                "fields": {
                    "request_id": request_id,
                    "method": request.method,
                    "route": request.url.path,
                    "status": status,
                    "total_ms": round((time.perf_counter() - start) * 1000, 1),
                    **fields,
                }
            },
        )
        end_request(token)


@app.exception_handler(LLMUnavailableError)
async def llm_unavailable(request: Request, exc: LLMUnavailableError) -> JSONResponse:
    log_fields(error=str(exc))
    return JSONResponse(status_code=503, content={"detail": "LLM unavailable"})


@app.exception_handler(LLMRateLimitedError)
async def llm_rate_limited(request: Request, exc: LLMRateLimitedError) -> JSONResponse:
    log_fields(error=str(exc))
    return JSONResponse(status_code=429, content={"detail": "LLM usage limit reached; try again shortly"})


@app.exception_handler(IngestInProgressError)
async def ingest_in_progress(request: Request, exc: IngestInProgressError) -> JSONResponse:
    return JSONResponse(status_code=409, content={"detail": str(exc)})


@app.get("/healthz")
def healthz() -> dict:
    """Liveness: the process is up and serving HTTP. Deliberately checks
    nothing else -- a down Ollama shouldn't get the API container killed."""
    return {"status": "ok"}


@app.get("/readyz", response_model=ReadinessOut)
def readyz() -> JSONResponse:
    """Readiness: Ollama reachable with the model pulled AND a non-empty index."""
    readiness = app.state.service.readiness()
    return JSONResponse(status_code=200 if readiness["ready"] else 503, content=readiness)


@app.post("/v1/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    timings: dict = {}
    try:
        result = app.state.service.ask(
            request.question,
            use_hybrid=request.use_hybrid,
            source_names=request.source_names,
            timings=timings,
        )
    finally:
        log_fields(**timings)
    log_fields(
        confidence=result["confidence"]["composite"],
        fallback_triggered=result["fallback_triggered"],
        chunks=len(result["chunks"]),
    )
    return AskResponse(**result)


@app.post("/v1/ingest", response_model=IngestResponse, dependencies=[Depends(require_api_key)])
def ingest() -> IngestResponse:
    """Re-index the server's configured corpus directory (RAG_CORPUS_DIR)."""
    stats = app.state.service.ingest()
    log_fields(**stats)
    return IngestResponse(**stats)


@app.post("/v1/documents", response_model=UploadResponse, dependencies=[Depends(require_api_key)])
def upload_documents(files: list[UploadFile] = File(...)) -> UploadResponse:
    """Upload files into the corpus and re-index. Every file is validated
    before any is written, so a bad file in a batch writes nothing."""
    max_bytes = app.state.service.config.max_upload_mb * 1024 * 1024
    accepted: list[tuple[str, bytes]] = []

    for upload in files:
        name = safe_filename(upload.filename)
        if name is None:
            raise HTTPException(status_code=400, detail=f"invalid filename: {upload.filename!r}")
        path = Path(name)
        extension = path.suffix.lstrip(".").lower()
        if extension not in SUPPORTED_EXTENSIONS:
            raise HTTPException(
                status_code=415,
                detail=f"{name}: unsupported type; allowed: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
            )
        content = upload.file.read(max_bytes + 1)
        if len(content) > max_bytes:
            raise HTTPException(
                status_code=413, detail=f"{name}: larger than {app.state.service.config.max_upload_mb} MB"
            )
        # Lowercase the extension: discover_files() matches extensions case-sensitively.
        accepted.append((f"{path.stem}.{extension}", content))

    stats = app.state.service.add_documents(accepted)
    uploaded = [name for name, _ in accepted]
    log_fields(**stats, uploaded=uploaded)
    return UploadResponse(**stats, uploaded=uploaded)


@app.get("/v1/documents", response_model=list[DocumentOut])
def documents() -> list[DocumentOut]:
    return [DocumentOut(**d) for d in app.state.service.list_documents()]
