from pydantic import BaseModel, Field, field_validator

# Hard caps on request size. A question longer than this is almost certainly
# abuse (it gets embedded, reranked against 20 chunks, and sent to the LLM
# several times), not a real question.
MAX_QUESTION_CHARS = 1000
MAX_SOURCE_NAMES = 100


class AskRequest(BaseModel):
    question: str = Field(max_length=MAX_QUESTION_CHARS)
    use_hybrid: bool = True
    source_names: list[str] | None = Field(default=None, max_length=MAX_SOURCE_NAMES)

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("question must not be blank")
        return stripped


class ChunkOut(BaseModel):
    chunk_id: str
    source_name: str
    section_heading: str | None
    text: str
    score: float


class CitationOut(BaseModel):
    claim: str
    citation_number: int
    chunk_id: str | None
    supported: bool


class ConfidenceBreakdownOut(BaseModel):
    retrieval: float
    citation: float
    completeness: float
    composite: float


class AskResponse(BaseModel):
    answer: str
    fallback_triggered: bool
    confidence: ConfidenceBreakdownOut
    chunks: list[ChunkOut]
    citations: list[CitationOut]


class IngestResponse(BaseModel):
    indexed: int
    deduped: int


class UploadResponse(IngestResponse):
    uploaded: list[str]


class ReadinessOut(BaseModel):
    ready: bool
    checks: dict[str, str]


class DocumentOut(BaseModel):
    source_name: str
    chunk_count: int
    total_tokens: int
