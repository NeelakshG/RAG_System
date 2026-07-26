from pydantic import BaseModel


class AskRequest(BaseModel):
    question: str
    use_hybrid: bool = True
    source_names: list[str] | None = None


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


class IngestRequest(BaseModel):
    corpus_dir: str = "data/corpus"


class IngestResponse(BaseModel):
    indexed: int
    deduped: int


class DocumentOut(BaseModel):
    source_name: str
    chunk_count: int
    total_tokens: int
