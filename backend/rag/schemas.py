from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)


class Citation(BaseModel):
    chunk_id: str
    source_title: str
    page_start: int | None = None
    page_end: int | None = None
    source_url: str | None = None
    similarity_score: float


class RetrievedChunk(Citation):
    text: str


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    retrieved_chunks: list[RetrievedChunk]
    similarity_scores: list[float]
    refusal: bool
    best_score: float | None
    top_k: int
    similarity_threshold: float


class HealthResponse(BaseModel):
    status: str
    backend_running: bool
    index_loaded: bool
    total_chunks: int
    embedding_model: str
    index_path: str
    metadata_path: str
    top_k: int
    similarity_threshold: float
    sources_count: int


class SourceMetadata(BaseModel):
    title: str
    authors_or_organization: str
    year: str
    url: str
    local_filename: str
    license_or_access_note: str
    category: str
