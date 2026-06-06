from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.rag.artifacts import (
    INDEX_MISSING_MESSAGE,
    ensure_index_artifacts,
    index_files_present,
)
from backend.rag.config import (
    EMBEDDING_MODEL,
    INDEX_DIR,
    METADATA_PATH,
    RAW_PDFS_DIR,
    REFUSAL_MESSAGE,
    SIMILARITY_THRESHOLD,
    TOP_K,
    VECTOR_INDEX_PATH,
    cors_origins,
    ensure_data_dirs,
)
from backend.rag.generator import AnswerGenerator
from backend.rag.retriever import Retriever
from backend.rag.schemas import (
    AskRequest,
    AskResponse,
    Citation,
    HealthResponse,
    RetrievedChunk,
    SourceMetadata,
)
from backend.rag.vector_store import VectorStore

@asynccontextmanager
async def lifespan(_: FastAPI):
    ensure_data_dirs()
    ensure_index_artifacts()
    yield


app = FastAPI(
    title="AI/ML Knowledge RAG API",
    version="2.0.0",
    description="FastAPI backend for grounded retrieval over indexed AI/ML documents.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    return Retriever()


@lru_cache(maxsize=1)
def get_generator() -> AnswerGenerator:
    return AnswerGenerator()


def _source_manifest_path() -> Path:
    return RAW_PDFS_DIR / "source_manifest.json"


def _load_metadata() -> list[dict[str, Any]]:
    if not METADATA_PATH.exists():
        return []

    import json

    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


def _load_sources() -> list[SourceMetadata]:
    manifest_path = _source_manifest_path()
    if manifest_path.exists():
        import json

        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        return [SourceMetadata(**item) for item in data]

    metadata = _load_metadata()
    seen: dict[str, dict[str, Any]] = {}
    for row in metadata:
        title = row.get("source_title") or row.get("file_name") or "Unknown source"
        if title in seen:
            continue
        seen[title] = {
            "title": title,
            "authors_or_organization": row.get("authors_or_organization", "Unknown"),
            "year": row.get("year", "Unknown"),
            "url": row.get("source_url", ""),
            "local_filename": row.get("file_name", ""),
            "license_or_access_note": row.get(
                "license_or_access_note", "Indexed local document."
            ),
            "category": row.get("category", "Uncategorized"),
        }
    return [SourceMetadata(**item) for item in seen.values()]


def _to_citation(chunk: dict[str, Any]) -> Citation:
    return Citation(
        chunk_id=chunk["chunk_id"],
        source_title=chunk["source_title"],
        file_name=chunk["file_name"],
        page_start=chunk.get("page_start"),
        page_end=chunk.get("page_end"),
        source_url=chunk.get("source_url"),
        category=chunk.get("category"),
        similarity_score=chunk["score"],
    )


def _to_retrieved_chunk(chunk: dict[str, Any]) -> RetrievedChunk:
    citation = _to_citation(chunk)
    return RetrievedChunk(
        chunk_id=citation.chunk_id,
        source_title=citation.source_title,
        file_name=citation.file_name,
        page_start=citation.page_start,
        page_end=citation.page_end,
        source_url=citation.source_url,
        category=citation.category,
        similarity_score=citation.similarity_score,
        text=chunk["text"],
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    index_loaded = index_files_present()
    total_chunks = len(_load_metadata()) if METADATA_PATH.exists() else 0
    sources_count = len(_load_sources())
    return HealthResponse(
        status="ok",
        backend_running=True,
        index_loaded=index_loaded,
        total_chunks=total_chunks,
        embedding_model=EMBEDDING_MODEL,
        index_path=str(VECTOR_INDEX_PATH),
        metadata_path=str(METADATA_PATH),
        top_k=TOP_K,
        similarity_threshold=SIMILARITY_THRESHOLD,
        sources_count=sources_count,
    )


@app.get("/ready")
def ready() -> dict[str, Any]:
    if not index_files_present():
        raise HTTPException(status_code=503, detail=INDEX_MISSING_MESSAGE)

    try:
        store = VectorStore.load()
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=INDEX_MISSING_MESSAGE) from exc

    if not store.metadata:
        raise HTTPException(status_code=503, detail=INDEX_MISSING_MESSAGE)

    return {
        "status": "ready",
        "index_loaded": True,
        "total_chunks": len(store.metadata),
        "embedding_model": EMBEDDING_MODEL,
    }


@app.get("/sources", response_model=list[SourceMetadata])
def sources() -> list[SourceMetadata]:
    return _load_sources()


@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest) -> AskResponse:
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    try:
        retriever = get_retriever()
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail=INDEX_MISSING_MESSAGE)

    retrieval = retriever.retrieve(question)
    citations = [_to_citation(chunk) for chunk in retrieval.chunks]
    retrieved_chunks = [_to_retrieved_chunk(chunk) for chunk in retrieval.chunks]
    similarity_scores = [chunk.similarity_score for chunk in citations]

    if retrieval.refusal:
        return AskResponse(
            answer=REFUSAL_MESSAGE,
            citations=citations,
            retrieved_chunks=retrieved_chunks,
            similarity_scores=similarity_scores,
            refusal=True,
            best_score=retrieval.best_score,
            top_k=TOP_K,
            similarity_threshold=SIMILARITY_THRESHOLD,
        )

    try:
        answer = get_generator().generate(question=question, chunks=retrieval.chunks)
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    final_answer = answer.strip() or REFUSAL_MESSAGE
    refused_by_generator = final_answer == REFUSAL_MESSAGE

    return AskResponse(
        answer=final_answer,
        citations=citations,
        retrieved_chunks=retrieved_chunks,
        similarity_scores=similarity_scores,
        refusal=refused_by_generator,
        best_score=retrieval.best_score,
        top_k=TOP_K,
        similarity_threshold=SIMILARITY_THRESHOLD,
    )


@app.get("/")
def root() -> dict[str, str]:
    return {
        "name": "AI/ML Knowledge RAG API",
        "health": "/health",
        "ask": "/ask",
        "sources": "/sources",
        "artifact_dir": str(INDEX_DIR),
    }
