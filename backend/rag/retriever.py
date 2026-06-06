from dataclasses import dataclass
from typing import Any

from backend.rag.config import SIMILARITY_THRESHOLD, TOP_K
from backend.rag.embedder import LocalEmbedder
from backend.rag.vector_store import VectorStore


@dataclass
class RetrievalResult:
    chunks: list[dict[str, Any]]
    best_score: float | None
    refusal: bool


def passes_threshold(chunks: list[dict[str, Any]], threshold: float) -> bool:
    return any(chunk.get("score", -1.0) >= threshold for chunk in chunks)


class Retriever:
    def __init__(
        self,
        top_k: int = TOP_K,
        similarity_threshold: float = SIMILARITY_THRESHOLD,
        embedder: LocalEmbedder | None = None,
        vector_store: VectorStore | None = None,
    ) -> None:
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.vector_store = vector_store or VectorStore.load()
        self.embedder = embedder or LocalEmbedder()

    def retrieve(self, question: str) -> RetrievalResult:
        query_embedding = self.embedder.encode_query(question)
        chunks = self.vector_store.search(query_embedding, self.top_k)
        best_score = chunks[0]["score"] if chunks else None
        refusal = not chunks or not passes_threshold(chunks, self.similarity_threshold)
        return RetrievalResult(chunks=chunks, best_score=best_score, refusal=refusal)
