from pathlib import Path
from typing import Any

import faiss
import numpy as np

from backend.rag.config import METADATA_PATH, VECTOR_INDEX_PATH
from backend.rag.metadata import (
    MetadataRow,
    expand_metadata_record,
    load_metadata,
    write_metadata,
)


class VectorStore:
    def __init__(
        self,
        index: faiss.Index,
        metadata: list[MetadataRow] | list[dict[str, Any]],
    ) -> None:
        if index.ntotal != len(metadata):
            raise ValueError(
                f"FAISS vectors ({index.ntotal}) do not match metadata rows ({len(metadata)})."
            )
        self.index = index
        self.metadata = metadata

    @classmethod
    def build(cls, embeddings: np.ndarray, metadata: list[dict[str, Any]]) -> "VectorStore":
        if embeddings.ndim != 2:
            raise ValueError(f"Expected 2D embeddings, got shape {embeddings.shape}.")
        if embeddings.shape[0] != len(metadata):
            raise ValueError("Embedding rows and metadata rows must match.")

        vectors = embeddings.astype(np.float32)
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        return cls(index=index, metadata=metadata)

    @classmethod
    def load(
        cls,
        index_path: Path = VECTOR_INDEX_PATH,
        metadata_path: Path = METADATA_PATH,
    ) -> "VectorStore":
        if not index_path.exists():
            raise FileNotFoundError(f"Missing FAISS index: {index_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing metadata file: {metadata_path}")

        index = faiss.read_index(str(index_path))
        metadata = load_metadata(metadata_path, compact=True)
        return cls(index=index, metadata=metadata)

    def save(
        self,
        index_path: Path = VECTOR_INDEX_PATH,
        metadata_path: Path = METADATA_PATH,
    ) -> None:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_path))
        write_metadata(self.metadata, metadata_path)

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[dict[str, Any]]:
        query = query_embedding.reshape(1, -1).astype(np.float32)
        scores, ids = self.index.search(query, top_k)

        hits: list[dict[str, Any]] = []
        for idx, score in zip(ids[0].tolist(), scores[0].tolist()):
            if idx == -1:
                continue
            record = expand_metadata_record(self.metadata[idx])
            record["score"] = float(score)
            hits.append(record)
        return hits
