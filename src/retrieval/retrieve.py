import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Tuple

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from src.utils.config import EMBEDDINGS_DIR

MODEL_NAME = "BAAI/bge-base-en-v1.5"  # HARD REQUIREMENT

FAISS_INDEX_PATH = EMBEDDINGS_DIR / "faiss.index"
META_PATH = EMBEDDINGS_DIR / "meta.jsonl"

TOP_K = 5

# Similarity threshold gate:
# Since we L2-normalize embeddings, FAISS IndexFlatIP returns cosine similarity in [-1, 1].
# Start with 0.25 as a reasonable default, and we will tune later using evaluation.
SIM_THRESHOLD = 0.60


@dataclass
class RetrievedChunk:
    chunk_id: str
    pdf_name: str
    page_number: int
    chunk_index: int
    score: float
    text: str


def _load_meta(meta_path: Path) -> List[Dict[str, Any]]:
    meta: List[Dict[str, Any]] = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            meta.append(json.loads(line))
    if not meta:
        raise ValueError(f"Empty metadata: {meta_path}")
    return meta


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return (x / norm).astype(np.float32)


class Retriever:
    """
    Loads:
      - FAISS index (CPU)
      - chunk metadata
      - embedding model (CPU) for query embeddings ONLY (online constraint)
    """
    def __init__(
        self,
        index_path: Path = FAISS_INDEX_PATH,
        meta_path: Path = META_PATH,
        top_k: int = TOP_K,
        sim_threshold: float = SIM_THRESHOLD,
        device: str = "cpu",
    ):
        if not index_path.exists():
            raise FileNotFoundError(f"Missing FAISS index: {index_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing metadata: {meta_path}")

        self.index = faiss.read_index(str(index_path))
        self.meta = _load_meta(meta_path)

        if self.index.ntotal != len(self.meta):
            raise ValueError(
                f"Index vectors ({self.index.ntotal}) != meta rows ({len(self.meta)})"
            )

        self.top_k = top_k
        self.sim_threshold = sim_threshold

        # Query embedding model MUST run on CPU in online mode
        if device != "cpu":
            raise ValueError("Retriever device must be 'cpu' (online constraint).")
        self.device = device

        # SentenceTransformer (PyTorch-based)
        self.model = SentenceTransformer(MODEL_NAME, device=self.device)

    @torch.no_grad()
    def embed_query(self, query: str) -> np.ndarray:
        q = query.strip()
        if not q:
            return np.zeros((1, self.index.d), dtype=np.float32)

        emb = self.model.encode(
            [q],
            convert_to_numpy=True,
            normalize_embeddings=False,  # normalize explicitly
            show_progress_bar=False,
        ).astype(np.float32)

        return _l2_normalize(emb)

    def search(self, query: str) -> Tuple[List[RetrievedChunk], float, bool]:
        """
        Returns:
          (retrieved_chunks, best_score, passed_threshold)
        """
        qv = self.embed_query(query)  # shape: (1, d)
        scores, ids = self.index.search(qv, self.top_k)

        ids_list = ids[0].tolist()
        scores_list = scores[0].tolist()

        results: List[RetrievedChunk] = []
        best_score = float(scores_list[0]) if scores_list and ids_list[0] != -1 else -1.0

        for idx, score in zip(ids_list, scores_list):
            if idx == -1:
                continue
            m = self.meta[idx]
            results.append(
                RetrievedChunk(
                    chunk_id=m["chunk_id"],
                    pdf_name=m["pdf_name"],
                    page_number=int(m["page_number"]),
                    chunk_index=int(m["chunk_index"]),
                    score=float(score),
                    text=m["text"],
                )
            )

        passed = best_score >= self.sim_threshold
        return results, best_score, passed
