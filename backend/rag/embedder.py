from collections.abc import Sequence

import numpy as np

from backend.rag.config import EMBEDDING_BATCH_SIZE, EMBEDDING_MODEL


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return (embeddings / norms).astype(np.float32)


def auto_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


class LocalEmbedder:
    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL,
        batch_size: int = EMBEDDING_BATCH_SIZE,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device or auto_device()
        print(f"Embedding device: {self.device}")
        import torch
        from sentence_transformers import SentenceTransformer

        torch.set_num_threads(1)
        self.model = SentenceTransformer(model_name, device=self.device)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            raise ValueError("No texts were provided for embedding.")

        embeddings = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=len(texts) > self.batch_size,
        ).astype(np.float32)
        return normalize_embeddings(embeddings)

    def encode_query(self, question: str) -> np.ndarray:
        return self.encode([question.strip()])[0]
