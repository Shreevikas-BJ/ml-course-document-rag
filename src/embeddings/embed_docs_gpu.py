import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from src.utils.config import CHUNKS_DIR, EMBEDDINGS_DIR

MODEL_NAME = "BAAI/bge-base-en-v1.5"  # HARD REQUIREMENT

# Tune for RTX 8GB if needed (start safe)
BATCH_SIZE = 32

CHUNKS_JSONL = CHUNKS_DIR / "chunks.jsonl"
OUT_EMB_NPY = EMBEDDINGS_DIR / "doc_embeddings.npy"
OUT_META_JSONL = EMBEDDINGS_DIR / "doc_chunks_meta.jsonl"


def load_chunks(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No chunks found in: {path}")
    return rows


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norm


@torch.no_grad()
def main() -> None:
    if not CHUNKS_JSONL.exists():
        raise FileNotFoundError(f"Missing chunks file. Run chunking first: {CHUNKS_JSONL}")

    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError(
            "CUDA not available. This step MUST run on GPU for offline doc embeddings."
        )

    print(f"✅ Using device: {device}")

    # SentenceTransformer uses transformers under the hood (PyTorch-based)
    model = SentenceTransformer(MODEL_NAME, device=device)

    chunks = load_chunks(CHUNKS_JSONL)
    texts = [c["text"] for c in chunks]

    all_embs: List[np.ndarray] = []

    for start in tqdm(range(0, len(texts), BATCH_SIZE), desc="Embedding (GPU)"):
        batch_texts = texts[start : start + BATCH_SIZE]

        # Encode -> returns numpy if convert_to_numpy=True
        emb = model.encode(
            batch_texts,
            batch_size=len(batch_texts),
            convert_to_numpy=True,
            normalize_embeddings=False,  # we normalize ourselves (explicit requirement)
            show_progress_bar=False,
        )
        all_embs.append(emb.astype(np.float32))

    doc_embs = np.vstack(all_embs)
    doc_embs = l2_normalize(doc_embs).astype(np.float32)

    # Save embeddings
    np.save(OUT_EMB_NPY, doc_embs)

    # Save metadata (aligned row-by-row with embeddings)
    with open(OUT_META_JSONL, "w", encoding="utf-8") as f:
        for c in chunks:
            # Keep only what runtime needs (citations)
            meta = {
                "chunk_id": c["chunk_id"],
                "doc_id": c["doc_id"],
                "pdf_name": c["pdf_name"],
                "page_number": c["page_number"],
                "chunk_index": c["chunk_index"],
                "token_count": c["token_count"],
                "text": c["text"],  # keep text for retrieval + citations
            }
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")

    print("\n✅ Saved outputs:")
    print(f" - Embeddings: {OUT_EMB_NPY}   shape={doc_embs.shape}")
    print(f" - Metadata:   {OUT_META_JSONL}   rows={len(chunks)}")


if __name__ == "__main__":
    main()
