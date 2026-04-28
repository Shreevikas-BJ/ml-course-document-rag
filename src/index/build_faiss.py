import json
from pathlib import Path
from typing import Dict, Any, List

import faiss
import numpy as np

from src.utils.config import EMBEDDINGS_DIR

EMB_NPY = EMBEDDINGS_DIR / "doc_embeddings.npy"
META_JSONL = EMBEDDINGS_DIR / "doc_chunks_meta.jsonl"

FAISS_INDEX_PATH = EMBEDDINGS_DIR / "faiss.index"
META_CACHE_PATH = EMBEDDINGS_DIR / "meta.jsonl"  # copy/standard name for runtime


def load_meta(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No metadata rows found in: {path}")
    return rows


def main() -> None:
    if not EMB_NPY.exists():
        raise FileNotFoundError(f"Missing embeddings file: {EMB_NPY}")
    if not META_JSONL.exists():
        raise FileNotFoundError(f"Missing metadata file: {META_JSONL}")

    # Load normalized embeddings (shape: [N, D])
    xb = np.load(EMB_NPY).astype(np.float32)
    if xb.ndim != 2:
        raise ValueError(f"Expected 2D embeddings array, got shape: {xb.shape}")

    n, d = xb.shape
    meta = load_meta(META_JSONL)
    if len(meta) != n:
        raise ValueError(f"Metadata rows ({len(meta)}) != embeddings rows ({n})")

    # Because embeddings are L2-normalized:
    # cosine_similarity(a,b) == inner_product(a,b)
    index = faiss.IndexFlatIP(d)
    index.add(xb)

    # Save FAISS index
    faiss.write_index(index, str(FAISS_INDEX_PATH))

    # Save metadata in a stable runtime filename
    # (so Streamlit app loads one predictable file)
    with open(META_CACHE_PATH, "w", encoding="utf-8") as f:
        for row in meta:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("✅ FAISS index built and saved:")
    print(f" - {FAISS_INDEX_PATH}   (vectors={n}, dim={d})")
    print("✅ Metadata saved for runtime:")
    print(f" - {META_CACHE_PATH}   (rows={len(meta)})")


if __name__ == "__main__":
    main()
