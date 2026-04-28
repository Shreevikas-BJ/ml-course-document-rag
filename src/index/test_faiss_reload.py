import faiss
import numpy as np

from src.utils.config import EMBEDDINGS_DIR

FAISS_INDEX_PATH = EMBEDDINGS_DIR / "faiss.index"
EMB_NPY = EMBEDDINGS_DIR / "doc_embeddings.npy"


def main() -> None:
    if not FAISS_INDEX_PATH.exists():
        raise FileNotFoundError(f"Missing index: {FAISS_INDEX_PATH}")

    index = faiss.read_index(str(FAISS_INDEX_PATH))
    print(f"✅ Loaded FAISS index: ntotal={index.ntotal}, d={index.d}")

    # simple search with a random existing vector
    xb = np.load(EMB_NPY).astype(np.float32)
    q = xb[0:1]  # already normalized

    scores, ids = index.search(q, k=5)
    print("Top-5 IDs:", ids[0].tolist())
    print("Top-5 scores:", [round(float(s), 4) for s in scores[0]])


if __name__ == "__main__":
    main()
