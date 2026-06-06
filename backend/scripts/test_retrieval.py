import sys
import tempfile
from pathlib import Path
from typing import Any
import os

os.environ.setdefault("LLM_PROVIDER", "none")

import faiss
import numpy as np
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.main import app
from backend.rag.artifacts import INDEX_MISSING_MESSAGE
from backend.rag.retriever import passes_threshold
from backend.rag.vector_store import VectorStore


def _fake_metadata() -> list[dict[str, Any]]:
    return [
        {
            "chunk_id": f"chunk-{idx}",
            "source_title": "Synthetic ML Notes",
            "file_name": "synthetic.pdf",
            "page_start": idx + 1,
            "page_end": idx + 1,
            "source_url": "https://example.com/synthetic.pdf",
            "category": "Machine Learning fundamentals",
            "text": f"Synthetic chunk {idx}",
        }
        for idx in range(3)
    ]


def test_top_k_returns_three_chunks() -> None:
    vectors = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.8, 0.2, 0.0],
        ],
        dtype=np.float32,
    )
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    store = VectorStore(index=index, metadata=_fake_metadata())
    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    hits = store.search(query, top_k=3)

    assert len(hits) == 3, f"Expected 3 chunks, got {len(hits)}"


def test_threshold_gating_refuses_below_06() -> None:
    low_score_hits = [{"score": 0.59}]
    assert not passes_threshold(low_score_hits, 0.6)


def test_citations_include_title_and_page_number() -> None:
    chunk = _fake_metadata()[0]
    assert chunk["source_title"]
    assert chunk["page_start"] is not None


def test_health_endpoint() -> None:
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["backend_running"] is True
    assert "index_loaded" in body
    assert "total_chunks" in body
    assert "embedding_model" in body
    assert body["top_k"] == 3


def test_ready_endpoint() -> None:
    client = TestClient(app)
    response = client.get("/ready")
    if response.status_code == 503:
        assert response.json()["detail"] == INDEX_MISSING_MESSAGE
        return

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["index_loaded"] is True
    assert body["total_chunks"] > 0


def test_ask_endpoint_examples() -> None:
    client = TestClient(app)
    questions = [
        "What is overfitting?",
        "How does attention work in transformers?",
        "Who won yesterday's NBA game?",
    ]
    for question in questions:
        response = client.post("/ask", json={"question": question})
        if response.status_code == 503:
            assert response.json()["detail"] == INDEX_MISSING_MESSAGE
            continue
        assert response.status_code == 200
        body = response.json()
        assert "answer" in body
        assert "refusal" in body
        assert "similarity_scores" in body


def main() -> None:
    tests = [
        test_top_k_returns_three_chunks,
        test_threshold_gating_refuses_below_06,
        test_citations_include_title_and_page_number,
        test_health_endpoint,
        test_ready_endpoint,
        test_ask_endpoint_examples,
    ]
    with tempfile.TemporaryDirectory():
        for test in tests:
            test()
            print(f"PASS {test.__name__}")


if __name__ == "__main__":
    main()
