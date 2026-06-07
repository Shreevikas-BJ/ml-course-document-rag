import argparse
import os
import sys

import requests


def _get(base_url: str, path: str) -> dict:
    response = requests.get(f"{base_url.rstrip('/')}{path}", timeout=60)
    response.raise_for_status()
    return response.json()


def _post(base_url: str, path: str, payload: dict) -> dict:
    response = requests.post(f"{base_url.rstrip('/')}{path}", json=payload, timeout=120)
    response.raise_for_status()
    return response.json()


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test a deployed RAG API.")
    parser.add_argument(
        "--api-base-url",
        default=os.getenv("API_BASE_URL", "http://localhost:8000"),
        help="Backend API base URL, or set API_BASE_URL.",
    )
    args = parser.parse_args()

    base_url = args.api_base_url
    print(f"Testing {base_url}")

    health = _get(base_url, "/health")
    assert health["backend_running"] is True
    assert health["top_k"] == 3
    assert health["similarity_threshold"] == 0.6
    assert health["index_loaded"] is True
    assert health["total_chunks"] > 0
    print("PASS /health")

    sources = _get(base_url, "/sources")
    assert isinstance(sources, list)
    assert len(sources) > 0
    print("PASS /sources")

    answer = _post(base_url, "/ask", {"question": "What is overfitting?"})
    assert answer["top_k"] <= 3
    assert answer["similarity_threshold"] == 0.6
    assert len(answer["citations"]) > 0
    assert len(answer["retrieved_chunks"]) <= 3
    best_score = answer["best_score"] or 0.0
    if best_score >= answer["similarity_threshold"]:
        assert answer["refusal"] is False
    else:
        assert answer["refusal"] is True
    print("PASS /ask grounded retrieval and threshold gate")

    unsupported = _post(
        base_url,
        "/ask",
        {"question": "Who won yesterday's NBA game?"},
    )
    assert unsupported["top_k"] <= 3
    assert unsupported["similarity_threshold"] == 0.6
    assert unsupported["refusal"] is True
    print("PASS /ask threshold/refusal question")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as exc:
        print(f"FAIL {exc}")
        sys.exit(1)
