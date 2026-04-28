from dataclasses import dataclass
from typing import List, Dict

from src.retrieval.retrieve import Retriever


@dataclass
class EvalCase:
    question: str
    expected_keywords: List[str]  # lightweight "relevance proxy"


def precision_at_k(retrieved_texts: List[str], expected_keywords: List[str], k: int) -> float:
    """
    Simple proxy:
    - A retrieved chunk counts as relevant if it contains ANY expected keyword (case-insensitive).
    Precision@K = (# relevant in top K) / K
    """
    kws = [kw.lower() for kw in expected_keywords]
    top = retrieved_texts[:k]
    relevant = 0
    for t in top:
        tl = t.lower()
        if any(kw in tl for kw in kws):
            relevant += 1
    return relevant / max(k, 1)


def main() -> None:
    r = Retriever(sim_threshold=0.60, top_k=5)

    cases: List[EvalCase] = [
        EvalCase(
            question="What is the goal of explainable machine learning and why is it important?",
            expected_keywords=["explain", "interpret", "transparency", "trust"],
        ),
        EvalCase(
            question="What is bias in machine learning and how can fairness be evaluated?",
            expected_keywords=["bias", "fairness", "disparate", "equal", "metrics"],
        ),
        EvalCase(
            question="What are common optimization problems in machine learning?",
            expected_keywords=["optimization", "convex", "gradient", "objective", "loss"],
        ),
    ]

    print("\n=== Precision@K (proxy) ===")
    for c in cases:
        chunks, best, passed = r.search(c.question)
        texts = [x.text for x in chunks]
        p5 = precision_at_k(texts, c.expected_keywords, k=5)
        print(f"\nQ: {c.question}")
        print(f"Best score: {best:.4f} | Passed gate: {passed}")
        print(f"Precision@5: {p5:.2f}")
        print("Top sources:")
        for x in chunks:
            print(f" - {x.score:.4f} | {x.pdf_name} p.{x.page_number} ({x.chunk_id})")


if __name__ == "__main__":
    main()
