from src.retrieval.retrieve import Retriever


def main() -> None:
    r = Retriever(sim_threshold=0.25)

    q = "What is the difference between convex and non-convex optimization?"
    chunks, best, passed = r.search(q)

    print(f"Query: {q}")
    print(f"Best cosine similarity: {best:.4f}")
    print(f"Passed threshold? {passed}")

    print("\nTop retrieved:")
    for c in chunks:
        cite = f"{c.pdf_name} p.{c.page_number} ({c.chunk_id})"
        print(f"- score={c.score:.4f} | {cite}")
        print(f"  preview: {c.text[:180]}...")
        print("")


if __name__ == "__main__":
    main()
