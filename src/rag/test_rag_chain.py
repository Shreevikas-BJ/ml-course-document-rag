import os

from src.retrieval.retrieve import Retriever
from src.rag.rag_chain import StrictRAG, REFUSAL_MESSAGE
from src.utils.langsmith_setup import enable_langsmith

enable_langsmith(project_name="ml-rag-docqa-phase1")

def main() -> None:
    # OpenAI key must be set in your environment:
    # setx OPENAI_API_KEY "your_key"
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("Missing OPENAI_API_KEY in environment.")

    retriever = Retriever(sim_threshold=0.60)
    rag = StrictRAG(retriever=retriever, openai_model="gpt-5-mini", temperature=0.0)

    q1 = "What is gradient descent?"
    a1 = rag.answer(q1)
    print("\n=== Q1 ===")
    print("Best score:", round(a1.best_score, 4), "Passed:", a1.passed_threshold)
    print(a1.answer)

    q2 = "What is the capital of France?"
    a2 = rag.answer(q2)
    print("\n=== Q2 ===")
    print("Best score:", round(a2.best_score, 4), "Passed:", a2.passed_threshold)
    print(a2.answer)
    assert a2.answer.strip() == REFUSAL_MESSAGE


if __name__ == "__main__":
    main()
