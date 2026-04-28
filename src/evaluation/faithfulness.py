import os
from dataclasses import dataclass
from typing import List

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from src.rag.rag_chain import StrictRAG, REFUSAL_MESSAGE
from src.retrieval.retrieve import Retriever


@dataclass
class FaithfulnessCase:
    question: str


CHECK_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You are a strict evaluator. "
         "You will be given a QUESTION, an ANSWER, and SOURCES (chunks). "
         "Decide if every claim in the ANSWER is supported by SOURCES.\n\n"
         "Return ONLY one of:\n"
         "- PASS\n"
         "- FAIL: <short reason>\n"
         ),
        ("human",
         "QUESTION:\n{question}\n\nANSWER:\n{answer}\n\nSOURCES:\n{sources}"
         )
    ]
)


def format_sources(retrieved) -> str:
    parts = []
    for c in retrieved:
        tag = f"[{c.pdf_name} p.{c.page_number} {c.chunk_id}]"
        parts.append(f"{tag}\n{c.text}")
    return "\n\n---\n\n".join(parts)


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("Missing OPENAI_API_KEY in environment.")

    retriever = Retriever(sim_threshold=0.25, top_k=5)
    rag = StrictRAG(retriever=retriever, openai_model="gpt-4o-mini", temperature=0.0)

    judge = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    judge_chain = CHECK_PROMPT | judge

    cases: List[FaithfulnessCase] = [
        FaithfulnessCase(question="Explain the main idea of explainable machine learning."),
        FaithfulnessCase(question="How do we define bias and fairness in ML?"),
    ]

    print("\n=== Faithfulness Check ===")
    for c in cases:
        out = rag.answer(c.question)

        print(f"\nQ: {c.question}")
        print(f"Best score: {out.best_score:.4f} | Passed gate: {out.passed_threshold}")

        if out.answer.strip() == REFUSAL_MESSAGE:
            print("Answer: REFUSAL (no sources strong enough) ✅")
            continue

        sources = format_sources(out.retrieved)
        verdict = judge_chain.invoke(
            {"question": c.question, "answer": out.answer, "sources": sources}
        ).content.strip()

        print("Verdict:", verdict)


if __name__ == "__main__":
    main()
