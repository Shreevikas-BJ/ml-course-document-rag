from dataclasses import dataclass
from typing import List, Tuple

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from src.retrieval.retrieve import Retriever, RetrievedChunk


REFUSAL_MESSAGE = "Not enough information in the uploaded documents."


@dataclass
class RAGAnswer:
    answer: str
    citations: List[str]          # human-readable citations
    best_score: float
    passed_threshold: bool
    retrieved: List[RetrievedChunk]


def _format_context(chunks: List[RetrievedChunk]) -> str:
    """
    Build the ONLY evidence the LLM is allowed to use.
    Each chunk includes a strict citation tag the model must reference.
    """
    blocks = []
    for c in chunks:
        cite = f"[{c.pdf_name} | p.{c.page_number} | {c.chunk_id}]"
        blocks.append(f"{cite}\n{c.text}")
    return "\n\n---\n\n".join(blocks)


SYSTEM_RULES = """You are a STRICT document-grounded QA assistant.

HARD RULES (DO NOT BREAK):
- You MUST answer ONLY using information explicitly present in the CONTEXT.
- You MUST NOT introduce methods, examples, terminology, or interpretations that do NOT appear verbatim or clearly implied in the CONTEXT.
- You MUST NOT use outside knowledge.
- You MUST NOT guess or generalize.
- If the CONTEXT does not clearly answer the question, respond EXACTLY with:
  Not enough information in the uploaded documents.

CITATION RULES (MANDATORY):
- EVERY sentence in the answer MUST end with at least one citation tag.
- A citation tag MUST be copied EXACTLY from the CONTEXT (including brackets).
- If multiple sentences use the same source, EACH sentence must still include the citation.

STYLE RULES:
- Prefer restating or lightly rephrasing the CONTEXT.
- If the CONTEXT is high-level, keep the answer high-level.
- Do NOT add summaries, lists, or examples unless they appear in the CONTEXT.

OUTPUT FORMAT (EXACT):
Answer:
<sentence 1> [citation]
<sentence 2> [citation]

Citations:
- <unique citation tag 1>
- <unique citation tag 2>
"""


PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_RULES),
        ("human", "Question: {question}\n\nCONTEXT:\n{context}"),
    ]
)

def _extract_answer_and_citations(resp: str) -> tuple[str, list[str]]:
    text = resp.strip()

    # Very simple parser for your enforced format
    answer = text
    citations: list[str] = []

    if "Answer:" in text:
        answer_part = text.split("Answer:", 1)[1]
        if "Citations:" in answer_part:
            answer_block, citations_block = answer_part.split("Citations:", 1)
            answer = answer_block.strip()
            citations = [
                line.strip("- ").strip()
                for line in citations_block.strip().splitlines()
                if line.strip().startswith("-")
            ]
        else:
            answer = answer_part.strip()

    return answer, citations

class StrictRAG:
    """
    Orchestrates:
      - retrieval (threshold-gated)
      - strict prompt
      - OpenAI chat model for synthesis (still grounded)
    """
    def __init__(
        self,
        retriever: Retriever,
        openai_model: str = "gpt-5-mini",
        temperature: float = 0.0,
    ):
        self.retriever = retriever
        self.llm = ChatOpenAI(model=openai_model, temperature=temperature)

    def answer(self, question: str) -> RAGAnswer:
        chunks, best_score, passed = self.retriever.search(question)

        # Gate BEFORE calling the LLM (no hallucination)
        if not passed or not chunks:
            return RAGAnswer(
                answer=REFUSAL_MESSAGE,
                citations=[],
                best_score=best_score,
                passed_threshold=False,
                retrieved=chunks,
            )

        context = _format_context(chunks)

        chain = PROMPT | self.llm
        resp = chain.invoke({"question": question, "context": context}).content.strip()

        # Basic post-guard: if model violated refusal rule, enforce it.
        # (We still keep it simple; later we can add stricter parsing.)
        final_answer, model_citations = _extract_answer_and_citations(resp)

        if final_answer.strip() == REFUSAL_MESSAGE:
            return RAGAnswer(
                answer=REFUSAL_MESSAGE,
                citations=[],
                best_score=best_score,
                passed_threshold=passed,
                retrieved=chunks,
    )

        citation_list = model_citations or [
        f"{c.pdf_name} p.{c.page_number} ({c.chunk_id})" for c in chunks
]

        return RAGAnswer(
            answer=final_answer,
            citations=citation_list,
            best_score=best_score,
            passed_threshold=passed,
            retrieved=chunks,

)

