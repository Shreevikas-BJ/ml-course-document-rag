from typing import Any

from backend.rag.config import (
    GROQ_API_KEY,
    GROQ_MODEL,
    LLM_PROVIDER,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    REFUSAL_MESSAGE,
    MAX_CONTEXT_CHARS,
)

SYSTEM_PROMPT = f"""You are a strict AI/ML course document assistant.

Rules:
- Answer only from the provided retrieved context.
- Do not use outside knowledge.
- If the context is insufficient, respond exactly with:
  {REFUSAL_MESSAGE}
- Include citations using the provided citation labels.
- Do not invent page numbers, source names, URLs, or facts.
- Keep answers student-friendly and concise.
"""


def citation_label(chunk: dict[str, Any], number: int) -> str:
    page_start = chunk.get("page_start")
    page_end = chunk.get("page_end")
    if page_start and page_end and page_start != page_end:
        page = f"pp. {page_start}-{page_end}"
    elif page_start:
        page = f"p. {page_start}"
    else:
        page = "page unavailable"
    return f"[C{number}: {chunk['source_title']}, {page}]"


def format_context(chunks: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    remaining_chars = MAX_CONTEXT_CHARS

    for idx, chunk in enumerate(chunks, start=1):
        if remaining_chars <= 0:
            break

        label = citation_label(chunk, idx)
        text = chunk["text"].strip()
        if len(text) > remaining_chars:
            text = text[:remaining_chars].rsplit(" ", 1)[0].strip() + "..."

        blocks.append(
            "\n".join(
                [
                    label,
                    f"Source URL: {chunk.get('source_url') or 'Unavailable'}",
                    f"Similarity: {chunk['score']:.4f}",
                    text,
                ]
            )
        )
        remaining_chars -= len(text)
    return "\n\n---\n\n".join(blocks)


def _extractive_answer(chunks: list[dict[str, Any]]) -> str:
    if not chunks:
        return REFUSAL_MESSAGE

    top = chunks[0]
    text = top["text"].strip()
    if len(text) > 900:
        text = text[:900].rsplit(" ", 1)[0].strip() + "..."
    return f"{text} {citation_label(top, 1)}"


class AnswerGenerator:
    def __init__(self) -> None:
        self.provider = LLM_PROVIDER
        self.openai_model = OPENAI_MODEL
        self.groq_model = GROQ_MODEL
        self.ollama_model = OLLAMA_MODEL
        self.ollama_base_url = OLLAMA_BASE_URL

    def generate(self, question: str, chunks: list[dict[str, Any]]) -> str:
        if self.provider in {"", "none", "extractive"}:
            return _extractive_answer(chunks)

        context = format_context(chunks)
        user_prompt = f"Question: {question}\n\nRetrieved context:\n{context}"

        if self.provider == "openai":
            return self._openai(user_prompt)
        if self.provider == "groq":
            return self._groq(user_prompt)
        if self.provider == "ollama":
            return self._ollama(user_prompt)

        raise ValueError(
            "Unsupported LLM_PROVIDER. Use one of: none, openai, groq, ollama."
        )

    def _openai(self, user_prompt: str) -> str:
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is required when LLM_PROVIDER=openai.")

        from openai import OpenAI

        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model=self.openai_model,
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content or REFUSAL_MESSAGE

    def _groq(self, user_prompt: str) -> str:
        if not GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY is required when LLM_PROVIDER=groq.")

        from openai import OpenAI

        client = OpenAI(
            api_key=GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1",
        )
        response = client.chat.completions.create(
            model=self.groq_model,
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content or REFUSAL_MESSAGE

    def _ollama(self, user_prompt: str) -> str:
        import requests

        response = requests.post(
            f"{self.ollama_base_url.rstrip('/')}/api/generate",
            json={
                "model": self.ollama_model,
                "prompt": f"{SYSTEM_PROMPT}\n\n{user_prompt}",
                "stream": False,
                "options": {"temperature": 0},
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json().get("response", "").strip() or REFUSAL_MESSAGE
