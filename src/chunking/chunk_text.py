import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any

import spacy
import tiktoken
from tqdm import tqdm

from src.utils.config import PAGES_JSONL_PATH, CHUNKS_DIR

# Chunking rules (your requirements)
CHUNK_TOKEN_BUDGET = 600
CHUNK_OVERLAP_TOKENS = 100

# We'll use a stable tokenizer to count tokens.
# (Token counting is for sizing only; embeddings use the HF model later.)
ENC = tiktoken.get_encoding("cl100k_base")


@dataclass
class ChunkRecord:
    chunk_id: str
    doc_id: str
    pdf_name: str
    page_number: int
    chunk_index: int
    text: str
    token_count: int


def count_tokens(text: str) -> int:
    return len(ENC.encode(text))


def split_into_sentences(nlp, text: str) -> List[str]:
    doc = nlp(text)
    sents = [s.text.strip() for s in doc.sents]
    return [s for s in sents if s]


def make_chunks_from_sentences(
    sentences: List[str],
    token_budget: int,
    overlap_tokens: int,
) -> List[str]:
    """
    Build chunks by packing sentences until token_budget.
    Overlap: carry last ~overlap_tokens worth of text into the next chunk.
    """
    chunks: List[str] = []
    cur: List[str] = []
    cur_tokens = 0

    def cur_text() -> str:
        return " ".join(cur).strip()

    for sent in sentences:
        sent_tokens = count_tokens(sent)

        # If one sentence is huge, still keep it alone (rare but possible with derivations)
        if not cur and sent_tokens > token_budget:
            chunks.append(sent)
            continue

        # If adding sentence exceeds budget → finalize chunk
        if cur_tokens + sent_tokens > token_budget and cur:
            chunks.append(cur_text())

            # Build overlap starter for next chunk
            # Take the end of the current chunk text such that it is ~overlap_tokens
            overlap_text = ""
            if overlap_tokens > 0:
                full = chunks[-1]
                ids = ENC.encode(full)
                tail_ids = ids[-overlap_tokens:] if len(ids) > overlap_tokens else ids
                overlap_text = ENC.decode(tail_ids).strip()

            cur = [overlap_text] if overlap_text else []
            cur_tokens = count_tokens(" ".join(cur)) if cur else 0

        # Add sentence
        cur.append(sent)
        cur_tokens += sent_tokens

    # Flush remainder
    if cur:
        chunks.append(cur_text())

    # Final cleanup: remove empties
    chunks = [c.strip() for c in chunks if c.strip()]
    return chunks


def main() -> None:
    if not PAGES_JSONL_PATH.exists():
        raise FileNotFoundError(
            f"Missing pages file. Run ingestion first: {PAGES_JSONL_PATH}"
        )

    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    chunks_path = CHUNKS_DIR / "chunks.jsonl"

    # spaCy pipeline: we only need sentence boundaries (fast + simple)
    nlp = spacy.load("en_core_web_sm")
    # Ensure sentence segmentation is enabled
    if "parser" not in nlp.pipe_names and "senter" not in nlp.pipe_names:
        # Fallback if model config changes (rare)
        nlp.add_pipe("sentencizer")

    total_chunks = 0

    with open(PAGES_JSONL_PATH, "r", encoding="utf-8") as f_in, \
         open(chunks_path, "w", encoding="utf-8") as f_out:

        for line in tqdm(f_in, desc="Chunking pages"):
            page: Dict[str, Any] = json.loads(line)

            doc_id = page["doc_id"]
            pdf_name = page["pdf_name"]
            page_number = int(page["page_number"])
            text = page["text"]

            sentences = split_into_sentences(nlp, text)
            chunk_texts = make_chunks_from_sentences(
                sentences,
                token_budget=CHUNK_TOKEN_BUDGET,
                overlap_tokens=CHUNK_OVERLAP_TOKENS,
            )

            for idx, chunk in enumerate(chunk_texts):
                tok_ct = count_tokens(chunk)
                chunk_id = f"{doc_id}::p{page_number}::c{idx}"

                rec = ChunkRecord(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    pdf_name=pdf_name,
                    page_number=page_number,
                    chunk_index=idx,
                    text=chunk,
                    token_count=tok_ct,
                )
                f_out.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
                total_chunks += 1

    print(f"\n✅ Wrote {total_chunks} chunks to:")
    print(f"   {chunks_path}")


if __name__ == "__main__":
    main()
