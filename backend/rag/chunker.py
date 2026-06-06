import hashlib
import json
from pathlib import Path
from typing import Any

from tqdm import tqdm

from backend.rag.config import (
    CHUNK_OVERLAP_CHARS,
    CHUNK_OVERLAP_TOKENS,
    CHUNK_SIZE_CHARS,
    CHUNK_SIZE_TOKENS,
    CHUNK_STATE_PATH,
    CHUNKS_PATH,
    PAGES_PATH,
)


def _read_json(path: Path, fallback: Any) -> Any:
    if not path.exists():
        return fallback
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _find_chunk_end(text: str, start: int, target_end: int) -> int:
    if target_end >= len(text):
        return len(text)

    min_end = min(len(text), start + int(CHUNK_SIZE_CHARS * 0.75))
    split_at = text.rfind(" ", min_end, target_end)
    if split_at == -1:
        return target_end
    return split_at


def _window_text(text: str) -> list[str]:
    if len(text) <= CHUNK_SIZE_CHARS:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        target_end = min(len(text), start + CHUNK_SIZE_CHARS)
        end = _find_chunk_end(text, start, target_end)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - CHUNK_OVERLAP_CHARS, start + 1)
    return chunks


def _estimated_tokens(text: str) -> int:
    return max(1, round(len(text) / 4))


def build_chunks(force: bool = False) -> Path:
    if not PAGES_PATH.exists():
        raise FileNotFoundError(f"Missing extracted pages file: {PAGES_PATH}")

    pages_hash = _file_sha256(PAGES_PATH)
    state = {
        "pages_hash": pages_hash,
        "chunk_size_tokens": CHUNK_SIZE_TOKENS,
        "chunk_overlap_tokens": CHUNK_OVERLAP_TOKENS,
    }
    previous_state = _read_json(CHUNK_STATE_PATH, {})
    if not force and CHUNKS_PATH.exists() and previous_state == state:
        print(f"Pages and chunk settings unchanged. Reusing chunks at {CHUNKS_PATH}")
        return CHUNKS_PATH

    total = 0
    with PAGES_PATH.open("r", encoding="utf-8") as pages, CHUNKS_PATH.open(
        "w", encoding="utf-8"
    ) as output:
        for line in tqdm(pages, desc="Chunking pages"):
            page = json.loads(line)
            for chunk_index, chunk_text in enumerate(_window_text(page["text"])):
                chunk_id = (
                    f"{Path(page['file_name']).stem}::"
                    f"p{page['page_number']}::c{chunk_index}"
                )
                record = {
                    "chunk_id": chunk_id,
                    "source_title": page["source_title"],
                    "authors_or_organization": page["authors_or_organization"],
                    "year": page["year"],
                    "source_url": page["source_url"],
                    "license_or_access_note": page["license_or_access_note"],
                    "category": page["category"],
                    "file_name": page["file_name"],
                    "file_hash": page["file_hash"],
                    "page_start": int(page["page_number"]),
                    "page_end": int(page["page_number"]),
                    "chunk_index": chunk_index,
                    "text": chunk_text,
                    "token_count_estimate": _estimated_tokens(chunk_text),
                }
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                total += 1

    _write_json(CHUNK_STATE_PATH, state)
    print(f"Wrote {total} chunks to {CHUNKS_PATH}")
    return CHUNKS_PATH
