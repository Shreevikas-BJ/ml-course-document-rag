import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.rag.chunker import build_chunks
from backend.rag.config import (
    CHUNK_OVERLAP_TOKENS,
    CHUNK_SIZE_TOKENS,
    CHUNKS_PATH,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MODEL,
    EMBEDDING_STATE_PATH,
    INDEX_MANIFEST_PATH,
    METADATA_PATH,
    SIMILARITY_THRESHOLD,
    TOP_K,
    VECTOR_INDEX_PATH,
    ensure_data_dirs,
)
from backend.rag.embedder import LocalEmbedder
from backend.rag.ingest import current_pdf_fingerprints, extract_pages
from backend.rag.vector_store import VectorStore


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path, fallback: Any) -> Any:
    if not path.exists():
        return fallback
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _load_chunks() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with CHUNKS_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No chunks found in {CHUNKS_PATH}")
    return rows


def _index_state() -> dict[str, Any]:
    return {
        "pdf_hashes": current_pdf_fingerprints(),
        "chunks_hash": _file_sha256(CHUNKS_PATH),
        "embedding_model": EMBEDDING_MODEL,
        "embedding_batch_size": EMBEDDING_BATCH_SIZE,
        "chunk_size_tokens": CHUNK_SIZE_TOKENS,
        "chunk_overlap_tokens": CHUNK_OVERLAP_TOKENS,
        "top_k": TOP_K,
        "similarity_threshold": SIMILARITY_THRESHOLD,
    }


def build_index(force: bool = False) -> None:
    ensure_data_dirs()
    extract_pages(force=force)
    build_chunks(force=force)

    state = _index_state()
    previous_state = _read_json(EMBEDDING_STATE_PATH, {})
    if (
        not force
        and VECTOR_INDEX_PATH.exists()
        and METADATA_PATH.exists()
        and previous_state == state
    ):
        print(f"Index artifacts are current. Reusing {VECTOR_INDEX_PATH}")
        return

    chunks = _load_chunks()
    texts = [chunk["text"] for chunk in chunks]

    embedder = LocalEmbedder(model_name=EMBEDDING_MODEL, batch_size=EMBEDDING_BATCH_SIZE)
    embeddings = embedder.encode(texts)

    metadata = [
        {
            "chunk_id": chunk["chunk_id"],
            "source_title": chunk["source_title"],
            "authors_or_organization": chunk["authors_or_organization"],
            "year": chunk["year"],
            "source_url": chunk["source_url"],
            "license_or_access_note": chunk["license_or_access_note"],
            "category": chunk["category"],
            "file_name": chunk["file_name"],
            "file_hash": chunk["file_hash"],
            "page_start": chunk["page_start"],
            "page_end": chunk["page_end"],
            "chunk_index": chunk["chunk_index"],
            "text": chunk["text"],
            "token_count_estimate": chunk["token_count_estimate"],
        }
        for chunk in chunks
    ]

    store = VectorStore.build(embeddings=embeddings, metadata=metadata)
    store.save()

    manifest = dict(state)
    manifest["vector_count"] = len(metadata)
    manifest["embedding_dimension"] = int(embeddings.shape[1])
    _write_json(EMBEDDING_STATE_PATH, state)
    _write_json(INDEX_MANIFEST_PATH, manifest)

    print(f"Saved FAISS index: {VECTOR_INDEX_PATH}")
    print(f"Saved metadata: {METADATA_PATH}")
    print(f"Vectors: {len(metadata)}, dimension: {embeddings.shape[1]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the local FAISS RAG index.")
    parser.add_argument("--force", action="store_true", help="Rebuild every artifact.")
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download/update source PDFs before building the index.",
    )
    args = parser.parse_args()

    if args.download:
        from backend.scripts.download_sources import download_all

        download_all(force=args.force)

    build_index(force=args.force)


if __name__ == "__main__":
    main()
