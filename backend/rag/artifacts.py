from pathlib import Path

import requests

from backend.rag.config import (
    INDEX_ARTIFACT_URL,
    METADATA_ARTIFACT_URL,
    METADATA_PATH,
    VECTOR_INDEX_PATH,
)

INDEX_MISSING_MESSAGE = (
    "FAISS index not found. Build the index locally and deploy/copy "
    "backend/data/index/faiss.index and metadata.json."
)


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".tmp")
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with tmp.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    tmp.replace(destination)


def ensure_index_artifacts() -> None:
    if not VECTOR_INDEX_PATH.exists() and INDEX_ARTIFACT_URL:
        print(f"Downloading FAISS index artifact to {VECTOR_INDEX_PATH}")
        _download(INDEX_ARTIFACT_URL, VECTOR_INDEX_PATH)

    if not METADATA_PATH.exists() and METADATA_ARTIFACT_URL:
        print(f"Downloading metadata artifact to {METADATA_PATH}")
        _download(METADATA_ARTIFACT_URL, METADATA_PATH)


def index_files_present() -> bool:
    return VECTOR_INDEX_PATH.exists() and METADATA_PATH.exists()
