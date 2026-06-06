import hashlib
import json
from pathlib import Path
from typing import Any

import fitz
from tqdm import tqdm

from backend.rag.config import (
    INGEST_STATE_PATH,
    PAGES_PATH,
    RAW_PDFS_DIR,
    ensure_data_dirs,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    return " ".join(text.split()).strip()


def _read_json(path: Path, fallback: Any) -> Any:
    if not path.exists():
        return fallback
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def source_manifest() -> dict[str, dict[str, Any]]:
    manifest_path = RAW_PDFS_DIR / "source_manifest.json"
    if not manifest_path.exists():
        return {}

    rows = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {row["local_filename"]: row for row in rows}


def current_pdf_fingerprints() -> dict[str, str]:
    return {
        pdf.name: file_sha256(pdf)
        for pdf in sorted(RAW_PDFS_DIR.glob("*.pdf"))
        if pdf.is_file()
    }


def extract_pages(force: bool = False) -> Path:
    ensure_data_dirs()
    pdf_paths = sorted(RAW_PDFS_DIR.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(
            f"No PDFs found in {RAW_PDFS_DIR}. Run backend/scripts/download_sources.py first."
        )

    fingerprints = current_pdf_fingerprints()
    previous_state = _read_json(INGEST_STATE_PATH, {})
    next_state = {"pdf_hashes": fingerprints}

    if (
        not force
        and PAGES_PATH.exists()
        and previous_state.get("pdf_hashes") == fingerprints
    ):
        print(f"PDFs unchanged. Reusing extracted pages at {PAGES_PATH}")
        return PAGES_PATH

    manifest_by_file = source_manifest()
    written = 0
    with PAGES_PATH.open("w", encoding="utf-8") as output:
        for pdf_path in tqdm(pdf_paths, desc="Extracting PDF pages"):
            source = manifest_by_file.get(
                pdf_path.name,
                {
                    "title": pdf_path.stem,
                    "authors_or_organization": "Unknown",
                    "year": "Unknown",
                    "url": "",
                    "local_filename": pdf_path.name,
                    "license_or_access_note": "Local PDF supplied by the project owner.",
                    "category": "Uncategorized",
                },
            )

            with fitz.open(pdf_path) as document:
                for page_index, page in enumerate(document, start=1):
                    text = _clean_text(page.get_text("text") or "")
                    if not text:
                        continue
                    record = {
                        "source_title": source["title"],
                        "authors_or_organization": source["authors_or_organization"],
                        "year": source["year"],
                        "source_url": source["url"],
                        "license_or_access_note": source["license_or_access_note"],
                        "category": source["category"],
                        "file_name": pdf_path.name,
                        "file_hash": fingerprints[pdf_path.name],
                        "page_number": page_index,
                        "text": text,
                    }
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                    written += 1

    _write_json(INGEST_STATE_PATH, next_state)
    print(f"Wrote {written} extracted page records to {PAGES_PATH}")
    return PAGES_PATH
