import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator, List

import pdfplumber
from tqdm import tqdm

from src.utils.config import RAW_PDFS_DIR, PAGES_JSONL_PATH


@dataclass
class PageRecord:
    doc_id: str           # stable id derived from filename
    pdf_name: str         # original filename
    page_number: int      # 1-based page index (humans like this)
    text: str             # extracted text


def _clean_text(text: str) -> str:
    """
    Minimal cleaning only:
    - normalize whitespace
    - keep content faithful (no rewriting)
    """
    text = text.replace("\x00", " ")
    text = " ".join(text.split())
    return text.strip()


def iter_pdf_pages(pdf_path: Path) -> Iterator[PageRecord]:
    doc_id = pdf_path.stem  # e.g., "AMAL-Deep-Learning-Gallinari"
    pdf_name = pdf_path.name

    with pdfplumber.open(str(pdf_path)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            raw = page.extract_text() or ""
            cleaned = _clean_text(raw)
            # Keep empty pages too? We'll skip truly empty to reduce noise.
            if cleaned:
                yield PageRecord(
                    doc_id=doc_id,
                    pdf_name=pdf_name,
                    page_number=i,
                    text=cleaned,
                )


def load_all_pdfs(pdf_dir: Path) -> List[Path]:
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in: {pdf_dir}")
    return pdfs


def main() -> None:
    PAGES_JSONL_PATH.parent.mkdir(parents=True, exist_ok=True)

    pdf_paths = load_all_pdfs(RAW_PDFS_DIR)

    total_written = 0
    with open(PAGES_JSONL_PATH, "w", encoding="utf-8") as f_out:
        for pdf_path in tqdm(pdf_paths, desc="Ingesting PDFs"):
            for rec in iter_pdf_pages(pdf_path):
                f_out.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
                total_written += 1

    print(f"\n✅ Wrote {total_written} page records to:")
    print(f"   {PAGES_JSONL_PATH}")


if __name__ == "__main__":
    main()
