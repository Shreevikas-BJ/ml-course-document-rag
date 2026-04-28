from pathlib import Path

# Root paths (relative to repo)
REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = REPO_ROOT / "data"
RAW_PDFS_DIR = DATA_DIR / "raw_pdfs"
CHUNKS_DIR = DATA_DIR / "chunks"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"

# Outputs for ingestion step
PAGES_JSONL_PATH = CHUNKS_DIR / "pages.jsonl"
