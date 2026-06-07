import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_DIR.parent

DATA_DIR = BACKEND_DIR / "data"
RAW_PDFS_DIR = DATA_DIR / "raw_pdfs"
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "index"

PAGES_PATH = PROCESSED_DIR / "pages.jsonl"
CHUNKS_PATH = PROCESSED_DIR / "chunks.jsonl"
INGEST_STATE_PATH = PROCESSED_DIR / "ingest_state.json"
CHUNK_STATE_PATH = PROCESSED_DIR / "chunk_state.json"
EMBEDDING_STATE_PATH = PROCESSED_DIR / "embedding_state.json"

VECTOR_INDEX_PATH = INDEX_DIR / "faiss.index"
METADATA_PATH = INDEX_DIR / "metadata.json"
INDEX_MANIFEST_PATH = INDEX_DIR / "index_manifest.json"

INDEX_ARTIFACT_URL = os.getenv("INDEX_ARTIFACT_URL", "").strip()
METADATA_ARTIFACT_URL = os.getenv("METADATA_ARTIFACT_URL", "").strip()

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-MiniLM-L3-v2"
LOW_MEMORY_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _env_flag(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


LOW_MEMORY_MODE = _env_flag("LOW_MEMORY_MODE")
_configured_embedding_model = os.getenv("EMBEDDING_MODEL") or os.getenv(
    "RAG_EMBEDDING_MODEL"
)
EMBEDDING_MODEL = (
    _configured_embedding_model.strip()
    if _configured_embedding_model
    else LOW_MEMORY_EMBEDDING_MODEL
    if LOW_MEMORY_MODE
    else DEFAULT_EMBEDDING_MODEL
)
EMBEDDING_BATCH_SIZE = int(
    os.getenv(
        "EMBEDDING_BATCH_SIZE",
        os.getenv("RAG_BATCH_SIZE", "16" if LOW_MEMORY_MODE else "32"),
    )
)
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "5000"))

TOP_K = int(os.getenv("TOP_K", os.getenv("RAG_TOP_K", "3")))
SIMILARITY_THRESHOLD = float(
    os.getenv("SIMILARITY_THRESHOLD", os.getenv("RAG_SIMILARITY_THRESHOLD", "0.6"))
)

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower().strip()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1").strip()

CHUNK_SIZE_TOKENS = int(os.getenv("RAG_CHUNK_SIZE_TOKENS", "1000"))
CHUNK_OVERLAP_TOKENS = int(os.getenv("RAG_CHUNK_OVERLAP_TOKENS", "200"))
APPROX_CHARS_PER_TOKEN = 4
CHUNK_SIZE_CHARS = CHUNK_SIZE_TOKENS * APPROX_CHARS_PER_TOKEN
CHUNK_OVERLAP_CHARS = CHUNK_OVERLAP_TOKENS * APPROX_CHARS_PER_TOKEN

REFUSAL_MESSAGE = (
    "Not enough information in the indexed AI/ML documents to answer confidently."
)


def ensure_data_dirs() -> None:
    for path in (RAW_PDFS_DIR, PROCESSED_DIR, INDEX_DIR):
        path.mkdir(parents=True, exist_ok=True)


def cors_origins() -> list[str]:
    values = ["http://localhost:3000"]
    raw_allowed = os.getenv("ALLOWED_ORIGINS", os.getenv("CORS_ORIGINS", ""))
    raw_allowed = raw_allowed.strip().strip("[]")
    values.extend(origin.strip() for origin in raw_allowed.split(",") if origin.strip())

    frontend_origin = os.getenv("FRONTEND_ORIGIN", "").strip()
    if frontend_origin:
        values.append(frontend_origin)

    return list(dict.fromkeys(values))
