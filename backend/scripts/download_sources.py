import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.rag.config import RAW_PDFS_DIR, ensure_data_dirs
from backend.rag.ingest import file_sha256

SOURCES: list[dict[str, str]] = [
    {
        "title": "CS229 Machine Learning Notes",
        "authors_or_organization": "Stanford University CS229",
        "year": "2022",
        "url": "https://cs229.stanford.edu/main_notes.pdf",
        "local_filename": "stanford-cs229-main-notes.pdf",
        "license_or_access_note": "Official Stanford course notes, publicly accessible for course use.",
        "category": "Machine Learning fundamentals",
    },
    {
        "title": "Dive into Deep Learning",
        "authors_or_organization": "Aston Zhang, Zachary C. Lipton, Mu Li, Alexander J. Smola",
        "year": "2024",
        "url": "https://d2l.ai/d2l-en.pdf",
        "local_filename": "dive-into-deep-learning.pdf",
        "license_or_access_note": "Official open-source book PDF from d2l.ai.",
        "category": "Deep Learning",
    },
    {
        "title": "Attention Is All You Need",
        "authors_or_organization": "Ashish Vaswani et al.",
        "year": "2017",
        "url": "https://arxiv.org/pdf/1706.03762",
        "local_filename": "attention-is-all-you-need.pdf",
        "license_or_access_note": "arXiv open-access preprint.",
        "category": "Modern AI / LLMs / Generative AI",
    },
    {
        "title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
        "authors_or_organization": "Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova",
        "year": "2018",
        "url": "https://arxiv.org/pdf/1810.04805",
        "local_filename": "bert.pdf",
        "license_or_access_note": "arXiv open-access preprint.",
        "category": "Modern AI / LLMs / Generative AI",
    },
    {
        "title": "Language Models are Few-Shot Learners",
        "authors_or_organization": "Tom B. Brown et al.",
        "year": "2020",
        "url": "https://arxiv.org/pdf/2005.14165",
        "local_filename": "gpt3-few-shot-learners.pdf",
        "license_or_access_note": "arXiv open-access preprint.",
        "category": "Modern AI / LLMs / Generative AI",
    },
    {
        "title": "Llama 2: Open Foundation and Fine-Tuned Chat Models",
        "authors_or_organization": "Hugo Touvron et al.",
        "year": "2023",
        "url": "https://arxiv.org/pdf/2307.09288",
        "local_filename": "llama-2-technical-report.pdf",
        "license_or_access_note": "arXiv open-access technical report.",
        "category": "Modern AI / LLMs / Generative AI",
    },
    {
        "title": "Retrieval-Augmented Generation for Large Language Models: A Survey",
        "authors_or_organization": "Yunfan Gao et al.",
        "year": "2023",
        "url": "https://arxiv.org/pdf/2312.10997",
        "local_filename": "rag-for-llms-survey.pdf",
        "license_or_access_note": "arXiv open-access survey.",
        "category": "RAG and AI agents",
    },
    {
        "title": "A Survey on RAG Meeting LLMs: Towards Retrieval-Augmented Large Language Models",
        "authors_or_organization": "Wenqi Fan, Yujuan Ding, Liangbo Ning, Shijie Wang, Hengyun Li, Dawei Yin, Tat-Seng Chua, Qing Li",
        "year": "2024",
        "url": "https://arxiv.org/pdf/2405.06211",
        "local_filename": "rag-meeting-llms-survey.pdf",
        "license_or_access_note": "arXiv open-access survey.",
        "category": "RAG and AI agents",
    },
    {
        "title": "A Survey of Graph Retrieval-Augmented Generation for Customized Large Language Models",
        "authors_or_organization": "Qinggang Zhang, Shengyuan Chen, Yuanchen Bei, Zheng Yuan, Huachi Zhou, Zijin Hong, Hao Chen, Yilin Xiao, Chuang Zhou, Junnan Dong, Yi Chang, Xiao Huang",
        "year": "2025",
        "url": "https://arxiv.org/pdf/2501.13958",
        "local_filename": "graph-rag-survey.pdf",
        "license_or_access_note": "arXiv open-access survey.",
        "category": "RAG and AI agents",
    },
    {
        "title": "The Rise and Potential of Large Language Model Based Agents: A Survey",
        "authors_or_organization": "Zhiheng Xi et al.",
        "year": "2023",
        "url": "https://arxiv.org/pdf/2309.07864",
        "local_filename": "llm-agents-survey.pdf",
        "license_or_access_note": "arXiv open-access survey.",
        "category": "RAG and AI agents",
    },
    {
        "title": "Artificial Intelligence Risk Management Framework (AI RMF 1.0)",
        "authors_or_organization": "National Institute of Standards and Technology",
        "year": "2023",
        "url": "https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf",
        "local_filename": "nist-ai-rmf-1-0.pdf",
        "license_or_access_note": "Official NIST publication, publicly accessible.",
        "category": "Responsible AI / Safety / Governance",
    },
    {
        "title": "AI RMF Generative AI Profile",
        "authors_or_organization": "National Institute of Standards and Technology",
        "year": "2024",
        "url": "https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf",
        "local_filename": "nist-ai-rmf-generative-ai-profile.pdf",
        "license_or_access_note": "Official NIST publication, publicly accessible.",
        "category": "Responsible AI / Safety / Governance",
    },
    {
        "title": "OWASP Top 10 for LLM Applications 2025",
        "authors_or_organization": "OWASP Foundation",
        "year": "2025",
        "url": "https://owasp.org/www-project-top-10-for-large-language-model-applications/assets/PDF/OWASP-Top-10-for-LLMs-v2025.pdf",
        "local_filename": "owasp-top-10-llm-applications-2025.pdf",
        "license_or_access_note": "Official OWASP project PDF, publicly accessible.",
        "category": "Responsible AI / Safety / Governance",
    },
    {
        "title": "AI Index Report 2025",
        "authors_or_organization": "Stanford Institute for Human-Centered Artificial Intelligence",
        "year": "2025",
        "url": "https://hai.stanford.edu/assets/files/hai_ai_index_report_2025.pdf",
        "local_filename": "stanford-ai-index-2025.pdf",
        "license_or_access_note": "Official Stanford HAI AI Index report PDF, publicly accessible.",
        "category": "Responsible AI / Safety / Governance",
    },
]


def _download_file(url: str, destination: Path, retries: int = 3) -> None:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=60) as response:
                response.raise_for_status()
                tmp = destination.with_suffix(destination.suffix + ".tmp")
                with tmp.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
                tmp.replace(destination)
                return
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(2 * attempt)
    raise RuntimeError(f"Failed to download {url}") from last_error


def download_all(force: bool = False) -> list[dict[str, Any]]:
    ensure_data_dirs()
    enriched: list[dict[str, Any]] = []

    for source in SOURCES:
        destination = RAW_PDFS_DIR / source["local_filename"]
        if destination.exists() and destination.stat().st_size > 0 and not force:
            print(f"Skipping unchanged local file: {destination.name}")
        else:
            print(f"Downloading {source['title']} -> {destination.name}")
            _download_file(source["url"], destination)

        record: dict[str, Any] = dict(source)
        record["sha256"] = file_sha256(destination)
        record["bytes"] = destination.stat().st_size
        enriched.append(record)

    manifest_path = RAW_PDFS_DIR / "source_manifest.json"
    manifest_path.write_text(
        json.dumps(enriched, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote source manifest: {manifest_path}")
    return enriched


def main() -> None:
    parser = argparse.ArgumentParser(description="Download legal AI/ML RAG source PDFs.")
    parser.add_argument("--force", action="store_true", help="Re-download existing PDFs.")
    args = parser.parse_args()
    download_all(force=args.force)


if __name__ == "__main__":
    main()
