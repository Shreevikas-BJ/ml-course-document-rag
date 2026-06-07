import json
import re
from pathlib import Path
from typing import Any, TypeAlias

from backend.rag.config import METADATA_PATH

METADATA_FIELDS = (
    "chunk_id",
    "source_title",
    "page_start",
    "page_end",
    "source_url",
    "text",
)

MetadataRow: TypeAlias = tuple[str, str, int | None, int | None, str | None, str]


def prune_metadata_record(record: dict[str, Any]) -> dict[str, Any]:
    return {field: record.get(field) for field in METADATA_FIELDS}


def compact_metadata_record(record: dict[str, Any]) -> MetadataRow:
    return (
        str(record.get("chunk_id") or ""),
        str(record.get("source_title") or ""),
        record.get("page_start"),
        record.get("page_end"),
        record.get("source_url"),
        str(record.get("text") or ""),
    )


def expand_metadata_record(record: MetadataRow | dict[str, Any]) -> dict[str, Any]:
    if isinstance(record, tuple):
        return dict(zip(METADATA_FIELDS, record, strict=True))
    return prune_metadata_record(record)


def load_metadata(
    metadata_path: Path = METADATA_PATH,
    *,
    compact: bool = False,
) -> list[MetadataRow] | list[dict[str, Any]]:
    with metadata_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, list):
        raise ValueError("metadata.json must contain a JSON list of chunk records.")

    if compact:
        return [compact_metadata_record(record) for record in data]
    return [prune_metadata_record(record) for record in data]


def write_metadata(
    records: list[MetadataRow] | list[dict[str, Any]],
    metadata_path: Path = METADATA_PATH,
) -> None:
    rows = [expand_metadata_record(record) for record in records]
    metadata_path.write_text(
        json.dumps(rows, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )


def count_metadata_records(metadata_path: Path = METADATA_PATH) -> int:
    if not metadata_path.exists():
        return 0

    count = 0
    marker = b'"chunk_id"'
    with metadata_path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            count += block.count(marker)
    return count


def count_unique_sources(metadata_path: Path = METADATA_PATH) -> int:
    if not metadata_path.exists():
        return 0

    sources: set[bytes] = set()
    pattern = re.compile(rb'"source_title"\s*:\s*"((?:\\.|[^"\\])*)"')
    carry = b""
    with metadata_path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            sample = carry + block
            sources.update(match.group(1) for match in pattern.finditer(sample))
            carry = sample[-512:]
    return len(sources)


def source_summaries_from_metadata(
    metadata_path: Path = METADATA_PATH,
) -> list[dict[str, str]]:
    rows = load_metadata(metadata_path, compact=True)
    seen: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        record = expand_metadata_record(row)
        title = str(record.get("source_title") or "").strip()
        url = str(record.get("source_url") or "").strip()
        if not title:
            continue
        key = (title, url)
        seen.setdefault(
            key,
            {
                "title": title,
                "authors_or_organization": "",
                "year": "",
                "url": url,
                "local_filename": "",
                "license_or_access_note": (
                    "Indexed source metadata is available in docs/SOURCE_MANIFEST.md."
                ),
                "category": "",
            },
        )
    return list(seen.values())
