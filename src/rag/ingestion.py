"""
Document ingestion pipeline.

Reads Markdown / text files from the corpus, chunks them with overlap,
and upserts into pgvector via the vectorstore module.
"""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path
from typing import Any

import psycopg

from src.config import get_settings
from src.rag.vectorstore import register_document, stable_document_id, upsert_chunks

logger = logging.getLogger(__name__)
settings = get_settings()

DOC_TYPE_MAP: dict[str, str] = {
    "well_reports": "Well Report",
    "maintenance_logs": "Maintenance Log",
    "hse_procedures": "HSE Procedure",
    "equipment_manuals": "Equipment Manual",
    "volve_real": "Volve Open Dataset (production)",
}


# ─── Chunking ─────────────────────────────────────────────────────────────────


def _extract_title(text: str, fallback: str) -> str:
    """Extract the first H1 or H2 heading from Markdown."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
        if stripped.startswith("## "):
            return stripped[3:].strip()
    return fallback


def _split_into_sections(text: str) -> list[tuple[str, str]]:
    """Split Markdown into (heading, content) pairs at H2 boundaries."""
    sections: list[tuple[str, str]] = []
    current_heading = "Introduction"
    current_lines: list[str] = []

    for line in text.splitlines():
        if line.startswith("## "):
            if current_lines:
                sections.append((current_heading, "\n".join(current_lines)))
            current_heading = line[3:].strip()
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_heading, "\n".join(current_lines)))

    return sections


def _chunk_text(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    section: str | None = None,
) -> list[dict[str, Any]]:
    """Split text into overlapping word-based chunks."""
    words = text.split()
    chunks: list[dict[str, Any]] = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunks.append(
            {
                "content": " ".join(chunk_words),
                "section": section,
                "token_count": len(chunk_words),
            }
        )
        if end == len(words):
            break
        start += chunk_size - chunk_overlap

    return chunks


def chunk_document(
    text: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[dict[str, Any]]:
    """Chunk a full document, preserving section headings."""
    cs = chunk_size or settings.rag_chunk_size
    co = chunk_overlap or settings.rag_chunk_overlap

    sections = _split_into_sections(text)
    all_chunks: list[dict[str, Any]] = []

    for heading, content in sections:
        if not content.strip():
            continue
        section_chunks = _chunk_text(content.strip(), cs, co, section=heading)
        all_chunks.extend(section_chunks)

    # Assign sequential chunk indices
    for i, chunk in enumerate(all_chunks):
        chunk["chunk_index"] = i

    return all_chunks


# ─── Ingestion ────────────────────────────────────────────────────────────────


def _detect_doc_type(path: Path) -> str:
    for dir_name, doc_type in DOC_TYPE_MAP.items():
        if dir_name in str(path):
            return doc_type
    return "Document"


def _detect_well_id(text: str) -> str | None:
    """Extract well ID from document content using heuristics."""
    patterns = [
        r"\*\*Well ID:\*\*\s*([\w/\-]+)",
        r"Well ID[:\s]+([\w/\-]+)",
        r"\b(1[45]/\d+-[A-Z]-\d+[A-Z]?)\b",  # Norwegian well format
        r"\b([A-Z]{1,3}-\d+[A-Z]{0,2}H?)\b",  # Draugen/Gullfaks format
    ]
    for pat in patterns:
        match = re.search(pat, text[:1000])
        if match:
            return match.group(1)
    return None


def _detect_field(text: str) -> str | None:
    """Extract field name from document content."""
    fields = ["Draugen", "Volve", "Gullfaks", "Oseberg"]
    text_lower = text[:2000].lower()
    for field in fields:
        if field.lower() in text_lower:
            return field
    return None


async def ingest_file(
    conn: psycopg.AsyncConnection[Any],
    file_path: Path,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> int:
    """Ingest a single document file. Returns number of chunks created."""
    text = file_path.read_text(encoding="utf-8")
    if not text.strip():
        return 0

    document_id = stable_document_id(str(file_path))
    title = _extract_title(text, fallback=file_path.stem)
    doc_type = _detect_doc_type(file_path)
    well_id = _detect_well_id(text)
    field_name = _detect_field(text)

    chunks = chunk_document(text, chunk_size, chunk_overlap)

    await register_document(
        conn,
        document_id=document_id,
        title=title,
        doc_type=doc_type,
        source_path=str(file_path),
        well_id=well_id,
        field_name=field_name,
        metadata={"filename": file_path.name, "category": doc_type},
    )
    await upsert_chunks(
        conn,
        document_id,
        chunks,
        document_title=title,
        well_id=well_id,
        field_name=field_name,
    )
    await conn.commit()

    logger.info("Ingested %s → %d chunks (doc_id: %s)", file_path.name, len(chunks), document_id)
    return len(chunks)


async def ingest_corpus(
    docs_dir: Path,
    db_url: str | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> dict[str, int]:
    """Ingest all documents in the corpus directory."""
    url = db_url or settings.database_url.replace("+psycopg", "")
    all_files = list(docs_dir.rglob("*.md")) + list(docs_dir.rglob("*.txt"))

    if not all_files:
        logger.warning("No documents found in %s", docs_dir)
        return {}

    results: dict[str, int] = {}

    async with await psycopg.AsyncConnection.connect(url) as conn:
        for file_path in all_files:
            try:
                n = await ingest_file(conn, file_path, chunk_size, chunk_overlap)
                results[file_path.name] = n
            except Exception:
                logger.exception("Failed to ingest %s", file_path)
                results[file_path.name] = -1

    return results


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Ingest documents into pgvector store")
    parser.add_argument("--docs-dir", default="data/docs", help="Document corpus directory")
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--chunk-overlap", type=int, default=None)
    args = parser.parse_args()

    results = asyncio.run(
        ingest_corpus(Path(args.docs_dir), chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    )
    sum(v for v in results.values() if v >= 0)
    sum(1 for v in results.values() if v < 0)


if __name__ == "__main__":
    main()
