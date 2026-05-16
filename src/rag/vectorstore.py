"""pgvector interface — stores and retrieves document chunk embeddings."""

from __future__ import annotations

import hashlib
import logging
from typing import Any
from uuid import uuid4

import psycopg
from openai import AsyncOpenAI
from psycopg.types.json import Json

from src.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

_openai_client: AsyncOpenAI | None = None


def _get_openai() -> AsyncOpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = AsyncOpenAI(api_key=settings.openai_api_key.get_secret_value())
    return _openai_client


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts using OpenAI embedding model."""
    client = _get_openai()
    response = await client.embeddings.create(
        model=settings.openai_embedding_model,
        input=texts,
    )
    return [item.embedding for item in response.data]


def _chunks_for_embedding(
    chunks: list[dict[str, Any]],
    *,
    document_title: str | None,
    well_id: str | None,
    field_name: str | None,
) -> list[str]:
    """Prefix corpus text seen by embedding so vectors align with retrieval queries mentioning wells/fields."""
    header_lines: list[str] = []
    if document_title:
        header_lines.append(f"Document: {document_title}")
    if field_name:
        header_lines.append(f"Field: {field_name}")
    if well_id:
        header_lines.append(f"Well: {well_id}")
    prefix = "\n".join(header_lines)
    sep = "\n\n" if prefix else ""
    return [f"{prefix}{sep}{c['content']}" for c in chunks]


async def upsert_chunks(
    conn: psycopg.AsyncConnection[Any],
    document_id: str,
    chunks: list[dict[str, Any]],
    *,
    document_title: str | None = None,
    well_id: str | None = None,
    field_name: str | None = None,
) -> None:
    """Upsert document chunks with embeddings into pgvector store (chunk text unchanged; embeddings enriched)."""
    texts = _chunks_for_embedding(
        chunks,
        document_title=document_title,
        well_id=well_id,
        field_name=field_name,
    )
    embeddings = await embed_texts(texts)

    async with conn.cursor() as cur:
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            await cur.execute(
                """
                INSERT INTO document_chunks
                    (id, document_id, chunk_index, content, section, page_number,
                     embedding, token_count)
                VALUES (%s, %s, %s, %s, %s, %s, %s::vector, %s)
                ON CONFLICT (document_id, chunk_index) DO UPDATE
                    SET content = EXCLUDED.content,
                        embedding = EXCLUDED.embedding,
                        section = EXCLUDED.section,
                        page_number = EXCLUDED.page_number,
                        token_count = EXCLUDED.token_count
                """,
                (
                    str(uuid4()),
                    document_id,
                    chunk["chunk_index"],
                    chunk["content"],
                    chunk.get("section"),
                    chunk.get("page_number"),
                    str(embedding),
                    chunk.get("token_count", len(chunk["content"].split())),
                ),
            )


async def semantic_search(
    conn: psycopg.AsyncConnection[Any],
    query: str,
    top_k: int | None = None,
    min_similarity: float | None = None,
) -> list[dict[str, Any]]:
    """Retrieve top-k chunks by cosine similarity to query embedding."""
    k = top_k or settings.rag_top_k
    min_sim = min_similarity if min_similarity is not None else settings.rag_similarity_threshold

    query_embedding = (await embed_texts([query]))[0]
    embedding_str = str(query_embedding)

    async with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
        await cur.execute(
            """
            SELECT
                dc.document_id,
                dc.chunk_index,
                dc.content,
                dc.section,
                dc.page_number,
                d.title,
                d.doc_type,
                d.well_id,
                1 - (dc.embedding <=> %s::vector) AS similarity
            FROM document_chunks dc
            JOIN documents d ON d.document_id = dc.document_id
            WHERE 1 - (dc.embedding <=> %s::vector) >= %s
            ORDER BY dc.embedding <=> %s::vector
            LIMIT %s
            """,
            (embedding_str, embedding_str, min_sim, embedding_str, k),
        )
        rows = await cur.fetchall()

    return [dict(row) for row in rows]


async def register_document(
    conn: psycopg.AsyncConnection[Any],
    document_id: str,
    title: str,
    doc_type: str,
    source_path: str,
    well_id: str | None = None,
    field_name: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Register a document in the documents table."""
    async with conn.cursor() as cur:
        await cur.execute(
            """
            INSERT INTO documents (document_id, title, doc_type, well_id, field_name,
                                   source_path, metadata)
            VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb)
            ON CONFLICT (document_id) DO UPDATE
                SET title = EXCLUDED.title,
                    doc_type = EXCLUDED.doc_type,
                    well_id = EXCLUDED.well_id,
                    field_name = EXCLUDED.field_name,
                    source_path = EXCLUDED.source_path,
                    metadata = EXCLUDED.metadata
            """,
            (
                document_id,
                title,
                doc_type,
                well_id,
                field_name,
                source_path,
                Json(metadata if metadata is not None else {}),
            ),
        )


async def clear_all(conn: psycopg.AsyncConnection[Any]) -> tuple[int, int]:
    """Delete all documents and their chunks from the vector store.

    Returns (documents_deleted, chunks_deleted) for confirmation output.
    Chunks are removed via CASCADE from the documents FK constraint.
    """
    async with conn.cursor() as cur:
        await cur.execute("SELECT COUNT(*) FROM document_chunks")
        chunks_row = await cur.fetchone()
        chunks_count: int = chunks_row[0] if chunks_row else 0

        await cur.execute("SELECT COUNT(*) FROM documents")
        docs_row = await cur.fetchone()
        docs_count: int = docs_row[0] if docs_row else 0

        # Chunks are deleted automatically via ON DELETE CASCADE.
        await cur.execute("DELETE FROM documents")
        await conn.commit()

    return docs_count, chunks_count


def stable_document_id(path: str) -> str:
    """Deterministic document ID from file path."""
    return hashlib.sha256(path.encode()).hexdigest()[:16]
