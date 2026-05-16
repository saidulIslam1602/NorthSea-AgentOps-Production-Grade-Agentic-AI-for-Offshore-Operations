"""
Hybrid retrieval: semantic (pgvector cosine) + BM25 lexical, fused via RRF.

Returns results with:
  - relevance_score  : fused rank score (0–1)
  - source_coverage  : fraction of query terms covered by top-k context
  - weak_evidence    : True when coverage < RAG_SIMILARITY_THRESHOLD
"""

from __future__ import annotations

import logging
from typing import Any

import psycopg

from src.config import get_settings
from src.rag.vectorstore import semantic_search
from src.schemas.domain import Citation

logger = logging.getLogger(__name__)
settings = get_settings()


def _tokenise(text: str) -> list[str]:
    return text.lower().split()


def _rrf_score(rank: int, k: int = 60) -> float:
    """Reciprocal Rank Fusion score for a result at position `rank` (1-indexed)."""
    return 1.0 / (k + rank)


async def _bm25_search(
    conn: psycopg.AsyncConnection[Any],
    query: str,
    top_k: int,
) -> list[dict[str, Any]]:
    """Lexical BM25 retrieval using PostgreSQL full-text search as a proxy."""
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
                ts_rank_cd(to_tsvector('english', dc.content),
                            plainto_tsquery('english', %s)) AS bm25_score
            FROM document_chunks dc
            JOIN documents d ON d.document_id = dc.document_id
            WHERE to_tsvector('english', dc.content) @@ plainto_tsquery('english', %s)
            ORDER BY bm25_score DESC
            LIMIT %s
            """,
            (query, query, top_k * 2),
        )
        rows = await cur.fetchall()
    return [dict(r) for r in rows]


def _fuse_results(
    semantic: list[dict[str, Any]],
    lexical: list[dict[str, Any]],
    alpha: float,
) -> list[dict[str, Any]]:
    """
    Fuse semantic and lexical results via Reciprocal Rank Fusion (RRF).

    alpha: weight on semantic score. (1-alpha) on lexical.
    """
    scores: dict[str, float] = {}
    chunk_data: dict[str, dict[str, Any]] = {}

    for rank, result in enumerate(semantic, start=1):
        key = f"{result['document_id']}:{result['chunk_index']}"
        scores[key] = scores.get(key, 0) + alpha * _rrf_score(rank)
        chunk_data[key] = result

    for rank, result in enumerate(lexical, start=1):
        key = f"{result['document_id']}:{result['chunk_index']}"
        scores[key] = scores.get(key, 0) + (1 - alpha) * _rrf_score(rank)
        if key not in chunk_data:
            chunk_data[key] = result

    sorted_keys = sorted(scores, key=lambda k: scores[k], reverse=True)

    fused: list[dict[str, Any]] = []
    for key in sorted_keys:
        item = chunk_data[key].copy()
        item["relevance_score"] = min(1.0, scores[key] * 120)  # normalise RRF to ~0-1
        fused.append(item)

    return fused


def _compute_coverage(query: str, chunks: list[dict[str, Any]]) -> float:
    """Fraction of unique query terms found in the retrieved context."""
    query_terms = set(_tokenise(query))
    if not query_terms:
        return 0.0

    context_text = " ".join(c["content"] for c in chunks).lower()
    found = sum(1 for term in query_terms if term in context_text)
    return found / len(query_terms)


async def retrieve(
    conn: psycopg.AsyncConnection[Any],
    query: str,
    top_k: int | None = None,
    min_similarity: float | None = None,
    alpha: float | None = None,
) -> dict[str, Any]:
    """
    Hybrid retrieval returning citations, coverage score, and weak-evidence flag.

    Returns:
        {
            "citations": list[Citation],
            "source_coverage": float,
            "weak_evidence": bool,
            "raw_chunks": list[dict],
        }
    """
    k = top_k or settings.rag_top_k
    min_sim = min_similarity if min_similarity is not None else settings.rag_similarity_threshold
    weight = alpha if alpha is not None else settings.rag_hybrid_alpha

    semantic_results = await semantic_search(conn, query, top_k=k * 2, min_similarity=0.0)
    lexical_results = await _bm25_search(conn, query, top_k=k)

    if not semantic_results and not lexical_results:
        return {
            "citations": [],
            "source_coverage": 0.0,
            "weak_evidence": True,
            "raw_chunks": [],
        }

    fused = _fuse_results(semantic_results, lexical_results, alpha=weight)
    top_chunks = [c for c in fused if c["relevance_score"] >= min_sim][:k]

    if not top_chunks:
        top_chunks = fused[: min(3, len(fused))]

    coverage = _compute_coverage(query, top_chunks)
    weak_evidence = coverage < settings.rag_similarity_threshold

    citations: list[Citation] = []
    for chunk in top_chunks:
        citations.append(
            Citation(
                document_id=chunk["document_id"],
                document_title=chunk.get("title", "Unknown Document"),
                document_type=chunk.get("doc_type", "Document"),
                section=chunk.get("section"),
                page=chunk.get("page_number"),
                relevance_score=round(chunk["relevance_score"], 4),
                excerpt=chunk["content"][:300],
            )
        )

    return {
        "citations": citations,
        "source_coverage": round(coverage, 4),
        "weak_evidence": weak_evidence,
        "raw_chunks": top_chunks,
    }
