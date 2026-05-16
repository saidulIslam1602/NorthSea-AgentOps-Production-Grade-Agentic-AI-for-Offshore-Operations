"""
Tool: retrieve_documents

Executes the hybrid RAG retrieval pipeline and returns formatted context
with citations for the Executor agent.
"""

from __future__ import annotations

import logging
from typing import Any

import psycopg

from src.config import get_settings
from src.rag.retriever import retrieve
from src.schemas.domain import Citation

logger = logging.getLogger(__name__)
settings = get_settings()


async def retrieve_documents(
    conn: psycopg.AsyncConnection[Any],
    query: str,
    top_k: int | None = None,
) -> dict[str, Any]:
    """
    Retrieve relevant documents using hybrid BM25 + semantic search.

    Returns:
        {
            "query": str,
            "context": str (formatted text for LLM injection),
            "citations": list[Citation],
            "source_coverage": float,
            "weak_evidence": bool,
            "num_chunks": int,
        }
    """
    result = await retrieve(conn, query, top_k=top_k)
    citations: list[Citation] = result["citations"]
    chunks: list[dict[str, Any]] = result["raw_chunks"]

    context_parts: list[str] = []
    for i, (citation, chunk) in enumerate(zip(citations, chunks, strict=False), start=1):
        context_parts.append(
            f"[Source {i}: {citation.document_title} "
            f"({citation.document_type}"
            f"{', ' + citation.section if citation.section else ''}"
            f")] (relevance: {citation.relevance_score:.2f})\n"
            f"{chunk['content']}"
        )

    context = "\n\n---\n\n".join(context_parts) if context_parts else "No relevant documents found."

    if result["weak_evidence"]:
        context += (
            "\n\nNOTE: Retrieval coverage for substantive query terms is below the configured "
            "evidence gate. Prefer confirming against primary sources or human review before acting."
        )

    return {
        "query": query,
        "context": context,
        "citations": citations,
        "source_coverage": result["source_coverage"],
        "weak_evidence": result["weak_evidence"],
        "num_chunks": len(chunks),
    }


async def query_similar_incidents(
    conn: psycopg.AsyncConnection[Any],
    anomaly_description: str,
    affected_features: list[str],
) -> dict[str, Any]:
    """Search for past similar incidents in the knowledge base."""
    # Build a targeted query combining anomaly features
    feature_text = ", ".join(affected_features)
    query = f"production anomaly incident {feature_text}: {anomaly_description}"

    result = await retrieve_documents(conn, query, top_k=5)

    # Filter specifically for incident reports and well performance reports
    incident_citations = [
        c
        for c in result["citations"]
        if any(
            keyword in (c.document_title + c.document_type).lower()
            for keyword in ["incident", "investigation", "failure", "upset", "anomaly"]
        )
    ]

    result["incident_citations"] = incident_citations
    result["query"] = query
    return result
