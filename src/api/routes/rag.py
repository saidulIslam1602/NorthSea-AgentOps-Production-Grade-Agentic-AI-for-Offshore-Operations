"""RAG Copilot endpoints — answer operational questions from the knowledge base."""

from __future__ import annotations

import logging

import psycopg
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from src.config import get_settings
from src.rag.retriever import retrieve
from src.safety.injection_guard import check_user_query
from src.schemas.domain import Citation

router = APIRouter()
logger = logging.getLogger(__name__)
settings = get_settings()


class RAGQueryRequest(BaseModel):
    query: str = Field(..., min_length=5, max_length=500)
    top_k: int = Field(8, ge=1, le=20)
    well_id: str | None = None


class RAGQueryResponse(BaseModel):
    query: str
    answer_context: str
    citations: list[Citation]
    source_coverage: float
    weak_evidence: bool
    num_chunks_retrieved: int


@router.post(
    "/rag/query",
    response_model=RAGQueryResponse,
    summary="Query operational knowledge base",
    description=(
        "Retrieves relevant documents from the operational knowledge base using "
        "hybrid BM25 + semantic search. Returns context with citations and "
        "source coverage score."
    ),
)
async def query_knowledge_base(request: RAGQueryRequest) -> RAGQueryResponse:
    # Safety check
    inj_check = check_user_query(request.query)
    if not inj_check.is_clean and inj_check.severity == "CRITICAL":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query contains prohibited content.",
        )

    db_url = settings.database_url.replace("+psycopg", "")
    try:
        async with await psycopg.AsyncConnection.connect(db_url) as conn:
            result = await retrieve(conn, request.query, top_k=request.top_k)
    except Exception as exc:
        logger.exception("RAG query failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"RAG retrieval failed: {exc}",
        ) from exc

    return RAGQueryResponse(
        query=request.query,
        answer_context="\n\n".join(c["content"] for c in result["raw_chunks"])
        if result["raw_chunks"]
        else "No relevant content found.",
        citations=result["citations"],
        source_coverage=result["source_coverage"],
        weak_evidence=result["weak_evidence"],
        num_chunks_retrieved=len(result["raw_chunks"]),
    )
