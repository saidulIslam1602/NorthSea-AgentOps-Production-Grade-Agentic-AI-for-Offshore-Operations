"""
Hybrid retrieval: semantic (pgvector cosine) + BM25 lexical, fused via RRF.

Returns results with:
  - relevance_score  : fused rank score (0–1), from RRF — not cosine similarity
  - source_coverage  : stopword‑aware overlap of substantive query tokens with context
  - weak_evidence    : True when coverage is low unless strong fused-score relief applies (see Settings)

Note:
  ``rag_similarity_threshold`` applies to **cosine similarity** for standalone semantic_search.
  Fused retrieval uses ``rag_min_fused_score`` (same 0–1 scale as relevance_score).
"""

from __future__ import annotations

import logging
import re
from typing import Any

import psycopg

from src.config import get_settings
from src.rag.vectorstore import semantic_search
from src.schemas.domain import Citation

logger = logging.getLogger(__name__)
settings = get_settings()


# Short common English stopwords; keeps coverage aligned with substantive query terms for ops Q&A.
_QUERY_STOPWORDS = frozenset(
    """
    a an the and or but if as at by for from in into of on onto with to too do does did during
    how what when where which who whom whose why is are was were be been being
    it its this that these those than then there their they them we you your he she his her
    not no nor any each every both few more most other some such same so can could should
    would will just about over under again further once here all both each few most other
    my our your i me mine ours yours yourself yourselves himself herself itself themselves
""".split()
)

_CHUNK_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\-_/]*")


def _substantive_query_tokens(query: str) -> list[str]:
    """Normalized tokens excluding stopwords and very short fragments."""
    q = query.lower()
    tokens: list[str] = []
    for raw in q.split():
        for m in _CHUNK_TOKEN_RE.finditer(raw):
            t = m.group(0)
            if len(t) < 3 or t in _QUERY_STOPWORDS:
                continue
            tokens.append(t)
    seen: set[str] = set()
    out: list[str] = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _tokenise_legacy(query: str) -> list[str]:
    """Single-token split retained for behavioural compatibility where needed."""
    return query.lower().split()


def _rrf_score(rank: int, k: int = 60) -> float:
    """Reciprocal Rank Fusion score for a result at position `rank` (1-indexed)."""
    return 1.0 / (k + rank)


_CHUNK_KEYSEP = ":"


_OVERVIEW_CHUNK_SELECT = """
    SELECT
        dc.document_id,
        dc.chunk_index,
        dc.content,
        dc.section,
        dc.page_number,
        d.title,
        d.doc_type,
        d.well_id
    FROM document_chunks dc
    JOIN documents d ON d.document_id = dc.document_id
    WHERE d.source_path ILIKE %s
    ORDER BY dc.chunk_index ASC
    LIMIT %s
"""


async def _fetch_chunks_by_source_ilike(
    conn: psycopg.AsyncConnection[Any],
    pattern: str,
    limit: int,
) -> list[dict[str, Any]]:
    async with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
        await cur.execute(_OVERVIEW_CHUNK_SELECT, (pattern, limit))
        rows = await cur.fetchall()
    return [dict(r) for r in rows]


def _chunk_key(chunk: dict[str, Any]) -> str:
    return f"{chunk['document_id']}{_CHUNK_KEYSEP}{chunk['chunk_index']}"


def _field_aggregate_query(query: str) -> bool:
    """Workbook totals / enumeration cues (VOLVE-wide)."""
    ql = query.lower()
    wells = "wells" in ql or "well " in ql or ql.rstrip().endswith("well")
    how_many = "how many" in ql or "what number of" in ql
    workbookish = "workbook" in ql or "enumerat" in ql or "enumeration" in ql or "summar" in ql
    aggregates = "field overview" in ql or "pooled across all wells" in ql or "total producing" in ql
    return (how_many and wells) or (workbookish and wells) or aggregates or (
        workbookish and "volve" in ql and ("producer" in ql or wells)
    )


def _field_overview_injection_heuristic(query: str) -> bool:
    """True when VOLVE_Field_Overview should be front-loaded (counts, dates, peaks across wells)."""
    if _field_aggregate_query(query):
        return True
    ql = query.lower()
    cross_well_peak = ("which" in ql or "what" in ql or "whose" in ql) and ("well" in ql or "producer" in ql) and (
        "peak" in ql or "bopd" in ql or "oil rate" in ql or "daily oil" in ql
    )
    superlative_well = ("highest" in ql or "lowest" in ql or "maximum" in ql or "minimum" in ql or "max " in ql) and (
        "well" in ql or "producer" in ql
    ) and ("peak" in ql or "oil" in ql or "bopd" in ql)
    comparison = ("compare" in ql or "rank" in ql or "versus" in ql or " vs " in ql) and (
        "well" in ql or "producer" in ql
    )
    return cross_well_peak or superlative_well or comparison


def _prioritize_chunks_for_query(
    query: str,
    chunks: list[dict[str, Any]],
    *,
    inject_n: int,
    min_content_chars: int = 180,
) -> list[dict[str, Any]]:
    """Prefer overview sections that lexical-overlap substantive query terms (drops header fluff)."""
    if not chunks:
        return chunks
    longish = [c for c in chunks if len((c.get("content") or "").strip()) >= min_content_chars]
    pool = longish if len(longish) >= min(4, inject_n // 2) else chunks

    decorated: list[tuple[float, float, int, dict[str, Any]]] = []
    for c in pool:
        frac = _lexical_overlap_frac(query, c.get("content", ""))
        size = len(c.get("content") or "")
        decorated.append((-frac, -size, int(c["chunk_index"]), c))
    decorated.sort()
    return [c for _, _, _, c in decorated[:inject_n]]


def _prepend_field_overview(
    fused: list[dict[str, Any]],
    overview: list[dict[str, Any]],
    *,
    fused_score_lookup: dict[str, float],
    boost_floor: float,
) -> list[dict[str, Any]]:
    """Front-load field-overview chunks (lexical-strong) while preserving fused scores where they overlap."""
    if not overview:
        return fused
    seen: set[str] = set()
    front: list[dict[str, Any]] = []
    for row in overview:
        k = _chunk_key(row)
        if k in seen:
            continue
        chunk = dict(row)
        fused_rel = fused_score_lookup.get(k, 0.0)
        chunk["relevance_score"] = max(float(fused_rel), boost_floor)
        front.append(chunk)
        seen.add(k)
    tail: list[dict[str, Any]] = []
    for c in fused:
        k = _chunk_key(c)
        if k in seen:
            continue
        tail.append(c)
        seen.add(k)
    return front + tail


def _lexical_overlap_frac(query: str, content: str) -> float:
    terms = _substantive_query_tokens(query)
    if not terms:
        return 0.0
    lowered = content.lower()
    hits = sum(1 for t in terms if t in lowered)
    return hits / len(terms)


_COUNT_WELL_LINE = re.compile(r"\*\*\d+\*\*\s+wells?\b", re.IGNORECASE)
_PEAK_ROW = re.compile(r"peak\s+oil\s+[0-9,]+(?:\.\d+)?\s*BOPD", re.IGNORECASE)


def _field_scope_rerank_bonus(query: str, chunk: dict[str, Any]) -> float:
    """Push substantive VOLVE_Field_Overview sections above title-only chunks when injecting overview."""
    if not _field_overview_injection_heuristic(query):
        return 0.0
    title = str(chunk.get("title") or "").lower()
    blob = chunk.get("content") or ""
    if "production overview" not in title and "overview" not in title:
        return 0.0
    ql = query.lower()
    bonus = min(len(blob), 520) / 900.0
    if "how many" in ql and ("well" in ql or "wells" in ql):
        if _COUNT_WELL_LINE.search(blob.replace(" ", " ")):
            bonus += 0.95
        if "producer" in blob.lower() or "wells represented" in blob.lower():
            bonus += 0.35
    if ("peak" in ql or "bopd" in ql) and (
        "highest" in ql or "maximum" in ql or "max " in ql or "lowest" in ql or "minimum" in ql
    ):
        if len(_PEAK_ROW.findall(blob)) >= 2:
            bonus += 1.05
    return bonus


def _rerank_with_lexical_overlap(query: str, chunks: list[dict[str, Any]], w_overlap: float) -> list[dict[str, Any]]:
    if not chunks or w_overlap <= 0:
        return chunks
    decorated: list[tuple[float, int, dict[str, Any]]] = []
    for i, c in enumerate(chunks):
        frac = _lexical_overlap_frac(query, c.get("content", ""))
        score = float(c.get("relevance_score", 0.0)) + w_overlap * frac + _field_scope_rerank_bonus(query, c)
        decorated.append((-score, i, c))
    decorated.sort()
    return [c for _, _, c in decorated]


_BM25_SELECT = """
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
"""


async def _bm25_search_rows(
    conn: psycopg.AsyncConnection[Any],
    needle: str,
    limit: int,
) -> list[dict[str, Any]]:
    async with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
        await cur.execute(_BM25_SELECT, (needle, needle, limit))
        rows = await cur.fetchall()
    return [dict(r) for r in rows]


async def _bm25_search(
    conn: psycopg.AsyncConnection[Any],
    query: str,
    top_k: int,
) -> list[dict[str, Any]]:
    """Lexical BM25 retrieval using PostgreSQL FTS; retries with distilled terms if plain fails."""
    limit = max(top_k * 3, top_k + 24)
    rows = await _bm25_search_rows(conn, query, limit)
    if rows:
        return rows

    terms = _substantive_query_tokens(query)
    if len(terms) >= 2:
        fallback = " ".join(terms[:14])
        rows = await _bm25_search_rows(conn, fallback, limit)
        if rows:
            return rows

    if terms:
        # Last resort: match on the longest substantive token (equipment / well jargon survives plainto gaps).
        for term in sorted(terms, key=len, reverse=True):
            rows = await _bm25_search_rows(conn, term, limit)
            if rows:
                return rows
    return []


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
    """Fraction of substantive query tokens found verbatim in concatenated retrieved context."""
    terms = _substantive_query_tokens(query)
    if not terms:
        legacy = [_t for _t in _tokenise_legacy(query) if len(_t) >= 4 and _t not in _QUERY_STOPWORDS]
        if not legacy:
            return 1.0
        terms = legacy

    context_text = " ".join(c["content"] for c in chunks).lower()
    found = sum(1 for term in terms if term in context_text)
    return found / len(terms)


async def retrieve(
    conn: psycopg.AsyncConnection[Any],
    query: str,
    top_k: int | None = None,
    min_fused_relevance: float | None = None,
    alpha: float | None = None,
) -> dict[str, Any]:
    """
    Hybrid retrieval returning citations, coverage score, and weak-evidence flag.

    Args:
        min_fused_relevance: Override ``rag_min_fused_score`` floor on fused RRF scores (not cosine similarity).

    Returns:
        {
            "citations": list[Citation],
            "source_coverage": float,
            "weak_evidence": bool,
            "raw_chunks": list[dict],
        }
    """
    k = top_k or settings.rag_top_k
    min_fused = (
        settings.rag_min_fused_score if min_fused_relevance is None else min_fused_relevance
    )
    if alpha is not None:
        weight = alpha
    elif _field_overview_injection_heuristic(query):
        weight = float(settings.rag_field_query_bm25_alpha)
    else:
        weight = settings.rag_hybrid_alpha

    pool = max(k * 4, 36)
    semantic_results = await semantic_search(conn, query, top_k=pool, min_similarity=0.0)
    lexical_results = await _bm25_search(conn, query, top_k=pool)

    if not semantic_results and not lexical_results:
        return {
            "citations": [],
            "source_coverage": 0.0,
            "weak_evidence": True,
            "raw_chunks": [],
        }

    fused = _fuse_results(semantic_results, lexical_results, alpha=weight)

    if _field_overview_injection_heuristic(query):
        pool_rows = await _fetch_chunks_by_source_ilike(
            conn,
            "%VOLVE_Field_Overview%",
            max(settings.rag_field_overview_inject_chunks * 5, 40),
        )
        overview = _prioritize_chunks_for_query(
            query,
            pool_rows,
            inject_n=settings.rag_field_overview_inject_chunks,
        )
        fused_score_lookup = {_chunk_key(c): float(c["relevance_score"]) for c in fused}
        fused = _prepend_field_overview(
            fused,
            overview,
            fused_score_lookup=fused_score_lookup,
            boost_floor=0.92,
        )

    fused_filtered = [c for c in fused if c["relevance_score"] >= min_fused]
    candidate = fused_filtered if fused_filtered else fused
    candidate = _rerank_with_lexical_overlap(
        query,
        candidate,
        float(settings.rag_lexical_overlap_rerank_weight),
    )
    top_chunks = candidate[:k]

    if len(top_chunks) < min(k, 3) and fused:
        top_chunks = fused[: max(k, 3)]

    coverage = _compute_coverage(query, top_chunks)
    max_rel = max((float(c["relevance_score"]) for c in top_chunks), default=0.0)
    relief_ok = coverage >= settings.rag_evidence_relief_min_coverage and max_rel >= float(
        settings.rag_evidence_relief_min_relevance
    )
    weak_evidence = (coverage < settings.rag_coverage_weak_threshold) and not relief_ok

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
