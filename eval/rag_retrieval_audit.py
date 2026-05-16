"""
Retrieval-oriented confusion-style metrics vs ``golden_testset.json`` sources.

Typically that JSON is VOLVE-aligned (built from Excel). Optionally pass ``--testset``.

Query-level (+ ``weak_evidence`` as operational gate):

  - **TP**: At least one top-k chunk is from an expected ``document_sources`` file (basename match).
  - **FN**: No retrieved chunk matched any listed gold source.

Chunk-level:

  - **tp_chunk**: Chunk whose document basename matches that testcase gold list.
  - **fp_chunk**: Chunk from a doc not in gold list.

**TN** at query-level is ambiguous for positives-only corpuses.

An optional abstention framing uses ``weak_evidence``: see ``cross_tab_retrieval_hit_vs_weak_gate``.

Usage:
  python -m eval.rag_retrieval_audit [--testset path] [--out path]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_TESTSET = Path("eval/golden_testset.json")
DEFAULT_OUT = Path("eval/results/rag_retrieval_confusion.json")


def _basename_key(path_like: str) -> str:
    return Path(path_like.strip()).name.lower()


def _expected_set(case: dict[str, Any]) -> set[str]:
    return {_basename_key(x) for x in case.get("document_sources") or []}


async def _source_by_document_id(conn: Any, ids: frozenset[str]) -> dict[str, str]:
    if not ids:
        return {}
    import psycopg

    q = """
        SELECT document_id::text AS document_id, source_path
        FROM documents
        WHERE document_id::text = ANY(%s)
    """
    async with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
        await cur.execute(q, (list(ids),))
        rows = await cur.fetchall()
    return {r["document_id"]: r["source_path"] for r in rows}


def _chunk_matches_expected(chunk_doc_id: str, source_paths: dict[str, str], expected: set[str]) -> bool:
    sp = source_paths.get(chunk_doc_id)
    if not sp:
        return False
    b = _basename_key(sp)
    return b in expected or any(b.endswith(e) or e.endswith(b) for e in expected)


async def audit_retrieval(testset_path: Path) -> dict[str, Any]:
    import psycopg

    from src.config import get_settings
    from src.rag.retriever import retrieve

    settings = get_settings()
    db_url = settings.database_url.replace("+psycopg", "")
    testcases: list[dict[str, Any]] = json.loads(testset_path.read_text())

    query_tp = 0
    query_fn = 0
    tp_chunk = 0
    fp_chunk = 0
    weak_when_fn = 0
    hits_when_weak = 0
    rows_out: list[dict[str, Any]] = []

    async with await psycopg.AsyncConnection.connect(db_url) as conn:
        for tc in testcases:
            qid = tc.get("id", "?")
            question = tc["question"]
            expected = _expected_set(tc)

            top_k_eff = max(16, settings.rag_top_k)
            result = await retrieve(conn, question, top_k=top_k_eff)
            raw = result["raw_chunks"]
            doc_ids = frozenset({str(c["document_id"]) for c in raw})
            paths = await _source_by_document_id(conn, doc_ids)

            hit = False
            for c in raw:
                cid = str(c["document_id"])
                if expected and _chunk_matches_expected(cid, paths, expected):
                    hit = True
                    tp_chunk += 1
                else:
                    fp_chunk += 1

            if expected:
                if hit:
                    query_tp += 1
                    if result["weak_evidence"]:
                        hits_when_weak += 1
                else:
                    query_fn += 1
                    if result["weak_evidence"]:
                        weak_when_fn += 1

            rows_out.append(
                {
                    "id": qid,
                    "document_hit_at_least_once": hit,
                    "chunks_returned": len(raw),
                    "weak_evidence": result["weak_evidence"],
                    "source_coverage": result["source_coverage"],
                }
            )

    denom_q = query_tp + query_fn
    denom_c = tp_chunk + fp_chunk
    miss_not_weak = sum(
        1 for r in rows_out if not r["document_hit_at_least_once"] and not r["weak_evidence"]
    )
    miss_weak = sum(1 for r in rows_out if not r["document_hit_at_least_once"] and r["weak_evidence"])
    hit_not_weak = sum(1 for r in rows_out if r["document_hit_at_least_once"] and not r["weak_evidence"])
    hit_weak = sum(1 for r in rows_out if r["document_hit_at_least_once"] and r["weak_evidence"])

    out: dict[str, Any] = {
        "golden_path": str(testset_path.resolve()),
        "num_queries_with_gold_sources": denom_q,
        "query_true_positive_hit": query_tp,
        "query_false_negative_miss": query_fn,
        "retrieval_recall": (query_tp / denom_q) if denom_q else 0.0,
        "chunk_true_positive_supporting_chunks": tp_chunk,
        "chunk_false_positive_other_chunks": fp_chunk,
        "chunk_precision": (tp_chunk / denom_c) if denom_c else 0.0,
        "hits_marked_weak_evidence": hits_when_weak,
        "misses_marked_weak_evidence": weak_when_fn,
        "cross_tab_retrieval_hit_vs_weak_gate": {
            "hit_and_weak_evidence": hit_weak,
            "hit_and_not_weak": hit_not_weak,
            "miss_and_weak_evidence": miss_weak,
            "miss_and_not_weak": miss_not_weak,
        },
        "interpretation_miss_not_weak_FP_style": (
            "miss_and_not_weak = retrieval missed all gold filenames but gate did NOT flag weak evidence "
            "(mis-calibrated optimism). miss_and_weak = consistent miss + weak gate (honest skepticism)."
        ),
        "per_query": rows_out,
    }
    # TN / FP for strict binary “no gold present” unavailable on this corpus (all positives).
    out["definitions"] = (
        "query TP = ≥1 retrieved chunk basename matches testcase document_sources; "
        "FN = zero matches. chunk TP/FP classify each returned chunk vs that testcase sources. "
        "Traditional TN (true absence of positives) requires negative queries not in golden_set. "
        "Use cross-tab rows as proxy: miss+weak behaves like guarded negative outcome; "
        "miss+not_weak is analogous to FP on the operational weak_evidence gate when gold existed."
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG retrieval audit vs golden document_sources")
    parser.add_argument("--testset", type=Path, default=DEFAULT_TESTSET)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    results = asyncio.run(audit_retrieval(args.testset))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))

    qp, qn = results["query_true_positive_hit"], results["query_false_negative_miss"]
    tc, fc = (
        results["chunk_true_positive_supporting_chunks"],
        results["chunk_false_positive_other_chunks"],
    )
    print("\n=== RAG retrieval vs golden sources ===")
    print(f"Queries hit (TP):          {qp}")
    print(f"Queries miss (FN):         {qn}")
    print(f"Retrieval recall:          {results['retrieval_recall']:.4f}")
    print(f"Chunk TP / FP (pooled):    {tc} / {fc}")
    print(f"Chunk precision (chunks): {results['chunk_precision']:.4f}")
    print(f"Written: {args.out.resolve()}")


if __name__ == "__main__":
    main()
