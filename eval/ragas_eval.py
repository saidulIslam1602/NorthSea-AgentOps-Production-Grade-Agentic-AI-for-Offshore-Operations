"""
RAGAS evaluation pipeline for NorthSea AgentOps RAG system.

``eval/golden_testset.json`` is normally **VOLVE-aligned**: rebuilt from real daily production via
``scripts/generate_volve_golden_testset.py`` when you run ``scripts/sync_real_volve_daily_rag.py``.
The archived synthetic scaffold lives at ``eval/golden_testset_legacy_synthetic.json`` (optional).

Evaluates:
  - Faithfulness: answers are grounded in retrieved context
  - Answer Relevancy: answers are relevant to the question
  - Context Precision: retrieved chunks are relevant (no noise)
  - Context Recall: retrieved chunks cover the expected answer

Integrates with MLflow for experiment tracking.
CI gate: fails if faithfulness < 0.80.

Throughput (optional): concurrent golden-set retrieval + OpenAI completions
(controlled by ``RAG_EVAL_GATHER_CONCURRENCY`` env or ``--gather-concurrency``),
and Ragas ``RunConfig`` workers via ``RAGAS_MAX_WORKERS`` / ``RAGAS_EVAL_TIMEOUT_SECONDS``.

Usage:
  python -m eval.ragas_eval [--threshold 0.80] [--experiment-name northsea-eval]
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

RAGAS_THRESHOLD = 0.80
EVAL_RESULTS_PATH = Path("eval/results/ragas_results.json")
PER_SAMPLE_PATH = Path("eval/results/ragas_per_sample.json")


def _local_mlflow_sqlite_uri() -> str:
    """File-backed tracking store under eval/.mlflow (works without a tracking server)."""
    root = Path(__file__).resolve().parent.parent
    db_path = (root / "eval" / ".mlflow" / "ragas_tracking.db").resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return "sqlite:///" + db_path.as_posix()


def _log_results_to_mlflow(
    mlflow: Any,
    tracking_uri_candidates: tuple[str, ...],
    experiment_name: str,
    results_dict: dict[str, Any],
    testset_path: str,
) -> None:
    logged = False
    for uri in tracking_uri_candidates:
        try:
            mlflow.set_tracking_uri(uri)
            mlflow.set_experiment(experiment_name)
            with mlflow.start_run(run_name="ragas-eval"):
                for metric, value in results_dict.items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(metric, value)
                mlflow.log_artifact(testset_path)
            logged = True
            if uri != tracking_uri_candidates[0]:
                logger.warning(
                    "MLflow logged to SQLite fallback — primary tracking URI was unreachable: %s",
                    tracking_uri_candidates[0],
                )
            break
        except Exception as e:
            logger.warning("MLflow logging failed for tracking URI %s: %s", uri, e)
    if not logged:
        logger.warning("MLflow unavailable — metrics saved only to %s", EVAL_RESULTS_PATH)


def _gather_concurrency_from_env(cli_value: int | None) -> int:
    if cli_value is not None:
        return max(1, min(64, cli_value))
    raw = os.environ.get("RAG_EVAL_GATHER_CONCURRENCY")
    if raw:
        return max(1, min(64, int(raw)))
    try:
        aff = os.sched_getaffinity(0)  # type: ignore[attr-defined]
        ncpu = len(aff) if aff else 4
    except (AttributeError, OSError, NotImplementedError):
        ncpu = 4
    return max(8, min(24, max(1, ncpu) * 4))


def _maybe_run_config():
    workers = max(2, min(32, int(os.environ.get("RAGAS_MAX_WORKERS", "16"))))
    timeout_sec = max(60, int(os.environ.get("RAGAS_EVAL_TIMEOUT_SECONDS", "240")))
    try:
        from ragas.run_config import RunConfig

        return RunConfig(max_workers=workers, timeout=timeout_sec)
    except ImportError:
        logger.warning("ragas.RunConfig not found — install a recent ragas release for tunable metric workers.")
        return None


def _rag_eval_completion_model() -> str:
    raw = os.environ.get("RAG_EVAL_COMPLETION_MODEL", "").strip()
    if raw:
        return raw
    from src.config import get_settings

    s = get_settings()
    if s.rag_completion_model:
        return s.rag_completion_model
    return s.openai_model


def _rag_eval_context_char_limit() -> int:
    raw = os.environ.get("RAG_EVAL_CONTEXT_CHARS", "").strip()
    if raw:
        try:
            return max(2000, min(100_000, int(raw)))
        except ValueError:
            pass
    return 12_000


async def _run_single_query(
    testcase: dict[str, Any],
    conn: Any,
    client: Any,
) -> dict[str, Any]:
    """Run a single RAG query and collect evaluation data."""
    from src.rag.retriever import retrieve

    query = testcase["question"]
    from src.config import get_settings as _rag_settings_fn

    _s = _rag_settings_fn()
    top_k_eff = max(16, _s.rag_top_k)

    result = await retrieve(conn, query, top_k=top_k_eff)

    # Include document title in each chunk so the LLM can attribute facts to specific wells/docs.
    def _fmt_chunk(c: dict) -> str:
        title = (c.get("title") or "").strip()
        body = (c.get("content") or "").strip()
        return f"[Source: {title}]\n{body}" if title else body

    # IMPORTANT: use the same titled format for both the LLM prompt and Ragas contexts
    # so faithfulness scoring checks claims against text the LLM actually saw.
    retrieved_contexts = [_fmt_chunk(c) for c in result["raw_chunks"]]
    answer_context = "\n\n---\n\n".join(retrieved_contexts) if retrieved_contexts else ""
    clip = _rag_eval_context_char_limit()

    completion_model = _rag_eval_completion_model()
    response = await client.chat.completions.create(
        model=completion_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a disciplined production-engineering assistant. "
                    "Each context block begins with [Source: <document name>] so you can attribute facts correctly. "
                    "Answer using ONLY facts that appear explicitly in the Context. "
                    "Copy exact numbers, dates, percentages, and well identifiers verbatim - do not round or rephrase. "
                    "For questions about a specific well, only report values from that well's source blocks. "
                    "For field-wide questions (no specific well named), summarise from all relevant source blocks. "
                    "If the specific fact requested is not present in the Context, reply with exactly: "
                    "Information not available in the knowledge base."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{answer_context[:clip]}\n\nQuestion: {query}",
            },
        ],
        temperature=0,
    )
    answer = response.choices[0].message.content or ""

    return {
        "id": testcase["id"],
        "question": query,
        "answer": answer,
        "contexts": retrieved_contexts,
        "ground_truth": testcase["ground_truth"],
        "source_coverage": result["source_coverage"],
        "weak_evidence": result["weak_evidence"],
    }


async def _gather_eval_rows(
    testset: list[dict[str, Any]],
    db_url: str,
    api_key: str,
    concurrency: int,
) -> list[dict[str, Any]]:
    """Concurrent retrieve + completion for each testcase (each task opens its own DB connection)."""
    import psycopg
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=api_key)
    sem = asyncio.Semaphore(concurrency)

    async def run_one(tc: dict[str, Any]) -> dict[str, Any] | None:
        async with sem:
            try:
                async with await psycopg.AsyncConnection.connect(db_url) as conn:
                    return await _run_single_query(tc, conn, client)
            except Exception:
                logger.exception("Failed to evaluate %s", tc.get("id"))
                return None

    merged = await asyncio.gather(*(run_one(tc) for tc in testset))
    return [row for row in merged if row is not None]


async def run_ragas_evaluation(
    testset_path: str = "eval/golden_testset.json",
    db_url: str | None = None,
    openai_api_key: str | None = None,
    sample_size: int | None = None,
    mlflow_experiment: str = "northsea-agentops-eval",
    gather_concurrency: int | None = None,
) -> dict[str, Any]:
    """Run the full RAGAS evaluation pipeline."""
    import mlflow
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    from src.config import get_settings

    settings = get_settings()
    api_key = openai_api_key or settings.openai_api_key.get_secret_value()
    url_raw = db_url or settings.database_url.replace("+psycopg", "")

    testset = json.loads(Path(testset_path).read_text())
    if sample_size:
        testset = testset[:sample_size]

    conc = max(1, _gather_concurrency_from_env(gather_concurrency))
    logger.info("Running RAGAS evaluation on %d test cases (gather concurrency=%d)", len(testset), conc)

    eval_data = await _gather_eval_rows(testset, url_raw, api_key, conc)

    if not eval_data:
        logger.error("No evaluation data collected — check database and API key")
        return {"error": "No evaluation data"}

    dataset = Dataset.from_list(
        [
            {
                "question": r["question"],
                "answer": r["answer"],
                "contexts": r["contexts"],
                "ground_truth": r["ground_truth"],
            }
            for r in eval_data
        ]
    )

    # Ragas constructs default LangChain ChatOpenAI from OPENAI_API_KEY; settings may omit it from os.environ.
    if not (os.environ.get("OPENAI_API_KEY") or "").strip():
        os.environ["OPENAI_API_KEY"] = api_key

    eval_kwargs: dict[str, Any] = {
        "dataset": dataset,
        "metrics": [faithfulness, answer_relevancy, context_precision, context_recall],
        "raise_exceptions": False,
    }
    metric_model = os.environ.get("RAGAS_METRIC_MODEL", "").strip() or settings.openai_model
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    eval_kwargs["llm"] = ChatOpenAI(model=metric_model, temperature=0, api_key=api_key)
    eval_kwargs["embeddings"] = OpenAIEmbeddings(model=settings.openai_embedding_model, api_key=api_key)

    logger.info(
        "Ragas metric/generation backends: ChatOpenAI model=%s, embeddings=%s",
        metric_model,
        settings.openai_embedding_model,
    )

    rc = _maybe_run_config()
    if rc is not None:
        eval_kwargs["run_config"] = rc

    scores = evaluate(**eval_kwargs)

    score_df = scores.to_pandas()
    numeric_means = score_df.mean(numeric_only=True)
    results_dict: dict[str, Any] = {k: float(v) for k, v in numeric_means.dropna().items()}
    results_dict["num_test_cases"] = len(eval_data)
    results_dict["avg_source_coverage"] = sum(r["source_coverage"] for r in eval_data) / len(eval_data)
    results_dict["weak_evidence_rate"] = sum(1 for r in eval_data if r["weak_evidence"]) / len(eval_data)

    fallback_sqlite = _local_mlflow_sqlite_uri()
    primary_uri = settings.mlflow_tracking_uri
    mlflow_uris = (primary_uri, fallback_sqlite) if primary_uri != fallback_sqlite else (fallback_sqlite,)
    _log_results_to_mlflow(mlflow, mlflow_uris, mlflow_experiment, results_dict, testset_path)

    # Save aggregate results
    EVAL_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVAL_RESULTS_PATH.write_text(json.dumps(results_dict, indent=2))

    # Save per-sample scores for diagnosis
    metric_cols = [
        c for c in score_df.columns if c in ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
    ]
    per_sample: list[dict[str, Any]] = []
    for i, row_data in enumerate(eval_data):
        entry: dict[str, Any] = {
            "id": row_data.get("id", f"row-{i}"),
            "question": row_data["question"],
            "answer": row_data["answer"],
            "ground_truth": row_data["ground_truth"],
            "source_coverage": row_data["source_coverage"],
            "weak_evidence": row_data["weak_evidence"],
        }
        if i < len(score_df):
            for col in metric_cols:
                val = score_df.iloc[i][col]
                entry[col] = float(val) if val == val else None  # NaN → None
        per_sample.append(entry)
    PER_SAMPLE_PATH.write_text(json.dumps(per_sample, indent=2))
    logger.info("Per-sample Ragas scores → %s", PER_SAMPLE_PATH)

    logger.info("RAGAS results: %s", json.dumps(results_dict, indent=2))
    return results_dict


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run RAGAS evaluation")
    parser.add_argument("--threshold", type=float, default=RAGAS_THRESHOLD)
    parser.add_argument("--testset", default="eval/golden_testset.json")
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--experiment", default="northsea-agentops-eval")
    parser.add_argument(
        "--gather-concurrency",
        type=int,
        default=None,
        metavar="N",
        help="Concurrent DB retrieve + completions (defaults from RAG_EVAL_GATHER_CONCURRENCY or CPU heuristic).",
    )
    parser.add_argument("--fail-on-threshold", action="store_true", default=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    results = asyncio.run(
        run_ragas_evaluation(
            testset_path=args.testset,
            sample_size=args.sample,
            mlflow_experiment=args.experiment,
            gather_concurrency=args.gather_concurrency,
        )
    )

    faithfulness_score = results.get("faithfulness", 0.0)
    print(f"\n{'=' * 60}")
    print("RAGAS Evaluation Results")
    print(f"{'=' * 60}")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print(f"{'=' * 60}")
    print(f"Faithfulness: {faithfulness_score:.4f} (threshold: {args.threshold})")

    if args.fail_on_threshold and faithfulness_score < args.threshold:
        print(f"\n❌ EVAL GATE FAILED: faithfulness {faithfulness_score:.4f} < {args.threshold}")
        sys.exit(1)
    else:
        print(f"\n✅ EVAL GATE PASSED: faithfulness {faithfulness_score:.4f} >= {args.threshold}")


if __name__ == "__main__":
    main()
