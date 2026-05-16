"""
RAGAS evaluation pipeline for NorthSea AgentOps RAG system.

Evaluates:
  - Faithfulness: answers are grounded in retrieved context
  - Answer Relevancy: answers are relevant to the question
  - Context Precision: retrieved chunks are relevant (no noise)
  - Context Recall: retrieved chunks cover the expected answer

Integrates with MLflow for experiment tracking.
CI gate: fails if faithfulness < 0.80.

Usage:
  python -m eval.ragas_eval [--threshold 0.80] [--experiment-name northsea-eval]
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

RAGAS_THRESHOLD = 0.80
EVAL_RESULTS_PATH = Path("eval/results/ragas_results.json")


async def _run_single_query(
    testcase: dict[str, Any],
    conn: Any,
    openai_api_key: str,
) -> dict[str, Any]:
    """Run a single RAG query and collect evaluation data."""
    from src.rag.retriever import retrieve

    query = testcase["question"]
    result = await retrieve(conn, query)

    retrieved_contexts = [c["content"] for c in result["raw_chunks"]]
    answer_context = "\n\n".join(retrieved_contexts) if retrieved_contexts else ""

    # Generate an answer using the retrieved context
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=openai_api_key)
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a production engineer AI assistant. Answer the question based strictly on the provided context. If the context doesn't contain the answer, say 'Information not available in the knowledge base.'",
            },
            {
                "role": "user",
                "content": f"Context:\n{answer_context[:3000]}\n\nQuestion: {query}",
            },
        ],
        temperature=0.1,
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


async def run_ragas_evaluation(
    testset_path: str = "eval/golden_testset.json",
    db_url: str | None = None,
    openai_api_key: str | None = None,
    sample_size: int | None = None,
    mlflow_experiment: str = "northsea-agentops-eval",
) -> dict[str, Any]:
    """Run the full RAGAS evaluation pipeline."""
    import mlflow
    import psycopg
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
    url = db_url or settings.database_url.replace("+psycopg", "")

    testset = json.loads(Path(testset_path).read_text())
    if sample_size:
        testset = testset[:sample_size]

    logger.info("Running RAGAS evaluation on %d test cases", len(testset))

    # Collect evaluation data
    eval_data: list[dict[str, Any]] = []
    async with await psycopg.AsyncConnection.connect(url) as conn:
        for tc in testset:
            try:
                row = await _run_single_query(tc, conn, api_key)
                eval_data.append(row)
            except Exception:
                logger.exception("Failed to evaluate %s", tc["id"])

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

    # Run RAGAS
    scores = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=None,  # uses default OpenAI
        raise_exceptions=False,
    )

    results_dict = scores.to_pandas().mean().to_dict()
    results_dict["num_test_cases"] = len(eval_data)
    results_dict["avg_source_coverage"] = sum(r["source_coverage"] for r in eval_data) / len(eval_data)
    results_dict["weak_evidence_rate"] = sum(1 for r in eval_data if r["weak_evidence"]) / len(eval_data)

    # Log to MLflow
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(mlflow_experiment)

    with mlflow.start_run(run_name="ragas-eval"):
        for metric, value in results_dict.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(metric, value)
        mlflow.log_artifact(testset_path)

    # Save results
    EVAL_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVAL_RESULTS_PATH.write_text(json.dumps(results_dict, indent=2))

    logger.info("RAGAS results: %s", json.dumps(results_dict, indent=2))
    return results_dict


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run RAGAS evaluation")
    parser.add_argument("--threshold", type=float, default=RAGAS_THRESHOLD)
    parser.add_argument("--testset", default="eval/golden_testset.json")
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--experiment", default="northsea-agentops-eval")
    parser.add_argument("--fail-on-threshold", action="store_true", default=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    results = asyncio.run(
        run_ragas_evaluation(
            testset_path=args.testset,
            sample_size=args.sample,
            mlflow_experiment=args.experiment,
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
