# ADR-003: Evaluation Stack — RAGAS + MLflow + Custom Agent Eval

**Status:** Accepted  
**Date:** 2024-Q4  

---

## Context

"Evaluation-driven development" is a first-class requirement at Aker BP (Track A).  
The agent produces two kinds of outputs that need independent evaluation:

1. **RAG retrieval quality** — Are the retrieved documents relevant? Is the answer faithful to the sources?
2. **Agent behaviour quality** — Does the Planner pick the right tools? Does the UncertaintyGate escalate correctly? Is the Critic's confidence calibrated?

No single framework covers both. A composite evaluation stack is needed.

---

## Decision

Use a **three-layer evaluation stack**:
1. **RAGAS** for retrieval and answer quality (LLM-as-judge on retrieved context)
2. **Custom agent eval** (`eval/agent_eval.py`) for plan quality, tool selection, and escalation calibration — deterministic, no LLM required in CI
3. **MLflow** for experiment tracking, metric history, and regression detection

---

## Layer 1: RAGAS (Retrieval Augmented Generation Assessment)

| Metric | What it measures | Why it matters |
|--------|-----------------|----------------|
| `answer_faithfulness` | Does the answer only use retrieved context? | Prevents hallucination of non-existent procedures |
| `answer_relevancy` | Is the answer on-topic for the question? | Catches generic answers that miss the specific anomaly |
| `context_precision` | Are retrieved chunks all relevant? | Measures retrieval precision — too many irrelevant chunks dilute focus |
| `context_recall` | Did we retrieve all necessary information? | Measures retrieval completeness — missing key docs causes wrong root cause |

RAGAS uses an LLM-as-judge — requires `OPENAI_API_KEY` in CI. Runs in the separate `eval-gate.yml` workflow (not the main CI) to avoid blocking every PR.

**Threshold**: `answer_faithfulness > 0.85`, `context_recall > 0.75`

## Layer 2: Custom Agent Evaluation (no LLM, runs in every CI)

| Metric | Implementation | Threshold |
|--------|---------------|-----------|
| Plan quality score | Compare generated tools against expected tools per anomaly type | ≥ 0.70 |
| Escalation recall | UncertaintyGate vs ground truth escalation labels | ≥ 0.85 (safety) |
| Escalation precision | Avoid unnecessary escalation (operator fatigue) | ≥ 0.70 |
| ECE (Expected Calibration Error) | Critic confidence vs actual correctness | ≤ 0.15 |
| Brier score | Probabilistic accuracy of confidence scores | ≤ 0.25 |

**Escalation recall is weighted higher than precision** because a missed escalation in offshore ops is a safety event. A false alarm (unnecessary escalation) costs operator time; a miss can cost lives.

## Layer 3: MLflow Experiment Tracking

- All RAGAS and agent eval metrics logged to MLflow per CI run
- `mlflow.set_experiment("northsea-agentops-eval")` groups runs by experiment
- Regression detection: if any metric degrades >5% from baseline, the eval gate fails
- Artifacts: calibration_report.json, agent_eval_results.json, RAGAS dataset

**Why MLflow over W&B / Neptune / Comet?**
- Self-hostable (regulatory environments prefer on-prem tooling)
- Docker Compose includes MLflow service (no additional cloud account needed)
- Integrates with Azure ML / Databricks if needed for scale
- Free and open source

---

## Consequences

- Every PR runs the deterministic agent eval (plan quality, ECE, escalation) — fast, no API key
- RAGAS runs only on main branch merges and scheduled eval gates — requires OpenAI key
- MLflow provides experiment history for regression detection
- The eval stack is self-contained — no third-party SaaS needed for core quality gates
