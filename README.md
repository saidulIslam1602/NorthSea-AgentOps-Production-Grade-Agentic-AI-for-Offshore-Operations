# NorthSea AgentOps: Production-Grade Agentic AI for Offshore Operations

> A production-grade agentic AI system for offshore oil and gas operations. Combines time-series anomaly detection, RAG over operational documents, planner-executor-critic agents, confidence thresholds, and human-in-the-loop escalation to support safer engineering decisions in industrial environments.

**Full README coming soon.**

---

## Quick Start

```bash
cp .env.example .env          # add your OPENAI_API_KEY
docker compose up -d          # starts Postgres+pgvector, Kafka, Grafana, MLflow
python -m src.data.doc_generator          # generate synthetic RAG corpus
python -m src.rag.ingestion --docs-dir data/docs   # embed documents
python -m src.data.synthetic_generator    # generate SCADA telemetry
uvicorn src.api.main:app --reload         # start API (http://localhost:8000/docs)
```

## Architecture

```
Kafka (SCADA telemetry)
  → Anomaly Detector (Isolation Forest + Z-score)
    → LangGraph Orchestrator
      → Planner Agent   (decomposes investigation into steps)
      → Executor Agent  (calls tools: timeseries, RAG, diagnostics)
      → Critic Agent    (scores confidence, evidence quality, risk)
      → Uncertainty Gate (confidence < 0.75 or HIGH risk → escalate)
        → Human Escalation Queue  OR  Recommendation + Citations
          → FastAPI Backend
            → Grafana + OpenTelemetry observability
            → RAGAS + MLflow evaluation
```

## Tech Stack

| Layer | Technology |
|---|---|
| Agent framework | LangGraph |
| LLM | OpenAI GPT-4o / GPT-4o-mini |
| Vector store | PostgreSQL + pgvector |
| Event streaming | Kafka / Azure Event Hubs |
| Backend | FastAPI + Pydantic v2 |
| Evaluation | RAGAS + MLflow |
| Observability | OpenTelemetry + Prometheus + Grafana |
| Infra | Docker Compose · Kubernetes · Terraform (Azure) |
| CI/CD | GitHub Actions |
