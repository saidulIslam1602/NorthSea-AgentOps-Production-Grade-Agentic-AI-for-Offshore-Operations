# NorthSea AgentOps — Production‑Grade Agentic AI for Offshore Operations

This **README is the canonical self‑documentation** for the repository: business context, architecture, behavioural strategies, operational flows, technical modules, governance, limitations, and how to run everything locally. Architectural trade‑offs live in **`docs/adr/`** (summarised below with links).

---

## How to use this document

| Audience | Jump to |
|----------|---------|
| Hiring manager / product | [Executive summary](#1-executive-summary), [Business context](#2-business-context--problem-space), [Value proposition](#26-value-proposition--business-outcomes) |
| Reservoir / production engineer | [User journeys](#3-stakeholders--user-journeys), [Detector strategy](#81-anomaly-detection-strategy-v1-vs-v2), [Glossary](#6-domain-language--glossary) |
| ML / MLOps engineer | [ML lifecycle](#82-model-lifecycle-mlflow--dvc), [Evaluation](#14-evaluation--experimentation-eval), [CI quality gates](#20-cicd-github-actions) |
| Security / compliance | [Security posture](#18-security-safety--compliance-posture), [ADRs](#19-architectural-decisions-adrs) |
| Developer onboarding | [Quick start](#24-quick-start--examples), [Repository map](#12-repository-layout), [Module reference](#13-python-modules-src) |

---

## Table of contents

1. [Executive summary](#1-executive-summary)  
2. [Business context & problem space](#2-business-context--problem-space)  
3. [Stakeholders & user journeys](#3-stakeholders--user-journeys)  
4. [Scope: in / out of scope](#4-scope-in--out-of-scope)  
5. [Requirements & constraints](#5-requirements--constraints)  
6. [Domain language & glossary](#6-domain-language--glossary)  
7. [Conceptual layers (mental model)](#7-conceptual-layers-mental-model)  
8. [Anomaly detection & ML lifecycle](#8-anomaly-detection--ml-lifecycle)  
9. [Agentic investigation strategy](#9-agentic-investigation-strategy)  
10. [System architecture (diagrams)](#10-system-architecture-diagrams)  
11. [Process flows & ordering](#11-process-flows--ordering)  
12. [Repository layout](#12-repository-layout)  
13. [Python modules (`src/`)](#13-python-modules-src)  
14. [Evaluation & experimentation (`eval/`)](#14-evaluation--experimentation-eval)  
15. [Scripts & DVC pipelines](#15-scripts--dvc-pipelines)  
16. [Data governance & licences](#16-data-governance--licences)  
17. [Configuration & environment](#17-configuration--environment-variables)  
18. [Security, safety & compliance posture](#18-security-safety--compliance-posture)  
19. [Architectural decisions (ADRs)](#19-architectural-decisions-adrs)  
20. [CI/CD (GitHub Actions)](#20-cicd-github-actions)  
21. [Observability & operations](#21-observability--operations-runbook-notes)  
22. [Testing strategy](#22-testing-strategy)  
23. [Honest limitations & risks](#23-honest-limitations--risks)  
24. [Roadmap](#24-roadmap)  
25. [Contributing & doc discipline](#25-contributing--documentation-discipline)  
26. [Value proposition recap](#26-value-proposition--business-outcomes)  
27. [Quick start & examples](#27-quick-start--examples)  
28. [Public HTTP API surface](#28-public-http-api-surface-overview)  

---

## 1. Executive summary

**NorthSea AgentOps** is a reference architecture and working codebase for **AI‑assisted production surveillance** on **North Sea–style offshore wells**. It connects three engineering concerns that operators usually silo:

1. **Detection** — multivariate anomalies (rates, choke, pressures, temperatures) using interpretable statistical learning, not black‑box novelty only.  
2. **Interpretation** — **agentic** workflows (Planner → Executor → **Critic** → **Challenger** → **Uncertainty Gate**) that gather evidence via tools (time series query, document RAG, diagnostic stubs) instead of hallucinating causal stories.  
3. **Governance** — escalation when uncertainty is high or risk is materially relevant; citations for document‑derived claims; observability hooks; MLflow registry and DVC for lineage.

The project is calibrated around **portfolio and interview depth** while remaining truthful about **daily public data granularity**: the architecture demonstrates how a major operator could wire agents, detectors, and safety layers; absolute detection precision on daily cells alone does not pretend to substitute sub‑daily SCADA in production KPIs ([limitations](#23-honest-limitations--risks)).

---

## 2. Business context & problem space

### 2.1 Industry backdrop

Oil and gas **offshore hubs**—platforms, subsea tie‑backs to FPSOs, electrified installations—produce under **tight HSE envelopes**, ageing brownfield declines, intermittent sand/wax/scale upset, choke management, compressor and **ESP reliability**, constraints from water handling and flare/vent limits, and **regulatory continuity** across handovers and contractors.

**Operating production surveillance** consumes significant engineering time:

- Routine review of dashboards and threshold alarms.  
- Correlating **tag data** with **reports, maintenance histories, commissioning notes, and HSE procedures**.  
- Deciding whether a deviation warrants **deferral**, **optimization**, **work order**, **shutdown**, **curtail**, or escalation to drilling/completions.

Mis‑prioritisation has direct **P&L** (undetected choke / pump issues → prolonged rate loss) and raises **major accident hazard** stakes when ambiguous situations are automated without escalation paths.

### 2.2 Core business problems addressed

These are the organisational pressures this codebase is designed **to speak to**:

| Problem | Why it hurts | Repo response |
|---------|---------------|---------------|
| **Noise / alarm fatigue** | Chronic false positives degrade trust; crews silence alarms → real events hide in plain sight | v2 **persistence + refractory** period, CUSUM for gradual drift, event‑level metrics; explicit trade‑off honesty in `eval/detector_performance_v2.json` |
| **Tacit knowledge loss** | Retiring experts held “mental models” tying tags to remedies and risk | **RAG** over ingestible corpuses + structured agent steps that surface citations |
| **Slow root‑cause brainstorming** | First‑principles RCA across documents and tags takes hours per event | Automated **investigation playbook** loop with capped iterations and escalation |
| **Black‑box “AI recommendations”** | Unexplained conclusions are unacceptable near safety‑critical envelopes | Confidence + evidence thresholds; challenger; explicit escalation reasons persisted to DB |
| **Audit pressure** | “Which model answered which alarm on which dataset?” matters after incidents | **MLflow Model Registry**, run tags, DVC pipeline stubs, Postgres persistence trails |
| **Integration reality** | SCADA historians, EH, document stores, and alerting are heterogeneous | Kafka‑shaped ingestion path (`Settings` topics), FastAPI façade, Postgres as system of record for investigations |

### 2.3 Operational story (composite, not fictional numbers)

Operators notice **rising water cut without matching choke widening**, or **BHP drifting down with GOR creeping up**. Each pattern suggests different failure modes (water breakthrough versus gas **coning** versus gauge drift). Investigation needs **historic analogues** (“when did we see comparable water‑cut ramps on neighbouring wells”) and procedures (“what is mandated sampling / verification before recommending chemical injection”).

NorthSea AgentOps encodes **that investigatory choreography**—not as a substitute for mandated procedures, but as a decision support scaffold that escalates rather than guesses when evidence is thin.

---

## 3. Stakeholders & user journeys

### 3.1 Personas

| Persona | Primary need | Typical interaction |
|---------|----------------|---------------------|
| **Production technologist / well analyst** | Fast triage, ranked hypotheses | Reviews investigation output and citations |
| **Offshore superintendent / HSE advisor** | Defensible escalation path when AI is unsure | Opens escalations from queue / tickets |
| **Data / ML engineer** | Retrain, register, rollback models | Runs `scripts/train_detector_v2.py`, MLflow transitions |
| **Platform / observability engineer** | SLOs on investigation latency | Prometheus Grafana stack |
| **Security / IAM** | Least‑privilege and traceability | Tool allowlists, audit logging |

### 3.2 User journeys (high level)

1. **Surveyance loop** — telemetry arrives (Kafka REST or simulated) → detector scores reading → anomaly record → optionally trigger investigation endpoint.  
2. **Insight loop** — operator asks `/rag/query` (or routed through ReAct for simple intents) → hybrid retrieval ranks chunks → concise answer cites sources in DB.  
3. **Remediation loop** — investigation yields recommended actions stored in Postgres; escalation creates work queue downstream (Kafka `agent.escalations` configurable).  

---

## 4. Scope (in / out of scope)

**In scope (by design)**

- Reference **detectors** trained on **real historical field data** paths (Equinor Volve), with evaluation harnesses publishing JSON metrics artefacts.  
- **Agent graphs** coordinating tools and escalation with explicit uncertainty handling.  
- **RAG** pipeline with Postgres/pgvector persistence.  
- **Safety** layers (prompt injection checks on user‑supplied descriptive text in investigation triggers, adversarial probes in CI where configured).  
- **MLOps** hooks: MLflow runs + detector registry scaffolding; DVC YAML for repeatable DAG semantics.  

**Explicitly out of scope (portfolio boundaries)**

- **Live operator SCADA adapters** proprietary to each licence—only interfaces and configs are modeled.  
- **Closed‑loop autonomous actuation** (choke moves, choke commands to DCS)—this codebase never writes back to controls.  
- **Full subsurface simulator coupling** — no Eclipse/Intersect integration; hypothetical narrative only through `ProductionOptimizer` style bridging.  

---

## 5. Requirements & constraints

### 5.1 Functional requirements (architecture‑level FRs)

| ID | Requirement | Implementation hint |
|----|-------------|---------------------|
| FR‑1 | Ingest multivariate telemetry and emit structured alerts | `WellAnomalyDetector`, `AdaptiveWellAnomalyDetector` |
| FR‑2 | Run multi‑step grounded investigation | LangGraph orchestrator (`orchestrator.py`) |
| FR‑3 | Retrieve operator knowledge grounded in citations | Hybrid RAG (`retriever.py`, `vectorstore.py`) |
| FR‑4 | Escalate on low confidence / high risk gates | `uncertainty_gate.py`, DB + optional Kafka escalation |
| FR‑5 | Track experiments and optionally register detector artefacts | MLflow integration + `model_registry.py` |

### 5.2 Non‑functional requirements (NFR targets)

These are **design targets**—tune per deployment:

| NFR | Target intent | Observability |
|-----|---------------|---------------|
| **Investigation bounded runtime** | `AGENT_TIMEOUT_SECONDS` (see `.env.example`) avoids runaway graphs | Histograms exposed via orchestrator + FastAPI timings |
| **Explainability bias** | Affected tags, anomaly types, cited docs visible in payloads | Persisted citations count in API responses |
| **Maintainability** | ADRs mandatory for paradigm shifts | `docs/adr/*.md` |
| **Isolation** | No silent LLM‑only physics claims beyond evidence thresholds | Challenger + Gate |

---

## 6. Domain language & glossary

| Term | Meaning |
|------|---------|
| **BOPD** | Barrels of oil per day (often mapped from reservoir SM³ in public datasets using fixed conversion conventions in loaders). |
| **Water cut** | Share of liquid volume that is produced water—influences separation, pumping, corrosion, polymer demand. |
| **GOR / GLR** | Gas‑oil ratio style signals reflecting breakthrough, liberation, choke effects, metering issues. |
| **Choke / 64ths** | Restrictor opening—primary short‑timescale production control observable on surfaces. |
| **BHP** | Bottom‑hole referenced pressure analogue in daily exports—drops can signal depletion/inflow/skin/pump impairment contextually. |
| **ESP** | Electrical submersible pump—bearing wear and gas locking affect rate stability; RUL‑style narration is illustrative in codebase. |
| **FPSO / platform surveillance** | This solution models **surveillance** not **control**. |
| **HSE** | Health, Safety, Environment—escalations align morally with “better false positives on critical uncertainty than invisible wrong confidence.” |

---

## 7. Conceptual layers (mental model)

```text
Layer 7  Human escalation & workflow tools (ticket systems — external)
Layer 6  Investigation API — FastAPI (REST)
Layer 5  Agents — Planner / Executor / Critic / Challenger / Gate / ReAct
Layer 4  Tools — Telemetry query, diagnostics stub, escalation writer, retrieval
Layer 3  Retrieval — ingestion + embeddings + BM25‑hybrid retrieval
Layer 2  Detection — Statistical + ensemble novelty + sequential SPC overlays
Layer 1  Transport — Kafka topics (mirrors Event Hub patterns); Postgres persistence
Layer 0  Data — Licenced corpus + CSV/Excel field exports + Sodir loaders
```

**Information flow order for a investigated anomaly:** Layers 2→5→6, with Layers 3–4 pulled as tools demand.

---

## 8. Anomaly detection & ML lifecycle

### 8.1 Anomaly detection strategy (v1 vs v2)

| Aspect | **v1** `WellAnomalyDetector` | **v2** `AdaptiveWellAnomalyDetector` |
|--------|---------------------------|--------------------------------------|
| Univariate shocks | Rolling Z‑score vs window | Adaptive thresholds calibrated from benign history |
| Multivariate correlations | Periodic Isolation Forest on scaled features | IF + richer operational policy |
| Slow trends | Easily missed vs window mean | **CUSUM** state machines per telemetry feature |
| Alert storms | Potential burst false positives after threshold crossings | **Persistence** (must hold N intervals) **+ refractory cooldown** |
| Offline training | Fits buffer windows | Dedicated `fit()` before deployment style registration |

**Design intent:** v1 anchors explainability benchmarks; v2 reflects **research‑grade increments** plausible in industrial time‑series practise (adapted SPC intuition).

Evaluation protocol in `scripts/train_detector_v2.py` uses **temporal splitting with a temporal gap**, **injections** modelling both **step** and **gradual** anomalies—mirroring abrupt trips vs creeping decline.

### 8.2 Model lifecycle (MLflow & DVC)

- **Runs** capture parameters and metrics (`eval/ragas_eval.py`, `eval/calibration.py`, detector training).  
- **Model Registry** (see `src/ml/model_registry.py`) formalises **`WellDetectorPyfunc`** artefacts with schemas and promotion gates outlined in **`docs/adr/ADR-006-mlops-model-registry.md`**.  
- **DVC** (`dvc.yaml`) encodes repeatable pipeline stages—even if large binary data remains local‑only due to licences.

Operational sequence (conceptual):

```text
Data pin (DVC/manual) → train/eval scripts → artefact JSON + MLflow Run
         → conditional register_detector() → staging/production transition
           → DetectorRegistry preload at runtime → Prometheus counter updates
```

---

## 9. Agentic investigation strategy

Patterns are documented in **`docs/adr/ADR-005-agentic-patterns.md`**:

| Strategy | Used when |
|----------|-----------|
| **Plan‑execute‑critique loop** | Multi‑evidence RCA needs structured steps (`planner`, `executor`, `critic`). |
| **Adversarial challenge** (`challenger.py`) | High severity / brittle causal chains questioned before escalation decision. |
| **ReAct** (`react_agent.py`) | Lightweight direct tool chains for narrowly scoped intents (cost‑aware routing story). |

**Uncertainty Gate** merges numeric confidence, evidence overlap, challenger notes, severity—escalating with human‑readable rationales persisted for audit—not only “because LLM said so.”

---

## 10. System architecture (diagrams)

### 10.1 Logical containers

```mermaid
flowchart TB
    subgraph Sources
        VOLVE[Volve Excel / loaders]
        NPD[NPD / Sodir loaders]
        DOCS[data/docs corpus]
        KAFKA[Kafka telemetry topic]
    end

    subgraph Runtime
        API[FastAPI src.api.main]
        ORCH[LangGraph orchestrator]
        DET[Detectors v1/v2]
        RAG[RAG retriever]
        PG[(PostgreSQL + pgvector)]
        ML[(MLflow + artefact store)]
    end

    subgraph Observability
        OTEL[OpenTelemetry OTLP exporter]
        PROM[Prometheus /metrics scrape]
        GRAF[Grafana dashboards]
    end

    VOLVE --> DET
    NPD -. optional .-> API
    DOCS --> RAG
    KAFKA --> DET
    DET --> ORCH
    ORCH --> RAG
    RAG --> PG
    API --> PG
    ORCH --> PG
    DET --> ML
    API --> PROM
    API --> OTEL
    OTEL --> GRAF
    PROM --> GRAF
```

### 10.2 Agent graph (investigation orchestration order)

Implementation order (`src/agents/orchestrator.py`):

```text
START → planner → executor ⟲ (until plan exhausted or max iterations hit)
      → critic → challenger → uncertainty_gate
      → escalate OR output → END
```

```mermaid
stateDiagram-v2
    [*] --> planner
    planner --> executor : plan issued
    executor --> executor : more steps remaining
    executor --> critic : plan complete or max iterations safeguard
    critic --> challenger
    challenger --> uncertainty_gate
    uncertainty_gate --> escalate : should_escalate
    uncertainty_gate --> output : cleared thresholds
    escalate --> [*]
    output --> [*]
```

---

## 11. Process flows & ordering

### 11.1 Greenfield bootstrap (developer)

Exact order minimizes “works on my machine” ambiguity:

1. Clone repository.  
2. `cp .env.example .env` and provide `OPENAI_API_KEY`. Align `DATABASE_URL` with Compose defaults or your Postgres.  
3. `pip install -e ".[dev]"`.  
4. `docker compose up -d` (Postgres/pgvector health, Kafka+Zookeeper, Grafana/Prometheus, MLflow depending on Compose profile).  
5. Apply schema: `infra/sql/init.sql` (volume mounted on first Postgres boot) **or** `agentops migrate head`.  
6. Author Markdown under `data/docs/` then `agentops ingest`.  
7. Place licenced **`data/Volve_Data/Volve production data.xlsx`** (Equinor open data terms apply).  
8. `python scripts/train_detector_v2.py` or `agentops detector-eval` → refresh `eval/detector_performance_v2.json`.  
9. `agentops serve` → open Swagger at `http://127.0.0.1:8000/docs`.

### 11.2 Request path: anomaly → investigation (REST sequence)

Typical synchronous flow for `POST /api/v1/investigate`:

```mermaid
sequenceDiagram
    participant Client
    participant API as FastAPI router
    participant Guard as injection_guard
    participant Orc as orchestrator LangGraph
    participant Tools as tools package
    participant DB as PostgreSQL

    Client->>API: POST /investigate (TriggerInvestigationRequest)
    API->>Guard: check_user_query(description)
    alt CRITICAL flagged patterns
        API-->>Client: 400 invalid description
    else clean / mild
        API->>Orc: investigate_anomaly(AnomalyAlert)
        Orc->>Tools: Planned tool steps
        loop Executor passes
            Tools->>DB: Telemetry / embeddings / escalation writers
            Tools-->>Orc: Observations
        end
        Orc->>Orc: critic + challenger + uncertainty_gate
        alt escalate
            Orc->>DB: escalation persistence / Kafka downstream
            API-->>Client: InvestigationResponse (should_escalate true)
        else output
            Orc->>DB: investigations row persisted
            API-->>Client: InvestigationResponse (recommendations + citations metric)
        end
    end
```

### 11.3 Streaming path (Kafka)

Consumer (`src/anomaly/kafka_consumer.py`) aligns with Compose topic names `KAFKA_TOPIC_TELEMETRY` → scoring → anomalies topic **downstream integrations must enforce dedupe + safety review in real facilities**.

---

## 12. Repository layout

| Path | Purpose |
|------|---------|
| `src/api/` | FastAPI bootstrap (`main.py`) + routers (`routes/investigations.py`, `telemetry.py`, `rag.py`, `escalations.py`). |
| `src/agents/` | Planner, executor, critic, orchestrator LangGraph compilation, challenger, uncertainty gate, ReAct shortcut, ProductionOptimizer bridging. |
| `src/anomaly/` | Detector v1 (`detector.py`), v2 (`adaptive_detector.py`), Kafka consumer. |
| `src/ml/` | MLflow **`model_registry`** and runtime **`DetectorRegistry` loader**. |
| `src/data/` | **`volve_loader.py`**, **`npd_loader.py`**. |
| `src/rag/` | ingestion, vectorstore embeddings, retrieval. |
| `src/tools/` | Callable affordances leveraged by Planner/Executor. |
| `src/safety/` | injection guard (**masking‑based allow‑lists**), auditing, tool allowlists. |
| `src/schemas/` | Domain models shared across detectors and API surfaces. |
| `src/db/` | Connection/session primitives. |
| `src/config.py` | Consolidated pydantic‑settings singleton. |
| `src/cli.py` | **`agentops`** Typer CLI. |
| `src/observability/` | OpenTelemetry bootstrap + Prometheus integration glue. |
| `eval/` | golden sets, Ragas orchestration, offline agent evaluation, calibration, adversarial probes, regression guards, detector JSON outputs. |
| `scripts/` | Training / benchmarking scripts. |
| `tests/unit/` | Unit tests incl. anomaly + safety. |
| `tests/integration/` | RAG/integration DB bound tests selectively marked. |
| `infra/` | SQL init (`infra/sql/init.sql`), k8s example, Grafana & Prometheus snippets, OTLP collector YAML. |
| `docs/adr/` | ADR‑001 … ADR‑006. |
| `.github/workflows/` | `ci.yml`, `eval-gate.yml`. |

---

## 13. Python modules (`src/`)

This section inventories **purpose + strategy**. For line‑level APIs read docstrings—the README stays stable at module granularity.

### Core configuration & schemas

| Module | Responsibility |
|--------|----------------|
| `config.py` | Environment‑backed **`Settings`**; cached via **`get_settings()`**. |
| `schemas/domain.py` | **`AnomalyAlert`**, severities, investigation result shapes used across routers and agents. |

### API layer

| Module | Responsibility |
|--------|----------------|
| `api/main.py` | Lifespan (telemetry + DB check), Prometheus **`Instrumentator`** on `/metrics`, CORS middleware, routers mount. |
| `api/routes/investigations.py` | Builds **`AnomalyAlert`** from **`TriggerInvestigationRequest`**, invokes **`investigate_anomaly`**; injection guard gate on textual description inputs. |

### Agents

| Module | Responsibility |
|--------|----------------|
| `agents/orchestrator.py` | Compiles **`StateGraph`**, route helpers `_should_continue_executing`, `_escalate_or_output`, challenger node placement, persists outputs and escalations, Prometheus metrics wiring. |
| `agents/planner.py` | Emits investigative plan respecting tool ergonomics / safety tone constraints. |
| `agents/executor.py` | Iterates executing tool calls with DB injected for grounded retrieval. |
| `agents/critic.py` | Produces calibrated qualitative scoring inputs for gate. |
| `agents/challenger.py` | Synthetic adversarial pass + reconciliation summary back into uncertainty features. |
| `agents/uncertainty_gate.py` | Deterministic escalation policy composed with LLM derived scores where applicable (`apply_uncertainty_gate`, `build_investigation_result`). |
| `agents/react_agent.py` | Lightweight ReAct path for narrow queries documented against full graph cost. |
| `agents/production_optimizer.py` | Bridges technical anomaly typing into **economics narration** placeholders (explicitly not pretending field economics truth without priors file). |

### Detection & ingestion

| Module | Responsibility |
|--------|----------------|
| `anomaly/detector.py` | **`WellAnomalyDetector`** Z‑score rolling window + scaler + **IsolationForest** with retrain cadence logic. |
| `anomaly/adaptive_detector.py` | **`AdaptiveWellAnomalyDetector`** adds **CUSUM**, persistence streak counter, cooldown counter resets after confirmed alerts aligning with classical SPC re‑arming practise. |
| `anomaly/kafka_consumer.py` | Poll → parse → ingest → anomaly emit contract (see constants for topic hygiene). |

### Data loaders

| Module | Responsibility |
|--------|----------------|
| `data/volve_loader.py` | Excel column validation, volumetric conversions, derived rates (water‑cut derivation defensively handles zero denominators via logic described in loader). |
| `data/npd_loader.py` | Optional external factual ingestion path for Sodir‑style artefacts. |

### RAG subsystem

| Path | Responsibility |
|------|----------------|
| `rag/ingestion.py` | Document discovery, deterministic chunk hashing via stable identifiers, chunked upserts. |
| `rag/vectorstore.py` | Embedding API calls batched appropriately, pgvector cosine storage. |
| `rag/retriever.py` | Semantic + lexical hybrid blend parameterised by **`rag_hybrid_alpha`**. |

### Tools / safety / observability / ML lifecycle

Summarised in earlier architecture tables—the critical safety nuance **`injection_guard.py` never short‑circuits** inspection simply because benign tokens matched allowlists; masking ensures adversarial payloads still scrutinised (**important fix path for interview storytelling**).

---

## 14. Evaluation & experimentation (`eval/`)

| File | Business question answered |
|------|-------------------------------|
| `golden_testset.json` | Controlled Q/A pairs for Retrieval faithfulness regressions vs evolving prompts. |
| `ragas_eval.py` | “Is grounding degrading?”—logs metrics optionally to MLflow when server reachable. |
| `agent_eval.py` | “Are agent policies slipping?” surrogate structural checks offline. |
| `calibration.py` | Calibration error quantification bridging probability statements to observable outcomes in offline harness. |
| `adversarial_tests.py` | Adversarial safety regression battery. |
| `regression_tests.py` | Prevents unintended behaviour drift merging via CI gates. |

**Committed metrics:** **`eval/detector_performance_v2.json`** is authoritative for anomaly offline experiments—portfolio reviewers expect numbers to reconcile with markdown narrative.

---

## 15. Scripts & DVC pipelines

| Script | Output / role |
|--------|---------------|
| `scripts/train_detector.py` | Simpler v1 benchmarking path historically useful for demos. |
| `scripts/train_detector_v2.py` | Authoritative KPI JSON + aggregates + commentary printed to stdout (`register_trained_models` optional MLflow linkage). |

`dvc.yaml` documents pipeline edges even if heavyweight binary artefacts remain unstored publicly.

---

## 16. Data governance & licences

| Dataset / artefact | Governance note |
|--------------------|----------------|
| Equinor Volve open dataset | Honour **Equinor + NLOD** licensing; do **not** commit spreadsheets to public git; document provenance internally. |
| Local `data/docs/` corpus | Controlled operator narrative—quality directly affects RAG trust. |
| NPD/Sodir style pulls | Respect API terms rate limits attribution in external talks. |

---

## 17. Configuration & environment variables

Copy **`.env.example`** → `.env`. Key knobs (non‑exhaustive—see source for full pydantic defaults):

| Variable group | Behavioural lever |
|----------------|---------------------|
| `OPENAI_*` | LLM routing + embeddings (swap to **`AZURE_*` stubs** commented for enterprise migration). |
| `DATABASE_*` | Pool connectivity for async routes and ingestion. |
| `KAFKA_*` | Telemetry / anomaly / escalation topic separation—mirrors segregation of reliability concerns operator side. |
| `AGENT_CONFIDENCE_THRESHOLD` etc. | **Escalation aggressiveness tuning** balancing automation vs nuisance human queueing. |
| `RAG_*` | Retrieval precision vs breadth + hybrid mixing. |
| `MLFLOW_TRACKING_URI` | Local Compose port `5001` mapped externally per `docker-compose.yml`. |

### Local service ports (`docker-compose.yml` highlights)

Services exposed for development include **PostgreSQL (`5432`)**, **Kafka (`9092`)**, **Kafka UI (`8090`)**, **Prometheus (`9090`)**, **Grafana (see compose)** and MLflow (**host `5001` → container 5000** pattern). Inspect compose labels when debugging port collisions.

---

## 18. Security, safety & compliance posture

| Control | Purpose |
|---------|---------|
| **Injection guarding** (`injection_guard.py`) | Mitigate textual prompt injection avenues from operator submitted descriptions when triggering investigations. |
| **Tool allowlisting** (`tool_allowlist.py`) | Prevent agents from spawning arbitrary unmanaged side effects. |
| **Audit logging scaffolding** (`audit_log.py`) | Trace escalation decisions—not a full SIEM substitution. |
| **Non‑automated choke / valve actuation** | Architecture isolates inference from control. |
| **Human escalation path** (`uncertainty_gate.py`) | Risk aware default when calibrated confidence dips or evidence breadth fails thresholds. |

> **Disclaimer:** Demo posture ≠ certified industrial cyber programme. Operators must overlay corporate IAM, segregated VPCs/VNets, private endpoints for LLMs, KMS secret rotation, egress controls, DPIA workflows, Norwegian **NORSOK** / **IEC 62443** overlays as applicable.

Align philosophically with **NORSOK Z‑013**: traceability narratives between analysis inputs and consequential recommendations—ADR‑006 cites registry traceability parallels.

---

## 19. Architectural decisions (ADRs)

| ADR | Concise rationale |
|-----|-------------------|
| [ADR‑001](docs/adr/ADR-001-langgraph-vs-alternatives.md) | Why LangGraph for structured loops & checkpointing ergonomics. |
| [ADR‑002](docs/adr/ADR-002-pgvector-vs-vector-dbs.md) | Operational simplicity vs SaaS specialised vector infra. |
| [ADR‑003](docs/adr/ADR-003-evaluation-stack.md) | Measuring regressions objectively instead of vibes. |
| [ADR‑004](docs/adr/ADR-004-kafka-streaming.md) | Decoupling ingestion bursts from investigative compute. |
| [ADR‑005](docs/adr/ADR-005-agentic-patterns.md) | When ReAct suffices vs heavyweight multi‑agents. |
| [ADR‑006](docs/adr/ADR-006-mlops-model-registry.md) | Lifecycle & promotion framing for audited ML artefacts. |

---

## 20. CI/CD (GitHub Actions)

**`.github/workflows/ci.yml`** (typical precedence):

| Step | Validates |
|------|-----------|
| `lint` | Ruff lint + formatter + strictness friendly mypy invocation on `src/`. |
| `unit-tests`, `regression-tests`, `adversarial-tests` | Core correctness + resilience probes. |
| `agent-eval` | Offline behavioural metrics with local file MLflow root to avoid flaky remote dependency. |
| `model-quality-gate` | Guards registry helper correctness + parses committed detector benchmark JSON tolerant if local data absent clones. |
| `integration-tests` | Postgres‑backed narrower integration scope (`pytest … -m “not requires_kafka”` style filtering). |

`eval-gate.yml` complements heavy evaluation regressions gated on secrets posture.

---

## 21. Observability & operations (runbook notes)

| Signal | Interpretation playbook |
|--------|--------------------------|
| **Prometheus scrape `/metrics`** | Latency buckets on HTTP instrumentation + custom counters incremented through orchestrator. |
| **OpenTelemetry OTLP exporter** | Spans bridging agent steps when collector configured—check `infra/otel`. |
| **Structured logs (`structlog` pattern in main)** | Correlate investigations by embedding investigation ids if extended in forks. |

**Rolling forward model versions:** Prefer MLflow **`transition_model_version_stage`** guarded by **`event_recall` / false positive KPIs**. Rollback leverages **`rollback_to_previous_production()`** semantics in **`model_registry`** path.

---

## 22. Testing strategy

| Layer | Tooling | Paths |
|-------|---------|-------|
| **Unit** | pytest | `tests/unit/` (anomaly correctness, injection guard regressions) |
| **Integration** | pytest + service containers in CI | `tests/integration/` |
| **Eval harness** | JSON golden sets + scripted metrics pipelines | `eval/` |
| **Quality gates** | GitHub workflows | `.github/workflows/*.yml` |

Target coverage thresholds are enforced via **`pyproject.toml`** → **`[tool.pytest.ini_options]`** together with **`--cov-fail-under`** in pytest **`addopts`**.

---

## 23. Honest limitations & risks

| Limitation | Impact | Mitigation path |
|-----------|--------|-----------------|
| **Daily aggregation** suppresses choke cycling sub‑patterns | Recall/precision KPI ceiling | Sub‑daily historian integration |
| LLM stochasticity shifts narrative style | Confidence calibration drift monitors | Calibration continuous eval loops |
| **Single cloud LLM key** portability | Sovereignty/regulatory optics | Hosted private LLM adapters |
| **Kafka single broker Compose** realism | Exactly‑once semantics not tackled | Extend to clustered mirror + transactional producer patterns |
| Corpus size / coverage | Retrieval blind spots unless docs invested | Operational knowledge engineering programme |

**Portfolio ethics:** Clearly separate **engineering architecture demonstration** versus **validated field deployment claim**. This README supports that differentiation.

---

## 24. Roadmap

Near‑term realistic extensions recruiters look for explicitly:

| Item | Benefit |
|------|---------|
| **Live historian adapter** abstraction | Operational credibility |
| Automated **concept drift sentinel** invoking MLflow downgrade | Operational safety |
| **Azure OpenAI** config profile parity | Matches Equinor / Aker operator clouds |
| **Shadow scoring** simultaneous Staging Production models | Controlled promotion |
| **PDF OCR ingestion** hardened pipeline beyond Markdown | Operational artefact ingestion |

Far‑research horizon: differentiable surrogate coupling (requires proprietary simulators licences).

---

## 25. Contributing & documentation discipline

Pull requests altering cross‑cutting behaviour **must** update or supersede relevant **ADR**.

Detector logic changes ideally refresh **`eval/detector_performance_v2.json`** reproducibly or justify stale metrics consciously.

Maintain **dual documentation**: README conceptual + ADR granular engineering proof.

---

## 26. Value proposition & business outcomes

**Why this earns attention in oil & gas AI hiring loops:**

Articulates bridging **engineering physics intuition** (**water cut trajectory**, choke symmetry arguments) → **trusted automation** layering **risk governance** (**escalations**, challenger, audit artefacts) → **MLOps operationalisation** aligning with large operator digital programmes.

Demonstrates familiarity with Norwegian offshore context subtly through dataset choice (public Volve) while remaining globally interpretable FPSO analogous story.

Financial outcomes are **scenario framing** helpers (`ProductionOptimizer`)—replace constants with calibrated field tariff models for serious economics studies.

---

## 27. Quick start & examples

```bash
# 1 Environment
cp .env.example .env   # populate OPENAI_API_KEY
pip install -e ".[dev]"

# 2 Infra (see docker-compose.yml for exact stack)
docker compose up -d

# 3 Database
agentops migrate head    # Alembic; or rely on infra/sql/init.sql on first Postgres boot

# 4 RAG corpus
mkdir -p data/docs && # add Markdown operational artefacts
agentops ingest

# 5 Real telemetry benchmarking (licenced workbook)
mkdir -p "data/Volve_Data"
# Copy Volve production workbook per Equinor open data licence
agentops detector-eval    # delegates to scripts/train_detector_v2.py

# 6 API
agentops serve --reload    # Swagger UI → http://127.0.0.1:8000/docs
```

Evaluation extras:

```bash
python -m eval.agent_eval --output eval/agent_eval_results.json
python -m eval.calibration
pytest tests/unit tests/integration -v
```

---

## 28. Public HTTP API surface (overview)

Full OpenAPI schemas appear at **`/docs`**. Behavioural summaries:

| Method & path | Intent |
|---------------|--------|
| `GET /health` | Liveness + DB connectivity degraded vs healthy signalling |
| `GET /metrics` | Prometheus exposition |
| `POST /api/v1/investigate` | Runs full investigative graph pathway after optional injection guard textual validation |
| `GET /api/v1/investigations/{id}` | Fetch persisted investigations |
| `POST /api/v1/rag/query` | Retrieval augmented QA path |
| `GET /api/v1/wells/{well_id}/telemetry` | Recent telemetry slice views |
| `GET/POST /api/v1/escalations` | Operational escalation backlog handling |
| `POST /api/v1/simulate/anomaly` | Development anomaly injection guarded / environment gated (see router docstrings) |

---

## Licence notes (project vs data)

This repository ships **research / portfolio code** authored by contributors. **Operational dataset files** retained under external licences (**Equinor Volve**) must be obtained and cited according to upstream terms—they are **`gitignored` by intention**.

---

_Last README structural refresh targets self‑containment for repository archival & hiring packet usage. For micro‑decisions, always reconcile with latest source + ADRs first._
