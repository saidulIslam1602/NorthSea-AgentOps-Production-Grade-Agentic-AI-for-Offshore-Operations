# ADR-006: MLOps Model Registry — MLflow + DVC

**Status:** Accepted  
**Date:** 2026-05-16  
**Authors:** NorthSea AgentOps team  
**Relates to:** ADR-003 (Evaluation Stack), ADR-001 (LangGraph)

---

## Context

The NorthSea AgentOps system trains a per-well `AdaptiveWellAnomalyDetector`
(Isolation Forest + CUSUM) on real Equinor Volve production data. In an
operational setting (Aker BP or equivalent NCS operator), multiple model
versions will co-exist:

- **Baseline v1** (Z-score Z=2.0 + IF): high recall, high FPR
- **Adaptive v2** (auto-calibrated + CUSUM): lower FPR, engineered drift detection
- Future versions trained on sub-daily SCADA data, new wells, or retrained on
  recent production history as field conditions change (late-life water cut, EOR phases)

Without model version control, the following problems arise in production:

| Problem | Consequence |
|---|---|
| No reproducibility | Cannot re-evaluate which model was running when a HSE incident occurred |
| No rollback | Degraded production model cannot be safely reverted |
| No lineage | Cannot trace: which data → which code → which deployed model |
| No A/B testing | Cannot compare new models against incumbent in shadow mode |
| No quality gate | Poor models can be promoted to production without automated validation |

NORSOK Z-013 §4.3 (Risk and emergency preparedness analysis) requires traceability
of safety-critical software decisions. A well anomaly detector that influences
escalation decisions is safety-adjacent; its version and validation state must
be auditable.

---

## Decision

### 1. MLflow Model Registry (model versioning and lifecycle)

Use the **MLflow Model Registry** already deployed in docker-compose
(`mlflow:v2.16.0`, backed by PostgreSQL) to manage the full model lifecycle.

**Model name convention:**
```
northsea-well-anomaly-detector-{well_id_sanitised}
```
e.g. `northsea-well-anomaly-detector-15_9-F-12`

**Lifecycle stages:**
```
Training run → [None] → Staging → Production → Archived
                           ↑
                    Quality Gate (automated):
                    event_recall ≥ 0.85 AND day_fpr ≤ 0.20
                    OR: daily-data override with documented justification
```

**Per-version tags (searchable, auditable):**
- `well_id`, `detector_version` (v1/v2)
- `data_source` (equinor_volve_open_dataset_2007_2016)
- `data_granularity` (daily / 15min / hourly)
- `event_recall`, `day_fpr`, `gate_passed`
- `justification` (human-readable promotion rationale)

**Model signature** (input/output schema enforced at load time):
```python
Input:  6 × float64  [oil_rate_bopd, water_cut_pct, gas_oil_ratio,
                       bhp_psi, wh_temp_f, choke_64ths]
Output: anomaly_score (float64), anomaly_type (string),
        severity (string), economic_impact_usd (float64),
        cusum_triggered (bool), consecutive_days (int64), is_anomaly (bool)
```

**Rollback:** `rollback_to_previous_production(well_id)` archives current
Production version and promotes the previous one — single function call,
no retraining required.

### 2. DVC (Data Version Control — dataset lineage)

Use **DVC** to version the Equinor Volve training data and link it to
trained model versions, enabling full data → model → metrics reproducibility.

```
data/Volve_Data/Volve production data.xlsx  ← tracked by DVC
    ↓ (dvc.yaml pipeline)
eval/detector_performance_v2.json           ← tracked by DVC
    ↓ (linked to MLflow run via run_id tag)
models:/northsea-well-anomaly-detector-*/   ← MLflow Model Registry
```

DVC remote: `./dvc-store/` locally; upgrades to Azure Blob Storage
(`azure://northsea-agentops/dvc`) for multi-developer environments.

### 3. CI Quality Gate (automated promotion)

After every training run in CI (`agent-eval` job), the pipeline:
1. Trains v2 detectors on current data
2. Evaluates on held-out test set
3. Runs `register_trained_models()` → MLflow Model Registry
4. Quality gate check:
   - `event_recall >= 0.85` AND `day_fpr <= 0.20` → **Promote to Production**
   - Otherwise → **Staging only** (human review required before Production)
5. Fails the CI step if no model reaches even Staging (regression guard)

---

## Alternatives Considered

| Option | Rejected Reason |
|---|---|
| **Weights & Biases (W&B)** | Paid SaaS; data sovereignty concern for NCS operational data; MLflow is self-hosted and already in docker-compose |
| **Neptune.ai** | Same SaaS concern; small team overhead |
| **Custom JSON version files** | No lifecycle management, no rollback, no signature enforcement; not auditable |
| **BentoML / Seldon** | Excellent serving tools but add infrastructure complexity; MLflow pyfunc + FastAPI is sufficient for current scale |
| **No DVC (just git)** | Cannot version binary Excel files; no data-to-model lineage |

---

## Consequences

**Positive:**
- Full audit trail: every deployed model version is linked to its training run,
  data commit, evaluation metrics, and promotion justification
- Rollback in < 60 seconds if production drift detected
- Shadow-mode A/B testing: load Staging model in parallel with Production
  without changing inference infrastructure
- Model signatures prevent schema drift bugs at load time (not at runtime)
- Meets NORSOK Z-013 §4.3 traceability requirements

**Negative / Accepted trade-offs:**
- MLflow Registry requires the PostgreSQL backend to be running; adds
  operational dependency vs. in-memory model loading
- DVC adds a git-like workflow for data files; requires team discipline
- pyfunc wrapping adds a serialisation step at registration time

**Monitoring hook (future):**
Add a `ModelDriftMonitor` that weekly re-evaluates the Production model
on the most recent 30 days of production data and triggers automatic
rollback if event_recall drops >10% from the registered baseline.
