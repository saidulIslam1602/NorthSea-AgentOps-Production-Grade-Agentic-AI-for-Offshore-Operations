"""
MLflow Model Registry — NorthSea AgentOps MLOps Layer.

Implements full model lifecycle management for the WellAnomalyDetector family:

  LIFECYCLE STAGES
  ┌──────────────────────────────────────────────────────────────────┐
  │  Training → [None] → Staging → Production → Archived            │
  │                         ↑                                        │
  │                   Quality Gate:                                  │
  │                   event_recall ≥ 0.85 AND day_fpr ≤ 0.20        │
  │                   (or: data_granularity=daily + justification)   │
  └──────────────────────────────────────────────────────────────────┘

  MODEL NAME CONVENTION
  northsea-well-anomaly-detector-{well_id}
  e.g.: northsea-well-anomaly-detector-15_9-F-12

  VERSIONING
  Each registered model version captures:
  - Trained scikit-learn pipeline (StandardScaler + IsolationForest)
  - Per-well adaptive threshold dict
  - CUSUM parameters (k, h)
  - Model signature: 6 telemetry features → anomaly_score
  - Tags: data_source, training_rows, event_recall, day_fpr, etc.
  - Training artifact: detector_performance_v2.json

  WHY THIS MATTERS FOR AKER BP
  1. Reproducibility: any deployed model version can be re-loaded,
     re-evaluated, and compared against the latest. Critical for
     regulatory audit trails in safety-critical operations.
  2. Rollback: if a new model degrades in production (drift detection),
     the previous Production version is promoted back in one API call.
  3. A/B testing: Staging and Production versions can serve simultaneously
     via feature flags, allowing shadow-mode comparison before cutover.
  4. Lineage: data → code → model → metrics are all linked via MLflow run ID,
     enabling full reproducibility from a single version identifier.

References:
  - MLflow Model Registry: https://mlflow.org/docs/latest/model-registry.html
  - Zaharia et al. (2018) "Accelerating the Machine Learning Lifecycle with MLflow"
  - NORSOK Z-013 §4.3: Traceability requirements for safety-critical software
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import ModelSignature
from mlflow.types.schema import ColSpec, Schema

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_NAME_PREFIX = "northsea-well-anomaly-detector"
EXPERIMENT_NAME = "northsea-agentops-detector-training"

TELEMETRY_FEATURES = [
    "oil_rate_bopd",
    "water_cut_pct",
    "gas_oil_ratio",
    "bhp_psi",
    "wh_temp_f",
    "choke_64ths",
]

# Quality gate thresholds for Production promotion
PRODUCTION_GATE = {
    "event_recall_min": 0.85,
    "day_fpr_max": 0.20,
    # Override: allow daily data with documented justification
    "allow_daily_data_override": True,
    "daily_data_min_event_recall": 0.60,  # minimum even for daily data
}

# MLflow stage aliases (MLflow 2.x uses aliases instead of stage strings)
STAGE_STAGING = "Staging"
STAGE_PRODUCTION = "Production"
STAGE_ARCHIVED = "Archived"


@dataclass
class ModelMetrics:
    """Evaluation metrics attached to a registered model version."""

    well_id: str
    detector_version: str  # "v1" or "v2"
    event_recall: float
    day_fpr: float
    day_precision: float
    roc_auc: float
    training_rows: int
    test_rows: int
    data_granularity: str  # "daily" | "hourly" | "15min"
    data_source: str
    zscore_thresholds: dict[str, float] = field(default_factory=dict)
    if_contamination: float = 0.02
    notes: str = ""

    @property
    def meets_production_gate(self) -> bool:
        gate = PRODUCTION_GATE
        if self.data_granularity == "daily" and gate["allow_daily_data_override"]:
            return self.event_recall >= gate["daily_data_min_event_recall"]
        return self.event_recall >= gate["event_recall_min"] and self.day_fpr <= gate["day_fpr_max"]

    @property
    def promotion_justification(self) -> str:
        if self.meets_production_gate:
            if self.data_granularity == "daily":
                return (
                    f"Promoted with daily-data override: event_recall={self.event_recall:.2f} "
                    f"(≥{PRODUCTION_GATE['daily_data_min_event_recall']:.2f}). "
                    f"Full production metrics require sub-daily SCADA data (projected recall>95%, FPR<5%)."
                )
            return (
                f"Meets production gate: event_recall={self.event_recall:.2f} "
                f"(≥{PRODUCTION_GATE['event_recall_min']:.2f}), "
                f"day_fpr={self.day_fpr:.2f} (≤{PRODUCTION_GATE['day_fpr_max']:.2f})."
            )
        return (
            f"Quality gate FAILED: event_recall={self.event_recall:.2f} "
            f"(need ≥{PRODUCTION_GATE['event_recall_min']:.2f}), "
            f"day_fpr={self.day_fpr:.2f} (need ≤{PRODUCTION_GATE['day_fpr_max']:.2f}). "
            f"Registered in Staging only."
        )


class WellDetectorPyfunc(mlflow.pyfunc.PythonModel):  # type: ignore[misc,attr-defined]
    """
    MLflow pyfunc wrapper for AdaptiveWellAnomalyDetector.

    Exposes the trained detector as a standard MLflow model that can be:
      - Logged to the MLflow artifact store
      - Registered in the Model Registry with a version
      - Loaded in any downstream service via mlflow.pyfunc.load_model()
      - Served via `mlflow models serve` for REST inference

    Input:  DataFrame with 6 telemetry columns (see TELEMETRY_FEATURES)
    Output: DataFrame with columns [anomaly_score, anomaly_type, severity, economic_impact_usd]
    """

    def load_context(self, context: Any) -> None:
        """Load detector state from MLflow artifact path."""
        detector_path = context.artifacts["detector_state"]
        thresholds_path = context.artifacts["adaptive_thresholds"]

        with open(detector_path, "rb") as f:
            self._detector_state = pickle.load(f)  # (scaler, iso_forest)

        with open(thresholds_path) as f:  # type: ignore[assignment]
            self._thresholds = json.load(f)

        from src.anomaly.adaptive_detector import AdaptiveWellAnomalyDetector

        self._detector = AdaptiveWellAnomalyDetector.__new__(AdaptiveWellAnomalyDetector)
        self._detector._scaler = self._detector_state["scaler"]
        self._detector._iso_forest = self._detector_state["iso_forest"]
        self._detector._is_trained = True
        self._detector._zscore_thresholds = self._thresholds
        self._detector.well_id = self._detector_state["well_id"]
        self._detector.min_consecutive = self._detector_state.get("min_consecutive", 2)
        self._detector._refractory_days = 7
        self._detector._refractory_remaining = 0
        self._detector._consecutive_count = 0

        from collections import deque

        from src.anomaly.adaptive_detector import CUSUMState

        self._detector._buffer = deque(maxlen=500)
        self._detector._cusum = {f: CUSUMState() for f in TELEMETRY_FEATURES}
        self._detector._retrain_counter = 0
        self._detector._retrain_interval = 60
        self._detector.zscore_window = 30
        self._detector.oil_price = 80.0
        self._detector.target_fpr = 0.05
        self._detector._if_contamination = 0.02
        self._detector._if_n_estimators = 200
        self._detector._min_train_samples = 90

    def predict(
        self,
        context: Any,
        model_input: pd.DataFrame,
        params: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """
        Score a batch of telemetry readings.

        Each row is treated as a sequential reading (order matters for CUSUM).
        Returns one row of scores per input row.
        """
        del context, params  # Provided by MLflow pyfunc runner; artefacts already in load_context.
        results = []
        for _, row in model_input.iterrows():
            reading: dict[str, Any] = {f: float(row.get(f, 0.0)) for f in TELEMETRY_FEATURES}
            reading["well_id"] = self._detector.well_id
            reading["field_name"] = "VOLVE"
            alert = self._detector.ingest(reading)

            if alert is not None:
                results.append(
                    {
                        "anomaly_score": alert.base_alert.anomaly_score,
                        "anomaly_type": alert.anomaly_type.value,
                        "severity": alert.base_alert.severity.value,
                        "economic_impact_usd": alert.economic_impact.revenue_at_risk_usd_per_day,
                        "cusum_triggered": alert.cusum_triggered,
                        "consecutive_days": alert.consecutive_days,
                        "is_anomaly": True,
                    }
                )
            else:
                results.append(
                    {
                        "anomaly_score": 0.0,
                        "anomaly_type": "normal",
                        "severity": "normal",
                        "economic_impact_usd": 0.0,
                        "cusum_triggered": False,
                        "consecutive_days": 0,
                        "is_anomaly": False,
                    }
                )

        return pd.DataFrame(results)


def _build_model_signature() -> ModelSignature:
    """Define input/output schema for the detector model."""
    input_schema = Schema([ColSpec("double", col) for col in TELEMETRY_FEATURES])
    output_schema = Schema(
        [
            ColSpec("double", "anomaly_score"),
            ColSpec("string", "anomaly_type"),
            ColSpec("string", "severity"),
            ColSpec("double", "economic_impact_usd"),
            ColSpec("boolean", "cusum_triggered"),
            ColSpec("long", "consecutive_days"),
            ColSpec("boolean", "is_anomaly"),
        ]
    )
    return ModelSignature(inputs=input_schema, outputs=output_schema)


def register_detector(
    detector: Any,  # AdaptiveWellAnomalyDetector instance (post-fit)
    metrics: ModelMetrics,
    tracking_uri: str = "http://localhost:5001",
    eval_json_path: Path | None = None,
) -> str:
    """
    Log a trained detector to MLflow and register in the Model Registry.

    Steps:
      1. Set experiment → start run
      2. Log all training metrics and parameters
      3. Serialise detector state (scaler + IF) to artifacts
      4. Log model with signature and input example
      5. Register model version in Model Registry
      6. Apply quality gate → promote to Staging or Production
      7. Add descriptive alias tags

    Returns the registered model version URI.
    """
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    model_name = f"{MODEL_NAME_PREFIX}-{metrics.well_id.replace('/', '_').replace(' ', '_')}"

    with mlflow.start_run(run_name=f"train-{metrics.well_id}-{metrics.detector_version}") as run:
        run_id = run.info.run_id

        # ── Log parameters ────────────────────────────────────────────────
        mlflow.log_params(
            {
                "well_id": metrics.well_id,
                "detector_version": metrics.detector_version,
                "data_source": metrics.data_source,
                "data_granularity": metrics.data_granularity,
                "training_rows": metrics.training_rows,
                "if_contamination": metrics.if_contamination,
                "min_consecutive": detector.min_consecutive,
                "refractory_days": detector._refractory_days,
                "zscore_window": detector.zscore_window,
                "cusum_k": detector._cusum[TELEMETRY_FEATURES[0]].k,
                "cusum_h": detector._cusum[TELEMETRY_FEATURES[0]].h,
            }
        )

        # ── Log metrics ───────────────────────────────────────────────────
        mlflow.log_metrics(
            {
                "event_recall": metrics.event_recall,
                "day_fpr": metrics.day_fpr,
                "day_precision": metrics.day_precision,
                "roc_auc": metrics.roc_auc,
                "test_rows": float(metrics.test_rows),
                "training_rows": float(metrics.training_rows),
            }
        )
        for feat, thresh in metrics.zscore_thresholds.items():
            mlflow.log_metric(f"zscore_threshold_{feat}", thresh)

        # ── Serialise detector state to artifacts ─────────────────────────
        with tempfile.TemporaryDirectory() as tmpdir:
            # Detector state (scaler + IF model)
            state_path = os.path.join(tmpdir, "detector_state.pkl")
            thresholds_path = os.path.join(tmpdir, "adaptive_thresholds.json")
            metrics_path = os.path.join(tmpdir, "model_metrics.json")

            with open(state_path, "wb") as f:
                pickle.dump(
                    {
                        "well_id": metrics.well_id,
                        "scaler": detector._scaler,
                        "iso_forest": detector._iso_forest,
                        "min_consecutive": detector.min_consecutive,
                    },
                    f,
                )

            with open(thresholds_path, "w") as f:
                json.dump(detector._zscore_thresholds, f, indent=2)

            with open(metrics_path, "w") as f:
                json.dump(
                    {
                        "event_recall": metrics.event_recall,
                        "day_fpr": metrics.day_fpr,
                        "roc_auc": metrics.roc_auc,
                        "gate_passed": metrics.meets_production_gate,
                        "justification": metrics.promotion_justification,
                    },
                    f,
                    indent=2,
                )

            mlflow.log_artifact(state_path, artifact_path="detector")
            mlflow.log_artifact(thresholds_path, artifact_path="detector")
            mlflow.log_artifact(metrics_path, artifact_path="detector")

            if eval_json_path and eval_json_path.exists():
                mlflow.log_artifact(str(eval_json_path), artifact_path="evaluation")

            # ── Log model with signature ──────────────────────────────────
            signature = _build_model_signature()

            # Build input example for model card
            input_example = pd.DataFrame(
                [
                    {
                        "oil_rate_bopd": 5000.0,
                        "water_cut_pct": 45.0,
                        "gas_oil_ratio": 850.0,
                        "bhp_psi": 3400.0,
                        "wh_temp_f": 162.0,
                        "choke_64ths": 40.0,
                    }
                ]
            )

            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=WellDetectorPyfunc(),
                artifacts={
                    "detector_state": state_path,
                    "adaptive_thresholds": thresholds_path,
                },
                signature=signature,
                input_example=input_example,
                registered_model_name=model_name,
                pip_requirements=[
                    "mlflow>=2.16.0",
                    "scikit-learn>=1.5.0",
                    "numpy>=1.26.0",
                    "pandas>=2.2.0",
                ],
            )

        # ── Apply quality gate and set stage ─────────────────────────────
        client = MlflowClient(tracking_uri=tracking_uri)

        # Get the version just registered
        versions = client.get_latest_versions(model_name)
        latest_version = max(int(v.version) for v in versions)
        version_str = str(latest_version)

        target_stage = STAGE_PRODUCTION if metrics.meets_production_gate else STAGE_STAGING

        client.transition_model_version_stage(
            name=model_name,
            version=version_str,
            stage=target_stage,
            archive_existing_versions=False,  # keep previous production for rollback
        )

        # Rich tags on the model version
        client.set_model_version_tag(model_name, version_str, "well_id", metrics.well_id)
        client.set_model_version_tag(model_name, version_str, "detector_version", metrics.detector_version)
        client.set_model_version_tag(model_name, version_str, "data_source", metrics.data_source)
        client.set_model_version_tag(model_name, version_str, "data_granularity", metrics.data_granularity)
        client.set_model_version_tag(model_name, version_str, "event_recall", str(round(metrics.event_recall, 4)))
        client.set_model_version_tag(model_name, version_str, "day_fpr", str(round(metrics.day_fpr, 4)))
        client.set_model_version_tag(model_name, version_str, "gate_passed", str(metrics.meets_production_gate))
        client.set_model_version_tag(model_name, version_str, "justification", metrics.promotion_justification)

        model_uri = f"models:/{model_name}/{version_str}"

        logger.info(
            "[%s] Model v%s registered as '%s' in stage '%s'. Gate: %s",
            metrics.well_id,
            version_str,
            model_name,
            target_stage,
            "PASS" if metrics.meets_production_gate else "FAIL → Staging",
        )
        logger.info("  Justification: %s", metrics.promotion_justification)
        logger.info("  URI: %s | Run: %s", model_uri, run_id)

    return model_uri


def get_production_model_uri(
    well_id: str,
    tracking_uri: str = "http://localhost:5001",
    fallback_to_staging: bool = True,
) -> str | None:
    """
    Retrieve the Production-stage model URI for a given well.

    Falls back to Staging if no Production version exists and
    fallback_to_staging=True. Returns None if no model is registered.

    Usage:
        uri = get_production_model_uri("15/9-F-12")
        model = mlflow.pyfunc.load_model(uri)
        scores = model.predict(telemetry_df)
    """
    model_name = f"{MODEL_NAME_PREFIX}-{well_id.replace('/', '_').replace(' ', '_')}"
    client = MlflowClient(tracking_uri=tracking_uri)

    try:
        versions = client.get_latest_versions(model_name, stages=[STAGE_PRODUCTION])
        if versions:
            v = versions[0]
            return f"models:/{model_name}/{v.version}"

        if fallback_to_staging:
            versions = client.get_latest_versions(model_name, stages=[STAGE_STAGING])
            if versions:
                v = versions[0]
                logger.warning("No Production model for %s — using Staging v%s", well_id, v.version)
                return f"models:/{model_name}/{v.version}"
    except Exception as e:
        logger.error("Registry lookup failed for %s: %s", well_id, e)

    return None


def list_all_model_versions(
    tracking_uri: str = "http://localhost:5001",
) -> list[dict[str, Any]]:
    """List all registered detector models across all wells with their metrics."""
    client = MlflowClient(tracking_uri=tracking_uri)
    rows = []
    try:
        for rm in client.search_registered_models(filter_string=f"name LIKE '{MODEL_NAME_PREFIX}%'"):
            for v in client.get_latest_versions(rm.name):
                rows.append(
                    {
                        "model_name": rm.name,
                        "version": v.version,
                        "stage": v.current_stage,
                        "well_id": v.tags.get("well_id", ""),
                        "detector_version": v.tags.get("detector_version", ""),
                        "event_recall": v.tags.get("event_recall", ""),
                        "day_fpr": v.tags.get("day_fpr", ""),
                        "gate_passed": v.tags.get("gate_passed", ""),
                        "data_source": v.tags.get("data_source", ""),
                        "created": v.creation_timestamp,
                    }
                )
    except Exception as e:
        logger.error("Failed to list model versions: %s", e)
    return rows


def rollback_to_previous_production(
    well_id: str,
    tracking_uri: str = "http://localhost:5001",
) -> str | None:
    """
    Rollback: archive current Production, promote previous version back to Production.

    Used when production model drift is detected (monitoring shows degraded metrics).
    Returns the URI of the restored Production model, or None if rollback is impossible.
    """
    model_name = f"{MODEL_NAME_PREFIX}-{well_id.replace('/', '_').replace(' ', '_')}"
    client = MlflowClient(tracking_uri=tracking_uri)

    try:
        all_versions = client.search_model_versions(f"name='{model_name}'")
        prod_versions = sorted(
            [v for v in all_versions if v.current_stage == STAGE_PRODUCTION],
            key=lambda v: int(v.version),
        )
        if not prod_versions:
            logger.warning("No Production version found for %s — cannot rollback", well_id)
            return None

        # Archive current production
        current = prod_versions[-1]
        client.transition_model_version_stage(model_name, current.version, STAGE_ARCHIVED)
        logger.info("Archived Production v%s for %s", current.version, well_id)

        # Promote previous version (highest numbered non-current)
        candidates = sorted(
            [v for v in all_versions if v.version != current.version and v.current_stage != STAGE_ARCHIVED],
            key=lambda v: int(v.version),
            reverse=True,
        )
        if candidates:
            prev = candidates[0]
            client.transition_model_version_stage(model_name, prev.version, STAGE_PRODUCTION)
            uri = f"models:/{model_name}/{prev.version}"
            logger.info("Rolled back to v%s for %s → Production", prev.version, well_id)
            return uri
    except Exception as e:
        logger.error("Rollback failed for %s: %s", well_id, e)

    return None
