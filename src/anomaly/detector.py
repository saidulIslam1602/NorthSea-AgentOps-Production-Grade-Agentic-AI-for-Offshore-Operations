"""
Well telemetry anomaly detector.

Uses a two-stage approach:
  1. Z-score on rolling window — fast, interpretable, per-feature
  2. Isolation Forest on multi-variate feature vector — catches correlated anomalies

Outputs an AnomalyAlert with severity, anomaly score, and affected features.
"""

from __future__ import annotations

import logging
from collections import deque
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from src.schemas.domain import AnomalyAlert, SeverityLevel

logger = logging.getLogger(__name__)

TELEMETRY_FEATURES = [
    "oil_rate_bopd",
    "water_cut_pct",
    "gas_oil_ratio",
    "bhp_psi",
    "wh_temp_f",
    "choke_64ths",
]

ZSCORE_THRESHOLDS: dict[SeverityLevel, float] = {
    SeverityLevel.LOW: 2.0,
    SeverityLevel.MEDIUM: 2.5,
    SeverityLevel.HIGH: 3.5,
    SeverityLevel.CRITICAL: 5.0,
}

IF_CONTAMINATION = 0.05  # expected fraction of anomalous points in training data


class WellAnomalyDetector:
    """
    Stateful per-well anomaly detector.

    Maintains a rolling buffer of readings to compute baselines and
    retrain the Isolation Forest periodically.
    """

    def __init__(
        self,
        well_id: str,
        window_size: int = 168,  # 7 days of hourly data
        zscore_window: int = 24,  # 24-hour rolling window for Z-score
        retrain_interval: int = 72,  # retrain IF every 72 readings
    ) -> None:
        self.well_id = well_id
        self.window_size = window_size
        self.zscore_window = zscore_window
        self.retrain_interval = retrain_interval

        self._buffer: deque[dict[str, float]] = deque(maxlen=window_size)
        self._scaler = StandardScaler()
        self._iso_forest: IsolationForest | None = None
        self._readings_since_retrain = 0
        self._is_trained = False

    def _buffer_df(self) -> pd.DataFrame:
        return pd.DataFrame(list(self._buffer))[TELEMETRY_FEATURES]

    def _compute_zscore(self, reading: dict[str, float]) -> dict[str, float]:
        """Compute Z-scores for each feature over the recent rolling window."""
        if len(self._buffer) < self.zscore_window:
            return dict.fromkeys(TELEMETRY_FEATURES, 0.0)

        window_data = list(self._buffer)[-self.zscore_window :]
        df = pd.DataFrame(window_data)[TELEMETRY_FEATURES]

        means = df.mean()
        stds = df.std().replace(0, np.nan)  # avoid division by zero

        zscores: dict[str, float] = {}
        for feat in TELEMETRY_FEATURES:
            if feat not in reading:
                zscores[feat] = 0.0
                continue
            mean_f = float(means.get(feat, 0.0))
            rv = float(reading[feat])
            std_raw = float(stds.get(feat, np.nan))
            if np.isnan(std_raw) or std_raw == 0.0:
                diff = abs(rv - mean_f)
                zscores[feat] = 0.0 if diff < 1e-6 else 12.0
            else:
                zscores[feat] = abs((rv - mean_f) / std_raw)

        return zscores

    def _train_isolation_forest(self) -> None:
        if len(self._buffer) < 50:
            return

        df = self._buffer_df().ffill().dropna()
        if len(df) < 20:
            return

        scaled_rows = self._scaler.fit_transform(df.values)
        self._iso_forest = IsolationForest(
            contamination=IF_CONTAMINATION,
            n_estimators=100,
            random_state=42,
        )
        self._iso_forest.fit(scaled_rows)
        self._is_trained = True
        self._readings_since_retrain = 0
        logger.debug("Isolation Forest retrained on %d samples for %s", len(df), self.well_id)

    def _isolation_score(self, reading: dict[str, float]) -> float:
        """Return anomaly score from Isolation Forest (0=normal, 1=highly anomalous)."""
        if not self._is_trained or self._iso_forest is None:
            return 0.0

        values = np.array([[reading.get(f, 0.0) for f in TELEMETRY_FEATURES]])
        scaled = self._scaler.transform(values)
        # IF decision_function returns negative for anomalies; convert to 0-1 score
        raw_score = self._iso_forest.decision_function(scaled)[0]
        # Normalise: score close to -1 = highly anomalous, +1 = normal
        anomaly_score = max(0.0, min(1.0, (-raw_score + 0.3) / 0.6))
        return float(anomaly_score)

    def _classify_severity(self, max_zscore: float, if_score: float) -> SeverityLevel:
        """Classify severity from combined Z-score and IF score."""
        combined = max(max_zscore / 6.0, if_score)  # normalise Z-score to ~0-1

        if combined >= 0.85 or max_zscore >= ZSCORE_THRESHOLDS[SeverityLevel.CRITICAL]:
            return SeverityLevel.CRITICAL
        if combined >= 0.65 or max_zscore >= ZSCORE_THRESHOLDS[SeverityLevel.HIGH]:
            return SeverityLevel.HIGH
        if combined >= 0.45 or max_zscore >= ZSCORE_THRESHOLDS[SeverityLevel.MEDIUM]:
            return SeverityLevel.MEDIUM
        return SeverityLevel.LOW

    def ingest(self, reading: dict[str, Any]) -> AnomalyAlert | None:
        """
        Ingest a new telemetry reading.

        Returns an AnomalyAlert if an anomaly is detected, else None.
        """
        clean: dict[str, float] = {f: float(reading.get(f, 0.0)) for f in TELEMETRY_FEATURES}

        zscores = self._compute_zscore(clean)
        affected_features = [f for f, z in zscores.items() if z >= ZSCORE_THRESHOLDS[SeverityLevel.LOW]]

        # Update buffer AFTER computing Z-scores (don't contaminate baseline)
        self._buffer.append(clean)
        self._readings_since_retrain += 1

        # Retrain periodically
        if self._readings_since_retrain >= self.retrain_interval:
            self._train_isolation_forest()

        # Initial training after enough data
        if not self._is_trained and len(self._buffer) >= 50:
            self._train_isolation_forest()

        if_score = self._isolation_score(clean)
        max_zscore = max(zscores.values(), default=0.0)

        if not affected_features and if_score < 0.35:
            return None  # Normal reading

        severity = self._classify_severity(max_zscore, if_score)
        anomaly_score = min(1.0, max(if_score, max_zscore / 6.0))

        # Compute baselines from recent buffer (exclude last reading)
        if len(self._buffer) > 1:
            recent = list(self._buffer)[-24:-1] if len(self._buffer) > 25 else list(self._buffer)[:-1]
            df_recent = pd.DataFrame(recent)[TELEMETRY_FEATURES]
            baseline_values = df_recent.mean().to_dict()
        else:
            baseline_values = clean.copy()

        deviation_pct = {}
        for f in affected_features:
            if baseline_values.get(f, 0) != 0:
                deviation_pct[f] = round(100 * (clean[f] - baseline_values[f]) / abs(baseline_values[f]), 1)

        description = _build_description(
            well_id=reading.get("well_id", self.well_id),
            severity=severity,
            affected_features=affected_features,
            deviation_pct=deviation_pct,
            current_values=clean,
        )

        return AnomalyAlert(
            timestamp=reading.get("timestamp", datetime.now(UTC)),
            well_id=reading.get("well_id", self.well_id),
            field_name=reading.get("field_name", "Unknown"),
            severity=severity,
            anomaly_score=round(anomaly_score, 4),
            affected_features=affected_features,
            baseline_values={k: round(v, 2) for k, v in baseline_values.items() if k in affected_features},
            current_values={k: round(clean[k], 2) for k in affected_features},
            deviation_pct=deviation_pct,
            description=description,
        )


def _build_description(
    well_id: str,
    severity: SeverityLevel,
    affected_features: list[str],
    deviation_pct: dict[str, float],
    current_values: dict[str, float],
) -> str:
    """Build a human-readable anomaly description."""
    feature_desc = ", ".join(f"{f.replace('_', ' ')} ({deviation_pct.get(f, 0):+.1f}%)" for f in affected_features)

    domain_hints: list[str] = []
    if "water_cut_pct" in affected_features and deviation_pct.get("water_cut_pct", 0) > 0:
        domain_hints.append("possible water breakthrough or coning")
    if "oil_rate_bopd" in affected_features and deviation_pct.get("oil_rate_bopd", 0) < -20:
        if current_values.get("bhp_psi", 3000) < 2500:
            domain_hints.append("possible pump failure or liquid loading")
        else:
            domain_hints.append("possible choke restriction or separator upset")
    if "gas_oil_ratio" in affected_features and deviation_pct.get("gas_oil_ratio", 0) > 50:
        domain_hints.append("possible gas coning or separator malfunction")
    if "bhp_psi" in affected_features and deviation_pct.get("bhp_psi", 0) < -15:
        domain_hints.append("possible reservoir pressure depletion or pump issue")

    hint_str = f" Possible causes: {'; '.join(domain_hints)}." if domain_hints else ""

    return f"{severity.value} anomaly on {well_id}: abnormal {feature_desc}.{hint_str}"
