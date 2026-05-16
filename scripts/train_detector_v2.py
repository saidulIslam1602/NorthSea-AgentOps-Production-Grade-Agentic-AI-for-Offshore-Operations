"""
Anomaly Detector v2 — Full Training, Evaluation & Business Impact Pipeline.

Evaluates both v1 (WellAnomalyDetector) and v2 (AdaptiveWellAnomalyDetector)
on real Equinor Volve production data to produce an honest A/B comparison.

Evaluation protocol (improvements over v1 script):
  - STEP anomaly injection: large overnight perturbation (5-day window) [same as v1]
  - GRADUAL anomaly injection: slow ramp over 20 days [NEW — harder, more realistic]
  - Metric suite: Precision, Recall, F1, FPR, ROC-AUC [NEW: AUC, gradual recall]
  - Economic impact: $revenue at risk per TP vs $alert cost per FP [NEW]
  - Time-series aware split: chronological 70/30 with 14-day gap [NEW]

Business impact framing:
  "Every 1 BOPD of undetected oil rate decline on F-12 costs $80/day. Over
   30 days before a planned well intervention, that's $2,400 per BOPD — and
   F-12 declines can be 500–1000 BOPD in a single equipment failure event."

Usage:
    python3 scripts/train_detector_v2.py
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.volve_loader import load_volve_daily, get_per_well_stats
from src.anomaly.detector import WellAnomalyDetector, TELEMETRY_FEATURES
from src.anomaly.adaptive_detector import AdaptiveWellAnomalyDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

VOLVE_XLSX   = Path("data/Volve_Data/Volve production data.xlsx")
OUTPUT_JSON  = Path("eval/detector_performance_v2.json")
OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)

TRAIN_SPLIT  = 0.70
GAP_DAYS     = 14         # buffer between train and test (prevents leakage)
MIN_TRAIN    = 90         # minimum training days required per well
MIN_TEST     = 40         # minimum test days required per well
OIL_PRICE    = 80.0       # USD/bbl

# ── Anomaly injection config ────────────────────────────────────────────────
# Step: large overnight perturbation (detectable by any decent detector)
STEP_ANOMALIES = {
    "water_breakthrough_step": ("water_cut_pct", 2.5, 5),
    "gor_spike_step":          ("gas_oil_ratio", 3.5, 5),
    "bhp_depletion_step":      ("bhp_psi",       0.70, 5),
    "oil_rate_collapse_step":  ("oil_rate_bopd", 0.30, 5),
}

# Gradual: slow ramp — the hard case that research-grade detectors should handle
GRADUAL_ANOMALIES = {
    "water_breakthrough_gradual": ("water_cut_pct", 1.8, 20),   # +80% over 20 days
    "bhp_depletion_gradual":      ("bhp_psi",       0.80, 20),  # −20% over 20 days
}


@dataclass
class WellEvalResult:
    well_id: str
    train_days: int
    test_days: int
    # Per anomaly category (day-level)
    step_tp: int = 0; step_fp: int = 0; step_fn: int = 0; step_tn: int = 0
    grad_tp: int = 0; grad_fp: int = 0; grad_fn: int = 0; grad_tn: int = 0
    per_type_detected: dict[str, bool] = field(default_factory=dict)
    # Event-level (operational)
    event_recall: float = 0.0
    alert_precision: float = 0.0
    tp_bursts: int = 0
    fp_bursts: int = 0
    events_detected: int = 0
    total_events: int = 0
    # For AUC
    y_true: list[int] = field(default_factory=list)
    y_score: list[float] = field(default_factory=list)
    # Economic
    total_oil_loss_bopd: float = 0.0
    total_revenue_at_risk_usd: float = 0.0

    def _metrics(self, tp, fp, fn, tn):
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        return p, r, f1, fpr

    @property
    def step_precision(self): return self._metrics(self.step_tp, self.step_fp, self.step_fn, self.step_tn)[0]
    @property
    def step_recall(self):    return self._metrics(self.step_tp, self.step_fp, self.step_fn, self.step_tn)[1]
    @property
    def step_f1(self):        return self._metrics(self.step_tp, self.step_fp, self.step_fn, self.step_tn)[2]
    @property
    def step_fpr(self):       return self._metrics(self.step_tp, self.step_fp, self.step_fn, self.step_tn)[3]
    @property
    def grad_recall(self):    return self._metrics(self.grad_tp, self.grad_fp, self.grad_fn, self.grad_tn)[1]

    @property
    def combined_precision(self):
        tp = self.step_tp + self.grad_tp
        fp = self.step_fp + self.grad_fp
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0
    @property
    def combined_recall(self):
        tp = self.step_tp + self.grad_tp
        fn = self.step_fn + self.grad_fn
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0
    @property
    def combined_f1(self):
        p, r = self.combined_precision, self.combined_recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    @property
    def combined_fpr(self):
        fp = self.step_fp + self.grad_fp
        tn = self.step_tn + self.grad_tn
        return fp / (fp + tn) if (fp + tn) > 0 else 0.0
    @property
    def roc_auc(self):
        if len(set(self.y_true)) < 2:
            return 0.5
        try:
            return float(roc_auc_score(self.y_true, self.y_score))
        except Exception:
            return 0.5


def inject_step_anomaly(
    df: pd.DataFrame, feature: str, multiplier: float, window: int,
    start_idx: int
) -> tuple[pd.DataFrame, np.ndarray]:
    df = df.copy()
    labels = np.zeros(len(df), dtype=int)
    end_idx = min(start_idx + window, len(df))
    base_val = df[feature].iloc[start_idx]
    if base_val <= 0:
        nonzero = df[feature].iloc[start_idx:].gt(0)
        if nonzero.any():
            start_idx += int(nonzero.idxmax()) - df.index[start_idx]
            end_idx = min(start_idx + window, len(df))
    df.iloc[start_idx:end_idx, df.columns.get_loc(feature)] *= multiplier
    labels[start_idx:end_idx] = 1
    return df, labels


def inject_gradual_anomaly(
    df: pd.DataFrame, feature: str, final_multiplier: float, window: int,
    start_idx: int
) -> tuple[pd.DataFrame, np.ndarray]:
    """Ramp the feature from 1.0× to final_multiplier× linearly over window days."""
    df = df.copy()
    labels = np.zeros(len(df), dtype=int)
    end_idx = min(start_idx + window, len(df))
    ramp = np.linspace(1.0, final_multiplier, end_idx - start_idx)
    for i, r in enumerate(ramp):
        idx = start_idx + i
        val = df.iloc[idx][feature]
        if val > 0:
            df.iloc[idx, df.columns.get_loc(feature)] = val * r
    labels[start_idx:end_idx] = 1
    return df, labels


def run_detector_on_test(
    detector,
    test_df: pd.DataFrame,
    labels: np.ndarray,
    is_adaptive: bool,
) -> tuple[list[int], list[float]]:
    """Feed test rows through detector; return (detections, scores)."""
    detections = []
    scores = []
    for i, (_, row) in enumerate(test_df.iterrows()):
        reading = {f: float(row.get(f, 0.0)) for f in TELEMETRY_FEATURES}
        reading["well_id"] = str(row.get("well_id", ""))
        reading["field_name"] = str(row.get("field_name", "VOLVE"))
        reading["timestamp"] = row["timestamp"]

        alert = detector.ingest(reading)
        if is_adaptive:
            detected = alert is not None
            score = alert.base_alert.anomaly_score if alert else 0.0
        else:
            detected = alert is not None
            score = alert.anomaly_score if alert else 0.0

        detections.append(1 if detected else 0)
        scores.append(score)
    return detections, scores


def compute_event_level_metrics(
    detections: list[int],
    labels: np.ndarray,
    event_windows: list[tuple[int, int]],
) -> dict:
    """
    Event-level evaluation: operationally correct for anomaly detection.

    An ALERT BURST = consecutive detected days (any cluster of detections).
    A TP burst = burst that overlaps ≥1 day of an anomaly window.
    A FP burst = burst that has no overlap with any anomaly window.
    Event recall = fraction of injected events where ≥1 day was detected.
    Alert precision = TP_bursts / (TP_bursts + FP_bursts).

    This mirrors how operators actually experience alerts: they see a cluster
    of alerts and investigate once — not once per detected day.
    """
    det = np.array(detections)

    # Identify alert bursts (consecutive detected days)
    bursts: list[tuple[int, int]] = []
    in_burst = False
    burst_start = 0
    for i, d in enumerate(det):
        if d == 1 and not in_burst:
            burst_start = i; in_burst = True
        elif d == 0 and in_burst:
            bursts.append((burst_start, i)); in_burst = False
    if in_burst:
        bursts.append((burst_start, len(det)))

    # Classify bursts
    tp_bursts = 0; fp_bursts = 0
    for bs, be in bursts:
        is_tp = any(
            not (be <= ws or bs >= we)   # burst overlaps event window
            for ws, we in event_windows
        )
        if is_tp: tp_bursts += 1
        else: fp_bursts += 1

    # Event recall: did at least 1 day in each event window get detected?
    events_detected = sum(
        1 for ws, we in event_windows
        if det[ws:we].any()
    )
    event_recall = events_detected / len(event_windows) if event_windows else 0.0
    alert_precision = tp_bursts / (tp_bursts + fp_bursts) if (tp_bursts + fp_bursts) > 0 else 0.0

    return {
        "event_recall": round(event_recall, 4),
        "alert_precision": round(alert_precision, 4),
        "tp_bursts": tp_bursts,
        "fp_bursts": fp_bursts,
        "events_detected": events_detected,
        "total_events": len(event_windows),
        "total_bursts": len(bursts),
    }


def evaluate_one_well(
    well_id: str,
    well_df: pd.DataFrame,
    rng: np.random.Generator,
    is_adaptive: bool,
) -> Optional[WellEvalResult]:

    prod = well_df[well_df["is_producing"] & (well_df["oil_rate_bopd"] > 0)].copy()
    prod = prod.sort_values("timestamp").reset_index(drop=True)

    if len(prod) < MIN_TRAIN + GAP_DAYS + MIN_TEST:
        logger.warning("Skipping %s: only %d rows", well_id, len(prod))
        return None

    split_idx = int(len(prod) * TRAIN_SPLIT)
    train_df = prod.iloc[:split_idx]
    # Gap: skip GAP_DAYS after training to prevent leakage
    test_start = min(split_idx + GAP_DAYS, len(prod) - MIN_TEST)
    test_base = prod.iloc[test_start:].reset_index(drop=True)

    if len(test_base) < MIN_TEST:
        return None

    result = WellEvalResult(
        well_id=well_id,
        train_days=len(train_df),
        test_days=len(test_base),
    )

    # ── Create and warm-up detector ────────────────────────────────────────
    if is_adaptive:
        det = AdaptiveWellAnomalyDetector(
            well_id=well_id,
            target_fpr=0.05,
            min_consecutive=2,
            zscore_window=30,
            if_contamination=0.02,
            if_n_estimators=200,
            min_train_samples=MIN_TRAIN,
        )
        det.fit(train_df)
    else:
        det = WellAnomalyDetector(
            well_id=well_id,
            window_size=min(168, len(train_df)),
            zscore_window=min(24, max(10, len(train_df) // 4)),
            retrain_interval=min(72, len(train_df) // 3),
        )
        for _, row in train_df.iterrows():
            reading = {f: float(row.get(f, 0.0)) for f in TELEMETRY_FEATURES}
            reading["well_id"] = well_id
            reading["timestamp"] = row["timestamp"]
            det.ingest(reading)

    # ── Plan injection positions (non-overlapping) ─────────────────────────
    max_window = 25   # largest anomaly window (gradual)
    # Minimum gap between events: persistence(2) + refractory(7) + max_window + buffer(3)
    min_spacing = 2 + 7 + max_window + 3   # = 37 days
    usable = len(test_base) - max_window - 2
    # Determine how many events actually fit
    max_events_possible = max(1, usable // min_spacing)

    all_anomaly_configs = list(STEP_ANOMALIES.items()) + list(GRADUAL_ANOMALIES.items())
    # Prioritise: 1 step per anomaly type first, then graduals
    n_events = min(len(all_anomaly_configs), max_events_possible)
    anomaly_configs = all_anomaly_configs[:n_events]

    positions = [2 + i * min_spacing for i in range(n_events)]
    positions = [min(p, len(test_base) - max_window - 1) for p in positions]
    rng.shuffle(positions)

    step_names = set(STEP_ANOMALIES.keys())
    step_positions = [pos for (nm, _), pos in zip(anomaly_configs, positions) if nm in step_names]
    grad_positions = [pos for (nm, _), pos in zip(anomaly_configs, positions) if nm not in step_names]

    # Only use step/grad configs that were included
    step_configs  = [(nm, cfg) for (nm, cfg), _ in zip(anomaly_configs, positions) if nm in step_names]
    grad_configs  = [(nm, cfg) for (nm, cfg), _ in zip(anomaly_configs, positions) if nm not in step_names]

    # ── Aggregate labels across all injection types ─────────────────────────
    combined_labels = np.zeros(len(test_base), dtype=int)
    type_label_map: dict[str, np.ndarray] = {}

    test_with_anomalies = test_base.copy()

    for (anom_name, (feature, mult, window)), pos in zip(step_configs, step_positions):
        test_with_anomalies, lbl = inject_step_anomaly(
            test_with_anomalies, feature, mult, window, pos
        )
        combined_labels = np.maximum(combined_labels, lbl)
        type_label_map[anom_name] = lbl

    for (anom_name, (feature, mult, window)), pos in zip(grad_configs, grad_positions):
        test_with_anomalies, lbl = inject_gradual_anomaly(
            test_with_anomalies, feature, mult, window, pos
        )
        combined_labels = np.maximum(combined_labels, lbl)
        type_label_map[anom_name] = lbl

    # ── Run detector ────────────────────────────────────────────────────────
    detections, scores = run_detector_on_test(
        det, test_with_anomalies, combined_labels, is_adaptive
    )
    det_arr = np.array(detections)

    # ── Step anomaly metrics ────────────────────────────────────────────────
    step_mask = np.zeros(len(test_base), dtype=bool)
    for nm in STEP_ANOMALIES:
        if nm in type_label_map:
            step_mask |= type_label_map[nm].astype(bool)
    step_tp = int(((det_arr == 1) & step_mask).sum())
    step_fp = int(((det_arr == 1) & ~combined_labels.astype(bool)).sum())
    step_fn = int(((det_arr == 0) & step_mask).sum())
    step_tn = int(((det_arr == 0) & ~combined_labels.astype(bool)).sum())
    result.step_tp = step_tp; result.step_fp = step_fp
    result.step_fn = step_fn; result.step_tn = step_tn

    # ── Gradual anomaly recall ──────────────────────────────────────────────
    grad_mask = np.zeros(len(test_base), dtype=bool)
    for nm in GRADUAL_ANOMALIES:
        if nm in type_label_map:
            grad_mask |= type_label_map[nm].astype(bool)
    grad_tp = int(((det_arr == 1) & grad_mask).sum())
    grad_fn = int(((det_arr == 0) & grad_mask).sum())
    result.grad_tp = grad_tp; result.grad_fn = grad_fn
    # FP already captured in step_fp (normal days are the same)

    # ── Per-type detection (event-level: any day in window detected) ────────
    for anom_name, lbl in type_label_map.items():
        anom_days = lbl.astype(bool)
        result.per_type_detected[anom_name] = bool((det_arr[anom_days] == 1).any())

    # ── Event-level metrics (operational, correct for anomaly detection) ───
    all_event_windows = []
    for nm, lbl in type_label_map.items():
        days = np.where(lbl.astype(bool))[0]
        if len(days) > 0:
            all_event_windows.append((int(days[0]), int(days[-1]) + 1))

    ev_metrics = compute_event_level_metrics(detections, combined_labels, all_event_windows)
    result.event_recall      = ev_metrics["event_recall"]
    result.alert_precision   = ev_metrics["alert_precision"]
    result.tp_bursts         = ev_metrics["tp_bursts"]
    result.fp_bursts         = ev_metrics["fp_bursts"]
    result.events_detected   = ev_metrics["events_detected"]
    result.total_events      = ev_metrics["total_events"]

    # ── AUC ────────────────────────────────────────────────────────────────
    result.y_true = combined_labels.tolist()
    result.y_score = scores

    # ── Economic impact on detected TPs ────────────────────────────────────
    if is_adaptive:
        # Re-run to collect economic signals from AdaptiveAlerts
        det2 = AdaptiveWellAnomalyDetector(
            well_id=well_id, target_fpr=0.05, min_consecutive=2,
            zscore_window=30, if_contamination=0.02, min_train_samples=MIN_TRAIN,
        )
        det2.fit(train_df)
        total_oil_loss = 0.0
        total_rev = 0.0
        for _, row in test_with_anomalies.iterrows():
            reading = {f: float(row.get(f, 0.0)) for f in TELEMETRY_FEATURES}
            reading["well_id"] = well_id
            reading["timestamp"] = row["timestamp"]
            alert = det2.ingest(reading)
            if alert and alert.economic_impact.oil_loss_bopd > 0:
                total_oil_loss += alert.economic_impact.oil_loss_bopd
                total_rev += alert.economic_impact.revenue_at_risk_usd_per_day
        result.total_oil_loss_bopd = total_oil_loss
        result.total_revenue_at_risk_usd = total_rev

    logger.info(
        "  [%s v%s] P=%.3f R=%.3f F1=%.3f FPR=%.3f AUC=%.3f GradRecall=%.3f",
        well_id, "2" if is_adaptive else "1",
        result.step_precision, result.step_recall, result.step_f1,
        result.step_fpr, result.roc_auc, result.grad_recall,
    )
    return result


def aggregate_results(results: list[WellEvalResult]) -> dict:
    def _mean(vals): return round(float(np.mean(vals)), 4) if vals else 0.0

    total_rev = sum(r.total_revenue_at_risk_usd for r in results)
    per_type_recall: dict[str, float] = {}
    all_types = set(k for r in results for k in r.per_type_detected)
    for t in all_types:
        vals = [1.0 if r.per_type_detected.get(t, False) else 0.0 for r in results]
        per_type_recall[t] = _mean(vals)

    total_events   = sum(r.total_events for r in results)
    events_detected = sum(r.events_detected for r in results)
    total_tp_bursts = sum(r.tp_bursts for r in results)
    total_fp_bursts = sum(r.fp_bursts for r in results)

    op_event_recall   = events_detected / total_events if total_events > 0 else 0.0
    op_alert_precision = total_tp_bursts / (total_tp_bursts + total_fp_bursts) if (total_tp_bursts + total_fp_bursts) > 0 else 0.0
    op_f1 = 2*op_event_recall*op_alert_precision/(op_event_recall+op_alert_precision) if (op_event_recall+op_alert_precision) > 0 else 0.0

    return {
        # Day-level metrics (for completeness)
        "day_precision": _mean([r.step_precision for r in results]),
        "day_recall":    _mean([r.step_recall for r in results]),
        "day_f1":        _mean([r.step_f1 for r in results]),
        "day_fpr":       _mean([r.step_fpr for r in results]),
        "macro_roc_auc": _mean([r.roc_auc for r in results]),
        "day_gradual_recall": _mean([r.grad_recall for r in results]),
        # Operational (event-level) metrics — the correct production metric
        "operational_event_recall":    round(op_event_recall, 4),
        "operational_alert_precision": round(op_alert_precision, 4),
        "operational_f1":              round(op_f1, 4),
        "total_events": total_events,
        "events_detected": events_detected,
        "total_tp_bursts": total_tp_bursts,
        "total_fp_bursts": total_fp_bursts,
        # Confusion counts
        "total_tp": sum(r.step_tp + r.grad_tp for r in results),
        "total_fp": sum(r.step_fp + r.grad_fp for r in results),
        "total_fn": sum(r.step_fn + r.grad_fn for r in results),
        "total_tn": sum(r.step_tn + r.grad_tn for r in results),
        "per_anomaly_type_recall": {k: round(v, 4) for k, v in per_type_recall.items()},
        "total_revenue_at_risk_usd": round(total_rev, 2),
        "n_wells": len(results),
    }


def main() -> None:
    logger.info("=== NorthSea AgentOps: Detector v1 vs v2 Evaluation ===")
    if not VOLVE_XLSX.exists():
        logger.error("Volve data not found at %s", VOLVE_XLSX); sys.exit(1)

    df = load_volve_daily(VOLVE_XLSX, producers_only=True, min_on_stream_hrs=1.0)
    logger.info("Loaded %d rows, %d wells", len(df), df["well_id"].nunique())

    rng = np.random.default_rng(seed=42)
    v1_results: list[WellEvalResult] = []
    v2_results: list[WellEvalResult] = []

    for well_id, well_df in df.groupby("well_id"):
        logger.info("=== Well: %s ===", well_id)

        logger.info("  Running v1 (baseline)...")
        r1 = evaluate_one_well(str(well_id), well_df, rng, is_adaptive=False)
        if r1:
            v1_results.append(r1)

        rng2 = np.random.default_rng(seed=42)   # same seed → same injection positions
        logger.info("  Running v2 (adaptive)...")
        r2 = evaluate_one_well(str(well_id), well_df, rng2, is_adaptive=True)
        if r2:
            v2_results.append(r2)

    if not v1_results:
        logger.error("No results"); sys.exit(1)

    agg_v1 = aggregate_results(v1_results)
    agg_v2 = aggregate_results(v2_results)

    per_well_v1 = [
        {
            "well_id": r.well_id, "train_days": r.train_days, "test_days": r.test_days,
            "precision": round(r.step_precision, 4), "recall": round(r.step_recall, 4),
            "f1": round(r.step_f1, 4), "fpr": round(r.step_fpr, 4),
            "roc_auc": round(r.roc_auc, 4), "gradual_recall": round(r.grad_recall, 4),
            "per_type_detected": r.per_type_detected,
        }
        for r in v1_results
    ]
    per_well_v2 = [
        {
            "well_id": r.well_id, "train_days": r.train_days, "test_days": r.test_days,
            "precision": round(r.step_precision, 4), "recall": round(r.step_recall, 4),
            "f1": round(r.step_f1, 4), "fpr": round(r.step_fpr, 4),
            "roc_auc": round(r.roc_auc, 4), "gradual_recall": round(r.grad_recall, 4),
            "per_type_detected": r.per_type_detected,
            "revenue_at_risk_usd": round(r.total_revenue_at_risk_usd, 2),
        }
        for r in v2_results
    ]

    output = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "data_source": "Equinor Volve Open Dataset — daily production 2007-2016",
        "train_split": TRAIN_SPLIT,
        "gap_days": GAP_DAYS,
        "step_anomaly_types": list(STEP_ANOMALIES.keys()),
        "gradual_anomaly_types": list(GRADUAL_ANOMALIES.keys()),
        "v1_aggregate": agg_v1,
        "v2_aggregate": agg_v2,
        "v1_per_well": per_well_v1,
        "v2_per_well": per_well_v2,
    }
    OUTPUT_JSON.write_text(json.dumps(output, indent=2))
    logger.info("Results written to %s", OUTPUT_JSON)

    # ── Summary table ────────────────────────────────────────────────────────
    print("\n" + "=" * 86)
    print(f"{'DETECTOR COMPARISON — VOLVE REAL DATA (v1 baseline vs v2 adaptive)':^86}")
    print("=" * 86)
    print(f"  NOTE: Operational (event-level) metrics are the production-correct measure.")
    print(f"  Day-level metrics are shown for completeness but not the primary KPI.")
    print("=" * 86)
    print(f"{'Metric':<34} {'v1 Baseline':>14} {'v2 Adaptive':>14} {'Delta':>10}")
    print("-" * 86)
    metrics = [
        ("-- OPERATIONAL (event-level) --", None),
        ("Event Recall [production KPI]",    "operational_event_recall"),
        ("Alert Precision [production KPI]", "operational_alert_precision"),
        ("Operational F1 [production KPI]",  "operational_f1"),
        ("-- DAY-LEVEL (informational) --", None),
        ("Day Precision",                    "day_precision"),
        ("Day Recall",                       "day_recall"),
        ("Day F1",                           "day_f1"),
        ("False Positive Rate (day)",        "day_fpr"),
        ("ROC-AUC",                          "macro_roc_auc"),
        ("Gradual drift recall (day)",       "day_gradual_recall"),
    ]
    for label, key in metrics:
        if key is None:
            print(f"\n  {label}")
            continue
        v1 = agg_v1[key]; v2 = agg_v2[key]
        delta = v2 - v1
        arrow = "▲" if delta > 0.005 else ("▼" if delta < -0.005 else "–")
        print(f"  {label:<32} {v1:>14.3f} {v2:>14.3f} {arrow}{abs(delta):>8.3f}")
    print("-" * 86)
    print(f"  {'Events detected / total':<32} "
          f"{agg_v1['events_detected']}/{agg_v1['total_events']:>10}  "
          f"{agg_v2['events_detected']}/{agg_v2['total_events']:>10}")
    print(f"  {'TP / FP alert bursts':<32} "
          f"{agg_v1['total_tp_bursts']}/{agg_v1['total_fp_bursts']:>10}  "
          f"{agg_v2['total_tp_bursts']}/{agg_v2['total_fp_bursts']:>10}")
    print(f"  {'Revenue at risk (v2)':<32} {'':>14} ${agg_v2['total_revenue_at_risk_usd']:>13,.0f}")
    print("=" * 86)

    print("\nPer-anomaly-type event detection (% wells where at least 1 day detected):")
    all_types = set(list(agg_v1["per_anomaly_type_recall"]) + list(agg_v2["per_anomaly_type_recall"]))
    for t in sorted(all_types):
        v1r = agg_v1["per_anomaly_type_recall"].get(t, 0)
        v2r = agg_v2["per_anomaly_type_recall"].get(t, 0)
        print(f"  {t:<38} v1={v1r:.2f}  v2={v2r:.2f}")


if __name__ == "__main__":
    main()
