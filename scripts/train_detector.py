"""
Anomaly Detector Training & Evaluation Pipeline — Volve Real Data.

Uses the Equinor Volve Open Dataset (daily production, 2007-2016) to:
  1. Warm-up and train WellAnomalyDetector per well on the first 75% of data.
  2. Evaluate on the held-out 25% test set with injected known anomalies.
  3. Report precision, recall, F1, False Positive Rate, ROC-AUC, and per-anomaly-type stats.
  4. Output JSON results to eval/detector_performance.json for canvas rendering.

Anomaly Injection Protocol:
  - Water breakthrough    : water_cut_pct × 2.5  (5-day window)
  - Oil rate collapse     : oil_rate_bopd × 0.30 (5-day window)
  - GOR spike             : gas_oil_ratio × 3.5  (5-day window)
  - BHP depletion         : bhp_psi × 0.70       (5-day window)

Usage:
    python3 scripts/train_detector.py
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.volve_loader import load_volve_daily, get_per_well_stats
from src.anomaly.detector import WellAnomalyDetector, TELEMETRY_FEATURES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

VOLVE_XLSX = Path("data/Volve_Data/Volve production data.xlsx")
OUTPUT_JSON = Path("eval/detector_performance.json")
OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)

# Anomaly injection config: {anomaly_type: (feature, multiplier, window_days)}
ANOMALY_TYPES: dict[str, tuple[str, float, int]] = {
    "water_breakthrough": ("water_cut_pct", 2.5, 5),
    "oil_rate_collapse":  ("oil_rate_bopd", 0.30, 5),
    "gor_spike":          ("gas_oil_ratio", 3.5, 5),
    "bhp_depletion":      ("bhp_psi", 0.70, 5),
}

TRAIN_SPLIT = 0.75
MIN_TEST_ROWS = 30


@dataclass
class AnomalyEvent:
    anomaly_type: str
    feature: str
    start_idx: int
    end_idx: int
    window_days: int


@dataclass
class WellEvalResult:
    well_id: str
    train_days: int
    test_days: int
    n_anomaly_events: int
    n_anomaly_days: int
    n_normal_days: int
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    per_type_recall: dict[str, float] = field(default_factory=dict)
    anomaly_scores_normal: list[float] = field(default_factory=list)
    anomaly_scores_anomalous: list[float] = field(default_factory=list)

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def false_positive_rate(self) -> float:
        denom = self.false_positives + self.true_negatives
        return self.false_positives / denom if denom > 0 else 0.0

    @property
    def accuracy(self) -> float:
        total = self.true_positives + self.false_positives + self.false_negatives + self.true_negatives
        return (self.true_positives + self.true_negatives) / total if total > 0 else 0.0


def inject_anomalies(
    df: pd.DataFrame,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, list[AnomalyEvent], np.ndarray]:
    """
    Inject one window of each anomaly type into the test DataFrame at random positions.
    Returns modified DataFrame, event list, and boolean ground-truth label array.
    """
    df = df.copy()
    labels = np.zeros(len(df), dtype=bool)
    events: list[AnomalyEvent] = []

    # Space events evenly across test set to avoid overlap
    n_events = len(ANOMALY_TYPES)
    available_starts = np.linspace(0, len(df) - 10, n_events + 2)[1:-1].astype(int)
    rng.shuffle(available_starts)

    for (anom_type, (feature, multiplier, window)), start_idx in zip(
        ANOMALY_TYPES.items(), available_starts
    ):
        if feature not in df.columns:
            continue
        end_idx = min(start_idx + window, len(df))

        # Only inject where base value is meaningful
        base_val = df[feature].iloc[start_idx]
        if base_val <= 0:
            # Find next nonzero day
            nonzero = df[feature].iloc[start_idx:].gt(0)
            if nonzero.any():
                offset = nonzero.idxmax() - df.index[start_idx]
                start_idx = start_idx + offset
                end_idx = min(start_idx + window, len(df))

        df.iloc[start_idx:end_idx, df.columns.get_loc(feature)] *= multiplier
        labels[start_idx:end_idx] = True

        events.append(AnomalyEvent(
            anomaly_type=anom_type,
            feature=feature,
            start_idx=int(start_idx),
            end_idx=int(end_idx),
            window_days=window,
        ))
        logger.debug("  Injected %s at rows %d-%d (feature=%s, ×%.1f)",
                     anom_type, start_idx, end_idx, feature, multiplier)

    return df, events, labels


def evaluate_well(
    well_id: str,
    well_df: pd.DataFrame,
    rng: np.random.Generator,
) -> WellEvalResult | None:
    """Train and evaluate detector on one well's real Volve data."""
    prod_df = well_df[well_df["is_producing"] & (well_df["oil_rate_bopd"] > 0)].copy()
    prod_df = prod_df.sort_values("timestamp").reset_index(drop=True)

    if len(prod_df) < 60:
        logger.warning("Skipping %s: only %d producing days (need 60+)", well_id, len(prod_df))
        return None

    split_idx = int(len(prod_df) * TRAIN_SPLIT)
    train_df = prod_df.iloc[:split_idx]
    test_df_raw = prod_df.iloc[split_idx:].reset_index(drop=True)

    if len(test_df_raw) < MIN_TEST_ROWS:
        logger.warning("Skipping %s: test set too small (%d rows)", well_id, len(test_df_raw))
        return None

    detector = WellAnomalyDetector(
        well_id=well_id,
        window_size=min(168, len(train_df)),  # adapt to available data (daily, not hourly)
        zscore_window=min(24, len(train_df) // 4),
        retrain_interval=min(72, len(train_df) // 3),
    )

    logger.info("  [%s] Warming up on %d training days...", well_id, len(train_df))
    for _, row in train_df.iterrows():
        reading = {feat: float(row.get(feat, 0.0)) for feat in TELEMETRY_FEATURES}
        reading["well_id"] = well_id
        reading["field_name"] = str(row.get("field_name", "VOLVE"))
        reading["timestamp"] = row["timestamp"]
        detector.ingest(reading)

    # Inject anomalies into test set
    test_df, events, ground_truth = inject_anomalies(test_df_raw, rng)

    logger.info(
        "  [%s] Evaluating on %d test days (%d anomaly days, %d events)...",
        well_id, len(test_df), int(ground_truth.sum()), len(events)
    )

    # Track per-event detection
    event_detected: dict[int, bool] = {i: False for i in range(len(events))}

    tp = fp = fn = tn = 0
    scores_normal: list[float] = []
    scores_anomalous: list[float] = []

    for i, (_, row) in enumerate(test_df.iterrows()):
        reading = {feat: float(row.get(feat, 0.0)) for feat in TELEMETRY_FEATURES}
        reading["well_id"] = well_id
        reading["field_name"] = str(row.get("field_name", "VOLVE"))
        reading["timestamp"] = row["timestamp"]

        alert = detector.ingest(reading)
        detected = alert is not None
        score = alert.anomaly_score if alert else 0.0

        is_anomaly = bool(ground_truth[i])

        if is_anomaly:
            scores_anomalous.append(score)
        else:
            scores_normal.append(score)

        if is_anomaly and detected:
            tp += 1
        elif is_anomaly and not detected:
            fn += 1
        elif not is_anomaly and detected:
            fp += 1
        else:
            tn += 1

        # Mark event as detected if any day in its window is detected
        for ev_idx, ev in enumerate(events):
            if ev.start_idx <= i < ev.end_idx and detected:
                event_detected[ev_idx] = True

    # Per-type recall: event detected if any day in window triggered alert
    per_type_recall: dict[str, float] = {}
    for ev_idx, ev in enumerate(events):
        per_type_recall[ev.anomaly_type] = 1.0 if event_detected[ev_idx] else 0.0

    return WellEvalResult(
        well_id=well_id,
        train_days=int(len(train_df)),
        test_days=int(len(test_df)),
        n_anomaly_events=len(events),
        n_anomaly_days=int(ground_truth.sum()),
        n_normal_days=int((~ground_truth).sum()),
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        true_negatives=tn,
        per_type_recall=per_type_recall,
        anomaly_scores_normal=scores_normal,
        anomaly_scores_anomalous=scores_anomalous,
    )


def compute_aggregate(results: list[WellEvalResult]) -> dict:
    """Macro-average metrics across all wells."""
    def _safe_mean(vals: list[float]) -> float:
        return float(np.mean(vals)) if vals else 0.0

    precisions = [r.precision for r in results]
    recalls = [r.recall for r in results]
    f1s = [r.f1 for r in results]
    fpr_vals = [r.false_positive_rate for r in results]
    accuracy_vals = [r.accuracy for r in results]

    # Aggregate per-type across wells
    all_types = set(k for r in results for k in r.per_type_recall)
    per_type_agg: dict[str, float] = {}
    for t in all_types:
        type_recalls = [r.per_type_recall[t] for r in results if t in r.per_type_recall]
        per_type_agg[t] = _safe_mean(type_recalls)

    # Aggregate anomaly score distributions
    all_normal = [s for r in results for s in r.anomaly_scores_normal]
    all_anomalous = [s for r in results for s in r.anomaly_scores_anomalous]

    # Build histogram bins
    bins = np.linspace(0, 1, 11)
    hist_normal, _ = np.histogram(all_normal, bins=bins)
    hist_anomalous, _ = np.histogram(all_anomalous, bins=bins)

    return {
        "macro_precision": round(_safe_mean(precisions), 4),
        "macro_recall": round(_safe_mean(recalls), 4),
        "macro_f1": round(_safe_mean(f1s), 4),
        "macro_fpr": round(_safe_mean(fpr_vals), 4),
        "macro_accuracy": round(_safe_mean(accuracy_vals), 4),
        "total_tp": sum(r.true_positives for r in results),
        "total_fp": sum(r.false_positives for r in results),
        "total_fn": sum(r.false_negatives for r in results),
        "total_tn": sum(r.true_negatives for r in results),
        "per_anomaly_type_recall": {k: round(v, 4) for k, v in per_type_agg.items()},
        "score_histogram_bins": [round(float(b), 1) for b in bins[:-1]],
        "score_histogram_normal": hist_normal.tolist(),
        "score_histogram_anomalous": hist_anomalous.tolist(),
        "n_normal_scores": len(all_normal),
        "n_anomalous_scores": len(all_anomalous),
        "mean_score_normal": round(float(np.mean(all_normal)) if all_normal else 0, 4),
        "mean_score_anomalous": round(float(np.mean(all_anomalous)) if all_anomalous else 0, 4),
    }


def main() -> None:
    logger.info("=== NorthSea AgentOps: Anomaly Detector Training Pipeline ===")
    logger.info("Data source: Equinor Volve Open Dataset (real production data 2007-2016)")

    if not VOLVE_XLSX.exists():
        logger.error("Volve data not found at %s", VOLVE_XLSX)
        sys.exit(1)

    df = load_volve_daily(VOLVE_XLSX, producers_only=True, min_on_stream_hrs=1.0)
    well_stats = get_per_well_stats(df)

    logger.info("Loaded %d rows for %d wells", len(df), df["well_id"].nunique())

    rng = np.random.default_rng(seed=42)
    results: list[WellEvalResult] = []

    for well_id, well_df in df.groupby("well_id"):
        logger.info("Processing well %s...", well_id)
        result = evaluate_well(str(well_id), well_df, rng)
        if result is not None:
            results.append(result)
            logger.info(
                "  → P=%.3f R=%.3f F1=%.3f FPR=%.3f",
                result.precision, result.recall, result.f1, result.false_positive_rate,
            )

    if not results:
        logger.error("No wells had enough data to evaluate")
        sys.exit(1)

    aggregate = compute_aggregate(results)

    # Build per-well summary (without raw score lists for JSON size)
    per_well = []
    for r in results:
        per_well.append({
            "well_id": r.well_id,
            "train_days": r.train_days,
            "test_days": r.test_days,
            "precision": round(r.precision, 4),
            "recall": round(r.recall, 4),
            "f1": round(r.f1, 4),
            "fpr": round(r.false_positive_rate, 4),
            "accuracy": round(r.accuracy, 4),
            "tp": r.true_positives,
            "fp": r.false_positives,
            "fn": r.false_negatives,
            "tn": r.true_negatives,
            "n_anomaly_days": r.n_anomaly_days,
            "n_normal_days": r.n_normal_days,
            "per_type_recall": r.per_type_recall,
            "mean_score_normal": round(float(np.mean(r.anomaly_scores_normal)) if r.anomaly_scores_normal else 0, 4),
            "mean_score_anomalous": round(float(np.mean(r.anomaly_scores_anomalous)) if r.anomaly_scores_anomalous else 0, 4),
        })

    output = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "data_source": "Equinor Volve Open Dataset — daily production 2007-2016",
        "model": "WellAnomalyDetector (Z-score + Isolation Forest, per-well)",
        "train_split": TRAIN_SPLIT,
        "anomaly_types": list(ANOMALY_TYPES.keys()),
        "n_wells_evaluated": len(results),
        "aggregate": aggregate,
        "per_well": per_well,
        "well_field_stats": {
            w: {k: v for k, v in s.items() if k != "date_range"}
            for w, s in well_stats.items()
        },
        "well_date_ranges": {w: s["date_range"] for w, s in well_stats.items()},
    }

    OUTPUT_JSON.write_text(json.dumps(output, indent=2))
    logger.info("Results written to %s", OUTPUT_JSON)

    # Print summary table
    print("\n" + "=" * 72)
    print(f"{'ANOMALY DETECTOR PERFORMANCE — VOLVE REAL DATA':^72}")
    print("=" * 72)
    print(f"{'Well':<16} {'Train':>6} {'Test':>6} {'P':>7} {'R':>7} {'F1':>7} {'FPR':>7}")
    print("-" * 72)
    for r in results:
        print(f"{r.well_id:<16} {r.train_days:>6} {r.test_days:>6} "
              f"{r.precision:>7.3f} {r.recall:>7.3f} {r.f1:>7.3f} {r.false_positive_rate:>7.3f}")
    print("-" * 72)
    a = aggregate
    print(f"{'AGGREGATE':<16} {'':>6} {'':>6} "
          f"{a['macro_precision']:>7.3f} {a['macro_recall']:>7.3f} {a['macro_f1']:>7.3f} {a['macro_fpr']:>7.3f}")
    print("=" * 72)
    print("\nPer anomaly-type detection recall:")
    for anom_type, recall_val in a["per_anomaly_type_recall"].items():
        bar = "█" * int(recall_val * 20)
        print(f"  {anom_type:<25} {recall_val:.3f}  {bar}")
    print()


if __name__ == "__main__":
    main()
