"""
Confidence calibration metrics for the Investigation Agent.

Implements Expected Calibration Error (ECE) — the primary metric for
"calibrated uncertainty estimation for agent decision points" (Aker BP Track A).

What calibration means:
  If the agent says confidence=0.8, it should be correct ~80% of the time.
  A perfectly calibrated agent has ECE = 0.0.
  An overconfident agent has ECE > 0 with high confidence on wrong answers.

References:
  - Guo et al., "On Calibration of Modern Neural Networks" (ICML 2017)
  - Naeini et al., "Obtaining Well Calibrated Probabilities Using Bayesian Binning" (AAAI 2015)
  - Lichtenstein & Fischhoff, calibration psychology literature (1977)

Usage:
  from eval.calibration import compute_ece, CalibrationResult
  result = compute_ece(confidences=[0.9, 0.7, 0.6], correct=[True, True, False])
  print(result)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class CalibrationBin:
    """A single bin in the ECE calculation."""

    bin_lower: float
    bin_upper: float
    count: int
    avg_confidence: float
    accuracy: float
    calibration_error: float  # |accuracy - avg_confidence|

    @property
    def label(self) -> str:
        return f"[{self.bin_lower:.1f}, {self.bin_upper:.1f})"


@dataclass
class CalibrationResult:
    """Full calibration analysis for a set of predictions."""

    ece: float  # Expected Calibration Error
    mce: float  # Maximum Calibration Error
    ace: float  # Average Calibration Error (unweighted)
    overconfidence_rate: float  # fraction of bins where conf > accuracy
    bins: list[CalibrationBin]
    n_samples: int
    n_bins: int
    avg_confidence: float
    avg_accuracy: float
    calibration_grade: str  # "EXCELLENT" | "GOOD" | "ACCEPTABLE" | "POOR"

    def __str__(self) -> str:
        return (
            f"CalibrationResult(ECE={self.ece:.4f}, MCE={self.mce:.4f}, "
            f"grade={self.calibration_grade}, n={self.n_samples})"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "ece": round(self.ece, 4),
            "mce": round(self.mce, 4),
            "ace": round(self.ace, 4),
            "overconfidence_rate": round(self.overconfidence_rate, 3),
            "avg_confidence": round(self.avg_confidence, 3),
            "avg_accuracy": round(self.avg_accuracy, 3),
            "n_samples": self.n_samples,
            "n_bins": self.n_bins,
            "calibration_grade": self.calibration_grade,
            "bins": [
                {
                    "range": b.label,
                    "count": b.count,
                    "avg_confidence": round(b.avg_confidence, 3),
                    "accuracy": round(b.accuracy, 3),
                    "calibration_error": round(b.calibration_error, 3),
                }
                for b in self.bins
            ],
        }


def _grade_ece(ece: float) -> str:
    """Map ECE to a human-readable calibration grade."""
    if ece <= 0.03:
        return "EXCELLENT"
    if ece <= 0.07:
        return "GOOD"
    if ece <= 0.12:
        return "ACCEPTABLE"
    return "POOR"


def compute_ece(
    confidences: list[float],
    correct: list[bool],
    n_bins: int = 10,
) -> CalibrationResult:
    """
    Compute Expected Calibration Error (ECE) using equal-width bins.

    ECE = Σ_b (|B_b| / n) * |acc(B_b) - conf(B_b)|

    Where:
      - B_b is the set of samples in bin b
      - acc(B_b) is fraction of correct predictions in that bin
      - conf(B_b) is average confidence in that bin
      - n is total number of samples

    Args:
        confidences: Model confidence scores in [0, 1].
        correct: Whether each prediction was correct (matches ground truth).
        n_bins: Number of equal-width bins (default 10 = 0.0-0.1, 0.1-0.2, ...).

    Returns:
        CalibrationResult with ECE, MCE, bin breakdown, and grade.
    """
    assert len(confidences) == len(correct), "confidences and correct must have same length"
    n = len(confidences)
    if n == 0:
        raise ValueError("Cannot compute ECE on empty input")

    confidences_arr = np.array(confidences, dtype=float)
    correct_arr = np.array(correct, dtype=float)

    # Bin boundaries
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    bins: list[CalibrationBin] = []
    ece_acc = 0.0
    calibration_errors: list[float] = []
    overconfident_bins = 0

    for i in range(n_bins):
        lower = bin_boundaries[i]
        upper = bin_boundaries[i + 1]

        # Include upper boundary in last bin
        if i == n_bins - 1:
            mask = (confidences_arr >= lower) & (confidences_arr <= upper)
        else:
            mask = (confidences_arr >= lower) & (confidences_arr < upper)

        bin_count = int(mask.sum())
        if bin_count == 0:
            continue

        avg_conf = float(confidences_arr[mask].mean())
        accuracy = float(correct_arr[mask].mean())
        cal_error = abs(accuracy - avg_conf)

        ece_acc += (bin_count / n) * cal_error
        calibration_errors.append(cal_error)

        if avg_conf > accuracy:
            overconfident_bins += 1

        bins.append(
            CalibrationBin(
                bin_lower=lower,
                bin_upper=upper,
                count=bin_count,
                avg_confidence=avg_conf,
                accuracy=accuracy,
                calibration_error=cal_error,
            )
        )

    mce = max(calibration_errors) if calibration_errors else 0.0
    ace = sum(calibration_errors) / len(calibration_errors) if calibration_errors else 0.0
    overconfidence_rate = overconfident_bins / len(bins) if bins else 0.0

    return CalibrationResult(
        ece=ece_acc,
        mce=mce,
        ace=ace,
        overconfidence_rate=overconfidence_rate,
        bins=bins,
        n_samples=n,
        n_bins=n_bins,
        avg_confidence=float(confidences_arr.mean()),
        avg_accuracy=float(correct_arr.mean()),
        calibration_grade=_grade_ece(ece_acc),
    )


def compute_escalation_calibration(
    confidences: list[float],
    should_escalate: list[bool],
    actually_required_escalation: list[bool],
    confidence_threshold: float = 0.75,
) -> dict[str, Any]:
    """
    Calibrate the escalation gate specifically.

    The agent escalates when confidence < threshold. Measures whether the agent
    escalates in the right cases — false positives (unnecessary escalation) and
    false negatives (missed escalations for cases that needed human review) are
    both costly in operations.

    Args:
        confidences: Agent confidence scores.
        should_escalate: Whether the agent decided to escalate.
        actually_required_escalation: Ground truth (did this case need a human?).
        confidence_threshold: The threshold below which the agent escalates.

    Returns:
        dict with precision, recall, F1, and ECE for the escalation decision.
    """
    assert len(confidences) == len(should_escalate) == len(actually_required_escalation)

    _ = confidence_threshold

    predicted = np.array(should_escalate, dtype=bool)
    actual = np.array(actually_required_escalation, dtype=bool)

    tp = int((predicted & actual).sum())
    fp = int((predicted & ~actual).sum())
    fn = int((~predicted & actual).sum())
    tn = int((~predicted & ~actual).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # ECE for escalation: treat "should escalate" as confidence=1-conf
    escalation_probs = [1.0 - c for c in confidences]
    ece_result = compute_ece(escalation_probs, list(actual))

    # Miss rate for safety: missing a real escalation is worse than a false alarm
    miss_rate = fn / (tp + fn) if (tp + fn) > 0 else 0.0
    false_alarm_rate = fp / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1_score": round(f1, 3),
        "specificity": round(specificity, 3),
        "miss_rate": round(miss_rate, 3),
        "false_alarm_rate": round(false_alarm_rate, 3),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "escalation_ece": round(ece_result.ece, 4),
        "safety_note": (
            f"Miss rate {miss_rate:.1%} — each false negative is a case where "
            f"the agent failed to escalate a situation requiring human judgment."
        ),
    }


def compute_brier_score(confidences: list[float], correct: list[bool]) -> float:
    """
    Compute Brier score — mean squared error of probabilistic predictions.

    Brier score ∈ [0, 1]. Lower is better.
    A perfect model scores 0.0; a random model scores 0.25; worst case 1.0.
    """
    n = len(confidences)
    if n == 0:
        return float("nan")
    return float(sum((c - float(o)) ** 2 for c, o in zip(confidences, correct, strict=False)) / n)


def log_calibration_to_mlflow(
    result: CalibrationResult,
    escalation_metrics: dict[str, Any] | None = None,
    brier_score: float | None = None,
    experiment_name: str = "northsea-agentops-eval",
    run_name: str = "calibration",
) -> None:
    """Log calibration metrics to MLflow."""
    try:
        import mlflow

        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=run_name):
            mlflow.log_metric("calibration_ece", result.ece)
            mlflow.log_metric("calibration_mce", result.mce)
            mlflow.log_metric("calibration_ace", result.ace)
            mlflow.log_metric("overconfidence_rate", result.overconfidence_rate)
            mlflow.log_metric("avg_confidence", result.avg_confidence)
            mlflow.log_metric("avg_accuracy", result.avg_accuracy)
            mlflow.log_param("calibration_grade", result.calibration_grade)
            mlflow.log_param("n_samples", result.n_samples)

            if brier_score is not None:
                mlflow.log_metric("brier_score", brier_score)

            if escalation_metrics:
                for k, v in escalation_metrics.items():
                    if isinstance(v, (int, float)):
                        mlflow.log_metric(f"escalation_{k}", v)

            mlflow.log_dict(result.to_dict(), "calibration_report.json")
    except Exception as exc:
        print(f"MLflow logging failed: {exc}")


# ── Offline test cases for calibration validation ─────────────────────────────

CALIBRATION_VALIDATION_CASES: list[dict[str, Any]] = [
    # (confidence, correct, should_escalate, actually_required_escalation)
    # These are deterministic ground-truth cases derivable from the golden testset
    # without calling an LLM — used for offline calibration sanity checks.
    {
        "description": "High confidence ESP diagnosis with strong evidence",
        "confidence": 0.88,
        "correct": True,
        "should_escalate": False,
        "actually_required_escalation": False,
    },
    {
        "description": "Water breakthrough — ambiguous source (aquifer vs injector)",
        "confidence": 0.65,
        "correct": True,
        "should_escalate": True,
        "actually_required_escalation": True,
    },
    {
        "description": "Separator GOR spike — Critic correctly identifies metering artefact",
        "confidence": 0.82,
        "correct": True,
        "should_escalate": False,
        "actually_required_escalation": False,
    },
    {
        "description": "ESD-related event — always requires escalation per HSE-OPS-005",
        "confidence": 0.91,
        "correct": True,
        "should_escalate": True,  # correctly escalated (HSE rule)
        "actually_required_escalation": True,
    },
    {
        "description": "Overconfident choke diagnosis with sparse evidence",
        "confidence": 0.85,
        "correct": False,  # wrong root cause
        "should_escalate": False,  # missed escalation
        "actually_required_escalation": True,
    },
    {
        "description": "Low confidence — missing telemetry data",
        "confidence": 0.45,
        "correct": False,
        "should_escalate": True,
        "actually_required_escalation": True,
    },
    {
        "description": "Wellhead pressure surge — correctly identified reservoir cause",
        "confidence": 0.78,
        "correct": True,
        "should_escalate": False,
        "actually_required_escalation": False,
    },
    {
        "description": "Slugging pattern — uncertain flowline vs separator origin",
        "confidence": 0.60,
        "correct": True,
        "should_escalate": True,
        "actually_required_escalation": False,  # false alarm (unnecessary escalation)
    },
]


def run_offline_calibration_check() -> None:
    """Run calibration check on static validation cases and print results."""
    confidences = [c["confidence"] for c in CALIBRATION_VALIDATION_CASES]
    correct = [c["correct"] for c in CALIBRATION_VALIDATION_CASES]
    should_escalate = [c["should_escalate"] for c in CALIBRATION_VALIDATION_CASES]
    actually_required = [c["actually_required_escalation"] for c in CALIBRATION_VALIDATION_CASES]

    result = compute_ece(confidences, correct)
    brier = compute_brier_score(confidences, correct)
    escalation = compute_escalation_calibration(confidences, should_escalate, actually_required)

    print("\n=== Offline Calibration Check ===")
    print(f"Samples: {result.n_samples}")
    print(f"ECE: {result.ece:.4f}  (grade: {result.calibration_grade})")
    print(f"MCE: {result.mce:.4f}")
    print(f"Brier Score: {brier:.4f}")
    print(f"Avg Confidence: {result.avg_confidence:.3f}")
    print(f"Avg Accuracy:   {result.avg_accuracy:.3f}")
    print("\nEscalation Gate:")
    print(f"  Precision: {escalation['precision']:.3f}")
    print(f"  Recall:    {escalation['recall']:.3f}  (miss rate: {escalation['miss_rate']:.3f})")
    print(f"  F1:        {escalation['f1_score']:.3f}")
    print(f"  Safety: {escalation['safety_note']}")
    print("\nBins:")
    for b in result.bins:
        print(
            f"  {b.label}: n={b.count}, conf={b.avg_confidence:.2f}, "
            f"acc={b.accuracy:.2f}, error={b.calibration_error:.2f}"
        )


if __name__ == "__main__":
    run_offline_calibration_check()
