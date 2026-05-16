"""
Uncertainty gate — determines whether to output a recommendation or escalate to human.

Rules (in priority order):
  1. If error occurred at any agent stage → ESCALATE (INJECTION_DETECTED or system error)
  2. If risk_level is HIGH or CRITICAL → ESCALATE (HSE_RISK) regardless of confidence
  3. If confidence_score < threshold → ESCALATE (LOW_CONFIDENCE)
  4. If evidence_coverage < threshold → ESCALATE (LOW_EVIDENCE_COVERAGE)
  5. If data was missing (no telemetry) → ESCALATE (MISSING_DATA)
  6. Otherwise → OUTPUT recommendation

All thresholds are configurable via Settings.
"""

from __future__ import annotations

import logging
from typing import Any

from src.config import get_settings
from src.schemas.domain import (
    AnomalyAlert,
    Citation,
    EscalationReason,
    InvestigationResult,
    RiskLevel,
)

logger = logging.getLogger(__name__)
settings = get_settings()


def evaluate_escalation(
    confidence_score: float,
    evidence_coverage: float,
    risk_level: RiskLevel,
    error: str | None,
    agent_steps: list[dict[str, Any]],
    alert: AnomalyAlert,
) -> tuple[bool, list[EscalationReason], str | None]:
    """
    Evaluate whether this investigation should be escalated.

    Returns:
        (should_escalate, reasons, escalation_message)
    """
    reasons: list[EscalationReason] = []

    # Rule 1: system error during investigation
    if error:
        reasons.append(EscalationReason.LOW_CONFIDENCE)
        reasons.append(EscalationReason.MISSING_DATA)

    # Rule 2: HIGH or CRITICAL HSE risk — always escalate
    if risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL):
        reasons.append(EscalationReason.HIGH_HSE_RISK)

    # Rule 3: low confidence
    if confidence_score < settings.agent_confidence_threshold:
        reasons.append(EscalationReason.LOW_CONFIDENCE)

    # Rule 4: low evidence coverage from RAG
    if evidence_coverage < settings.agent_evidence_coverage_threshold:
        reasons.append(EscalationReason.LOW_EVIDENCE_COVERAGE)

    # Rule 5: check for missing data indicators
    data_missing = any(
        "no data" in str(s.get("output", "")).lower()
        or "not found" in str(s.get("output", "")).lower()
        for s in agent_steps
    )
    if data_missing and len(agent_steps) > 0:
        reasons.append(EscalationReason.MISSING_DATA)

    # Deduplicate reasons
    unique_reasons = list(dict.fromkeys(reasons))

    should_escalate = len(unique_reasons) > 0

    message = None
    if should_escalate:
        reason_texts = {
            EscalationReason.LOW_CONFIDENCE: f"AI confidence {confidence_score:.0%} is below threshold {settings.agent_confidence_threshold:.0%}",
            EscalationReason.LOW_EVIDENCE_COVERAGE: f"Evidence coverage {evidence_coverage:.0%} is below threshold {settings.agent_evidence_coverage_threshold:.0%}",
            EscalationReason.HIGH_HSE_RISK: f"Risk level {risk_level.value} requires human review per HSE-OPS-001",
            EscalationReason.MISSING_DATA: "Insufficient telemetry or document data available",
            EscalationReason.INJECTION_DETECTED: "Potential prompt injection detected in retrieved content",
            EscalationReason.OPERATOR_OVERRIDE: "Operator manually requested human review",
        }
        reason_summary = "; ".join(
            reason_texts.get(r, r.value) for r in unique_reasons
        )
        message = (
            f"Investigation for well {alert.well_id} requires human review. "
            f"Reasons: {reason_summary}. "
            f"Per HSE-OPS-001, an engineer must review before any operational changes."
        )

    return should_escalate, unique_reasons, message


def apply_uncertainty_gate(state: dict[str, Any]) -> dict[str, Any]:
    """LangGraph node: applies escalation rules and finalises the investigation state."""
    alert: AnomalyAlert = state["alert"]
    confidence_score: float = state.get("confidence_score", 0.0)
    evidence_coverage: float = state.get("evidence_coverage", 0.0)
    risk_level: RiskLevel = state.get("risk_level", RiskLevel.MEDIUM)
    error: str | None = state.get("error")
    agent_steps: list[dict[str, Any]] = state.get("agent_steps", [])

    should_escalate, reasons, message = evaluate_escalation(
        confidence_score=confidence_score,
        evidence_coverage=evidence_coverage,
        risk_level=risk_level,
        error=error,
        agent_steps=agent_steps,
        alert=alert,
    )

    if should_escalate:
        logger.warning(
            "ESCALATING %s: confidence=%.2f, coverage=%.2f, risk=%s, reasons=%s",
            alert.well_id, confidence_score, evidence_coverage,
            risk_level.value, [r.value for r in reasons],
        )
    else:
        logger.info(
            "RECOMMENDING %s: confidence=%.2f, risk=%s",
            alert.well_id, confidence_score, risk_level.value,
        )

    return {
        "should_escalate": should_escalate,
        "escalation_reasons": reasons,
        "escalation_message": message,
    }


def build_investigation_result(state: dict[str, Any]) -> InvestigationResult:
    """Build the final InvestigationResult from completed agent state."""
    import time

    alert: AnomalyAlert = state["alert"]
    critic_review: dict[str, Any] = state.get("_critic_review", {})

    return InvestigationResult(
        investigation_id=state.get("investigation_id"),
        alert=alert,
        root_cause_hypothesis=state.get("recommendation", "Undetermined"),
        supporting_evidence=critic_review.get("supporting_evidence", state.get("evidence", [])),
        recommended_actions=critic_review.get("recommended_actions", []),
        citations=state.get("citations", []),
        confidence_score=state.get("confidence_score", 0.0),
        evidence_coverage=state.get("evidence_coverage", 0.0),
        risk_level=state.get("risk_level", RiskLevel.MEDIUM),
        should_escalate=state.get("should_escalate", True),
        escalation_reasons=state.get("escalation_reasons", []),
        escalation_message=state.get("escalation_message"),
        agent_steps=state.get("agent_steps", []),
        total_tokens_used=state.get("total_tokens", 0),
    )
