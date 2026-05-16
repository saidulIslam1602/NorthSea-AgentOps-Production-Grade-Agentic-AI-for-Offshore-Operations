"""
Critic agent — validates the investigation recommendation before output.

Pattern: Critic-Actor (critic validates, does NOT generate the recommendation).
  - Reviews all evidence and plan step results
  - Generates the root cause hypothesis and recommended actions
  - Scores: confidence (0-1), evidence quality, logical consistency
  - Identifies missing evidence or logical gaps
  - Assesses HSE/risk level

The Critic is the FINAL agent before the uncertainty gate.
It uses the primary (most capable) LLM because quality matters most here.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.config import get_settings
from src.schemas.domain import AnomalyAlert, Citation, RiskLevel

logger = logging.getLogger(__name__)
settings = get_settings()

CRITIC_SYSTEM_PROMPT = """You are a critical quality reviewer for AI-generated production \
engineering investigations. Your role is to:
1. Review the evidence collected during investigation
2. Draft the root cause hypothesis and recommended actions
3. Score the quality and confidence of the investigation
4. Identify any gaps, contradictions, or unsupported claims
5. Assess the HSE risk level

Return a JSON object with EXACTLY these fields:
{
  "root_cause_hypothesis": "string — most likely root cause based on evidence",
  "supporting_evidence": ["list", "of", "key", "evidence", "points"],
  "recommended_actions": ["list", "of", "specific", "actionable", "steps"],
  "confidence_score": 0.0-1.0,
  "evidence_quality_notes": "string — notes on evidence quality/gaps",
  "risk_level": "LOW|MEDIUM|HIGH|CRITICAL",
  "logical_gaps": ["list", "of", "missing", "information"],
  "hse_concerns": ["list", "of", "safety", "concerns", "if", "any"],
  "uncertainty_statement": "string — honest statement about what is unknown"
}

SCORING GUIDANCE:
- confidence_score < 0.5: major gaps, conflicting evidence, or primary cause unclear
- confidence_score 0.5-0.75: plausible hypothesis but some evidence missing
- confidence_score 0.75-0.9: well-supported by evidence and precedent
- confidence_score > 0.9: strong evidence, consistent pattern, clear root cause

RISK LEVEL GUIDANCE:
- CRITICAL: potential well control event, safety device failure, HSE regulation breach
- HIGH: significant production loss >20%, equipment failure risk, integrity concern
- MEDIUM: moderate production impact, equipment degradation, needs intervention
- LOW: minor variance, no immediate action required, monitoring sufficient

Be HONEST about uncertainty. It is better to escalate to a human than to
over-claim confidence when evidence is weak."""

RECOMMENDATION_BUILDER_PROMPT = """You are helping draft the final recommendation for a
production anomaly investigation. Review all evidence and create the recommendation."""


def _build_critic_prompt(
    alert: AnomalyAlert,
    evidence: list[str],
    citations: list[Citation],
    plan_steps: list[dict[str, Any]],
) -> str:
    evidence_text = "\n\n".join(evidence) if evidence else "No evidence collected."
    citation_text = (
        "\n".join(
            f"- [{c.document_title}] (relevance: {c.relevance_score:.2f}): {c.excerpt[:150]}" for c in citations[:8]
        )
        if citations
        else "No document citations."
    )

    completed_steps = [s for s in plan_steps if s.get("completed")]
    steps_summary = "\n".join(
        f"Step {s['step_id']}: {s['description']} → {(s.get('result') or '')[:200]}" for s in completed_steps
    )

    return f"""ANOMALY ALERT:
Well: {alert.well_id} | Field: {alert.field_name}
Severity: {alert.severity.value} | Score: {alert.anomaly_score:.3f}
Affected features: {", ".join(alert.affected_features)}
Deviations: {json.dumps(alert.deviation_pct)}
Alert description: {alert.description}

INVESTIGATION STEPS COMPLETED:
{steps_summary}

EVIDENCE COLLECTED:
{evidence_text[:4000]}

RELEVANT DOCUMENTS CITED:
{citation_text}

Review this investigation and provide your assessment."""


async def run_critic(state: dict[str, Any]) -> dict[str, Any]:
    """LangGraph node: Critic agent — validates evidence and scores confidence."""
    alert: AnomalyAlert = state["alert"]
    evidence: list[str] = state.get("evidence", [])
    citations: list[Citation] = state.get("citations", [])
    plan_steps: list[dict[str, Any]] = state.get("plan_steps", [])
    agent_steps = list(state.get("agent_steps", []))
    total_tokens: int = state.get("total_tokens", 0)

    step_record: dict[str, Any] = {
        "agent": "critic",
        "action": "validate_and_score",
        "input": {
            "evidence_count": len(evidence),
            "citation_count": len(citations),
            "steps_completed": sum(1 for s in plan_steps if s.get("completed")),
        },
    }

    llm = ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        temperature=0.1,
        response_format={"type": "json_object"},  # type: ignore[call-arg]  # accepted at runtime; stub lags behind SDK
    )

    messages = [
        SystemMessage(content=CRITIC_SYSTEM_PROMPT),
        HumanMessage(content=_build_critic_prompt(alert, evidence, citations, plan_steps)),
    ]

    try:
        response = await llm.ainvoke(messages)
        content = response.content if isinstance(response.content, str) else str(response.content)
        review = json.loads(content)
        tokens = response.usage_metadata.get("total_tokens", 0) if response.usage_metadata else 0  # type: ignore[attr-defined]
        total_tokens += tokens

        confidence_score = float(review.get("confidence_score", 0.5))
        confidence_score = max(0.0, min(1.0, confidence_score))

        risk_level_str = review.get("risk_level", "MEDIUM").upper()
        try:
            risk_level = RiskLevel(risk_level_str)
        except ValueError:
            risk_level = RiskLevel.MEDIUM

        # Compute average source coverage from evidence step records
        coverage_values = [s.get("source_coverage", 1.0) for s in agent_steps if "source_coverage" in s]
        evidence_coverage = sum(coverage_values) / len(coverage_values) if coverage_values else 0.5

        # Penalise confidence if there are major gaps
        gaps: list[str] = review.get("logical_gaps", [])
        if len(gaps) >= 3:
            confidence_score = max(0.1, confidence_score - 0.15)

        step_record.update(
            {
                "output": f"confidence={confidence_score:.2f}, risk={risk_level.value}",
                "tokens": tokens,
                "success": True,
            }
        )

        recommendation = review.get("root_cause_hypothesis", "Root cause undetermined.")

        logger.info(
            "Critic scored %s: confidence=%.2f, risk=%s, gaps=%d",
            alert.well_id,
            confidence_score,
            risk_level.value,
            len(gaps),
        )

        return {
            "recommendation": recommendation,
            "confidence_score": confidence_score,
            "evidence_coverage": evidence_coverage,
            "risk_level": risk_level,
            "agent_steps": agent_steps + [step_record],
            "total_tokens": total_tokens,
            "messages": [
                HumanMessage(
                    content=f"Critic review complete. Confidence: {confidence_score:.2f}, "
                    f"Risk: {risk_level.value}. "
                    f"Root cause: {recommendation[:100]}"
                )
            ],
            # Store full review in evidence for audit
            "_critic_review": review,
        }

    except Exception as exc:
        logger.exception("Critic agent failed for %s", alert.well_id)
        step_record.update({"output": str(exc), "success": False})
        return {
            "recommendation": "Critic agent failed — investigation incomplete. Escalating to human.",
            "confidence_score": 0.0,
            "evidence_coverage": 0.0,
            "risk_level": RiskLevel.HIGH,
            "agent_steps": agent_steps + [step_record],
            "error": f"Critic failed: {exc}",
        }
