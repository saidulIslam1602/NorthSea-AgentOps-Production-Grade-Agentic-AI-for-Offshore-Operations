"""
Challenger agent — adversarial multi-agent coordination for critic validation.

Pattern: Critic-Actor with adversarial challenge (multi-agent coordination).
  - The Critic produces a root cause hypothesis and confidence score.
  - The Challenger independently reviews the SAME evidence and tries to:
      1. Identify alternative root cause hypotheses the Critic missed.
      2. Find weaknesses in the Critic's evidence chain.
      3. Estimate a competing confidence score.
  - A Reconciler (lightweight LLM call) synthesises both positions into a
    final agreed confidence and recommendation.

Why this matters for offshore operations:
  In safety-critical environments (NORSOK, PSA regulations), a second
  independent verification of any AI-generated recommendation is essential
  before it reaches an operator. The Challenger is the software equivalent
  of a "second pair of eyes" mandated by HSE procedures for HIGH-risk events.

Multi-agent coordination taxonomy (Aker BP Track A requirement):
  - Pattern: critic-actor with adversarial validation
  - Agent roles: Critic (primary), Challenger (adversarial), Reconciler (synthesis)
  - Communication: state-based (shared evidence + citations via LangGraph state dict)
  - Activation: only on HIGH/CRITICAL risk_level or when Critic confidence < 0.80
    (saves token cost for low-risk events where overhead is unjustified)
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

# ── System prompts ────────────────────────────────────────────────────────────

CHALLENGER_SYSTEM_PROMPT = """You are an adversarial reviewer for AI-generated production \
engineering investigations. Your job is to CHALLENGE the primary Critic's conclusion.

You are given:
1. The original anomaly alert
2. The evidence collected during investigation
3. The primary Critic's hypothesis and confidence score

Your task:
1. Identify alternative root cause hypotheses the Critic may have missed.
2. Find weaknesses, unsupported assumptions, or logical gaps in the Critic's reasoning.
3. Assess whether the Critic's confidence score is overconfident or underconfident.
4. Propose what ADDITIONAL evidence would be needed to distinguish between hypotheses.

Return a JSON object with EXACTLY:
{
  "alternative_hypotheses": ["list", "of", "alternative", "root", "causes"],
  "critique_of_primary": "string — specific weaknesses in the Critic's reasoning",
  "competing_confidence_score": 0.0-1.0,
  "missing_evidence": ["list", "of", "data", "gaps", "that", "would", "help"],
  "challenger_agrees": true/false,
  "agreement_level": "AGREE|PARTIAL|DISAGREE",
  "recommended_escalation": true/false,
  "challenge_summary": "string — 1-2 sentence summary of your challenge position"
}

Be rigorous. False confidence in offshore operations costs lives and production.
If the evidence genuinely supports the Critic, it is acceptable to AGREE — but
only if you have checked thoroughly for alternatives."""

RECONCILER_SYSTEM_PROMPT = """You are a senior production engineer reconciling two independent \
AI reviews of a production anomaly investigation.

You are given the primary Critic's assessment and the Challenger's challenge. 
Your task is to produce a RECONCILED final assessment that:
1. Takes the strongest evidence from both positions.
2. Adjusts confidence appropriately — be CONSERVATIVE (lower confidence when challenged).
3. Identifies the SINGLE most actionable root cause hypothesis.
4. Lists the top 3 recommended actions.

Return a JSON object:
{
  "reconciled_root_cause": "string",
  "reconciled_confidence": 0.0-1.0,
  "reconciled_risk_level": "LOW|MEDIUM|HIGH|CRITICAL",
  "reconciled_actions": ["action1", "action2", "action3"],
  "reconciliation_notes": "string — why you resolved the disagreement this way",
  "requires_human_review": true/false,
  "human_review_reason": "string or null"
}

If Critic and Challenger substantially disagree (agreement_level=DISAGREE), you MUST
set requires_human_review=true and lower confidence below the Critic's score."""


# ── Challenge activation logic ────────────────────────────────────────────────

def should_invoke_challenger(
    critic_confidence: float,
    risk_level: RiskLevel,
    evidence_count: int,
) -> bool:
    """
    Decide whether to run the Challenger.

    Cost-aware: only activates when the stakes justify the extra LLM call.
    Activation criteria (any one triggers):
      - risk_level is HIGH or CRITICAL (HSE threshold)
      - critic_confidence < 0.80 (Critic is uncertain; challenger may clarify)
      - evidence_count < 2 (sparse evidence; adversarial check catches overconfidence)
    """
    return (
        risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL)
        or critic_confidence < 0.80
        or evidence_count < 2
    )


# ── Challenger agent ──────────────────────────────────────────────────────────

async def run_challenger(
    alert: AnomalyAlert,
    evidence: list[str],
    citations: list[Citation],
    critic_review: dict[str, Any],
    critic_confidence: float,
) -> dict[str, Any]:
    """
    Run the adversarial Challenger agent against the Critic's conclusion.

    Args:
        alert: The original anomaly alert.
        evidence: Evidence strings collected by the Executor.
        citations: Document citations from RAG retrieval.
        critic_review: Full Critic review dict (from _critic_review in state).
        critic_confidence: The Critic's stated confidence score.

    Returns:
        Challenger review dict with alternative hypotheses, critique, and
        agreement level.
    """
    llm = ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key.get_secret_value(),
        temperature=0.2,  # slightly higher than Critic to surface alternatives
        response_format={"type": "json_object"},
    )

    evidence_text = "\n\n".join(evidence[:6]) if evidence else "No evidence collected."
    citation_text = "\n".join(
        f"- [{c.document_title}]: {c.excerpt[:120]}"
        for c in citations[:5]
    ) if citations else "No citations."

    critic_hypothesis = critic_review.get("root_cause_hypothesis", "Not provided")
    critic_actions = critic_review.get("recommended_actions", [])
    critic_notes = critic_review.get("evidence_quality_notes", "")

    prompt = f"""ANOMALY ALERT:
Well: {alert.well_id} | Field: {alert.field_name}
Severity: {alert.severity.value} | Score: {alert.anomaly_score:.3f}
Features affected: {", ".join(alert.affected_features)}
Deviations: {json.dumps(alert.deviation_pct)}
Description: {alert.description}

PRIMARY CRITIC'S CONCLUSION (to be challenged):
Root cause: {critic_hypothesis}
Recommended actions: {json.dumps(critic_actions)}
Confidence score: {critic_confidence:.2f}
Evidence quality notes: {critic_notes}
Logical gaps identified by Critic: {json.dumps(critic_review.get("logical_gaps", []))}

EVIDENCE AVAILABLE:
{evidence_text[:3000]}

CITATIONS:
{citation_text}

Challenge the Critic's conclusion. Look for what was missed."""

    messages = [
        SystemMessage(content=CHALLENGER_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]

    step_record: dict[str, Any] = {
        "agent": "challenger",
        "action": "adversarial_review",
        "input": {
            "critic_confidence": critic_confidence,
            "evidence_count": len(evidence),
        },
    }

    try:
        response = await llm.ainvoke(messages)
        content = response.content if isinstance(response.content, str) else str(response.content)
        challenge = json.loads(content)
        tokens = response.usage_metadata.get("total_tokens", 0) if response.usage_metadata else 0

        step_record.update({
            "output": f"agreement={challenge.get('agreement_level')}, "
                      f"competing_conf={challenge.get('competing_confidence_score', 0):.2f}",
            "tokens": tokens,
            "success": True,
        })

        logger.info(
            "Challenger reviewed %s: agreement=%s, alt_hypotheses=%d",
            alert.well_id,
            challenge.get("agreement_level"),
            len(challenge.get("alternative_hypotheses", [])),
        )

        return {
            "challenge": challenge,
            "step_record": step_record,
            "tokens": tokens,
        }

    except Exception as exc:
        logger.exception("Challenger agent failed for %s", alert.well_id)
        step_record.update({"output": str(exc), "success": False})
        return {
            "challenge": {
                "agreement_level": "AGREE",
                "challenger_agrees": True,
                "competing_confidence_score": critic_confidence,
                "challenge_summary": f"Challenger failed: {exc}",
                "recommended_escalation": False,
                "alternative_hypotheses": [],
                "missing_evidence": [],
                "critique_of_primary": "Challenger unavailable",
            },
            "step_record": step_record,
            "tokens": 0,
        }


# ── Reconciler ────────────────────────────────────────────────────────────────

async def run_reconciler(
    alert: AnomalyAlert,
    critic_review: dict[str, Any],
    challenge: dict[str, Any],
    critic_confidence: float,
) -> dict[str, Any]:
    """
    Synthesise Critic and Challenger positions into a final reconciled assessment.

    Called only when Challenger has PARTIAL or DISAGREE agreement level — for AGREE
    cases the Critic's output is accepted directly (no extra LLM call needed).
    """
    agreement = challenge.get("agreement_level", "AGREE")
    if agreement == "AGREE":
        # No reconciliation needed — accept Critic's output unchanged
        return {
            "reconciled_root_cause": critic_review.get("root_cause_hypothesis"),
            "reconciled_confidence": critic_confidence,
            "reconciled_risk_level": critic_review.get("risk_level", "MEDIUM"),
            "reconciled_actions": critic_review.get("recommended_actions", []),
            "reconciliation_notes": "Challenger agreed with primary Critic — no adjustment needed.",
            "requires_human_review": False,
            "human_review_reason": None,
            "tokens": 0,
        }

    llm = ChatOpenAI(
        model=settings.openai_mini_model,  # mini model adequate for synthesis
        api_key=settings.openai_api_key.get_secret_value(),
        temperature=0.1,
        response_format={"type": "json_object"},
    )

    prompt = f"""RECONCILIATION TASK:

PRIMARY CRITIC:
- Root cause: {critic_review.get("root_cause_hypothesis")}
- Confidence: {critic_confidence:.2f}
- Risk level: {critic_review.get("risk_level")}
- Actions: {json.dumps(critic_review.get("recommended_actions", []))}

CHALLENGER:
- Agreement: {agreement}
- Alternative hypotheses: {json.dumps(challenge.get("alternative_hypotheses", []))}
- Critique: {challenge.get("critique_of_primary")}
- Competing confidence: {challenge.get("competing_confidence_score", 0):.2f}
- Challenge summary: {challenge.get("challenge_summary")}
- Missing evidence: {json.dumps(challenge.get("missing_evidence", []))}

Well: {alert.well_id} | Field: {alert.field_name} | Severity: {alert.severity.value}

Reconcile these two positions into a final assessment."""

    messages = [
        SystemMessage(content=RECONCILER_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]

    try:
        response = await llm.ainvoke(messages)
        content = response.content if isinstance(response.content, str) else str(response.content)
        reconciled = json.loads(content)
        tokens = response.usage_metadata.get("total_tokens", 0) if response.usage_metadata else 0
        reconciled["tokens"] = tokens

        logger.info(
            "Reconciler: %s reconciled_confidence=%.2f, requires_human=%s",
            alert.well_id,
            reconciled.get("reconciled_confidence", 0),
            reconciled.get("requires_human_review"),
        )
        return reconciled

    except Exception as exc:
        logger.exception("Reconciler failed for %s", alert.well_id)
        # Conservative fallback: reduce confidence, recommend human review
        return {
            "reconciled_root_cause": critic_review.get("root_cause_hypothesis"),
            "reconciled_confidence": min(critic_confidence, 0.6),
            "reconciled_risk_level": "HIGH",
            "reconciled_actions": critic_review.get("recommended_actions", []),
            "reconciliation_notes": f"Reconciler failed ({exc}) — applying conservative fallback.",
            "requires_human_review": True,
            "human_review_reason": "Reconciler unavailable; cannot resolve Critic/Challenger disagreement.",
            "tokens": 0,
        }


# ── LangGraph node entry point ────────────────────────────────────────────────

async def run_challenger_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    LangGraph node: Challenger + Reconciler pass.

    Inserts between the Critic node and the UncertaintyGate node.
    Updates state with reconciled confidence, root cause, and actions.
    Only runs when should_invoke_challenger() returns True (cost-aware).
    """
    alert: AnomalyAlert = state["alert"]
    critic_review: dict[str, Any] = state.get("_critic_review", {})
    critic_confidence: float = state.get("confidence_score", 0.5)
    evidence: list[str] = state.get("evidence", [])
    citations: list[Citation] = state.get("citations", [])
    risk_level: RiskLevel = state.get("risk_level", RiskLevel.MEDIUM)
    agent_steps = list(state.get("agent_steps", []))
    total_tokens: int = state.get("total_tokens", 0)

    if not should_invoke_challenger(critic_confidence, risk_level, len(evidence)):
        logger.info(
            "Challenger skipped for %s (confidence=%.2f, risk=%s) — within acceptable threshold",
            alert.well_id, critic_confidence, risk_level.value,
        )
        return {}  # no-op: state unchanged

    logger.info(
        "Invoking Challenger for %s (confidence=%.2f, risk=%s)",
        alert.well_id, critic_confidence, risk_level.value,
    )

    # 1. Run Challenger
    challenge_result = await run_challenger(
        alert=alert,
        evidence=evidence,
        citations=citations,
        critic_review=critic_review,
        critic_confidence=critic_confidence,
    )
    challenge = challenge_result["challenge"]
    total_tokens += challenge_result.get("tokens", 0)
    agent_steps.append(challenge_result["step_record"])

    # 2. Run Reconciler (only if there is actual disagreement)
    reconciled = await run_reconciler(
        alert=alert,
        critic_review=critic_review,
        challenge=challenge,
        critic_confidence=critic_confidence,
    )
    total_tokens += reconciled.pop("tokens", 0)

    # 3. Update state with reconciled values
    updates: dict[str, Any] = {
        "agent_steps": agent_steps,
        "total_tokens": total_tokens,
        "_challenger_review": challenge,
        "_reconciler_review": reconciled,
    }

    # Override Critic's outputs if Reconciler changed them
    if reconciled.get("reconciled_confidence") is not None:
        updates["confidence_score"] = float(reconciled["reconciled_confidence"])
    if reconciled.get("reconciled_root_cause"):
        updates["recommendation"] = reconciled["reconciled_root_cause"]
    if reconciled.get("reconciled_actions"):
        updates["_critic_review"] = {
            **critic_review,
            "recommended_actions": reconciled["reconciled_actions"],
        }

    # If Reconciler demands human review, force escalation
    if reconciled.get("requires_human_review"):
        from src.schemas.domain import EscalationReason
        existing_reasons = list(state.get("escalation_reasons", []))
        if EscalationReason.LOW_CONFIDENCE not in existing_reasons:
            existing_reasons.append(EscalationReason.LOW_CONFIDENCE)
        updates["escalation_reasons"] = existing_reasons
        updates["should_escalate"] = True
        logger.warning(
            "Reconciler requires human review for %s: %s",
            alert.well_id, reconciled.get("human_review_reason"),
        )

    return updates
