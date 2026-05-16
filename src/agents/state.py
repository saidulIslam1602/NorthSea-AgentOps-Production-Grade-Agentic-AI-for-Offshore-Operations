"""
LangGraph state definition for the NorthSea investigation pipeline.

The AgentState is the single typed message object that flows through
every node in the graph. All agents read from and write to this state.
"""

from __future__ import annotations

from typing import Annotated, Any
from uuid import UUID

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from src.schemas.domain import AnomalyAlert, Citation, EscalationReason, RiskLevel


class AgentState(dict[str, Any]):
    """
    Typed state dict for the investigation graph.

    Fields populated progressively as the graph executes:
      - alert: the triggering anomaly alert
      - messages: LangChain message history (add_messages reducer for thread safety)
      - plan_steps: structured investigation plan from the Planner
      - evidence: accumulated evidence from Executor tool calls
      - citations: RAG citations collected during investigation
      - recommendation: final recommendation text
      - confidence_score: 0-1 score from Critic
      - evidence_coverage: 0-1 RAG coverage from retriever
      - risk_level: assessed risk level
      - should_escalate: escalation decision
      - escalation_reasons: list of reasons driving escalation
      - agent_steps: audit trail of every agent action
      - error: non-None if a recoverable error occurred
    """

    messages: Annotated[list[BaseMessage], add_messages]
    alert: AnomalyAlert
    plan_steps: list[dict[str, Any]]
    current_step_index: int
    evidence: list[str]
    citations: list[Citation]
    recommendation: str
    confidence_score: float
    evidence_coverage: float
    risk_level: RiskLevel
    should_escalate: bool
    escalation_reasons: list[EscalationReason]
    agent_steps: list[dict[str, Any]]
    total_tokens: int
    error: str | None
    investigation_id: UUID | None


def initial_state(alert: AnomalyAlert) -> dict[str, Any]:
    """Return a fresh AgentState for a new investigation."""
    from uuid import uuid4

    return {
        "messages": [],
        "alert": alert,
        "plan_steps": [],
        "current_step_index": 0,
        "evidence": [],
        "citations": [],
        "recommendation": "",
        "confidence_score": 0.0,
        "evidence_coverage": 0.0,
        "risk_level": RiskLevel.LOW,
        "should_escalate": False,
        "escalation_reasons": [],
        "agent_steps": [],
        "total_tokens": 0,
        "error": None,
        "investigation_id": uuid4(),
    }
