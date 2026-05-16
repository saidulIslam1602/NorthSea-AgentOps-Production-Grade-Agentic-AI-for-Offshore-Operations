"""
LangGraph orchestrator — wires Planner → Executor loop → Critic → Uncertainty Gate.

Graph structure:

  START
    ↓
  [planner]          — generates investigation plan
    ↓
  [executor]         — executes one step at a time (loops)
    ↓ (steps remaining?)
  ┌─ YES → back to executor
  └─ NO  → [critic]
              ↓
           [uncertainty_gate]
              ↓
           ┌─ escalate? → [escalate]
           └─ output   → [output]
              ↓
             END

Human-in-the-loop checkpoint is placed between critic and uncertainty_gate,
allowing a human to review the raw evidence before escalation is finalized.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import psycopg
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.agents.critic import run_critic
from src.agents.executor import run_executor
from src.agents.planner import run_planner
from src.agents.state import AgentState, initial_state
from src.agents.uncertainty_gate import apply_uncertainty_gate, build_investigation_result
from src.config import get_settings
from src.schemas.domain import AnomalyAlert, InvestigationResult

logger = logging.getLogger(__name__)
settings = get_settings()


def _should_continue_executing(state: dict[str, Any]) -> str:
    """Routing function: continue executor loop or move to critic."""
    plan_steps = state.get("plan_steps", [])
    current_idx = state.get("current_step_index", 0)
    total_steps = len(plan_steps)

    # Safety: cap at max_iterations
    agent_step_count = len(state.get("agent_steps", []))
    if agent_step_count >= settings.agent_max_iterations:
        logger.warning("Max iterations reached — forcing critic")
        return "critic"

    if current_idx < total_steps:
        return "executor"
    return "critic"


def _escalate_or_output(state: dict[str, Any]) -> str:
    """Route to escalation queue or direct output."""
    return "escalate" if state.get("should_escalate", False) else "output"


# ─── Node wrappers ────────────────────────────────────────────────────────────

async def planner_node(state: dict[str, Any]) -> dict[str, Any]:
    return await run_planner(state)


async def executor_node(
    state: dict[str, Any],
    conn: psycopg.AsyncConnection[Any],
) -> dict[str, Any]:
    return await run_executor(state, conn)


async def critic_node(state: dict[str, Any]) -> dict[str, Any]:
    return await run_critic(state)


def uncertainty_gate_node(state: dict[str, Any]) -> dict[str, Any]:
    return apply_uncertainty_gate(state)


async def escalate_node(state: dict[str, Any]) -> dict[str, Any]:
    """Persist escalation to database and Kafka."""
    from src.tools.escalation import create_escalation

    try:
        await create_escalation(state)
    except Exception:
        logger.exception("Failed to persist escalation")
    return state


async def output_node(state: dict[str, Any]) -> dict[str, Any]:
    """Persist successful investigation result."""
    logger.info(
        "Investigation complete for %s: confidence=%.2f",
        state["alert"].well_id,
        state.get("confidence_score", 0),
    )
    return state


# ─── Graph Builder ────────────────────────────────────────────────────────────

def build_investigation_graph(
    conn: psycopg.AsyncConnection[Any],
) -> Any:
    """Build and compile the investigation LangGraph."""

    async def _executor_with_conn(state: dict[str, Any]) -> dict[str, Any]:
        return await executor_node(state, conn)

    graph = StateGraph(AgentState)  # type: ignore[arg-type]

    graph.add_node("planner", planner_node)
    graph.add_node("executor", _executor_with_conn)
    graph.add_node("critic", critic_node)
    graph.add_node("uncertainty_gate", uncertainty_gate_node)
    graph.add_node("escalate", escalate_node)
    graph.add_node("output", output_node)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "executor")

    graph.add_conditional_edges(
        "executor",
        _should_continue_executing,
        {"executor": "executor", "critic": "critic"},
    )

    # Human-in-the-loop interrupt point: before uncertainty_gate
    # In production, a human can review the state here before proceeding
    graph.add_edge("critic", "uncertainty_gate")

    graph.add_conditional_edges(
        "uncertainty_gate",
        _escalate_or_output,
        {"escalate": "escalate", "output": "output"},
    )

    graph.add_edge("escalate", END)
    graph.add_edge("output", END)

    checkpointer = MemorySaver()
    return graph.compile(
        checkpointer=checkpointer,
        interrupt_before=["uncertainty_gate"],  # human review checkpoint
    )


# ─── High-level entry point ───────────────────────────────────────────────────

async def investigate_anomaly(
    alert: AnomalyAlert,
    conn: psycopg.AsyncConnection[Any],
    skip_human_checkpoint: bool = True,
) -> InvestigationResult:
    """
    Run the full investigation pipeline for an anomaly alert.

    Args:
        alert: The triggering anomaly alert
        conn: Active PostgreSQL connection
        skip_human_checkpoint: If True, auto-proceeds past human checkpoint.
                                Set False in production for real human review.

    Returns:
        InvestigationResult with recommendation, scores, and escalation decision.
    """
    start_time = time.monotonic()
    thread_id = str(alert.alert_id)
    config = {"configurable": {"thread_id": thread_id}}

    app = build_investigation_graph(conn)
    state = initial_state(alert)

    try:
        if skip_human_checkpoint:
            # Run to completion in one pass (automated mode)
            final_state: dict[str, Any] = {}
            async for chunk in app.astream(state, config=config):
                for node_name, node_output in chunk.items():
                    final_state.update(node_output)
                    logger.debug("Node '%s' completed", node_name)

            # If interrupted at human checkpoint, resume
            snapshot = await app.aget_state(config)
            if snapshot.next:
                async for chunk in app.astream(None, config=config):
                    for _, node_output in chunk.items():
                        final_state.update(node_output)
        else:
            # Run until human checkpoint
            async for chunk in app.astream(state, config=config):
                for node_name, node_output in chunk.items():
                    final_state = node_output
                    logger.info("Pausing at human checkpoint after '%s'", node_name)
                    break

            logger.info("Awaiting human review for %s (thread_id=%s)", alert.well_id, thread_id)
            return _partial_result(alert, final_state, start_time)

    except Exception as exc:
        logger.exception("Investigation pipeline failed for %s", alert.well_id)
        final_state = {
            **state,
            "error": str(exc),
            "confidence_score": 0.0,
            "should_escalate": True,
            "escalation_reasons": [],
            "risk_level": "HIGH",
        }

    elapsed_ms = (time.monotonic() - start_time) * 1000
    result = build_investigation_result(final_state)
    result.latency_ms = elapsed_ms

    return result


def _partial_result(
    alert: AnomalyAlert,
    state: dict[str, Any],
    start_time: float,
) -> InvestigationResult:
    """Return a partial result when paused at human checkpoint."""
    from src.schemas.domain import EscalationReason, RiskLevel

    elapsed_ms = (time.monotonic() - start_time) * 1000
    return InvestigationResult(
        alert=alert,
        root_cause_hypothesis="Awaiting human review",
        supporting_evidence=state.get("evidence", []),
        recommended_actions=["Human engineer review required before action"],
        citations=state.get("citations", []),
        confidence_score=state.get("confidence_score", 0.0),
        evidence_coverage=state.get("evidence_coverage", 0.0),
        risk_level=state.get("risk_level", RiskLevel.MEDIUM),
        should_escalate=True,
        escalation_reasons=[EscalationReason.OPERATOR_OVERRIDE],
        escalation_message="Investigation paused at human review checkpoint.",
        agent_steps=state.get("agent_steps", []),
        total_tokens_used=state.get("total_tokens", 0),
        latency_ms=elapsed_ms,
    )
