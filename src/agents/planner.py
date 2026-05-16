"""
Planner agent — decomposes an anomaly alert into a structured investigation plan.

Pattern: Plan-and-Execute (LangChain blog, 2023)
  - Receives the AnomalyAlert and field context
  - Uses the primary LLM to generate 3-5 investigation steps
  - Each step specifies: description, tool_to_use, expected_output

The Planner runs once at the start of each investigation. It does NOT
call tools itself — that is the Executor's responsibility.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.agents.state import AgentState
from src.config import get_settings
from src.schemas.domain import AnomalyAlert

logger = logging.getLogger(__name__)
settings = get_settings()

AVAILABLE_TOOLS = [
    "query_timeseries",
    "retrieve_documents",
    "query_similar_incidents",
    "get_well_metadata",
    "get_equipment_status",
]

PLANNER_SYSTEM_PROMPT = """You are a senior production engineer AI assistant specialising in \
North Sea oil and gas operations. Your role is to create structured investigation plans \
for production anomalies.

When given an anomaly alert, produce a JSON investigation plan with 3-5 steps.
Each step must specify:
- step_id: integer (1-based)
- description: clear description of what to investigate
- tool_to_use: one of {tools}
- expected_output: what information we expect to find

IMPORTANT RULES:
1. Always start with querying historical timeseries data to confirm the anomaly.
2. Always retrieve relevant operational documents (maintenance logs, well reports, HSE procedures).
3. Query for similar past incidents when severity is MEDIUM or higher.
4. Keep steps focused — avoid redundant steps.
5. Consider HSE implications for HIGH and CRITICAL severity anomalies.

Return ONLY a JSON object with key "steps" containing the list of step objects.
Do not include any explanation outside the JSON.""".format(tools=AVAILABLE_TOOLS)


def _build_planner_prompt(alert: AnomalyAlert) -> str:
    return f"""ANOMALY ALERT:
Well: {alert.well_id}
Field: {alert.field_name}
Severity: {alert.severity.value}
Anomaly Score: {alert.anomaly_score:.3f}
Affected Features: {', '.join(alert.affected_features)}
Current Values: {json.dumps(alert.current_values, indent=2)}
Baseline Values: {json.dumps(alert.baseline_values, indent=2)}
Deviations: {json.dumps(alert.deviation_pct, indent=2)}
Description: {alert.description}

Create a structured investigation plan for this anomaly."""


async def run_planner(state: dict[str, Any]) -> dict[str, Any]:
    """LangGraph node: Planner agent."""
    alert: AnomalyAlert = state["alert"]

    llm = ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key.get_secret_value(),
        temperature=0.1,
        response_format={"type": "json_object"},
    )

    messages = [
        SystemMessage(content=PLANNER_SYSTEM_PROMPT),
        HumanMessage(content=_build_planner_prompt(alert)),
    ]

    step_record: dict[str, Any] = {
        "agent": "planner",
        "action": "generate_investigation_plan",
        "input": {"alert_id": str(alert.alert_id), "severity": alert.severity.value},
    }

    try:
        response = await llm.ainvoke(messages)
        content = response.content if isinstance(response.content, str) else str(response.content)
        plan_data = json.loads(content)
        plan_steps = plan_data.get("steps", [])

        if not isinstance(plan_steps, list) or len(plan_steps) == 0:
            raise ValueError("Planner returned empty or invalid steps")

        # Validate each step has required fields
        validated_steps = []
        for i, step in enumerate(plan_steps):
            validated_steps.append({
                "step_id": step.get("step_id", i + 1),
                "description": step.get("description", f"Step {i + 1}"),
                "tool_to_use": step.get("tool_to_use", "query_timeseries"),
                "expected_output": step.get("expected_output", ""),
                "completed": False,
                "result": None,
            })

        tokens = response.usage_metadata.get("total_tokens", 0) if response.usage_metadata else 0

        step_record.update({
            "output": f"Generated {len(validated_steps)} investigation steps",
            "tokens": tokens,
            "success": True,
        })

        logger.info(
            "Planner created %d steps for %s [%s]",
            len(validated_steps), alert.well_id, alert.severity.value
        )

        return {
            "plan_steps": validated_steps,
            "current_step_index": 0,
            "total_tokens": state.get("total_tokens", 0) + tokens,
            "agent_steps": state.get("agent_steps", []) + [step_record],
            "messages": [HumanMessage(content=f"Investigation plan created: {len(validated_steps)} steps")],
        }

    except Exception as exc:
        step_record.update({"output": str(exc), "success": False})
        logger.exception("Planner failed for %s", alert.well_id)
        # Fallback plan
        fallback_steps = _fallback_plan(alert)
        return {
            "plan_steps": fallback_steps,
            "current_step_index": 0,
            "agent_steps": state.get("agent_steps", []) + [step_record],
            "error": f"Planner LLM failed: {exc}",
        }


def _fallback_plan(alert: AnomalyAlert) -> list[dict[str, Any]]:
    """Rule-based fallback plan when LLM planner fails."""
    steps = [
        {
            "step_id": 1,
            "description": f"Query 72-hour telemetry history for {alert.well_id} to confirm anomaly trend",
            "tool_to_use": "query_timeseries",
            "expected_output": "Trend data confirming or ruling out anomaly",
            "completed": False,
            "result": None,
        },
        {
            "step_id": 2,
            "description": f"Search operational documents for {alert.well_id} maintenance history and well reports",
            "tool_to_use": "retrieve_documents",
            "expected_output": "Recent maintenance events, known issues, intervention history",
            "completed": False,
            "result": None,
        },
        {
            "step_id": 3,
            "description": f"Search for similar {', '.join(alert.affected_features)} anomalies in historical incidents",
            "tool_to_use": "query_similar_incidents",
            "expected_output": "Past incidents with similar signatures and their root causes",
            "completed": False,
            "result": None,
        },
    ]

    if alert.severity.value in ("HIGH", "CRITICAL"):
        steps.append({
            "step_id": 4,
            "description": "Retrieve HSE procedures relevant to this type of anomaly",
            "tool_to_use": "retrieve_documents",
            "expected_output": "Safety response requirements and escalation thresholds",
            "completed": False,
            "result": None,
        })

    return steps
