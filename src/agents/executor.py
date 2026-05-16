"""
Executor agent — executes the investigation plan step by step.

Pattern: Plan-and-Execute executor node.
  - Picks the next incomplete step from plan_steps
  - Calls the appropriate tool (enforced by tool allowlist)
  - Appends evidence and citations to state
  - Loops until all steps are complete or max_iterations reached
"""

from __future__ import annotations

import logging
from typing import Any

import psycopg
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.config import get_settings
from src.schemas.domain import AnomalyAlert
from src.tools.diagnostic_api import get_equipment_status, get_well_metadata
from src.tools.rag_retrieval import query_similar_incidents, retrieve_documents
from src.tools.timeseries_query import query_timeseries

logger = logging.getLogger(__name__)
settings = get_settings()

# ─── Tool allowlist for Executor ─────────────────────────────────────────────
EXECUTOR_ALLOWED_TOOLS = frozenset(
    {
        "query_timeseries",
        "retrieve_documents",
        "query_similar_incidents",
        "get_well_metadata",
        "get_equipment_status",
    }
)

EXECUTOR_SYSTEM_PROMPT = """You are a North Sea production operations engineer AI.
Your task is to execute one investigation step and synthesise the tool results
into a concise evidence summary.

Given the tool output, write a brief (3-5 sentence) evidence summary that:
1. States what was found
2. Notes any anomalies or patterns in the data
3. Highlights anything relevant to the anomaly being investigated
4. Flags any data quality issues

Be factual. Do not recommend actions — that comes later. If data is unavailable, say so clearly."""


async def _dispatch_tool(
    conn: psycopg.AsyncConnection[Any],
    tool_name: str,
    alert: AnomalyAlert,
    step_description: str,
) -> dict[str, Any]:
    """Dispatch tool call with allowlist enforcement."""
    if tool_name not in EXECUTOR_ALLOWED_TOOLS:
        return {
            "error": f"Tool '{tool_name}' is not in the executor allowlist.",
            "summary_text": f"Tool call blocked: '{tool_name}' is not permitted for the Executor agent.",
        }

    try:
        if tool_name == "query_timeseries":
            return await query_timeseries(conn, alert.well_id, hours_back=72)

        elif tool_name == "retrieve_documents":
            return await retrieve_documents(conn, step_description)

        elif tool_name == "query_similar_incidents":
            return await query_similar_incidents(conn, alert.description, alert.affected_features)

        elif tool_name == "get_well_metadata":
            return await get_well_metadata(conn, alert.well_id)

        elif tool_name == "get_equipment_status":
            return await get_equipment_status(conn, alert.well_id)

        else:
            return {"summary_text": f"Unknown tool: {tool_name}"}

    except Exception as exc:
        logger.exception("Tool %s failed for %s", tool_name, alert.well_id)
        return {"error": str(exc), "summary_text": f"Tool '{tool_name}' failed: {exc}"}


async def run_executor(
    state: dict[str, Any],
    conn: psycopg.AsyncConnection[Any],
) -> dict[str, Any]:
    """LangGraph node: Executor agent — executes the next pending plan step."""
    plan_steps: list[dict[str, Any]] = state.get("plan_steps", [])
    current_idx: int = state.get("current_step_index", 0)
    alert: AnomalyAlert = state["alert"]
    evidence: list[str] = list(state.get("evidence", []))
    citations = list(state.get("citations", []))
    agent_steps = list(state.get("agent_steps", []))
    total_tokens: int = state.get("total_tokens", 0)

    if current_idx >= len(plan_steps):
        logger.info("All steps complete for %s", alert.well_id)
        return {"current_step_index": current_idx}

    step = plan_steps[current_idx]
    tool_name = step.get("tool_to_use", "retrieve_documents")
    step_desc = step.get("description", "")

    step_record: dict[str, Any] = {
        "agent": "executor",
        "action": f"execute_step_{step['step_id']}",
        "tool": tool_name,
        "description": step_desc,
    }

    # 1. Execute the tool
    tool_result = await _dispatch_tool(conn, tool_name, alert, step_desc)

    # 2. Synthesise tool output with LLM (mini model for cost efficiency)
    llm = ChatOpenAI(
        model=settings.openai_mini_model,
        api_key=settings.openai_api_key,
        temperature=0.1,
    )

    raw_text = tool_result.get("summary_text", "") or tool_result.get("context", "")
    if not raw_text and "error" in tool_result:
        raw_text = f"Tool error: {tool_result['error']}"

    synthesis_prompt = f"""Investigation step: {step_desc}
Expected output: {step.get("expected_output", "")}

Tool output:
{raw_text[:3000]}

Summarise the key findings relevant to this investigation step."""

    messages = [
        SystemMessage(content=EXECUTOR_SYSTEM_PROMPT),
        HumanMessage(content=synthesis_prompt),
    ]

    try:
        response = await llm.ainvoke(messages)
        synthesis = response.content if isinstance(response.content, str) else str(response.content)
        _usage = getattr(response, "usage_metadata", None)
        tokens = int(_usage.get("total_tokens", 0)) if _usage is not None else 0
        total_tokens += tokens
    except Exception:
        logger.exception("Executor LLM synthesis failed")
        synthesis = raw_text[:500]
        tokens = 0

    # 3. Collect citations if this was a RAG step
    if "citations" in tool_result:
        citations.extend(tool_result["citations"])

    # 4. Track coverage
    step_coverage = tool_result.get("source_coverage", 1.0)

    # Update step result in plan
    updated_steps = list(plan_steps)
    updated_steps[current_idx] = {
        **step,
        "completed": True,
        "result": synthesis,
    }

    evidence.append(f"Step {step['step_id']} ({step_desc}):\n{synthesis}")

    step_record.update(
        {
            "output": synthesis[:200],
            "tokens": tokens,
            "source_coverage": step_coverage,
            "success": "error" not in tool_result,
        }
    )

    logger.info(
        "Executor completed step %d/%d for %s: %s", current_idx + 1, len(plan_steps), alert.well_id, synthesis[:80]
    )

    return {
        "plan_steps": updated_steps,
        "current_step_index": current_idx + 1,
        "evidence": evidence,
        "citations": citations,
        "total_tokens": total_tokens,
        "agent_steps": agent_steps + [step_record],
        "messages": [HumanMessage(content=f"Step {step['step_id']} complete: {synthesis[:100]}")],
    }
