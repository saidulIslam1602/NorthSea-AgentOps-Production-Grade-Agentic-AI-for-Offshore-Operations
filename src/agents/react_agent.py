"""
ReAct Agent — Reasoning + Acting pattern for single-turn operational queries.

Pattern: ReAct (Yao et al., 2022 — https://arxiv.org/abs/2210.03629)
  - Alternates between: Thought → Action → Observation
  - Each Thought explains the reasoning behind the next Action
  - Each Observation feeds back from the tool into the next Thought
  - Terminates on a Final Answer or when max_steps is reached

Complements the Plan-Execute-Critic pipeline:
  - Plan-Execute-Critic: used for complex, multi-step anomaly investigations
    that need upfront planning and a separate critic validation pass
  - ReAct: used for single-turn operational Q&A and quick diagnostic checks
    where upfront planning is overkill (e.g. "What is the current BHP on F-4?",
    "Retrieve the maintenance log for D-3H", "Check the HSE procedure for H2S")

Architecture note:
  ReAct runs as a standalone entry point — it does NOT enter the LangGraph
  Plan-Execute pipeline. Both patterns share the same tool registry and
  allowlist enforcement (safety is not pattern-specific).
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

import psycopg
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.config import get_settings
from src.safety.injection_guard import check_user_query

logger = logging.getLogger(__name__)
settings = get_settings()

MAX_REACT_STEPS = 6  # safety cap to prevent runaway loops

# ── ReAct system prompt ───────────────────────────────────────────────────────

REACT_SYSTEM_PROMPT = """You are an expert production engineer AI for North Sea oil and gas \
operations. You answer operational queries by reasoning step-by-step and using tools.

Use the following format EXACTLY for each step:

Thought: <your reasoning about what to do next>
Action: <tool_name>
Action Input: <JSON input for the tool>
Observation: <tool result will appear here — you do not generate this>

When you have enough information to answer, use:
Thought: I now have enough information to answer.
Final Answer: <your complete, factual answer with sources cited>

Available tools and their inputs:
- query_timeseries: {{"well_id": "15/9-F-4", "hours_back": 72}}
- retrieve_documents: {{"query": "ESP failure scale deposition", "k": 5}}
- query_similar_incidents: {{"description": "water cut increase", "features": ["water_cut_pct"]}}
- get_well_metadata: {{"well_id": "15/9-F-4"}}
- get_equipment_status: {{"well_id": "15/9-F-4"}}

RULES:
1. Never fabricate tool results — only use what appears in Observation.
2. Always cite document sources when using retrieve_documents.
3. For HSE/safety questions, always defer to the documented procedure.
4. If uncertain after max steps, say so clearly in Final Answer.
5. AI must NOT recommend ESD activation or well control actions — escalate to human.
"""


# ── Data structures ───────────────────────────────────────────────────────────


@dataclass
class ReactStep:
    """Single Thought→Action→Observation cycle."""

    thought: str
    action: str
    action_input: dict[str, Any]
    observation: str
    tokens_used: int = 0


@dataclass
class ReactResult:
    """Final result from the ReAct agent."""

    query: str
    final_answer: str
    steps: list[ReactStep] = field(default_factory=list)
    total_tokens: int = 0
    latency_ms: float = 0.0
    citations: list[str] = field(default_factory=list)
    halted_early: bool = False
    halt_reason: str = ""


# ── Tool dispatch (shared with Executor) ─────────────────────────────────────

REACT_ALLOWED_TOOLS = frozenset(
    {
        "query_timeseries",
        "retrieve_documents",
        "query_similar_incidents",
        "get_well_metadata",
        "get_equipment_status",
    }
)


async def _call_tool(
    tool_name: str,
    action_input: dict[str, Any],
    conn: psycopg.AsyncConnection[Any],
) -> str:
    """Dispatch a single tool call and return a string observation."""
    if tool_name not in REACT_ALLOWED_TOOLS:
        return f"[BLOCKED] Tool '{tool_name}' is not in the ReAct allowlist."

    try:
        if tool_name == "query_timeseries":
            from src.tools.timeseries_query import query_timeseries

            result = await query_timeseries(
                conn,
                action_input.get("well_id", ""),
                hours_back=action_input.get("hours_back", 72),
            )
        elif tool_name == "retrieve_documents":
            from src.tools.rag_retrieval import retrieve_documents

            result = await retrieve_documents(conn, action_input.get("query", ""))
        elif tool_name == "query_similar_incidents":
            from src.tools.rag_retrieval import query_similar_incidents

            result = await query_similar_incidents(
                conn,
                action_input.get("description", ""),
                action_input.get("features", []),
            )
        elif tool_name == "get_well_metadata":
            from src.tools.diagnostic_api import get_well_metadata

            result = await get_well_metadata(conn, action_input.get("well_id", ""))
        elif tool_name == "get_equipment_status":
            from src.tools.diagnostic_api import get_equipment_status

            result = await get_equipment_status(conn, action_input.get("well_id", ""))
        else:
            return "Unknown tool."

        return result.get("summary_text") or result.get("context") or json.dumps(result)[:800]

    except Exception as exc:
        logger.exception("ReAct tool %s failed", tool_name)
        return f"[ERROR] Tool '{tool_name}' failed: {exc}"


# ── Response parser ───────────────────────────────────────────────────────────


def _parse_react_response(text: str) -> dict[str, Any]:
    """
    Parse the LLM's ReAct-format response.

    Returns dict with keys: thought, action, action_input, final_answer
    (final_answer is set if the model produced a Final Answer line).
    """
    result: dict[str, Any] = {
        "thought": "",
        "action": "",
        "action_input": {},
        "final_answer": None,
    }

    # Check for final answer first
    final_match = re.search(r"Final Answer:\s*(.+)", text, re.DOTALL | re.IGNORECASE)
    if final_match:
        result["final_answer"] = final_match.group(1).strip()
        return result

    # Extract Thought
    thought_match = re.search(r"Thought:\s*(.+?)(?=Action:|Final Answer:|$)", text, re.DOTALL)
    if thought_match:
        result["thought"] = thought_match.group(1).strip()

    # Extract Action
    action_match = re.search(r"Action:\s*(\w+)", text)
    if action_match:
        result["action"] = action_match.group(1).strip()

    # Extract Action Input
    input_match = re.search(r"Action Input:\s*(\{.+?\})", text, re.DOTALL)
    if input_match:
        try:
            result["action_input"] = json.loads(input_match.group(1))
        except json.JSONDecodeError:
            result["action_input"] = {}

    return result


# ── Main ReAct loop ───────────────────────────────────────────────────────────


async def run_react(
    query: str,
    conn: psycopg.AsyncConnection[Any],
    max_steps: int = MAX_REACT_STEPS,
) -> ReactResult:
    """
    Run the ReAct agent on a single operational query.

    Args:
        query: Natural-language operational question from operator / API caller.
        conn: Active PostgreSQL connection (for tool dispatch).
        max_steps: Safety cap on Thought→Action→Observation cycles.

    Returns:
        ReactResult with final answer, all steps, tokens, and latency.
    """
    start_time = time.monotonic()
    total_tokens = 0
    steps: list[ReactStep] = []
    citations: list[str] = []

    # Safety check on user query
    inj = check_user_query(query)
    if not inj.is_clean and inj.severity == "CRITICAL":
        return ReactResult(
            query=query,
            final_answer="[BLOCKED] Query contains prohibited content and cannot be processed.",
            halted_early=True,
            halt_reason="INJECTION_DETECTED",
            latency_ms=(time.monotonic() - start_time) * 1000,
        )

    llm = ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key.get_secret_value(),
        temperature=0.0,  # deterministic for operational queries
        max_tokens=1024,
    )

    messages: list[Any] = [
        SystemMessage(content=REACT_SYSTEM_PROMPT),
        HumanMessage(content=f"Query: {query}"),
    ]

    final_answer: str | None = None
    halt_reason = ""

    for step_num in range(1, max_steps + 1):
        logger.debug("ReAct step %d/%d", step_num, max_steps)

        try:
            response = await llm.ainvoke(messages)
            raw_text = response.content if isinstance(response.content, str) else str(response.content)
            tokens = response.usage_metadata.get("total_tokens", 0) if response.usage_metadata else 0
            total_tokens += tokens
        except Exception as exc:
            logger.exception("ReAct LLM call failed at step %d", step_num)
            final_answer = f"Agent error at step {step_num}: {exc}"
            halt_reason = "LLM_ERROR"
            break

        parsed = _parse_react_response(raw_text)

        if parsed["final_answer"]:
            final_answer = parsed["final_answer"]
            break

        action = parsed.get("action", "")
        action_input = parsed.get("action_input", {})
        thought = parsed.get("thought", raw_text[:200])

        if not action:
            logger.warning("ReAct step %d: no action parsed from LLM response", step_num)
            if step_num == max_steps:
                final_answer = thought or "Unable to determine an answer — insufficient data."
                halt_reason = "MAX_STEPS"
            continue

        # Execute the tool
        observation = await _call_tool(action, action_input, conn)

        react_step = ReactStep(
            thought=thought,
            action=action,
            action_input=action_input,
            observation=observation[:600],
            tokens_used=tokens,
        )
        steps.append(react_step)

        # Feed observation back into the conversation
        messages.append(AIMessage(content=raw_text))
        messages.append(HumanMessage(content=f"Observation: {observation[:800]}"))

        # Collect citations from document retrieval steps
        if action == "retrieve_documents" and "Source:" in observation:
            for line in observation.split("\n"):
                if line.strip().startswith("Source:"):
                    citations.append(line.strip().replace("Source:", "").strip())

    if final_answer is None:
        final_answer = "Maximum reasoning steps reached without a conclusive answer. Consult a production engineer."
        halt_reason = "MAX_STEPS"

    elapsed_ms = (time.monotonic() - start_time) * 1000

    logger.info(
        "ReAct complete: query='%s...' steps=%d tokens=%d latency=%.0fms",
        query[:60],
        len(steps),
        total_tokens,
        elapsed_ms,
    )

    return ReactResult(
        query=query,
        final_answer=final_answer,
        steps=steps,
        total_tokens=total_tokens,
        latency_ms=elapsed_ms,
        citations=citations,
        halted_early=bool(halt_reason),
        halt_reason=halt_reason,
    )
