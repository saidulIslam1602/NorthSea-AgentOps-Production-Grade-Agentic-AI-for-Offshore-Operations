"""
Tool allowlist — per-agent permitted tool registry.

Each agent declares exactly which tools it is permitted to call.
The orchestrator enforces this at dispatch time.
Any attempt by an agent to call an unlisted tool is blocked and logged.
"""

from __future__ import annotations

from typing import FrozenSet

AGENT_TOOL_PERMISSIONS: dict[str, FrozenSet[str]] = {
    "planner": frozenset({
        # Planner generates plans but does NOT call tools directly
    }),
    "executor": frozenset({
        "query_timeseries",
        "retrieve_documents",
        "query_similar_incidents",
        "get_well_metadata",
        "get_equipment_status",
    }),
    "critic": frozenset({
        # Critic reviews evidence already collected — no tool calls
    }),
    "uncertainty_gate": frozenset({
        # Gate applies rules — no tool calls
    }),
    "escalation": frozenset({
        "create_escalation",
        "notify_engineer",
    }),
}

# Tools that require explicit human approval before execution
HUMAN_APPROVAL_REQUIRED: FrozenSet[str] = frozenset({
    "modify_choke",
    "adjust_gaslift_rate",
    "initiate_esd",
    "restart_esp",
    "modify_chemical_injection",
    "notify_psa",
})

# Tools completely blocked in all contexts
BLOCKED_TOOLS: FrozenSet[str] = frozenset({
    "execute_shell",
    "write_file",
    "delete_record",
    "modify_config",
    "send_external_email",
})


def is_tool_permitted(agent_id: str, tool_name: str) -> bool:
    """Check whether an agent is permitted to call a tool."""
    if tool_name in BLOCKED_TOOLS:
        return False
    permitted = AGENT_TOOL_PERMISSIONS.get(agent_id, frozenset())
    return tool_name in permitted


def requires_human_approval(tool_name: str) -> bool:
    """Check whether a tool requires human approval before execution."""
    return tool_name in HUMAN_APPROVAL_REQUIRED


def get_permitted_tools(agent_id: str) -> list[str]:
    """Return sorted list of permitted tools for an agent."""
    return sorted(AGENT_TOOL_PERMISSIONS.get(agent_id, frozenset()))


def validate_tool_call(
    agent_id: str,
    tool_name: str,
    audit_logger: "AuditLogger | None" = None,  # noqa: F821
) -> tuple[bool, str]:
    """
    Validate a tool call, returning (is_permitted, reason).

    Logs violations to audit logger if provided.
    """
    if tool_name in BLOCKED_TOOLS:
        reason = f"Tool '{tool_name}' is in the global blocked list"
        if audit_logger:
            audit_logger.log_blocked_tool_call(agent_id, tool_name, reason)
        return False, reason

    permitted = AGENT_TOOL_PERMISSIONS.get(agent_id, frozenset())
    if tool_name not in permitted:
        reason = (
            f"Agent '{agent_id}' is not permitted to call '{tool_name}'. "
            f"Permitted: {sorted(permitted)}"
        )
        if audit_logger:
            audit_logger.log_blocked_tool_call(agent_id, tool_name, reason)
        return False, reason

    if requires_human_approval(tool_name):
        reason = f"Tool '{tool_name}' requires explicit human approval"
        return False, reason

    return True, "permitted"
