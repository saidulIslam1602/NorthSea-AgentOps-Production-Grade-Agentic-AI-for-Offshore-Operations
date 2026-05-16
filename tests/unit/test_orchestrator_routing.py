"""Deterministic tests for orchestrator routing and initial LangGraph state (no LLM calls)."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from src.agents.orchestrator import _should_continue_executing
from src.agents.state import initial_state
from src.schemas.domain import AnomalyAlert, SeverityLevel


def _dummy_alert() -> AnomalyAlert:
    return AnomalyAlert(
        timestamp=datetime.now(UTC),
        well_id="15/9-F-12",
        field_name="Volve",
        severity=SeverityLevel.MEDIUM,
        anomaly_score=0.7,
        affected_features=["oil_rate_bopd"],
        baseline_values={"oil_rate_bopd": 1000.0},
        current_values={"oil_rate_bopd": 600.0},
        deviation_pct={"oil_rate_bopd": -40.0},
        description="Unit test anomaly",
    )


def test_initial_state_contains_expected_keys() -> None:
    """Sanity-check graph bootstrap fields after ``initial_state``."""
    alert = _dummy_alert()
    state = initial_state(alert)

    assert state["messages"] == []
    assert state["alert"].well_id == "15/9-F-12"
    assert state["plan_steps"] == []
    assert state["current_step_index"] == 0
    assert state["investigation_id"] is not None


def test_should_continue_executing_with_steps_remaining(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "src.agents.orchestrator.settings",
        SimpleNamespace(agent_max_iterations=10),
    )
    state = {"plan_steps": [{"step_id": 1}, {"step_id": 2}], "current_step_index": 0, "agent_steps": []}
    assert _should_continue_executing(state) == "executor"


def test_should_continue_executing_moves_to_critic_when_done(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "src.agents.orchestrator.settings",
        SimpleNamespace(agent_max_iterations=10),
    )
    state = {"plan_steps": [{"step_id": 1}], "current_step_index": 1, "agent_steps": []}
    assert _should_continue_executing(state) == "critic"


def test_should_continue_executing_respects_iteration_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "src.agents.orchestrator.settings",
        SimpleNamespace(agent_max_iterations=3),
    )
    state = {"plan_steps": [{"step_id": i} for i in range(20)], "current_step_index": 0, "agent_steps": [{}, {}, {}]}
    assert _should_continue_executing(state) == "critic"
