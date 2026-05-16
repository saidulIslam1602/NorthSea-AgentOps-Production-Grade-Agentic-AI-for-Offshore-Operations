"""Simulation endpoints — inject synthetic anomaly alerts for dev/test purposes.

This router is only active when APP_ENV != "production". It provides a way to
drive the full investigation pipeline (planner → executor → critic → escalation)
without waiting for the real anomaly detector to fire on live telemetry.

Routes:
  POST /api/v1/simulate/anomaly  — build a synthetic AnomalyAlert and run investigation
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

import psycopg
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from src.agents.orchestrator import investigate_anomaly
from src.config import get_settings
from src.schemas.domain import AnomalyAlert, SeverityLevel

router = APIRouter()
logger = logging.getLogger(__name__)
settings = get_settings()

# Realistic baseline sensor readings for the Volve field (15/9-F wells).
_VOLVE_BASELINES: dict[str, float] = {
    "oil_rate_bopd": 1200.0,
    "water_cut_pct": 42.0,
    "gas_oil_ratio": 850.0,
    "bhp_psi": 3200.0,
    "wh_temp_f": 145.0,
    "choke_64ths": 32.0,
}

# Pre-defined anomaly scenarios so callers can request a named scenario rather
# than crafting raw sensor values from scratch.
_SCENARIOS: dict[str, dict[str, Any]] = {
    "oil_rate_drop": {
        "affected_features": ["oil_rate_bopd", "bhp_psi"],
        "current_values": {"oil_rate_bopd": 480.0, "bhp_psi": 2800.0},
        "deviation_pct": {"oil_rate_bopd": -60.0, "bhp_psi": -12.5},
        "severity": SeverityLevel.HIGH,
        "anomaly_score": 0.87,
        "description": (
            "Simulated 60% oil rate drop with declining BHP — possible tubing restriction or wax deposition."
        ),
    },
    "high_water_cut": {
        "affected_features": ["water_cut_pct", "oil_rate_bopd"],
        "current_values": {"water_cut_pct": 78.0, "oil_rate_bopd": 820.0},
        "deviation_pct": {"water_cut_pct": 85.7, "oil_rate_bopd": -31.7},
        "severity": SeverityLevel.MEDIUM,
        "anomaly_score": 0.72,
        "description": "Simulated water-cut surge to 78% — possible water breakthrough from aquifer.",
    },
    "pressure_anomaly": {
        "affected_features": ["bhp_psi", "wh_temp_f"],
        "current_values": {"bhp_psi": 2100.0, "wh_temp_f": 118.0},
        "deviation_pct": {"bhp_psi": -34.4, "wh_temp_f": -18.6},
        "severity": SeverityLevel.CRITICAL,
        "anomaly_score": 0.94,
        "description": "Simulated critical pressure and temperature drop — possible wellbore integrity concern.",
    },
    "gor_spike": {
        "affected_features": ["gas_oil_ratio"],
        "current_values": {"gas_oil_ratio": 2400.0},
        "deviation_pct": {"gas_oil_ratio": 182.4},
        "severity": SeverityLevel.MEDIUM,
        "anomaly_score": 0.69,
        "description": "Simulated GOR spike to 2400 scf/bbl — possible gas coning from cap gas.",
    },
}


class SimulateAnomalyRequest(BaseModel):
    """Request body for injecting a synthetic anomaly alert."""

    well_id: str = Field("15/9-F-12", description="Well identifier (Volve field convention)")
    field_name: str = Field("Volve", description="Field name")
    scenario: str | None = Field(
        None,
        description=(
            "Named scenario: 'oil_rate_drop', 'high_water_cut', 'pressure_anomaly', 'gor_spike'. "
            "When set, sensor_overrides are merged on top of the scenario defaults."
        ),
    )
    sensor_overrides: dict[str, float] = Field(
        default_factory=dict,
        description="Explicit sensor values that override scenario defaults or baselines.",
    )
    severity: SeverityLevel | None = Field(
        None,
        description="Override alert severity (defaults to scenario severity or MEDIUM).",
    )
    anomaly_score: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Override anomaly score (0–1). Defaults to scenario value or 0.75.",
    )
    skip_human_checkpoint: bool = Field(
        True,
        description="Skip the human-in-the-loop pause gate (recommended for automated testing).",
    )


class SimulateAnomalyResponse(BaseModel):
    """Response envelope for a simulated investigation."""

    simulated: bool = True
    scenario: str | None
    alert_id: str
    investigation_id: str | None
    well_id: str
    severity: str
    anomaly_score: float
    root_cause_hypothesis: str
    recommended_actions: list[str]
    confidence_score: float
    should_escalate: bool
    escalation_reasons: list[str]
    total_tokens_used: int
    latency_ms: float


@router.post(
    "/simulate/anomaly",
    response_model=SimulateAnomalyResponse,
    status_code=status.HTTP_200_OK,
    summary="Inject a synthetic anomaly and run investigation (dev only)",
    description=(
        "Builds a synthetic AnomalyAlert from a named scenario or explicit sensor overrides "
        "and runs the full planner-executor-critic investigation pipeline. "
        "**Only available when APP_ENV != 'production'.**"
    ),
    tags=["Simulation"],
)
async def simulate_anomaly(
    request: SimulateAnomalyRequest,
) -> SimulateAnomalyResponse:
    if settings.app_env == "production":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Simulation endpoints are disabled in production.",
        )

    # Resolve scenario defaults
    scenario_data: dict[str, Any] = {}
    if request.scenario:
        if request.scenario not in _SCENARIOS:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(f"Unknown scenario '{request.scenario}'. Valid options: {sorted(_SCENARIOS)}"),
            )
        scenario_data = _SCENARIOS[request.scenario]

    # Merge: baselines → scenario overrides → explicit request overrides
    current_values: dict[str, float] = {**_VOLVE_BASELINES}
    current_values.update(scenario_data.get("current_values", {}))
    current_values.update(request.sensor_overrides)

    affected_features: list[str] = scenario_data.get(
        "affected_features",
        list(request.sensor_overrides.keys()) or ["oil_rate_bopd"],
    )
    deviation_pct: dict[str, float] = {
        k: round(
            (current_values[k] - _VOLVE_BASELINES.get(k, current_values[k]))
            / max(abs(_VOLVE_BASELINES.get(k, current_values[k])), 1e-9)
            * 100,
            1,
        )
        for k in affected_features
        if k in current_values
    }
    deviation_pct.update(scenario_data.get("deviation_pct", {}))

    severity = request.severity or scenario_data.get("severity", SeverityLevel.MEDIUM)
    anomaly_score = (
        request.anomaly_score if request.anomaly_score is not None else scenario_data.get("anomaly_score", 0.75)
    )
    description = scenario_data.get(
        "description",
        f"Simulated anomaly on {request.well_id} — sensor overrides: {request.sensor_overrides}",
    )

    alert = AnomalyAlert(
        timestamp=datetime.now(UTC),
        well_id=request.well_id,
        field_name=request.field_name,
        severity=severity,
        anomaly_score=anomaly_score,
        affected_features=affected_features,
        baseline_values={k: _VOLVE_BASELINES.get(k, 0.0) for k in affected_features},
        current_values={k: current_values[k] for k in affected_features if k in current_values},
        deviation_pct=deviation_pct,
        description=description,
    )

    db_url = settings.database_url.replace("+psycopg", "")
    try:
        async with await psycopg.AsyncConnection.connect(db_url) as conn:
            result = await investigate_anomaly(
                alert=alert,
                conn=conn,
                skip_human_checkpoint=request.skip_human_checkpoint,
            )
    except Exception as exc:
        logger.exception("Simulated investigation failed for well %s", request.well_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Simulation pipeline failed: {exc}",
        ) from exc

    return SimulateAnomalyResponse(
        scenario=request.scenario,
        alert_id=str(alert.alert_id),
        investigation_id=str(result.investigation_id) if result.investigation_id else None,
        well_id=result.alert.well_id,
        severity=result.alert.severity.value,
        anomaly_score=result.alert.anomaly_score,
        root_cause_hypothesis=result.root_cause_hypothesis,
        recommended_actions=result.recommended_actions,
        confidence_score=result.confidence_score,
        should_escalate=result.should_escalate,
        escalation_reasons=[r.value for r in result.escalation_reasons],
        total_tokens_used=result.total_tokens_used,
        latency_ms=result.latency_ms,
    )
