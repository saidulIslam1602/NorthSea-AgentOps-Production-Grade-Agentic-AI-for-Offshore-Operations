"""Investigation endpoints."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any
from uuid import UUID

import psycopg
import psycopg.rows
from fastapi import APIRouter, BackgroundTasks, HTTPException, status
from pydantic import BaseModel, Field

from src.agents.orchestrator import investigate_anomaly
from src.config import get_settings
from src.safety.injection_guard import check_user_query
from src.schemas.domain import AnomalyAlert, InvestigationResult, SeverityLevel

router = APIRouter()
logger = logging.getLogger(__name__)
settings = get_settings()


class TriggerInvestigationRequest(BaseModel):
    """Request body for manually triggering an investigation."""
    well_id: str
    field_name: str
    severity: SeverityLevel = SeverityLevel.MEDIUM
    anomaly_score: float = Field(0.7, ge=0, le=1)
    affected_features: list[str] = Field(default_factory=lambda: ["oil_rate_bopd"])
    baseline_values: dict[str, float] = Field(default_factory=dict)
    current_values: dict[str, float] = Field(default_factory=dict)
    deviation_pct: dict[str, float] = Field(default_factory=dict)
    description: str = "Manual investigation trigger"
    skip_human_checkpoint: bool = True


class InvestigationResponse(BaseModel):
    """API response for an investigation result."""
    investigation_id: str | None
    well_id: str
    severity: str
    root_cause_hypothesis: str
    recommended_actions: list[str]
    confidence_score: float
    evidence_coverage: float
    risk_level: str
    should_escalate: bool
    escalation_reasons: list[str]
    escalation_message: str | None
    citations_count: int
    total_tokens_used: int
    latency_ms: float
    timestamp: datetime


@router.post(
    "/investigate",
    response_model=InvestigationResponse,
    status_code=status.HTTP_200_OK,
    summary="Trigger anomaly investigation",
    description=(
        "Runs the full planner-executor-critic pipeline for the given anomaly alert. "
        "Returns a recommendation with confidence score and escalation decision."
    ),
)
async def trigger_investigation(
    request: TriggerInvestigationRequest,
) -> InvestigationResponse:
    # Safety check on description input
    inj_check = check_user_query(request.description)
    if not inj_check.is_clean and inj_check.severity == "CRITICAL":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Request description contains prohibited content.",
        )

    alert = AnomalyAlert(
        timestamp=datetime.utcnow(),
        well_id=request.well_id,
        field_name=request.field_name,
        severity=request.severity,
        anomaly_score=request.anomaly_score,
        affected_features=request.affected_features,
        baseline_values=request.baseline_values,
        current_values=request.current_values,
        deviation_pct=request.deviation_pct,
        description=request.description,
    )

    db_url = settings.database_url.replace("+psycopg", "")
    try:
        async with await psycopg.AsyncConnection.connect(db_url) as conn:
            result: InvestigationResult = await investigate_anomaly(
                alert=alert,
                conn=conn,
                skip_human_checkpoint=request.skip_human_checkpoint,
            )
    except Exception as exc:
        logger.exception("Investigation failed for %s", request.well_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Investigation pipeline failed: {exc}",
        ) from exc

    return InvestigationResponse(
        investigation_id=str(result.investigation_id) if result.investigation_id else None,
        well_id=result.alert.well_id,
        severity=result.alert.severity.value,
        root_cause_hypothesis=result.root_cause_hypothesis,
        recommended_actions=result.recommended_actions,
        confidence_score=result.confidence_score,
        evidence_coverage=result.evidence_coverage,
        risk_level=result.risk_level.value,
        should_escalate=result.should_escalate,
        escalation_reasons=[r.value for r in result.escalation_reasons],
        escalation_message=result.escalation_message,
        citations_count=len(result.citations),
        total_tokens_used=result.total_tokens_used,
        latency_ms=result.latency_ms,
        timestamp=result.timestamp,
    )


@router.get(
    "/investigations",
    summary="List recent investigations",
)
async def list_investigations(
    limit: int = 20,
    well_id: str | None = None,
) -> list[dict[str, Any]]:
    db_url = settings.database_url.replace("+psycopg", "")
    async with await psycopg.AsyncConnection.connect(db_url) as conn:
        async with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
            if well_id:
                await cur.execute(
                    "SELECT * FROM investigations WHERE well_id = %s ORDER BY created_at DESC LIMIT %s",
                    (well_id, limit),
                )
            else:
                await cur.execute(
                    "SELECT * FROM investigations ORDER BY created_at DESC LIMIT %s",
                    (limit,),
                )
            rows = await cur.fetchall()
    return [dict(r) for r in rows]
