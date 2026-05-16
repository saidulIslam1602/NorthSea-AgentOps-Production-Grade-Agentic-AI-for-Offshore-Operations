"""Core domain schemas shared across the system."""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class SeverityLevel(str, enum.Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class RiskLevel(str, enum.Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class EscalationReason(str, enum.Enum):
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    LOW_EVIDENCE_COVERAGE = "LOW_EVIDENCE_COVERAGE"
    HIGH_HSE_RISK = "HIGH_HSE_RISK"
    MISSING_DATA = "MISSING_DATA"
    INJECTION_DETECTED = "INJECTION_DETECTED"
    OPERATOR_OVERRIDE = "OPERATOR_OVERRIDE"


# ─── Telemetry ────────────────────────────────────────────────────────────────

class WellTelemetry(BaseModel):
    """Single time-series reading from a well instrument."""

    event_id: UUID = Field(default_factory=uuid4)
    timestamp: datetime
    well_id: str
    field_name: str
    oil_rate_bopd: float = Field(..., description="Oil production rate (barrels/day)")
    water_cut_pct: float = Field(..., ge=0, le=100, description="Water cut percentage")
    gas_oil_ratio: float = Field(..., description="GOR (scf/bbl)")
    bottomhole_pressure_psi: float = Field(..., description="BHP (psi)")
    wellhead_temperature_f: float = Field(..., description="Wellhead temperature (°F)")
    choke_size_64ths: float = Field(..., description="Choke opening (64ths inch)")
    is_injector: bool = Field(False)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AnomalyAlert(BaseModel):
    """Anomaly detected in well telemetry."""

    alert_id: UUID = Field(default_factory=uuid4)
    timestamp: datetime
    well_id: str
    field_name: str
    severity: SeverityLevel
    anomaly_score: float = Field(..., ge=0.0, le=1.0)
    affected_features: list[str]
    baseline_values: dict[str, float]
    current_values: dict[str, float]
    deviation_pct: dict[str, float]
    description: str


# ─── Investigation ────────────────────────────────────────────────────────────

class Citation(BaseModel):
    """Document citation from RAG retrieval."""

    document_id: str
    document_title: str
    document_type: str
    section: str | None = None
    page: int | None = None
    relevance_score: float = Field(..., ge=0.0, le=1.0)
    excerpt: str


class InvestigationStep(BaseModel):
    """Single step in an investigation plan."""

    step_id: int
    description: str
    tool_to_use: str
    expected_output: str
    completed: bool = False
    result: str | None = None


class InvestigationResult(BaseModel):
    """Final output from the agent investigation pipeline."""

    investigation_id: UUID = Field(default_factory=uuid4)
    alert: AnomalyAlert
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Agent outputs
    root_cause_hypothesis: str
    supporting_evidence: list[str]
    recommended_actions: list[str]
    citations: list[Citation]

    # Scoring
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    evidence_coverage: float = Field(..., ge=0.0, le=1.0)
    risk_level: RiskLevel

    # Escalation
    should_escalate: bool
    escalation_reasons: list[EscalationReason]
    escalation_message: str | None = None

    # Audit
    agent_steps: list[dict[str, Any]] = Field(default_factory=list)
    total_tokens_used: int = 0
    latency_ms: float = 0.0


# ─── Escalation ───────────────────────────────────────────────────────────────

class EscalationRecord(BaseModel):
    """Human escalation record stored in the queue."""

    escalation_id: UUID = Field(default_factory=uuid4)
    investigation_id: UUID
    well_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    resolved_at: datetime | None = None
    resolved_by: str | None = None
    resolution_notes: str | None = None
    reasons: list[EscalationReason]
    risk_level: RiskLevel
    confidence_score: float
    summary: str
    status: str = "PENDING"
