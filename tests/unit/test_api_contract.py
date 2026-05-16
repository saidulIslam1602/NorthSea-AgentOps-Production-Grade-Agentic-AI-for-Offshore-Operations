"""HTTP API smoke and contract checks (no DB required for mocks; health may degrade)."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient


def _minimal_investigation_result(alert):  # type: ignore[no-untyped-def]
    from src.schemas.domain import InvestigationResult, RiskLevel

    return InvestigationResult(
        investigation_id=uuid4(),
        alert=alert,
        timestamp=datetime.now(UTC),
        root_cause_hypothesis="Suspected choke drift (test fixture)",
        supporting_evidence=["Fixture evidence"],
        recommended_actions=["Review choke calibration"],
        citations=[],
        confidence_score=0.82,
        evidence_coverage=0.71,
        risk_level=RiskLevel.MEDIUM,
        should_escalate=False,
        escalation_reasons=[],
        escalation_message=None,
        agent_steps=[],
        total_tokens_used=42,
        latency_ms=150.0,
    )


@pytest.fixture
def api_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Fresh TestClient — clears settings cache after env tweaks."""

    from src.api import main as api_main
    from src.config import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("APP_ENV", "development")
    return TestClient(api_main.app)


def test_openapi_json_available(api_client: TestClient) -> None:
    resp = api_client.get("/openapi.json")
    assert resp.status_code == 200
    data = resp.json()
    assert "openapi" in data
    assert "paths" in data
    assert "/api/v1/investigate" in data["paths"]


def test_metrics_endpoint_returns_200(api_client: TestClient) -> None:
    resp = api_client.get("/metrics")
    assert resp.status_code == 200
    body = resp.text
    # Prometheus exposition format exposes HELP/TYPE comments; instrumentation adds http_* spans.
    assert "#" in body or "http_" in body


def test_health_payload_shape(api_client: TestClient) -> None:
    resp = api_client.get("/health")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["status"] in ("healthy", "degraded")
    assert payload["version"] == "0.1.0"
    assert payload["database"] in ("connected", "disconnected")
    assert isinstance(payload["env"], str) and payload["env"]


def test_post_investigate_returns_structured_payload(api_client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Contract for trigger investigation endpoint with orchestrator mocked."""

    class _DummyAsyncConnCtx:
        __slots__ = ()

        async def __aenter__(self) -> MagicMock:
            return MagicMock()

        async def __aexit__(self, *_exc: object) -> None:
            return None

    async def fake_connect(*_args: object, **_kwargs: object) -> _DummyAsyncConnCtx:
        return _DummyAsyncConnCtx()

    async def fake_investigate(*_args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        from src.schemas.domain import AnomalyAlert, SeverityLevel

        alert = kwargs.get("alert")
        fake_alert = alert or AnomalyAlert(
            timestamp=datetime.now(UTC),
            well_id="15/9-F-12",
            field_name="Volve",
            severity=SeverityLevel.MEDIUM,
            anomaly_score=0.7,
            affected_features=["oil_rate_bopd"],
            baseline_values={"oil_rate_bopd": 1000.0},
            current_values={"oil_rate_bopd": 600.0},
            deviation_pct={"oil_rate_bopd": -40.0},
            description="Test alert",
        )
        return _minimal_investigation_result(fake_alert)

    monkeypatch.setattr(
        "src.api.routes.investigations.psycopg.AsyncConnection.connect",
        fake_connect,
    )

    monkeypatch.setattr(
        "src.api.routes.investigations.investigate_anomaly",
        fake_investigate,
    )

    resp = api_client.post(
        "/api/v1/investigate",
        json={
            "well_id": "15/9-F-12",
            "field_name": "Volve",
            "severity": "MEDIUM",
            "description": "Test investigation via API contract test",
            "skip_human_checkpoint": True,
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["well_id"] == "15/9-F-12"
    assert body["confidence_score"] == 0.82
    assert body["should_escalate"] is False
    assert "root_cause_hypothesis" in body


def test_simulate_requires_unknown_scenario_422(api_client: TestClient) -> None:
    resp = api_client.post(
        "/api/v1/simulate/anomaly",
        json={"scenario": "not-a-real-scenario-key"},
    )
    assert resp.status_code == 422
    body = resp.json()
    detail = body.get("detail", "")
    if isinstance(detail, str):
        assert "Unknown scenario" in detail or "scenario" in detail.lower()
    else:
        payload = repr(detail)
        assert "Unknown scenario" in payload


def test_utc_now_returns_timezone_aware() -> None:
    """Covers default_factory path for domain models (keeps unit cov gate stable)."""
    from src.schemas.domain import _utc_now

    dt = _utc_now()
    assert dt.tzinfo is UTC
    """Dev-only simulate route returns 403 when route settings report production."""
    import src.api.routes.simulate as sim_mod
    from src.api import main as api_main

    monkeypatch.setattr(
        sim_mod,
        "settings",
        MagicMock(app_env="production"),
    )

    client = TestClient(api_main.app)
    resp = client.post("/api/v1/simulate/anomaly", json={"scenario": "oil_rate_drop"})
    assert resp.status_code == 403
    assert resp.json()["detail"] == "Simulation endpoints are disabled in production."
