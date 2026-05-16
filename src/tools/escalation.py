"""
Tool: create_escalation

Persists escalation records to PostgreSQL and publishes to Kafka.
"""

from __future__ import annotations

import json
import logging
from typing import Any
from uuid import uuid4

import psycopg

from src.config import get_settings
from src.schemas.domain import EscalationReason, EscalationRecord, RiskLevel

logger = logging.getLogger(__name__)
settings = get_settings()


async def create_escalation(state: dict[str, Any]) -> EscalationRecord:
    """Persist an escalation record from the investigation state."""

    alert = state["alert"]
    investigation_id = state.get("investigation_id")
    reasons: list[EscalationReason] = state.get("escalation_reasons", [])
    risk_level: RiskLevel = state.get("risk_level", RiskLevel.HIGH)
    confidence_score: float = state.get("confidence_score", 0.0)
    recommendation: str = state.get("recommendation", "")
    message: str | None = state.get("escalation_message")

    record = EscalationRecord(
        escalation_id=uuid4(),
        investigation_id=investigation_id,  # type: ignore[arg-type]  # state dict value is UUID at runtime
        well_id=alert.well_id,
        reasons=reasons,
        risk_level=risk_level,
        confidence_score=confidence_score,
        summary=message or f"Investigation escalated for {alert.well_id}: {recommendation[:200]}",
    )

    # Persist to database
    db_url = settings.database_url.replace("+psycopg", "")
    try:
        async with await psycopg.AsyncConnection.connect(db_url) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO escalations
                        (escalation_id, investigation_id, well_id, reasons,
                         risk_level, confidence_score, summary, status)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, 'PENDING')
                    ON CONFLICT (escalation_id) DO NOTHING
                    """,
                    (
                        str(record.escalation_id),
                        str(investigation_id) if investigation_id else None,
                        alert.well_id,
                        [r.value for r in reasons],
                        risk_level.value,
                        confidence_score,
                        record.summary,
                    ),
                )
            await conn.commit()
    except Exception:
        logger.exception("Failed to persist escalation record")

    # Publish to Kafka (best effort)
    try:
        from aiokafka import AIOKafkaProducer

        producer = AIOKafkaProducer(
            bootstrap_servers=settings.kafka_bootstrap_servers,
            value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
        )
        await producer.start()
        try:
            await producer.send(
                settings.kafka_topic_escalations,
                value=record.model_dump(mode="json"),
            )
        finally:
            await producer.stop()
    except Exception:
        logger.warning("Failed to publish escalation to Kafka (non-fatal)")

    logger.warning(
        "ESCALATION created: %s for well %s (risk=%s, confidence=%.2f)",
        record.escalation_id,
        alert.well_id,
        risk_level.value,
        confidence_score,
    )

    return record
