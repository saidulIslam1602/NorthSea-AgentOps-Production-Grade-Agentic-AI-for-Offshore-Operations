"""
Kafka consumer for well telemetry → anomaly detection pipeline.

Consumes from `well.telemetry`, runs WellAnomalyDetector,
publishes AnomalyAlerts to `well.anomalies`.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from typing import Any

import pandas as pd
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer

from src.anomaly.detector import WellAnomalyDetector
from src.config import get_settings
from src.schemas.domain import AnomalyAlert

logger = logging.getLogger(__name__)
settings = get_settings()

# One detector instance per well, lazily created
_detectors: dict[str, WellAnomalyDetector] = {}


def _get_detector(well_id: str) -> WellAnomalyDetector:
    if well_id not in _detectors:
        _detectors[well_id] = WellAnomalyDetector(well_id=well_id)
    return _detectors[well_id]


def _parse_reading(msg_value: bytes) -> dict[str, Any] | None:
    try:
        data: dict[str, Any] = json.loads(msg_value.decode("utf-8"))
        if "timestamp" in data and isinstance(data["timestamp"], str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return data
    except Exception:
        logger.exception("Failed to parse telemetry message")
        return None


async def run_anomaly_detector() -> None:
    """Run the anomaly detection consumer loop."""
    consumer = AIOKafkaConsumer(
        settings.kafka_topic_telemetry,
        bootstrap_servers=settings.kafka_bootstrap_servers,
        group_id=f"{settings.kafka_consumer_group}-anomaly",
        value_deserializer=None,
        auto_offset_reset="latest",
    )
    producer = AIOKafkaProducer(
        bootstrap_servers=settings.kafka_bootstrap_servers,
        value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
    )

    await consumer.start()
    await producer.start()
    logger.info("Anomaly detector consumer started on topic: %s", settings.kafka_topic_telemetry)

    try:
        async for msg in consumer:
            reading = _parse_reading(msg.value)
            if reading is None:
                continue

            well_id = reading.get("well_id", "unknown")
            detector = _get_detector(well_id)
            alert: AnomalyAlert | None = detector.ingest(reading)

            if alert is not None:
                alert_dict = alert.model_dump(mode="json")
                await producer.send(settings.kafka_topic_anomalies, value=alert_dict)
                logger.warning(
                    "ANOMALY [%s] %s: %s (score=%.3f)",
                    alert.severity.value,
                    well_id,
                    alert.description[:100],
                    alert.anomaly_score,
                )
    finally:
        await consumer.stop()
        await producer.stop()


class TelemetryProducer:
    """Produces well telemetry messages to Kafka (for simulation and testing)."""

    def __init__(self) -> None:
        self._producer: AIOKafkaProducer | None = None

    async def start(self) -> None:
        self._producer = AIOKafkaProducer(
            bootstrap_servers=settings.kafka_bootstrap_servers,
            value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
        )
        await self._producer.start()

    async def stop(self) -> None:
        if self._producer:
            await self._producer.stop()

    async def send_reading(self, reading: dict[str, Any]) -> None:
        if not self._producer:
            raise RuntimeError("Producer not started")
        await self._producer.send(settings.kafka_topic_telemetry, value=reading)

    async def send_dataframe(self, df: pd.DataFrame, delay_ms: int = 100) -> None:
        """Stream a DataFrame of readings to Kafka with a delay between each."""
        for _, row in df.iterrows():
            await self.send_reading(row.to_dict())
            if delay_ms > 0:
                await asyncio.sleep(delay_ms / 1000)


if __name__ == "__main__":
    asyncio.run(run_anomaly_detector())
