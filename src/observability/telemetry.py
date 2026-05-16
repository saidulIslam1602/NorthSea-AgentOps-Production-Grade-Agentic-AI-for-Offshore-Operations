"""
OpenTelemetry setup and custom metric helpers.

Instruments:
  - FastAPI HTTP traces (via opentelemetry-instrumentation-fastapi)
  - Custom business metrics:
      * agent_investigation_duration_seconds
      * agent_confidence_score (histogram)
      * agent_escalation_total (counter)
      * agent_tokens_used_total (counter)
      * rag_retrieval_duration_seconds
      * rag_source_coverage (histogram)
      * anomaly_detected_total (counter, labelled by severity)
"""

from __future__ import annotations

import logging
from typing import Any

from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from prometheus_client import Counter, Gauge, Histogram

from src.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# ─── OpenTelemetry ────────────────────────────────────────────────────────────

_tracer: trace.Tracer | None = None
_meter: metrics.Meter | None = None


def setup_telemetry(app: Any) -> None:
    """Configure OpenTelemetry tracing and metrics, instrument FastAPI."""
    global _tracer, _meter

    resource = Resource.create({
        "service.name": settings.otel_service_name,
        "service.version": "0.1.0",
        "deployment.environment": settings.app_env,
    })

    # ─── Tracing ──────────────────────────────────────────────────────────────
    tracer_provider = TracerProvider(resource=resource)
    try:
        span_exporter = OTLPSpanExporter(
            endpoint=f"{settings.otel_exporter_otlp_endpoint}/v1/traces",
        )
        tracer_provider.add_span_processor(BatchSpanProcessor(span_exporter))
    except Exception:
        logger.warning("OTLP trace exporter unavailable — traces will not be exported")

    trace.set_tracer_provider(tracer_provider)
    _tracer = trace.get_tracer(settings.otel_service_name)

    # ─── Metrics ──────────────────────────────────────────────────────────────
    try:
        metric_reader = PeriodicExportingMetricReader(
            OTLPMetricExporter(
                endpoint=f"{settings.otel_exporter_otlp_endpoint}/v1/metrics",
            ),
            export_interval_millis=30_000,
        )
        meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
        metrics.set_meter_provider(meter_provider)
    except Exception:
        logger.warning("OTLP metric exporter unavailable")

    _meter = metrics.get_meter(settings.otel_service_name)

    # ─── FastAPI instrumentation ───────────────────────────────────────────────
    FastAPIInstrumentor.instrument_app(app)

    logger.info("OpenTelemetry configured: service=%s", settings.otel_service_name)


def get_tracer() -> trace.Tracer:
    if _tracer is None:
        return trace.get_tracer(settings.otel_service_name)
    return _tracer


# ─── Prometheus Metrics ───────────────────────────────────────────────────────

investigation_duration = Histogram(
    "agent_investigation_duration_seconds",
    "Time taken for full investigation pipeline",
    labelnames=["well_id", "severity"],
    buckets=[1, 5, 10, 30, 60, 120, 300],
)

confidence_score_histogram = Histogram(
    "agent_confidence_score",
    "Confidence score distribution from Critic agent",
    labelnames=["severity", "escalated"],
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0],
)

escalation_counter = Counter(
    "agent_escalations_total",
    "Total escalations triggered",
    labelnames=["reason", "risk_level"],
)

tokens_used_counter = Counter(
    "agent_tokens_used_total",
    "Total LLM tokens consumed",
    labelnames=["model", "agent"],
)

rag_retrieval_duration = Histogram(
    "rag_retrieval_duration_seconds",
    "Time taken for hybrid RAG retrieval",
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
)

rag_source_coverage = Histogram(
    "rag_source_coverage",
    "Source coverage fraction from retrieval",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.8, 0.9, 1.0],
)

anomaly_detected_counter = Counter(
    "anomaly_detected_total",
    "Total anomalies detected by the detector",
    labelnames=["severity", "field_name"],
)

prompt_injection_counter = Counter(
    "prompt_injection_detected_total",
    "Total prompt injection attempts detected",
    labelnames=["severity"],
)

active_investigations = Gauge(
    "agent_active_investigations",
    "Number of investigations currently running",
)
