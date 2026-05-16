"""
FastAPI application entry point.

Routes:
  POST /api/v1/investigate          — trigger investigation for an anomaly alert
  GET  /api/v1/investigations/{id}  — get investigation result
  GET  /api/v1/escalations          — list open escalations
  POST /api/v1/escalations/{id}/resolve — resolve an escalation
  POST /api/v1/rag/query            — RAG copilot endpoint
  GET  /api/v1/wells/{well_id}/telemetry — get recent telemetry
  POST /api/v1/simulate/anomaly     — inject a test anomaly (dev only)
  GET  /health                      — health check
  GET  /metrics                     — Prometheus metrics
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Any

import psycopg
import structlog
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator

from src.api.routes import investigations, rag, telemetry, escalations
from src.config import get_settings
from src.observability.telemetry import setup_telemetry

logger = structlog.get_logger(__name__)
settings = get_settings()

# ─── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle."""
    # Setup OpenTelemetry
    setup_telemetry(app)

    # Verify database connectivity
    db_url = settings.database_url.replace("+psycopg", "")
    try:
        async with await psycopg.AsyncConnection.connect(db_url) as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT 1")
        logger.info("database_connected", url=db_url.split("@")[-1])
    except Exception as exc:
        logger.error("database_connection_failed", error=str(exc))

    logger.info(
        "northsea_agentops_started",
        env=settings.app_env,
        host=settings.api_host,
        port=settings.api_port,
    )

    yield

    logger.info("northsea_agentops_shutdown")


# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="NorthSea AgentOps API",
    description=(
        "Production-grade agentic AI for offshore oil & gas operations. "
        "Detects production anomalies, reasons about root causes, retrieves "
        "operational knowledge, and escalates uncertain decisions to human engineers."
    ),
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ─── Middleware ────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.app_env == "development" else [],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Any, call_next: Any) -> Any:
    start = time.monotonic()
    response = await call_next(request)
    latency_ms = (time.monotonic() - start) * 1000
    logger.info(
        "http_request",
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        latency_ms=round(latency_ms, 1),
    )
    return response


# ─── Prometheus ───────────────────────────────────────────────────────────────

Instrumentator().instrument(app).expose(app, endpoint="/metrics")

# ─── Routers ──────────────────────────────────────────────────────────────────

app.include_router(investigations.router, prefix="/api/v1", tags=["Investigations"])
app.include_router(rag.router, prefix="/api/v1", tags=["RAG Copilot"])
app.include_router(telemetry.router, prefix="/api/v1", tags=["Telemetry"])
app.include_router(escalations.router, prefix="/api/v1", tags=["Escalations"])

# ─── Health Check ─────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
async def health_check() -> dict[str, Any]:
    db_url = settings.database_url.replace("+psycopg", "")
    db_ok = False
    try:
        async with await psycopg.AsyncConnection.connect(db_url, connect_timeout=3) as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT 1")
        db_ok = True
    except Exception:
        pass

    return {
        "status": "healthy" if db_ok else "degraded",
        "version": "0.1.0",
        "env": settings.app_env,
        "database": "connected" if db_ok else "disconnected",
    }
