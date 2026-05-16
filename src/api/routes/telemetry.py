"""Telemetry endpoints — well production data access."""

from __future__ import annotations

import logging
from typing import Any

import psycopg
import psycopg.rows
from fastapi import APIRouter, HTTPException, status

from src.config import get_settings
from src.tools.timeseries_query import query_timeseries

router = APIRouter()
logger = logging.getLogger(__name__)
settings = get_settings()


@router.get(
    "/wells/{well_id:path}/telemetry",
    summary="Get recent well telemetry",
)
async def get_well_telemetry(
    well_id: str,
    hours_back: int = 72,
) -> dict[str, Any]:
    if hours_back > 8760:  # max 1 year
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="hours_back cannot exceed 8760 (1 year)",
        )

    db_url = settings.database_url.replace("+psycopg", "")
    async with await psycopg.AsyncConnection.connect(db_url) as conn:
        result = await query_timeseries(conn, well_id, hours_back=hours_back)

    return result


@router.get(
    "/wells",
    summary="List available wells",
)
async def list_wells() -> list[dict[str, Any]]:
    db_url = settings.database_url.replace("+psycopg", "")
    async with await psycopg.AsyncConnection.connect(db_url) as conn:
        async with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
            await cur.execute(
                """
                SELECT DISTINCT well_id, field_name,
                       MIN(timestamp) AS first_reading,
                       MAX(timestamp) AS last_reading,
                       COUNT(*) AS total_readings
                FROM well_telemetry
                GROUP BY well_id, field_name
                ORDER BY field_name, well_id
                """
            )
            rows = await cur.fetchall()
    return [dict(r) for r in rows]
