"""Escalation queue endpoints."""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

import psycopg
import psycopg.rows
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from src.config import get_settings

router = APIRouter()
logger = logging.getLogger(__name__)
settings = get_settings()


class ResolveEscalationRequest(BaseModel):
    resolved_by: str
    resolution_notes: str
    new_status: str = "RESOLVED"  # RESOLVED or DISMISSED


@router.get(
    "/escalations",
    summary="List escalations",
)
async def list_escalations(
    status_filter: str = "PENDING",
    limit: int = 50,
) -> list[dict[str, Any]]:
    valid_statuses = {"PENDING", "IN_REVIEW", "RESOLVED", "DISMISSED", "ALL"}
    if status_filter.upper() not in valid_statuses:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid status. Must be one of: {valid_statuses}",
        )

    db_url = settings.database_url.replace("+psycopg", "")
    async with await psycopg.AsyncConnection.connect(db_url) as conn:
        async with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
            if status_filter.upper() == "ALL":
                await cur.execute(
                    "SELECT * FROM escalations ORDER BY created_at DESC LIMIT %s",
                    (limit,),
                )
            else:
                await cur.execute(
                    "SELECT * FROM escalations WHERE status = %s ORDER BY created_at DESC LIMIT %s",
                    (status_filter.upper(), limit),
                )
            rows = await cur.fetchall()
    return [dict(r) for r in rows]


@router.post(
    "/escalations/{escalation_id}/resolve",
    summary="Resolve an escalation",
)
async def resolve_escalation(
    escalation_id: UUID,
    request: ResolveEscalationRequest,
) -> dict[str, Any]:
    valid_statuses = {"RESOLVED", "DISMISSED"}
    if request.new_status.upper() not in valid_statuses:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="new_status must be RESOLVED or DISMISSED",
        )

    db_url = settings.database_url.replace("+psycopg", "")
    async with await psycopg.AsyncConnection.connect(db_url) as conn:
        async with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
            await cur.execute(
                """
                UPDATE escalations
                SET status = %s,
                    resolved_by = %s,
                    resolution_notes = %s,
                    resolved_at = NOW()
                WHERE escalation_id = %s
                RETURNING *
                """,
                (request.new_status.upper(), request.resolved_by,
                 request.resolution_notes, str(escalation_id)),
            )
            row = await cur.fetchone()
        await conn.commit()

    if not row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Escalation {escalation_id} not found",
        )

    return dict(row)
