"""
Structured agent audit logger.

Every agent action is logged with:
  - investigation_id
  - agent_id
  - step_name
  - tool_called (if any)
  - input_hash (SHA-256 of input for tamper-evidence without storing PII)
  - output_hash
  - tokens_used
  - latency_ms
  - success / error_message

Logs are written to:
  1. PostgreSQL agent_audit_log table
  2. structlog structured stdout (consumed by OpenTelemetry)
"""

from __future__ import annotations

import hashlib
import logging
import time
from typing import Any
from uuid import UUID

import structlog

logger = structlog.get_logger(__name__)
std_logger = logging.getLogger(__name__)


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


class AuditLogger:
    """
    Lightweight audit logger that writes to structlog and (optionally) to DB.

    Instantiate per-investigation with the investigation_id.
    """

    def __init__(
        self,
        investigation_id: UUID | str | None,
        conn: Any | None = None,
    ) -> None:
        self.investigation_id = str(investigation_id) if investigation_id else None
        self._conn = conn
        self._bound_logger = logger.bind(
            investigation_id=self.investigation_id,
            service="northsea-agentops",
        )

    def log_agent_step(
        self,
        agent_id: str,
        step_name: str,
        input_text: str,
        output_text: str,
        tool_called: str | None = None,
        tokens_used: int = 0,
        latency_ms: float = 0.0,
        success: bool = True,
        error_message: str | None = None,
    ) -> dict[str, Any]:
        """Log a single agent step. Returns the log record."""
        record: dict[str, Any] = {
            "investigation_id": self.investigation_id,
            "agent_id": agent_id,
            "step_name": step_name,
            "tool_called": tool_called,
            "input_hash": _hash(input_text),
            "output_hash": _hash(output_text),
            "tokens_used": tokens_used,
            "latency_ms": latency_ms,
            "success": success,
            "error_message": error_message,
        }

        level = "warning" if not success else "info"
        getattr(self._bound_logger, level)(
            "agent_step",
            agent_id=agent_id,
            step_name=step_name,
            tool=tool_called,
            tokens=tokens_used,
            latency_ms=round(latency_ms, 1),
            success=success,
        )

        return record

    def log_injection_detection(
        self,
        chunk_id: str,
        severity: str,
        matches: list[str],
        input_hash: str,
    ) -> None:
        """Log a prompt injection detection event."""
        self._bound_logger.warning(
            "prompt_injection_detected",
            chunk_id=chunk_id,
            severity=severity,
            num_matches=len(matches),
            input_hash=input_hash,
        )

    def log_blocked_tool_call(
        self,
        agent_id: str,
        tool_name: str,
        reason: str,
    ) -> None:
        """Log a blocked tool call attempt."""
        self._bound_logger.warning(
            "blocked_tool_call",
            agent_id=agent_id,
            tool_name=tool_name,
            reason=reason,
        )

    async def persist_to_db(self, records: list[dict[str, Any]]) -> None:
        """Persist audit log records to the database."""
        if not self._conn:
            return
        try:
            async with self._conn.cursor() as cur:
                for record in records:
                    await cur.execute(
                        """
                        INSERT INTO agent_audit_log
                            (investigation_id, agent_id, step_name, tool_called,
                             input_hash, output_hash, tokens_used, latency_ms,
                             success, error_message)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            record.get("investigation_id"),
                            record.get("agent_id"),
                            record.get("step_name"),
                            record.get("tool_called"),
                            record.get("input_hash"),
                            record.get("output_hash"),
                            record.get("tokens_used", 0),
                            record.get("latency_ms", 0.0),
                            record.get("success", True),
                            record.get("error_message"),
                        ),
                    )
            await self._conn.commit()
        except Exception:
            std_logger.exception("Failed to persist audit log to DB")


class TimedStep:
    """Context manager for timing an agent step and logging to AuditLogger."""

    def __init__(
        self,
        audit_logger: AuditLogger,
        agent_id: str,
        step_name: str,
        tool_called: str | None = None,
    ) -> None:
        self._audit = audit_logger
        self._agent_id = agent_id
        self._step_name = step_name
        self._tool = tool_called
        self._start: float = 0.0
        self._input: str = ""
        self._record: dict[str, Any] = {}

    def set_input(self, text: str) -> None:
        self._input = text

    def __enter__(self) -> TimedStep:
        self._start = time.monotonic()
        return self

    def __exit__(
        self,
        exc_type: type | None,
        exc_val: Exception | None,
        exc_tb: Any,
    ) -> None:
        latency_ms = (time.monotonic() - self._start) * 1000
        self._record = self._audit.log_agent_step(
            agent_id=self._agent_id,
            step_name=self._step_name,
            input_text=self._input,
            output_text="",
            tool_called=self._tool,
            latency_ms=latency_ms,
            success=exc_type is None,
            error_message=str(exc_val) if exc_val else None,
        )
