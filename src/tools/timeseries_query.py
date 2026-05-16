"""
Tool: query_timeseries

Queries well production telemetry from PostgreSQL for a specified time window.
Returns a formatted summary including trend analysis and statistical profile.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import psycopg
import psycopg.rows

from src.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

TELEMETRY_FEATURES = [
    "oil_rate_bopd",
    "water_cut_pct",
    "gas_oil_ratio",
    "bhp_psi",
    "wh_temp_f",
    "choke_64ths",
]


async def query_timeseries(
    conn: psycopg.AsyncConnection[Any],
    well_id: str,
    hours_back: int = 72,
    end_time: datetime | None = None,
) -> dict[str, Any]:
    """
    Retrieve and summarise well telemetry for the specified lookback window.

    Returns:
        {
            "well_id": str,
            "period": {"start": str, "end": str, "hours": int},
            "record_count": int,
            "statistics": dict per feature (mean, std, min, max, trend_pct),
            "data_quality": {"completeness": float, "gaps": list},
            "summary_text": str (human-readable summary for LLM context),
        }
    """
    end = end_time or datetime.now(UTC)
    start = end - timedelta(hours=hours_back)

    async with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
        await cur.execute(
            """
            SELECT timestamp, oil_rate_bopd, water_cut_pct, gas_oil_ratio,
                   bhp_psi, wh_temp_f, choke_64ths
            FROM well_telemetry
            WHERE well_id = %s AND timestamp BETWEEN %s AND %s
            ORDER BY timestamp ASC
            """,
            (well_id, start, end),
        )
        rows = await cur.fetchall()

    if not rows:
        return {
            "well_id": well_id,
            "period": {"start": start.isoformat(), "end": end.isoformat(), "hours": hours_back},
            "record_count": 0,
            "statistics": {},
            "data_quality": {"completeness": 0.0, "gaps": []},
            "summary_text": f"No telemetry data found for well {well_id} in the last {hours_back} hours.",
        }

    data = {feat: [float(r[feat]) for r in rows if r.get(feat) is not None] for feat in TELEMETRY_FEATURES}
    timestamps = [r["timestamp"] for r in rows]

    statistics: dict[str, dict[str, float]] = {}
    for feat, values in data.items():
        if not values:
            continue
        arr = np.array(values)
        # Trend: % change from first 10% to last 10% of the window
        n = len(arr)
        early_mean = np.mean(arr[: max(1, n // 10)])
        late_mean = np.mean(arr[max(0, -n // 10) :])
        trend_pct = ((late_mean - early_mean) / abs(early_mean) * 100) if early_mean != 0 else 0.0

        statistics[feat] = {
            "mean": round(float(np.mean(arr)), 2),
            "std": round(float(np.std(arr)), 2),
            "min": round(float(np.min(arr)), 2),
            "max": round(float(np.max(arr)), 2),
            "trend_pct": round(float(trend_pct), 1),
        }

    # Data quality
    expected_readings = hours_back
    completeness = min(1.0, len(rows) / max(1, expected_readings))
    gaps = _find_gaps(timestamps, expected_gap_hours=2)

    summary = _build_summary_text(well_id, hours_back, statistics, len(rows), completeness)

    return {
        "well_id": well_id,
        "period": {"start": start.isoformat(), "end": end.isoformat(), "hours": hours_back},
        "record_count": len(rows),
        "statistics": statistics,
        "data_quality": {"completeness": round(completeness, 3), "gaps": gaps},
        "summary_text": summary,
    }


def _find_gaps(timestamps: list[datetime], expected_gap_hours: int = 2) -> list[dict[str, Any]]:
    """Find time gaps larger than expected_gap_hours in the timestamp series."""
    if len(timestamps) < 2:
        return []
    gaps = []
    for i in range(1, len(timestamps)):
        delta = (timestamps[i] - timestamps[i - 1]).total_seconds() / 3600
        if delta > expected_gap_hours:
            gaps.append(
                {
                    "from": timestamps[i - 1].isoformat(),
                    "to": timestamps[i].isoformat(),
                    "duration_hours": round(delta, 1),
                }
            )
    return gaps[:10]  # cap at 10 gaps


def _build_summary_text(
    well_id: str,
    hours: int,
    stats: dict[str, dict[str, float]],
    n_records: int,
    completeness: float,
) -> str:
    lines = [
        (
            f"Telemetry summary for {well_id} over the last {hours} hours "
            f"({n_records} readings, {completeness:.0%} complete):"
        )
    ]

    for feat, s in stats.items():
        trend_dir = "▲" if s["trend_pct"] > 2 else ("▼" if s["trend_pct"] < -2 else "→")
        lines.append(
            f"  {feat}: mean={s['mean']}, std={s['std']}, "
            f"range=[{s['min']}, {s['max']}], trend={trend_dir}{abs(s['trend_pct']):.1f}%"
        )

    # Add domain interpretation
    if "oil_rate_bopd" in stats and stats["oil_rate_bopd"]["trend_pct"] < -15:
        lines.append("  ⚠ ALERT: Oil rate declining trend >15% — investigate.")
    if "water_cut_pct" in stats and stats["water_cut_pct"]["trend_pct"] > 10:
        lines.append("  ⚠ ALERT: Water cut increasing trend >10% — possible water breakthrough.")
    if "bhp_psi" in stats and stats["bhp_psi"]["trend_pct"] < -10:
        lines.append("  ⚠ ALERT: BHP declining >10% — check reservoir pressure or pump health.")
    if "gas_oil_ratio" in stats and stats["gas_oil_ratio"]["trend_pct"] > 30:
        lines.append("  ⚠ ALERT: GOR increasing >30% — possible gas coning or separator issue.")

    return "\n".join(lines)
