"""
Tool: get_equipment_status / get_well_metadata

Simulates equipment diagnostic API calls. In production these would call
real SCADA system APIs, asset management systems, or well databases.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import psycopg
import psycopg.rows

logger = logging.getLogger(__name__)


async def get_well_metadata(
    conn: psycopg.AsyncConnection[Any],
    well_id: str,
) -> dict[str, Any]:
    """Retrieve well metadata and latest production status."""
    async with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
        # Latest reading
        await cur.execute(
            """
            SELECT * FROM well_telemetry
            WHERE well_id = %s
            ORDER BY timestamp DESC
            LIMIT 1
            """,
            (well_id,),
        )
        latest = await cur.fetchone()

        # Recent alert count
        await cur.execute(
            """
            SELECT severity, COUNT(*) as count
            FROM anomaly_alerts
            WHERE well_id = %s AND timestamp > NOW() - INTERVAL '7 days'
            GROUP BY severity
            """,
            (well_id,),
        )
        alert_rows = await cur.fetchall()

    alert_summary = {row["severity"]: row["count"] for row in alert_rows}

    if not latest:
        return {
            "well_id": well_id,
            "status": "NO_DATA",
            "summary_text": f"No telemetry data available for well {well_id}.",
        }

    latest_dict = dict(latest)
    return {
        "well_id": well_id,
        "status": "ACTIVE",
        "latest_reading": {
            "timestamp": str(latest_dict.get("timestamp")),
            "oil_rate_bopd": latest_dict.get("oil_rate_bopd"),
            "water_cut_pct": latest_dict.get("water_cut_pct"),
            "gas_oil_ratio": latest_dict.get("gas_oil_ratio"),
            "bhp_psi": latest_dict.get("bhp_psi"),
            "wh_temp_f": latest_dict.get("wh_temp_f"),
            "choke_64ths": latest_dict.get("choke_64ths"),
        },
        "recent_alerts_7d": alert_summary,
        "summary_text": _well_status_summary(well_id, latest_dict, alert_summary),
    }


def _well_status_summary(
    well_id: str,
    latest: dict[str, Any],
    alerts: dict[str, int],
) -> str:
    alert_text = ""
    if alerts:
        parts = [f"{count} {sev}" for sev, count in sorted(alerts.items())]
        alert_text = f" Recent alerts (7d): {', '.join(parts)}."

    return (
        f"Well {well_id} last reading at {latest.get('timestamp', 'unknown')}: "
        f"oil={latest.get('oil_rate_bopd', 'N/A')} BOPD, "
        f"WC={latest.get('water_cut_pct', 'N/A')}%, "
        f"GOR={latest.get('gas_oil_ratio', 'N/A')} scf/bbl, "
        f"BHP={latest.get('bhp_psi', 'N/A')} psi."
        f"{alert_text}"
    )


async def get_equipment_status(
    conn: psycopg.AsyncConnection[Any],
    well_id: str,
    equipment_type: str = "all",
) -> dict[str, Any]:
    """
    Retrieve equipment status from maintenance logs.

    In production: would call CMMS (SAP PM, Maximo) API.
    Here: queries the document knowledge base for maintenance history.
    """
    # Query for recent maintenance events from the knowledge base
    async with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
        await cur.execute(
            """
            SELECT d.title, d.doc_type, dc.content, d.indexed_at
            FROM document_chunks dc
            JOIN documents d ON d.document_id = dc.document_id
            WHERE d.doc_type = 'Maintenance Log'
              AND (d.well_id = %s OR dc.content ILIKE %s)
            ORDER BY d.indexed_at DESC
            LIMIT 5
            """,
            (well_id, f"%{well_id}%"),
        )
        rows = await cur.fetchall()

    if not rows:
        return {
            "well_id": well_id,
            "equipment_type": equipment_type,
            "maintenance_records": [],
            "summary_text": f"No maintenance records found for well {well_id}.",
        }

    records = [
        {
            "title": row["title"],
            "doc_type": row["doc_type"],
            "excerpt": row["content"][:200],
        }
        for row in rows
    ]

    summary = (
        f"Found {len(records)} maintenance records for {well_id}: "
        + "; ".join(r["title"][:60] for r in records[:3])
    )

    return {
        "well_id": well_id,
        "equipment_type": equipment_type,
        "maintenance_records": records,
        "summary_text": summary,
    }
