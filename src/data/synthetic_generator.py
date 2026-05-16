"""
Synthetic well-level timeseries for tests and demos (not production-facing Volve ingestion).
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

TELEMETRY_COLS = [
    "oil_rate_bopd",
    "water_cut_pct",
    "gas_oil_ratio",
    "bhp_psi",
    "wh_temp_f",
    "choke_64ths",
]


def generate_well_timeseries(
    well_id: str,
    field_name: str,
    start_ts: datetime,
    n_hours: int,
    *,
    anomaly_type: str | None = None,
    anomaly_onset_hour: int = 100,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """
    Build an hourly DataFrame resembling well telemetry suitable for anomaly tests.

    After ``anomaly_onset_hour``, ``water_breakthrough`` ramps water_cut and reduces oil_rate.
    """
    if n_hours <= 0:
        return pd.DataFrame(columns=["timestamp", "well_id", "field_name", *TELEMETRY_COLS])
    rnd = rng or np.random.default_rng(42)

    rows: list[dict[str, Any]] = []
    for h in range(n_hours):
        t = start_ts + timedelta(hours=h)
        base_oil = 2500.0 + rnd.normal(0, 50.0)
        base_water = 15.0 + rnd.uniform(-1.5, 1.5)
        gor = 650.0 + rnd.uniform(-25, 25)
        bhp = 3200.0 + rnd.uniform(-30, 30)
        temp_f = 145.0 + rnd.uniform(-3, 3)
        choke = 48.0 + rnd.uniform(-1, 1)

        if anomaly_type == "water_breakthrough" and h >= anomaly_onset_hour:
            progress = min(1.0, (h - anomaly_onset_hour) / max(48, n_hours - anomaly_onset_hour))
            base_water += 45.0 * progress
            base_oil *= max(0.35, 1.0 - 0.55 * progress)

        base_water = float(np.clip(base_water, 0.0, 100.0))
        choke = float(np.clip(choke, 0.0, 64.0))

        rows.append(
            {
                "timestamp": t,
                "well_id": well_id,
                "field_name": field_name,
                "oil_rate_bopd": max(0.0, base_oil),
                "water_cut_pct": base_water,
                "gas_oil_ratio": max(50.0, gor),
                "bhp_psi": max(200.0, bhp),
                "wh_temp_f": temp_f,
                "choke_64ths": choke,
            },
        )

    return pd.DataFrame(rows)
