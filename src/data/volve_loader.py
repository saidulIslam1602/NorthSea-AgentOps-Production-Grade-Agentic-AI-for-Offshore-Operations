"""
Volve Open Dataset Loader — Real Production Data from Equinor.

Source: Equinor Volve Data Village (Databricks Marketplace)
License: Equinor Open Data License
Field: Volve, Block 15/9, Norwegian North Sea, 2007–2016

Converts the real Volve production Excel into the internal telemetry
schema used by WellAnomalyDetector and the training pipeline.

Column mapping (Volve → internal):
  BORE_OIL_VOL   Sm³/day  → oil_rate_bopd    (× 6.28981)
  BORE_WAT_VOL   Sm³/day  → water_cut_pct     (derived)
  BORE_GAS_VOL   Sm³/day  → gas_oil_ratio     (scf/bbl)
  AVG_DOWNHOLE_PRESSURE bar → bhp_psi         (× 14.5038)
  AVG_WHT_P      °C       → wh_temp_f         (×9/5 +32)
  AVG_CHOKE_SIZE_P  %     → choke_64ths       (×0.64)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Unit conversions
SM3_TO_BBL = 6.28981
BAR_TO_PSI = 14.5038
SM3_PER_MSCF = 35.315  # 1000 scf per Sm³ of gas

# Volve daily sheet columns we use
VOLVE_COLS = [
    "DATEPRD",
    "NPD_WELL_BORE_NAME",
    "NPD_FIELD_NAME",
    "FLOW_KIND",
    "ON_STREAM_HRS",
    "BORE_OIL_VOL",
    "BORE_WAT_VOL",
    "BORE_GAS_VOL",
    "AVG_DOWNHOLE_PRESSURE",
    "AVG_WHP_P",
    "AVG_WHT_P",
    "AVG_CHOKE_SIZE_P",
]

# Only these wells are oil producers in Volve
PRODUCER_WELLS = {"15/9-F-1 C", "15/9-F-11", "15/9-F-12", "15/9-F-14", "15/9-F-15 D", "15/9-F-5"}
INJECTOR_WELLS = {"15/9-F-4"}


def load_volve_daily(
    xlsx_path: Path,
    producers_only: bool = True,
    min_on_stream_hrs: float = 1.0,
) -> pd.DataFrame:
    """
    Load the Volve daily production sheet and return a clean telemetry DataFrame.

    Args:
        xlsx_path: Path to 'Volve production data.xlsx'.
        producers_only: If True, filter to FLOW_KIND='production' wells only.
        min_on_stream_hrs: Drop rows where well was effectively offline.

    Returns:
        DataFrame with columns: timestamp, well_id, field_name,
        oil_rate_bopd, water_cut_pct, gas_oil_ratio, bhp_psi,
        wh_temp_f, choke_64ths, on_stream_hrs, is_producing
    """
    logger.info("Loading Volve daily production data from %s", xlsx_path)

    raw = pd.read_excel(xlsx_path, sheet_name="Daily Production Data", usecols=VOLVE_COLS)
    raw["DATEPRD"] = pd.to_datetime(raw["DATEPRD"], errors="coerce")
    raw = raw.dropna(subset=["DATEPRD", "NPD_WELL_BORE_NAME"])

    if producers_only:
        raw = raw[raw["FLOW_KIND"] == "production"]

    # Drop fully offline days
    raw = raw[raw["ON_STREAM_HRS"] >= min_on_stream_hrs]

    # Numeric coercion
    for col in [
        "BORE_OIL_VOL",
        "BORE_WAT_VOL",
        "BORE_GAS_VOL",
        "AVG_DOWNHOLE_PRESSURE",
        "AVG_WHP_P",
        "AVG_WHT_P",
        "AVG_CHOKE_SIZE_P",
    ]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce").fillna(0.0).clip(lower=0)

    df = pd.DataFrame()

    # ── Identifiers ──────────────────────────────────────────────────────────
    df["timestamp"] = raw["DATEPRD"].values
    df["well_id"] = raw["NPD_WELL_BORE_NAME"].str.strip().values
    df["field_name"] = raw["NPD_FIELD_NAME"].fillna("VOLVE").values
    df["on_stream_hrs"] = raw["ON_STREAM_HRS"].values

    # ── Oil rate: Sm³/day → BOPD ─────────────────────────────────────────────
    df["oil_rate_bopd"] = (raw["BORE_OIL_VOL"] * SM3_TO_BBL).round(1).values

    # ── Water cut: WAT/(OIL+WAT) × 100 ───────────────────────────────────────
    total_liquid = raw["BORE_OIL_VOL"] + raw["BORE_WAT_VOL"].clip(lower=0)
    df["water_cut_pct"] = np.where(
        total_liquid > 0,
        (raw["BORE_WAT_VOL"].clip(lower=0) / total_liquid * 100).round(1),
        0.0,
    )

    # ── GOR: Sm³_gas / Sm³_oil → scf/bbl ────────────────────────────────────
    # 1 Sm³ gas = 35.315 scf; 1 Sm³ oil = 6.28981 bbl
    # GOR [scf/bbl] = (GAS_Sm3 × 35.315) / (OIL_Sm3 × 6.28981)
    df["gas_oil_ratio"] = np.where(
        raw["BORE_OIL_VOL"] > 0,
        (raw["BORE_GAS_VOL"] * SM3_PER_MSCF / (raw["BORE_OIL_VOL"] * SM3_TO_BBL)).round(0),
        0.0,
    )

    # ── BHP: bar → psi ────────────────────────────────────────────────────────
    df["bhp_psi"] = (raw["AVG_DOWNHOLE_PRESSURE"] * BAR_TO_PSI).round(0).values

    # ── Wellhead temp: °C → °F ───────────────────────────────────────────────
    df["wh_temp_f"] = (raw["AVG_WHT_P"] * 9 / 5 + 32).round(1).values

    # ── Choke: % → 64ths (approximate) ──────────────────────────────────────
    df["choke_64ths"] = (raw["AVG_CHOKE_SIZE_P"] * 0.64).round(0).values

    # ── Producing flag ────────────────────────────────────────────────────────
    df["is_producing"] = (raw["BORE_OIL_VOL"] > 0).values

    df = df.sort_values(["well_id", "timestamp"]).reset_index(drop=True)

    logger.info(
        "Volve data loaded: %d rows, %d wells, %s to %s",
        len(df),
        df["well_id"].nunique(),
        df["timestamp"].min().date(),
        df["timestamp"].max().date(),
    )
    return df


def get_per_well_stats(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Compute summary statistics per well from real Volve data."""
    stats = {}
    for well, g in df[df["is_producing"]].groupby("well_id"):
        stats[str(well)] = {
            "producing_days": int(len(g)),
            "peak_oil_bopd": float(g["oil_rate_bopd"].max()),
            "avg_oil_bopd": float(g["oil_rate_bopd"].mean()),
            "peak_water_cut_pct": float(g["water_cut_pct"].max()),
            "avg_water_cut_pct": float(g["water_cut_pct"].mean()),
            "avg_gor_scf_bbl": float(g["gas_oil_ratio"].mean()),
            "avg_bhp_psi": float(g["bhp_psi"][g["bhp_psi"] > 0].mean()) if (g["bhp_psi"] > 0).any() else 0,
            "date_range": f"{g['timestamp'].min().date()} → {g['timestamp'].max().date()}",
        }
    return stats
