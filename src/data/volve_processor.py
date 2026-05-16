"""
Volve Open Dataset Processor.

Processes the Equinor Volve open dataset (equinor.com/energy/volve-data-sharing).
The Volve field produced from 2008-2016 in Block 15/9, Norwegian North Sea.

Usage:
  1. Download the Volve dataset from https://www.equinor.com/energy/volve-data-sharing
  2. Place production CSV files in data/volve/
  3. Run: python -m src.data.volve_processor --input data/volve/ --output data/synthetic/

If the dataset is not available, this module gracefully falls back to the
synthetic generator.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

VOLVE_COLUMN_MAP: dict[str, str] = {
    "DATEPRD": "timestamp",
    "NPD_WELL_BORE_NAME": "well_id",
    "BORE_OIL_VOL": "oil_rate_bopd",
    "BORE_WAT_VOL": "water_rate_bwpd",
    "BORE_GAS_VOL": "gas_rate_mmscfd",
    "BORE_WI_VOL": "water_injection_bwpd",
    "FLOW_KIND": "flow_kind",
}

VOLVE_FIELD_NAME = "Volve"


def _compute_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Derive water cut, GOR, and approximate BHP from raw Volve production columns."""
    df = df.copy()

    total_liquid = df["oil_rate_bopd"] + df.get("water_rate_bwpd", 0)
    df["water_cut_pct"] = np.where(
        total_liquid > 0,
        100.0 * df.get("water_rate_bwpd", 0) / total_liquid,
        0.0,
    )

    df["gas_oil_ratio"] = np.where(
        df["oil_rate_bopd"] > 0,
        df["gas_rate_mmscfd"] * 1_000_000 / df["oil_rate_bopd"],
        0.0,
    )

    # Approximate BHP using a linear PI model (placeholder — real BHP needs gauges)
    df["bhp_psi"] = 3_200.0 - df["oil_rate_bopd"] * 0.04
    df["wh_temp_f"] = 140.0
    df["choke_64ths"] = 48.0
    df["is_injector"] = df.get("flow_kind", "production").str.lower().str.contains("inject")

    return df


def load_volve_production(volve_dir: Path) -> pd.DataFrame | None:
    """Load and normalise Volve production CSV files."""
    csv_files = list(volve_dir.glob("*.csv"))
    if not csv_files:
        logger.warning("No Volve CSV files found in %s — falling back to synthetic data", volve_dir)
        return None

    dfs: list[pd.DataFrame] = []
    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path, sep=";", encoding="latin-1", low_memory=False)

            # Rename known columns
            rename = {k: v for k, v in VOLVE_COLUMN_MAP.items() if k in df.columns}
            df.rename(columns=rename, inplace=True)

            if "timestamp" not in df.columns:
                logger.warning("No date column in %s — skipping", csv_path)
                continue

            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df.dropna(subset=["timestamp"], inplace=True)
            df["field_name"] = VOLVE_FIELD_NAME

            for col in ["oil_rate_bopd", "water_rate_bwpd", "gas_rate_mmscfd"]:
                if col not in df.columns:
                    df[col] = 0.0
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

            df = df[df["oil_rate_bopd"] > 0]  # drop non-producing days
            df = _compute_derived_metrics(df)
            dfs.append(df)

        except Exception:
            logger.exception("Failed to load %s", csv_path)
            continue

    if not dfs:
        return None

    combined = pd.concat(dfs, ignore_index=True)
    combined.sort_values(["well_id", "timestamp"], inplace=True)

    keep_cols = [
        "timestamp", "well_id", "field_name", "oil_rate_bopd", "water_cut_pct",
        "gas_oil_ratio", "bhp_psi", "wh_temp_f", "choke_64ths", "is_injector",
    ]
    return combined[[c for c in keep_cols if c in combined.columns]]


def export_normalised(volve_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Load Volve data and export normalised telemetry CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_volve_production(volve_dir)

    if df is None:
        logger.warning("Volve data unavailable — generating synthetic Volve-shaped data instead")
        from src.data.synthetic_generator import generate_well_timeseries
        from datetime import datetime, timezone

        synthetic_wells = ["15/9-F-1C", "15/9-F-4", "15/9-F-5", "15/9-F-11H", "15/9-F-12H"]
        dfs = []
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        for well in synthetic_wells:
            well_df = generate_well_timeseries(well, "Volve", start, n_hours=720)
            dfs.append(well_df)
        df = pd.concat(dfs, ignore_index=True)

    out_path = output_dir / "volve_production.csv"
    df.to_csv(out_path, index=False)

    summary: dict[str, Any] = {
        "source": str(volve_dir),
        "output": str(out_path),
        "total_rows": len(df),
        "wells": sorted(df["well_id"].unique().tolist()),
        "date_range": [
            str(df["timestamp"].min()),
            str(df["timestamp"].max()),
        ],
    }

    (output_dir / "volve_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/volve/", help="Directory containing Volve CSVs")
    parser.add_argument("--output", default="data/synthetic/", help="Output directory")
    args = parser.parse_args()

    summary = export_normalised(Path(args.input), Path(args.output))
    print(json.dumps(summary, indent=2))
