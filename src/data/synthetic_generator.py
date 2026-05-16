"""
Synthetic SCADA / well-telemetry generator.

Produces realistic North Sea well production data shaped after the Equinor Volve
field (2008-2016). Three anomaly injection modes are supported:

  - water_breakthrough  : sudden step increase in water-cut
  - liquid_loading      : declining oil rate with rising GHP
  - separator_upset     : correlated drop in oil rate + GOR spike
  - choke_restriction   : gradual decline matching reduced choke
  - downhole_pump_failure: exponential decline in BHP + rate
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ─── Field / Well Master Data ─────────────────────────────────────────────────

FIELDS: list[dict[str, Any]] = [
    {"name": "Draugen", "wells": ["D-1H", "D-2H", "D-3H", "D-4AH", "D-5H"]},
    {"name": "Volve", "wells": ["15/9-F-1C", "15/9-F-4", "15/9-F-5", "15/9-F-11H", "15/9-F-12H"]},
    {"name": "Gullfaks", "wells": ["GF-A-1H", "GF-A-2H", "GF-B-1H", "GF-C-1H"]},
    {"name": "Oseberg", "wells": ["OS-1H", "OS-2H", "OS-3H", "OS-4H"]},
]

WELL_BASELINES: dict[str, dict[str, float]] = {
    "default": {
        "oil_rate_bopd": 2500.0,
        "water_cut_pct": 15.0,
        "gas_oil_ratio": 650.0,
        "bhp_psi": 3200.0,
        "wh_temp_f": 145.0,
        "choke_64ths": 48.0,
    }
}

# Per-well variation from baseline
WELL_VARIANCE: dict[str, dict[str, float]] = {
    "high_rate": {"oil_rate_bopd": 1.6, "bhp_psi": 1.1},
    "mature": {"oil_rate_bopd": 0.6, "water_cut_pct": 2.5, "gas_oil_ratio": 0.8},
    "injector": {"oil_rate_bopd": 0.0, "water_cut_pct": 0.0},
}


def _baseline_for_well(well_id: str) -> dict[str, float]:
    base = WELL_BASELINES["default"].copy()
    # Add deterministic per-well noise so each well has a distinct signature
    rng = np.random.default_rng(seed=abs(hash(well_id)) % 2**31)
    base["oil_rate_bopd"] += rng.normal(0, 300)
    base["water_cut_pct"] = max(2, base["water_cut_pct"] + rng.normal(0, 8))
    base["gas_oil_ratio"] += rng.normal(0, 80)
    base["bhp_psi"] += rng.normal(0, 200)
    base["wh_temp_f"] += rng.normal(0, 10)
    base["choke_64ths"] = float(random.choice([32, 40, 48, 56, 64]))
    return base


def _add_process_noise(row: dict[str, float], rng: np.random.Generator) -> dict[str, float]:
    noise = {
        "oil_rate_bopd": rng.normal(0, 30),
        "water_cut_pct": rng.normal(0, 0.5),
        "gas_oil_ratio": rng.normal(0, 15),
        "bhp_psi": rng.normal(0, 20),
        "wh_temp_f": rng.normal(0, 1.5),
        "choke_64ths": 0.0,
    }
    return {k: max(0.0, row[k] + noise.get(k, 0.0)) for k in row}


# ─── Anomaly Injection ────────────────────────────────────────────────────────

def inject_water_breakthrough(
    df: pd.DataFrame,
    onset_idx: int,
    magnitude: float = 3.0,
) -> pd.DataFrame:
    """Step increase in water cut with corresponding oil rate decline."""
    df = df.copy()
    n = len(df)
    for i in range(onset_idx, n):
        ramp = min(1.0, (i - onset_idx) / 24)
        df.at[i, "water_cut_pct"] = min(95.0, df.at[i, "water_cut_pct"] + magnitude * 15 * ramp)
        df.at[i, "oil_rate_bopd"] = max(0, df.at[i, "oil_rate_bopd"] * (1 - 0.4 * ramp))
    return df


def inject_liquid_loading(df: pd.DataFrame, onset_idx: int) -> pd.DataFrame:
    """Gradual oil rate decline with rising wellhead pressure symptoms."""
    df = df.copy()
    n = len(df)
    for i in range(onset_idx, n):
        decay = min(1.0, (i - onset_idx) / 48)
        df.at[i, "oil_rate_bopd"] = max(0, df.at[i, "oil_rate_bopd"] * (1 - 0.6 * decay))
        df.at[i, "bhp_psi"] = df.at[i, "bhp_psi"] * (1 + 0.15 * decay)
    return df


def inject_separator_upset(df: pd.DataFrame, onset_idx: int) -> pd.DataFrame:
    """Correlated oil rate drop and GOR spike from separator malfunction."""
    df = df.copy()
    n = len(df)
    for i in range(onset_idx, n):
        factor = min(1.0, (i - onset_idx) / 12)
        df.at[i, "oil_rate_bopd"] = max(0, df.at[i, "oil_rate_bopd"] * (1 - 0.5 * factor))
        df.at[i, "gas_oil_ratio"] = df.at[i, "gas_oil_ratio"] * (1 + 1.2 * factor)
    return df


def inject_choke_restriction(df: pd.DataFrame, onset_idx: int) -> pd.DataFrame:
    """Gradual choke closure → declining rate + pressure build-up."""
    df = df.copy()
    n = len(df)
    for i in range(onset_idx, n):
        factor = min(1.0, (i - onset_idx) / 36)
        df.at[i, "choke_64ths"] = max(4, df.at[i, "choke_64ths"] * (1 - 0.7 * factor))
        df.at[i, "oil_rate_bopd"] = max(0, df.at[i, "oil_rate_bopd"] * (1 - 0.55 * factor))
        df.at[i, "bhp_psi"] = df.at[i, "bhp_psi"] * (1 + 0.1 * factor)
    return df


def inject_pump_failure(df: pd.DataFrame, onset_idx: int) -> pd.DataFrame:
    """Exponential decline in BHP and production from ESP/pump failure."""
    df = df.copy()
    n = len(df)
    for i in range(onset_idx, n):
        t = i - onset_idx
        decay = np.exp(-t / 24)
        df.at[i, "bhp_psi"] = max(500, df.at[i, "bhp_psi"] * (0.3 + 0.7 * decay))
        df.at[i, "oil_rate_bopd"] = max(0, df.at[i, "oil_rate_bopd"] * (0.1 + 0.9 * decay))
    return df


ANOMALY_INJECTORS = {
    "water_breakthrough": inject_water_breakthrough,
    "liquid_loading": inject_liquid_loading,
    "separator_upset": inject_separator_upset,
    "choke_restriction": inject_choke_restriction,
    "pump_failure": inject_pump_failure,
}


# ─── Main Generator ───────────────────────────────────────────────────────────

def generate_well_timeseries(
    well_id: str,
    field_name: str,
    start_date: datetime,
    n_hours: int = 720,
    anomaly_type: str | None = None,
    anomaly_onset_hour: int | None = None,
) -> pd.DataFrame:
    """Generate hourly telemetry for a single well, optionally injecting an anomaly."""
    rng = np.random.default_rng(seed=abs(hash(well_id + str(start_date))) % 2**31)
    baseline = _baseline_for_well(well_id)

    records = []
    ts = start_date
    current = baseline.copy()

    for _ in range(n_hours):
        row = _add_process_noise(current, rng)
        records.append(
            {
                "timestamp": ts,
                "well_id": well_id,
                "field_name": field_name,
                "oil_rate_bopd": round(row["oil_rate_bopd"], 2),
                "water_cut_pct": round(min(99.9, max(0.0, row["water_cut_pct"])), 2),
                "gas_oil_ratio": round(max(0, row["gas_oil_ratio"]), 2),
                "bhp_psi": round(max(100, row["bhp_psi"]), 1),
                "wh_temp_f": round(row["wh_temp_f"], 1),
                "choke_64ths": round(row["choke_64ths"], 1),
                "is_anomaly": False,
            }
        )
        ts = ts + timedelta(hours=1)

    df = pd.DataFrame(records)

    if anomaly_type and anomaly_type in ANOMALY_INJECTORS:
        onset = anomaly_onset_hour or (n_hours // 2)
        df = ANOMALY_INJECTORS[anomaly_type](df, onset)
        df.loc[onset:, "is_anomaly"] = True

    return df


def generate_full_dataset(
    output_dir: Path,
    n_hours: int = 720,
    include_anomalies: bool = True,
) -> dict[str, Path]:
    """Generate telemetry for all fields and wells."""
    output_dir.mkdir(parents=True, exist_ok=True)
    start = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)

    output_files: dict[str, Path] = {}
    anomaly_types = list(ANOMALY_INJECTORS.keys())

    for field in FIELDS:
        field_name = field["name"]
        all_well_dfs = []

        for i, well_id in enumerate(field["wells"]):
            anomaly = None
            onset = None
            if include_anomalies and i < len(anomaly_types):
                anomaly = anomaly_types[i % len(anomaly_types)]
                onset = random.randint(n_hours // 3, 2 * n_hours // 3)

            df = generate_well_timeseries(
                well_id=well_id,
                field_name=field_name,
                start_date=start,
                n_hours=n_hours,
                anomaly_type=anomaly,
                anomaly_onset_hour=onset,
            )
            all_well_dfs.append(df)

        combined = pd.concat(all_well_dfs, ignore_index=True)
        out_path = output_dir / f"{field_name.lower()}_telemetry.csv"
        combined.to_csv(out_path, index=False)
        output_files[field_name] = out_path
        print(f"  Generated {len(combined)} rows → {out_path}")

    # Also generate a combined manifest
    manifest = {
        "generated_at": datetime.utcnow().isoformat(),
        "n_hours": n_hours,
        "fields": [f["name"] for f in FIELDS],
        "total_wells": sum(len(f["wells"]) for f in FIELDS),
        "anomaly_types": anomaly_types,
        "files": {k: str(v) for k, v in output_files.items()},
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    return output_files


def main() -> None:
    import typer
    app = typer.Typer()

    @app.command()
    def generate(
        output: str = "data/synthetic",
        hours: int = 720,
        no_anomalies: bool = False,
    ) -> None:
        """Generate synthetic well telemetry."""
        output_dir = Path(output)
        print(f"Generating synthetic telemetry → {output_dir}")
        files = generate_full_dataset(output_dir, n_hours=hours, include_anomalies=not no_anomalies)
        print(f"\nGenerated {len(files)} field datasets.")

    app()


if __name__ == "__main__":
    main()
