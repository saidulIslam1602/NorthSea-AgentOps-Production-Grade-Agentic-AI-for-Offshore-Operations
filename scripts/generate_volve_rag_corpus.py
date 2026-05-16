"""
Build Markdown artefacts from the Equinor Volve **production** Excel for RAG ingestion.

Uses ``src.data.volve_loader.load_volve_daily`` … Output goes under ``data/docs/volve_real/``.
Golden eval labels (`eval/golden_testset.json`) are regenerated from the same workbook via
``scripts/generate_volve_golden_testset.py`` (normally called from ``sync_real_volve_daily_rag.py``).

Requirements:
    data/Volve_Data/Volve production data.xlsx

License: obey Equinor Open Data terms.

Usage:
    python scripts/generate_volve_rag_corpus.py
    python -m src.rag.ingestion --docs-dir data/docs
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_XLSX = ROOT / "data" / "Volve_Data" / "Volve production data.xlsx"
DEFAULT_OUT = ROOT / "data" / "docs" / "volve_real"

logger = logging.getLogger(__name__)


def _slug(well_id: str) -> str:
    return well_id.strip().replace("/", "-").replace(" ", "_")


def _monthly_table(g: Any) -> str:
    """Return compact monthly mean oil/water-cut markdown table."""
    import pandas as pd

    gg = g.copy()
    gg = gg.sort_values("timestamp")
    gg["ym"] = pd.to_datetime(gg["timestamp"]).dt.to_period("M")
    piv = gg.groupby("ym", as_index=False).agg(
        days=("oil_rate_bopd", "count"),
        oil_bopd_mean=("oil_rate_bopd", "mean"),
        wct_mean=("water_cut_pct", "mean"),
    )
    piv = piv.tail(48)  # last 48 months keeps docs bounded
    lines = ["| Month | Days | Mean oil BOPD | Mean WCT % |", "|-------|-----:|--------------:|-----------:|"]
    for _, row in piv.iterrows():
        m = str(row["ym"])
        lines.append(f"| {m} | {int(row['days'])} | {row['oil_bopd_mean']:.1f} | {row['wct_mean']:.1f} |")
    return "\n".join(lines)


def _well_markdown(well_id: str, g: Any, stats: dict[str, Any]) -> str:
    field = str(g["field_name"].iloc[0])
    tmin = g["timestamp"].min().date()
    tmax = g["timestamp"].max().date()

    peaks = []
    peak_idx = int(g["oil_rate_bopd"].idxmax())
    peak_day = g.loc[peak_idx]
    peaks.append(
        f"Peak oil **{peak_day['oil_rate_bopd']:.1f} BOPD** on "
        f"{peak_day['timestamp'].date()} (WCT ~{peak_day['water_cut_pct']:.1f} %, "
        f"GOR ~{peak_day['gas_oil_ratio']:.0f} scf/bbl)."
    )

    wc_peak_idx = int(g["water_cut_pct"].idxmax())
    wc_peak = g.loc[wc_peak_idx]
    peaks.append(
        f"Peak water cut **{wc_peak['water_cut_pct']:.1f} %** on {wc_peak['timestamp'].date()} "
        f"(oil {wc_peak['oil_rate_bopd']:.1f} BOPD)."
    )

    body = f"""# Volve daily production — {well_id.strip()}

Operational knowledge extracted from **Equinor Volve Open Dataset** daily production workbook
(machine-generated summary for retrieval; verify against source files for decisions).

## Identifiers

- **Field:** {field}
- **NPD well / bore:** {well_id.strip()}
- **Flow:** production (`FLOW_KIND == production`)
- **Data window:** {tmin} → {tmax}
- **Producing day rows:** {stats["producing_days"]}

## Summary statistics ({tmin} to {tmax})

| Metric | Value |
|:-------|------:|
| Peak oil BOPD | {stats["peak_oil_bopd"]:.1f} |
| Mean oil BOPD | {stats["avg_oil_bopd"]:.1f} |
| Mean water cut % | {stats["avg_water_cut_pct"]:.1f} |
| Peak water cut % | {stats["peak_water_cut_pct"]:.1f} |
| Mean GOR scf/bbl | {stats["avg_gor_scf_bbl"]:.0f} |
| Mean BHP (psi, where reported) | {stats["avg_bhp_psi"]:.0f} |

## Notable extrema (same window)

""" + "\n".join(f"- {p}" for p in peaks) + f"""

## Recent monthly aggregates (subset)

{_monthly_table(g)}

## Usage note

Answers about **rates, choke, downtime, averages, peaks** grounded in these tables are supported.
This file does **not** replace geological reports, HSE procedures, or post-2017 operations.
"""

    return body


def _field_markdown(df: Any, stats: dict[str, dict[str, Any]]) -> str:
    tmin = df["timestamp"].min().date()
    tmax = df["timestamp"].max().date()
    wells = sorted(stats.keys())
    n_wells = int(df["well_id"].nunique())

    bullets = []
    for w in wells[:20]:
        s = stats[w]
        bullets.append(
            f"- **{w.strip()}**: {s['date_range']} — peak oil {s['peak_oil_bopd']:.0f} BOPD, "
            f"avg WCT {s['avg_water_cut_pct']:.1f} %"
        )
    if len(wells) > 20:
        bullets.append(f"- … +{len(wells) - 20} further wells")

    return f"""# Volve field — production overview (real daily data)

**Source:** Equinor Volve Open Dataset Excel (`Daily Production Data`).
**Coverage:** {tmin} → {tmax}.

**Producer wells enumerated in this workbook-derived extract:** **{n_wells}**.

## Wells represented (production subset)

""" + "\n".join(bullets) + f"""

## Field-level aggregates

Total producing day-rows ingested across file: **{len(df)}** over **{n_wells}** wells (**producer count in this workbook extract: {n_wells}**).

## Interpretation

Use this narrative for retrieval about **historic Volve throughput, relative well performance,
and water-cut escalation** bounded to the spreadsheet. It is **not** a substitute for WR-style
engineering narratives absent from this extract.
"""


def generate(xlsx_path: Path, out_dir: Path, producers_only: bool = True) -> list[Path]:
    sys.path.insert(0, str(ROOT))

    from src.data.volve_loader import get_per_well_stats, load_volve_daily

    df = load_volve_daily(xlsx_path, producers_only=producers_only, min_on_stream_hrs=1.0)
    stats = get_per_well_stats(df)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    field_md = _field_markdown(df, stats)
    p0 = out_dir / "VOLVE_Field_Overview_RealDailyProduction.md"
    p0.write_text(field_md, encoding="utf-8")
    paths.append(p0)

    # Per-well substantive docs for embedding precision
    for well_id, st in sorted(stats.items(), key=lambda x: x[0]):
        gw = df[(df["well_id"] == well_id) & (df["is_producing"])]
        if gw.empty:
            continue
        text = _well_markdown(str(well_id), gw, st)
        dest = out_dir / f"Volve_RealDaily_{_slug(str(well_id))}.md"
        dest.write_text(text, encoding="utf-8")
        paths.append(dest)

    logger.info("Wrote %d markdown files under %s", len(paths), out_dir)
    return paths


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Generate Volve Excel → Markdown corpus for RAG")
    p.add_argument("--xlsx", type=Path, default=DEFAULT_XLSX, help="Path to Volve production data.xlsx")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT, help="Output directory for .md files")
    args = p.parse_args()

    if not args.xlsx.exists():
        logger.error("Missing workbook: %s", args.xlsx)
        logger.error("Download Equinor Volve Open Dataset and place the daily production workbook there.")
        sys.exit(1)

    generate(args.xlsx, args.out_dir, producers_only=True)
    print(
        "\nProduction corpus: clear DB + ingest this folder only —\n"
        "  .venv/bin/python scripts/sync_real_volve_daily_rag.py\n\n"
        "Or merge into existing KB (not production-only):\n"
        "  .venv/bin/python -m src.rag.ingestion --docs-dir data/docs\n",
        flush=True,
    )


if __name__ == "__main__":
    main()

