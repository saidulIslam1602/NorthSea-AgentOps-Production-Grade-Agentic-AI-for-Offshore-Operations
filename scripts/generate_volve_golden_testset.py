"""
Build ``eval/golden_testset.json`` from **real** Volve daily production (same Excel as RAG ingestion).

Questions, ``ground_truth``, and ``document_sources`` align with ``data/docs/volve_real/*.md``.
Invoked automatically from ``scripts/sync_real_volve_daily_rag.py``.

Requires ``data/Volve_Data/Volve production data.xlsx``.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_XLSX = ROOT / "data" / "Volve_Data" / "Volve production data.xlsx"
GOLDEN_OUT = ROOT / "eval" / "golden_testset.json"

logger = logging.getLogger(__name__)
FIELD_DOC = "VOLVE_Field_Overview_RealDailyProduction.md"


def _slug(well_id: str) -> str:
    return well_id.strip().replace("/", "-").replace(" ", "_")


def _well_doc(well_id: str) -> str:
    return f"Volve_RealDaily_{_slug(well_id)}.md"


def _nid(counter: itertools.count) -> str:
    return f"VGT-{next(counter):03d}"


def _field_cases(df: Any, stats: dict[str, dict[str, Any]], c: itertools.count) -> list[dict[str, Any]]:
    tmin = df["timestamp"].min().date()
    tmax = df["timestamp"].max().date()
    n_well = int(df["well_id"].nunique())
    n_rows = len(df)
    bw_key, bs = max(stats.items(), key=lambda kv: float(kv[1]["peak_oil_bopd"]))

    return [
        {
            "id": _nid(c),
            "category": "field_overview",
            "question": "Between which calendar dates does the VOLVE field overview summarise daily production excerpts?",
            "expected_topics": [str(tmin), str(tmax)],
            "ground_truth": f"VOLVE overview excerpts summarise data from **{tmin}** through **{tmax}**.",
            "document_sources": [FIELD_DOC],
        },
        {
            "id": _nid(c),
            "category": "field_overview",
            "question": "How many VOLVE wells are enumerated in workbook-derived production summaries?",
            "expected_topics": ["well"],
            "ground_truth": f"The VOLVE ingestion enumerates **{n_well}** producer wells.",
            "document_sources": [FIELD_DOC],
        },
        {
            "id": _nid(c),
            "category": "field_overview",
            "question": "Which VOLVE well shows the highest peak daily oil in the excerpts, and roughly what magnitude?",
            "expected_topics": ["peak", "BOPD"],
            "ground_truth": (
                f"Highest workbook peak belongs to **{str(bw_key).strip()}** at about "
                f"**{float(bs['peak_oil_bopd']):.1f} BOPD**."
            ),
            "document_sources": [FIELD_DOC, _well_doc(str(bw_key))],
        },
        {
            "id": _nid(c),
            "category": "field_overview",
            "question": "How many producing-day observations were pooled across all wells before chunking?",
            "expected_topics": ["rows"],
            "ground_truth": f"The workbook extract pools **{n_rows}** producing-day observations across wells.",
            "document_sources": [FIELD_DOC],
        },
    ]


def _well_cases(well_key: str, g: Any, st: dict[str, Any], c: itertools.count) -> list[dict[str, Any]]:
    well = str(well_key).strip()
    wdoc = _well_doc(str(well_key))
    peak_idx = int(g["oil_rate_bopd"].idxmax())
    rp = g.loc[peak_idx]
    wc_peak_idx = int(g["water_cut_pct"].idxmax())
    rw = g.loc[wc_peak_idx]
    tmin = g["timestamp"].min().date()
    tmax = g["timestamp"].max().date()
    gor_m = float(st["avg_gor_scf_bbl"])
    bh = float(st["avg_bhp_psi"]) if float(st["avg_bhp_psi"]) > 0 else 0.0

    rows = [
        {
            "id": _nid(c),
            "category": "production_metrics",
            "question": f"What peak daily oil rate (BOPD) appears for VOLVE producer {well} in the workbook-backed summary?",
            "expected_topics": ["BOPD", "peak", "oil"],
            "ground_truth": (
                f"Peak oil for {well} is {float(rp['oil_rate_bopd']):.1f} BOPD on {rp['timestamp'].date()}, "
                f"water cut roughly {float(rp['water_cut_pct']):.1f} %."
            ),
            "document_sources": [wdoc, FIELD_DOC],
        },
        {
            "id": _nid(c),
            "category": "production_metrics",
            "question": f"What producing-day date range covers well {well} in ingested VOLVE excerpts?",
            "expected_topics": [str(tmin), str(tmax)],
            "ground_truth": f"{well} has producing-day excerpts from {tmin} through {tmax}.",
            "document_sources": [wdoc, FIELD_DOC],
        },
        {
            "id": _nid(c),
            "category": "production_metrics",
            "question": f"What average water-cut percentage appears across producing rows summarised for {well}?",
            "expected_topics": ["water cut"],
            "ground_truth": (
                f"Average producing-day water cut for {well} is about "
                f"{float(st['avg_water_cut_pct']):.1f}% over {int(st['producing_days'])} summarised producing rows."
            ),
            "document_sources": [wdoc],
        },
        {
            "id": _nid(c),
            "category": "production_metrics",
            "question": (
                f"What maximum water-cut percentage is summarised for producer {well} and on what date did it occur?"
            ),
            "expected_topics": ["water cut"],
            "ground_truth": (
                f"Highest summarised water cut for {well} is roughly {float(rw['water_cut_pct']):.1f}% on "
                f"{rw['timestamp'].date()}, with oil rate about {float(rw['oil_rate_bopd']):.1f} BOPD."
            ),
            "document_sources": [wdoc],
        },
        {
            "id": _nid(c),
            "category": "production_metrics",
            "question": (
                f"What mean gas-oil ratio (scf/bbl) summarises producer {well} in the VOLVE workbook-derived text?"
            ),
            "expected_topics": ["GOR", "scf/bbl"],
            "ground_truth": f"Mean GOR across producing rows for {well} is roughly {gor_m:.0f} scf/bbl.",
            "document_sources": [wdoc],
        },
    ]
    if bh > 0:
        rows.append(
            {
                "id": _nid(c),
                "category": "production_metrics",
                "question": f"What approximate mean bottom-hole pressure (psi) summary exists for producer {well}?",
                "expected_topics": ["BHP"],
                "ground_truth": f"Mean reporting BHP among producing rows summarised for {well} is about {bh:.0f} psi.",
                "document_sources": [wdoc],
            }
        )
    return rows


def build_golden_records(xlsx_path: Path) -> list[dict[str, Any]]:
    sys.path.insert(0, str(ROOT))
    from src.data.volve_loader import get_per_well_stats, load_volve_daily

    df = load_volve_daily(xlsx_path, producers_only=True, min_on_stream_hrs=1.0)
    stats_map = get_per_well_stats(df)
    c = itertools.count(1)

    rows: list[dict[str, Any]] = _field_cases(df, stats_map, c)

    for well_key in sorted(stats_map.keys(), key=str):
        st = stats_map[well_key]
        g = df[(df["well_id"] == well_key) & df["is_producing"]].copy()
        if g.empty:
            continue
        rows.extend(_well_cases(well_key, g, st, c))

    return sorted(rows, key=lambda r: r["id"])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", type=Path, default=DEFAULT_XLSX)
    ap.add_argument("--out", type=Path, default=GOLDEN_OUT)
    args = ap.parse_args()

    if not args.xlsx.exists():
        logger.error("Workbook missing: %s", args.xlsx)
        sys.exit(1)

    legacy = args.out.parent / "golden_testset_legacy_synthetic.json"
    if args.out.exists() and not legacy.exists():
        sniff = args.out.read_text(encoding="utf-8", errors="replace")
        if "WR-" in sniff or '"WR-' in sniff:
            shutil.copy2(args.out, legacy)
            logger.info("Archived previous synthetic golden set → %s", legacy.resolve())

    golden = build_golden_records(args.xlsx)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(golden, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    logger.info("Wrote %d cases → %s", len(golden), args.out.resolve())


if __name__ == "__main__":
    main()
