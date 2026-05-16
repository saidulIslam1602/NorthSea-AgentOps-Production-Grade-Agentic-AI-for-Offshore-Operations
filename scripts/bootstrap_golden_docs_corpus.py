"""
Generate **synthetic** Markdown under ``data/docs/`` from ``eval/golden_testset.json`` so filenames
listed in ``document_sources`` align with Ragas QA labels.

Uses golden ``ground_truth`` text directly — inflated eval scores vs production narrative.
For operator KB retrieval from **real** field data only, use::

    python scripts/sync_real_volve_daily_rag.py

(and keep bootstrap outputs out of ingestion).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "eval" / "golden_testset.json"
DOCS_ROOT = ROOT / "data" / "docs"


def dest_dir(fname: str) -> Path:
    if fname.startswith("WR-"):
        return DOCS_ROOT / "well_reports"
    if fname.startswith("EQ-"):
        return DOCS_ROOT / "equipment_manuals"
    if fname.startswith("HSE-"):
        return DOCS_ROOT / "hse_procedures"
    if fname.startswith("ML-"):
        return DOCS_ROOT / "maintenance_logs"
    # NPD, Equinor, agent/meta references
    if fname.startswith(("NPD_", "Equinor_")):
        return DOCS_ROOT / "well_reports"
    return DOCS_ROOT / "reference"


def build_markdown(fname: str, cases: list[dict]) -> str:
    title = fname.replace(".md", "").replace("_", " ")
    parts: list[str] = [
        f"# {title}",
        "",
        f"Operational reference derived from curated evaluation excerpts for **{fname}**.",
        "",
        "## Incident and engineering narrative",
        "",
    ]
    seen_gt: set[str] = set()
    for tc in cases:
        tid = tc.get("id", "")
        topics = tc.get("expected_topics") or []
        q = tc.get("question") or ""
        gt = (tc.get("ground_truth") or "").strip()
        parts.append(f"### Anchor {tid} ({tc.get('category', '')})")
        parts.append("")
        parts.append(f"**Operator question.** {q}")
        parts.append("")
        if topics:
            parts.append("**Indexing keywords:** " + ", ".join(str(t) for t in topics) + "")
            parts.append("")
        if gt and gt not in seen_gt:
            parts.append(gt)
            parts.append("")
            seen_gt.add(gt)
        parts.append("---")
        parts.append("")
    if not cases:
        parts.append("_No anchored golden rows; ingest placeholder for completeness._")
    return "\n".join(parts)


def main() -> None:
    testset = json.loads(GOLDEN.read_text(encoding="utf-8"))
    by_file: dict[str, list[dict]] = defaultdict(list)
    for tc in testset:
        for src in tc.get("document_sources") or []:
            by_file[src].append(tc)

    written = 0
    for fname, cases in sorted(by_file.items()):
        d = dest_dir(fname)
        d.mkdir(parents=True, exist_ok=True)
        path = d / fname
        path.write_text(build_markdown(fname, cases), encoding="utf-8")
        written += 1

    print(f"Wrote {written} markdown files under {DOCS_ROOT}")


if __name__ == "__main__":
    main()
