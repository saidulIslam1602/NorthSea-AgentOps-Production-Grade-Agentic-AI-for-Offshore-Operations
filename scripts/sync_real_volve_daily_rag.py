"""
Replace the pgvector knowledge base with **only** artefacts from real Volve daily production.

Steps:
  1) DELETE FROM ``documents`` (cascades ``document_chunks``)
  2) Regenerate Markdown via ``scripts/generate_volve_rag_corpus.py``
  3) Ingest ``data/docs/volve_real`` only

Run from repo root. Requires workbook at ``data/Volve_Data/Volve production data.xlsx``.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


async def wipe_document_store() -> None:
    import psycopg

    sys.path.insert(0, str(ROOT))
    from src.config import get_settings

    url = get_settings().database_url.replace("+psycopg", "")
    async with await psycopg.AsyncConnection.connect(url) as conn:
        async with conn.cursor() as cur:
            await cur.execute("DELETE FROM documents")
            await conn.commit()
    logging.info("Cleared documents + chunks (FK cascade).")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    asyncio.run(wipe_document_store())

    subprocess.check_call(
        [sys.executable, str(ROOT / "scripts" / "generate_volve_rag_corpus.py")],
        cwd=ROOT,
    )
    subprocess.check_call(
        [sys.executable, str(ROOT / "scripts" / "generate_volve_golden_testset.py")],
        cwd=ROOT,
    )
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "src.rag.ingestion",
            "--docs-dir",
            str(ROOT / "data" / "docs" / "volve_real"),
        ],
        cwd=ROOT,
    )
    logging.info(
        "RAG corpus + ``eval/golden_testset.json`` synced from VOLVE workbook only (%s)",
        ROOT / "data/docs/volve_real",
    )


if __name__ == "__main__":
    main()
