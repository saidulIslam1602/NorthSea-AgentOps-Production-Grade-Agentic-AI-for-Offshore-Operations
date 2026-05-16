"""
Command-line interface for NorthSea AgentOps.

Entry points:
  agentops serve          — Start the FastAPI server
  agentops ingest         — Ingest documents into the RAG vector store
  agentops detector-eval — Run anomaly detector train/eval (Volve, scripts/train_detector_v2.py)
  agentops evaluate       — Run RAGAS evaluation suite
  agentops migrate        — Apply database migrations (Alembic)

Synthetic telemetry/doc generation was removed — use real data under data/Volve_Data/
and corpus files under data/docs/. See README.md.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import typer
from rich.console import Console

app = typer.Typer(
    name="agentops",
    help="NorthSea AgentOps — Production-Grade Agentic AI for Offshore Operations",
    add_completion=False,
)
console = Console()


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Bind host"),
    port: int = typer.Option(8000, help="Bind port"),
    reload: bool = typer.Option(False, help="Enable auto-reload (development only)"),
    workers: int = typer.Option(1, help="Number of Uvicorn workers"),
) -> None:
    """Start the FastAPI investigation API server."""
    import uvicorn

    console.print(f"[bold]Starting NorthSea AgentOps API[/bold] on {host}:{port}")
    uvicorn.run(
        "src.api.main:app",
        host=host,
        port=port,
        reload=reload,
        workers=workers if not reload else 1,
    )


@app.command()
def ingest(
    docs_dir: Path = typer.Option(
        Path("data/docs"), "--docs-dir", "-d", help="Directory containing documents to ingest"
    ),
    clear: bool = typer.Option(False, "--clear", help="Clear existing vectors before ingesting"),
) -> None:
    """Ingest documents into the RAG vector store."""
    import asyncio

    from src.rag.ingestion import main as ingest_main

    if clear:
        import psycopg

        from src.config import get_settings
        from src.rag.vectorstore import clear_all

        settings = get_settings()
        db_url = settings.database_url.replace("+psycopg", "")

        async def _clear() -> tuple[int, int]:
            async with await psycopg.AsyncConnection.connect(db_url) as conn:
                return await clear_all(conn)

        docs_deleted, chunks_deleted = asyncio.run(_clear())
        console.print(
            f"[yellow]Cleared vector store:[/yellow] "
            f"{docs_deleted} document(s), {chunks_deleted} chunk(s) removed."
        )

    console.print(f"[bold]Ingesting documents[/bold] from {docs_dir}")
    ingest_main()


@app.command("detector-eval")
def detector_eval() -> None:
    """Train and evaluate anomaly detectors v1 vs v2 on real Volve data (writes eval/detector_performance_v2.json)."""
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "train_detector_v2.py"
    if not script.is_file():
        console.print("[red]scripts/train_detector_v2.py not found — run from repo root[/red]")
        raise typer.Exit(code=2)
    console.print(f"[bold]Running detector evaluation[/bold]: {script}")
    result = subprocess.run([sys.executable, str(script)], cwd=str(repo_root))
    raise typer.Exit(code=result.returncode)


@app.command()
def evaluate(
    golden_set: Path = typer.Option(
        Path("eval/golden_testset.json"), "--golden-set", help="Path to golden test set JSON"
    ),
    experiment: str = typer.Option("northsea-rag-eval", "--experiment", help="MLflow experiment name"),
) -> None:
    """Run RAGAS evaluation against the golden test set and log to MLflow."""
    from eval.ragas_eval import main as eval_main

    _ = golden_set
    console.print(f"[bold]Running RAGAS evaluation[/bold] — experiment: {experiment}")
    eval_main()


@app.command()
def migrate(
    revision: str = typer.Argument("head", help="Alembic revision to migrate to"),
) -> None:
    """Apply Alembic database migrations."""
    console.print(f"[bold]Running database migration[/bold] to revision: {revision}")
    result = subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", revision],
        check=False,
    )
    if result.returncode != 0:
        console.print("[red]Migration failed — check Alembic logs above.[/red]")
        raise typer.Exit(code=result.returncode)
    console.print("[green]Migration complete.[/green]")


if __name__ == "__main__":
    app()
