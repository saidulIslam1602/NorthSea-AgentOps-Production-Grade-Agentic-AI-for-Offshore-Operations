"""
Command-line interface for NorthSea AgentOps.

Entry points:
  agentops serve      — Start the FastAPI server
  agentops ingest     — Ingest documents into the RAG vector store
  agentops generate   — Generate synthetic telemetry / documents
  agentops eval       — Run RAGAS evaluation suite
  agentops migrate    — Apply database migrations (Alembic)
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
    from src.rag.ingestion import main as ingest_main

    console.print(f"[bold]Ingesting documents[/bold] from {docs_dir}")
    ingest_main()


@app.command()
def generate(
    output_dir: Path = typer.Option(
        Path("data"), "--output-dir", "-o", help="Output directory for generated data"
    ),
    num_wells: int = typer.Option(5, "--wells", help="Number of wells to simulate"),
    days: int = typer.Option(30, "--days", help="Number of days of telemetry to generate"),
    docs: bool = typer.Option(True, "--docs/--no-docs", help="Generate synthetic documents"),
) -> None:
    """Generate synthetic telemetry data and well documents."""
    from src.data.synthetic_generator import main as gen_main

    console.print(f"[bold]Generating synthetic data[/bold] — {num_wells} wells × {days} days")
    gen_main()

    if docs:
        from src.data.doc_generator import generate_all_documents

        console.print("[bold]Generating synthetic operational documents[/bold]")
        generate_all_documents()


@app.command()
def evaluate(
    golden_set: Path = typer.Option(
        Path("eval/golden_testset.json"), "--golden-set", help="Path to golden test set JSON"
    ),
    experiment: str = typer.Option("northsea-rag-eval", "--experiment", help="MLflow experiment name"),
) -> None:
    """Run RAGAS evaluation against the golden test set and log to MLflow."""
    from eval.ragas_eval import main as eval_main

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
