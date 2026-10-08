"""
CLI entrypoint.

    python app.py ingest --source data/sample_docs
    python app.py ask "your question"
    python app.py ask "your question" --trace
    python app.py chat
"""
import typer
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from src.config import settings
from src.ingestion import load_and_chunk
from src.vectorstore import build_vectorstore
from src.graph import run

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def ingest(source: str = typer.Option("data/sample_docs", help="Folder of .txt/.md/.pdf files")):
    """Load, chunk, embed, and persist documents into the vector store."""
    settings.validate()
    chunks = load_and_chunk(source)
    build_vectorstore(chunks)
    console.print(f"[green]Ingested and persisted {len(chunks)} chunks from {source}[/green]")


@app.command()
def ask(
    question: str,
    trace: bool = typer.Option(False, "--trace", help="Print the full node-by-node execution trace"),
):
    """Ask a single question against the ingested corpus."""
    settings.validate()
    result = run(question)
    console.print(Panel(Markdown(result["generation"]), title="Answer", border_style="cyan"))

    if trace:
        console.print("\n[bold]Execution trace:[/bold]")
        for i, line in enumerate(result.get("trace", []), start=1):
            console.print(f"  {i}. {line}")
        console.print(
            f"\n[dim]retries={result.get('retries', 0)}  "
            f"hallucination_ok={result.get('hallucination_ok')}  "
            f"answers_question={result.get('answers_question')}  "
            f"used_web_search={result.get('used_web_search')}[/dim]"
        )


@app.command()
def chat():
    """Interactive loop -- ask multiple questions in one session."""
    settings.validate()
    console.print("[bold cyan]Self-Corrective RAG[/bold cyan] -- type 'exit' to quit.\n")
    while True:
        question = console.input("[bold]> [/bold]")
        if question.strip().lower() in ("exit", "quit"):
            break
        result = run(question)
        console.print(Panel(Markdown(result["generation"]), title="Answer", border_style="cyan"))
        console.print(
            f"[dim]retries={result.get('retries', 0)}  "
            f"hallucination_ok={result.get('hallucination_ok')}  "
            f"answers_question={result.get('answers_question')}[/dim]\n"
        )


if __name__ == "__main__":
    app()
