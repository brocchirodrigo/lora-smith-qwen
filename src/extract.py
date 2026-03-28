"""
Entrypoint do pipeline de extração e formatação.

Uso:
    uv run python -m src.extract
    # ou
    python src/extract.py
"""

import asyncio
import json
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from src.config.settings import settings
from src.infrastructure.wp_client import WordPressClient
from src.services.formatter import ChatMLFormatter

console = Console()

OUTPUT_DIR  = Path("data/processed")
TRAIN_FILE  = OUTPUT_DIR / "train.jsonl"
VALID_FILE  = OUTPUT_DIR / "valid.jsonl"
VALID_SPLIT = 0.05


async def main() -> None:
    console.print("[bold cyan]═══ Pipeline de Extração — WordPress Help ═══[/bold cyan]\n")
    console.print(f"[dim]Fonte: {settings.wp_base_url}[/dim]")
    console.print(f"[dim]Destino: {TRAIN_FILE}[/dim]\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    client = WordPressClient(settings)
    formatter = ChatMLFormatter(settings)

    total_posts = 0
    skipped = 0
    entries: list[str] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Extraindo posts...", total=None)

        async for post in client.iter_posts():
            total_posts += 1
            progress.update(
                task,
                description=f"[cyan]Processando post #{post.id}[/cyan] — {total_posts} encontrados",
            )

            entry = formatter.format_post(post)
            if entry is None:
                skipped += 1
                continue

            entries.append(entry)

    n_valid = max(1, int(len(entries) * VALID_SPLIT))
    valid_entries = entries[:n_valid]
    train_entries = entries[n_valid:]

    def write_jsonl(path: Path, items: list[str]) -> None:
        with path.open("w", encoding="utf-8") as f:
            for text in items:
                f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

    write_jsonl(TRAIN_FILE, train_entries)
    write_jsonl(VALID_FILE, valid_entries)

    console.print("\n[bold green]✓ Extração concluída![/bold green]")
    console.print(f"  Posts encontrados : [white]{total_posts}[/white]")
    console.print(f"  Posts descartados : [yellow]{skipped}[/yellow] (conteúdo muito curto)")
    console.print(f"  Entradas treino   : [bold white]{len(train_entries)}[/bold white] → {TRAIN_FILE.resolve()}")
    console.print(f"  Entradas validação: [bold white]{len(valid_entries)}[/bold white] → {VALID_FILE.resolve()}\n")


if __name__ == "__main__":
    asyncio.run(main())
