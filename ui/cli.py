from __future__ import annotations

import asyncio
import logging

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

from orchestrator.orchestrator import Orchestrator
from config.settings import get_settings

console = Console()
logger = logging.getLogger(__name__)


def _print_result(result: dict) -> None:
    success = result.get("success", False)
    code = result.get("code", "")
    review = result.get("review", {})
    task_id = result.get("task_id", "")
    dry_run = result.get("dry_run", False)

    status = "[green]SUCCESS[/green]" if success else "[red]FAILED[/red]"
    console.print(Panel(f"{status}  |  task={task_id}  |  dry_run={dry_run}", title="Result"))

    if review:
        risk = review.get("risk_level", "?")
        approved = review.get("approved", False)
        summary = review.get("summary", "")
        color = {"low": "green", "medium": "yellow", "high": "red"}.get(risk, "white")
        console.print(
            f"Review: [{color}]{risk.upper()}[/{color}] | approved={approved} | {summary}"
        )

    if code:
        console.print(Syntax(code, "python", theme="monokai", line_numbers=True))


async def _run_cli(dry_run: bool) -> None:
    settings = get_settings()
    orchestrator = Orchestrator()

    console.print(Panel(
        Text("UE5 Agent — CLI", style="bold cyan"),
        subtitle=f"dry_run={dry_run} | ollama={settings.ollama_base_url}",
    ))
    console.print("Type your prompt and press Enter. Type [bold]exit[/bold] to quit.\n")

    while True:
        try:
            prompt = console.input("[bold green]>>> [/bold green]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye[/dim]")
            break

        if not prompt:
            continue
        if prompt.lower() in ("exit", "quit", "q"):
            break

        with console.status("[bold yellow]Thinking...[/bold yellow]"):
            result = await orchestrator.run(prompt, dry_run=dry_run)

        _print_result(result)
        console.print()


def run(dry_run: bool = False) -> None:
    asyncio.run(_run_cli(dry_run))
