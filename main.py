#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import logging
import sys
from typing import Optional

import typer

from config.settings import get_settings

app = typer.Typer(help="UE5 Agent — turn plain English into Unreal Engine 5 code.")


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


@app.command()
def main(
    prompt: Optional[str] = typer.Option(None, "--prompt", "-p", help="Single prompt (headless mode)"),
    ui: str = typer.Option("cli", "--ui", help="Interface: cli | web"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Simulate without writing to UE5"),
    share: bool = typer.Option(False, "--share", help="Share web UI publicly via Gradio tunnel"),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    settings = get_settings()
    _setup_logging(log_level or settings.log_level)

    effective_dry_run = dry_run or settings.dry_run

    if prompt:
        _run_headless(prompt, effective_dry_run)
        return

    if ui == "web":
        from ui.web_ui import launch
        launch(share=share)
    else:
        from ui.cli import run
        run(dry_run=effective_dry_run)


def _run_headless(prompt: str, dry_run: bool) -> None:
    from orchestrator.orchestrator import Orchestrator
    from rich.console import Console
    from rich.syntax import Syntax

    console = Console()
    orchestrator = Orchestrator()

    with console.status("[bold yellow]Running agent pipeline...[/bold yellow]"):
        result = asyncio.run(orchestrator.run(prompt, dry_run=dry_run))

    if result.get("success"):
        console.print("[green]Success[/green]")
        code = result.get("code", "")
        if code:
            console.print(Syntax(code, "python", theme="monokai", line_numbers=True))
    else:
        console.print(f"[red]Failed:[/red] {result.get('error', 'unknown error')}")
        sys.exit(1)


if __name__ == "__main__":
    app()
