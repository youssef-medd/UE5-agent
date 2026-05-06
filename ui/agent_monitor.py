from __future__ import annotations

import asyncio
import logging
from collections import defaultdict

from rich.console import Console
from rich.live import Live
from rich.table import Table

from events.bus import get_bus
from events.event_types import Event, EventKind

logger = logging.getLogger(__name__)
console = Console()


class AgentMonitor:
    """Live terminal dashboard showing all agent activity via the event bus."""

    def __init__(self) -> None:
        self._bus = get_bus()
        self._events: list[Event] = []
        self._agent_states: dict[str, str] = defaultdict(lambda: "idle")

    def _register_handlers(self) -> None:
        for kind in EventKind:
            self._bus.subscribe(kind, self._handle_event)

    async def _handle_event(self, event: Event) -> None:
        self._events.append(event)
        if event.agent:
            self._agent_states[event.agent] = event.kind.value

    def _build_table(self) -> Table:
        table = Table(title="Agent Monitor", expand=True)
        table.add_column("Agent", style="cyan")
        table.add_column("Status", style="yellow")
        table.add_column("Last Event")

        for agent, status in sorted(self._agent_states.items()):
            table.add_row(agent, status, "")

        return table

    async def run(self) -> None:
        self._register_handlers()
        bus_task = asyncio.create_task(self._bus.run())

        with Live(self._build_table(), refresh_per_second=4, console=console) as live:
            try:
                while True:
                    await asyncio.sleep(0.25)
                    live.update(self._build_table())
            except asyncio.CancelledError:
                pass

        bus_task.cancel()
