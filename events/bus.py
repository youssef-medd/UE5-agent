from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Callable, Coroutine

from events.event_types import Event, EventKind

logger = logging.getLogger(__name__)

_Handler = Callable[[Event], Coroutine]


class EventBus:
    def __init__(self) -> None:
        self._subscribers: dict[EventKind, list[_Handler]] = defaultdict(list)
        self._queue: asyncio.Queue[Event] = asyncio.Queue()
        self._running = False

    def subscribe(self, kind: EventKind, handler: _Handler) -> None:
        self._subscribers[kind].append(handler)

    async def publish(self, event: Event) -> None:
        await self._queue.put(event)

    async def run(self) -> None:
        self._running = True
        while self._running:
            try:
                event = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            handlers = self._subscribers.get(event.kind, [])
            for handler in handlers:
                try:
                    await handler(event)
                except Exception as exc:
                    logger.error("Event handler error for %s: %s", event.kind, exc)

    def stop(self) -> None:
        self._running = False


_bus: EventBus | None = None


def get_bus() -> EventBus:
    global _bus
    if _bus is None:
        _bus = EventBus()
    return _bus
