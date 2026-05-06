from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from enum import IntEnum

logger = logging.getLogger(__name__)


class Priority(IntEnum):
    LOW = 3
    NORMAL = 2
    HIGH = 1


@dataclass(order=True)
class QueuedTask:
    priority: Priority
    task_id: str = field(compare=False)
    prompt: str = field(compare=False)
    context: dict = field(compare=False, default_factory=dict)


class TaskQueue:
    def __init__(self) -> None:
        self._queue: asyncio.PriorityQueue[QueuedTask] = asyncio.PriorityQueue()

    async def put(
        self,
        task_id: str,
        prompt: str,
        context: dict | None = None,
        priority: Priority = Priority.NORMAL,
    ) -> None:
        item = QueuedTask(priority=priority, task_id=task_id, prompt=prompt, context=context or {})
        await self._queue.put(item)
        logger.debug("Queued task %s (priority=%s)", task_id, priority.name)

    async def get(self) -> QueuedTask:
        return await self._queue.get()

    def task_done(self) -> None:
        self._queue.task_done()

    def empty(self) -> bool:
        return self._queue.empty()

    def size(self) -> int:
        return self._queue.qsize()
