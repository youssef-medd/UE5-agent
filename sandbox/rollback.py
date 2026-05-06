from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

_ROLLBACK_FILE = Path("logs/rollback_stack.json")


@dataclass
class RollbackEntry:
    task_id: str
    timestamp: float
    code: str
    undo_code: str
    description: str = ""


class RollbackManager:
    """Tracks executed UE5 operations and can undo the last N changes."""

    def __init__(self) -> None:
        self._stack: list[RollbackEntry] = []
        self._load()

    def push(self, entry: RollbackEntry) -> None:
        self._stack.append(entry)
        self._save()
        logger.debug("Rollback push: %r", entry.description)

    def pop(self) -> RollbackEntry | None:
        if not self._stack:
            return None
        entry = self._stack.pop()
        self._save()
        return entry

    def peek(self, n: int = 1) -> list[RollbackEntry]:
        return self._stack[-n:]

    def clear(self) -> None:
        self._stack.clear()
        self._save()

    def _save(self) -> None:
        _ROLLBACK_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(_ROLLBACK_FILE, "w") as f:
            json.dump([asdict(e) for e in self._stack], f, indent=2)

    def _load(self) -> None:
        if not _ROLLBACK_FILE.exists():
            return
        try:
            with open(_ROLLBACK_FILE) as f:
                data = json.load(f)
            self._stack = [RollbackEntry(**d) for d in data]
        except Exception as exc:
            logger.warning("Could not load rollback stack: %s", exc)
