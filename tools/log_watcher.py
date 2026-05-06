from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import AsyncIterator

logger = logging.getLogger(__name__)

_DEFAULT_LOG_GLOB = "Saved/Logs/UnrealEditor.log"


class LogWatcher:
    """Tails the UE5 output log file and yields new lines as they appear."""

    def __init__(self, log_path: str | None = None, ue5_project_dir: str = ".") -> None:
        if log_path:
            self._log_path = Path(log_path)
        else:
            self._log_path = Path(ue5_project_dir) / _DEFAULT_LOG_GLOB

    async def tail(self, poll_interval: float = 0.5) -> AsyncIterator[str]:
        if not self._log_path.exists():
            logger.warning("UE5 log not found at %s", self._log_path)
            return

        with open(self._log_path, "r", encoding="utf-8", errors="replace") as fh:
            fh.seek(0, os.SEEK_END)
            while True:
                line = fh.readline()
                if line:
                    yield line.rstrip()
                else:
                    await asyncio.sleep(poll_interval)

    async def tail_for(self, seconds: float, poll_interval: float = 0.5) -> list[str]:
        lines: list[str] = []
        try:
            async with asyncio.timeout(seconds):
                async for line in self.tail(poll_interval):
                    lines.append(line)
        except asyncio.TimeoutError:
            pass
        return lines
