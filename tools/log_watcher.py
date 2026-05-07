from __future__ import annotations

import asyncio
import logging
import os
import re
from pathlib import Path
from typing import AsyncIterator

logger = logging.getLogger(__name__)

_DEFAULT_LOG_GLOB = "Saved/Logs/UnrealEditor.log"

_ERROR_PATTERNS = [
    re.compile(r"\bError\b", re.I),
    re.compile(r"\bFatal\b", re.I),
    re.compile(r"\bCrash\b", re.I),
    re.compile(r"\bException\b", re.I),
]

_WARNING_PATTERN = re.compile(r"\bWarning\b", re.I)


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

    async def tail_filtered(
        self,
        pattern: str,
        poll_interval: float = 0.5,
    ) -> AsyncIterator[str]:
        """Yield only lines matching the given regex pattern."""
        compiled = re.compile(pattern, re.I)
        async for line in self.tail(poll_interval):
            if compiled.search(line):
                yield line

    async def tail_for(self, seconds: float, poll_interval: float = 0.5) -> list[str]:
        lines: list[str] = []
        try:
            async with asyncio.timeout(seconds):
                async for line in self.tail(poll_interval):
                    lines.append(line)
        except asyncio.TimeoutError:
            pass
        return lines

    async def scan_for_errors(self, seconds: float = 5.0) -> list[str]:
        """Collect all error/fatal/crash lines emitted within a time window."""
        errors: list[str] = []
        for line in await self.tail_for(seconds):
            if any(p.search(line) for p in _ERROR_PATTERNS):
                errors.append(line)
        return errors

    async def wait_for_pattern(
        self,
        pattern: str,
        timeout: float = 30.0,
        poll_interval: float = 0.5,
    ) -> str | None:
        """Block until a log line matches pattern, then return that line."""
        compiled = re.compile(pattern, re.I)
        try:
            async with asyncio.timeout(timeout):
                async for line in self.tail(poll_interval):
                    if compiled.search(line):
                        logger.debug("wait_for_pattern matched: %s", line)
                        return line
        except asyncio.TimeoutError:
            logger.warning("wait_for_pattern timed out after %.1fs for %r", timeout, pattern)
        return None

    async def scan_for_warnings(self, seconds: float = 5.0) -> list[str]:
        warnings: list[str] = []
        for line in await self.tail_for(seconds):
            if _WARNING_PATTERN.search(line):
                warnings.append(line)
        return warnings
