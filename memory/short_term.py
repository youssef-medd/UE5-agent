from __future__ import annotations

import logging
from collections import deque
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_WINDOW = 20


class ShortTermMemory:
    """Per-session sliding window of conversation turns and agent results."""

    def __init__(self, max_turns: int = _DEFAULT_WINDOW) -> None:
        self._turns: deque[dict[str, Any]] = deque(maxlen=max_turns)

    def add_turn(self, role: str, content: str, metadata: dict | None = None) -> None:
        self._turns.append({
            "role": role,
            "content": content,
            "metadata": metadata or {},
        })

    def add_agent_result(self, agent_role: str, output: Any, success: bool) -> None:
        self._turns.append({
            "role": f"agent:{agent_role}",
            "content": str(output)[:500],
            "metadata": {"success": success},
        })

    def get_messages(self) -> list[dict[str, str]]:
        return [
            {"role": t["role"], "content": t["content"]}
            for t in self._turns
            if t["role"] in ("user", "assistant")
        ]

    def get_context_string(self) -> str:
        lines = []
        for t in self._turns:
            lines.append(f"[{t['role']}]: {t['content'][:200]}")
        return "\n".join(lines)

    def clear(self) -> None:
        self._turns.clear()

    def __len__(self) -> int:
        return len(self._turns)
