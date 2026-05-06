from __future__ import annotations

import abc
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from llm.base_llm import GenerationConfig, LLMResponse
from llm.model_router import get_router
from llm.prompt_templates import get_system_prompt
from events.bus import get_bus
from events.event_types import Event, EventKind

logger = logging.getLogger(__name__)


@dataclass
class AgentResult:
    role: str
    output: Any
    raw_response: Optional[LLMResponse] = None
    success: bool = True
    error: str = ""
    latency_ms: float = 0.0
    metadata: dict = field(default_factory=dict)


class BaseAgent(abc.ABC):
    role: str = ""

    def __init__(self, task_id: str = "") -> None:
        self.task_id = task_id
        self._router = get_router()
        self._bus = get_bus()

    async def run(self, prompt: str, context: dict | None = None) -> AgentResult:
        start = time.monotonic()
        await self._bus.publish(Event(
            kind=EventKind.AGENT_THINKING,
            task_id=self.task_id,
            agent=self.role,
            payload={"prompt_preview": prompt[:120]},
        ))
        try:
            result = await self._execute(prompt, context or {})
            result.latency_ms = (time.monotonic() - start) * 1000
            await self._bus.publish(Event(
                kind=EventKind.AGENT_DONE,
                task_id=self.task_id,
                agent=self.role,
                payload={"latency_ms": result.latency_ms},
            ))
            return result
        except Exception as exc:
            latency = (time.monotonic() - start) * 1000
            logger.error("[%s] agent failed: %s", self.role, exc)
            await self._bus.publish(Event(
                kind=EventKind.AGENT_ERROR,
                task_id=self.task_id,
                agent=self.role,
                payload={"error": str(exc)},
            ))
            return AgentResult(
                role=self.role,
                output=None,
                success=False,
                error=str(exc),
                latency_ms=latency,
            )

    @abc.abstractmethod
    async def _execute(self, prompt: str, context: dict) -> AgentResult:
        ...

    async def _call_llm(
        self,
        user_prompt: str,
        system_prompt: str = "",
        config: GenerationConfig | None = None,
    ) -> LLMResponse:
        backend = await self._router.get_backend(self.role)
        cfg = config or GenerationConfig.for_role(self.role)
        if not system_prompt:
            system_prompt = get_system_prompt(self.role)
        messages = backend._build_messages(user_prompt, system_prompt)
        return await backend.generate(messages, cfg)
