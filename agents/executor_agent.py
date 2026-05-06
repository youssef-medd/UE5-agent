from __future__ import annotations

import logging

from agents.base_agent import AgentResult, BaseAgent
from config.settings import get_settings
from events.bus import get_bus
from events.event_types import Event, EventKind

logger = logging.getLogger(__name__)


class ExecutorAgent(BaseAgent):
    role = "executor"

    def __init__(self, task_id: str = "") -> None:
        super().__init__(task_id)
        self._settings = get_settings()

    async def _execute(self, prompt: str, context: dict) -> AgentResult:
        code = context.get("code", "")
        review = context.get("review", {})
        dry_run = context.get("dry_run", self._settings.dry_run)

        if not code:
            return AgentResult(
                role=self.role,
                output=None,
                success=False,
                error="No code provided to executor",
            )

        if not review.get("approved", False):
            reason = review.get("summary", "Reviewer did not approve")
            logger.warning("Executor: skipping execution — %s", reason)
            return AgentResult(
                role=self.role,
                output={"action": "skip", "reason": reason},
                success=False,
                error=reason,
            )

        if dry_run:
            logger.info("Executor: DRY RUN — would execute %d lines", len(code.splitlines()))
            return AgentResult(
                role=self.role,
                output={"action": "dry_run", "code": code},
                success=True,
                metadata={"dry_run": True},
            )

        result = await self._send_to_ue5(code)
        return result

    async def _send_to_ue5(self, code: str) -> AgentResult:
        from tools.remote_control import RemoteControlClient

        bus = get_bus()
        await bus.publish(Event(
            kind=EventKind.UE5_REQUEST,
            task_id=self.task_id,
            agent=self.role,
        ))

        async with RemoteControlClient() as client:
            try:
                response = await client.execute_python(code)
                await bus.publish(Event(
                    kind=EventKind.UE5_RESPONSE,
                    task_id=self.task_id,
                    agent=self.role,
                    payload={"response": response},
                ))
                return AgentResult(
                    role=self.role,
                    output=response,
                    success=True,
                )
            except Exception as exc:
                await bus.publish(Event(
                    kind=EventKind.UE5_ERROR,
                    task_id=self.task_id,
                    agent=self.role,
                    payload={"error": str(exc)},
                ))
                return AgentResult(
                    role=self.role,
                    output=None,
                    success=False,
                    error=str(exc),
                )
