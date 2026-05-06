from __future__ import annotations

import json
import logging
from typing import Any

from agents.base_agent import AgentResult, BaseAgent
from config.constants import MAX_PLAN_STEPS

logger = logging.getLogger(__name__)


class PlannerAgent(BaseAgent):
    role = "planner"

    async def _execute(self, prompt: str, context: dict) -> AgentResult:
        system = (
            f"You are a senior UE5 project planner. Break the request into an ordered "
            f"list of subtasks (max {MAX_PLAN_STEPS}). "
            f"Output ONLY a JSON array: "
            f'[{{"step": 1, "action": "...", "agent": "coder|executor", "details": "..."}}]'
        )
        response = await self._call_llm(prompt, system_prompt=system)

        steps = self._parse_steps(response.text)
        logger.info("Planner produced %d steps for task %r", len(steps), self.task_id)

        return AgentResult(
            role=self.role,
            output=steps,
            raw_response=response,
        )

    def _parse_steps(self, text: str) -> list[dict[str, Any]]:
        start = text.find("[")
        end = text.rfind("]") + 1
        if start == -1 or end == 0:
            logger.warning("Planner: no JSON array found in response")
            return [{"step": 1, "action": text.strip(), "agent": "coder", "details": ""}]
        try:
            steps = json.loads(text[start:end])
            return steps[:MAX_PLAN_STEPS]
        except json.JSONDecodeError as exc:
            logger.error("Planner: JSON parse error: %s", exc)
            return [{"step": 1, "action": text.strip(), "agent": "coder", "details": ""}]
