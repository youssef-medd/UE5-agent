from __future__ import annotations

import logging
import re

from agents.base_agent import AgentResult, BaseAgent

logger = logging.getLogger(__name__)

_CODE_BLOCK = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)


class CoderAgent(BaseAgent):
    role = "coder"

    async def _execute(self, prompt: str, context: dict) -> AgentResult:
        rag_context = context.get("rag_context", "")
        plan_step = context.get("plan_step", {})

        user_prompt = prompt
        if plan_step:
            user_prompt = (
                f"Task: {plan_step.get('action', prompt)}\n"
                f"Details: {plan_step.get('details', '')}\n\n"
                f"Original request: {prompt}"
            )
        if rag_context:
            user_prompt = f"Relevant UE5 API context:\n{rag_context}\n\n{user_prompt}"

        response = await self._call_llm(user_prompt)
        code = self._extract_code(response.text)

        logger.info(
            "Coder generated %d lines of code for task %r",
            len(code.splitlines()), self.task_id,
        )

        return AgentResult(
            role=self.role,
            output=code,
            raw_response=response,
            metadata={"lines": len(code.splitlines())},
        )

    def _extract_code(self, text: str) -> str:
        match = _CODE_BLOCK.search(text)
        if match:
            return match.group(1).strip()
        return text.strip()
