from __future__ import annotations

import json
import logging

from agents.base_agent import AgentResult, BaseAgent

logger = logging.getLogger(__name__)


class ReviewerAgent(BaseAgent):
    role = "reviewer"

    async def _execute(self, prompt: str, context: dict) -> AgentResult:
        code = context.get("code", prompt)
        user_prompt = f"Review the following UE5 Python code:\n\n```python\n{code}\n```"

        response = await self._call_llm(user_prompt)
        review = self._parse_review(response.text)

        approved = review.get("approved", False)
        risk = review.get("risk_level", "high")
        logger.info(
            "Reviewer: approved=%s risk=%s for task %r",
            approved, risk, self.task_id,
        )

        return AgentResult(
            role=self.role,
            output=review,
            raw_response=response,
            success=approved,
            metadata={"risk_level": risk},
        )

    def _parse_review(self, text: str) -> dict:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start == -1 or end == 0:
            return {
                "approved": False,
                "risk_level": "high",
                "issues": ["Reviewer returned non-JSON output"],
                "suggestions": [],
                "summary": text[:200],
            }
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            return {
                "approved": False,
                "risk_level": "high",
                "issues": ["Could not parse reviewer JSON"],
                "suggestions": [],
                "summary": text[:200],
            }
