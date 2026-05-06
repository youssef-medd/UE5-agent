import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from agents.planner_agent import PlannerAgent
from agents.coder_agent import CoderAgent
from agents.reviewer_agent import ReviewerAgent
from llm.base_llm import LLMResponse


def _mock_response(text: str) -> LLMResponse:
    return LLMResponse(text=text, model="mock", tokens_used=10)


@pytest.mark.asyncio
class TestPlannerAgent:
    async def test_parses_valid_json_steps(self):
        agent = PlannerAgent(task_id="t1")
        steps_json = '[{"step": 1, "action": "spawn light", "agent": "coder", "details": ""}]'

        with patch.object(agent, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = _mock_response(steps_json)
            result = await agent.run("add a light")

        assert result.success
        assert isinstance(result.output, list)
        assert result.output[0]["step"] == 1

    async def test_falls_back_on_bad_json(self):
        agent = PlannerAgent(task_id="t2")

        with patch.object(agent, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = _mock_response("no json here")
            result = await agent.run("do something")

        assert result.success
        assert len(result.output) == 1


@pytest.mark.asyncio
class TestCoderAgent:
    async def test_extracts_code_block(self):
        agent = CoderAgent(task_id="t3")
        response_text = "Here is the code:\n```python\nimport unreal\nunreal.log('hi')\n```"

        with patch.object(agent, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = _mock_response(response_text)
            result = await agent.run("add a log message")

        assert result.success
        assert "import unreal" in result.output

    async def test_returns_raw_text_if_no_block(self):
        agent = CoderAgent(task_id="t4")

        with patch.object(agent, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = _mock_response("import unreal")
            result = await agent.run("raw code")

        assert "import unreal" in result.output


@pytest.mark.asyncio
class TestReviewerAgent:
    async def test_approved_review(self):
        agent = ReviewerAgent(task_id="t5")
        review_json = '{"approved": true, "risk_level": "low", "issues": [], "suggestions": [], "summary": "looks good"}'

        with patch.object(agent, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = _mock_response(review_json)
            result = await agent.run("review this", context={"code": "import unreal"})

        assert result.success
        assert result.output["approved"] is True

    async def test_rejected_review(self):
        agent = ReviewerAgent(task_id="t6")
        review_json = '{"approved": false, "risk_level": "high", "issues": ["mass delete"], "suggestions": [], "summary": "too risky"}'

        with patch.object(agent, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = _mock_response(review_json)
            result = await agent.run("review this", context={"code": "delete everything"})

        assert not result.success
