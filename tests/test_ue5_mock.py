import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from agents.executor_agent import ExecutorAgent


@pytest.mark.asyncio
class TestExecutorAgent:
    async def test_skips_when_not_approved(self):
        agent = ExecutorAgent(task_id="e1")
        result = await agent.run(
            "run",
            context={
                "code": "import unreal",
                "review": {"approved": False, "summary": "too risky"},
                "dry_run": False,
            },
        )
        assert not result.success
        assert result.output["action"] == "skip"

    async def test_dry_run_returns_success(self):
        agent = ExecutorAgent(task_id="e2")
        result = await agent.run(
            "run",
            context={
                "code": "import unreal\nunreal.log('hi')",
                "review": {"approved": True},
                "dry_run": True,
            },
        )
        assert result.success
        assert result.output["action"] == "dry_run"
        assert result.metadata.get("dry_run") is True

    async def test_no_code_returns_error(self):
        agent = ExecutorAgent(task_id="e3")
        result = await agent.run(
            "run",
            context={"code": "", "review": {"approved": True}, "dry_run": False},
        )
        assert not result.success
        assert "No code" in result.error

    async def test_live_execution_calls_remote_control(self):
        agent = ExecutorAgent(task_id="e4")
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.execute_python = AsyncMock(return_value={"status": "ok"})

        with patch("agents.executor_agent.RemoteControlClient", return_value=mock_client):
            result = await agent.run(
                "run",
                context={
                    "code": "import unreal",
                    "review": {"approved": True},
                    "dry_run": False,
                },
            )

        assert result.success
        mock_client.execute_python.assert_called_once()
