from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from tools.remote_control import RemoteControlClient, _CircuitBreaker, _CBState


class TestCircuitBreaker:
    def test_starts_closed(self):
        cb = _CircuitBreaker()
        assert cb.state == _CBState.CLOSED

    def test_opens_after_threshold(self):
        cb = _CircuitBreaker(fail_threshold=3)
        for _ in range(3):
            cb.record_failure()
        assert cb.state == _CBState.OPEN

    def test_resets_on_success(self):
        cb = _CircuitBreaker(fail_threshold=2)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == _CBState.OPEN
        cb.record_success()
        assert cb.state == _CBState.CLOSED

    def test_blocks_requests_when_open(self):
        cb = _CircuitBreaker(fail_threshold=1)
        cb.record_failure()
        assert not cb.allow_request()

    def test_allows_requests_when_closed(self):
        cb = _CircuitBreaker()
        assert cb.allow_request()

    def test_transitions_to_half_open_after_timeout(self):
        import time
        cb = _CircuitBreaker(fail_threshold=1, recovery_timeout=0.01)
        cb.record_failure()
        assert cb.state == _CBState.OPEN
        time.sleep(0.02)
        assert cb.state == _CBState.HALF_OPEN


class TestRemoteControlClient:
    @pytest.fixture
    def client(self):
        with patch("tools.remote_control.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(ue5_base_url="http://localhost:30010")
            with patch("httpx.AsyncClient"):
                return RemoteControlClient()

    @pytest.mark.asyncio
    async def test_circuit_open_raises(self, client):
        client._cb._state = _CBState.OPEN
        client._cb._opened_at = 9e18
        with pytest.raises(RuntimeError, match="circuit breaker is OPEN"):
            await client._request("GET", "/remote/info")

    @pytest.mark.asyncio
    async def test_execute_python_builds_correct_payload(self, client):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"returnValue": "ok"}
        mock_resp.raise_for_status = MagicMock()
        client._client.request = AsyncMock(return_value=mock_resp)

        result = await client.execute_python("import unreal")

        call_args = client._client.request.call_args
        assert call_args[0][0] == "PUT"
        assert "/remote/object/call" in call_args[0][1]
        body = call_args[1]["json"]
        assert body["functionName"] == "ExecutePythonScript"
        assert "import unreal" in body["parameters"]["PythonScript"]

    @pytest.mark.asyncio
    async def test_batch_sends_requests_array(self, client):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"responses": [{"status": 200}]}
        mock_resp.raise_for_status = MagicMock()
        client._client.request = AsyncMock(return_value=mock_resp)

        reqs = [{"url": "/remote/object/describe", "verb": "GET", "body": {}}]
        result = await client.batch(reqs)

        body = client._client.request.call_args[1]["json"]
        assert "requests" in body
        assert result == [{"status": 200}]

    @pytest.mark.asyncio
    async def test_health_check_returns_true_on_200(self, client):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        client._client.get = AsyncMock(return_value=mock_resp)

        assert await client.health_check() is True

    @pytest.mark.asyncio
    async def test_health_check_returns_false_on_exception(self, client):
        client._client.get = AsyncMock(side_effect=Exception("connection refused"))
        assert await client.health_check() is False

    def test_circuit_state_property(self, client):
        assert client.circuit_state == "closed"
