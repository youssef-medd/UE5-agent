from __future__ import annotations

import enum
import logging
import time
from typing import Any

import httpx

from config.settings import get_settings

logger = logging.getLogger(__name__)


class _CBState(enum.Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class _CircuitBreaker:
    """Simple three-state circuit breaker for the UE5 HTTP connection."""

    def __init__(self, fail_threshold: int = 3, recovery_timeout: float = 15.0) -> None:
        self._fail_threshold = fail_threshold
        self._recovery_timeout = recovery_timeout
        self._failures = 0
        self._opened_at: float = 0.0
        self._state = _CBState.CLOSED

    @property
    def state(self) -> _CBState:
        if self._state == _CBState.OPEN:
            if time.monotonic() - self._opened_at >= self._recovery_timeout:
                self._state = _CBState.HALF_OPEN
        return self._state

    def record_success(self) -> None:
        self._failures = 0
        self._state = _CBState.CLOSED

    def record_failure(self) -> None:
        self._failures += 1
        if self._failures >= self._fail_threshold:
            self._state = _CBState.OPEN
            self._opened_at = time.monotonic()
            logger.warning("CircuitBreaker: opened after %d failures", self._failures)

    def allow_request(self) -> bool:
        return self.state in (_CBState.CLOSED, _CBState.HALF_OPEN)


class RemoteControlClient:
    """HTTP client for UE5 Remote Control Plugin (port 30010)."""

    def __init__(self) -> None:
        settings = get_settings()
        self._base_url = settings.ue5_base_url
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(connect=5.0, read=30.0, write=10.0, pool=5.0),
            headers={"Content-Type": "application/json"},
        )
        self._cb = _CircuitBreaker()

    async def _request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        if not self._cb.allow_request():
            raise RuntimeError(
                "RemoteControlClient circuit breaker is OPEN — UE5 is unreachable"
            )
        try:
            resp = await self._client.request(method, url, **kwargs)
            resp.raise_for_status()
            self._cb.record_success()
            return resp
        except Exception as exc:
            self._cb.record_failure()
            raise exc

    async def execute_python(self, code: str) -> dict:
        payload = {
            "objectPath": "/Script/PythonScriptPlugin",
            "functionName": "ExecutePythonScript",
            "parameters": {"PythonScript": code},
        }
        resp = await self._request("PUT", "/remote/object/call", json=payload)
        return resp.json()

    async def get_actor_list(self) -> list[dict]:
        resp = await self._request("GET", "/remote/object/describe")
        return resp.json().get("objects", [])

    async def set_property(self, object_path: str, property_name: str, value: Any) -> dict:
        payload = {
            "objectPath": object_path,
            "propertyName": property_name,
            "propertyValue": {"value": value},
        }
        resp = await self._request("PUT", "/remote/object/property", json=payload)
        return resp.json()

    async def get_property(self, object_path: str, property_name: str) -> dict:
        params = {"objectPath": object_path, "propertyName": property_name}
        resp = await self._request("GET", "/remote/object/property", params=params)
        return resp.json()

    async def call_function(
        self,
        object_path: str,
        function_name: str,
        parameters: dict[str, Any] | None = None,
    ) -> dict:
        payload = {
            "objectPath": object_path,
            "functionName": function_name,
            "parameters": parameters or {},
        }
        resp = await self._request("PUT", "/remote/object/call", json=payload)
        return resp.json()

    async def batch(self, requests: list[dict[str, Any]]) -> list[dict]:
        """Execute multiple remote control requests in a single HTTP call."""
        resp = await self._request("PUT", "/remote/batch", json={"requests": requests})
        return resp.json().get("responses", [])

    async def list_presets(self) -> list[dict]:
        resp = await self._request("GET", "/remote/presets")
        return resp.json().get("presets", [])

    async def get_preset(self, preset_name: str) -> dict:
        resp = await self._request("GET", f"/remote/preset/{preset_name}")
        return resp.json()

    async def set_preset_property(
        self, preset_name: str, property_label: str, value: Any
    ) -> dict:
        payload = {"propertyValues": [{property_label: value}]}
        resp = await self._request(
            "PUT", f"/remote/preset/{preset_name}/property", json=payload
        )
        return resp.json()

    async def health_check(self) -> bool:
        try:
            resp = await self._client.get("/remote/info", timeout=3.0)
            healthy = resp.status_code == 200
            if healthy:
                self._cb.record_success()
            return healthy
        except Exception:
            return False

    @property
    def circuit_state(self) -> str:
        return self._cb.state.value

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "RemoteControlClient":
        return self

    async def __aexit__(self, *args) -> None:
        await self.close()
