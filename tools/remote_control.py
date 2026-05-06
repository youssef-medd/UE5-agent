from __future__ import annotations

import logging

import httpx

from config.settings import get_settings

logger = logging.getLogger(__name__)


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

    async def execute_python(self, code: str) -> dict:
        payload = {"objectPath": "/Script/PythonScriptPlugin", "functionName": "ExecutePythonScript", "parameters": {"PythonScript": code}}
        resp = await self._client.put("/remote/object/call", json=payload)
        resp.raise_for_status()
        return resp.json()

    async def get_actor_list(self) -> list[dict]:
        resp = await self._client.get("/remote/object/describe")
        resp.raise_for_status()
        return resp.json().get("objects", [])

    async def set_property(self, object_path: str, property_name: str, value) -> dict:
        payload = {
            "objectPath": object_path,
            "propertyName": property_name,
            "propertyValue": {"value": value},
        }
        resp = await self._client.put("/remote/object/property", json=payload)
        resp.raise_for_status()
        return resp.json()

    async def health_check(self) -> bool:
        try:
            resp = await self._client.get("/remote/info", timeout=3.0)
            return resp.status_code == 200
        except Exception:
            return False

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "RemoteControlClient":
        return self

    async def __aexit__(self, *args) -> None:
        await self.close()
