from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_STATE_FILE = Path("logs/state.json")


class StateManager:
    """Simple JSON-backed shared state for agent coordination.

    Falls back to file-based storage when Redis is not available.
    """

    def __init__(self, redis_url: str = "") -> None:
        self._redis_url = redis_url
        self._redis = None
        self._local: dict[str, Any] = {}
        self._use_redis = False

        if redis_url:
            try:
                import redis.asyncio as aioredis
                self._redis = aioredis.from_url(redis_url, decode_responses=True)
                self._use_redis = True
                logger.info("StateManager: using Redis at %s", redis_url)
            except ImportError:
                logger.warning("redis package not installed, using file-based state")
            except Exception as exc:
                logger.warning("Redis unavailable (%s), using file-based state", exc)

        if not self._use_redis:
            self._load_local()

    async def set(self, key: str, value: Any) -> None:
        if self._use_redis:
            await self._redis.set(key, json.dumps(value))
        else:
            self._local[key] = value
            self._save_local()

    async def get(self, key: str, default: Any = None) -> Any:
        if self._use_redis:
            raw = await self._redis.get(key)
            return json.loads(raw) if raw else default
        return self._local.get(key, default)

    async def delete(self, key: str) -> None:
        if self._use_redis:
            await self._redis.delete(key)
        else:
            self._local.pop(key, None)
            self._save_local()

    def _save_local(self) -> None:
        _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(_STATE_FILE, "w") as f:
            json.dump(self._local, f, indent=2)

    def _load_local(self) -> None:
        if _STATE_FILE.exists():
            try:
                with open(_STATE_FILE) as f:
                    self._local = json.load(f)
            except Exception:
                self._local = {}
