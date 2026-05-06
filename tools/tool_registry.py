from __future__ import annotations

import logging
from typing import Callable, Any

logger = logging.getLogger(__name__)

_REGISTRY: dict[str, Callable] = {}


def register(name: str):
    def decorator(fn: Callable) -> Callable:
        _REGISTRY[name] = fn
        logger.debug("Registered tool: %s", name)
        return fn
    return decorator


def get_tool(name: str) -> Callable:
    if name not in _REGISTRY:
        raise KeyError(f"Tool {name!r} not found. Available: {list(_REGISTRY)}")
    return _REGISTRY[name]


def list_tools() -> list[str]:
    return list(_REGISTRY.keys())


async def call_tool(name: str, **kwargs: Any) -> Any:
    tool = get_tool(name)
    result = tool(**kwargs)
    import inspect
    if inspect.isawaitable(result):
        return await result
    return result
