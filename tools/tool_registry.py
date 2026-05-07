from __future__ import annotations

import inspect
import logging
from typing import Any, Callable

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
    if inspect.isawaitable(result):
        return await result
    return result


def _register_all() -> None:
    from tools.ue5_python_bridge import UE5PythonBridge
    from tools.world_query import WorldQuery
    from tools.asset_manager import AssetManager
    from tools.blueprint_writer import BlueprintWriter

    bridge = UE5PythonBridge()
    wq = WorldQuery()
    am = AssetManager()
    bw = BlueprintWriter()

    _REGISTRY["spawn_actor"] = bridge.spawn_actor
    _REGISTRY["delete_actor"] = bridge.delete_actor
    _REGISTRY["move_actor"] = bridge.move_actor
    _REGISTRY["rotate_actor"] = bridge.rotate_actor
    _REGISTRY["scale_actor"] = bridge.scale_actor
    _REGISTRY["duplicate_actor"] = bridge.duplicate_actor
    _REGISTRY["set_material"] = bridge.set_material
    _REGISTRY["set_visibility"] = bridge.set_visibility
    _REGISTRY["set_physics"] = bridge.set_physics
    _REGISTRY["attach_actor"] = bridge.attach_actor
    _REGISTRY["detach_actor"] = bridge.detach_actor
    _REGISTRY["set_label"] = bridge.set_label
    _REGISTRY["set_tags"] = bridge.set_tags
    _REGISTRY["run_script"] = bridge.run_script

    _REGISTRY["list_actors"] = wq.list_actors
    _REGISTRY["get_level_info"] = wq.get_level_info
    _REGISTRY["get_actor"] = wq.get_actor
    _REGISTRY["find_by_class"] = wq.find_by_class
    _REGISTRY["get_actor_bounds"] = wq.get_actor_bounds
    _REGISTRY["get_components"] = wq.get_components

    _REGISTRY["import_asset"] = am.import_asset
    _REGISTRY["list_assets"] = am.list_assets
    _REGISTRY["delete_asset"] = am.delete_asset
    _REGISTRY["rename_asset"] = am.rename_asset
    _REGISTRY["duplicate_asset"] = am.duplicate_asset
    _REGISTRY["asset_exists"] = am.asset_exists
    _REGISTRY["get_asset_metadata"] = am.get_metadata

    _REGISTRY["create_blueprint"] = bw.create_blueprint
    _REGISTRY["compile_blueprint"] = bw.compile_blueprint
    _REGISTRY["add_component"] = bw.add_component
    _REGISTRY["add_event_graph"] = bw.add_event_graph
    _REGISTRY["set_bp_variable"] = bw.set_variable
    _REGISTRY["get_bp_variables"] = bw.get_variables
    _REGISTRY["run_bp_script"] = bw.run_custom_bp_script

    logger.info("Tool registry loaded %d tools", len(_REGISTRY))


_register_all()
