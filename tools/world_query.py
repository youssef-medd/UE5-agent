from __future__ import annotations

import logging
from typing import Any

from tools.remote_control import RemoteControlClient

logger = logging.getLogger(__name__)


_LIST_ACTORS_SCRIPT = """\
import unreal, json

actors = unreal.EditorLevelLibrary.get_all_level_actors()
result = []
for a in actors:
    loc = a.get_actor_location()
    result.append({
        'name': a.get_name(),
        'class': a.get_class().get_name(),
        'location': {'x': loc.x, 'y': loc.y, 'z': loc.z},
    })
print(json.dumps(result))
"""

_GET_LEVEL_INFO_SCRIPT = """\
import unreal, json

world = unreal.EditorLevelLibrary.get_editor_world()
info = {
    'world_name': world.get_name() if world else 'unknown',
    'actor_count': len(unreal.EditorLevelLibrary.get_all_level_actors()),
}
print(json.dumps(info))
"""

_FIND_BY_CLASS_TEMPLATE = """\
import unreal, json

actors = unreal.EditorLevelLibrary.get_all_level_actors()
matches = []
for a in actors:
    if '{class_name}' in a.get_class().get_name():
        loc = a.get_actor_location()
        matches.append({{
            'name': a.get_name(),
            'class': a.get_class().get_name(),
            'location': {{'x': loc.x, 'y': loc.y, 'z': loc.z}},
        }})
print(json.dumps(matches))
"""

_GET_ACTOR_BOUNDS_TEMPLATE = """\
import unreal, json

actors = unreal.EditorLevelLibrary.get_all_level_actors()
result = None
for a in actors:
    if a.get_name() == '{actor_name}':
        origin, extent = a.get_actor_bounds(False)
        result = {{
            'name': a.get_name(),
            'origin': {{'x': origin.x, 'y': origin.y, 'z': origin.z}},
            'extent': {{'x': extent.x, 'y': extent.y, 'z': extent.z}},
        }}
        break
print(json.dumps(result))
"""

_GET_ACTOR_SCRIPT_TEMPLATE = """\
import unreal, json

actors = unreal.EditorLevelLibrary.get_all_level_actors()
result = None
for a in actors:
    if a.get_name() == '{actor_name}':
        loc = a.get_actor_location()
        rot = a.get_actor_rotation()
        scale = a.get_actor_scale3d()
        result = {{
            'name': a.get_name(),
            'class': a.get_class().get_name(),
            'location': {{'x': loc.x, 'y': loc.y, 'z': loc.z}},
            'rotation': {{'pitch': rot.pitch, 'yaw': rot.yaw, 'roll': rot.roll}},
            'scale': {{'x': scale.x, 'y': scale.y, 'z': scale.z}},
            'hidden': a.is_hidden_ed(),
        }}
        break
print(json.dumps(result))
"""


class WorldQuery:
    """Reads the current UE5 world state via Remote Control."""

    def __init__(self) -> None:
        self._rc = RemoteControlClient()

    async def list_actors(self) -> dict[str, Any]:
        logger.debug("Querying actor list from UE5")
        return await self._rc.execute_python(_LIST_ACTORS_SCRIPT)

    async def get_level_info(self) -> dict[str, Any]:
        return await self._rc.execute_python(_GET_LEVEL_INFO_SCRIPT)

    async def get_actor(self, actor_name: str) -> dict[str, Any]:
        code = _GET_ACTOR_SCRIPT_TEMPLATE.format(actor_name=actor_name)
        return await self._rc.execute_python(code)

    async def find_by_class(self, class_name: str) -> dict[str, Any]:
        code = _FIND_BY_CLASS_TEMPLATE.format(class_name=class_name)
        logger.debug("Finding actors of class %r", class_name)
        return await self._rc.execute_python(code)

    async def get_actor_bounds(self, actor_name: str) -> dict[str, Any]:
        code = _GET_ACTOR_BOUNDS_TEMPLATE.format(actor_name=actor_name)
        return await self._rc.execute_python(code)

    async def close(self) -> None:
        await self._rc.close()

    async def __aenter__(self) -> "WorldQuery":
        return self

    async def __aexit__(self, *args) -> None:
        await self.close()
