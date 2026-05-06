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


class WorldQuery:
    """Reads the current UE5 world state via Remote Control."""

    def __init__(self) -> None:
        self._rc = RemoteControlClient()

    async def list_actors(self) -> dict[str, Any]:
        logger.debug("Querying actor list from UE5")
        return await self._rc.execute_python(_LIST_ACTORS_SCRIPT)

    async def get_level_info(self) -> dict[str, Any]:
        return await self._rc.execute_python(_GET_LEVEL_INFO_SCRIPT)

    async def close(self) -> None:
        await self._rc.close()

    async def __aenter__(self) -> "WorldQuery":
        return self

    async def __aexit__(self, *args) -> None:
        await self.close()
