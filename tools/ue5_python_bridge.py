from __future__ import annotations

import logging
from typing import Any

from tools.remote_control import RemoteControlClient

logger = logging.getLogger(__name__)

_SPAWN_TEMPLATE = """\
import unreal

asset_path = '{asset_path}'
location = unreal.Vector({x}, {y}, {z})
rotation = unreal.Rotator(0, 0, 0)
scale = unreal.Vector(1, 1, 1)

actor_class = unreal.EditorAssetLibrary.load_blueprint_class(asset_path)
if actor_class:
    actor = unreal.EditorLevelLibrary.spawn_actor_from_class(actor_class, location, rotation)
    actor.set_actor_scale3d(scale)
    print(f'Spawned {{actor_class}} at {{location}}')
else:
    print(f'Asset not found: {asset_path}')
"""

_DELETE_TEMPLATE = """\
import unreal

actors = unreal.EditorLevelLibrary.get_all_level_actors()
to_delete = [a for a in actors if a.get_name() == '{actor_name}']
for actor in to_delete:
    unreal.EditorLevelLibrary.destroy_actor(actor)
print(f'Deleted {{len(to_delete)}} actor(s) named {actor_name}')
"""


class UE5PythonBridge:
    """High-level UE5 operations using Remote Control to execute Python."""

    def __init__(self) -> None:
        self._rc = RemoteControlClient()

    async def spawn_actor(
        self,
        asset_path: str,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
    ) -> dict[str, Any]:
        code = _SPAWN_TEMPLATE.format(asset_path=asset_path, x=x, y=y, z=z)
        logger.info("Spawning actor %r at (%.0f, %.0f, %.0f)", asset_path, x, y, z)
        return await self._rc.execute_python(code)

    async def delete_actor(self, actor_name: str) -> dict[str, Any]:
        code = _DELETE_TEMPLATE.format(actor_name=actor_name)
        logger.info("Deleting actor %r", actor_name)
        return await self._rc.execute_python(code)

    async def run_script(self, code: str) -> dict[str, Any]:
        return await self._rc.execute_python(code)

    async def close(self) -> None:
        await self._rc.close()

    async def __aenter__(self) -> "UE5PythonBridge":
        return self

    async def __aexit__(self, *args) -> None:
        await self.close()
