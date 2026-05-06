from __future__ import annotations

import logging
from typing import Any

from tools.remote_control import RemoteControlClient

logger = logging.getLogger(__name__)

_CREATE_BP_TEMPLATE = """\
import unreal

factory = unreal.BlueprintFactory()
factory.set_editor_property('ParentClass', unreal.Actor)

asset_tools = unreal.AssetToolsHelpers.get_asset_tools()
blueprint = asset_tools.create_asset('{bp_name}', '{package_path}', unreal.Blueprint, factory)

if blueprint:
    print(f'Created Blueprint: {{blueprint.get_path_name()}}')
else:
    print('Failed to create Blueprint')
"""

_COMPILE_BP_TEMPLATE = """\
import unreal

bp_path = '{bp_path}'
blueprint = unreal.load_asset(bp_path)
if blueprint:
    unreal.BlueprintEditorLibrary.compile_blueprint(blueprint)
    print(f'Compiled: {{bp_path}}')
else:
    print(f'Blueprint not found: {{bp_path}}')
"""


class BlueprintWriter:
    """Creates and compiles UE5 Blueprint assets via the Python API."""

    def __init__(self) -> None:
        self._rc = RemoteControlClient()

    async def create_blueprint(
        self,
        name: str,
        package_path: str = "/Game/Blueprints",
    ) -> dict[str, Any]:
        code = _CREATE_BP_TEMPLATE.format(bp_name=name, package_path=package_path)
        logger.info("Creating Blueprint %r at %r", name, package_path)
        return await self._rc.execute_python(code)

    async def compile_blueprint(self, bp_path: str) -> dict[str, Any]:
        code = _COMPILE_BP_TEMPLATE.format(bp_path=bp_path)
        logger.info("Compiling Blueprint %r", bp_path)
        return await self._rc.execute_python(code)

    async def run_custom_bp_script(self, code: str) -> dict[str, Any]:
        return await self._rc.execute_python(code)

    async def close(self) -> None:
        await self._rc.close()

    async def __aenter__(self) -> "BlueprintWriter":
        return self

    async def __aexit__(self, *args) -> None:
        await self.close()
