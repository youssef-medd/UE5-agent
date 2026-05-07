from __future__ import annotations

import logging
from typing import Any

from tools.remote_control import RemoteControlClient

logger = logging.getLogger(__name__)

_CREATE_BP_TEMPLATE = """\
import unreal

parent_class = getattr(unreal, '{parent_class}', unreal.Actor)
factory = unreal.BlueprintFactory()
factory.set_editor_property('ParentClass', parent_class)

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

_ADD_COMPONENT_TEMPLATE = """\
import unreal

blueprint = unreal.load_asset('{bp_path}')
if not blueprint:
    print(f'Blueprint not found: {bp_path}')
else:
    component_class = getattr(unreal, '{component_class}', None)
    if not component_class:
        print(f'Component class not found: {component_class}')
    else:
        subsystem = unreal.get_editor_subsystem(unreal.SubobjectDataSubsystem)
        root_data_handle = subsystem.k2_gather_subobject_data_for_blueprint(blueprint)[0]
        new_handle, fail_reason = subsystem.add_new_subobject(
            unreal.AddNewSubobjectParams(
                parent_handle=root_data_handle,
                new_class=component_class,
                blueprint_context=blueprint
            )
        )
        if fail_reason:
            print(f'Failed to add component: {{fail_reason}}')
        else:
            print(f'Added {{component_class}} to {{blueprint.get_name()}}')
"""


class BlueprintWriter:
    """Creates and compiles UE5 Blueprint assets via the Python API."""

    def __init__(self) -> None:
        self._rc = RemoteControlClient()

    async def create_blueprint(
        self,
        name: str,
        package_path: str = "/Game/Blueprints",
        parent_class: str = "Actor",
    ) -> dict[str, Any]:
        code = _CREATE_BP_TEMPLATE.format(
            bp_name=name, package_path=package_path, parent_class=parent_class
        )
        logger.info("Creating Blueprint %r (parent=%s) at %r", name, parent_class, package_path)
        return await self._rc.execute_python(code)

    async def compile_blueprint(self, bp_path: str) -> dict[str, Any]:
        code = _COMPILE_BP_TEMPLATE.format(bp_path=bp_path)
        logger.info("Compiling Blueprint %r", bp_path)
        return await self._rc.execute_python(code)

    async def add_component(self, bp_path: str, component_class: str) -> dict[str, Any]:
        code = _ADD_COMPONENT_TEMPLATE.format(
            bp_path=bp_path, component_class=component_class
        )
        logger.info("Adding %r component to Blueprint %r", component_class, bp_path)
        return await self._rc.execute_python(code)

    async def run_custom_bp_script(self, code: str) -> dict[str, Any]:
        return await self._rc.execute_python(code)

    async def close(self) -> None:
        await self._rc.close()

    async def __aenter__(self) -> "BlueprintWriter":
        return self

    async def __aexit__(self, *args) -> None:
        await self.close()
