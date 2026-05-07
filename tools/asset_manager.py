from __future__ import annotations

import logging
from typing import Any

from tools.remote_control import RemoteControlClient

logger = logging.getLogger(__name__)

_IMPORT_ASSET_TEMPLATE = """\
import unreal

task = unreal.AssetImportTask()
task.set_editor_property('filename', '{source_path}')
task.set_editor_property('destination_path', '{dest_path}')
task.set_editor_property('replace_existing', True)
task.set_editor_property('automated', True)
task.set_editor_property('save', True)

unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])
print(f'Imported: {source_path} -> {dest_path}')
"""

_DELETE_ASSET_TEMPLATE = """\
import unreal

result = unreal.EditorAssetLibrary.delete_asset('{asset_path}')
print(f'Deleted {asset_path}: {{result}}')
"""

_RENAME_ASSET_TEMPLATE = """\
import unreal

result = unreal.EditorAssetLibrary.rename_asset('{source_path}', '{dest_path}')
print(f'Renamed {source_path} -> {dest_path}: {{result}}')
"""

_LIST_ASSETS_TEMPLATE = """\
import unreal, json

ar = unreal.AssetRegistryHelpers.get_asset_registry()
assets = ar.get_assets_by_path('{package_path}', recursive={recursive})
result = [str(a.package_name) for a in assets]
print(json.dumps(result))
"""


class AssetManager:
    """Import and query UE5 assets via the Python API."""

    def __init__(self) -> None:
        self._rc = RemoteControlClient()

    async def import_asset(self, source_path: str, dest_path: str = "/Game/Imported") -> dict[str, Any]:
        code = _IMPORT_ASSET_TEMPLATE.format(source_path=source_path, dest_path=dest_path)
        logger.info("Importing asset %r to %r", source_path, dest_path)
        return await self._rc.execute_python(code)

    async def list_assets(self, package_path: str = "/Game", recursive: bool = True) -> dict[str, Any]:
        code = _LIST_ASSETS_TEMPLATE.format(
            package_path=package_path,
            recursive="True" if recursive else "False",
        )
        return await self._rc.execute_python(code)

    async def delete_asset(self, asset_path: str) -> dict[str, Any]:
        code = _DELETE_ASSET_TEMPLATE.format(asset_path=asset_path)
        logger.info("Deleting asset %r", asset_path)
        return await self._rc.execute_python(code)

    async def rename_asset(self, source_path: str, dest_path: str) -> dict[str, Any]:
        code = _RENAME_ASSET_TEMPLATE.format(
            source_path=source_path, dest_path=dest_path
        )
        logger.info("Renaming asset %r -> %r", source_path, dest_path)
        return await self._rc.execute_python(code)

    async def close(self) -> None:
        await self._rc.close()

    async def __aenter__(self) -> "AssetManager":
        return self

    async def __aexit__(self, *args) -> None:
        await self.close()
