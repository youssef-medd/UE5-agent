from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


def _make_am():
    with patch("tools.asset_manager.RemoteControlClient") as MockRC:
        mock_rc = MagicMock()
        MockRC.return_value = mock_rc
        from tools.asset_manager import AssetManager
        am = AssetManager()
        am._rc = mock_rc
        return am, mock_rc


class TestAssetManager:
    @pytest.mark.asyncio
    async def test_import_asset_includes_paths(self):
        am, rc = _make_am()
        rc.execute_python = AsyncMock(return_value={})
        await am.import_asset("/tmp/mesh.fbx", "/Game/Imported")
        code = rc.execute_python.call_args[0][0]
        assert "/tmp/mesh.fbx" in code
        assert "/Game/Imported" in code

    @pytest.mark.asyncio
    async def test_list_assets_includes_package_path(self):
        am, rc = _make_am()
        rc.execute_python = AsyncMock(return_value={})
        await am.list_assets("/Game/Meshes", recursive=False)
        code = rc.execute_python.call_args[0][0]
        assert "/Game/Meshes" in code
        assert "False" in code

    @pytest.mark.asyncio
    async def test_delete_asset_includes_path(self):
        am, rc = _make_am()
        rc.execute_python = AsyncMock(return_value={})
        await am.delete_asset("/Game/Old/OldMesh")
        code = rc.execute_python.call_args[0][0]
        assert "/Game/Old/OldMesh" in code

    @pytest.mark.asyncio
    async def test_rename_asset_includes_both_paths(self):
        am, rc = _make_am()
        rc.execute_python = AsyncMock(return_value={})
        await am.rename_asset("/Game/Old/Mesh", "/Game/New/Mesh")
        code = rc.execute_python.call_args[0][0]
        assert "/Game/Old/Mesh" in code
        assert "/Game/New/Mesh" in code

    @pytest.mark.asyncio
    async def test_duplicate_asset_includes_both_paths(self):
        am, rc = _make_am()
        rc.execute_python = AsyncMock(return_value={})
        await am.duplicate_asset("/Game/Mesh", "/Game/MeshCopy")
        code = rc.execute_python.call_args[0][0]
        assert "/Game/Mesh" in code
        assert "/Game/MeshCopy" in code

    @pytest.mark.asyncio
    async def test_asset_exists_includes_path(self):
        am, rc = _make_am()
        rc.execute_python = AsyncMock(return_value={})
        await am.asset_exists("/Game/MyAsset")
        code = rc.execute_python.call_args[0][0]
        assert "/Game/MyAsset" in code

    @pytest.mark.asyncio
    async def test_get_metadata_includes_path(self):
        am, rc = _make_am()
        rc.execute_python = AsyncMock(return_value={})
        await am.get_metadata("/Game/MyMaterial")
        code = rc.execute_python.call_args[0][0]
        assert "/Game/MyMaterial" in code
