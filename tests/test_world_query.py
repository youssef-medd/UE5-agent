from __future__ import annotations

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


def _make_wq():
    with patch("tools.world_query.RemoteControlClient") as MockRC:
        mock_rc = MagicMock()
        MockRC.return_value = mock_rc
        from tools.world_query import WorldQuery
        wq = WorldQuery()
        wq._rc = mock_rc
        return wq, mock_rc


class TestWorldQuery:
    @pytest.mark.asyncio
    async def test_list_actors_calls_execute_python(self):
        wq, rc = _make_wq()
        rc.execute_python = AsyncMock(return_value={"output": "[]"})
        await wq.list_actors()
        rc.execute_python.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_level_info_calls_execute_python(self):
        wq, rc = _make_wq()
        rc.execute_python = AsyncMock(return_value={"output": "{}"})
        await wq.get_level_info()
        rc.execute_python.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_actor_passes_actor_name(self):
        wq, rc = _make_wq()
        rc.execute_python = AsyncMock(return_value={})
        await wq.get_actor("BP_Cube_1")
        code = rc.execute_python.call_args[0][0]
        assert "BP_Cube_1" in code

    @pytest.mark.asyncio
    async def test_find_by_class_includes_class_name(self):
        wq, rc = _make_wq()
        rc.execute_python = AsyncMock(return_value={})
        await wq.find_by_class("PointLight")
        code = rc.execute_python.call_args[0][0]
        assert "PointLight" in code

    @pytest.mark.asyncio
    async def test_get_actor_bounds_includes_name(self):
        wq, rc = _make_wq()
        rc.execute_python = AsyncMock(return_value={})
        await wq.get_actor_bounds("StaticMesh_Floor")
        code = rc.execute_python.call_args[0][0]
        assert "StaticMesh_Floor" in code

    @pytest.mark.asyncio
    async def test_get_components_includes_actor_name(self):
        wq, rc = _make_wq()
        rc.execute_python = AsyncMock(return_value={})
        await wq.get_components("BP_Player")
        code = rc.execute_python.call_args[0][0]
        assert "BP_Player" in code
