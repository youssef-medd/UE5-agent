from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


def _make_bw():
    with patch("tools.blueprint_writer.RemoteControlClient") as MockRC:
        mock_rc = MagicMock()
        MockRC.return_value = mock_rc
        from tools.blueprint_writer import BlueprintWriter
        bw = BlueprintWriter()
        bw._rc = mock_rc
        return bw, mock_rc


class TestBlueprintWriter:
    @pytest.mark.asyncio
    async def test_create_blueprint_uses_parent_class(self):
        bw, rc = _make_bw()
        rc.execute_python = AsyncMock(return_value={})
        await bw.create_blueprint("BP_Enemy", parent_class="Pawn")
        code = rc.execute_python.call_args[0][0]
        assert "BP_Enemy" in code
        assert "Pawn" in code

    @pytest.mark.asyncio
    async def test_create_blueprint_default_parent_is_actor(self):
        bw, rc = _make_bw()
        rc.execute_python = AsyncMock(return_value={})
        await bw.create_blueprint("BP_Simple")
        code = rc.execute_python.call_args[0][0]
        assert "Actor" in code

    @pytest.mark.asyncio
    async def test_compile_blueprint_includes_path(self):
        bw, rc = _make_bw()
        rc.execute_python = AsyncMock(return_value={})
        await bw.compile_blueprint("/Game/Blueprints/BP_Enemy")
        code = rc.execute_python.call_args[0][0]
        assert "/Game/Blueprints/BP_Enemy" in code

    @pytest.mark.asyncio
    async def test_add_component_includes_class(self):
        bw, rc = _make_bw()
        rc.execute_python = AsyncMock(return_value={})
        await bw.add_component("/Game/BP_Enemy", "StaticMeshComponent")
        code = rc.execute_python.call_args[0][0]
        assert "StaticMeshComponent" in code

    @pytest.mark.asyncio
    async def test_add_event_graph_includes_event_name(self):
        bw, rc = _make_bw()
        rc.execute_python = AsyncMock(return_value={})
        await bw.add_event_graph("/Game/BP_Enemy", "OnDeath")
        code = rc.execute_python.call_args[0][0]
        assert "OnDeath" in code

    @pytest.mark.asyncio
    async def test_set_variable_includes_name_and_value(self):
        bw, rc = _make_bw()
        rc.execute_python = AsyncMock(return_value={})
        await bw.set_variable("/Game/BP_Enemy", "Health", 100)
        code = rc.execute_python.call_args[0][0]
        assert "Health" in code
        assert "100" in code

    @pytest.mark.asyncio
    async def test_get_variables_includes_bp_path(self):
        bw, rc = _make_bw()
        rc.execute_python = AsyncMock(return_value={})
        await bw.get_variables("/Game/BP_Enemy")
        code = rc.execute_python.call_args[0][0]
        assert "/Game/BP_Enemy" in code
