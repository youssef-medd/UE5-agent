from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


def _make_bridge():
    with patch("tools.ue5_python_bridge.RemoteControlClient") as MockRC:
        mock_rc = MagicMock()
        MockRC.return_value = mock_rc
        from tools.ue5_python_bridge import UE5PythonBridge
        bridge = UE5PythonBridge()
        bridge._rc = mock_rc
        return bridge, mock_rc


class TestUE5PythonBridge:
    @pytest.mark.asyncio
    async def test_spawn_actor_contains_asset_path(self):
        bridge, rc = _make_bridge()
        rc.execute_python = AsyncMock(return_value={})
        await bridge.spawn_actor("/Game/BP_Cube", x=100, y=200, z=0)
        code = rc.execute_python.call_args[0][0]
        assert "/Game/BP_Cube" in code
        assert "100" in code

    @pytest.mark.asyncio
    async def test_delete_actor_contains_name(self):
        bridge, rc = _make_bridge()
        rc.execute_python = AsyncMock(return_value={})
        await bridge.delete_actor("BP_Cube_1")
        code = rc.execute_python.call_args[0][0]
        assert "BP_Cube_1" in code

    @pytest.mark.asyncio
    async def test_move_actor_includes_coordinates(self):
        bridge, rc = _make_bridge()
        rc.execute_python = AsyncMock(return_value={})
        await bridge.move_actor("MyActor", x=500, y=-200, z=100)
        code = rc.execute_python.call_args[0][0]
        assert "500" in code and "-200" in code and "100" in code

    @pytest.mark.asyncio
    async def test_rotate_actor_includes_angles(self):
        bridge, rc = _make_bridge()
        rc.execute_python = AsyncMock(return_value={})
        await bridge.rotate_actor("MyActor", pitch=45, yaw=90, roll=0)
        code = rc.execute_python.call_args[0][0]
        assert "45" in code and "90" in code

    @pytest.mark.asyncio
    async def test_set_visibility_hidden_true(self):
        bridge, rc = _make_bridge()
        rc.execute_python = AsyncMock(return_value={})
        await bridge.set_visibility("MyActor", hidden=True)
        code = rc.execute_python.call_args[0][0]
        assert "True" in code

    @pytest.mark.asyncio
    async def test_set_physics_simulate_false(self):
        bridge, rc = _make_bridge()
        rc.execute_python = AsyncMock(return_value={})
        await bridge.set_physics("MyActor", simulate=False)
        code = rc.execute_python.call_args[0][0]
        assert "False" in code

    @pytest.mark.asyncio
    async def test_attach_actor_both_names_in_script(self):
        bridge, rc = _make_bridge()
        rc.execute_python = AsyncMock(return_value={})
        await bridge.attach_actor("ChildActor", "ParentActor")
        code = rc.execute_python.call_args[0][0]
        assert "ChildActor" in code and "ParentActor" in code

    @pytest.mark.asyncio
    async def test_set_label_contains_label(self):
        bridge, rc = _make_bridge()
        rc.execute_python = AsyncMock(return_value={})
        await bridge.set_label("MyActor", "MyCustomLabel")
        code = rc.execute_python.call_args[0][0]
        assert "MyCustomLabel" in code

    @pytest.mark.asyncio
    async def test_set_tags_contains_tags(self):
        bridge, rc = _make_bridge()
        rc.execute_python = AsyncMock(return_value={})
        await bridge.set_tags("MyActor", ["enemy", "boss"])
        code = rc.execute_python.call_args[0][0]
        assert "enemy" in code and "boss" in code
