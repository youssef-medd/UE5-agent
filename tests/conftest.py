import pytest


@pytest.fixture(autouse=True)
def reset_tool_registry():
    """Isolate the tool registry between tests."""
    from tools import tool_registry
    original = dict(tool_registry._REGISTRY)
    yield
    tool_registry._REGISTRY.clear()
    tool_registry._REGISTRY.update(original)
