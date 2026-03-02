"""Direct module-level tests for ``thenvoi.runtime.tools``."""

from __future__ import annotations

import pytest

import thenvoi.runtime.tool_definitions as tool_definitions
import thenvoi.runtime.tools as runtime_tools

pytestmark = pytest.mark.contract_gate


def test_runtime_tools_module_exports_core_symbols() -> None:
    assert runtime_tools.AgentTools.__name__ == "AgentTools"
    assert runtime_tools.MCP_TOOL_PREFIX == "mcp__thenvoi__"
    assert "thenvoi_send_message" in runtime_tools.TOOL_MODELS


def test_runtime_tools_mcp_tool_names_are_sorted_and_prefixed() -> None:
    names = frozenset({"thenvoi_zeta", "thenvoi_alpha"})

    result = runtime_tools.mcp_tool_names(names)

    assert result == [
        "mcp__thenvoi__thenvoi_alpha",
        "mcp__thenvoi__thenvoi_zeta",
    ]


def test_runtime_tools_description_fallback_for_unknown_tool() -> None:
    assert runtime_tools.get_tool_description("unknown_tool_name") == (
        "Execute unknown_tool_name"
    )


def test_runtime_tools_reexports_tool_registry_symbols() -> None:
    assert runtime_tools.TOOL_MODELS is tool_definitions.TOOL_MODELS
    assert runtime_tools.CHAT_TOOL_NAMES is tool_definitions.CHAT_TOOL_NAMES
