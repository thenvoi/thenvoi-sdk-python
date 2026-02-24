"""Conformance tests that detect tool-name drift between the central registry and adapters.

When a new tool is added to ``TOOL_MODELS`` in ``runtime/tools.py``, these
tests will fail for any adapter or integration that is missing it — surfacing
the gap before it reaches production.

Strategy
--------
Each adapter / integration file is scanned for *string literals* that match
known tool names from the central ``TOOL_MODELS`` registry.  The found set is
compared against an *expected* set that describes which tools the file should
cover.

Files that derive their tool lists from the central constants (e.g. via
``mcp_tool_names()``) are verified to have the correct import, not rescanned
for individual names.
"""

from __future__ import annotations

from pathlib import Path

from thenvoi.runtime.tools import (
    ALL_TOOL_NAMES,
    BASE_TOOL_NAMES,
    CHAT_TOOL_NAMES,
    CONTACT_TOOL_NAMES,
    MEMORY_TOOL_NAMES,
)

_SRC_ROOT = Path(__file__).resolve().parents[2] / "src" / "thenvoi"


def _extract_tool_names(source: str) -> set[str]:
    """Extract all tool names from ``ALL_TOOL_NAMES`` referenced in source."""
    return {name for name in ALL_TOOL_NAMES if name in source}


# ---------------------------------------------------------------------------
# Per-adapter / integration expected tool sets
# ---------------------------------------------------------------------------


class TestClaudeSDKAdapterToolDrift:
    """Claude SDK adapter (adapters/claude_sdk.py) must cover all tools."""

    _FILE = _SRC_ROOT / "adapters" / "claude_sdk.py"

    def test_derives_base_tools_from_central_registry(self):
        """Verify the adapter imports BASE_TOOL_NAMES instead of hardcoding."""
        source = self._FILE.read_text()
        assert "BASE_TOOL_NAMES" in source, (
            "Claude SDK adapter should import BASE_TOOL_NAMES from "
            "thenvoi.runtime.tools instead of hardcoding MCP tool names."
        )

    def test_derives_memory_tools_from_central_registry(self):
        """Verify the adapter imports MEMORY_TOOL_NAMES instead of hardcoding."""
        source = self._FILE.read_text()
        assert "MEMORY_TOOL_NAMES" in source, (
            "Claude SDK adapter should import MEMORY_TOOL_NAMES from "
            "thenvoi.runtime.tools instead of hardcoding MCP tool names."
        )

    def test_all_tools_registered_as_mcp_handlers(self):
        """Every tool in TOOL_MODELS has a @tool decorator in the MCP server."""
        source = self._FILE.read_text()
        found = _extract_tool_names(source)
        missing = ALL_TOOL_NAMES - found
        assert not missing, (
            f"Claude SDK adapter is missing @tool handlers for: {sorted(missing)}. "
            f"Add the tool implementation inside _create_mcp_server()."
        )


class TestClaudeSDKIntegrationToolDrift:
    """Claude SDK integration (integrations/claude_sdk/tools.py) — chat tools only."""

    _FILE = _SRC_ROOT / "integrations" / "claude_sdk" / "tools.py"

    def test_derives_tool_list_from_central_registry(self):
        """Verify THENVOI_TOOLS is derived from CHAT_TOOL_NAMES."""
        source = self._FILE.read_text()
        assert "CHAT_TOOL_NAMES" in source, (
            "Claude SDK integration tools should import CHAT_TOOL_NAMES from "
            "thenvoi.runtime.tools instead of hardcoding MCP tool names."
        )

    def test_all_chat_tools_implemented(self):
        """Every chat tool has a @tool handler."""
        source = self._FILE.read_text()
        found = _extract_tool_names(source)
        missing = CHAT_TOOL_NAMES - found
        assert not missing, (
            f"Claude SDK integration is missing @tool handlers for: {sorted(missing)}. "
            f"Add the tool implementation inside create_thenvoi_mcp_server()."
        )


class TestClaudeSDKPromptsToolDrift:
    """Claude SDK prompts (integrations/claude_sdk/prompts.py) — chat tools only."""

    _FILE = _SRC_ROOT / "integrations" / "claude_sdk" / "prompts.py"

    def test_all_chat_tools_documented_in_prompt(self):
        """Every chat tool is mentioned in the system prompt template."""
        source = self._FILE.read_text()
        found = _extract_tool_names(source)
        missing = CHAT_TOOL_NAMES - found
        assert not missing, (
            f"Claude SDK prompts are missing documentation for: {sorted(missing)}. "
            f"Add tool documentation to the system prompt in prompts.py."
        )


class TestLangGraphToolDrift:
    """LangGraph integration (integrations/langgraph/langchain_tools.py)."""

    _FILE = _SRC_ROOT / "integrations" / "langgraph" / "langchain_tools.py"

    def test_all_tools_registered(self):
        """Every tool in TOOL_MODELS has a StructuredTool wrapper."""
        source = self._FILE.read_text()
        found = _extract_tool_names(source)
        missing = ALL_TOOL_NAMES - found
        assert not missing, (
            f"LangGraph integration is missing StructuredTool wrappers for: "
            f"{sorted(missing)}. Add wrappers in create_langchain_tools()."
        )


class TestCrewAIToolDrift:
    """CrewAI adapter (adapters/crewai.py)."""

    _FILE = _SRC_ROOT / "adapters" / "crewai.py"

    def test_all_tools_registered(self):
        """Every tool in TOOL_MODELS has a CrewAI tool class."""
        source = self._FILE.read_text()
        found = _extract_tool_names(source)
        missing = ALL_TOOL_NAMES - found
        assert not missing, (
            f"CrewAI adapter is missing tool classes for: {sorted(missing)}. "
            f"Add tool classes in _register_tools()."
        )


class TestPydanticAIToolDrift:
    """PydanticAI adapter (adapters/pydantic_ai.py)."""

    _FILE = _SRC_ROOT / "adapters" / "pydantic_ai.py"

    def test_all_tools_registered(self):
        """Every tool in TOOL_MODELS has a PydanticAI tool function."""
        source = self._FILE.read_text()
        found = _extract_tool_names(source)
        missing = ALL_TOOL_NAMES - found
        assert not missing, (
            f"PydanticAI adapter is missing tool functions for: {sorted(missing)}. "
            f"Add tool registrations in _register_tools()."
        )


class TestParlantToolDrift:
    """Parlant integration (integrations/parlant/tools.py) — chat tools only."""

    _FILE = _SRC_ROOT / "integrations" / "parlant" / "tools.py"

    def test_all_chat_tools_registered(self):
        """Every chat tool has a Parlant tool function."""
        source = self._FILE.read_text()
        found = _extract_tool_names(source)
        missing = CHAT_TOOL_NAMES - found
        assert not missing, (
            f"Parlant integration is missing tool functions for: {sorted(missing)}. "
            f"Add tool implementations in create_parlant_tools()."
        )


# ---------------------------------------------------------------------------
# Cross-cutting sanity checks
# ---------------------------------------------------------------------------


class TestToolRegistryConsistency:
    """Verify the derived sets are consistent with TOOL_MODELS."""

    def test_all_equals_base_plus_memory(self):
        assert ALL_TOOL_NAMES == BASE_TOOL_NAMES | MEMORY_TOOL_NAMES

    def test_base_equals_chat_plus_contact(self):
        assert BASE_TOOL_NAMES == CHAT_TOOL_NAMES | CONTACT_TOOL_NAMES

    def test_no_overlap_chat_contact(self):
        assert not (CHAT_TOOL_NAMES & CONTACT_TOOL_NAMES)

    def test_no_overlap_base_memory(self):
        assert not (BASE_TOOL_NAMES & MEMORY_TOOL_NAMES)

    def test_memory_tools_count(self):
        assert len(MEMORY_TOOL_NAMES) == 5

    def test_contact_tools_count(self):
        assert len(CONTACT_TOOL_NAMES) == 5

    def test_chat_tools_count(self):
        assert len(CHAT_TOOL_NAMES) == 7

    def test_all_tools_count(self):
        assert len(ALL_TOOL_NAMES) == 17
