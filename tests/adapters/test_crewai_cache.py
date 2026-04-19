"""Regression tests for CrewAI tool caching behavior (INT-340).

These tests intentionally bypass the mocked `crewai.tools.BaseTool` fixture in
tests/adapters/test_crewai_adapter.py and exercise the **real** CrewAI SDK.
CrewAI's CacheHandler keys cached tool outputs by (tool_name, input_string).
Thenvoi tools are room-scoped via a ContextVar, not via arguments, so caching
would silently serve room A's result back in room B. Every Thenvoi-supplied
CrewAI tool (including the dynamic CustomCrewAITool) must opt out by returning
False from cache_function.
"""

from __future__ import annotations

import importlib
import sys

import pytest
from pydantic import BaseModel, Field

from thenvoi.core.types import AdapterFeatures, Capability


class CustomToolInput(BaseModel):
    """A custom tool for exercising the CustomCrewAITool cache override."""

    query: str = Field(..., description="Query string")


def _handler(validated: CustomToolInput) -> str:
    return f"got {validated.query}"


@pytest.fixture
def crewai_adapter_cls():
    """Import CrewAIAdapter against the real CrewAI SDK.

    tests/adapters/test_crewai_adapter.py installs a mocked `crewai.tools`
    in sys.modules and pops `thenvoi.adapters.crewai` during teardown — which
    leaves Pydantic unable to resolve forward refs on any previously-imported
    adapter class. Re-importing here guarantees we get a module bound to the
    real `crewai.tools.BaseTool`, not the mock.
    """
    sys.modules.pop("thenvoi.adapters.crewai", None)
    module = importlib.import_module("thenvoi.adapters.crewai")
    return module.CrewAIAdapter


@pytest.fixture
def adapter_with_memory_and_custom_tool(crewai_adapter_cls):
    """Adapter with all tools enabled: platform + memory + one custom tool."""
    return crewai_adapter_cls(
        model="gpt-4o-mini",
        additional_tools=[(CustomToolInput, _handler)],
        features=AdapterFeatures(
            emit=frozenset(),
            capabilities=frozenset({Capability.MEMORY}),
        ),
    )


class TestToolCaching:
    """CrewAI tools must never be cached — their context is a ContextVar."""

    def test_every_platform_and_memory_tool_disables_cache(
        self, adapter_with_memory_and_custom_tool
    ) -> None:
        """cache_function must return False for every Thenvoi tool."""
        tools = adapter_with_memory_and_custom_tool._create_crewai_tools()

        # Sanity check: full set = 12 platform + 5 memory + 1 custom = 18
        tool_names = [t.name for t in tools]
        assert len(tools) == 18, (
            f"Expected 18 tools (12 platform + 5 memory + 1 custom), "
            f"got {len(tools)}: {tool_names}"
        )

        for tool in tools:
            # CrewAI calls cache_function(calling.arguments, result); the exact
            # args don't matter — our override ignores them.
            assert tool.cache_function({"any": "input"}, "any output") is False, (
                f"Tool {tool.name!r} did NOT disable caching — "
                "CrewAI will leak results across rooms."
            )

    def test_custom_tool_disables_cache(
        self, adapter_with_memory_and_custom_tool
    ) -> None:
        """The dynamic CustomCrewAITool factory must also opt out of caching."""
        tools = adapter_with_memory_and_custom_tool._create_crewai_tools()

        platform_names = {
            "thenvoi_send_message",
            "thenvoi_send_event",
            "thenvoi_add_participant",
            "thenvoi_remove_participant",
            "thenvoi_get_participants",
            "thenvoi_lookup_peers",
            "thenvoi_create_chatroom",
            "thenvoi_list_contacts",
            "thenvoi_add_contact",
            "thenvoi_remove_contact",
            "thenvoi_list_contact_requests",
            "thenvoi_respond_contact_request",
            "thenvoi_list_memories",
            "thenvoi_store_memory",
            "thenvoi_get_memory",
            "thenvoi_supersede_memory",
            "thenvoi_archive_memory",
        }
        custom_tools = [t for t in tools if t.name not in platform_names]
        assert len(custom_tools) == 1, (
            f"Expected exactly one CustomCrewAITool, got: {[t.name for t in custom_tools]}"
        )
        assert custom_tools[0].cache_function({"query": "x"}, "any output") is False

    def test_cache_override_is_not_the_crewai_default(
        self, adapter_with_memory_and_custom_tool
    ) -> None:
        """Guard against regression to CrewAI's default (always-True) cache_function."""
        from crewai.tools import BaseTool

        default_cf = BaseTool.model_fields["cache_function"].default
        assert default_cf(None, None) is True, (
            "CrewAI's BaseTool default cache_function should return True — "
            "if this fails, CrewAI's upstream behavior changed and this test "
            "suite's assumptions need re-verification."
        )

        tools = adapter_with_memory_and_custom_tool._create_crewai_tools()
        for tool in tools:
            assert tool.cache_function is not default_cf, (
                f"Tool {tool.name!r} still references CrewAI's default "
                "cache_function — override did not land."
            )
