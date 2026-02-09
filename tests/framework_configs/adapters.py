"""Adapter configuration registry for parameterized conformance tests.

Each AdapterConfig describes a framework adapter's properties, default values,
custom initialization kwargs, and factory function so that conformance tests can
run identical logic across all six adapters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable
from unittest.mock import MagicMock


@dataclass(frozen=True)
class AdapterConfig:
    """Describes a framework adapter for parameterized testing."""

    # Identity
    framework_id: str
    display_name: str

    # Factory: (**kwargs) -> adapter instance (handles mocking internally)
    adapter_factory: Callable[..., Any]

    # {attr_name: expected_default} for test_default_initialization
    default_values: dict[str, Any] = field(default_factory=dict)

    # For test_custom_initialization
    custom_kwargs: dict[str, Any] = field(default_factory=dict)
    custom_expected: dict[str, Any] = field(default_factory=dict)

    # Custom tools support
    has_custom_tools_attr: bool = True
    custom_tools_attr: str = "_custom_tools"

    # History converter presence
    has_history_converter: bool = True

    # Skip on_started conformance test when adapter needs live client (e.g. PydanticAI)
    skip_on_started_conformance: bool = False


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


def _anthropic_factory(**kw: Any) -> Any:
    from thenvoi.adapters.anthropic import AnthropicAdapter

    return AnthropicAdapter(**kw)


def _langgraph_factory(**kw: Any) -> Any:
    from thenvoi.adapters.langgraph import LangGraphAdapter

    if "llm" not in kw and "graph_factory" not in kw and "graph" not in kw:
        kw["llm"] = MagicMock()
        kw["checkpointer"] = MagicMock()
    return LangGraphAdapter(**kw)


# WARNING: Conformance-created CrewAI instances must NOT call runtime methods
# (e.g. on_message, _invoke_crew).  They are only safe for inspecting primitive
# attributes (model, role, etc.).  For runtime/method tests, use the
# monkeypatch-based fixtures in tests/adapters/test_crewai_adapter.py.
_crewai_adapter_cls: type | None = None


def _get_crewai_adapter_cls() -> type:
    """Import CrewAIAdapter once with mocked crewai dependencies.

    The class is cached after the first import so subsequent calls do not
    manipulate sys.modules.  This mirrors the monkeypatch-based approach in
    tests/adapters/test_crewai_adapter.py but uses unittest.mock.patch.dict
    since we are outside a pytest fixture context.

    SAFETY: The cached class references a module that is later removed from
    sys.modules (so real imports in other tests get a fresh module).  This
    means isinstance() checks between conformance-created instances and
    instances from the real module will fail.  This is acceptable because
    conformance tests only inspect primitive attributes (model, role, etc.)
    and never mix instances across import boundaries.  For runtime tests
    that invoke CrewAI methods or need isinstance compatibility, use the
    ``crewai_mocks`` and ``CrewAIAdapter`` monkeypatch-based fixtures in
    tests/adapters/test_crewai_adapter.py.
    """
    global _crewai_adapter_cls
    if _crewai_adapter_cls is not None:
        return _crewai_adapter_cls

    import importlib
    import sys
    from unittest.mock import patch

    mock_crewai_module = MagicMock()
    mock_crewai_tools_module = MagicMock()
    mock_nest_asyncio = MagicMock()

    mock_crewai_module.Agent = MagicMock()
    mock_crewai_module.LLM = MagicMock()

    class MockBaseTool:
        name: str = ""
        description: str = ""

        def __init__(self):
            pass

    mock_crewai_tools_module.BaseTool = MockBaseTool

    mock_modules = {
        "crewai": mock_crewai_module,
        "crewai.tools": mock_crewai_tools_module,
        "nest_asyncio": mock_nest_asyncio,
    }

    with patch.dict(sys.modules, mock_modules):
        sys.modules.pop("thenvoi.adapters.crewai", None)
        module = importlib.import_module("thenvoi.adapters.crewai")
        _crewai_adapter_cls = module.CrewAIAdapter

    # patch.dict restores the original crewai/nest_asyncio entries;
    # also clear the adapter module so later real imports aren't stale.
    sys.modules.pop("thenvoi.adapters.crewai", None)

    return _crewai_adapter_cls


def _crewai_factory(**kw: Any) -> Any:
    cls = _get_crewai_adapter_cls()
    return cls(**kw)


def _claude_sdk_factory(**kw: Any) -> Any:
    from thenvoi.adapters.claude_sdk import ClaudeSDKAdapter

    return ClaudeSDKAdapter(**kw)


def _pydantic_ai_factory(**kw: Any) -> Any:
    from thenvoi.adapters.pydantic_ai import PydanticAIAdapter

    if "model" not in kw:
        kw["model"] = "openai:gpt-4o"
    return PydanticAIAdapter(**kw)


def _parlant_factory(**kw: Any) -> Any:
    from thenvoi.adapters.parlant import ParlantAdapter

    if "server" not in kw:
        kw["server"] = MagicMock()
    if "parlant_agent" not in kw:
        mock_agent = MagicMock()
        mock_agent.id = "parlant-agent-123"
        mock_agent.name = "TestBot"
        kw["parlant_agent"] = mock_agent
    return ParlantAdapter(**kw)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ADAPTER_CONFIGS: list[AdapterConfig] = [
    AdapterConfig(
        framework_id="anthropic",
        display_name="Anthropic",
        adapter_factory=_anthropic_factory,
        default_values={
            "model": "claude-sonnet-4-5-20250929",
            "max_tokens": 4096,
            "enable_execution_reporting": False,
        },
        custom_kwargs={
            "model": "claude-opus-4-20250514",
            "max_tokens": 8192,
            "custom_section": "Be helpful.",
            "enable_execution_reporting": True,
        },
        custom_expected={
            "model": "claude-opus-4-20250514",
            "max_tokens": 8192,
            "custom_section": "Be helpful.",
            "enable_execution_reporting": True,
        },
    ),
    AdapterConfig(
        framework_id="langgraph",
        display_name="LangGraph",
        adapter_factory=_langgraph_factory,
        default_values={
            "prompt_template": "default",
            "custom_section": "",
        },
        custom_kwargs={
            "custom_section": "Be helpful.",
        },
        custom_expected={
            "custom_section": "Be helpful.",
        },
        has_custom_tools_attr=False,
        has_history_converter=True,
    ),
    AdapterConfig(
        framework_id="crewai",
        display_name="CrewAI",
        adapter_factory=_crewai_factory,
        default_values={
            "model": "gpt-4o",
            "role": None,
            "goal": None,
            "backstory": None,
            "enable_execution_reporting": False,
            "verbose": False,
            "max_iter": 20,
            "allow_delegation": False,
        },
        custom_kwargs={
            "model": "gpt-4o-mini",
            "role": "Research Analyst",
            "goal": "Find and analyze information",
            "backstory": "Expert researcher",
            "custom_section": "Be thorough.",
            "enable_execution_reporting": True,
            "verbose": True,
            "max_iter": 30,
            "max_rpm": 10,
            "allow_delegation": True,
        },
        custom_expected={
            "model": "gpt-4o-mini",
            "role": "Research Analyst",
            "goal": "Find and analyze information",
            "backstory": "Expert researcher",
            "custom_section": "Be thorough.",
            "enable_execution_reporting": True,
            "verbose": True,
            "max_iter": 30,
            "max_rpm": 10,
            "allow_delegation": True,
        },
    ),
    AdapterConfig(
        framework_id="claude_sdk",
        display_name="ClaudeSDK",
        adapter_factory=_claude_sdk_factory,
        default_values={
            "model": "claude-sonnet-4-5-20250929",
            "custom_section": None,
            "max_thinking_tokens": None,
            "permission_mode": "acceptEdits",
            "enable_execution_reporting": False,
        },
        custom_kwargs={
            "model": "claude-opus-4-20250514",
            "custom_section": "Be helpful.",
            "max_thinking_tokens": 10000,
            "permission_mode": "bypassPermissions",
            "enable_execution_reporting": True,
        },
        custom_expected={
            "model": "claude-opus-4-20250514",
            "custom_section": "Be helpful.",
            "max_thinking_tokens": 10000,
            "permission_mode": "bypassPermissions",
            "enable_execution_reporting": True,
        },
    ),
    AdapterConfig(
        framework_id="pydantic_ai",
        display_name="PydanticAI",
        adapter_factory=_pydantic_ai_factory,
        default_values={
            "model": "openai:gpt-4o",
            "system_prompt": None,
            "custom_section": None,
            "enable_execution_reporting": False,
            "_agent": None,
        },
        custom_kwargs={
            "model": "anthropic:claude-sonnet-4-5-20250929",
            "system_prompt": "You are a helpful bot.",
            "custom_section": "Be concise.",
            "enable_execution_reporting": True,
        },
        custom_expected={
            "model": "anthropic:claude-sonnet-4-5-20250929",
            "system_prompt": "You are a helpful bot.",
            "custom_section": "Be concise.",
            "enable_execution_reporting": True,
        },
        skip_on_started_conformance=True,  # on_started creates real OpenAI client; tested in test_pydantic_ai_adapter
    ),
    AdapterConfig(
        framework_id="parlant",
        display_name="Parlant",
        adapter_factory=_parlant_factory,
        default_values={
            "system_prompt": None,
            "custom_section": None,
        },
        custom_kwargs={
            "system_prompt": "Custom system prompt",
            "custom_section": "Be helpful.",
        },
        custom_expected={
            "system_prompt": "Custom system prompt",
            "custom_section": "Be helpful.",
        },
        has_custom_tools_attr=False,
    ),
]
