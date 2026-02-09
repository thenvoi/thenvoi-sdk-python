"""Adapter configuration registry for parameterized conformance tests.

Each AdapterConfig describes a framework adapter's properties, default values,
custom initialization kwargs, and factory function so that conformance tests can
run identical logic across all six adapters.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Any, Callable
from unittest.mock import MagicMock

# Default model strings — keep in sync with the adapter __init__ defaults.
# Centralised here so a model bump requires only one change in the test config.
# Source: src/thenvoi/adapters/<framework>.py  __init__(model=...)
_ANTHROPIC_DEFAULT_MODEL = "claude-sonnet-4-5-20250929"
_CLAUDE_SDK_DEFAULT_MODEL = "claude-sonnet-4-5-20250929"
_CREWAI_DEFAULT_MODEL = "gpt-4o"
_PYDANTIC_AI_DEFAULT_MODEL = (
    "openai:gpt-4o"  # factory-injected (adapter requires model)
)


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


@functools.cache
def _get_crewai_adapter_cls() -> type:
    """Import CrewAIAdapter once with mocked crewai dependencies.

    Uses ``importlib.util.spec_from_file_location`` to load the adapter
    module into an isolated namespace.  ``sys.modules`` is temporarily
    mutated to inject mock dependencies during ``exec_module`` and
    restored in a ``finally`` block immediately after.

    The result is cached via ``@functools.cache`` so subsequent calls
    are cheap and thread-safe.  Conformance tests only inspect primitive
    attributes (model, role, etc.) and never mix instances across import
    boundaries.  For runtime tests that invoke CrewAI methods or need
    isinstance compatibility, use the ``crewai_mocks`` and
    ``CrewAIAdapter`` monkeypatch-based fixtures in
    tests/adapters/test_crewai_adapter.py.
    """

    import importlib.util
    import pathlib
    import sys
    import types

    # Build mock crewai modules the adapter imports at module level.
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

    # Load the adapter module in an isolated namespace so sys.modules is
    # never mutated.  The loader's exec_module will use the module's own
    # __dict__ for its top-level imports, which we pre-populate with mocks.
    adapter_path = (
        pathlib.Path(__file__).resolve().parents[2]
        / "src"
        / "thenvoi"
        / "adapters"
        / "crewai.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_conformance_crewai_adapter", adapter_path
    )
    isolated_module = importlib.util.module_from_spec(spec)

    # Inject mocked dependencies into the isolated module's namespace
    # *before* exec_module runs its top-level imports.
    saved = {}
    mock_entries = {
        "crewai": mock_crewai_module,
        "crewai.tools": mock_crewai_tools_module,
        "nest_asyncio": mock_nest_asyncio,
    }
    for name, mock in mock_entries.items():
        saved[name] = sys.modules.get(name)
        sys.modules[name] = mock

    # Also ensure the parent package is importable so relative imports work.
    adapters_pkg_name = "thenvoi.adapters"
    if adapters_pkg_name not in sys.modules:
        adapters_pkg = types.ModuleType(adapters_pkg_name)
        adapters_pkg.__path__ = [str(adapter_path.parent)]
        sys.modules[adapters_pkg_name] = adapters_pkg

    try:
        spec.loader.exec_module(isolated_module)
    finally:
        # Restore original sys.modules entries (or remove mocks).
        for name, original in saved.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original

    return isolated_module.CrewAIAdapter


def _crewai_factory(**kw: Any) -> Any:
    cls = _get_crewai_adapter_cls()
    return cls(**kw)


def _claude_sdk_factory(**kw: Any) -> Any:
    from thenvoi.adapters.claude_sdk import ClaudeSDKAdapter

    return ClaudeSDKAdapter(**kw)


def _pydantic_ai_factory(**kw: Any) -> Any:
    from thenvoi.adapters.pydantic_ai import PydanticAIAdapter

    if "model" not in kw:
        kw["model"] = _PYDANTIC_AI_DEFAULT_MODEL
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
            "model": _ANTHROPIC_DEFAULT_MODEL,
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
            "model": _CREWAI_DEFAULT_MODEL,
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
            "model": _CLAUDE_SDK_DEFAULT_MODEL,
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
            "model": _PYDANTIC_AI_DEFAULT_MODEL,
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
