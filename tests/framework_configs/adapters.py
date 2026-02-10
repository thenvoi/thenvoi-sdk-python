"""Adapter configuration registry for parameterized conformance tests.

Each AdapterConfig describes a framework adapter's properties, default values,
custom initialization kwargs, and factory function so that conformance tests can
run identical logic across all registered adapters.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Any, Callable
from unittest.mock import MagicMock

__all__ = ["AdapterConfig", "ADAPTER_CONFIGS"]

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

    Uses ``unittest.mock.patch.dict`` to temporarily inject mock
    dependencies into ``sys.modules`` during ``exec_module``.
    ``patch.dict`` guarantees atomic restore of the original state
    on exit (even on exception), which is safe for parallel test
    collection and avoids the fragility of manual save/restore.

    The adapter module is loaded into an isolated namespace via
    ``importlib.util.spec_from_file_location`` so it does not
    pollute the real ``thenvoi.adapters.crewai`` entry.

    The result is cached via ``@functools.cache`` so subsequent calls
    are cheap.  Conformance tests only inspect primitive attributes
    (model, role, etc.) and never mix instances across import
    boundaries.  For runtime tests that invoke CrewAI methods or need
    isinstance compatibility, use the ``crewai_mocks`` and
    ``CrewAIAdapter`` monkeypatch-based fixtures in
    tests/adapters/test_crewai_adapter.py.

    **Divergence risk (guarded):** Because this function loads the
    adapter module into an isolated namespace, the returned class is
    *not* the same object as ``thenvoi.adapters.crewai.CrewAIAdapter``.
    A signature-equality check at the end of this function compares the
    isolated class's ``__init__`` against the real one and raises
    ``RuntimeError`` on mismatch.  When modifying
    ``CrewAIAdapter.__init__``, also update the ``default_values``,
    ``custom_kwargs``, and ``custom_expected`` in the CrewAI
    ``AdapterConfig`` entry below and re-run conformance tests.
    """

    import importlib.util
    import pathlib
    import sys
    import types
    from unittest.mock import patch

    # Build mock crewai modules the adapter imports at module level.
    # These correspond to the top-level imports in src/thenvoi/adapters/crewai.py:
    #   from crewai import Agent as CrewAIAgent   -> crewai.Agent
    #   from crewai import LLM                    -> crewai.LLM
    #   from crewai.tools import BaseTool          -> crewai.tools.BaseTool
    #   import nest_asyncio                        -> nest_asyncio
    # If crewai.py adds new module-level imports, add matching mocks here.
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

    # Load the adapter module in an isolated namespace so the real
    # thenvoi.adapters.crewai entry is never overwritten.
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

    # Use patch.dict to inject mock dependencies into sys.modules for the
    # duration of exec_module.  patch.dict atomically restores the original
    # state on exit, even if exec_module raises.
    mock_entries: dict[str, types.ModuleType] = {
        "crewai": mock_crewai_module,
        "crewai.tools": mock_crewai_tools_module,
        "nest_asyncio": mock_nest_asyncio,
    }

    # Ensure the parent package is importable so relative imports work.
    # Include it in patch.dict so it is cleaned up if it was not already
    # present (avoids permanently injecting a fake module into sys.modules).
    adapters_pkg_name = "thenvoi.adapters"
    if adapters_pkg_name not in sys.modules:
        adapters_pkg = types.ModuleType(adapters_pkg_name)
        adapters_pkg.__path__ = [str(adapter_path.parent)]
        mock_entries[adapters_pkg_name] = adapters_pkg

    with patch.dict(sys.modules, mock_entries):
        spec.loader.exec_module(isolated_module)

    isolated_cls = isolated_module.CrewAIAdapter

    # --- Signature drift guard ---
    # Compare __init__ signatures so that changes to the real adapter
    # (new required params, renamed kwargs, removed defaults) are caught
    # immediately rather than producing silent conformance-test drift.
    # Wrapped in try/except so the guard is skipped when the real crewai
    # package is not installed (the isolated import above uses mocks, but
    # the real import below requires the actual package).
    import inspect

    try:
        import crewai as _crewai_probe  # noqa: F401 — probe whether crewai is installed
    except ImportError:
        # crewai package not installed — skip drift check (isolated import is sufficient)
        pass
    else:
        # crewai IS installed, so a failure importing the real adapter is a real error
        # that should propagate (not be silently swallowed).
        from thenvoi.adapters.crewai import CrewAIAdapter as _RealCrewAIAdapter

        real_sig = inspect.signature(_RealCrewAIAdapter.__init__)
        isolated_sig = inspect.signature(isolated_cls.__init__)
        if real_sig != isolated_sig:
            raise RuntimeError(
                "CrewAIAdapter __init__ signature drift detected!\n"
                f"  Real:     {real_sig}\n"
                f"  Isolated: {isolated_sig}\n"
                "Update the CrewAI AdapterConfig (default_values, custom_kwargs, "
                "custom_expected) and re-run conformance tests."
            )

    return isolated_cls


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
        has_custom_tools_attr=True,
        custom_tools_attr="additional_tools",
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
