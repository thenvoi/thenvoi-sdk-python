"""Adapter configuration registry for parameterized conformance tests.

Each AdapterConfig describes a framework adapter's properties, default values,
custom initialization kwargs, and factory function so that conformance tests can
run identical logic across all registered adapters.
"""

from __future__ import annotations

import functools
import inspect
from dataclasses import dataclass, field
from typing import Any, Callable
from unittest.mock import MagicMock

__all__ = ["AdapterConfig", "ADAPTER_CONFIGS"]

# Populated lazily via __getattr__ to avoid top-level adapter imports.
ADAPTER_CONFIGS: list[AdapterConfig]


_MISSING = object()


def _default_from_init(cls: type, param: str, fallback: Any = _MISSING) -> Any:
    """Extract the default value of *param* from *cls.__init__* signature.

    Keeps test configs in sync with adapter source automatically — no need
    to hard-code model strings or other defaults here.

    Relies on ``inspect.signature(cls.__init__)``. It does not work for
    classes that use ``__init_subclass__``, custom metaclasses, or
    constructor patterns that hide defaults (e.g. attrs, Pydantic). If an
    adapter switches to such a pattern, expected_initial_values must be
    updated manually or a fallback passed.

    If the parameter is not found (e.g. the adapter accepts ``**kwargs``
    and forwards it to an underlying client), *fallback* is returned when
    provided.  Without a fallback, ``ValueError`` is raised.
    """
    sig = inspect.signature(cls.__init__)
    p = sig.parameters.get(param)
    if p is None or p.default is inspect.Parameter.empty:
        if fallback is not _MISSING:
            return fallback
        raise ValueError(
            f"{cls.__name__}.__init__ has no default for {param!r}. "
            f"If the adapter uses **kwargs, pass an explicit fallback value."
        )
    return p.default


@dataclass(frozen=True)
class AdapterConfig:
    """Describes a framework adapter for parameterized testing."""

    # Identity
    framework_id: str
    display_name: str

    # Factory: (**kwargs) -> adapter instance (handles mocking internally)
    adapter_factory: Callable[..., Any]

    # {attr_name: expected_value} verified by test_default_initialization.
    # For most adapters these are true defaults from __init__; for PydanticAI
    # ``model`` is a required kwarg injected by the factory (not a real default).
    expected_initial_values: dict[str, Any] = field(default_factory=dict)

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
# (e.g. on_message, _invoke_crew).  The class is imported with mocked crewai
# dependencies (CrewAIAgent, LLM, BaseTool, nest_asyncio are all MagicMock
# objects).  These mock references persist in the module's global scope even
# after sys.modules is cleaned up, so any runtime method call would silently
# operate on MagicMock objects and produce meaningless results.
#
# Conformance instances are ONLY safe for inspecting primitive attributes
# (model, role, etc.).  For runtime/method tests, use the monkeypatch-based
# fixtures in tests/adapters/test_crewai_adapter.py.


class _MockBaseTool:
    """Minimal stand-in for ``crewai.tools.BaseTool`` used only at import time.

    The CrewAI adapter module does ``from crewai.tools import BaseTool`` and
    later subclasses it.  This class provides just enough for the import to
    succeed.  It is never instantiated in conformance tests — only the
    *class object* is referenced as a base class.  Class-level annotations
    (not instance attrs) are intentional for this reason.
    """

    name: str = ""
    description: str = ""

    def __init__(self):
        pass


def _get_crewai_adapter_cls() -> type:
    """Import CrewAIAdapter with mocked crewai dependencies.

    Uses ``patch.dict(sys.modules, ...)`` to temporarily inject mock
    crewai/nest_asyncio modules, then imports the real adapter module
    via ``importlib.import_module``.  ``patch.dict`` atomically restores
    ``sys.modules`` on exit (even on exception).

    The adapter module entry is removed from ``sys.modules`` after import
    so subsequent (non-conformance) imports are not polluted by mocks.

    **Mock reference lifetime:** The returned class's module globals still
    hold references to the mock ``CrewAIAgent``, ``LLM``, ``BaseTool``,
    and ``nest_asyncio`` objects — those bindings were created at import
    time and survive ``patch.dict`` cleanup.  This is intentional: the
    class is only used for **primitive attribute inspection** (model,
    role, etc.) in conformance tests.  Any call to a runtime method
    (``on_message``, ``_invoke_crew``, etc.) would hit MagicMock objects
    and produce nonsense results.

    This function is NOT cached itself — the result is cached in
    ``_build_adapter_configs()`` (which is ``@functools.cache``'d).
    This avoids permanently binding to whichever mocked module imports
    first if test collection order changes.  For runtime tests that
    invoke CrewAI methods, use the monkeypatch-based fixtures in
    ``tests/adapters/test_crewai_adapter.py``.
    """
    import importlib
    import sys
    from unittest.mock import patch

    adapter_module_name = "thenvoi.adapters.crewai"
    mock_dep_names = ("crewai", "crewai.tools", "nest_asyncio")

    # Pre-flight: clear any stale adapter module AND mock dependency modules
    # that may linger from a prior framework-specific test run.  This ensures
    # a clean import regardless of test collection order.
    sys.modules.pop(adapter_module_name, None)
    for dep in mock_dep_names:
        if dep in sys.modules and isinstance(sys.modules[dep], MagicMock):
            sys.modules.pop(dep)

    # Build mock crewai modules matching the adapter's top-level imports:
    #   from crewai import Agent as CrewAIAgent, LLM
    #   from crewai.tools import BaseTool
    #   import nest_asyncio
    mock_crewai = MagicMock()
    mock_crewai.Agent = MagicMock()
    mock_crewai.LLM = MagicMock()

    mock_crewai_tools = MagicMock()
    mock_crewai_tools.BaseTool = _MockBaseTool

    mock_entries = {
        "crewai": mock_crewai,
        "crewai.tools": mock_crewai_tools,
        "nest_asyncio": MagicMock(),
    }

    with patch.dict(sys.modules, mock_entries):
        module = importlib.import_module(adapter_module_name)
        cls = module.CrewAIAdapter

    # Clean up so non-conformance imports are not polluted.
    sys.modules.pop(adapter_module_name, None)

    # Verify mock entries were cleaned from sys.modules by patch.dict.
    # If any leaked, subsequent real crewai imports would silently get
    # MagicMock objects instead of the real package.
    for mock_key in mock_entries:
        assert mock_key not in sys.modules or not isinstance(
            sys.modules[mock_key], MagicMock
        ), (
            f"Mock module {mock_key!r} leaked into sys.modules after "
            f"patch.dict cleanup — this would pollute real crewai imports"
        )

    return cls


async def _crewai_conformance_guard(*_args: Any, **_kw: Any) -> None:
    raise RuntimeError(
        "This CrewAI adapter instance was created by the conformance test "
        "factory with mocked dependencies.  Calling on_message or "
        "_invoke_crew is not safe because CrewAIAgent, LLM, and BaseTool "
        "are MagicMock objects.  Use the monkeypatch-based fixtures in "
        "tests/adapters/test_crewai_adapter.py for runtime tests instead."
    )


def _crewai_factory(**kw: Any) -> Any:
    cls = _get_crewai_adapter_cls()
    instance = cls(**kw)
    # Guard the runtime methods that use mocked CrewAI dependencies.
    # on_message and _invoke_crew invoke the model and would silently
    # operate on MagicMock objects without this guard.
    # Note: on_started and on_cleanup are safe (they only set attributes
    # or clean up dicts) so they are NOT guarded — conformance tests
    # call them directly.
    for method_name in ("on_message", "_invoke_crew"):
        if hasattr(instance, method_name):
            setattr(instance, method_name, _crewai_conformance_guard)
    return instance


def _claude_sdk_factory(**kw: Any) -> Any:
    from thenvoi.adapters.claude_sdk import ClaudeSDKAdapter

    return ClaudeSDKAdapter(**kw)


def _pydantic_ai_factory(**kw: Any) -> Any:
    from thenvoi.adapters.pydantic_ai import PydanticAIAdapter

    if "model" not in kw:
        kw["model"] = _PYDANTIC_AI_INJECTED_MODEL
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
# Registry  (built lazily to avoid top-level adapter imports)
# ---------------------------------------------------------------------------

# PydanticAI requires model as a mandatory kwarg — the factory injects this
# value when not provided.  It is NOT extracted via _default_from_init because
# `model` has no default in the adapter's __init__.
_PYDANTIC_AI_INJECTED_MODEL = "openai:gpt-4o"


@functools.lru_cache(maxsize=1)
def _build_adapter_configs() -> list[AdapterConfig]:
    """Build configs lazily so adapter imports (and _default_from_init) happen
    only when the conformance tests actually need them.

    Uses ``lru_cache(maxsize=1)`` instead of ``cache`` so that
    ``.cache_clear()`` is available — useful if a test teardown needs to
    discard the cached CrewAI adapter class (whose module globals hold
    references to mocked ``crewai`` dependencies).
    """
    from thenvoi.adapters.anthropic import AnthropicAdapter
    from thenvoi.adapters.claude_sdk import ClaudeSDKAdapter
    from thenvoi.adapters.langgraph import LangGraphAdapter
    from thenvoi.adapters.parlant import ParlantAdapter
    from thenvoi.adapters.pydantic_ai import PydanticAIAdapter

    crewai_cls = _get_crewai_adapter_cls()

    return [
        AdapterConfig(
            framework_id="anthropic",
            display_name="Anthropic",
            adapter_factory=_anthropic_factory,
            expected_initial_values={
                "model": _default_from_init(AnthropicAdapter, "model"),
                "max_tokens": _default_from_init(AnthropicAdapter, "max_tokens"),
                "enable_execution_reporting": _default_from_init(
                    AnthropicAdapter, "enable_execution_reporting"
                ),
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
            expected_initial_values={
                "prompt_template": _default_from_init(
                    LangGraphAdapter, "prompt_template"
                ),
                "custom_section": _default_from_init(
                    LangGraphAdapter, "custom_section"
                ),
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
            expected_initial_values={
                "model": _default_from_init(crewai_cls, "model"),
                "role": _default_from_init(crewai_cls, "role"),
                "goal": _default_from_init(crewai_cls, "goal"),
                "backstory": _default_from_init(crewai_cls, "backstory"),
                "enable_execution_reporting": _default_from_init(
                    crewai_cls, "enable_execution_reporting"
                ),
                "verbose": _default_from_init(crewai_cls, "verbose"),
                "max_iter": _default_from_init(crewai_cls, "max_iter"),
                "allow_delegation": _default_from_init(crewai_cls, "allow_delegation"),
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
            expected_initial_values={
                "model": _default_from_init(ClaudeSDKAdapter, "model"),
                "custom_section": _default_from_init(
                    ClaudeSDKAdapter, "custom_section"
                ),
                "max_thinking_tokens": _default_from_init(
                    ClaudeSDKAdapter, "max_thinking_tokens"
                ),
                "permission_mode": _default_from_init(
                    ClaudeSDKAdapter, "permission_mode"
                ),
                "enable_execution_reporting": _default_from_init(
                    ClaudeSDKAdapter, "enable_execution_reporting"
                ),
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
            skip_on_started_conformance=True,  # on_started creates real MCP server + ClaudeSessionManager; tested in test_claude_sdk_adapter
        ),
        AdapterConfig(
            framework_id="pydantic_ai",
            display_name="PydanticAI",
            adapter_factory=_pydantic_ai_factory,
            expected_initial_values={
                # NOTE: injected, not a default — model is a required kwarg
                # (no default); the factory injects _PYDANTIC_AI_INJECTED_MODEL
                # so the conformance test verifies the injected value is stored.
                "model": _PYDANTIC_AI_INJECTED_MODEL,
                "system_prompt": _default_from_init(PydanticAIAdapter, "system_prompt"),
                "custom_section": _default_from_init(
                    PydanticAIAdapter, "custom_section"
                ),
                "enable_execution_reporting": _default_from_init(
                    PydanticAIAdapter, "enable_execution_reporting"
                ),
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
            expected_initial_values={
                "system_prompt": _default_from_init(ParlantAdapter, "system_prompt"),
                "custom_section": _default_from_init(ParlantAdapter, "custom_section"),
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


# Public API: consumers import ADAPTER_CONFIGS as a module-level list.
# The property-on-module trick is not worth the complexity; instead, callers
# that need the list call _build_adapter_configs().  For backward compat the
# module attribute is set once on first access.


def __getattr__(name: str) -> Any:
    if name == "ADAPTER_CONFIGS":
        configs = _build_adapter_configs()
        globals()["ADAPTER_CONFIGS"] = configs
        return configs
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
