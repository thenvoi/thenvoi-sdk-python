"""CrewAI adapter-specific tests.

Tests for CrewAI adapter-specific behavior that isn't covered by conformance tests:
- system_prompt deprecation warning
- role/goal/backstory configuration
- Platform instructions in backstory
- Agent name as default role
"""

from __future__ import annotations

import sys
import warnings

import pytest

from tests.framework_configs.adapters import setup_crewai_mocks


@pytest.fixture
def crewai_mocks(monkeypatch):
    """Set up CrewAI module mocks."""
    mock_crewai_module = setup_crewai_mocks(monkeypatch)

    yield mock_crewai_module

    # Cleanup
    sys.modules.pop("thenvoi.adapters.crewai", None)


class TestSystemPromptDeprecation:
    """Tests for system_prompt parameter deprecation."""

    def test_system_prompt_emits_deprecation_warning(self, crewai_mocks):
        """system_prompt parameter should emit DeprecationWarning."""
        import importlib

        CrewAIAdapter = importlib.import_module("thenvoi.adapters.crewai").CrewAIAdapter

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            CrewAIAdapter(system_prompt="Custom prompt")

            # Check that a deprecation warning was issued
            deprecation_warnings = [
                warning
                for warning in w
                if issubclass(warning.category, DeprecationWarning)
            ]
            assert len(deprecation_warnings) >= 1

            warning_msg = str(deprecation_warnings[0].message)
            assert "system_prompt" in warning_msg
            assert "deprecated" in warning_msg.lower()

    def test_system_prompt_does_not_override_backstory(self, crewai_mocks):
        """system_prompt should not override explicit backstory."""
        import importlib

        CrewAIAdapter = importlib.import_module("thenvoi.adapters.crewai").CrewAIAdapter

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adapter = CrewAIAdapter(
                backstory="My custom backstory",
                system_prompt="This should not be used",
            )

        assert adapter.backstory == "My custom backstory"

    def test_system_prompt_used_as_backstory_when_backstory_not_provided(
        self, crewai_mocks
    ):
        """system_prompt should be used as backstory when backstory not provided."""
        import importlib

        CrewAIAdapter = importlib.import_module("thenvoi.adapters.crewai").CrewAIAdapter

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adapter = CrewAIAdapter(system_prompt="Fallback backstory")

        assert adapter.backstory == "Fallback backstory"


class TestRoleGoalBackstory:
    """Tests for role/goal/backstory configuration."""

    @pytest.mark.asyncio
    async def test_uses_custom_role_goal_backstory(self, crewai_mocks):
        """Should use custom role, goal, backstory when provided."""
        import importlib

        CrewAIAdapter = importlib.import_module("thenvoi.adapters.crewai").CrewAIAdapter

        adapter = CrewAIAdapter(
            role="Research Assistant",
            goal="Find and analyze information",
            backstory="Expert researcher with 10 years experience",
        )

        crewai_mocks.Agent.reset_mock()
        await adapter.on_started(agent_name="TestBot", agent_description="A test bot")

        # Check that Agent was called with custom values
        call_kwargs = crewai_mocks.Agent.call_args.kwargs

        assert call_kwargs["role"] == "Research Assistant"
        assert call_kwargs["goal"] == "Find and analyze information"
        assert "Expert researcher" in call_kwargs["backstory"]

    @pytest.mark.asyncio
    async def test_uses_agent_name_as_default_role(self, crewai_mocks):
        """Should use agent_name as role when role not provided."""
        import importlib

        CrewAIAdapter = importlib.import_module("thenvoi.adapters.crewai").CrewAIAdapter

        adapter = CrewAIAdapter()  # No role specified

        crewai_mocks.Agent.reset_mock()
        await adapter.on_started(
            agent_name="WeatherBot", agent_description="Gets weather"
        )

        call_kwargs = crewai_mocks.Agent.call_args.kwargs
        assert call_kwargs["role"] == "WeatherBot"

    @pytest.mark.asyncio
    async def test_uses_agent_description_as_default_goal(self, crewai_mocks):
        """Should use agent_description as goal when goal not provided."""
        import importlib

        CrewAIAdapter = importlib.import_module("thenvoi.adapters.crewai").CrewAIAdapter

        adapter = CrewAIAdapter()  # No goal specified

        crewai_mocks.Agent.reset_mock()
        await adapter.on_started(
            agent_name="TestBot", agent_description="Help users with tasks"
        )

        call_kwargs = crewai_mocks.Agent.call_args.kwargs
        assert call_kwargs["goal"] == "Help users with tasks"


class TestPlatformInstructions:
    """Tests for platform instructions in backstory."""

    @pytest.mark.asyncio
    async def test_includes_platform_instructions_in_backstory(self, crewai_mocks):
        """Should include PLATFORM_INSTRUCTIONS in backstory."""
        import importlib

        CrewAIAdapter = importlib.import_module("thenvoi.adapters.crewai").CrewAIAdapter

        adapter = CrewAIAdapter(backstory="Custom backstory")

        crewai_mocks.Agent.reset_mock()
        await adapter.on_started(agent_name="TestBot", agent_description="A test bot")

        call_kwargs = crewai_mocks.Agent.call_args.kwargs
        backstory = call_kwargs["backstory"]

        # Check for key platform instruction content
        assert "thenvoi_send_message" in backstory
        assert "thenvoi_lookup_peers" in backstory
        assert "Multi-participant chat" in backstory

    @pytest.mark.asyncio
    async def test_includes_custom_section_in_backstory(self, crewai_mocks):
        """Should include custom_section in backstory."""
        import importlib

        CrewAIAdapter = importlib.import_module("thenvoi.adapters.crewai").CrewAIAdapter

        adapter = CrewAIAdapter(custom_section="Always be helpful and friendly.")

        crewai_mocks.Agent.reset_mock()
        await adapter.on_started(agent_name="TestBot", agent_description="A test bot")

        call_kwargs = crewai_mocks.Agent.call_args.kwargs
        backstory = call_kwargs["backstory"]

        assert "Always be helpful and friendly." in backstory


class TestCrewAIAgentCreation:
    """Tests for CrewAI agent creation."""

    @pytest.mark.asyncio
    async def test_creates_crewai_agent_with_tools(self, crewai_mocks):
        """Should create CrewAI agent with platform tools."""
        import importlib

        CrewAIAdapter = importlib.import_module("thenvoi.adapters.crewai").CrewAIAdapter

        adapter = CrewAIAdapter()

        crewai_mocks.Agent.reset_mock()
        await adapter.on_started(agent_name="TestBot", agent_description="A test bot")

        # Verify Agent was created
        assert crewai_mocks.Agent.called

        call_kwargs = crewai_mocks.Agent.call_args.kwargs

        # Check tools were passed
        assert "tools" in call_kwargs
        tools = call_kwargs["tools"]
        assert len(tools) > 0

        # Check tool names
        tool_names = [t.name for t in tools]
        assert "thenvoi_send_message" in tool_names
        assert "thenvoi_lookup_peers" in tool_names

    @pytest.mark.asyncio
    async def test_creates_llm_with_model(self, crewai_mocks):
        """Should create LLM with specified model."""
        import importlib

        CrewAIAdapter = importlib.import_module("thenvoi.adapters.crewai").CrewAIAdapter

        adapter = CrewAIAdapter(model="gpt-4o-mini")

        await adapter.on_started(agent_name="TestBot", agent_description="A test bot")

        # Check LLM was created with correct model
        crewai_mocks.LLM.assert_called_with(model="gpt-4o-mini")
