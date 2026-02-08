"""LangGraph adapter-specific tests.

Tests for LangGraph adapter-specific behavior that isn't covered by conformance tests:
- Required parameter validation (llm or graph_factory/graph)
- Simple vs advanced pattern initialization
- Graph factory creation
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from thenvoi.adapters.langgraph import LangGraphAdapter


class TestRequiredParameterValidation:
    """Tests for required parameter validation."""

    def test_raises_without_llm_or_graph(self):
        """Should raise ValueError when neither llm nor graph_factory/graph provided."""
        with pytest.raises(ValueError) as exc_info:
            LangGraphAdapter()

        error_msg = str(exc_info.value)
        assert "Must provide either llm" in error_msg
        assert "graph_factory" in error_msg

    def test_accepts_llm_parameter(self):
        """Should accept llm parameter without error."""
        mock_llm = MagicMock()
        mock_checkpointer = MagicMock()

        adapter = LangGraphAdapter(llm=mock_llm, checkpointer=mock_checkpointer)

        assert adapter is not None
        assert adapter.graph_factory is not None

    def test_accepts_graph_factory_parameter(self):
        """Should accept graph_factory parameter without error."""

        def custom_factory(tools):
            return MagicMock()

        adapter = LangGraphAdapter(graph_factory=custom_factory)

        assert adapter is not None
        assert adapter.graph_factory == custom_factory

    def test_accepts_static_graph_parameter(self):
        """Should accept graph parameter without error."""
        mock_graph = MagicMock()

        adapter = LangGraphAdapter(graph=mock_graph)

        assert adapter is not None
        assert adapter._static_graph == mock_graph


class TestSimplePattern:
    """Tests for simple pattern initialization (llm + checkpointer)."""

    def test_simple_pattern_creates_graph_factory(self):
        """Simple pattern should create a graph_factory from llm."""
        mock_llm = MagicMock()
        mock_checkpointer = MagicMock()

        adapter = LangGraphAdapter(llm=mock_llm, checkpointer=mock_checkpointer)

        assert adapter.graph_factory is not None
        assert callable(adapter.graph_factory)

    def test_simple_pattern_with_custom_section(self):
        """Simple pattern should accept custom_section."""
        mock_llm = MagicMock()
        mock_checkpointer = MagicMock()

        adapter = LangGraphAdapter(
            llm=mock_llm,
            checkpointer=mock_checkpointer,
            custom_section="Be helpful and concise.",
        )

        assert adapter.custom_section == "Be helpful and concise."

    def test_simple_pattern_with_additional_tools(self):
        """Simple pattern should bake additional_tools into graph_factory."""
        mock_llm = MagicMock()
        mock_checkpointer = MagicMock()

        def custom_tool():
            pass

        adapter = LangGraphAdapter(
            llm=mock_llm,
            checkpointer=mock_checkpointer,
            additional_tools=[custom_tool],
        )

        # additional_tools should be cleared (baked into factory)
        assert adapter.additional_tools == []


class TestAdvancedPattern:
    """Tests for advanced pattern initialization (graph_factory or graph)."""

    def test_advanced_pattern_with_graph_factory(self):
        """Advanced pattern should use provided graph_factory."""

        def my_factory(tools):
            return MagicMock()

        adapter = LangGraphAdapter(graph_factory=my_factory)

        assert adapter.graph_factory == my_factory
        assert adapter._static_graph is None

    def test_advanced_pattern_with_static_graph(self):
        """Advanced pattern should use provided static graph."""
        mock_graph = MagicMock()

        adapter = LangGraphAdapter(graph=mock_graph)

        assert adapter._static_graph == mock_graph
        assert adapter.graph_factory is None

    def test_advanced_pattern_preserves_additional_tools(self):
        """Advanced pattern should preserve additional_tools."""

        def custom_tool():
            pass

        adapter = LangGraphAdapter(
            graph_factory=lambda tools: MagicMock(),
            additional_tools=[custom_tool],
        )

        # additional_tools should NOT be cleared in advanced pattern
        assert len(adapter.additional_tools) == 1


class TestPromptTemplate:
    """Tests for prompt template configuration."""

    def test_default_prompt_template(self):
        """Should default to 'default' prompt template."""
        mock_llm = MagicMock()

        adapter = LangGraphAdapter(llm=mock_llm, checkpointer=MagicMock())

        assert adapter.prompt_template == "default"

    def test_custom_prompt_template(self):
        """Should accept custom prompt template."""
        mock_llm = MagicMock()

        adapter = LangGraphAdapter(
            llm=mock_llm,
            checkpointer=MagicMock(),
            prompt_template="minimal",
        )

        assert adapter.prompt_template == "minimal"


class TestOnStarted:
    """Tests for on_started behavior."""

    @pytest.mark.asyncio
    async def test_renders_system_prompt(self):
        """Should render system prompt with agent info."""
        mock_llm = MagicMock()

        adapter = LangGraphAdapter(llm=mock_llm, checkpointer=MagicMock())
        await adapter.on_started(
            agent_name="TestBot",
            agent_description="A helpful test bot",
        )

        assert adapter._system_prompt is not None
        assert "TestBot" in adapter._system_prompt

    @pytest.mark.asyncio
    async def test_includes_custom_section_in_prompt(self):
        """Should include custom_section in rendered prompt."""
        mock_llm = MagicMock()

        adapter = LangGraphAdapter(
            llm=mock_llm,
            checkpointer=MagicMock(),
            custom_section="Always respond in haiku format.",
        )
        await adapter.on_started(
            agent_name="TestBot",
            agent_description="A test bot",
        )

        assert "Always respond in haiku format." in adapter._system_prompt
