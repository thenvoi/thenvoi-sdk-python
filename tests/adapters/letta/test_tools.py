"""Tests for Letta adapter tools module."""

import pytest
from unittest.mock import MagicMock

from thenvoi.adapters.letta.tools import (
    THENVOI_TOOL_NAMES,
    register_thenvoi_tools,
    get_letta_tool_ids,
    CustomToolBuilder,
    _get_letta_tool_schema,
)


class TestThenvoiToolNames:
    """Tests for THENVOI_TOOL_NAMES constant."""

    def test_contains_all_platform_tools(self):
        """Should contain all Thenvoi platform tools exposed to LLM.

        Note: send_event is NOT exposed to LLM - it's used internally by the adapter.
        """
        assert "send_message" in THENVOI_TOOL_NAMES
        assert "add_participant" in THENVOI_TOOL_NAMES
        assert "remove_participant" in THENVOI_TOOL_NAMES
        assert "lookup_peers" in THENVOI_TOOL_NAMES
        assert "get_participants" in THENVOI_TOOL_NAMES
        assert "create_chatroom" in THENVOI_TOOL_NAMES


class TestLettaToolSchemas:
    """Tests for Letta tool schema generation from runtime.tools."""

    def test_generates_schema_from_tool_models(self):
        """Should generate schemas from TOOL_MODELS."""
        for name in THENVOI_TOOL_NAMES:
            schema = _get_letta_tool_schema(name)
            assert "type" in schema
            assert schema["type"] == "object"

    def test_send_message_has_message_parameter(self):
        """send_message schema should have 'message' (renamed from 'content')."""
        schema = _get_letta_tool_schema("send_message")
        assert "message" in schema["properties"]
        assert "mentions" in schema["properties"]
        # 'content' should be renamed to 'message'
        assert "content" not in schema["properties"]

    def test_send_message_has_required_fields(self):
        """send_message should require message and mentions."""
        schema = _get_letta_tool_schema("send_message")
        assert "message" in schema.get("required", [])
        assert "mentions" in schema.get("required", [])


class TestRegisterThenvoiTools:
    """Tests for register_thenvoi_tools function."""

    @pytest.fixture
    def mock_client(self):
        """Create mock Letta client."""
        client = MagicMock()

        # Mock tools.upsert to return a tool with id
        def make_tool(**kwargs):
            tool = MagicMock()
            # Extract function name from source code
            import re

            source_code = kwargs.get("source_code", "")
            match = re.search(r"def (\w+)\(", source_code)
            tool.name = match.group(1) if match else "unknown"
            tool.id = f"tool-{tool.name}-123"
            return tool

        client.tools.upsert.side_effect = make_tool
        return client

    def test_registers_all_thenvoi_tools(self, mock_client):
        """Should register all Thenvoi tools with Letta."""
        tool_ids = register_thenvoi_tools(mock_client)

        assert len(tool_ids) == len(THENVOI_TOOL_NAMES)
        assert "send_message" in tool_ids
        assert "add_participant" in tool_ids

    def test_calls_upsert_with_thenvoi_tags(self, mock_client):
        """Should use 'thenvoi' tag for tools."""
        register_thenvoi_tools(mock_client)

        # Check that upsert was called with thenvoi tags
        call_args = mock_client.tools.upsert.call_args_list[0]
        assert call_args.kwargs.get("tags") == ["thenvoi"]


class TestGetLettaToolIds:
    """Tests for get_letta_tool_ids function."""

    @pytest.fixture
    def mock_client(self):
        """Create mock Letta client."""
        client = MagicMock()
        # Mock tools.list to return the default base tools
        tool1 = MagicMock()
        tool1.name = "memory"
        tool1.id = "tool-memory-123"
        tool1.tags = ["letta_memory_core"]
        tool2 = MagicMock()
        tool2.name = "conversation_search"
        tool2.id = "tool-conv-search-123"
        tool2.tags = ["letta_core"]
        tool3 = MagicMock()
        tool3.name = "archival_memory_insert"
        tool3.id = "tool-archival-insert-123"
        tool3.tags = ["letta_core"]
        tool4 = MagicMock()
        tool4.name = "archival_memory_search"
        tool4.id = "tool-archival-search-123"
        tool4.tags = ["letta_core"]
        client.tools.list.return_value = [tool1, tool2, tool3, tool4]
        return client

    def test_includes_thenvoi_tool_ids(self, mock_client):
        """Should include Thenvoi tool IDs."""
        thenvoi_ids = {
            "send_message": "tool-send-123",
            "add_participant": "tool-add-123",
        }
        tool_ids = get_letta_tool_ids(mock_client, thenvoi_ids, letta_base_tools=[])

        assert "tool-send-123" in tool_ids
        assert "tool-add-123" in tool_ids

    def test_includes_base_tools_by_default(self, mock_client):
        """Should include Letta base tools by default."""
        thenvoi_ids = {"send_message": "tool-send-123"}
        tool_ids = get_letta_tool_ids(mock_client, thenvoi_ids)

        # Should include default base tools
        assert "tool-memory-123" in tool_ids
        assert "tool-conv-search-123" in tool_ids
        assert "tool-archival-insert-123" in tool_ids
        assert "tool-archival-search-123" in tool_ids

    def test_uses_custom_base_tools_when_provided(self, mock_client):
        """Should use custom base tools when provided."""
        thenvoi_ids = {"send_message": "tool-send-123"}
        tool_ids = get_letta_tool_ids(
            mock_client, thenvoi_ids, letta_base_tools=["memory"]
        )

        # Should only include memory tool
        assert "tool-memory-123" in tool_ids
        assert "tool-conv-search-123" not in tool_ids
        assert len(tool_ids) == 2  # send_message + memory


class TestCustomToolBuilder:
    """Tests for CustomToolBuilder class."""

    def test_creates_empty_builder(self):
        """Should create empty builder."""
        builder = CustomToolBuilder()
        assert builder.get_tool_names() == []
        assert builder.get_tool_definitions() == []

    def test_decorator_registers_tool(self):
        """Should register tool via decorator."""
        builder = CustomToolBuilder()

        @builder.tool
        def my_tool(arg1: str, arg2: int) -> str:
            """My tool description."""
            return f"{arg1}-{arg2}"

        names = builder.get_tool_names()
        assert "my_tool" in names

    def test_extracts_docstring_as_description(self):
        """Should use docstring as tool description."""
        builder = CustomToolBuilder()

        @builder.tool
        def calculate(a: float, b: float) -> float:
            """Perform a calculation on two numbers."""
            return a + b

        definitions = builder.get_tool_definitions()
        assert len(definitions) == 1
        assert definitions[0]["description"] == "Perform a calculation on two numbers."

    def test_extracts_parameters_from_hints(self):
        """Should extract parameter types from type hints."""
        builder = CustomToolBuilder()

        @builder.tool
        def process(name: str, count: int, enabled: bool) -> str:
            """Process data."""
            return f"{name}"

        definitions = builder.get_tool_definitions()
        params = definitions[0]["parameters"]

        assert params["properties"]["name"]["type"] == "string"
        assert params["properties"]["count"]["type"] == "integer"
        assert params["properties"]["enabled"]["type"] == "boolean"

    def test_marks_required_parameters(self):
        """Should mark parameters without defaults as required."""
        builder = CustomToolBuilder()

        @builder.tool
        def mixed(required_arg: str, optional_arg: str = "default") -> str:
            """Mix of required and optional."""
            return required_arg

        definitions = builder.get_tool_definitions()
        params = definitions[0]["parameters"]

        assert "required_arg" in params["required"]
        assert "optional_arg" not in params["required"]

    def test_register_programmatically(self):
        """Should register tool programmatically."""
        builder = CustomToolBuilder()

        def my_func(x: int) -> int:
            return x * 2

        builder.register(
            name="double",
            description="Double a number",
            func=my_func,
            parameters={
                "type": "object",
                "properties": {"x": {"type": "integer"}},
                "required": ["x"],
            },
        )

        assert builder.has_tool("double")
        assert "double" in builder.get_tool_names()

    def test_has_tool(self):
        """Should check if tool exists."""
        builder = CustomToolBuilder()

        @builder.tool
        def existing() -> None:
            """Existing tool."""
            pass

        assert builder.has_tool("existing") is True
        assert builder.has_tool("nonexistent") is False

    @pytest.mark.asyncio
    async def test_executes_sync_tool(self):
        """Should execute synchronous tool."""
        builder = CustomToolBuilder()

        @builder.tool
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        result = await builder.execute("add", {"a": 2, "b": 3})
        assert result == 5

    @pytest.mark.asyncio
    async def test_executes_async_tool(self):
        """Should execute asynchronous tool."""
        builder = CustomToolBuilder()

        @builder.tool
        async def fetch(url: str) -> str:
            """Fetch URL."""
            return f"fetched: {url}"

        result = await builder.execute("fetch", {"url": "https://example.com"})
        assert result == "fetched: https://example.com"

    @pytest.mark.asyncio
    async def test_raises_on_unknown_tool_execution(self):
        """Should raise ValueError for unknown tool execution."""
        builder = CustomToolBuilder()

        with pytest.raises(ValueError, match="Unknown tool"):
            await builder.execute("nonexistent", {})

    def test_python_type_to_json_mapping(self):
        """Should map Python types to JSON schema types."""
        builder = CustomToolBuilder()

        assert builder._python_type_to_json(str) == "string"
        assert builder._python_type_to_json(int) == "integer"
        assert builder._python_type_to_json(float) == "number"
        assert builder._python_type_to_json(bool) == "boolean"
        assert builder._python_type_to_json(list) == "array"
        assert builder._python_type_to_json(dict) == "object"
        # Unknown types default to string
        assert builder._python_type_to_json(object) == "string"
