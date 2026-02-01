"""Integration tests for history converters with real platform data.

These tests verify that history converters work correctly with real session
history from the platform. This catches any mismatches between the expected
format in the converters and the actual format stored by the platform.

Run with: uv run pytest tests/integration/test_history_converters.py -v -s

End-to-end tests (require ANTHROPIC_API_KEY):
    uv run pytest tests/integration/test_history_converters.py::TestEndToEndWithRealLLM -v -s
"""

from __future__ import annotations

import asyncio
import json
import logging

import pytest
from thenvoi_rest import ChatEventRequest, ChatMessageRequest, ChatRoomRequest
from thenvoi_rest.types import (
    ChatMessageRequestMentionsItem as Mention,
    ParticipantRequest,
)

from tests.integration.conftest import requires_api, requires_llm_api

logger = logging.getLogger(__name__)


def _create_tool_call_content(
    tool_name: str,
    args: dict,
    tool_call_id: str,
) -> str:
    """Create tool_call content in the format adapters use.

    This is the same format used by:
    - AnthropicAdapter (anthropic.py:310-317)
    - PydanticAIAdapter (pydantic_ai.py:273-282)

    Format: {"name": "...", "args": {...}, "tool_call_id": "..."}
    """
    return json.dumps(
        {
            "name": tool_name,
            "args": args,
            "tool_call_id": tool_call_id,
        }
    )


def _create_tool_result_content(
    tool_name: str,
    output: str,
    tool_call_id: str,
) -> str:
    """Create tool_result content in the format adapters use.

    This is the same format used by:
    - AnthropicAdapter (anthropic.py:340-348)
    - PydanticAIAdapter (pydantic_ai.py:288-297)

    Format: {"name": "...", "output": "...", "tool_call_id": "..."}
    """
    return json.dumps(
        {
            "name": tool_name,
            "output": output,
            "tool_call_id": tool_call_id,
        }
    )


@requires_api
class TestAnthropicConverterIntegration:
    """Integration tests for AnthropicHistoryConverter with real platform data."""

    async def test_converter_with_real_tool_history(self, api_client):
        """Verify Anthropic converter handles real platform tool history."""
        from thenvoi.converters.anthropic import AnthropicHistoryConverter

        # === SETUP: Create chat and add peer ===
        logger.info("Creating test chat...")
        response = await api_client.agent_api.create_agent_chat(chat=ChatRoomRequest())
        chat_id = response.data.id
        logger.info("Created chat: %s", chat_id)

        # Get agent info for name filtering
        agent_me = await api_client.agent_api.get_agent_me()
        agent_name = agent_me.data.name
        logger.info("Agent name: %s", agent_name)

        # Add a peer to the chat
        peers = await api_client.agent_api.list_agent_peers()
        assert peers.data and len(peers.data) > 0, "Need at least one peer"
        peer = peers.data[0]
        await api_client.agent_api.add_agent_chat_participant(
            chat_id,
            participant=ParticipantRequest(participant_id=peer.id, role="member"),
        )
        logger.info("Added peer: %s", peer.name)

        # === STEP 1: Send a user message (with mention to trigger it) ===
        await api_client.agent_api.create_agent_chat_message(
            chat_id,
            message=ChatMessageRequest(
                content=f"@{peer.name} What's the weather in Tokyo?",
                mentions=[Mention(id=peer.id, name=peer.name)],
            ),
        )
        logger.info("Sent initial message")

        # === STEP 2: Create tool_call event ===
        tool_call_content = _create_tool_call_content(
            tool_name="get_weather",
            args={"location": "Tokyo", "unit": "celsius"},
            tool_call_id="toolu_test_123",
        )
        await api_client.agent_api.create_agent_chat_event(
            chat_id,
            event=ChatEventRequest(
                content=tool_call_content,
                message_type="tool_call",
            ),
        )
        logger.info("Created tool_call event")

        # === STEP 3: Create tool_result event ===
        tool_result_content = _create_tool_result_content(
            tool_name="get_weather",
            output="Tokyo is 22°C and sunny",
            tool_call_id="toolu_test_123",
        )
        await api_client.agent_api.create_agent_chat_event(
            chat_id,
            event=ChatEventRequest(
                content=tool_result_content,
                message_type="tool_result",
            ),
        )
        logger.info("Created tool_result event")

        # === STEP 4: Fetch history from platform ===
        context_response = await api_client.agent_api.get_agent_chat_context(chat_id)
        raw_history = [msg.model_dump() for msg in context_response.data]
        logger.info("Fetched %d messages from platform", len(raw_history))

        # === STEP 5: Convert using Anthropic converter ===
        converter = AnthropicHistoryConverter(agent_name=agent_name)
        result = converter.convert(raw_history)
        logger.info("Converted to %d Anthropic messages", len(result))

        # === STEP 6: Verify conversion ===
        # Find tool_use message
        tool_use_messages = [
            m
            for m in result
            if m.get("role") == "assistant"
            and isinstance(m.get("content"), list)
            and any(b.get("type") == "tool_use" for b in m["content"])
        ]
        assert len(tool_use_messages) >= 1, "Should have at least one tool_use message"

        tool_use_block = tool_use_messages[0]["content"][0]
        assert tool_use_block["type"] == "tool_use"
        assert tool_use_block["id"] == "toolu_test_123"
        assert tool_use_block["name"] == "get_weather"
        assert tool_use_block["input"] == {"location": "Tokyo", "unit": "celsius"}

        # Find tool_result message
        tool_result_messages = [
            m
            for m in result
            if m.get("role") == "user"
            and isinstance(m.get("content"), list)
            and any(b.get("type") == "tool_result" for b in m["content"])
        ]
        assert len(tool_result_messages) >= 1, (
            "Should have at least one tool_result message"
        )

        tool_result_block = tool_result_messages[0]["content"][0]
        assert tool_result_block["type"] == "tool_result"
        assert tool_result_block["tool_use_id"] == "toolu_test_123"
        assert tool_result_block["content"] == "Tokyo is 22°C and sunny"

        logger.info("SUCCESS: Anthropic converter correctly handled real platform data")

    async def test_converter_batches_parallel_tool_calls(self, api_client):
        """Verify converter batches parallel tool calls from platform."""
        from thenvoi.converters.anthropic import AnthropicHistoryConverter

        # === SETUP ===
        response = await api_client.agent_api.create_agent_chat(chat=ChatRoomRequest())
        chat_id = response.data.id

        agent_me = await api_client.agent_api.get_agent_me()
        agent_name = agent_me.data.name

        # === Create multiple tool_call events (simulating parallel tool use) ===
        await api_client.agent_api.create_agent_chat_event(
            chat_id,
            event=ChatEventRequest(
                content=_create_tool_call_content(
                    "get_weather", {"location": "Tokyo"}, "toolu_1"
                ),
                message_type="tool_call",
            ),
        )
        await api_client.agent_api.create_agent_chat_event(
            chat_id,
            event=ChatEventRequest(
                content=_create_tool_call_content(
                    "get_weather", {"location": "London"}, "toolu_2"
                ),
                message_type="tool_call",
            ),
        )

        # === Create corresponding tool_result events ===
        await api_client.agent_api.create_agent_chat_event(
            chat_id,
            event=ChatEventRequest(
                content=_create_tool_result_content(
                    "get_weather", "Tokyo: 22°C", "toolu_1"
                ),
                message_type="tool_result",
            ),
        )
        await api_client.agent_api.create_agent_chat_event(
            chat_id,
            event=ChatEventRequest(
                content=_create_tool_result_content(
                    "get_weather", "London: 15°C", "toolu_2"
                ),
                message_type="tool_result",
            ),
        )

        # === Fetch and convert ===
        context_response = await api_client.agent_api.get_agent_chat_context(chat_id)
        raw_history = [msg.model_dump() for msg in context_response.data]

        converter = AnthropicHistoryConverter(agent_name=agent_name)
        result = converter.convert(raw_history)

        # === Verify batching ===
        # Should have batched tool_use blocks in single assistant message
        tool_use_messages = [
            m
            for m in result
            if m.get("role") == "assistant"
            and isinstance(m.get("content"), list)
            and any(b.get("type") == "tool_use" for b in m["content"])
        ]
        assert len(tool_use_messages) == 1, "Should batch tool_use into single message"
        assert len(tool_use_messages[0]["content"]) == 2, (
            "Should have 2 tool_use blocks"
        )

        # Should have batched tool_result blocks in single user message
        tool_result_messages = [
            m
            for m in result
            if m.get("role") == "user"
            and isinstance(m.get("content"), list)
            and any(b.get("type") == "tool_result" for b in m["content"])
        ]
        assert len(tool_result_messages) == 1, (
            "Should batch tool_result into single message"
        )
        assert len(tool_result_messages[0]["content"]) == 2, (
            "Should have 2 tool_result blocks"
        )

        logger.info(
            "SUCCESS: Anthropic converter correctly batches parallel tool calls"
        )


@requires_api
class TestPydanticAIConverterIntegration:
    """Integration tests for PydanticAIHistoryConverter with real platform data."""

    async def test_converter_with_real_tool_history(self, api_client):
        """Verify PydanticAI converter handles real platform tool history."""
        pydantic_ai_messages = pytest.importorskip("pydantic_ai.messages")
        ModelRequest = pydantic_ai_messages.ModelRequest
        ModelResponse = pydantic_ai_messages.ModelResponse

        from thenvoi.converters.pydantic_ai import PydanticAIHistoryConverter

        # === SETUP ===
        response = await api_client.agent_api.create_agent_chat(chat=ChatRoomRequest())
        chat_id = response.data.id

        agent_me = await api_client.agent_api.get_agent_me()
        agent_name = agent_me.data.name

        peers = await api_client.agent_api.list_agent_peers()
        assert peers.data and len(peers.data) > 0, "Need at least one peer"
        peer = peers.data[0]
        await api_client.agent_api.add_agent_chat_participant(
            chat_id,
            participant=ParticipantRequest(participant_id=peer.id, role="member"),
        )

        # === Create tool events ===
        await api_client.agent_api.create_agent_chat_message(
            chat_id,
            message=ChatMessageRequest(
                content=f"@{peer.name} Search for Python tutorials",
                mentions=[Mention(id=peer.id, name=peer.name)],
            ),
        )

        await api_client.agent_api.create_agent_chat_event(
            chat_id,
            event=ChatEventRequest(
                content=_create_tool_call_content(
                    "search", {"query": "Python tutorials"}, "call_abc123"
                ),
                message_type="tool_call",
            ),
        )

        await api_client.agent_api.create_agent_chat_event(
            chat_id,
            event=ChatEventRequest(
                content=_create_tool_result_content(
                    "search", '["tutorial1", "tutorial2"]', "call_abc123"
                ),
                message_type="tool_result",
            ),
        )

        # === Fetch and convert ===
        context_response = await api_client.agent_api.get_agent_chat_context(chat_id)
        raw_history = [msg.model_dump() for msg in context_response.data]

        converter = PydanticAIHistoryConverter(agent_name=agent_name)
        result = converter.convert(raw_history)

        # === Verify conversion ===
        # Find ModelResponse with ToolCallPart
        tool_call_responses = [m for m in result if isinstance(m, ModelResponse)]
        assert len(tool_call_responses) >= 1, "Should have ModelResponse for tool_call"

        tool_call_parts = [
            p
            for r in tool_call_responses
            for p in r.parts
            if hasattr(p, "tool_call_id")
        ]
        assert len(tool_call_parts) >= 1, "Should have ToolCallPart"
        assert tool_call_parts[0].tool_name == "search"
        assert tool_call_parts[0].tool_call_id == "call_abc123"
        assert tool_call_parts[0].args == {"query": "Python tutorials"}

        # Find ModelRequest with ToolReturnPart
        tool_result_requests = [
            m
            for m in result
            if isinstance(m, ModelRequest)
            and any(hasattr(p, "tool_call_id") for p in m.parts)
        ]
        assert len(tool_result_requests) >= 1, (
            "Should have ModelRequest for tool_result"
        )

        tool_return_parts = [
            p
            for r in tool_result_requests
            for p in r.parts
            if hasattr(p, "tool_call_id") and hasattr(p, "content")
        ]
        assert len(tool_return_parts) >= 1, "Should have ToolReturnPart"
        assert tool_return_parts[0].tool_name == "search"
        assert tool_return_parts[0].tool_call_id == "call_abc123"

        logger.info(
            "SUCCESS: PydanticAI converter correctly handled real platform data"
        )


@requires_api
class TestMixedConversationIntegration:
    """Test converters with mixed conversation patterns (text + tools)."""

    async def test_full_conversation_flow(self, api_client):
        """Test converter handles realistic conversation with multiple turns."""
        from thenvoi.converters.anthropic import AnthropicHistoryConverter

        # === SETUP ===
        response = await api_client.agent_api.create_agent_chat(chat=ChatRoomRequest())
        chat_id = response.data.id

        agent_me = await api_client.agent_api.get_agent_me()
        agent_name = agent_me.data.name

        peers = await api_client.agent_api.list_agent_peers()
        assert peers.data and len(peers.data) > 0
        peer = peers.data[0]
        await api_client.agent_api.add_agent_chat_participant(
            chat_id,
            participant=ParticipantRequest(participant_id=peer.id, role="member"),
        )

        # === Simulate multi-turn conversation ===
        # Turn 1: User asks question
        await api_client.agent_api.create_agent_chat_message(
            chat_id,
            message=ChatMessageRequest(
                content=f"@{peer.name} What's the weather in NYC?",
                mentions=[Mention(id=peer.id, name=peer.name)],
            ),
        )

        # Turn 1: Agent uses tool
        await api_client.agent_api.create_agent_chat_event(
            chat_id,
            event=ChatEventRequest(
                content=_create_tool_call_content(
                    "get_weather", {"location": "NYC"}, "toolu_turn1"
                ),
                message_type="tool_call",
            ),
        )

        await api_client.agent_api.create_agent_chat_event(
            chat_id,
            event=ChatEventRequest(
                content=_create_tool_result_content(
                    "get_weather", "NYC: 18°C, cloudy", "toolu_turn1"
                ),
                message_type="tool_result",
            ),
        )

        # Turn 2: User asks follow-up
        await api_client.agent_api.create_agent_chat_message(
            chat_id,
            message=ChatMessageRequest(
                content=f"@{peer.name} What about San Francisco?",
                mentions=[Mention(id=peer.id, name=peer.name)],
            ),
        )

        # Turn 2: Agent uses tool again
        await api_client.agent_api.create_agent_chat_event(
            chat_id,
            event=ChatEventRequest(
                content=_create_tool_call_content(
                    "get_weather", {"location": "San Francisco"}, "toolu_turn2"
                ),
                message_type="tool_call",
            ),
        )

        await api_client.agent_api.create_agent_chat_event(
            chat_id,
            event=ChatEventRequest(
                content=_create_tool_result_content(
                    "get_weather", "SF: 15°C, foggy", "toolu_turn2"
                ),
                message_type="tool_result",
            ),
        )

        # === Fetch and convert ===
        context_response = await api_client.agent_api.get_agent_chat_context(chat_id)
        raw_history = [msg.model_dump() for msg in context_response.data]

        converter = AnthropicHistoryConverter(agent_name=agent_name)
        result = converter.convert(raw_history)

        # === Verify structure ===
        # Should have: user1, tool_use1, tool_result1, user2, tool_use2, tool_result2
        # (at minimum, may have more user messages depending on API behavior)
        user_messages = [
            m
            for m in result
            if m.get("role") == "user" and isinstance(m.get("content"), str)
        ]
        tool_use_count = sum(
            1
            for m in result
            if m.get("role") == "assistant"
            and isinstance(m.get("content"), list)
            and any(b.get("type") == "tool_use" for b in m["content"])
        )
        tool_result_count = sum(
            1
            for m in result
            if m.get("role") == "user"
            and isinstance(m.get("content"), list)
            and any(b.get("type") == "tool_result" for b in m["content"])
        )

        assert len(user_messages) >= 2, "Should have at least 2 user messages"
        assert tool_use_count == 2, "Should have 2 tool_use messages"
        assert tool_result_count == 2, "Should have 2 tool_result messages"

        logger.info(
            "SUCCESS: Converter handled multi-turn conversation "
            "(user=%d, tool_use=%d, tool_result=%d)",
            len(user_messages),
            tool_use_count,
            tool_result_count,
        )


@requires_api
class TestEdgeCasesIntegration:
    """Test converter edge cases with real platform data."""

    async def test_handles_thought_events(self, api_client):
        """Verify thought events are properly skipped."""
        from thenvoi.converters.anthropic import AnthropicHistoryConverter

        response = await api_client.agent_api.create_agent_chat(chat=ChatRoomRequest())
        chat_id = response.data.id

        # Create thought event
        await api_client.agent_api.create_agent_chat_event(
            chat_id,
            event=ChatEventRequest(
                content="Let me think about this request...",
                message_type="thought",
            ),
        )

        # Create tool_call event
        await api_client.agent_api.create_agent_chat_event(
            chat_id,
            event=ChatEventRequest(
                content=_create_tool_call_content(
                    "analyze", {"text": "hello"}, "toolu_thought_test"
                ),
                message_type="tool_call",
            ),
        )

        # Fetch and convert
        context_response = await api_client.agent_api.get_agent_chat_context(chat_id)
        raw_history = [msg.model_dump() for msg in context_response.data]

        converter = AnthropicHistoryConverter()
        result = converter.convert(raw_history)

        # Verify thought is not in output
        for msg in result:
            if isinstance(msg.get("content"), str):
                assert "Let me think" not in msg["content"], (
                    "Thought event should be skipped"
                )

        # But tool_call should be present
        tool_use_count = sum(
            1
            for m in result
            if m.get("role") == "assistant"
            and isinstance(m.get("content"), list)
            and any(b.get("type") == "tool_use" for b in m["content"])
        )
        assert tool_use_count >= 1, "Tool call should be present"

        logger.info("SUCCESS: Thought events are properly skipped")

    async def test_handles_error_events(self, api_client):
        """Verify error events are properly skipped."""
        from thenvoi.converters.anthropic import AnthropicHistoryConverter

        response = await api_client.agent_api.create_agent_chat(chat=ChatRoomRequest())
        chat_id = response.data.id

        # Create error event
        await api_client.agent_api.create_agent_chat_event(
            chat_id,
            event=ChatEventRequest(
                content="Error: API rate limit exceeded",
                message_type="error",
            ),
        )

        # Fetch and convert
        context_response = await api_client.agent_api.get_agent_chat_context(chat_id)
        raw_history = [msg.model_dump() for msg in context_response.data]

        converter = AnthropicHistoryConverter()
        result = converter.convert(raw_history)

        # Verify error is not in output (error message_type is not 'text')
        for msg in result:
            if isinstance(msg.get("content"), str):
                assert "rate limit" not in msg["content"], (
                    "Error event should be skipped"
                )

        logger.info("SUCCESS: Error events are properly skipped")


@requires_llm_api
class TestEndToEndWithRealLLM:
    """End-to-end tests that run real adapters with actual LLM calls.

    These tests verify the full flow:
    1. Create an Agent with enable_execution_reporting=True
    2. Agent processes a message and makes real tool calls
    3. Tool events are emitted to the platform
    4. Fetch history and convert with our converters
    5. Verify the converted output is valid

    Requires: THENVOI_API_KEY and ANTHROPIC_API_KEY
    """

    @pytest.fixture
    def get_time_tool(self):
        """A simple custom tool that returns the current time."""
        from thenvoi.runtime.custom_tools import CustomToolDef

        async def get_current_time() -> str:
            """Get the current UTC time."""
            from datetime import datetime, timezone

            return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        return CustomToolDef(
            name="get_current_time",
            description="Get the current UTC time. Call this when asked for the time.",
            handler=get_current_time,
            parameters={},
        )

    async def test_anthropic_adapter_emits_tool_events(
        self, api_client, integration_settings, get_time_tool
    ):
        """Verify AnthropicAdapter emits tool_call and tool_result events.

        This test:
        1. Creates an AnthropicAdapter with enable_execution_reporting=True
        2. Runs the agent and sends a message that triggers a tool call
        3. Waits for the agent to process and emit tool events
        4. Fetches history from platform
        5. Converts with AnthropicHistoryConverter
        6. Verifies the tool_use and tool_result blocks are present
        """
        from thenvoi import Agent
        from thenvoi.adapters import AnthropicAdapter
        from thenvoi.converters.anthropic import AnthropicHistoryConverter

        # === SETUP: Create chat and get agent info ===
        response = await api_client.agent_api.create_agent_chat(chat=ChatRoomRequest())
        chat_id = response.data.id
        logger.info("Created chat: %s", chat_id)

        agent_me = await api_client.agent_api.get_agent_me()
        agent_name = agent_me.data.name
        agent_id = str(agent_me.data.id)
        logger.info("Agent: %s (id=%s)", agent_name, agent_id)

        # === Create adapter with execution reporting enabled ===
        adapter = AnthropicAdapter(
            model="claude-sonnet-4-5-20250929",
            custom_section=(
                "You are a helpful assistant. When asked about the time, "
                "you MUST use the get_current_time tool. Always use tools when available."
            ),
            enable_execution_reporting=True,
            additional_tools=[get_time_tool],
        )

        # === Create and start agent ===
        agent = Agent.create(
            adapter=adapter,
            agent_id=agent_id,
            api_key=integration_settings.thenvoi_api_key,
            ws_url=integration_settings.thenvoi_ws_url,
            rest_url=integration_settings.thenvoi_base_url,
        )

        # Start agent in background
        agent_task = asyncio.create_task(agent.run(shutdown_timeout=5.0))
        logger.info("Started agent in background")

        try:
            # Wait for agent to connect
            await asyncio.sleep(2)

            # === Send message that triggers tool call ===
            await api_client.agent_api.create_agent_chat_message(
                chat_id,
                message=ChatMessageRequest(
                    content=f"@{agent_name} What time is it right now?",
                    mentions=[Mention(id=agent_id, name=agent_name)],
                ),
            )
            logger.info("Sent message to trigger tool call")

            # === Wait for tool events to appear ===
            # Poll for tool_call and tool_result events
            max_wait = 30  # seconds
            poll_interval = 1  # second
            tool_call_found = False
            tool_result_found = False

            for _ in range(max_wait // poll_interval):
                await asyncio.sleep(poll_interval)

                context_response = await api_client.agent_api.get_agent_chat_context(
                    chat_id
                )
                raw_history = [msg.model_dump() for msg in context_response.data]

                for msg in raw_history:
                    msg_type = msg.get("message_type", "text")
                    if msg_type == "tool_call":
                        tool_call_found = True
                        logger.info("Found tool_call event")
                    elif msg_type == "tool_result":
                        tool_result_found = True
                        logger.info("Found tool_result event")

                if tool_call_found and tool_result_found:
                    break

            assert tool_call_found, "Tool call event was not emitted to platform"
            assert tool_result_found, "Tool result event was not emitted to platform"

            # === Fetch final history and convert ===
            context_response = await api_client.agent_api.get_agent_chat_context(
                chat_id
            )
            raw_history = [msg.model_dump() for msg in context_response.data]
            logger.info("Fetched %d messages from platform", len(raw_history))

            # Log raw history for debugging
            for i, msg in enumerate(raw_history):
                logger.debug(
                    "Raw[%d]: type=%s, content=%s...",
                    i,
                    msg.get("message_type"),
                    str(msg.get("content", ""))[:50],
                )

            # === Convert using Anthropic converter ===
            converter = AnthropicHistoryConverter(agent_name=agent_name)
            result = converter.convert(raw_history)
            logger.info("Converted to %d Anthropic messages", len(result))

            # === Verify conversion ===
            # Find tool_use message
            tool_use_messages = [
                m
                for m in result
                if m.get("role") == "assistant"
                and isinstance(m.get("content"), list)
                and any(b.get("type") == "tool_use" for b in m["content"])
            ]
            assert len(tool_use_messages) >= 1, (
                "Converter should produce at least one tool_use message"
            )

            tool_use_block = next(
                b
                for b in tool_use_messages[0]["content"]
                if b.get("type") == "tool_use"
            )
            assert tool_use_block["name"] == "get_current_time"
            assert "id" in tool_use_block
            logger.info("tool_use block: %s", tool_use_block)

            # Find tool_result message
            tool_result_messages = [
                m
                for m in result
                if m.get("role") == "user"
                and isinstance(m.get("content"), list)
                and any(b.get("type") == "tool_result" for b in m["content"])
            ]
            assert len(tool_result_messages) >= 1, (
                "Converter should produce at least one tool_result message"
            )

            tool_result_block = next(
                b
                for b in tool_result_messages[0]["content"]
                if b.get("type") == "tool_result"
            )
            assert tool_result_block["tool_use_id"] == tool_use_block["id"]
            assert "UTC" in tool_result_block["content"]
            logger.info("tool_result block: %s", tool_result_block)

            logger.info(
                "SUCCESS: End-to-end test passed - real LLM tool events "
                "were correctly converted"
            )

        finally:
            # Stop the agent
            await agent.stop(timeout=5.0)
            agent_task.cancel()
            try:
                await agent_task
            except asyncio.CancelledError:
                pass
            logger.info("Agent stopped")

    async def test_pydantic_ai_converter_with_anthropic_events(
        self, api_client, integration_settings, get_time_tool
    ):
        """Verify PydanticAIHistoryConverter works with real Anthropic tool events.

        This test uses the AnthropicAdapter to generate real tool events, then
        verifies that the PydanticAIHistoryConverter can also parse them correctly.
        This ensures format compatibility across converters.
        """
        pydantic_ai_messages = pytest.importorskip("pydantic_ai.messages")
        ModelRequest = pydantic_ai_messages.ModelRequest
        ModelResponse = pydantic_ai_messages.ModelResponse

        from thenvoi import Agent
        from thenvoi.adapters import AnthropicAdapter
        from thenvoi.converters.pydantic_ai import PydanticAIHistoryConverter

        # === SETUP ===
        response = await api_client.agent_api.create_agent_chat(chat=ChatRoomRequest())
        chat_id = response.data.id

        agent_me = await api_client.agent_api.get_agent_me()
        agent_name = agent_me.data.name
        agent_id = str(agent_me.data.id)

        # === Create adapter ===
        adapter = AnthropicAdapter(
            model="claude-sonnet-4-5-20250929",
            custom_section=(
                "You are a helpful assistant. When asked about the time, "
                "you MUST use the get_current_time tool."
            ),
            enable_execution_reporting=True,
            additional_tools=[get_time_tool],
        )

        agent = Agent.create(
            adapter=adapter,
            agent_id=agent_id,
            api_key=integration_settings.thenvoi_api_key,
            ws_url=integration_settings.thenvoi_ws_url,
            rest_url=integration_settings.thenvoi_base_url,
        )

        agent_task = asyncio.create_task(agent.run(shutdown_timeout=5.0))

        try:
            await asyncio.sleep(2)

            # Send message
            await api_client.agent_api.create_agent_chat_message(
                chat_id,
                message=ChatMessageRequest(
                    content=f"@{agent_name} Tell me what time it is",
                    mentions=[Mention(id=agent_id, name=agent_name)],
                ),
            )

            # Wait for tool events
            max_wait = 30
            tool_result_found = False

            for _ in range(max_wait):
                await asyncio.sleep(1)
                context_response = await api_client.agent_api.get_agent_chat_context(
                    chat_id
                )
                raw_history = [msg.model_dump() for msg in context_response.data]

                for msg in raw_history:
                    if msg.get("message_type") == "tool_result":
                        tool_result_found = True
                        break
                if tool_result_found:
                    break

            assert tool_result_found, "Tool result event was not emitted"

            # === Convert with PydanticAI converter ===
            context_response = await api_client.agent_api.get_agent_chat_context(
                chat_id
            )
            raw_history = [msg.model_dump() for msg in context_response.data]

            converter = PydanticAIHistoryConverter(agent_name=agent_name)
            result = converter.convert(raw_history)

            # === Verify conversion ===
            # Find ModelResponse with ToolCallPart
            tool_call_responses = [m for m in result if isinstance(m, ModelResponse)]
            assert len(tool_call_responses) >= 1, "Should have ModelResponse"

            tool_call_parts = [
                p
                for r in tool_call_responses
                for p in r.parts
                if hasattr(p, "tool_call_id") and hasattr(p, "tool_name")
            ]
            assert len(tool_call_parts) >= 1, "Should have ToolCallPart"
            assert tool_call_parts[0].tool_name == "get_current_time"

            # Find ModelRequest with ToolReturnPart
            tool_return_requests = [
                m
                for m in result
                if isinstance(m, ModelRequest)
                and any(
                    hasattr(p, "tool_call_id") and hasattr(p, "content")
                    for p in m.parts
                )
            ]
            assert len(tool_return_requests) >= 1, "Should have ModelRequest"

            tool_return_parts = [
                p
                for r in tool_return_requests
                for p in r.parts
                if hasattr(p, "tool_call_id") and hasattr(p, "content")
            ]
            assert len(tool_return_parts) >= 1, "Should have ToolReturnPart"
            assert "UTC" in tool_return_parts[0].content

            logger.info(
                "SUCCESS: PydanticAI converter correctly handles real "
                "Anthropic tool events"
            )

        finally:
            await agent.stop(timeout=5.0)
            agent_task.cancel()
            try:
                await agent_task
            except asyncio.CancelledError:
                pass
