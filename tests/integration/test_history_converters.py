"""Integration tests for history converters with real platform data.

These tests verify that history converters work correctly with real session
history from the platform. This catches any mismatches between the expected
format in the converters and the actual format stored by the platform.

Run with: uv run pytest tests/integration/test_history_converters.py -v -s

Skip cleanup (for debugging):
    uv run pytest tests/integration/test_history_converters.py -v -s --no-clean
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager

import pytest
from thenvoi_rest import (
    AsyncRestClient,
    ChatEventRequest,
    ChatMessageRequest,
    ChatRoomRequest,
)
from thenvoi_rest.types import (
    ChatMessageRequestMentionsItem as Mention,
    ParticipantRequest,
)

from tests.integration.conftest import requires_api

logger = logging.getLogger(__name__)


def pytest_addoption(parser):
    """Add --no-clean option to pytest."""
    try:
        parser.addoption(
            "--no-clean",
            action="store_true",
            default=False,
            help="Skip cleanup of test chats (for debugging)",
        )
    except ValueError:
        # Option already added (e.g., by conftest.py)
        pass


@pytest.fixture
def no_clean(request):
    """Fixture to check if --no-clean flag was passed."""
    return request.config.getoption("--no-clean", default=False)


@asynccontextmanager
async def create_test_chat(api_client: AsyncRestClient, skip_cleanup: bool = False):
    """Context manager that creates a chat and cleans up by leaving it.

    Since the API doesn't support deleting chats, cleanup is done by
    having the agent leave the chat (remove itself as participant).

    Args:
        api_client: The API client to use
        skip_cleanup: If True, skip cleanup (for debugging with --no-clean flag)
    """
    # Create chat
    response = await api_client.agent_api.create_agent_chat(chat=ChatRoomRequest())
    chat_id = response.data.id
    logger.info("Created test chat: %s", chat_id)

    # Get agent ID for cleanup
    agent_me = await api_client.agent_api.get_agent_me()
    agent_id = agent_me.data.id

    try:
        yield chat_id, agent_me.data
    finally:
        if skip_cleanup:
            logger.info("Skipping cleanup for chat %s (--no-clean)", chat_id)
            return

        # Cleanup: agent leaves the chat
        try:
            await api_client.agent_api.remove_agent_chat_participant(chat_id, agent_id)
            logger.info("Cleanup: left chat %s", chat_id)
        except Exception as e:
            logger.warning("Cleanup failed for chat %s: %s", chat_id, e)


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

    async def test_converter_with_real_tool_history(self, api_client, no_clean):
        """Verify Anthropic converter handles real platform tool history."""
        from thenvoi.converters.anthropic import AnthropicHistoryConverter

        async with create_test_chat(api_client, skip_cleanup=no_clean) as (
            chat_id,
            agent_me,
        ):
            agent_name = agent_me.name

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
            context_response = await api_client.agent_api.get_agent_chat_context(
                chat_id
            )
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
            assert len(tool_use_messages) >= 1, (
                "Should have at least one tool_use message"
            )

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

            logger.info(
                "SUCCESS: Anthropic converter correctly handled real platform data"
            )

    async def test_converter_batches_parallel_tool_calls(self, api_client, no_clean):
        """Verify converter batches parallel tool calls from platform."""
        from thenvoi.converters.anthropic import AnthropicHistoryConverter

        async with create_test_chat(api_client, skip_cleanup=no_clean) as (
            chat_id,
            agent_me,
        ):
            agent_name = agent_me.name

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
            context_response = await api_client.agent_api.get_agent_chat_context(
                chat_id
            )
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
            assert len(tool_use_messages) == 1, (
                "Should batch tool_use into single message"
            )
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

    async def test_converter_with_real_tool_history(self, api_client, no_clean):
        """Verify PydanticAI converter handles real platform tool history."""
        pydantic_ai_messages = pytest.importorskip("pydantic_ai.messages")
        ModelRequest = pydantic_ai_messages.ModelRequest
        ModelResponse = pydantic_ai_messages.ModelResponse

        from thenvoi.converters.pydantic_ai import PydanticAIHistoryConverter

        async with create_test_chat(api_client, skip_cleanup=no_clean) as (
            chat_id,
            agent_me,
        ):
            agent_name = agent_me.name

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
            context_response = await api_client.agent_api.get_agent_chat_context(
                chat_id
            )
            raw_history = [msg.model_dump() for msg in context_response.data]

            converter = PydanticAIHistoryConverter(agent_name=agent_name)
            result = converter.convert(raw_history)

            # === Verify conversion ===
            # Find ModelResponse with ToolCallPart
            tool_call_responses = [m for m in result if isinstance(m, ModelResponse)]
            assert len(tool_call_responses) >= 1, (
                "Should have ModelResponse for tool_call"
            )

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

    async def test_full_conversation_flow(self, api_client, no_clean):
        """Test converter handles realistic conversation with multiple turns."""
        from thenvoi.converters.anthropic import AnthropicHistoryConverter

        async with create_test_chat(api_client, skip_cleanup=no_clean) as (
            chat_id,
            agent_me,
        ):
            agent_name = agent_me.name

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
            context_response = await api_client.agent_api.get_agent_chat_context(
                chat_id
            )
            raw_history = [msg.model_dump() for msg in context_response.data]

            converter = AnthropicHistoryConverter(agent_name=agent_name)
            result = converter.convert(raw_history)

            # === Verify structure ===
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

    async def test_handles_thought_events(self, api_client, no_clean):
        """Verify thought events are properly skipped."""
        from thenvoi.converters.anthropic import AnthropicHistoryConverter

        async with create_test_chat(api_client, skip_cleanup=no_clean) as (
            chat_id,
            _agent_me,
        ):
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
            context_response = await api_client.agent_api.get_agent_chat_context(
                chat_id
            )
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

    async def test_handles_error_events(self, api_client, no_clean):
        """Verify error events are properly skipped."""
        from thenvoi.converters.anthropic import AnthropicHistoryConverter

        async with create_test_chat(api_client, skip_cleanup=no_clean) as (
            chat_id,
            _agent_me,
        ):
            # Create error event
            await api_client.agent_api.create_agent_chat_event(
                chat_id,
                event=ChatEventRequest(
                    content="Error: API rate limit exceeded",
                    message_type="error",
                ),
            )

            # Fetch and convert
            context_response = await api_client.agent_api.get_agent_chat_context(
                chat_id
            )
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
