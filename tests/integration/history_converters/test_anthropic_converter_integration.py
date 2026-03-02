"""Anthropic history converter integration scenarios."""

from __future__ import annotations

from thenvoi_rest import ChatEventRequest, ChatMessageRequest
from thenvoi_rest.types import (
    ChatMessageRequestMentionsItem as Mention,
    ParticipantRequest,
)

from tests.integration.history_converters.support import (
    _create_tool_call_content,
    _create_tool_result_content,
    create_test_chat,
    logger,
)
from tests.support.integration.markers import requires_api

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
            peers = await api_client.agent_api_peers.list_agent_peers()
            assert peers.data and len(peers.data) > 0, "Need at least one peer"
            peer = peers.data[0]
            await api_client.agent_api_participants.add_agent_chat_participant(
                chat_id,
                participant=ParticipantRequest(participant_id=peer.id, role="member"),
            )
            logger.info("Added peer: %s", peer.name)

            # === STEP 1: Send a user message (with mention to trigger it) ===
            await api_client.agent_api_messages.create_agent_chat_message(
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
            await api_client.agent_api_events.create_agent_chat_event(
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
            await api_client.agent_api_events.create_agent_chat_event(
                chat_id,
                event=ChatEventRequest(
                    content=tool_result_content,
                    message_type="tool_result",
                ),
            )
            logger.info("Created tool_result event")

            # === STEP 4: Fetch history from platform ===
            context_response = (
                await api_client.agent_api_context.get_agent_chat_context(chat_id)
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
            await api_client.agent_api_events.create_agent_chat_event(
                chat_id,
                event=ChatEventRequest(
                    content=_create_tool_call_content(
                        "get_weather", {"location": "Tokyo"}, "toolu_1"
                    ),
                    message_type="tool_call",
                ),
            )
            await api_client.agent_api_events.create_agent_chat_event(
                chat_id,
                event=ChatEventRequest(
                    content=_create_tool_call_content(
                        "get_weather", {"location": "London"}, "toolu_2"
                    ),
                    message_type="tool_call",
                ),
            )

            # === Create corresponding tool_result events ===
            await api_client.agent_api_events.create_agent_chat_event(
                chat_id,
                event=ChatEventRequest(
                    content=_create_tool_result_content(
                        "get_weather", "Tokyo: 22°C", "toolu_1"
                    ),
                    message_type="tool_result",
                ),
            )
            await api_client.agent_api_events.create_agent_chat_event(
                chat_id,
                event=ChatEventRequest(
                    content=_create_tool_result_content(
                        "get_weather", "London: 15°C", "toolu_2"
                    ),
                    message_type="tool_result",
                ),
            )

            # === Fetch and convert ===
            context_response = (
                await api_client.agent_api_context.get_agent_chat_context(chat_id)
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
