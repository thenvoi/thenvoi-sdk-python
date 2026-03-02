"""Mixed text/tool conversation converter integration scenarios."""

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

            peers = await api_client.agent_api_peers.list_agent_peers()
            assert peers.data and len(peers.data) > 0
            peer = peers.data[0]
            await api_client.agent_api_participants.add_agent_chat_participant(
                chat_id,
                participant=ParticipantRequest(participant_id=peer.id, role="member"),
            )

            # === Simulate multi-turn conversation ===
            # Turn 1: User asks question
            await api_client.agent_api_messages.create_agent_chat_message(
                chat_id,
                message=ChatMessageRequest(
                    content=f"@{peer.name} What's the weather in NYC?",
                    mentions=[Mention(id=peer.id, name=peer.name)],
                ),
            )

            # Turn 1: Agent uses tool
            await api_client.agent_api_events.create_agent_chat_event(
                chat_id,
                event=ChatEventRequest(
                    content=_create_tool_call_content(
                        "get_weather", {"location": "NYC"}, "toolu_turn1"
                    ),
                    message_type="tool_call",
                ),
            )

            await api_client.agent_api_events.create_agent_chat_event(
                chat_id,
                event=ChatEventRequest(
                    content=_create_tool_result_content(
                        "get_weather", "NYC: 18°C, cloudy", "toolu_turn1"
                    ),
                    message_type="tool_result",
                ),
            )

            # Turn 2: User asks follow-up
            await api_client.agent_api_messages.create_agent_chat_message(
                chat_id,
                message=ChatMessageRequest(
                    content=f"@{peer.name} What about San Francisco?",
                    mentions=[Mention(id=peer.id, name=peer.name)],
                ),
            )

            # Turn 2: Agent uses tool again
            await api_client.agent_api_events.create_agent_chat_event(
                chat_id,
                event=ChatEventRequest(
                    content=_create_tool_call_content(
                        "get_weather", {"location": "San Francisco"}, "toolu_turn2"
                    ),
                    message_type="tool_call",
                ),
            )

            await api_client.agent_api_events.create_agent_chat_event(
                chat_id,
                event=ChatEventRequest(
                    content=_create_tool_result_content(
                        "get_weather", "SF: 15°C, foggy", "toolu_turn2"
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
