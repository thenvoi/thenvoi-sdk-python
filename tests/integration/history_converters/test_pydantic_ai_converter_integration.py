"""PydanticAI history converter integration scenarios."""

from __future__ import annotations

import pytest
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

            peers = await api_client.agent_api_peers.list_agent_peers()
            assert peers.data and len(peers.data) > 0, "Need at least one peer"
            peer = peers.data[0]
            await api_client.agent_api_participants.add_agent_chat_participant(
                chat_id,
                participant=ParticipantRequest(participant_id=peer.id, role="member"),
            )

            # === Create tool events ===
            await api_client.agent_api_messages.create_agent_chat_message(
                chat_id,
                message=ChatMessageRequest(
                    content=f"@{peer.name} Search for Python tutorials",
                    mentions=[Mention(id=peer.id, name=peer.name)],
                ),
            )

            await api_client.agent_api_events.create_agent_chat_event(
                chat_id,
                event=ChatEventRequest(
                    content=_create_tool_call_content(
                        "search", {"query": "Python tutorials"}, "call_abc123"
                    ),
                    message_type="tool_call",
                ),
            )

            await api_client.agent_api_events.create_agent_chat_event(
                chat_id,
                event=ChatEventRequest(
                    content=_create_tool_result_content(
                        "search", '["tutorial1", "tutorial2"]', "call_abc123"
                    ),
                    message_type="tool_result",
                ),
            )

            # === Fetch and convert ===
            context_response = (
                await api_client.agent_api_context.get_agent_chat_context(chat_id)
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
