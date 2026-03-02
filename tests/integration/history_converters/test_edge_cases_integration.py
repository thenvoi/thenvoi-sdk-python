"""History converter edge-case integration scenarios."""

from __future__ import annotations

from thenvoi_rest import ChatEventRequest

from tests.integration.history_converters.support import (
    _create_tool_call_content,
    create_test_chat,
    logger,
)
from tests.support.integration.markers import requires_api

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
            await api_client.agent_api_events.create_agent_chat_event(
                chat_id,
                event=ChatEventRequest(
                    content="Let me think about this request...",
                    message_type="thought",
                ),
            )

            # Create tool_call event
            await api_client.agent_api_events.create_agent_chat_event(
                chat_id,
                event=ChatEventRequest(
                    content=_create_tool_call_content(
                        "analyze", {"text": "hello"}, "toolu_thought_test"
                    ),
                    message_type="tool_call",
                ),
            )

            # Fetch and convert
            context_response = (
                await api_client.agent_api_context.get_agent_chat_context(chat_id)
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
            await api_client.agent_api_events.create_agent_chat_event(
                chat_id,
                event=ChatEventRequest(
                    content="Error: API rate limit exceeded",
                    message_type="error",
                ),
            )

            # Fetch and convert
            context_response = (
                await api_client.agent_api_context.get_agent_chat_context(chat_id)
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
