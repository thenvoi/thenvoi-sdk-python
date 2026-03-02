"""Mention replacement integration scenarios for formatted history."""

from __future__ import annotations

from thenvoi_rest import ChatMessageRequest
from thenvoi_rest.types import (
    ChatMessageRequestMentionsItem as Mention,
    ParticipantRequest,
)

from tests.integration.history_converters.support import (
    create_test_chat,
    logger,
)
from tests.support.integration.markers import requires_api

@requires_api
class TestMentionReplacementIntegration:
    """Integration tests for UUID mention replacement in message history."""

    async def test_replaces_uuid_mentions_with_handles(self, api_client, no_clean):
        """Verify platform stores mentions as UUIDs and SDK converts back to handles.

        Flow:
        1. Send message with @handle mention
        2. Platform prepends @[[uuid]] to the content (keeps original handle too)
        3. SDK's format_history_for_llm converts @[[uuid]] to @handle
        """
        from thenvoi.runtime.formatters import format_history_for_llm

        async with create_test_chat(api_client, skip_cleanup=no_clean) as (
            chat_id,
            _agent_me,
        ):
            # Get a peer to add as participant
            peers = await api_client.agent_api_peers.list_agent_peers()
            assert peers.data and len(peers.data) > 0, "Need at least one peer"
            peer = peers.data[0]

            # Get participants to find peer's handle
            await api_client.agent_api_participants.add_agent_chat_participant(
                chat_id,
                participant=ParticipantRequest(participant_id=peer.id, role="member"),
            )

            participants_response = (
                await api_client.agent_api_participants.list_agent_chat_participants(
                    chat_id
                )
            )
            participants = [
                {
                    "id": p.id,
                    "name": p.name,
                    "type": p.type,
                    "handle": getattr(p, "handle", None),
                }
                for p in participants_response.data
            ]

            peer_participant = next(
                (p for p in participants if p["id"] == peer.id), None
            )
            assert peer_participant is not None, "Peer should be in participants"
            peer_handle = peer_participant.get("handle")
            assert peer_handle, "Peer must have a handle for this test"
            logger.info(
                "Added peer: %s (id: %s, handle: %s)", peer.name, peer.id, peer_handle
            )

            # === STEP 1: Send message with mention in array (not in content) ===
            # The platform will prepend @[[uuid]] to the content based on mentions array
            message_content = "Hey, can you help me?"
            await api_client.agent_api_messages.create_agent_chat_message(
                chat_id,
                message=ChatMessageRequest(
                    content=message_content,
                    mentions=[Mention(id=peer.id, name=peer.name)],
                ),
            )
            logger.info("Sent message: %s (with mention in array)", message_content)

            # === STEP 2: Verify raw history contains UUID format ===
            context_response = (
                await api_client.agent_api_context.get_agent_chat_context(chat_id)
            )
            raw_history = [msg.model_dump() for msg in context_response.data]
            logger.info("Fetched %d messages from platform", len(raw_history))

            raw_message = next(
                (m for m in raw_history if "can you help me" in m.get("content", "")),
                None,
            )
            assert raw_message is not None, "Should find the message in raw history"
            raw_content = raw_message["content"]
            logger.info("Raw content from platform: %s", raw_content)

            # Platform prepends @[[uuid]] to content
            assert f"@[[{peer.id}]]" in raw_content, (
                f"Raw content should contain UUID mention @[[{peer.id}]], "
                f"got: {raw_content}"
            )

            # === STEP 3: Verify formatted history converts UUID to handle ===
            formatted_history = format_history_for_llm(
                raw_history, participants=participants
            )

            formatted_message = next(
                (m for m in formatted_history if "can you help me" in m["content"]),
                None,
            )
            assert formatted_message is not None, "Should find formatted message"
            formatted_content = formatted_message["content"]

            # UUID should be replaced with @handle
            assert f"@[[{peer.id}]]" not in formatted_content, (
                f"Formatted content should NOT contain UUID @[[{peer.id}]], "
                f"got: {formatted_content}"
            )
            assert f"@{peer_handle}" in formatted_content, (
                f"Formatted content should contain @{peer_handle}, "
                f"got: {formatted_content}"
            )

            logger.info(
                "SUCCESS: Platform stored @[[%s]], SDK converted to @%s",
                peer.id,
                peer_handle,
            )
            logger.info("  Raw:       %s", raw_content)
            logger.info("  Formatted: %s", formatted_content)
