"""LangGraph tools for Thenvoi platform integration.

These tools allow LangGraph agents to interact with the Thenvoi platform.

CRITICAL: room_id comes from RunnableConfig (thread_id), NOT from LLM parameter.
The LLM should NOT decide which room to send to - it's determined by context.
"""

import json
import logging
from typing import List, TypedDict, cast
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from thenvoi.client.rest import (
    AsyncRestClient,
    ChatMessageRequest,
    ParticipantRequest,
)

logger = logging.getLogger(__name__)


class ThenvoiConfigurable(TypedDict):
    """Configurable dict with required thread_id for room context."""

    thread_id: str  # Room ID - always required for Thenvoi tools


def get_thenvoi_tools(client: AsyncRestClient, agent_id: str) -> List:
    """Get Thenvoi tools for LangGraph agents.

    Returns list of LangChain tools that interact with Thenvoi platform.

    IMPORTANT: room_id comes from RunnableConfig (thread_id), not from LLM!
    The adapter sets thread_id when invoking the agent, making it available
    to tools via the config parameter.

    Args:
        client: AsyncRestClient instance to use for API calls
        agent_id: Agent ID on Thenvoi platform

    Returns:
        List of LangChain tool functions
    """

    def _get_room_id_from_config(config: RunnableConfig) -> str:
        """Extract room_id from config.

        The adapter passes room_id as thread_id in config.
        LLM doesn't choose this - it comes from the message context!
        """
        configurable = config.get("configurable", {})
        thenvoi_config = cast(ThenvoiConfigurable, configurable)
        return thenvoi_config["thread_id"]

    def _format_participant_info(participant) -> str:
        """Format participant information with proper None handling.

        Args:
            participant: Participant object from API response

        Returns:
            Formatted participant string
        """
        participant_info = f"- {participant.type}: "
        if participant.type == "User":
            first_name = participant.first_name or ""
            last_name = participant.last_name or ""
            full_name = f"{first_name} {last_name}".strip()
            participant_info += f"{full_name or 'Unknown User'} (ID: {participant.id})"
        elif participant.type == "Agent":
            agent_name = (
                getattr(participant, "agent_name", None)
                or getattr(participant, "name", None)
                or "Unknown Agent"
            )
            participant_info += f"{agent_name} (ID: {participant.id})"
        else:
            participant_info += f"(ID: {participant.id})"
        return participant_info

    @tool
    async def send_message(
        content: str,
        mentions: str,
        config: RunnableConfig,  # Automatically injected by LangChain, hidden from LLM
    ) -> str:
        """Send a message to the current chat room.

        The room is automatically determined from context - you don't need to specify it.

        Args:
            content: The message content to send
            mentions: JSON string of mentions (at least one required). Format: '[{"id":"uuid","username":"name"}]'

        Returns:
            Success message with details
        """
        room_id = _get_room_id_from_config(config)

        mentions_list = json.loads(mentions)
        if not isinstance(mentions_list, list):
            raise ValueError(
                "Mentions must be a list of objects with 'id' and 'username'"
            )
        if len(mentions_list) == 0:
            raise ValueError(
                "At least one mention is required. Use get_participants to find users to mention."
            )
        for mention in mentions_list:
            if (
                not isinstance(mention, dict)
                or "id" not in mention
                or "username" not in mention
            ):
                raise ValueError("Each mention must have 'id' and 'username' fields")

        message_type = "text"  # Always use text type for agent messages

        logger.debug(
            f"[send_message] room_id: {room_id}, content_length: {len(content)}, mentions_count: {len(mentions_list)}"
        )

        message_request = ChatMessageRequest(
            content=content,
            message_type=message_type,
            mentions=mentions_list,
        )
        result = await client.agent_api.create_agent_chat_message(
            chat_id=room_id, message=message_request
        )

        return f"Message sent successfully to room {room_id}: {result}"

    @tool
    async def add_participant(
        participant_id: str,
        config: RunnableConfig,  # Automatically injected by LangChain, hidden from LLM
        role: str = "member",
    ) -> str:
        """Add a participant to the current chat room.

        The room is automatically determined from context.

        Args:
            participant_id: UUID of participant to add
            role: Role to assign (member, admin, owner)

        Returns:
            Success message with details
        """
        room_id = _get_room_id_from_config(config)

        logger.debug(f"[add_participant] room_id: {room_id}, role: {role}")

        participant_request = ParticipantRequest(
            participant_id=participant_id, role=role
        )
        result = await client.agent_api.add_agent_chat_participant(
            chat_id=room_id, participant=participant_request
        )

        return f"Successfully added participant {participant_id} to room {room_id} with role {role}: {result}"

    @tool
    async def remove_participant(
        participant_id: str,
        config: RunnableConfig,  # Automatically injected by LangChain, hidden from LLM
    ) -> str:
        """Remove a participant from the current chat room.

        The room is automatically determined from context.

        Args:
            participant_id: UUID of participant to remove

        Returns:
            Success message with details
        """
        room_id = _get_room_id_from_config(config)

        logger.debug(f"[remove_participant] room_id: {room_id}")

        result = await client.agent_api.remove_agent_chat_participant(
            chat_id=room_id, id=participant_id
        )

        return f"Successfully removed participant {participant_id} from room {room_id}: {result}"

    @tool
    async def get_participants(
        config: RunnableConfig,  # Automatically injected by LangChain, hidden from LLM
    ) -> str:
        """Get list of participants in the current chat room.

        The room is automatically determined from context.

        Returns:
            List of participants with details
        """
        room_id = _get_room_id_from_config(config)

        logger.debug(f"[get_participants] room_id: {room_id}")

        result = await client.agent_api.list_agent_chat_participants(chat_id=room_id)

        if not result or not result.data:
            return f"No participants found in room {room_id}"

        participants = [_format_participant_info(p) for p in result.data]

        return f"Participants in room {room_id}:\n" + "\n".join(participants)

    @tool
    async def list_available_participants(
        participant_type: str,
        config: RunnableConfig,  # Automatically injected by LangChain, hidden from LLM
    ) -> str:
        """List participants available to add to the current chat room.

        Shows participants NOT currently in the room.
        The room is automatically determined from context.

        Args:
            participant_type: Type of participant (User or Agent)

        Returns:
            List of available participants
        """
        room_id = _get_room_id_from_config(config)

        if participant_type not in ["User", "Agent"]:
            raise ValueError(
                f"participant_type must be 'User' or 'Agent', got: {participant_type}"
            )

        logger.debug(
            f"[list_available_participants] room_id: {room_id}, type: {participant_type}"
        )

        result = await client.agent_api.list_agent_peers(not_in_chat=room_id)

        if not result or not result.data:
            return f"No available {participant_type}s found for room {room_id}"

        # Filter by participant type
        filtered_peers = [p for p in result.data if p.type == participant_type]
        if not filtered_peers:
            return f"No available {participant_type}s found for room {room_id}"

        participants = [_format_participant_info(p) for p in filtered_peers]

        return (
            f"Found {len(participants)} available {participant_type}(s) for room {room_id}:\n"
            + "\n".join(participants)
        )

    return [
        send_message,
        add_participant,
        remove_participant,
        get_participants,
        list_available_participants,
    ]
