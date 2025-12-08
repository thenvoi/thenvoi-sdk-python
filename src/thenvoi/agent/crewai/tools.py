"""CrewAI tools for Thenvoi platform integration.

These tools allow CrewAI agents to interact with the Thenvoi platform.

CRITICAL: room_id comes from ThenvoiContext, NOT from LLM parameter.
The LLM should NOT decide which room to send to - it's determined by context.
"""

import asyncio
import json
import logging
from typing import List, Optional, Type
from pydantic import BaseModel, Field
from crewai.tools import BaseTool

from thenvoi.client.rest import (
    AsyncRestClient,
    ChatMessageRequest,
    AddChatParticipantRequestParticipant,
)

logger = logging.getLogger(__name__)


class ThenvoiContext:
    """Context holder for room_id - equivalent to RunnableConfig thread_id in LangGraph."""

    _room_id: Optional[str] = None

    @classmethod
    def set_room_id(cls, room_id: str):
        cls._room_id = room_id

    @classmethod
    def get_room_id(cls) -> str:
        if cls._room_id is None:
            raise ValueError(
                "Room ID not set. Ensure ThenvoiContext.set_room_id() is called before using tools."
            )
        return cls._room_id

    @classmethod
    def clear(cls):
        cls._room_id = None


def get_thenvoi_tools(client: AsyncRestClient, agent_id: str) -> List[BaseTool]:
    """Get Thenvoi tools for CrewAI agents.

    Returns list of CrewAI tools that interact with Thenvoi platform.

    IMPORTANT: room_id comes from ThenvoiContext, not from LLM!
    The adapter sets ThenvoiContext.set_room_id() when invoking the agent,
    making room context available to tools.

    Args:
        client: AsyncRestClient instance to use for API calls
        agent_id: Agent ID on Thenvoi platform

    Returns:
        List of CrewAI tool instances
    """

    def _run_async(coro):
        """Run an async coroutine from sync context.

        CrewAI calls sync _run() methods, so we need to bridge to async.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # We're in an async context, use nest_asyncio
            import nest_asyncio

            nest_asyncio.apply()
            return asyncio.get_event_loop().run_until_complete(coro)
        else:
            # No running loop, create a new one
            return asyncio.run(coro)

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

    # Input schemas for CrewAI tools
    class SendMessageInput(BaseModel):
        content: str = Field(description="The message content to send")
        mentions: str = Field(
            description='JSON string of mentions (at least one required). Format: \'[{"id":"uuid","username":"name"}]\''
        )

    class AddParticipantInput(BaseModel):
        participant_id: str = Field(description="UUID of participant to add")
        role: str = Field(
            default="member", description="Role to assign (member, admin, owner)"
        )

    class RemoveParticipantInput(BaseModel):
        participant_id: str = Field(description="UUID of participant to remove")

    class GetParticipantsInput(BaseModel):
        pass

    class ListAvailableParticipantsInput(BaseModel):
        participant_type: str = Field(description="Type of participant (User or Agent)")

    # CrewAI Tool: send_message
    class SendMessageTool(BaseTool):
        name: str = "send_message"
        description: str = """Send a message to the current chat room.

The room is automatically determined from context - you don't need to specify it.

Args:
    content: The message content to send
    mentions: JSON string of mentions (at least one required). Format: '[{"id":"uuid","username":"name"}]'

Returns:
    Success message with details"""
        args_schema: Type[BaseModel] = SendMessageInput

        def _run(self, *args, **kwargs) -> str:
            content: str = kwargs.get("content", args[0] if args else "")
            mentions: str = kwargs.get("mentions", args[1] if len(args) > 1 else "[]")
            room_id = ThenvoiContext.get_room_id()

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
                    raise ValueError(
                        "Each mention must have 'id' and 'username' fields"
                    )

            message_type = "text"  # Always use text type for agent messages

            logger.debug(
                f"[send_message] room_id: {room_id}, content_length: {len(content)}, mentions_count: {len(mentions_list)}"
            )

            async def _send():
                message_request = ChatMessageRequest(
                    content=content,
                    message_type=message_type,
                    mentions=mentions_list,
                )
                return await client.chat_messages.create_chat_message(
                    chat_id=room_id, message=message_request
                )

            result = _run_async(_send())
            return f"Message sent successfully to room {room_id}: {result}"

    # CrewAI Tool: add_participant
    class AddParticipantTool(BaseTool):
        name: str = "add_participant"
        description: str = """Add a participant to the current chat room.

The room is automatically determined from context.

Args:
    participant_id: UUID of participant to add
    role: Role to assign (member, admin, owner)

Returns:
    Success message with details"""
        args_schema: Type[BaseModel] = AddParticipantInput

        def _run(self, *args, **kwargs) -> str:
            participant_id: str = kwargs.get("participant_id", args[0] if args else "")
            role: str = kwargs.get("role", args[1] if len(args) > 1 else "member")
            room_id = ThenvoiContext.get_room_id()

            logger.debug(f"[add_participant] room_id: {room_id}, role: {role}")

            async def _add():
                participant_request = AddChatParticipantRequestParticipant(
                    participant_id=participant_id, role=role
                )
                return await client.chat_participants.add_chat_participant(
                    chat_id=room_id, participant=participant_request
                )

            result = _run_async(_add())
            return f"Successfully added participant {participant_id} to room {room_id} with role {role}: {result}"

    # CrewAI Tool: remove_participant
    class RemoveParticipantTool(BaseTool):
        name: str = "remove_participant"
        description: str = """Remove a participant from the current chat room.

The room is automatically determined from context.

Args:
    participant_id: UUID of participant to remove

Returns:
    Success message with details"""
        args_schema: Type[BaseModel] = RemoveParticipantInput

        def _run(self, *args, **kwargs) -> str:
            participant_id: str = kwargs.get("participant_id", args[0] if args else "")
            room_id = ThenvoiContext.get_room_id()

            logger.debug(f"[remove_participant] room_id: {room_id}")

            async def _remove():
                return await client.chat_participants.remove_chat_participant(
                    chat_id=room_id, id=participant_id
                )

            result = _run_async(_remove())
            return f"Successfully removed participant {participant_id} from room {room_id}: {result}"

    # CrewAI Tool: get_participants
    class GetParticipantsTool(BaseTool):
        name: str = "get_participants"
        description: str = """Get list of participants in the current chat room.

The room is automatically determined from context.

Returns:
    List of participants with details"""
        args_schema: Type[BaseModel] = GetParticipantsInput

        def _run(self, *args, **kwargs) -> str:
            room_id = ThenvoiContext.get_room_id()

            logger.debug(f"[get_participants] room_id: {room_id}")

            async def _get():
                return await client.chat_participants.list_chat_participants(
                    chat_id=room_id
                )

            result = _run_async(_get())

            if not result or not result.data:
                return f"No participants found in room {room_id}"

            participants = [_format_participant_info(p) for p in result.data]

            return f"Participants in room {room_id}:\n" + "\n".join(participants)

    # CrewAI Tool: list_available_participants
    class ListAvailableParticipantsTool(BaseTool):
        name: str = "list_available_participants"
        description: str = """List participants available to add to the current chat room.

Shows participants NOT currently in the room.
The room is automatically determined from context.

Args:
    participant_type: Type of participant (User or Agent)

Returns:
    List of available participants"""
        args_schema: Type[BaseModel] = ListAvailableParticipantsInput

        def _run(self, *args, **kwargs) -> str:
            participant_type: str = kwargs.get(
                "participant_type", args[0] if args else ""
            )
            room_id = ThenvoiContext.get_room_id()

            if participant_type not in ["User", "Agent"]:
                raise ValueError(
                    f"participant_type must be 'User' or 'Agent', got: {participant_type}"
                )

            logger.debug(
                f"[list_available_participants] room_id: {room_id}, type: {participant_type}"
            )

            async def _list():
                return await client.chat_participants.get_available_chat_participants(
                    chat_id=room_id, participant_type=participant_type
                )

            result = _run_async(_list())

            if not result or not result.data:
                return f"No available {participant_type}s found for room {room_id}"

            participants = [_format_participant_info(p) for p in result.data]

            return (
                f"Found {len(participants)} available {participant_type}(s) for room {room_id}:\n"
                + "\n".join(participants)
            )

    return [
        SendMessageTool(),
        AddParticipantTool(),
        RemoveParticipantTool(),
        GetParticipantsTool(),
        ListAvailableParticipantsTool(),
    ]
