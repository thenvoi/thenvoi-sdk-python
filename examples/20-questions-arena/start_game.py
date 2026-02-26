"""Start a 20 Questions Arena game as a user by creating a room, adding all agents, and sending a message."""

from __future__ import annotations

import asyncio
import sys

from thenvoi_rest import AsyncRestClient, ChatMessageRequest, ParticipantRequest
from thenvoi_rest.human_api_chats.types.create_my_chat_room_request_chat import (
    CreateMyChatRoomRequestChat,
)
from thenvoi_rest.types import ChatMessageRequestMentionsItem as Mention

THINKER_AGENT_ID = "3f39b82c-b6ab-4300-bfd6-f3ff5792e348"
GUESSER_AGENT_IDS = [
    "f00d355b-df9b-4659-aa8e-9f5abad4fb5d",  # Guesser GPT 5-nano
    "8b6510bc-b678-4ee4-8f71-d84de19b43b8",  # Guesser GPT 5.2 pro
    "c10a872d-075a-488b-8611-8d9eda3a0b94",  # Guesser Haiku 4.5
]
REST_URL = "https://app.thenvoi.com"


async def start_game(user_api_key: str) -> str:
    client = AsyncRestClient(api_key=user_api_key, base_url=REST_URL)

    # Create a chat room
    chat_resp = await client.human_api_chats.create_my_chat_room(
        chat=CreateMyChatRoomRequestChat(),
    )
    chat_id = chat_resp.data.id
    print(f"Created chat room: {chat_id}")

    # Add thinker agent to the room
    await client.human_api_participants.add_my_chat_participant(
        chat_id,
        participant=ParticipantRequest(
            participant_id=THINKER_AGENT_ID,
            role="member",
        ),
    )
    print("Added Thinker to room")

    # Add all guesser agents to the room
    for guesser_id in GUESSER_AGENT_IDS:
        await client.human_api_participants.add_my_chat_participant(
            chat_id,
            participant=ParticipantRequest(
                participant_id=guesser_id,
                role="member",
            ),
        )
    print(f"Added {len(GUESSER_AGENT_IDS)} guessers to room")

    # Get participants to build mention list
    parts_resp = await client.human_api_participants.list_my_chat_participants(chat_id)

    # Build a map of agent_id -> participant info
    agents = {}
    for p in parts_resp.data:
        pid = str(p.id)
        agents[pid] = p

    thinker = agents.get(THINKER_AGENT_ID)
    thinker_name = thinker.name if thinker else "Thinker"
    thinker_id = str(thinker.id) if thinker else THINKER_AGENT_ID

    # Build mentions for thinker + all guessers
    mentions = [Mention(id=thinker_id, name=thinker_name)]
    guesser_names = []
    for gid in GUESSER_AGENT_IDS:
        g = agents.get(gid)
        if g:
            mentions.append(Mention(id=str(g.id), name=g.name))
            guesser_names.append(g.name)

    print(f"Thinker: {thinker_name}")
    print(f"Guessers: {', '.join(guesser_names)}")

    # Send "start a game" message mentioning everyone
    msg_resp = await client.human_api_messages.send_my_chat_message(
        chat_id,
        message=ChatMessageRequest(
            content=f"@{thinker_name} start a new game of 20 questions with all the guessers in this room!",
            mentions=mentions,
        ),
    )
    print(f"Sent start message: {msg_resp.data.id}")
    return chat_id


async def main() -> None:
    user_api_key = sys.argv[1] if len(sys.argv) > 1 else ""
    if not user_api_key:
        print("Usage: python start_game.py <user_api_key>")
        sys.exit(1)

    chat_id = await start_game(user_api_key)
    print(f"\nGame started in room: {chat_id}")


if __name__ == "__main__":
    asyncio.run(main())
