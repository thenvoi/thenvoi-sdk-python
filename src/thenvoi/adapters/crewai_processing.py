"""CrewAI adapter message-processing helpers."""

from __future__ import annotations

import logging
from typing import Any, Protocol

from thenvoi.converters.crewai import CrewAIMessages
from thenvoi.core.protocols import MessagingDispatchToolsProtocol
from thenvoi.core.types import PlatformMessage

logger = logging.getLogger(__name__)


class CrewAIProcessingProtocol(Protocol):
    """Adapter state/method surface required for processing helpers."""

    _crewai_agent: Any
    _message_history: dict[str, list[dict[str, Any]]]

    def stage_room_history(
        self,
        history_by_room: dict[str, list[dict[str, Any]]],
        *,
        room_id: str,
        is_session_bootstrap: bool,
        hydrated_history: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]]: ...

    def apply_metadata_updates(
        self,
        messages: list[dict[str, str]],
        *,
        participants_msg: str | None,
        contacts_msg: str | None,
        make_entry: Any,
    ) -> int: ...

    async def report_adapter_error(
        self,
        tools: MessagingDispatchToolsProtocol,
        *,
        error: Exception,
        operation: str,
    ) -> None: ...


def build_backstory(
    *,
    agent_name: str,
    backstory: str | None,
    custom_section: str | None,
    platform_instructions: str,
) -> str:
    """Compose CrewAI backstory from user config and platform instructions."""
    backstory_parts: list[str] = []
    if backstory:
        backstory_parts.append(backstory)
    else:
        backstory_parts.append(
            f"You are {agent_name}, a collaborative AI agent on the Thenvoi platform."
        )

    if custom_section:
        backstory_parts.append(custom_section)
    backstory_parts.append(platform_instructions)
    return "\n\n".join(backstory_parts)


async def process_message(
    adapter: CrewAIProcessingProtocol,
    *,
    msg: PlatformMessage,
    tools: MessagingDispatchToolsProtocol,
    history: CrewAIMessages,
    participants_msg: str | None,
    contacts_msg: str | None,
    is_session_bootstrap: bool,
    room_id: str,
) -> None:
    """Process a single room message with optional bootstrap history."""
    room_history = adapter.stage_room_history(
        adapter._message_history,
        room_id=room_id,
        is_session_bootstrap=is_session_bootstrap,
        hydrated_history=(
            [{"role": h["role"], "content": h["content"]} for h in history]
            if history
            else None
        ),
    )

    if is_session_bootstrap:
        if history:
            logger.info(
                "Room %s: Loaded %s historical messages",
                room_id,
                len(history),
            )
        else:
            logger.info("Room %s: No historical messages found", room_id)

    messages: list[dict[str, str]] = []
    if is_session_bootstrap and room_history:
        history_text = "\n".join(f"{m['role']}: {m['content']}" for m in room_history)
        messages.append(
            {
                "role": "user",
                "content": f"[Previous conversation:]\n{history_text}",
            }
        )

    system_update_count = adapter.apply_metadata_updates(
        messages,
        participants_msg=participants_msg,
        contacts_msg=contacts_msg,
        make_entry=lambda update: {"role": "user", "content": update},
    )
    if system_update_count:
        logger.info(
            "Room %s: Injected %d system updates",
            room_id,
            system_update_count,
        )

    user_message = msg.format_for_llm()
    messages.append({"role": "user", "content": user_message})
    room_history.append({"role": "user", "content": user_message})

    logger.info(
        "Room %s: Processing with %s messages (first_msg=%s)",
        room_id,
        len(room_history),
        is_session_bootstrap,
    )

    try:
        # CrewAI's stubs claim only str prompt input, but runtime accepts
        # OpenAI-style message lists for multi-turn context.
        result = await adapter._crewai_agent.kickoff_async(messages)  # type: ignore[arg-type]
        if result and result.raw:
            room_history.append({"role": "assistant", "content": result.raw})
        logger.info(
            "Room %s: CrewAI agent completed (output_length=%s)",
            room_id,
            len(result.raw) if result and result.raw else 0,
        )
    except Exception as error:
        logger.error("Error processing message: %s", error, exc_info=True)
        await adapter.report_adapter_error(
            tools,
            error=error,
            operation="error_event",
        )
        raise

    logger.debug(
        "Message %s processed successfully (history now has %s messages)",
        msg.id,
        len(room_history),
    )

