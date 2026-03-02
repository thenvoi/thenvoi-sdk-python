"""@mention routing for bridge message dispatch."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from thenvoi.client.streaming import MessageCreatedPayload
    from thenvoi.platform.link import ThenvoiLink
    from thenvoi.runtime.tools import AgentTools

    from .handler import BaseHandler
    from .session import SessionStore

from .agent_mapping import parse_agent_mapping
from .platform_message_factory import build_platform_message
from .route_dispatch import (
    build_dispatch_targets,
    execute_dispatch_targets,
    summarize_dispatch_failures,
)

logger = logging.getLogger(__name__)

# Maximum length for individual handler error messages sent to the platform.
_MAX_ERR_LEN = 500


class MentionRouter:
    """Routes @mention messages to registered handlers.

    Parses AGENT_MAPPING config to map agent usernames to handler instances.
    When a message arrives, inspects mention metadata and dispatches to
    the appropriate handler(s). Integrates with ThenvoiLink for message
    lifecycle marking (processing/processed/failed).
    """

    def __init__(
        self,
        agent_mapping: dict[str, str],
        handlers: dict[str, BaseHandler],
        session_store: SessionStore,
        agent_id: str,
        link: ThenvoiLink,
        handler_timeout: float | None = None,
    ) -> None:
        """Initialize the mention router.

        Args:
            agent_mapping: Map of agent username -> handler name.
            handlers: Map of handler name -> handler instance.
            session_store: Session store for tracking conversations.
            agent_id: The bridge agent's own ID (for self-message filtering).
            link: ThenvoiLink for message lifecycle marking.
            handler_timeout: Optional timeout in seconds for handler execution.
                None means no timeout (handlers can run indefinitely).
        """
        self._agent_mapping = agent_mapping
        self._handlers = handlers
        self._session_store = session_store
        self._agent_id = agent_id
        self._link = link
        self._handler_timeout = handler_timeout

    def set_link(self, link: ThenvoiLink) -> None:
        """Rebind ThenvoiLink dependency (used by bridge reconnect/test wiring)."""
        self._link = link

    @staticmethod
    def parse_agent_mapping(mapping_str: str) -> dict[str, str]:
        """Parse AGENT_MAPPING environment variable.

        Format: "alice:alice_handler,bob:bob_handler"

        Args:
            mapping_str: Comma-separated key:value pairs.

        Returns:
            Dict mapping agent usernames to handler names.

        Raises:
            ValueError: If the mapping string is empty or has invalid entries.
        """
        return parse_agent_mapping(mapping_str)

    async def route(
        self,
        payload: MessageCreatedPayload,
        room_id: str,
        tools: AgentTools,
        sender_name: str | None = None,
    ) -> None:
        """Route a message to handlers based on @mentions.

        Filters self-messages, then checks each mention against the
        agent mapping. For each match, creates/updates a session and
        dispatches to the handler. Marks the message as
        processing/processed/failed once per message (not per handler).

        Args:
            payload: The message payload from the platform.
            room_id: The room where the message was received.
            tools: AgentTools instance for the handler to send responses.
            sender_name: Display name of the sender, or None if unresolvable.
        """
        # Filter self-messages
        if payload.sender_id == self._agent_id:
            logger.debug("Skipping self-message from %s", payload.sender_id)
            return

        # Extract mentions from metadata
        if not payload.metadata or not payload.metadata.mentions:
            logger.debug("No mentions in message %s, skipping", payload.id)
            return
        mentions = payload.metadata.mentions

        dispatch_list = build_dispatch_targets(
            mentions,
            agent_mapping=self._agent_mapping,
            handlers=self._handlers,
            logger=logger,
        )
        if not dispatch_list:
            logger.debug("No mapped handlers for mentions in message %s", payload.id)
            return

        # Mark processing once for the whole message.
        # Note: ThenvoiLink.mark_processing swallows exceptions internally,
        # so this try/except is defensive against future changes to the link.
        try:
            await self._link.mark_processing(room_id, payload.id)
        except Exception:
            logger.warning(
                "Failed to mark message %s as processing",
                payload.id,
                exc_info=True,
            )

        platform_message = build_platform_message(
            payload,
            room_id,
            sender_name=sender_name,
        )
        await self._session_store.get_or_create(room_id)

        errors = await execute_dispatch_targets(
            dispatch_list,
            platform_message=platform_message,
            tools=tools,
            room_id=room_id,
            handler_timeout=self._handler_timeout,
            logger=logger,
        )

        # Mark processed/failed once for the whole message.
        # Wrap in try-except so API failures don't crash the router.
        if errors:
            all_failed, internal_message, user_message = summarize_dispatch_failures(
                errors,
                total_targets=len(dispatch_list),
                max_error_len=_MAX_ERR_LEN,
            )

            if all_failed:
                # Defensive try/except: ThenvoiLink swallows exceptions
                # internally, kept for robustness against future changes.
                try:
                    await self._link.mark_failed(room_id, payload.id, internal_message)
                except Exception:
                    logger.warning(
                        "Failed to mark message %s as failed",
                        payload.id,
                        exc_info=True,
                    )
            else:
                # Partial success: some handlers succeeded, some failed.
                # Mark as processed (the message was handled) and report
                # failures via send_event so users see them. Full details
                # are logged for operators.
                logger.warning(
                    "Partial failure for message %s: %s", payload.id, internal_message
                )
                # Defensive try/except: see comment on mark_processing above.
                try:
                    await self._link.mark_processed(room_id, payload.id)
                except Exception:
                    logger.warning(
                        "Failed to mark message %s as processed",
                        payload.id,
                        exc_info=True,
                    )

            try:
                await tools.send_event(
                    content=f"Handler failures: {user_message}",
                    message_type="error",
                )
            except Exception:
                logger.warning(
                    "Failed to send error event to room %s",
                    room_id,
                    exc_info=True,
                )
        else:
            # Defensive try/except: see comment on mark_processing above.
            try:
                await self._link.mark_processed(room_id, payload.id)
            except Exception:
                logger.warning(
                    "Failed to mark message %s as processed",
                    payload.id,
                    exc_info=True,
                )
