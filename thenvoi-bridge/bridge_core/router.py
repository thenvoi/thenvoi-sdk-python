"""@mention routing for bridge message dispatch."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from thenvoi.client.streaming import MessageCreatedPayload
    from thenvoi.platform.link import ThenvoiLink
    from thenvoi.runtime.tools import AgentTools

    from .handler import Handler
    from .session import SessionStore

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
        handlers: dict[str, Handler],
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
        if not mapping_str or not mapping_str.strip():
            raise ValueError("AGENT_MAPPING cannot be empty")

        result: dict[str, str] = {}
        for entry in mapping_str.split(","):
            entry = entry.strip()
            if not entry:
                continue

            parts = entry.split(":")
            if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
                raise ValueError(
                    f"Invalid AGENT_MAPPING entry: '{entry}'. "
                    "Expected format: 'agent_name:handler_name'"
                )
            username = parts[0].strip()
            if username in result:
                raise ValueError(
                    f"Duplicate username '{username}' in AGENT_MAPPING. "
                    "Each username must map to exactly one handler."
                )
            result[username] = parts[1].strip()

        if not result:
            raise ValueError("AGENT_MAPPING produced no valid entries")

        return result

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

        # Resolve which mentions map to registered handlers (deduplicate by username
        # so that repeated @mentions in one message don't dispatch the same handler twice)
        dispatch_list: list[tuple[str, str, Handler]] = []
        seen_usernames: set[str] = set()
        for mention in mentions:
            username = mention.username
            if username is None:
                # Platform sends username=null for agents; try handle/name fallback
                username = mention.handle or mention.name
                if username is None:
                    logger.debug(
                        "Could not resolve username for mention %s, skipping",
                        mention.id,
                    )
                    continue
            if username in seen_usernames:
                continue
            seen_usernames.add(username)

            handler_name = self._agent_mapping.get(username)

            if handler_name is None:
                logger.debug("No handler mapped for @%s", username)
                continue

            # handler must exist: the bridge validates at startup that
            # every mapped handler_name is present in self._handlers.
            handler = self._handlers[handler_name]

            dispatch_list.append((username, handler_name, handler))

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

        # Track session activity (sessions are per-room for lifecycle tracking).
        # Thread ID is passed directly from the payload to handlers so each
        # message carries the correct thread context regardless of session state.
        thread_id = payload.thread_id or room_id
        await self._session_store.get_or_create(room_id)

        # Dispatch to all matched handlers concurrently, collect failures
        async def _dispatch(
            username: str, handler_name: str, handler: Handler
        ) -> tuple[str, str, Exception] | None:
            logger.info(
                "Routing message to handler '%s' for @%s in room %s",
                handler_name,
                username,
                room_id,
            )
            try:
                coro = handler.handle(
                    content=payload.content,
                    room_id=room_id,
                    thread_id=thread_id,
                    message_id=payload.id,
                    sender_id=payload.sender_id,
                    sender_name=sender_name,
                    sender_type=payload.sender_type,
                    mentioned_agent=username,
                    tools=tools,
                )
                if self._handler_timeout is not None:
                    await asyncio.wait_for(coro, timeout=self._handler_timeout)
                else:
                    await coro
            except asyncio.TimeoutError:
                logger.error(
                    "Handler '%s' timed out after %.1fs for @%s in room %s",
                    handler_name,
                    self._handler_timeout,
                    username,
                    room_id,
                )
                return (
                    handler_name,
                    username,
                    TimeoutError(f"timed out after {self._handler_timeout}s"),
                )
            except asyncio.CancelledError:
                # Let cancellation propagate so asyncio.gather cancels all
                # sibling handlers.  This is the shutdown path — the bridge
                # consume loop cancels handle_fut when shutdown is requested.
                raise
            except Exception as e:
                logger.exception(
                    "Handler '%s' failed for @%s in room %s",
                    handler_name,
                    username,
                    room_id,
                )
                return (handler_name, username, e)
            return None

        # Note: return_exceptions is intentionally omitted.  _dispatch()
        # catches all exceptions internally except CancelledError, which must
        # propagate so asyncio.gather cancels sibling handlers on shutdown.
        results = await asyncio.gather(
            *[_dispatch(u, n, h) for u, n, h in dispatch_list]
        )
        errors = [r for r in results if r is not None]

        # Mark processed/failed once for the whole message.
        # Wrap in try-except so API failures don't crash the router.
        if errors:
            all_failed = len(errors) == len(dispatch_list)
            # Full details for platform operators (mark_failed) and logs
            internal_summaries = [
                f"'{name}' (@{user}): {str(err)[:_MAX_ERR_LEN]}{'...' if len(str(err)) > _MAX_ERR_LEN else ''}"
                for name, user, err in errors
            ]
            internal_message = "; ".join(internal_summaries)
            # Sanitized message for chat users (send_event) — avoids
            # leaking internal handler names
            user_summaries = [f"@{user}: processing failed" for _, user, _ in errors]
            user_message = "; ".join(user_summaries)

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
