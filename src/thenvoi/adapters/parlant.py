"""
Parlant adapter using the official Parlant SDK directly.

This adapter integrates the Parlant framework (https://github.com/emcie-co/parlant)
with the Thenvoi platform using the SDK's internal components (no HTTP).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from thenvoi.core.protocols import AgentToolsProtocol
from thenvoi.core.simple_adapter import SimpleAdapter
from thenvoi.core.types import PlatformMessage
from thenvoi.converters.parlant import ParlantHistoryConverter, ParlantMessages
from thenvoi.integrations.parlant.tools import set_session_tools
from thenvoi.runtime.prompts import render_system_prompt

if TYPE_CHECKING:
    import parlant.sdk as p
    from parlant.core.application import Application
    from parlant.core.sessions import SessionId

logger = logging.getLogger(__name__)


class ParlantAdapter(SimpleAdapter[ParlantMessages]):
    """
    Parlant adapter using the official Parlant SDK directly.

    This adapter uses the Parlant SDK's internal components for message processing
    without HTTP communication. It integrates directly with the Parlant engine.

    Example:
        import parlant.sdk as p

        async with p.Server() as server:
            agent = await server.create_agent(
                name="Assistant",
                description="A helpful assistant",
            )

            adapter = ParlantAdapter(
                server=server,
                parlant_agent=agent,
            )

            thenvoi_agent = Agent.create(
                adapter=adapter,
                agent_id="...",
                api_key="...",
            )
            await thenvoi_agent.run()
    """

    def __init__(
        self,
        server: p.Server,
        parlant_agent: p.Agent,
        system_prompt: str | None = None,
        custom_section: str | None = None,
        history_converter: ParlantHistoryConverter | None = None,
    ):
        """
        Initialize the Parlant SDK adapter.

        Args:
            server: The Parlant SDK Server instance
            parlant_agent: The Parlant Agent instance
            system_prompt: Full system prompt override
            custom_section: Custom instructions appended to agent description
            history_converter: Custom history converter (optional)
        """
        super().__init__(
            history_converter=history_converter or ParlantHistoryConverter()
        )

        self._server = server
        self._parlant_agent = parlant_agent
        self.system_prompt = system_prompt
        self.custom_section = custom_section

        # Parlant application (accessed via container)
        self._app: Application | None = None

        # Per-room session mapping (room_id -> parlant session_id)
        self._room_sessions: dict[str, SessionId] = {}

        # Per-room customer mapping (room_id -> parlant customer_id)
        self._room_customers: dict[str, str] = {}

        # Rendered system prompt (set after start)
        self._system_prompt: str = ""

    async def on_started(self, agent_name: str, agent_description: str) -> None:
        """Initialize after agent metadata is fetched."""
        await super().on_started(agent_name, agent_description)

        # Render system prompt
        self._system_prompt = self.system_prompt or render_system_prompt(
            agent_name=agent_name,
            agent_description=agent_description,
            custom_section=self.custom_section or "",
        )

        # Get Application from Parlant container
        try:
            from parlant.core.application import Application

            self._app = self._server.container[Application]
            logger.info(
                f"Parlant SDK adapter started for agent: {agent_name} "
                f"(parlant_agent_id={self._parlant_agent.id})"
            )
        except Exception as e:
            logger.error(f"Failed to get Parlant Application: {e}", exc_info=True)
            raise

    async def on_message(
        self,
        msg: PlatformMessage,
        tools: AgentToolsProtocol,
        history: ParlantMessages,
        participants_msg: str | None,
        *,
        is_session_bootstrap: bool,
        room_id: str,
    ) -> None:
        """
        Handle incoming message using the Parlant SDK directly.

        Uses Parlant's internal Application for session and message management.
        """
        logger.debug(f"Handling message {msg.id} in room {room_id}")

        if not self._app:
            logger.error("Parlant Application not initialized")
            return

        app = self._app
        sender_name = msg.sender_name or msg.sender_id or "User"

        # Get or create Parlant session for this room (need session_id first)
        session_id = await self._get_or_create_session(room_id, sender_name)
        session_id_str = str(session_id)

        # Set tools for this session (keyed by session_id for cross-task access)
        set_session_tools(session_id_str, tools)
        logger.info(f"Room {room_id}: Set tools for session_id={session_id_str}")

        # On bootstrap, inject historical context
        if is_session_bootstrap and history:
            injected = await self._inject_history(session_id, history)
            logger.info(f"Room {room_id}: Injected {injected} messages from history")

        # Send customer message to Parlant
        user_message = msg.format_for_llm()
        logger.info(f"Room {room_id}: Sending message to Parlant: {user_message[:100]}...")

        try:
            from parlant.core.app_modules.sessions import Moderation
            from parlant.core.sessions import EventSource

            # Create customer message event (triggers processing)
            logger.info(f"Room {room_id}: Creating customer message event...")
            event = await app.sessions.create_customer_message(
                session_id=session_id,
                moderation=Moderation.NONE,
                message=user_message,
                source=EventSource.CUSTOMER,
                trigger_processing=True,
                metadata=None,
            )
            logger.info(f"Room {room_id}: Customer message created, offset={event.offset}")

            # Wait for and process agent response
            await self._process_agent_response(
                session_id=session_id,
                room_id=room_id,
                min_offset=event.offset,
                tools=tools,
                sender_name=sender_name,
            )

        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)
            await self._report_error(tools, str(e))
            raise
        finally:
            # Clear tools after message processing
            set_session_tools(session_id_str, None)
            logger.info(f"Room {room_id}: Cleared tools for session_id={session_id_str}")

        logger.debug(f"Message {msg.id} processed successfully")

    async def _get_or_create_session(
        self,
        room_id: str,
        customer_name: str,
    ) -> SessionId:
        """Get existing session for room or create a new one."""
        if room_id in self._room_sessions:
            return self._room_sessions[room_id]

        if not self._app:
            raise RuntimeError("Parlant Application not initialized")

        app = self._app
        logger.info(f"Creating Parlant session for room: {room_id}")

        # Create or get customer
        customer_id = await self._get_or_create_customer(room_id, customer_name)

        # Create session
        session = await app.sessions.create(
            customer_id=customer_id,
            agent_id=self._parlant_agent.id,
            title=f"Thenvoi Room {room_id[:8]}",
        )

        self._room_sessions[room_id] = session.id
        logger.info(f"Session created: {session.id} for room {room_id}")

        return session.id

    async def _get_or_create_customer(
        self,
        room_id: str,
        customer_name: str,
    ) -> Any:
        """Get or create a Parlant customer."""
        if room_id in self._room_customers:
            return self._room_customers[room_id]

        # Create customer via server
        customer = await self._server.create_customer(
            name=customer_name,
            id=f"thenvoi-{room_id[:8]}",
        )

        self._room_customers[room_id] = customer.id
        return customer.id

    async def _inject_history(
        self,
        session_id: SessionId,
        history: ParlantMessages,
    ) -> int:
        """Inject historical messages into a Parlant session.

        Only injects COMPLETE exchanges (user message + assistant response).
        User messages without a following assistant response are NOT injected,
        as they represent pending/unanswered questions that should be handled
        by the current message flow.
        """
        if not self._app:
            return 0

        app = self._app
        from parlant.core.app_modules.sessions import Moderation
        from parlant.core.sessions import EventKind, EventSource

        # First, filter to only complete exchanges
        # A user message is only injected if it has a following assistant response
        complete_history: ParlantMessages = []
        i = 0
        while i < len(history):
            msg = history[i]
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "user" and content:
                # Check if there's a following assistant response
                if i + 1 < len(history) and history[i + 1].get("role") == "assistant":
                    # Complete exchange - include both
                    complete_history.append(msg)
                    complete_history.append(history[i + 1])
                    i += 2
                else:
                    # User message without response - skip (it's pending)
                    logger.debug(f"Skipping unanswered user message: {content[:50]}...")
                    i += 1
            elif role == "assistant" and content:
                # Standalone assistant message (rare) - include it
                complete_history.append(msg)
                i += 1
            else:
                i += 1

        # Now inject the filtered history
        count = 0
        for hist in complete_history:
            role = hist.get("role", "user")
            content = hist.get("content", "")

            if not content:
                continue

            try:
                if role == "user":
                    await app.sessions.create_customer_message(
                        session_id=session_id,
                        moderation=Moderation.NONE,
                        message=content,
                        source=EventSource.CUSTOMER,
                        trigger_processing=False,
                        metadata={"historical": True},
                    )
                    count += 1
                elif role == "assistant":
                    # Parlant requires participant info for AI_AGENT messages
                    sender = hist.get("sender", self.agent_name or "Assistant")
                    await app.sessions.create_event(
                        session_id=session_id,
                        kind=EventKind.MESSAGE,
                        source=EventSource.AI_AGENT,
                        data={
                            "message": content,
                            "participant": {"display_name": sender},
                        },
                        metadata={"historical": True},
                        trigger_processing=False,
                    )
                    count += 1
            except Exception as e:
                logger.warning(f"Failed to inject history message ({role}): {e}")

        return count

    async def _process_agent_response(
        self,
        session_id: SessionId,
        room_id: str,
        min_offset: int,
        tools: AgentToolsProtocol,
        sender_name: str,
    ) -> None:
        """Wait for and process agent response events.

        Parlant may send multiple messages:
        1. A preamble message (tagged with __preamble__) - acknowledgment before tool execution
        2. Final message(s) after tool execution

        We need to keep waiting until we get a non-preamble message, which indicates
        tool execution is complete.
        """
        if not self._app:
            logger.error(f"Room {room_id}: No Parlant Application available")
            return

        app = self._app
        from parlant.core.async_utils import Timeout
        from parlant.core.sessions import EventKind, EventSource

        current_offset = min_offset
        max_iterations = 10  # Safety limit to prevent infinite loops
        iteration = 0

        while iteration < max_iterations:
            iteration += 1

            # Wait for agent response
            logger.info(f"Room {room_id}: Waiting for agent response (min_offset={current_offset + 1}, iteration={iteration})...")

            try:
                has_update = await app.sessions.wait_for_update(
                    session_id=session_id,
                    min_offset=current_offset + 1,
                    kinds=[EventKind.MESSAGE],
                    source=EventSource.AI_AGENT,
                    timeout=Timeout(120),  # Increased timeout for tool execution
                )
                logger.info(f"Room {room_id}: wait_for_update returned: {has_update}")
            except Exception as e:
                logger.error(f"Room {room_id}: Error waiting for update: {e}", exc_info=True)
                return

            if not has_update:
                logger.warning(f"Room {room_id}: Timeout waiting for agent response")
                return

            # Get new events
            try:
                events = await app.sessions.find_events(
                    session_id=session_id,
                    min_offset=current_offset + 1,
                    source=EventSource.AI_AGENT,
                    kinds=[EventKind.MESSAGE],
                    trace_id=None,  # Required by Parlant SDK v3.x
                )
                logger.info(f"Room {room_id}: Found {len(events)} agent events")
            except Exception as e:
                logger.error(f"Room {room_id}: Error finding events: {e}", exc_info=True)
                return

            if not events:
                logger.warning(f"Room {room_id}: No events found despite update signal")
                return

            # Process events and track if we got a non-preamble message
            got_final_message = False

            for event in events:
                logger.debug(f"Room {room_id}: Event kind={event.kind}, source={event.source}, data={event.data}")

                # Update offset for next iteration
                if hasattr(event, 'offset') and event.offset > current_offset:
                    current_offset = event.offset

                if event.kind == EventKind.MESSAGE and event.source == EventSource.AI_AGENT:
                    data = event.data
                    message_content = ""
                    tags = []

                    if isinstance(data, dict):
                        message_content = str(data.get("message", ""))
                        tags = data.get("tags", [])
                    elif isinstance(data, str):
                        message_content = data

                    # Check if this is a preamble message
                    is_preamble = "__preamble__" in tags

                    if is_preamble:
                        logger.info(f"Room {room_id}: Skipping preamble message: {message_content[:50]}...")
                        continue

                    # This is a final message (after tool execution)
                    got_final_message = True

                    if message_content:
                        logger.info(
                            f"Room {room_id}: Sending agent response to platform: {message_content[:100]}..."
                        )
                        try:
                            await tools.send_message(message_content, mentions=[sender_name])
                            logger.info(f"Room {room_id}: Message sent successfully")
                        except Exception as e:
                            logger.error(f"Room {room_id}: Error sending message: {e}", exc_info=True)
                    else:
                        logger.warning(f"Room {room_id}: Empty message content in event")

            # If we got a final (non-preamble) message, we're done
            if got_final_message:
                logger.info(f"Room {room_id}: Got final message, processing complete")
                return

            # Otherwise, continue waiting for the final message after tool execution
            logger.info(f"Room {room_id}: Only got preamble, continuing to wait for final message...")

        logger.warning(f"Room {room_id}: Reached max iterations ({max_iterations}) waiting for response")

    async def on_cleanup(self, room_id: str) -> None:
        """Clean up session when agent leaves a room."""
        if room_id in self._room_sessions:
            del self._room_sessions[room_id]
        if room_id in self._room_customers:
            del self._room_customers[room_id]

        logger.debug(f"Room {room_id}: Cleaned up Parlant session")

    async def _report_error(self, tools: AgentToolsProtocol, error: str) -> None:
        """Send error event (best effort)."""
        try:
            await tools.send_event(content=f"Error: {error}", message_type="error")
        except Exception:
            pass

    async def cleanup_all(self) -> None:
        """Cleanup all sessions (call on stop)."""
        self._room_sessions.clear()
        self._room_customers.clear()
        logger.info("Parlant adapter cleanup complete")
