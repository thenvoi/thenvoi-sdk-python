"""
Parlant adapter using the official Parlant SDK.

This adapter integrates the Parlant framework (https://github.com/emcie-co/parlant)
with the Thenvoi platform using the official Parlant SDK for proper guideline-based
agent behavior.
"""

from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING, Any

from thenvoi.core.protocols import AgentToolsProtocol
from thenvoi.core.simple_adapter import SimpleAdapter
from thenvoi.core.types import PlatformMessage
from thenvoi.converters.parlant import ParlantHistoryConverter, ParlantMessages
from thenvoi.runtime.prompts import render_system_prompt

if TYPE_CHECKING:
    from parlant.client import AsyncParlantClient

    from thenvoi.integrations.parlant.session_manager import ParlantSessionManager

logger = logging.getLogger(__name__)


class ParlantAdapter(SimpleAdapter[ParlantMessages]):
    """
    Parlant adapter using the official Parlant SDK.

    Integrates the Parlant framework (https://github.com/emcie-co/parlant)
    with Thenvoi platform for controlled, guideline-based agent behavior.

    This adapter uses the official Parlant SDK to ensure proper guideline
    matching, tool invocation, and session management rather than simulating
    Parlant behavior through prompt engineering.

    Parlant provides:
    - Behavioral guidelines for consistent agent responses
    - Tool integration with conditional activation
    - Built-in guardrails against hallucination
    - Explainability for agent decisions
    - Session-based conversation management

    Requires a running Parlant server (local or remote).

    Example:
        adapter = ParlantAdapter(
            parlant_url="http://localhost:8000",
            agent_id="my-agent-id",  # Pre-configured agent on Parlant server
            custom_section="You are a helpful customer support agent.",
        )
        agent = Agent.create(adapter=adapter, agent_id="...", api_key="...")
        await agent.run()

    Example (with guidelines registered at runtime):
        adapter = ParlantAdapter(
            parlant_url="http://localhost:8000",
            agent_id="my-agent-id",
            guidelines=[
                {
                    "condition": "Customer asks about refunds",
                    "action": "Check order status first to see if eligible",
                }
            ],
        )
    """

    def __init__(
        self,
        # Parlant SDK configuration
        parlant_url: str | None = None,
        agent_id: str | None = None,
        # Agent configuration
        system_prompt: str | None = None,
        custom_section: str | None = None,
        guidelines: list[dict[str, str]] | None = None,
        # Options
        enable_execution_reporting: bool = False,
        wait_timeout: int = 60,
        history_converter: ParlantHistoryConverter | None = None,
    ):
        """
        Initialize the Parlant adapter.

        Args:
            parlant_url: URL of the Parlant server (uses PARLANT_URL env var if not provided)
            agent_id: ID of the pre-configured agent on Parlant server
                     (uses PARLANT_AGENT_ID env var if not provided)
            system_prompt: Full system prompt override. When provided, this completely
                          replaces the default prompt (agent description + platform instructions).
                          Use this for complete control over the agent's instructions.
            custom_section: Custom instructions appended to the agent description.
                           Ignored if system_prompt is provided. Use this to add specific
                           behaviors while keeping default platform instructions.
            guidelines: List of behavioral guidelines with condition/action pairs.
                       These are registered with Parlant at startup if provided.
            enable_execution_reporting: If True, sends tool_call/tool_result events
            wait_timeout: Timeout in seconds when waiting for agent responses
            history_converter: Custom history converter (optional)
        """
        super().__init__(
            history_converter=history_converter or ParlantHistoryConverter()
        )

        self.parlant_url = parlant_url or os.getenv(
            "PARLANT_URL", "http://localhost:8000"
        )
        self.agent_id = agent_id or os.getenv("PARLANT_AGENT_ID")
        self.system_prompt = system_prompt
        self.custom_section = custom_section
        self.guidelines = guidelines or []
        self.enable_execution_reporting = enable_execution_reporting
        self.wait_timeout = wait_timeout

        # Parlant client and session manager (initialized on start)
        self._client: AsyncParlantClient | None = None
        self._session_manager: ParlantSessionManager | None = None

        # Per-room tools storage for tool execution
        self._room_tools: dict[str, AgentToolsProtocol] = {}

        # Rendered system prompt (set after start)
        self._system_prompt: str = ""

    async def on_started(self, agent_name: str, agent_description: str) -> None:
        """Initialize Parlant client and session manager after metadata is fetched."""
        await super().on_started(agent_name, agent_description)

        # Render system prompt
        self._system_prompt = self.system_prompt or render_system_prompt(
            agent_name=agent_name,
            agent_description=agent_description,
            custom_section=self.custom_section or "",
        )

        # Initialize Parlant client
        try:
            from parlant.client import AsyncParlantClient
        except ImportError:
            raise ImportError(
                "parlant package required for ParlantAdapter. "
                "Install with: pip install 'thenvoi-sdk[parlant]' or pip install parlant"
            )

        try:
            self._client = AsyncParlantClient(base_url=self.parlant_url)

            # Get or create agent
            if not self.agent_id:
                # Create agent dynamically
                logger.info(f"Creating Parlant agent: {agent_name}")

                # Prepare agent description for Parlant
                # Parlant uses the description field for agent prompting
                if self.system_prompt:
                    # User provided a full system prompt override
                    description_with_context = self.system_prompt
                    logger.debug("Using custom system_prompt for agent creation")
                elif self.custom_section:
                    # Append custom section to base description
                    description_with_context = (
                        f"{agent_description}\n\n"
                        f"Additional instructions:\n{self.custom_section}"
                    )
                    logger.debug("Using agent_description with custom_section")
                else:
                    # Use just the agent description (basic mode)
                    # Note: The full _system_prompt with platform instructions is available
                    # but not used unless explicitly requested via system_prompt or custom_section
                    description_with_context = agent_description
                    logger.debug("Using basic agent_description")

                agent_response = await self._client.agents.create(
                    name=agent_name,
                    description=description_with_context,
                )

                # Validate agent creation
                if (
                    not agent_response
                    or not hasattr(agent_response, "id")
                    or not agent_response.id
                ):
                    raise ValueError(
                        f"Failed to create Parlant agent '{agent_name}': "
                        "Invalid response from Parlant server"
                    )

                self.agent_id = agent_response.id
                logger.info(f"Created Parlant agent with ID: {self.agent_id}")

                # Register guidelines if provided
                await self._register_guidelines()
            else:
                logger.info(f"Using existing Parlant agent: {self.agent_id}")

            # Register tools with Parlant (for both new and existing agents)
            # This ensures tools are available even if the agent was pre-configured
            await self._register_tools()

            # Initialize session manager
            from thenvoi.integrations.parlant.session_manager import (
                ParlantSessionManager,
            )

            self._session_manager = ParlantSessionManager(
                client=self._client,
                agent_id=self.agent_id,
            )

            logger.info(
                f"Parlant adapter started for agent: {agent_name} "
                f"(parlant_url={self.parlant_url}, agent_id={self.agent_id})"
            )

        except Exception as e:
            # Cleanup on error
            logger.error(f"Failed to initialize Parlant adapter: {e}", exc_info=True)
            if self._client:
                try:
                    if hasattr(self._client, "close"):
                        await self._client.close()
                except Exception:
                    pass
                self._client = None
            self._session_manager = None
            raise

    async def _register_guidelines(self) -> None:
        """Register guidelines with the Parlant server."""
        if not self.guidelines or not self._client:
            return

        logger.info(f"Registering {len(self.guidelines)} guidelines with Parlant")

        for i, guideline in enumerate(self.guidelines, 1):
            condition = guideline.get("condition", "")
            action = guideline.get("action", "")

            if not condition or not action:
                logger.warning(
                    f"Skipping invalid guideline {i}: missing condition or action"
                )
                continue

            try:
                await self._client.agents.create_guideline(
                    agent_id=self.agent_id,
                    condition=condition,
                    action=action,
                )
                logger.debug(f"Registered guideline {i}: {condition[:50]}...")
            except Exception as e:
                # Guideline may already exist, log and continue
                logger.warning(f"Failed to register guideline {i}: {e}")

        logger.info("Guidelines registration complete")

    async def _register_tools(self) -> None:
        """
        Register platform tools with the Parlant server.

        Requires Parlant SDK with the following API methods:
        - client.agents.create_tool(agent_id, name, description, parameters)

        Note: This method handles gracefully if tools already exist.
        """
        if not self._client or not self.agent_id:
            return

        from thenvoi.integrations.parlant.tools import get_tool_schemas_for_parlant

        logger.info("Registering platform tools with Parlant")

        try:
            tool_schemas = get_tool_schemas_for_parlant()

            # Register each tool with Parlant
            # Tool schemas are expected in OpenAI function format:
            # {"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}
            for tool_schema in tool_schemas:
                # Validate schema structure
                if not isinstance(tool_schema, dict) or "function" not in tool_schema:
                    logger.warning(
                        f"Invalid tool schema format, skipping: {tool_schema}"
                    )
                    continue

                function_def = tool_schema["function"]
                if not isinstance(function_def, dict):
                    logger.warning(
                        f"Invalid function definition, skipping: {function_def}"
                    )
                    continue

                tool_name = function_def.get("name", "")
                if not tool_name:
                    logger.warning(
                        f"Tool schema missing name, skipping: {function_def}"
                    )
                    continue

                try:
                    await self._client.agents.create_tool(
                        agent_id=self.agent_id,
                        name=tool_name,
                        description=function_def.get("description", ""),
                        parameters=function_def.get("parameters", {}),
                    )
                    logger.debug(f"Registered tool: {tool_name}")
                except Exception as e:
                    # Tool may already exist, log and continue
                    logger.warning(f"Failed to register tool {tool_name}: {e}")

            logger.info(f"Registered {len(tool_schemas)} tools with Parlant")
        except Exception as e:
            logger.error(f"Failed to register tools: {e}", exc_info=True)

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
        Handle incoming message using the Parlant SDK.

        Uses Parlant's session-based messaging to ensure proper guideline
        matching and tool execution.

        Thread Safety:
            This method assumes messages for the same room are processed sequentially.
            Concurrent calls for different rooms are safe due to per-room session management.
            If messages for the same room arrive concurrently, they should be queued
            by the caller to prevent race conditions.
        """
        logger.debug(f"Handling message {msg.id} in room {room_id}")

        if not self._client or not self._session_manager:
            logger.error("Parlant client not initialized")
            return

        # Store tools for this room (used by tool execution)
        self._room_tools[room_id] = tools

        # Get or create session for this room
        customer_id = msg.sender_id or "anonymous"
        customer_name = msg.sender_name or customer_id

        session = await self._session_manager.get_or_create_session(
            room_id=room_id,
            customer_id=customer_id,
            customer_name=customer_name,
        )

        # On bootstrap, inject historical context
        if is_session_bootstrap and history:
            injected_count = await self._inject_history(session.session_id, history)
            logger.info(
                f"Room {room_id}: Injected {injected_count} customer messages "
                f"from {len(history)} total historical messages"
            )

        # Inject participants update if provided
        if participants_msg:
            await self._send_system_message(
                session.session_id,
                f"[Participant Update]: {participants_msg}",
            )
            logger.info(f"Room {room_id}: Participants updated")

        # Send customer message to Parlant
        user_message = msg.format_for_llm()
        logger.info(f"Room {room_id}: Sending message to Parlant session")

        try:
            # Create customer event
            event = await self._client.sessions.create_event(
                session_id=session.session_id,
                kind="message",
                source="customer",
                message=user_message,
            )

            # Wait for agent response events
            await self._process_agent_response(
                session_id=session.session_id,
                room_id=room_id,
                min_offset=event.offset,
                tools=tools,
            )

        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)
            await self._report_error(tools, str(e))
            raise

        logger.debug(f"Message {msg.id} processed successfully")

    async def _inject_history(
        self,
        session_id: str,
        history: ParlantMessages,
    ) -> int:
        """
        Inject historical messages into a Parlant session.

        Note: Only customer messages are injected. Assistant messages are skipped because:
        1. Parlant generates ai_agent messages internally based on its processing
        2. Externally injecting ai_agent events can cause state inconsistencies
        3. Customer messages provide sufficient context for Parlant to understand conversation history

        Returns:
            Number of customer messages successfully injected
        """
        customer_count = 0
        skipped_count = 0

        for hist in history:
            role = hist.get("role", "user")
            content = hist.get("content", "")

            if role == "user":
                try:
                    await self._client.sessions.create_event(
                        session_id=session_id,
                        kind="message",
                        source="customer",
                        message=content,
                    )
                    customer_count += 1
                except Exception as e:
                    logger.error(f"Failed to inject customer message into history: {e}")
            elif role == "assistant":
                # Skip assistant messages - Parlant doesn't support external injection of ai_agent events
                skipped_count += 1

        if skipped_count > 0:
            logger.debug(
                f"Session {session_id}: Injected {customer_count} customer messages, "
                f"skipped {skipped_count} assistant messages (not supported by Parlant)"
            )

        return customer_count

    async def _send_system_message(
        self,
        session_id: str,
        content: str,
    ) -> None:
        """Send a system message to a Parlant session."""
        try:
            await self._client.sessions.create_event(
                session_id=session_id,
                kind="message",
                source="system",
                message=content,
            )
        except Exception as e:
            logger.warning(f"Could not send system message: {e}")

    async def _process_agent_response(
        self,
        session_id: str,
        room_id: str,
        min_offset: int,
        tools: AgentToolsProtocol,
    ) -> None:
        """
        Process agent response events from Parlant.

        Waits for and processes events until we receive a complete agent response.
        """
        processed_offset = min_offset
        max_iterations = 50  # Prevent infinite loops
        iteration_count = 0

        while iteration_count < max_iterations:
            iteration_count += 1

            # Wait for new events
            try:
                events = await self._client.sessions.list_events(
                    session_id=session_id,
                    min_offset=processed_offset,
                    wait_for_data=self.wait_timeout,
                )
            except Exception as e:
                logger.error(
                    f"Room {room_id}: Failed to fetch events: {e}", exc_info=True
                )
                break

            if not events:
                logger.warning(
                    f"Room {room_id}: No events received (timeout after {self.wait_timeout}s)"
                )
                break

            agent_responded = False

            for event in events:
                processed_offset = max(processed_offset, event.offset + 1)

                # Process based on event type
                if event.kind == "message" and event.source == "ai_agent":
                    # Agent message - send to Thenvoi platform
                    agent_responded = True
                    message_content = (
                        str(event.message) if hasattr(event, "message") else ""
                    )

                    if message_content:
                        logger.debug(
                            f"Room {room_id}: Agent message: {message_content[:100]}..."
                        )
                        # Send agent's response to the chat
                        await tools.send_message(message_content)
                    else:
                        logger.warning(
                            f"Room {room_id}: Agent message event has no content"
                        )

                elif event.kind == "tool_calls":
                    # Tool invocation by Parlant
                    await self._handle_tool_calls(event, room_id, tools)

                elif event.kind == "status":
                    # Status update
                    logger.debug(f"Room {room_id}: Status: {event.data}")

            # Update session offset
            self._session_manager.update_offset(room_id, processed_offset)

            # If agent has responded, we're done
            if agent_responded:
                logger.debug(
                    f"Room {room_id}: Agent response complete after {iteration_count} iterations"
                )
                break

            # Log warning if taking too long
            if iteration_count >= max_iterations * 0.8:
                logger.warning(
                    f"Room {room_id}: Event processing taking longer than expected "
                    f"({iteration_count}/{max_iterations} iterations)"
                )

        if iteration_count >= max_iterations:
            logger.error(
                f"Room {room_id}: Event processing stopped after {max_iterations} iterations "
                "without receiving agent response"
            )

    async def _handle_tool_calls(
        self,
        event: Any,
        room_id: str,
        tools: AgentToolsProtocol,
    ) -> None:
        """
        Handle tool calls from Parlant.

        Executes the requested tools and returns results to Parlant.
        """
        from thenvoi.integrations.parlant.tools import (
            ParlantToolContext,
            create_parlant_tools,
        )

        # Validate event structure
        if not hasattr(event, "session_id") or not event.session_id:
            logger.error(f"Room {room_id}: Tool call event missing session_id")
            return

        if not hasattr(event, "data") or not event.data:
            logger.error(f"Room {room_id}: Tool call event missing data")
            return

        # Get tool definitions
        parlant_tools = {t.name: t for t in create_parlant_tools()}

        # Create tool context
        ctx = ParlantToolContext(
            room_id=room_id,
            tools=tools,
            session_id=event.session_id,
        )

        # Process each tool call
        tool_calls = event.data

        for tc in tool_calls:
            tool_name = tc.get("name", "")
            tool_args = tc.get("arguments", {})
            tool_call_id = tc.get("id", "")

            # Validate tool call structure
            if not tool_name:
                logger.error(
                    f"Room {room_id}: Tool call missing 'name' field, skipping: {tc}"
                )
                continue

            if not tool_call_id:
                logger.error(
                    f"Room {room_id}: Tool call '{tool_name}' missing 'id' field, skipping: {tc}"
                )
                continue

            logger.info(f"Room {room_id}: Executing tool: {tool_name}")

            # Report tool call if enabled
            if self.enable_execution_reporting:
                await tools.send_event(
                    content=json.dumps({"tool": tool_name, "input": tool_args}),
                    message_type="tool_call",
                    metadata={"tool": tool_name, "input": tool_args},
                )

            # Execute tool
            if tool_name in parlant_tools:
                tool_def = parlant_tools[tool_name]
                try:
                    result = await tool_def.func(ctx, tool_args)
                    result_data = result.to_dict()
                    is_error = False
                except Exception as e:
                    result_data = {"status": "error", "message": str(e)}
                    is_error = True
                    logger.error(f"Tool {tool_name} failed: {e}")
            else:
                result_data = {
                    "status": "error",
                    "message": f"Unknown tool: {tool_name}",
                }
                is_error = True

            # Report tool result if enabled
            if self.enable_execution_reporting:
                await tools.send_event(
                    content=json.dumps(
                        {"tool": tool_name, "result": result_data, "is_error": is_error}
                    ),
                    message_type="tool_result",
                    metadata={"tool": tool_name, "is_error": is_error},
                )

            # Return result to Parlant
            try:
                await self._client.sessions.submit_tool_result(
                    session_id=event.session_id,
                    tool_call_id=tool_call_id,
                    result=json.dumps(result_data, default=str),
                    is_error=is_error,
                )
            except Exception as e:
                logger.error(
                    f"Failed to submit tool result for {tool_name}: {e}", exc_info=True
                )

    async def on_cleanup(self, room_id: str) -> None:
        """Clean up Parlant session when agent leaves a room."""
        if self._session_manager:
            await self._session_manager.cleanup_session(room_id)

        if room_id in self._room_tools:
            del self._room_tools[room_id]

        logger.debug(f"Room {room_id}: Cleaned up Parlant session")

    async def _report_error(self, tools: AgentToolsProtocol, error: str) -> None:
        """Send error event (best effort)."""
        try:
            await tools.send_event(content=f"Error: {error}", message_type="error")
        except Exception:
            pass

    async def cleanup_all(self) -> None:
        """Cleanup all sessions and close client (call on stop)."""
        if self._session_manager:
            await self._session_manager.cleanup_all()

        if self._client:
            # Close client connection if supported
            try:
                if hasattr(self._client, "close"):
                    await self._client.close()
            except Exception:
                pass

        self._room_tools.clear()
        logger.info("Parlant adapter cleanup complete")
