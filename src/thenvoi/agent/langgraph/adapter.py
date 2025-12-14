"""
LangGraph adapter using the new ThenvoiAgent architecture.

This adapter uses the three-class architecture:
- ThenvoiAgent: Coordinator
- AgentSession: Per-room processing
- AgentTools: Tools for LLM

KEY DESIGN:
    SDK does NOT send messages.
    LangGraph agent uses tools.send_message() to respond.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, List

from langgraph.pregel import Pregel

from thenvoi.agent.core import (
    ThenvoiAgent,
    AgentTools,
    AgentConfig,
    PlatformMessage,
    SessionConfig,
    render_system_prompt,
)
from .langchain_tools import agent_tools_to_langchain

logger = logging.getLogger(__name__)


class LangGraphAdapter:
    """
    LangGraph adapter using graph_factory pattern.

    The graph_factory receives Thenvoi tools so the LLM can use
    them to communicate. SDK does NOT send messages directly.

    Example:
        from langchain_openai import ChatOpenAI
        from langgraph.prebuilt import create_react_agent
        from langgraph.checkpoint.memory import MemorySaver

        def my_graph_factory(tools):
            return create_react_agent(
                ChatOpenAI(model="gpt-4o"),
                tools,
                checkpointer=MemorySaver(),
            )

        adapter = LangGraphAdapter(
            graph_factory=my_graph_factory,
            agent_id="...",
            api_key="...",
        )
        await adapter.run()
    """

    def __init__(
        self,
        graph_factory: Callable[[List[Any]], Pregel] | None = None,
        graph: Pregel | None = None,
        agent_id: str = "",
        api_key: str = "",
        ws_url: str = "wss://api.thenvoi.com/ws",
        rest_url: str = "https://api.thenvoi.com",
        prompt_template: str = "default",
        custom_section: str = "",
        additional_tools: List[Any] | None = None,
        config: AgentConfig | None = None,
        session_config: SessionConfig | None = None,
    ):
        """
        Initialize LangGraph adapter.

        Args:
            graph_factory: Function that receives tools and returns compiled graph.
                          Recommended - allows Thenvoi tools to be included.
            graph: Pre-built graph (alternative to graph_factory).
                   Note: Won't have Thenvoi tools unless you add them manually.
            agent_id: Agent ID from Thenvoi platform
            api_key: Agent API key
            ws_url: WebSocket URL
            rest_url: REST API URL
            prompt_template: Named template or custom template string
            custom_section: Custom instructions for prompt
            additional_tools: Extra tools to add alongside Thenvoi tools
            config: Agent configuration
            session_config: Session configuration
        """
        if not graph_factory and not graph:
            raise ValueError("Must provide either graph_factory or graph")

        self.graph_factory = graph_factory
        self._static_graph = graph
        self.prompt_template = prompt_template
        self.custom_section = custom_section
        self.additional_tools = additional_tools or []

        # Create ThenvoiAgent coordinator
        self.thenvoi = ThenvoiAgent(
            agent_id=agent_id,
            api_key=api_key,
            ws_url=ws_url,
            rest_url=rest_url,
            config=config,
            session_config=session_config,
        )

        # Will be set after start()
        self._system_prompt: str = ""

    @property
    def system_prompt(self) -> str:
        """Get rendered system prompt."""
        return self._system_prompt

    async def start(self) -> None:
        """Start the adapter."""
        # Start thenvoi (fetches agent metadata)
        # Pass cleanup callback to clear checkpointer on session destroy
        self.thenvoi._on_session_cleanup = self._cleanup_session
        await self.thenvoi.start(on_message=self._handle_message)

        # Render system prompt with agent info
        self._system_prompt = render_system_prompt(
            template=self.prompt_template,
            agent_name=self.thenvoi.agent_name,
            agent_description=self.thenvoi.agent_description,
            custom_section=self.custom_section,
        )

        logger.info(f"LangGraph adapter started for agent: {self.thenvoi.agent_name}")

    async def stop(self) -> None:
        """Stop the adapter."""
        await self.thenvoi.stop()

    async def run(self) -> None:
        """Start and run until interrupted."""
        await self.start()
        try:
            await self.thenvoi.run()
        finally:
            await self.stop()

    async def _handle_message(self, msg: PlatformMessage, tools: AgentTools) -> None:
        """
        Handle incoming message.

        Handler receives AgentTools, NOT ThenvoiAgent.
        LLM uses tools.send_message() to respond.

        KEY DESIGN:
        - System prompt sent ONLY on first message (session tracks via is_llm_initialized)
        - Historical messages injected on first message to prime checkpointer
        - Participant list injected ONLY when it changes (session tracks)
        - Subsequent messages: user message only (unless participants changed)
        """
        room_id = msg.room_id  # = thread_id for LangGraph

        logger.debug(f"Handling message {msg.id} in room {room_id}")

        # Get session for this room
        session = self.thenvoi.active_sessions.get(room_id)
        if not session:
            logger.error(f"No session for room {room_id}")
            return

        # Check session state
        is_first_message = not session.is_llm_initialized
        participants_changed = session.participants_changed()

        logger.debug(
            f"Room {room_id}: is_first_message={is_first_message}, "
            f"participants_changed={participants_changed}"
        )

        # Get LangChain tools from AgentTools
        langchain_tools = agent_tools_to_langchain(tools) + self.additional_tools

        # Build or get graph
        if self.graph_factory:
            graph = self.graph_factory(langchain_tools)
        else:
            graph = self._static_graph

        if not graph:
            raise RuntimeError("No graph available")

        # Build messages list for this invocation
        messages = []

        if is_first_message:
            # FIRST MESSAGE: Include system prompt + historical context
            messages.append(("system", self._system_prompt))
            logger.info(f"Room {room_id}: Sending system prompt (first message)")
            logger.debug(f"System prompt:\n{self._system_prompt}")

            # Load and inject historical messages from session
            try:
                history = await session.get_history_for_llm(exclude_message_id=msg.id)
                for hist in history:
                    role = hist["role"]
                    content = hist["content"]
                    sender_name = hist["sender_name"]

                    # Format for LangGraph
                    if role == "assistant":
                        messages.append(("assistant", content))
                    else:
                        messages.append(("user", f"[{sender_name}]: {content}"))
            except Exception as e:
                logger.warning(f"Room {room_id}: Failed to load history: {e}")

            session.mark_llm_initialized()

        # Inject participants message ONLY if changed
        if participants_changed:
            messages.append(("system", session.build_participants_message()))
            logger.info(
                f"Room {room_id}: Participants updated: {[p.get('name') for p in session.participants]}"
            )
            session.mark_participants_sent()

        # Add current message (always)
        messages.append(("user", msg.format_for_llm()))

        graph_input = {"messages": messages}

        logger.debug(f"Room {room_id}: Sending {len(messages)} messages to LangGraph")

        # Run LangGraph - LLM uses tools.send_message() internally
        try:
            # Use astream_events for visibility into tool calls
            async for event in graph.astream_events(
                graph_input,
                config={"configurable": {"thread_id": room_id}},
                version="v2",
            ):
                await self._handle_stream_event(event, room_id, tools)

            logger.debug(f"Message {msg.id} processed successfully")

        except Exception as e:
            logger.error(f"Error processing message {msg.id}: {e}", exc_info=True)
            # Send error via event (no mentions required)
            try:
                await tools.send_event(
                    content=f"Error processing message: {e}",
                    message_type="error",
                    metadata={"error_type": type(e).__name__},
                )
            except Exception:
                pass  # Best effort

    async def _handle_stream_event(
        self, event: Any, room_id: str, tools: AgentTools
    ) -> None:
        """Handle streaming events from LangGraph - send raw events to platform."""
        event_type = event.get("event")

        if event_type == "on_tool_start":
            tool_name = event.get("name", "unknown")
            logger.debug(f"[{room_id}] Tool started: {tool_name}")

            # Send raw LangGraph event
            try:
                await tools.send_event(
                    content=json.dumps(event, default=str),
                    message_type="tool_call",
                    metadata=None,
                )
            except Exception as e:
                logger.warning(f"Failed to send tool_call event: {e}")

        elif event_type == "on_tool_end":
            tool_name = event.get("name", "unknown")
            logger.debug(f"[{room_id}] Tool ended: {tool_name}")

            # Send raw LangGraph event
            try:
                await tools.send_event(
                    content=json.dumps(event, default=str),
                    message_type="tool_result",
                    metadata=None,
                )
            except Exception as e:
                logger.warning(f"Failed to send tool_result event: {e}")

    async def _cleanup_session(self, room_id: str) -> None:
        """Clean up LangGraph checkpointer data for session."""
        if not self.graph_factory or not hasattr(self.graph_factory, "checkpointer"):
            return

        checkpointer = self.graph_factory.checkpointer
        try:
            # LangGraph uses thread_id internally, which maps to our room_id
            await checkpointer.adelete_thread(room_id)
            logger.info(f"Cleared LangGraph session for room {room_id}")
        except Exception as e:
            logger.warning(f"Failed to clear session {room_id}: {e}")


class LangGraphMCPAdapter(LangGraphAdapter):
    """
    LangGraph adapter that loads tools from MCP server.

    For setups using thenvoi-mcp-server instead of direct tools.
    """

    def __init__(
        self,
        mcp_server_url: str = "http://localhost:3000/mcp",
        **kwargs,
    ):
        """
        Initialize MCP-based LangGraph adapter.

        Args:
            mcp_server_url: URL for thenvoi-mcp-server
            **kwargs: Arguments passed to LangGraphAdapter
        """
        super().__init__(**kwargs)
        self.mcp_server_url = mcp_server_url

    async def _handle_message(self, msg: PlatformMessage, tools: AgentTools) -> None:
        """Handle message with MCP tools."""
        # TODO: Load tools from MCP server instead of AgentTools
        # For now, delegate to parent
        await super()._handle_message(msg, tools)


# Convenience function
async def with_langgraph(
    graph_factory: Callable[[List[Any]], Pregel] | None = None,
    graph: Pregel | None = None,
    agent_id: str = "",
    api_key: str = "",
    ws_url: str = "wss://api.thenvoi.com/ws",
    rest_url: str = "https://api.thenvoi.com",
    **kwargs,
) -> LangGraphAdapter:
    """
    Quick integration for LangGraph agents.

    Provide either graph_factory (recommended) or pre-built graph.

    Example:
        # With factory (recommended - gets Thenvoi tools)
        adapter = await with_langgraph(
            graph_factory=lambda tools: create_react_agent(ChatOpenAI(), tools),
            agent_id="...",
            api_key="...",
        )

        # With pre-built graph
        adapter = await with_langgraph(
            graph=my_prebuilt_graph,
            agent_id="...",
            api_key="...",
        )

    Returns:
        LangGraphAdapter (already started)
    """
    adapter = LangGraphAdapter(
        graph_factory=graph_factory,
        graph=graph,
        agent_id=agent_id,
        api_key=api_key,
        ws_url=ws_url,
        rest_url=rest_url,
        **kwargs,
    )
    await adapter.start()
    return adapter
