"""
LangGraph integration for Thenvoi platform.

Uses composition with PlatformClient and RoomManager for clean separation.
Only implements LangGraph-specific logic.
"""

import logging
from typing import Optional, Any, List

from langchain.agents import create_agent as langgraph_create_agent
from langchain_core.runnables.schema import StreamEvent
from langgraph.pregel import Pregel

from thenvoi.agent.core import ThenvoiPlatformClient, RoomManager
from thenvoi.client.streaming import MessageCreatedPayload
from thenvoi.agent.langgraph.prompts import generate_langgraph_agent_prompt
from thenvoi.agent.langgraph.tools import get_thenvoi_tools
from thenvoi.agent.langgraph.message_formatters import (
    MessageFormatter,
    default_messages_state_formatter,
)
from thenvoi.client.rest import ChatEventRequest

logger = logging.getLogger(__name__)


# Shared utilities for both agent types
async def _send_platform_event(
    api_client, room_id: str, content: str, message_type: str
):
    """Send an event to the platform via API (shared utility)."""
    event_request = ChatEventRequest(
        content=content,
        message_type=message_type,
    )

    await api_client.agent_api.create_agent_chat_event(
        chat_id=room_id, event=event_request
    )


async def _handle_streaming_event(event: StreamEvent, room_id: str, api_client):
    """
    Handle streaming events from LangGraph - shared by both agent types.

    Streams tool calls and results back to the chat room for visibility.
    """
    event_type = event["event"]

    if event_type == "on_tool_start":
        tool_name = event["name"]
        tool_input = event["data"].get("input", {})

        # Format tool call message
        if isinstance(tool_input, dict):
            args_str = ", ".join(
                [f"{k}={v}" for k, v in tool_input.items() if k != "config"]
            )
        else:
            args_str = str(tool_input)
        content = f"Calling {tool_name}({args_str})"

        logger.debug(f"[{room_id}] Tool call: {tool_name} with args: {args_str}")
        await _send_platform_event(api_client, room_id, content, "tool_call")

    elif event_type == "on_tool_end":
        tool_name = event["name"]
        output = event["data"].get("output")

        # Format tool result message (truncate if too long)
        if output is None:
            output_str = ""
        elif hasattr(output, "content"):
            output_str = str(output.content)
        else:
            output_str = str(output)
        if len(output_str) > 500:
            output_str = output_str[:500] + "..."

        content = f"{tool_name} result: {output_str}"

        logger.debug(
            f"[{room_id}] Tool result from {tool_name}: {output_str[:100]}{'...' if len(output_str) > 100 else ''}"
        )
        await _send_platform_event(api_client, room_id, content, "tool_result")


async def create_langgraph_agent(
    agent_id: str,
    api_key: str,
    llm: Any,
    checkpointer: Any,
    ws_url: str,
    thenvoi_restapi_url: str,
    additional_tools: Optional[List] = None,
    custom_instructions: Optional[str] = None,
):
    """
    Create and start a Thenvoi LangGraph agent (functional API).

    This is a convenience function that creates a ThenvoiLangGraphAgent
    and starts it immediately.

    Args:
        agent_id: Agent ID from platform (agent must be created externally)
        api_key: Agent-specific API key from platform
        llm: Language model (e.g., ChatOpenAI(model="gpt-4o"))
        checkpointer: LangGraph checkpointer (e.g., InMemorySaver())
        ws_url: WebSocket URL
        thenvoi_restapi_url: REST API base URL
        additional_tools: Custom tools to add to agent
        custom_instructions: Additional instructions appended to base system prompt

    Returns:
        ThenvoiLangGraphAgent instance (already running)

    Example:
        >>> agent_id, api_key = load_agent_config("my_agent")
        >>> agent = await create_langgraph_agent(
        ...     agent_id=agent_id,
        ...     api_key=api_key,
        ...     llm=ChatOpenAI(model="gpt-4o"),
        ...     checkpointer=InMemorySaver(),
        ...     ws_url=os.getenv("THENVOI_WS_URL"),
        ...     thenvoi_restapi_url=os.getenv("THENVOI_REST_API_URL"),
        ...     custom_instructions="You are a friendly assistant with a sense of humor.",
        ... )
        # Agent is now listening for messages
    """
    agent = ThenvoiLangGraphAgent(
        agent_id=agent_id,
        api_key=api_key,
        llm=llm,
        checkpointer=checkpointer,
        ws_url=ws_url,
        thenvoi_restapi_url=thenvoi_restapi_url,
        additional_tools=additional_tools,
        custom_instructions=custom_instructions,
    )
    await agent.start()
    return agent


class ThenvoiLangGraphAgent:
    """
    LangGraph agent integration with Thenvoi platform.

    Builds a default LangGraph agent with LLM and tools, then uses
    DirectGraphAgent to run it. This is a convenience wrapper that
    creates a standard agent architecture.
    """

    def __init__(
        self,
        agent_id: str,
        api_key: str,
        llm: Any,
        checkpointer: Any,
        ws_url: str,
        thenvoi_restapi_url: str,
        additional_tools: Optional[List] = None,
        custom_instructions: Optional[str] = None,
    ):
        """
        Initialize a Thenvoi LangGraph agent.

        Args:
            agent_id: Agent ID from platform (agent must be created externally)
            api_key: Agent-specific API key from platform
            llm: Language model (e.g., ChatOpenAI(model="gpt-4o"))
            checkpointer: LangGraph checkpointer (e.g., InMemorySaver())
            ws_url: WebSocket URL
            thenvoi_restapi_url: REST API base URL
            additional_tools: Custom tools to add to agent (in addition to platform tools)
            custom_instructions: Additional instructions appended to base system prompt
        """
        # Store configuration for building the graph
        self.agent_id = agent_id
        self.api_key = api_key
        self.ws_url = ws_url
        self.thenvoi_restapi_url = thenvoi_restapi_url
        self.llm = llm
        self.checkpointer = checkpointer
        self.additional_tools = additional_tools or []
        self.custom_instructions = custom_instructions

    def _build_system_prompt(self, agent_name: str) -> str:
        """Build system prompt: base prompt + optional custom instructions."""
        base_prompt = generate_langgraph_agent_prompt(agent_name)

        if self.custom_instructions:
            return (
                f"{base_prompt}\n\nAdditional Instructions:\n{self.custom_instructions}"
            )

        return base_prompt

    def _build_graph(self, platform_client: ThenvoiPlatformClient) -> Pregel:
        """Build the default LangGraph agent with all tools.

        Args:
            platform_client: ThenvoiPlatformClient with fetched metadata
        """
        # Create platform tools using the existing client
        platform_tools = get_thenvoi_tools(
            client=platform_client.api_client, agent_id=platform_client.agent_id
        )

        all_tools = platform_tools + self.additional_tools

        logger.debug(f"Building LangGraph agent with {len(all_tools)} tools")

        # Build system prompt using agent name from platform client
        system_prompt = self._build_system_prompt(platform_client.name)

        # Build the LangGraph agent
        graph = langgraph_create_agent(
            self.llm,
            tools=all_tools,
            system_prompt=system_prompt,
            checkpointer=self.checkpointer,
        )

        logger.debug("LangGraph agent built")
        return graph

    def _create_message_formatter(self) -> MessageFormatter:
        """Create message formatter that formats messages for the built-in agent."""

        def formatter(message: MessageCreatedPayload, sender_name: str):
            formatted_message = (
                f"Message from {sender_name} ({message.sender_type}, ID: {message.sender_id}) "
                f"in room {message.chat_room_id}: {message.content}"
            )
            return {"messages": [{"role": "user", "content": formatted_message}]}

        return formatter

    async def start(self):
        """
        Start the agent.

        Steps:
        1. Fetch agent metadata (to get agent name for system prompt)
        2. Build the LangGraph agent
        3. Delegate to DirectGraphAgent to run it
        """
        # Step 1: Fetch agent metadata to get agent name
        self._platform_client = ThenvoiPlatformClient(
            agent_id=self.agent_id,
            api_key=self.api_key,
            ws_url=self.ws_url,
            thenvoi_restapi_url=self.thenvoi_restapi_url,
        )
        await self._platform_client.fetch_agent_metadata()

        logger.debug(
            f"Building default LangGraph agent for '{self._platform_client.name}'"
        )

        # Step 2: Build the graph with the platform client
        built_graph = self._build_graph(self._platform_client)

        # Step 3: Create ConnectedGraphAgent and delegate to it (reuse platform_client)
        self.connected_agent = ConnectedGraphAgent(
            graph=built_graph,
            platform_client=self._platform_client,
            message_formatter=self._create_message_formatter(),
        )

        logger.debug("Starting ConnectedGraphAgent with built-in graph")
        await self.connected_agent.start()


class ConnectedGraphAgent:
    """
    Runs a user-provided LangGraph directly with Thenvoi platform.

    This allows users to bring their own compiled LangGraph and connect it
    directly to chat room messages, bypassing the default agent architecture.

    The user's graph receives chat messages as input and can use Thenvoi
    platform tools from any node.
    """

    def __init__(
        self,
        graph: Pregel,
        platform_client: ThenvoiPlatformClient,
        message_formatter: MessageFormatter = default_messages_state_formatter,
    ):
        """
        Initialize a direct graph agent.

        Args:
            graph: User's compiled LangGraph (must have checkpointer)
            platform_client: ThenvoiPlatformClient instance to use for platform communication
            message_formatter: Function to convert platform messages to graph input.
                              Default: converts to MessagesState format
        """
        # Use the provided platform client
        self.platform = platform_client

        # User's graph
        self.user_graph = graph
        self.message_formatter = message_formatter

        # Validate graph has checkpointer
        if not hasattr(graph, "checkpointer") or graph.checkpointer is None:
            raise ValueError(
                "Graph must be compiled with a checkpointer. "
                "Example: graph.compile(checkpointer=InMemorySaver())"
            )

    async def _handle_streaming_event(self, event: StreamEvent, room_id: str):
        """Handle streaming events from user's graph (delegates to shared utility)."""
        await _handle_streaming_event(event, room_id, self.platform.api_client)

    async def _handle_room_message(self, message: MessageCreatedPayload):
        """Handle incoming message - invoke user's graph directly."""
        logger.debug(f"Received message in room {message.chat_room_id}")

        # Get sender name for formatting
        sender_name = await self.room_manager.get_participant_name(
            message.sender_id, message.sender_type, message.chat_room_id
        )

        # Format message using user's formatter
        graph_input = self.message_formatter(message, sender_name)

        logger.debug(f"Invoking user graph in room {message.chat_room_id}")
        logger.debug(f"Graph input: {graph_input}")

        # Invoke user's graph with streaming to capture tool calls
        try:
            async for event in self.user_graph.astream_events(
                graph_input,
                {"configurable": {"thread_id": message.chat_room_id}},
                version="v2",
            ):
                await self._handle_streaming_event(event, message.chat_room_id)

            logger.debug("User graph processed message successfully")

        except Exception as e:
            logger.error(
                f"Error invoking user graph: {type(e).__name__}: {str(e)}",
                exc_info=True,
            )
            # Send error message to chat room
            error_content = f"Error processing message: {type(e).__name__}: {str(e)}"
            await _send_platform_event(
                self.platform.api_client,
                message.chat_room_id,
                error_content,
                "error",
            )

    async def _on_room_removed(self, room_id: str):
        """Handle room removal (optional cleanup)."""
        logger.debug(f"Room removed: {room_id}")

    async def start(self):
        """
        Start the agent.

        Steps:
        1. Validate agent on platform
        2. Connect to WebSocket
        3. Subscribe to rooms
        4. Listen for messages
        """
        # Step 1: Fetch agent metadata from platform
        await self.platform.fetch_agent_metadata()
        logger.info(f"Agent '{self.platform.name}' validated on platform")

        # Step 2: Connect WebSocket
        ws_client = await self.platform.connect_websocket()

        try:
            async with ws_client:
                logger.debug("WebSocket connected")

                # Step 3: Create room manager with our message handler
                self.room_manager = RoomManager(
                    agent_id=self.platform.agent_id,
                    agent_name=self.platform.name,
                    api_client=self.platform.api_client,
                    ws_client=ws_client,
                    message_handler=self._handle_room_message,
                    on_room_removed=self._on_room_removed,
                )

                # Step 4: Subscribe to all rooms and room events
                room_count = await self.room_manager.subscribe_to_all_rooms()
                await self.room_manager.subscribe_to_room_events()

                if room_count == 0:
                    logger.warning(
                        "No rooms found. Add agent to a room via the platform."
                    )

                # Keep running
                logger.info(
                    f"Agent '{self.platform.name}' is now listening for messages..."
                )
                logger.info("Messages will be passed directly to user's graph")
                await ws_client.run_forever()
        except Exception as e:
            logger.error(
                f"Agent '{self.platform.name}' disconnected: {type(e).__name__}: {e}",
                exc_info=True,
            )
            raise
        finally:
            logger.info(f"Agent '{self.platform.name}' stopped")


async def connect_graph_to_platform(
    graph: Pregel,
    platform_client: ThenvoiPlatformClient,
    message_formatter: MessageFormatter = default_messages_state_formatter,
):
    """
    Connect a custom LangGraph to the Thenvoi platform.

    This function provides the input from Thenvoi chat rooms to your graph.
    Inside your graph, you decide when and how to use platform tools.

    The integration is split into two parts:
    1. **Input (Automatic)**: Chat room messages are automatically delivered to your graph
    2. **Output (Your Control)**: Inside your graph, you decide when/how to use platform tools

    Args:
        graph: Your compiled LangGraph (must have a checkpointer).
               Example: graph.compile(checkpointer=InMemorySaver())
        platform_client: ThenvoiPlatformClient instance for platform communication
        message_formatter: Function to convert platform messages to your graph's input format.
                          Default: converts to MessagesState format {"messages": [...]}

    Returns:
        ConnectedGraphAgent instance (already running)

    Example (Simple - MessagesState):
        >>> from langgraph.graph import StateGraph, MessagesState, START, END
        >>> from langgraph.checkpoint.memory import InMemorySaver
        >>> from thenvoi.agent.langgraph import connect_graph_to_platform
        >>> from thenvoi.agent.core import ThenvoiPlatformClient
        >>>
        >>> # Create platform client
        >>> platform_client = ThenvoiPlatformClient(
        ...     agent_id=agent_id,
        ...     api_key=api_key,
        ...     ws_url=ws_url,
        ...     thenvoi_restapi_url=thenvoi_restapi_url,
        ... )
        >>>
        >>> # Create your graph
        >>> def create_echo_graph():
        ...     def echo_node(state: MessagesState):
        ...         return {"messages": [{"role": "assistant", "content": "Echo!"}]}
        ...
        ...     graph = StateGraph(MessagesState)
        ...     graph.add_node("echo", echo_node)
        ...     graph.add_edge(START, "echo")
        ...     graph.add_edge("echo", END)
        ...     return graph.compile(checkpointer=InMemorySaver())
        >>>
        >>> agent = await connect_graph_to_platform(
        ...     graph=create_echo_graph(),
        ...     platform_client=platform_client,
        ... )

    Example (Advanced - Custom State with Tools):
        >>> from thenvoi.agent.langgraph import get_thenvoi_tools
        >>> from thenvoi.agent.core import ThenvoiPlatformClient
        >>>
        >>> # Create platform client (single client for everything)
        >>> platform_client = ThenvoiPlatformClient(
        ...     agent_id=agent_id,
        ...     api_key=api_key,
        ...     ws_url=ws_url,
        ...     thenvoi_restapi_url=thenvoi_restapi_url,
        ... )
        >>>
        >>> # Get platform tools using the same client
        >>> platform_tools = get_thenvoi_tools(
        ...     client=platform_client.api_client,
        ...     agent_id=agent_id
        ... )
        >>>
        >>> # Create graph with platform tools
        >>> graph = StateGraph(MyState)
        >>> graph.add_node("process", my_node)
        >>> graph.add_node("tools", ToolNode(platform_tools))  # Use Thenvoi tools!
        >>> my_graph = graph.compile(checkpointer=InMemorySaver())
        >>>
        >>> # Custom formatter to match your state
        >>> def my_formatter(message, sender_name):
        ...     return {"content": message.content, "room_id": message.chat_room_id}
        >>>
        >>> agent = await connect_graph_to_platform(
        ...     graph=my_graph,
        ...     platform_client=platform_client,
        ...     message_formatter=my_formatter,
        ... )
    """
    # Create agent with the provided platform client
    agent = ConnectedGraphAgent(
        graph=graph,
        platform_client=platform_client,
        message_formatter=message_formatter,
    )
    await agent.start()
    return agent
