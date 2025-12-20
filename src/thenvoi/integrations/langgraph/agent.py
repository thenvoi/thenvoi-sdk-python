"""
ThenvoiLangGraphAgent - LangGraph agent connected to Thenvoi platform.

Uses the SDK's runtime layer:
- ThenvoiLink: WebSocket + REST transport
- AgentRuntime: Room presence + execution management
- ExecutionContext: Per-room context and event handling
- AgentTools: Tool interface for LLM

KEY DESIGN:
    SDK does NOT send messages.
    LangGraph agent uses tools.send_message() to respond.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, List

from langgraph.pregel import Pregel
from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.base import BaseCheckpointSaver

from thenvoi.agents import BaseFrameworkAgent
from thenvoi.runtime import (
    AgentConfig,
    AgentTools,
    ExecutionContext,
    PlatformMessage,
    SessionConfig,
    render_system_prompt,
)
from .langchain_tools import agent_tools_to_langchain

logger = logging.getLogger(__name__)


class ThenvoiLangGraphAgent(BaseFrameworkAgent):
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

        adapter = ThenvoiLangGraphAgent(
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

        super().__init__(
            agent_id=agent_id,
            api_key=api_key,
            ws_url=ws_url,
            rest_url=rest_url,
            config=config,
            session_config=session_config,
        )

        self.graph_factory = graph_factory
        self._static_graph = graph
        self.prompt_template = prompt_template
        self.custom_section = custom_section
        self.additional_tools = additional_tools or []

        # Will be set after start()
        self._system_prompt: str = ""

    @property
    def system_prompt(self) -> str:
        """Get rendered system prompt."""
        return self._system_prompt

    async def _on_started(self) -> None:
        """Render system prompt after agent metadata is fetched."""
        self._system_prompt = render_system_prompt(
            template=self.prompt_template,
            agent_name=self.agent_name,
            agent_description=self.agent_description,
            custom_section=self.custom_section,
        )
        logger.info(f"LangGraph adapter started for agent: {self.agent_name}")

    def _extract_tool_call_id(self, output: str) -> str | None:
        """Extract tool_call_id from tool output string.

        Output format: "content='...' name='...' tool_call_id='call_xxx'"
        """
        match = re.search(r"tool_call_id='([^']+)'", output)
        return match.group(1) if match else None

    def _reconstruct_messages(
        self, history: list[dict[str, Any]]
    ) -> list[AIMessage | HumanMessage | ToolMessage]:
        """Reconstruct LangGraph messages from platform history.

        Parses stored tool events and creates proper LangChain message types:
        - tool_call + tool_result pairs -> AIMessage with tool_calls + ToolMessage
        - text -> HumanMessage or AIMessage

        Tool calls and results are paired by run_id (most reliable) or by name
        as fallback. The tool_call_id is extracted from the tool_result output.
        """
        messages: list[AIMessage | HumanMessage | ToolMessage] = []
        # Map run_id -> tool_call event for reliable matching
        pending_by_run_id: dict[str, dict[str, Any]] = {}
        # Fallback: list of tool_calls without run_id, matched by name (LIFO)
        pending_by_name: dict[str, list[dict[str, Any]]] = {}

        for hist in history:
            message_type = hist.get("message_type", "text")
            content = hist.get("content", "")
            role = hist.get("role")
            sender_name = hist.get("sender_name", "")

            if message_type == "tool_call":
                # Store pending - indexed by run_id for reliable matching
                try:
                    event = json.loads(content)
                    run_id = event.get("run_id")
                    tool_name = event.get("name", "unknown")

                    if run_id:
                        pending_by_run_id[run_id] = event
                    else:
                        # Fallback: store by name (as stack for LIFO matching)
                        if tool_name not in pending_by_name:
                            pending_by_name[tool_name] = []
                        pending_by_name[tool_name].append(event)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse tool_call event: {content[:100]}")

            elif message_type == "tool_result":
                # Parse stored event JSON
                try:
                    event = json.loads(content)
                    tool_name = event.get("name", "unknown")
                    run_id = event.get("run_id")
                    output = event.get("data", {}).get("output", "")

                    # Extract tool_call_id from output string
                    tool_call_id = self._extract_tool_call_id(str(output))
                    if not tool_call_id:
                        # Fallback to run_id if no tool_call_id found
                        tool_call_id = run_id or "unknown"

                    # Find matching pending tool_call
                    matching_call = None

                    # First: try to match by run_id (most reliable)
                    if run_id and run_id in pending_by_run_id:
                        matching_call = pending_by_run_id.pop(run_id)

                    # Fallback: match by name (LIFO - pop from end of stack)
                    if not matching_call and tool_name in pending_by_name:
                        name_stack = pending_by_name[tool_name]
                        if name_stack:
                            matching_call = name_stack.pop()
                            if not name_stack:
                                del pending_by_name[tool_name]

                    if matching_call:
                        tool_input = matching_call.get("data", {}).get("input", {})

                        # Create AIMessage with tool_calls using correct tool_call_id
                        messages.append(
                            AIMessage(
                                content="",
                                tool_calls=[
                                    {
                                        "id": tool_call_id,
                                        "name": tool_name,
                                        "args": tool_input,
                                    }
                                ],
                            )
                        )
                    else:
                        # No matching tool_call - emit ToolMessage only, don't fabricate AIMessage
                        logger.warning(
                            f"tool_result without matching tool_call: "
                            f"name={tool_name}, run_id={run_id}"
                        )

                    # Create ToolMessage with matching tool_call_id
                    messages.append(
                        ToolMessage(
                            content=str(output),
                            tool_call_id=tool_call_id,
                        )
                    )
                except json.JSONDecodeError:
                    logger.warning(
                        f"Failed to parse tool_result event: {content[:100]}"
                    )

            elif message_type == "text":
                if role == "assistant":
                    # SKIP assistant text messages - they're redundant with tool_call/tool_result
                    # Including them teaches LLM to respond with text instead of using tools
                    logger.debug(f"Skipping redundant assistant text: {content[:50]}")
                else:
                    messages.append(HumanMessage(content=f"[{sender_name}]: {content}"))

            # Skip other message types (thought, error, task, etc.)

        # Warn about unmatched tool calls
        unmatched_count = len(pending_by_run_id) + sum(
            len(v) for v in pending_by_name.values()
        )
        if unmatched_count:
            logger.warning(
                f"Found {unmatched_count} tool_calls without matching tool_results"
            )
            for run_id, call in pending_by_run_id.items():
                logger.warning(
                    f"Unmatched tool_call: name={call.get('name')}, run_id={run_id}"
                )
            for name, calls in pending_by_name.items():
                for call in calls:
                    logger.warning(
                        f"Unmatched tool_call: name={name}, "
                        f"input={call.get('data', {}).get('input', {})}"
                    )

        return messages

    async def _handle_message(
        self,
        msg: PlatformMessage,
        tools: AgentTools,
        ctx: ExecutionContext,
        history: list[dict[str, Any]] | None,
        participants_msg: str | None,
    ) -> None:
        """
        Handle incoming message.

        Handler receives AgentTools, NOT ThenvoiAgent.
        LLM uses tools.send_message() to respond.

        KEY DESIGN:
        - System prompt sent ONLY on first message (history is not None)
        - Historical messages injected on first message to prime checkpointer
        - Participant list injected ONLY when it changes (participants_msg is not None)
        - Subsequent messages: user message only (unless participants changed)
        """
        room_id = msg.room_id  # = thread_id for LangGraph
        is_first_message = history is not None

        logger.info(
            f"[HANDLE] Message {msg.id} in room {room_id}, is_first={history is not None}"
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
        # Mixed types: tuples for system/user, LangChain messages for history
        messages: list[Any] = []

        if is_first_message:
            # FIRST MESSAGE: Include system prompt + historical context
            # Only inject system prompt if using graph_factory (pre-compiled graph has its own)
            if self.graph_factory:
                messages.append(("system", self._system_prompt))
                logger.info(f"Room {room_id}: Sending system prompt (first message)")
            logger.debug(f"System prompt:\n{self._system_prompt}")

            # Inject historical messages - reconstruct proper LangGraph message types
            if history:
                logger.info(f"[HISTORY] Raw history has {len(history)} items")
                for i, h in enumerate(history):
                    logger.info(
                        f"[HISTORY] [{i}] type={h.get('message_type', 'text')} role={h.get('role')} sender={h.get('sender_name', '?')}"
                    )
                reconstructed = self._reconstruct_messages(history)
                messages.extend(reconstructed)
                logger.info(
                    f"[HISTORY] Reconstructed {len(reconstructed)} LangChain messages"
                )
                for i, m in enumerate(reconstructed):
                    logger.info(f"[HISTORY] [{i}] {type(m).__name__}: {str(m)[:200]}")

        # Inject participants message ONLY if changed
        if participants_msg:
            messages.append(("system", participants_msg))
            logger.info(
                f"Room {room_id}: Participants updated: "
                f"{[p.get('name') for p in ctx.participants]}"
            )

        # Add current message (always)
        messages.append(("user", msg.format_for_llm()))

        graph_input = {"messages": messages}

        logger.info(f"[INVOKE] Sending {len(messages)} messages to LangGraph")
        for i, m in enumerate(messages):
            if isinstance(m, tuple):
                logger.info(f"[INVOKE] [{i}] tuple: {m[0]} - {str(m[1])[:100]}")
            else:
                logger.info(f"[INVOKE] [{i}] {type(m).__name__}: {str(m)[:100]}")

        # Run LangGraph - LLM uses tools.send_message() internally
        try:
            # Use astream_events for visibility into tool calls
            async for event in graph.astream_events(
                graph_input,
                config={"configurable": {"thread_id": room_id}},
                version="v2",
            ):
                await self._handle_stream_event(event, room_id, tools)

            logger.info(f"[DONE] Message {msg.id} processed successfully")

        except Exception as e:
            logger.error(f"Error processing message {msg.id}: {e}", exc_info=True)
            await self._report_error(tools, f"Error processing message: {e}")
            raise

    async def _handle_stream_event(
        self, event: Any, room_id: str, tools: AgentTools
    ) -> None:
        """Handle streaming events from LangGraph - send raw events to platform."""
        event_type = event.get("event")

        # Log all events for debugging
        if event_type in (
            "on_chat_model_start",
            "on_chat_model_end",
            "on_chat_model_stream",
        ):
            logger.info(f"[STREAM] {event_type}: {event.get('name', '?')}")
        elif event_type in ("on_chain_start", "on_chain_end"):
            logger.info(f"[STREAM] {event_type}: {event.get('name', '?')}")

        if event_type == "on_tool_start":
            tool_name = event.get("name", "unknown")
            logger.info(f"[STREAM] on_tool_start: {tool_name}")

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
            logger.info(f"[STREAM] on_tool_end: {tool_name}")

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


class ThenvoiLangGraphMCPAgent(ThenvoiLangGraphAgent):
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
            **kwargs: Arguments passed to ThenvoiLangGraphAgent
        """
        super().__init__(**kwargs)
        self.mcp_server_url = mcp_server_url

    async def _handle_message(
        self,
        msg: PlatformMessage,
        tools: AgentTools,
        ctx: ExecutionContext,
        history: list[dict[str, Any]] | None,
        participants_msg: str | None,
    ) -> None:
        """Handle message with MCP tools."""
        # TODO: Load tools from MCP server instead of AgentTools
        # For now, delegate to parent
        await super()._handle_message(msg, tools, ctx, history, participants_msg)


async def create_langgraph_agent(
    agent_id: str,
    api_key: str,
    llm: BaseChatModel,
    checkpointer: BaseCheckpointSaver | None = None,
    ws_url: str = "wss://api.thenvoi.com/ws",
    thenvoi_restapi_url: str = "https://api.thenvoi.com",
    additional_tools: List[Any] | None = None,
    custom_instructions: str = "",
    **kwargs,
) -> None:
    """
    Create and run a LangGraph agent connected to Thenvoi platform.

    This is a high-level convenience function that:
    1. Creates a ReAct agent with the provided LLM
    2. Adds Thenvoi platform tools automatically
    3. Connects to the platform and runs until interrupted

    Args:
        agent_id: Thenvoi agent ID
        api_key: Thenvoi API key
        llm: LangChain chat model (e.g., ChatOpenAI)
        checkpointer: LangGraph checkpointer for conversation memory
        ws_url: WebSocket URL for real-time events
        thenvoi_restapi_url: REST API URL
        additional_tools: Extra tools to add alongside platform tools
        custom_instructions: Custom instructions to add to system prompt
        **kwargs: Additional arguments for ThenvoiLangGraphAgent

    Example:
        await create_langgraph_agent(
            agent_id="...",
            api_key="...",
            llm=ChatOpenAI(model="gpt-4o"),
            checkpointer=InMemorySaver(),
        )
    """
    additional = additional_tools or []

    def graph_factory(thenvoi_tools: List[Any]) -> Pregel:
        all_tools = thenvoi_tools + additional
        return create_agent(model=llm, tools=all_tools, checkpointer=checkpointer)

    adapter = ThenvoiLangGraphAgent(
        graph_factory=graph_factory,
        agent_id=agent_id,
        api_key=api_key,
        ws_url=ws_url,
        rest_url=thenvoi_restapi_url,
        custom_section=custom_instructions,
        **kwargs,
    )
    await adapter.run()
