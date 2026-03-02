"""LangGraph adapter."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, List

from thenvoi.adapters.optional_dependencies import ensure_optional_dependency

from thenvoi.core.nonfatal import NonFatalErrorRecorder
from thenvoi.core.protocols import MessagingDispatchToolsProtocol
from thenvoi.core.room_state import RoomFlagStore
from thenvoi.core.simple_adapter import SimpleAdapter, legacy_chat_turn_compat
from thenvoi.core.types import ChatMessageTurnContext
from thenvoi.converters.langchain import LangChainHistoryConverter, LangChainMessages
from thenvoi.runtime.prompts import render_system_prompt

_LANGGRAPH_IMPORT_ERROR: ImportError | None = None
_LANGGRAPH_INSTALL_COMMANDS = (
    "pip install 'thenvoi-sdk[langgraph]'",
    "uv add langgraph langchain-core",
)

try:
    from langgraph.pregel import Pregel
except ImportError as exc:
    _LANGGRAPH_IMPORT_ERROR = exc
    Pregel = Any

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langgraph.checkpoint.base import BaseCheckpointSaver

logger = logging.getLogger(__name__)


_BOOTSTRAP_TRACKING_WARN_THRESHOLD = 1000


def _ensure_langgraph_available() -> None:
    """Raise a consistent runtime error when LangGraph extras are missing."""
    ensure_optional_dependency(
        _LANGGRAPH_IMPORT_ERROR,
        package="langgraph",
        integration="LangGraph",
        install_commands=_LANGGRAPH_INSTALL_COMMANDS,
    )


class LangGraphAdapter(
    NonFatalErrorRecorder,
    SimpleAdapter[LangChainMessages, MessagingDispatchToolsProtocol],
):
    """
    LangGraph adapter using SimpleAdapter pattern.

    Two usage patterns:

    1. Simple (recommended for most users):
        adapter = LangGraphAdapter(
            llm=ChatOpenAI(model="gpt-4o"),
            checkpointer=InMemorySaver(),
            custom_section="You are a helpful assistant.",
        )

    2. Advanced (custom graph):
        def graph_factory(tools):
            return create_react_agent(llm, tools, checkpointer=checkpointer)

        adapter = LangGraphAdapter(graph_factory=graph_factory)

    Example:
        from langchain_openai import ChatOpenAI
        from langgraph.checkpoint.memory import InMemorySaver

        adapter = LangGraphAdapter(
            llm=ChatOpenAI(model="gpt-4o"),
            checkpointer=InMemorySaver(),
        )
        agent = Agent.create(adapter=adapter, agent_id="...", api_key="...")
        await agent.run()
    """

    def __init__(
        self,
        # Simple pattern: just provide llm and checkpointer
        llm: "BaseChatModel | None" = None,
        checkpointer: "BaseCheckpointSaver | None" = None,
        # Advanced pattern: provide a graph factory or static graph
        graph_factory: Callable[[List[Any]], Pregel] | None = None,
        graph: Pregel | None = None,
        # Common options
        prompt_template: str = "default",
        custom_section: str = "",
        additional_tools: List[Any] | None = None,
        enable_memory_tools: bool = False,
        history_converter: LangChainHistoryConverter | None = None,
        recursion_limit: int = 50,
    ):
        _ensure_langgraph_available()

        # Use default LangChain converter if not provided
        super().__init__(
            history_converter=history_converter or LangChainHistoryConverter()
        )

        # Simple pattern: create graph_factory from llm + checkpointer
        if llm is not None and graph_factory is None and graph is None:
            try:
                from langgraph.prebuilt import create_react_agent
            except ImportError as exc:
                ensure_optional_dependency(
                    exc,
                    package="langgraph.prebuilt",
                    integration="LangGraph",
                    install_commands=_LANGGRAPH_INSTALL_COMMANDS,
                )
                raise AssertionError("unreachable")

            additional = additional_tools or []

            def _make_graph_factory(llm, checkpointer, additional):
                def factory(thenvoi_tools: List[Any]) -> Pregel:
                    all_tools = thenvoi_tools + additional
                    return create_react_agent(
                        model=llm, tools=all_tools, checkpointer=checkpointer
                    )

                return factory

            graph_factory = _make_graph_factory(llm, checkpointer, additional)
            # Clear additional_tools since they're now baked into the factory
            additional_tools = []

        if not graph_factory and not graph:
            raise ValueError(
                "Must provide either llm (simple pattern) or graph_factory/graph (advanced pattern)"
            )

        self.graph_factory = graph_factory
        self._static_graph = graph
        self.prompt_template = prompt_template
        self.custom_section = custom_section
        self.additional_tools = additional_tools or []
        self.enable_memory_tools = enable_memory_tools
        self.recursion_limit = recursion_limit
        self._system_prompt: str = ""
        # Track rooms that have already been bootstrapped to avoid injecting
        # duplicate system prompts when the checkpointer retains state across
        # reconnections (on_cleanup doesn't clear checkpointer state).
        self._bootstrapped_rooms = RoomFlagStore()
        self._init_nonfatal_errors()

    async def on_started(self, agent_name: str, agent_description: str) -> None:
        """Render system prompt after agent metadata is fetched."""
        await super().on_started(agent_name, agent_description)
        self._system_prompt = render_system_prompt(
            template=self.prompt_template,
            agent_name=agent_name,
            agent_description=agent_description,
            custom_section=self.custom_section,
        )
        logger.info("LangGraph adapter started for agent: %s", agent_name)

    @legacy_chat_turn_compat
    async def on_message(
        self,
        turn: ChatMessageTurnContext[LangChainMessages, MessagingDispatchToolsProtocol],
    ) -> None:
        """Handle message with LangGraph."""
        msg = turn.msg
        tools = turn.tools
        history = turn.history
        participants_msg = turn.participants_msg
        contacts_msg = turn.contacts_msg
        is_session_bootstrap = turn.is_session_bootstrap
        room_id = turn.room_id

        from thenvoi.integrations.langgraph.langchain_tools import (
            agent_tools_to_langchain,
        )

        logger.info("[HANDLE] Message %s in room %s", msg.id, room_id)

        # Get LangChain tools
        langchain_tools = (
            agent_tools_to_langchain(
                tools, include_memory_tools=self.enable_memory_tools
            )
            + self.additional_tools
        )

        # Build or get graph
        if self.graph_factory:
            graph = self.graph_factory(langchain_tools)
        else:
            graph = self._static_graph

        if not graph:
            raise RuntimeError("No graph available")

        # Build messages
        messages: list[Any] = []

        # Session bootstrap: inject system prompt and any hydrated history.
        # Only inject the system prompt once per room to avoid duplicate system
        # messages when the checkpointer retains state across reconnections.
        if is_session_bootstrap:
            if self.graph_factory and self.mark_bootstrap_room(
                self._bootstrapped_rooms,
                room_id=room_id,
                is_session_bootstrap=is_session_bootstrap,
            ):
                messages.append(("system", self._system_prompt))
                if len(self._bootstrapped_rooms) == _BOOTSTRAP_TRACKING_WARN_THRESHOLD:
                    logger.warning(
                        "Bootstrap tracking has %d rooms; "
                        "on_cleanup may not be called for all rooms",
                        len(self._bootstrapped_rooms),
                    )
            if history:
                messages.extend(history)  # Already converted by history_converter

        # Inject metadata updates as user messages with canonical [System]: prefix.
        # Many providers require a single system message at the start; additional
        # updates are modeled as user-role metadata events.
        self.apply_metadata_updates(
            messages,
            participants_msg=participants_msg,
            contacts_msg=contacts_msg,
            make_entry=lambda update: ("user", update),
        )

        messages.append(("user", msg.format_for_llm()))

        graph_input = {"messages": messages}

        try:
            async for event in graph.astream_events(
                graph_input,
                config={
                    "configurable": {"thread_id": room_id},
                    "recursion_limit": self.recursion_limit,
                },
                version="v2",
            ):
                await self._handle_stream_event(event, room_id, tools)

            logger.info("[DONE] Message %s processed successfully", msg.id)

        except Exception as e:
            logger.error("Error processing message %s: %s", msg.id, e, exc_info=True)
            await self.report_adapter_error(
                tools,
                error=e,
                operation="report_error_event",
                room_id=room_id,
            )
            raise

    async def _handle_stream_event(
        self,
        event: Any,
        room_id: str,
        tools: MessagingDispatchToolsProtocol,
    ) -> None:
        """Handle streaming events from LangGraph."""
        event_type = event.get("event")

        if event_type == "on_tool_start":
            tool_name = event.get("name", "unknown")
            logger.info("[STREAM] on_tool_start: %s", tool_name)
            await self.send_tool_call_event(
                tools,
                payload=event,
                room_id=room_id,
                tool_name=tool_name,
            )

        elif event_type == "on_tool_end":
            tool_name = event.get("name", "unknown")
            logger.info("[STREAM] on_tool_end: %s", tool_name)
            await self.send_tool_result_event(
                tools,
                payload=event,
                room_id=room_id,
                tool_name=tool_name,
            )

    async def on_cleanup(self, room_id: str) -> None:
        """Clean up LangGraph state for a room."""
        self._bootstrapped_rooms.discard(room_id)
        if not self.graph_factory:
            return
        # Future graph_factory-specific cleanup (e.g. checkpointer) goes here
