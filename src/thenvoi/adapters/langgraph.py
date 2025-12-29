"""LangGraph adapter."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Callable, List

from langgraph.pregel import Pregel

from thenvoi.core.protocols import AgentToolsProtocol
from thenvoi.core.simple_adapter import SimpleAdapter
from thenvoi.core.types import PlatformMessage
from thenvoi.converters.langchain import LangChainHistoryConverter, LangChainMessages
from thenvoi.runtime.prompts import render_system_prompt

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langgraph.checkpoint.base import BaseCheckpointSaver

logger = logging.getLogger(__name__)


class LangGraphAdapter(SimpleAdapter[LangChainMessages]):
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
        history_converter: LangChainHistoryConverter | None = None,
    ):
        # Use default LangChain converter if not provided
        super().__init__(
            history_converter=history_converter or LangChainHistoryConverter()
        )

        # Simple pattern: create graph_factory from llm + checkpointer
        if llm is not None and graph_factory is None and graph is None:
            from langgraph.prebuilt import create_react_agent

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
        self._system_prompt: str = ""

    async def on_started(self, agent_name: str, agent_description: str) -> None:
        """Render system prompt after agent metadata is fetched."""
        await super().on_started(agent_name, agent_description)
        self._system_prompt = render_system_prompt(
            template=self.prompt_template,
            agent_name=agent_name,
            agent_description=agent_description,
            custom_section=self.custom_section,
        )
        logger.info(f"LangGraph adapter started for agent: {agent_name}")

    async def on_message(
        self,
        msg: PlatformMessage,
        tools: AgentToolsProtocol,
        history: LangChainMessages,  # Fully typed!
        participants_msg: str | None,
        *,
        is_session_bootstrap: bool,
        room_id: str,
    ) -> None:
        """Handle message with LangGraph."""
        from thenvoi.integrations.langgraph.langchain_tools import (
            agent_tools_to_langchain,
        )

        logger.info(f"[HANDLE] Message {msg.id} in room {room_id}")

        # Get LangChain tools
        langchain_tools = agent_tools_to_langchain(tools) + self.additional_tools

        # Build or get graph
        if self.graph_factory:
            graph = self.graph_factory(langchain_tools)
        else:
            graph = self._static_graph

        if not graph:
            raise RuntimeError("No graph available")

        # Build messages
        messages: list[Any] = []

        # Session bootstrap: inject system prompt and any hydrated history
        if is_session_bootstrap:
            if self.graph_factory:
                messages.append(("system", self._system_prompt))
            if history:
                messages.extend(history)  # Already converted by history_converter

        if participants_msg:
            messages.append(("system", participants_msg))

        messages.append(("user", msg.format_for_llm()))

        graph_input = {"messages": messages}

        try:
            async for event in graph.astream_events(
                graph_input,
                config={"configurable": {"thread_id": room_id}},
                version="v2",
            ):
                await self._handle_stream_event(event, room_id, tools)

            logger.info(f"[DONE] Message {msg.id} processed successfully")

        except Exception as e:
            logger.error(f"Error processing message {msg.id}: {e}", exc_info=True)
            try:
                await tools.send_event(content=f"Error: {e}", message_type="error")
            except Exception:
                pass
            raise

    async def _handle_stream_event(
        self,
        event: Any,
        room_id: str,
        tools: AgentToolsProtocol,
    ) -> None:
        """Handle streaming events from LangGraph."""
        event_type = event.get("event")

        if event_type == "on_tool_start":
            tool_name = event.get("name", "unknown")
            logger.info(f"[STREAM] on_tool_start: {tool_name}")
            try:
                await tools.send_event(
                    content=json.dumps(event, default=str),
                    message_type="tool_call",
                )
            except Exception as e:
                logger.warning(f"Failed to send tool_call event: {e}")

        elif event_type == "on_tool_end":
            tool_name = event.get("name", "unknown")
            logger.info(f"[STREAM] on_tool_end: {tool_name}")
            try:
                await tools.send_event(
                    content=json.dumps(event, default=str),
                    message_type="tool_result",
                )
            except Exception as e:
                logger.warning(f"Failed to send tool_result event: {e}")

    async def on_cleanup(self, room_id: str) -> None:
        """Clean up LangGraph checkpointer."""
        if not self.graph_factory:
            return
        # Cleanup logic here
