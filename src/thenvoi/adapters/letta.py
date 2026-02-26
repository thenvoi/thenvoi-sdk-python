"""Letta adapter using AsyncLetta SDK with client_tools pattern."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from pydantic import ValidationError

from thenvoi.converters.letta import LettaHistoryConverter, LettaSessionState
from thenvoi.core.protocols import AgentToolsProtocol
from thenvoi.core.simple_adapter import SimpleAdapter
from thenvoi.core.types import PlatformMessage
from thenvoi.runtime.prompts import render_system_prompt

logger = logging.getLogger(__name__)

# Maximum number of tool execution rounds before aborting. Prevents infinite loops
# if Letta Cloud keeps returning approval_request_messages.
_MAX_TOOL_ROUNDS: int = 50

# Platform tools whose execution should not be reported as tool_call/tool_result
# events — they already produce visible output (messages or events) on the platform.
_SILENT_REPORTING_TOOLS: frozenset[str] = frozenset(
    {
        "thenvoi_send_message",
        "thenvoi_send_event",
    }
)

# Letta-specific preamble prepended to the system prompt when writing to the
# agent's instruction block.  Letta models tend to respond with plain
# assistant_message text instead of calling client_tools — this enforces
# strict tool usage so messages actually reach the platform.
_LETTA_TOOL_ENFORCEMENT = """\
## MANDATORY: You MUST use tools to communicate

You are connected to a multi-agent chat platform via client tools.
Your plain text responses (assistant_message) are NOT delivered to anyone.
The ONLY way to communicate is by calling the provided tools.

EVERY response MUST include at least one tool call. Specifically:
- To send a message: call `thenvoi_send_message(content, mentions)` — this is REQUIRED
- To share your thinking: call `thenvoi_send_event(content, message_type="thought")`
- NEVER respond with just plain text — it will be silently discarded

## WRONG (message is lost):
Just responding with plain text like this.

## CORRECT:
Call thenvoi_send_message(content="Your response", mentions=["@user-id"])

If you respond without calling `thenvoi_send_message`, the user sees NOTHING.

"""


@dataclass
class LettaAdapterConfig:
    """Configuration for the Letta adapter."""

    agent_id: str | None = None
    model: str | None = None
    api_key: str | None = None
    base_url: str = "https://api.letta.com"
    custom_section: str = ""
    include_base_instructions: bool = True
    enable_execution_reporting: bool = False
    enable_task_events: bool = True
    enable_memory_tools: bool = False
    persona: str | None = None
    turn_timeout_s: float = 300.0
    memory_blocks: list[dict[str, str]] = field(default_factory=list)
    summary_max_length: int = 150


@dataclass
class _RoomContext:
    """Per-room state for a Letta agent."""

    agent_id: str
    conversation_id: str | None = None
    last_interaction: datetime | None = None
    summary: str | None = None


class LettaAdapter(SimpleAdapter[LettaSessionState]):
    """
    Letta adapter using the Letta Python SDK (letta-client).

    Uses AsyncLetta REST API with client_tools for bidirectional tool execution.
    When the Letta agent calls a platform tool, the adapter executes it locally
    and returns the result via the approval_request_message pattern.

    Example:
        adapter = LettaAdapter(
            config=LettaAdapterConfig(
                model="openai/gpt-4o",
                api_key="sk-let-...",
            ),
        )
        agent = Agent.create(adapter=adapter, agent_id="...", api_key="...")
        await agent.run()
    """

    def __init__(
        self,
        config: LettaAdapterConfig | None = None,
        history_converter: LettaHistoryConverter | None = None,
    ) -> None:
        super().__init__(history_converter=history_converter or LettaHistoryConverter())
        self.config = config or LettaAdapterConfig()

        # Letta SDK async client (shared across rooms)
        self._client: Any = None

        # Per-room state
        self._rooms: dict[str, _RoomContext] = {}

        # Concurrency
        self._rpc_lock = asyncio.Lock()

        # Built during on_started
        self._system_prompt: str = ""
        self._client_tools: list[dict[str, Any]] = []

    async def on_started(self, agent_name: str, agent_description: str) -> None:
        """Build system prompt and create Letta SDK client."""
        await super().on_started(agent_name, agent_description)

        self._system_prompt = render_system_prompt(
            agent_name=agent_name,
            agent_description=agent_description,
            custom_section=self.config.custom_section,
            include_base_instructions=self.config.include_base_instructions,
        )

        try:
            from letta_client import AsyncLetta  # type: ignore[import-not-found]  # optional dependency
        except ImportError:
            raise ImportError(
                "letta-client is required for LettaAdapter. "
                "Install with: pip install thenvoi-sdk[letta]"
            )

        client_kwargs: dict[str, Any] = {"api_key": self.config.api_key}
        if self.config.base_url != "https://api.letta.com":
            client_kwargs["base_url"] = self.config.base_url
        self._client = AsyncLetta(**client_kwargs)

        logger.info("Letta adapter started for agent: %s", agent_name)

    async def on_message(
        self,
        msg: PlatformMessage,
        tools: AgentToolsProtocol,
        history: LettaSessionState,
        participants_msg: str | None,
        contacts_msg: str | None,
        *,
        is_session_bootstrap: bool,
        room_id: str,
    ) -> None:
        """Handle incoming message via Letta API with client_tools."""
        async with self._rpc_lock:
            await self._handle_message(
                msg=msg,
                tools=tools,
                history=history,
                participants_msg=participants_msg,
                contacts_msg=contacts_msg,
                is_session_bootstrap=is_session_bootstrap,
                room_id=room_id,
            )

    async def _handle_message(
        self,
        msg: PlatformMessage,
        tools: AgentToolsProtocol,
        history: LettaSessionState,
        participants_msg: str | None,
        contacts_msg: str | None,
        *,
        is_session_bootstrap: bool,
        room_id: str,
    ) -> None:
        """Inner message handler (called under lock)."""
        if not self._client:
            logger.error("Letta client not initialized, dropping message %s", msg.id)
            await self._report_error(tools, "Letta adapter not initialized")
            return

        # Build client_tools from platform tool schemas (once, lazily)
        if not self._client_tools:
            self._client_tools = self._build_client_tools(tools)

        # Ensure Letta agent exists for this room
        agent_id = await self._ensure_agent(room_id, history, tools)

        # Build user message content
        # NOTE: Unlike other adapters that pass participants_msg and contacts_msg
        # as separate system messages, Letta uses a single-message API where each
        # call sends one user message.  We inject system context as [System]-
        # prefixed lines in the user message body so the agent sees participant
        # and contact updates inline with the conversation.
        parts: list[str] = []

        # Inject rejoin context when resuming after absence
        room_ctx = self._rooms.get(room_id)
        if is_session_bootstrap and room_ctx and room_ctx.last_interaction:
            time_ago = self._format_time_ago(room_ctx.last_interaction)
            rejoin_msg = f"[System: You have rejoined this room after {time_ago}."
            if room_ctx.summary:
                rejoin_msg += f" Previous topic: {room_ctx.summary}"
            rejoin_msg += "]"
            parts.append(rejoin_msg)

        # Inject participants update
        if participants_msg:
            parts.append(f"[System]: {participants_msg}")

        # Inject contacts update
        if contacts_msg:
            parts.append(f"[System]: {contacts_msg}")

        # Add the actual message
        user_message = msg.format_for_llm()
        parts.append(user_message)

        content = "\n\n".join(parts)

        logger.info(
            "Room %s: Sending message to Letta agent %s",
            room_id,
            agent_id,
        )

        # Send message and handle tool execution loop
        try:
            final_text_parts = await asyncio.wait_for(
                self._send_and_handle_tools(
                    agent_id=agent_id,
                    content=content,
                    tools=tools,
                    room_id=room_id,
                    reply_to_sender_id=msg.sender_id,
                ),
                timeout=self.config.turn_timeout_s,
            )

            # Update room context after successful turn
            room_ctx = self._rooms.get(room_id)
            if room_ctx:
                room_ctx.last_interaction = datetime.now(timezone.utc)
                if final_text_parts:
                    room_ctx.summary = self._extract_summary(
                        final_text_parts, self.config.summary_max_length
                    )
        except asyncio.TimeoutError:
            logger.error(
                "Room %s: Letta turn timed out after %ss",
                room_id,
                self.config.turn_timeout_s,
            )
            await self._report_error(
                tools,
                f"Letta agent response timed out after {self.config.turn_timeout_s}s",
            )
        except Exception as e:
            logger.exception("Room %s: Error during Letta turn: %s", room_id, e)
            await self._report_error(tools, str(e))

    async def _send_and_handle_tools(
        self,
        agent_id: str,
        content: str,
        tools: AgentToolsProtocol,
        room_id: str,
        reply_to_sender_id: str = "",
    ) -> list[str]:
        """Send message to Letta and handle the tool execution loop.

        Returns the list of assistant text parts collected during the turn.
        """
        messages = [{"role": "user", "content": content}]
        final_text_parts: list[str] = []
        used_send_message = False  # tracks if agent called thenvoi_send_message

        for _round in range(_MAX_TOOL_ROUNDS):
            response = await self._client.agents.messages.create(
                agent_id=agent_id,
                messages=messages,
                client_tools=self._client_tools,
            )

            has_approval_request = False

            for resp_msg in response.messages:
                msg_type = getattr(resp_msg, "message_type", None)
                logger.debug(
                    "Room %s: Letta response message type=%s", room_id, msg_type
                )

                if msg_type == "assistant_message":
                    text = getattr(resp_msg, "content", "") or ""
                    logger.debug(
                        "Room %s: assistant_message content=%r", room_id, text[:200]
                    )
                    if text:
                        final_text_parts.append(text)

                elif msg_type == "tool_call_message":
                    # Letta's internal tool calls (e.g. memory tools) — distinct from
                    # client_tools which arrive as approval_request_message.
                    # Report tool call if enabled
                    if self.config.enable_execution_reporting:
                        tool_name = getattr(
                            getattr(resp_msg, "tool_call", None), "name", "unknown"
                        )
                        if tool_name not in _SILENT_REPORTING_TOOLS:
                            await tools.send_event(
                                content=json.dumps(
                                    {
                                        "name": tool_name,
                                        "args": getattr(
                                            getattr(resp_msg, "tool_call", None),
                                            "arguments",
                                            "{}",
                                        ),
                                    }
                                ),
                                message_type="tool_call",
                            )

                elif msg_type == "tool_return_message":
                    # Report tool result if enabled
                    if self.config.enable_execution_reporting:
                        tool_name = getattr(resp_msg, "tool_name", "unknown")
                        if tool_name not in _SILENT_REPORTING_TOOLS:
                            await tools.send_event(
                                content=json.dumps(
                                    {
                                        "name": tool_name,
                                        "output": getattr(resp_msg, "tool_return", ""),
                                    }
                                ),
                                message_type="tool_result",
                            )

                elif msg_type == "approval_request_message":
                    has_approval_request = True
                    tool_call = getattr(resp_msg, "tool_call", None)
                    if not tool_call:
                        logger.warning(
                            "Room %s: approval_request_message without tool_call",
                            room_id,
                        )
                        continue

                    tool_name = getattr(tool_call, "name", "")
                    tool_call_id = getattr(tool_call, "tool_call_id", "")
                    raw_args = getattr(tool_call, "arguments", "{}")

                    try:
                        args = (
                            json.loads(raw_args)
                            if isinstance(raw_args, str)
                            else raw_args
                        )
                    except json.JSONDecodeError:
                        args = {}

                    logger.debug(
                        "Room %s: Executing tool %s with args %s",
                        room_id,
                        tool_name,
                        args,
                    )

                    # Report tool call if enabled
                    if self.config.enable_execution_reporting:
                        if tool_name not in _SILENT_REPORTING_TOOLS:
                            await tools.send_event(
                                content=json.dumps({"name": tool_name, "args": args}),
                                message_type="tool_call",
                            )

                    # Track if agent used send_message itself
                    if tool_name == "thenvoi_send_message":
                        used_send_message = True

                    # Execute the tool
                    try:
                        result = await tools.execute_tool_call(tool_name, args)
                        result_str = (
                            json.dumps(result, default=str)
                            if not isinstance(result, str)
                            else result
                        )
                        status = "success"
                    except ValidationError as e:
                        errors = "; ".join(
                            f"{err['loc'][0]}: {err['msg']}" for err in e.errors()
                        )
                        result_str = f"Invalid arguments for {tool_name}: {errors}"
                        status = "error"
                        logger.error("Validation error for tool %s: %s", tool_name, e)
                    except Exception as e:
                        result_str = f"Error: {e}"
                        status = "error"
                        logger.exception("Tool %s failed: %s", tool_name, e)

                    # Report tool result if enabled
                    if self.config.enable_execution_reporting:
                        if tool_name not in _SILENT_REPORTING_TOOLS:
                            await tools.send_event(
                                content=json.dumps(
                                    {"name": tool_name, "output": result_str}
                                ),
                                message_type="tool_result",
                            )

                    # Send approval result back to Letta
                    messages = [
                        {
                            "type": "approval",
                            "approvals": [
                                {
                                    "type": "tool",
                                    "tool_call_id": tool_call_id,
                                    "tool_return": result_str,
                                    "status": status,
                                }
                            ],
                        }
                    ]

            # If no approval request was found, we're done
            if not has_approval_request:
                break
        else:
            logger.error(
                "Room %s: Exceeded %d tool execution rounds",
                room_id,
                _MAX_TOOL_ROUNDS,
            )
            raise RuntimeError(
                f"Exceeded maximum tool execution rounds ({_MAX_TOOL_ROUNDS})"
            )

        # If the agent already called thenvoi_send_message via client_tools,
        # the message is on the platform — no auto-relay needed.  Otherwise
        # fall back to relaying the assistant_message text so the user still
        # sees a response.
        if used_send_message:
            logger.debug(
                "Room %s: Agent used thenvoi_send_message, skipping auto-relay",
                room_id,
            )
        elif final_text_parts:
            final_text = "\n\n".join(final_text_parts)
            mentions = [reply_to_sender_id] if reply_to_sender_id else None
            logger.info(
                "Room %s: Auto-relaying assistant text "
                "(agent did not use send_message)",
                room_id,
            )
            await tools.send_message(final_text, mentions=mentions)
        else:
            logger.debug("Room %s: Letta turn complete, no output", room_id)

        return final_text_parts

    async def _ensure_agent(
        self,
        room_id: str,
        history: LettaSessionState,
        tools: AgentToolsProtocol,
    ) -> str:
        """Ensure a Letta agent exists for this room, creating or resuming."""
        # Already have an agent for this room
        if room_id in self._rooms:
            return self._rooms[room_id].agent_id

        # Try to resume: prefer history agent_id, fall back to config agent_id
        resume_agent_id = (
            history.agent_id if history.has_agent() else self.config.agent_id
        )
        if resume_agent_id:
            try:
                await self._client.agents.retrieve(resume_agent_id)
                self._rooms[room_id] = _RoomContext(
                    agent_id=resume_agent_id,
                    conversation_id=history.conversation_id or None,
                )

                # Update an instruction block with the Thenvoi system prompt so
                # the agent has tool-use instructions even when reusing a
                # pre-existing agent.  Try common label names in priority order.
                await self._update_instruction_block(resume_agent_id, room_id)

                logger.info("Room %s: Resumed Letta agent %s", room_id, resume_agent_id)
                await self._emit_task_event(tools, room_id, resume_agent_id)
                return resume_agent_id
            except Exception as e:
                logger.warning(
                    "Room %s: Failed to resume agent %s: %s",
                    room_id,
                    resume_agent_id,
                    e,
                )

        # Create new agent
        memory_blocks = (
            list(self.config.memory_blocks) if self.config.memory_blocks else []
        )

        # Add persona block with system prompt + tool enforcement
        base_prompt = self.config.persona or self._system_prompt
        persona_value = _LETTA_TOOL_ENFORCEMENT + base_prompt
        memory_blocks.insert(0, {"label": "persona", "value": persona_value})

        create_kwargs: dict[str, Any] = {
            "memory_blocks": memory_blocks,
            "include_base_tools": True,
        }
        if self.config.model:
            create_kwargs["model"] = self.config.model

        agent = await self._client.agents.create(**create_kwargs)
        agent_id = agent.id

        self._rooms[room_id] = _RoomContext(agent_id=agent_id)
        logger.info("Room %s: Created Letta agent %s", room_id, agent_id)

        await self._emit_task_event(tools, room_id, agent_id)
        return agent_id

    # Labels tried (in order) when injecting the system prompt into a
    # pre-existing agent's memory.  "persona" is the Letta default; others
    # are common alternatives created by Letta Cloud templates.
    _INSTRUCTION_BLOCK_LABELS: tuple[str, ...] = (
        "persona",
        "custom_instructions",
        "system_instructions",
    )

    async def _update_instruction_block(self, agent_id: str, room_id: str) -> None:
        """Update (or create) a memory block with the Thenvoi system prompt."""
        base = self.config.persona or self._system_prompt
        value = _LETTA_TOOL_ENFORCEMENT + base

        # Try known instruction-block labels in priority order
        for label in self._INSTRUCTION_BLOCK_LABELS:
            try:
                await self._client.agents.blocks.update(
                    label,
                    agent_id=agent_id,
                    value=value,
                )
                logger.debug(
                    "Room %s: Updated %r block for agent %s",
                    room_id,
                    label,
                    agent_id,
                )
                return
            except Exception:
                # Label not found on this agent, try next
                logger.debug(
                    "Room %s: Block %r not found for agent %s, trying next",
                    room_id,
                    label,
                    agent_id,
                )
                continue

        # None of the known labels exist — create a "persona" block
        try:
            await self._client.agents.blocks.create(
                agent_id=agent_id,
                label="persona",
                value=value,
            )
            logger.debug(
                "Room %s: Created persona block for agent %s", room_id, agent_id
            )
        except Exception as e:
            logger.warning(
                "Room %s: Could not update or create instruction block: %s",
                room_id,
                e,
            )

    async def _emit_task_event(
        self,
        tools: AgentToolsProtocol,
        room_id: str,
        agent_id: str,
    ) -> None:
        """Emit a task event with agent/room mapping metadata."""
        if not self.config.enable_task_events:
            return
        try:
            room_ctx = self._rooms.get(room_id)
            conversation_id = room_ctx.conversation_id if room_ctx else None
            metadata: dict[str, Any] = {
                "letta_agent_id": agent_id,
                "letta_room_id": room_id,
                "letta_created_at": datetime.now(timezone.utc).isoformat(),
            }
            if conversation_id:
                metadata["letta_conversation_id"] = conversation_id
            await tools.send_event(
                content=f"Letta agent {agent_id} active for room {room_id}",
                message_type="task",
                metadata=metadata,
            )
        except Exception as e:
            logger.warning("Failed to emit task event: %s", e)

    async def on_cleanup(self, room_id: str) -> None:
        """Clean up per-room state. Does NOT delete the Letta agent."""
        async with self._rpc_lock:
            room_ctx = self._rooms.get(room_id)
            if room_ctx and self._client:
                await self._consolidate_memory(room_ctx.agent_id, room_id)
            self._rooms.pop(room_id, None)
        logger.debug("Room %s: Cleaned up Letta adapter state", room_id)

    async def _consolidate_memory(self, agent_id: str, room_id: str) -> None:
        """Send a consolidation prompt so the agent saves key context to memory.

        Best-effort: failures are logged but do not propagate.
        """
        try:
            await self._client.agents.messages.create(
                agent_id=agent_id,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "[System: You are leaving this room. Consolidate key "
                            "decisions, action items, and important context into "
                            "your memory now.]"
                        ),
                    }
                ],
            )
            logger.debug(
                "Room %s: Sent memory consolidation prompt to agent %s",
                room_id,
                agent_id,
            )
        except Exception as e:
            logger.warning("Room %s: Memory consolidation failed: %s", room_id, e)

    def _build_client_tools(self, tools: AgentToolsProtocol) -> list[dict[str, Any]]:
        """Build client_tools list from platform tool schemas."""
        schemas = tools.get_openai_tool_schemas(
            include_memory=self.config.enable_memory_tools
        )
        client_tools: list[dict[str, Any]] = []
        for schema in schemas:
            func = schema.get("function", {})
            client_tools.append(
                {
                    "name": func.get("name", ""),
                    "description": func.get("description", ""),
                    "parameters": func.get("parameters", {}),
                }
            )
        return client_tools

    @staticmethod
    def _format_time_ago(dt: datetime) -> str:
        """Format a datetime as a human-readable time-ago string."""
        now = datetime.now(timezone.utc)
        # Ensure dt is timezone-aware for comparison
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        delta = now - dt
        total_seconds = int(delta.total_seconds())
        if total_seconds < 60:
            return f"{total_seconds}s"
        minutes = total_seconds // 60
        if minutes < 60:
            return f"{minutes}m"
        hours = minutes // 60
        if hours < 24:
            return f"{hours}h" if hours > 1 else "1 hour"
        days = hours // 24
        return f"{days}d" if days > 1 else "1 day"

    @staticmethod
    def _extract_summary(parts: list[str], max_length: int = 150) -> str:
        """Extract a brief summary from assistant text parts.

        Uses the first sentence if available, otherwise truncates to max_length.
        """
        text = " ".join(parts).strip()
        if not text:
            return ""

        # Find the earliest sentence-ending delimiter
        earliest_idx = -1
        for delimiter in (".", "!", "?"):
            idx = text.find(delimiter)
            if idx != -1 and (earliest_idx == -1 or idx < earliest_idx):
                earliest_idx = idx
        if earliest_idx != -1:
            sentence = text[: earliest_idx + 1].strip()
            if len(sentence) <= max_length:
                return sentence

        # No sentence delimiter found or sentence too long — truncate
        if len(text) <= max_length:
            return text
        return text[:max_length].rsplit(" ", 1)[0] + "..."

    async def _report_error(self, tools: AgentToolsProtocol, error: str) -> None:
        """Send error event (best effort)."""
        try:
            await tools.send_event(content=f"Error: {error}", message_type="error")
        except Exception:
            logger.debug("Failed to report error to platform: %s", error)
