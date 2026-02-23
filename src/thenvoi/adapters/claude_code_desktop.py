"""
Claude Code Desktop adapter using SimpleAdapter pattern.

Enables Claude Code CLI (with MAX subscription) to act as an AI agent
participant in Thenvoi conversations WITHOUT requiring an Anthropic API key.

This adapter invokes Claude Code CLI as a subprocess and parses the JSON output.
It supports session persistence via --session-id and --resume for room conversations.

Requirements:
    - Claude Code CLI installed (npm install -g @anthropic-ai/claude-code)
    - Claude MAX subscription (for API-key-free usage)
    - CLI accessible in PATH or set via CLAUDE_CODE_PATH env var

Example:
    adapter = ClaudeCodeDesktopAdapter(
        custom_section="You are a helpful assistant.",
    )
    agent = Agent.create(adapter=adapter, agent_id="...", api_key="...")
    await agent.run()
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
from typing import Any

from thenvoi.core.protocols import AgentToolsProtocol
from thenvoi.core.simple_adapter import SimpleAdapter
from thenvoi.core.types import PlatformMessage
from thenvoi.converters.claude_sdk import ClaudeSDKHistoryConverter
from thenvoi.runtime.tools import get_tool_description

logger = logging.getLogger(__name__)


class ClaudeCodeDesktopAdapter(SimpleAdapter[str]):
    """
    Claude Code Desktop adapter using SimpleAdapter pattern.

    Uses Claude Code CLI as a subprocess for LLM interactions.
    The CLI is invoked with --print --output-format stream-json for non-interactive use.

    Session Management:
    - First message in a room: Uses --no-session-persistence (new session)
    - Subsequent messages: Uses --resume with saved session_id

    Note: This adapter stores session IDs per-room for conversation continuity.
    The history is converted to a text string for context injection.

    Example:
        adapter = ClaudeCodeDesktopAdapter(
            custom_section="You are a helpful assistant.",
        )
        agent = Agent.create(adapter=adapter, agent_id="...", api_key="...")
        await agent.run()
    """

    def __init__(
        self,
        custom_section: str | None = None,
        cli_path: str | None = None,
        cli_timeout: int = 120000,
        history_converter: ClaudeSDKHistoryConverter | None = None,
        allowed_tools: list[str] | None = None,
        verbose: bool = False,
    ):
        """
        Initialize adapter.

        Args:
            custom_section: Optional custom instructions to append to system prompt.
            cli_path: Path to Claude Code CLI. If not set, uses CLAUDE_CODE_PATH
                     env var or searches PATH.
            cli_timeout: CLI invocation timeout in milliseconds (default: 2 minutes).
            history_converter: Optional history converter. Defaults to ClaudeSDKHistoryConverter.
            allowed_tools: Optional list of Claude Code tools to enable (e.g.
                          ["Read", "Write", "Edit"]). When set, these tools are
                          auto-approved via --allowedTools. Default: None (no tools).
            verbose: Pass --verbose to the CLI for detailed output (default: False).
        """
        super().__init__(
            history_converter=history_converter or ClaudeSDKHistoryConverter()  # type: ignore[arg-type]  # desktop adapter consumes .text only
        )

        self.custom_section = custom_section
        self.cli_path = cli_path
        self.cli_timeout = cli_timeout
        self.allowed_tools = allowed_tools or []
        self.verbose = verbose

        # Per-room session IDs (for CLI session resume)
        self._session_ids: dict[str, str] = {}

        self._sensitive_pattern = re.compile(
            r"(/[\w./\\-]+)"  # file paths
            r"|(sk-[a-zA-Z0-9]{8,})"  # API keys (sk-...)
            r"|(key-[a-zA-Z0-9]{8,})"  # API keys (key-...)
            r"|(\b[A-Za-z_]*(token|secret|password|key)\s*[:=]\s*\S+)",  # assignments
            re.IGNORECASE,
        )

        logger.info("ClaudeCodeDesktopAdapter initialized")

    def _get_cli_path(self) -> str:
        """
        Get path to Claude Code CLI.

        Resolution order:
        1. Explicit cli_path from __init__
        2. CLAUDE_CODE_PATH environment variable
        3. 'claude' command in PATH

        Returns:
            Path to Claude Code CLI

        Raises:
            RuntimeError: If CLI cannot be found
        """
        # 1. Explicit path
        if self.cli_path:
            return self.cli_path

        # 2. Environment variable
        env_path = os.environ.get("CLAUDE_CODE_PATH")
        if env_path:
            return env_path

        # 3. Search PATH
        which_result = shutil.which("claude")
        if which_result:
            return which_result

        raise RuntimeError(
            "Claude Code CLI not found. Please either:\n"
            "  1. Set CLAUDE_CODE_PATH environment variable\n"
            "  2. Install Claude Code CLI: npm install -g @anthropic-ai/claude-code\n"
            "  3. Pass cli_path to ClaudeCodeDesktopAdapter"
        )

    def _sanitize_error(self, error: str, max_length: int = 200) -> str:
        """Sanitize error message before sending to the platform.

        Redacts file paths, API keys, and token/secret assignments to avoid
        leaking sensitive information into chat. The full error is still
        available in server-side logs via logger.error().

        Args:
            error: Raw error message string.
            max_length: Maximum length of the returned message.

        Returns:
            Sanitized, truncated error string.
        """
        sanitized = self._sensitive_pattern.sub("[redacted]", error)
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length] + "..."
        return sanitized

    async def on_started(self, agent_name: str, agent_description: str) -> None:
        """Store agent metadata after agent info is fetched."""
        await super().on_started(agent_name, agent_description)

        logger.info(
            "Claude Code Desktop adapter started for agent: %s (cli_timeout=%sms)",
            agent_name,
            self.cli_timeout,
        )

    def _build_cli_command(
        self,
        session_id: str | None,
    ) -> list[str]:
        """
        Build CLI command arguments.

        Args:
            session_id: Optional session ID for resume

        Returns:
            List of command arguments
        """
        cli_path = self._get_cli_path()
        cmd = [cli_path, "--print", "--output-format", "stream-json"]

        if self.verbose:
            cmd.append("--verbose")

        if self.allowed_tools:
            cmd.extend(["--allowedTools", ",".join(self.allowed_tools)])

        if session_id:
            cmd.extend(["--resume", session_id])
        else:
            cmd.append("--no-session-persistence")

        return cmd

    def _parse_cli_response(self, output: str) -> dict[str, Any]:
        """
        Parse stream-json (NDJSON) response from CLI.

        The stream-json format emits one JSON object per line. We look for
        the ``type=result`` line which contains session_id, cost, and the
        final result text. Tool-use blocks from ``type=assistant`` lines are
        logged for visibility.

        Falls back to single-JSON parsing for backward compatibility.

        Args:
            output: Raw NDJSON output from CLI

        Returns:
            Parsed response dict with at least 'result' key
        """
        result_data: dict[str, Any] = {}
        tool_calls: list[dict[str, Any]] = []

        for line in output.strip().split("\n"):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            msg_type = data.get("type")

            if msg_type == "result":
                # Final result line - has session_id, cost, result text
                result_data = data

            elif msg_type == "assistant":
                # Capture tool calls for logging
                message = data.get("message", {})
                for block in message.get("content", []):
                    if block.get("type") == "tool_use":
                        tool_calls.append(
                            {
                                "tool": block.get("name"),
                                "input": block.get("input"),
                            }
                        )

        if not result_data:
            # Fallback: try parsing entire output as single JSON (backward compat)
            try:
                result_data = json.loads(output)
            except json.JSONDecodeError:
                logger.error("Failed to parse CLI response as JSON or NDJSON")
                return {"result": output, "is_error": True}

        # Log tool calls for visibility
        if tool_calls:
            logger.info("Tools used: %s", [tc["tool"] for tc in tool_calls])

        return result_data

    def _generate_prompt(
        self,
        room_id: str,
        message: str,
        history: str,
        participants_msg: str | None,
    ) -> str:
        """
        Generate full prompt for Claude Code CLI.

        Args:
            room_id: Room identifier for context
            message: Current message content
            history: Converted history text
            participants_msg: Optional participants update message

        Returns:
            Full prompt string
        """
        custom_text = (
            f"\n\n## Custom Instructions\n\n{self.custom_section}"
            if self.custom_section
            else ""
        )

        # Tool descriptions for Thenvoi platform
        tool_descriptions = self._get_tool_descriptions_text()

        prompt_parts = [
            "## Thenvoi Platform Integration",
            "",
            f"You are **{self.agent_name}**, {self.agent_description}, "
            f"operating in a Thenvoi chat room.",
            "",
            "### Room Context",
            "",
            f"room_id: {room_id}",
            "",
            "### Available Actions",
            "",
            "To interact with the Thenvoi platform, respond with structured JSON "
            "commands. The adapter will execute them for you.",
            "",
            "**IMPORTANT: To send a message to the chat, respond with:**",
            "```json",
            '{"action": "send_message", "content": "Your message here"}',
            "```",
            "",
            "To perform multiple actions in one response, use a JSON array:",
            "```json",
            '[{"action": "send_event", "content": "Thinking...", "message_type": "thought"}, '
            '{"action": "send_message", "content": "Here is my answer."}]',
            "```",
            "",
            "Available actions:",
            tool_descriptions,
            "",
            "### Rules",
            "",
            "1. **Always use JSON action format** to send messages",
            "2. **Extract room_id** from context - it's provided above",
            "3. **Don't respond to yourself** - avoid message loops",
            custom_text,
        ]

        # Add history if available
        if history:
            prompt_parts.extend(
                [
                    "",
                    "### Previous Conversation",
                    "",
                    history,
                ]
            )

        # Add participants update if available
        if participants_msg:
            prompt_parts.extend(
                [
                    "",
                    "### Participants Update",
                    "",
                    participants_msg,
                ]
            )

        # Add current message
        prompt_parts.extend(
            [
                "",
                "### Current Message",
                "",
                f"[room_id: {room_id}]{message}",
            ]
        )

        return "\n".join(prompt_parts)

    def _get_tool_descriptions_text(self) -> str:
        """Get tool descriptions formatted for the prompt.

        Descriptions are pulled from the centralized get_tool_description() registry.
        Examples are adapter-specific (JSON action format unique to this CLI adapter).
        """
        descriptions = [
            f"- `send_message`: {get_tool_description('send_message')}",
            '  Example: {"action": "send_message", "content": "Hello!", "mentions": []}',
            f"- `send_event`: {get_tool_description('send_event')}",
            '  Example: {"action": "send_event", "content": "Thinking...", "message_type": "thought"}',
            f"- `add_participant`: {get_tool_description('add_participant')}",
            '  Example: {"action": "add_participant", "name": "Weather Agent", "role": "member"}',
            f"- `remove_participant`: {get_tool_description('remove_participant')}",
            '  Example: {"action": "remove_participant", "name": "Weather Agent"}',
            f"- `get_participants`: {get_tool_description('get_participants')}",
            '  Example: {"action": "get_participants"}',
            f"- `lookup_peers`: {get_tool_description('lookup_peers')}",
            '  Example: {"action": "lookup_peers", "page": 1, "page_size": 50}',
            f"- `create_chatroom`: {get_tool_description('create_chatroom')}",
            '  Example: {"action": "create_chatroom", "task_id": "optional-task-id"}',
        ]
        return "\n".join(descriptions)

    async def _execute_cli(self, prompt: str, session_id: str | None) -> dict[str, Any]:
        """
        Execute Claude Code CLI with the given prompt.

        Args:
            prompt: Prompt to send to Claude
            session_id: Optional session ID for resume

        Returns:
            Parsed response dict
        """
        cmd = self._build_cli_command(session_id)
        timeout_seconds = self.cli_timeout / 1000

        logger.debug("Executing CLI: %s...", " ".join(cmd[:3]))

        process = None
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(input=prompt.encode()),
                timeout=timeout_seconds,
            )

            if process.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown error"
                logger.error(
                    "CLI failed with code %s: %s", process.returncode, error_msg
                )
                return {
                    "result": f"CLI error: {error_msg}",
                    "is_error": True,
                }

            return self._parse_cli_response(stdout.decode())

        except asyncio.TimeoutError:
            logger.error("CLI timed out after %ss", timeout_seconds)
            # Kill the orphaned process
            if process:
                try:
                    process.kill()
                    await process.wait()
                    logger.debug("Killed timed-out CLI process")
                except Exception as kill_error:
                    logger.warning("Failed to kill timed-out process: %s", kill_error)
            return {
                "result": f"CLI timed out after {timeout_seconds} seconds",
                "is_error": True,
            }
        except Exception as e:
            logger.error("CLI execution failed: %s", e, exc_info=True)
            return {
                "result": f"CLI execution failed: {e}",
                "is_error": True,
            }

    async def _process_response(
        self,
        response: dict[str, Any],
        tools: AgentToolsProtocol,
        room_id: str,
    ) -> None:
        """
        Process CLI response and execute any actions.

        Args:
            response: Parsed CLI response
            tools: Agent tools for executing actions
            room_id: Room identifier
        """
        result = response.get("result", "")

        if response.get("is_error"):
            logger.error("Room %s: CLI returned error: %s", room_id, result)
            await tools.send_event(
                content=f"Error: {self._sanitize_error(result)}",
                message_type="error",
            )
            return

        # Try to extract JSON actions from response
        actions = self._extract_actions(result)

        if actions:
            for action_data in actions:
                await self._execute_action(action_data, tools)
                logger.debug(
                    "Room %s: Executed action '%s'", room_id, action_data.get("action")
                )
        else:
            # No structured action, send as message
            if result.strip():
                await tools.send_message(result.strip(), [])
                logger.debug("Room %s: Sent plain text response", room_id)

    def _extract_actions(self, result: str) -> list[dict[str, Any]]:
        """
        Extract all JSON actions from CLI response.

        Supports:
        - Single action object: {"action": "send_message", ...}
        - Array of actions: [{"action": ...}, {"action": ...}]
        - Multiple ```json``` code blocks each containing an action or array
        - Raw JSON without code fences

        Args:
            result: Raw result string

        Returns:
            List of parsed action dicts (empty if no actions found)
        """
        actions: list[dict[str, Any]] = []

        # Look for all ```json ... ``` blocks
        for match in re.finditer(r"```json\s*(.*?)\s*```", result, re.DOTALL):
            try:
                data = json.loads(match.group(1))
                if isinstance(data, dict) and "action" in data:
                    actions.append(data)
                elif isinstance(data, list):
                    actions.extend(
                        d for d in data if isinstance(d, dict) and "action" in d
                    )
            except json.JSONDecodeError as e:
                logger.warning(
                    "Failed to parse JSON from code block: %s. Content: %s...",
                    e,
                    match.group(1)[:100],
                )

        if actions:
            return actions

        # Fallback: try parsing the whole result as JSON
        try:
            data = json.loads(result)
            if isinstance(data, dict) and "action" in data:
                return [data]
            if isinstance(data, list):
                return [d for d in data if isinstance(d, dict) and "action" in d]
        except json.JSONDecodeError:
            # Not JSON or no action key - this is expected for plain text responses
            logger.debug("Result is not a JSON action, treating as plain text")

        return []

    async def _execute_action(
        self,
        action_data: dict[str, Any],
        tools: AgentToolsProtocol,
    ) -> None:
        """
        Execute a parsed action.

        Args:
            action_data: Action data with 'action' key
            tools: Agent tools for execution
        """
        action = action_data.get("action")

        try:
            if action == "send_message":
                content = action_data.get("content", "")
                mentions = action_data.get("mentions", [])
                await tools.send_message(content, mentions)
                logger.info(
                    "Sent message: %s%s",
                    content[:50],
                    "..." if len(content) > 50 else "",
                )

            elif action == "send_event":
                content = action_data.get("content", "")
                message_type = action_data.get("message_type", "thought")
                await tools.send_event(content, message_type)
                logger.debug(
                    "Sent event (%s): %s",
                    message_type,
                    content[:50] + ("..." if len(content) > 50 else ""),
                )

            elif action == "add_participant":
                name = action_data.get("name", "")
                role = action_data.get("role", "member")
                await tools.add_participant(name, role)
                logger.info("Added participant: %s as %s", name, role)

            elif action == "remove_participant":
                name = action_data.get("name", "")
                await tools.remove_participant(name)
                logger.info("Removed participant: %s", name)

            elif action == "get_participants":
                participants = await tools.get_participants()
                logger.debug("Got participants: %s", len(participants))

            elif action == "lookup_peers":
                page = action_data.get("page", 1)
                page_size = action_data.get("page_size", 50)
                await tools.lookup_peers(page, page_size)
                logger.debug("Looked up peers")

            elif action == "create_chatroom":
                task_id = action_data.get("task_id") or None
                new_room_id = await tools.create_chatroom(task_id)
                logger.info("Created chatroom: %s", new_room_id)

            else:
                logger.warning("Unknown action: %s", action)
                await tools.send_event(
                    content=f"Unknown action type: {action}. "
                    "Available actions: send_message, send_event, add_participant, "
                    "remove_participant, get_participants, lookup_peers, "
                    "create_chatroom",
                    message_type="error",
                )

        except Exception as e:
            logger.error("Action execution failed: %s", e, exc_info=True)
            await tools.send_event(
                content=f"Action failed: {self._sanitize_error(str(e))}",
                message_type="error",
            )

    async def on_message(
        self,
        msg: PlatformMessage,
        tools: AgentToolsProtocol,
        history: str,
        participants_msg: str | None,
        contacts_msg: str | None,
        *,
        is_session_bootstrap: bool,
        room_id: str,
    ) -> None:
        """
        Handle incoming message.

        Args:
            msg: Platform message
            tools: Agent tools (send_message, send_event, etc.)
            history: Converted history as text
            participants_msg: Participants update message, or None
            contacts_msg: Contact changes broadcast message, or None
            is_session_bootstrap: True if first message from this room
            room_id: The room identifier
        """
        logger.debug("Handling message %s in room %s", msg.id, room_id)

        # Get stored session_id for potential resume (only on subsequent messages)
        session_id = None
        if not is_session_bootstrap:
            session_id = self._session_ids.get(room_id)

        # Generate prompt
        prompt = self._generate_prompt(
            room_id=room_id,
            message=msg.format_for_llm(),
            history=history if is_session_bootstrap else "",
            participants_msg=participants_msg,
        )

        logger.info(
            "Room %s: Sending query to Claude Code CLI (first_msg=%s, session_id=%s)",
            room_id,
            is_session_bootstrap,
            session_id,
        )

        try:
            # Execute CLI
            response = await self._execute_cli(prompt, session_id)

            # Capture session_id for future resume
            new_session_id = response.get("session_id")
            if new_session_id:
                self._session_ids[room_id] = new_session_id
                logger.debug("Room %s: Captured session_id %s", room_id, new_session_id)

            # Log cost if available
            cost = response.get("total_cost_usd")
            if cost:
                logger.info("Room %s: Cost $%.4f", room_id, cost)

            # Process response (execute actions)
            await self._process_response(response, tools, room_id)

        except Exception as e:
            logger.error("Error processing message: %s", e, exc_info=True)
            await tools.send_event(
                content=f"Error: {self._sanitize_error(str(e))}",
                message_type="error",
            )
            raise

        logger.debug("Message %s processed successfully", msg.id)

    async def on_cleanup(self, room_id: str) -> None:
        """Clean up session ID when agent leaves a room."""
        if room_id in self._session_ids:
            del self._session_ids[room_id]
        logger.debug("Room %s: Cleaned up Claude Code Desktop session", room_id)

    async def cleanup_all(self) -> None:
        """Cleanup all sessions (call on stop)."""
        self._session_ids.clear()
        logger.info("Cleaned up all Claude Code Desktop sessions")
