"""AgentCore handler — invokes AWS Bedrock AgentCore agent runtimes."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from thenvoi.runtime.tools import AgentTools


class AgentCoreClient(Protocol):
    """Minimal interface for the boto3 bedrock-agentcore client."""

    def invoke_agent_runtime(self, **kwargs: Any) -> dict[str, Any]: ...


logger = logging.getLogger(__name__)

_MAX_RESPONSE_BYTES = 1_048_576  # 1 MB
_READ_CHUNK_SIZE = 65_536  # 64 KB


class AgentCoreHandler:
    """Handler that invokes an AWS Bedrock AgentCore agent runtime.

    Each instance is bound to a single agent runtime ARN.
    Create multiple instances for multiple agents.

    **Timeout layering**: This handler applies its own ``timeout`` (default
    120 s) via ``asyncio.wait_for`` around the boto3 invocation.  The bridge
    router may apply an *additional* outer timeout (``BridgeConfig.handler_timeout``,
    default 300 s) around the entire ``handle()`` call.  The inner handler
    timeout fires first; the outer router timeout acts as a safety net.
    """

    def __init__(
        self,
        agent_runtime_arn: str,
        region: str,
        timeout: float = 120.0,
        max_response_bytes: int = _MAX_RESPONSE_BYTES,
        boto3_client: AgentCoreClient | None = None,
    ) -> None:
        if not agent_runtime_arn or not agent_runtime_arn.strip():
            raise ValueError("agent_runtime_arn must be a non-empty string")
        if not region or not region.strip():
            raise ValueError("region must be a non-empty string")
        if timeout <= 0:
            raise ValueError("timeout must be a positive number")
        if max_response_bytes <= 0:
            raise ValueError("max_response_bytes must be a positive integer")

        self._agent_runtime_arn = agent_runtime_arn.strip()
        self._region = region.strip()
        self._timeout = timeout
        self._max_response_bytes = max_response_bytes
        self._boto3_client = boto3_client

    def _get_client(self) -> AgentCoreClient:
        """Return the boto3 client, creating it lazily if not injected.

        Called from the asyncio event loop thread (before ``to_thread``),
        so concurrent coroutines are serialised by cooperative scheduling
        and no lock is needed.
        """
        if self._boto3_client is None:
            try:
                import boto3
            except ImportError:
                raise ImportError(
                    "boto3 is required for AgentCoreHandler. "
                    "Install with: pip install thenvoi-sdk[bridge_agentcore]"
                )
            self._boto3_client = boto3.client(
                "bedrock-agentcore", region_name=self._region
            )
        return self._boto3_client

    def _build_payload(
        self,
        content: str,
        sender_id: str,
        sender_name: str | None,
        sender_type: str,
        thread_id: str,
        mentioned_agent: str,
    ) -> dict[str, str]:
        """Build the JSON payload for the AgentCore invocation.

        Keys with ``None`` values are omitted so the AgentCore API receives
        a clean payload without explicit nulls.
        """
        payload: dict[str, str] = {
            "prompt": content,
            "actor_id": sender_id,
            "actor_type": sender_type,
            "thread_id": thread_id,
            "mentioned_agent": mentioned_agent,
        }
        if sender_name is not None:
            payload["actor_name"] = sender_name
        return payload

    def _read_streaming_response(self, response: dict[str, Any]) -> str:
        """Read and parse the streaming response from AgentCore.

        Extracts text from known response keys or falls back to raw text.
        Reads in 64 KB chunks to avoid buffering large responses in a single
        allocation, and enforces the 1 MB size limit incrementally.
        """
        body = response.get("body")
        if body is None:
            raise RuntimeError("Response missing 'body' (StreamingBody)")

        try:
            chunks: list[bytes] = []
            total = 0
            while True:
                chunk = body.read(_READ_CHUNK_SIZE)
                if not chunk:
                    break
                if total + len(chunk) > self._max_response_bytes:
                    raise RuntimeError(
                        f"Response exceeds {self._max_response_bytes} byte limit"
                    )
                total += len(chunk)
                chunks.append(chunk)
            raw = b"".join(chunks)
            text = raw.decode("utf-8")
        finally:
            body.close()

        # Try to parse as JSON and extract from known keys
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                for key in ("output", "response", "text", "content", "message"):
                    if key in data:
                        value = data[key]
                        return str(value) if not isinstance(value, str) else value
        except (json.JSONDecodeError, TypeError):
            logger.debug("Response is not JSON, using raw text")

        return text

    async def _invoke_agent(
        self,
        payload: dict[str, str],
        session_id: str,
    ) -> str:
        """Invoke the AgentCore runtime in a thread (boto3 is sync).

        Both the API call and the response body read happen inside the
        thread so that ``StreamingBody.read()`` never blocks the event loop.
        """
        client = self._get_client()

        def _call() -> str:
            response = client.invoke_agent_runtime(
                agentRuntimeArn=self._agent_runtime_arn,
                runtimeSessionId=session_id,
                inputText=json.dumps(payload),
            )
            return self._read_streaming_response(response)

        return await asyncio.to_thread(_call)

    async def handle(
        self,
        content: str,
        room_id: str,
        thread_id: str,
        message_id: str,
        sender_id: str,
        sender_name: str | None,
        sender_type: str,
        mentioned_agent: str,
        tools: AgentTools,
    ) -> None:
        """Handle a routed @mention message by invoking AgentCore."""
        # Send a thought event (non-fatal if it fails)
        try:
            await tools.send_event(
                content=f"Invoking AgentCore for @{mentioned_agent}...",
                message_type="thought",
            )
        except Exception as exc:
            logger.warning("Failed to send thought event: %s", exc)

        # Resolve sender info from the pre-cached participant list injected
        # by the bridge.  This avoids a redundant REST API call — the bridge
        # already populated tools._participants from its participant cache.
        resolved_name, sender_handle = self._resolve_sender(sender_id, tools)
        # Use sender_name (pre-resolved by the bridge) when available,
        # otherwise fall back to the participant cache lookup.  Both are
        # None when the sender is unresolvable, which causes actor_name
        # to be omitted from the payload rather than sending a raw UUID.
        payload_name = sender_name if sender_name is not None else resolved_name

        payload = self._build_payload(
            content=content,
            sender_id=sender_id,
            sender_name=payload_name,
            sender_type=sender_type,
            thread_id=thread_id,
            mentioned_agent=mentioned_agent,
        )

        try:
            response_text = await asyncio.wait_for(
                self._invoke_agent(payload, session_id=thread_id),
                timeout=self._timeout,
            )
        except TimeoutError:
            raise TimeoutError(
                f"AgentCore invocation timed out after {self._timeout}s "
                f"for @{mentioned_agent}"
            ) from None

        if not response_text or not response_text.strip():
            raise RuntimeError(
                f"AgentCore returned empty response for @{mentioned_agent}"
            )

        # Prefer handle for mention resolution (most reliable for the
        # platform), fall back to sender_name or resolved display name.
        # If none are available (sender not in participants and
        # sender_name was None), send without mentions rather than
        # mentioning a raw UUID.
        mention_identifier = sender_handle or sender_name or resolved_name
        kwargs: dict[str, Any] = {"content": response_text}
        if mention_identifier:
            kwargs["mentions"] = [mention_identifier]
        await tools.send_message(**kwargs)

    def _resolve_sender(
        self,
        sender_id: str,
        tools: AgentTools,
    ) -> tuple[str | None, str | None]:
        """Resolve sender display name and handle from pre-cached participants.

        Uses the public ``tools.participants`` property (populated by the
        bridge), avoiding a redundant REST API call.

        Returns:
            Tuple of ``(display_name, handle)``.  Both are ``None`` when the
            sender is not found in the participant cache.  ``handle`` is also
            ``None`` when the participant was added via WebSocket event
            (which doesn't include handle).
        """
        participants = tools.participants
        for p in participants:
            if p.get("id") == sender_id:
                return p.get("name"), p.get("handle")
        logger.debug(
            "Sender %s not found in participant cache (%d entries)",
            sender_id,
            len(participants),
        )
        return None, None
