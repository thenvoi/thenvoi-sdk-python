"""HTTP client for Kore.ai webhook API."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

import aiohttp

from thenvoi.integrations.koreai.auth import generate_jwt

if TYPE_CHECKING:
    from thenvoi.integrations.koreai.types import KoreAIConfig

logger = logging.getLogger(__name__)

_RETRYABLE_STATUS_CODES = frozenset({502, 503, 504})
_MAX_RETRIES = 3
_BACKOFF_BASE_SECONDS = 1.0


class KoreAIClient:
    """HTTP client for sending messages to a Kore.ai webhook bot.

    Handles JWT authentication, request formatting, and retry logic
    for transient errors.
    """

    def __init__(self, config: KoreAIConfig) -> None:
        self._config = config
        self._session: aiohttp.ClientSession | None = None
        self._webhook_url = "%s/chatbot/v2/webhook/%s" % (
            config.api_host.rstrip("/"),
            config.bot_id,
        )

    async def start(self) -> None:
        """Create the underlying HTTP session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()

    async def close(self) -> None:
        """Close the underlying HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def send_message(
        self,
        room_id: str,
        text: str,
        *,
        new_session: bool = False,
    ) -> None:
        """Send a text message to the Kore.ai bot.

        Args:
            room_id: Thenvoi room ID (used as from.id and JWT userIdentity).
            text: Message text to send.
            new_session: If True, force a new Kore.ai conversation session.
        """
        body: dict[str, Any] = {
            "session": {"new": new_session},
            "message": {"type": "text", "val": text},
            "from": {"id": room_id},
        }
        await self._post(room_id, body)

    async def close_session(self, room_id: str) -> None:
        """Send a SESSION_CLOSURE event to Kore.ai.

        Args:
            room_id: Thenvoi room ID.
        """
        body: dict[str, Any] = {
            "message": {"type": "event", "val": "SESSION_CLOSURE"},
            "from": {"id": room_id},
        }
        try:
            await self._post(room_id, body, retries=0)
        except Exception:
            logger.warning("Failed to send session closure for room %s", room_id)

    async def _post(
        self,
        room_id: str,
        body: dict[str, Any],
        *,
        retries: int = _MAX_RETRIES,
        force_new_session_on_auth_error: bool = True,
    ) -> None:
        """POST to the Kore.ai webhook with retry logic.

        Retry strategy:
        - 502/503/504: retry up to ``retries`` times with exponential backoff.
        - 429: honor Retry-After header, then retry.
        - 401/403: retry once with session.new=true (session may have expired).
        - Network errors: treated like 502 (retry with backoff).
        """
        if self._session is None or self._session.closed:
            raise RuntimeError("KoreAIClient not started. Call start() first.")

        last_error: Exception | None = None

        for attempt in range(retries + 1):
            token = generate_jwt(
                client_id=self._config.client_id,
                client_secret=self._config.client_secret,
                user_identity=room_id,
                algorithm=self._config.jwt_algorithm,
            )
            headers = {
                "Authorization": "Bearer %s" % token,
                "Content-Type": "application/json",
            }

            try:
                async with self._session.post(
                    self._webhook_url,
                    json=body,
                    headers=headers,
                ) as resp:
                    if resp.status == 200:
                        return

                    if resp.status == 429:
                        retry_after = _parse_retry_after(resp)
                        logger.warning(
                            "Kore.ai rate limited (429), retrying after %ss",
                            retry_after,
                        )
                        await asyncio.sleep(retry_after)
                        continue

                    if resp.status in (401, 403) and force_new_session_on_auth_error:
                        logger.warning(
                            "Kore.ai auth error (%d), retrying with new session",
                            resp.status,
                        )
                        body_copy = dict(body)
                        body_copy["session"] = {"new": True}
                        await self._post(
                            room_id,
                            body_copy,
                            retries=0,
                            force_new_session_on_auth_error=False,
                        )
                        return

                    if resp.status in _RETRYABLE_STATUS_CODES and attempt < retries:
                        wait = _BACKOFF_BASE_SECONDS * (2**attempt)
                        logger.warning(
                            "Kore.ai returned %d, retrying in %ss (attempt %d/%d)",
                            resp.status,
                            wait,
                            attempt + 1,
                            retries,
                        )
                        await asyncio.sleep(wait)
                        continue

                    resp_text = await resp.text()
                    last_error = KoreAIAPIError(resp.status, resp_text)

            except aiohttp.ClientError as exc:
                if attempt < retries:
                    wait = _BACKOFF_BASE_SECONDS * (2**attempt)
                    logger.warning(
                        "Network error sending to Kore.ai: %s, retrying in %ss (attempt %d/%d)",
                        exc,
                        wait,
                        attempt + 1,
                        retries,
                    )
                    await asyncio.sleep(wait)
                    continue
                last_error = exc

        if last_error is not None:
            raise last_error


def _parse_retry_after(resp: aiohttp.ClientResponse) -> float:
    """Parse the Retry-After header, defaulting to 1 second."""
    raw = resp.headers.get("Retry-After", "1")
    try:
        return max(float(raw), 0.1)
    except (ValueError, TypeError):
        return 1.0


class KoreAIAPIError(Exception):
    """Error from the Kore.ai API."""

    def __init__(self, status: int, body: str) -> None:
        self.status = status
        self.body = body
        super().__init__("Kore.ai API returned %d: %s" % (status, body[:200]))
