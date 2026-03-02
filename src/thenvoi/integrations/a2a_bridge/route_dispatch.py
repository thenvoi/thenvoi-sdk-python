"""Mention routing dispatch helpers."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .handler import HandlerResult

if TYPE_CHECKING:
    from thenvoi.runtime.tools import AgentTools
    from thenvoi.runtime.types import PlatformMessage

    from .handler import BaseHandler


@dataclass(frozen=True)
class DispatchTarget:
    """A resolved mention dispatch target."""

    username: str
    handler_name: str
    handler: "BaseHandler"


@dataclass(frozen=True)
class DispatchFailure:
    """A dispatch failure for one target."""

    handler_name: str
    username: str
    error: Exception


def _normalize_handler_result(result: object) -> HandlerResult:
    """Validate explicit bridge handler outcome contract."""
    if isinstance(result, HandlerResult):
        return result
    if result is None:
        raise TypeError(
            "Bridge handlers must return HandlerResult, got None. "
            "Use HandlerResult.handled() for successful handling."
        )
    raise TypeError(
        "Bridge handlers must return HandlerResult, "
        f"got {type(result).__name__}."
    )


def build_dispatch_targets(
    mentions: list[object],
    *,
    agent_mapping: dict[str, str],
    handlers: dict[str, "BaseHandler"],
    logger: logging.Logger,
) -> list[DispatchTarget]:
    """Resolve mentions to unique handler dispatch targets."""
    targets: list[DispatchTarget] = []
    seen_usernames: set[str] = set()

    for mention in mentions:
        username = getattr(mention, "username", None)
        if username is None or username in seen_usernames:
            continue
        seen_usernames.add(username)

        handler_name = agent_mapping.get(username)
        if handler_name is None:
            logger.debug("No handler mapped for @%s", username)
            continue

        targets.append(
            DispatchTarget(
                username=username,
                handler_name=handler_name,
                handler=handlers[handler_name],
            )
        )

    return targets


async def execute_dispatch_targets(
    targets: list[DispatchTarget],
    *,
    platform_message: "PlatformMessage",
    tools: "AgentTools",
    room_id: str,
    handler_timeout: float | None,
    logger: logging.Logger,
) -> list[DispatchFailure]:
    """Execute mention handlers concurrently and collect failures."""

    async def _dispatch(target: DispatchTarget) -> DispatchFailure | None:
        logger.info(
            "Routing message to handler '%s' for @%s in room %s",
            target.handler_name,
            target.username,
            room_id,
        )
        try:
            coro = target.handler.handle(
                message=platform_message,
                mentioned_agent=target.username,
                tools=tools,
            )
            if handler_timeout is not None:
                raw_result = await asyncio.wait_for(coro, timeout=handler_timeout)
            else:
                raw_result = await coro
            result = _normalize_handler_result(raw_result)
            if result.status == "ignored":
                logger.debug(
                    "Handler '%s' ignored @%s in room %s (%s)",
                    target.handler_name,
                    target.username,
                    room_id,
                    result.detail or "no detail",
                )
                return None
            if result.status == "error":
                return DispatchFailure(
                    handler_name=target.handler_name,
                    username=target.username,
                    error=RuntimeError(result.detail or "handler returned error"),
                )
        except asyncio.TimeoutError:
            logger.error(
                "Handler '%s' timed out after %.1fs for @%s in room %s",
                target.handler_name,
                handler_timeout,
                target.username,
                room_id,
            )
            return DispatchFailure(
                handler_name=target.handler_name,
                username=target.username,
                error=TimeoutError(f"timed out after {handler_timeout}s"),
            )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            logger.exception(
                "Handler '%s' failed for @%s in room %s",
                target.handler_name,
                target.username,
                room_id,
            )
            return DispatchFailure(
                handler_name=target.handler_name,
                username=target.username,
                error=error,
            )
        return None

    results = await asyncio.gather(*[_dispatch(target) for target in targets])
    return [result for result in results if result is not None]


def summarize_dispatch_failures(
    failures: list[DispatchFailure],
    *,
    total_targets: int,
    max_error_len: int,
) -> tuple[bool, str, str]:
    """Build operator and user-facing failure summaries."""
    all_failed = len(failures) == total_targets
    internal_summaries = []
    user_summaries = []
    for failure in failures:
        error_text = str(failure.error)
        truncated = error_text[:max_error_len]
        suffix = "..." if len(error_text) > max_error_len else ""
        internal_summaries.append(
            f"'{failure.handler_name}' (@{failure.username}): {truncated}{suffix}"
        )
        user_summaries.append(f"@{failure.username}: processing failed")

    return all_failed, "; ".join(internal_summaries), "; ".join(user_summaries)


__all__ = [
    "DispatchFailure",
    "DispatchTarget",
    "build_dispatch_targets",
    "execute_dispatch_targets",
    "summarize_dispatch_failures",
]
