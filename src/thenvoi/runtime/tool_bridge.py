"""Shared platform tool bridge used by framework adapters.

This module centralizes tool argument normalization and execution dispatch so
adapters only need to manage framework-specific binding and response shapes.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from collections.abc import Awaitable, Callable
from typing import Any, Mapping

from pydantic import ValidationError

from thenvoi.core.protocols import PlatformToolOperationsProtocol, ToolDispatchProtocol
from thenvoi.runtime.tool_definitions import BASE_TOOL_NAMES, MEMORY_TOOL_NAMES

# Explicit order keeps tool registration deterministic for adapters.
PLATFORM_BASE_TOOL_ORDER: tuple[str, ...] = (
    "thenvoi_send_message",
    "thenvoi_send_event",
    "thenvoi_add_participant",
    "thenvoi_remove_participant",
    "thenvoi_get_participants",
    "thenvoi_lookup_peers",
    "thenvoi_create_chatroom",
    "thenvoi_list_contacts",
    "thenvoi_add_contact",
    "thenvoi_remove_contact",
    "thenvoi_list_contact_requests",
    "thenvoi_respond_contact_request",
)

PLATFORM_MEMORY_TOOL_ORDER: tuple[str, ...] = (
    "thenvoi_list_memories",
    "thenvoi_store_memory",
    "thenvoi_get_memory",
    "thenvoi_supersede_memory",
    "thenvoi_archive_memory",
)

PLATFORM_TOOL_ORDER: tuple[str, ...] = (
    *PLATFORM_BASE_TOOL_ORDER,
    *PLATFORM_MEMORY_TOOL_ORDER,
)

if frozenset(PLATFORM_BASE_TOOL_ORDER) != BASE_TOOL_NAMES:
    raise ValueError("PLATFORM_BASE_TOOL_ORDER does not match BASE_TOOL_NAMES")
if frozenset(PLATFORM_MEMORY_TOOL_ORDER) != MEMORY_TOOL_NAMES:
    raise ValueError("PLATFORM_MEMORY_TOOL_ORDER does not match MEMORY_TOOL_NAMES")


@dataclass(frozen=True)
class ToolFailure:
    """Structured failure payload for all tool integrations."""

    tool_name: str
    arguments: dict[str, Any]
    cause: Exception
    message: str


class ToolExecutionError(RuntimeError):
    """Typed error raised for normalized platform tool failures."""

    def __init__(self, failure: ToolFailure) -> None:
        super().__init__(failure.message)
        self.failure = failure


@dataclass(frozen=True)
class ToolContract:
    """Authoritative tool contract for defaults, dispatch, and error formatting."""

    defaults: Mapping[str, Any]
    none_if_empty: tuple[str, ...] = ()
    normalizer: Callable[[dict[str, Any]], None] | None = None
    dispatcher: (
        Callable[[PlatformToolOperationsProtocol, Mapping[str, Any]], Awaitable[Any]]
        | None
    ) = None
    error_formatter: Callable[[Mapping[str, Any], Exception], str] | None = None


def get_platform_tool_order(*, include_memory_tools: bool) -> list[str]:
    """Return platform tool names in deterministic order."""
    names = list(PLATFORM_BASE_TOOL_ORDER)
    if include_memory_tools:
        names.extend(PLATFORM_MEMORY_TOOL_ORDER)
    return names


def _none_if_empty(value: Any) -> Any:
    return None if value == "" else value


def _normalize_send_message(args: dict[str, Any]) -> None:
    mentions = args.get("mentions")
    if mentions is None or mentions == "":
        args["mentions"] = []
    elif isinstance(mentions, list):
        args["mentions"] = mentions
    else:
        raise ValueError("mentions must be a list")


def _build_message_tool_contracts() -> dict[str, ToolContract]:
    return {
        "thenvoi_send_message": ToolContract(
            defaults={"content": ""},
            normalizer=_normalize_send_message,
            dispatcher=lambda tools, args: tools.send_message(
                args["content"],
                args.get("mentions"),
            ),
            error_formatter=lambda _args, error: f"Error sending message: {error}",
        ),
        "thenvoi_send_event": ToolContract(
            defaults={"content": "", "message_type": "thought", "metadata": None},
            dispatcher=lambda tools, args: tools.send_event(
                args["content"],
                args["message_type"],
                args.get("metadata"),
            ),
            error_formatter=lambda _args, error: f"Error sending event: {error}",
        ),
    }


def _build_participant_tool_contracts() -> dict[str, ToolContract]:
    return {
        "thenvoi_add_participant": ToolContract(
            defaults={"name": "", "role": "member"},
            dispatcher=lambda tools, args: tools.add_participant(
                args["name"],
                args.get("role", "member"),
            ),
            error_formatter=lambda args, error: (
                f"Error adding participant '{args.get('name', '')}': {error}"
            ),
        ),
        "thenvoi_remove_participant": ToolContract(
            defaults={"name": ""},
            dispatcher=lambda tools, args: tools.remove_participant(args["name"]),
            error_formatter=lambda args, error: (
                f"Error removing participant '{args.get('name', '')}': {error}"
            ),
        ),
        "thenvoi_get_participants": ToolContract(
            defaults={},
            dispatcher=lambda tools, _args: tools.get_participants(),
            error_formatter=lambda _args, error: f"Error getting participants: {error}",
        ),
        "thenvoi_lookup_peers": ToolContract(
            defaults={"page": 1, "page_size": 50},
            dispatcher=lambda tools, args: tools.lookup_peers(
                page=args["page"],
                page_size=args["page_size"],
            ),
            error_formatter=lambda _args, error: f"Error looking up peers: {error}",
        ),
        "thenvoi_create_chatroom": ToolContract(
            defaults={"task_id": None},
            none_if_empty=("task_id",),
            dispatcher=lambda tools, args: tools.create_chatroom(args.get("task_id")),
            error_formatter=lambda args, error: (
                f"Error creating chatroom (task_id={args.get('task_id')}): {error}"
            ),
        ),
    }


def _build_contact_tool_contracts() -> dict[str, ToolContract]:
    return {
        "thenvoi_list_contacts": ToolContract(
            defaults={"page": 1, "page_size": 50},
            dispatcher=lambda tools, args: tools.list_contacts(
                page=args["page"],
                page_size=args["page_size"],
            ),
            error_formatter=lambda _args, error: f"Error listing contacts: {error}",
        ),
        "thenvoi_add_contact": ToolContract(
            defaults={"handle": "", "message": None},
            none_if_empty=("message",),
            dispatcher=lambda tools, args: tools.add_contact(
                handle=args["handle"],
                message=args.get("message"),
            ),
            error_formatter=lambda args, error: (
                f"Error adding contact '{args.get('handle', '')}': {error}"
            ),
        ),
        "thenvoi_remove_contact": ToolContract(
            defaults={"handle": None, "contact_id": None},
            none_if_empty=("handle", "contact_id"),
            dispatcher=lambda tools, args: tools.remove_contact(
                handle=args.get("handle"),
                contact_id=args.get("contact_id"),
            ),
            error_formatter=lambda _args, error: f"Error removing contact: {error}",
        ),
        "thenvoi_list_contact_requests": ToolContract(
            defaults={"page": 1, "page_size": 50, "sent_status": "pending"},
            dispatcher=lambda tools, args: tools.list_contact_requests(
                page=args["page"],
                page_size=args["page_size"],
                sent_status=args["sent_status"],
            ),
            error_formatter=lambda _args, error: (
                f"Error listing contact requests: {error}"
            ),
        ),
        "thenvoi_respond_contact_request": ToolContract(
            defaults={"action": "", "handle": None, "request_id": None},
            none_if_empty=("handle", "request_id"),
            dispatcher=lambda tools, args: tools.respond_contact_request(
                action=args["action"],
                handle=args.get("handle"),
                request_id=args.get("request_id"),
            ),
            error_formatter=lambda _args, error: (
                f"Error responding to contact request: {error}"
            ),
        ),
    }


def _build_base_tool_contracts() -> dict[str, ToolContract]:
    return {
        **_build_message_tool_contracts(),
        **_build_participant_tool_contracts(),
        **_build_contact_tool_contracts(),
    }


def _build_memory_tool_contracts() -> dict[str, ToolContract]:
    return {
        "thenvoi_list_memories": ToolContract(
            defaults={
                "subject_id": None,
                "scope": None,
                "system": None,
                "type": None,
                "segment": None,
                "content_query": None,
                "page_size": 50,
                "status": None,
            },
            none_if_empty=(
                "subject_id",
                "scope",
                "system",
                "type",
                "segment",
                "content_query",
                "status",
            ),
            dispatcher=lambda tools, args: tools.list_memories(
                subject_id=args.get("subject_id"),
                scope=args.get("scope"),
                system=args.get("system"),
                type=args.get("type"),
                segment=args.get("segment"),
                content_query=args.get("content_query"),
                page_size=args["page_size"],
                status=args.get("status"),
            ),
            error_formatter=lambda _args, error: f"Error listing memories: {error}",
        ),
        "thenvoi_store_memory": ToolContract(
            defaults={
                "content": "",
                "system": "",
                "type": "",
                "segment": "",
                "thought": "",
                "scope": "subject",
                "subject_id": None,
                "metadata": None,
            },
            none_if_empty=("subject_id",),
            dispatcher=lambda tools, args: tools.store_memory(
                content=args["content"],
                system=args["system"],
                type=args["type"],
                segment=args["segment"],
                thought=args["thought"],
                scope=args["scope"],
                subject_id=args.get("subject_id"),
                metadata=args.get("metadata"),
            ),
            error_formatter=lambda _args, error: f"Error storing memory: {error}",
        ),
        "thenvoi_get_memory": ToolContract(
            defaults={"memory_id": ""},
            dispatcher=lambda tools, args: tools.get_memory(args["memory_id"]),
            error_formatter=lambda _args, error: f"Error getting memory: {error}",
        ),
        "thenvoi_supersede_memory": ToolContract(
            defaults={"memory_id": ""},
            dispatcher=lambda tools, args: tools.supersede_memory(args["memory_id"]),
            error_formatter=lambda _args, error: f"Error superseding memory: {error}",
        ),
        "thenvoi_archive_memory": ToolContract(
            defaults={"memory_id": ""},
            dispatcher=lambda tools, args: tools.archive_memory(args["memory_id"]),
            error_formatter=lambda _args, error: f"Error archiving memory: {error}",
        ),
    }


def _build_tool_contracts() -> dict[str, ToolContract]:
    return {
        **_build_base_tool_contracts(),
        **_build_memory_tool_contracts(),
    }


TOOL_CONTRACTS: dict[str, ToolContract] = _build_tool_contracts()


def _format_tool_error_message(
    tool_name: str,
    args: Mapping[str, Any],
    error: Exception,
) -> str:
    """Return consistent user-visible error text for tool wrappers."""
    raw_error = str(error)
    if raw_error.startswith("Unknown tool:"):
        return raw_error
    if isinstance(error, ValidationError):
        details = "; ".join(
            f"{'.'.join(str(loc) for loc in err['loc'])}: {err['msg']}"
            for err in error.errors()
        )
        return f"Invalid arguments for {tool_name}: {details}"

    contract = TOOL_CONTRACTS.get(tool_name)
    if contract is not None and contract.error_formatter is not None:
        return contract.error_formatter(args, error)
    return f"Error executing {tool_name}: {error}"


def build_tool_failure(
    tool_name: str,
    args: Mapping[str, Any] | None,
    error: Exception,
) -> ToolFailure:
    """Create a structured failure payload for adapter/tool boundaries."""
    if isinstance(error, ToolExecutionError):
        return error.failure

    normalized_args = dict(args or {})
    message = _format_tool_error_message(tool_name, normalized_args, error)
    return ToolFailure(
        tool_name=tool_name,
        arguments=normalized_args,
        cause=error,
        message=message,
    )


def as_tool_execution_error(
    tool_name: str,
    args: Mapping[str, Any] | None,
    error: Exception,
) -> ToolExecutionError:
    """Coerce unknown errors into the shared ToolExecutionError contract."""
    if isinstance(error, ToolExecutionError):
        return error
    return ToolExecutionError(build_tool_failure(tool_name, args, error))


def format_tool_error(
    tool_name: str,
    args: Mapping[str, Any] | None,
    error: Exception,
) -> str:
    """Render user-visible tool error text from the shared failure contract."""
    return build_tool_failure(tool_name, args, error).message


def normalize_tool_arguments(
    tool_name: str, arguments: dict[str, Any]
) -> dict[str, Any]:
    """Normalize framework-specific argument shapes into canonical tool args."""
    args: dict[str, Any] = dict(arguments)

    contract = TOOL_CONTRACTS.get(tool_name)
    if contract is None:
        return args

    for key, value in contract.defaults.items():
        args.setdefault(key, value)
    for key in contract.none_if_empty:
        args[key] = _none_if_empty(args.get(key))
    if contract.normalizer is not None:
        contract.normalizer(args)

    return args


async def dispatch_platform_tool_call(
    tools: PlatformToolOperationsProtocol,
    tool_name: str,
    arguments: Mapping[str, Any],
) -> Any:
    """Dispatch a normalized platform tool call using canonical runtime methods."""
    args = normalize_tool_arguments(tool_name, dict(arguments))
    contract = TOOL_CONTRACTS.get(tool_name)
    if contract is None or contract.dispatcher is None:
        raise ValueError(f"Unknown tool: {tool_name}")
    return await contract.dispatcher(tools, args)


async def invoke_platform_tool(
    tools: ToolDispatchProtocol,
    tool_name: str,
    arguments: dict[str, Any],
) -> Any:
    """Execute a Thenvoi platform tool using normalized arguments."""
    raw_arguments = dict(arguments)
    try:
        args = normalize_tool_arguments(tool_name, arguments)
    except Exception as error:
        raise as_tool_execution_error(tool_name, raw_arguments, error) from error

    typed_dispatch = getattr(tools, "execute_tool_call_or_raise", None)
    if callable(typed_dispatch) and inspect.iscoroutinefunction(typed_dispatch):
        try:
            return await typed_dispatch(tool_name, args)
        except Exception as error:
            raise as_tool_execution_error(tool_name, args, error) from error

    try:
        return await tools.execute_tool_call(tool_name, args)
    except Exception as error:
        raise as_tool_execution_error(tool_name, args, error) from error
