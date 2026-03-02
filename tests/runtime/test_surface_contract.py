"""Contract tests for the public ``thenvoi.runtime`` package surface."""

from __future__ import annotations

import pytest

import thenvoi.runtime as runtime_module
from thenvoi.runtime.contacts.contact_handler import ContactEventHandler
from thenvoi.runtime.contacts.contact_tools import ContactTools
from thenvoi.runtime.contacts.service import ContactService
from thenvoi.runtime.tooling.custom_tools import get_custom_tool_name

pytestmark = pytest.mark.contract_gate

_EXPECTED_RUNTIME_EXPORTS = {
    "AgentConfig",
    "SessionConfig",
    "PlatformMessage",
    "ConversationContext",
    "MessageHandler",
    "RoomPresence",
    "Execution",
    "ExecutionContext",
    "ExecutionHandler",
    "AgentRuntime",
    "AgentTools",
    "GracefulShutdown",
    "run_with_graceful_shutdown",
}

_FORBIDDEN_RUNTIME_ROOT_EXPORTS = {
    "ContactEventHandler",
    "ContactService",
    "ContactTools",
    "HUB_ROOM_SYSTEM_PROMPT",
}


def test_runtime_public_export_set_is_stable() -> None:
    assert set(runtime_module.__all__) == _EXPECTED_RUNTIME_EXPORTS


def test_runtime_public_exports_resolve_to_declared_modules() -> None:
    assert runtime_module.AgentRuntime.__module__ == "thenvoi.runtime.runtime"
    assert runtime_module.ExecutionContext.__module__ == "thenvoi.runtime.execution"
    assert runtime_module.RoomPresence.__module__ == "thenvoi.runtime.presence"
    assert runtime_module.SessionConfig.__module__ == "thenvoi.runtime.types"


def test_runtime_root_keeps_contacts_domain_symbols_out_of_surface() -> None:
    for symbol in _FORBIDDEN_RUNTIME_ROOT_EXPORTS:
        assert not hasattr(runtime_module, symbol)


def test_runtime_compat_import_paths_warn_and_forward_symbols() -> None:
    with pytest.warns(DeprecationWarning, match=r"runtime\.contact_tools\.ContactTools"):
        from thenvoi.runtime.contact_tools import ContactTools as LegacyContactTools

    with pytest.warns(
        DeprecationWarning,
        match=r"runtime\.custom_tools\.get_custom_tool_name",
    ):
        from thenvoi.runtime.custom_tools import (
            get_custom_tool_name as LegacyGetCustomToolName,
        )

    with pytest.warns(
        DeprecationWarning,
        match=r"runtime\.compat\.contact_handler\.ContactEventHandler",
    ):
        from thenvoi.runtime.compat.contact_handler import (
            ContactEventHandler as LegacyContactEventHandler,
        )

    with pytest.warns(
        DeprecationWarning,
        match=r"runtime\.compat\.contact_service\.ContactService",
    ):
        from thenvoi.runtime.compat.contact_service import (
            ContactService as LegacyContactService,
        )

    assert LegacyContactTools is ContactTools
    assert LegacyGetCustomToolName is get_custom_tool_name
    assert LegacyContactEventHandler is ContactEventHandler
    assert LegacyContactService is ContactService
