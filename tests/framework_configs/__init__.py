"""Framework config registries for parameterized conformance tests.

Shared constants used by both config modules and conftest.py live here
to avoid duplication.

Public API:
  - ``CONVERTER_ID_FOR_ADAPTER``: adapter → converter id mapping
  - ``ADAPTER_CONFIGS``: adapter config registry (lazy)
  - ``CONVERTER_CONFIGS``: converter config registry (lazy)
  - ``converters.SenderBehavior``: enum for sender_name handling
  - ``fixtures``: shared tool-event payloads for conformance tests
  - ``output_adapters``: output adapter classes for uniform assertions
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tests.framework_configs.adapters import AdapterConfig  # noqa: F401
    from tests.framework_configs.converters import ConverterConfig  # noqa: F401

__all__ = [
    "CONVERTER_ID_FOR_ADAPTER",
    "ADAPTER_CONFIGS",
    "CONVERTER_CONFIGS",
    "fixtures",
    "output_adapters",
]

# Canonical mapping of adapter framework_id -> converter framework_id.
# Most adapters use a converter with the same id; overrides go here.
CONVERTER_ID_FOR_ADAPTER: dict[str, str] = {
    "langgraph": "langchain",
}


def __getattr__(name: str) -> Any:
    if name == "ADAPTER_CONFIGS":
        from tests.framework_configs.adapters import ADAPTER_CONFIGS

        return ADAPTER_CONFIGS
    if name == "CONVERTER_CONFIGS":
        from tests.framework_configs.converters import CONVERTER_CONFIGS

        return CONVERTER_CONFIGS
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
