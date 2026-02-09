"""Framework config registries for parameterized conformance tests.

Shared constants used by both config modules and conftest.py live here
to avoid duplication.

Public API:
  - ``CONVERTER_ID_FOR_ADAPTER``: adapter → converter id mapping
  - ``adapters.ADAPTER_CONFIGS``: adapter config registry
  - ``converters.CONVERTER_CONFIGS``: converter config registry
  - ``_fixtures``: shared tool-event payloads for conformance tests
"""

from __future__ import annotations

__all__ = ["CONVERTER_ID_FOR_ADAPTER"]

# Canonical mapping of adapter framework_id -> converter framework_id.
# Most adapters use a converter with the same id; overrides go here.
# Referenced by conftest.py (_get_framework_run_map) and documented in
# FRAMEWORK_VERIFICATION.md ("Converter framework_ids" note).
CONVERTER_ID_FOR_ADAPTER: dict[str, str] = {
    "langgraph": "langchain",
}
