"""Framework config registries for parameterized conformance tests.

Shared constants used by both config modules and conftest.py live here
to avoid duplication.
"""

from __future__ import annotations

# Canonical mapping of adapter framework_id -> converter framework_id.
# Most adapters use a converter with the same id; overrides go here.
# Referenced by conftest.py (_get_framework_run_map) and documented in
# FRAMEWORK_VERIFICATION.md ("Converter framework_ids" note).
CONVERTER_ID_FOR_ADAPTER: dict[str, str] = {
    "langgraph": "langchain",
}
