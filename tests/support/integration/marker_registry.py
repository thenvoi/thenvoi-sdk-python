"""Canonical marker policy for integration fixtures and decorators."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IntegrationMarkerSpec:
    """Marker metadata shared by decorators and fixture enforcement."""

    name: str
    env_vars: tuple[str, ...]
    reason: str


INTEGRATION_MARKER_SPECS: dict[str, IntegrationMarkerSpec] = {
    "requires_api": IntegrationMarkerSpec(
        name="requires_api",
        env_vars=("THENVOI_API_KEY",),
        reason="THENVOI_API_KEY is required",
    ),
    "requires_multi_agent": IntegrationMarkerSpec(
        name="requires_multi_agent",
        env_vars=("THENVOI_API_KEY", "THENVOI_API_KEY_2"),
        reason="Both THENVOI_API_KEY and THENVOI_API_KEY_2 required for multi-agent tests",
    ),
    "requires_user_api": IntegrationMarkerSpec(
        name="requires_user_api",
        env_vars=("THENVOI_API_KEY_USER",),
        reason="THENVOI_API_KEY_USER is required",
    ),
}


LIVE_INTEGRATION_FIXTURE_MARKERS: dict[str, tuple[str, ...]] = {
    "api_client": ("requires_api", "requires_multi_agent"),
    "api_client_2": ("requires_multi_agent",),
    "user_api_client": ("requires_user_api",),
    "integration_settings": (
        "requires_api",
        "requires_multi_agent",
        "requires_user_api",
    ),
    "test_chat": ("requires_api", "requires_multi_agent"),
    "test_peer_id": ("requires_api", "requires_multi_agent"),
}
