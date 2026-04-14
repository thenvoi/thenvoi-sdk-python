"""Types and compatibility aliases for outbound ACP client integration."""

from __future__ import annotations

from dataclasses import dataclass, field

from thenvoi.integrations.acp.client_profiles import CursorACPClientProfile
from thenvoi.integrations.acp.client_runtime import ACPCollectingClient


@dataclass
class ACPClientSessionState:
    """Session state for ACP client adapter rehydration."""

    room_to_session: dict[str, str] = field(default_factory=dict)


class ThenvoiACPClient(ACPCollectingClient):
    """Transitional compatibility wrapper with Cursor extensions enabled.

    Existing tests and e2e helpers still construct ``ThenvoiACPClient``
    directly. Keep this alias stable while the runtime split lands.
    """

    def __init__(self) -> None:
        super().__init__(profile=CursorACPClientProfile())
