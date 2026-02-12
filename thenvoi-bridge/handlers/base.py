"""Protocol for bridge message handlers.

Re-exports BaseHandler from bridge_core.handler for backwards compatibility.
"""

from __future__ import annotations

from bridge_core.handler import BaseHandler

__all__ = ["BaseHandler"]
