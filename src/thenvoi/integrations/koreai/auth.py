"""JWT authentication for Kore.ai webhook API."""

from __future__ import annotations

import logging
import random
import time

logger = logging.getLogger(__name__)


def generate_jwt(
    client_id: str,
    client_secret: str,
    user_identity: str,
    algorithm: str = "HS256",
    ttl_seconds: int = 300,
) -> str:
    """Generate a JWT for authenticating with Kore.ai webhook API.

    Each request requires a fresh JWT because the ``userIdentity`` claim
    must match the ``from.id`` in the request body, which varies per room.

    Args:
        client_id: Kore.ai Client ID (appId claim).
        client_secret: Kore.ai Client Secret used for signing.
        user_identity: Value for the userIdentity claim (must match from.id).
        algorithm: Signing algorithm (HS256 or HS512).
        ttl_seconds: Token time-to-live in seconds.

    Returns:
        Encoded JWT string.
    """
    import jwt

    now = int(time.time())
    payload = {
        "appId": client_id,
        "sub": str(random.randint(10000000, 99999999)),
        "userIdentity": user_identity,
        "iat": now,
        "exp": now + ttl_seconds,
    }

    token: str = jwt.encode(payload, client_secret, algorithm=algorithm)
    logger.debug("Generated JWT for userIdentity=%s", user_identity)
    return token
