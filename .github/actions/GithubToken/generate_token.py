from __future__ import annotations

import os
import time
from typing import Any

import jwt
import requests


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"{name} environment variable is required")
    return value


def _generate_jwt(app_id: str, private_key: str) -> str:
    now = int(time.time())
    payload: dict[str, Any] = {
        "iat": now,
        "exp": now + (10 * 60),
        "iss": app_id,
    }
    try:
        return jwt.encode(payload, private_key, algorithm="RS256")
    except Exception as exc:
        raise ValueError(f"Error encoding JWT: {exc}") from exc


def _request_installation_token(jwt_token: str, installation_id: str) -> str:
    headers = {
        "Authorization": f"Bearer {jwt_token}",
        "Accept": "application/vnd.github+json",
    }
    url = f"https://api.github.com/app/installations/{installation_id}/access_tokens"
    try:
        response = requests.post(url, headers=headers, timeout=30)
        response.raise_for_status()
    except requests.exceptions.HTTPError as exc:
        raise SystemExit(f"HTTP error occurred: {exc}") from exc
    except requests.RequestException as exc:
        raise SystemExit(f"Request error occurred: {exc}") from exc

    token = response.json().get("token")
    if not token:
        raise ValueError("Access token not found in the response")
    return token


def _write_output_token(token: str) -> None:
    github_output = _require_env("GITHUB_OUTPUT")
    with open(github_output, "a", encoding="utf-8") as output_file:
        output_file.write(f"token={token}\n")


def main() -> None:
    app_id = _require_env("github_app_id")
    installation_id = _require_env("github_app_installation_id")
    private_key = _require_env("github_app_private_key")

    jwt_token = _generate_jwt(app_id, private_key)
    token = _request_installation_token(jwt_token, installation_id)
    _write_output_token(token)


if __name__ == "__main__":
    main()
