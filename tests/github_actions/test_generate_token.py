"""Unit tests for .github/actions/GithubToken/generate_token.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest
import requests


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "github_token_generate_token",
        Path(".github/actions/GithubToken/generate_token.py"),
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load generate_token module spec")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_require_env_returns_value(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module()
    monkeypatch.setenv("TEST_VALUE", "abc")

    assert module._require_env("TEST_VALUE") == "abc"


def test_require_env_raises_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module()
    monkeypatch.delenv("MISSING_VALUE", raising=False)

    with pytest.raises(ValueError, match="MISSING_VALUE environment variable is required"):
        module._require_env("MISSING_VALUE")


def test_generate_jwt_builds_expected_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module()
    mock_encode = MagicMock(return_value="jwt-token")
    monkeypatch.setattr(module.time, "time", lambda: 1_700_000_000)
    monkeypatch.setattr(module.jwt, "encode", mock_encode)

    token = module._generate_jwt("12345", "private-key")

    assert token == "jwt-token"
    mock_encode.assert_called_once_with(
        {"iat": 1_700_000_000, "exp": 1_700_000_600, "iss": "12345"},
        "private-key",
        algorithm="RS256",
    )


def test_generate_jwt_wraps_encode_error(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module()
    monkeypatch.setattr(module.jwt, "encode", MagicMock(side_effect=RuntimeError("bad key")))

    with pytest.raises(ValueError, match="Error encoding JWT: bad key"):
        module._generate_jwt("12345", "private-key")


def test_request_installation_token_returns_token(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module()
    mock_response = SimpleNamespace(
        raise_for_status=MagicMock(),
        json=MagicMock(return_value={"token": "ghs_token"}),
    )
    mock_post = MagicMock(return_value=mock_response)
    monkeypatch.setattr(module.requests, "post", mock_post)

    token = module._request_installation_token("jwt-value", "install-1")

    assert token == "ghs_token"
    mock_post.assert_called_once_with(
        "https://api.github.com/app/installations/install-1/access_tokens",
        headers={
            "Authorization": "Bearer jwt-value",
            "Accept": "application/vnd.github+json",
        },
        timeout=30,
    )
    mock_response.raise_for_status.assert_called_once()


def test_request_installation_token_raises_system_exit_on_http_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    mock_response = SimpleNamespace(
        raise_for_status=MagicMock(side_effect=requests.exceptions.HTTPError("403")),
        json=MagicMock(return_value={"token": "unused"}),
    )
    monkeypatch.setattr(module.requests, "post", MagicMock(return_value=mock_response))

    with pytest.raises(SystemExit, match="HTTP error occurred: 403"):
        module._request_installation_token("jwt-value", "install-1")


def test_request_installation_token_raises_system_exit_on_request_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    monkeypatch.setattr(
        module.requests,
        "post",
        MagicMock(side_effect=requests.exceptions.Timeout("timed out")),
    )

    with pytest.raises(SystemExit, match="Request error occurred: timed out"):
        module._request_installation_token("jwt-value", "install-1")


def test_request_installation_token_raises_when_response_has_no_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    mock_response = SimpleNamespace(
        raise_for_status=MagicMock(),
        json=MagicMock(return_value={"message": "ok"}),
    )
    monkeypatch.setattr(module.requests, "post", MagicMock(return_value=mock_response))

    with pytest.raises(ValueError, match="Access token not found in the response"):
        module._request_installation_token("jwt-value", "install-1")


def test_write_output_token_appends_to_github_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_module()
    output_file = tmp_path / "github_output.txt"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_file))

    module._write_output_token("ghs_123")

    assert output_file.read_text(encoding="utf-8") == "token=ghs_123\n"


def test_main_orchestrates_generation_and_output(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module()

    values = {
        "github_app_id": "123",
        "github_app_installation_id": "456",
        "github_app_private_key": "pem",
    }

    def _fake_require_env(name: str) -> str:
        return values[name]

    mock_generate_jwt = MagicMock(return_value="jwt-token")
    mock_request_token = MagicMock(return_value="install-token")
    mock_write_output = MagicMock()

    monkeypatch.setattr(module, "_require_env", _fake_require_env)
    monkeypatch.setattr(module, "_generate_jwt", mock_generate_jwt)
    monkeypatch.setattr(module, "_request_installation_token", mock_request_token)
    monkeypatch.setattr(module, "_write_output_token", mock_write_output)

    module.main()

    mock_generate_jwt.assert_called_once_with("123", "pem")
    mock_request_token.assert_called_once_with("jwt-token", "456")
    mock_write_output.assert_called_once_with("install-token")
