"""Contract tests for the public integration policy API."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from tests.support.integration import config as integration_config
from tests.support.integration import plugin as integration_plugin
from tests.support.integration import policy as integration_policy


@dataclass
class _FakeConfig:
    mark_expression: str = ""

    def getoption(self, name: str) -> str:
        if name == "-m":
            return self.mark_expression
        raise ValueError(name)


@dataclass
class _FakeItem:
    fixturenames: list[str]
    markers: set[str] = field(default_factory=set)
    mark_expression: str = ""
    nodeid: str = "tests/fake_test.py::test_case"

    def __post_init__(self) -> None:
        self.config = _FakeConfig(mark_expression=self.mark_expression)

    def get_closest_marker(self, name: str) -> object | None:
        return object() if name in self.markers else None


@dataclass
class _FakeRequest:
    node: _FakeItem


def test_policy_api_references_are_import_stable() -> None:
    assert integration_plugin.is_truthy_env is integration_policy.is_truthy_env
    assert integration_plugin.is_integration_mode is integration_policy.is_integration_mode
    assert (
        integration_plugin.enforce_live_fixture_environment
        is integration_policy.enforce_live_fixture_environment
    )
    assert (
        integration_plugin.enforce_live_fixture_policy
        is integration_policy.enforce_live_fixture_policy
    )
    assert integration_config._is_truthy_env is integration_policy.is_truthy_env
    assert integration_config._is_integration_mode is integration_policy.is_integration_mode
    assert (
        integration_config._enforce_live_fixture_environment
        is integration_policy.enforce_live_fixture_environment
    )
    assert (
        integration_config._enforce_live_fixture_policy
        is integration_policy.enforce_live_fixture_policy
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [("1", True), ("true", True), ("yes", True), ("on", True), ("0", False), ("", False)],
)
def test_truthy_env_contract(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
    expected: bool,
) -> None:
    monkeypatch.setenv("THENVOI_RUN_INTEGRATION", value)
    assert integration_policy.is_truthy_env("THENVOI_RUN_INTEGRATION") is expected


def test_integration_mode_contract_respects_mark_expression() -> None:
    config = _FakeConfig(mark_expression="integration and requires_api")
    assert integration_policy.is_integration_mode(config) is True


def test_enforce_live_fixture_policy_contract() -> None:
    item = _FakeItem(
        fixturenames=["api_client"],
        markers={"requires_api"},
        mark_expression="integration",
    )
    with pytest.raises(pytest.fail.Exception, match="not marked"):
        integration_policy.enforce_live_fixture_policy(item)


def test_enforce_live_fixture_environment_uses_active_markers_for_env_requirements(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("THENVOI_API_KEY", "token-1")
    monkeypatch.delenv("THENVOI_API_KEY_2", raising=False)
    item = _FakeItem(
        fixturenames=["api_client"],
        markers={"integration", "requires_api"},
        mark_expression="integration",
    )

    integration_policy.enforce_live_fixture_environment(
        _FakeRequest(node=item),
        "api_client",
    )


def test_enforce_live_fixture_environment_skips_missing_env_for_marked_tests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("THENVOI_API_KEY", "token-1")
    monkeypatch.delenv("THENVOI_API_KEY_2", raising=False)
    item = _FakeItem(
        fixturenames=["api_client"],
        markers={"integration", "requires_multi_agent"},
        mark_expression="integration",
    )

    with pytest.raises(pytest.skip.Exception, match="THENVOI_API_KEY_2"):
        integration_policy.enforce_live_fixture_environment(
            _FakeRequest(node=item),
            "api_client",
        )
