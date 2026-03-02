"""Unit tests for integration fixture usage policy enforcement."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from tests.support.integration.contracts.markers import (
    _enforce_live_fixture_environment,
    _enforce_live_fixture_policy,
    _is_integration_mode,
)

pytestmark = pytest.mark.contract_gate


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


def test_policy_ignores_tests_without_live_fixtures() -> None:
    item = _FakeItem(fixturenames=["tmp_path"], mark_expression="not integration")
    _enforce_live_fixture_policy(item)


def test_policy_requires_integration_marker_for_live_fixture() -> None:
    item = _FakeItem(
        fixturenames=["api_client"],
        markers={"requires_api"},
        mark_expression="integration",
    )

    with pytest.raises(pytest.fail.Exception, match="not marked"):
        _enforce_live_fixture_policy(item)


def test_policy_requires_integration_mode_for_live_fixture() -> None:
    item = _FakeItem(
        fixturenames=["api_client"],
        markers={"integration", "requires_api"},
        mark_expression="not integration",
    )

    with pytest.raises(pytest.fail.Exception, match="outside integration mode"):
        _enforce_live_fixture_policy(item)


def test_policy_requires_fixture_specific_marker() -> None:
    item = _FakeItem(
        fixturenames=["api_client_2"],
        markers={"integration"},
        mark_expression="integration",
    )

    with pytest.raises(pytest.fail.Exception, match="requires_multi_agent"):
        _enforce_live_fixture_policy(item)


def test_policy_allows_live_fixture_with_required_markers() -> None:
    item = _FakeItem(
        fixturenames=["api_client"],
        markers={"integration", "requires_api"},
        mark_expression="integration",
    )

    _enforce_live_fixture_policy(item)


def test_environment_guard_fails_unmarked_live_fixture_request() -> None:
    item = _FakeItem(
        fixturenames=["api_client"],
        markers={"requires_api"},
        mark_expression="integration",
    )
    request = _FakeRequest(node=item)

    with pytest.raises(pytest.fail.Exception, match="not marked"):
        _enforce_live_fixture_environment(request, "api_client")


def test_environment_guard_skips_marked_live_fixture_with_missing_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("THENVOI_API_KEY", raising=False)
    item = _FakeItem(
        fixturenames=["api_client"],
        markers={"integration", "requires_api"},
        mark_expression="integration",
    )
    request = _FakeRequest(node=item)

    with pytest.raises(pytest.skip.Exception, match="THENVOI_API_KEY"):
        _enforce_live_fixture_environment(request, "api_client")


def test_integration_mode_env_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _FakeConfig(mark_expression="not integration")
    monkeypatch.setenv("THENVOI_RUN_INTEGRATION", "1")

    assert _is_integration_mode(config) is True


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("integration", True),
        ("integration and requires_api", True),
        ("requires_api and integration", True),
        ("integration or smoke", True),
        ("not integration", False),
        ("not (integration and requires_api)", True),
        ("requires_api", False),
        ("notintegration", False),
    ],
)
def test_integration_mode_expression_parsing(
    expression: str,
    expected: bool,
) -> None:
    config = _FakeConfig(mark_expression=expression)
    assert _is_integration_mode(config) is expected
