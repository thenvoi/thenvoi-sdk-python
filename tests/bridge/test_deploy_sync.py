"""Sync check: ecs-task-definition.json must match cloudformation.yml.

Both files define the same ECS task definition — the JSON file for standalone
``aws ecs register-task-definition`` usage, and the CloudFormation template
for infrastructure-as-code deployments.  This test catches drift between
them so CI fails before a misconfigured deployment reaches production.
"""

from __future__ import annotations

import json
import os
from typing import Any

import pytest
import yaml

_DEPLOY_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "thenvoi-bridge", "deploy"
)
_CF_PATH = os.path.join(_DEPLOY_DIR, "cloudformation.yml")
_JSON_PATH = os.path.join(_DEPLOY_DIR, "ecs-task-definition.json")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_cf_task_definition() -> dict[str, Any]:
    """Load the TaskDefinition Properties from the CloudFormation template.

    Uses a custom YAML loader that resolves CloudFormation intrinsic
    functions (``!Ref``, ``!Sub``, ``!GetAtt``) to placeholder strings
    so structural comparison with the JSON file is possible.
    """

    class _CFLoader(yaml.SafeLoader):
        pass

    # Register handlers for CF intrinsic functions that appear in the template.
    # Each returns a recognisable placeholder so we can compare structure
    # without needing the actual parameter values.
    _CFLoader.add_constructor(
        "!Ref", lambda loader, node: f"<CF_REF:{loader.construct_scalar(node)}>"
    )
    _CFLoader.add_constructor(
        "!Sub", lambda loader, node: f"<CF_SUB:{loader.construct_scalar(node)}>"
    )
    _CFLoader.add_constructor(
        "!GetAtt", lambda loader, node: f"<CF_GETATT:{loader.construct_scalar(node)}>"
    )
    _CFLoader.add_constructor(
        "!Not", lambda loader, node: f"<CF_NOT:{loader.construct_sequence(node)}>"
    )
    _CFLoader.add_constructor(
        "!Equals",
        lambda loader, node: f"<CF_EQUALS:{loader.construct_sequence(node)}>",
    )
    _CFLoader.add_constructor(
        "!If", lambda loader, node: f"<CF_IF:{loader.construct_sequence(node)}>"
    )

    with open(_CF_PATH) as f:
        template = yaml.load(f, Loader=_CFLoader)  # noqa: S506

    return template["Resources"]["TaskDefinition"]["Properties"]


def _load_json_task_definition() -> dict[str, Any]:
    with open(_JSON_PATH) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_files_exist = os.path.isfile(_CF_PATH) and os.path.isfile(_JSON_PATH)
_skip_reason = "deploy files not found (expected in thenvoi-bridge/deploy/)"

pytestmark = pytest.mark.skipif(not _files_exist, reason=_skip_reason)


@pytest.fixture(scope="module")
def cf_props() -> dict[str, Any]:
    return _load_cf_task_definition()


@pytest.fixture(scope="module")
def json_def() -> dict[str, Any]:
    return _load_json_task_definition()


@pytest.fixture(scope="module")
def cf_container(cf_props: dict[str, Any]) -> dict[str, Any]:
    return cf_props["ContainerDefinitions"][0]


@pytest.fixture(scope="module")
def json_container(json_def: dict[str, Any]) -> dict[str, Any]:
    return json_def["containerDefinitions"][0]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTaskDefinitionSync:
    """Verify top-level task definition fields match."""

    def test_family(self, cf_props: dict[str, Any], json_def: dict[str, Any]) -> None:
        assert cf_props["Family"] == json_def["family"]

    def test_requires_compatibilities(
        self, cf_props: dict[str, Any], json_def: dict[str, Any]
    ) -> None:
        assert (
            cf_props["RequiresCompatibilities"] == json_def["requiresCompatibilities"]
        )

    def test_network_mode(
        self, cf_props: dict[str, Any], json_def: dict[str, Any]
    ) -> None:
        assert cf_props["NetworkMode"] == json_def["networkMode"]

    def test_runtime_platform(
        self, cf_props: dict[str, Any], json_def: dict[str, Any]
    ) -> None:
        cf_platform = cf_props["RuntimePlatform"]
        json_platform = json_def["runtimePlatform"]
        assert cf_platform["CpuArchitecture"] == json_platform["cpuArchitecture"]
        assert (
            cf_platform["OperatingSystemFamily"]
            == json_platform["operatingSystemFamily"]
        )

    def test_cpu(self, cf_props: dict[str, Any], json_def: dict[str, Any]) -> None:
        assert cf_props["Cpu"] == json_def["cpu"]

    def test_memory(self, cf_props: dict[str, Any], json_def: dict[str, Any]) -> None:
        assert cf_props["Memory"] == json_def["memory"]


class TestContainerDefinitionSync:
    """Verify the container definition fields match."""

    def test_container_name(
        self, cf_container: dict[str, Any], json_container: dict[str, Any]
    ) -> None:
        assert cf_container["Name"] == json_container["name"]

    def test_essential(
        self, cf_container: dict[str, Any], json_container: dict[str, Any]
    ) -> None:
        assert cf_container["Essential"] == json_container["essential"]

    def test_port_mappings(
        self, cf_container: dict[str, Any], json_container: dict[str, Any]
    ) -> None:
        cf_ports = cf_container["PortMappings"]
        json_ports = json_container["portMappings"]
        assert len(cf_ports) == len(json_ports)
        assert cf_ports[0]["ContainerPort"] == json_ports[0]["containerPort"]
        assert cf_ports[0]["Protocol"] == json_ports[0]["protocol"]

    def test_healthcheck_command(
        self, cf_container: dict[str, Any], json_container: dict[str, Any]
    ) -> None:
        assert (
            cf_container["HealthCheck"]["Command"]
            == json_container["healthCheck"]["command"]
        )

    def test_healthcheck_interval(
        self, cf_container: dict[str, Any], json_container: dict[str, Any]
    ) -> None:
        assert (
            cf_container["HealthCheck"]["Interval"]
            == json_container["healthCheck"]["interval"]
        )

    def test_healthcheck_timeout(
        self, cf_container: dict[str, Any], json_container: dict[str, Any]
    ) -> None:
        assert (
            cf_container["HealthCheck"]["Timeout"]
            == json_container["healthCheck"]["timeout"]
        )

    def test_healthcheck_retries(
        self, cf_container: dict[str, Any], json_container: dict[str, Any]
    ) -> None:
        assert (
            cf_container["HealthCheck"]["Retries"]
            == json_container["healthCheck"]["retries"]
        )

    def test_healthcheck_start_period(
        self, cf_container: dict[str, Any], json_container: dict[str, Any]
    ) -> None:
        assert (
            cf_container["HealthCheck"]["StartPeriod"]
            == json_container["healthCheck"]["startPeriod"]
        )

    def test_secret_names_match(
        self, cf_container: dict[str, Any], json_container: dict[str, Any]
    ) -> None:
        """Both files must reference the same set of secret environment variables."""
        cf_secret_names = sorted(s["Name"] for s in cf_container["Secrets"])
        json_secret_names = sorted(s["name"] for s in json_container["secrets"])
        assert cf_secret_names == json_secret_names

    def test_environment_variable_names_match(
        self, cf_container: dict[str, Any], json_container: dict[str, Any]
    ) -> None:
        """Both files must define the same set of environment variables."""
        cf_env_names = sorted(e["Name"] for e in cf_container["Environment"])
        json_env_names = sorted(e["name"] for e in json_container["environment"])
        assert cf_env_names == json_env_names

    def test_static_environment_values_match(
        self, cf_container: dict[str, Any], json_container: dict[str, Any]
    ) -> None:
        """Environment variables with static values (not CF refs) must match."""
        # Build lookup dicts
        cf_env = {e["Name"]: e["Value"] for e in cf_container["Environment"]}
        json_env = {e["name"]: e["value"] for e in json_container["environment"]}

        # Only compare values that are static strings in CF (not !Ref / !Sub)
        for name in cf_env:
            cf_val = cf_env[name]
            if isinstance(cf_val, str) and not cf_val.startswith("<CF_"):
                assert cf_val == json_env.get(name), (
                    f"Environment variable '{name}' differs: "
                    f"CF={cf_val!r}, JSON={json_env.get(name)!r}"
                )

    def test_log_driver(
        self, cf_container: dict[str, Any], json_container: dict[str, Any]
    ) -> None:
        assert (
            cf_container["LogConfiguration"]["LogDriver"]
            == json_container["logConfiguration"]["logDriver"]
        )

    def test_log_stream_prefix(
        self, cf_container: dict[str, Any], json_container: dict[str, Any]
    ) -> None:
        cf_opts = cf_container["LogConfiguration"]["Options"]
        json_opts = json_container["logConfiguration"]["options"]
        assert cf_opts["awslogs-stream-prefix"] == json_opts["awslogs-stream-prefix"]
