from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from urllib.request import urlopen


def _load_bridge_service_module():
    module_path = (
        Path(__file__).resolve().parents[2]
        / "examples"
        / "coding_agents"
        / "bridge_service.py"
    )
    spec = importlib.util.spec_from_file_location(
        "bridge_service_test_module", module_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec for {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_health_reports_starting_until_bridge_ready() -> None:
    bridge_service = _load_bridge_service_module()
    readiness = bridge_service._BridgeReadiness()
    server = bridge_service._start_health_server("127.0.0.1", 0, readiness)
    port = server.server_address[1]

    try:
        try:
            urlopen(f"http://127.0.0.1:{port}/health")
        except Exception as exc:
            response = exc
        else:
            raise AssertionError("Expected /health to be unavailable before readiness")

        assert response.code == 503
        payload = json.loads(response.read().decode("utf-8"))
        assert payload["status"] == "starting"

        readiness.mark_ready()
        with urlopen(f"http://127.0.0.1:{port}/health") as ready_response:
            assert ready_response.status == 200
            ready_payload = json.loads(ready_response.read().decode("utf-8"))
        assert ready_payload["status"] == "ok"
    finally:
        server.shutdown()
        server.server_close()
