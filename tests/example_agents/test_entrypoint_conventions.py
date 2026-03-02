"""Conventions for example entrypoint modules."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

ENTRYPOINTS_WITH_SHARED_LOGGING = [
    REPO_ROOT / "examples" / "run_agent.py",
    REPO_ROOT / "examples" / "coding_agents" / "create_agents.py",
    REPO_ROOT / "examples" / "coding_agents" / "test_communication.py",
    REPO_ROOT / "examples" / "anthropic" / "01_basic_agent.py",
    REPO_ROOT / "examples" / "codex" / "basic_agent.py",
]


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_entrypoints_use_shared_logging_helper() -> None:
    for path in ENTRYPOINTS_WITH_SHARED_LOGGING:
        contents = _read(path)
        assert "logging.basicConfig(" not in contents, (
            f"{path} should use thenvoi.testing.example_logging helpers"
        )
        assert "setup_logging_profile(" in contents, (
            f"{path} should call setup_logging_profile()"
        )


def test_codex_basic_agent_uses_public_adapter_namespace() -> None:
    contents = _read(REPO_ROOT / "examples" / "codex" / "basic_agent.py")
    assert "from thenvoi.adapters.codex import" not in contents
    assert "from thenvoi.adapters import CodexAdapter, CodexAdapterConfig" in contents


def test_run_agent_uses_shared_runtime_loader() -> None:
    contents = _read(REPO_ROOT / "examples" / "run_agent.py")
    assert "load_runtime_config(" in contents
    assert "load_agent_config(" not in contents
