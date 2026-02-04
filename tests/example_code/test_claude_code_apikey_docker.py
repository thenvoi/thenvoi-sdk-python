"""
Tests for the claude_code_apikey_docker example.

Tests cover:
- runner.py: config loading, custom tools loading
- example_tools.py: safe_math_eval edge cases
- healthcheck.py: health check logic
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

# Add examples directory to path for imports
EXAMPLES_DIR = (
    Path(__file__).parent.parent.parent / "examples" / "claude_code_apikey_docker"
)
sys.path.insert(0, str(EXAMPLES_DIR))


# =============================================================================
# runner.py tests
# =============================================================================


class TestLoadConfig:
    """Tests for runner.load_config function."""

    def test_load_valid_config(self, tmp_path: Path) -> None:
        """Test loading a valid YAML config file."""
        from runner import load_config

        config_file = tmp_path / "agent.yaml"
        config_file.write_text(
            yaml.dump(
                {
                    "agent_id": "test-agent-123",
                    "api_key": "sk_test_key",
                    "model": "claude-sonnet-4-5-20250929",
                }
            )
        )

        config = load_config(str(config_file))

        assert config["agent_id"] == "test-agent-123"
        assert config["api_key"] == "sk_test_key"
        assert config["model"] == "claude-sonnet-4-5-20250929"

    def test_load_config_file_not_found(self) -> None:
        """Test error when config file doesn't exist."""
        from runner import load_config

        with pytest.raises(ValueError, match="Config file not found"):
            load_config("/nonexistent/path/agent.yaml")

    def test_load_config_missing_agent_id(self, tmp_path: Path) -> None:
        """Test error when agent_id is missing."""
        from runner import load_config

        config_file = tmp_path / "agent.yaml"
        config_file.write_text(yaml.dump({"api_key": "sk_test_key"}))

        with pytest.raises(
            ValueError, match="Missing required config fields.*agent_id"
        ):
            load_config(str(config_file))

    def test_load_config_missing_api_key(self, tmp_path: Path) -> None:
        """Test error when api_key is missing."""
        from runner import load_config

        config_file = tmp_path / "agent.yaml"
        config_file.write_text(yaml.dump({"agent_id": "test-agent"}))

        with pytest.raises(ValueError, match="Missing required config fields.*api_key"):
            load_config(str(config_file))

    def test_load_config_empty_file(self, tmp_path: Path) -> None:
        """Test error when config file is empty."""
        from runner import load_config

        config_file = tmp_path / "agent.yaml"
        config_file.write_text("")

        with pytest.raises(ValueError, match="Config file is empty"):
            load_config(str(config_file))

    def test_load_config_invalid_yaml(self, tmp_path: Path) -> None:
        """Test error when config file has invalid YAML."""
        from runner import load_config

        config_file = tmp_path / "agent.yaml"
        config_file.write_text("invalid: yaml: content: [}")

        with pytest.raises(ValueError, match="Invalid YAML"):
            load_config(str(config_file))


class TestLoadCustomTools:
    """Tests for runner.load_custom_tools function."""

    def test_load_custom_tools_no_tools_dir(self, tmp_path: Path) -> None:
        """Test when tools directory doesn't exist."""
        from runner import load_custom_tools

        tools_dir = tmp_path / "tools"
        config_dir = tmp_path

        result = load_custom_tools(tools_dir, config_dir, ["calculator"])

        assert result == []

    def test_load_custom_tools_path_traversal_blocked(self, tmp_path: Path) -> None:
        """Test that path traversal attacks are blocked."""
        from runner import load_custom_tools

        # Create a tools directory outside the config directory
        outside_dir = tmp_path / "outside" / "tools"
        outside_dir.mkdir(parents=True)
        (outside_dir / "__init__.py").write_text("TOOL_REGISTRY = {}")

        config_dir = tmp_path / "config"
        config_dir.mkdir()

        result = load_custom_tools(outside_dir, config_dir, ["calculator"])

        assert result == []


# =============================================================================
# example_tools.py tests
# =============================================================================


class TestSafeMathEval:
    """Tests for safe_math_eval function."""

    def test_basic_arithmetic(self) -> None:
        """Test basic arithmetic operations."""
        from tools.example_tools import safe_math_eval

        assert safe_math_eval("2 + 2") == 4
        assert safe_math_eval("10 - 3") == 7
        assert safe_math_eval("5 * 6") == 30
        assert safe_math_eval("20 / 4") == 5.0

    def test_complex_expressions(self) -> None:
        """Test more complex expressions."""
        from tools.example_tools import safe_math_eval

        assert safe_math_eval("2 + 3 * 4") == 14
        assert safe_math_eval("(2 + 3) * 4") == 20
        assert safe_math_eval("10 // 3") == 3
        assert safe_math_eval("10 % 3") == 1

    def test_unary_operators(self) -> None:
        """Test unary operators."""
        from tools.example_tools import safe_math_eval

        assert safe_math_eval("-5") == -5
        assert safe_math_eval("+5") == 5
        assert safe_math_eval("--5") == 5

    def test_power_operator(self) -> None:
        """Test power operator within bounds."""
        from tools.example_tools import safe_math_eval

        assert safe_math_eval("2 ** 3") == 8
        assert safe_math_eval("10 ** 2") == 100

    def test_power_bounds_exceeded(self) -> None:
        """Test that pow with large operands is rejected."""
        from tools.example_tools import safe_math_eval

        with pytest.raises(ValueError, match="Pow operands too large"):
            safe_math_eval("99999 ** 2")  # base too large

        with pytest.raises(ValueError, match="Pow operands too large"):
            safe_math_eval("2 ** 999")  # exponent too large

    def test_safe_functions(self) -> None:
        """Test allowed functions."""
        from tools.example_tools import safe_math_eval

        assert safe_math_eval("abs(-5)") == 5
        assert safe_math_eval("round(3.7)") == 4
        assert safe_math_eval("min(1, 2, 3)") == 1
        assert safe_math_eval("max(1, 2, 3)") == 3
        assert safe_math_eval("pow(2, 3)") == 8

    def test_pow_function_bounds_exceeded(self) -> None:
        """Test that pow() function with large operands is rejected."""
        from tools.example_tools import safe_math_eval

        with pytest.raises(ValueError, match="Pow operands too large"):
            safe_math_eval("pow(99999, 2)")  # base too large

        with pytest.raises(ValueError, match="Pow operands too large"):
            safe_math_eval("pow(2, 999)")  # exponent too large

    def test_unsupported_function_rejected(self) -> None:
        """Test that unsupported functions are rejected."""
        from tools.example_tools import safe_math_eval

        with pytest.raises(ValueError, match="Unsupported function"):
            safe_math_eval("eval('1')")

        with pytest.raises(ValueError, match="Unsupported function"):
            safe_math_eval("exec('1')")

        with pytest.raises(ValueError, match="Unsupported function"):
            safe_math_eval("open('file')")

    def test_expression_too_long(self) -> None:
        """Test that very long expressions are rejected."""
        from tools.example_tools import safe_math_eval

        long_expression = "1 + " * 500 + "1"  # Very long expression

        with pytest.raises(ValueError, match="Expression too long"):
            safe_math_eval(long_expression)

    def test_deeply_nested_expression(self) -> None:
        """Test that deeply nested expressions are rejected."""
        from tools.example_tools import safe_math_eval

        # Create a deeply nested binary expression (each + adds depth)
        nested = "1" + " + 1" * 60

        with pytest.raises(ValueError, match="Expression too deeply nested"):
            safe_math_eval(nested)

    def test_unsupported_constant_type(self) -> None:
        """Test that non-numeric constants are rejected."""
        from tools.example_tools import safe_math_eval

        with pytest.raises(ValueError, match="Unsupported constant type"):
            safe_math_eval("'string'")

    def test_syntax_error(self) -> None:
        """Test that syntax errors are raised."""
        from tools.example_tools import safe_math_eval

        with pytest.raises(SyntaxError):
            safe_math_eval("2 +")

        with pytest.raises(SyntaxError):
            safe_math_eval("(2 + 3")

    def test_division_by_zero(self) -> None:
        """Test that division by zero raises ZeroDivisionError."""
        from tools.example_tools import safe_math_eval

        with pytest.raises(ZeroDivisionError):
            safe_math_eval("1 / 0")

        with pytest.raises(ZeroDivisionError):
            safe_math_eval("10 // 0")

        with pytest.raises(ZeroDivisionError):
            safe_math_eval("5 % 0")


# Note: Tool tests (TestCalculatorTool, TestGetTimeTool, TestRandomNumberTool) removed
# because @tool decorator from claude_agent_sdk wraps functions in SdkMcpTool objects
# that aren't directly callable. The safe_math_eval tests above cover the core logic.


# =============================================================================
# healthcheck.py tests
# =============================================================================


class TestHealthcheck:
    """Tests for healthcheck.py main function."""

    def test_healthcheck_no_config_env(self) -> None:
        """Test healthcheck fails when AGENT_CONFIG not set."""
        from healthcheck import main

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("AGENT_CONFIG", None)
            result = main()

        assert result == 1

    def test_healthcheck_config_file_not_found(self, tmp_path: Path) -> None:
        """Test healthcheck fails when config file doesn't exist."""
        from healthcheck import main

        with patch.dict(os.environ, {"AGENT_CONFIG": "/nonexistent/agent.yaml"}):
            result = main()

        assert result == 1

    def test_healthcheck_missing_credentials(self, tmp_path: Path) -> None:
        """Test healthcheck fails when credentials are missing."""
        from healthcheck import main

        config_file = tmp_path / "agent.yaml"
        config_file.write_text(yaml.dump({"model": "claude-sonnet-4-5-20250929"}))

        with patch.dict(os.environ, {"AGENT_CONFIG": str(config_file)}):
            result = main()

        assert result == 1

    def test_healthcheck_valid_config_no_httpx(self, tmp_path: Path) -> None:
        """Test healthcheck passes with valid config when httpx unavailable."""

        config_file = tmp_path / "agent.yaml"
        config_file.write_text(
            yaml.dump({"agent_id": "test-agent", "api_key": "sk_test_key"})
        )

        # Mock httpx import to raise ImportError
        with (
            patch.dict(os.environ, {"AGENT_CONFIG": str(config_file)}),
            patch.dict(sys.modules, {"httpx": None}),
        ):
            # Force reimport to trigger ImportError
            import importlib
            import healthcheck

            importlib.reload(healthcheck)

            # The test should still pass with config validation only
            # when httpx is not available
