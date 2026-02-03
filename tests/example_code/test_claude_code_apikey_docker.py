"""
Tests for claude_code_apikey_docker example.

Tests cover:
- safe_math_eval: AST-based math expression evaluator
- load_config: YAML configuration loader
- load_custom_tools: Custom tools loader
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add examples to path for imports
EXAMPLES_DIR = (
    Path(__file__).parent.parent.parent / "examples" / "claude_code_apikey_docker"
)
sys.path.insert(0, str(EXAMPLES_DIR))

from tools.example_tools import safe_math_eval  # noqa: E402


class TestSafeMathEval:
    """Tests for the safe math expression evaluator."""

    # --- Basic operations ---

    def test_addition(self):
        """Should evaluate addition."""
        assert safe_math_eval("2 + 3") == 5

    def test_subtraction(self):
        """Should evaluate subtraction."""
        assert safe_math_eval("10 - 4") == 6

    def test_multiplication(self):
        """Should evaluate multiplication."""
        assert safe_math_eval("6 * 7") == 42

    def test_division(self):
        """Should evaluate division."""
        assert safe_math_eval("15 / 3") == 5.0

    def test_floor_division(self):
        """Should evaluate floor division."""
        assert safe_math_eval("17 // 5") == 3

    def test_modulo(self):
        """Should evaluate modulo."""
        assert safe_math_eval("17 % 5") == 2

    def test_power(self):
        """Should evaluate exponentiation."""
        assert safe_math_eval("2 ** 10") == 1024

    # --- Unary operators ---

    def test_unary_negative(self):
        """Should evaluate unary negative."""
        assert safe_math_eval("-5") == -5

    def test_unary_positive(self):
        """Should evaluate unary positive."""
        assert safe_math_eval("+5") == 5

    def test_negative_in_expression(self):
        """Should handle negative numbers in expressions."""
        assert safe_math_eval("10 + -3") == 7

    # --- Complex expressions ---

    def test_operator_precedence(self):
        """Should respect operator precedence."""
        assert safe_math_eval("2 + 3 * 4") == 14

    def test_parentheses(self):
        """Should respect parentheses."""
        assert safe_math_eval("(2 + 3) * 4") == 20

    def test_nested_parentheses(self):
        """Should handle nested parentheses."""
        assert safe_math_eval("((2 + 3) * (4 - 1))") == 15

    def test_float_numbers(self):
        """Should handle float numbers."""
        assert safe_math_eval("3.14 * 2") == 6.28

    # --- Allowed functions ---

    def test_abs_function(self):
        """Should evaluate abs() function."""
        assert safe_math_eval("abs(-10)") == 10

    def test_round_function(self):
        """Should evaluate round() function."""
        assert safe_math_eval("round(3.7)") == 4

    def test_min_function(self):
        """Should evaluate min() function."""
        assert safe_math_eval("min(1, 2, 3)") == 1

    def test_max_function(self):
        """Should evaluate max() function."""
        assert safe_math_eval("max(1, 2, 3)") == 3

    def test_pow_function(self):
        """Should evaluate pow() function."""
        assert safe_math_eval("pow(2, 8)") == 256

    def test_function_in_expression(self):
        """Should allow functions in larger expressions."""
        assert safe_math_eval("abs(-5) + max(1, 2, 3)") == 8

    # --- Security: blocked expressions ---

    def test_blocks_import(self):
        """Should block __import__ attempts."""
        with pytest.raises(ValueError):
            safe_math_eval('__import__("os")')

    def test_blocks_open(self):
        """Should block open() function."""
        with pytest.raises(ValueError):
            safe_math_eval('open("/etc/passwd")')

    def test_blocks_eval(self):
        """Should block eval() function."""
        with pytest.raises(ValueError):
            safe_math_eval('eval("1+1")')

    def test_blocks_exec(self):
        """Should block exec() function."""
        with pytest.raises(ValueError):
            safe_math_eval('exec("print(1)")')

    def test_blocks_getattr(self):
        """Should block getattr() function."""
        with pytest.raises(ValueError):
            safe_math_eval('getattr(str, "join")')

    def test_blocks_string_literals(self):
        """Should block string literals."""
        with pytest.raises(ValueError):
            safe_math_eval('"hello"')

    def test_blocks_list_literals(self):
        """Should block list literals."""
        with pytest.raises(ValueError):
            safe_math_eval("[1, 2, 3]")

    def test_blocks_dict_literals(self):
        """Should block dict literals."""
        with pytest.raises(ValueError):
            safe_math_eval('{"a": 1}')

    def test_blocks_attribute_access(self):
        """Should block attribute access."""
        with pytest.raises(ValueError):
            safe_math_eval("(1).__class__")

    def test_blocks_lambda(self):
        """Should block lambda expressions."""
        with pytest.raises(ValueError):
            safe_math_eval("lambda x: x + 1")

    # --- Error handling ---

    def test_syntax_error(self):
        """Should raise SyntaxError for invalid syntax."""
        with pytest.raises(SyntaxError):
            safe_math_eval("2 +")

    def test_division_by_zero(self):
        """Should raise ZeroDivisionError."""
        with pytest.raises(ZeroDivisionError):
            safe_math_eval("1 / 0")


class TestLoadConfig:
    """Tests for the YAML configuration loader."""

    def test_load_valid_config(self, tmp_path):
        """Should load valid config successfully."""
        # Import here to avoid import errors if runner not available
        sys.path.insert(0, str(EXAMPLES_DIR))
        from runner import load_config

        config_file = tmp_path / "agent.yaml"
        config_file.write_text("""
agent_id: "test-agent-123"
api_key: "test-api-key"
model: "claude-sonnet-4-5-20250929"
prompt: "You are helpful."
""")

        config = load_config(str(config_file))

        assert config["agent_id"] == "test-agent-123"
        assert config["api_key"] == "test-api-key"
        assert config["model"] == "claude-sonnet-4-5-20250929"
        assert config["prompt"] == "You are helpful."

    def test_missing_file_exits(self, tmp_path):
        """Should exit when config file doesn't exist."""
        sys.path.insert(0, str(EXAMPLES_DIR))
        from runner import load_config

        with pytest.raises(SystemExit) as exc_info:
            load_config(str(tmp_path / "nonexistent.yaml"))

        assert exc_info.value.code == 1

    def test_invalid_yaml_exits(self, tmp_path):
        """Should exit on invalid YAML syntax."""
        sys.path.insert(0, str(EXAMPLES_DIR))
        from runner import load_config

        config_file = tmp_path / "invalid.yaml"
        config_file.write_text("""
agent_id: "test"
api_key: [unclosed bracket
""")

        with pytest.raises(SystemExit) as exc_info:
            load_config(str(config_file))

        assert exc_info.value.code == 1

    def test_empty_file_exits(self, tmp_path):
        """Should exit when config file is empty."""
        sys.path.insert(0, str(EXAMPLES_DIR))
        from runner import load_config

        config_file = tmp_path / "empty.yaml"
        config_file.write_text("")

        with pytest.raises(SystemExit) as exc_info:
            load_config(str(config_file))

        assert exc_info.value.code == 1

    def test_missing_agent_id_exits(self, tmp_path):
        """Should exit when agent_id is missing."""
        sys.path.insert(0, str(EXAMPLES_DIR))
        from runner import load_config

        config_file = tmp_path / "missing_id.yaml"
        config_file.write_text("""
api_key: "test-api-key"
""")

        with pytest.raises(SystemExit) as exc_info:
            load_config(str(config_file))

        assert exc_info.value.code == 1

    def test_missing_api_key_exits(self, tmp_path):
        """Should exit when api_key is missing."""
        sys.path.insert(0, str(EXAMPLES_DIR))
        from runner import load_config

        config_file = tmp_path / "missing_key.yaml"
        config_file.write_text("""
agent_id: "test-agent-123"
""")

        with pytest.raises(SystemExit) as exc_info:
            load_config(str(config_file))

        assert exc_info.value.code == 1

    def test_empty_agent_id_exits(self, tmp_path):
        """Should exit when agent_id is empty string."""
        sys.path.insert(0, str(EXAMPLES_DIR))
        from runner import load_config

        config_file = tmp_path / "empty_id.yaml"
        config_file.write_text("""
agent_id: ""
api_key: "test-api-key"
""")

        with pytest.raises(SystemExit) as exc_info:
            load_config(str(config_file))

        assert exc_info.value.code == 1


class TestLoadCustomTools:
    """Tests for the custom tools loader."""

    def test_load_existing_tools(self):
        """Should load tools from registry."""
        sys.path.insert(0, str(EXAMPLES_DIR))
        from runner import load_custom_tools

        # Use the actual tools directory
        tools_dir = EXAMPLES_DIR / "tools"
        tools = load_custom_tools(tools_dir, ["calculator", "get_time"])

        assert len(tools) == 2
        # Verify tool names (SdkMcpTool uses 'name' attribute)
        tool_names = [getattr(t, "name", None) for t in tools]
        assert "calculator" in tool_names
        assert "get_time" in tool_names

    def test_load_nonexistent_tool_skipped(self):
        """Should skip tools that don't exist in registry."""
        sys.path.insert(0, str(EXAMPLES_DIR))
        from runner import load_custom_tools

        tools_dir = EXAMPLES_DIR / "tools"
        tools = load_custom_tools(tools_dir, ["calculator", "nonexistent_tool"])

        assert len(tools) == 1
        tool_names = [getattr(t, "name", None) for t in tools]
        assert "calculator" in tool_names

    def test_load_from_nonexistent_dir(self, tmp_path):
        """Should return empty list for nonexistent tools directory."""
        sys.path.insert(0, str(EXAMPLES_DIR))
        from runner import load_custom_tools

        tools_dir = tmp_path / "nonexistent_tools"
        tools = load_custom_tools(tools_dir, ["calculator"])

        assert tools == []

    def test_load_empty_tool_list(self):
        """Should return empty list when no tools requested."""
        sys.path.insert(0, str(EXAMPLES_DIR))
        from runner import load_custom_tools

        tools_dir = EXAMPLES_DIR / "tools"
        tools = load_custom_tools(tools_dir, [])

        assert tools == []
