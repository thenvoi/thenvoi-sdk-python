# Framework Configurations for Parameterized Tests

This directory contains configuration modules for the parameterized conformance tests. Instead of writing hundreds of lines of tests for each framework adapter/converter, you define a configuration that describes the framework's behavior.

## Structure

- `converters.py` - Configuration for history converter conformance tests
- `adapters.py` - Configuration for adapter conformance tests
- `_output_adapters.py` - Helpers for asserting on different output formats
- `base_converter_tests.py` - Base class for standalone converter testing
- `base_adapter_tests.py` - Base class for standalone adapter testing

## Quick Start: Testing a New Converter (Recommended)

The easiest way to test a new converter is to inherit from `BaseConverterTests`:

```python
# tests/converters/test_my_framework.py
from tests.framework_configs.base_converter_tests import BaseConverterTests
from thenvoi.converters.my_framework import MyFrameworkHistoryConverter

class TestMyFrameworkConverter(BaseConverterTests):
    converter_class = MyFrameworkHistoryConverter
    output_type = "dict_list"  # or "langchain_messages", "pydantic_ai_messages", "string"

    # Optional - override defaults as needed:
    tool_handling_mode = "structured"  # or "skip", "langchain", "raw_json"
    skips_own_messages = True
    batches_tool_calls = True
```

Run: `uv run pytest tests/converters/test_my_framework.py -v`

**That's it - ~6 lines and you get 20+ conformance tests automatically.**

### Available Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `converter_class` | (required) | Your converter class |
| `output_type` | (required) | `"dict_list"`, `"langchain_messages"`, `"pydantic_ai_messages"`, `"string"` |
| `tool_handling_mode` | `"skip"` | `"skip"`, `"structured"`, `"langchain"`, `"raw_json"` |
| `skips_own_messages` | `True` | Skip agent's own assistant messages |
| `converts_other_agents_to_user` | `True` | Convert other agents to user role |
| `batches_tool_calls` | `False` | Batch consecutive tool_call messages |
| `batches_tool_results` | `False` | Batch consecutive tool_result messages |
| `supports_is_error` | `False` | Preserve is_error field in tool results |
| `skips_empty_content` | `False` | Skip messages with empty content |
| `empty_sender_prefix_behavior` | `"no_prefix"` | `"no_prefix"` or `"empty_brackets"` |

---

## Quick Start: Testing a New Adapter (Recommended)

The easiest way to test a new adapter is to inherit from `BaseAdapterTests`:

```python
# tests/adapters/test_my_framework.py
from tests.framework_configs.base_adapter_tests import BaseAdapterTests
from thenvoi.adapters.my_framework import MyFrameworkAdapter

class TestMyFrameworkAdapter(BaseAdapterTests):
    # Required
    adapter_class = MyFrameworkAdapter

    # Optional overrides (defaults shown)
    has_history_converter = True
    has_custom_tools = True
    custom_tools_attr = "_custom_tools"
    custom_tool_format = "tuple"  # or "callable"
    supports_enable_execution_reporting = True
    supports_system_prompt_override = True
    history_storage_attr = "_message_history"
    system_prompt_attr = "_system_prompt"
    cleanup_storage_attrs = ["_message_history"]

    # Required - implement these methods for framework-specific behavior
    def create_adapter(self, **kwargs):
        return MyFrameworkAdapter(**kwargs)

    async def setup_on_started(self, adapter):
        await adapter.on_started(agent_name="TestBot", agent_description="A test bot")

    async def setup_on_message(self, adapter, mock_tools):
        await self.setup_on_started(adapter)
        return {}  # Return any mocks needed for mock_llm_call

    def mock_llm_call(self, adapter, mocks):
        # Return a context manager that mocks LLM calls
        from unittest.mock import patch, MagicMock
        mock_response = MagicMock(stop_reason="end_turn", content=[])
        return patch.object(adapter, "_call_llm", return_value=mock_response)
```

Run: `uv run pytest tests/adapters/test_my_framework.py -v`

**With ~30 lines you get 16 conformance tests automatically.**

### Available Adapter Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `adapter_class` | (required) | Your adapter class |
| `has_history_converter` | `True` | Adapter has history_converter attribute |
| `has_custom_tools` | `True` | Supports additional_tools parameter |
| `custom_tools_attr` | `"_custom_tools"` | Attribute name storing custom tools |
| `custom_tool_format` | `"tuple"` | `"tuple"` for (Model, func) or `"callable"` for func |
| `supports_enable_execution_reporting` | `True` | Has enable_execution_reporting parameter |
| `supports_system_prompt_override` | `True` | system_prompt parameter works |
| `history_storage_attr` | `"_message_history"` | Dict storing room history |
| `system_prompt_attr` | `"_system_prompt"` | Attribute for rendered prompt |
| `cleanup_storage_attrs` | `["_message_history"]` | Attributes to check in cleanup tests |
| `default_model` | `None` | Expected default model value |
| `additional_init_checks` | `{}` | Dict of {attr: expected_value} to verify |

---

## Alternative: Adding to Shared Config (for SDK maintainers)

If you want your converter to be included in the SDK's parameterized test suite:

1. **Create your converter** in `src/thenvoi/converters/`

2. **Add a configuration** in `converters.py`:

```python
from thenvoi.converters.your_framework import YourFrameworkHistoryConverter

CONVERTER_CONFIGS["your_framework"] = ConverterConfig(
    name="your_framework",
    converter_class=YourFrameworkHistoryConverter,
    output_type="dict_list",  # or "string", "langchain_messages", "pydantic_ai_messages"

    # Behavior flags (all optional, shown with defaults):
    skips_own_messages=True,              # Skip agent's own text messages
    converts_other_agents_to_user=True,   # Convert other agents to "user" role
    skips_empty_content=False,            # Skip messages with empty content
    tool_handling_mode=ToolHandlingMode.SKIP,  # How to handle tool messages
    batches_tool_calls=False,             # Batch consecutive tool_calls
    batches_tool_results=False,           # Batch consecutive tool_results
    supports_is_error=False,              # Preserve is_error in tool results
    logs_malformed_json=False,            # Log when JSON parsing fails
)
```

3. **If your output format is new**, add an output adapter in `_output_adapters.py`:

```python
class YourOutputAdapter(OutputTypeAdapter):
    def get_length(self, result: Any) -> int:
        ...
    def get_content(self, result: Any, index: int) -> str:
        ...
    # etc.
```

4. **Run the tests**:
```bash
uv run pytest tests/converters/test_converter_conformance.py -k "your_framework" -v
```

## Adding a New Adapter

1. **Create your adapter** in `src/thenvoi/adapters/`

2. **Create a factory function** in `adapters.py`:

```python
def _create_your_adapter(**kwargs: Any) -> Any:
    from thenvoi.adapters.your_framework import YourAdapter
    return YourAdapter(**kwargs)
```

3. **Add a configuration** using `make_standard_adapter_config`:

```python
ADAPTER_CONFIGS["your_framework"] = make_standard_adapter_config(
    "your_framework",
    _create_your_adapter,

    # Optional configuration (shown with defaults):
    default_model=None,                    # Default model string
    has_history_converter=True,            # Has history_converter attribute
    has_custom_tools=True,                 # Supports additional_tools param
    custom_tools_attr="_custom_tools",     # Attribute storing custom tools
    custom_tool_format="tuple",            # "tuple" or "callable"
    supports_enable_execution_reporting=True,
    supports_system_prompt_override=True,
    history_storage_attr="_message_history",  # Dict storing room history
    system_prompt_attr="_system_prompt",   # Attribute for rendered prompt
    cleanup_storage_attrs=["_message_history"],

    # Callbacks for framework-specific behavior:
    on_started_callback=_your_on_started,
    mock_llm_callback=_your_mock_llm,
    error_setup_callback=_your_error_setup,
    verify_participants_injection=_your_verify_participants,
)
```

4. **Implement callbacks** for your adapter:

```python
async def _your_on_started(adapter: Any, config: AdapterConfig) -> None:
    """Set up any mocks needed before calling on_started."""
    await adapter.on_started(agent_name="TestBot", agent_description="A test bot")

def _your_mock_llm(adapter: Any, mocks: dict | None, captured_input: dict | None) -> Any:
    """Return an async context manager that mocks LLM calls."""
    @asynccontextmanager
    async def ctx() -> AsyncIterator[None]:
        # Set up mocks
        yield
    return ctx()
```

5. **Run the tests**:
```bash
uv run pytest tests/adapters/test_adapter_conformance.py -k "your_framework" -v
```

## Configuration Reference

### ConverterConfig Fields

| Field | Type | Description |
|-------|------|-------------|
| `name` | str | Framework identifier |
| `converter_class` | type | The converter class |
| `output_type` | OutputType | Output format type |
| `skips_own_messages` | bool | Skip agent's own assistant messages |
| `converts_other_agents_to_user` | bool | Convert other agents to user role |
| `skips_empty_content` | bool | Skip messages with empty content |
| `tool_handling_mode` | ToolHandlingMode | How to handle tool messages |
| `batches_tool_calls` | bool | Batch consecutive tool_call messages |
| `batches_tool_results` | bool | Batch consecutive tool_result messages |
| `supports_is_error` | bool | Preserve is_error field in tool results |
| `logs_malformed_json` | bool | Log warning for malformed JSON |

### ToolHandlingMode Values

| Mode | Description |
|------|-------------|
| `SKIP` | Ignore tool_call and tool_result messages |
| `STRUCTURED` | Convert to structured format (Anthropic API format) |
| `LANGCHAIN` | Convert to LangChain message format |
| `RAW_JSON` | Include as raw JSON string |

### AdapterConfig Key Fields

| Field | Type | Description |
|-------|------|-------------|
| `name` | str | Framework identifier |
| `factory` | Callable | Factory function to create adapter |
| `has_history_converter` | bool | Adapter has history_converter |
| `has_custom_tools` | bool | Supports additional_tools parameter |
| `supports_cleanup_all` | bool | Has cleanup_all() method |
| `on_started_callback` | Callable | Custom on_started setup |
| `mock_llm_callback` | Callable | Mock LLM for on_message tests |
| `error_setup_callback` | Callable | Set up error conditions for tests |
| `verify_participants_injection` | Callable | Verify participants message handling |

## Tips

- Look at existing configs for similar frameworks as starting points
- Use `pytest -k "your_framework" -v` to run just your framework's tests
- The test output will tell you which configuration flags need adjustment
- Framework-specific tests that don't fit the conformance model go in separate files (e.g., `test_crewai_specific.py`)
