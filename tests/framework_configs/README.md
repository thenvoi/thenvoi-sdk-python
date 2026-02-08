# Framework Configurations for Parameterized Tests

This directory contains configuration modules for the parameterized conformance tests. Instead of writing hundreds of lines of tests for each framework adapter/converter, you define a configuration that describes the framework's behavior.

## Structure

- `converters.py` - Configuration for history converter conformance tests
- `adapters.py` - Configuration for adapter conformance tests
- `_output_adapters.py` - Helpers for asserting on different output formats

## Quick Start: Testing a New Converter

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

## Quick Start: Testing a New Adapter

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
