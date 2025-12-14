# Pydantic AI Examples

Examples showing how to use the Thenvoi SDK with Pydantic AI.

## Prerequisites

Install with Pydantic AI support:
```bash
pip install "thenvoi-sdk[pydantic-ai]"
# Plus your model provider:
pip install "pydantic-ai-slim[openai]"  # or [anthropic], [google], etc.
```

## Configuration

Create an `agent_config.yaml` in the examples directory:

```yaml
pydantic_agent:
  agent_id: "your-agent-id"
  api_key: "your-api-key"

support_agent:
  agent_id: "your-support-agent-id"
  api_key: "your-support-api-key"
```

Set environment variables:
```bash
export THENVOI_WS_URL="wss://api.thenvoi.com/ws"
export THENVOI_REST_API_URL="https://api.thenvoi.com"
export OPENAI_API_KEY="your-openai-key"  # for OpenAI models
export ANTHROPIC_API_KEY="your-anthropic-key"  # for Anthropic models
```

## Examples

| File | Description |
|------|-------------|
| `01_basic_agent.py` | Simple agent that responds to messages |
| `02_custom_instructions.py` | Agent with custom system prompt |

## Running Examples

```bash
cd examples/pydantic_ai
python 01_basic_agent.py
```

## Model Strings

Pydantic AI uses model strings in the format `provider:model-name`:

- `openai:gpt-4o`
- `openai:gpt-4o-mini`
- `anthropic:claude-3-5-sonnet-latest`
- `anthropic:claude-3-5-haiku-latest`
- `google:gemini-1.5-pro`

See [Pydantic AI documentation](https://ai.pydantic.dev/) for more model options.

## Known Issues

### OpenAI `content: null` Error

When using OpenAI models with complex multi-turn tool usage, you may encounter:

```
Invalid value for 'content': expected a string, got null.
```

This is a [known issue in Pydantic AI](https://github.com/pydantic/pydantic-ai/issues/149) where assistant messages with `tool_calls` but no text content are sent with `content: null`, which OpenAI sometimes rejects.

**Workarounds:**

1. **Use Anthropic instead** (recommended for production):
   ```python
   adapter = PydanticAIAdapter(
       model="anthropic:claude-3-5-sonnet-latest",
       ...
   )
   ```

2. **Use the LangGraph adapter** - handles message history differently:
   ```python
   from thenvoi.agent.langgraph import LangGraphAdapter
   ```

3. **Simple conversations work fine** - the issue mainly occurs with complex multi-turn tool sequences.
