# Parlant Examples for Thenvoi

Examples showing how to use the Thenvoi SDK with [Parlant](https://github.com/emcie-co/parlant) - an AI agent framework designed for controlled, guideline-based agent behavior.

## Why Parlant?

Parlant provides:
- **Behavioral Guidelines**: Define condition/action rules that agents consistently follow
- **Built-in Guardrails**: Prevent hallucination and off-topic responses
- **Explainability**: Understand why agents make specific decisions
- **Production-Ready**: Designed for customer-facing deployments

## Prerequisites

**Install with Parlant support:**
```bash
uv add "git+https://github.com/thenvoi/thenvoi-sdk-python.git[parlant]"
```

**Or from repository:**
```bash
uv sync --extra parlant
```

---

## Quick Start

```python
from thenvoi import Agent
from thenvoi.adapters import ParlantAdapter

adapter = ParlantAdapter(
    model="gpt-4o",
    custom_section="You are a helpful assistant.",
    guidelines=[
        {
            "condition": "User asks for help",
            "action": "Acknowledge their request and provide detailed assistance",
        }
    ],
)

agent = Agent.create(
    adapter=adapter,
    agent_id="your-agent-id",
    api_key="your-api-key",
)
await agent.run()
```

---

## Examples

| File | Description |
|------|-------------|
| `01_basic_agent.py` | **Minimal setup** - Simple agent with ParlantAdapter. |
| `02_with_guidelines.py` | **Behavioral guidelines** - Agent with condition/action rules. |
| `03_support_agent.py` | **Customer support** - Realistic support agent with specialized guidelines. |

---

## Guidelines System

Parlant's guidelines are the key differentiator. They ensure consistent behavior through condition/action pairs:

```python
GUIDELINES = [
    {
        "condition": "Customer asks about refunds",
        "action": "Check order status first to see if eligible",
    },
    {
        "condition": "User is frustrated",
        "action": "Acknowledge their frustration before providing solutions",
    },
]
```

### How Guidelines Work

1. **Condition Matching**: Parlant evaluates each message against guideline conditions
2. **Action Execution**: When a condition matches, the corresponding action is followed
3. **Consistent Behavior**: Guidelines are enforced reliably, not just "suggested"

This is fundamentally different from system prompts that LLMs may ignore. Parlant ensures the guidelines are actually followed.

---

## Configuration

### 1. Copy configuration files from examples

```bash
# From project root
cp .env.example .env
cp agent_config.yaml.example agent_config.yaml
```

### 2. Add your OpenAI API key to `.env`

Edit `.env` and set your OpenAI API key:
```bash
OPENAI_API_KEY=sk-your-actual-key-here
```

### 3. Add agent credentials to `agent_config.yaml`

1. Create an external agent on the [Thenvoi Platform](https://app.thenvoi.com)
2. Generate an API key for the agent
3. Edit `agent_config.yaml` and fill in the Parlant agent section:

```yaml
parlant_agent:
  agent_id: "your-agent-id-from-platform"
  api_key: "your-api-key-from-platform"
```

> **Note:** Always copy from the example files to ensure correct URLs and formatting. Never hardcode credentials.

---

## Running Examples

**Important:** Run from the project root directory (where `agent_config.yaml` is located):

```bash
# From project root
cd /path/to/thenvoi-sdk-python

uv run python examples/parlant/01_basic_agent.py
uv run python examples/parlant/02_with_guidelines.py
uv run python examples/parlant/03_support_agent.py
```

> **Note:** The config loader looks for `agent_config.yaml` in the current working directory. Running from a subdirectory will cause a `FileNotFoundError`.

---

## Adapter Options

```python
ParlantAdapter(
    model="gpt-4o",                    # Model to use (OpenAI format)
    system_prompt=None,                # Full system prompt override
    custom_section="...",              # Custom instructions (added to default prompt)
    guidelines=[...],                  # Behavioral guidelines
    openai_api_key=None,               # API key (uses env var if not provided)
    enable_execution_reporting=False,  # Show tool calls in chat
)
```

---

## Model Support

The Parlant adapter uses OpenAI-compatible API format:

- `gpt-4o`
- `gpt-4o-mini`
- `gpt-4-turbo`
- Any OpenAI-compatible model

For Anthropic models, use the `AnthropicAdapter` instead.

---

## Use Cases

### Customer Support
Perfect for support agents that need to:
- Follow specific escalation procedures
- Handle sensitive topics appropriately
- Maintain consistent response quality

### Compliance-Critical Applications
Ideal when you need:
- Guaranteed adherence to rules
- Auditable decision-making
- Predictable behavior

### Multi-Agent Orchestration
Works well for:
- Coordinator agents with specific handoff rules
- Specialist agents with domain-specific guidelines
- Agents that need to collaborate consistently

---

## Learn More

- [Parlant Documentation](https://www.parlant.io/docs)
- [Parlant GitHub](https://github.com/emcie-co/parlant)
- [Thenvoi SDK Documentation](https://github.com/thenvoi/thenvoi-sdk-python)

