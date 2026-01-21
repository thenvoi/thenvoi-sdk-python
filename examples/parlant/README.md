# Parlant Examples for Thenvoi

Examples showing how to use the Thenvoi SDK with [Parlant](https://github.com/emcie-co/parlant) - an AI agent framework designed for controlled, guideline-based agent behavior.

## Why Parlant?

Parlant provides:
- **Behavioral Guidelines**: Define condition/action rules that agents consistently follow
- **Built-in Guardrails**: Prevent hallucination and off-topic responses
- **Explainability**: Understand why agents make specific decisions
- **Production-Ready**: Designed for customer-facing deployments
- **Session Management**: Proper conversation context through the SDK

## Prerequisites

### 1. Install with Parlant support

```bash
uv add "git+https://github.com/thenvoi/thenvoi-sdk-python.git[parlant]"
```

**Or from repository:**
```bash
uv sync --extra parlant
```

### 2. Run a Parlant Server

The adapter requires a running Parlant server. You can run one locally:

```bash
# Using Docker
docker run -p 8000:8000 emcie/parlant

# Or install and run directly
pip install parlant
parlant serve --port 8000
```

The adapter connects to `http://localhost:8000` by default. Set `PARLANT_URL` environment variable to use a different server.

---

## Quick Start

```python
from thenvoi import Agent
from thenvoi.adapters import ParlantAdapter

adapter = ParlantAdapter(
    parlant_url="http://localhost:8000",  # Your Parlant server
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

Parlant's guidelines are the key differentiator. They ensure consistent behavior through condition/action pairs that are **actually enforced** by the Parlant SDK, not just suggested in prompts:

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

1. **Registration**: Guidelines are registered with the Parlant server at startup
2. **Condition Matching**: Parlant evaluates each message against guideline conditions
3. **Action Enforcement**: When a condition matches, the corresponding action is enforced
4. **Consistent Behavior**: Guidelines are reliably followed, not just "suggested"

This is fundamentally different from system prompts that LLMs may ignore. The Parlant SDK ensures guidelines are actually followed through its guideline matching engine.

---

## Configuration

### 1. Copy configuration files from examples

```bash
# From project root
cp .env.example .env
cp agent_config.yaml.example agent_config.yaml
```

### 2. Set up environment variables in `.env`

```bash
# Parlant server URL (required)
PARLANT_URL=http://localhost:8000

# Optional: Pre-configured Parlant agent ID
# If not set, the adapter creates an agent dynamically
PARLANT_AGENT_ID=your-parlant-agent-id
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

# Make sure Parlant server is running
PARLANT_URL=http://localhost:8000 uv run python examples/parlant/01_basic_agent.py
PARLANT_URL=http://localhost:8000 uv run python examples/parlant/02_with_guidelines.py
PARLANT_URL=http://localhost:8000 uv run python examples/parlant/03_support_agent.py
```

> **Note:** The config loader looks for `agent_config.yaml` in the current working directory. Running from a subdirectory will cause a `FileNotFoundError`.

---

## Adapter Options

```python
ParlantAdapter(
    # Parlant SDK configuration
    parlant_url="http://localhost:8000",  # Parlant server URL (or PARLANT_URL env)
    agent_id=None,                         # Pre-configured agent ID (or PARLANT_AGENT_ID env)
    
    # Agent configuration
    system_prompt=None,                    # Full system prompt override
    custom_section="...",                  # Custom instructions (added to default prompt)
    guidelines=[...],                      # Behavioral guidelines (registered with Parlant)
    
    # Options
    enable_execution_reporting=False,      # Show tool calls in chat
    wait_timeout=60,                       # Timeout waiting for agent responses
)
```

---

## Agent ID Modes

### Dynamic Agent Creation (Default)

If you don't provide `agent_id`, the adapter creates a new Parlant agent automatically:

```python
adapter = ParlantAdapter(
    parlant_url="http://localhost:8000",
    guidelines=[...],  # Registered with the new agent
)
```

### Pre-configured Agent

If you have a pre-configured agent on your Parlant server:

```python
adapter = ParlantAdapter(
    parlant_url="http://localhost:8000",
    agent_id="my-parlant-agent-id",  # Use existing agent
)
```

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

## Troubleshooting

### Cannot connect to Parlant server

```
ConnectionError: Cannot connect to http://localhost:8000
```

Make sure your Parlant server is running:
```bash
# Check if server is running
curl http://localhost:8000/health

# Start Parlant server
parlant serve --port 8000
```

### Guidelines not being followed

Guidelines are registered with the Parlant server. If they're not being followed:

1. Check the Parlant server logs for guideline registration
2. Verify the condition matches your test messages
3. Try more specific conditions

### Import errors

```
ImportError: parlant package required for ParlantAdapter
```

Install the Parlant extra:
```bash
uv sync --extra parlant
# or
pip install 'thenvoi-sdk[parlant]'
```

---

## Learn More

- [Parlant Documentation](https://www.parlant.io/docs)
- [Parlant GitHub](https://github.com/emcie-co/parlant)
- [Thenvoi SDK Documentation](https://github.com/thenvoi/thenvoi-sdk-python)
