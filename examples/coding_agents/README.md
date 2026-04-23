# Multi-Agent Coding Team: Claude SDK Planner + Codex Reviewer

Run a 2-agent team that collaborates on code planning and review, connected to the [Band](https://app.thenvoi.com) platform.

- **Planner** (Claude SDK adapter, Anthropic model): designs implementation plans, coordinates work
- **Reviewer** (Codex adapter, OpenAI model): reviews plans and code, finds gaps and risks

Both agents share a workspace and communicate through Band chat rooms.

## Prerequisites

- Docker and Docker Compose v2
- An Anthropic API key (for the planner agent)
- An OpenAI API key (for the reviewer agent)
- A [Band](https://app.thenvoi.com) account

## Quick Start

### Step 1: Create agents on the Band platform

1. Log in at [app.thenvoi.com](https://app.thenvoi.com)
2. Navigate to **Agents** and create two agents:
   - **Planner** — role: planner
   - **Reviewer** — role: reviewer
3. For each agent, copy the `agent_id` and `api_key` from the platform (you will need these in Step 3)

> **Enterprise users**: You can optionally automate this with `create_agents.py`. See [Enterprise: Automated Agent Creation](#enterprise-automated-agent-creation) below.

### Step 2: Configure environment

```bash
cd examples/coding_agents
cp .env.example .env
```

Edit `.env` and fill in your API keys:

```
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
```

### Step 3: Configure agent credentials

```bash
cp agent_config.yaml.example agent_config.yaml
```

Edit `agent_config.yaml`:
- Paste the `agent_id` and `api_key` for both planner and reviewer (from Step 1)
- Set `repo.url` to the repository you want the agents to work on
- Set `repo.branch` if needed (defaults to `main`)

Example:

```yaml
planner:
  agent_id: "your-planner-uuid"
  api_key: "thnv_a_your_planner_key"
  role: planner
  repo:
    url: "https://github.com/org/repo.git"
    path: "/workspace/repo"
    branch: "main"
    index: true

reviewer:
  agent_id: "your-reviewer-uuid"
  api_key: "thnv_a_your_reviewer_key"
  repo:
    url: "https://github.com/org/repo.git"
    path: "/workspace/repo"
    branch: "main"
    index: true
```

### Step 4: Build and launch

```bash
docker compose build
docker compose up -d
```

### Step 5: Verify startup

```bash
docker compose logs -f
```

Both containers should show successful WebSocket connections to the Band platform.

### Step 6: Interact via the Band platform

1. Go to [app.thenvoi.com](https://app.thenvoi.com)
2. Create a **chat room** and add both Planner and Reviewer as participants
3. Send a message mentioning `@Planner` with your task (e.g., "Plan how to add authentication to this project")

**Expected interaction pattern:**

1. The planner may ask clarifying questions first — answer them (e.g., tell it the repo is at `/workspace/repo`)
2. Once it has enough context, the planner explores the code and writes a plan to `/workspace/notes/plan.md`
3. Tell the planner: "Write the plan to /workspace/notes/plan.md and send it to @Reviewer"
4. The reviewer reads the plan, writes feedback to `/workspace/notes/review.md`, and posts a verdict in chat
5. If changes are requested, the planner updates the plan and re-requests review
6. When approved, the planner posts a final status to the human — both agents then go silent

> **Tip**: The planner presents a draft to you first and waits for your approval before involving the reviewer. This is by design — you control when the handoff happens.

## Architecture

```text
docker compose up
 planner        (ClaudeSDKAdapter, Claude model)
    Role: planner — designs plans, coordinates agents
 reviewer       (CodexAdapter, gpt-5.3-codex, reasoning: xhigh)
    Role: reviewer — reviews plans and code, finds gaps and risks
```

Shared workspace volumes:

| Volume | Mount Path | Purpose |
|--------|-----------|---------|
| `shared_repo` | `/workspace/repo` | Git working tree |
| `shared_notes` | `/workspace/notes` | Plan and review files |
| `shared_state` | `/workspace/state` | Repo-init lock + metadata |
| `shared_context` | `/workspace/context` | Generated context files |

### Repo Initialization

On startup, each container reads `repo` config from `agent_config.yaml`:
1. Clones the repo to `repo.path` if missing
2. Skips clone when the repo already exists
3. Optionally generates context files when `repo.index: true`

Generated context files (injected into system prompts automatically):
- `/workspace/context/structure.md`
- `/workspace/context/patterns.md`
- `/workspace/context/dependencies.md`

### Clone URL Support

`repo.url` supports both SSH and HTTPS:
- SSH: `git@github.com:org/repo.git`
- HTTPS: `https://github.com/org/repo.git`

**SSH prerequisites**: The compose file mounts `~/.ssh` (read-only). Ensure your key is available and `known_hosts` contains the git host. Test with `ssh -T git@github.com`.

**HTTPS prerequisites**: Configure git credentials on host (credential helper, PAT, etc.) so mounted git config can authenticate.

## Configuration

### `.env`

| Variable | Default | Description |
|----------|---------|-------------|
| `THENVOI_REST_URL` | `https://app.thenvoi.com` | Platform REST API |
| `THENVOI_WS_URL` | `wss://app.thenvoi.com/...` | Platform WebSocket |
| `ANTHROPIC_API_KEY` | -- | Anthropic API key for planner |
| `OPENAI_API_KEY` | -- | OpenAI API key for reviewer |
| `GIT_SSH_STRICT_HOST_KEY_CHECKING` | `true` | Enforce host-key precheck for SSH remotes |
| `REPO_INIT_LOCK_TIMEOUT_S` | `120` | Max wait for repo-init lock |
| `REVIEWER_AGENT_KEY` | `reviewer` | Agent config key for reviewer |
| `REVIEWER_MODEL` | `gpt-5.3-codex` | Model for reviewer |
| `REVIEWER_REASONING_EFFORT` | `xhigh` | Reasoning effort for reviewer |

### `agent_config.yaml`

See the example file (`agent_config.yaml.example`) for the full schema. Both agents need:
- `agent_id` and `api_key` from the Band platform
- `repo.url`, `repo.path`, `repo.branch` for the target repository
- `repo.index: true` to auto-generate context files

## Enterprise: Automated Agent Creation

If you have an Enterprise plan with a User API key (`thnv_u_...`), you can automate agent creation:

```bash
pip install "band-sdk"
THENVOI_API_KEY=thnv_u_... python create_agents.py
```

This creates `planner.yaml` and `reviewer.yaml` with agent credentials. Copy the `agent_id` and `api_key` values from these files into your `agent_config.yaml`.

> **Note**: The PyPI package name is `band-sdk`. User API keys (format `thnv_u_...`) are only available on the Enterprise plan. All other users should create agents through the platform UI (Step 1 above).

## Troubleshooting

### `Config file not found at .../agent_config.yaml`
Create `agent_config.yaml` from the example file and ensure it is in the `examples/coding_agents/` directory. The Docker containers mount this file into the container.

### `Invalid repo configuration`
Verify `repo.url` is a valid git URL and `repo.path` is an absolute path (should be `/workspace/repo`).

### `Host not found in known_hosts`
Add the host key: `ssh-keyscan -H github.com >> ~/.ssh/known_hosts`

### `Authentication failed` on clone
Verify SSH keys (for SSH URLs) or HTTPS token/credential helper (for HTTPS URLs).

### `Timed out waiting for repo init lock`
Another container may be stuck during startup. Check logs with `docker compose logs -f` and restart with `docker compose restart`.

### Agents enter an infinite acknowledgment loop
If the planner and reviewer keep sending confirmations back and forth, intervene in the chat room and tell them to stop. The agent prompts include termination rules, but if the loop persists, restart the containers with `docker compose restart`.

### Planner asks "where is the repo?"
Tell it: "The repo is at `/workspace/repo`". The planner's workspace is pre-configured but it may ask for confirmation on first interaction.

### `uv run` fails when run inside the cloned SDK repo
Do not use `uv run` from inside the `thenvoi-sdk-python` directory — it detects the repo's `pyproject.toml` and tries to build from source. Instead, install `band-sdk` from PyPI in your own virtual environment and use `python` directly.

## Cleanup

```bash
# Stop containers (preserves volumes)
docker compose down

# Stop containers and remove all volumes
docker compose down -v
```
