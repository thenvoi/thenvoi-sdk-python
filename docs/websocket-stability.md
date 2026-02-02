# WebSocket Connection Stability (INT-93)

This document describes the WebSocket connection stability features and how to test them.

## Problem

Agent WebSocket connections were dropping unpredictably after ~60 seconds of inactivity. This is because Phoenix servers expect periodic heartbeat messages to detect stale connections.

## Solution

Two features were added to the `phoenix-channels-python-client`:

### 1. Heartbeat Mechanism

Sends periodic heartbeat messages to keep connections alive.

- **Default interval:** 30 seconds (matches Phoenix JS client)
- **Message format:** `{topic: "phoenix", event: "heartbeat", payload: {}}`
- **Server response:** `phx_reply` with `{status: "ok"}`

```python
from phoenix_channels_python_client.client import PHXChannelsClient

client = PHXChannelsClient(
    ws_url,
    api_key,
    heartbeat_interval_secs=30,  # Default, can be changed
    # Set to None to disable heartbeat
)
```

### 2. Automatic Reconnection

Automatically reconnects when connection drops with exponential backoff.

- **Default:** Enabled
- **Max attempts:** 10 (set to 0 for unlimited)
- **Backoff:** 1s → 2s → 4s → 8s → ... → 30s max

```python
client = PHXChannelsClient(
    ws_url,
    api_key,
    auto_reconnect=True,              # Enable/disable
    reconnect_max_attempts=10,        # 0 = unlimited
    reconnect_backoff_base=1.0,       # Initial delay
    reconnect_backoff_max=30.0,       # Maximum delay
    on_disconnect=my_disconnect_handler,  # Optional callback
    on_reconnect=my_reconnect_handler,    # Optional callback
)
```

## Testing

### Automated Tests (Recommended)

Run the integration tests with a mock server (no external dependencies):

```bash
# Run all WebSocket stability tests
uv run pytest tests/integration/test_websocket_stability.py -v

# Run with detailed logs
uv run pytest tests/integration/test_websocket_stability.py -v -s

# Run a specific test
uv run pytest tests/integration/test_websocket_stability.py::test_automatic_reconnection_after_disconnect -v -s
```

### Manual Tests Against Real Server

For testing against a real Phoenix server (localhost or production):

#### 1. Heartbeat Simulation Test

Tests that heartbeat keeps the connection alive beyond the 60-second timeout.

```bash
# Full test (compares with/without heartbeat, ~160 seconds)
uv run python scripts/websocket_tests/test_int93_simulation.py \
  --ws-url ws://localhost:4000/api/v1/socket/websocket \
  --api-key YOUR_API_KEY

# Quick test (only with heartbeat, ~90 seconds)
uv run python scripts/websocket_tests/test_int93_simulation.py \
  --skip-no-heartbeat \
  --ws-url ws://localhost:4000/api/v1/socket/websocket \
  --api-key YOUR_API_KEY
```

**Expected output:**
```
[PASS] With Heartbeat | Duration: 90s | Connection stays alive beyond 60s
```

#### 2. Reconnection Test (Interactive)

Tests automatic reconnection by requiring manual server restart.

```bash
uv run python scripts/websocket_tests/test_int93_simulation.py \
  --test-reconnection \
  --ws-url ws://localhost:4000/api/v1/socket/websocket \
  --api-key YOUR_API_KEY
```

**Steps:**
1. Script connects and waits
2. **You restart the Phoenix server** (Ctrl+C twice, then restart)
3. Script detects disconnect and automatically reconnects
4. Press Ctrl+C when done

**Expected output:**
```
[!] DISCONNECTED (count: 1)
Reconnection attempt 1/∞ in 1.0 seconds...
[OK] RECONNECTED! (count: 1)
```

#### 3. Automated Reconnection Test (Mock Server)

Tests reconnection without needing a real server:

```bash
uv run python scripts/websocket_tests/test_reconnection_automated.py
```

**Expected output:**
```
STEP 1: Forcing server disconnect...
[OK] Disconnect detected!

STEP 2: Waiting for automatic reconnection...
[OK] Reconnected successfully!

STEP 3: Verifying connection is working...
[OK] Connection is active!

[PASS] RECONNECTION TEST PASSED!
```

## Test Files

| File | Type | Description |
|------|------|-------------|
| `tests/integration/test_websocket_stability.py` | pytest | Automated tests with mock server |
| `scripts/websocket_tests/test_int93_simulation.py` | Manual | Full simulation against real server |
| `scripts/websocket_tests/test_reconnection_automated.py` | Standalone | Automated reconnection test with mock |
| `scripts/websocket_tests/test_heartbeat_simulation.py` | Manual | Heartbeat-only test |
| `scripts/websocket_tests/test_reconnection_localhost.py` | Manual | Reconnection test for localhost |

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `heartbeat_interval_secs` | `30` | Seconds between heartbeats. Set to `None` to disable. |
| `auto_reconnect` | `True` | Enable automatic reconnection on disconnect. |
| `reconnect_max_attempts` | `10` | Max reconnection attempts. `0` = unlimited. |
| `reconnect_backoff_base` | `1.0` | Initial backoff delay in seconds. |
| `reconnect_backoff_max` | `30.0` | Maximum backoff delay in seconds. |
| `on_disconnect` | `None` | Async callback when disconnected. Receives exception. |
| `on_reconnect` | `None` | Async callback when reconnected. |

## Callbacks Example

```python
async def handle_disconnect(error: Optional[Exception]):
    logger.warning("Disconnected: %s", error)
    # Notify monitoring, update UI, etc.

async def handle_reconnect():
    logger.info("Reconnected!")
    # Resume operations, refresh state, etc.

client = PHXChannelsClient(
    ws_url,
    api_key,
    on_disconnect=handle_disconnect,
    on_reconnect=handle_reconnect,
)
```

## Troubleshooting

### Connection still drops with heartbeat enabled

1. Check that `heartbeat_interval_secs` is not `None`
2. Verify server is responding to heartbeats (check logs for "Heartbeat acknowledged")
3. Server timeout might be shorter than 60s - try reducing `heartbeat_interval_secs`

### Reconnection not working

1. Check that `auto_reconnect=True`
2. Verify `reconnect_max_attempts` hasn't been reached
3. Check logs for "Reconnection attempt" messages
4. Ensure server is back up before max attempts exhausted

### Logs to enable for debugging

```python
import logging
logging.getLogger("phoenix_channels_python_client").setLevel(logging.DEBUG)
```
