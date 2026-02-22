#!/usr/bin/env python3
"""
WebSocket harness for codex app-server.

Prerequisite:
    codex app-server --listen ws://127.0.0.1:8765

Run:
    python scripts/codex_ws_harness.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from thenvoi.integrations.codex import CodexJsonRpcError, CodexWebSocketClient, RpcEvent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Codex WebSocket harness")
    parser.add_argument("--url", default="ws://127.0.0.1:8765", help="Codex WS URL")
    parser.add_argument(
        "--model",
        default=None,
        help="Model id. If omitted, auto-selects first visible codex model from model/list.",
    )
    parser.add_argument("--cwd", default=os.getcwd())
    parser.add_argument("--approval-policy", default="never")
    parser.add_argument("--personality", default="pragmatic")
    parser.add_argument(
        "--prompt",
        default="Reply with exactly: harness-ok",
        help="User prompt for turn/start",
    )
    parser.add_argument(
        "--experimental-api",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable initialize.capabilities.experimentalApi",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Idle timeout (seconds) while waiting for next event",
    )
    parser.add_argument(
        "--skip-turn",
        action="store_true",
        help="Only run initialize + thread/start (no turn/start)",
    )
    return parser.parse_args()


async def handle_server_request(client: CodexWebSocketClient, event: RpcEvent) -> None:
    if event.id is None:
        return

    if event.method == "item/tool/call":
        await client.respond(
            event.id,
            {
                "contentItems": [
                    {
                        "type": "inputText",
                        "text": "WS harness has no dynamic tool dispatcher configured.",
                    }
                ],
                "success": False,
            },
        )
        return

    if event.method in {
        "item/commandExecution/requestApproval",
        "item/fileChange/requestApproval",
    }:
        await client.respond(event.id, {"decision": "decline"})
        return

    await client.respond_error(
        event.id,
        code=-32601,
        message=f"Unhandled server request: {event.method}",
    )


def _as_dict(params: dict[str, Any] | list[Any] | None) -> dict[str, Any]:
    if isinstance(params, dict):
        return params
    return {}


def _pick_model(args_model: str | None, model_list_result: dict[str, Any]) -> str:
    if args_model:
        return args_model

    data = model_list_result.get("data") or []
    if not isinstance(data, list):
        return "gpt-5.2-codex"

    for model in data:
        if not isinstance(model, dict):
            continue
        model_id = model.get("id")
        hidden = bool(model.get("hidden", False))
        if hidden or not isinstance(model_id, str):
            continue
        if "codex" in model_id:
            return model_id

    return "gpt-5.2-codex"


async def run() -> int:
    args = parse_args()
    client = CodexWebSocketClient(ws_url=args.url)

    final_text = ""
    turn_status = None
    thread_id = None
    chosen_model = None
    last_event_method = None
    last_event_at = time.monotonic()

    try:
        await client.connect()
        init = await client.initialize(
            client_name="thenvoi_codex_ws_harness",
            client_title="Thenvoi Codex WS Harness",
            client_version="0.1.0",
            experimental_api=args.experimental_api,
        )
        print(f"initialize.result={json.dumps(init)}")

        model_list = await client.request("model/list", {})
        chosen_model = _pick_model(args.model, model_list)
        print(f"model.selected={chosen_model}")

        thread_result = await client.request(
            "thread/start",
            {
                "model": chosen_model,
                "cwd": args.cwd,
                "approvalPolicy": args.approval_policy,
                "personality": args.personality,
            },
        )
        thread = thread_result.get("thread", {})
        thread_id = thread.get("id")
        print(f"thread.id={thread_id}")

        if args.skip_turn:
            print("skip-turn enabled; stopping after thread/start")
            return 0

        turn_result = await client.request(
            "turn/start",
            {
                "threadId": thread_id,
                "input": [{"type": "text", "text": args.prompt}],
                "model": chosen_model,
                "cwd": args.cwd,
                "approvalPolicy": args.approval_policy,
                "personality": args.personality,
            },
        )
        print(f"turn.started={json.dumps(turn_result)}")

        while True:
            event = await client.recv_event(timeout_s=args.timeout)
            last_event_method = event.method
            last_event_at = time.monotonic()

            if event.kind == "request":
                print(f"[request] {event.method}")
                await handle_server_request(client, event)
                continue

            params = _as_dict(event.params)
            if event.method == "item/agentMessage/delta":
                delta = str(params.get("delta", ""))
                final_text += delta
                print(delta, end="", flush=True)
                continue

            if event.method == "turn/completed":
                turn = _as_dict(params.get("turn")) if isinstance(params, dict) else {}
                turn_status = turn.get("status")
                print(f"\n[turn/completed] status={turn_status}")
                break

            print(f"[notify] {event.method}")

        print("\nsummary:")
        print(f"  thread_id={thread_id}")
        print(f"  turn_status={turn_status}")
        print(f"  final_text={final_text!r}")
        return 0
    except asyncio.TimeoutError:
        idle_for = int(time.monotonic() - last_event_at)
        print("error: timed out waiting for next event")
        print(f"  idle_timeout_s={args.timeout}")
        print(f"  idle_elapsed_s={idle_for}")
        print(f"  last_event={last_event_method}")
        print(f"  selected_model={chosen_model}")
        print(
            "hint: verify model access/network; or run with --skip-turn for protocol-only validation."
        )
        return 1
    except (CodexJsonRpcError, RuntimeError) as exc:
        print(f"error: {exc}")
        return 1
    finally:
        await client.close()


def main() -> None:
    raise SystemExit(asyncio.run(run()))


if __name__ == "__main__":
    main()
