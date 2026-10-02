#!/usr/bin/env python3
"""Functional validation for the local, authenticated oMLX target service."""

from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BASE_URL = os.environ.get("OMLX_BASE_URL", "http://127.0.0.1:8013")
MODEL = os.environ.get("OMLX_MODEL_ID", "qwen38-27b-mtp-4bit")
OUT = Path(os.environ.get("BENCH_OUTPUT_DIR", Path(__file__).resolve().parent / "output")) / "validation.json"
ACCOUNT = os.environ.get("OMLX_KEYCHAIN_ACCOUNT", os.environ.get("USER", ""))
SERVICE = os.environ.get("OMLX_KEYCHAIN_SERVICE", "oMLX API Key 127.0.0.1:8013")


def api_key() -> str:
    if os.environ.get("OMLX_API_KEY"):  # set this, or store the key in the macOS Keychain
        return os.environ["OMLX_API_KEY"]
    value = subprocess.run(
        ["/usr/bin/security", "find-generic-password", "-a", ACCOUNT, "-s", SERVICE, "-w"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not value.startswith("sk-omlx-") or len(value) < 32:
        raise RuntimeError("Unexpected Keychain value")
    return value


def call(
    path: str,
    payload: dict[str, Any] | None = None,
    *,
    authenticated: bool = True,
    timeout: int = 180,
) -> tuple[int, dict[str, Any]]:
    headers = {"Content-Type": "application/json"}
    if authenticated:
        headers["Authorization"] = f"Bearer {api_key()}"
    req = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=None if payload is None else json.dumps(payload).encode(),
        headers=headers,
        method="GET" if payload is None else "POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            body = json.loads(response.read().decode())
            return response.status, body
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode(errors="replace")
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            body = {"detail": raw}
        return exc.code, body


def chat_payload(content: str, **extra: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": MODEL,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_tokens": 128,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    payload.update(extra)
    return payload


def stream_exact() -> dict[str, Any]:
    payload = chat_payload("Reply with exactly STREAM_OK", stream=True, max_tokens=16)
    req = urllib.request.Request(
        f"{BASE_URL}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {api_key()}", "Content-Type": "application/json"},
        method="POST",
    )
    pieces: list[str] = []
    events = 0
    with urllib.request.urlopen(req, timeout=180) as response:
        for raw in response:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            events += 1
            event = json.loads(data)
            choices = event.get("choices") or []
            if choices:
                pieces.append((choices[0].get("delta") or {}).get("content") or "")
    output = "".join(pieces)
    return {"passed": output == "STREAM_OK", "output": output, "sse_events": events}


def cancel_stream() -> dict[str, Any]:
    payload = chat_payload(
        "Write a long numbered reliability checklist with at least 200 items.",
        stream=True,
        max_tokens=512,
    )
    req = urllib.request.Request(
        f"{BASE_URL}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {api_key()}", "Content-Type": "application/json"},
        method="POST",
    )
    received_content = False
    response = urllib.request.urlopen(req, timeout=180)
    try:
        for raw in response:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            event = json.loads(line[5:].strip())
            choices = event.get("choices") or []
            if choices and (choices[0].get("delta") or {}).get("content"):
                received_content = True
                break
    finally:
        response.close()
    time.sleep(1.0)
    health_status, health = call("/health")
    return {
        "passed": received_content and health_status == 200 and health.get("status") == "healthy",
        "received_before_cancel": received_content,
        "health_after_cancel": health.get("status"),
    }


def main() -> None:
    report: dict[str, Any] = {"generated_at": datetime.now(timezone.utc).isoformat(), "tests": {}}

    unauth_status, _ = call("/v1/models", authenticated=False)
    auth_status, models = call("/v1/models")
    listed = [item.get("id") for item in models.get("data", [])]
    report["tests"]["authentication"] = {
        "passed": unauth_status == 401 and auth_status == 200 and MODEL in listed,
        "unauthenticated_status": unauth_status,
        "authenticated_status": auth_status,
        "target_listed": MODEL in listed,
    }

    json_schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "validation",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "status": {"type": "string", "enum": ["ok"]},
                    "count": {"type": "integer", "enum": [3]},
                },
                "required": ["status", "count"],
                "additionalProperties": False,
            },
        },
    }
    status, body = call(
        "/v1/chat/completions",
        chat_payload("Return an object with status exactly ok and count exactly 3.", response_format=json_schema),
    )
    content = ((body.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        parsed = None
    report["tests"]["structured_json"] = {
        "passed": status == 200 and parsed == {"status": "ok", "count": 3},
        "parsed": parsed,
    }

    tools = [
        {
            "type": "function",
            "function": {
                "name": "add_numbers",
                "description": "Add two integers",
                "parameters": {
                    "type": "object",
                    "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                    "required": ["a", "b"],
                    "additionalProperties": False,
                },
            },
        }
    ]
    first_payload = chat_payload("What is 19 plus 23? You must use the supplied tool.", tools=tools, tool_choice="required")
    first_status, first = call("/v1/chat/completions", first_payload)
    message = ((first.get("choices") or [{}])[0].get("message") or {})
    tool_calls = message.get("tool_calls") or []
    arguments: dict[str, Any] | None = None
    if tool_calls:
        arguments = json.loads(tool_calls[0]["function"]["arguments"])
    roundtrip_messages = first_payload["messages"] + [message]
    if tool_calls:
        roundtrip_messages.append(
            {"role": "tool", "tool_call_id": tool_calls[0]["id"], "content": json.dumps({"sum": 42})}
        )
        roundtrip_messages.append(
            {"role": "user", "content": "Reply with exactly TOOL_ROUNDTRIP_OK after using the tool result."}
        )
    second_payload = chat_payload("unused", tools=tools, tool_choice="none", max_tokens=32)
    second_payload["messages"] = roundtrip_messages
    second_status, second = call("/v1/chat/completions", second_payload)
    final_text = (((second.get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip()
    report["tests"]["tool_roundtrip"] = {
        "passed": (
            first_status == 200
            and arguments == {"a": 19, "b": 23}
            and second_status == 200
            and final_text == "TOOL_ROUNDTRIP_OK"
        ),
        "arguments": arguments,
        "final_text": final_text,
    }

    report["tests"]["streaming"] = stream_exact()
    report["tests"]["cancellation"] = cancel_stream()

    status, body = call(
        "/v1/chat/completions",
        chat_payload("Reply with exactly TARGET_ONLY_OK", max_tokens=16),
    )
    exact = (((body.get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip()
    report["tests"]["post_cancel_completion"] = {"passed": status == 200 and exact == "TARGET_ONLY_OK", "output": exact}

    report["passed"] = all(test.get("passed") is True for test in report["tests"].values())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    os.chmod(OUT, 0o600)
    print(json.dumps({"passed": report["passed"], "tests": {k: v["passed"] for k, v in report["tests"].items()}}, indent=2))


if __name__ == "__main__":
    main()
