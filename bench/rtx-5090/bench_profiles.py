#!/usr/bin/env python3
"""Apply a runtime profile through the dashboard and run the representative suite.

Usage:
  python3 bench_profiles.py --profile dflash2 [--context 32768] [--skip-apply]

Measurements go directly to the model endpoint (BENCH_BASE_URL) so TTFT and
decode throughput are not skewed by the dashboard. Speculative acceptance is
read from the server log over SSH (BENCH_SSH_HOST, BENCH_SERVER_LOG).
Results are written to output/<profile>-<stamp>.json and appended to
output/summary.md.

This is the harness exactly as I ran it. Profile switching, the served-model
status and the VRAM readings come from my local dashboard's API
(BENCH_DASHBOARD_URL), so treat it as a reference: the reusable part is
run_suite(), which only needs an OpenAI-compatible endpoint.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

DASHBOARD = os.environ.get("BENCH_DASHBOARD_URL", "http://127.0.0.1:7319")
MODEL = os.environ.get("BENCH_BASE_URL", "http://127.0.0.1:8002")
ROOT = Path(__file__).resolve().parent
OUT = Path(os.environ.get("BENCH_OUTPUT_DIR", ROOT / "output"))
REMOTE_LOG = os.environ.get("BENCH_SERVER_LOG", "/opt/llm/console/vector-model.log")
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", os.environ.get("BENCH_SSH_HOST", "llm-box")]


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def http(url: str, payload: dict | None = None, timeout: float = 60) -> dict:
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def wsl(script: str, timeout: float = 60) -> str:
    encoded = base64.b64encode(script.encode()).decode()
    command = [*SSH, f'wsl.exe -d Ubuntu-24.04 -- bash -lc "echo {encoded} | base64 -d | bash"']
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    return result.stdout


def remote_log_lines() -> int:
    try:
        return int(wsl(f"wc -l < {REMOTE_LOG}").strip() or 0)
    except (ValueError, subprocess.TimeoutExpired):
        return 0


def remote_log_since(line: int, limit: int = 4000) -> str:
    try:
        return wsl(f"tail -n +{line + 1} {REMOTE_LOG} | tail -n {limit}")
    except subprocess.TimeoutExpired:
        return ""


def apply_profile(profile: str, context: int, timeout: float = 900) -> dict:
    log(f"Applying profile {profile} @ {context}")
    try:
        started = http(f"{DASHBOARD}/api/profile/apply", {"profile": profile, "context": context}, timeout=120)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise SystemExit(f"apply rejected: HTTP {exc.code} {detail}")
    if started.get("unchanged"):
        log("Profile already active")
        return {"unchanged": True, "phases": [], "elapsed_seconds": 0}
    phases = []
    deadline = time.monotonic() + timeout
    last_phase = None
    while time.monotonic() < deadline:
        job = http(f"{DASHBOARD}/api/profile/apply/status", timeout=30).get("job") or {}
        phase = f"{job.get('phase')}: {job.get('detail')}"
        if phase != last_phase:
            log(f"  {job.get('elapsed_seconds', 0):>6}s  {phase}")
            phases.append({"at": job.get("elapsed_seconds"), "phase": job.get("phase"), "detail": job.get("detail")})
            last_phase = phase
        if job.get("status") in {"done", "failed"}:
            return {
                "status": job.get("status"),
                "phases": phases,
                "elapsed_seconds": job.get("elapsed_seconds"),
                "result": job.get("result"),
                "error": job.get("error"),
            }
        time.sleep(3)
    return {"status": "timeout", "phases": phases, "elapsed_seconds": timeout}


def stream_completion(messages: list, *, max_tokens: int, thinking: bool, tools: list | None = None, temperature: float | None = None, model: str) -> dict:
    mode_temperature = 1.0 if thinking else 0.7
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": mode_temperature if temperature is None else temperature,
        "top_p": 0.95 if thinking else 0.8,
        "top_k": 20,
        "presence_penalty": 0.0 if thinking else 1.5,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": thinking},
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    request = urllib.request.Request(f"{MODEL}/v1/chat/completions", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    first = None
    content = ""
    reasoning = ""
    tool_calls: dict[int, dict] = {}
    usage = {}
    finish = None
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            for raw in response:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                event = json.loads(data)
                usage = event.get("usage") or usage
                choices = event.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                delta = choice.get("delta") or {}
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]
                piece = delta.get("content") or ""
                thought = delta.get("reasoning_content") or delta.get("reasoning") or ""
                for call in delta.get("tool_calls") or []:
                    index = call.get("index", 0)
                    entry = tool_calls.setdefault(index, {"name": "", "arguments": ""})
                    function = call.get("function") or {}
                    entry["name"] += function.get("name") or ""
                    entry["arguments"] += function.get("arguments") or ""
                if piece or thought or delta.get("tool_calls"):
                    if first is None:
                        first = time.perf_counter()
                content += piece
                reasoning += thought
    except urllib.error.HTTPError as exc:
        return {"ok": False, "error": f"HTTP {exc.code}: {exc.read().decode(errors='replace')[:500]}"}
    except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
        return {"ok": False, "error": f"transport: {exc}"}
    ended = time.perf_counter()
    completion = int(usage.get("completion_tokens") or 0)
    decode = (ended - first) if first else 0
    return {
        "ok": True,
        "content": content,
        "reasoning_chars": len(reasoning),
        "tool_calls": [tool_calls[key] for key in sorted(tool_calls)],
        "finish_reason": finish,
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": completion,
        "ttft_seconds": round(first - started, 3) if first else None,
        "elapsed_seconds": round(ended - started, 3),
        "tokens_per_second": round(completion / decode, 1) if decode > 0 and completion > 1 else None,
        "usage_details": {
            key: value
            for key, value in usage.items()
            if key not in {"prompt_tokens", "completion_tokens", "total_tokens"}
        },
    }


def test_image() -> str:
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        raise SystemExit("Pillow is required for the vision probe")
    image = Image.new("RGB", (640, 200), "white")
    draw = ImageDraw.Draw(image)
    draw.text((20, 80), "CODE: ZEBRA-7741", fill="black")
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


LONG_PASSAGE = (
    "The harbour town kept its records in a brick archive beside the lighthouse. Every ledger listed the "
    "tides, the catch, the names of boats, and the weather noted by the keeper at dawn and dusk. "
)
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
                "required": ["city"],
            },
        },
    }
]


def build_long_prompt(target_tokens: int = 12000) -> str:
    chunks = []
    approx = 0
    inserted = False
    while approx < target_tokens * 4:
        chunks.append(LONG_PASSAGE)
        approx += len(LONG_PASSAGE)
        if not inserted and approx > target_tokens * 4 * 0.4:
            chunks.append("Keeper's private note: the access code for the archive vault is 7741-ZEBRA. ")
            inserted = True
    chunks.append("\n\nQuestion: What is the access code for the archive vault? Reply with only the code.")
    return "".join(chunks)


def parse_acceptance(text: str, engine_hint: str) -> dict:
    sglang = [
        (float(length), float(rate), float(tps))
        for length, rate, tps in re.findall(r"accept len: ([0-9.]+), accept rate: ([0-9.]+), cuda graph: \w+, gen throughput \(token/s\): ([0-9.]+)", text)
    ]
    vllm = re.findall(r"SpecDecoding metrics: (.*)", text)
    vllm_rows = []
    for row in vllm:
        numbers = dict(re.findall(r"([A-Za-z ]+?): ([0-9.]+)", row))
        per_position = re.findall(r"Per-position acceptance rate: \[([^\]]*)\]", row)
        vllm_rows.append({"fields": numbers, "per_position": per_position[0] if per_position else None})
    result = {"sglang_samples": len(sglang), "vllm_samples": len(vllm_rows)}
    if sglang:
        result["sglang_mean_accept_len"] = round(statistics.mean(row[0] for row in sglang), 2)
        result["sglang_mean_accept_rate"] = round(statistics.mean(row[1] for row in sglang), 3)
        result["sglang_mean_gen_tps"] = round(statistics.mean(row[2] for row in sglang), 1)
    if vllm_rows:
        result["vllm_last"] = vllm_rows[-1]
        lengths = []
        for row in vllm_rows:
            for key, value in row["fields"].items():
                if "acceptance length" in key.lower():
                    lengths.append(float(value))
        if lengths:
            result["vllm_mean_acceptance_length"] = round(statistics.mean(lengths), 3)
    return result


def run_suite(model: str, served_engine: str) -> dict:
    suite: dict = {}
    log("warmup")
    suite["warmup"] = stream_completion([{"role": "user", "content": "Reply with the single word READY."}], max_tokens=8, thinking=False, model=model)

    log("short prompt")
    suite["short"] = stream_completion(
        [{"role": "user", "content": "Explain in three sentences why the sky is blue."}],
        max_tokens=256, thinking=False, model=model,
    )

    log("long generation (throughput)")
    suite["generation"] = stream_completion(
        [{"role": "user", "content": "Write a vivid 600-word short story about a lighthouse keeper who discovers an old logbook. Use plain prose, no headings."}],
        max_tokens=1024, thinking=False, temperature=0.7, model=model,
    )

    log("thinking prompt")
    suite["thinking"] = stream_completion(
        [{"role": "user", "content": "What is 17 multiplied by 23? Show the final answer as 'Answer: N'."}],
        max_tokens=768, thinking=True, model=model,
    )
    if suite["thinking"].get("ok"):
        suite["thinking"]["answer_correct"] = "391" in suite["thinking"]["content"]

    log("long-context recall (~12K tokens)")
    suite["long_context"] = stream_completion(
        [{"role": "user", "content": build_long_prompt()}], max_tokens=64, thinking=False, model=model,
    )
    if suite["long_context"].get("ok"):
        suite["long_context"]["recall_correct"] = "7741-ZEBRA" in suite["long_context"]["content"].replace(" ", "")

    log("tool call")
    suite["tool_call"] = stream_completion(
        [{"role": "user", "content": "What's the weather in Dubai right now? Use the weather tool."}],
        max_tokens=256, thinking=False, tools=TOOLS, model=model,
    )
    if suite["tool_call"].get("ok"):
        calls = suite["tool_call"]["tool_calls"]
        parsed = None
        if calls:
            try:
                parsed = json.loads(calls[0]["arguments"] or "{}")
            except json.JSONDecodeError:
                parsed = "INVALID JSON"
        suite["tool_call"]["structured_ok"] = bool(calls) and calls[0]["name"] == "get_weather" and isinstance(parsed, dict) and "city" in parsed
        suite["tool_call"]["parsed_arguments"] = parsed

    log("vision")
    suite["vision"] = stream_completion(
        [{"role": "user", "content": [
            {"type": "text", "text": "Read any text in this image and reply with just that text."},
            {"type": "image_url", "image_url": {"url": test_image()}},
        ]}],
        max_tokens=48, thinking=False, model=model,
    )
    if suite["vision"].get("ok"):
        suite["vision"]["ocr_correct"] = "ZEBRA-7741" in suite["vision"]["content"].upper()

    log("quality spot-checks")
    suite["quality"] = {
        "factual": stream_completion([{"role": "user", "content": "Name the capital of Australia and the year it became the capital. One sentence."}], max_tokens=96, thinking=False, model=model),
        "coding": stream_completion([{"role": "user", "content": "Write a Python function is_palindrome(s) that ignores case and non-alphanumeric characters. Return only a fenced python code block."}], max_tokens=300, thinking=False, model=model),
        "format": stream_completion([{"role": "user", "content": "Return a JSON object with keys 'city' (string) and 'population_millions' (number) for Tokyo. Output only the JSON, no prose, no code fence."}], max_tokens=96, thinking=False, model=model),
    }
    fmt = suite["quality"]["format"]
    if fmt.get("ok"):
        try:
            parsed = json.loads(fmt["content"].strip().strip("`"))
            fmt["json_ok"] = isinstance(parsed, dict) and "city" in parsed and "population_millions" in parsed
        except json.JSONDecodeError:
            fmt["json_ok"] = False
    code = suite["quality"]["coding"]
    if code.get("ok"):
        code["has_fence"] = "```" in code["content"] and "def is_palindrome" in code["content"]

    log("stability x5")
    runs = []
    for index in range(5):
        runs.append(stream_completion([{"role": "user", "content": f"Give me one fun fact about the number {index + 11}, in one sentence."}], max_tokens=96, thinking=False, temperature=0.7, model=model))
    ok_runs = [run for run in runs if run.get("ok") and run.get("tokens_per_second")]
    suite["stability"] = {
        "runs": len(runs),
        "ok": len([run for run in runs if run.get("ok")]),
        "errors": [run.get("error") for run in runs if not run.get("ok")],
        "tps_mean": round(statistics.mean(run["tokens_per_second"] for run in ok_runs), 1) if ok_runs else None,
        "tps_stdev": round(statistics.pstdev(run["tokens_per_second"] for run in ok_runs), 1) if len(ok_runs) > 1 else None,
        "ttft_mean": round(statistics.mean(run["ttft_seconds"] for run in ok_runs if run["ttft_seconds"]), 3) if ok_runs else None,
        "empty_outputs": len([run for run in ok_runs if not run["content"].strip()]),
    }
    return suite


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--context", type=int, default=32768)
    parser.add_argument("--skip-apply", action="store_true")
    parser.add_argument("--label", default="")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    record: dict = {"profile": args.profile, "context": args.context, "label": args.label, "started_at": stamp}

    log_start = remote_log_lines()
    if not args.skip_apply:
        record["apply"] = apply_profile(args.profile, args.context)
        if record["apply"].get("status") == "failed" or record["apply"].get("status") == "timeout":
            record["startup_log_tail"] = remote_log_since(log_start)[-12000:]
            path = OUT / f"{args.profile}-{stamp}.json"
            path.write_text(json.dumps(record, indent=2))
            log(f"Profile failed to start; details in {path}")
            return 2

    status = http(f"{DASHBOARD}/api/status", timeout=60)
    record["served_model"] = status["model"].get("model")
    record["runtime"] = status.get("runtime")
    record["vram_after_load_mib"] = (status["machine"].get("gpu") or {}).get("memory_used_mib")
    model_id = (status["model"].get("model") or {}).get("id")
    if not model_id:
        log("Endpoint has no model; aborting")
        return 3
    log(f"Served model: {model_id} ({status.get('runtime')})")
    served_engine = "sglang" if "sglang" in json.dumps(status["model"]) else "vllm"
    suite_start_line = remote_log_lines()
    record["suite"] = run_suite(model_id, served_engine)
    status_after = http(f"{DASHBOARD}/api/status", timeout=60)
    record["vram_after_suite_mib"] = (status_after["machine"].get("gpu") or {}).get("memory_used_mib")
    record["acceptance"] = parse_acceptance(remote_log_since(suite_start_line), served_engine)
    record["finished_at"] = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    path = OUT / f"{args.profile}-{stamp}.json"
    path.write_text(json.dumps(record, indent=2))
    suite = record["suite"]
    short = suite["short"]
    gen = suite["generation"]
    row = (
        f"| {args.profile} | {model_id} | {short.get('ttft_seconds')} | {short.get('tokens_per_second')} | {gen.get('tokens_per_second')} | "
        f"{suite['long_context'].get('ttft_seconds')} / {suite['long_context'].get('recall_correct')} | {suite['tool_call'].get('structured_ok')} | "
        f"{suite['vision'].get('ocr_correct')} | {suite['stability']['ok']}/5 ±{suite['stability']['tps_stdev']} | {record['vram_after_suite_mib']} | "
        f"{record['acceptance'].get('sglang_mean_accept_len') or record['acceptance'].get('vllm_mean_acceptance_length')} |"
    )
    summary = OUT / "summary.md"
    if not summary.exists():
        summary.write_text("| profile | served | short TTFT s | short tok/s | 1K-gen tok/s | 12K TTFT s / recall | tool call | vision | stability | VRAM MiB | accept len |\n|---|---|---|---|---|---|---|---|---|---|---|\n")
    with summary.open("a") as handle:
        handle.write(row + "\n")
    log(f"Wrote {path}")
    print(row)
    return 0


if __name__ == "__main__":
    sys.exit(main())
