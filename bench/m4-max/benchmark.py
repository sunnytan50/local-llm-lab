#!/usr/bin/env python3
"""Repeatable target-only vs DFlash2 benchmark for the local oMLX service."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


# Everything machine-specific comes from the environment; the defaults match a stock oMLX install.
BASE_URL = os.environ.get("OMLX_BASE_URL", "http://127.0.0.1:8013")
MODEL_ID = os.environ.get("OMLX_MODEL_ID", "qwen38-27b-mtp-4bit")
OMLX_HOME = Path(os.environ.get("OMLX_HOME", Path.home() / ".omlx"))
MODEL_PATH = Path(os.environ.get("OMLX_MODEL_PATH", OMLX_HOME / "models" / MODEL_ID))
OUTPUT_DIR = Path(os.environ.get("BENCH_OUTPUT_DIR", Path(__file__).resolve().parent / "output"))
SERVER_LOG = OMLX_HOME / "logs" / "server.log"
KEYCHAIN_ACCOUNT = os.environ.get("OMLX_KEYCHAIN_ACCOUNT", os.environ.get("USER", ""))
KEYCHAIN_SERVICE = os.environ.get("OMLX_KEYCHAIN_SERVICE", "oMLX API Key 127.0.0.1:8013")
CONTEXTS = (4096, 16384, 32768)

BASE_DOCUMENT = (
    "Apple Silicon uses a unified memory architecture in which CPU, GPU, and "
    "accelerators share a physical memory pool. Reliable local inference depends "
    "on bounded concurrency, measured context growth, deterministic recovery, and "
    "clear separation between model-serving engines. Prefix reuse can reduce prompt "
    "work, while speculative decoding can reduce serial decode steps when the draft "
    "and target agree. Every performance claim must be tied to observed output parity. "
)

TASKS = {
    4096: (
        "\nUsing the supplied technical context, write a precise implementation note "
        "covering isolation, deterministic validation, and rollback. Continue until "
        "the output limit; do not use a thinking preamble."
    ),
    16384: (
        "\nSynthesize the supplied document into an engineering review that identifies "
        "the performance hypothesis, the correctness invariant, and the memory safety "
        "gate. Continue until the output limit; do not use a thinking preamble."
    ),
    32768: (
        "\nProduce a detailed reliability analysis of the supplied long document. Explain "
        "how to validate fallback across an acceleration context threshold while "
        "preserving deterministic output. Continue until the output limit; do not use "
        "a thinking preamble."
    ),
}


def keychain_api_key() -> str:
    if os.environ.get("OMLX_API_KEY"):  # set this, or store the key in the macOS Keychain
        return os.environ["OMLX_API_KEY"]
    result = subprocess.run(
        [
            "/usr/bin/security",
            "find-generic-password",
            "-a",
            KEYCHAIN_ACCOUNT,
            "-s",
            KEYCHAIN_SERVICE,
            "-w",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    key = result.stdout.strip()
    if not key.startswith("sk-omlx-") or len(key) < 32:
        raise RuntimeError("Keychain returned an unexpected oMLX API key")
    return key


def request_json(path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=data,
        headers={
            "Authorization": f"Bearer {keychain_api_key()}",
            "Content-Type": "application/json",
        },
        method="GET" if data is None else "POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=900) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {path}: {body}") from exc


def chat_token_count(tokenizer: Any, content: str) -> int:
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if hasattr(encoded, "get"):
        ids = encoded.get("input_ids")
        if ids is not None:
            return len(ids[0]) if ids and isinstance(ids[0], list) else len(ids)
    if hasattr(encoded, "encodings") and encoded.encodings:
        return len(encoded.encodings[0].ids)
    return len(encoded)


def build_prompt(tokenizer: Any, target_tokens: int) -> tuple[str, int]:
    task = TASKS[target_tokens]
    base_ids = tokenizer.encode(BASE_DOCUMENT, add_special_tokens=False)
    task_ids = tokenizer.encode(task, add_special_tokens=False)
    task_only_tokens = chat_token_count(tokenizer, task)
    template_overhead = max(task_only_tokens - len(task_ids), 0)
    filler_budget = max(target_tokens - template_overhead - len(task_ids), 1)
    repeats = (filler_budget // len(base_ids)) + 2
    filler_ids = (base_ids * repeats)[:filler_budget]
    content = tokenizer.decode(filler_ids, skip_special_tokens=True) + task

    # Token boundaries can merge after decode. Nudge toward the requested size.
    for _ in range(8):
        actual = chat_token_count(tokenizer, content)
        delta = target_tokens - actual
        if abs(delta) <= 2:
            return content, actual
        if delta > 0:
            content = ("context " * delta) + content
        else:
            content = content[abs(delta) * 8 :]
    return content, chat_token_count(tokenizer, content)


def server_pid() -> int | None:
    result = subprocess.run(
        ["/usr/sbin/lsof", "-t", "-iTCP:8013", "-sTCP:LISTEN"],
        check=False,
        capture_output=True,
        text=True,
    )
    for line in result.stdout.splitlines():
        if line.strip().isdigit():
            return int(line.strip())
    return None


def rss_bytes(pid: int | None) -> int:
    if pid is None:
        return 0
    result = subprocess.run(
        ["/bin/ps", "-o", "rss=", "-p", str(pid)],
        check=False,
        capture_output=True,
        text=True,
    )
    value = result.stdout.strip()
    return int(value) * 1024 if value.isdigit() else 0


def system_memory_snapshot() -> dict[str, Any]:
    pressure = subprocess.run(
        ["/usr/bin/memory_pressure", "-Q"],
        check=False,
        capture_output=True,
        text=True,
    ).stdout
    match = re.search(r"free percentage:\s*(\d+)%", pressure)
    swap = subprocess.run(
        ["/usr/sbin/sysctl", "-n", "vm.swapusage"],
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "free_percentage": int(match.group(1)) if match else None,
        "swapusage": swap,
    }


class MemorySampler:
    def __init__(self, pid: int | None) -> None:
        self.pid = pid
        self.peak_rss_bytes = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.peak_rss_bytes = max(self.peak_rss_bytes, rss_bytes(self.pid))
            self._stop.wait(0.25)

    def __enter__(self) -> MemorySampler:
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self._thread.join(timeout=2)
        self.peak_rss_bytes = max(self.peak_rss_bytes, rss_bytes(self.pid))


def new_dflash_metrics(log_offset: int) -> list[dict[str, Any]]:
    if not SERVER_LOG.exists():
        return []
    with SERVER_LOG.open("r", encoding="utf-8", errors="replace") as handle:
        handle.seek(log_offset)
        lines = handle.readlines()
    metrics: list[dict[str, Any]] = []
    pattern = re.compile(
        r"DFlash(?:2)? generation complete: (?P<tokens>\d+) tokens, "
        r"(?P<tps>[0-9.]+) tok/s, acceptance=(?P<acceptance>[0-9.]+)%, "
        r"cycles=(?P<cycles>\d+)"
    )
    for line in lines:
        match = pattern.search(line)
        if match:
            row: dict[str, Any] = match.groupdict()
            row.update(
                tokens=int(row["tokens"]),
                tok_per_s=float(row["tps"]),
                acceptance_percent=float(row["acceptance"]),
                cycles=int(row["cycles"]),
            )
            row.pop("tps", None)
            row.pop("acceptance", None)
            metrics.append(row)
    return metrics


def current_dflash_stats() -> dict[str, Any] | None:
    """Return only the target's DFlash counters, never the admin API key."""
    try:
        stats = request_json("/admin/api/stats")
    except RuntimeError as exc:
        # The managed desktop runtime protects admin routes with a session
        # cookie even when the OpenAI API bearer key is valid.
        if "HTTP 401" in str(exc):
            return None
        raise
    active_models = stats.get("active_models") or {}
    for model in active_models.get("models") or []:
        if model.get("id") == MODEL_ID:
            return model.get("dflash")
    return None


def stream_completion(prompt: str, max_tokens: int = 128) -> dict[str, Any]:
    payload = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "seed": 42,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = urllib.request.Request(
        f"{BASE_URL}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {keychain_api_key()}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    started = time.perf_counter()
    first_token_at: float | None = None
    chunks: list[str] = []
    usage: dict[str, Any] = {}
    finish_reason: str | None = None
    pid = server_pid()
    log_offset = SERVER_LOG.stat().st_size if SERVER_LOG.exists() else 0
    memory_before = system_memory_snapshot()

    try:
        with MemorySampler(pid) as sampler:
            with urllib.request.urlopen(request, timeout=900) as response:
                for raw_line in response:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    event = json.loads(data)
                    if event.get("usage"):
                        usage = event["usage"]
                    choices = event.get("choices") or []
                    if not choices:
                        continue
                    choice = choices[0]
                    delta = choice.get("delta") or {}
                    piece = delta.get("content") or ""
                    if piece:
                        if first_token_at is None:
                            first_token_at = time.perf_counter()
                        chunks.append(piece)
                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]
        peak_rss = sampler.peak_rss_bytes
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from chat completion: {body}") from exc

    finished = time.perf_counter()
    output = "".join(chunks)
    completion_tokens = usage.get("completion_tokens") or usage.get("output_tokens") or 0
    decode_seconds = max(finished - (first_token_at or started), 1e-9)
    return {
        "output": output,
        "output_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
        "finish_reason": finish_reason,
        "usage": usage,
        "ttft_seconds": None if first_token_at is None else first_token_at - started,
        "wall_seconds": finished - started,
        "observed_decode_tok_per_s": completion_tokens / decode_seconds,
        "peak_server_rss_bytes": peak_rss,
        "memory_before": memory_before,
        "memory_after": system_memory_snapshot(),
        "dflash_metrics": new_dflash_metrics(log_offset),
        "dflash_stats": current_dflash_stats(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("baseline", "dflash", "dflash_quant"))
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument(
        "--contexts",
        type=int,
        nargs="+",
        choices=CONTEXTS,
        default=list(CONTEXTS),
    )
    args = parser.parse_args()

    health = request_json("/health")
    if health.get("status") != "healthy":
        raise RuntimeError(f"oMLX is not healthy: {health}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=False)
    rows: list[dict[str, Any]] = []
    for target_tokens in args.contexts:
        prompt, actual_tokens = build_prompt(tokenizer, target_tokens)
        result = stream_completion(prompt, max_tokens=args.max_tokens)
        result["target_context_tokens"] = target_tokens
        result["tokenizer_context_tokens"] = actual_tokens
        rows.append(result)
        print(
            f"{args.phase} {target_tokens}: "
            f"prompt={result['usage'].get('prompt_tokens')} "
            f"output={result['usage'].get('completion_tokens')} "
            f"ttft={result['ttft_seconds']:.3f}s "
            f"decode={result['observed_decode_tok_per_s']:.2f} tok/s "
            f"sha={result['output_sha256'][:12]}",
            flush=True,
        )

    report: dict[str, Any] = {
        "phase": args.phase,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "base_url": BASE_URL,
        "model": MODEL_ID,
        "max_tokens": args.max_tokens,
        "rows": rows,
    }

    if args.phase.startswith("dflash"):
        baseline_path = OUTPUT_DIR / "baseline.json"
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        baseline_by_context = {
            row["target_context_tokens"]: row for row in baseline["rows"]
        }
        for row in rows:
            old = baseline_by_context[row["target_context_tokens"]]
            row["parity_with_baseline"] = row["output_sha256"] == old["output_sha256"]
            old_rate = old.get("observed_decode_tok_per_s") or 0
            row["speedup_vs_baseline"] = (
                row["observed_decode_tok_per_s"] / old_rate if old_rate else None
            )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{args.phase}.json"
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    os.chmod(output_path, 0o600)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
