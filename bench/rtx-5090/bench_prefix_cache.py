#!/usr/bin/env python3
"""Measure cold versus repeated-prefix TTFT on a vLLM profile."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

from bench_profiles import DASHBOARD, OUT, apply_profile, build_long_prompt, http, stream_completion


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="baseline", choices=("baseline", "mtp", "alt-baseline", "alt-mtp"))
    parser.add_argument("--context", type=int, default=32768)
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    apply_result = apply_profile(args.profile, args.context, timeout=1200)
    if apply_result.get("status") in {"failed", "timeout"}:
        return 2
    status = http(f"{DASHBOARD}/api/status", timeout=60)
    model = (status.get("model", {}).get("model") or {}).get("id")
    if not model:
        return 3
    messages = [{"role": "user", "content": build_long_prompt(16000)}]
    cold = stream_completion(messages, max_tokens=32, thinking=False, temperature=0.0, model=model)
    warm = stream_completion(messages, max_tokens=32, thinking=False, temperature=0.0, model=model)
    result = {
        "profile": args.profile,
        "context": args.context,
        "cold": cold,
        "warm": warm,
        "ttft_improvement_pct": round((cold["ttft_seconds"] - warm["ttft_seconds"]) / cold["ttft_seconds"] * 100, 1)
        if cold.get("ttft_seconds") and warm.get("ttft_seconds") else None,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"prefix-cache-{args.profile}-{stamp}.json"
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps({"path": str(path), "cold_ttft": cold.get("ttft_seconds"), "warm_ttft": warm.get("ttft_seconds"), "improvement_pct": result["ttft_improvement_pct"]}, indent=2))
    return 0 if cold.get("ok") and warm.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
