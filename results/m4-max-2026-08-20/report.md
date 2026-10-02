# Qwen3.8-27B + DFlash2 on oMLX — implementation report

Generated: 2026-08-20 (Asia/Dubai)

## Verdict

The signed oMLX 0.6.3rc1 desktop app is installed and running as a managed, localhost-only service at `http://127.0.0.1:8013/v1`. The Qwen3.8-27B MTP 4-bit target is accepted for text inference. DFlash2 is downloaded, paired, parity-validated, and retained, but **disabled** because neither the unquantized nor draft-only 4-bit profile met the specification's 15% performance gate.

The active production-safe state is therefore **target-only BatchedEngine**, maximum concurrency 1, Safe memory guard, 32K initial context, 8K maximum output, global cache off, and API-key authentication on.

## Installed custody

- App: `/Applications/oMLX.app`, version 0.6.3rc1, notarized Developer ID build.
- Installer: `~/Downloads/oMLX-0.6.3rc1-macos26-27.dmg`.
- Installer SHA-256: `263099b656cfe9f79337f6054a276c3a1a89c26ac8ebca7b864a7aaabff11dc0`.
- Target source: `~/.lmstudio/models/community/Qwen3.8-27B-MTP-4bit`.
- oMLX target import: `~/.omlx/models/qwen38-27b-mtp-4bit` (read-only symlink; source model files were not edited or copied).
- Draft: `z-lab/Qwen3.8-27B-DFlash2`, pinned revision `50307d4c4cde6860d4eee73e2547cd786fe8e8a4`.
- Draft model-file SHA-256: `67fc76d68dc5a9415511a4f394ef744d67510cd20e93b37cc2cc7d28e4bab65c`.

## Security and persistence

- API key authentication is enabled; unauthenticated `/v1/models` returns HTTP 401.
- The key is stored in owner-only `~/.omlx/settings.json` (mode 600).
- A second copy is stored in macOS Keychain under service `oMLX API Key 127.0.0.1:8013`, account `<your macOS user>`.
- The key is intentionally absent from every exported report file.
- The desktop-managed server successfully restarted and returned exact output `FINAL_RESTART_OK` after the final rollback.

## Final settings

| Setting | Value |
|---|---:|
| Bind address | 127.0.0.1:8013 |
| Concurrency | 1 |
| Memory guard | Safe |
| Initial context window | 32,768 |
| Maximum output tokens | 8,192 |
| Temperature / top-p / top-k | 1.0 / 0.95 / 20 |
| Global oMLX cache | Disabled |
| DFlash private L1 cache | Config retained; inactive while DFlash is disabled |
| DFlash maximum context | 32,768 |
| DFlash enabled | No |

## Benchmark result

Decode throughput is observed completion throughput. All runs used the same tokenizer-calibrated prompts, temperature 0, seed 42, thinking disabled, 128 output tokens, and exact SHA-256 output comparison.

| Context | Target-only tok/s | DFlash tok/s (delta) | Acceptance | Q4 draft DFlash tok/s (delta) | Acceptance | Exact parity |
|---:|---:|---:|---:|---:|---:|---:|
| 4,096 | 16.15 | 14.93 (-7.5%) | 57.8% | 16.39 (+1.5%) | 57.8% | Yes |
| 16,384 | 9.35 | 10.20 (+9.1%) | 62.5% | 11.79 (+26.1%) | 61.7% | Yes |

At 32K, the configured threshold correctly switched to the target-only fallback: 15.55 tok/s versus the 14.72 tok/s baseline with exact output parity. The unquantized DFlash profile's 4K/16K median decode gain was only about 0.8%. The valid draft-Q4 profile's 4K/16K median decode gain was about 13.8%, and it was slower end to end at both contexts (32.82s versus 32.04s at 4K; 176.13s versus 158.70s at 16K). Average accepted-draft rate remained healthy at roughly 60%, so the rollback is a performance decision, not a correctness failure.

The machine was intentionally kept in its real multi-service state. Safe-memory prefill throttling was observed while LM Studio and other services remained active; no existing service was terminated for a cleaner benchmark.

## Functional validation

| Test | Result |
|---|---|
| Authentication | Pass |
| Structured Json | Pass |
| Tool Roundtrip | Pass |
| Streaming | Pass |
| Cancellation | Pass |
| Post Cancel Completion | Pass |

Structured output used a strict JSON schema. Tool validation included a parsed function call and a tool-result round trip. Streaming, cancellation, health recovery, and post-cancel generation all passed.

## RC1 observations

- The native first-run form briefly retained a stale Port validation value. Re-entering the field and relaunching the app completed desktop management; subsequent managed start/restart passed.
- The model's metadata makes auto-discovery try the VLM loader first. Its tensor layout is not accepted by RC1's VLM path, but oMLX safely falls back to the text LLM loader and all required text/API tests pass. Vision was not accepted in this deployment.
- The model-settings dialog cleared the draft path and DFlash threshold while the experimental draft-quantization toggle was changed. The path and threshold were restored through the admin API before the valid quantized benchmark. Final exported settings preserve both fields.

## Rollback and next validation stage

The current rollback is already applied: `dflash_enabled=false`; draft files and pairing metadata remain available. After a newer oMLX build or DFlash implementation is installed, rerun `benchmark.py` at 4K and 16K before enabling DFlash. Only then stage 65K, 131K, and 262K context tests with the same parity, cancellation, and memory-pressure gates.

## Evidence files

- `baseline.json` — target-only 4K/16K/32K benchmark.
- `dflash.json` — unquantized DFlash and 32K fallback benchmark.
- `dflash_quant.json` — valid draft-only Q4 DFlash benchmark.
- `validation.json` — authentication, structured output, tool, streaming, cancellation, and recovery results.
- `settings.redacted.json` — export-safe global settings.
- `model_settings.export.json` — final model settings, including disabled DFlash state.
- `manifest.json` — version, revision, checksum, custody, and artifact hashes.
