# local-llm-lab

The serving configs and benchmark scripts behind my local-LLM write-ups. One 27B model family runs on two very different machines: one RTX 5090 (32 GB) and one M4 Max (128 GB). Every run is single-user, at concurrency 1, on my own hardware.

The write-ups these files support:
- Same model, two machines (M4 Max vs RTX 5090): https://suhailmohebi.com/notes/same-model-two-machines/
- SGLang vs vLLM serving configs on an RTX 5090: https://suhailmohebi.com/insights/rtx-5090-llm-serving/
- Speculative decoding measured on an M4 Max: https://suhailmohebi.com/notes/speculative-decoding-on-my-m4-max/
- KV cache precision (why an FP8 KV cache is free and FP4 is not): https://suhailmohebi.com/notes/kv-cache-precision/

## Headline results

### RTX 5090 (32 GB): Qwen3.8-27B NVFP4, FP8 KV cache

| Profile | Engine | Sustained output | VRAM after suite | Mean accepted length |
|---|---|---:|---:|---:|
| D-Flash 2 drafter | SGLang | 139.7 tok/s | 29,150 MiB | 2.95 |
| DSpark v2 drafter | SGLang | 110.2 tok/s | 26,494 MiB | 2.05 |
| Native MTP | vLLM | 100.8 tok/s | 30,293 MiB | 2.29 |
| Baseline | vLLM | 84.5 tok/s | 30,503 MiB | — |

The raw runs behind this table are in `results/rtx-5090/`, from 22 Aug 2026. `summary.md` there lists every profile I tried, including some not in the table.

### M4 Max (128 GB): oMLX, community 4-bit MLX builds

| Configuration | Decode tok/s | Change |
|---|---:|---:|
| Speculation off | 16.7 | baseline |
| + DFlash 2, greedy | 35.5 | +113% |
| + DFlash 2, reasoning lane, thinking on | 47.2 | +183% |

These Mac figures come from the runs described in the speculative-decoding note; their raw logs are not in this repo. What is here is the first DFlash 2 trial on the Mac (20 Aug 2026), in `results/m4-max-2026-08-20/`, with its own report.

Reading the prompt (prefill) is where the machines differ most: about 9,081 tok/s on the 5090 against about 196 tok/s on the Mac, roughly 46×. A warm prefix cache cuts the time to first token by about 90% on both.

## What's in here

| Path | What it is |
|---|---|
| `configs/rtx-5090/sglang-dflash2.sh` | SGLang + D-Flash 2 drafter (32K / 64K / 112K) |
| `configs/rtx-5090/sglang-dspark-v2.sh` | SGLang + DSpark v2 NVFP4 drafter (32K / 64K / 112K) |
| `configs/rtx-5090/vllm-native-mtp.sh` | vLLM with the model's native MTP (32K) |
| `configs/rtx-5090/vllm-baseline-long.sh` | vLLM, no speculation, prefix cache (32K up to 262K) |
| `configs/m4-max/omlx-model-settings.json` | oMLX model settings for the two Mac builds (target-only with MTP, and DFlash 2) |
| `bench/rtx-5090/bench_profiles.py` | The 5090 suite: TTFT, decode, 12K recall, tool call, vision, stability, acceptance |
| `bench/rtx-5090/bench_prefix_cache.py` | Cold vs repeated-prefix TTFT on a vLLM profile (imports `bench_profiles.py`) |
| `bench/m4-max/benchmark.py` | Target-only vs DFlash 2 on oMLX at 4K, 16K and 32K context |
| `bench/m4-max/validation.py` | Functional checks for the local, authenticated oMLX service |
| `bench/long-context-256k/` | A 256k-context tool-calling and reasoning bench: a ~200k-token dossier with planted needles, bugs to find and a decoy, a grader, and 17 deterministic tests that never call a model |
| `results/` | Raw JSON from the runs above |

Model and profile names in the configs and results are neutral labels for the community 4-bit builds I tested. The launch scripts are my current versions (last changed 1 Oct 2026); some flags were tuned after the 22 Aug runs in the results table. Paths such as `/opt/llm/models/hot/...` and `/opt/llm/venvs/...` are my layout, so change them to yours. The servers bind to `127.0.0.1` only.

## Running the benchmarks

The scripts read their machine-specific settings from environment variables:

| Variable | Used by | Default |
|---|---|---|
| `BENCH_BASE_URL` | `bench_profiles.py` | `http://127.0.0.1:8002` |
| `BENCH_DASHBOARD_URL` | `bench_profiles.py` | `http://127.0.0.1:7319` (my local dashboard; see below) |
| `BENCH_SSH_HOST`, `BENCH_SERVER_LOG` | `bench_profiles.py` (reads speculative acceptance from the server log) | `llm-box`, `/opt/llm/console/vector-model.log` |
| `OMLX_BASE_URL`, `OMLX_MODEL_ID`, `OMLX_HOME` | the Mac scripts | `http://127.0.0.1:8013`, the MTP build, `~/.omlx` |
| `OMLX_API_KEY` | the Mac scripts | read from the macOS Keychain if unset |
| `BENCH_OUTPUT_DIR` | all | an `output/` folder next to the script |

`bench_profiles.py` is the harness as I ran it. It switches profiles and reads status and VRAM through my local dashboard's API, so treat it as a reference: `run_suite()` is the reusable part, and it only needs an OpenAI-compatible endpoint.

```bash
# the 256k bench's own tests (no model needed)
cd bench/long-context-256k && python3 -m unittest test_hermes_256k_bench
```

## Method, in short

- Concurrency 1, one user.
- RTX 5090 decode is matched-run throughput. Mac decode is completion tokens ÷ (wall time − TTFT), as a mean of 3 runs.
- Context, KV cache and drafter settings are in each config.
- The two machines run the same model family and the same drafter family, but not identical weights or runtimes. The two-machines note has the comparison table.

## Caveats

These are one person's measurements on one machine of each kind. Re-measure on your own hardware, and open an issue if your numbers disagree; that's the most useful thing this repo can get.

## Licence

Scripts and configs: MIT (see `LICENSE`). Measurements and text: CC BY 4.0, with a link back to the source note.
