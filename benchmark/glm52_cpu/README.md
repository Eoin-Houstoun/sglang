# GLM-5.2 on CPU: end-to-end serving benchmark for Artemis

Launches the model through this checkout's SGLang server on CPU, gates on GSM8K, measures the
1k to 8k in, 1k out workload with `sglang.bench_serving`, and writes `artemis_results.json`.
Discovery's edits to the checkout are what the server runs, because SGLang is imported from
the checkout, not from an installed package.

## Commands

```bash
sh benchmark/glm52_cpu/run_e2e.sh compile   # import check: the DSA CPU module and the server entry point
sh benchmark/glm52_cpu/run_e2e.sh test      # the layer-level DSA gate (seconds), before the expensive load
sh benchmark/glm52_cpu/run_e2e.sh bench     # launch, GSM8K gate, serving benchmark, artemis_results.json
```

These are the three Artemis project commands. Build the CPU environment first with
`sh benchmark/kernels/attention/dsa_cpu/setup_env.sh` (see that README); these commands stop
with a message if it is not there.

Unlike the layer benchmark, this one keeps a wrapper: it layers the committed defaults in
`config.env` under the machine-local `$HOME/.artemis/glm52_cpu.env` under the environment,
and preloads the allocator libraries SGLang's CPU docs use for serving.

## Configuration

`config.env` holds the committed defaults: `zai-org/GLM-5.2-FP8`, one TP rank per sub-NUMA
cluster (`--tp 6`), the docs' launch flags, 200 GSM8K questions, inputs of 1k and 8k tokens
with 1k output, four prompts at concurrency four. Put machine-specific values in
`$HOME/.artemis/glm52_cpu.env` on the runner host (a local model path, a different TP,
`SGLANG_CPU_OMP_THREADS_BIND` for core pinning, `--max-total-tokens` via
`SGLANG_E2E_SERVER_ARGS`); environment variables override both.

Set `SGLANG_E2E_GSM8K_MIN_ACC` from a CUDA run of the same model with the same scorer
(5-shot, greedy, last number), minus the run-to-run noise of about three points at 200
questions. A version below it fails.

## Metrics (`artemis_results.json`)

| Metric | Better | Meaning |
|---|---|---|
| `gsm8k_ok` | 1 | 1 when GSM8K accuracy reaches the gate; the benchmark exits non-zero otherwise |
| `gsm8k_accuracy` | higher | 5-shot GSM8K accuracy on the configured subset |
| `server_load_s` | lower | seconds from launch to healthy |
| `ttft_ms_1k_in`, `ttft_ms_8k_in` | lower | median time to first token |
| `tpot_ms_1k_in`, `tpot_ms_8k_in` | lower | median time per output token |
| `e2e_s_1k_in`, `e2e_s_8k_in` | lower | median request latency for the configured output length, the target workload |
| `output_tps_1k_in`, `output_tps_8k_in` | higher | output tokens per second across the concurrent prompts |

## Memory

The CPU engine reserves most of the free memory for the KV cache. Bound it with
`--max-total-tokens` in `SGLANG_E2E_SERVER_ARGS` when sharing a machine, and start the Artemis
runner with `--ram-limit-mb` above weights plus cache: the runner's memory monitor kills the
benchmark otherwise ("Memory limit exceeded").

## Cost per Discovery version

One server launch per version: weight load plus warm-up, then about a thousand generated
tokens per GSM8K question and the two serving points. For a 700 GB model expect tens of
minutes per version, so size the version budget accordingly and keep the layer benchmark for
fast iteration on the attention path.

## Checked before handover

The flow was exercised on a single-socket Xeon with a small model (`Qwen/Qwen2.5-0.5B-Instruct`,
TP 1, 40 questions, 256 output tokens) through the same commands: launch from the checkout,
GSM8K gate, both serving points, JSON written, server torn down. GLM-5.2 itself and TP=6 were
not run here; the first run on the target machine sets the GSM8K gate and the load timeout.
