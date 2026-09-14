# Artemis example: GLM-5.2 DSA on CPU

This branch is a working starting point for using Artemis Discovery to optimize
DeepSeek Sparse Attention (DSA) in SGLang on an Intel Xeon CPU. Unlike a stub
or dense fallback, the baseline already runs both registered C++ DSA
operations, passes an independent FP32 correctness gate, and reports real
latencies.

Repository:
`https://github.com/turintech/sglang/tree/cpu-dsa-glm52-ratio-discovery`

## Scope

The harness uses GLM-5.2 dimensions:

- hidden size 6144 and query LoRA rank 2048;
- 64 attention heads and 32 indexer heads;
- indexer head dimension 128;
- latent KV dimension 512 and RoPE dimension 64;
- top-k 2048 and page size 64;
- BF16 model tensors with FP32 reference calculations.

The measured DSA path includes SGLang's indexer projections, rotary embedding,
causal top-k, and sparse latent MLA. It returns `[tokens, 64, 512]` latent
output. It does not yet exercise the complete serving backend, production DSA
KV pool, final output projection, or a full GLM-5.2 model.

The dense comparison starts from already projected MLA tensors. It is useful
as an algorithm-level target, but it is not an identical kernel entry-point
comparison: DSA pays for its additional indexer while dense MLA does not.

## Prerequisites

Use a Linux x86-64 machine with a recent GCC/G++, CMake, Ninja, libnuma, TBB,
and network access to PyPI and the PyTorch CPU wheel index. On Debian or Ubuntu:

```bash
sudo apt-get install -y build-essential cmake ninja-build libnuma-dev libtbb-dev
```

Clone the example branch:

```bash
git clone --branch cpu-dsa-glm52-ratio-discovery \
  https://github.com/turintech/sglang.git
cd sglang
```

## Build once

Create the default environment at
`$HOME/.artemis/sglang-cpu-venv`:

```bash
sh benchmark/kernels/attention/dsa_cpu/setup_env.sh
```

To choose another environment path or rebuild it:

```bash
sh benchmark/kernels/attention/dsa_cpu/setup_env.sh /path/to/venv
sh benchmark/kernels/attention/dsa_cpu/setup_env.sh --force
sh benchmark/kernels/attention/dsa_cpu/setup_env.sh /path/to/venv --force
```

Initial setup takes roughly ten minutes on a 32-core Xeon. Candidate builds
reuse `$HOME/.artemis/sglang-dsa-cpu-cache`.

## Test and benchmark

Run these commands from the repository root:

```bash
benchmark/kernels/attention/dsa_cpu/artemis_compile.sh
benchmark/kernels/attention/dsa_cpu/artemis_test.sh
benchmark/kernels/attention/dsa_cpu/artemis_benchmark.sh
```

On the reference C4 runner, incremental compilation takes about one minute,
tests take about five seconds, and the benchmark takes two to three minutes.
The test command currently runs five correctness tests.

The benchmark performs one warmup and reports the median of three iterations.
Override this only when running manually:

```bash
DSA_BENCH_WARMUP=2 DSA_BENCH_ITERS=5 \
  benchmark/kernels/attention/dsa_cpu/artemis_benchmark.sh
```

Results are written atomically to `artemis_results.json`.

## Metrics

For prefill lengths 1k, 2k, 4k, and 8k, and 8k decode batches 1, 4, and 16,
the result contains:

- `cpp_*_ms`: the candidate C++ DSA latency;
- `torch_*_ms`: the unchanged PyTorch DSA reference latency;
- `dense_*_ms`: the unchanged SGLang dense C++ MLA reference latency;
- `cpp_over_torch_*`: `cpp_ms / torch_ms`;
- `cpp_over_dense_*`: `cpp_ms / dense_ms`;
- `cpp_dsa_correct`: `1` only when the C++ path passes the FP32 gate.

All timings and ratios are lower-is-better. A ratio below `1.0` means C++ DSA
beat that reference. Reference timings are remeasured for each version and
contain system noise, so select winners using raw C++ timings first and ratios
as supporting evidence.

One baseline run on an Emerald Rapids Xeon Platinum 8581C measured:

- prefill C++ DSA: 110, 356, 964, and 2647 ms at 1k through 8k;
- C++ DSA over PyTorch DSA: 0.032, 0.051, 0.124, and 0.268;
- C++ DSA over dense C++ MLA: 4.98, 8.40, 5.28, and 3.87;
- decode C++ DSA: 2.81, 10.36, and 39.64 ms at batches 1, 4, and 16.

These are representative measurements, not portable performance claims.
The reference host is Emerald Rapids, not Xeon 6.

## Run with Artemis Discovery

Register an Artemis runner on the target machine and add a Git credential that
can read `turintech/sglang`. Then import this branch:

```bash
artemis project import \
  --git-url https://github.com/turintech/sglang.git \
  --key-id <git-key-id> \
  --name sglang-glm52-cpu-dsa-example \
  --branch cpu-dsa-glm52-ratio-discovery
```

Record the returned project UUID, then configure the runner and commands:

```bash
artemis project runner set \
  --project <project-id> --runner <runner-name>

artemis project commands set --project <project-id> \
  --compile "benchmark/kernels/attention/dsa_cpu/artemis_compile.sh" \
  --test "benchmark/kernels/attention/dsa_cpu/artemis_test.sh" \
  --benchmark "benchmark/kernels/attention/dsa_cpu/artemis_benchmark.sh"
```

Launch a constrained Discovery:

```bash
artemis discovery create \
  --project <project-id> \
  --runner <runner-name> \
  --model <model-id> \
  --versions 12 \
  --mode automatic \
  --target-files python/sglang/kernels/aot/csrc/cpu/dsa.cpp \
  --compile-cmd "benchmark/kernels/attention/dsa_cpu/artemis_compile.sh" \
  --test-cmd "benchmark/kernels/attention/dsa_cpu/artemis_test.sh" \
  --benchmark-cmd "benchmark/kernels/attention/dsa_cpu/artemis_benchmark.sh" \
  --task "Optimize the correct GLM-5.2 CPU DSA C++ kernels. Preserve the FP32 correctness gate and operation schemas. Minimize raw cpp prefill/decode latency and cpp_over_torch/cpp_over_dense ratios. Do not modify tests, references, synthetic geometry, registration, or benchmark scripts."
```

## Discovery contract

Discovery may modify only:

- `python/sglang/kernels/aot/csrc/cpu/dsa.cpp`.

It must preserve:

- operation schemas and registration;
- FP32 oracles and correctness tests;
- synthetic GLM-5.2 geometry;
- benchmark points, metric definitions, and command scripts.
