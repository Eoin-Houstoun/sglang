# CPU chunk_kda benchmark

Measures KDA prefill (`chunk_kda`) on CPU at Kimi-Linear-48B-A3B's shape (32 heads,
key and value dim 128, bf16), so a kernel for `sgl-kernel-cpu` can be built, checked and
timed on every change.

The entry point is `python/sglang/kernels/ops/attention/fla/kda_cpu.py`, which SGLang's
`kda_triton.py` calls for prefill on CPU. It starts as a plain PyTorch token loop.

## Setup

Nothing to run by hand. On first use the Build command runs `setup_env.sh`, which builds
`~/.artemis/sglang-cpu-venv` (set `SGLANG_CPU_VENV` to move it) with SGLang's CPU dependencies,
the CPU `sgl_kernel`, CMake >= 3.26 and pytest. That takes a few minutes once; afterwards the
check costs about a second. The machine needs a C/C++ toolchain, `cmake`, `ninja`, `libnuma-dev`
and `libtbb-dev`, and network access on that first run; the script names anything missing.

## Commands

| | |
|---|---|
| Build | `bash benchmark/kernels/linear_attention/kda_cpu/artemis_compile.sh` |
| Test | `bash benchmark/kernels/linear_attention/kda_cpu/artemis_test.sh` |
| Benchmark | `bash benchmark/kernels/linear_attention/kda_cpu/artemis_benchmark.sh` |

- **Build** rebuilds the CPU `sgl_kernel` incrementally (any new `.cpp` in
  `python/sglang/kernels/aot/csrc/cpu/` is picked up) and checks the entry point imports.
- **Test** compares output and the updated state pool with a float64 reference, checks
  untouched pool rows, covers tails, odd dims, ragged batches, bf16/fp16 and the in-kernel
  gate activations. A skipped test fails the command.
- **Benchmark** calls the entry point the way Kimi Linear prefill does (raw gate and beta,
  activated in the kernel with `A_log` and `dt_bias`) at 512, 2k, 8k and a ragged 4x512 batch.
  It times the entry point, the frozen PyTorch fallback and `torch.compile` of it, refuses to
  report if the entry point disagrees with the fallback, and writes `artemis_results.json`.
  The headline metric is `kda_prefill_2k_ms` (lower is better); edit `POINTS` in
  `bench_kda_cpu.py` for other shapes, e.g. fewer heads per rank under tensor parallelism.

Each point is measured in `KDA_BENCH_REPEATS` (default 3) interleaved rounds of warmup plus the median of
`KDA_BENCH_ITERS` (default 5) calls; results report the mean and, as `*_ms_std`, the spread across rounds.
`OMP_NUM_THREADS` defaults to 16. Set it to the cores you want measured.
