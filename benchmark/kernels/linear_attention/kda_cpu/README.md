# CPU chunk_kda benchmark

Measures KDA prefill (`chunk_kda`) on CPU at Kimi-Linear-48B-A3B's shape (32 heads,
key and value dim 128, bf16), so a kernel for `sgl-kernel-cpu` can be built, checked and
timed on every change.

The entry point is `python/sglang/kernels/ops/attention/fla/kda_cpu.py`, which SGLang's
`kda_triton.py` calls for prefill on CPU. It starts as a plain PyTorch token loop.

## Setup, once per machine

As the user the runner runs as:

```
sh benchmark/kernels/linear_attention/kda_cpu/setup_env.sh
```

It builds `~/.artemis/sglang-cpu-venv` (set `SGLANG_CPU_VENV` to move it), with SGLang's CPU
dependencies, the CPU `sgl_kernel` and CMake >= 3.26. About ten minutes; later runs are a no-op.

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
- **Benchmark** times the entry point, the frozen PyTorch fallback and `torch.compile` of it,
  refuses to report if the entry point disagrees with the fallback, and writes
  `artemis_results.json`. The metric to optimise is `kda_prefill_2k_ms` (lower is better).

`OMP_NUM_THREADS` defaults to 16. Set it to the cores you want measured.
