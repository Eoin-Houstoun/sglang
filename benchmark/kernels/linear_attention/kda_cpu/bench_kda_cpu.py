"""Benchmark CPU chunk_kda at Kimi Linear shapes and write artemis_results.json.

Times the SGLang entry point (whatever implementation it currently has)
against the frozen PyTorch fallback and torch.compile of that fallback.
Fails if the entry point disagrees with the fallback, so a fast but wrong
version never gets a number.
"""

import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(__file__))
from reference import make_inputs, pytorch_fallback, pytorch_step  # noqa: E402

import sgl_kernel  # noqa: E402,F401  registers the torch.ops.sgl_kernel CPU ops
from sglang.kernels.ops.attention.fla.kda_cpu import chunk_kda  # noqa: E402

WARMUP = int(os.environ.get("KDA_BENCH_WARMUP", "1"))
ITERS = int(os.environ.get("KDA_BENCH_ITERS", "5"))
TOLERANCE = 3e-2

# name -> sequence lengths; 32 heads, key/value dim 128, bf16.
POINTS = {
    "prefill_2k": [2048],
    "prefill_512": [512],
    "ragged_4x512": [512, 512, 512, 512],
}
PRIMARY = "prefill_2k"


def log(message: str):
    print(message, file=sys.stderr, flush=True)


def timed(function) -> float:
    samples = []
    for iteration in range(WARMUP + ITERS):
        started = time.perf_counter()
        function()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if iteration >= WARMUP:
            samples.append(elapsed_ms)
    return statistics.median(samples)


@torch.inference_mode()
def benchmark_point(name, lengths, compiled_step):
    inputs = make_inputs(lengths, seed=1)

    expected_state = inputs.state.clone()
    expected = pytorch_fallback(inputs, expected_state)
    actual_state = inputs.state.clone()
    actual = chunk_kda(**inputs.kwargs(actual_state))
    error = max(
        (actual.float() - expected.float()).abs().max().item(),
        (actual_state - expected_state).abs().max().item(),
    )
    if not math.isfinite(error) or error > TOLERANCE:
        raise RuntimeError(f"{name}: entry point differs from the PyTorch fallback by {error}")

    # Each call mutates the pool in place, so every call gets a fresh copy; it is tiny next to KDA.
    kernel_ms = timed(lambda: chunk_kda(**inputs.kwargs(inputs.state.clone())))
    pytorch_ms = timed(lambda: pytorch_fallback(inputs, inputs.state.clone()))
    compile_ms = timed(lambda: pytorch_fallback(inputs, inputs.state.clone(), compiled_step))
    log(f"{name}: kernel {kernel_ms:.1f} ms, pytorch {pytorch_ms:.1f} ms, "
        f"torch.compile {compile_ms:.1f} ms, max error {error:.2e}")
    return kernel_ms, pytorch_ms, compile_ms, error


def main():
    threads = int(os.environ.get("OMP_NUM_THREADS", str(max(1, (os.cpu_count() or 2) // 2))))
    torch.set_num_threads(threads)
    output_path = Path.cwd() / "artemis_results.json"
    output_path.unlink(missing_ok=True)
    compiled_step = torch.compile(pytorch_step, dynamic=False)
    log(f"threads {threads}, torch {torch.__version__}, "
        f"capability {torch.backends.cpu.get_cpu_capability()}")

    results = {}
    worst_error = 0.0
    for name, lengths in POINTS.items():
        kernel_ms, pytorch_ms, compile_ms, error = benchmark_point(name, lengths, compiled_step)
        results[f"kda_{name}_ms"] = kernel_ms
        results[f"pytorch_{name}_ms"] = pytorch_ms
        results[f"compile_{name}_ms"] = compile_ms
        results[f"speedup_vs_pytorch_{name}"] = pytorch_ms / kernel_ms
        results[f"speedup_vs_compile_{name}"] = compile_ms / kernel_ms
        worst_error = max(worst_error, error)
    results["kda_max_abs_error"] = worst_error
    results["kda_correct"] = 1

    temporary = output_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(results, indent=2) + "\n")
    temporary.replace(output_path)
    log(f"primary metric kda_{PRIMARY}_ms = {results[f'kda_{PRIMARY}_ms']:.1f}")
    log(json.dumps(results, sort_keys=True))


if __name__ == "__main__":
    main()
