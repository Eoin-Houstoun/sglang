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
import sgl_kernel  # noqa: E402,F401  registers the torch.ops.sgl_kernel CPU ops
from reference import make_inputs, pytorch_fallback, pytorch_step  # noqa: E402

from sglang.kernels.ops.attention.fla.kda_cpu import chunk_kda  # noqa: E402

WARMUP = int(os.environ.get("KDA_BENCH_WARMUP", "1"))
ITERS = int(os.environ.get("KDA_BENCH_ITERS", "5"))
# Independent rounds per point, interleaved across implementations; results report their mean and spread.
REPEATS = int(os.environ.get("KDA_BENCH_REPEATS", "3"))
TOLERANCE = 3e-2

# name -> sequence lengths; 32 heads, key/value dim 128, bf16, called the way Kimi Linear
# calls extend(): raw gate and beta, activated in the kernel with A_log and dt_bias.
POINTS = {
    "prefill_2k": [2048],
    "prefill_8k": [8192],
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
    inputs = make_inputs(lengths, raw_gate=True, seed=1)

    expected_state = inputs.state.clone()
    expected = pytorch_fallback(inputs, expected_state)
    actual_state = inputs.state.clone()
    actual = chunk_kda(**inputs.kwargs(actual_state))
    error = max(
        (actual.float() - expected.float()).abs().max().item(),
        (actual_state - expected_state).abs().max().item(),
    )
    if not math.isfinite(error) or error > TOLERANCE:
        raise RuntimeError(
            f"{name}: entry point differs from the PyTorch fallback by {error}"
        )

    # Each call mutates the pool in place, so every call gets a fresh copy; it is tiny next to KDA.
    implementations = {
        "kda": lambda: chunk_kda(**inputs.kwargs(inputs.state.clone())),
        "pytorch": lambda: pytorch_fallback(inputs, inputs.state.clone()),
        "compile": lambda: pytorch_fallback(
            inputs, inputs.state.clone(), compiled_step
        ),
    }
    rounds = {key: [] for key in implementations}
    for _ in range(REPEATS):
        for key, function in implementations.items():
            rounds[key].append(timed(function))
    log(
        f"{name}: "
        + ", ".join(
            f"{key} {statistics.mean(ms):.1f} +/- {_spread(ms):.1f} ms"
            for key, ms in rounds.items()
        )
        + f", max error {error:.2e}"
    )
    return rounds, error


def _spread(samples) -> float:
    return statistics.stdev(samples) if len(samples) > 1 else 0.0


def main():
    threads = int(
        os.environ.get("OMP_NUM_THREADS", str(max(1, (os.cpu_count() or 2) // 2)))
    )
    torch.set_num_threads(threads)
    output_path = Path.cwd() / "artemis_results.json"
    output_path.unlink(missing_ok=True)
    compiled_step = torch.compile(pytorch_step, dynamic=False)
    log(
        f"threads {threads}, torch {torch.__version__}, "
        f"capability {torch.backends.cpu.get_cpu_capability()}"
    )

    results = {}
    worst_error = 0.0
    for name, lengths in POINTS.items():
        rounds, error = benchmark_point(name, lengths, compiled_step)
        mean = {key: statistics.mean(ms) for key, ms in rounds.items()}
        for key, ms in rounds.items():
            results[f"{key}_{name}_ms"] = mean[key]
            results[f"{key}_{name}_ms_std"] = _spread(ms)
        results[f"speedup_vs_pytorch_{name}"] = mean["pytorch"] / mean["kda"]
        results[f"speedup_vs_compile_{name}"] = mean["compile"] / mean["kda"]
        worst_error = max(worst_error, error)
    results["kda_max_abs_error"] = worst_error
    results["kda_correct"] = 1
    results["bench_repeats"] = REPEATS

    temporary = output_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(results, indent=2) + "\n")
    temporary.replace(output_path)
    log(f"primary metric kda_{PRIMARY}_ms = {results[f'kda_{PRIMARY}_ms']:.1f}")
    log(json.dumps(results, sort_keys=True))


if __name__ == "__main__":
    main()
