"""DSA CPU layer benchmark for Artemis: writes artemis_results.json (numeric only) to the
working directory. Diagnostics go to stderr.

Run from the repository root:
    SGLANG_USE_CPU_ENGINE=1 PYTHONPATH=$PWD/python python benchmark/kernels/attention/dsa_cpu/bench_dsa_cpu.py
"""

import json
import os
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
REPO = HERE.parents[3]

import torch  # noqa: E402

from sglang.srt.runtime_context import publish  # noqa: E402
from sglang.srt.server_args import ServerArgs  # noqa: E402

publish(ServerArgs(model_path="dummy"), role="test")

import sglang  # noqa: E402
from sglang.srt.layers.attention.dsa.dsa_cpu import sparse_mla_attention_cpu  # noqa: E402

import goldens  # noqa: E402
from synthetic import GLM52_SMALL, build_indexer, dense_mla_attention, make_batches, make_weights  # noqa: E402

assert Path(sglang.__file__).resolve().is_relative_to(REPO), f"sglang imported from {sglang.__file__}, not this checkout"

WARMUP, ITERS = 2, 5
CFG = GLM52_SMALL


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def dsa_attention(indexer, weights, batch):
    """One attention layer forward: sparse DSA when implemented, else the dense fallback.

    Returns (out [T, H, v_head], topk_indices [T, index_topk] or None).
    """
    try:
        topk = indexer(batch.x, batch.q_lora, batch.positions, batch.cpu_batch(), 0)
        out = sparse_mla_attention_cpu(batch.q_nope, batch.q_pe, batch.c_kv, batch.k_pe, topk, weights.W_UK, weights.W_UV, CFG.softmax_scale)
        return out, topk
    except NotImplementedError:
        out = dense_mla_attention(batch.q_nope, batch.q_pe, batch.c_kv, batch.k_pe, weights.W_UK, weights.W_UV, CFG.softmax_scale, batch.causal_offset)
        return out, None


def sparse_available(indexer, weights) -> bool:
    """Capability probe: True when the indexer and sparse attention both run on a tiny input."""
    g = torch.Generator().manual_seed(1)
    n = 16
    rand = lambda *shape: torch.randn(*shape, generator=g).to(CFG.dtype)  # noqa: E731
    from synthetic import DSABatch

    b = DSABatch(
        name="probe",
        x=rand(n, CFG.hidden_size),
        q_lora=rand(n, CFG.q_lora_rank),
        positions=torch.arange(n),
        q_nope=rand(n, CFG.num_attention_heads, CFG.qk_nope_head_dim),
        q_pe=rand(n, CFG.num_attention_heads, CFG.qk_rope_head_dim),
        c_kv=rand(n, CFG.kv_lora_rank),
        k_pe=rand(n, CFG.qk_rope_head_dim),
        index_k_cache=torch.empty(0, CFG.index_head_dim, dtype=CFG.dtype),
        causal_offset=0,
    )
    _, topk = dsa_attention(indexer, weights, b)
    return topk is not None


def timed_point(name, indexer, weights):
    times = []
    for i in range(WARMUP + ITERS):
        batches = make_batches(name, CFG, indexer, seed=1000 + i)
        t0 = time.perf_counter()
        for b in batches:
            dsa_attention(indexer, weights, b)
        times.append((time.perf_counter() - t0) * 1000)
    med = statistics.median(times[WARMUP:])
    log(f"{name}: {med:.2f} ms (median of {ITERS}: {[round(t, 1) for t in times[WARMUP:]]})")
    return med


def timed_indexer_8k(indexer):
    times = []
    for i in range(WARMUP + ITERS):
        b = make_batches("prefill_8k", CFG, indexer, seed=2000 + i)[0]
        t0 = time.perf_counter()
        indexer(b.x, b.q_lora, b.positions, b.cpu_batch(), 0)
        times.append((time.perf_counter() - t0) * 1000)
    med = statistics.median(times[WARMUP:])
    log(f"indexer at 8k: {med:.2f} ms")
    return med


def sparse_correct(indexer, weights) -> int:
    if not sparse_available(indexer, weights):
        log("sparse path not implemented: dsa_sparse_correct = 0")
        return 0
    problems = []
    for name in goldens.POINTS:
        outs, topks = [], []
        for b in make_batches(name, CFG, indexer):
            out, topk = dsa_attention(indexer, weights, b)
            outs.append(out)
            topks.append(topk)
        problems += goldens.check_sparse_point(name, torch.cat(outs), None if topks[0] is None else torch.cat(topks))
    for p in problems:
        log(f"golden check: {p}")
    log(f"dsa_sparse_correct = {0 if problems else 1}")
    return 0 if problems else 1


def main():
    torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
    out_path = Path.cwd() / "artemis_results.json"
    out_path.unlink(missing_ok=True)
    weights = make_weights(CFG)
    indexer = build_indexer(CFG, weights)
    with torch.no_grad():
        results = {"dsa_sparse_correct": sparse_correct(indexer, weights)}
        results["prefill_ms_4k"] = timed_point("prefill_4k", indexer, weights)
        results["prefill_ms_8k"] = timed_point("prefill_8k", indexer, weights)
        results["decode_ms_8k"] = timed_point("decode_8k", indexer, weights)
        results["indexer_ms_8k"] = timed_indexer_8k(indexer) if results["dsa_sparse_correct"] else 0.0
    # Intel's target configuration: 8k tokens in, 1k tokens out. Time to first token plus 1,000 decode steps.
    results["gen_ms_8k_in_1k_out"] = results["prefill_ms_8k"] + 1000 * results["decode_ms_8k"]
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log(json.dumps(results))


if __name__ == "__main__":
    main()
