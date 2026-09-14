"""Production-shaped CPU DSA benchmark for Artemis."""

from __future__ import annotations

import gc
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(HERE))

import torch  # noqa: E402

from sglang.srt.runtime_context import publish  # noqa: E402
from sglang.srt.server_args import ServerArgs  # noqa: E402

publish(ServerArgs(model_path="dummy"), role="test")

import sgl_kernel  # noqa: F401, E402
from goldens import (  # noqa: E402
    indexer_topk_fp32,
    minimum_row_overlap,
    output_errors,
    sparse_mla_latent_fp32,
)
from synthetic import (  # noqa: E402
    GLM52,
    build_indexer,
    dense_mla_latent_torch,
    make_batches,
    make_weights,
    projected_indexer_inputs,
)
from sglang.srt.layers.attention.dsa.dsa_cpu import (  # noqa: E402
    indexer_topk_torch,
    sparse_mla_attention_cpu,
    sparse_mla_latent_torch,
)

WARMUP = int(os.environ.get("DSA_BENCH_WARMUP", "1"))
ITERS = int(os.environ.get("DSA_BENCH_ITERS", "3"))


def log(message: str):
    print(message, file=sys.stderr, flush=True)


def timed(function) -> float:
    samples: list[float] = []
    for iteration in range(WARMUP + ITERS):
        started = time.perf_counter()
        function()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if iteration >= WARMUP:
            samples.append(elapsed_ms)
    return statistics.median(samples)


def pytorch_dsa_layer(indexer, batch):
    q, keys, gates = projected_indexer_inputs(indexer, batch)
    indices = indexer_topk_torch(
        q, keys, gates, batch.positions, GLM52.index_topk
    )
    return sparse_mla_latent_torch(
        batch.q_abs,
        batch.q_pe,
        batch.c_kv,
        batch.k_pe,
        indices,
        GLM52.softmax_scale,
    )


def cpp_dsa_layer(indexer, batch):
    indices = indexer(
        batch.x, batch.q_lora, batch.positions, batch.cpu_batch(), 0
    )
    return sparse_mla_attention_cpu(
        batch.q_abs,
        batch.q_pe,
        batch.c_kv,
        batch.k_pe,
        indices,
        GLM52.softmax_scale,
    )


def production_dense_extend(batch):
    """Call SGLang's registered dense C++ absorbed-MLA prefill kernel."""
    query = torch.cat((batch.q_abs, batch.q_pe), dim=-1).contiguous()
    kv = torch.cat((batch.c_kv, batch.k_pe), dim=-1).unsqueeze(1).contiguous()
    v_buffer = kv[..., : GLM52.kv_lora_rank]
    query_tokens = batch.q_abs.shape[0]
    k_extend = kv[-query_tokens:].contiguous()
    v_extend = batch.c_kv[-query_tokens:].unsqueeze(1).contiguous()
    output = torch.empty_like(batch.q_abs)
    kv_tokens = batch.c_kv.shape[0]
    req_to_token = torch.arange(kv_tokens, dtype=torch.int32).unsqueeze(0)
    torch.ops.sgl_kernel.extend_attention_cpu(
        query,
        k_extend,
        v_extend,
        output,
        kv,
        v_buffer,
        1.0,
        1.0,
        req_to_token,
        torch.zeros(1, dtype=torch.int64),
        torch.tensor([kv_tokens], dtype=torch.int64),
        torch.tensor([query_tokens], dtype=torch.int32),
        torch.zeros(1, dtype=torch.int32),
        query_tokens,
        GLM52.softmax_scale,
        0.0,
        False,
        0,
        None,
        None,
        None,
        True,
    )
    return output


def production_dense_decode(batches):
    """Call SGLang's registered dense C++ absorbed-MLA decode kernel."""
    batch_size = len(batches)
    kv_tokens = batches[0].c_kv.shape[0]
    q_abs = torch.cat([batch.q_abs for batch in batches], dim=0)
    q_pe = torch.cat([batch.q_pe for batch in batches], dim=0)
    query = torch.cat((q_abs, q_pe), dim=-1).contiguous()
    kv_parts = [
        torch.cat((batch.c_kv, batch.k_pe), dim=-1) for batch in batches
    ]
    kv = torch.cat(kv_parts, dim=0).unsqueeze(1).contiguous()
    v_buffer = kv[..., : GLM52.kv_lora_rank]
    new_key = torch.stack([part[-1] for part in kv_parts]).unsqueeze(1)
    new_value = torch.stack([batch.c_kv[-1] for batch in batches]).unsqueeze(1)
    output = torch.empty_like(q_abs)
    req_to_token = torch.arange(
        batch_size * kv_tokens, dtype=torch.int32
    ).view(batch_size, kv_tokens)
    torch.ops.sgl_kernel.decode_attention_cpu(
        query,
        kv,
        v_buffer,
        1.0,
        1.0,
        output,
        new_key,
        new_value,
        torch.arange(batch_size, dtype=torch.int64) * kv_tokens + kv_tokens - 1,
        torch.empty(
            batch_size,
            GLM52.num_attention_heads,
            16,
            GLM52.kv_lora_rank + 1,
            dtype=torch.float32,
        ),
        req_to_token,
        torch.arange(batch_size, dtype=torch.int64),
        torch.full((batch_size,), kv_tokens, dtype=torch.int64),
        GLM52.softmax_scale,
        0.0,
        False,
        0,
        None,
        None,
    )
    return output


def cpp_correctness() -> int:
    generator = torch.Generator().manual_seed(17)
    tokens, kv_tokens, topk = 4, 64, 16
    q = torch.randn(
        tokens,
        GLM52.index_n_heads,
        GLM52.index_head_dim,
        generator=generator,
        dtype=GLM52.dtype,
    )
    keys = torch.randn(
        kv_tokens,
        GLM52.index_head_dim,
        generator=generator,
        dtype=GLM52.dtype,
    )
    gates = torch.randn(
        tokens, GLM52.index_n_heads, generator=generator
    )
    positions = torch.arange(kv_tokens - tokens, kv_tokens)
    q_abs = torch.randn(
        tokens,
        GLM52.num_attention_heads,
        GLM52.kv_lora_rank,
        generator=generator,
        dtype=GLM52.dtype,
    )
    q_pe = torch.randn(
        tokens,
        GLM52.num_attention_heads,
        GLM52.qk_rope_head_dim,
        generator=generator,
        dtype=GLM52.dtype,
    )
    c_kv = torch.randn(
        kv_tokens,
        GLM52.kv_lora_rank,
        generator=generator,
        dtype=GLM52.dtype,
    )
    k_pe = torch.randn(
        kv_tokens,
        GLM52.qk_rope_head_dim,
        generator=generator,
        dtype=GLM52.dtype,
    )
    expected_indices = indexer_topk_fp32(q, keys, gates, positions, topk)
    try:
        actual_indices = torch.ops.sgl_kernel.dsa_indexer_topk_cpu(
            q, keys, gates, positions, topk
        )
        actual_output = sparse_mla_attention_cpu(
            q_abs,
            q_pe,
            c_kv,
            k_pe,
            actual_indices,
            GLM52.softmax_scale,
        )
    except (AttributeError, NotImplementedError, RuntimeError) as exc:
        log(f"C++ DSA unavailable: {exc}")
        return 0
    expected_output = sparse_mla_latent_fp32(
        q_abs,
        q_pe,
        c_kv,
        k_pe,
        expected_indices,
        GLM52.softmax_scale,
    )
    maximum_absolute, relative_l2 = output_errors(actual_output, expected_output)
    return int(
        minimum_row_overlap(actual_indices, expected_indices) >= 0.97
        and maximum_absolute <= 0.15
        and relative_l2 <= 0.03
    )


def benchmark_point(name: str, indexer):
    batches = make_batches(name, GLM52, indexer, seed=29)
    is_decode = name.startswith("decode")

    torch_ms = timed(
        lambda: [pytorch_dsa_layer(indexer, batch) for batch in batches]
    )
    if is_decode:
        dense_ms = timed(lambda: production_dense_decode(batches))
    else:
        dense_ms = timed(lambda: production_dense_extend(batches[0]))
    cpp_ms = timed(
        lambda: [cpp_dsa_layer(indexer, batch) for batch in batches]
    )
    log(
        f"{name}: cpp={cpp_ms:.3f}ms torch={torch_ms:.3f}ms "
        f"dense={dense_ms:.3f}ms"
    )
    del batches
    gc.collect()
    return cpp_ms, torch_ms, dense_ms


def main():
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "16")))
    output_path = Path.cwd() / "artemis_results.json"
    output_path.unlink(missing_ok=True)
    weights = make_weights(GLM52)
    indexer = build_indexer(GLM52, weights)
    cpp_available = bool(cpp_correctness())
    if not cpp_available:
        raise RuntimeError("C++ DSA correctness gate failed")
    results: dict[str, float | int] = {"cpp_dsa_correct": int(cpp_available)}

    for point in (
        "prefill_1k",
        "prefill_2k",
        "prefill_4k",
        "prefill_8k",
        "decode_8k_b1",
        "decode_8k_b4",
        "decode_8k_b16",
    ):
        cpp_ms, torch_ms, dense_ms = benchmark_point(point, indexer)
        if not math.isfinite(torch_ms) or torch_ms <= 0.0:
            raise RuntimeError(f"invalid PyTorch DSA timing for {point}: {torch_ms}")
        if not math.isfinite(dense_ms) or dense_ms <= 0.0:
            raise RuntimeError(f"invalid dense C++ MLA timing for {point}: {dense_ms}")
        results[f"cpp_{point}_ms"] = cpp_ms
        results[f"torch_{point}_ms"] = torch_ms
        results[f"dense_{point}_ms"] = dense_ms
        results[f"cpp_over_torch_{point}"] = cpp_ms / torch_ms
        results[f"cpp_over_dense_{point}"] = cpp_ms / dense_ms

    temporary = output_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(results, indent=2) + "\n")
    temporary.replace(output_path)
    log(json.dumps(results, sort_keys=True))


if __name__ == "__main__":
    main()
