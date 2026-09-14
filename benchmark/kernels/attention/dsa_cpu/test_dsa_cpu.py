"""Correctness gates for the GLM-5.2 CPU DSA operation boundary."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from sglang.srt.runtime_context import publish
from sglang.srt.server_args import ServerArgs

publish(ServerArgs(model_path="dummy"), role="test")

import sgl_kernel  # noqa: F401, E402
from goldens import (  # noqa: E402
    indexer_topk_fp32,
    minimum_row_overlap,
    output_errors,
    sparse_mla_latent_fp32,
)
from synthetic import GLM52, build_indexer, make_weights  # noqa: E402
from sglang.srt.layers.attention.dsa.dsa_cpu import (  # noqa: E402
    indexer_topk_torch,
    sparse_mla_attention_cpu,
    sparse_mla_latent_torch,
)


def make_case(tokens: int = 8, kv_tokens: int = 96, topk: int = 32):
    generator = torch.Generator().manual_seed(20260914)
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
        tokens,
        GLM52.index_n_heads,
        generator=generator,
        dtype=torch.float32,
    )
    positions = torch.arange(kv_tokens - tokens, kv_tokens, dtype=torch.int64)
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
    return q, keys, gates, positions, q_abs, q_pe, c_kv, k_pe, expected_indices


def cpp_ops_available(case) -> bool:
    q, keys, gates, positions, q_abs, q_pe, c_kv, k_pe, _ = case
    try:
        indices = torch.ops.sgl_kernel.dsa_indexer_topk_cpu(
            q, keys, gates, positions, 32
        )
        sparse_mla_attention_cpu(
            q_abs, q_pe, c_kv, k_pe, indices, GLM52.softmax_scale
        )
    except (AttributeError, NotImplementedError, RuntimeError):
        return False
    return True


CASE = make_case()
CPP_OPS_AVAILABLE = cpp_ops_available(CASE)


def test_exact_glm52_geometry():
    weights = make_weights()
    indexer = build_indexer(GLM52, weights)
    assert tuple(indexer.wq_b.weight.shape) == (4096, 2048)
    assert tuple(indexer.wk.weight.shape) == (128, 6144)
    assert GLM52.index_topk == 2048
    assert GLM52.page_size == 64


def test_pytorch_indexer_matches_fp32_oracle():
    q, keys, gates, positions, *_, expected = CASE
    actual = indexer_topk_torch(q, keys, gates, positions, 32)
    assert minimum_row_overlap(actual, expected) >= 0.97


def test_pytorch_sparse_mla_matches_fp32_oracle():
    _, _, _, _, q_abs, q_pe, c_kv, k_pe, indices = CASE
    expected = sparse_mla_latent_fp32(
        q_abs, q_pe, c_kv, k_pe, indices, GLM52.softmax_scale
    )
    actual = sparse_mla_latent_torch(
        q_abs, q_pe, c_kv, k_pe, indices, GLM52.softmax_scale
    )
    maximum_absolute, relative_l2 = output_errors(actual, expected)
    assert maximum_absolute <= 0.15
    assert relative_l2 <= 0.03


@pytest.mark.skipif(not CPP_OPS_AVAILABLE, reason="C++ DSA stubs not implemented")
def test_cpp_indexer_matches_fp32_oracle():
    q, keys, gates, positions, *_, expected = CASE
    actual = torch.ops.sgl_kernel.dsa_indexer_topk_cpu(
        q, keys, gates, positions, 32
    )
    assert actual.dtype == torch.int32
    assert minimum_row_overlap(actual, expected) >= 0.97


@pytest.mark.skipif(not CPP_OPS_AVAILABLE, reason="C++ DSA stubs not implemented")
def test_cpp_sparse_mla_matches_fp32_oracle():
    _, _, _, _, q_abs, q_pe, c_kv, k_pe, indices = CASE
    expected = sparse_mla_latent_fp32(
        q_abs, q_pe, c_kv, k_pe, indices, GLM52.softmax_scale
    )
    actual = sparse_mla_attention_cpu(
        q_abs, q_pe, c_kv, k_pe, indices, GLM52.softmax_scale
    )
    maximum_absolute, relative_l2 = output_errors(actual, expected)
    assert maximum_absolute <= 0.15
    assert relative_l2 <= 0.03
