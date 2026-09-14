"""Deterministic GLM-5.2-shaped inputs for CPU DSA validation."""

from __future__ import annotations

import msgspec
import torch

from sglang.srt.layers.attention.dsa.dsa_cpu import (
    DSACPUBatch,
    index_head_gates_cpu,
    index_k_cpu,
    index_q_cpu,
)
from sglang.srt.layers.attention.dsa.dsa_indexer import Indexer


class DSAConfig(msgspec.Struct, frozen=True):
    hidden_size: int = 6144
    q_lora_rank: int = 2048
    num_attention_heads: int = 64
    kv_lora_rank: int = 512
    qk_rope_head_dim: int = 64
    index_n_heads: int = 32
    index_head_dim: int = 128
    index_topk: int = 2048
    page_size: int = 64
    rope_theta: float = 8_000_000.0
    max_position_embeddings: int = 131_072
    dtype: torch.dtype = torch.bfloat16

    @property
    def softmax_scale(self) -> float:
        return (self.kv_lora_rank + self.qk_rope_head_dim) ** -0.5


GLM52 = DSAConfig()
WEIGHT_SEED = 20260914


class DSAWeights(msgspec.Struct):
    wq_b: torch.Tensor
    wk: torch.Tensor
    weights_proj: torch.Tensor
    k_norm_weight: torch.Tensor
    k_norm_bias: torch.Tensor


class DSABatch(msgspec.Struct):
    name: str
    x: torch.Tensor
    q_lora: torch.Tensor
    positions: torch.Tensor
    q_abs: torch.Tensor
    q_pe: torch.Tensor
    c_kv: torch.Tensor
    k_pe: torch.Tensor
    index_k_cache: torch.Tensor

    def cpu_batch(self) -> DSACPUBatch:
        return DSACPUBatch(index_k_cache=self.index_k_cache)


OPERATING_POINTS = {
    "prefill_1k": (1024, 1024, 1),
    "prefill_2k": (2048, 2048, 1),
    "prefill_4k": (4096, 4096, 1),
    "prefill_8k": (8192, 8192, 1),
    "decode_8k_b1": (1, 8192, 1),
    "decode_8k_b4": (1, 8192, 4),
    "decode_8k_b16": (1, 8192, 16),
}


def make_weights(config: DSAConfig = GLM52, seed: int = WEIGHT_SEED) -> DSAWeights:
    generator = torch.Generator().manual_seed(seed)

    def linear(out_features: int, in_features: int) -> torch.Tensor:
        return (
            torch.randn(out_features, in_features, generator=generator)
            * in_features**-0.5
        ).to(config.dtype)

    return DSAWeights(
        wq_b=linear(config.index_n_heads * config.index_head_dim, config.q_lora_rank),
        wk=linear(config.index_head_dim, config.hidden_size),
        weights_proj=linear(config.index_n_heads, config.hidden_size),
        k_norm_weight=torch.ones(config.index_head_dim, dtype=config.dtype),
        k_norm_bias=torch.zeros(config.index_head_dim, dtype=config.dtype),
    )


def build_indexer(
    config: DSAConfig = GLM52, weights: DSAWeights | None = None
) -> Indexer:
    weights = weights or make_weights(config)
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(config.dtype)
    try:
        indexer = Indexer(
            hidden_size=config.hidden_size,
            index_n_heads=config.index_n_heads,
            index_head_dim=config.index_head_dim,
            rope_head_dim=config.qk_rope_head_dim,
            index_topk=config.index_topk,
            q_lora_rank=config.q_lora_rank,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=config.rope_theta,
            layer_id=0,
            scale_fmt=None,
            block_size=128,
            rope_scaling=None,
            is_neox_style=False,
            prefix="indexer",
            quant_config=None,
            alt_stream=None,
            config=None,
        )
    finally:
        torch.set_default_dtype(previous_dtype)

    with torch.no_grad():
        indexer.wq_b.weight.copy_(weights.wq_b)
        indexer.wk.weight.copy_(weights.wk)
        indexer.weights_proj.weight.copy_(weights.weights_proj)
        indexer.k_norm.weight.copy_(
            weights.k_norm_weight.to(indexer.k_norm.weight.dtype)
        )
        indexer.k_norm.bias.copy_(
            weights.k_norm_bias.to(indexer.k_norm.bias.dtype)
        )
    indexer.eval()
    return indexer


def make_batches(
    name: str,
    config: DSAConfig,
    indexer: Indexer,
    seed: int = 0,
) -> list[DSABatch]:
    query_tokens, kv_tokens, batch_size = OPERATING_POINTS[name]
    batches: list[DSABatch] = []
    point_seed = list(OPERATING_POINTS).index(name)
    for sequence in range(batch_size):
        generator = torch.Generator().manual_seed(
            seed * 100_003 + point_seed * 101 + sequence
        )
        positions = torch.arange(
            kv_tokens - query_tokens, kv_tokens, dtype=torch.int64
        )
        x = torch.randn(
            query_tokens,
            config.hidden_size,
            generator=generator,
            dtype=config.dtype,
        )
        q_lora = torch.randn(
            query_tokens,
            config.q_lora_rank,
            generator=generator,
            dtype=config.dtype,
        )
        q_abs = torch.randn(
            query_tokens,
            config.num_attention_heads,
            config.kv_lora_rank,
            generator=generator,
            dtype=config.dtype,
        )
        q_pe = torch.randn(
            query_tokens,
            config.num_attention_heads,
            config.qk_rope_head_dim,
            generator=generator,
            dtype=config.dtype,
        )
        c_kv = torch.randn(
            kv_tokens,
            config.kv_lora_rank,
            generator=generator,
            dtype=config.dtype,
        )
        k_pe = torch.randn(
            kv_tokens,
            config.qk_rope_head_dim,
            generator=generator,
            dtype=config.dtype,
        )
        cached_keys = torch.randn(
            kv_tokens - query_tokens,
            config.index_head_dim,
            generator=generator,
            dtype=config.dtype,
        )
        batches.append(
            DSABatch(
                name=name,
                x=x,
                q_lora=q_lora,
                positions=positions,
                q_abs=q_abs,
                q_pe=q_pe,
                c_kv=c_kv,
                k_pe=k_pe,
                index_k_cache=cached_keys,
            )
        )
    return batches


def projected_indexer_inputs(indexer: Indexer, batch: DSABatch):
    q = index_q_cpu(indexer, batch.q_lora, batch.positions)
    new_keys = index_k_cpu(indexer, batch.x, batch.positions)
    keys = torch.cat((batch.index_k_cache, new_keys), dim=0)
    gates = index_head_gates_cpu(indexer, batch.x)
    return q, keys, gates


def dense_mla_latent_torch(
    q_abs: torch.Tensor,
    q_pe: torch.Tensor,
    c_kv: torch.Tensor,
    k_pe: torch.Tensor,
    scale: float,
    chunk_size: int = 16,
) -> torch.Tensor:
    """Dense causal absorbed MLA, used only as a portable diagnostic baseline."""
    token_count = q_abs.shape[0]
    kv_count = c_kv.shape[0]
    offset = kv_count - token_count
    output = torch.empty_like(q_abs)
    key_positions = torch.arange(kv_count)
    for start in range(0, token_count, chunk_size):
        stop = min(token_count, start + chunk_size)
        scores = torch.einsum(
            "thr,sr->ths", q_abs[start:stop], c_kv
        ) + torch.einsum("thd,sd->ths", q_pe[start:stop], k_pe)
        scores = scores.float() * scale
        limits = torch.arange(start + offset, stop + offset)[:, None, None]
        scores.masked_fill_(key_positions[None, None, :] > limits, float("-inf"))
        probabilities = torch.softmax(scores, dim=-1).to(c_kv.dtype)
        output[start:stop] = torch.einsum(
            "ths,sr->thr", probabilities, c_kv
        )
    return output
