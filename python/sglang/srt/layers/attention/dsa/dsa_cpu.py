"""CPU boundary for DeepSeek Sparse Attention.

The production-shaped C++ operations consume already projected indexer tensors and
absorbed MLA queries:

* ``dsa_indexer_topk_cpu(q, keys, gates, positions, topk) -> [T, topk]``
* ``sparse_mla_attention_cpu(q_abs, q_pe, c_kv, k_pe, indices, scale)
  -> [T, H, kv_lora_rank]``

The Python reference functions in this module are deliberately separate from the
registered operations. They are correctness oracles and the secondary PyTorch
performance baseline; Discovery is expected to implement and optimize the C++ ops.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import msgspec
import torch

if TYPE_CHECKING:
    from sglang.srt.layers.attention.dsa.dsa_indexer import Indexer


class DSACPUBatch(msgspec.Struct, frozen=True):
    """What the CPU indexer path reads from the forward batch for one sequence.

    A stand-in for the DSA token-to-KV pool until the CPU pool integration lands:
    `index_k_cache` holds the index keys of the tokens before this step, as produced by
    `index_k_cpu`, one row per position.
    """

    index_k_cache: torch.Tensor  # [S - T, Di]


def index_q_cpu(indexer: "Indexer", q_lora: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Indexer queries [T, Hi, Di]: wq_b, then rotary on the first R dims."""
    if q_lora.shape[0] == 0:
        return q_lora.new_empty(0, indexer.n_heads, indexer.head_dim)
    q, _ = indexer.wq_b(q_lora)
    q = q.view(-1, indexer.n_heads, indexer.head_dim)
    q_rope = q[..., : indexer.rope_head_dim].contiguous()
    q_rope, _ = indexer.rotary_emb.forward_native(positions, q_rope, q_rope.clone())
    return torch.cat((q_rope.view(q.shape[0], indexer.n_heads, -1), q[..., indexer.rope_head_dim :]), dim=-1)


def index_k_cpu(indexer: "Indexer", x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Indexer keys [T, Di]: wk, layer norm, rotary on the first R dims. This is what the cache stores."""
    if x.shape[0] == 0:
        return x.new_empty(0, indexer.head_dim)
    k, _ = indexer.wk(x)
    k = indexer.k_norm.forward_native(k)
    k_rope = k[..., : indexer.rope_head_dim].contiguous()
    _, k_rope = indexer.rotary_emb.forward_native(positions, k_rope.clone(), k_rope)
    return torch.cat((k_rope.view(k.shape[0], -1), k[..., indexer.rope_head_dim :]), dim=-1)


def index_head_gates_cpu(indexer: "Indexer", x: torch.Tensor) -> torch.Tensor:
    """Per-head gates [T, Hi] fp32: weights_proj, Hi**-0.5, softmax scale (q_scale is 1 in bf16)."""
    w, _ = indexer.weights_proj(x)
    return w.float() * indexer.n_heads**-0.5 * indexer.softmax_scale


def indexer_topk_cpu(
    indexer: "Indexer",
    x: torch.Tensor,
    q_lora: torch.Tensor,
    positions: torch.Tensor,
    index_k_cache: torch.Tensor,
) -> torch.Tensor:
    """Run indexer projections, then dispatch selection to the registered CPU op."""
    q = index_q_cpu(indexer, q_lora, positions)
    new_k = index_k_cpu(indexer, x, positions)
    keys = torch.cat((index_k_cache, new_k), dim=0)
    gates = index_head_gates_cpu(indexer, x)
    try:
        return torch.ops.sgl_kernel.dsa_indexer_topk_cpu(
            q, keys, gates, positions, indexer.index_topk
        )
    except (AttributeError, RuntimeError) as exc:
        if "not implemented" in str(exc).lower() or "no such operator" in str(exc).lower():
            raise NotImplementedError("DSA indexer C++ op is not implemented") from exc
        raise


def indexer_topk_torch(
    q: torch.Tensor,
    keys: torch.Tensor,
    gates: torch.Tensor,
    positions: torch.Tensor,
    topk: int,
    chunk_size: int = 128,
) -> torch.Tensor:
    """Independent PyTorch implementation of gated causal indexer top-k."""
    token_count = q.shape[0]
    result = torch.full(
        (token_count, topk), -1, dtype=torch.int32, device=q.device
    )
    for start in range(0, token_count, chunk_size):
        stop = min(token_count, start + chunk_size)
        valid_max = min(keys.shape[0], int(positions[stop - 1]) + 1)
        scores = torch.zeros(
            (stop - start, valid_max), dtype=torch.float32, device=q.device
        )
        key_t = keys[:valid_max].transpose(0, 1)
        for head in range(q.shape[1]):
            dots = torch.mm(q[start:stop, head], key_t)
            scores.add_(
                torch.relu(dots).float() * gates[start:stop, head, None]
            )
        for row in range(stop - start):
            valid = min(valid_max, int(positions[start + row]) + 1)
            count = min(topk, valid)
            if count == valid:
                selected = torch.arange(valid, device=q.device)
            else:
                selected = torch.topk(scores[row, :valid], count, sorted=False).indices
            result[start + row, :count] = selected.to(torch.int32)
    return result


def sparse_mla_attention_cpu(
    q_abs: torch.Tensor,  # [T, H, r]
    q_pe: torch.Tensor,  # [T, H, R], rotary applied
    c_kv: torch.Tensor,  # [S, r] latent KV of the whole sequence
    k_pe: torch.Tensor,  # [S, R], rotary applied
    topk_indices: torch.Tensor,  # [T, index_topk] int32, -1 padded
    softmax_scale: float,
) -> torch.Tensor:
    """Dispatch sparse absorbed MLA and return latent output ``[T, H, r]``."""
    try:
        return torch.ops.sgl_kernel.sparse_mla_attention_cpu(
            q_abs, q_pe, c_kv, k_pe, topk_indices, softmax_scale
        )
    except (AttributeError, RuntimeError) as exc:
        if "not implemented" in str(exc).lower() or "no such operator" in str(exc).lower():
            raise NotImplementedError("DSA sparse MLA C++ op is not implemented") from exc
        raise


def sparse_mla_latent_torch(
    q_abs: torch.Tensor,
    q_pe: torch.Tensor,
    c_kv: torch.Tensor,
    k_pe: torch.Tensor,
    topk_indices: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Independent PyTorch sparse absorbed-MLA implementation."""
    output = torch.empty_like(q_abs)
    for token in range(q_abs.shape[0]):
        indices = topk_indices[token]
        indices = indices[indices >= 0].to(torch.int64)
        selected_c = c_kv.index_select(0, indices)
        selected_pe = k_pe.index_select(0, indices)
        scores = torch.mm(q_abs[token], selected_c.transpose(0, 1))
        scores.add_(torch.mm(q_pe[token], selected_pe.transpose(0, 1)))
        probs = torch.softmax(scores.float() * softmax_scale, dim=-1).to(c_kv.dtype)
        output[token] = torch.mm(probs, selected_c)
    return output
