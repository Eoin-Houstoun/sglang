"""CPU path of DeepSeek Sparse Attention (DSA): lightning-indexer top-k and sparse MLA.

`Indexer.forward` dispatches to `Indexer.forward_cpu` on a CPU with AMX (dispatch key
"cpu", `SGLANG_USE_CPU_ENGINE=1`), which delegates to `indexer_topk_cpu` below. The
projections the indexer stores and scores with (`index_k_cpu`, `index_q_cpu`,
`index_head_gates_cpu`) are wired and run on CPU. The two functions that compute the
selection and the attention over it, `indexer_topk_cpu` and `sparse_mla_attention_cpu`,
have no CPU implementation yet and raise NotImplementedError, as `Indexer.forward_native`
does upstream.

Shapes: T query tokens, S = len(index_k_cache) + T key positions, Hi indexer heads, Di
indexer head dim, R rope dim, H attention heads, Dn = qk_nope_head_dim, r = kv_lora_rank,
Dv = v_head_dim. All activations bf16 unless stated.

Contract for the selection. For query t (position p_t = positions[t]) and key position
s <= p_t:

    I[t, s] = sum_h gates[t, h] * relu(q[t, h] . k[s])

with q = index_q_cpu(indexer, q_lora, positions), k = cat(index_k_cache,
index_k_cpu(indexer, x, positions)) and gates = index_head_gates_cpu(indexer, x). Select the
index_topk largest I[t, :]; when p_t + 1 <= index_topk, select every valid position.
Return [T, index_topk] int32, -1 padded. The gate compares sets, so order is free.

Contract for the attention. sparse_mla_attention_cpu must equal absorbed MLA restricted to
the selected positions: scores[t, h, k] = (q_nope[t, h] @ w_kc[h]) . c_kv[idx] +
q_pe[t, h] . k_pe[idx], softmax over k with scale, o_lat = p @ c_kv[idx], out[t, h] =
o_lat @ w_vc[h]. Output [T, H, Dv] in the input dtype.
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
    """Top-k key positions per query token: [T, index_topk] int32, -1 padded. See the module docstring."""
    raise NotImplementedError("DSA indexer has no CPU path")


def sparse_mla_attention_cpu(
    q_nope: torch.Tensor,  # [T, H, Dn]
    q_pe: torch.Tensor,  # [T, H, R], rotary applied
    c_kv: torch.Tensor,  # [S, r] latent KV of the whole sequence
    k_pe: torch.Tensor,  # [S, R], rotary applied
    topk_indices: torch.Tensor,  # [T, index_topk] int32, -1 padded
    w_kc: torch.Tensor,  # [H, Dn, r]
    w_vc: torch.Tensor,  # [H, r, Dv]
    softmax_scale: float,
) -> torch.Tensor:
    """Absorbed MLA attention where query t attends only to c_kv[topk_indices[t]]. [T, H, Dv]."""
    raise NotImplementedError("DSA sparse attention has no CPU path")
