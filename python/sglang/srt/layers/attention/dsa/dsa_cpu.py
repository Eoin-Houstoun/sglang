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
    T = x.shape[0]
    topk = indexer.index_topk
    out = torch.full((T, topk), -1, dtype=torch.int32, device=x.device)

    # A causal row with at most topk valid positions has a predetermined
    # answer. Besides being exact, handling these rows here avoids both the
    # indexer GEMMs and topk for short prompts and the early prefill prefix.
    valid_counts = (positions + 1).clamp(min=0)
    deterministic = valid_counts <= topk
    for t in deterministic.nonzero(as_tuple=False).flatten().tolist():
        n = min(int(valid_counts[t]), topk)
        if n:
            out[t, :n] = torch.arange(n, dtype=torch.int32, device=x.device)

    work = (~deterministic).nonzero(as_tuple=False).flatten()
    if work.numel() == 0:
        return out

    q = index_q_cpu(indexer, q_lora, positions)
    new_k = index_k_cpu(indexer, x, positions)
    keys = torch.cat((index_k_cache, new_k), dim=0)
    gates = index_head_gates_cpu(indexer, x)

    # With one decode query, materialising all bf16 head logits is small and
    # one matmul is substantially cheaper than 32 tiny GEMM dispatches.
    if T == 1:
        dots = torch.mm(q[0], keys.T)
        dots.relu_()
        logits = (dots.float() * gates[0, :, None]).sum(dim=0)
        selected = torch.topk(logits[: int(valid_counts[0])], topk, sorted=False).indices
        out[0] = selected.sort().values.to(torch.int32)
        return out

    # One fp32 [query chunk, KV] accumulator avoids materialising the much
    # larger [query, index head, KV] logits tensor.
    chunk = 256
    for i0 in range(0, work.numel(), chunk):
        rows = work[i0 : i0 + chunk]
        max_valid = min(int(valid_counts[rows].max()), keys.shape[0])
        logits = torch.zeros((rows.numel(), max_valid), dtype=torch.float32, device=x.device)
        qr = q[rows]
        for h in range(indexer.n_heads):
            dots = qr[:, h] @ keys[:max_valid].T
            logits.add_(torch.relu(dots).float() * gates[rows, h, None])
        cols = torch.arange(max_valid, device=x.device)
        logits.masked_fill_(cols[None, :] >= valid_counts[rows, None], float("-inf"))
        selected = torch.topk(logits, topk, dim=-1, sorted=False).indices
        selected = selected.sort(dim=-1).values.to(torch.int32)
        out[rows] = selected
    return out


_DETERMINISTIC_BLOCK = 1024
_SPARSE_QUERY_CHUNK = 16


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
    T = q_nope.shape[0]
    H = w_vc.shape[0]
    scale = softmax_scale
    out = torch.empty((T, H, w_vc.shape[-1]), dtype=q_nope.dtype, device=q_nope.device)

    # Decode uses one fully-populated selection row. Keep heads as matrix rows
    # and issue direct matrix products rather than entering the general gather
    # chunk and dispatching four einsums.
    if T == 1 and bool((topk_indices[0] >= 0).all()):
        idx = topk_indices[0].long()
        c_sel = c_kv.index_select(0, idx)
        pe_sel = k_pe.index_select(0, idx)
        q_abs = torch.bmm(q_nope[0].unsqueeze(1), w_kc).squeeze(1)
        scores = torch.mm(q_abs, c_sel.T)
        scores.add_(torch.mm(q_pe[0], pe_sel.T))
        p = torch.softmax(scores.float().mul_(scale), dim=-1).to(c_kv.dtype)
        o_lat = torch.mm(p, c_sel)
        out[0] = torch.bmm(o_lat.unsqueeze(1), w_vc).squeeze(1)
        return out

    q_abs = torch.einsum("thd,hdr->thr", q_nope, w_kc)

    # Leading causal rows select their complete prefix. Compute these in large
    # rectangular blocks so every KV row is shared by many queries instead of
    # being copied by a per-row gather.
    dense_end = 0
    max_dense = min(T, topk_indices.shape[1])
    while dense_end < max_dense:
        row = topk_indices[dense_end]
        n = int((row >= 0).sum())
        if n != dense_end + 1 or not torch.equal(row[:n], torch.arange(n, dtype=row.dtype, device=row.device)):
            break
        dense_end += 1

    block = _DETERMINISTIC_BLOCK
    for t0 in range(0, dense_end, block):
        t1 = min(dense_end, t0 + block)
        S = t1
        scores = torch.einsum("thr,sr->ths", q_abs[t0:t1], c_kv[:S])
        scores.add_(torch.einsum("thd,sd->ths", q_pe[t0:t1], k_pe[:S]))
        scores = scores.float().mul_(scale)
        limit = torch.arange(t0, t1, device=q_nope.device)[:, None, None]
        cols = torch.arange(S, device=q_nope.device)
        scores.masked_fill_(cols[None, None, :] > limit, float("-inf"))
        p = torch.softmax(scores, dim=-1).to(c_kv.dtype)
        o_lat = torch.einsum("ths,sr->thr", p, c_kv[:S])
        out[t0:t1] = torch.einsum("thr,hrv->thv", o_lat, w_vc)

    # A small row chunk bounds the irregular gathered working set while all
    # heads remain vectorised so the contractions still reach bf16 AMX kernels.
    chunk = _SPARSE_QUERY_CHUNK
    for t0 in range(dense_end, T, chunk):
        t1 = min(T, t0 + chunk)
        idx = topk_indices[t0:t1].long()
        valid = idx >= 0
        safe = idx.clamp_min(0)
        c_sel = c_kv[safe]
        pe_sel = k_pe[safe]
        scores = torch.einsum("thr,tkr->thk", q_abs[t0:t1], c_sel)
        scores.add_(torch.einsum("thd,tkd->thk", q_pe[t0:t1], pe_sel))
        scores = scores.float().mul_(scale)
        scores.masked_fill_(~valid[:, None, :], float("-inf"))
        p = torch.softmax(scores, dim=-1).to(c_kv.dtype)
        o_lat = torch.einsum("thk,tkr->thr", p, c_sel)
        out[t0:t1] = torch.einsum("thr,hrv->thv", o_lat, w_vc)
    return out
