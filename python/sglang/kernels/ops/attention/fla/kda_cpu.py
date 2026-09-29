from typing import Optional

import torch
import torch.nn.functional as F


def _activate_gate(g, A_log, dt_bias, lower_bound):
    x = g.float() + dt_bias.float().view(1, 1, *g.shape[-2:])
    decay = A_log.float().exp().view(1, 1, -1, 1)
    if lower_bound is not None:
        return lower_bound * torch.sigmoid(decay * x)
    return -decay * F.softplus(x)


def chunk_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    initial_state_indices: torch.Tensor = None,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
    A_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    lower_bound: Optional[float] = None,
    output_intermediate_states: bool = False,
    track_state: Optional[torch.Tensor] = None,
    track_chunk_idx: Optional[torch.Tensor] = None,
    beta_is_raw: bool = False,
    **kwargs,
):
    """KDA prefill on CPU. Updates the selected rows of ``initial_state`` in place.

    q, k, g: [1, T, H, K]; v: [1, T, H, V]; beta: [1, T, H];
    initial_state: fp32 [N, H, V, K]; initial_state_indices: [num_seqs].
    Returns the output [1, T, H, V] in v's dtype.
    """
    if (
        output_intermediate_states
        or track_state is not None
        or track_chunk_idx is not None
    ):
        raise NotImplementedError(
            "CPU chunk_kda does not support intermediate state snapshots"
        )
    if initial_state is None or initial_state_indices is None:
        raise ValueError(
            "CPU chunk_kda requires initial_state and initial_state_indices"
        )
    if scale is None:
        scale = k.shape[-1] ** -0.5

    tokens = q.shape[1]
    offsets = [0, tokens] if cu_seqlens is None else cu_seqlens.tolist()
    q = q[0].float()
    k = k[0].float()
    if use_qk_l2norm_in_kernel:
        q = F.normalize(q, dim=-1, eps=1e-6)
        k = F.normalize(k, dim=-1, eps=1e-6)
    gate = (
        _activate_gate(g, A_log, dt_bias, lower_bound)[0]
        if A_log is not None
        else g[0].float()
    )
    decay = gate.exp()
    beta = beta[0].float().sigmoid() if beta_is_raw else beta[0].float()
    values = v[0].float()
    output = torch.empty_like(v[0])

    for sequence, state_index in enumerate(initial_state_indices.tolist()):
        state = initial_state[state_index].clone()
        for t in range(offsets[sequence], offsets[sequence + 1]):
            state.mul_(decay[t].unsqueeze(-2))
            residual = values[t] - torch.einsum("hvk,hk->hv", state, k[t])
            state.add_(torch.einsum("hv,hk->hvk", residual * beta[t, :, None], k[t]))
            output[t] = (torch.einsum("hvk,hk->hv", state, q[t]) * scale).to(
                output.dtype
            )
        initial_state[state_index] = state
    return output.unsqueeze(0)
