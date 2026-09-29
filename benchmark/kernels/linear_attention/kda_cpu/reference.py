"""Inputs and references for the CPU chunk_kda benchmark.

`reference_kda` is the correctness gate: a per-token, per-head float64 loop
of the KDA delta rule, kept deliberately independent of any implementation
under test. `pytorch_step` is the frozen PyTorch fallback the benchmark times
against (and compiles with torch.compile); it never changes between versions.
"""

from typing import NamedTuple

import torch
import torch.nn.functional as F

# Kimi-Linear-48B-A3B: 20 KDA layers, 32 heads, key and value dim 128.
KIMI_HEADS = 32
KIMI_DIM = 128


class KDAInputs(NamedTuple):
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    g: torch.Tensor
    beta: torch.Tensor
    state: torch.Tensor
    indices: torch.Tensor
    cu_seqlens: torch.Tensor
    A_log: torch.Tensor | None = None
    dt_bias: torch.Tensor | None = None
    lower_bound: float | None = None
    beta_is_raw: bool = False

    def kwargs(self, state: torch.Tensor) -> dict:
        return dict(
            q=self.q,
            k=self.k,
            v=self.v,
            g=self.g,
            beta=self.beta,
            initial_state=state,
            initial_state_indices=self.indices,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=self.cu_seqlens,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            lower_bound=self.lower_bound,
            beta_is_raw=self.beta_is_raw,
        )


def make_inputs(
    lengths,
    heads=KIMI_HEADS,
    key_dim=KIMI_DIM,
    value_dim=KIMI_DIM,
    dtype=torch.bfloat16,
    raw_gate=False,
    lower_bound=None,
    pool=None,
    seed=0,
) -> KDAInputs:
    generator = torch.Generator().manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, generator=generator)

    tokens = sum(lengths)
    pool = pool or len(lengths) + 2
    q = randn(1, tokens, heads, key_dim).to(dtype)
    k = randn(1, tokens, heads, key_dim).to(dtype)
    v = (randn(1, tokens, heads, value_dim) * 0.5).to(dtype)
    if raw_gate:
        g = (randn(1, tokens, heads, key_dim) * 0.5 - 1.0).to(dtype)
        beta = randn(1, tokens, heads).to(dtype)
        A_log = randn(heads) * 0.1
        dt_bias = randn(heads, key_dim) * 0.1
    else:
        g = (-(randn(1, tokens, heads, key_dim) * 0.05).abs() - 0.02).to(dtype)
        beta = torch.rand(1, tokens, heads, generator=generator).to(dtype)
        A_log = dt_bias = None
    state = randn(pool, heads, value_dim, key_dim) * 0.05
    # Scatter sequences over non-contiguous pool rows so untouched rows exist.
    indices = torch.randperm(pool, generator=generator)[: len(lengths)].to(torch.int32)
    cu_seqlens = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32)
    return KDAInputs(
        q, k, v, g, beta, state, indices, cu_seqlens, A_log, dt_bias, lower_bound, raw_gate
    )


def activated(inputs: KDAInputs):
    """Gate (log decay) and beta after the optional in-kernel activations."""
    g, beta = inputs.g.double(), inputs.beta.double()
    if inputs.A_log is not None:
        x = g + inputs.dt_bias.double().view(1, 1, *g.shape[-2:])
        decay = inputs.A_log.double().exp().view(1, 1, -1, 1)
        if inputs.lower_bound is not None:
            g = inputs.lower_bound * torch.sigmoid(decay * x)
        else:
            g = -decay * F.softplus(x)
    if inputs.beta_is_raw:
        beta = beta.sigmoid()
    return g, beta


def reference_kda(inputs: KDAInputs):
    """Float64 golden output and state pool."""
    g, beta = activated(inputs)
    q = F.normalize(inputs.q.double(), dim=-1, eps=1e-6)
    k = F.normalize(inputs.k.double(), dim=-1, eps=1e-6)
    v = inputs.v.double()
    scale = inputs.q.shape[-1] ** -0.5
    output = torch.zeros(v.shape, dtype=torch.float64)
    pool = inputs.state.double().clone()
    offsets = inputs.cu_seqlens.tolist()
    for sequence, row in enumerate(inputs.indices.tolist()):
        for head in range(q.shape[2]):
            state = pool[row, head]
            for t in range(offsets[sequence], offsets[sequence + 1]):
                state = state * g[0, t, head].exp()[None, :]
                residual = v[0, t, head] - state @ k[0, t, head]
                state = state + beta[0, t, head] * torch.outer(residual, k[0, t, head])
                output[0, t, head] = (state @ q[0, t, head]) * scale
            pool[row, head] = state
    return output, pool


def pytorch_step(state, decay, k, v, beta, q, scale: float):
    """One token of KDA for all heads, fp32. Returns the new state and output."""
    state = state * decay.unsqueeze(-2)
    residual = v - torch.einsum("hvk,hk->hv", state, k)
    state = state + torch.einsum("hv,hk->hvk", residual * beta[:, None], k)
    return state, torch.einsum("hvk,hk->hv", state, q) * scale


def pytorch_fallback(inputs: KDAInputs, state_pool: torch.Tensor, step=pytorch_step):
    """The frozen PyTorch baseline: token loop over `step`, heads vectorised."""
    g, beta = activated(inputs)
    q = F.normalize(inputs.q[0].float(), dim=-1, eps=1e-6)
    k = F.normalize(inputs.k[0].float(), dim=-1, eps=1e-6)
    v = inputs.v[0].float()
    decay = g[0].float().exp()
    beta = beta[0].float()
    scale = inputs.q.shape[-1] ** -0.5
    output = torch.empty_like(inputs.v[0])
    offsets = inputs.cu_seqlens.tolist()
    for sequence, row in enumerate(inputs.indices.tolist()):
        state = state_pool[row]
        for t in range(offsets[sequence], offsets[sequence + 1]):
            state, out = step(state, decay[t], k[t], v[t], beta[t], q[t], scale)
            output[t] = out.to(output.dtype)
        state_pool[row] = state
    return output.unsqueeze(0)
