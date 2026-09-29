import sys

import pytest
import sgl_kernel  # noqa: F401
import torch
import torch.nn.functional as F
from einops import rearrange

from sglang.kernels.ops.attention.fla.kda_cpu import chunk_kda
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.cpu_test_utils import precision

register_cpu_ci(est_time=10, suite="stage-a-test-cpu-intel")

# [NB]: State-layout convention for this test file:
# - SGLang's state pool is VK: [pool, H, V, K], same as the triton impl.
# - The torch naive references are ported from
#   https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/kda/naive.py (MIT)
#   and follow its KV semantics; transposes only bridge the two views.

# Kimi-Linear-48B-A3B KDA layers: 32 heads, key and value dim 128.
KIMI_HEADS = 32
KIMI_DIM = 128


def naive_recurrent_kda(q, k, v, g, beta, scale=None, initial_state=None):
    dtype = v.dtype
    B, T, H, K, V = *q.shape, v.shape[-1]
    if scale is None:
        scale = K**-0.5

    q, k, v, g, beta = map(lambda x: x.to(torch.float), [q, k, v, g, beta])
    q = q * scale

    S = k.new_zeros(B, H, K, V)
    if initial_state is not None:
        S += initial_state
    o = torch.zeros_like(v)
    for i in range(0, T):
        q_i, k_i, v_i, g_i, b_i = q[:, i], k[:, i], v[:, i], g[:, i], beta[:, i]
        S = S * g_i[..., None].exp()
        S = S + torch.einsum(
            "b h k, b h v -> b h k v",
            b_i[..., None] * k_i,
            v_i - (k_i[..., None] * S).sum(-2),
        )
        o[:, i] = torch.einsum("b h k, b h k v -> b h v", q_i, S)
    return o.to(dtype), S


def naive_chunk_kda(
    q, k, v, g, beta, scale=None, initial_state=None, chunk_size: int = 64
):
    dtype = v.dtype
    B, T, H, K, V = *q.shape, v.shape[-1]
    BT = chunk_size
    NT = T // BT
    if scale is None:
        scale = K**-0.5
    assert T % BT == 0

    q, k = [
        rearrange(x, "b (n c) h ... -> b h n c ...", c=BT).to(torch.float)
        for x in [q, k]
    ]
    v, g, beta = [
        rearrange(x, "b (n c) h ... -> b h n c ...", c=BT).to(torch.float)
        for x in [v, g, beta]
    ]
    q = q * scale
    g = g.cumsum(-2)

    # note that diagonal is masked.
    mask = torch.triu(torch.ones(BT, BT, dtype=torch.bool), diagonal=0)

    A = torch.zeros(*g.shape[:-1], BT, dtype=torch.float)
    for i in range(BT):
        k_i = k[..., i, :]
        g_i = g[..., i : i + 1, :]
        A[..., i] = torch.einsum("... c d, ... d -> ... c", k * (g - g_i).exp(), k_i)
    A = A * beta[..., None]

    A = -A.masked_fill(mask, 0)
    for i in range(1, BT):
        A[..., i, :i] = A[..., i, :i].clone() + (
            A[..., i, :, None].clone() * A[..., :, :i].clone()
        ).sum(-2)
    A = (A + torch.eye(BT, dtype=torch.float)) * beta[..., None, :]

    w = A @ (g.exp() * k)
    u = A @ v

    S = k.new_zeros(B, H, K, V)
    if initial_state is not None:
        S += initial_state
    o = torch.zeros_like(v)
    mask = torch.triu(torch.ones(BT, BT, dtype=torch.bool), diagonal=1)
    for i in range(0, NT):
        q_i = q[:, :, i]
        k_i = k[:, :, i]
        u_i = u[:, :, i]
        g_i = g[:, :, i]
        w_i = w[:, :, i]
        Aqk = torch.zeros(B, H, BT, BT, dtype=torch.float)
        for j in range(BT):
            k_j = k[:, :, i, j]
            g_j = g[:, :, i, j : j + 1, :]
            Aqk[..., j] = torch.einsum(
                "... c d, ... d -> ... c", q_i * (g_i - g_j).exp(), k_j
            )
        Aqk = Aqk.masked_fill(mask, 0)
        v_i = u_i - w_i @ S
        o[:, :, i] = (q_i * g_i.exp()) @ S + Aqk @ v_i
        S = S * rearrange(g_i[:, :, -1].exp(), "b h k -> b h k 1")
        S += rearrange((g_i[:, :, -1:] - g_i).exp() * k_i, "b h c k -> b h k c") @ v_i
    return rearrange(o, "b h n c d -> b (n c) h d").to(dtype), S


def make_inputs(lengths, heads, key_dim, value_dim, dtype, raw_gate, seed):
    """Packed inputs as KimiDeltaAttention passes them to extend()."""
    torch.manual_seed(seed)
    tokens = sum(lengths)
    pool = len(lengths) + 2
    inputs = dict(
        q=torch.randn(1, tokens, heads, key_dim).to(dtype),
        k=torch.randn(1, tokens, heads, key_dim).to(dtype),
        v=(torch.randn(1, tokens, heads, value_dim) * 0.5).to(dtype),
        initial_state=torch.randn(pool, heads, value_dim, key_dim) * 0.05,
        # Non-contiguous pool rows, so some rows must stay untouched.
        initial_state_indices=torch.randperm(pool)[: len(lengths)].to(torch.int32),
        cu_seqlens=torch.tensor(
            [0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32
        ),
        use_qk_l2norm_in_kernel=True,
    )
    if raw_gate:
        inputs.update(
            g=(torch.randn(1, tokens, heads, key_dim) * 0.5 - 1.0).to(dtype),
            beta=torch.randn(1, tokens, heads).to(dtype),
            A_log=torch.randn(1, 1, heads, 1) * 0.1,
            dt_bias=torch.randn(heads * key_dim) * 0.1,
            beta_is_raw=True,
        )
    else:
        inputs.update(
            g=(-(torch.randn(1, tokens, heads, key_dim) * 0.05).abs() - 0.02).to(dtype),
            beta=torch.rand(1, tokens, heads).to(dtype),
        )
    return inputs


def activated_gate_and_beta(inputs, lower_bound):
    g, beta = inputs["g"].float(), inputs["beta"].float()
    if "A_log" in inputs:
        x = g + inputs["dt_bias"].view(1, 1, *g.shape[-2:])
        decay = inputs["A_log"].exp().view(1, 1, -1, 1)
        if lower_bound is not None:
            g = lower_bound * torch.sigmoid(decay * x)
        else:
            g = -decay * F.softplus(x)
    if inputs.get("beta_is_raw", False):
        beta = beta.sigmoid()
    return g, beta


def reference(inputs, chunk_size=None, lower_bound=None):
    """Per sequence: naive recurrent (chunk_size None) or chunked KDA; returns output and pool."""
    g, beta = activated_gate_and_beta(inputs, lower_bound)
    q = F.normalize(inputs["q"].float(), dim=-1, eps=1e-6)
    k = F.normalize(inputs["k"].float(), dim=-1, eps=1e-6)
    v = inputs["v"].float()
    pool = inputs["initial_state"].clone()
    output = torch.empty_like(inputs["v"])
    offsets = inputs["cu_seqlens"].tolist()
    for sequence, row in enumerate(inputs["initial_state_indices"].tolist()):
        span = slice(offsets[sequence], offsets[sequence + 1])
        args = [x[:, span] for x in (q, k, v, g, beta)]
        state = pool[row].transpose(-1, -2).unsqueeze(0)
        if chunk_size is None:
            out, final = naive_recurrent_kda(*args, initial_state=state)
        else:
            out, final = naive_chunk_kda(
                *args, initial_state=state, chunk_size=chunk_size
            )
        output[:, span] = out.to(output.dtype)
        pool[row] = final[0].transpose(-1, -2)
    return output, pool


def check_against(inputs, expected, expected_pool, lower_bound=None):
    state = inputs["initial_state"].clone()
    kernel_inputs = {**inputs, "initial_state": state}
    actual = chunk_kda(**kernel_inputs, lower_bound=lower_bound)
    atol = rtol = precision[inputs["q"].dtype]
    assert actual.shape == inputs["v"].shape and actual.dtype == inputs["v"].dtype
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    rows = inputs["initial_state_indices"].long()
    torch.testing.assert_close(state[rows], expected_pool[rows], atol=atol, rtol=rtol)
    untouched = sorted(set(range(state.shape[0])) - set(rows.tolist()))
    torch.testing.assert_close(
        state[untouched], inputs["initial_state"][untouched], atol=0, rtol=0
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "lengths, heads, key_dim, value_dim",
    [
        ([1, 63, 64, 65], 2, 64, 64),  # chunk-boundary tails
        ([3, 17], 2, 47, 33),  # K and V not multiples of the vector width
        ([130], KIMI_HEADS, KIMI_DIM, KIMI_DIM),
        ([40, 1, 90], KIMI_HEADS, KIMI_DIM, KIMI_DIM),  # ragged batch
    ],
)
@torch.inference_mode()
def test_chunk_kda_matches_recurrent_reference(
    dtype, lengths, heads, key_dim, value_dim
):
    inputs = make_inputs(
        lengths, heads, key_dim, value_dim, dtype, raw_gate=False, seed=len(lengths)
    )
    check_against(inputs, *reference(inputs))


@pytest.mark.parametrize("chunk_size", [16, 32, 64])
@torch.inference_mode()
def test_chunk_kda_matches_chunked_reference(chunk_size):
    inputs = make_inputs(
        [64, 192],
        KIMI_HEADS,
        KIMI_DIM,
        KIMI_DIM,
        torch.bfloat16,
        raw_gate=False,
        seed=chunk_size,
    )
    check_against(inputs, *reference(inputs, chunk_size=chunk_size))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("lower_bound", [None, -5.0])
@torch.inference_mode()
def test_chunk_kda_kimi_linear_call(dtype, lower_bound):
    """The path Kimi Linear prefill takes: raw gate and beta, activated in the kernel."""
    inputs = make_inputs(
        [70, 1, 57], KIMI_HEADS, KIMI_DIM, KIMI_DIM, dtype, raw_gate=True, seed=5
    )
    check_against(
        inputs, *reference(inputs, lower_bound=lower_bound), lower_bound=lower_bound
    )


def test_kda_triton_dispatches_to_cpu_kernel_on_cpu():
    from sglang.srt.layers.attention.linear.kernels import kda_triton

    assert kda_triton.chunk_kda is chunk_kda


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
