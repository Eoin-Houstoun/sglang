"""Correctness gate for the CPU chunk_kda entry point.

Compares output and the updated state pool against the float64 reference,
and checks that state rows no sequence points at are left untouched.
Nothing here skips: a skipped gate would pass silently.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))
from reference import make_inputs, reference_kda  # noqa: E402

import sgl_kernel  # noqa: E402,F401  registers the torch.ops.sgl_kernel CPU ops
from sglang.kernels.ops.attention.fla.kda_cpu import chunk_kda  # noqa: E402

TOLERANCE = {torch.bfloat16: 3e-2, torch.float16: 1e-2}

CASES = [
    # (lengths, heads, key_dim, value_dim)
    ([1, 63, 64, 65], 2, 64, 64),  # chunk-boundary tails
    ([3, 17], 2, 47, 33),  # K and V not multiples of the vector width
    ([130], 32, 128, 128),  # Kimi Linear shape, one sequence
    ([40, 1, 90], 32, 128, 128),  # Kimi Linear shape, ragged batch
]


def run_and_check(inputs):
    state = inputs.state.clone()
    actual = chunk_kda(**inputs.kwargs(state))
    expected, expected_pool = reference_kda(inputs)
    tolerance = TOLERANCE[inputs.q.dtype]
    assert actual.shape == inputs.v.shape and actual.dtype == inputs.v.dtype
    torch.testing.assert_close(actual.double(), expected, atol=tolerance, rtol=tolerance)
    touched = inputs.indices.long()
    torch.testing.assert_close(
        state[touched].double(), expected_pool[touched], atol=tolerance, rtol=tolerance
    )
    untouched = sorted(set(range(state.shape[0])) - set(touched.tolist()))
    assert untouched, "case must leave some pool rows unused"
    torch.testing.assert_close(state[untouched], inputs.state[untouched], atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("lengths,heads,key_dim,value_dim", CASES)
@torch.inference_mode()
def test_matches_reference(dtype, lengths, heads, key_dim, value_dim):
    run_and_check(make_inputs(lengths, heads, key_dim, value_dim, dtype=dtype, seed=len(lengths)))


@pytest.mark.parametrize("lower_bound", [None, -5.0])
@torch.inference_mode()
def test_in_kernel_gate_and_beta_activation(lower_bound):
    run_and_check(
        make_inputs([9, 24], 4, 32, 24, raw_gate=True, lower_bound=lower_bound, seed=11)
    )


@torch.inference_mode()
def test_sglang_cpu_extend_dispatches_here():
    from sglang.srt.layers.attention.linear.kernels import kda_triton

    assert kda_triton.chunk_kda is chunk_kda
