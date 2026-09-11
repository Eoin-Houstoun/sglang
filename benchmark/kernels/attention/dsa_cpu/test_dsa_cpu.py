"""Correctness gate for the DSA CPU path against fp32 reference goldens.

Run from the repository root:
    SGLANG_USE_CPU_ENGINE=1 PYTHONPATH=$PWD/python pytest -q benchmark/kernels/attention/dsa_cpu/test_dsa_cpu.py
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import pytest
import torch

from sglang.srt.runtime_context import publish
from sglang.srt.server_args import ServerArgs

publish(ServerArgs(model_path="dummy"), role="test")

import goldens  # noqa: E402
from bench_dsa_cpu import dsa_attention, sparse_available  # noqa: E402
from synthetic import GLM52_SMALL, build_indexer, dense_mla_attention, make_batches, make_weights  # noqa: E402

CFG = GLM52_SMALL
W = make_weights(CFG)
INDEXER = build_indexer(CFG, W)
SPARSE = sparse_available(INDEXER, W)


def _run_point(name):
    outs, topks = [], []
    with torch.no_grad():
        for b in make_batches(name, CFG, INDEXER):
            out, topk = dsa_attention(INDEXER, W, b)
            outs.append(out)
            topks.append(topk)
    return torch.cat(outs), (None if topks[0] is None else torch.cat(topks))


@pytest.mark.parametrize("name", goldens.POINTS)
def test_dense_matches_golden(name):
    with torch.no_grad():
        outs = [dense_mla_attention(b.q_nope, b.q_pe, b.c_kv, b.k_pe, W.W_UK, W.W_UV, CFG.softmax_scale, b.causal_offset) for b in make_batches(name, CFG, INDEXER)]
    assert goldens.check_dense_point(name, torch.cat(outs)) == []


@pytest.mark.skipif(not SPARSE, reason="sparse path not implemented (upstream state)")
@pytest.mark.parametrize("name", goldens.POINTS)
def test_sparse_matches_golden(name):
    out, topk = _run_point(name)
    assert goldens.check_sparse_point(name, out, topk) == []


@pytest.mark.skipif(not SPARSE, reason="sparse path not implemented (upstream state)")
def test_sparse_indices_causal_and_in_range():
    with torch.no_grad():
        for b in make_batches("prefill_4k", CFG, INDEXER):
            _, topk = dsa_attention(INDEXER, W, b)
            assert topk.dtype == torch.int32
            valid = topk >= 0
            limit = b.positions[:, None].expand_as(topk)
            assert bool((topk[valid] <= limit[valid]).all())
            assert bool((topk[valid] >= 0).all())
