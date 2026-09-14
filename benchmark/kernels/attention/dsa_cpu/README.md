# DSA on CPU: layer benchmark and correctness gate

DeepSeek Sparse Attention (DSA), as used by GLM-5.2, has no CPU path in SGLang: on a CPU
with AMX `Indexer.forward` dispatches to `Indexer.forward_cpu`, which delegates to
`sglang/srt/layers/attention/dsa/dsa_cpu.py`, where `indexer_topk_cpu` and
`sparse_mla_attention_cpu` raise `NotImplementedError`. This directory drives that gap at
GLM-5.2's attention shape (64 attention heads, 32 indexer heads, `index_topk` 2048, real
MLA dims, bf16; only the hidden width and `q_lora_rank` are reduced) with SGLang's own
`Indexer` module, so the missing functions can be implemented and made fast with a benchmark
and a correctness gate in the loop.

## Setup, once per machine

Builds a CPU environment at `$HOME/.artemis/sglang-cpu-venv` (override with
`SGLANG_CPU_VENV`): SGLang's CPU dependencies from `pyproject_cpu.toml` and the CPU
`sgl-kernel`, per `docs/hardware-platforms/cpu_server.mdx`. About ten minutes on a 32-core
Xeon, and the only step that needs the build toolchain (gcc-13 or a recent gcc, cmake, ninja,
libnuma-dev, libtbb-dev) or the network. It checks for those first and names anything missing.

SGLang itself is not installed into the environment: `PYTHONPATH` points at the checkout, so
the code under test is what runs.

```bash
sh benchmark/kernels/attention/dsa_cpu/setup_env.sh
```

## Run

From the repository root, with `$VENV` as the environment above:

```bash
VENV=$HOME/.artemis/sglang-cpu-venv

# compile: the layer imports
SGLANG_USE_CPU_ENGINE=1 PYTHONPATH=$PWD/python \
  $VENV/bin/python -c "import sglang.srt.layers.attention.dsa.dsa_cpu"

# test: dense goldens pass, sparse tests skip until implemented
SGLANG_USE_CPU_ENGINE=1 PYTHONPATH=$PWD/python OMP_NUM_THREADS=16 \
  $VENV/bin/python -m pytest -q -p no:cacheprovider \
  benchmark/kernels/attention/dsa_cpu/test_dsa_cpu.py

# benchmark: writes artemis_results.json to the repository root
SGLANG_USE_CPU_ENGINE=1 PYTHONPATH=$PWD/python OMP_NUM_THREADS=16 \
  $VENV/bin/python benchmark/kernels/attention/dsa_cpu/bench_dsa_cpu.py
```

Those three are the compile, test and benchmark commands for an Artemis project on this
branch, with `$VENV` and `$PWD` written out in full. `SGLANG_USE_CPU_ENGINE=1` is required:
without it the CPU dispatch key is empty and nothing reaches `forward_cpu`. Set
`OMP_NUM_THREADS` to the cores you want the benchmark to use; it changes the timings, so keep
it the same across versions being compared.

## Metrics (`artemis_results.json`)

| Metric | Better | Meaning |
|---|---|---|
| `dsa_sparse_correct` | 1 | 1 only when the sparse path runs on all five operating points and matches the goldens |
| `prefill_ms_4k` | lower | median time of one attention layer, 4096-token prefill |
| `prefill_ms_8k` | lower | same at 8192 tokens |
| `decode_ms_8k` | lower | one decode step for 4 sequences with an 8k KV cache |
| `indexer_ms_8k` | lower | indexer alone at 8k; 0 until the sparse path is correct |
| `gen_ms_8k_in_1k_out` | lower | 8k tokens in, 1,000 out: prefill at 8k plus 1,000 decode steps, derived from the two above |

Until the sparse path exists the timings are those of the dense stand-in in `synthetic.py`
(absorbed MLA over every position, pure torch), which is the computation the CPU falls back
to today.

## Correctness gate

`fixtures/*.pt` hold, for 48 sampled query rows per operating point, the reference top-k sets
and the sparse and dense outputs computed by an fp32 reference of the DSA maths (kept outside
the repository). `goldens.py` checks index-set overlap (at least 0.97 per row; rows with fewer
than 2048 valid positions must select all of them) and output error. Tolerances were set from
a bf16 run of the reference with a 3x margin and verified to reject a dropped ReLU, missing
head gates, non-causal selection and a wrong attention scale. `goldens.py`, `synthetic.py`,
`test_dsa_cpu.py`, `bench_dsa_cpu.py` and `fixtures/` are the referee: do not modify them.

## What to implement

`dsa_cpu.py` states the contract in its module docstring. The projections (`index_q_cpu`,
`index_k_cpu`, `index_head_gates_cpu`) are wired; `indexer_topk_cpu` and
`sparse_mla_attention_cpu` are the gap. The dense fallback at 8k does four times the
arithmetic of the sparse path, but a per-row gather of 2,048 KV rows moves about 19 GB, so
beating dense needs gathers shared across query rows, not a naive select.

## Boundary

This is the attention layer, not the serving path: `DSACPUBatch` stands in for the DSA
token-to-KV pool, and the AMX dense kernels are not the baseline here. Wiring the CPU path
into `forward_absorb` and the pool is the step after the two functions exist.
