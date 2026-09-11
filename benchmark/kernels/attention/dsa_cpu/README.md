# DSA on CPU: layer benchmark and correctness gate

DeepSeek Sparse Attention (DSA), as used by GLM-5.2, has no CPU path in SGLang: on a CPU
with AMX `Indexer.forward` dispatches to `Indexer.forward_cpu`, which delegates to
`sglang/srt/layers/attention/dsa/dsa_cpu.py`, where `indexer_topk_cpu` and
`sparse_mla_attention_cpu` raise `NotImplementedError`. This directory drives that gap at
GLM-5.2's attention shape (64 attention heads, 32 indexer heads, `index_topk` 2048, real
MLA dims, bf16; only the hidden width and `q_lora_rank` are reduced) with SGLang's own
`Indexer` module, so the missing functions can be implemented and made fast with a benchmark
and a correctness gate in the loop.

## Run

From the repository root, in an environment built per `docs/hardware-platforms/cpu_server.mdx`
(the CPU `sgl-kernel` must be installed; the indexer's import chain needs it):

```bash
export SGLANG_USE_CPU_ENGINE=1 PYTHONPATH=$PWD/python
python -c "import sglang.srt.layers.attention.dsa.dsa_cpu"          # import check
pytest -q benchmark/kernels/attention/dsa_cpu/test_dsa_cpu.py       # dense goldens pass; sparse tests skip until implemented
python benchmark/kernels/attention/dsa_cpu/bench_dsa_cpu.py         # writes artemis_results.json to the working directory
```

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
