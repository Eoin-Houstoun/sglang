# GLM-5.2 DeepSeek Sparse Attention on CPU

This harness defines the production-shaped CPU boundary missing from SGLang:

- exact GLM-5.2 dimensions (`hidden_size=6144`, `q_lora_rank=2048`,
  64 attention heads, 32 index heads, index dimension 128, latent KV 512,
  RoPE 64, top-k 2048, page size 64);
- SGLang's real indexer projections and rotary embedding;
- a registered C++ indexer/top-k operation;
- a registered C++ sparse absorbed-MLA operation returning latent
  `[tokens, 64, 512]` output;
- independent FP32 correctness oracles;
- pure-PyTorch DSA and SGLang dense CPU MLA performance baselines.

The C++ operations are intentionally stubs in the seed. A functional Discovery
must implement them without changing their schemas, tests, or benchmark.

## One-time runner setup

From a checkout on the runner:

```bash
sh benchmark/kernels/attention/dsa_cpu/setup_env.sh
```

This creates `$HOME/.artemis/sglang-cpu-venv`, installs CPU dependencies, and
builds the initial `sgl_kernel`. Artemis candidate builds then reuse
`$HOME/.artemis/sglang-dsa-cpu-cache`.

## Artemis commands

Run all commands from the repository root:

```bash
benchmark/kernels/attention/dsa_cpu/artemis_compile.sh
benchmark/kernels/attention/dsa_cpu/artemis_test.sh
benchmark/kernels/attention/dsa_cpu/artemis_benchmark.sh
```

The benchmark writes numeric metrics to `artemis_results.json`. Missing C++ ops
receive a large latency sentinel and `cpp_dsa_correct=0`; implemented candidates
must set that gate to `1` before their latency is meaningful.

The C4 runner uses an Emerald Rapids Xeon Platinum 8581C. Results from this
harness are evidence for that CPU and should not be described as Xeon 6 results.

## Protected contract

Discovery may modify:

- `python/sglang/kernels/aot/csrc/cpu/dsa.cpp`;
- supporting private C++ helpers under the same CPU source directory;
- the thin dispatch wrapper in
  `python/sglang/srt/layers/attention/dsa/dsa_cpu.py`.

It must not modify operation schemas, FP32 oracles, synthetic shapes, tests,
benchmark metric definitions, or the Artemis command scripts.
