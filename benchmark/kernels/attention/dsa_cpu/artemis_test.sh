#!/usr/bin/env bash
set -euo pipefail

VENV="${SGLANG_CPU_VENV:-$HOME/.artemis/sglang-cpu-venv}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"

export SGLANG_USE_CPU_ENGINE=1
export PYTHONPATH="$ROOT/python"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"

"$VENV/bin/python" -m pytest -q -p no:cacheprovider \
  "$ROOT/benchmark/kernels/attention/dsa_cpu/test_dsa_cpu.py"
