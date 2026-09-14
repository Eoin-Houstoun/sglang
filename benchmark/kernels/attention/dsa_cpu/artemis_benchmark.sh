#!/usr/bin/env bash
set -euo pipefail

VENV="${SGLANG_CPU_VENV:-$HOME/.artemis/sglang-cpu-venv}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
ORIGINAL_PWD="$PWD"

rm -f "$ORIGINAL_PWD/artemis_results.json" "$ORIGINAL_PWD/artemis_results.csv"
export SGLANG_USE_CPU_ENGINE=1
export PYTHONPATH="$ROOT/python"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"

"$VENV/bin/python" \
  "$ROOT/benchmark/kernels/attention/dsa_cpu/bench_dsa_cpu.py"
test -s "$ORIGINAL_PWD/artemis_results.json"
