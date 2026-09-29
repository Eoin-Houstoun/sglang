#!/usr/bin/env bash
set -euo pipefail

VENV="${SGLANG_CPU_VENV:-$HOME/.artemis/sglang-cpu-venv}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
REPORT="$(mktemp)"
trap 'rm -f "$REPORT"' EXIT

export SGLANG_USE_CPU_ENGINE=1
export PYTHONPATH="$ROOT/python"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"

"$VENV/bin/python" -m pytest -q -rs -p no:cacheprovider \
  "$ROOT/test/registered/cpu/test_kda.py" | tee "$REPORT"
# A skipped test is a gate that did not run.
if grep -q -E '[0-9]+ skipped' "$REPORT"; then
  echo "kda_cpu: tests were skipped, failing" >&2
  exit 1
fi
