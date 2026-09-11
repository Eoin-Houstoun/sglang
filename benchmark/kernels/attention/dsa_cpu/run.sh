#!/bin/sh
# Run the DSA CPU benchmark commands from any working directory, in a CPU
# environment that is built on first use (see setup_env.sh).
#
#   sh benchmark/kernels/attention/dsa_cpu/run.sh compile|test|bench
#
# SGLANG_CPU_VENV overrides where the environment lives
# (default $HOME/.artemis/sglang-cpu-venv). The checkout containing this
# script is what runs: SGLang is not installed into the environment.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../../../.." && pwd)
VENV=${SGLANG_CPU_VENV:-$HOME/.artemis/sglang-cpu-venv}
PY=$VENV/bin/python
if [ ! -x "$PY" ] || ! "$PY" -c "import torch, sgl_kernel" >/dev/null 2>&1; then
  sh "$HERE/setup_env.sh" "$VENV"
fi
export SGLANG_USE_CPU_ENGINE=1
export PYTHONPATH="$ROOT/python${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16}
cd "$ROOT"
case "$1" in
  compile) exec "$PY" -c "import sglang.srt.layers.attention.dsa.dsa_cpu" ;;
  test) exec "$PY" -m pytest -q -p no:cacheprovider benchmark/kernels/attention/dsa_cpu/test_dsa_cpu.py ;;
  bench) exec "$PY" benchmark/kernels/attention/dsa_cpu/bench_dsa_cpu.py ;;
  *) echo "usage: run.sh compile|test|bench" >&2; exit 2 ;;
esac
