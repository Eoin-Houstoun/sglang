#!/bin/sh
# Artemis commands for the end-to-end GLM-5.2 CPU benchmark:
#
#   sh benchmark/glm52_cpu/run_e2e.sh compile|test|bench
#
# Settings come from config.env next to this script (committed, GLM-5.2 on
# six sub-NUMA nodes by default), then $HOME/.artemis/glm52_cpu.env if it
# exists (machine-local: model path, TP, core binding), then the environment.
# The CPU environment is built on first use by the layer benchmark's
# setup_env.sh; SGLANG_CPU_VENV overrides its location.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
LAYER="$ROOT/benchmark/kernels/attention/dsa_cpu"
VENV=${SGLANG_CPU_VENV:-$HOME/.artemis/sglang-cpu-venv}
PY=$VENV/bin/python
if [ ! -x "$PY" ] || ! "$PY" -c "import torch, sgl_kernel" >/dev/null 2>&1; then
  sh "$LAYER/setup_env.sh" "$VENV"
fi
# Configuration precedence: environment > machine-local file > committed defaults.
_ENV_SNAPSHOT=$(export -p)
set -a
[ -f "$HERE/config.env" ] && . "$HERE/config.env"
[ -f "$HOME/.artemis/glm52_cpu.env" ] && . "$HOME/.artemis/glm52_cpu.env"
set +a
eval "$_ENV_SNAPSHOT"
export SGLANG_USE_CPU_ENGINE=1
export PYTHONPATH="$ROOT/python${PYTHONPATH:+:$PYTHONPATH}"
# Runtime libraries the CPU docs preload for serving, when present.
LIBDIR=${SGLANG_E2E_LIBDIR:-/usr/lib/x86_64-linux-gnu}
PRELOAD=""
for lib in "$VENV/lib/libiomp5.so" "$LIBDIR/libtcmalloc.so.4" "$LIBDIR/libtbbmalloc.so.2"; do
  [ -f "$lib" ] && PRELOAD="${PRELOAD:+$PRELOAD:}$lib"
done
[ -n "$PRELOAD" ] && export LD_PRELOAD="${LD_PRELOAD:+$LD_PRELOAD:}$PRELOAD" LD_LIBRARY_PATH="$LIBDIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd "$ROOT"
case "$1" in
  compile) exec "$PY" -c "import sglang.srt.layers.attention.dsa.dsa_cpu, sglang.launch_server" ;;
  test) exec sh "$LAYER/run.sh" test ;;
  bench) exec "$PY" benchmark/glm52_cpu/bench_e2e.py ;;
  *) echo "usage: run_e2e.sh compile|test|bench" >&2; exit 2 ;;
esac
