#!/usr/bin/env bash
set -euo pipefail

VENV="${SGLANG_CPU_VENV:-$HOME/.artemis/sglang-cpu-venv}"
CACHE="${ARTEMIS_CACHE_ROOT:-$HOME/.artemis/sglang-dsa-cpu-cache}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"

test -x "$VENV/bin/python"
mkdir -p "$CACHE/src" "$CACHE/build"

rsync -a --delete "$ROOT/python/sglang/kernels/aot/" "$CACHE/src/"

export VIRTUAL_ENV="$VENV"
export PATH="$VENV/bin:$PATH"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-$(nproc)}"

cmake -S "$CACHE/src/csrc/cpu" -B "$CACHE/build" \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython_EXECUTABLE="$VENV/bin/python"
cmake --build "$CACHE/build" --parallel "$CMAKE_BUILD_PARALLEL_LEVEL"

SITE_PACKAGES="$("$VENV/bin/python" -c \
  'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
cmake --install "$CACHE/build" --prefix "$SITE_PACKAGES"

PYTHONPATH="$ROOT/python" "$VENV/bin/python" -m py_compile \
  "$ROOT/python/sglang/srt/layers/attention/dsa/dsa_cpu.py" \
  "$ROOT/python/sglang/srt/layers/attention/dsa/dsa_indexer.py" \
  "$ROOT/benchmark/kernels/attention/dsa_cpu/"*.py
"$VENV/bin/python" -c \
  "import sgl_kernel, torch; assert hasattr(torch.ops.sgl_kernel, 'dsa_indexer_topk_cpu')"
