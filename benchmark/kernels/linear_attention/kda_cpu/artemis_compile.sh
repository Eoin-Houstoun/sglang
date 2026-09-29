#!/usr/bin/env bash
set -euo pipefail

VENV="${SGLANG_CPU_VENV:-$HOME/.artemis/sglang-cpu-venv}"
CACHE="${ARTEMIS_CACHE_ROOT:-$HOME/.artemis/sglang-kda-cpu-cache}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"

# Builds the environment on first use; a no-op in about a second once it exists.
sh "$(dirname "${BASH_SOURCE[0]}")/setup_env.sh" "$VENV"
mkdir -p "$CACHE/src" "$CACHE/build"

# Compare by content, not mtime, and stamp changed files now: ninja rebuilds by mtime, and a
# runner workspace can carry any timestamps, so this rebuilds exactly the files a version changed.
rsync -rl --checksum --delete "$ROOT/python/sglang/kernels/aot/" "$CACHE/src/"

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
  "$ROOT/python/sglang/kernels/ops/attention/fla/kda_cpu.py" \
  "$ROOT/python/sglang/srt/layers/attention/linear/kernels/kda_triton.py" \
  "$ROOT/benchmark/kernels/linear_attention/kda_cpu/"*.py
SGLANG_USE_CPU_ENGINE=1 PYTHONPATH="$ROOT/python" "$VENV/bin/python" -c \
  "import sgl_kernel, torch; from sglang.kernels.ops.attention.fla.kda_cpu import chunk_kda"
