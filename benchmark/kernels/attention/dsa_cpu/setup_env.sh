#!/bin/sh
# Build the CPU environment for the DSA benchmark once: SGLang's CPU
# dependencies (pyproject_cpu.toml) and the CPU sgl-kernel, following
# docs/hardware-platforms/cpu_server.mdx. Needs uv (installed if missing),
# gcc-13 or a recent gcc, cmake, ninja, libnuma-dev and libtbb-dev.
#
#   sh benchmark/kernels/attention/dsa_cpu/setup_env.sh [venv path]
#
# Run this once per machine before the benchmark's three commands. It takes
# about ten minutes on a 32-core Xeon and is the only step that needs the
# build toolchain or the network.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../../../.." && pwd)
VENV=${1:-${SGLANG_CPU_VENV:-$HOME/.artemis/sglang-cpu-venv}}
export PATH="$HOME/.local/bin:$PATH"

# Already built: re-running this is a no-op, so it is safe to put in front of
# anything. Delete the directory (or pass --force) to rebuild after changing
# pyproject_cpu.toml or the CPU kernel.
if [ "$2" != "--force" ] && [ -x "$VENV/bin/python" ] &&
   "$VENV/bin/python" -c "import torch, sgl_kernel" >/dev/null 2>&1; then
  echo "[dsa_cpu] environment already built at $VENV" >&2
  exit 0
fi

# Preflight. Everything below this point assumes a toolchain and an index to
# install from; say which piece is missing rather than failing inside cmake.
MISSING=""
# tbb/tbb.h is a C++ header and pulls in the C++ standard library, so each
# header is probed with the compiler that will actually consume it.
have_c_header() { echo "#include <$1>" | ${CC:-cc} -E -x c - >/dev/null 2>&1; }
have_cxx_header() { echo "#include <$1>" | ${CXX:-c++} -E -x c++ - >/dev/null 2>&1; }
for tool in cmake ninja; do
  command -v "$tool" >/dev/null 2>&1 || MISSING="$MISSING $tool"
done
command -v gcc >/dev/null 2>&1 || command -v cc >/dev/null 2>&1 || MISSING="$MISSING gcc"
command -v c++ >/dev/null 2>&1 || command -v g++ >/dev/null 2>&1 || MISSING="$MISSING g++"
have_c_header numa.h || MISSING="$MISSING libnuma-dev(numa.h)"
have_cxx_header tbb/tbb.h || MISSING="$MISSING libtbb-dev(tbb/tbb.h)"
if [ -n "$MISSING" ]; then
  echo "[dsa_cpu] missing build prerequisites:$MISSING" >&2
  echo "[dsa_cpu] Debian/Ubuntu: apt-get install -y build-essential cmake ninja-build libnuma-dev libtbb-dev" >&2
  echo "[dsa_cpu] RHEL/Fedora:   dnf install -y gcc-c++ cmake ninja-build numactl-devel tbb-devel" >&2
  exit 1
fi
for url in https://pypi.org/simple/ https://download.pytorch.org/whl/cpu; do
  curl -fsS --max-time 20 -o /dev/null "$url" 2>/dev/null || {
    echo "[dsa_cpu] cannot reach $url" >&2
    echo "[dsa_cpu] behind a proxy, export https_proxy/http_proxy, or point uv at a mirror with UV_INDEX_URL" >&2
    exit 1
  }
done
if ! command -v gcc-13 >/dev/null 2>&1; then
  echo "[dsa_cpu] note: gcc-13 not found, building with $(${CC:-cc} --version | head -1)" >&2
fi

command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
mkdir -p "$(dirname "$VENV")"
[ -d "$VENV" ] || uv venv --python 3.12 "$VENV"
cat > "$VENV/uv.toml" <<'EOF'
[[index]]
name = "torch"
url = "https://download.pytorch.org/whl/cpu"

[[index]]
name = "torchvision"
url = "https://download.pytorch.org/whl/cpu"

[[index]]
name = "torchaudio"
url = "https://download.pytorch.org/whl/cpu"

[[index]]
name = "triton"
url = "https://download.pytorch.org/whl/cpu"
EOF
export UV_CONFIG_FILE="$VENV/uv.toml" VIRTUAL_ENV="$VENV"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

echo "[dsa_cpu] installing SGLang CPU dependencies into $VENV" >&2
cp -r "$ROOT/python/." "$TMP/pkg"
cd "$TMP/pkg" && cp pyproject_cpu.toml pyproject.toml
uv pip install --upgrade pip setuptools
uv pip install .
uv pip uninstall sglang >/dev/null 2>&1 || true

echo "[dsa_cpu] building the CPU sgl-kernel (a few minutes)" >&2
cp -r "$ROOT/python/sglang/kernels/aot/." "$TMP/kernel"
cd "$TMP/kernel" && cp pyproject_cpu.toml pyproject.toml
if command -v gcc-13 >/dev/null 2>&1; then CC=${CC:-gcc-13}; CXX=${CXX:-g++-13}; fi
CC=${CC:-gcc} CXX=${CXX:-g++} CMAKE_BUILD_PARALLEL_LEVEL=${CMAKE_BUILD_PARALLEL_LEVEL:-$(nproc)} uv pip install .
git -C "$ROOT" rev-parse HEAD > "$VENV/.sglang-commit" 2>/dev/null || true
"$VENV/bin/python" -c "import torch, sgl_kernel; print('[dsa_cpu] environment ready: torch', torch.__version__)" >&2
echo "[dsa_cpu] environment at $VENV; rebuild it after changing pyproject_cpu.toml or the CPU kernel" >&2
