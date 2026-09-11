#!/bin/sh
# Build the CPU environment for the DSA benchmark once: SGLang's CPU
# dependencies (pyproject_cpu.toml) and the CPU sgl-kernel, following
# docs/hardware-platforms/cpu_server.mdx. Needs uv (installed if missing),
# gcc-13 or a recent gcc, cmake, ninja, libnuma-dev and libtbb-dev.
#
#   sh benchmark/kernels/attention/dsa_cpu/setup_env.sh <venv path>
set -e
VENV=${1:?venv path}
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../../../.." && pwd)
export PATH="$HOME/.local/bin:$PATH"
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
"$VENV/bin/python" -c "import torch, sgl_kernel; print('[dsa_cpu] environment ready: torch', torch.__version__)" >&2
