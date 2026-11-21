#!/bin/zsh
# Reproducible build for the _ane_linear nanobind extension.
# Requirements: macOS, project venv with MLX (uv sync), cmake >= 3.27, Xcode toolchain.
# nanobind is NOT a project dependency: it is installed into a scratch dir, pinned to
# the ABI the installed MLX needs (CMakeLists.txt checks the pairing and fails fast).
set -e
STAGE3=$(cd "$(dirname "$0")/.." && pwd)          # spikes/ane_matmul/stage3b
SPIKE=$(cd "$STAGE3/.." && pwd)                    # spikes/ane_matmul
REPO=$(cd "$SPIKE/../.." && pwd)
PY=$REPO/.venv/bin/python
SRC=$SPIKE/ane_linear
BUILD=$SRC/build
SCRATCH=${ANE_NANOBIND_DIR:-/tmp/ane_build/nanobind}

MLX_VER=$($PY -c "import mlx.core as mx; print(mx.__version__)")
case "$MLX_VER" in
  0.32.4|0.32.5|0.32.6) NB_PIN=3.1.0 ;;
  0.32.[0-3]) NB_PIN=2.14.2 ;;
  *) echo "MLX $MLX_VER: pick the nanobind pin in this script per CMakeLists.txt rules" ; exit 1 ;;
esac

# version-scoped cache: a stale dir from another MLX pin must never be reused
SCRATCH_NB="$SCRATCH/nb-$NB_PIN"
if [ ! -d "$SCRATCH_NB/nanobind" ]; then
  mkdir -p "$SCRATCH_NB"
  uv pip install --target "$SCRATCH_NB" "nanobind==$NB_PIN"
fi

MLX_DIR=$($PY -c "
import glob, site
for d in site.getsitepackages():
    g = glob.glob(d + '/mlx/share/cmake/MLX')
    if g:
        print(g[0])
        break
")
[ -d "$MLX_DIR" ] || { echo "MLX cmake dir not found at $MLX_DIR"; exit 1; }

mkdir -p "$BUILD"
PYTHONPATH="$SCRATCH_NB" cmake -S "$SRC" -B "$BUILD" \
  -DPython_EXECUTABLE="$PY" \
  -DMLX_DIR="$MLX_DIR" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD" -j

# Place the .so next to the sources so `import _ane_linear` resolves via the spike path.
cp "$BUILD"/_ane_linear.cpython-*.so "$SRC/"
echo "Built:"
ls -la "$SRC"/_ane_linear.cpython-*.so
$PY -c "import sys; sys.path.insert(0, '$SRC'); import _ane_linear; print('import OK'); print(_ane_linear.ane_linear_q8.__doc__ or 'q8 binding present')"
