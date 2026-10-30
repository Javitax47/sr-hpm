#!/usr/bin/env bash
# SR-HPM installer for Linux: llama.cpp at the pinned commit + the SR-HPM patch, built natively (-march=native) and
# with CUDA when nvcc is available, plus the Python launcher's dependencies.
# Requirements: git, cmake >= 3.24, a C++17 compiler (gcc >= 11 or clang >= 14), python3 >= 3.10.
# Optional: CUDA toolkit 12.x. (The dGPU co-verifier is Windows-only for now.)
#
#   bash scripts/install_linux.sh [DEST] [--no-cuda]
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${1:-$REPO/llama.cpp-srhpm}"
[ "${1:-}" = "--no-cuda" ] && DEST="$REPO/llama.cpp-srhpm"
CUDA=1
for a in "$@"; do [ "$a" = "--no-cuda" ] && CUDA=0; done
COMMIT="$(head -1 "$REPO/patch/UPSTREAM_COMMIT" | tr -d '[:space:]')"
JOBS="$(nproc)"
step() { printf '\033[36m==> %s\033[0m\n' "$*"; }

step "llama.cpp $COMMIT -> $DEST"
[ -d "$DEST/.git" ] || git clone https://github.com/ggml-org/llama.cpp.git "$DEST"
git -C "$DEST" fetch --quiet origin "$COMMIT"
git -C "$DEST" checkout --quiet --force "$COMMIT"
git -C "$DEST" clean -fdq -e build
step "applying the SR-HPM patch"
git -C "$DEST" apply --whitespace=nowarn "$REPO/patch/srhpm.patch"

ARGS=(-S "$DEST" -B "$DEST/build" -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DGGML_NATIVE=ON)
if [ $CUDA = 1 ] && command -v nvcc >/dev/null; then
  ARGS+=(-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native)
  step "configure (CUDA)"
else
  step "configure (CPU only)"
fi
cmake "${ARGS[@]}"
step "build"
cmake --build "$DEST/build" -j "$JOBS" --target llama-server llama-cli llama-bench llama-quantize llama-perplexity

step "Python dependencies"
python3 -m pip install --quiet -r "$REPO/requirements.txt"

printf '\n\033[32mSR-HPM is ready: %s/build/bin/llama-server\033[0m\n' "$DEST"
echo "  python3 -m srhpm.launch <model.gguf> [--draft <drafter.gguf>] --run"
echo "  (export SRHPM_LLAMA_DIR=$DEST if you installed outside the repository)"
echo "  Transparent huge pages: echo madvise | sudo tee /sys/kernel/mm/transparent_hugepage/enabled"
