#!/usr/bin/env bash
# Build the llama.cpp variants ArmForge benchmarks against.
#
# Three builds, because "which kernels ran" is the question ArmForge exists to
# answer and you cannot answer it with one binary:
#
#   cpu          ggml's own Arm CPU kernels only. The controlled baseline.
#   kleidiai     ggml + Arm's KleidiAI micro-kernels (SME2/i8mm paths).
#   accelerate   Apple's Accelerate BLAS. What a default macOS build gives you,
#                included because it is the real-world reference point -- but it
#                bypasses ggml's Arm kernels for prefill, so it must never be
#                compared against the other two as if it were the same thing.
#
# Metal is disabled everywhere: ArmForge measures the CPU.
#
# Usage:  scripts/setup-llama-cpp.sh [--jobs N] [--repo-dir DIR]

set -euo pipefail

CACHE_DIR="${ARMFORGE_CACHE:-$HOME/.cache/armforge}"
REPO_DIR="$CACHE_DIR/llama.cpp"
JOBS="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"
LLAMA_REPO="https://github.com/ggml-org/llama.cpp.git"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --jobs) JOBS="$2"; shift 2 ;;
        --repo-dir) REPO_DIR="$2"; shift 2 ;;
        -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

command -v cmake >/dev/null 2>&1 || {
    echo "error: cmake not found. Install it (brew install cmake / apt install cmake)." >&2
    exit 1
}

if [[ ! -d "$REPO_DIR/.git" ]]; then
    echo "==> cloning llama.cpp into $REPO_DIR"
    mkdir -p "$(dirname "$REPO_DIR")"
    git clone --depth 1 "$LLAMA_REPO" "$REPO_DIR"
fi

COMMIT="$(git -C "$REPO_DIR" rev-parse --short HEAD)"
echo "==> llama.cpp at $COMMIT"

# variant : extra cmake flags
build_variant() {
    local name="$1"; shift
    local build_dir="$REPO_DIR/build-$name"

    echo "==> configuring $name"
    cmake -S "$REPO_DIR" -B "$build_dir" \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_METAL=OFF \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=OFF \
        -DLLAMA_BUILD_SERVER=OFF \
        "$@" >"$build_dir.configure.log" 2>&1 || {
            echo "!! configure failed for $name; tail of log:" >&2
            tail -30 "$build_dir.configure.log" >&2
            return 1
        }

    echo "==> building $name (-j $JOBS)"
    # llama-completion is built alongside llama-bench purely as a capability
    # probe: it prints ggml's compiled-in feature flags on startup
    # ("CPU : NEON = 1 | MATMUL_INT8 = 1 | SME = 1 | REPACK = 1 |"), which
    # llama-bench does not. llama-cli would do too, but it is gated behind
    # LLAMA_BUILD_SERVER and would drag in the server and web UI.
    cmake --build "$build_dir" --target llama-bench llama-completion -j "$JOBS" \
        >"$build_dir.build.log" 2>&1 || {
            echo "!! build failed for $name; tail of log:" >&2
            tail -30 "$build_dir.build.log" >&2
            return 1
        }
    echo "    ok: $build_dir/bin/llama-bench"
}

build_variant cpu        -DGGML_CPU_KLEIDIAI=OFF -DGGML_ACCELERATE=OFF -DGGML_BLAS=OFF
build_variant kleidiai   -DGGML_CPU_KLEIDIAI=ON  -DGGML_ACCELERATE=OFF -DGGML_BLAS=OFF
build_variant accelerate -DGGML_CPU_KLEIDIAI=OFF -DGGML_ACCELERATE=ON

echo
echo "==> built variants:"
for d in "$REPO_DIR"/build-*/; do
    [[ -x "$d/bin/llama-bench" ]] && echo "    $(basename "$d")"
done
echo
echo "Run 'armforge runtimes' to see what ArmForge discovered."
