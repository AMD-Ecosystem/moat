#!/usr/bin/env bash
# Repeatable HIP (ROCm) build for k2 on gfx90a. Run from anywhere.
# Usage: bash projects/k2/build_hip.sh [configure|build|all]   (default: all)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"          # MOAT repo root
SRC="$ROOT/projects/k2/src"
BUILD="$SRC/build"
LIBHIPCXX="$ROOT/_deps/libhipcxx/include"
ARCH="${K2_HIP_ARCH:-gfx90a}"
JOBS="${K2_JOBS:-16}"

step="${1:-all}"

export PYTHONPATH=""   # use the ROCm torch in the active env, not k2's source tree

if [ "$step" = "configure" ] || [ "$step" = "all" ]; then
  rm -rf "$BUILD"
  mkdir -p "$BUILD"
  # Do NOT pass CMAKE_HIP_COMPILER: CMake's HIP toolchain canonicalizes the
  # compiler path (/opt/rocm/lib/llvm/bin/clang++), and passing a different
  # spelling (/opt/rocm/llvm/bin/clang++) makes CMake "re-run configure and
  # reset variables", which silently flips K2_WITH_HIP back OFF.
  cmake -S "$SRC" -B "$BUILD" \
    -DCMAKE_BUILD_TYPE=Release \
    -DK2_WITH_HIP:BOOL=ON -DK2_WITH_CUDA:BOOL=OFF \
    -DCMAKE_HIP_ARCHITECTURES="$ARCH" \
    -DCMAKE_CXX_STANDARD=20 \
    -DK2_LIBHIPCXX_INCLUDE_DIR="$LIBHIPCXX" \
    -DPYTHON_EXECUTABLE="$(which python3)" \
    -DK2_ENABLE_TESTS=ON -DK2_ENABLE_BENCHMARK=OFF
fi

if [ "$step" = "build" ] || [ "$step" = "all" ]; then
  cmake --build "$BUILD" -j "$JOBS"
fi
