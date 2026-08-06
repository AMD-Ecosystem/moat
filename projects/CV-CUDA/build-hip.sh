#!/usr/bin/env bash
# CV-CUDA ROCm/HIP build (lead: linux-gfx90a). Configurable arch via ARCH env.
# Core libs + C++ gtest, no Python; cuOSD-backed ops scoped out at configure.
set -euo pipefail
SRC="$(cd "$(dirname "$0")/src" && pwd)"
ARCH="${ARCH:-gfx90a}"
BUILD="${BUILD:-$SRC/build-hip}"
JOBS="${JOBS:-16}"

cmake -S "$SRC" -B "$BUILD" -G Ninja \
  -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES="$ARCH" \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DBUILD_PYTHON=OFF -DBUILD_TESTS=ON -DBUILD_TESTS_CPP=ON \
  -DBUILD_TESTS_PYTHON=OFF -DBUILD_TESTS_WHEELS=OFF \
  -DBUILD_BENCH=OFF -DBUILD_DOCS=OFF -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ \
  -DPUBLIC_API_COMPILERS=

cmake --build "$BUILD" -j"$JOBS"
