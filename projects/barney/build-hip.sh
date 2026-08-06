#!/usr/bin/env bash
# Repeatable Backend-1 (software tracer -> HIP) build for barney on Linux gfx90a.
# Consumes the ported cuBQL from _deps/cuBQL (AMD-Ecosystem/cuBQL @ moat-port).
set -euo pipefail

MOAT=/var/lib/jenkins/moat
SRC=$MOAT/projects/barney/src
BUILD=$SRC/build-hip
CUBQL=$MOAT/_deps/cuBQL
ARCH=${CMAKE_HIP_ARCHITECTURES:-gfx90a}

export ROCM_PATH=${ROCM_PATH:-/opt/rocm}
export HIP_PATH=${HIP_PATH:-/opt/rocm}

cmake -S "$SRC" -B "$BUILD" \
  -DUSE_HIP=ON \
  -DBARNEY_USE_EXTERNAL_CUBQL=ON \
  -DBARNEY_EXTERNAL_CUBQL_DIR="$CUBQL" \
  -DCMAKE_HIP_ARCHITECTURES="$ARCH" \
  -DCMAKE_BUILD_TYPE=Release \
  "$@"

cmake --build "$BUILD" -j"$(nproc)"
