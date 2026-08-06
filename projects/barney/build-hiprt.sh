#!/usr/bin/env bash
# Backend-2 build: barney with the HIPRT (AMD hardware-RT) rtcore backend on
# Linux gfx90a (ROCm 7.2.1). HIPRT supplies BVH build + traversal; barney keeps
# its function-pointer program dispatch. See notes.md "Backend 2".
#
# Prereqs:
#   - submodules/owl inited
#   - _deps/cuBQL = AMD-Ecosystem/cuBQL @ moat-port (barney_config still links it)
#   - HIPRT built from source -DBITCODE=OFF (pinned commit cb09c56, API 3001);
#     point hiprt_ROOT at the HIPRT checkout (its dist/bin/Release holds
#     libhiprt0300164.so). See notes.md "HIPRT dependency" for the recipe.
#   - ANARI SDK installed (anari_DIR) to also build the ANARI device + smoke.
set -euo pipefail
MOAT=/var/lib/jenkins/moat
SRC=$MOAT/projects/barney/src
BUILD=$SRC/build-hiprt
HIPRT_ROOT=${HIPRT_ROOT:-$MOAT/agent_space/hiprt_probe/HIPRT}
CUBQL=${CUBQL:-$MOAT/_deps/cuBQL}
ANARI=${ANARI:-$MOAT/_deps/anari-install}

HIP_VISIBLE_DEVICES=0 cmake -S "$SRC" -B "$BUILD" \
  -DUSE_HIP=ON \
  -DBARNEY_BACKEND_HIPRT=ON \
  -Dhiprt_ROOT="$HIPRT_ROOT" \
  -DBARNEY_USE_EXTERNAL_CUBQL=ON \
  -DBARNEY_EXTERNAL_CUBQL_DIR="$CUBQL" \
  -Danari_DIR="$ANARI/lib/cmake/anari-0.16.0" \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a -DCMAKE_BUILD_TYPE=Release

HIP_VISIBLE_DEVICES=0 cmake --build "$BUILD" -j"$(nproc)"
