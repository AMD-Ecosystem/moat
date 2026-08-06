#!/usr/bin/env bash
# Backend-2 smoke render on gfx90a: render ANARI scenes through the HIPRT-backed
# barney device and confirm non-trivial output. /tmp/anariTest and /tmp/test_sphere
# are the validator's harnesses (anariTest.cpp + a sphere user-geom scene),
# compiled against the ANARI SDK; they dlopen whichever libanari_library_barney.so
# LD_LIBRARY_PATH resolves -- here the HIPRT build's.
set -euo pipefail
MOAT=/var/lib/jenkins/moat
BUILD=$MOAT/projects/barney/src/build-hiprt
ANARI=$MOAT/_deps/anari-install
HIPRT_ROOT=${HIPRT_ROOT:-$MOAT/agent_space/hiprt_probe/HIPRT}
cd /tmp
rm -f anariTest.png
HIP_VISIBLE_DEVICES=0 HIPRT_PATH=$HIPRT_ROOT \
LD_LIBRARY_PATH="$BUILD:$BUILD/anari:$ANARI/lib:$HIPRT_ROOT/dist/bin/Release" \
  /tmp/anariTest
echo "--- triangle scene -> /tmp/anariTest.png"
HIP_VISIBLE_DEVICES=0 HIPRT_PATH=$HIPRT_ROOT \
LD_LIBRARY_PATH="$BUILD:$BUILD/anari:$ANARI/lib:$HIPRT_ROOT/dist/bin/Release" \
  /tmp/test_sphere
echo "--- sphere (user geom) scene -> /tmp/test_sphere.png"
