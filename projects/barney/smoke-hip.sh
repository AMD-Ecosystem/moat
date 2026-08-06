#!/usr/bin/env bash
# Backend-1 smoke render on gfx90a: render one ANARI scene through the barney
# HIP device (the cuBQL software tracer) and confirm non-trivial output.
set -euo pipefail
MOAT=/var/lib/jenkins/moat
BUILD=$MOAT/projects/barney/src/build-hip
ANARI=$MOAT/_deps/anari-install
cd /tmp
rm -f anariTest.png
HIP_VISIBLE_DEVICES=0 \
LD_LIBRARY_PATH="$BUILD:$BUILD/anari:$ANARI/lib" \
  /tmp/anariTest
