#!/bin/bash
# Build anari_validate and run every scene with both renderers, one scene per
# process, under a timeout. See README.md in this directory.
#
# usage: run_all.sh <anari-install> <anari-visionaray-install> <fork-src> [outdir]
#
# env: RUNS (default 8), TIMEOUT seconds per process (default 600),
#      SCENES / RENDERERS (space-separated overrides), CXX (default c++).
set -u
ANARI=$1; AV=$2; SRC=$3; OUT=${4:-$PWD/gpu_tests_out}
RUNS=${RUNS:-8}; TIMEOUT=${TIMEOUT:-600}
# surface runs last: a case that hangs the GPU then cannot cost the other results.
SCENES=${SCENES:-"volume-nearest volume-linear nanovdb-nearest nanovdb-linear gaussian-splat surface"}
RENDERERS=${RENDERERS:-"default raycast"}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"

${CXX:-c++} -O2 -std=c++17 "$HERE/anari_validate.cpp" -I"$ANARI/include" -I"$SRC/external" \
  -L"$ANARI/lib" -lanari -Wl,-rpath,"$ANARI/lib" -o "$OUT/anari_validate" || exit 2

ulimit -c 0
export ANARI_LIBRARY_PATH=$AV/lib
export LD_LIBRARY_PATH=$AV/lib:$ANARI/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
echo "HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-<unset>} RUNS=$RUNS TIMEOUT=$TIMEOUT"
fails=0
for r in $RENDERERS; do
  for sc in $SCENES; do
    # The CPU device crashes on Gaussian splats in upstream's own code, so the
    # splat scene has no CPU reference.
    devs="visionaray_hip visionaray"; [ "$sc" = gaussian-splat ] && devs=visionaray_hip
    log=$OUT/$r-$sc.log
    t0=$(date +%s)
    RENDERER=$r ONLY=$sc RUNS=$RUNS PPM_PREFIX=$OUT/$r-$sc \
      timeout -k 30 "$TIMEOUT" "$OUT/anari_validate" $devs > "$log" 2>&1
    rc=$?
    t1=$(date +%s)
    case $rc in
      0) verdict=PASS ;;
      124|137) verdict="FAIL (timeout ${TIMEOUT}s)" ;;
      *) verdict="FAIL (rc=$rc)" ;;
    esac
    [ $rc -ne 0 ] && fails=$((fails+1))
    echo "=== renderer=$r scene=$sc: $verdict, $((t1-t0)) s"
    grep -E '^\[' "$log"
  done
done
echo "TOTAL: $fails failing cases"
[ $fails -eq 0 ]
