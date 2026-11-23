#!/usr/bin/env bash
# Run every case in its own process. usage: run_all.sh <python> <builddir> [case ...]
# Prints the RESULT lines and a summary; exit status is the number of FAIL cases
# (SKIP is not a failure, it is reported).
PY=${1:?python}; OUT=${2:?builddir}; shift 2
HERE=$(cd "$(dirname "$0")" && pwd)
cases=${*:-t1_lif t2_poisson_delay t3_recurrent t4_timedarray t5_poisson_lambda t6_binomial t7_int_math}
mkdir -p "$OUT"
fails=0
for c in $cases; do
  ( cd "$HERE" && timeout 1800 "$PY" "$c.py" "$OUT" ) > "$OUT/$c.log" 2>&1
  rc=$?
  line=$(grep '^RESULT' "$OUT/$c.log" | tail -1)
  [ -n "$line" ] || line="RESULT $c FAIL rc=$rc (no result line; see $OUT/$c.log)"
  echo "$line"
  case "$line" in *" FAIL "*) fails=$((fails+1));; esac
done
echo "failed: $fails"
exit $fails
