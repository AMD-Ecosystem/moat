#!/usr/bin/env bash
# Generate every case with two brian2cuda trees and diff the generated projects.
# usage: compare.sh <python> <old_tree> <new_tree> <outdir> [cuda|hip]
# For `cuda` on a host without nvcc, put a stub nvcc first on PATH and leave
# USE_HIP unset. Exit status: number of cases whose generated lines differ
# (a pure reordering of lines is reported but not counted).
PY=$1; OLD=$2; NEW=$3; OUT=$4; BACKEND=${5:-cuda}
HERE=$(cd "$(dirname "$0")" && pwd)
# fixed hash seed: codegen orders some lines by set iteration
export SETUPTOOLS_SCM_PRETEND_VERSION=1.0.0 PYTHONHASHSEED=0
diffs=0
cd /tmp
for c in smoke cuba stdp timed_poisson binomial spatial; do
  for t in old new; do
    tree=$OLD; [ $t = new ] && tree=$NEW
    rm -rf "$OUT/$t/$c"; mkdir -p "$OUT/$t"
    PYTHONPATH=$tree "$PY" -P "$HERE/gen_case.py" $c "$OUT/$t/$c" $BACKEND > "$OUT/$t/$c.log" 2>&1 \
      || { echo "$c: generation failed with $t tree (see $OUT/$t/$c.log)"; diffs=$((diffs+1)); continue 2; }
  done
  if diff -r -q "$OUT/old/$c" "$OUT/new/$c" > "$OUT/$c.diff.txt"; then
    echo "$c: generated text identical ($(find "$OUT/new/$c" -type f | wc -l) files)"
  else
    # codegen orders some lines by iterating sets of objects, which can change
    # between two runs of the same tree; compare each differing file's sorted lines
    content=0
    while read -r _ a _ b _; do
      if [ -f "$a" ] && [ -f "$b" ] && cmp -s <(sort "$a") <(sort "$b"); then
        echo "$c: same lines, different order: ${b#$OUT/new/}"
      else
        echo "$c: CONTENT DIFFERS: $a $b"; content=1
      fi
    done < "$OUT/$c.diff.txt"
    if [ $content = 0 ]; then
      echo "$c: generated text identical up to line order"
    else
      diffs=$((diffs+1))
    fi
  fi
done
exit $diffs
