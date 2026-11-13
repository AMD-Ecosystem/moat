#!/bin/bash
ulimit -c 0
export HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0
M=/var/lib/jenkins/fleet/jobs/gfx1201-20261112-201654-mmseqs-gfx1201/moat/agent_space/mm
cd $M/bench
L=$M/raw/bench_t8.log
for v in "$@"; do
  B=$M/build-$v/src/mmseqs; rm -rf p8_$v* o8_$v*
  echo "$(date -u +%FT%TZ) $v load: $(cut -d' ' -f1-3 /proc/loadavg)" >> $L
  s=$(date +%s.%N)
  $B ungappedprefilter q200 t8 p8_$v --gpu 1 --max-seqs 1000 -e 10000 --threads 8 > log8_$v 2>&1; rc=$?
  e=$(date +%s.%N)
  $B createtsv q200 t8 p8_$v o8_$v.tsv --threads 8 >/dev/null 2>&1
  echo "$v rc=$rc wall=$(echo "$e - $s" | bc) s rows=$(wc -l < o8_$v.tsv) md5sorted=$(sort o8_$v.tsv | md5sum | cut -c1-32)" >> $L
done
