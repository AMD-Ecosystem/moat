#!/bin/bash
ulimit -c 0
export HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0
M=/var/lib/jenkins/fleet/jobs/gfx1201-20261112-201654-mmseqs-gfx1201/moat/agent_space/mm
cd $M/bench
L=$M/raw/bench_q1k.log
for v in "$@"; do
  B=$M/build-$v/src/mmseqs; d=run_$v; rm -rf $d p_$v* out_$v*; mkdir -p $d
  echo "$(date -u +%FT%TZ) $v load: $(cut -d' ' -f1-3 /proc/loadavg)" >> $L
  s=$(date +%s.%N)
  $B ungappedprefilter q1k t p_$v --gpu 1 --max-seqs 1000 -e 10000 --threads 8 > $d/log 2>&1; rc=$?
  e=$(date +%s.%N)
  $B createtsv q1k t p_$v out_$v.tsv --threads 8 >/dev/null 2>&1
  echo "$v rc=$rc wall=$(echo "$e - $s" | bc) s rows=$(wc -l < out_$v.tsv) md5sorted=$(sort out_$v.tsv | md5sum | cut -c1-32)" >> $L
done
