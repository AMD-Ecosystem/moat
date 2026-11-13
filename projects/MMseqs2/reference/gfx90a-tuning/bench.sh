#!/bin/bash
# bench.sh <q1k|t8> <variant>...   variants: generic tuned (builds b-<variant>)
ulimit -c 0
export HIP_VISIBLE_DEVICES=1 CUDA_VISIBLE_DEVICES=0
A=<moat>/agent_space
cd $A/bench
set=$1; shift
if [ $set = q1k ]; then Q=q1k; T=t; else Q=q200; T=t8; fi
L=$A/bench_$set.txt
for v in "$@"; do
  B=$A/b-$v/src/mmseqs; rm -rf p_${set}_$v* o_${set}_$v*
  echo "$(date -u +%FT%TZ) $v load: $(cut -d' ' -f1-3 /proc/loadavg)" >> $L
  s=$(date +%s.%N)
  $B ungappedprefilter $Q $T p_${set}_$v --gpu 1 --max-seqs 1000 -e 10000 --threads 8 > log_${set}_$v 2>&1; rc=$?
  e=$(date +%s.%N)
  $B createtsv $Q $T p_${set}_$v o_${set}_$v.tsv --threads 8 >/dev/null 2>&1
  echo "$v rc=$rc wall=$(echo "$e - $s" | bc) s rows=$(wc -l < o_${set}_$v.tsv) md5sorted=$(sort o_${set}_$v.tsv | md5sum | cut -c1-32)" >> $L
done
