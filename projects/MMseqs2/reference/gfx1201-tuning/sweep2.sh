#!/bin/bash
ulimit -c 0
export HIP_VISIBLE_DEVICES=0
M=/var/lib/jenkins/fleet/jobs/gfx1201-20261112-201654-mmseqs-gfx1201/moat/agent_space/mm
T=$M/build-tune/tuning_gapless
run(){ name=$1; bin=$2; shift 2; s=$(date +%s); echo "$(date -u +%FT%TZ) start $name: $bin $*" >> $M/raw/sweeps.log
  $bin "$@" --printBestConfigs --sleep 10 > $M/raw/$name.csv 2> $M/raw/$name.err; rc=$?
  echo "$(date -u +%FT%TZ) end $name rc=$rc $(( $(date +%s)-s )) s" >> $M/raw/sweeps.log; }
for job in "$@"; do
  case $job in
    f32_t4) run f32_t4 $T --float --int32 --tiles 4;;
    endpos_t4_a) run endpos_t4_a $M/build-narrow-a/tuning_gapless --float --int32 --endpos --tiles 4;;
    endpos_t4_b) run endpos_t4_b $M/build-narrow-b/tuning_gapless --float --int32 --endpos --tiles 4;;
  esac
done
