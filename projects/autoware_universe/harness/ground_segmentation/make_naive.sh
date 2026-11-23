#!/bin/bash
# naive.cu: the fork's kernel with the group fix undone, as a plain hipify would leave it
set -e
src=$1; out=$2
sed -e 's/, WARP_SIZE)/)/' \
    -e 's/__ballot_sync(FULL_MASK, predicate) >> (__lane_id() & ~(WARP_SIZE - 1))/__ballot_sync(FULL_MASK, predicate)/' "$src" > "$out"
grep -c 'WARP_SIZE)' "$out" || true
