#!/usr/bin/env bash
# Validate the full EGGROLL HIP port (the transformer / multi-GPU Muon / d-eggs
# distributed trainers, now folded into the moat-port branch) on a real AMD GPU.
# egg.c is pipeline-tracked, so a gfx90a host can also pick this up via
# /port-next (egg.c -> revalidate); this script runs all components in one shot.
#
# Usage:   bash projects/egg.c/validate_advanced_rocm.sh <hiparch> [ngpu]
# Example: bash projects/egg.c/validate_advanced_rocm.sh gfx90a 4
#
# Run from the MOAT repo root. Assumes the fork clone is at projects/egg.c/src.
# It checks out moat-port and builds/runs each component for <hiparch>.
# Decisive wave64 (gfx90a) check: the width-32 logical-warp reductions must keep
# the loss CONVERGING -- a wrong-width reduction folds two perturbations together
# on a 64-lane wavefront and the loss diverges or stalls.
set -u
ARCH="${1:-gfx90a}"
NGPU="${2:-1}"
SRC="projects/egg.c/src"
STEP_BUDGET="${STEP_BUDGET:-420}"   # seconds per trainer run; trainers are slow at default population
PASS=0; FAIL=0
say(){ printf '\n=== %s ===\n' "$*"; }
verdict(){ if [ "$1" = ok ]; then PASS=$((PASS+1)); echo "RESULT: PASS - $2"; else FAIL=$((FAIL+1)); echo "RESULT: FAIL - $2"; fi; }

cd "$(git rev-parse --show-toplevel)" || exit 1
git -C "$SRC" fetch origin --quiet
git -C "$SRC" checkout moat-port 2>/dev/null || git -C "$SRC" checkout -b moat-port origin/moat-port
git -C "$SRC" reset --hard origin/moat-port
echo "branch HEAD: $(git -C "$SRC" rev-parse --short HEAD)  arch=$ARCH  ngpu=$NGPU"
mkdir -p agent_space/egg_adv_val && cd agent_space/egg_adv_val
python3 -c "open('input.txt','w').write(('The quick brown fox jumps over the lazy dog. '*1000)[:18000])"
SRCABS="$(git -C ../../$SRC rev-parse --show-toplevel)/$SRC" 2>/dev/null || SRCABS="../../$SRC"

# loss_decreases <logfile> : true if the captured "Loss:" sequence ends below it starts
loss_decreases(){ awk '/Loss:/{match($0,/Loss:[ ]*[-0-9.]+/);s=substr($0,RSTART+5);gsub(/[^0-9.\-]/,"",s);v=s+0;if(n==0)first=v;last=v;n++} END{exit !(n>=2 && last<first)}' "$1"; }

run_trainer(){ # name srcfile extra_hipcc_flags extra_run_env
  local name="$1" file="$2" flags="$3" runenv="$4"
  say "$name ($file) -- build for $ARCH"
  hipcc -O3 --offload-arch="$ARCH" -x hip "$SRCABS/$file" $flags -o "./$name" 2>build_$name.log
  if [ ! -x "./$name" ]; then echo "BUILD FAILED (see build_$name.log):"; tail -5 build_$name.log; verdict fail "$name build"; return; fi
  say "$name -- train (EGG_FIXED_SEED=12345, ${STEP_BUDGET}s budget)"
  env $runenv EGG_FIXED_SEED=12345 timeout "$STEP_BUDGET" stdbuf -oL "./$name" > run_$name.log 2>&1
  grep -m6 'Loss:' run_$name.log || true
  if grep -q 'Loss:' run_$name.log && loss_decreases run_$name.log && ! grep -qiE 'nan|illegal|fault|HSA_STATUS' run_$name.log; then
    verdict ok "$name: loss converges on $ARCH (no NaN/fault)"
  else
    echo "--- tail ---"; tail -8 run_$name.log; verdict fail "$name: no monotone loss decrease / NaN / fault"
  fi
}

# 1+2: single-GPU transformer trainers (width-32 head reductions; adam also exercises the __dp4a shim)
run_trainer egg_xf            full_cuda_train_egg_transformer.cu        ""           "HIP_VISIBLE_DEVICES=0"
run_trainer egg_xf_adam       full_cuda_train_egg_transformer_adam.cu   ""           "HIP_VISIBLE_DEVICES=0"

# 3: multi-GPU Int8NativeFormer + Muon (hipBLAS). Uses up to NGPU GPUs.
GPUS=$(seq -s, 0 $((NGPU-1)))
run_trainer egg_mgpu          full_cuda_train_transformer_adam_mgpu.cu  "-lhipblas"             "HIP_VISIBLE_DEVICES=$GPUS"
run_trainer egg_mgpu_muon     full_cuda_train_transformer_adam_mgpu.cu  "-lhipblas -DUSE_MUON=1" "HIP_VISIBLE_DEVICES=$GPUS"

# 4: d-eggs distributed (small CHUNK_SIZE so steps finish quickly). coordinator + NGPU workers.
say "d-eggs -- build for $ARCH (CHUNK_SIZE=256)"
( cd "$SRCABS/d-eggs" && make USE_HIP=1 HIPARCH="$ARCH" coordinator worker print_arch \
    CFLAGS="-O3 -Iinclude -DVOCAB_SIZE=256 -DCHUNK_SIZE=256" \
    GPUFLAGS="-O3 -Iinclude -x hip --offload-arch=$ARCH -lhipblas -DVOCAB_SIZE=256 -DCHUNK_SIZE=256" >/tmp/deggs_build.log 2>&1 )
if [ ! -x "$SRCABS/d-eggs/worker" ]; then echo "d-eggs BUILD FAILED:"; tail -8 /tmp/deggs_build.log; verdict fail "d-eggs build"; else
  say "d-eggs -- distributed run ($NGPU workers, 120s)"
  ( cd "$SRCABS/d-eggs" && stdbuf -oL ./coordinator > /tmp/deggs_coord.log 2>&1 & echo $! > /tmp/deggs_coord.pid )
  sleep 3
  for i in $(seq 0 $((NGPU-1))); do
    ( cd "$SRCABS/d-eggs" && HIP_VISIBLE_DEVICES=$i stdbuf -oL ./worker 127.0.0.1 > /tmp/deggs_w$i.log 2>&1 & )
  done
  sleep 117; kill $(cat /tmp/deggs_coord.pid) 2>/dev/null; pkill -f './worker' 2>/dev/null
  echo "coordinator loss lines:"; grep -m8 'Loss:' /tmp/deggs_coord.log || true
  echo "worker hipGraph capture:"; grep -m1 'Graph captured' /tmp/deggs_w0.log || true
  if loss_decreases /tmp/deggs_coord.log && grep -qE 'Graph captured: [1-9]' /tmp/deggs_w0.log; then
    verdict ok "d-eggs: distributed loss converges + hipGraph captured N>0 nodes on $ARCH"
  else
    echo "--- coord tail ---"; tail -8 /tmp/deggs_coord.log; verdict fail "d-eggs: no loss decrease or hipGraph 0 nodes"
  fi
fi

# non-GPU regression
say "test_ternary (non-GPU regression)"
g++ -O2 -I"$SRCABS/d-eggs/include" "$SRCABS/d-eggs/test_ternary.cpp" -o ./test_ternary 2>/dev/null && ./test_ternary | tail -1

say "SUMMARY: $PASS passed, $FAIL failed (arch=$ARCH)"
[ "$FAIL" -eq 0 ] && echo "OVERALL: PASS" || echo "OVERALL: FAIL (see logs in agent_space/egg_adv_val/)"
