# warp-1.14: scripts of the evidence session (2026-11-24, linux-gfx1250): final head `1c94e68b`, default and strict mode

Referenced from that section of `notes.md`. Scratch copies were in the job's `agent_space/f/`; `<job>` stands for the job directory.

## `tailspill.hip`

```cpp
// gfx1250: a device function calls a non-inlined callee, its SGPRs are clobbered so the callee address
// is spilled to a VGPR lane, and it ends with a tail call through the reloaded address.
#include <hip/hip_runtime.h>
#include <cstdio>
__device__ __noinline__ void g(float* p, float v) { atomicAdd(p, v); }
#define CLOBBER asm volatile("" ::: "s0","s1","s2","s3","s4","s5","s6","s7","s8","s9","s10","s11","s12","s13","s14","s15", \
    "s16","s17","s18","s19","s20","s21","s22","s23","s24","s25","s26","s27","s28","s29","s34","s35","s36","s37","s38","s39", \
    "s40","s41","s42","s43","s44","s45","s46","s47","s48","s49","s50","s51","s52","s53","s54","s55","s56","s57","s58","s59", \
    "s60","s61","s62","s63","s64","s65","s66","s67","s68","s69","s70","s71","s72","s73","s74","s75","s76","s77","s78","s79", \
    "s80","s81","s82","s83","s84","s85","s86","s87","s88","s89","s90","s91","s92","s93","s94","s95","s96","s97","s98","s99", \
    "s100","s101","s102","s103","s104","s105")
__device__ __noinline__ void f(float* p, float v)
{
    g(p, v);
    CLOBBER;
    g(p, 2.0f * v);
    CLOBBER;
    g(p, 3.0f * v);  // tail call
}
__global__ void k(float* p) { f(p, 1.0f); }
int main()
{
    float* p;
    if (hipMallocManaged(&p, 4) != hipSuccess) return 2;
    *p = 0.0f;
    k<<<1, 32>>>(p);
    hipError_t l = hipGetLastError();
    hipError_t e = hipDeviceSynchronize();
    printf("launch=%d (%s) sync=%d (%s) sum=%g (expect 192)\n", l, hipGetErrorString(l), e, hipGetErrorString(e), *p);
    return l != hipSuccess || e != hipSuccess || *p != 192.0f;
}
```

## `load_co.hip`

```cpp
// load a code object with hipModuleLoad and report the result (no launch)
#include <hip/hip_runtime.h>
#include <cstdio>
int main(int argc, char** argv)
{
    hipModule_t m;
    hipError_t e = hipModuleLoad(&m, argv[1]);
    printf("hipModuleLoad(%s): %d (%s)\n", argv[1], e, hipGetErrorString(e));
    if (e == hipSuccess && argc > 2) {
        hipFunction_t f;
        e = hipModuleGetFunction(&f, m, argv[2]);
        printf("hipModuleGetFunction(%s): %d (%s)\n", argv[2], e, hipGetErrorString(e));
    }
    return e != hipSuccess;
}
```

## `noinl.hip`

```cpp
#include <hip/hip_runtime.h>
#include <cstdio>
__device__ __noinline__ float f(float x) { return x * 2.0f + 1.0f; }
__global__ void k(float* a) { int i = threadIdx.x; a[i] = f(a[i]); }
int main() {
  float* a; hipMallocManaged(&a, 64 * 4);
  for (int i = 0; i < 64; ++i) a[i] = i;
  k<<<1, 64>>>(a);
  hipError_t e = hipDeviceSynchronize();
  printf("err=%d (%s) a[3]=%g (expect 7)\n", e, hipGetErrorString(e), a[3]);
  return 0;
}
```

## `idle_wait.sh`

```bash
#!/bin/bash
# wait until GPU 0 shows no process other than the PIDs in $ALLOW_PIDS (space-separated host PIDs of this job's
# known orphan) in <n> consecutive samples 10 s apart; give up after <max> s.
n=${1:-3}; max=${2:-600}; ok=0; t0=$(date +%s)
while true; do
  other=$(amd-smi process -g 0 2>&1 | grep -oE 'PID: [0-9]+' | cut -d' ' -f2 | grep -vxF -f <(printf '%s\n' ${ALLOW_PIDS:-none}) | head -1)
  if [ -n "$other" ]; then ok=0; else ok=$((ok+1)); fi
  [ $ok -ge $n ] && { echo "idle $(date -u +%T)"; exit 0; }
  [ $(( $(date +%s) - t0 )) -ge $max ] && { echo "busy $(date -u +%T) pid $other"; exit 1; }
  sleep 10
done
```

## `suite2.sh`

```bash
#!/bin/bash
# usage: suite2.sh <tag> [env...]; full default suite once from a scratch dir outside the tree, fresh cache,
# with a 20 s occupancy log: amd-smi process count on GPU 0 against this job's python process count
ulimit -c 0
F=$(dirname $(readlink -f $0)); export S=$(readlink -f ${SRC:-$F/../../projects/warp-1.14/src})
export PATH=$FLEET_VENV/bin:$PATH
tag=$1; shift
mkdir -p $F/run/$tag; cd $F/run/$tag
$F/idle_wait.sh 2 600 || { echo "GPU busy, suite not started"; exit 2; }
( while true; do
    n=$(amd-smi process -g 0 2>&1 | grep -c 'PID:'); o=$(pgrep -u $(id -u) -x python | wc -l)
    echo "$(date -u +%T) amd-smi_pids=$n our_python=$o"; sleep 20
  done ) > occupancy.log 2>&1 &
echo $! > occ.pid
export WARP_CACHE_PATH=$(mktemp -d $F/run/$tag/kc.XXXX)
env "$@" PYTHONPATH=$S python -c "import warp,os; assert warp.__file__.startswith(os.environ['S']); print(warp.__file__)" > suite.log 2>&1
env "$@" PYTHONPATH=$S python -m warp.tests -s autodetect >> suite.log 2>&1
rc=$?
kill $(cat occ.pid)
echo "suite rc=$rc $(date -u +%T)"
```

## `one.sh`

```bash
#!/bin/bash
# usage: one.sh <tag> <test file> [env...]; one test file alone (normal parallel runner), fresh cache, outside the tree
ulimit -c 0
F=$(dirname $(readlink -f $0)); export S=$(readlink -f ${SRC:-$F/../../projects/warp-1.14/src})
export PATH=$FLEET_VENV/bin:$PATH
tag=$1; p=$2; shift 2
mkdir -p $F/run/$tag; cd $F/run/$tag
$F/idle_wait.sh 1 600 >/dev/null || { echo "GPU busy, $p not started"; exit 2; }
export WARP_CACHE_PATH=$(mktemp -d $F/run/$tag/kc.XXXX)
env "$@" PYTHONPATH=$S python -m warp.tests -s autodetect -p "$p" > $p.log 2>&1
echo "$p rc=$? $(grep -E '^Ran ' $p.log) $(grep -E '^(OK|FAILED)' $p.log)"
```

## `dev.sh`

```bash
#!/bin/bash
# usage: dev.sh <tag> [env...]; short device check from a scratch dir, fresh cache
ulimit -c 0
F=$(dirname $(readlink -f $0)); export S=${SRC:-$F/../../projects/warp-1.14/src}; S=$(readlink -f $S)
export PATH=$FLEET_VENV/bin:$PATH R=$(rocm-sdk path --root)
tag=$1; shift
mkdir -p $F/run/$tag; cd $F/run/$tag
export WARP_CACHE_PATH=$(mktemp -d $F/run/$tag/kc.XXXX)
env "$@" PYTHONPATH=$S python $F/dev.py
```

## `dev.py`

```python
import os, sys, glob, subprocess
import warp as wp
assert wp.__file__.startswith(os.environ["S"]), wp.__file__
wp.init()
d = wp.get_device("cuda:0")
print("arch_name", d.arch, getattr(d, "_arch_name", None), d.name)
@wp.kernel
def k(a: wp.array(dtype=float)):
    i = wp.tid()
    a[i] = a[i] * 2.0 + 1.0
a = wp.ones(1024, dtype=float, device="cuda:0")
wp.launch(k, dim=1024, inputs=[a], device="cuda:0")
print("result", a.numpy()[:3])
# library sort
keys = wp.array([5, 3, 1, 4, 2, 0, 7, 6] + [0] * 8, dtype=int, device="cuda:0")
vals = wp.array(list(range(8)) + [0] * 8, dtype=int, device="cuda:0")
wp.utils.radix_sort_pairs(keys, vals, 8)
print("sorted", keys.numpy()[:8])
for f in glob.glob(os.environ["WARP_CACHE_PATH"] + "/**/*.hsaco", recursive=True):
    print(os.path.relpath(f, os.environ["WARP_CACHE_PATH"]))
    print(subprocess.run([os.environ["R"] + "/lib/llvm/bin/llvm-readelf", "-h", f], capture_output=True, text=True).stdout.split("Flags:")[1].split("\n")[0])
```

## `fnsize.py`

```python
# usage: fnsize.py <objdump bin> <hsaco>...; for each s_set_pc_i64, print function name, function size, text size, register
import re, subprocess, sys
O = sys.argv[1]
for f in sys.argv[2:]:
    asm = subprocess.run([O, "-d", "--mcpu=gfx1250", f], capture_output=True, text=True).stdout.splitlines()
    starts = [(int(m.group(1), 16), m.group(2)) for l in asm if (m := re.match(r"^([0-9a-f]+) <(.*)>:", l))]
    for l in asm:
        m = re.search(r"s_set_pc_i64 (s\[\d+:\d+\])\s+// ([0-9A-F]+):", l)
        if not m: continue
        a = int(m.group(2), 16)
        i = max(j for j, (s, _) in enumerate(starts) if s <= a)
        end = starts[i + 1][0] if i + 1 < len(starts) else None
        print(f.split("/")[-1][:60], m.group(1), "fn_size", (end - starts[i][0]) if end else "last", "offset_in_fn", a - starts[i][0], starts[i][1][:50])
```
