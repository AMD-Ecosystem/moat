# warp-1.14 porting session 20: scripts (linux-gfx950)

Scripts referenced by "Porting session 20" in notes.md, reproduced because agent_space/ is wiped. Plan scripts (bench_bvh.py, bench_cholesky.py, bench_matmul.py, bench_examples.py, common.py, compare.py, cmp_bvh.py) are inline in jeff's plan warp-amd113-vs-moat114-perf-comparison-mi300.md; bench_bvh.py here gained two lines (TAG from $TREE when the tag is "-", constructors from $BVH_CONS).

## `bench/run.sh`

```
#!/bin/bash
# usage: run.sh <tree> <script> [args...]
#   abl<N> = the ablation tree agent_space/wabl with WP_ABL=N (builder variant bitmask)
#   tree = name of a Warp worktree agent_space/w<tree>; trees starting with 113 are AMD's release/1.13.0.amd0
#   and run on the image's ROCm 10.1 (as in the gfx942 comparison), every other tree on the job venv's ROCm 10.2.
A=/var/lib/jenkins/fleet/jobs/mi355-20261119-154230-warp114-perf-porter-mi355-prompt/moat/agent_space
R101=/opt/python-3.11-venv/lib/python3.11/site-packages/_rocm_sdk_devel
T=$1; shift
export TREE=$T
ulimit -c 0
D=$T
case $T in abl[0-9]*) export WP_ABL=${T#abl}; D=abl;; esac
export PYTHONPATH=$A/w$D
export WARP_CACHE_PATH=$A/cache/$D
case $T in 113*) export ROCM_PATH=$R101 WARP_ROCM_PATH=$R101;; esac
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-1}
cd $A/bench
exec $FLEET_VENV/bin/python "$@"
```

## `bench/drive.sh`

```
#!/bin/bash
# usage: drive.sh <script> <outfile> <trees...> -- [script args]; runs trees one at a time on the pinned GPU
S=$1; O=$2; shift 2
TREES=(); while [ $# -gt 0 ] && [ "$1" != "--" ]; do TREES+=("$1"); shift; done; shift
cd /var/lib/jenkins/fleet/jobs/mi355-20261119-154230-warp114-perf-porter-mi355-prompt/moat/agent_space/bench
for t in "${TREES[@]}"; do
  echo "### $t $(date +%T) idle-check: $(amd-smi process -g ${HIP_VISIBLE_DEVICES:-1} 2>&1 | tr -s ' \n' ' ') gfx: $(amd-smi metric -g ${HIP_VISIBLE_DEVICES:-1} -u 2>&1 | grep GFX_ACTIVITY | tr -s ' ')" | tee -a $O.log
  BENCH_OUT=$O ./run.sh $t $S "$@" >> $O.log 2>&1
  echo "### $t exit $? $(date +%T)" | tee -a $O.log
done
```

## `bench/rounds.sh`

```
#!/bin/bash
# usage: rounds.sh <script> <outfile> <rounds> <trees...> -- [args]; runs every tree once per round, rounds interleaved
S=$1; O=$2; R=$3; shift 3
TREES=(); while [ $# -gt 0 ] && [ "$1" != "--" ]; do TREES+=("$1"); shift; done; shift
for r in $(seq $R); do ./drive.sh $S $O "${TREES[@]}" -- "$@" > /dev/null; done
```

## `bench/prof.sh`

```
#!/bin/bash
# usage: prof.sh <tree> <outname> <script> [args]; rocprofv3 kernel trace + stats (ROCm 10.2 profiler), CSV in ../prof/<outname>
A=/var/lib/jenkins/fleet/jobs/mi355-20261119-154230-warp114-perf-porter-mi355-prompt/moat/agent_space
R=$($FLEET_VENV/bin/python -m rocm_sdk path --root)
T=$1; O=$2; shift 2
case $T in 113*) R=/opt/python-3.11-venv/lib/python3.11/site-packages/_rocm_sdk_devel;; esac
ulimit -c 0
D=$T
case $T in abl[0-9]*) export WP_ABL=${T#abl}; D=abl;; esac
export TREE=$T PYTHONPATH=$A/w$D WARP_CACHE_PATH=$A/cache/$D HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-1}
case $T in 113*) export ROCM_PATH=$R WARP_ROCM_PATH=$R;; esac
cd $A/bench
$R/bin/rocprofv3 --kernel-trace --stats --output-format csv -d $A/prof/$O -o run -- $FLEET_VENV/bin/python "$@" > $A/prof/$O.log 2>&1
f=$(find $A/prof/$O -name "*kernel_stats.csv" | head -1)
python3 -c "
import csv,sys
rows=list(csv.DictReader(open('$f')))
for r in sorted(rows,key=lambda r:-float(r['TotalDurationNs']))[:14]:
    print('%10.3f ms %6s calls %8.3f ms avg  %s'%(float(r['TotalDurationNs'])/1e6, r['Calls'], float(r['AverageNs'])/1e6, r['Name'][:90]))
"
```

## `bench/trace_median.py`

```
# usage: trace_median.py <prof dir> <kernel substring> [skip]; median, min and max per-dispatch duration (us) from a rocprofv3
# kernel trace, ignoring the first <skip> dispatches (warm-up)
import csv, glob, statistics, sys
f = glob.glob(f"{sys.argv[1]}/**/*kernel_trace.csv", recursive=True)[0]
d = [(int(r["End_Timestamp"]) - int(r["Start_Timestamp"])) / 1e3 for r in csv.DictReader(open(f)) if sys.argv[2] in r["Kernel_Name"]]
d = d[int(sys.argv[3]) if len(sys.argv) > 3 else 0:]
print(f"{statistics.median(d):.1f} {min(d):.1f} {max(d):.1f} {len(d)}")
```

## `bench/rtable.py`

```
# usage: rtable.py <jsonl> <op> <tree,...>; per N: median over runs of each run's median (ms), its spread over runs, ratio to the first tree
import json, statistics, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
op = sys.argv[2]; trees = sys.argv[3].split(",")
d = {}
for r in rows:
    if r.get("op") == op and "med" in r:
        d.setdefault((r["tree"], r["N"]), []).append(r["med"])
Ns = sorted({k[1] for k in d})
print("| N | " + " | ".join(trees) + " |"); print("|---" * (len(trees) + 1) + "|")
for N in Ns:
    ref = statistics.median(d[(trees[0], N)]) if (trees[0], N) in d else None
    cells = []
    for t in trees:
        v = d.get((t, N))
        if not v: cells.append("-"); continue
        m = statistics.median(v); sp = (max(v) - min(v)) / m * 50
        c = f"{m*1e3:.3f} (+-{sp:.0f}%, n={len(v)})"
        if ref and t != trees[0]: c += f" {m/ref:.2f}x"
        cells.append(c)
    print(f"| {N} | " + " | ".join(cells) + " |")
```

## `bench/three.py`

```
# usage: three.py <jsonl> <keyfields comma> [filter key=value ...]; columns 1.14 before (base), after (src), AMD 1.13 (113),
# after/before and after/AMD (median times), plus the correctness check where the script records one
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
keys = sys.argv[2].split(","); flt = dict(a.split("=") for a in sys.argv[3:])
d = {}; cells = []
for r in rows:
    if any(str(r.get(k)) != v for k, v in flt.items()): continue
    c = tuple(r.get(k) for k in keys); d[(r["tree"],) + c] = r
    if c not in cells: cells.append(c)
def v(r): return None if r is None or "error" in r else (r["med"] if "med" in r else r.get("s"))
def fmt(r):
    if r is None: return "-"
    if "error" in r: return "ERR"
    x = v(r); return f"{x*1e3:.3f} ms" if x < 1 else f"{x:.2f} s"
print("| " + " | ".join(keys + ["1.14 before", "1.14 after", "AMD 1.13", "after/before", "after/AMD", "check"]) + " |")
print("|" + "---|" * (len(keys) + 6))
for c in cells:
    b, a, m = (d.get((t,) + c) for t in ("base", "src", "113"))
    ab = f"{v(a)/v(b):.2f}" if v(a) and v(b) else "-"; am = f"{v(a)/v(m):.2f}" if v(a) and v(m) else "-"
    chk = []
    for t, r in (("before", b), ("after", a), ("113", m)):
        if r and "ok" in r and not r["ok"]: chk.append(f"{t} FAIL {r['relerr']:.1e}")
        if r and "error" in r: chk.append(f"{t}: {r['error'][:60]}")
    print("| " + " | ".join([str(x) for x in c] + [fmt(b), fmt(a), fmt(m), ab, am, "; ".join(chk) or "ok"]) + " |")
```

## `bench/bench_bvh_geom.py`

```
import numpy as np

def grid_mesh(ntri, rng):
    # subdivided unit-square grid, 2*r*r triangles, small height displacement plus noise
    r = int(round((ntri / 2) ** 0.5))
    x, y = np.meshgrid(np.linspace(0, 1, r + 1), np.linspace(0, 1, r + 1), indexing="ij")
    z = 0.05 * np.sin(6 * x) * np.cos(5 * y) + 0.002 * rng.standard_normal(x.shape)
    pts = np.stack([x, y, z], -1).reshape(-1, 3).astype(np.float32)
    i = np.arange(r)[:, None] * (r + 1) + np.arange(r)[None, :]
    a, b, c, d = i, i + 1, i + r + 1, i + r + 2
    tris = np.concatenate([np.stack([a, c, b], -1).reshape(-1, 3), np.stack([b, c, d], -1).reshape(-1, 3)])
    return pts, tris.astype(np.int32).reshape(-1), r
```

## `bench/free_race.py`

```
# Too-early-free probe. A kernel on stream s1 writes 1.0 into every element of x after a long spin.
# While it runs, x is freed with stream s2 current, and a new array y of the same size is allocated and
# zero-filled on s2. If the free does not wait for s1, the pool hands x's memory straight back to y and
# the late writes land in y (y.max() == 1). Cases:
#   alloc_on_s1: x allocated with s1 current (the kernel's stream), freed with s2 current
#   alloc_on_s2: x allocated with s2 current, written on s1 (third-party stream), freed with s2 current
#   same_stream: control, everything on s1
# Prints, per case, how many of TRIALS trials reused x's pointer for y and how many saw late writes.
import sys, time
import numpy as np
import warp as wp

wp.config.quiet = True
wp.init()
N = 1 << 20
ITERS = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
TRIALS = 10

@wp.kernel
def late_write(x: wp.array(dtype=float), iters: int):
    i = wp.tid()
    v = float(i)
    for _k in range(iters):
        v = wp.sin(v) + 1.0
    x[i] = 1.0 + 0.0 * v

dev = wp.get_device("cuda:0")
s1 = wp.Stream(dev)
s2 = wp.Stream(dev)

# kernel duration
x = wp.empty(N, dtype=float, device=dev)
wp.launch(late_write, dim=N, inputs=[x, ITERS], stream=s1)
wp.synchronize_device(dev)
t0 = time.perf_counter()
wp.launch(late_write, dim=N, inputs=[x, ITERS], stream=s1)
wp.synchronize_device(dev)
print(f"late_write kernel: {(time.perf_counter() - t0) * 1e3:.1f} ms")
del x
wp.synchronize_device(dev)

def trial(case):
    alloc_stream = s2 if case == "alloc_on_s2" else s1
    free_stream = s1 if case == "same_stream" else s2
    with wp.ScopedStream(alloc_stream):
        x = wp.empty(N, dtype=float, device=dev)
    wp.launch(late_write, dim=N, inputs=[x, ITERS], stream=s1)
    xptr = x.ptr
    with wp.ScopedStream(free_stream):
        del x
        y = wp.zeros(N, dtype=float, device=dev)
    reused = y.ptr == xptr
    wp.synchronize_device(dev)
    bad = float(y.numpy().max()) != 0.0
    del y
    wp.synchronize_device(dev)
    return reused, bad

for case in ("same_stream", "alloc_on_s1", "alloc_on_s2"):
    res = [trial(case) for _ in range(TRIALS)]
    print(f"{case}: reused {sum(r for r, _ in res)}/{TRIALS}, late writes seen {sum(b for _, b in res)}/{TRIALS}")
```

## `bench/free_cost.py`

```
# Host time per wp.empty() + free of a 4 KiB device array (mempool allocator), with K extra Warp streams
# alive on the device (K = 0, 3, 15), plus mesh.refit() per call at 10^4 triangles. Medians of 7 x 500 calls.
import time
import numpy as np
import warp as wp
from bench_bvh_geom import grid_mesh

wp.config.quiet = True
wp.init()
dev = wp.get_device("cuda:0")

def per_call(fn, it=500):
    for _ in range(20):
        fn()
    wp.synchronize_device()
    res = []
    for _ in range(7):
        t0 = time.perf_counter()
        for _ in range(it):
            fn()
        wp.synchronize_device()
        res.append((time.perf_counter() - t0) / it * 1e6)
    return sorted(res)[3]

def alloc_free():
    a = wp.empty(1024, dtype=float, device=dev)
    del a

streams = []
for k in (0, 3, 15):
    while len(streams) < k:
        streams.append(wp.Stream(dev))
    print(f"alloc+free, {k} extra streams: {per_call(alloc_free):.1f} us")
streams = []
pts_np, ind_np, r = grid_mesh(10**4, np.random.default_rng(0))
mesh = wp.Mesh(points=wp.array(pts_np, dtype=wp.vec3), indices=wp.array(ind_np, dtype=int), bvh_constructor="sah")
print(f"mesh.refit 10^4: {per_call(mesh.refit):.1f} us")
```

## `bench/free_loop.py`

```
# Per-iteration host time (issue) and issue+drain of {allocate a 4 KiB array, launch a kernel on it, free it},
# 500 iterations back to back, median of 7 (us). Then the same with the free of iteration i deferred to i+1.
import time
import warp as wp
wp.config.quiet = True
wp.init()

@wp.kernel
def tiny(x: wp.array(dtype=float)):
    x[wp.tid()] = 1.0

def run(defer):
    res = []
    for _ in range(7):
        prev = None
        wp.synchronize_device()
        t0 = time.perf_counter()
        for _ in range(500):
            a = wp.empty(1024, dtype=float)
            wp.launch(tiny, dim=1024, inputs=[a])
            if defer:
                prev = a
            del a
        t1 = time.perf_counter()
        prev = None
        wp.synchronize_device()
        t2 = time.perf_counter()
        res.append(((t1 - t0) / 500 * 1e6, (t2 - t0) / 500 * 1e6))
    return sorted(res)[3]

run(False)
print("alloc+launch+free: issue %.1f us, issue+drain %.1f us" % run(False))
```

## `bench/free_loop2.py`

```
# The loop of free_loop.py with each call timed: mean host us per call of allocate, launch, free (500 iterations).
import time
import warp as wp
wp.config.quiet = True
wp.init()

@wp.kernel
def tiny(x: wp.array(dtype=float)):
    x[wp.tid()] = 1.0

for rep in range(3):
    acc = [0.0, 0.0, 0.0]
    wp.synchronize_device()
    for _ in range(500):
        t0 = time.perf_counter(); a = wp.empty(1024, dtype=float)
        t1 = time.perf_counter(); wp.launch(tiny, dim=1024, inputs=[a])
        t2 = time.perf_counter(); del a
        t3 = time.perf_counter()
        acc[0] += t1 - t0; acc[1] += t2 - t1; acc[2] += t3 - t2
    wp.synchronize_device()
print("alloc %.1f us, launch %.1f us, free %.1f us" % tuple(v / 500 * 1e6 for v in acc))
```

## `bench/free_block8.py`

```
# free_block7's batch with K=4, each call timed (ms, median of 5): which call waits for the ~10 ms kernel.
import time
import warp as wp
wp.config.quiet = True
wp.init()
N = 1 << 20

@wp.kernel
def spin(x: wp.array(dtype=float), iters: int):
    i = wp.tid()
    v = float(i)
    for _k in range(iters):
        v = wp.sin(v) + 1.0
    x[i] = v

@wp.kernel
def tiny(x: wp.array(dtype=float)):
    x[wp.tid()] = 1.0

x = wp.empty(N, dtype=float)
wp.launch(spin, dim=N, inputs=[x, 20000]); wp.launch(tiny, dim=16, inputs=[x]); wp.synchronize_device()
K = 4
rows = {}
for _ in range(5):
    wp.synchronize_device()
    wp.launch(spin, dim=N, inputs=[x, 20000])
    for k in range(K):
        t0 = time.perf_counter(); a = wp.empty(1024, dtype=float)
        t1 = time.perf_counter(); wp.launch(tiny, dim=1024, inputs=[a])
        t2 = time.perf_counter(); del a
        t3 = time.perf_counter()
        for n, v in ((f"alloc{k}", t1 - t0), (f"launch{k}", t2 - t1), (f"free{k}", t3 - t2)):
            rows.setdefault(n, []).append(v * 1e3)
    wp.synchronize_device()
print(" ".join(f"{n}={sorted(v)[2]:.2f}" for n, v in rows.items()))
```

## `bench/bvh_one.py`

```
# usage: bvh_one.py <ntri> [reps]; builds wp.Bvh (lbvh) over the grid-mesh triangle boxes reps times (for profiling)
import sys
import numpy as np
import warp as wp
from bench_bvh_geom import grid_mesh
wp.config.quiet = True
wp.init()
N = int(float(sys.argv[1])); reps = int(sys.argv[2]) if len(sys.argv) > 2 else 3
pts_np, ind_np, r = grid_mesh(N, np.random.default_rng(0))
tri = pts_np[ind_np.reshape(-1, 3)]
lo = wp.array(tri.min(axis=1), dtype=wp.vec3); hi = wp.array(tri.max(axis=1), dtype=wp.vec3)
for _ in range(reps):
    b = wp.Bvh(lo, hi, constructor="lbvh")
    wp.synchronize_device()
    del b
```

## `bench/refit_one.py`

```
# usage: refit_one.py <ntri> [reps]; one lbvh wp.Bvh over the grid-mesh triangle boxes, then reps refits (for profiling)
import sys
import numpy as np
import warp as wp
from bench_bvh_geom import grid_mesh
wp.config.quiet = True
wp.init()
N = int(float(sys.argv[1])); reps = int(sys.argv[2]) if len(sys.argv) > 2 else 50
pts_np, ind_np, r = grid_mesh(N, np.random.default_rng(0))
tri = pts_np[ind_np.reshape(-1, 3)]
lo = wp.array(tri.min(axis=1), dtype=wp.vec3); hi = wp.array(tri.max(axis=1), dtype=wp.vec3)
b = wp.Bvh(lo, hi, constructor="lbvh")
for _ in range(reps):
    b.refit()
wp.synchronize_device()
```

## `bench/refit_wall.py`

```
# Wall us per wp.Bvh.refit() and wp.Mesh.refit() (lbvh), 200 back-to-back calls then a sync, median of 7, at several sizes.
import sys, time
import numpy as np
import warp as wp
from bench_bvh_geom import grid_mesh
from common import record
wp.config.quiet = True
wp.init()
for N in [int(float(s)) for s in sys.argv[1:]] or [10**4, 10**5, 10**6]:
    pts_np, ind_np, r = grid_mesh(N, np.random.default_rng(0))
    pts = wp.array(pts_np, dtype=wp.vec3); ind = wp.array(ind_np, dtype=int)
    tri = pts_np[ind_np.reshape(-1, 3)]
    lo = wp.array(tri.min(axis=1), dtype=wp.vec3); hi = wp.array(tri.max(axis=1), dtype=wp.vec3)
    for op, obj in (("bvh_refit", wp.Bvh(lo, hi, constructor="lbvh")),
                    ("mesh_refit", wp.Mesh(points=pts, indices=ind, bvh_constructor="lbvh"))):
        for _ in range(20):
            obj.refit()
        wp.synchronize_device()
        res = []
        for _ in range(7):
            t0 = time.perf_counter()
            for _ in range(200):
                obj.refit()
            wp.synchronize_device()
            res.append((time.perf_counter() - t0) / 200)
        res.sort()
        record(area="refit", op=op, N=N, med=res[3], min=res[0], max=res[-1], reps=7)
```

## `bench/bench_build.py`

```
# usage: bench_build.py [sizes...]; wall ms per wp.Bvh / wp.Mesh lbvh build (sync before and after), 3 warm-up builds,
# then median and min of 30 builds per size; one JSON line per cell (common.record)
import statistics, sys, time
import numpy as np
import warp as wp
from bench_bvh_geom import grid_mesh
from common import record
wp.config.quiet = True
wp.init()
SIZES = [int(float(s)) for s in sys.argv[1:]] or [10**4, 10**5, 10**6, 10**7]
for N in SIZES:
    pts_np, ind_np, r = grid_mesh(N, np.random.default_rng(0))
    pts = wp.array(pts_np, dtype=wp.vec3); ind = wp.array(ind_np, dtype=int)
    tri = pts_np[ind_np.reshape(-1, 3)]
    lo = wp.array(tri.min(axis=1), dtype=wp.vec3); hi = wp.array(tri.max(axis=1), dtype=wp.vec3)
    reps = 30 if N < 10**7 else 10
    for op, make in (("bvh_build", lambda: wp.Bvh(lo, hi, constructor="lbvh")),
                     ("mesh_build", lambda: wp.Mesh(points=pts, indices=ind, bvh_constructor="lbvh"))):
        ts = []
        obj = None
        for k in range(reps + 3):
            obj = None
            wp.synchronize_device()
            t0 = time.perf_counter()
            obj = make()
            wp.synchronize_device()
            if k >= 3:
                ts.append(time.perf_counter() - t0)
        record(area="build", op=op, N=N, med=statistics.median(ts), min=min(ts), max=max(ts), reps=reps)
```

## `bench/small_n.py`

```
# lbvh wp.Bvh and wp.Mesh with very few items (1, 2, 3, 5, 17 boxes or triangles): AABB query hit sets and closest
# points against brute force, after build and after rebuild.
import numpy as np
import warp as wp
wp.config.quiet = True
wp.init()

@wp.kernel
def q_aabb(bvh: wp.uint64, lo: wp.vec3, hi: wp.vec3, hits: wp.array(dtype=int)):
    q = wp.bvh_query_aabb(bvh, lo, hi)
    i = int(0)
    while wp.bvh_query_next(q, i):
        hits[i] = 1

@wp.kernel
def q_point(mesh: wp.uint64, p: wp.array(dtype=wp.vec3), face: wp.array(dtype=int)):
    t = wp.tid()
    r = wp.mesh_query_point_no_sign(mesh, p[t], 100.0)
    face[t] = wp.where(r.result, r.face, -1)

rng = np.random.default_rng(1)
bad = 0
for n in (1, 2, 3, 5, 17):
    lo_np = rng.uniform(0, 1, (n, 3)).astype(np.float32); hi_np = lo_np + 0.1
    lo = wp.array(lo_np, dtype=wp.vec3); hi = wp.array(hi_np, dtype=wp.vec3)
    bvh = wp.Bvh(lo, hi, constructor="lbvh")
    for rebuild in (False, True):
        if rebuild:
            bvh.rebuild()
        for _ in range(20):
            c = rng.uniform(0, 1, 3).astype(np.float32); qlo, qhi = c - 0.2, c + 0.2
            hits = wp.zeros(n, dtype=int)
            wp.launch(q_aabb, dim=1, inputs=[bvh.id, wp.vec3(qlo), wp.vec3(qhi), hits])
            ref = np.all((lo_np <= qhi) & (hi_np >= qlo), axis=1).astype(int)
            bad += int(np.any(hits.numpy() != ref))
    pts = rng.uniform(0, 1, (3 * n, 3)).astype(np.float32)
    mesh = wp.Mesh(points=wp.array(pts, dtype=wp.vec3), indices=wp.array(np.arange(3 * n, dtype=np.int32)), bvh_constructor="lbvh")
    qp = wp.array(rng.uniform(0, 1, (64, 3)).astype(np.float32), dtype=wp.vec3)
    face = wp.zeros(64, dtype=int)
    wp.launch(q_point, dim=64, inputs=[mesh.id, qp, face])
    bad += int(np.any(face.numpy() < 0))
print("small-n mismatches:", bad)
```

## `bench/hang_repro.py`

```
# usage: hang_repro.py <ntri> <refit 0|1> <kind point|aabb>; lbvh Mesh (point) or Bvh (aabb) at ntri, optional refit, one query launch
import sys, time
import numpy as np
import warp as wp
from bench_bvh_geom import grid_mesh
wp.config.quiet = True
wp.init()
N = int(float(sys.argv[1])); refit = sys.argv[2] == "1"; kind = sys.argv[3]

@wp.kernel
def k_query_point(mesh: wp.uint64, qp: wp.array(dtype=wp.vec3), dist: wp.array(dtype=float)):
    t = wp.tid()
    q = wp.mesh_query_point_no_sign(mesh, qp[t], 10.0)
    dist[t] = wp.where(q.result, float(q.face), -1.0)

@wp.kernel
def k_query_aabb(bvh: wp.uint64, lo: wp.array(dtype=wp.vec3), hi: wp.array(dtype=wp.vec3), cnt: wp.array(dtype=int)):
    t = wp.tid()
    q = wp.bvh_query_aabb(bvh, lo[t], hi[t])
    idx = int(0)
    n = int(0)
    while wp.bvh_query_next(q, idx):
        n += 1
    cnt[t] = n

rng = np.random.default_rng(0)
pts_np, ind_np, r = grid_mesh(N, rng)
NQ = 10**6
if kind == "point":
    obj = wp.Mesh(points=wp.array(pts_np, dtype=wp.vec3), indices=wp.array(ind_np, dtype=int), bvh_constructor="lbvh")
    if refit: obj.refit()
    qp = wp.array(np.concatenate([rng.uniform(0, 1, (NQ, 2)), rng.uniform(-0.2, 0.2, (NQ, 1))], 1).astype(np.float32), dtype=wp.vec3)
    out = wp.zeros(NQ, dtype=float)
    wp.launch(k_query_point, dim=NQ, inputs=[obj.id, qp, out])
else:
    tri = pts_np[ind_np.reshape(-1, 3)]
    obj = wp.Bvh(wp.array(tri.min(axis=1), dtype=wp.vec3), wp.array(tri.max(axis=1), dtype=wp.vec3), constructor="lbvh")
    if refit: obj.refit()
    c = rng.uniform(0, 1, (NQ, 2)); h = 2.0 / r
    lo = wp.array(np.concatenate([c - h, np.full((NQ, 1), -1.0)], 1).astype(np.float32), dtype=wp.vec3)
    hi = wp.array(np.concatenate([c + h, np.full((NQ, 1), 1.0)], 1).astype(np.float32), dtype=wp.vec3)
    out = wp.zeros(NQ, dtype=int)
    wp.launch(k_query_aabb, dim=NQ, inputs=[obj.id, lo, hi, out])
t0 = time.perf_counter(); wp.synchronize_device()
print(f"{kind} N={N} refit={refit}: ok {time.perf_counter()-t0:.3f}s sum={float(out.numpy().sum()):.6g}")
```

## `bench/ray_one.py`

```
# usage: ray_one.py <ntri> <constructor> [launches]; one Mesh, then the bench_bvh ray-query kernel launched N times over
# 10^6 rays (for rocprofv3 kernel timing); saves the hit distances to res_ray_<TREE>_<ntri>_<constructor>.npy
import os, sys
import numpy as np
import warp as wp
from bench_bvh_geom import grid_mesh
wp.config.quiet = True
wp.init()
N = int(float(sys.argv[1])); con = sys.argv[2]; L = int(sys.argv[3]) if len(sys.argv) > 3 else 20

@wp.kernel
def k_query_ray(mesh: wp.uint64, ro: wp.array(dtype=wp.vec3), rd: wp.array(dtype=wp.vec3), tout: wp.array(dtype=float)):
    t = wp.tid()
    q = wp.mesh_query_ray(mesh, ro[t], rd[t], 10.0)
    if q.result:
        tout[t] = q.t
    else:
        tout[t] = -1.0

rng = np.random.default_rng(0)
NQ = 10**6
ro_np = np.concatenate([rng.uniform(0, 1, (NQ, 2)), np.ones((NQ, 1))], 1).astype(np.float32)
rd_np = np.concatenate([0.1 * rng.uniform(-1, 1, (NQ, 2)), -np.ones((NQ, 1))], 1)
rd_np = (rd_np / np.linalg.norm(rd_np, axis=1, keepdims=True)).astype(np.float32)
pts_np, ind_np, r = grid_mesh(N, rng)
mesh = wp.Mesh(points=wp.array(pts_np, dtype=wp.vec3), indices=wp.array(ind_np, dtype=int), bvh_constructor=con)
ro = wp.array(ro_np, dtype=wp.vec3); rd = wp.array(rd_np, dtype=wp.vec3); tout = wp.zeros(NQ, dtype=float)
for _ in range(L):
    wp.launch(k_query_ray, dim=NQ, inputs=[mesh.id, ro, rd, tout])
wp.synchronize_device()
np.save(f"res_ray_{os.environ['TREE']}_{N}_{con}.npy", tout.numpy())
```

## `bench/ray_ab.sh`

```
#!/bin/bash
# usage: ray_ab.sh <out> <treeA> <treeB> [reps]; per size and constructor, reps alternating A,B pairs of ray_one.py
# (20 launches each, rocprofv3); per pair the ratio B/A of per-dispatch median k_query_ray durations.
# Writes "N cons medA medB ratio" per pair.
O=$1; A=$2; B=$3; REPS=${4:-3}
for N in 1e4 1e5 1e6 1e7; do for c in sah median lbvh; do for r in $(seq $REPS); do
  m=()
  for t in $A $B; do
    d=pab_${t}_${N}_${c}_$r; rm -rf ../prof/$d
    ./prof.sh $t $d ray_one.py $N $c 20 > /dev/null 2>&1
    m+=($(python3 trace_median.py ../prof/$d k_query_ray | cut -d' ' -f1))
  done
  echo "$N $c ${m[0]} ${m[1]} $(echo "scale=3; ${m[1]}/${m[0]}" | bc)" >> $O
done; done; done
```

## `bench/chol_one.py`

```
# usage: chol_one.py <fp32|fp64> <n> <block_dim> [launches] [inplace 0|1]; 8192 blocks of tile_cholesky (or
# tile_cholesky_inplace) launched N times (for rocprofv3), result checked against NumPy (relative max error)
import sys
import numpy as np
import warp as wp
wp.config.quiet = True
wp.config.enable_backward = False
wp.init()
dtn, n, bd = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
L = int(sys.argv[4]) if len(sys.argv) > 4 else 20
inplace = len(sys.argv) > 5 and sys.argv[5] == "1"
dt, npdt, tol = (wp.float32, np.float32, 1e-3) if dtn == "fp32" else (wp.float64, np.float64, 1e-9)
BATCH = 8192

@wp.kernel(module="unique")
def chol(A: wp.array3d(dtype=dt), Lo: wp.array3d(dtype=dt)):
    i = wp.tid()
    a = wp.tile_load(A[i], shape=(n, n))
    if wp.static(inplace):
        wp.tile_cholesky_inplace(a)
        wp.tile_store(Lo[i], a)
    else:
        l = wp.tile_cholesky(a)
        wp.tile_store(Lo[i], l)

rng = np.random.default_rng(0)
X = rng.standard_normal((BATCH, n, n))
Anp = (X @ X.transpose(0, 2, 1) + n * np.eye(n)).astype(npdt)
Lref = np.linalg.cholesky(Anp.astype(np.float64))
A = wp.array(Anp, dtype=dt); Lo = wp.zeros_like(A)
for _ in range(L):
    wp.launch_tiled(chol, dim=[BATCH], inputs=[A, Lo], block_dim=bd)
out = np.tril(Lo.numpy().astype(np.float64))
err = float(np.max(np.abs(out - Lref)) / max(1.0, np.max(np.abs(Lref))))
print(f"{dtn} n={n} bd={bd} inplace={inplace} relerr={err:.2e} {'ok' if err < tol else 'FAIL'}")
```

## `bench/chol_ab.sh`

```
#!/bin/bash
# usage: chol_ab.sh <out> <treeA> <treeB> [reps]; per dtype, n, block_dim: reps alternating A,B pairs of chol_one.py
# (20 launches, rocprofv3); writes "dtype n bd medA medB ratio checkA checkB" per pair (median per-dispatch us).
O=$1; A=$2; B=$3; REPS=${4:-3}
for dt in fp32 fp64; do for n in 4 8 16 32; do for bd in 32 64 128 256; do for r in $(seq $REPS); do
  m=(); ck=()
  for t in $A $B; do
    d=pch_${t}_${dt}_${n}_${bd}_$r; rm -rf ../prof/$d
    ./prof.sh $t $d chol_one.py $dt $n $bd 20 > /dev/null 2>&1
    m+=($(python3 trace_median.py ../prof/$d chol | cut -d' ' -f1))
    ck+=($(grep -oE "relerr=[0-9.e+-]+ (ok|FAIL)" ../prof/$d.log | tr ' ' ':'))
  done
  echo "$dt $n $bd ${m[0]} ${m[1]} $(echo "scale=3; ${m[1]}/${m[0]}" | bc) ${ck[0]} ${ck[1]}" >> $O
done; done; done; done
```

## `bench/bench_chol_only.py`

```
# bench_cholesky.py restricted to the tile_cholesky op (warm-up launch, then 7 x 20 launches), all n and block_dim
import time
import numpy as np
import warp as wp
from common import timeit, record

wp.config.quiet = True
wp.config.enable_backward = False
wp.init()
BATCH = 8192

def make_kernels(n, dt):
    @wp.kernel(module="unique")
    def chol(A: wp.array3d(dtype=dt), L: wp.array3d(dtype=dt)):
        i = wp.tid()
        a = wp.tile_load(A[i], shape=(n, n))
        l = wp.tile_cholesky(a)
        wp.tile_store(L[i], l)

    @wp.kernel(module="unique")
    def csolve(L: wp.array3d(dtype=dt), y: wp.array2d(dtype=dt), x: wp.array2d(dtype=dt)):
        i = wp.tid()
        l = wp.tile_load(L[i], shape=(n, n))
        yy = wp.tile_load(y[i], shape=n)
        xx = wp.tile_cholesky_solve(l, yy)
        wp.tile_store(x[i], xx)

    @wp.kernel(module="unique")
    def lsolve(L: wp.array3d(dtype=dt), y: wp.array2d(dtype=dt), x: wp.array2d(dtype=dt)):
        i = wp.tid()
        l = wp.tile_load(L[i], shape=(n, n))
        yy = wp.tile_load(y[i], shape=n)
        xx = wp.tile_lower_solve(l, yy)
        wp.tile_store(x[i], xx)

    @wp.kernel(module="unique")
    def usolve(U: wp.array3d(dtype=dt), y: wp.array2d(dtype=dt), x: wp.array2d(dtype=dt)):
        i = wp.tid()
        u = wp.tile_load(U[i], shape=(n, n))
        yy = wp.tile_load(y[i], shape=n)
        xx = wp.tile_upper_solve(u, yy)
        wp.tile_store(x[i], xx)

    return chol, csolve, lsolve, usolve

rng = np.random.default_rng(0)
for dtn, dt, npdt, tol in [("fp32", wp.float32, np.float32, 1e-3), ("fp64", wp.float64, np.float64, 1e-9)]:
    for n in (4, 8, 16, 32):
        X = rng.standard_normal((BATCH, n, n))
        Anp = (X @ X.transpose(0, 2, 1) + n * np.eye(n)).astype(npdt)
        Lref = np.linalg.cholesky(Anp.astype(np.float64))
        Y = rng.standard_normal((BATCH, n)).astype(npdt)
        xref = np.linalg.solve(Anp.astype(np.float64), Y.astype(np.float64)[..., None])[..., 0]
        zl = np.linalg.solve(Lref, Y.astype(np.float64)[..., None])[..., 0]
        Ut = Lref.transpose(0, 2, 1).copy()
        zu = np.linalg.solve(Ut, Y.astype(np.float64)[..., None])[..., 0]
        A = wp.array(Anp, dtype=dt); L = wp.zeros_like(A); U = wp.array(Ut.astype(npdt), dtype=dt)
        Lin = wp.array(Lref.astype(npdt), dtype=dt)
        y = wp.array(Y, dtype=dt); x = wp.zeros_like(y)
        for bd in (32, 64, 128, 256):
            try:
                chol, csolve, lsolve, usolve = make_kernels(n, dt)
            except Exception as e:
                record(area="cholesky", dtype=dtn, n=n, block_dim=bd, op="build", error=str(e)[-400:]); continue
            ops = [
                ("cholesky", chol, [A, L], lambda: L.numpy(), Lref),
            ]
            for op, k, args, get, ref in ops:
                cell = dict(area="cholesky", dtype=dtn, n=n, block_dim=bd, op=op)
                try:
                    t0 = time.perf_counter()
                    wp.launch_tiled(k, dim=[BATCH], inputs=args, block_dim=bd)
                    wp.synchronize_device()
                    first = time.perf_counter() - t0
                    out = get().astype(np.float64)
                    err = float(np.max(np.abs(out - ref)) / max(1.0, np.max(np.abs(ref))))
                    r = timeit(lambda: wp.launch_tiled(k, dim=[BATCH], inputs=args, block_dim=bd), reps=7, inner=20)
                    record(**cell, first_launch_s=first, relerr=err, ok=bool(err < tol), **r)
                except Exception as e:
                    record(**cell, error=f"{type(e).__name__}: {str(e)[-600:]}")
```

## `hipbuild.sh`

```
#!/bin/bash
# Build the Warp tree $1 for gfx950 only with the job venv's ROCm 10.2; log to $1/build_gfx950.log
set -o pipefail
T=$1
P=$FLEET_VENV/bin/python
R=$($P -m rocm_sdk path --root)
cd "$T" && $P build_lib.py --rocm-path "$R" --hip-arch gfx950 -j 64 > build_gfx950.log 2>&1
rc=$?; tail -2 build_gfx950.log; exit $rc
```

## `runtests.sh`

```
#!/bin/bash
# Run test files (patterns) against the Warp tree $1 on the pinned GPU, one warp.tests run per pattern,
# private kernel cache; prints the last lines of each. Usage: runtests.sh <tree> <log> <pattern>...
T=$1; L=$2; shift 2
ulimit -c 0
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-1}
export WARP_CACHE_PATH=$T/../cache_tests_$(basename $T)
cd $T
for p in "$@"; do
  echo "=== $p" >> $L
  $FLEET_VENV/bin/python -m warp.tests -s autodetect -p "$p" >> $L 2>&1
  echo "=== $p exit $?" >> $L
  echo "$p: $(grep -E '^(Ran|OK|FAILED)' $L | tail -2 | tr '\n' ' ')"
done
```

## `suite.sh`

```
#!/bin/bash
# Full default suite from the source build in the fork clone, pinned GPU, private kernel cache.
A=/var/lib/jenkins/fleet/jobs/mi355-20261119-154230-warp114-perf-porter-mi355-prompt/moat/agent_space
ulimit -c 0
cd $A/../projects/warp-1.14/src
HIP_VISIBLE_DEVICES=1 WARP_CACHE_PATH=$A/cache_full $FLEET_VENV/bin/python -m warp.tests -s autodetect
```

## `cudabuild.sh`

```
#!/bin/bash
# CUDA build (nvcc 12.8, no libmathdx) of the worktree agent_space/wcuda, as in session 19.
cd /var/lib/jenkins/fleet/jobs/mi355-20261119-154230-warp114-perf-porter-mi355-prompt/moat/agent_space/wcuda
$FLEET_VENV/bin/python build_lib.py --cuda-path /usr/local/cuda --no-use-libmathdx -j 32
```

## `pp_cmp.sh`

```
#!/bin/bash
# Compare two cuda_pp.sh output directories: build paths and __LINE__ values in check_* expansions normalised.
# Prints IDENTICAL/LINE-ONLY/DIFF per translation unit. Usage: pp_cmp.sh <dirA> <dirB>
norm() { sed -E 's#"[^"]*tree/warp/native/#"tree/warp/native/#g' "$1"; }
for f in "$1"/*.i; do
  b=$(basename "$f")
  if cmp -s <(norm "$f") <(norm "$2/$b"); then echo "IDENTICAL $b"
  elif cmp -s <(norm "$f" | sed -E 's/, [0-9]+\)/, L)/g') <(norm "$2/$b" | sed -E 's/, [0-9]+\)/, L)/g'); then echo "LINE-ONLY $b"
  else echo "DIFF $b"; fi
done
```

## `monitor.sh`

```
#!/bin/bash
# Every 5 s: time, GFX_ACTIVITY of GPU $1, and the PIDs amd-smi reports on it. Usage: monitor.sh <gpu> <log>
while true; do
  echo "$(date +%T) $(amd-smi metric -g $1 -u 2>&1 | grep GFX_ACTIVITY | tr -s ' ') $(amd-smi process -g $1 2>&1 | grep -E 'PID|No running' | tr -s ' \n' ' ')" >> $2
  sleep 5
done
```

## `fmad_probe.py`

```
# Does NVRTC (CUDA 12.8, default options and --fmad=true/false) fuse a multiply and an add written in two
# separate statements? Counts fma.rn.f32 / mul.rn.f32 / add.rn.f32 in the PTX of each kernel.
import ctypes, re
nvrtc = ctypes.CDLL("/usr/local/cuda/lib64/libnvrtc.so")
src = rb'''
extern "C" __global__ void same_stmt(float* o, const float* a, const float* b, const float* c)
{ int i = threadIdx.x; o[i] = a[i] * b[i] + c[i]; }
extern "C" __global__ void two_stmts(float* o, const float* a, const float* b, const float* c)
{ int i = threadIdx.x; float t = a[i] * b[i]; o[i] = t + c[i]; }
extern "C" __device__ __noinline__ float mul(float x, float y) { return x * y; }
extern "C" __global__ void across_inline(float* o, const float* a, const float* b, const float* c)
{ int i = threadIdx.x; float t = a[i] * b[i]; float u = t; o[i] = u + c[i]; }
'''
def ptx(opts):
    prog = ctypes.c_void_p()
    assert nvrtc.nvrtcCreateProgram(ctypes.byref(prog), src, b"p.cu", 0, None, None) == 0
    arr = (ctypes.c_char_p * len(opts))(*opts)
    rc = nvrtc.nvrtcCompileProgram(prog, len(opts), arr)
    n = ctypes.c_size_t(); nvrtc.nvrtcGetPTXSize(prog, ctypes.byref(n))
    buf = ctypes.create_string_buffer(n.value); nvrtc.nvrtcGetPTX(prog, buf)
    return rc, buf.value.decode()
for opts in ([b"--gpu-architecture=compute_90"], [b"--gpu-architecture=compute_90", b"--fmad=true"], [b"--gpu-architecture=compute_90", b"--fmad=false"]):
    rc, p = ptx(opts)
    out = []
    for k in ("same_stmt", "two_stmts", "across_inline"):
        body = p[p.index(f".entry {k}"):]; body = body[:body.index("ret;")]
        out.append(f"{k}: fma {len(re.findall(r'fma.rn.f32', body))} mul {len(re.findall(r'mul.rn.f32|mul.f32', body))} add {len(re.findall(r'add.rn.f32|add.f32', body))}")
    print(b" ".join(opts).decode(), "rc", rc, "|", "; ".join(out))
```

## `hip_contract_probe.py`

```
# The same kernels through hipRTC (ROCm 10.2) for gfx950 with -ffp-contract=on and =fast; counts v_fma*/v_mul_f32/
# v_add_f32 in each kernel's disassembly (llvm-objdump). Usage: python hip_contract_probe.py <rocm root>
import ctypes, os, re, subprocess, sys
R = sys.argv[1]
h = ctypes.CDLL(f"{R}/lib/libhiprtc.so")
src = rb'''
extern "C" __global__ void same_stmt(float* o, const float* a, const float* b, const float* c)
{ int i = threadIdx.x; o[i] = a[i] * b[i] + c[i]; }
extern "C" __global__ void two_stmts(float* o, const float* a, const float* b, const float* c)
{ int i = threadIdx.x; float t = a[i] * b[i]; o[i] = t + c[i]; }
'''
for mode in (b"on", b"fast"):
    prog = ctypes.c_void_p()
    assert h.hiprtcCreateProgram(ctypes.byref(prog), src, b"p.cu", 0, None, None) == 0
    opts = [b"--offload-arch=gfx950", b"-O3", b"-ffp-contract=" + mode]
    arr = (ctypes.c_char_p * len(opts))(*opts)
    rc = h.hiprtcCompileProgram(prog, len(opts), arr)
    n = ctypes.c_size_t(); h.hiprtcGetCodeSize(prog, ctypes.byref(n))
    buf = ctypes.create_string_buffer(n.value); h.hiprtcGetCode(prog, buf)
    open("k.co", "wb").write(buf.raw)
    dis = subprocess.run([f"{R}/lib/llvm/bin/llvm-objdump", "-d", "--mcpu=gfx950", "k.co"], capture_output=True, text=True).stdout
    out = []
    for k in ("same_stmt", "two_stmts"):
        body = dis[dis.index(f"<{k}>:"):]; body = body[:body.index("s_endpgm")]
        out.append(f"{k}: fma {len(re.findall(r'v_fma', body))} mul {len(re.findall(r'v_mul_f32', body))} add {len(re.findall(r'v_add_f32', body))}")
    print(f"-ffp-contract={mode.decode()} rc {rc} | " + "; ".join(out))
```

## Ablation tree builder (agent_space/wabl, `git diff dd41f3d2 -- warp/native/bvh.cu`; scratch only, never committed)

```diff
diff --git a/warp/native/bvh.cu b/warp/native/bvh.cu
index e0229a6f..1c7c9a01 100644
--- a/warp/native/bvh.cu
+++ b/warp/native/bvh.cu
@@ -280,7 +280,8 @@ __global__ void build_hierarchy(
     volatile int* __restrict__ range_rights,
     volatile int* __restrict__ parents,
     volatile BVHPackedNodeHalf* __restrict__ lowers,
-    volatile BVHPackedNodeHalf* __restrict__ uppers
+    volatile BVHPackedNodeHalf* __restrict__ uppers,
+    int fence
 )
 {
     int index = blockDim.x * blockIdx.x + threadIdx.x;
@@ -353,7 +354,8 @@ __global__ void build_hierarchy(
                 range_lefts[parent] = left;
 
                 // ensure above writes are visible to all threads
-                __threadfence();
+                if (fence)
+                    __threadfence();
 
                 childCount = atomicAdd(&num_children[parent], 1);
             } else {
@@ -365,7 +367,8 @@ __global__ void build_hierarchy(
                 range_rights[parent] = right;
 
                 // ensure above writes are visible to all threads
-                __threadfence();
+                if (fence)
+                    __threadfence();
 
                 childCount = atomicAdd(&num_children[parent], 1);
             }
@@ -418,7 +421,8 @@ __global__ void mark_packed_leaf_nodes(
     const uint64_t* __restrict__ keys,
     BVHPackedNodeHalf* __restrict__ lowers,
     BVHPackedNodeHalf* __restrict__ uppers,
-    const int leaf_size
+    const int leaf_size,
+    const int* __restrict__ precomputed_depths
 )
 {
     int node_index = blockDim.x * blockIdx.x + threadIdx.x;
@@ -428,11 +432,15 @@ __global__ void mark_packed_leaf_nodes(
 
         // calculate depth
         int depth = 1;
+        if (precomputed_depths)
+            depth = precomputed_depths[node_index];
+        else {
         int parent = parents[node_index];
         while (parent != -1) {
             parent = parents[parent];
             depth++;
         }
+        }
 
         int left = range_lefts[node_index];
         // the LBVH constructor's range is defined as left <= i <= right
@@ -457,6 +465,277 @@ __global__ void mark_packed_leaf_nodes(
 CUDA_CALLABLE inline vec3 Vec3Max(const vec3& a, const vec3& b) { return wp::max(a, b); }
 CUDA_CALLABLE inline vec3 Vec3Min(const vec3& a, const vec3& b) { return wp::min(a, b); }
 
+static __device__ __forceinline__ int delta_prefix(const uint64_t* __restrict__ keys, int n, int i, int j)
+{
+    if (j < 0 || j >= n)
+        return -1;
+    const uint64_t diff = keys[i] ^ keys[j];
+    if (diff == 0ull) {
+        const unsigned int idiff = (unsigned int)(i ^ j);
+        return 64 + __clz(idiff);
+    }
+    return __clzll(diff);
+}
+
+// Variant that accepts a pre-loaded key to avoid redundant global loads
+// when the same index is queried repeatedly (common in Karras construction).
+static __device__ __forceinline__ int delta_prefix_cached(
+    const uint64_t* __restrict__ keys, int n, uint64_t key_i, int i, int j)
+{
+    if (j < 0 || j >= n)
+        return -1;
+    const uint64_t diff = key_i ^ keys[j];
+    if (diff == 0ull) {
+        const unsigned int idiff = (unsigned int)(i ^ j);
+        return 64 + __clz(idiff);
+    }
+    return __clzll(diff);
+}
+
+__global__ void build_karras_topology(
+    int n,
+    int* root,
+    const uint64_t* __restrict__ keys,
+    volatile int* __restrict__ range_lefts,
+    volatile int* __restrict__ range_rights,
+    volatile int* __restrict__ parents,
+    volatile BVHPackedNodeHalf* __restrict__ lowers,
+    volatile BVHPackedNodeHalf* __restrict__ uppers
+)
+{
+    const int i = blockDim.x * blockIdx.x + threadIdx.x;
+    if (i >= n - 1)
+        return;  // n-1 internal nodes
+
+    const int internal_offset = n;
+
+    const uint64_t key_i = keys[i];
+
+    // Determine direction of the range (+1 or -1)
+    const int delta_next = delta_prefix_cached(keys, n, key_i, i, i + 1);
+    const int delta_prev = delta_prefix_cached(keys, n, key_i, i, i - 1);
+    const int d = (delta_next - delta_prev) >= 0 ? 1 : -1;
+
+    // Compute upper bound for the length of the range
+    const int delta_min = delta_prefix_cached(keys, n, key_i, i, i - d);
+
+    int l_max = 2;
+    while (delta_prefix_cached(keys, n, key_i, i, i + l_max * d) > delta_min)
+        l_max *= 2;
+
+    // Find the other end using binary search
+    int l = 0;
+    for (int t = l_max / 2; t >= 1; t /= 2) {
+        if (delta_prefix_cached(keys, n, key_i, i, i + (l + t) * d) > delta_min)
+            l += t;
+    }
+    const int j = i + l * d;
+
+    const int first = min(i, j);
+    const int last = max(i, j);
+
+    // Find split position (canonical Karras) — cache keys[first] for binary search
+    const uint64_t key_first = keys[first];
+    const int delta_node = delta_prefix_cached(keys, n, key_first, first, last);
+
+    int split = first;
+    int step = last - first;
+    do {
+        step = (step + 1) >> 1;
+        const int new_split = split + step;
+        if (new_split < last) {
+            const int delta_split = delta_prefix_cached(keys, n, key_first, first, new_split);
+            if (delta_split > delta_node)
+                split = new_split;
+        }
+    } while (step > 1);
+
+    const int internal_index = internal_offset + i;
+
+    const int left_child = (split == first) ? first : (internal_offset + split);
+    const int right_child = (split + 1 == last) ? last : (internal_offset + split + 1);
+
+    // Write internal node children
+    lowers[internal_index].i = left_child;
+    lowers[internal_index].b = 0;
+    uppers[internal_index].i = right_child;
+    uppers[internal_index].b = 0;
+
+    // Parent pointers
+    parents[left_child] = internal_index;
+    parents[right_child] = internal_index;
+
+    // Range (inclusive)
+    range_lefts[internal_index] = first;
+    range_rights[internal_index] = last;
+
+    // Identify root: the node whose range covers [0, n-1]
+    if (first == 0 && last == n - 1) {
+        *root = internal_index;
+        parents[internal_index] = -1;
+    }
+}
+
+// Fused kernel: compute depth-from-root for every node AND find the global
+// maximum depth in a single launch (avoids a second kernel + D2H copy for the
+// max-reduction that was previously required).  The per-block max is folded
+// into the depth-walk itself using shared-memory reduction + atomicMax.
+__global__ void compute_node_depths_and_max(
+    int n_nodes, const int* __restrict__ parents, int* __restrict__ depths, int* __restrict__ out_max, int max_steps
+)
+{
+    __shared__ int sdata[WP_BVH_BLOCK_DIM];
+
+    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
+
+    int depth = 0;  // 0 for out-of-range threads – won't affect max
+    if (idx < n_nodes) {
+        depth = 1;
+        int p = parents[idx];
+        int steps = 0;
+        while (p != -1 && steps < max_steps) {
+            p = parents[p];
+            ++depth;
+            ++steps;
+        }
+        depths[idx] = depth;
+    }
+
+    // Block-level max reduction
+    sdata[threadIdx.x] = depth;
+    __syncthreads();
+
+    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
+        if (threadIdx.x < s)
+            sdata[threadIdx.x] = max(sdata[threadIdx.x], sdata[threadIdx.x + s]);
+        __syncthreads();
+    }
+
+    if (threadIdx.x == 0)
+        atomicMax(out_max, sdata[0]);
+}
+
+__global__ void refit_nodes_at_depth(
+    int n_internal,
+    int internal_offset,
+    int target_depth,
+    const int* __restrict__ depths,
+    BVHPackedNodeHalf* __restrict__ node_lowers,
+    BVHPackedNodeHalf* __restrict__ node_uppers)
+{
+    int tid = blockDim.x * blockIdx.x + threadIdx.x;
+    if (tid >= n_internal)
+        return;
+
+    int idx = internal_offset + tid;
+
+    if (depths[idx] != target_depth)
+        return;
+
+    const int left = node_lowers[idx].i;
+    const int right = node_uppers[idx].i;
+
+    BVHPackedNodeHalf ll = node_lowers[left];
+    BVHPackedNodeHalf lu = node_uppers[left];
+    BVHPackedNodeHalf rl = node_lowers[right];
+    BVHPackedNodeHalf ru = node_uppers[right];
+
+    const vec3 out_lower = min(vec3(ll.x, ll.y, ll.z), vec3(rl.x, rl.y, rl.z));
+    const vec3 out_upper = max(vec3(lu.x, lu.y, lu.z), vec3(ru.x, ru.y, ru.z));
+
+    make_node(node_lowers + idx, out_lower, left, false);
+    make_node(node_uppers + idx, out_upper, right, false);
+}
+
+
+__global__ void refit_nodes_at_depth_bounded(
+    int n_internal, int internal_offset, int target_depth, const int* __restrict__ max_depth,
+    const int* __restrict__ depths, BVHPackedNodeHalf* __restrict__ node_lowers, BVHPackedNodeHalf* __restrict__ node_uppers)
+{
+    if (target_depth > *max_depth)
+        return;
+    int tid = blockDim.x * blockIdx.x + threadIdx.x;
+    if (tid >= n_internal)
+        return;
+    int idx = internal_offset + tid;
+    if (depths[idx] != target_depth)
+        return;
+    const int left = node_lowers[idx].i;
+    const int right = node_uppers[idx].i;
+    BVHPackedNodeHalf ll = node_lowers[left];
+    BVHPackedNodeHalf lu = node_uppers[left];
+    BVHPackedNodeHalf rl = node_lowers[right];
+    BVHPackedNodeHalf ru = node_uppers[right];
+    const vec3 out_lower = min(vec3(ll.x, ll.y, ll.z), vec3(rl.x, rl.y, rl.z));
+    const vec3 out_upper = max(vec3(lu.x, lu.y, lu.z), vec3(ru.x, ru.y, ru.z));
+    make_node(node_lowers + idx, out_lower, left, false);
+    make_node(node_uppers + idx, out_upper, right, false);
+}
+
+// Deterministic bounds reduction for HIP (avoids vec3 atomics which are not safe/portable on HIP).
+__global__ void compute_block_bounds(
+    const vec3* item_lowers, const vec3* item_uppers, vec3* block_lowers, vec3* block_uppers, int num_items
+)
+{
+    typedef cub::BlockReduce<vec3, WP_BVH_BLOCK_DIM> BlockReduce;
+    __shared__ typename BlockReduce::TempStorage temp_storage;
+
+    const int blockStart = blockDim.x * blockIdx.x;
+    const int numValid = ::min(num_items - blockStart, (int)blockDim.x);
+    const int tid = blockStart + threadIdx.x;
+
+    vec3 lower = vec3(FLT_MAX);
+    vec3 upper = vec3(-FLT_MAX);
+    if (tid < num_items) {
+        lower = item_lowers[tid];
+        upper = item_uppers[tid];
+    }
+
+    vec3 block_upper = BlockReduce(temp_storage).Reduce(upper, Vec3Max, numValid);
+    __syncthreads();
+    vec3 block_lower = BlockReduce(temp_storage).Reduce(lower, Vec3Min, numValid);
+
+    if (threadIdx.x == 0) {
+        block_lowers[blockIdx.x] = block_lower;
+        block_uppers[blockIdx.x] = block_upper;
+    }
+}
+
+// Block size for the final reduction kernel
+#define REDUCE_BLOCK_DIM 256
+
+// Fused final reduction: block bounds -> total bounds + inverse edge lengths.
+// Eliminates the separate compute_total_inv_edges kernel launch.
+__global__ void reduce_block_bounds(
+    const vec3* block_lowers, const vec3* block_uppers, int num_blocks,
+    vec3* total_lower, vec3* total_upper, vec3* total_inv_edges
+)
+{
+    typedef cub::BlockReduce<vec3, REDUCE_BLOCK_DIM> BlockReduce;
+    __shared__ typename BlockReduce::TempStorage temp_storage;
+
+    vec3 lo = vec3(FLT_MAX);
+    vec3 hi = vec3(-FLT_MAX);
+
+    for (int i = threadIdx.x; i < num_blocks; i += blockDim.x) {
+        lo = min(lo, block_lowers[i]);
+        hi = max(hi, block_uppers[i]);
+    }
+
+    vec3 block_upper = BlockReduce(temp_storage).Reduce(hi, Vec3Max);
+    __syncthreads();
+    vec3 block_lower = BlockReduce(temp_storage).Reduce(lo, Vec3Min);
+
+    if (threadIdx.x == 0) {
+        total_lower[0] = block_lower;
+        total_upper[0] = block_upper;
+        if (total_inv_edges) {
+            vec3 edges = block_upper - block_lower + vec3(0.0001f);
+            total_inv_edges[0] = vec3(1.0f / edges[0], 1.0f / edges[1], 1.0f / edges[2]);
+        }
+    }
+}
+
 __global__ void compute_total_bounds(
     const vec3* item_lowers, const vec3* item_uppers, vec3* total_lower, vec3* total_upper, int num_items
 )
@@ -499,6 +778,20 @@ __global__ void compute_total_inv_edges(const vec3* total_lower, const vec3* tot
 }
 
 
+
+// ABLATION ONLY: WP_ABL bitmask. 1 = no __threadfence in build_hierarchy (timing only, not correct);
+// 2 = Karras topology + depths + per-depth refit, max depth read back to the host;
+// 4 = same, capture-safe (fixed number of refit launches, device-side early exit);
+// 8 = fused block bounds + reduce with inverse edges; 16 = one pooled scratch allocation;
+// 32 = sort into bvh.primitive_indices (no d2d copy), end_bit 30 without groups;
+// 64 = mark_packed_leaf_nodes walks the parents itself even when depths were computed;
+// 128 = with 4, bound the refit launches by key bits + index bits instead of 97.
+static int abl_flags()
+{
+    const char* s = getenv("WP_ABL");
+    return s ? atoi(s) : 0;
+}
+
 LinearBVHBuilderGPU::LinearBVHBuilderGPU()
     : indices(NULL)
     , keys(NULL)
@@ -510,6 +803,8 @@ LinearBVHBuilderGPU::LinearBVHBuilderGPU()
     , total_upper(NULL)
     , total_inv_edges(NULL)
 {
+    if (abl_flags() & 16)
+        return;
     total_lower = (vec3*)wp_alloc_device(WP_CURRENT_CONTEXT, sizeof(vec3), "(native:bvh)");
     total_upper = (vec3*)wp_alloc_device(WP_CURRENT_CONTEXT, sizeof(vec3), "(native:bvh)");
     total_inv_edges = (vec3*)wp_alloc_device(WP_CURRENT_CONTEXT, sizeof(vec3), "(native:bvh)");
@@ -517,110 +812,210 @@ LinearBVHBuilderGPU::LinearBVHBuilderGPU()
 
 LinearBVHBuilderGPU::~LinearBVHBuilderGPU()
 {
+    if (!total_lower)
+        return;
     wp_free_device(WP_CURRENT_CONTEXT, total_lower);
     wp_free_device(WP_CURRENT_CONTEXT, total_upper);
     wp_free_device(WP_CURRENT_CONTEXT, total_inv_edges);
 }
 
-
 void LinearBVHBuilderGPU::build(
     BVH& bvh, const vec3* item_lowers, const vec3* item_uppers, int num_items, bounds3* total_bounds, int* item_groups
 )
 {
-    // allocate temporary memory used during  building
-    indices
-        = (int*)wp_alloc_device(WP_CURRENT_CONTEXT, sizeof(int) * num_items * 2, "(native:bvh)");  // *2 for radix sort
-    keys = (uint64_t*)wp_alloc_device(
-        WP_CURRENT_CONTEXT, sizeof(uint64_t) * num_items * 2, "(native:bvh)"
-    );  // *2 for radix sort
-    deltas = (int*)wp_alloc_device(
-        WP_CURRENT_CONTEXT, sizeof(int) * num_items, "(native:bvh)"
-    );  // highest differentiating bit between keys for item i and i+1
-    range_lefts = (int*)wp_alloc_device(WP_CURRENT_CONTEXT, sizeof(int) * bvh.max_nodes, "(native:bvh)");
-    range_rights = (int*)wp_alloc_device(WP_CURRENT_CONTEXT, sizeof(int) * bvh.max_nodes, "(native:bvh)");
-    num_children = (int*)wp_alloc_device(WP_CURRENT_CONTEXT, sizeof(int) * bvh.max_nodes, "(native:bvh)");
-
-    // if total bounds supplied by the host then we just
-    // compute our edge length and upload it to the GPU directly
+    const int F = abl_flags();
+    const int fence = (F & 1) ? 0 : 1;
+    const bool karras = (F & 6) != 0;
+    const bool capsafe = (F & 4) != 0;
+    const bool fbounds = (F & 8) != 0;
+    const bool pooled = (F & 16) != 0;
+    const bool fsort = (F & 32) != 0;
+    const int nb_bounds = (num_items + WP_BVH_BLOCK_DIM - 1) / WP_BVH_BLOCK_DIM;
+    const int sort_end_bit = (fsort && !item_groups) ? 30 : 64;
+
+    size_t sort_temp_bytes = 0;
+    if (fsort)
+        (void)cub::DeviceRadixSort::SortPairs(
+            nullptr, sort_temp_bytes, (const uint64_t*)nullptr, (uint64_t*)nullptr, (const int*)nullptr, (int*)nullptr,
+            num_items, 0, sort_end_bit
+        );
+
+    vec3* block_lowers = NULL;
+    vec3* block_uppers = NULL;
+    int* max_depth_dev = NULL;
+    void* sort_temp = NULL;
+    char* pool = NULL;
+
+    auto align_up = [](size_t x) -> size_t { return (x + 255) & ~size_t(255); };
+    if (pooled) {
+        const size_t sizes[] = { align_up(sizeof(int) * num_items * 2), align_up(sizeof(uint64_t) * num_items * 2),
+                                 align_up(sizeof(int) * num_items),     align_up(sizeof(int) * bvh.max_nodes),
+                                 align_up(sizeof(int) * bvh.max_nodes), align_up(sizeof(int) * bvh.max_nodes),
+                                 align_up(sizeof(vec3)),                align_up(sizeof(vec3)),
+                                 align_up(sizeof(vec3)),                align_up(sizeof(vec3) * nb_bounds),
+                                 align_up(sizeof(vec3) * nb_bounds),    align_up(sizeof(int)),
+                                 align_up(sort_temp_bytes) };
+        size_t total = 0;
+        for (size_t z : sizes)
+            total += z;
+        pool = (char*)wp_alloc_device(WP_CURRENT_CONTEXT, total, "(native:bvh)");
+        char* q = pool;
+        void** outs[] = { (void**)&indices,       (void**)&keys,         (void**)&deltas,      (void**)&range_lefts,
+                          (void**)&range_rights,  (void**)&num_children, (void**)&total_lower, (void**)&total_upper,
+                          (void**)&total_inv_edges, (void**)&block_lowers, (void**)&block_uppers,
+                          (void**)&max_depth_dev, (void**)&sort_temp };
+        for (int k = 0; k < 13; ++k) {
+            *outs[k] = q;
+            q += sizes[k];
+        }
+    } else {
+        indices = (int*)wp_alloc_device(WP_CURRENT_CONTEXT, sizeof(int) * num_items * 2, "(native:bvh)");
+        keys = (uint64_t*)wp_alloc_device(WP_CURRENT_CONTEXT, sizeof(uint64_t) * num_items * 2, "(native:bvh)");
+        deltas = (int*)wp_alloc_device(WP_CURRENT_CONTEXT, sizeof(int) * num_items, "(native:bvh)");
+        range_lefts = (int*)wp_alloc_device(WP_CURRENT_CONTEXT, sizeof(int) * bvh.max_nodes, "(native:bvh)");
+        range_rights = (int*)wp_alloc_device(WP_CURRENT_CONTEXT, sizeof(int) * bvh.max_nodes, "(native:bvh)");
+        num_children = (int*)wp_alloc_device(WP_CURRENT_CONTEXT, sizeof(int) * bvh.max_nodes, "(native:bvh)");
+        if (fbounds) {
+            block_lowers = (vec3*)wp_alloc_device(WP_CURRENT_CONTEXT, sizeof(vec3) * nb_bounds, "(native:bvh)");
+            block_uppers = (vec3*)wp_alloc_device(WP_CURRENT_CONTEXT, sizeof(vec3) * nb_bounds, "(native:bvh)");
+        }
+        if (karras)
+            max_depth_dev = (int*)wp_alloc_device(WP_CURRENT_CONTEXT, sizeof(int), "(native:bvh)");
+        if (fsort)
+            sort_temp = wp_alloc_device(WP_CURRENT_CONTEXT, sort_temp_bytes, "(native:bvh)");
+    }
+
     if (total_bounds) {
-        // calculate Morton codes
         vec3 edges = (*total_bounds).edges();
         edges += vec3(0.0001f);
-
         vec3 inv_edges = vec3(1.0f / edges[0], 1.0f / edges[1], 1.0f / edges[2]);
-
         wp_memcpy_h2d(WP_CURRENT_CONTEXT, total_lower, &total_bounds->lower[0], sizeof(vec3));
         wp_memcpy_h2d(WP_CURRENT_CONTEXT, total_upper, &total_bounds->upper[0], sizeof(vec3));
         wp_memcpy_h2d(WP_CURRENT_CONTEXT, total_inv_edges, &inv_edges[0], sizeof(vec3));
+    } else if (fbounds) {
+        wp_launch_device(
+            WP_CURRENT_CONTEXT, compute_block_bounds, num_items,
+            (item_lowers, item_uppers, block_lowers, block_uppers, num_items)
+        );
+        wp_launch_device(
+            WP_CURRENT_CONTEXT, reduce_block_bounds, 1,
+            (block_lowers, block_uppers, nb_bounds, total_lower, total_upper, total_inv_edges)
+        );
     } else {
-        // IEEE-754 bit patterns for ± FLT_MAX
         constexpr int FLT_MAX_BITS = 0x7f7fffff;
         constexpr int NEG_FLT_MAX_BITS = 0xff7fffff;
-
-        // total_lower := ( +FLT_MAX, +FLT_MAX, +FLT_MAX )
         wp_launch_device(
             WP_CURRENT_CONTEXT, memset_kernel, sizeof(vec3) / 4, ((int*)total_lower, FLT_MAX_BITS, sizeof(vec3) / 4)
         );
-
-        // total_upper := ( -FLT_MAX, -FLT_MAX, -FLT_MAX )
         wp_launch_device(
             WP_CURRENT_CONTEXT, memset_kernel, sizeof(vec3) / 4, ((int*)total_upper, NEG_FLT_MAX_BITS, sizeof(vec3) / 4)
         );
-
-        // compute the total bounds on the GPU
         wp_launch_device(
             WP_CURRENT_CONTEXT, compute_total_bounds, num_items,
             (item_lowers, item_uppers, total_lower, total_upper, num_items)
         );
-
-        // compute the total edge length
         wp_launch_device(WP_CURRENT_CONTEXT, compute_total_inv_edges, 1, (total_lower, total_upper, total_inv_edges));
     }
 
-    // assign 30-bit Morton code based on the centroid of each triangle and bounds for each leaf
     wp_launch_device(
         WP_CURRENT_CONTEXT, compute_morton_codes, num_items,
         (item_lowers, item_uppers, num_items, total_lower, total_inv_edges, indices, keys, item_groups)
     );
 
-    // sort items based on Morton key (note the 64-bit sort key includes group in upper 32 bits and morton code in lower
-    // 32 bits)
-    radix_sort_pairs_device(WP_CURRENT_CONTEXT, keys, indices, num_items);
-    wp_memcpy_d2d(WP_CURRENT_CONTEXT, bvh.primitive_indices, indices, sizeof(int) * num_items);
-
-    // calculate deltas between adjacent keys
-    wp_launch_device(WP_CURRENT_CONTEXT, compute_key_deltas, num_items, (keys, deltas, num_items - 1));
+    uint64_t* sorted_keys = keys;
+    const int* sorted_indices = indices;
+    if (fsort) {
+        sorted_keys = keys + num_items;
+        check_cuda(
+            cub::DeviceRadixSort::SortPairs(
+                sort_temp, sort_temp_bytes, keys, sorted_keys, indices, bvh.primitive_indices, num_items, 0,
+                sort_end_bit, (cudaStream_t)wp_cuda_stream_get_current()
+            )
+        );
+        sorted_indices = bvh.primitive_indices;
+    } else {
+        radix_sort_pairs_device(WP_CURRENT_CONTEXT, keys, indices, num_items);
+        wp_memcpy_d2d(WP_CURRENT_CONTEXT, bvh.primitive_indices, indices, sizeof(int) * num_items);
+    }
 
-    // initialize leaf nodes
     wp_launch_device(
         WP_CURRENT_CONTEXT, build_leaves, num_items,
-        (item_lowers, item_uppers, num_items, indices, range_lefts, range_rights, bvh.node_lowers, bvh.node_uppers)
+        (item_lowers, item_uppers, num_items, sorted_indices, range_lefts, range_rights, bvh.node_lowers,
+         bvh.node_uppers)
     );
 
-    // reset children count, this is our atomic counter so we know when an internal node is complete, only used during
-    // building
-    wp_memset_device(WP_CURRENT_CONTEXT, num_children, 0, sizeof(int) * bvh.max_nodes);
-
-    // build the tree and internal node bounds
-    wp_launch_device(
-        WP_CURRENT_CONTEXT, build_hierarchy, num_items,
-        (num_items, bvh.root, deltas, keys, num_children, bvh.primitive_indices, range_lefts, range_rights,
-         bvh.node_parents, bvh.node_lowers, bvh.node_uppers)
-    );
+    int* depths = NULL;
+    if (karras) {
+        wp_memset_device(WP_CURRENT_CONTEXT, bvh.node_parents, 0xFF, sizeof(int) * bvh.max_nodes);
+        wp_launch_device(
+            WP_CURRENT_CONTEXT, build_karras_topology, num_items,
+            (num_items, bvh.root, sorted_keys, range_lefts, range_rights, bvh.node_parents, bvh.node_lowers,
+             bvh.node_uppers)
+        );
+        depths = num_children;
+        wp_memset_device(WP_CURRENT_CONTEXT, max_depth_dev, 0, sizeof(int));
+        wp_launch_device(
+            WP_CURRENT_CONTEXT, compute_node_depths_and_max, bvh.max_nodes,
+            (bvh.max_nodes, bvh.node_parents, depths, max_depth_dev, 1024)
+        );
+        const int num_internal = num_items - 1;
+        if (!capsafe) {
+            int max_depth = 0;
+            wp_memcpy_d2h(WP_CURRENT_CONTEXT, &max_depth, max_depth_dev, sizeof(int));
+            for (int d = max_depth; d >= 1; --d)
+                wp_launch_device(
+                    WP_CURRENT_CONTEXT, refit_nodes_at_depth, num_internal,
+                    (num_internal, num_items, d, depths, bvh.node_lowers, bvh.node_uppers)
+                );
+        } else {
+            // internal depth <= key bits + index bits (Karras: the common prefix grows strictly with depth)
+            int bound = 97;
+            if (F & 128) {
+                int k = 0;
+                while ((1 << k) < num_items)
+                    ++k;
+                bound = (item_groups ? 64 : 30) + k;
+            }
+            for (int d = bound; d >= 1; --d)
+                wp_launch_device(
+                    WP_CURRENT_CONTEXT, refit_nodes_at_depth_bounded, num_internal,
+                    (num_internal, num_items, d, max_depth_dev, depths, bvh.node_lowers, bvh.node_uppers)
+                );
+        }
+    } else {
+        wp_launch_device(WP_CURRENT_CONTEXT, compute_key_deltas, num_items, (sorted_keys, deltas, num_items - 1));
+        wp_memset_device(WP_CURRENT_CONTEXT, num_children, 0, sizeof(int) * bvh.max_nodes);
+        wp_launch_device(
+            WP_CURRENT_CONTEXT, build_hierarchy, num_items,
+            (num_items, bvh.root, deltas, sorted_keys, num_children, bvh.primitive_indices, range_lefts, range_rights,
+             bvh.node_parents, bvh.node_lowers, bvh.node_uppers, fence)
+        );
+    }
     wp_launch_device(
         WP_CURRENT_CONTEXT, mark_packed_leaf_nodes, bvh.max_nodes,
-        (bvh.max_nodes, range_lefts, range_rights, bvh.node_parents, keys, bvh.node_lowers, bvh.node_uppers,
-         bvh.leaf_size)
+        (bvh.max_nodes, range_lefts, range_rights, bvh.node_parents, sorted_keys, bvh.node_lowers, bvh.node_uppers,
+         bvh.leaf_size, (F & 64) ? (const int*)NULL : depths)
     );
 
-    // free temporary memory
-    wp_free_device(WP_CURRENT_CONTEXT, indices);
-    wp_free_device(WP_CURRENT_CONTEXT, keys);
-    wp_free_device(WP_CURRENT_CONTEXT, deltas);
-
-    wp_free_device(WP_CURRENT_CONTEXT, range_lefts);
-    wp_free_device(WP_CURRENT_CONTEXT, range_rights);
-    wp_free_device(WP_CURRENT_CONTEXT, num_children);
+    if (pooled) {
+        wp_free_device(WP_CURRENT_CONTEXT, pool);
+        total_lower = total_upper = total_inv_edges = NULL;
+    } else {
+        wp_free_device(WP_CURRENT_CONTEXT, indices);
+        wp_free_device(WP_CURRENT_CONTEXT, keys);
+        wp_free_device(WP_CURRENT_CONTEXT, deltas);
+        wp_free_device(WP_CURRENT_CONTEXT, range_lefts);
+        wp_free_device(WP_CURRENT_CONTEXT, range_rights);
+        wp_free_device(WP_CURRENT_CONTEXT, num_children);
+        if (block_lowers) {
+            wp_free_device(WP_CURRENT_CONTEXT, block_lowers);
+            wp_free_device(WP_CURRENT_CONTEXT, block_uppers);
+        }
+        if (max_depth_dev)
+            wp_free_device(WP_CURRENT_CONTEXT, max_depth_dev);
+        if (sort_temp)
+            wp_free_device(WP_CURRENT_CONTEXT, sort_temp);
+    }
 }
 
 // buffer_size is the number of T, not the number of bytes
```
