# warp-1.14: contraction scripts (porting session 33 and its addendum, linux-gfx90a)

Complete, runnable copies of the scripts behind "Porting session 33" in `notes.md` (the abbreviated copies there are superseded by these). Layout as run: `r.sh` and `t.sh` in `<moat>/agent_space/`, the Python scripts in `<moat>/agent_space/s/` (`hel.py` imports `twice.py`, `wnres.py` imports `perf.py`, so keep them together). Every script pins one GPU (`HIP_VISIBLE_DEVICES`, default 0), runs under `ulimit -c 0` through `r.sh`/`t.sh`, and gets a fresh `WARP_CACHE_PATH` per tag. `TREE` selects the Warp tree imported through `PYTHONPATH` (default: the fork clone `projects/warp-1.14/src`, built with `build_lib.py`); hipRTC reads `warp/native/*.h` from that tree at JIT time, so a header-only variant needs only a copy of the package with the edited headers and the same `warp/bin`. Needs numpy, usd-core (`pxr`, for the spiky.usd test asset) and the tree's `warp.tests` package.

Pass script paths to `r.sh` as absolute paths (it runs from `agent_space/run`).

## The hipRTC `on` tree

A worktree of the fork at the head under test (`git -C projects/warp-1.14/src worktree add --detach <dir> 4e0d02ab`, then `git -C <dir> lfs pull`), this one-line change in `warp/native/warp.cu` (`cuda_compile_program`'s hipRTC options), and a rebuild with the usual command (`python build_lib.py --rocm-path "$(rocm-sdk path --root)" --hip-arch gfx90a -j 64 --no-rocm-rpath`):

```diff
-    stored_options.push_back(fuse_fp ? "-ffp-contract=fast-honor-pragmas" : "-ffp-contract=off");
+    stored_options.push_back(fuse_fp ? "-ffp-contract=on" : "-ffp-contract=off");
```

Equivalent: `sed -i 's/"-ffp-contract=fast-honor-pragmas"/"-ffp-contract=on"/' warp/native/warp.cu`. Run with `TREE=<dir>`.

## Spill counts

From the cached code object of a run (the `.hsaco` in the tag's kernel cache), the AMDGPU metadata of one kernel:

```bash
LLVM="$(rocm-sdk path --root)/lib/llvm/bin"
"$LLVM/llvm-readelf" --notes agent_space/cache-<tag>/*/*/*.hsaco \
  | grep -A40 'k_wn_.*_cuda_kernel_forward' \
  | grep -E '\.name:|vgpr_count|sgpr_count|private_segment_fixed_size|vgpr_spill|sgpr_spill'
```

(`k_wn_` is `perf.py`'s winding-number kernel; for the test kernel use `sample_mesh_query_sign_normal`.) Instruction counts: `"$LLVM/llvm-objdump" -d --mcpu=gfx90a <hsaco>`, cut to one kernel's symbol, `grep -cE '\bv_pk_fma_f32\b'` and so on.

## `r.sh`

Usage: `[TREE=<tree>] agent_space/r.sh <tag> <abs path to script> [args]`: one script, fresh cache, prints `rc` and the `expect_eq() failed` count; output in `agent_space/<tag>.out`.

```bash
#!/bin/bash
# usage: [TREE=<warp tree>] r.sh <tag> <script> [args]
# Runs one script on one pinned GPU with a fresh kernel cache and prints its exit code and the count of device-side
# "expect_eq() failed" lines. Lives in <moat>/agent_space/; TREE defaults to the fork clone projects/warp-1.14/src.
# PY overrides the interpreter (default: $FLEET_VENV/bin/python, else python3); HIP_VISIBLE_DEVICES is kept if set.
A=$(cd "$(dirname "$0")" && pwd)
ulimit -c 0
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0} PYTHONPATH=${TREE:-$A/../projects/warp-1.14/src}
PY=${PY:-${FLEET_VENV:+$FLEET_VENV/bin/python}}; PY=${PY:-python3}
tag=$1; shift; C=$A/cache-$tag; rm -rf "$C"; mkdir -p "$A/run"
cd "$A/run" && WARP_CACHE_PATH=$C "$PY" -s "$@" > "$A/$tag.out" 2>&1
echo "$tag rc=$? expect_eq=$(grep -c 'expect_eq() failed' "$A/$tag.out")"
```

## `t.sh`

Usage: `[TREE=<tree>] agent_space/t.sh <tag> -p <test_file.py>`: `python -m warp.tests -s autodetect` with a fresh cache (`-p` keeps only its last pattern: one file per run).

```bash
#!/bin/bash
# usage: [TREE=<warp tree>] t.sh <tag> <warp.tests args...>
# python -m warp.tests -s autodetect on one pinned GPU with a fresh kernel cache, from a directory outside the tree.
# Same conventions as r.sh (TREE, PY, HIP_VISIBLE_DEVICES); output goes to stdout.
A=$(cd "$(dirname "$0")" && pwd)
ulimit -c 0
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0} PYTHONPATH=${TREE:-$A/../projects/warp-1.14/src}
PY=${PY:-${FLEET_VENV:+$FLEET_VENV/bin/python}}; PY=${PY:-python3}
tag=$1; shift; C=$A/cache-$tag; rm -rf "$C"; mkdir -p "$A/run"
cd "$A/run" && WARP_CACHE_PATH=$C "$PY" -s -m warp.tests -s autodetect "$@"
```

## `mqp.py`

Usage: `r.sh <tag> <s>/mqp.py [nofuse]`: the test module's own `sample_mesh_query_sign_normal` over spiky.usd and the test's 32^3 grid, nine BVH configurations; the failure shows as 60 `expect_eq` lines (0 when fixed). The validator's `mqp_diag4.py`.

```python
# usage: mqp.py [nofuse]; runs the test module's sample_mesh_query_sign_normal kernel over spiky.usd, 9 configs
import itertools, os, sys
import numpy as np
import warp as wp
from warp.tests.geometry import test_mesh_query_point as T
from pxr import Usd, UsdGeom
dev = wp.get_device("cuda:0")
if "nofuse" in sys.argv:
    wp.set_module_options({"fuse_fp": False}, module=T)
st = Usd.Stage.Open(os.path.join(os.path.dirname(T.__file__), "..", "assets", "spiky.usd"))
g = UsdGeom.Mesh(st.GetPrimAtPath("/Cube/Cube"))
tri = T.triangulate(g.GetFaceVertexCountsAttr().Get(), g.GetFaceVertexIndicesAttr().Get())
P = wp.array(np.array(g.GetPointsAttr().Get()), dtype=wp.vec3, device=dev)
I = wp.array(np.array(tri), dtype=int, device=dev)
p = T.particle_grid(32, 32, 32, np.array([-1.1, -1.1, -1.1]), 0.05, 0.0)
n = len(p)
Q = wp.array(p, dtype=wp.vec3, device=dev)
z = lambda t: wp.zeros(n, dtype=t, device=dev)
for leaf, con in itertools.product([1, 2, 4], ["sah", "median", "lbvh"]):
    m = wp.Mesh(points=P, velocities=None, indices=I, support_winding_number=True, bvh_constructor=con, bvh_leaf_size=leaf)
    f, s, d = z(int), z(float), z(float)
    wp.launch(T.sample_mesh_query_sign_normal, n, inputs=[m.id, Q, f, s, d], device=dev)
    wp.synchronize_device(dev)
```

## `twice.py`

Usage: `r.sh <tag> <s>/twice.py <out.json> [nofuse]`: every geometry query builtin called twice in one kernel; prints `<disagreements> / <checks>  <mesh>/<builtin>` (session 33 section 2). Note `nofuse` must be its own argument.

```python
# usage: twice.py <out.json> [nofuse]
# Calls each Warp geometry query builtin twice in one kernel (out-parameter form against struct form where both exist,
# otherwise the same call twice with a store in between) and counts query points whose two results differ in any field.
# Meshes: spiky.usd (test_mesh_query_point's mesh) and a regular 24x24x24 cube surface grid (many exact ties).
# Points: the test's 32^3 grid and 200,000 uniform random points in the inflated bounding box.
# BVH: leaf sizes 1, 2, 4 x constructors sah, median, lbvh.
import itertools, json, os, sys
import numpy as np
import warp as wp
from warp.tests.geometry import test_mesh_query_point as T
from pxr import Usd, UsdGeom

wp.init()
dev = wp.get_device("cuda:0")
NOFUSE = "nofuse" in sys.argv
if NOFUSE:
    wp.set_module_options({"fuse_fp": False})

MAXD = 10012.0
DIRS = [wp.vec3(1.0, 0.0, 0.0), wp.vec3(0.0, -1.0, 0.0), wp.vec3(0.0, 0.0, 1.0), wp.vec3(0.57735, 0.57735, 0.57735),
        wp.vec3(0.3, -0.8, 0.52)]


@wp.func
def fneq(a: float, b: float) -> int:
    if a == b:
        return 0
    return 1


@wp.func
def vneq(a: wp.vec3, b: wp.vec3) -> int:
    return fneq(a[0], b[0]) + fneq(a[1], b[1]) + fneq(a[2], b[2])


@wp.kernel
def k_point(m: wp.uint64, pts: wp.array[wp.vec3], bad: wp.array[int]):
    i = wp.tid(); p = pts[i]
    s = float(0.0); f = int(0); u = float(0.0); v = float(0.0)
    ok = wp.mesh_query_point(m, p, MAXD, s, f, u, v)
    q = wp.mesh_query_point(m, p, MAXD)
    bad[i] = wp.where(ok == q.result, 0, 1) + fneq(s, q.sign) + wp.where(f == q.face, 0, 1) + fneq(u, q.u) + fneq(v, q.v)


@wp.kernel
def k_nosign(m: wp.uint64, pts: wp.array[wp.vec3], bad: wp.array[int]):
    i = wp.tid(); p = pts[i]
    f = int(0); u = float(0.0); v = float(0.0)
    ok = wp.mesh_query_point_no_sign(m, p, MAXD, f, u, v)
    q = wp.mesh_query_point_no_sign(m, p, MAXD)
    bad[i] = wp.where(ok == q.result, 0, 1) + wp.where(f == q.face, 0, 1) + fneq(u, q.u) + fneq(v, q.v)


@wp.kernel
def k_furthest(m: wp.uint64, pts: wp.array[wp.vec3], bad: wp.array[int]):
    i = wp.tid(); p = pts[i]
    f = int(0); u = float(0.0); v = float(0.0)
    ok = wp.mesh_query_furthest_point_no_sign(m, p, 0.0, f, u, v)
    q = wp.mesh_query_furthest_point_no_sign(m, p, 0.0)
    bad[i] = wp.where(ok == q.result, 0, 1) + wp.where(f == q.face, 0, 1) + fneq(u, q.u) + fneq(v, q.v)


@wp.kernel
def k_parity(m: wp.uint64, pts: wp.array[wp.vec3], bad: wp.array[int]):
    i = wp.tid(); p = pts[i]
    s = float(0.0); f = int(0); u = float(0.0); v = float(0.0)
    ok = wp.mesh_query_point_sign_parity(m, p, MAXD, s, f, u, v)
    q = wp.mesh_query_point_sign_parity(m, p, MAXD)
    bad[i] = wp.where(ok == q.result, 0, 1) + fneq(s, q.sign) + wp.where(f == q.face, 0, 1) + fneq(u, q.u) + fneq(v, q.v)


@wp.kernel
def k_normal(m: wp.uint64, pts: wp.array[wp.vec3], bad: wp.array[int]):
    i = wp.tid(); p = pts[i]
    s = float(0.0); f = int(0); u = float(0.0); v = float(0.0)
    ok = wp.mesh_query_point_sign_normal(m, p, MAXD, s, f, u, v)
    q = wp.mesh_query_point_sign_normal(m, p, MAXD)
    bad[i] = wp.where(ok == q.result, 0, 1) + fneq(s, q.sign) + wp.where(f == q.face, 0, 1) + fneq(u, q.u) + fneq(v, q.v)


@wp.kernel
def k_normal_ctx(m: wp.uint64, pts: wp.array[wp.vec3], dist: wp.array[float], bad: wp.array[int]):
    # the test kernel's shape: an eval and a store between the two calls
    i = wp.tid(); p = pts[i]
    s = float(0.0); f = int(0); u = float(0.0); v = float(0.0)
    wp.mesh_query_point_sign_normal(m, p, MAXD, s, f, u, v)
    cp = wp.mesh_eval_position(m, f, u, v)
    dist[i] = wp.length(cp - p)
    q = wp.mesh_query_point_sign_normal(m, p, MAXD)
    bad[i] = fneq(s, q.sign) + wp.where(f == q.face, 0, 1) + fneq(u, q.u) + fneq(v, q.v)


@wp.kernel
def k_wn(m: wp.uint64, pts: wp.array[wp.vec3], bad: wp.array[int]):
    i = wp.tid(); p = pts[i]
    s = float(0.0); f = int(0); u = float(0.0); v = float(0.0)
    ok = wp.mesh_query_point_sign_winding_number(m, p, MAXD, s, f, u, v)
    q = wp.mesh_query_point_sign_winding_number(m, p, MAXD)
    bad[i] = wp.where(ok == q.result, 0, 1) + fneq(s, q.sign) + wp.where(f == q.face, 0, 1) + fneq(u, q.u) + fneq(v, q.v)


@wp.kernel
def k_ray(m: wp.uint64, pts: wp.array[wp.vec3], d: wp.vec3, bad: wp.array[int]):
    i = wp.tid(); p = pts[i]
    t = float(0.0); u = float(0.0); v = float(0.0); s = float(0.0); n = wp.vec3(); f = int(0)
    ok = wp.mesh_query_ray(m, p, d, MAXD, t, u, v, s, n, f)
    q = wp.mesh_query_ray(m, p, d, MAXD)
    bad[i] = (wp.where(ok == q.result, 0, 1) + fneq(t, q.t) + fneq(u, q.u) + fneq(v, q.v) + fneq(s, q.sign)
              + vneq(n, q.normal) + wp.where(f == q.face, 0, 1))


@wp.kernel
def k_ray_misc(m: wp.uint64, pts: wp.array[wp.vec3], d: wp.vec3, tmp: wp.array[int], bad: wp.array[int]):
    i = wp.tid(); p = pts[i]
    a = wp.mesh_query_ray_anyhit(m, p, d, MAXD)
    c = wp.mesh_query_ray_count_intersections(m, p, d)
    tmp[i] = c
    a2 = wp.mesh_query_ray_anyhit(m, p, d, MAXD)
    c2 = wp.mesh_query_ray_count_intersections(m, p, d)
    bad[i] = wp.where(a == a2, 0, 1) + wp.where(c == c2, 0, 1)


@wp.func
def maabb_hash(m: wp.uint64, lo: wp.vec3, hi: wp.vec3) -> wp.vec2i:
    q = wp.mesh_query_aabb(m, lo, hi)
    idx = int(0); k = int(0); h = int(0)
    while wp.mesh_query_aabb_next(q, idx):
        k += 1
        h = h * 31 + idx
    return wp.vec2i(k, h)


@wp.kernel
def k_maabb(m: wp.uint64, pts: wp.array[wp.vec3], r: float, tmp: wp.array[int], bad: wp.array[int]):
    i = wp.tid(); p = pts[i]
    e = wp.vec3(r, r, r)
    a = maabb_hash(m, p - e, p + e)
    tmp[i] = a[0]
    b = maabb_hash(m, p - e, p + e)
    bad[i] = wp.where(a[0] == b[0], 0, 1) + wp.where(a[1] == b[1], 0, 1)


@wp.func
def bray_hash(b: wp.uint64, p: wp.vec3, d: wp.vec3, md: float) -> wp.vec2i:
    q = wp.bvh_query_ray(b, p, d)
    idx = int(0); k = int(0); h = int(0)
    while wp.bvh_query_next(q, idx, md):
        k += 1
        h = h * 31 + idx
    return wp.vec2i(k, h)


@wp.func
def baabb_hash(b: wp.uint64, lo: wp.vec3, hi: wp.vec3) -> wp.vec2i:
    q = wp.bvh_query_aabb(b, lo, hi)
    idx = int(0); k = int(0); h = int(0)
    while wp.bvh_query_next(q, idx):
        k += 1
        h = h * 31 + idx
    return wp.vec2i(k, h)


@wp.kernel
def k_bvh(b: wp.uint64, pts: wp.array[wp.vec3], d: wp.vec3, r: float, tmp: wp.array[int], bad: wp.array[int]):
    i = wp.tid(); p = pts[i]
    e = wp.vec3(r, r, r)
    a = bray_hash(b, p, d, 1.0e6)
    a2 = bray_hash(b, p, d, 0.5)
    c = baabb_hash(b, p - e, p + e)
    tmp[i] = a[0] + a2[0] + c[0]
    x = bray_hash(b, p, d, 1.0e6)
    x2 = bray_hash(b, p, d, 0.5)
    y = baabb_hash(b, p - e, p + e)
    bad[i] = (wp.where(a[0] == x[0], 0, 1) + wp.where(a[1] == x[1], 0, 1) + wp.where(a2[0] == x2[0], 0, 1)
              + wp.where(a2[1] == x2[1], 0, 1) + wp.where(c[0] == y[0], 0, 1) + wp.where(c[1] == y[1], 0, 1))


@wp.kernel
def k_helpers(pts: wp.array[wp.vec3], tmp: wp.array[wp.vec3], bad: wp.array[int]):
    i = wp.tid(); n = pts.shape[0]
    p1 = pts[i]; q1 = pts[(i * 7 + 1) % n]; p2 = pts[(i * 13 + 5) % n]; q2 = pts[(i * 29 + 11) % n]
    a = wp.closest_point_edge_edge(p1, q1, p2, q2, 1.0e-6)
    t = wp.intersect_tri_tri(p1, q1, p2, q2, pts[(i * 3 + 2) % n], pts[(i * 17 + 3) % n])
    tmp[i] = a
    b = wp.closest_point_edge_edge(p1, q1, p2, q2, 1.0e-6)
    t2 = wp.intersect_tri_tri(p1, q1, p2, q2, pts[(i * 3 + 2) % n], pts[(i * 17 + 3) % n])
    bad[i] = vneq(a, b) + wp.where(t == t2, 0, 1)


@wp.kernel
def tri_bounds(P: wp.array[wp.vec3], I: wp.array[int], lo: wp.array[wp.vec3], hi: wp.array[wp.vec3]):
    t = wp.tid()
    a = P[I[3 * t]]; b = P[I[3 * t + 1]]; c = P[I[3 * t + 2]]
    lo[t] = wp.min(wp.min(a, b), c)
    hi[t] = wp.max(wp.max(a, b), c)


def spiky():
    st = Usd.Stage.Open(os.path.join(os.path.dirname(T.__file__), "..", "assets", "spiky.usd"))
    g = UsdGeom.Mesh(st.GetPrimAtPath("/Cube/Cube"))
    tri = T.triangulate(g.GetFaceVertexCountsAttr().Get(), g.GetFaceVertexIndicesAttr().Get())
    return np.array(g.GetPointsAttr().Get(), dtype=np.float32), np.array(tri, dtype=np.int32)


def gridcube(n=24):
    # surface of [-1,1]^3 as a regular grid, two triangles per cell, outward winding
    pts, idx, ix = [], [], {}
    def vid(x):
        k = tuple(np.round(x, 6))
        if k not in ix:
            ix[k] = len(pts); pts.append(x)
        return ix[k]
    s = np.linspace(-1.0, 1.0, n + 1)
    for ax in range(3):
        for sg in (-1.0, 1.0):
            for a in range(n):
                for b in range(n):
                    def P(i, j):
                        x = np.zeros(3); x[ax] = sg; x[(ax + 1) % 3] = s[i]; x[(ax + 2) % 3] = s[j]; return vid(x)
                    v00, v10, v11, v01 = P(a, b), P(a + 1, b), P(a + 1, b + 1), P(a, b + 1)
                    if sg > 0:
                        idx += [v00, v10, v11, v00, v11, v01]
                    else:
                        idx += [v00, v11, v10, v00, v01, v11]
    return np.array(pts, dtype=np.float32), np.array(idx, dtype=np.int32)


def points(P, rng):
    g = T.particle_grid(32, 32, 32, np.array([-1.1, -1.1, -1.1]), 0.05, 0.0).astype(np.float32)
    lo, hi = P.min(0), P.max(0); c = (lo + hi) / 2; h = (hi - lo) / 2 * 1.2
    r = (c + (rng.random((200000, 3)) * 2 - 1) * h).astype(np.float32)
    return np.concatenate([g, r])


def main():
    out = {"nofuse": NOFUSE, "results": {}}
    rng = np.random.default_rng(1)
    R = out["results"]
    for mname, (Pn, In) in [("spiky", spiky()), ("gridcube", gridcube())]:
        pn = points(Pn, rng)
        n = len(pn)
        P = wp.array(Pn, dtype=wp.vec3, device=dev); I = wp.array(In.flatten(), dtype=int, device=dev)
        Q = wp.array(pn, dtype=wp.vec3, device=dev)
        bad = wp.zeros(n, dtype=int, device=dev); tmp = wp.zeros(n, dtype=int, device=dev)
        dist = wp.zeros(n, dtype=float, device=dev); tmpv = wp.zeros(n, dtype=wp.vec3, device=dev)
        ntri = len(In) // 3
        lo = wp.zeros(ntri, dtype=wp.vec3, device=dev); hi = wp.zeros(ntri, dtype=wp.vec3, device=dev)
        wp.launch(tri_bounds, ntri, inputs=[P, I, lo, hi], device=dev)
        edge = float(np.linalg.norm(Pn[In[1::3]] - Pn[In[0::3]], axis=1).mean())

        def acc(key, cnt):
            R.setdefault(key, [0, 0]); R[key][0] += int(cnt); R[key][1] += n

        def run(key, k, inputs):
            bad.zero_()
            wp.launch(k, n, inputs=inputs + [bad], device=dev)
            acc(f"{mname}/{key}", np.count_nonzero(bad.numpy()))

        run("helpers(closest_point_edge_edge,intersect_tri_tri)", k_helpers, [Q, tmpv])
        for leaf, con in itertools.product([1, 2, 4], ["sah", "median", "lbvh"]):
            m = wp.Mesh(points=P, indices=I, support_winding_number=True, bvh_constructor=con, bvh_leaf_size=leaf)
            run("mesh_query_point", k_point, [m.id, Q])
            run("mesh_query_point_no_sign", k_nosign, [m.id, Q])
            run("mesh_query_furthest_point_no_sign", k_furthest, [m.id, Q])
            run("mesh_query_point_sign_parity", k_parity, [m.id, Q])
            run("mesh_query_point_sign_normal", k_normal, [m.id, Q])
            run("mesh_query_point_sign_normal(test shape)", k_normal_ctx, [m.id, Q, dist])
            run("mesh_query_point_sign_winding_number", k_wn, [m.id, Q])
            for d in DIRS:
                run("mesh_query_ray", k_ray, [m.id, Q, d])
                run("mesh_query_ray_anyhit/count_intersections", k_ray_misc, [m.id, Q, d, tmp])
            run("mesh_query_aabb", k_maabb, [m.id, Q, edge, tmp])
            b = wp.Bvh(lo, hi, constructor=con, leaf_size=leaf)
            for d in DIRS[:3]:
                run("bvh_query_ray/aabb", k_bvh, [b.id, Q, d, edge, tmp])
    json.dump(out, open(sys.argv[1], "w"), indent=1)
    for k, (c, t) in R.items():
        print(f"{c:8d} / {t:9d}  {k}")


if __name__ == "__main__":
    main()
```

## `hel.py`

Usage: `r.sh <tag> <s>/hel.py [nofuse]`: splits `twice.py`'s helper kernel into `intersect_tri_tri` alone and `closest_point_edge_edge` alone; prints mismatch counts and the first indices.

```python
# usage: hel.py [nofuse]; locate the helper mismatch: k_helpers verbatim from twice.py, but the flag says which helper
import sys
import numpy as np
import warp as wp
sys.path.insert(0, sys.path[0])
import twice as tw
if "nofuse" in sys.argv:
    wp.set_module_options({"fuse_fp": False})

@wp.kernel
def k_helpers2(pts: wp.array[wp.vec3], tmp: wp.array[wp.vec3], bad: wp.array[int]):
    i = wp.tid(); n = pts.shape[0]
    p1 = pts[i]; q1 = pts[(i * 7 + 1) % n]; p2 = pts[(i * 13 + 5) % n]; q2 = pts[(i * 29 + 11) % n]
    a = wp.closest_point_edge_edge(p1, q1, p2, q2, 1.0e-6)
    t = wp.intersect_tri_tri(p1, q1, p2, q2, pts[(i * 3 + 2) % n], pts[(i * 17 + 3) % n])
    tmp[i] = a
    b = wp.closest_point_edge_edge(p1, q1, p2, q2, 1.0e-6)
    t2 = wp.intersect_tri_tri(p1, q1, p2, q2, pts[(i * 3 + 2) % n], pts[(i * 17 + 3) % n])
    bad[i] = tw.vneq(a, b) + 16 * wp.where(t == t2, 0, 1)

@wp.kernel
def k_tt(pts: wp.array[wp.vec3], tmp: wp.array[int], bad: wp.array[int]):
    i = wp.tid(); n = pts.shape[0]
    p1 = pts[i]; q1 = pts[(i * 7 + 1) % n]; p2 = pts[(i * 13 + 5) % n]; q2 = pts[(i * 29 + 11) % n]
    t = wp.intersect_tri_tri(p1, q1, p2, q2, pts[(i * 3 + 2) % n], pts[(i * 17 + 3) % n])
    tmp[i] = t
    t2 = wp.intersect_tri_tri(p1, q1, p2, q2, pts[(i * 3 + 2) % n], pts[(i * 17 + 3) % n])
    bad[i] = wp.where(t == t2, 0, 1)

@wp.kernel
def k_ee(pts: wp.array[wp.vec3], tmp: wp.array[wp.vec3], bad: wp.array[int]):
    i = wp.tid(); n = pts.shape[0]
    p1 = pts[i]; q1 = pts[(i * 7 + 1) % n]; p2 = pts[(i * 13 + 5) % n]; q2 = pts[(i * 29 + 11) % n]
    a = wp.closest_point_edge_edge(p1, q1, p2, q2, 1.0e-6)
    tmp[i] = a
    b = wp.closest_point_edge_edge(p1, q1, p2, q2, 1.0e-6)
    bad[i] = tw.vneq(a, b)

dev = tw.dev
rng = np.random.default_rng(1)
for name, (Pn, In) in [("spiky", tw.spiky()), ("gridcube", tw.gridcube())]:
    pn = tw.points(Pn, rng); n = len(pn)
    Q = wp.array(pn, dtype=wp.vec3, device=dev)
    tv = wp.zeros(n, dtype=wp.vec3, device=dev); ti = wp.zeros(n, dtype=int, device=dev)
    for k, t in ((tw.k_helpers, tv), (k_helpers2, tv), (k_tt, ti), (k_ee, tv)):
        bad = wp.zeros(n, dtype=int, device=dev)
        wp.launch(k, n, inputs=[Q, t, bad], device=dev)
        b = bad.numpy(); d = np.nonzero(b)[0]
        print(name, k.key, len(d), [(int(j), int(b[j])) for j in d[:5]])
```

## `user2.py`

Usage: `r.sh <tag> <s>/user2.py [nofuse]`: user code evaluating one expression twice in a kernel (box distance, 2x2 determinant, `wp.dot`); prints the bitwise-differing count of 2^22 inputs for each.

```python
# usage: user2.py [nofuse]; can user code that evaluates one expression twice in a kernel get two answers?
# Each kernel evaluates a user @wp.func twice on the same inputs: once in straight-line code, once inside a loop whose
# trip count is a runtime value (1), with a different neighbourhood of other uses. Counts inputs whose two results
# differ bitwise, over 2^22 random inputs.
import sys
import numpy as np
import warp as wp

wp.init()
dev = wp.get_device("cuda:0")
if "nofuse" in sys.argv:
    wp.set_module_options({"fuse_fp": False})


@wp.func
def box_d2(p: wp.vec3, lo: wp.vec3, hi: wp.vec3) -> float:
    # a user's copy of the usual point-to-box squared distance
    dx = wp.min(hi[0], wp.max(lo[0], p[0])) - p[0]
    dy = wp.min(hi[1], wp.max(lo[1], p[1])) - p[1]
    dz = wp.min(hi[2], wp.max(lo[2], p[2])) - p[2]
    return dx * dx + dy * dy + dz * dz


@wp.func
def det2(a: float, b: float, c: float, d: float) -> float:
    return a * d - b * c


@wp.kernel
def k_box(p: wp.array[wp.vec3], lo: wp.array[wp.vec3], hi: wp.array[wp.vec3], trip: int,
          o1: wp.array[float], o2: wp.array[float], o3: wp.array[float]):
    i = wp.tid()
    n = p.shape[0]
    j = (i + 1) % n
    a = box_d2(p[i], lo[i], hi[i])
    b = box_d2(p[i], lo[j], hi[j])
    o1[i] = a
    acc = wp.where(a < b, a, b)
    for _k in range(trip):
        c = box_d2(p[i], lo[i], hi[i])
        e = box_d2(p[j], lo[i], hi[i])
        acc += c * 0.5 + e
        o2[i] = c
    o3[i] = acc


@wp.kernel
def k_det(x: wp.array[wp.vec4], trip: int, o1: wp.array[float], o2: wp.array[float], o3: wp.array[float]):
    i = wp.tid()
    v = x[i]
    a = det2(v[0], v[1], v[2], v[3])
    o1[i] = a
    acc = float(0.0)
    for _k in range(trip):
        u = x[(i + 1) % x.shape[0]]
        c = det2(v[0], v[1], v[2], v[3])
        acc += c * u[0] + v[1] * v[2]
        o2[i] = c
    o3[i] = acc


@wp.kernel
def k_dot(x: wp.array[wp.vec3], y: wp.array[wp.vec3], trip: int, o1: wp.array[float], o2: wp.array[float],
          o3: wp.array[float]):
    i = wp.tid()
    a = wp.dot(x[i], y[i])
    o1[i] = a
    acc = float(0.0)
    for _k in range(trip):
        c = wp.dot(x[i], y[i])
        acc += c + x[i][0] * y[i][0]
        o2[i] = c
    o3[i] = acc


n = 1 << 22
rng = np.random.default_rng(3)
f = lambda *s: rng.standard_normal(s).astype(np.float32)
o1, o2, o3 = (wp.zeros(n, dtype=float, device=dev) for _ in range(3))

P = wp.array(f(n, 3), dtype=wp.vec3, device=dev)
c = f(n, 3); h = np.abs(f(n, 3))
LO = wp.array(c - h, dtype=wp.vec3, device=dev); HI = wp.array(c + h, dtype=wp.vec3, device=dev)
wp.launch(k_box, n, inputs=[P, LO, HI, 1, o1, o2, o3], device=dev)
print("box_d2: straight-line vs loop copy differ:", int(np.count_nonzero(o1.numpy() != o2.numpy())), "of", n)

X = wp.array(f(n, 4), dtype=wp.vec4, device=dev)
wp.launch(k_det, n, inputs=[X, 1, o1, o2, o3], device=dev)
print("det2  : straight-line vs loop copy differ:", int(np.count_nonzero(o1.numpy() != o2.numpy())), "of", n)

Y = wp.array(f(n, 3), dtype=wp.vec3, device=dev)
wp.launch(k_dot, n, inputs=[P, Y, 1, o1, o2, o3], device=dev)
print("dot   : straight-line vs loop copy differ:", int(np.count_nonzero(o1.numpy() != o2.numpy())), "of", n)
```

## `perf.py`

Usage: `r.sh <tag> <s>/perf.py <out.json>`: per-launch ms of the mesh query kernels on a 2-million-triangle torus (lbvh and sah), `intersect_tri_tri` and an elementwise control. Run trees interleaved, one process per tree per round.

```python
# usage: perf.py <out.json>; time per launch of mesh query kernels on a large noisy closed torus mesh
# (about 2*10^6 triangles, lbvh and sah, leaf size default), 10^6 query points uniform in the inflated bounding box.
# Each kernel: 2 warm-up launches, then 5 samples of N back-to-back launches + sync; reports the median sample / N.
# Controls: mesh_query_ray (code unchanged by the patch) and an elementwise kernel.
import json, sys, time
import numpy as np
import warp as wp

wp.init()
dev = wp.get_device("cuda:0")


@wp.kernel
def k_point(m: wp.uint64, pts: wp.array[wp.vec3], o: wp.array[float]):
    i = wp.tid()
    q = wp.mesh_query_point(m, pts[i], 1.0e6)
    o[i] = q.u + float(q.face)


@wp.kernel
def k_nosign(m: wp.uint64, pts: wp.array[wp.vec3], o: wp.array[float]):
    i = wp.tid()
    q = wp.mesh_query_point_no_sign(m, pts[i], 1.0e6)
    o[i] = q.u + float(q.face)


@wp.kernel
def k_normal(m: wp.uint64, pts: wp.array[wp.vec3], o: wp.array[float]):
    i = wp.tid()
    q = wp.mesh_query_point_sign_normal(m, pts[i], 1.0e6)
    o[i] = q.sign + q.u + float(q.face)


@wp.kernel
def k_wn(m: wp.uint64, pts: wp.array[wp.vec3], o: wp.array[float]):
    i = wp.tid()
    q = wp.mesh_query_point_sign_winding_number(m, pts[i], 1.0e6)
    o[i] = q.sign + q.u + float(q.face)


@wp.kernel
def k_furthest(m: wp.uint64, pts: wp.array[wp.vec3], o: wp.array[float]):
    i = wp.tid()
    q = wp.mesh_query_furthest_point_no_sign(m, pts[i], 0.0)
    o[i] = q.u + float(q.face)


@wp.kernel
def k_ray(m: wp.uint64, pts: wp.array[wp.vec3], o: wp.array[float]):
    i = wp.tid()
    p = pts[i]
    q = wp.mesh_query_ray(m, p, wp.normalize(-p + wp.vec3(0.01, 0.02, 0.03)), 1.0e6)
    o[i] = q.t + float(q.face)


@wp.kernel
def k_tritri(P: wp.array[wp.vec3], I: wp.array[int], o: wp.array[float]):
    i = wp.tid()
    n = I.shape[0] // 3
    j = (i * 7919) % n
    k = (j + 1) % n
    o[i] = float(wp.intersect_tri_tri(P[I[3 * j]], P[I[3 * j + 1]], P[I[3 * j + 2]], P[I[3 * k]], P[I[3 * k + 1]] + wp.vec3(0.0, 0.001, 0.0), P[I[3 * k + 2]]))


@wp.kernel
def k_axpy(a: wp.array[wp.vec3], o: wp.array[float]):
    i = wp.tid()
    o[i] = o[i] * 0.5 + wp.length(a[i])


def torus(nu, nv, R=1.0, r=0.35, noise=0.01, seed=0):
    rng = np.random.default_rng(seed)
    u = np.linspace(0, 2 * np.pi, nu, endpoint=False)
    v = np.linspace(0, 2 * np.pi, nv, endpoint=False)
    U, V = np.meshgrid(u, v, indexing="ij")
    rr = r * (1 + noise * rng.standard_normal(U.shape))
    P = np.stack([(R + rr * np.cos(V)) * np.cos(U), (R + rr * np.cos(V)) * np.sin(U), rr * np.sin(V)], -1).reshape(-1, 3)
    a = np.arange(nu)[:, None] * nv
    b = np.arange(nv)[None, :]
    a1 = ((np.arange(nu)[:, None] + 1) % nu) * nv
    b1 = (b + 1) % nv
    v00, v10, v11, v01 = a + b, a1 + b, a1 + b1, a + b1
    I = np.stack([v00, v10, v11, v00, v11, v01], -1).reshape(-1)
    return P.astype(np.float32), I.astype(np.int32)


def bench(fn, n):
    for _ in range(2):
        fn()
    wp.synchronize_device(dev)
    s = []
    for _ in range(5):
        t = time.perf_counter()
        for _ in range(n):
            fn()
        wp.synchronize_device(dev)
        s.append((time.perf_counter() - t) / n)
    return float(np.median(s)) * 1e3


def main():
    Pn, In = torus(1000, 1000)
    P = wp.array(Pn, dtype=wp.vec3, device=dev)
    I = wp.array(In, dtype=int, device=dev)
    rng = np.random.default_rng(1)
    nq = 1000000
    q = (rng.random((nq, 3)) * 2 - 1) * np.array([1.5, 1.5, 0.5])
    Q = wp.array(q.astype(np.float32), dtype=wp.vec3, device=dev)
    o = wp.zeros(nq, dtype=float, device=dev)
    res = {}
    for con in ["lbvh", "sah"]:
        m = wp.Mesh(points=P, indices=I, support_winding_number=True, bvh_constructor=con)
        for name, k, N in [("point", k_point, 3), ("no_sign", k_nosign, 3), ("sign_normal", k_normal, 3),
                           ("sign_winding_number", k_wn, 1), ("furthest_no_sign", k_furthest, 1), ("ray", k_ray, 5)]:
            res[f"{con}/{name}"] = bench(lambda: wp.launch(k, nq, inputs=[m.id, Q, o], device=dev), N)
    res["tri_tri"] = bench(lambda: wp.launch(k_tritri, nq, inputs=[P, I, o], device=dev), 50)
    res["elementwise"] = bench(lambda: wp.launch(k_axpy, nq, inputs=[Q, o], device=dev), 200)
    json.dump(res, open(sys.argv[1], "w"), indent=1)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
```

## `perf2.py`

Usage: `r.sh <tag> <s>/perf2.py <out.json>`: `perf.py` reduced to `sign_winding_number`, `sign_normal`, `ray`, `tri_tri` (200 launches per sample) and the control.

```python
# usage: perf2.py (perf.py reduced to sign_winding_number, sign_normal, ray, tri_tri x200, elementwise)
# was: perf.py <out.json>; time per launch of mesh query kernels on a large noisy closed torus mesh
# (about 2*10^6 triangles, lbvh and sah, leaf size default), 10^6 query points uniform in the inflated bounding box.
# Each kernel: 2 warm-up launches, then 5 samples of N back-to-back launches + sync; reports the median sample / N.
# Controls: mesh_query_ray (code unchanged by the patch) and an elementwise kernel.
import json, sys, time
import numpy as np
import warp as wp

wp.init()
dev = wp.get_device("cuda:0")


@wp.kernel
def k_point(m: wp.uint64, pts: wp.array[wp.vec3], o: wp.array[float]):
    i = wp.tid()
    q = wp.mesh_query_point(m, pts[i], 1.0e6)
    o[i] = q.u + float(q.face)


@wp.kernel
def k_nosign(m: wp.uint64, pts: wp.array[wp.vec3], o: wp.array[float]):
    i = wp.tid()
    q = wp.mesh_query_point_no_sign(m, pts[i], 1.0e6)
    o[i] = q.u + float(q.face)


@wp.kernel
def k_normal(m: wp.uint64, pts: wp.array[wp.vec3], o: wp.array[float]):
    i = wp.tid()
    q = wp.mesh_query_point_sign_normal(m, pts[i], 1.0e6)
    o[i] = q.sign + q.u + float(q.face)


@wp.kernel
def k_wn(m: wp.uint64, pts: wp.array[wp.vec3], o: wp.array[float]):
    i = wp.tid()
    q = wp.mesh_query_point_sign_winding_number(m, pts[i], 1.0e6)
    o[i] = q.sign + q.u + float(q.face)


@wp.kernel
def k_furthest(m: wp.uint64, pts: wp.array[wp.vec3], o: wp.array[float]):
    i = wp.tid()
    q = wp.mesh_query_furthest_point_no_sign(m, pts[i], 0.0)
    o[i] = q.u + float(q.face)


@wp.kernel
def k_ray(m: wp.uint64, pts: wp.array[wp.vec3], o: wp.array[float]):
    i = wp.tid()
    p = pts[i]
    q = wp.mesh_query_ray(m, p, wp.normalize(-p + wp.vec3(0.01, 0.02, 0.03)), 1.0e6)
    o[i] = q.t + float(q.face)


@wp.kernel
def k_tritri(P: wp.array[wp.vec3], I: wp.array[int], o: wp.array[float]):
    i = wp.tid()
    n = I.shape[0] // 3
    j = (i * 7919) % n
    k = (j + 1) % n
    o[i] = float(wp.intersect_tri_tri(P[I[3 * j]], P[I[3 * j + 1]], P[I[3 * j + 2]], P[I[3 * k]], P[I[3 * k + 1]] + wp.vec3(0.0, 0.001, 0.0), P[I[3 * k + 2]]))


@wp.kernel
def k_axpy(a: wp.array[wp.vec3], o: wp.array[float]):
    i = wp.tid()
    o[i] = o[i] * 0.5 + wp.length(a[i])


def torus(nu, nv, R=1.0, r=0.35, noise=0.01, seed=0):
    rng = np.random.default_rng(seed)
    u = np.linspace(0, 2 * np.pi, nu, endpoint=False)
    v = np.linspace(0, 2 * np.pi, nv, endpoint=False)
    U, V = np.meshgrid(u, v, indexing="ij")
    rr = r * (1 + noise * rng.standard_normal(U.shape))
    P = np.stack([(R + rr * np.cos(V)) * np.cos(U), (R + rr * np.cos(V)) * np.sin(U), rr * np.sin(V)], -1).reshape(-1, 3)
    a = np.arange(nu)[:, None] * nv
    b = np.arange(nv)[None, :]
    a1 = ((np.arange(nu)[:, None] + 1) % nu) * nv
    b1 = (b + 1) % nv
    v00, v10, v11, v01 = a + b, a1 + b, a1 + b1, a + b1
    I = np.stack([v00, v10, v11, v00, v11, v01], -1).reshape(-1)
    return P.astype(np.float32), I.astype(np.int32)


def bench(fn, n):
    for _ in range(2):
        fn()
    wp.synchronize_device(dev)
    s = []
    for _ in range(5):
        t = time.perf_counter()
        for _ in range(n):
            fn()
        wp.synchronize_device(dev)
        s.append((time.perf_counter() - t) / n)
    return float(np.median(s)) * 1e3


def main():
    Pn, In = torus(1000, 1000)
    P = wp.array(Pn, dtype=wp.vec3, device=dev)
    I = wp.array(In, dtype=int, device=dev)
    rng = np.random.default_rng(1)
    nq = 1000000
    q = (rng.random((nq, 3)) * 2 - 1) * np.array([1.5, 1.5, 0.5])
    Q = wp.array(q.astype(np.float32), dtype=wp.vec3, device=dev)
    o = wp.zeros(nq, dtype=float, device=dev)
    res = {}
    for con in ["lbvh", "sah"]:
        m = wp.Mesh(points=P, indices=I, support_winding_number=True, bvh_constructor=con)
        for name, k, N in [("sign_winding_number", k_wn, 1), ("sign_normal", k_normal, 3), ("ray", k_ray, 5)]:
            res[f"{con}/{name}"] = bench(lambda: wp.launch(k, nq, inputs=[m.id, Q, o], device=dev), N)
    res["tri_tri"] = bench(lambda: wp.launch(k_tritri, nq, inputs=[P, I, o], device=dev), 200)
    res["elementwise"] = bench(lambda: wp.launch(k_axpy, nq, inputs=[Q, o], device=dev), 200)
    json.dump(res, open(sys.argv[1], "w"), indent=1)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
```

## `wnres.py`

Usage: `r.sh <tag> <s>/wnres.py <out.npz>`: saves `perf.py`'s winding-number, sign_normal and point results (lbvh) for one tree; compare two trees' files with numpy.

```python
# usage: wnres.py <out.npz>; perf.py's torus mesh and points: sign_winding_number / sign_normal / point results (lbvh)
import sys
import numpy as np
import warp as wp
sys.path.insert(0, sys.path[0])
import perf as pf

Pn, In = pf.torus(1000, 1000)
P = wp.array(Pn, dtype=wp.vec3, device=pf.dev); I = wp.array(In, dtype=int, device=pf.dev)
q = (np.random.default_rng(1).random((1000000, 3)) * 2 - 1) * np.array([1.5, 1.5, 0.5])
Q = wp.array(q.astype(np.float32), dtype=wp.vec3, device=pf.dev)
o = wp.zeros(len(q), dtype=float, device=pf.dev)
m = wp.Mesh(points=P, indices=I, support_winding_number=True, bvh_constructor="lbvh")
r = {}
for name, k in [("wn", pf.k_wn), ("normal", pf.k_normal), ("point", pf.k_point)]:
    wp.launch(k, len(q), inputs=[m.id, Q, o], device=pf.dev)
    r[name] = o.numpy().copy()
np.savez(sys.argv[1], **r)
```

## `pfsum.py`

Usage: `python3 pfsum.py <dir>`: summarizes `pf-{before,after}-<round>.json` from `perf.py` into the median/ratio/spread table.

```python
# usage: pfsum.py <dir>; summarize pf-{before,after}-<r>.json: per kernel, median over rounds, after/before, and the
# spread (max/min) of each side over rounds as the noise floor
import glob, json, sys
import numpy as np
d = sys.argv[1]
side = {s: [json.load(open(f)) for f in sorted(glob.glob(f"{d}/pf-{s}-*.json"))] for s in ("before", "after")}
print(f"rounds: before {len(side['before'])}, after {len(side['after'])}")
print("| kernel | before ms (median) | after ms (median) | after/before | before spread | after spread |")
print("|---|---|---|---|---|---|")
for k in side["before"][0]:
    b = np.array([r[k] for r in side["before"]]); a = np.array([r[k] for r in side["after"]])
    print(f"| {k} | {np.median(b):.4g} | {np.median(a):.4g} | {np.median(a) / np.median(b):.3f} | "
          f"{b.max() / b.min():.3f} | {a.max() / a.min():.3f} |")
```

## `exsum.py`

Usage: `python3 exsum.py <dir> <prefix> <sideA> <sideB>`: summarizes `<prefix>-<side>-<round>.json` from `exbench.py` (or `perf2.py`) into min/median per side, ratios, and per-round ratios.

```python
# usage: exsum.py <dir> <prefix> <sideA> <sideB>; per case: min/median per side, B/A of mins and medians, per-round B/A
import glob, json, sys
import numpy as np
d, pre, a, b = sys.argv[1:5]
R = {s: [json.load(open(f)) for f in sorted(glob.glob(f"{d}/{pre}-{s}-*.json"), key=lambda f: int(f.rsplit("-", 1)[1][:-5]))] for s in (a, b)}
print(f"rounds: {len(R[a])} / {len(R[b])}")
print(f"| case | {a} us (min / median) | {b} us (min / median) | {b}/{a} (min) | {b}/{a} (median) | per-round {b}/{a} |")
print("|---|---|---|---|---|---|")
for k in R[a][0]:
    x = np.array([r[k] for r in R[a]]); y = np.array([r[k] for r in R[b]])
    pr = " / ".join(f"{q:.3f}" for q in y / x)
    print(f"| {k} | {x.min():.1f} / {np.median(x):.1f} | {y.min():.1f} / {np.median(y):.1f} | {y.min() / x.min():.3f} | {np.median(y) / np.median(x):.3f} | {pr} |")
```

## `exbench.py`

Usage: `r.sh <tag> <s>/exbench.py <out.json> [case ...]`: per-call us of Warp example steps with default module options (addendum to session 33); cases sph, dem, raycast, fluid, wave, marching_cubes, mesh_intersect, navier_stokes, tile_nbody, tile_mcgp, cloth, and the `floor` noise kernel.

```python
# usage: exbench.py <out.json> [name ...]; GPU wall time per call of Warp example steps (default module options)
# Each case: build the example, 3 warm-up calls + sync, then 5 samples of N back-to-back calls + sync; median per call (us).
# Cases follow the fast_math sessions' list (sph, dem, cloth 16 substeps, raycast, a tile kernel) plus fluid, wave,
# marching cubes, mesh_intersect, tile nbody, tile mcgp, FFT Navier-Stokes; noise floor = an unchanged elementwise kernel.
import json, os, sys, time
import numpy as np
import warp as wp

wp.init()
wp.set_device("cuda:0")
dev = wp.get_device("cuda:0")
sys.path.insert(0, os.path.join(os.path.dirname(wp.__file__), "examples", "benchmarks"))


@wp.kernel
def k_floor(a: wp.array[float], b: wp.array[float]):
    i = wp.tid()
    b[i] = b[i] * 0.5 + a[i]


def bench(fn, n):
    for _ in range(3):
        fn()
    wp.synchronize_device(dev)
    s = []
    for _ in range(5):
        t = time.perf_counter()
        for _ in range(n):
            fn()
        wp.synchronize_device(dev)
        s.append((time.perf_counter() - t) / n)
    return float(np.median(s)) * 1e6


def case_floor():
    a = wp.ones(1 << 24, dtype=float); b = wp.zeros(1 << 24, dtype=float)
    return lambda: wp.launch(k_floor, a.shape[0], inputs=[a, b]), 200


def case_sph():
    from warp.examples.core import example_sph as m
    e = m.Example(stage_path=None)
    return e.step, 10


def case_dem():
    from warp.examples.core import example_dem as m
    e = m.Example(stage_path=None)
    return e.step, 5


def case_raycast():
    from warp.examples.core import example_raycast as m
    e = m.Example()
    return e.render, 50


def case_fluid():
    from warp.examples.core import example_fluid as m
    e = m.Example()
    return e.step, 50


def case_wave():
    from warp.examples.core import example_wave as m
    e = m.Example(stage_path=None)
    return e.step, 50


def case_marching_cubes():
    from warp.examples.core import example_marching_cubes as m
    e = m.Example(stage_path=None)
    return e.step, 50


def case_mesh_intersect():
    from warp.examples.core import example_mesh_intersect as m
    e = m.Example(stage_path=None)
    return e.step, 50


def case_navier_stokes():
    from warp.examples.core import example_fft_poisson_navier_stokes_2d as m
    e = m.Example()
    return e.step, 50


def case_tile_nbody():
    from warp.examples.tile import example_tile_nbody as m
    e = m.Example(headless=True)
    return e.step, 20


def case_tile_mcgp():
    from warp.examples.tile import example_tile_mcgp as m
    e = m.Example(seed=42)
    return (lambda: e.render(0)), 3


def case_cloth():
    import benchmark_cloth_warp as bw
    # benchmark_cloth.py runs its benchmark at import; take only the code before that (the Cloth class)
    src = open(os.path.join(os.path.dirname(bw.__file__), "benchmark_cloth.py")).read()
    ns = {"__name__": "benchmark_cloth_defs", "__file__": os.path.join(os.path.dirname(bw.__file__), "benchmark_cloth.py")}
    exec(src.split("# record profiling information")[0], ns)
    cloth = ns["Cloth"](lower=(0.0, 0.0, 0.0), dx=256, dy=256, radius=0.1, stretch_stiffness=1000.0, bend_stiffness=1000.0,
                     shear_stiffness=1000.0, mass=0.1, fix_corners=True)
    integ = bw.WpIntegrator(cloth, "cuda")
    return (lambda: integ.simulate(1.0 / 60.0, 16)), 20


CASES = {k[5:]: v for k, v in globals().items() if k.startswith("case_")}
names = sys.argv[2:] or list(CASES)
res = {}
for name in names:
    try:
        fn, n = CASES[name]()
        res[name] = bench(fn, n)
    except Exception as exc:  # record and continue; a broken case must not hide the others
        res[name] = f"error: {type(exc).__name__}: {exc}"
    print(name, res[name], flush=True)
json.dump(res, open(sys.argv[1], "w"), indent=1)
```

## Interleaved rounds

As run for the timing tables (trees: `before` = a copy of the package with the 4e0d02ab headers and the clone's `warp/bin`; the candidate; the hipRTC `on` worktree), from the MOAT checkout:

```bash
for r in 1 2 3 4 5; do
  for side in before on; do
    case $side in before) T=<before tree>;; on) T=<on worktree>;; esac
    TREE=$T bash utils/timeit.sh warp-1.14 bench -- agent_space/r.sh ex-$side-$r "$PWD/agent_space/s/exbench.py" "$PWD/agent_space/ex-$side-$r.json" >/dev/null
  done
done
```

