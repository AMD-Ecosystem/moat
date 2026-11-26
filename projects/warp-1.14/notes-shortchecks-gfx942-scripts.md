# warp-1.14 short checks (linux-gfx942, 2026-11-26): scripts

Scripts for "Short checks from AMD-Ecosystem/warp issues and PR #21 (linux-gfx942)" in notes.md.

## `c1_scan_rle.py`

```
import os
import sys
import threading
import time

import numpy as np
import warp as wp

ITERS = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0
TAG = sys.argv[3] if len(sys.argv) > 3 else "p0"
TIMEOUT = 60.0
NMAX = 20_000_000

wp.config.log_level = wp.LOG_WARNING
wp.init()
dev = wp.get_device("cuda:0")
assert wp.is_mempool_enabled(dev)
# release threshold 0: the pool returns memory at each synchronize, so the next
# allocation is freshly mapped
wp.set_mempool_release_threshold(dev, 0)

rng = np.random.default_rng(SEED)
h_i = rng.integers(0, 4, NMAX, dtype=np.int32)
h_f = rng.integers(0, 4, NMAX).astype(np.float32)
vals = rng.integers(-1000, 1000, NMAX // 2, dtype=np.int32)
lens = rng.integers(1, 9, NMAX // 2)
h_r = np.repeat(vals, lens)[:NMAX].astype(np.int32)
assert h_r.size == NMAX

state = {"it": -1, "desc": "", "t0": time.monotonic()}


def watchdog():
    while True:
        time.sleep(1)
        if state["it"] >= ITERS:
            return
        if time.monotonic() - state["t0"] > TIMEOUT:
            print(f"[{TAG}] HANG it={state['it']} {state['desc']} > {TIMEOUT}s", flush=True)
            os._exit(2)


threading.Thread(target=watchdog, daemon=True).start()

errors = 0
t_start = time.monotonic()
for it in range(ITERS):
    n = int(np.exp(rng.uniform(np.log(1e5), np.log(2e7))))
    dt = wp.int32 if it % 2 == 0 else wp.float32
    inclusive = (it // 2) % 2 == 0
    state.update(it=it, desc=f"n={n} dtype={dt.__name__} inclusive={inclusive}", t0=time.monotonic())
    check = it % 50 == 0
    try:
        src = h_i[:n] if dt == wp.int32 else h_f[:n]
        a = wp.array(src, dtype=dt, device=dev)
        o = wp.empty(n, dtype=dt, device=dev)
        wp.utils.array_scan(a, o, inclusive=inclusive)
        r = wp.array(h_r[:n], dtype=wp.int32, device=dev)
        rv = wp.empty(n, dtype=wp.int32, device=dev)
        rl = wp.empty(n, dtype=wp.int32, device=dev)
        cnt = wp.utils.runlength_encode(r, rv, rl)
        if check:
            got = o.numpy()
            ref = np.cumsum(src.astype(np.float64))
            if not inclusive:
                ref = np.concatenate(([0.0], ref[:-1]))
            if dt == wp.int32:
                ok_scan = np.array_equal(got.astype(np.int64), ref.astype(np.int64))
            else:
                ok_scan = np.allclose(got, ref, rtol=1e-4, atol=1.0)
            x = h_r[:n]
            starts = np.flatnonzero(np.concatenate(([True], x[1:] != x[:-1])))
            ref_l = np.diff(np.concatenate((starts, [n])))
            ok_rle = (
                cnt == starts.size
                and np.array_equal(rv.numpy()[:cnt], x[starts])
                and np.array_equal(rl.numpy()[:cnt], ref_l)
            )
            if not (ok_scan and ok_rle):
                errors += 1
                print(f"[{TAG}] WRONG it={it} {state['desc']} scan_ok={ok_scan} rle_ok={ok_rle} cnt={cnt} ref_cnt={starts.size}", flush=True)
        del a, o, r, rv, rl
        wp.synchronize()
    except Exception as e:
        errors += 1
        print(f"[{TAG}] ERROR it={it} {state['desc']}: {type(e).__name__}: {e}", flush=True)
        break
    if it % 250 == 0:
        print(f"[{TAG}] it={it} {state['desc']} ok elapsed={time.monotonic() - t_start:.1f}s", flush=True)
state["it"] = ITERS
print(f"[{TAG}] DONE iters={ITERS} errors={errors} elapsed={time.monotonic() - t_start:.1f}s", flush=True)
```

## `c2_event_external_same_stream.py`

```
# test_event_external_same_stream from tomasthoresen/warp 51764a81b3
# (added in d03ac24a, warp/tests/cuda/test_streams.py), run standalone.
import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import StdErrCapture, assert_np_equal

wp.config.log_level = wp.LOG_WARNING


@wp.kernel
def inc(a: wp.array[float]):
    tid = wp.tid()
    a[tid] = a[tid] + 1.0


@wp.kernel
def inc_new(src: wp.array[float], dst: wp.array[float]):
    tid = wp.tid()
    dst[tid] = src[tid] + 1.0


class T(unittest.TestCase):
    def test_event_external_same_stream(self):
        test = self
        device = "cuda:0"
        with wp.ScopedDevice(device):
            event = wp.Event()
            n = 1024
            a = wp.zeros(n, dtype=float)
            b = wp.zeros(n, dtype=float)

            capture_stderr = StdErrCapture()
            capture_stderr.begin()
            try:
                with wp.ScopedCapture() as capture:
                    wp.launch(inc, dim=n, inputs=[a])
                    wp.record_event(event, external=True)
                    wp.wait_event(event, external=True)
                    wp.launch(inc_new, dim=n, inputs=[a, b])
            finally:
                errors = capture_stderr.end()
            if errors:
                print("CAPTURED STDERR:", errors.strip())
            test.assertNotIn("error", errors.lower())

            for _ in range(3):
                wp.capture_launch(capture.graph)
            wp.synchronize_device()

            assert_np_equal(a.numpy(), np.full(n, 3.0, dtype=np.float32))
            assert_np_equal(b.numpy(), np.full(n, 4.0, dtype=np.float32))


if __name__ == "__main__":
    wp.init()
    unittest.main()
```

## `c3_tile_query.py`

```
# GH-2181 probe: tiled BVH / mesh AABB queries vs thread-level queries on a
# 59,168-triangle UV sphere (SAH, leaf_size=1), large query boxes.
import sys

import numpy as np

import warp as wp

wp.config.log_level = wp.LOG_WARNING
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
NQ = 512
BLOCK_DIMS = [64, 128, 256]


def uv_sphere(slices=172, stacks=173):
    pts = [(0.0, 0.0, 1.0)]
    for i in range(1, stacks):
        th = np.pi * i / stacks
        for j in range(slices):
            ph = 2 * np.pi * j / slices
            pts.append((np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)))
    pts.append((0.0, 0.0, -1.0))
    pts = np.array(pts, dtype=np.float32)
    s = pts.shape[0] - 1
    tris = []
    ring = lambda i, j: 1 + (i - 1) * slices + (j % slices)
    for j in range(slices):
        tris.append((0, ring(1, j), ring(1, j + 1)))
    for i in range(1, stacks - 1):
        for j in range(slices):
            a, b, c, d = ring(i, j), ring(i, j + 1), ring(i + 1, j), ring(i + 1, j + 1)
            tris.append((a, c, b))
            tris.append((b, c, d))
    for j in range(slices):
        tris.append((ring(stacks - 1, j), s, ring(stacks - 1, j + 1)))
    return pts, np.array(tris, dtype=np.int32)


@wp.kernel
def bvh_thread(bid: wp.uint64, lo: wp.array[wp.vec3], hi: wp.array[wp.vec3], cnt: wp.array[int]):
    i = wp.tid()
    q = wp.bvh_query_aabb(bid, lo[i], hi[i])
    k = int(0)
    c = int(0)
    while wp.bvh_query_next(q, k):
        c += 1
    cnt[i] = c


@wp.kernel
def mesh_thread(mid: wp.uint64, lo: wp.array[wp.vec3], hi: wp.array[wp.vec3], cnt: wp.array[int]):
    i = wp.tid()
    q = wp.mesh_query_aabb(mid, lo[i], hi[i])
    k = int(0)
    c = int(0)
    while wp.mesh_query_aabb_next(q, k):
        c += 1
    cnt[i] = c


@wp.kernel
def bvh_tile(bid: wp.uint64, lo: wp.array[wp.vec3], hi: wp.array[wp.vec3], cnt: wp.array[int]):
    i, _j = wp.tid()
    q = wp.tile_bvh_query_aabb(bid, lo[i], hi[i])
    while wp.tile_query_valid(q):
        r = wp.untile(wp.tile_bvh_query_next(q))
        if r >= 0:
            wp.atomic_add(cnt, i, 1)


@wp.kernel
def mesh_tile(mid: wp.uint64, lo: wp.array[wp.vec3], hi: wp.array[wp.vec3], cnt: wp.array[int]):
    i, _j = wp.tid()
    q = wp.tile_mesh_query_aabb(mid, lo[i], hi[i])
    while wp.tile_query_valid(q):
        r = wp.untile(wp.tile_mesh_query_aabb_next(q))
        if r >= 0:
            wp.atomic_add(cnt, i, 1)


wp.init()
pts, tris = uv_sphere()
assert tris.shape[0] == 59168, tris.shape
tri_pts = pts[tris]
tlo, thi = tri_pts.min(axis=1), tri_pts.max(axis=1)

rng = np.random.default_rng(SEED)
c = rng.uniform(-1.0, 1.0, (NQ, 3)).astype(np.float32)
h = rng.uniform(0.3, 1.5, (NQ, 3)).astype(np.float32)
qlo, qhi = c - h, c + h
# brute-force reference: triangle AABB overlaps query box
ref = np.array([np.count_nonzero(np.all((tlo <= qhi[k]) & (thi >= qlo[k]), axis=1)) for k in range(NQ)])

with wp.ScopedDevice("cuda:0"):
    lo = wp.array(qlo, dtype=wp.vec3)
    hi = wp.array(qhi, dtype=wp.vec3)
    bvh = wp.Bvh(wp.array(tlo, dtype=wp.vec3), wp.array(thi, dtype=wp.vec3), constructor="sah", leaf_size=1)
    mesh = wp.Mesh(wp.array(pts, dtype=wp.vec3), wp.array(tris.flatten(), dtype=int), bvh_constructor="sah", bvh_leaf_size=1)
    out = {}
    for name, obj, tk, kk in (("bvh", bvh, bvh_thread, bvh_tile), ("mesh", mesh, mesh_thread, mesh_tile)):
        t = wp.zeros(NQ, dtype=int)
        wp.launch(tk, dim=NQ, inputs=[obj.id, lo, hi, t])
        tcount = t.numpy()
        print(f"seed={SEED} {name} thread: wrong_vs_bruteforce={np.count_nonzero(tcount != ref)}/{NQ} mean_hits={tcount.mean():.0f} max_hits={tcount.max()}", flush=True)
        for bd in BLOCK_DIMS:
            g = wp.zeros(NQ, dtype=int)
            wp.launch_tiled(kk, dim=NQ, inputs=[obj.id, lo, hi, g], block_dim=bd)
            gcount = g.numpy()
            bad = np.flatnonzero(gcount != tcount)
            ex = "" if bad.size == 0 else f" e.g. q{bad[0]} tile={gcount[bad[0]]} thread={tcount[bad[0]]}"
            print(f"seed={SEED} {name} tile bd={bd}: wrong={bad.size}/{NQ}{ex}", flush=True)
```

## `run_cf2.sh`

```
#!/bin/bash
# run_cf2.sh <label> <loads> <iterations> [fragoff]: start <loads> copy_flake load processes on the pinned GPU,
# keep their PIDs, run copy_flake2 <iterations> (timeout 900 s), then kill only those PIDs.
# fragoff: HSA_DISABLE_FRAGMENT_ALLOCATOR=1 for copy_flake2 and the loads.
set -u
ulimit -c 0
cd "$(dirname "$0")"
export LD_LIBRARY_PATH=$FLEET_VENV/lib/python3.11/site-packages/_rocm_sdk_core/lib
L=$1; NL=$2; IT=$3; MODE=${4:-default}
if [ "$MODE" = fragoff ]; then export HSA_DISABLE_FRAGMENT_ALLOCATOR=1; else unset HSA_DISABLE_FRAGMENT_ALLOCATOR; fi
pids=()
for k in $(seq $NL); do ./copy_flake load 1200 4 > logs/${L}_load$k.log 2>&1 & pids+=($!); done
echo "label=$L mode=$MODE HSA_DISABLE_FRAGMENT_ALLOCATOR=${HSA_DISABLE_FRAGMENT_ALLOCATOR:-unset} HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES load_pids=${pids[*]}" > logs/$L.log
sleep 3
alive=0; for p in "${pids[@]}"; do kill -0 $p 2>/dev/null && alive=$((alive+1)); done
echo "loads alive at start: $alive/$NL" >> logs/$L.log
t0=$(date +%s)
timeout 900 ./copy_flake2 $IT >> logs/$L.log 2>&1; rc=$?
t1=$(date +%s)
alive=0; for p in "${pids[@]}"; do kill -0 $p 2>/dev/null && alive=$((alive+1)); done
echo "rc=$rc time=$((t1-t0))s loads alive at end: $alive/$NL" >> logs/$L.log
for p in "${pids[@]}"; do kill $p 2>/dev/null; done
wait
```
