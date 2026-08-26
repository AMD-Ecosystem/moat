#!/usr/bin/env python3
"""Performance harness for the cubvh BVH core rewrite (round 2).

Measures BVH build time (host) and GPU query throughput for ray_intersect,
closest_triangle (UDF), and both SDF modes, at three mesh scales, so the
rewrite can be compared old-vs-new on the same host.

Usage: python bench.py --out results.json [--reps 5] [--queries 500000]
"""

import argparse
import json
import time

import numpy as np
import torch

from golden import make_torus, make_queries


def bench_op(fn, reps, warmup=2):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))  # ms
    return {"median_ms": float(np.median(times)),
            "min_ms": float(np.min(times)),
            "max_ms": float(np.max(times)),
            "times_ms": times}


def main():
    import cubvh

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--queries", type=int, default=500000)
    args = ap.parse_args()

    scales = {
        "small10k": (72, 72),        # 10,368 tris
        "med200k": (320, 320),       # 204,800 tris
        "big2m": (1000, 1000),       # 2,000,000 tris
    }

    results = {"device": torch.cuda.get_device_name(0),
               "torch": torch.__version__,
               "n_queries": args.queries, "reps": args.reps}

    for name, (mj, mn) in scales.items():
        v, f = make_torus(mj, mn)
        points, origins, dirs = make_queries(v, f, n_points=args.queries,
                                             n_rays=args.queries, seed=1)
        tp = torch.from_numpy(points).cuda()
        to = torch.from_numpy(origins).cuda()
        td = torch.from_numpy(dirs).cuda()

        # host-side build time (includes triangle prep + BVH construction)
        build_times = []
        for _ in range(args.reps):
            t0 = time.perf_counter()
            bvh = cubvh.cuBVH(v, f)
            build_times.append((time.perf_counter() - t0) * 1000)
        r = {"build": {"median_ms": float(np.median(build_times)),
                       "min_ms": float(np.min(build_times)),
                       "max_ms": float(np.max(build_times)),
                       "times_ms": build_times}}

        # one throwaway query to trigger the lazy GPU upload
        bvh.unsigned_distance(tp[:1024])
        torch.cuda.synchronize()

        r["ray_trace"] = bench_op(lambda: bvh.ray_trace(to, td), args.reps)
        r["udf"] = bench_op(lambda: bvh.unsigned_distance(tp), args.reps)
        r["sdf_watertight"] = bench_op(
            lambda: bvh.signed_distance(tp, mode="watertight"), args.reps)
        r["sdf_raystab"] = bench_op(
            lambda: bvh.signed_distance(tp, mode="raystab"), args.reps)

        results[name] = r
        print(f"[bench] {name} ({len(f)} tris): "
              + ", ".join(f"{k} {v_['median_ms']:.2f}ms"
                          for k, v_ in r.items()))

    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"[bench] wrote {args.out}")


if __name__ == "__main__":
    main()
