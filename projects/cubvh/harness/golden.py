#!/usr/bin/env python3
"""Golden harness for the cubvh BVH core rewrite (round 2).

Captures the observable behavior of the cuBVH query API on deterministic
meshes and fixed query sets, so an independent reimplementation can be
checked against the pre-rewrite build.

Usage:
  python golden.py capture --out DIR    # run against the CURRENT build, save goldens
  python golden.py check   --ref DIR    # run against the CURRENT build, compare to goldens
  python golden.py crossload --ref DIR  # load REF state_dicts with CURRENT build, compare queries

Epsilon policy (documented per op; scene scale is O(1), all float32):
  distances (UDF/SDF/depth): atol 2e-5. Rewrite may reorder float math and
    contract FMAs differently; observed baseline noise vs trimesh is ~2.4e-7,
    so 2e-5 is generous but still catches algorithmic errors.
  face_id: equidistant faces (shared edges/vertices, symmetric meshes)
    legitimately tie, and tie winners are decided by inlining-sensitive
    1-ulp FMA differences, so exact-match cannot be required across builds.
    Policy: NON-TIE mismatches (|dist_a| vs |dist_b| gap > 2e-5) must be
    <= 0.1%; tie mismatches are reported but accepted.
  uvw: compared only where face_id matches, atol 1e-4 (barycentric of the
    closest point; conditioning is worse near degenerate faces).
  ray hit/miss: >= 99.9% agreement; depth/positions compared where both
    builds hit the SAME face.
  raystab sign: >= 99.9% agreement (same pcg32 stream is used by the rewrite,
    so borderline flips should be rare); mismatches reported.
  degenerate (zero-area) faces: the reference build's handling was
    numerically undefined (signed-zero dependent: sometimes NaN, sometimes a
    finite segment distance). The rewrite makes them unable to win distance
    queries, an intentional documented change. Queries the reference resolved
    to a degenerate face (and rays where either build reports a degenerate
    face) are excluded from old-vs-new comparison; instead the current build
    must agree with ITSELF on the same mesh with degenerate faces removed
    (face ids of real faces are unchanged), same tolerances.
  state_dict: cross-load goldens' serialized BVHs into the current build and
    require the same query behavior (validates node layout + traversal of
    foreign node arrays, the compatibility contract).
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

MAX_DIST = 1000.0

ATOL_DIST = 2e-5
ATOL_UVW = 1e-4
ATOL_POS = 1e-4
ID_MATCH_MIN = 0.999


def make_torus(major, minor, R=1.0, r=0.4):
    """Deterministic torus triangulation, watertight, 2*major*minor faces."""
    i = np.arange(major)
    j = np.arange(minor)
    theta = (2 * np.pi / major) * i
    phi = (2 * np.pi / minor) * j
    T, P = np.meshgrid(theta, phi, indexing="ij")
    x = (R + r * np.cos(P)) * np.cos(T)
    y = (R + r * np.cos(P)) * np.sin(T)
    z = r * np.sin(P)
    verts = np.stack([x, y, z], axis=-1).reshape(-1, 3).astype(np.float32)

    ii, jj = np.meshgrid(i, j, indexing="ij")
    v00 = ii * minor + jj
    v01 = ii * minor + (jj + 1) % minor
    v10 = ((ii + 1) % major) * minor + jj
    v11 = ((ii + 1) % major) * minor + (jj + 1) % minor
    f1 = np.stack([v00, v10, v11], axis=-1).reshape(-1, 3)
    f2 = np.stack([v00, v11, v01], axis=-1).reshape(-1, 3)
    faces = np.concatenate([f1, f2], axis=0).astype(np.uint32)
    return verts, faces


def make_meshes():
    """name -> (verts, faces, degen_face_ids). Degenerate faces are always
    appended last so real-face ids are unchanged when they are dropped."""
    meshes = {}

    # small watertight torus ~10k faces
    v, f = make_torus(72, 72)
    meshes["torus10k"] = (v, f, ())

    # open mesh: same torus with a patch removed (open boundary)
    v2, f2 = make_torus(48, 48)
    cent = v2[f2.astype(np.int64)].mean(axis=1)
    keep = ~((cent[:, 0] > 0.9) & (cent[:, 2] > 0.15))
    meshes["open"] = (v2, f2[keep], ())

    # degenerate faces: torus plus zero-area triangles (repeated vertex,
    # collinear) appended at known indices
    v3, f3 = make_torus(24, 24)
    n3 = len(v3)
    extra_v = np.array(
        [[2.0, 0.0, 0.0], [2.5, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=np.float32
    )
    v3 = np.concatenate([v3, extra_v], axis=0)
    extra_f = np.array(
        [
            [n3, n3, n3 + 1],          # repeated vertex
            [n3, n3 + 1, n3 + 2],      # collinear (zero area)
            [5, 5, 5],                 # fully degenerate on the surface
        ],
        dtype=np.uint32,
    )
    meshes["degen"] = (v3, np.concatenate([f3, extra_f], axis=0),
                       tuple(range(len(f3), len(f3) + len(extra_f))))

    return meshes


def make_queries(verts, faces, n_points=20000, n_rays=20000, seed=0):
    rng = np.random.Generator(np.random.PCG64(seed))
    lo = verts.min(axis=0) - 0.25
    hi = verts.max(axis=0) + 0.25

    # volume points + near-surface points (offset face centroids)
    n_vol = n_points // 2
    pts_vol = rng.uniform(lo, hi, size=(n_vol, 3)).astype(np.float32)
    fi = rng.integers(0, len(faces), size=n_points - n_vol)
    cent = verts[faces[fi].astype(np.int64)].mean(axis=1)
    pts_near = (cent + rng.normal(0, 0.02, cent.shape)).astype(np.float32)
    points = np.concatenate([pts_vol, pts_near], axis=0)

    # rays: origins on a sphere outside the mesh, directed at jittered
    # interior targets; plus some guaranteed-miss rays pointing outward
    center = (lo + hi) / 2
    radius = float(np.linalg.norm(hi - lo))
    d = rng.normal(size=(n_rays, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    origins = (center + radius * d).astype(np.float32)
    targets = rng.uniform(lo, hi, size=(n_rays, 3))
    dirs = targets - origins
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    n_miss = n_rays // 10
    dirs[:n_miss] = d[:n_miss]  # outward: cannot hit
    dirs = dirs.astype(np.float32)

    return points, origins, dirs


def run_queries(bvh, points, origins, dirs, device="cuda"):
    tp = torch.from_numpy(points).to(device)
    to = torch.from_numpy(origins).to(device)
    td = torch.from_numpy(dirs).to(device)

    out = {}
    ud, uf, uu = bvh.unsigned_distance(tp, return_uvw=True)
    out["ud_dist"], out["ud_fid"], out["ud_uvw"] = ud.cpu(), uf.cpu(), uu.cpu()
    sd, sf, su = bvh.signed_distance(tp, return_uvw=True, mode="watertight")
    out["sdw_dist"], out["sdw_fid"], out["sdw_uvw"] = sd.cpu(), sf.cpu(), su.cpu()
    rd_, rf_, ru_ = bvh.signed_distance(tp, return_uvw=True, mode="raystab")
    out["sdr_dist"], out["sdr_fid"], out["sdr_uvw"] = rd_.cpu(), rf_.cpu(), ru_.cpu()
    pos, fid, dep = bvh.ray_trace(to, td)
    out["rt_pos"], out["rt_fid"], out["rt_depth"] = pos.cpu(), fid.cpu(), dep.cpu()
    torch.cuda.synchronize()
    return out


def capture(outdir):
    import cubvh

    os.makedirs(outdir, exist_ok=True)
    meta = {"torch": torch.__version__, "device": torch.cuda.get_device_name(0)}
    for name, (v, f, _degen) in make_meshes().items():
        points, origins, dirs = make_queries(v, f)
        bvh = cubvh.cuBVH(v, f)
        out = run_queries(bvh, points, origins, dirs)
        out["state_dict"] = {k: t.cpu() for k, t in bvh.state_dict().items()}
        torch.save(out, os.path.join(outdir, f"{name}.pt"))
        meta[name] = {
            "n_verts": len(v),
            "n_faces": len(f),
            "nodes_shape": list(out["state_dict"]["nodes"].shape),
            "tris_shape": list(out["state_dict"]["triangles"].shape),
        }
        print(f"[capture] {name}: {len(f)} faces, "
              f"nodes {meta[name]['nodes_shape']}, saved")
    with open(os.path.join(outdir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)


def _compare(name, ref, out, results, degen_ids=(), skip_rt=False):
    ok = True
    degen = np.asarray(sorted(degen_ids), dtype=np.int64)

    def rec(key, passed, detail):
        nonlocal ok
        results.append({"mesh": name, "check": key, "pass": bool(passed),
                        "detail": detail})
        if not passed:
            ok = False
            print(f"[FAIL] {name}/{key}: {detail}")
        else:
            print(f"[ ok ] {name}/{key}: {detail}")

    for op, dk, fk, uk in [
        ("ud", "ud_dist", "ud_fid", "ud_uvw"),
        ("sdw", "sdw_dist", "sdw_fid", "sdw_uvw"),
        ("sdr", "sdr_dist", "sdr_fid", "sdr_uvw"),
    ]:
        d_ref, d_new = ref[dk].numpy(), out[dk].numpy()
        f_ref, f_new = ref[fk].numpy(), out[fk].numpy()

        # exclude queries either build resolved to a degenerate face; those
        # are covered by the no-degenerate self-consistency check instead
        cmpmask = ~(np.isin(f_ref, degen) | np.isin(f_new, degen))
        n_masked = int((~cmpmask).sum())
        d_ref, d_new = d_ref[cmpmask], d_new[cmpmask]
        f_ref, f_new = f_ref[cmpmask], f_new[cmpmask]

        if op == "sdr":
            sign_match = (np.sign(d_ref) == np.sign(d_new)).mean()
            rec(f"{op}.sign", sign_match >= ID_MATCH_MIN,
                f"sign match {sign_match:.5f}")
            dmax = np.abs(np.abs(d_ref) - np.abs(d_new)).max()
        else:
            dmax = np.abs(d_ref - d_new).max()
        rec(f"{op}.dist", dmax <= ATOL_DIST,
            f"max |d| diff {dmax:.3e}"
            + (f" ({n_masked} degenerate-won queries excluded)"
               if n_masked else ""))

        # ties (equal distances, winner decided by ulps) are legitimate;
        # only non-tie mismatches count against the budget
        mism = f_ref != f_new
        gap = np.abs(np.abs(d_ref) - np.abs(d_new))
        real = mism & (gap > ATOL_DIST)
        rec(f"{op}.face_id", real.mean() <= 1.0 - ID_MATCH_MIN,
            f"non-tie mismatches {real.sum()}/{len(f_ref)} "
            f"({int(mism.sum())} total mismatches, "
            f"{int(mism.sum() - real.sum())} ties)")

        same_full = (ref[fk].numpy() == out[fk].numpy()) & cmpmask
        u_ref = ref[uk].numpy()[same_full]
        u_new = out[uk].numpy()[same_full]
        nan_same = np.array_equal(np.isnan(u_ref), np.isnan(u_new))
        both_fin = np.isfinite(u_ref) & np.isfinite(u_new)
        umax = np.abs(u_ref[both_fin] - u_new[both_fin]).max()
        rec(f"{op}.uvw", nan_same and umax <= ATOL_UVW,
            f"max uvw diff {umax:.3e}, nan mask equal={nan_same}")

    if skip_rt:
        return ok

    # ray trace: degenerate-face hits are excluded the same way
    rf_ref, rf_new = ref["rt_fid"].numpy(), out["rt_fid"].numpy()
    rmask = ~(np.isin(rf_ref, degen) | np.isin(rf_new, degen))
    n_masked = int((~rmask).sum())
    dep_ref, dep_new = ref["rt_depth"].numpy()[rmask], out["rt_depth"].numpy()[rmask]
    rf_ref, rf_new = rf_ref[rmask], rf_new[rmask]
    hit_ref, hit_new = rf_ref >= 0, rf_new >= 0
    hm = (hit_ref == hit_new).mean()
    rec("rt.hitmiss", hm >= ID_MATCH_MIN,
        f"hit/miss match {hm:.5f}"
        + (f" ({n_masked} degenerate-face rays excluded)" if n_masked else ""))
    both = hit_ref & hit_new
    idm = (rf_ref[both] == rf_new[both]).mean()
    rec("rt.face_id", idm >= ID_MATCH_MIN, f"id match {idm:.5f} (hits)")
    same = both & (rf_ref == rf_new)
    dmax = np.abs(dep_ref[same] - dep_new[same]).max()
    rec("rt.depth", dmax <= ATOL_DIST * 10,
        f"max depth diff {dmax:.3e} (same-face hits)")
    pos_ref = ref["rt_pos"].numpy()[rmask][same]
    pos_new = out["rt_pos"].numpy()[rmask][same]
    pmax = np.abs(pos_ref - pos_new).max()
    rec("rt.pos", pmax <= ATOL_POS, f"max position diff {pmax:.3e}")
    if (~hit_new).any():
        rec("rt.miss_depth", np.allclose(dep_new[~hit_new], MAX_DIST),
            f"miss depth == {MAX_DIST}")
    return ok


def check(refdir):
    import cubvh

    all_ok = True
    results = []
    for name, (v, f, degen_ids) in make_meshes().items():
        ref = torch.load(os.path.join(refdir, f"{name}.pt"), weights_only=False)
        points, origins, dirs = make_queries(v, f)
        bvh = cubvh.cuBVH(v, f)
        out = run_queries(bvh, points, origins, dirs)
        if not _compare(name, ref, out, results, degen_ids):
            all_ok = False

        if degen_ids:
            # the current build must behave as if degenerate faces were
            # absent: same distances/ids against a build on the filtered
            # mesh (real-face ids unchanged; rays excluded -- degenerate
            # ray behavior is not part of the invisibility contract)
            f_filt = np.delete(f, list(degen_ids), axis=0)
            bvh_f = cubvh.cuBVH(v, f_filt)
            out_f = run_queries(bvh_f, points, origins, dirs)
            if not _compare(f"{name}.nodegen", out_f, out, results,
                            skip_rt=True):
                all_ok = False

        # round-trip through the CURRENT build's own state_dict
        sd = bvh.state_dict()
        bvh2 = cubvh.cuBVH.from_state_dict(sd)
        out2 = run_queries(bvh2, points, origins, dirs)
        rt_ok = all(
            torch.equal(out[k].nan_to_num(-12345.0), out2[k].nan_to_num(-12345.0))
            for k in out if not k.startswith("state")
        )
        results.append({"mesh": name, "check": "own_roundtrip", "pass": rt_ok,
                        "detail": "state_dict round-trip bitwise"})
        print(f"[{' ok ' if rt_ok else 'FAIL'}] {name}/own_roundtrip")
        all_ok = all_ok and rt_ok
    return all_ok, results


def crossload(refdir):
    """Load the REFERENCE build's serialized BVHs with the CURRENT build."""
    import cubvh

    all_ok = True
    results = []
    for name, (v, f, degen_ids) in make_meshes().items():
        ref = torch.load(os.path.join(refdir, f"{name}.pt"), weights_only=False)
        points, origins, dirs = make_queries(v, f)
        bvh = cubvh.cuBVH.from_state_dict(ref["state_dict"])
        out = run_queries(bvh, points, origins, dirs)
        if not _compare(f"{name}.crossload", ref, out, results, degen_ids):
            all_ok = False
    return all_ok, results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["capture", "check", "crossload"])
    ap.add_argument("--out", default="goldens")
    ap.add_argument("--ref", default="goldens")
    ap.add_argument("--json", default=None, help="write results as JSON")
    args = ap.parse_args()

    if args.mode == "capture":
        capture(args.out)
        return 0
    fn = check if args.mode == "check" else crossload
    ok, results = fn(args.ref)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(results, fh, indent=2)
    print(f"[golden] {args.mode}: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
