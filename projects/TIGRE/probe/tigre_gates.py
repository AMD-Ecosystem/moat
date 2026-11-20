"""GPU gates for the TIGRE HIP port, recreated 2026-11-20 from the descriptions in
notes.md (the earlier tigre_validate.py / tigre_solvers.py / tigre_parallel.py /
repeat_par*.py were host-local and lost). Kept here so later rounds run the same
checks.

usage: python tigre_gates.py <TIGRE>/Python cone|solvers|parallel|repeat|all

cone     256^3 head phantom, dDetector 1.6, 32 angles + 0.3: interp vs Siddon < 1%,
         FDK finite and < 0.5, OS-SART-20 finite and < FDK, adjointness
         (interpolated Ax, matched Atb; random x and phantom x) < 0.05.
solvers  64^3 default low-res geometry, 60 angles, 5 iterations, nine solvers; bar = finite.
parallel 128^3 parallel geometry, head phantom; angle sets axis (linspace(0,pi,64)),
         offset (linspace(0,pi,64,endpoint=False)+0.3), Euler (offset alphas, theta 0.4,
         psi 0.2): interp vs Siddon < 1%, Siddon and interp adjointness < 0.05; FBP and
         SIRT-20 at 180 angles; missed rays (64^3, 129^2 detector, offOrigin (0,7.3,-5.1),
         5 angles) return in < 30 s and finite.
repeat   128^3, offset angle set, 20 repeated Ax calls compared bitwise with the first,
         phantom and random input, Siddon and interpolated, parallel and cone.
"""
import sys
import time

import numpy as np

sys.path.insert(0, sys.argv[1])
import tigre  # noqa: E402
import tigre.algorithms as algs  # noqa: E402
from tigre.utilities import sample_loader  # noqa: E402

FAIL = []


def nrmse(x, ref):
    return float(np.linalg.norm(x - ref) / np.linalg.norm(ref))


def check(name, ok):
    if not ok:
        FAIL.append(name)


def adjoint(geo, angles, ptype, x):
    rng = np.random.default_rng(1)
    y = rng.random((len(angles), int(geo.nDetector[0]), int(geo.nDetector[1])), dtype=np.float32)
    ax = tigre.Ax(x, geo, angles, ptype)
    aty = tigre.Atb(y, geo, angles, "matched")
    lhs = float(np.dot(ax.ravel().astype(np.float64), y.ravel()))
    rhs = float(np.dot(x.ravel().astype(np.float64), aty.ravel()))
    return abs(lhs - rhs) / abs(lhs)


def cone():
    geo = tigre.geometry_default(high_resolution=False, nVoxel=np.array([256, 256, 256]))
    geo.dDetector = np.array([1.6, 1.6])
    geo.sDetector = geo.nDetector * geo.dDetector
    angles = (np.linspace(0, 2 * np.pi, 32, endpoint=False) + 0.3).astype(np.float32)
    head = sample_loader.load_head_phantom(geo.nVoxel).astype(np.float32)
    sid = tigre.Ax(head, geo, angles, "Siddon")
    itp = tigre.Ax(head, geo, angles, "interpolated")
    fdk = algs.fdk(sid, geo, angles)
    oss = algs.ossart(sid, geo, angles, 20, computel2=False)
    rnd = np.random.default_rng(0).random(tuple(geo.nVoxel), dtype=np.float32)
    r = dict(sid_mean=float(sid.mean()), sid_max=float(sid.max()), interp_vs_siddon=nrmse(itp, sid),
             fdk=nrmse(fdk, head), ossart20=nrmse(oss, head),
             adj_random=adjoint(geo, angles, "interpolated", rnd),
             adj_phantom=adjoint(geo, angles, "interpolated", head))
    print("cone: Siddon mean/max %.2f/%.1f, interp vs Siddon %.4f, FDK %.4f, OS-SART-20 %.4f, "
          "adjointness %.3e / %.3e" % tuple(r.values()))
    check("cone interp", r["interp_vs_siddon"] < 0.01)
    check("cone fdk", np.isfinite(r["fdk"]) and r["fdk"] < 0.5)
    check("cone ossart", np.isfinite(r["ossart20"]) and r["ossart20"] < r["fdk"])
    check("cone adjoint", r["adj_random"] < 0.05 and r["adj_phantom"] < 0.05)


def solvers():
    geo = tigre.geometry_default(high_resolution=False)
    angles = np.linspace(0, 2 * np.pi, 60, endpoint=False).astype(np.float32)
    head = sample_loader.load_head_phantom(geo.nVoxel).astype(np.float32)
    proj = tigre.Ax(head, geo, angles)
    out = []
    for name in ("sart", "ossart", "sirt", "cgls", "fista", "asd_pocs", "awasd_pocs", "os_asd_pocs", "ossart_tv"):
        rec = getattr(algs, name)(proj, geo, angles, 5, computel2=False) if name not in ("cgls", "fista") \
            else getattr(algs, name)(proj, geo, angles, 5)
        if isinstance(rec, tuple):
            rec = rec[0]
        e = nrmse(rec, head)
        out.append("%s %.4f" % (name, e))
        check("solver " + name, np.isfinite(e) and np.all(np.isfinite(rec)))
    print("solvers: " + ", ".join(out))


def parallel():
    geo = tigre.geometry(mode="parallel", nVoxel=np.array([128, 128, 128]), default=True)
    head = sample_loader.load_head_phantom(geo.nVoxel).astype(np.float32)
    rnd = np.random.default_rng(0).random(tuple(geo.nVoxel), dtype=np.float32)
    axis = np.linspace(0, np.pi, 64).astype(np.float32)
    offset = (np.linspace(0, np.pi, 64, endpoint=False) + 0.3).astype(np.float32)
    euler = np.stack([offset, np.full_like(offset, 0.4), np.full_like(offset, 0.2)], axis=1).astype(np.float32)
    for name, a in (("axis", axis), ("offset", offset), ("Euler", euler)):
        sid = tigre.Ax(head, geo, a, "Siddon")
        itp = tigre.Ax(head, geo, a, "interpolated")
        e, js, ji = nrmse(itp, sid), adjoint(geo, a, "Siddon", rnd), adjoint(geo, a, "interpolated", rnd)
        print("parallel %s: interp vs Siddon %.4f, adjointness Siddon %.3e interp %.3e" % (name, e, js, ji))
        check("parallel " + name, e < 0.01 and js < 0.05 and ji < 0.05)
    a180 = np.linspace(0, np.pi, 180, endpoint=False).astype(np.float32)
    p = tigre.Ax(head, geo, a180)
    fbp = nrmse(algs.fbp(p, geo, a180), head)
    sirt = nrmse(algs.sirt(p, geo, a180, 20, computel2=False), head)
    print("parallel FBP %.4f, SIRT-20 %.4f" % (fbp, sirt))
    check("parallel fbp/sirt", np.isfinite(fbp) and np.isfinite(sirt))
    g = tigre.geometry(mode="parallel", nVoxel=np.array([64, 64, 64]), default=True)
    g.nDetector = np.array([129, 129])
    g.sDetector = g.nDetector * g.dDetector
    g.offOrigin = np.array([0, 7.3, -5.1])
    x = sample_loader.load_head_phantom(g.nVoxel).astype(np.float32)
    a5 = np.linspace(0, np.pi, 5, endpoint=False).astype(np.float32)
    t0 = time.time()
    ps = tigre.Ax(x, g, a5, "Siddon")
    pi_ = tigre.Ax(x, g, a5, "interpolated")
    dt = time.time() - t0
    print("parallel missed rays: %.2f s, finite %s, zero fraction %.3f Siddon / %.3f interp"
          % (dt, bool(np.all(np.isfinite(ps)) and np.all(np.isfinite(pi_))), float((ps == 0).mean()),
             float((pi_ == 0).mean())))
    check("parallel missed", dt < 30 and np.all(np.isfinite(ps)) and np.all(np.isfinite(pi_)))


def repeat():
    a = (np.linspace(0, np.pi, 64, endpoint=False) + 0.3).astype(np.float32)
    for mode in ("parallel", "cone"):
        geo = tigre.geometry(mode=mode, nVoxel=np.array([128, 128, 128]), default=True)
        head = sample_loader.load_head_phantom(geo.nVoxel).astype(np.float32)
        rnd = np.random.default_rng(0).random(tuple(geo.nVoxel), dtype=np.float32)
        for ptype in ("Siddon", "interpolated"):
            for xname, x in (("phantom", head), ("random", rnd)):
                ref = tigre.Ax(x, geo, a, ptype)
                bad, worst = 0, 0.0
                for _ in range(20):
                    r = tigre.Ax(x, geo, a, ptype)
                    if not np.array_equal(r, ref):
                        bad += 1
                        worst = max(worst, float(np.abs(r - ref).max() / np.abs(ref).max()))
                print("repeat %s %s %s: %d/20 differ, worst %.3e" % (mode, ptype, xname, bad, worst))
                check("repeat %s %s %s" % (mode, ptype, xname), bad == 0)


what = sys.argv[2]
for fn in (cone, solvers, parallel, repeat):
    if what in (fn.__name__, "all"):
        fn()
print("RESULT", "PASS" if not FAIL else "FAIL " + ", ".join(FAIL))
sys.exit(1 if FAIL else 0)
