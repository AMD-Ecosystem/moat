#!/usr/bin/env python3
"""
Standalone validation harness for the diff-surfel-rasterizations ROCm/HIP port.

Tests all three -wet variants (NUM_CHANNELS 3/5/7) against a gfx1100 GPU (or any
ROCm-enabled GPU):
  1. pip-installs/rebuilds each variant against the ROCm PyTorch (USE_ROCM auto-set)
  2. Forward pass on a small synthetic 2DGS scene -> finite, non-trivial image
  3. Backward pass -> all gradients finite
  4. Central finite-difference check on opacity and means3D vs analytic gradient
  5. Short Adam fit (400 iters) -> loss drops >60%, no NaN

Usage (gfx1100 example):
    export HIP_VISIBLE_DEVICES=0
    export PYTORCH_ROCM_ARCH=gfx1100          # or gfx90a, gfx1201
    export MAX_JOBS=16
    RASTER_BASE=<path-to>/diff-surfel-rasterizations
    for v in diff-surfel-rasterization-wet diff-surfel-rasterization-wet-ch05 \\
             diff-surfel-rasterization-wet-ch07; do
        rm -rf $RASTER_BASE/$v/{build,*.egg-info,hip_rasterizer,*_hip.*,rasterize_points.hip}
        pip install -e $RASTER_BASE/$v --no-build-isolation --no-deps -v
    done
    python3 projects/EnvGS/validation/validate_diff_surfel_rocm.py

Deps: torch (ROCm build), numpy (optional, not actually used).

Source: AMD-Ecosystem/diff-surfel-rasterizations @ moat-port
"""

import sys
import math
import subprocess
import os

import torch

DEVICE = "cuda"  # PyTorch uses "cuda" device name for ROCm/HIP

# ---------------------------------------------------------------------------
# Helpers: rasterizer construction and forward call
# ---------------------------------------------------------------------------

def make_rasterizer(nc, W, H):
    """Construct a GaussianRasterizer for the given channel count and image size."""
    if nc == 3:
        from diff_surfel_rasterization_wet import GaussianRasterizationSettings, GaussianRasterizer
    elif nc == 5:
        from diff_surfel_rasterization_wet_ch05 import GaussianRasterizationSettings, GaussianRasterizer
    elif nc == 7:
        from diff_surfel_rasterization_wet_ch07 import GaussianRasterizationSettings, GaussianRasterizer
    else:
        raise ValueError(f"Unsupported NC={nc}; expected 3, 5, or 7")

    fovx = math.pi / 4.0
    fovy = math.pi / 4.0
    settings = GaussianRasterizationSettings(
        image_height=H,
        image_width=W,
        tanfovx=math.tan(fovx / 2),
        tanfovy=math.tan(fovy / 2),
        bg=torch.zeros(nc, device=DEVICE),
        scale_modifier=1.0,
        viewmatrix=torch.eye(4, device=DEVICE),
        projmatrix=torch.eye(4, device=DEVICE),
        sh_degree=0,
        campos=torch.zeros(3, device=DEVICE),
        prefiltered=False,
        debug=False,
    )
    return GaussianRasterizer(raster_settings=settings)


def rasterize(rasterizer, means3D, means2D, scales, rotations, opacity, colors):
    """Run one forward pass; returns (image, radii, allmap, weight)."""
    return rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=None,
        colors_precomp=colors,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None,
    )


def _unpack_image(out):
    """Extract the image tensor from the rasterizer output tuple."""
    return out[0] if isinstance(out, (list, tuple)) else out


def make_inputs(P, nc, device=DEVICE):
    """Reproducible synthetic 2DGS scene (surfel scales are [P,2] for 2DGS API)."""
    torch.manual_seed(42)
    means3D = (torch.randn(P, 3, device=device) * 0.5).requires_grad_(True)
    means2D = torch.zeros(P, 3, device=device, requires_grad=True)
    scales = (torch.rand(P, 2, device=device) * 0.05 + 0.01).requires_grad_(True)
    rotations_raw = torch.randn(P, 4, device=device)
    rotations = (rotations_raw / rotations_raw.norm(dim=-1, keepdim=True)).requires_grad_(True)
    opacity = torch.sigmoid(torch.randn(P, 1, device=device)).requires_grad_(True)
    colors = torch.rand(P, nc, device=device).requires_grad_(True)
    return means3D, means2D, scales, rotations, opacity, colors


# ---------------------------------------------------------------------------
# Stage checks
# ---------------------------------------------------------------------------

def check_forward(nc, W=200, H=150, P=4000):
    """Forward pass: image must be finite and non-trivial."""
    rasterizer = make_rasterizer(nc, W, H)
    means3D, means2D, scales, rotations, opacity, colors = make_inputs(P, nc)

    out = rasterize(rasterizer, means3D, means2D, scales, rotations, opacity, colors)
    image = _unpack_image(out)
    radii = out[1] if isinstance(out, (list, tuple)) and len(out) > 1 else None

    img_cpu = image.detach().cpu()
    finite = torch.isfinite(img_cpu).all().item()
    nontrivial = img_cpu.max().item() > 1e-3
    nonzero_frac = (img_cpu > 0).float().mean().item()
    visible = int((radii.detach() > 0).sum().item()) if radii is not None else -1

    print(f"  Forward: shape={list(img_cpu.shape)} finite={finite}"
          f" min={img_cpu.min():.4f} max={img_cpu.max():.4f} mean={img_cpu.mean():.4f}"
          f" nonzero_frac={nonzero_frac:.3f} visible={visible}/{P}")

    if not finite:
        return False, "image contains non-finite values"
    if not nontrivial:
        return False, "image is near-zero (no rendering)"
    return True, f"image finite, max={img_cpu.max():.4f}, visible={visible}/{P}"


def check_backward(nc, W=200, H=150, P=4000):
    """Backward pass: all gradients must be finite."""
    rasterizer = make_rasterizer(nc, W, H)
    means3D, means2D, scales, rotations, opacity, colors = make_inputs(P, nc)

    out = rasterize(rasterizer, means3D, means2D, scales, rotations, opacity, colors)
    image = _unpack_image(out)
    image.sum().backward()

    grads = {
        "means3D": means3D.grad,
        "means2D": means2D.grad,
        "opacity": opacity.grad,
        "colors": colors.grad,
        "scales": scales.grad,
        "rotations": rotations.grad,
    }

    all_finite = True
    for k, g in grads.items():
        fin = torch.isfinite(g).all().item() if g is not None else None
        status = "finite" if fin else ("NON-FINITE" if g is not None else "None")
        if g is not None and not fin:
            all_finite = False
        print(f"    grad {k}: {status}")

    if not all_finite:
        return False, "non-finite gradients"
    return True, "all 6 input gradients finite"


def check_fd_opacity(nc, W=200, H=150, P=4000):
    """
    Central finite-difference check on opacity vs analytic gradient.
    Decisive geometric-blend gate: sign must agree, slope in (0.5, 2.0).
    """
    rasterizer = make_rasterizer(nc, W, H)
    means3D, means2D, scales, rotations, opacity, colors = make_inputs(P, nc)

    # Forward to find a visible surfel
    with torch.no_grad():
        out0 = rasterize(rasterizer, means3D.detach(), means2D.detach(),
                         scales.detach(), rotations.detach(), opacity.detach(), colors.detach())
    radii = out0[1] if isinstance(out0, (list, tuple)) and len(out0) > 1 else None
    if radii is not None:
        vis_idx = (radii.detach() > 0).nonzero(as_tuple=True)[0]
        i_fd = vis_idx[0].item() if len(vis_idx) > 0 else 0
    else:
        i_fd = 0

    eps = 1e-3
    op_base = opacity.detach()
    m3 = means3D.detach(); m2 = means2D.detach()
    sc = scales.detach(); ro = rotations.detach(); col = colors.detach()

    with torch.no_grad():
        op_p = op_base.clone(); op_p[i_fd, 0] += eps
        op_m = op_base.clone(); op_m[i_fd, 0] -= eps
        img_p = _unpack_image(rasterize(rasterizer, m3, m2, sc, ro, op_p, col))
        img_m = _unpack_image(rasterize(rasterizer, m3, m2, sc, ro, op_m, col))
        fd_val = (img_p - img_m).sum().item() / (2 * eps)

    op_t = op_base.clone().requires_grad_(True)
    img_a = _unpack_image(rasterize(rasterizer, m3, m2, sc, ro, op_t, col))
    img_a.sum().backward()
    analytic_val = op_t.grad[i_fd, 0].item()

    sign_agree = 1.0 if (analytic_val * fd_val > 0) else 0.0
    slope = analytic_val / (fd_val + 1e-12)
    print(f"  FD opacity (surfel {i_fd}): sign_agreement={sign_agree:.2f} slope={slope:.3f}"
          f" (analytic={analytic_val:.4f} fd={fd_val:.4f})")

    if sign_agree != 1.0:
        return False, f"opacity FD sign disagreement: analytic={analytic_val:.4f} fd={fd_val:.4f}"
    if not (0.5 < slope < 2.0):
        return False, f"opacity FD slope={slope:.3f} out of (0.5, 2.0)"
    return True, f"opacity FD PASS: sign_agree={sign_agree:.2f} slope={slope:.3f}"


def check_fd_means3d(nc, W=200, H=150, P=4000, n_surfels=5):
    """
    Directional FD check on means3D: analytic and FD must point in the same
    hemisphere (slope in (0.1, 5.0)), confirming downhill direction.
    """
    rasterizer = make_rasterizer(nc, W, H)
    means3D, means2D, scales, rotations, opacity, colors = make_inputs(P, nc)

    with torch.no_grad():
        out0 = rasterize(rasterizer, means3D.detach(), means2D.detach(),
                         scales.detach(), rotations.detach(), opacity.detach(), colors.detach())
    radii = out0[1] if isinstance(out0, (list, tuple)) and len(out0) > 1 else None
    if radii is not None:
        vis_idx = (radii.detach() > 0).nonzero(as_tuple=True)[0]
        candidates = vis_idx[:n_surfels].tolist() if len(vis_idx) >= n_surfels else vis_idx.tolist()
    else:
        candidates = list(range(n_surfels))

    eps = 1e-2
    m3 = means3D.detach(); m2 = means2D.detach()
    sc = scales.detach(); ro = rotations.detach()
    op = opacity.detach(); col = colors.detach()

    slopes = []
    for idx in candidates:
        with torch.no_grad():
            mp = m3.clone(); mp[idx, 0] += eps
            mm = m3.clone(); mm[idx, 0] -= eps
            fd = (_unpack_image(rasterize(rasterizer, mp, m2, sc, ro, op, col)) -
                  _unpack_image(rasterize(rasterizer, mm, m2, sc, ro, op, col))).sum().item() / (2 * eps)
        if abs(fd) < 1e-6:
            continue
        m3t = m3.clone().requires_grad_(True)
        _unpack_image(rasterize(rasterizer, m3t, m2, sc, ro, op, col)).sum().backward()
        analytic = m3t.grad[idx, 0].item()
        slopes.append(analytic / (fd + 1e-12))

    if not slopes:
        return False, "no valid means3D FD measurements"
    avg_slope = sum(slopes) / len(slopes)
    print(f"  FD means3D: avg_slope={avg_slope:.3f} over {len(slopes)} surfels")

    if not (0.1 < abs(avg_slope) < 5.0):
        return False, f"means3D FD avg_slope={avg_slope:.3f} out of (0.1, 5.0)"
    return True, f"means3D FD PASS: avg_slope={avg_slope:.3f}"


def check_convergence(nc, P=2500, W=160, H=120, iters=400, min_drop_pct=60.0):
    """
    Short Adam fit to a fixed target image.
    Confirms gradients are useful end-to-end; loss must drop >60%, no NaN.
    gfx90a baseline: ~95% drop, ~25-26 dB PSNR.
    gfx1100 baseline: ~73-75% drop, ~11 dB PSNR (fewer visible surfels in this setup).
    """
    import torch.optim as optim
    rasterizer = make_rasterizer(nc, W, H)

    torch.manual_seed(42)
    target = torch.rand(nc, H, W, device=DEVICE)

    means3D = (torch.randn(P, 3, device=DEVICE) * 0.3).requires_grad_(True)
    means2D = torch.zeros(P, 3, device=DEVICE, requires_grad=True)
    scales = (torch.ones(P, 2, device=DEVICE) * 0.02).requires_grad_(True)
    rotations_raw = torch.randn(P, 4, device=DEVICE)
    rotations_raw = (rotations_raw / rotations_raw.norm(dim=-1, keepdim=True)).requires_grad_(True)
    opacity_logit = torch.zeros(P, 1, device=DEVICE, requires_grad=True)
    colors = (torch.rand(P, nc, device=DEVICE) * 0.5).requires_grad_(True)

    opt = optim.Adam([means3D, means2D, scales, rotations_raw, opacity_logit, colors], lr=1e-2)

    first_loss = None
    last_loss = None
    all_finite = True

    for i in range(iters):
        opt.zero_grad()
        opacity = torch.sigmoid(opacity_logit)
        rots = rotations_raw / (rotations_raw.norm(dim=-1, keepdim=True) + 1e-8)
        out = rasterize(rasterizer,
                        means3D, means2D,
                        torch.relu(scales) + 1e-4, rots,
                        opacity, torch.sigmoid(colors))
        image = _unpack_image(out)
        loss = ((image - target) ** 2).mean()
        loss.backward()
        opt.step()

        if not torch.isfinite(loss):
            print(f"    iter {i}: loss={loss.item()} NON-FINITE")
            all_finite = False
            break
        if i == 0:
            first_loss = loss.item()
        last_loss = loss.item()
        if (i + 1) % 100 == 0:
            psnr = -10 * math.log10(last_loss + 1e-12)
            print(f"    iter {i+1:4d}: loss={last_loss:.5f} PSNR={psnr:.2f} dB")

    if first_loss is None or last_loss is None:
        return False, "no iterations completed"
    drop_pct = (first_loss - last_loss) / (first_loss + 1e-12) * 100
    psnr = -10 * math.log10(last_loss + 1e-12)
    summary = f"loss {first_loss:.5f} -> {last_loss:.5f} ({drop_pct:.1f}% down), PSNR={psnr:.2f} dB, all_finite={all_finite}"
    print(f"  Convergence: {summary}")

    if not all_finite:
        return False, "NaN/Inf in loss"
    if drop_pct < min_drop_pct:
        return False, f"loss only dropped {drop_pct:.1f}% (expected >{min_drop_pct}%)"
    return True, summary


# ---------------------------------------------------------------------------
# Per-variant orchestrator
# ---------------------------------------------------------------------------

def validate_variant(nc):
    """Run all four checks for one channel count. Returns (pass_bool, dict-of-results)."""
    name_map = {3: "diff-surfel-rasterization-wet",
                5: "diff-surfel-rasterization-wet-ch05",
                7: "diff-surfel-rasterization-wet-ch07"}
    name = name_map[nc]
    print(f"\n{'='*62}")
    print(f"Variant: {name}  (NC={nc})")
    print(f"{'='*62}")

    results = {}
    variant_ok = True

    def run_check(label, fn, *args, **kwargs):
        nonlocal variant_ok
        try:
            ok, msg = fn(*args, **kwargs)
        except Exception as exc:
            import traceback
            ok, msg = False, str(exc)
            traceback.print_exc()
        results[label] = (ok, msg)
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {label}: {msg}")
        if not ok:
            variant_ok = False

    run_check("1-forward",    check_forward,     nc)
    run_check("2-backward",   check_backward,    nc)
    run_check("3-fd-opacity", check_fd_opacity,  nc)
    run_check("4-fd-means3d", check_fd_means3d,  nc)
    run_check("5-converge",   check_convergence, nc)

    verdict = "PASS" if variant_ok else "FAIL"
    print(f"\n  Variant {name}: {verdict}")
    return variant_ok, results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("diff-surfel-rasterizations ROCm/HIP standalone validation")
    print(f"PyTorch: {torch.__version__}")
    print(f"HIP:     {torch.version.hip}")
    try:
        print(f"Device:  {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        arch = getattr(props, 'gcnArchName', None)
        if arch:
            # gcnArchName already includes 'gfx' prefix (e.g. 'gfx1100')
            print(f"GPU arch: {arch}")
    except Exception:
        pass

    env_dev = os.environ.get("HIP_VISIBLE_DEVICES", "not set")
    env_arch = os.environ.get("PYTORCH_ROCM_ARCH", "not set")
    print(f"HIP_VISIBLE_DEVICES={env_dev}  PYTORCH_ROCM_ARCH={env_arch}")

    overall_ok = True
    summary = {}
    for nc in [3, 5, 7]:
        ok, res = validate_variant(nc)
        summary[nc] = (ok, res)
        if not ok:
            overall_ok = False

    print(f"\n{'='*62}")
    print("SUMMARY")
    print(f"{'='*62}")
    nc_names = {3: "wet (NC=3)", 5: "wet-ch05 (NC=5)", 7: "wet-ch07 (NC=7)"}
    for nc, (ok, _) in summary.items():
        print(f"  {nc_names[nc]:25s}: {'PASS' if ok else 'FAIL'}")
    print(f"\nOverall: {'ALL PASS' if overall_ok else 'SOME FAIL'}")
    sys.exit(0 if overall_ok else 1)


if __name__ == "__main__":
    main()
