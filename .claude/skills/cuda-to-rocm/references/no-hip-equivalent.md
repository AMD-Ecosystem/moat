# When hipify has nothing to translate

Some CUDA code has no HIP analogue to translate INTO. The build succeeds at finding nothing to do, which reads like a small port and is the opposite. This is the most common reason a project ends up dispositioned `cant-port`, and it is worth separating two very different verdicts that get written the same way:

- **No AMD-native target exists.** Genuinely not portable. NVIDIA-proprietary runtimes (TensorRT, OptiX), closed binary kernels shipped as wheels, or a codegen backend emitting PTX with no AMDGPU counterpart.
- **A port would be a REIMPLEMENTATION against an AMD-native target.** Tractable, sometimes valuable, but not a translation -- it needs its own scope, its own estimate, and a person's decision. Recording it as `cant-port` throws away the fact that a target exists.

## CUTLASS -> Composable Kernel

CUTLASS does not port; upstream projects say so themselves (NATTEN's own porting guide states it). The AMD-native equivalent is Composable Kernel (`ROCm/composable_kernel`), including `ck_tile` for tile-level programming. A CUTLASS-based GEMM or attention kernel becomes a CK reimplementation: the algorithm and the tiling strategy carry over, the source does not.

## CuTe -> FlyDSL

`ROCm/FlyDSL` is AMD's Flexible Layout Python DSL, an MLIR-based Python front end for expressing tiling, partitioning, data movement and kernel structure. Its core, FLIR (Flexible Layout Intermediate Representation), is a layout algebra **explicitly inspired by CuTe**, expressing tiling, swizzling and vectorization through composable `(Shape, Stride)` abstractions -- the same design vocabulary. AMD built it in part because so many community and customer workloads depend on CUTLASS and CuTe.

That correspondence is the useful part. Nothing translates CuTe to FlyDSL automatically, but the concepts map: a layout is a layout, a tiled copy is a tiled copy. So a CuTe kernel is not a blank-page rewrite the way a CUTLASS-to-CK move can feel -- there is a structural correspondence to follow, which changes the estimate.

Treat FlyDSL as new: nightly wheels, blogs dated 2026, and a moving surface. Check what it currently supports for the specific kernels in question before promising anything, and prefer it where the source is CuTe-shaped rather than as a general-purpose target.

- Repository: https://github.com/ROCm/FlyDSL
- Porting guide: https://rocm.blogs.amd.com/software-tools-optimization/porting-hip-flydsl/README.html
- Worked example (fused MoE, MI300X): https://rocm.blogs.amd.com/artificial-intelligence/kimi-k2.5-optimize/README.html

## What this means at intake

`cant-port` should mean "no AMD-native target exists", not "hipify will not do it". When the blocker is CUTLASS or CuTe, say which target a reimplementation would aim at and roughly what it would cost, and let a person decide whether that is work worth taking on. A decline that does not name the alternative is hiding a decision rather than making one.

Existing dispositions that rest on this and predate FlyDSL being available: FlashKDA, NATTEN, mirage and spconv are all recorded `cant-port` on CUTLASS/CuTe grounds. They may deserve revisiting on that basis -- which is a person's call, not an agent's.

Raw PTX tensor-core intrinsics are a separate case and stay blocked: `mma.sync`, `ldmatrix.sync`, Hopper `wgmma`/TMA and `tcgen05` have no HIP-level equivalent, and the AMD analogue is MFMA/WMMA reached through rocWMMA, CK or FlyDSL rather than a direct instruction swap.
