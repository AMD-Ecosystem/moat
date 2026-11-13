# MASt3R-SLAM porting plan (linux-gfx90a lead)

## Project
- Name: MASt3R-SLAM
- Upstream: https://github.com/rmurai0610/MASt3R-SLAM (CVPR 2025; real-time dense SLAM on the MASt3R 3D-vision foundation model)
- Default branch: main
- Cloned read-only at projects/MASt3R-SLAM/src (depth=1)

## Disposition
CLEAN STRATEGY-B PORT, LOW-to-MODERATE effort. Two torch CUDAExtension modules, three `.cu` translation units, no CUDA libraries, no CUTLASS/CuTe, no warp intrinsics, no atomics, no textures. The only real porting risk is one DROID-SLAM-style `volatile` tree reduction whose correctness must be VALIDATED on wave64 (gfx90a) and wave32 (RDNA). Set platform state to `planned`.

The dominant uncertainty is VALIDATION FEASIBILITY, not the port: full-SLAM trajectory evaluation needs ~2.6 GB of MASt3R checkpoints plus multi-GB SLAM datasets, impractical on this host's ~40-160 KB/s egress. The plan therefore recommends a KERNEL-LEVEL correctness gate (synthetic inputs, HIP-vs-CPU/torch reference) that fully exercises the ported custom kernels without the model or datasets. Flagged as the key open question for the validator.

## Existing AMD support
- Community fork: https://github.com/EmmanuelMess/MASt3R-SLAM-ROCm -- a GitHub fork of upstream, 1 star, single dev, last push 2025-11-25. README instructs `export HSA_OVERRIDE_GFX_VERSION=10.3.0` (forces consumer gfx1030/RDNA2 identity), no datacenter-GPU validation.
- Judgment: NON-AUTHORITATIVE community hack. No AMD-official effort exists -- checked the ROCm, AMD, GPUOpen, and ROCm-DS orgs (no MASt3R-SLAM repo in any), scanned the full upstream fork network (only `EmmanuelMess/MASt3R-SLAM-ROCm` and `hippoley/-ER-MASt3R-SLAM` carry rocm/hip-ish names; the latter is unrelated), and web-searched "MASt3R-SLAM ROCm/AMD/HIP" (no AMD docs, blog, or release notes). Per PORTING_GUIDE's authoritative-vs-community rule we do NOT adopt the fork as a base and do NOT treat its `.cu` edits as SOTA. Port FROM SCRATCH our way.
- Non-authoritative HINT extracted from the fork's diff (reference only, not inherited):
  - It made NO HIP-specific kernel edits. Its `.cu` changes are pure newer-torch-API compatibility (`torch::linalg::linalg_norm` -> `torch::linalg_norm`, `::cuda::std::numeric_limits` -> `std::numeric_limits` + `<limits>`, `D11.type()` -> `D11.scalar_type()`). These are torch-version drift, not ROCm fixes; whether we need them depends only on the torch version we build against.
  - It added a `torch.version.hip` branch to setup.py that builds the same sources via `CUDAExtension` -- confirming ROCm torch hipifies and builds these unchanged (Strategy B).
  - It did NOT touch the `warpReduce`/`blockReduce` volatile reduction, and it only ever ran on a wave32 GPU via the gfx1030 override; so it provides ZERO evidence about wave64 (gfx90a) correctness of that reduction. Our gfx90a validation is the first real test of it.
  - The `HSA_OVERRIDE_GFX_VERSION` crutch is exactly the forbidden hazard class; we ignore it.

## Build classification: torch-extension (Strategy B)
Evidence:
- `setup.py` lines 4-5: `import torch` + `from torch.utils.cpp_extension import BuildExtension, CppExtension`; lines 24-45: `from torch.utils.cpp_extension import CUDAExtension` and `ext_modules = [CUDAExtension("mast3r_slam_backends", sources=[gn.cpp, gn_kernels.cu, matching_kernels.cu], ...)]`, `cmdclass={"build_ext": BuildExtension}`.
- Second extension: `thirdparty/mast3r/dust3r/croco/models/curope/setup.py` -- its own `CUDAExtension(name='curope', sources=["curope.cpp","kernels.cu"])` + `BuildExtension`, built by `pip install -e thirdparty/mast3r`.
- `pyproject.toml`: `requires = ["setuptools==70.0.0", "torch"]`; no CMake anywhere; no `find_package(Torch)`.
On a ROCm torch, `BuildExtension` runs `torch.utils.hipify` on the `.cu`/`.cuh` sources and links amdhip64/c10_hip/torch_hip automatically. Keep sources in CUDA spelling; fix only what hipify cannot, guarded by `USE_ROCM`.

## Port strategy: B (torch hipify), rationale
Both modules are torch CUDAExtensions, so hipify is the mechanism. No `cuda_to_hip.h` shim, no manual symbol renaming. The work is:
1. Build against a ROCm torch (`torch.version.hip` set). Add a HIP branch in the two setup.py files mirroring the CUDA branch (the upstream `setup.py` gates the kernel sources behind `has_cuda = torch.cuda.is_available()` which is True on ROCm, but the nvcc-only `-gencode` args are CUDA-specific and must not be passed on ROCm). Drop the `-gencode` flags on the HIP path; let BuildExtension supply `--offload-arch`. This is the minimal-footprint equivalent of the fork's `torch.version.hip` branch, written our way.
2. Verify hipify output compiles; fix any residual fault-class issue (see risks). Expect very few -- there are no libraries, intrinsics, atomics, or textures to swap.
3. The DROID-SLAM `volatile` reduction: leave as-is first and validate; if it miscomputes, the correct minimal fix is an explicit `__syncwarp()` between the volatile steps in `warpReduce` (and the final `tid<32` entry) so the port is correct on both wave widths without hardcoding a width.

No CUTLASS/CuTe present, so no CK rewrite is warranted. A mechanical HIP port is the right and sufficient first step; these are small custom kernels (matrix-free Gauss-Newton assembly + a feature-match search), not GEMM/attention, so there is no large AMD-native-rewrite upside to chase.

## CUDA surface inventory
Three translation units (all torch CUDAExtension):
- `mast3r_slam/backend/src/gn_kernels.cu` (1637 lines) -- the Gauss-Newton core. 4 `__global__` kernels (`point_align_kernel`, `ray_align_kernel`, a calib variant, `pose_retr_kernel`) + device helpers. Uses `__shared__` arrays, `__syncthreads()`, and a `warpReduce`/`blockReduce` tree reduction over `__shared__ float sdata[THREADS]` with `THREADS=256`. Host side builds the reduced per-edge H/g blocks into an `Eigen::SparseMatrix<double>` and solves with `Eigen::SparseCholesky` on CPU (Eigen, not cuSPARSE/cuSOLVER). `torch::linalg::linalg_norm` for the delta-norm stop test. `sqrtf` in residual weighting (fp-drift relevant).
- `mast3r_slam/backend/src/matching_kernels.cu` (315 lines) -- 2 `__global__` kernels (`refine_matches_kernel`, `iter_proj_kernel`). Pure per-thread elementwise: bilinear ray interpolation + a 2x2 LM solve per pixel + a local descriptor-dot-product neighborhood search. Uses `torch::PackedTensorAccessor32`, `cuda_fp16.h`, `cuda/std/limits`. No shared mem, no intrinsics, no reductions, no atomics.
- `thirdparty/mast3r/dust3r/croco/models/curope/kernels.cu` (109 lines, vendored CroCo/DUSt3R) -- 1 `__global__` `rope_2d_cuda_kernel` (2D rotary position embedding). `extern __shared__`, proper `__syncthreads()`, `powf/cosf/sinf`. Built by `pip install -e thirdparty/mast3r`. Compiled with `--use_fast_math` on CUDA (fp-drift relevant; on HIP clang defaults to `-ffp-contract=fast` similarly).
Negative findings (verified by grep across all three .cu): NO `__shfl*`, `__ballot`, `__activemask`, `warpSize`, `__popc`, cooperative groups, `tiled_partition`; NO atomics; NO cuBLAS/cuSPARSE/cuSOLVER/cuFFT/cuRAND/Thrust/CUB; NO CUTLASS/CuTe; NO textures/surfaces/`__ldg`; NO pinned/managed memory; NO explicit streams/events (default stream only). The only library is Eigen, used HOST-side.

ROCm/HIP mapping: every runtime/symbol used (`cudaGetLastError`, `cudaSuccess`, `<cuda_fp16.h>`, `PackedTensorAccessor32`, `AT_DISPATCH_*`) is handled automatically by torch hipify. `cuda/std/limits` -> hipify maps to the equivalent or fall back to `<limits>` + `std::numeric_limits` (the fork's hint). Eigen is CPU and untouched.

## Risk list
1. WAVE-SIZE / volatile reduction (PRIMARY, validate-first). `warpReduce(volatile float* sdata, tid)` in gn_kernels.cu does `sdata[tid]+=sdata[tid+32]; +16; +8; +4; +2; +1;` with no `__syncwarp`, entered under `if(tid<32)` after a `__syncthreads()`-terminated `tid<64` step. Analysis: on wave64 (gfx90a) tid 0..31 lie in one 64-lane wavefront (lockstep) and the `+32` read of sdata[32..63] is covered by the preceding `__syncthreads()`; on wave32 (gfx1100/gfx1151) tid 0..31 ARE one wavefront and the cross-wavefront `+32` read is likewise covered by that `__syncthreads()`. So it is wave-agnostic-CORRECT as written -- BUT it leans on `volatile` + implicit intra-wavefront lockstep that the ROCm compiler is not contractually required to preserve. Treat as the thing to PROVE on real GPU; if it drifts, the minimal wave-generic fix is an explicit `__syncwarp()` between the volatile steps. Do NOT introduce any hardcoded-64 or hardcoded-32 geometry. This is the only kernel that can diverge by arch, so it is the cross-arch consistency target for the followers.
2. fp drift / fast-math. curope builds `--use_fast_math` on CUDA; HIP clang defaults to `-ffp-contract=fast`. The GN residual weighting uses `sqrtf`. SLAM is iterative and tolerant, so ~1 ULP drift is benign for a trajectory gate, but a bit-exact kernel-level compare must use a tolerance (atol/rtol), not exact equality. (PORTING_GUIDE `__fsqrt_rn` / `-ffp-contract` classes.) No exact-equality-on-approx-division control flow found (the TIGRE hang class does not apply).
3. setup.py CUDA-only flags on the ROCm path. The `-gencode=arch=compute_*` nvcc args (setup.py lines 30-37; curope `get_gencode_flags()`) are CUDA-specific and break/clutter the ROCm build. Gate them behind `torch.version.cuda` and add a `torch.version.hip` branch with no gencode (BuildExtension injects `--offload-arch`). Mirror in both setup.py files.
4. torch-version API drift (NOT a ROCm issue, but will surface at build). `torch::linalg::linalg_norm`, `::cuda::std::numeric_limits`, `D11.type()` are spellings that changed across torch releases (the fork rewrote them). Whether we touch them depends solely on the ROCm torch version we build against; fix only if the compile fails, and gate nothing on arch.
5. OOB neighbor reads. `iter_proj_kernel` reads `rays_img[b][v11+1][u11+1]` but clamps `u in [1,w-2]`, `v in [1,h-2]` first, so the +1 stays in bounds -- SAFE (no fix). `refine_matches_kernel` guards every gather with `inside_image()` -- SAFE. Noted because AMD faults where CUDA tolerates a one-past read; re-confirm if either clamp is ever loosened.
6. No rule-of-five / texture / pitch / layered-array / smid risks apply (none of those constructs are present).

## File-by-file change list
- `setup.py` -- add a `torch.version.hip` build branch (sources unchanged) that omits the nvcc `-gencode` args; keep the CUDA branch behind `torch.version.cuda`. Minimal, mirrors upstream structure.
- `thirdparty/mast3r/dust3r/croco/models/curope/setup.py` -- same: a HIP branch dropping `get_gencode_flags()`/`--use_fast_math`-only nvcc flags (keep `-O3`); let BuildExtension hipify + supply arch.
- `mast3r_slam/backend/src/gn_kernels.cu` -- expected NO change; ONLY if validation shows the volatile reduction drifts, add `__syncwarp()` between the `warpReduce` volatile steps (wave-generic). Possibly the torch-API spelling fixes (risk 4) if the build torch requires them, guarded by nothing arch-specific.
- `mast3r_slam/backend/src/matching_kernels.cu` -- expected NO change; possibly `<cuda/std/limits>` -> `<limits>`/`std::numeric_limits` and `.type()`->`.scalar_type()` only if the build torch requires (torch-version, not ROCm).
- `thirdparty/mast3r/dust3r/croco/models/curope/kernels.cu` -- expected NO change.
The intent is the smallest possible diff: two setup.py HIP branches; kernel `.cu` edits only if forced by the compiler or by a proven GPU miscompute.

## Build commands (gfx90a)
Prereq: a ROCm PyTorch (`python -c "import torch; print(torch.version.hip)"` must be non-None). Use the existing ROCm 7.2.x torch env on this host (check notes/local memory for the env name; do not reinstall over slow egress).
```
# (one-time) python deps without the heavy model/datasets:
cd projects/MASt3R-SLAM/src
pip install --no-build-isolation -e thirdparty/mast3r   # builds curope (CUDAExtension -> hipify)
pip install --no-build-isolation -e thirdparty/in3d
pip install --no-build-isolation -e .                   # builds mast3r_slam_backends
```
BuildExtension targets the local GPU arch automatically; to force gfx90a set `PYTORCH_ROCM_ARCH=gfx90a` (and `gfx90a;gfx1100` for a fat-binary multi-arch correctness check per the warp-size policy). Wrap each build phase in utils/timeit.sh.

## Test plan
Upstream ships NO unit tests -- only full-SLAM eval scripts (`scripts/eval_{tum,7_scenes,euroc,eth3d}.sh`) that require ~2.6 GB MASt3R checkpoints + multi-GB datasets (TUM fr1 alone is ~9 .tgz sequences). On ~40-160 KB/s egress that is many hours to days; treat full-SLAM as ASPIRATIONAL, not the gate.

RECOMMENDED GATE -- kernel-level correctness (runnable here, no model/dataset):
- The five pybind entry points are directly callable on synthetic tensors: `mast3r_slam_backends.{gauss_newton_points, gauss_newton_rays, gauss_newton_calib, iter_proj, refine_matches}` and `curope`'s rope_2d. Write a small pytest in agent_space that:
  - feeds deterministic random tensors (fixed seed) of representative shapes to each backend op on `cuda` (ROCm),
  - compares against a CPU/torch reference where one exists cheaply: `iter_proj` (bilinear interp + 2x2 LM solve) and `refine_matches` (descriptor-dot neighborhood argmax) are straightforward to reproduce in pure torch on CPU; the GN ops can be checked by (a) the documented H symmetry / finite delta-norm decrease and (b) a CPU recomputation of the per-edge H/g assembly, and rope_2d against a pure-torch RoPE.
  - asserts allclose with a tolerance (rtol/atol ~1e-3 fp32) to absorb fast-math/`-ffp-contract` drift (risk 2); never exact equality.
- This exercises every ported kernel including the volatile reduction (the GN ops) and is the cross-arch consistency target: the followers diff their op outputs against the gfx90a outputs for the same seeded inputs (catches a wave32 reduction divergence the loose "sane output" gate would miss).
- Multi-arch build check: build `PYTORCH_ROCM_ARCH="gfx90a;gfx1100"` and confirm both code objects are emitted (`llvm-objdump --offloading mast3r_slam_backends*.so`).
- Non-GPU regression: there is no non-GPU test suite to regress; the python import of `mast3r_slam` and a CPU-only construction path are the only smoke checks.

If the validator CAN obtain at least the checkpoints + ONE short TUM sequence within egress budget, a single `python main.py --dataset datasets/tum/rgbd_dataset_freiburg1_xyz/ --config config/calib.yaml` headless run producing a finite ATE is the stronger end-to-end confirmation -- but it is a bonus on top of the kernel gate, not a prerequisite.

## Open questions
1. EGRESS / data feasibility (KEY, for the validator): can the host fetch the ~2.6 GB MASt3R checkpoints and any one SLAM sequence at ~40-160 KB/s in acceptable time? If not, the kernel-level gate above is the validation of record; document the egress constraint and the substitution explicitly. Confirm whether a ROCm torch env already exists on the host (avoid re-downloading torch).
2. Which ROCm torch version is installed -- determines whether the torch-API spelling fixes (risk 4) are needed at all.
3. The volatile-reduction correctness on real gfx90a: does the ROCm compiler preserve the implicit-lockstep `volatile` semantics, or is the `__syncwarp()` fix required? Pure validate-then-fix; cannot be decided statically.
4. curope: built transitively via `thirdparty/mast3r`; confirm the runtime actually imports the compiled `curope` (vs a pure-torch fallback) so the kernel is in the validated surface.
