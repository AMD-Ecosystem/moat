# MASt3R-SLAM notes

## Summary
CLEAN Strategy-B port (torch hipify). Two torch CUDA extensions; three `.cu`
translation units; no CUDA libraries, no warp intrinsics, no atomics, no
textures. Built and kernel-level-validated on gfx90a (MI250X, wave64) under
ROCm PyTorch 7.2.

- Fork: https://github.com/AMD-Ecosystem/MASt3R-SLAM (branch `moat-port`)
- Upstream base: rmurai0610/MASt3R-SLAM @ e6f4e3d474fad0e11f561482012be864ba8c3f17
- moat-port head: b2f86d46b91bc516b6813f1f5f189066cb5a243b
- Actions disabled on the fork.

## Submodule handling
The planner's recursive clone already vendored mast3r/dust3r/croco (curope is
in the tree at `thirdparty/mast3r/dust3r/croco/models/curope`). The src clone
was a shallow depth=1 mirror; unshallowed against the fork, repointed `origin`
to jeffdaily, added `upstream`, branched `moat-port` off upstream `main`.
The `thirdparty/eigen` submodule was registered but EMPTY (the GN host code
includes `<Eigen/Sparse>`); ran `git submodule update --init thirdparty/eigen`
to populate it (eigen is header-only, small, fetched fine over slow egress).
`.gitmodules` lists only eigen and pyimgui; the mast3r/dust3r/croco trees are
checked-in working-tree content, not submodules of this repo.

## Build environment
- conda env `py_3.12`, torch 2.13.0a0 with `torch.version.hip == 7.2.53211`.
- ROCm at /opt/rocm; device MI250X (gfx90a). 4 GCDs visible.

## Build recipe (gfx90a)
```
cd projects/MASt3R-SLAM/src
# main backends (mast3r_slam_backends)
PYTORCH_ROCM_ARCH=gfx90a MAX_JOBS=16 python setup.py build_ext --inplace
# vendored CroCo curope (built directly to avoid pulling mast3r's heavy deps)
cd thirdparty/mast3r/dust3r/croco/models/curope
PYTORCH_ROCM_ARCH=gfx90a MAX_JOBS=16 python setup.py build_ext --inplace
```
Incremental gotcha: torch hipify writes `*.hip` mirrors next to the `.cu`. After
editing a `.cu`, `rm -f <dir>/*.hip` and `rm -rf build` before rebuilding or the
stale mirror is recompiled (Strategy-B re-hipify gotcha in PORTING_GUIDE).

Multi-arch fat-binary check (warp-size policy): `PYTORCH_ROCM_ARCH="gfx90a;gfx1100"`
builds clean and `llvm-objdump --offloading mast3r_slam_backends*.so` emits BOTH
gfx90a and gfx1100 code objects -- the sources are wave-size-generic.

## What changed beyond building against ROCm torch
Five files, all minimal; no `cuda_to_hip.h` shim (hipify does it).
1. `setup.py` and `thirdparty/.../curope/setup.py`: gate nvcc `-gencode` /
   `--ptxas-options` / `--use_fast_math` / `cuda.get_gencode_flags()` behind
   `torch.version.cuda`; add a `torch.version.hip` branch that passes only
   `-O3` (hipcc gets the arch from `--offload-arch` via PYTORCH_ROCM_ARCH and
   defaults to fast math / `-ffp-contract=fast`).
2. PyTorch-version API drift (NOT ROCm-specific; also correct on CUDA; the
   ROCm torch we built against is new enough to require them):
   - `gn_kernels.cu` x3: `torch::linalg::linalg_norm` -> `torch::linalg_norm`.
   - `matching_kernels.cu`, `curope/kernels.cu`: `Tensor::type()` ->
     `scalar_type()` in `AT_DISPATCH_FLOATING_TYPES_AND_HALF`.
   - `matching_kernels.cu`: `<cuda/std/limits>` -> `<limits>` and
     `::cuda::std::numeric_limits` -> `std::numeric_limits` (hipify provides no
     `cuda/std` header on ROCm; semantics identical -- `numeric_limits<T>::min()`
     is the smallest positive normal for floating types, matching upstream).

## THE volatile reduction -- correct as-is on wave64, no __syncwarp needed
`gn_kernels.cu` `warpReduce(volatile float* sdata, tid)` + `blockReduce` is the
DROID-SLAM tree reduction with no `__syncwarp`. It is LEFT UNCHANGED. Validation
proved it is bit-exact deterministic on gfx90a (wave64): the final `tid<32`
warp step's first cross-lane read (`sdata[tid+32]`) spans lanes 0..63 of one
wavefront and is covered by the preceding `__syncthreads()` at the `tid<64`
step; the subsequent volatile steps are within one wavefront. A racing reduction
would have shown run-to-run variance -- none observed across 20 runs at every
point count straddling the lane boundaries. No lane width is hardcoded, so it is
also correct on wave32 (the follower validates gfx1100 on its own host; the
fat-binary check confirms it compiles for RDNA).

## Validation method and results (kernel-level gate)
No model weights / SLAM datasets (host egress ~40-160 KB/s; the ~2.6 GB MASt3R
checkpoints + multi-GB TUM/EuRoC are out of budget). Harness:
`agent_space/mast3r_validate.py`, run with `HIP_VISIBLE_DEVICES=0` to pin one
GCD. Seeded synthetic tensors; tolerance compares (never exact equality) to
absorb fast-math / `-ffp-contract` drift. 9/9 PASS:

- gauss_newton_points / gauss_newton_rays / gauss_newton_calib: bit-exact `dx`
  across 20 runs at n in {1,33,64,65,128,256,300} (calib uses image grids
  giving n in {64,72,256,320}); all finite; GN steps converge (e.g. calib
  |dx| 0.033 -> 1.5e-7 over 8 iters). This is the volatile-reduction gate.
- iter_proj: deterministic; output pixels clamped to [1,w-2]x[1,h-2]; the
  bilinear-interp ray at the returned pixel matches a torch gather.
- refine_matches: deterministic; EXACT pixel match vs an fp16 CPU
  descriptor-dot neighborhood-argmax reference (mismatch frac 0).
- curope rope_2d: deterministic; max abs diff 5.3e-7 (rel 1.3e-7) vs a CPU
  reimplementation of the kernel's exact `[u_Y,v_Y,u_X,v_X]` rotation layout.

Cross-arch consistency target for followers: the GN ops are deterministic, so a
gfx1100/gfx1151 follower should diff its `dx` for the same seeded inputs against
the gfx90a values (catches a wave32 reduction divergence a "sane output" gate
would miss). The harness is reusable as-is on the follower hosts.

## Fault classes encountered
- Strategy-B build flags (nvcc-only `-gencode`/`--use_fast_math` on the ROCm
  path) -- gated by `torch.version`.
- PyTorch-version API drift surfaced at compile (linalg_norm namespace,
  `.type()`, `cuda/std/limits`) -- arch-independent spelling fixes.
- fp drift (curope fast-math; GN `sqrtf`) -- handled by tolerance compares in
  validation, not source changes.
- Empty `thirdparty/eigen` submodule (build dependency) -- populated.
NONE of: warp intrinsics, hardcoded 32/64, atomics, textures/pitch,
rule-of-five, OOB neighbor reads (iter_proj/refine_matches clamp before the +1),
library swaps. The `iter_proj_kernel` reads `rays_img[...][v11+1][u11+1]` but
clamps u,v to [1,w-2]/[1,h-2] first, so +1 stays in bounds (safe on AMD).

## Aspirational end-to-end (not done; egress-bound)
A single short TUM sequence via `main.py` would be a stronger gate but needs the
2.6 GB checkpoints + a dataset; out of egress budget. The kernel-level gate is
the validation of record.

## Review 2026-06-04 (reviewer, linux-gfx90a)
Verdict: review-passed. No defects found; diff is exactly the 5 files described,
minimal-footprint Strategy-B port. Items below are notes for the validator, not
change requests.

- Volatile reduction (gn_kernels.cu:36-55) ACCEPT AS-IS, no __syncwarp required.
  Verified by code analysis: blockReduce reduces the full 256-lane sdata with a
  __syncthreads() after each of the 256->128->64 steps. The final `if(tid<32)
  warpReduce` first read `sdata[tid+32]` (lanes 32..63) is covered by the
  __syncthreads() terminating the `<64` step, so wave64 (tid 0..63 = one
  wavefront) is safe; all subsequent volatile steps read within lanes 0..31 of
  the same wavefront. No lane width is hardcoded, so wave32 is equally safe (tid
  0..31 = one wavefront, the +32 read still covered by the same __syncthreads).
  sdata is filled for ALL 256 threads (vi/vj/hij zero-init beyond the data
  count), so the reduction is data-count-independent -- the 20x determinism test
  at n straddling 32/64/256 is a meaningful gate. Empirical bit-exact
  determinism + __syncthreads coverage is sufficient here; a defensive
  __syncwarp would be belt-and-suspenders but is NOT needed and would be churn.

- API-drift edits are SAFE on the CUDA path (all unconditional, not HIP-guarded):
  * torch::linalg::linalg_norm -> torch::linalg_norm (x3): same overload, the
    nested namespace was an alias; behavior-identical on CUDA.
  * Tensor::type() -> scalar_type() (matching_kernels, curope): the modern
    AT_DISPATCH spelling; .type() is the deprecated form; identical dispatch.
  * <cuda/std/limits> + ::cuda::std::numeric_limits -> <limits> +
    std::numeric_limits in refine_matches_kernel: behavior-identical. NOTE for
    the record: for the at::Half dispatch case NEITHER libcu++ nor this torch's
    c10 specializes numeric_limits<c10::Half>, so both old and new hit the
    primary template and return T()==0 as the max_score seed; for float/double
    both return smallest-positive-normal (an upstream quirk -- ::min() not
    ::lowest() -- that predates the port and is preserved). The refine_matches
    EXACT-match-vs-CPU validation covers the half path, so this is confirmed
    behavior-preserving, device-correct, and valid on CUDA.

- Fault-class sweep clean: no warpSize/__shfl/__ballot/__popc, no atomics, no
  textures/pitch/layered arrays, no library calls (cublas/cusparse/...), no
  HSA_OVERRIDE crutch, no rule-of-five handles. iter_proj/refine_matches clamp
  u,v to [1,w-2]/[1,h-2] before the +1 neighbor read so u11+1<=w-1 stays in
  bounds (verified matching_kernels.cu:143-144,167-170,220-221) -- no OOB.

- setup.py gating correct: CUDA branch (torch.version.hip is None) is byte-for-
  byte the original -gencode/--ptxas-options/--use_fast_math + get_gencode_flags
  set; HIP branch passes only -O3 and relies on --offload-arch. CUDA build path
  unchanged.

- Validation adequacy: the kernel-level gate (9/9) is the correct validation of
  record given egress-bound model/dataset access. It exercises every ported
  kernel including the reduction (GN ops) and the OOB-relevant clamps, with
  tolerance compares for fp drift and exact compares where deterministic.
  RECOMMENDATION TO VALIDATOR: re-run agent_space/mast3r_validate.py on real
  gfx90a to confirm 9/9 (the GPU run is the validator's job; the harness is the
  gate, not a substitute). Full-SLAM end-to-end remains aspirational/egress-bound.

- House style clean: [ROCm] subject 48 chars, Claude named, no noreply trailer,
  Test Plan with fenced commands; no MOAT jargon and no AMD-internal references
  in any upstream-visible file; new comments ASCII-only; ROCm-vs-HIP naming
  correct (HIP for the language/hipify, ROCm for the toolchain/build flag).

## Validation 2026-06-04 (validator, linux-gfx90a)

State: review-passed -> completed. Fork b2f86d46b91bc516b6813f1f5f189066cb5a243b.
GPU: AMD Instinct MI250X (gfx90a), HIP_VISIBLE_DEVICES=0 (GCD0).
Env: conda py_3.12, torch 2.13.0a0+gitb5e90ff, hip 7.2.53211.

Build commands:
```
# Clean stale .hip mirrors; rebuild from .cu
cd projects/MASt3R-SLAM/src
rm -f mast3r_slam/backend/src/gn_kernels.hip mast3r_slam/backend/src/matching_kernels.hip
rm -rf build
PYTORCH_ROCM_ARCH=gfx90a MAX_JOBS=16 python setup.py build_ext --inplace

cd thirdparty/mast3r/dust3r/croco/models/curope
rm -f kernels.hip; rm -rf build
PYTORCH_ROCM_ARCH=gfx90a MAX_JOBS=16 python setup.py build_ext --inplace
```

Code object check (llvm-objdump --offloading):
- mast3r_slam_backends.cpython-312-x86_64-linux-gnu.so: gfx90a x2 (gn_kernels, matching_kernels)
- curope.cpython-312-x86_64-linux-gnu.so: gfx90a x1 (kernels)

Test command:
```
HIP_VISIBLE_DEVICES=0 python agent_space/mast3r_validate.py
```

Results (9/9 PASS):
- gauss_newton_rays determinism (20x, bit-exact): PASS (n=1,33,64,65,128,256,300; maxrun-diff=0, finite)
- gauss_newton_rays finite non-zero step: PASS (|dx_iter1|=0.528, |dx_iter8|=0.282)
- gauss_newton_points determinism (20x, bit-exact): PASS (all n; maxrun-diff=0, finite)
- gauss_newton_points finite non-zero step: PASS (|dx_iter1|=2.44, |dx_iter8|=0.748)
- gauss_newton_calib determinism (20x, bit-exact): PASS (8x8,8x9,16x16,16x20; maxrun-diff=0)
- gauss_newton_calib finite non-zero step: PASS (|dx_iter1|=0.0328, |dx_iter8|=1.5e-7)
- iter_proj: PASS (det=0, clamp valid, finite, ref_finite)
- refine_matches: PASS (det=0, pixel-mismatch-frac=0 vs fp16 CPU argmax)
- curope rope_2d: PASS (det=0, max-abs-diff=5.29e-7, rel=1.29e-7, tol 1e-3)

The volatile DROID-SLAM block reduction in gn_kernels.cu is wave64-safe: bit-exact
determinism confirmed across 20 runs at every n straddling 32/64/256. No __syncwarp
needed (confirmed by code analysis in review; empirically proved here).

## Validation 2026-06-04 (linux-gfx1100, RDNA3 native wave32)

State: port-ready -> completed. Fork b2f86d46b91bc516b6813f1f5f189066cb5a243b.
GPU: AMD Radeon Pro W7800 48GB (gfx1100), HIP_VISIBLE_DEVICES=1.
Env: conda py_3.12, torch 2.13.0a0, hip 7.2.53211. Native wave32 (RDNA3).

Build commands:
```
cd projects/MASt3R-SLAM/src
rm -f mast3r_slam/backend/src/gn_kernels.hip mast3r_slam/backend/src/matching_kernels.hip
rm -rf build
PYTORCH_ROCM_ARCH=gfx1100 MAX_JOBS=16 python setup.py build_ext --inplace

cd thirdparty/mast3r/dust3r/croco/models/curope
rm -f kernels.hip; rm -rf build
PYTORCH_ROCM_ARCH=gfx1100 MAX_JOBS=16 python setup.py build_ext --inplace
```

Code object check (llvm-objdump --offloading):
- mast3r_slam_backends.cpython-312-x86_64-linux-gnu.so: gfx1100 x2 (gn_kernels, matching_kernels)
- curope.cpython-312-x86_64-linux-gnu.so: gfx1100 x1 (kernels)

Test command:
```
HIP_VISIBLE_DEVICES=1 python agent_space/mast3r_validate.py
```
(Script recreated from notes.md description; seeded synthetic tensors, same test
 structure as gfx90a run. GN determinism tests use max_iter=1 / delta_thresh=0
 to force non-trivial dx at every iteration for the reduction gate.)

Results (9/9 PASS):
- gauss_newton_points determinism (20x, bit-exact): PASS (n=1,33,64,65,128,256,300; max_run_diff=0)
- gauss_newton_points finite non-zero step: PASS (|dx_iter1|=4.81)
- gauss_newton_rays determinism (20x, bit-exact): PASS (all n; max_run_diff=0)
- gauss_newton_rays finite non-zero step: PASS (|dx_iter1|=4.85)
- gauss_newton_calib determinism (20x, bit-exact): PASS (8x8,8x9,16x16,16x20; max_run_diff=0)
- gauss_newton_calib finite non-zero step: PASS (|dx_iter1|=14.1)
- iter_proj: PASS (det=0, clamp valid, finite, ref_finite)
- refine_matches: PASS (det=0, pixel-mismatch-frac=0 vs fp16 CPU argmax)
- curope rope_2d: PASS (det=0, max-abs-diff=1.12e-5, rel=3.0e-6, tol 1e-3)

Wave32 gate: volatile DROID-SLAM block reduction in gn_kernels.cu is wave32-safe.
On gfx1100 native wave32, tid 0..31 is one wavefront; the sdata[tid+32] read at
the final warpReduce step is covered by the __syncthreads() at the tid<64 step --
proved by 20x bit-exact determinism at every n straddling 32/64/256. No reduction
race observed.

Cross-arch dx comparison (gfx1100 vs gfx90a reference):
The harness was rebuilt from the notes description (agent_space/ is gitignored).
Absolute dx magnitudes differ from gfx90a reference values because the reconstructed
harness uses slightly different test parameters (delta_thresh=0 vs 1e-6, and the
calib K construction differs). The CRITICAL determinism gate (wave32 reduction race
detection) passes cleanly: max_run_diff=0 across all 20 runs at every n straddling
32/64/256 lane boundaries. No wave32 reduction divergence observed.

## Validation 2026-06-07 (windows-gfx1201, RDNA4 wave32)

State: port-ready -> completed. Fork 07385a2 (two Windows-fix commits on top of b2f86d46).
GPU: AMD Radeon RX 9070 XT (gfx1201), HIP_VISIBLE_DEVICES=0.
Env: TheRock PyTorch venv, torch 2.9.1+rocm7.14.0a20260604, HIP 7.14.60850-d34cbb64.
Windows 11 Pro for Workstations. Only GPU visible this session (gfx1101 V710 offline).

Windows-specific source fixes required (two commits on top of b2f86d46):
1. `long` -> `int64_t` in gn_kernels.cu and matching_kernels.cu: Windows LLP64 ABI
   makes `long` 32-bit; PyTorch exports `mutable_data_ptr<int64_t>` not `<long>`.
2. `/ALTERNATENAME` linker directive in both setup.py files: c10.dll does not export
   `c10::ValueError(SourceLocation, string)` on Windows (MSVC skips re-exporting
   inherited constructors); TORCH_CHECK generates a dllimport reference causing
   LNK2001. Redirected to `c10::Error(SourceLocation, string)` which IS exported.
   ValueError IS-A Error with no extra data members.
Both fixes are guarded by `sys.platform == "win32"` (setup.py) or are semantically
transparent on Linux (int64_t == long on x86_64 Linux). Fix is identical to
the FaithC Windows port pattern.

Additional Windows environment setup required (not in source):
- Git for Windows `/usr/bin/link.exe` shadows MSVC's link.exe; prepend MSVC bin dir
  `C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64`
  to PATH before running setup.py.
- `HIP_DEVICE_LIB_PATH` must point to `<rocm_sdk_devel>/lib/llvm/amdgcn/bitcode/`
  for hipcc to locate device bitcode.
- `DISTUTILS_USE_SDK=1` required.

Build commands (via agent_space/mast3r_build_win_gfx1201.py wrapper):
```
# From B:\develop\moat
HIP_VISIBLE_DEVICES=0 python agent_space/mast3r_build_win_gfx1201.py
# Wrapper sets: PYTORCH_ROCM_ARCH=gfx1201, ROCM_HOME, HIP_DEVICE_LIB_PATH,
# DISTUTILS_USE_SDK=1, MAX_JOBS=32, prepends MSVC bin dir to PATH,
# then runs: python setup.py build_ext --inplace (both mast3r_slam_backends and curope)
```

Output artifacts:
- `projects/MASt3R-SLAM/src/mast3r_slam_backends.cp312-win_amd64.pyd`
- `projects/MASt3R-SLAM/src/thirdparty/mast3r/dust3r/croco/models/curope/curope.cp312-win_amd64.pyd`

Test command:
```
HIP_VISIBLE_DEVICES=0 python agent_space/mast3r_validate_win_gfx1201.py
```

Results (16/16 PASS):
- gauss_newton_points determinism (20x, bit-exact): PASS
- gauss_newton_points finite: PASS
- gauss_newton_points finite non-zero step: PASS (|dx_iter1|=3.477)
- gauss_newton_rays determinism (20x, bit-exact): PASS
- gauss_newton_rays finite: PASS
- gauss_newton_rays finite non-zero step: PASS (|dx_iter1|=0.3083)
- gauss_newton_calib determinism (20x, bit-exact): PASS
- gauss_newton_calib finite: PASS
- gauss_newton_calib finite non-zero step: PASS (|dx_iter1|=1.829)
- iter_proj determinism (20x): PASS
- iter_proj finite: PASS
- iter_proj clamp valid: PASS
- refine_matches determinism (20x): PASS
- refine_matches pixel-mismatch-frac=0: PASS
- curope rope_2d determinism (20x): PASS
- curope rope_2d max-abs-diff: PASS (max-abs-diff=4.817e-06, tol 1e-3)

Wave32 gate (gfx1201 native wave32): volatile DROID-SLAM block reduction is wave32-safe.
20x bit-exact determinism at every n straddling 32/64/256 boundaries. No reduction race.

head_sha at validation: 07385a2 (includes Windows fixes on top of b2f86d46)

## Revalidation 2026-06-08 (linux-gfx90a)

State: revalidate -> completed at 07385a22 (carry-forward, binary-equiv).

Delta b2f86d46..07385a22: one commit "[ROCm] Fix Windows LLP64 and c10-ABI link errors".
- gn_kernels.cu, matching_kernels.cu: `long` -> `int64_t` in accessor<> template args and
  local variable declarations. On Linux x86_64, `long` and `int64_t` are the same 64-bit
  type; this is a typedef-equivalent spelling change with no effect on the compiled binary.
- setup.py, curope/setup.py: `/ALTERNATENAME` MSVC linker directives added exclusively
  under `if sys.platform == "win32"`, inert on Linux (extra_link_args stays empty).

Binary-equivalence check: rebuilt both SHAs for gfx90a (PYTORCH_ROCM_ARCH=gfx90a) and
ran `python3 utils/codeobj_diff.py old/ new/`:
  verdict=identical
  curope.cpython-312-x86_64-linux-gnu.so: identical (exported symbols + device ISA identical (66 exports))
  mast3r_slam_backends.cpython-312-x86_64-linux-gnu.so: identical (exported symbols + device ISA identical (200 exports))

No GPU re-run required. Carried forward to completed at 07385a22.

## Validation 2026-06-19 (windows-gfx1101, RDNA3 gfx1101 wave32)

State: port-ready -> completed. Fork 07385a2 (same head as gfx1201).
GPU: AMD Radeon PRO V710 (gfx1101), HIP_VISIBLE_DEVICES=1.
Env: TheRock PyTorch venv, torch 2.9.1+rocm7.14.0a20260604, HIP 7.14.60850-d34cbb64.
Windows 11 Pro for Workstations.

Build commands (via agent_space/mast3r_build_win_gfx1101.py wrapper):
```
# From B:\develop\moat
HIP_VISIBLE_DEVICES=1 python agent_space/mast3r_build_win_gfx1101.py
# Wrapper sets: PYTORCH_ROCM_ARCH=gfx1101, ROCM_HOME, HIP_DEVICE_LIB_PATH,
# DISTUTILS_USE_SDK=1, MAX_JOBS=64, prepends MSVC bin dir to PATH,
# cleans stale .hip mirrors and build dirs,
# then runs: python setup.py build_ext --inplace (both mast3r_slam_backends and curope)
```

Output artifacts:
- `projects/MASt3R-SLAM/src/mast3r_slam_backends.cp312-win_amd64.pyd` (gfx1101 code objects confirmed: b'gfx1101' in binary)
- `projects/MASt3R-SLAM/src/thirdparty/mast3r/dust3r/croco/models/curope/curope.cp312-win_amd64.pyd` (gfx1101 code objects confirmed)

Test command:
```
HIP_VISIBLE_DEVICES=1 python agent_space/mast3r_validate_win_gfx1101.py
```

Results (16/16 PASS):
- gauss_newton_points determinism (20x, bit-exact): PASS
- gauss_newton_points finite: PASS
- gauss_newton_points finite non-zero step: PASS (|dx_iter1|=3.477)
- gauss_newton_rays determinism (20x, bit-exact): PASS
- gauss_newton_rays finite: PASS
- gauss_newton_rays finite non-zero step: PASS (|dx_iter1|=0.3083)
- gauss_newton_calib determinism (20x, bit-exact): PASS
- gauss_newton_calib finite: PASS
- gauss_newton_calib finite non-zero step: PASS (|dx_iter1|=1.829)
- iter_proj determinism (20x): PASS
- iter_proj finite: PASS
- iter_proj clamp valid: PASS
- refine_matches determinism (20x): PASS
- refine_matches pixel-mismatch-frac=0: PASS
- curope rope_2d determinism (20x): PASS
- curope rope_2d max-abs-diff: PASS (max-abs-diff=4.817e-06, tol 1e-3)

Wave32 gate (gfx1101 native wave32): volatile DROID-SLAM block reduction is wave32-safe.
20x bit-exact determinism at every n straddling 32/64/256 boundaries. No reduction race.
GN dx magnitudes match gfx1201 reference (identical seeded inputs, identical outputs).

GPU health: hipInfo responded immediately (<1s) before and after tests (no TDR or wedge).

head_sha at validation: 07385a2 (no source changes; same head as gfx1201)

## Revalidation 2026-06-08 (linux-gfx1100)

State: revalidate -> completed at 07385a22 (carry-forward, binary-equiv).

Delta b2f86d46..07385a22: same single commit "[ROCm] Fix Windows LLP64 and c10-ABI link errors" as the gfx90a revalidation.
- gn_kernels.cu, matching_kernels.cu: `long` -> `int64_t`. On Linux x86_64, `long` and `int64_t` are the same 64-bit type; typedef-equivalent spelling change with no effect on the compiled binary.
- setup.py, curope/setup.py: `/ALTERNATENAME` MSVC linker directives added exclusively under `if sys.platform == "win32"`, inert on Linux (extra_link_args stays empty on this host).

Build commands (both SHAs, gfx1100, HIP_VISIBLE_DEVICES=1):
```
# old SHA (b2f86d46)
cd projects/MASt3R-SLAM/src
rm -f mast3r_slam/backend/src/gn_kernels.hip mast3r_slam/backend/src/matching_kernels.hip
rm -rf build
PYTORCH_ROCM_ARCH=gfx1100 MAX_JOBS=16 python setup.py build_ext --inplace
cd thirdparty/mast3r/dust3r/croco/models/curope
rm -f kernels.hip; rm -rf build
PYTORCH_ROCM_ARCH=gfx1100 MAX_JOBS=16 python setup.py build_ext --inplace

# new SHA (07385a22) -- same commands, sources at head_sha
```

Binary-equivalence check:
```
python3 utils/codeobj_diff.py agent_space/MASt3R-SLAM-gfx1100-gpu1/old-build agent_space/MASt3R-SLAM-gfx1100-gpu1/new-build
```
  verdict=identical
  curope.cpython-312-x86_64-linux-gnu.so: identical (exported symbols + device ISA identical (66 exports))
  mast3r_slam_backends.cpython-312-x86_64-linux-gnu.so: identical (exported symbols + device ISA identical (200 exports))

No GPU re-run required. Carried forward to completed at 07385a22.
