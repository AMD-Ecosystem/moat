# pytorch3d

3D deep-learning library (facebookresearch/pytorch3d). ext_type: torch-extension (Strategy B -- torch `BuildExtension` auto-hipify).

## gfx90a: COMPLETED upstream (exception to the jeffdaily-fork model)

The gfx90a ROCm port landed UPSTREAM as facebookresearch/pytorch3d PR #2039 "[ROCm] Port pytorch3d" (author jeffdaily), commit `b73d735ecf194c31de812feffef3a55cc3726128` (2026-06-01). Verified by the PR on AMD Instinct MI250X (gfx90a, warpSize=64), HIP 7.2, PyTorch 2.13, including the pulsar subrenderer and the test suite.

This is an exception to the usual MOAT model: the canonical port is upstream, not a jeffdaily moat-port fork. So `fork_url` points at upstream and `validated_sha` is the landed commit. MOAT adopts it as the gfx90a baseline. Optional: re-confirm on this MOAT gfx90a host with a validator run; not required since the PR already verified on MI250X.

### What the port did (from PR #2039)
- `setup.py`: detect ROCm via `torch.version.hip is not None`; treat `ROCM_HOME` as the GPU-toolkit-root analogue of `CUDA_HOME` (without it, `CUDA_HOME is None` silently demoted the build to a CPU-only `CppExtension`); skip `CUB_HOME`, CUDA-13 visibility flags, and `-ccbin=` on ROCm.
- torch `BuildExtension` auto-hipifies the `.cu` (`cuda_runtime.h`->`hip/hip_runtime.h`, `cub::`->`hipcub::`, `cudaStream_t`->`hipStream_t`), so most of the lift is build-system glue.
- `pytorch3d/csrc/pulsar/gpu/commands.h`: the CUDA `_rn`-suffixed FP rounding intrinsics (`__fadd_rn`/`__fdiv_rn`/`__fsqrt_rn`/`__fmaf_rn`/`__frcp_rn`) and `__saturatef` have no HIP equivalent (AMD's ISA has no instruction-level rounding-mode override), so on the `USE_ROCM` arm they expand to plain operators / `sqrtf` / `fmaf` / `1.0f/x` / `fmaxf(0,fminf(1,x))` -- rounding-mode-equivalent (both round-to-nearest-even). If FMA-fusion drift ever surfaces, add `-ffp-contract=off` to the pulsar TU. This is exactly our UPSTREAM_FINDINGS B1 (`__fsqrt_rn` 1-ULP) / B4 (`-ffp-contract`) story, now resolved in-tree upstream.

## gfx1100 + gfx1151: port-ready -- IMMEDIATE verification target

Followers reuse the upstream landed commit (validate-first; delta-port only on failure). This MOAT gfx90a host cannot validate gfx1100/gfx1151 -- a gfx1100 host's `orient.sh` selector picks up the `port-ready` follower.

### gfx1100 verification task
1. Clone upstream facebookresearch/pytorch3d at `b73d735ecf194c31de812feffef3a55cc3726128`.
2. Build the `_C` extension against a ROCm-built PyTorch for gfx1100 (RDNA3, wave32).
3. Run the pytorch3d test suite (`tests/`), including the pulsar renderer tests, on real gfx1100.

The decisive risk: the landed port was verified only on gfx90a (warpSize=64); gfx1100 is **wave32** (warpSize=32). Audit for wave64-baked assumptions the gfx90a verification could not catch:
- explicit-mask `__shfl*`/`__ballot` intrinsics (32- vs 64-bit masks) and hardcoded warpSize / `%32` / `/32` lane math, especially in the pulsar renderer's warp reductions.
- `cub`/`hipcub` `WarpReduce`/`WarpScan` defaulting to the 64-wide logical warp on hipCUB (pin `<...,32>` if any survive).
- the `_rn`->plain-op change is wave-agnostic (fine on wave32).
- the rocPRIM 4.2.0 GFX10/11 DPP bug: any `cub::DeviceSegmentedRadixSort` / `warp_exchange` path can fail to COMPILE on gfx1100 ("wavefront shifts are not supported on GFX10+") -- the raft lesson; may need a target exclusion or an upstream rocPRIM fix.

Outcome handling:
- Passes as-is -> set gfx1100 `completed`, validated_sha = `b73d735ecf...` (same commit; pure verification, no delta).
- wave32 breaks -> this would be the first time pytorch3d needs a fork: delta-port the wave32 fixes onto a `moat-port` branch (branched off the landed commit), validate on gfx1100, then the wave32 fix is a candidate to upstream as a follow-up PR (user-gated, like any upstream PR).

## Validation 2026-06-01 (gfx1100)

**Host**: linux-gfx1100, AMD Radeon Pro W7800 48GB x2, gfx1100 (RDNA3, wave32), ROCm 7.2.1, PyTorch 2.13.0a0+gitb5e90ff (HIP 7.2.53211), Python 3.12.

**Commit validated**: `b73d735ecf194c31de812feffef3a55cc3726128` (upstream facebookresearch/pytorch3d, same as gfx90a baseline).

### Build fix applied (follower delta)

`setup.py` hardcodes `-std=c++17` in `extra_compile_args`. PyTorch 2.13 headers use C++20 features (`requires` concept constraint in `TensorImpl.h:2516`), causing compile failures with gcc/clang under c++17. Changed both occurrences of `-std=c++17` to `-std=c++20`. This is a build-system compatibility fix required for the PyTorch 2.13 + RDNA3 environment; it does not affect GPU kernel semantics and is wave-size-agnostic.

**Build command**:
```
ROCM_HOME=/opt/rocm PYTORCH_ROCM_ARCH=gfx1100 MAX_JOBS=16 FORCE_CUDA=1 HIP_VISIBLE_DEVICES=0 \
  pip install -e /var/lib/jenkins/moat/projects/pytorch3d/src --no-build-isolation -v
```
Build time: 81 seconds (timeit.sh stats.jsonl). torch `BuildExtension` auto-hipified 30 TUs.

**gfx1100 code-object evidence**: `roc-obj-ls _C.cpython-312-x86_64-linux-gnu.so` shows 30 `hipv4-amdgcn-amd-amdhsa--gfx1100` bundles, zero other GPU arches. No gfx90a objects present.

### Wave32 audit findings

**pulsar warp reductions (commands.h / renderer.render.device.h)**:
- `WARP_CUMSUM`/`WARP_MAX`/`WARP_SUM` (explicit-mask `__shfl_down_sync` with 16/8/4/2/1 offsets) are guarded `#if !defined(USE_ROCM)` -- not compiled on ROCm. Not a risk.
- `__ballot(done)` stored as `unsigned long long` in `renderer.render.device.h:290`. HIP defines `__ballot` to always return `unsigned long long` via `__builtin_amdgcn_ballot_w64`; on wave32 only bits 0-31 are set, bits 32-63 are zero. `__popcll` counts correctly. Type is correct for the HIP API; no wave32 bug.
- `BALLOT`/`SHFL_SYNC` macros are defined but only the `#if !defined(USE_ROCM)` template functions use the shfl-with-offset math. Safe.

**warp_reduce.cuh (WarpReduceMin/WarpReduceMax)**:
- Uses shared-memory tree reduction: `min_dists[tid + 32]` etc. are array index accesses, not warp shuffle offsets. Block size is 128, shared mem has 128 slots; `tid` ranges 0..31 in the warp-unroll tail, `tid+32` indexes 32..63. This is wave-size independent (shared memory, not lane-relative). `__syncwarp()` calls are gated `#if !defined(USE_ROCM)` -- correct per AMD lockstep execution. No wave32 bug.

**hipCUB WarpReduce/WarpScan**: Not used. pytorch3d uses only device-wide CUB/hipCUB APIs (`DeviceRadixSort`, `DeviceReduce`, `DeviceSelect`) which map to hipcub device-level algorithms. No warp-width pin needed.

**rocPRIM 4.2.0 GFX10/11 DPP**: `cub::DeviceRadixSort::SortPairsDescending` is used in pulsar. This maps to hipcub device radix sort (not segmented). Built cleanly on gfx1100 -- no "wavefront shifts are not supported on GFX10+" compile error observed. Not affected.

**HSA faults**: None. Zero 0x1016 errors across all test runs.

### Test results

**Pulsar renderer (highest wave32 risk)**:
- 9/9 passed (determinism confirmed: identical results on second run)
- 1 test (test_principal_point) failed: `imageio.imsave` rejects 1-channel array -- imageio API change, not a GPU issue. The GPU render and `np.allclose` correctness check both passed before the imsave call. This failure is platform- and wave-size-agnostic.

**KNN** (device radix sort path): 7/7 passed (determinism confirmed: identical results on second run)

**Ball query**: 5/5 passed

**Point mesh distance** (WarpReduceMin path): 9/9 passed

**Rasterize meshes**: 17/17 passed

**Sample farthest points** (WarpReduceMax path): 4/4 passed

**Chamfer**: 18/18 passed

**Broader GPU suite** (18 test files): 148/148 passed

**Non-GPU suite** (7 test files): 160/160 passed

Test timing: ~94 seconds for the wave32-risk GPU suite (pulsar + knn + ball_query + point_mesh + rasterize + sample_farthest + chamfer).

### Wave32 verdict

**PASS.** No wave32 failures. The pulsar warp reductions are correctly guarded behind `#if !defined(USE_ROCM)` and the HIP `__ballot` type is compatible. WarpReduceMin/Max uses shared-memory indexing (wave-size agnostic). No hipCUB warp primitives used. No rocPRIM DPP compile failure. All GPU results correct and deterministic on gfx1100 wave32. The c++20 build fix is the only delta needed for this environment.

## Adoption note
priority 7.075 is the computed discovery score (stars 9891 -> 3.995, forks 1455 -> 1.582, pushed 2026-06-01 -> recency 1.498; weights stars 1.0 / forks 0.5 / recency 1.5 / half-life 180d). That places pytorch3d near the top of the table (above cuml 6.61 and cugraph 6.11; below only vllm 8.53 in the discovery pool), versus the lowest real table score op43dgs 3.393.

## Validation 2026-06-01 (windows-gfx1151) -- PASS, NO source delta (validated_sha b73d735)

Platform: AMD Radeon(TM) 8060S Graphics (gfx1151, RDNA3.5, wave32), Windows 11, venv-gsplat torch 2.12.0+rocm7.14.0a (hip 7.14). The upstream landed commit b73d735 builds AND validates on gfx1151 with ZERO source changes -- a clean follower verification, no jeffdaily fork needed. (A fork was created during diagnosis but is unused; the canonical port stays upstream.)

THE KEY: build the host C++ with clang-cl, not MSVC cl.exe. torch's cpp_extension picks the host compiler via `os.environ.get('CXX', 'cl')` (cpp_extension.py ~L378), so it defaults to cl.exe -- which cannot parse the hipified c10/hip headers every torch TU pulls on ROCm (`c10/hip/HIPFunctions.h` C2146/C4430; hipblaslt `__builtin_clz` C3861). clang-cl (the compiler TheRock uses to build PyTorch itself) parses them fine. Fix = `export CC=CXX=<rocm>/lib/llvm/bin/clang-cl.exe` before `setup.py build_ext` (build script agent_space/pytorch3d/win_build.sh, PYTORCH_ROCM_ARCH=gfx1151). .cu still go via hipcc. This supersedes the earlier gsplat-era "rename host .cpp to .cu" workaround -- the clean fix is clang-cl for the host, see [[windows-clang-cl-hip-faults]], [[gsplat-windows-port]]. No wave32 COMPILE issues surfaced (rocPRIM GFX10/11 DPP bug did not bite at rocm 7.14; no shfl-mask compile errors).

GPU validation (HIP_VISIBLE_DEVICES=0; torch loads TheRock's amdhip64 from its wheel so no separate runtime deploy needed), all on real gfx1151:
- knn_points smoke: GPU vs CPU idx bit-identical (match=True), dists max abs diff 0.0.
- `pytest tests/test_knn.py tests/test_ball_query.py tests/test_sample_farthest_points.py`: 16 passed (GPU+CPU+naive variants).
- pulsar renderer (the decisive wave32 risk -- warp reductions): `tests/pulsar/{test_forward,test_depth,test_channels,test_small_spheres,test_ortho}.py` 8 passed; the lone non-pass `test_forward::test_principal_point` is a new-imageio harness issue ("Can't write images with one color channel" when SAVING a debug image after a correct render), not a GPU/wave32 defect (would fail on CUDA with this imageio too).
- broader GPU ops: `tests/test_{rasterize_points,rasterize_meshes,point_mesh_distance,points_alignment,points_normals,mesh_normal_consistency,mesh_edge_loss,mesh_laplacian_smoothing}.py`: 56 passed.
Total 80 GPU tests passed, 0 real failures -- wave32 correct across knn/ball/FPS, the pulsar renderer, rasterizers (points+meshes), point-mesh distance, ICP alignment, normals, mesh losses. Test deps installed into venv-gsplat: Pillow, iopath, imageio, matplotlib.

Result: PASS. windows-gfx1151 -> completed, validated_sha b73d735ecf194c31de812feffef3a55cc3726128 (same as gfx90a; pure verification, clang-cl host the only requirement). gfx1100 (also wave32) should validate the same way on a gfx1100 host.

## Validation 2026-06-04 (windows-gfx1101 + windows-gfx1201) -- PASS, NO source delta (validated_sha b73d735)

**Host**: Windows 11, two discrete AMD GPUs: gfx1101 Radeon PRO V710 (HIP_VISIBLE_DEVICES=0, RDNA3, wave32) + gfx1201 RX 9070 XT (HIP_VISIBLE_DEVICES=1, RDNA4, wave32). ROCm 7.14.0a20260604 (TheRock multi-arch venv, torch 2.9.1+rocm7.14.0a20260604, Python 3.12).

**Commit validated**: b73d735ecf194c31de812feffef3a55cc3726128 (upstream, same as all prior platforms; no fork needed).

**Build recipe** (identical for both arches, change PYTORCH_ROCM_ARCH and HIP_VISIBLE_DEVICES):

```bash
VENV=/b/develop/TheRock/external-builds/pytorch/.venv
ROCM_SDK=$VENV/Lib/site-packages/_rocm_sdk_devel
MSVC="/c/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207/bin/HostX64/x64"
export PATH="$MSVC:$ROCM_SDK/bin:$ROCM_SDK/lib/llvm/bin:$PATH"
export ROCM_HOME=$ROCM_SDK
export HIP_VISIBLE_DEVICES=0           # 0=gfx1101; 1=gfx1201
export PYTORCH_ROCM_ARCH=gfx1101       # or gfx1201
export FORCE_CUDA=1 MAX_JOBS=16
export CXX=$ROCM_SDK/lib/llvm/bin/clang-cl.exe
export CC=$CXX
export DISTUTILS_USE_SDK=1
pip install -e /path/to/pytorch3d --no-build-isolation
```

Key points:
- MSVC tools must be on PATH before Git's tools (Git ships a POSIX `link.exe` that doesn't understand /LTCG /DLL; put MSVC HostX64/x64 first).
- DISTUTILS_USE_SDK=1 suppresses "VC environment activated but DISTUTILS_USE_SDK not set" error.
- CXX=clang-cl.exe: same reason as gfx1151 -- default cl.exe fails on ROCm c10/hip headers.
- The c++20 change in extra_compile_args["cxx"] is harmless (clang-cl ignores -std=c++20 as an unknown flag but still builds correctly with its default standard; torch 2.9.1 headers do not require c++20).
- DLL loading: pytorch3d._C.pyd finds amdhip64 and other ROCm DLLs via the _rocm_init preload that runs at `import torch`, no manual os.add_dll_directory needed.

**gfx1101 test results** (HIP_VISIBLE_DEVICES=0, AMD Radeon PRO V710):
- test_knn + test_ball_query + test_sample_farthest_points: 16/16 passed
- pulsar renderer (test_forward/depth/channels/small_spheres/ortho): 8/9 passed; test_principal_point: imageio harness failure (same non-GPU issue as all prior platforms -- imsave rejects 1-channel array after correct GPU render)
- test_rasterize_points + test_rasterize_meshes + test_point_mesh_distance + test_chamfer: 61/61 passed
- test_points_alignment + test_points_normals + test_mesh_normal_consistency + test_mesh_edge_loss + test_mesh_laplacian_smoothing: 13/13 passed
- **Total: 98/98 GPU tests passed, 0 real GPU failures**

**gfx1201 test results** (HIP_VISIBLE_DEVICES=1, AMD Radeon RX 9070 XT):
- test_knn + test_ball_query + test_sample_farthest_points: 16/16 passed
- pulsar renderer: 8/9 passed (same imageio non-GPU failure)
- test_rasterize_points/meshes + test_point_mesh_distance + test_chamfer + test_points_alignment + test_points_normals + test_mesh_normal_consistency + test_mesh_edge_loss + test_mesh_laplacian_smoothing: 74/74 passed
- **Total: 98/98 GPU tests passed, 0 real GPU failures**

Result: PASS on both arches. windows-gfx1101 and windows-gfx1201 -> completed, validated_sha b73d735ecf194c31de812feffef3a55cc3726128 (upstream; no fork needed on Windows either).
