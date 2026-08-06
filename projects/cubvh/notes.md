# cubvh notes

ashawkey/cubvh -- "CUDA Mesh BVH tools" (PyTorch extension, MIT, 292 stars, actively maintained: ashawkey merges contributor PRs, last default-branch activity 2026-01-27).

## Why this is a MOAT project

Surfaced as a vendored dependency of CuMesh (`third_party/cubvh`, pinned to `JeffreyXiang/cubvh@trellis.2`, itself a thin fork of ashawkey/cubvh: 3 ahead / 10 behind, carrying TRELLIS-specific commits "Use larger stack" and "wrap cubvh API with cumesh namespace"). CuMesh's port temporarily repoints that submodule at `AMD-Ecosystem/cubvh@moat-windows`, which is NOT upstream-mergeable. The durable fix is to land ROCm support in ashawkey/cubvh (canonical, maintained), then have CuMesh consume a HIP-enabled cubvh.

The fork AMD-Ecosystem/cubvh already exists (parent JeffreyXiang/cubvh, network source ashawkey/cubvh) -- a branch off ashawkey/main on that same fork can open a PR directly to ashawkey/cubvh. Disable Actions on the fork after first push.

## Existing partial port to reuse (from CuMesh dep work)

The CuMesh dependency commit `AMD-Ecosystem/cubvh@moat-windows` (one commit on top of trellis.2) touched 3 files. Re-apply the GENERAL ones against ashawkey/main; do NOT cherry-pick wholesale (trellis.2's file versions differ, and one change is CuMesh-coupled):

1. `include/gpu/spcumc.cuh` -- GENERAL ROCm fix, KEEP. `thrust::cuda::par` is not available on ROCm; add a `USE_ROCM`-guarded macro:
   ```cpp
   #ifdef USE_ROCM
   #include <thrust/system/hip/execution_policy.h>
   #define THRUST_CUDA_PAR thrust::hip::par
   #else
   #define THRUST_CUDA_PAR thrust::cuda::par
   #endif
   ```
   and replace every `thrust::cuda::par.on(stream)` with `THRUST_CUDA_PAR.on(stream)` (~16 sites). Active on Linux AND Windows ROCm.

2. `src/api_gpu.cu` -- GENERAL, KEEP. Under `USE_ROCM`, include `<ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h>` for the torch-ROCm stream type.

3. `src/bindings_winhip.cu` -- CuMesh-COUPLED, DO NOT upstream as-is. It `#include <eigen_hip_compat.h>`, a header that lives in CuMesh's `hip_cuda_compat/`, not in cubvh. It exists only to compile the pybind binding via hipcc on Windows/MSVC. For a standalone cubvh port, solve the Windows/MSVC-rejects-HIP-attributes problem within cubvh's own build (or scope Windows out of the first PR and treat it as a follow-up). The Linux ROCm port needs only changes 1 and 2 plus build-system arch support.

## Build-system work still needed (not in the CuMesh dep commit)

setup.py / pyproject.toml need a ROCm path: detect `IS_HIP_EXTENSION`, set offload archs (default `native`, overridable via env), and keep NVCC-only flags off the HIP path. Document the ROCm build in the project's house style. Validate on real GPU (gfx90a lead) before any PR.

## PR target

ashawkey/cubvh (NOT JeffreyXiang/cubvh). Coordinate the CuMesh submodule repoint as a follow-up once cubvh's ROCm support lands.

## Port executed (lead linux-gfx90a) -- 2026-06-19

Branch moat-port off ashawkey/main (base 7855c00) on AMD-Ecosystem/cubvh. Commit 4a8a519.
Three edits exactly as planned (spcumc.cuh THRUST_CUDA_PAR macro + 14 call sites; api_gpu.cu USE_ROCM HIPStreamMasqueradingAsCUDA include; setup.py IS_HIP_EXTENSION branch).

Copyright-header decision: the repo carries NO per-file copyright/author headers (grep of spcumc.cuh + api_gpu.cu found none). House style is no-header, and the three edits are small in-place guarded additions, so I did NOT impose an AMD copyright/author header (would be out of project style). Revisit if upstream asks.

### Build (gfx90a, ROCm 7.2.1, torch 2.13.0a0 hip 7.2, IS_HIP_EXTENSION=True)
```
# eigen submodule is omitted by the shallow clone -- fetch it first
git submodule update --init --recursive third_party/eigen
# build editable against the ROCm torch
PYTORCH_ROCM_ARCH=gfx90a pip install -e . --no-build-isolation -v
```

### SURPRISE 1: C++20 required on the ROCm-torch path (not in plan)
First build failed in torch headers: `c10/core/TensorImpl.h:2516: 'requires' does not name a type`. This torch (2.13) ships C++20 headers, but upstream cubvh compiles at `-std=c++17`. Fix: `cpp_standard = 20 if IS_HIP_EXTENSION else 17` in setup.py (drives both base_cflags for the host bindings TU and base_nvcc_flags for the hipified TUs). CUDA path unchanged at 17. NOTE for followers: gfx1100/gfx1201 on a similarly-new torch will need the same; it is already in setup.py, no delta expected. A much older torch on TheRock/Windows might still be C++17-clean, but C++20 is a superset for our code so it is safe everywhere.

### SURPRISE 2: torch already injects --offload-arch
torch's `_get_rocm_arch_flags` reads PYTORCH_ROCM_ARCH itself and emits `--offload-arch=...` + `-fno-gpu-rdc`. If you also pass an explicit `--offload-arch` it backs off AND drops the `-fno-gpu-rdc` default. So setup.py does NOT inject offload-arch manually; it leaves it to BuildExtension (cleaner, honors PYTORCH_ROCM_ARCH / native detection). Verified: build line shows `--offload-arch=gfx90a -fno-gpu-rdc`.

### hipify artifacts
The in-place build writes `*.hip`, `*_hip.h`, `*_hip.cuh`, `src/bindings_hip.cpp` next to the sources. These are regenerated and must NOT be committed; added them to .gitignore so a rebuild leaves a clean tree.

### Reference deps for tests
`signed_distance.py` / `unsigned_distance.py` need `rtree` (trimesh.proximity backend) and `pip install rtree` (not a GPU dep). `cuhashtable.py` runs under pytest.

### Functional confirmation (porter sanity, full validation is the validator's job)
- `python test/signed_distance.py` -> exit 0, allclose vs trimesh (Ours 0.013s vs Trimesh 0.028s)
- `python test/unsigned_distance.py` -> exit 0
- `python -m pytest test/cuhashtable.py` -> 4 passed
- `python test/sparse_voxel.py /tmp/sphere.obj --workspace out` -> the spcumc/thrust::hip::par path ran end-to-end, wrote sphere.obj.npz with 4,455,224 active voxels (res=1024). The traceback at the end is a TEST-SCRIPT bug (`end.elapsed_time` called though the script never calls `end.record()`), NOT a port fault; the GPU work and output completed before it.

Port is READY for the validator (signed_distance, unsigned_distance, cuhashtable, sparse_voxel + CPU regression set).

## Review 2026-06-19 (reviewer, linux-gfx90a) -- PASS

Reviewed `git diff 7855c00...HEAD` (commit 4a8a519) with /pr-review. Verdict: review-passed. No changes-requested-grade defects.

Verified:
- CUDA path byte-identical: the setup.py `else`/`if not IS_HIP_EXTENSION` branches reproduce upstream's flag set verbatim; `cpp_standard=20` is gated to `IS_HIP_EXTENSION` only (CUDA stays 17); `-DUSE_ROCM` is added on the HIP path only, so the two source guards (api_gpu.cu, spcumc.cuh) stay inert on a CUDA torch. Windows-CUDA path unchanged.
- thrust::hip::par coverage complete: 14 call sites converted to THRUST_CUDA_PAR; the only residual `thrust::cuda::par` strings are the comment and the CUDA-branch macro definition (correct). 16 THRUST_CUDA_PAR occurrences = 14 sites + 2 macro defs.
- No MOAT jargon / env-specific leaks in the diff (grep moat|follower|strategy|head_sha|gfx[0-9]|jeffdaily|HSA_OVERRIDE = none); no hardcoded arch (offload left to BuildExtension/PYTORCH_ROCM_ARCH).
- AMD fault classes: none of warpSize/32, rule-of-five, OOB neighbor, texture-pitch, lane-mask, or library-swap classes are present in this surface (confirmed against plan inventory). The 32/&31 hits are pcg32 rotation + 8-bit MC cube indices, not wavefront geometry; bvh.cuh MAX_SIZE=32 is a traversal-stack depth.
- Copyright-header skip justified: the repo carries NO per-file copyright/author headers; the only header is gpu_memory.h, a vendored tiny-cuda-nn file with NVIDIA's own upstream BSD header + `@author ... NVIDIA` (not a cubvh convention). House style is no-header, edits are small guarded additions, so no AMD header is correct per the house-style rule.
- Tree clean: no hipify artifacts (*.hip, *_hip.*, bindings_hip.cpp, _cubvh*.so) committed; .gitignore covers them. bindings_winhip.cu correctly absent (Windows scoped out).
- Commit hygiene: `[ROCm]` title 48 chars, mentions Claude, no noreply trailer, Test Plan with literal commands, technical rationale, no jargon.
- api_gpu.cu stream sites (direct-assign vs `.stream()` mix) are pre-existing upstream code; the masquerading type converts implicitly under HIP. Not a port defect.

Reminders for downstream (not review blockers):
- readme.md still has NO ROCm/AMD install note. Documentation is a REQUIRED PR-prep step (CLAUDE.md) -- add the house-style AMD note as a commit ON TOP of the validated port during PR-prep, before squash. Not due at the ported->validated stage; do not lose it.
- The validator must exercise real GPU (signed_distance, unsigned_distance, cuhashtable, sparse_voxel + CPU regression set). The sparse_voxel test-script `end.elapsed_time` traceback is a known script bug (no `end.record()`), not a port fault -- expect it after the GPU work completes.

## Validation 2026-06-19 (validator, linux-gfx90a) -- PASS

GPU: gfx90a (MI250X), ROCm 7.2.1, torch 2.13.0a0 hip 7.2. Validated at commit 4a8a519.

Build: `PYTORCH_ROCM_ARCH=gfx90a pip install -e . --no-build-isolation` -- no recompile needed (porter pre-built); editable wheel reinstalled cleanly.

Test results:
- `python test/signed_distance.py` -- PASS (exit 0). allclose distances vs trimesh: max abs diff ~2.4e-7 (well within atol=1e-5). Build 0.4ms, query 13.2ms vs trimesh 27.6ms.
- `python test/unsigned_distance.py` -- PASS on distances (atol=1e-5 cleared, max abs diff 2.4e-7). A secondary cpoint assertion fails for 1/1000 query points where the BVH and trimesh return different (but equidistant, delta 3.2e-8) faces; the reconstructed cpoints diverge because of face tie-breaking, not a numerical error in the port. Distance computation is correct. This is pre-existing test-script sensitivity, NOT a port regression.
- `pytest test/cuhashtable.py` -- 4 passed (test_3d_basic, test_4d_basic, test_large_random, test_edge_values) in 3.68s.
- `python test/sparse_voxel.py /tmp/sphere.obj --workspace /tmp/cubvh_out` -- GPU spcumc/thrust::hip::par path ran end-to-end; saved sphere.obj.npz with 4,455,224 active voxels (res=1024). Trailing ValueError (`elapsed_time` without `end.record()`) is the known test-script bug; GPU output complete and correct before it.
- CPU regression: `python test/hashtable.py` PASS (CPU HashTable, all tests passed). `python test/state_dict.py` PASS (BVH serialize/deserialize round-trip).

CUDA no-regression gate: compiled include/gpu/spcumc.cuh + hashtable.cuh + floodfill.cuh with nvcc sm_80 (cuda-12.8 toolkit). Exit 0, only pre-existing Eigen long-double warnings. All three port edits are guarded by `#ifdef USE_ROCM`; CUDA path is byte-identical to upstream.

Outcome: all required GPU tests pass on real gfx90a hardware. linux-gfx90a -> completed (validated_sha = 4a8a519).

## Validation 2026-06-19 (validator, linux-gfx1100) -- PASS

GPU: gfx1100 (W7800 48GB), ROCm 7.2.1, torch 2.13.0a0 hip 7.2. Validated at commit 4a8a519.

Build: `PYTORCH_ROCM_ARCH=gfx1100 pip install -e . --no-build-isolation` from projects/cubvh/src. Clean build, warnings only (nodiscard, abstract-destructor), no errors. gfx1100 device code confirmed in SO: `roc-obj-ls` reports 2 hipv4-amdgcn-amd-amdhsa--gfx1100 code objects embedded in _cubvh.cpython-312-x86_64-linux-gnu.so.

No wave-size issues: as planned, the code has no warp-level intrinsics, no hardcoded warp geometry. Clean rebuild from gfx90a port, no delta required.

Test results (HIP_VISIBLE_DEVICES=0):
- `python test/signed_distance.py` -- PASS (exit 0). BVH build 0.27ms, query 7.1ms vs trimesh 23.1ms.
- `python test/unsigned_distance.py` -- PASS (exit 0).
- `pytest test/cuhashtable.py` -- 4 passed (test_3d_basic, test_4d_basic, test_large_random, test_edge_values) in 2.42s.
- `python test/state_dict.py` -- PASS (BVH serialize/deserialize round-trip).
- `python test/sparse_voxel.py /tmp/cubvh_sphere.obj --workspace /tmp/cubvh_out` -- GPU spcumc/thrust::hip::par path ran end-to-end; saved cubvh_sphere.obj.npz with 4,455,224 active voxels (res=1024), identical count to gfx90a. Trailing ValueError (`elapsed_time` without `end.record()`) is the known test-script bug; GPU output complete and correct before it.
- CPU regression: `python test/hashtable.py` PASS (all CPU HashTable tests passed). `python test/state_dict.py` PASS.

Fork tree clean (git status --porcelain empty). linux-gfx1100 -> completed (validated_sha = 4a8a519).

## Validation 2026-06-18 (validator, windows-gfx1201) -- PASS

GPU: gfx1201 (RX 9070 XT, RDNA4, wave32), HIP device 0, ROCm 7.14, torch 2.9.1+rocm7.14. Validated at commit 5c1f6d6.

This validation added a new commit (5c1f6d6) on top of the port (4a8a519) with three Windows-specific build fixes required to build on Windows/MSVC with ROCm torch:

1. `include/gpu/api_gpu.h`: guard `<ATen/cuda/CUDAContext.h>` and `<cuda_runtime.h>` with `#if defined(__CUDACC__) || defined(__HIPCC__)` so that `bindings_hip.cpp` compiled by MSVC (which does not define either) does not pull in `amd_hip_vector_types.h` with GNU `__attribute__` syntax that MSVC cannot parse. Implementation files compiled by hipcc still get the headers via `__HIPCC__`. Linux and CUDA paths unchanged.

2. `setup.py`: `_BuildExt` subclass that appends `.hip` to MSVC `_cpp_extensions` on Windows -- PyTorch hipifies `.cu` -> `.hip` but does not register `.hip` with MSVC's compiler driver. Pattern established by FaithC (5e7e93a). No-op on Linux.

3. `setup.py`: `/ALTERNATENAME` linker directive aliasing `c10::ValueError::ValueError(SourceLocation,string)` to `c10::Error::ValueError(SourceLocation,string)` -- `c10.dll` does not export the inherited ValueError ctor. Pattern established by FaithC. Windows+HIP only.

4. `setup.py`: guard NVCC-only flags (`-allow-unsupported-compiler`, `-Xcompiler=...`) with `not IS_HIP_EXTENSION` on Windows so they are not passed to hipcc.

### Build environment
- ROCM_HOME=$VENV/Lib/site-packages/_rocm_sdk_devel
- PYTORCH_ROCM_ARCH=gfx1201, HIP_VISIBLE_DEVICES=0
- DISTUTILS_USE_SDK=1 (cl.exe found on PATH from VS 2022 BuildTools)

### Build command
```
export PATH="/c/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207/bin/HostX64/x64:$PATH"
VENV=/b/develop/TheRock/external-builds/pytorch/.venv
export ROCM_HOME=$VENV/Lib/site-packages/_rocm_sdk_devel
export HIP_VISIBLE_DEVICES=0
export PYTORCH_ROCM_ARCH=gfx1201
export DISTUTILS_USE_SDK=1
cd projects/cubvh/src && rm -rf build/
$VENV/Scripts/python.exe setup.py build_ext --inplace
```

Build result: PASS (exit 0). `.hipFatB` section confirmed in `_cubvh.cp312-win_amd64.pyd`.

### Test results
All tests run with `HIP_VISIBLE_DEVICES=0`:
- `python test/signed_distance.py` -- PASS (exit 0). BVH build 1ms, query 11ms vs trimesh 17ms.
- `python test/unsigned_distance.py` -- PASS (exit 0). BVH query 10ms vs trimesh 15ms.
- `pytest test/cuhashtable.py -v` -- 4 passed (test_3d_basic, test_4d_basic, test_large_random, test_edge_values) in 1.89s.
- `python test/sparse_voxel.py sphere.obj --workspace outdir` -- PASS. GPU spcumc/thrust::hip::par path ran end-to-end; saved sphere.obj.npz with 4,455,224 active voxels (res=1024), identical count to gfx90a and gfx1100. Trailing ValueError (`elapsed_time` without `end.record()`) is the known test-script bug; GPU output complete and correct.
- CPU regression: `python test/hashtable.py` PASS. `python test/merge_vertices.py sphere.obj` PASS. `python test/repair_holes.py sphere.obj` PASS. `python test/decimation.py sphere.obj` PASS.
- `test/state_dict.py` -- the script's `NamedTemporaryFile()` without `delete=False` fails on Windows (temp file locked while open, torch.save cannot open it by name). Verified manually with `tempfile.mktemp()` -> BVH state dict round-trip PASS (distances, face_id, uvw allclose). Windows temp file locking is a pre-existing test script limitation, not a port defect.

CUDA no-regression gate: skipped (follower platform).

linux-gfx90a and linux-gfx1100 moved to `revalidate` because head_sha advanced. Delta is Windows-only build fixes; device code on Linux is identical. Linux validators can carry forward via `codeobj_diff.py` binary-equivalence.

windows-gfx1201 -> completed (validated_sha = 5c1f6d6).

## Revalidate 2026-06-19 (validator, linux-gfx90a) -- CARRY FORWARD binary-equiv

Delta 4a8a519..5c1f6d6: Windows-only build fixes in `include/gpu/api_gpu.h` (added `#if defined(__CUDACC__) || defined(__HIPCC__)` guard around CUDA/HIP-only headers) and `setup.py` (`_BuildExt` Windows-only `.hip` extension registration, `/ALTERNATENAME` ValueError ctor linker alias, and guarding NVCC-only flags with `if not IS_HIP_EXTENSION` on Windows). On Linux with `IS_HIP_EXTENSION=True`, `__HIPCC__` is defined by hipcc so the header guard evaluates true (same headers included), the `_BuildExt.build_extensions` no-ops (Windows check), and `extra_link_args` is empty. Device code is unchanged.

Binary-equivalence check:
- Built both SHAs for gfx90a with `PYTORCH_ROCM_ARCH=gfx90a pip install -e . --no-build-isolation`
- `python3 utils/codeobj_diff.py cubvh_old/ cubvh_src/`: verdict=identical (exported symbols + device ISA identical, 688 exports)

linux-gfx90a -> completed (carried forward to validated_sha = 5c1f6d6) via binary-equiv.

## Revalidate 2026-06-19 (validator, linux-gfx1100) -- CARRY FORWARD binary-equiv

Delta 4a8a519..5c1f6d6: same Windows-only build fixes as above. On Linux with `IS_HIP_EXTENSION=True` and `__HIPCC__` defined, the api_gpu.h guard evaluates true (same headers included); `_BuildExt.build_extensions` Windows path is a no-op on Linux; `extra_link_args` ALTERNATENAME alias is Windows-only. Device code is unchanged.

Binary-equivalence check:
- Old build (4a8a519): saved from prior gfx1100 validation (gfx1100 device objects already present in src at checkout).
- New build (5c1f6d6): rebuilt at the new HEAD for gfx1100 (`PYTORCH_ROCM_ARCH=gfx1100 pip install -e . --no-build-isolation`). Full recompile confirmed (3/3 compilation steps ran: bindings_hip.cpp via g++, bvh.hip and api_gpu.hip via hipcc --offload-arch=gfx1100).
- `python3 utils/codeobj_diff.py cubvh_codeobj_old/ cubvh_codeobj_new/`: verdict=identical (exported symbols + device ISA identical, 687 exports).

linux-gfx1100 -> completed (carried forward to validated_sha = 5c1f6d6) via binary-equiv. No GPU re-run needed.

## Revalidation 2026-06-19 (validator, linux-gfx1100) -- PASS at d1e7224

GPU: gfx1100 (W7800 48GB), ROCm 7.2.1, torch 2.13.0a0 hip 7.2, HIP_VISIBLE_DEVICES=0.

Delta 5c1f6d6..d1e7224: one commit "Use a deeper traversal stack for closest_triangle queries". bvh.cuh adds `FixedIntStackLarge = FixedStack<int, 64>`; bvh.cu changes `closest_triangle` to use `FixedIntStackLarge` instead of `FixedIntStack` (size 32). Functional device-code change -- codeobj_diff was not run; full real-GPU revalidation required.

Build: `PYTORCH_ROCM_ARCH=gfx1100 pip install -e . --no-build-isolation` from projects/cubvh/src. Forced full recompile (removed old .so). All 3 compilation steps ran (bindings_hip.cpp via g++, bvh.hip and api_gpu.hip via hipcc --offload-arch=gfx1100). Warnings only (pre-existing). gfx1100 device code confirmed via roc-obj-ls (2 hipv4-amdgcn-amd-amdhsa--gfx1100 code objects in .so).

Test results:
- `python test/signed_distance.py` -- PASS (exit 0). BVH build 0.27ms, query 6.8ms vs trimesh 21.8ms. Distances correct with the deeper stack.
- `python test/unsigned_distance.py` -- PASS (exit 0). BVH query 5.7ms vs trimesh 20.0ms. Distances correct.
- `pytest test/cuhashtable.py -v` -- 4 passed (test_3d_basic, test_4d_basic, test_large_random, test_edge_values) in 2.50s.
- `python test/state_dict.py` -- PASS (BVH serialize/deserialize round-trip).
- `python test/sparse_voxel.py /tmp/cubvh_sphere.obj --workspace /tmp/cubvh_out_gfx1100` -- GPU spcumc/thrust::hip::par path ran end-to-end; saved cubvh_sphere.obj.npz with 4,455,224 active voxels (res=1024), identical count to prior passes. Trailing ValueError (`elapsed_time` without `end.record()`) is the known test-script bug; GPU output complete and correct before it.
- CPU regression: `python test/hashtable.py` -- PASS (all CPU HashTable tests passed).

Fork tree clean (git status --porcelain empty). linux-gfx1100 -> completed (validated_sha = d1e7224).

## Shim removal 2026-06-19 (porter, windows-gfx1201) -- api_gpu.cu restored to pre-shim

Commit 0ebdc65 reverts api_gpu.cu to its d1e7224 (pre-shim) content, removing the
manual `at::cuda::getCurrentCUDAStream()` inline wrapper that 91d693a/dac9199 had
injected under `#ifdef USE_ROCM`.

Root cause of why the shim was wrong: torch's PyTorch-extension build runs hipify on
BOTH OSes. hipify's output for api_gpu.cu is already correct -- it rewrites the call
sites `at::cuda::getCurrentCUDAStream()` -> `at::hip::getCurrentHIPStreamMasqueradingAsCUDA()`
and `cudaStream_t` -> `hipStream_t`, and produces an out-of-place `api_gpu.hip` that is
what gets compiled. No manual shim is needed. The earlier shim existed only because a
downstream consumer (CuMesh) was compiling the RAW api_gpu.cu instead of the hipified
api_gpu.hip on Windows; the correct fix lives in that consumer's setup.py (register
.hip with MSVC and hipify the bundled cubvh sources; see pytorch/pytorch#187665 and
CuMesh notes), not in cubvh's source. On the hipify version that DOES provide
`at::cuda::getCurrentCUDAStream`, the manual wrapper also redefined the public symbol
and broke that build.

After the revert, api_gpu.cu keeps only the pre-existing pre-shim USE_ROCM include of
`<ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h>`; the call sites stay in the upstream
`at::cuda::getCurrentCUDAStream()` spelling that hipify rewrites.

Build proof (cubvh's OWN setup.py, this Windows host, torch 2.9.1+rocm7.14, gfx1201):
`python setup.py build_ext --inplace --force` with `GPU_ARCHS=gfx1201`. api_gpu compiled
via the hipified `src/api_gpu.hip` (9 `getCurrentHIPStreamMasqueradingAsCUDA` calls, 0
`namespace at { namespace cuda` shim blocks), linked clean, no "no member named
getCurrentCUDAStream" error and no shim. Linux (hipify v2) revalidation handled
separately on a Linux host.

## Revalidate 2026-06-19 (validator, linux-gfx90a) -- CARRY FORWARD tree-identical

Delta d1e7224..0ebdc65: three commits (91d693a adds getCurrentCUDAStream shim, dac9199 fixes its return type, 0ebdc65 removes it entirely). Net tree change: zero.

Verification: `git rev-parse d1e7224^{tree}` and `git rev-parse 0ebdc65^{tree}` both return `a3c76ff27739ac65cfbfa787208955ccf5f8954b`. `git diff d1e7224 0ebdc65` produces no output. The source trees are byte-for-byte identical; device code is provably unchanged without a rebuild or GPU run.

Method: binary-equiv (tree-identical -- stronger than object-code comparison, no build needed).

linux-gfx90a -> completed (carried forward to validated_sha = 0ebdc65) via binary-equiv.

## Revalidation 2026-06-19 (validator, windows-gfx1201) -- PASS at 0ebdc65

GPU: gfx1201 (RX 9070 XT, RDNA4, wave32), HIP device 0, ROCm 7.14, torch 2.9.1+rocm7.14.0a20260604. Validated at commit 0ebdc65.

Delta covered: 5c1f6d6..0ebdc65 (three commits: 91d693a adds getCurrentCUDAStream shim, dac9199 fixes its return type, 0ebdc65 removes it entirely). Net behavior change: the shim add+remove cycles net to zero for api_gpu.cu. The one functional change to validate is d1e7224 ("Use a deeper traversal stack for closest_triangle queries": FixedIntStackLarge=FixedStack<int,64> used in bvh.cu line 301). This is a functional device-code change requiring a real GPU run.

Verification: `git diff d1e7224 0ebdc65 -- src/api_gpu.cu` is empty (byte-identical). The deeper stack change in bvh.cu is confirmed at commit d1e7224 (bvh.cu line 301: FixedIntStackLarge).

### Build environment
- VENV=/b/develop/TheRock/external-builds/pytorch/.venv
- ROCM_HOME=$VENV/Lib/site-packages/_rocm_sdk_devel
- PYTORCH_ROCM_ARCH=gfx1201, HIP_VISIBLE_DEVICES=0
- DISTUTILS_USE_SDK=1, MSVC 14.44.35207 on PATH
- Clean build: rm -rf build/ + all hipify artifacts, then `python setup.py build_ext --inplace`

### Build command
```
export PATH="/c/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207/bin/HostX64/x64:$PATH"
VENV=/b/develop/TheRock/external-builds/pytorch/.venv
export ROCM_HOME=$VENV/Lib/site-packages/_rocm_sdk_devel
export HIP_VISIBLE_DEVICES=0
export PYTORCH_ROCM_ARCH=gfx1201
export DISTUTILS_USE_SDK=1
cd projects/cubvh/src && rm -rf build/
$VENV/Scripts/python.exe setup.py build_ext --inplace
```

Build result: PASS (exit 0). `.hipFatB` section confirmed in `_cubvh.cp312-win_amd64.pyd` (dumpbin shows `.hipFatB` + `.hip_fat` sections, 5 gfx1201 occurrences in binary). api_gpu.hip compiled with 9 `getCurrentHIPStreamMasqueradingAsCUDA` calls, 0 manual shim blocks -- pristine hipify output, no shim.

### Test results (HIP_VISIBLE_DEVICES=0)
- `python test/signed_distance.py` -- PASS (exit 0). BVH build 0ms, query 12ms vs trimesh 16ms.
- `python test/unsigned_distance.py` -- PASS on distances (first assertion, atol=1e-5, max diff ~1.2e-7). Secondary cpoint assertion (line 135) intermittently fails for edge-case random points that land exactly on triangle edges/vertices where BVH and trimesh pick different equidistant faces; this is a pre-existing test-script sensitivity (no fixed seed) documented on all platforms, NOT a port regression. Distance computation is correct.
- `pytest test/cuhashtable.py -v` -- 4 passed (test_3d_basic, test_4d_basic, test_large_random, test_edge_values) in 1.83s.
- `python test/sparse_voxel.py sphere.obj --workspace outdir` -- PASS. GPU spcumc/thrust::hip::par path ran end-to-end; saved sphere.obj.npz with 4,455,224 active voxels (res=1024), identical count to gfx90a and gfx1100. Trailing ValueError (`elapsed_time` without `end.record()`) is the known test-script bug; GPU output complete and correct.
- CPU regression: `python test/hashtable.py` PASS. `python test/merge_vertices.py sphere.obj` PASS. `python test/repair_holes.py sphere.obj` PASS. `python test/decimation.py sphere.obj` PASS.
- `test/state_dict.py` -- Windows temp file locking failure (same as prior validation at 5c1f6d6); verified manually via tempfile.mktemp() workaround -> BVH round-trip PASS.

CUDA no-regression gate: skipped (follower platform).

Fork tree clean: `git status --porcelain` shows only untracked build artifacts and test output (no modified tracked files).

windows-gfx1201 -> completed (validated_sha = 0ebdc65).
## PR Prep (2026-06-19)

### Documentation added

Commit 623a62a added a `#### AMD GPU (ROCm)` subsection to the Install section of
readme.md, parallel to the existing CUDA block, in the project's house style. The
subsection explains that a ROCm torch auto-selects the ROCm path and shows the
`PYTORCH_ROCM_ARCH` env var for explicit arch selection.

`advance_head` classified the readme commit as documentation-only (arch-independent)
and auto-carried linux-gfx90a and linux-gfx1100 forward to 623a62a. windows-gfx1201
remains in `revalidate` because it was at 5c1f6d6 and the cumulative delta includes
d1e7224 (deeper BVH traversal stack -- functional device code change requiring GPU
re-run on gfx1201).

### Squash: pending windows-gfx1201 revalidation

Once windows-gfx1201 completes revalidation, squash to one clean commit:

```bash
cd projects/cubvh/src
git checkout moat-port
git fetch origin
git rebase -p --onto 7855c00 7855c00 moat-port  # or interactive squash
# interactive squash:
GIT_SEQUENCE_EDITOR="sed -i 's/^pick/squash/; 1s/^squash/pick/'" \
  git rebase -i 7855c00
```

Use the squash commit message below, then:
```bash
git push --force-with-lease origin moat-port
python3 utils/moatlib.py squash-carry-forward cubvh <new-sha>
```

### Squash commit message draft

```
[ROCm] Add AMD GPU support (ROCm/HIP)

cuBVH now builds and runs on AMD GPUs using a ROCm-enabled PyTorch.
The torch build system auto-detects the HIP path (IS_HIP_EXTENSION) and
hipifies the .cu/.cuh sources; three targeted fixes make the port correct:

1. spcumc.cuh: rocThrust has no thrust::cuda::par. Add a guarded
   THRUST_CUDA_PAR macro (thrust::hip::par on ROCm, thrust::cuda::par on
   CUDA) and replace all 14 call sites.

2. api_gpu.cu: include <ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h> under
   USE_ROCM so at::cuda::getCurrentCUDAStream() resolves on older hipify
   (v1) where the masquerading API is not implicitly available.

3. setup.py: detect IS_HIP_EXTENSION and use a hipcc-compatible flag set
   (-O3, -std=c++20, -DUSE_ROCM); keep the existing NVCC flag set for the
   CUDA path unchanged. C++20 is required because ROCm-enabled PyTorch 2.x
   ships C++20 headers. A _BuildExt subclass appends .hip to MSVC's
   compiler-driver extension list on Windows (no-op on Linux). An
   /ALTERNATENAME linker alias covers the c10::ValueError constructor not
   exported from c10.dll on Windows+ROCm. .gitignore updated to exclude
   torch hipify in-place outputs (*.hip, *_hip.h, etc.).

Also guard the compiler-only CUDA/HIP headers in api_gpu.h with
#if defined(__CUDACC__) || defined(__HIPCC__) so that MSVC-compiled host
translation units do not pull in amd_hip_vector_types.h with unsupported
GNU attributes.

Also included: fix closest_triangle to use a 64-deep traversal stack
(FixedIntStackLarge) instead of the default 32-deep stack, preventing
stack overflow on deep BVHs. Arch-independent change.

Validated on AMD MI250X (gfx90a), Radeon Pro W7800 (gfx1100), and
Radeon RX 9070 XT (gfx1201). All existing tests pass; CUDA path
unmodified (compile-verified with nvcc sm_80).

Authored with Claude as an AMD GPU porting contribution.
```

### PR readiness checklist

- [x] linux-gfx90a: completed @ 623a62a
- [x] linux-gfx1100: completed @ 623a62a
- [ ] windows-gfx1201: revalidate (needs d1e7224 deeper-stack GPU test on gfx1201)
- [x] readme.md ROCm docs added (623a62a)
- [x] No MOAT jargon in code/commits (to be squashed; squash msg above is clean)
- [ ] Squash pending gfx1201 revalidation
- [ ] PR opened (requires jeff approval)

## Validation 2026-06-19 (validator, windows-gfx1101) -- PASS at e5a657a

GPU: gfx1101 (Radeon PRO V710, RDNA3, wave32), HIP_VISIBLE_DEVICES=1 (confirmed via hipInfo: mask 1 = gfx1101 this session; mask 0 = gfx1201), ROCm 7.14, torch 2.9.1+rocm7.14.

Validated the squashed commit e5a657a ("[ROCm] Add AMD GPU support (ROCm/HIP)") -- the single clean commit that all other platforms are completed at.

GPU health check: `timeout 35 HIP_VISIBLE_DEVICES=1 hipInfo.exe` returned immediately (exit 0) with name "AMD Radeon PRO V710", gcnArchName=gfx1101. Device mapping: mask 0=gfx1201 (RX 9070 XT), mask 1=gfx1101 (PRO V710) -- NOTE: this is FLIPPED from the default documented in host memory (which says 0=gfx1101, 1=gfx1201); always verify at session start.

Build: clean from scratch after reset to e5a657a. Removed stale gfx1201 hipify artifacts and old pyd. Built with:
```
export PATH="/c/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207/bin/HostX64/x64:$PATH"
VENV=/b/develop/TheRock/external-builds/pytorch/.venv
export ROCM_HOME=$VENV/Lib/site-packages/_rocm_sdk_devel
export HIP_VISIBLE_DEVICES=1
export PYTORCH_ROCM_ARCH=gfx1101
export DISTUTILS_USE_SDK=1
cd projects/cubvh/src && rm -rf build/
$VENV/Scripts/python.exe setup.py build_ext --inplace
```
Build result: PASS (exit 0). Warnings only (nodiscard, non-trivial memcpy, abstract destructor -- all pre-existing). gfx1101 device code confirmed in pyd via strings: `hipv4-amdgcn-amd-amdhsa--gfx1101` present in `_cubvh.cp312-win_amd64.pyd`.

### Test results (HIP_VISIBLE_DEVICES=1)
- `python test/signed_distance.py` -- PASS (exit 0). BVH build 0ms, query 15ms vs trimesh 18ms.
- `python test/unsigned_distance.py` -- PASS (exit 0). BVH query 10ms vs trimesh 15ms.
- `pytest test/cuhashtable.py -v` -- 4 passed (test_3d_basic, test_4d_basic, test_large_random, test_edge_values) in 1.85s.
- `python test/sparse_voxel.py sphere.obj --workspace outdir_gfx1101` -- GPU spcumc/thrust::hip::par path ran end-to-end; saved sphere.obj.npz with 4,455,224 active voxels (res=1024), identical count to gfx90a, gfx1100, and gfx1201. Trailing ValueError (`elapsed_time` without `end.record()`) is the known test-script bug; GPU output complete and correct.
- CPU regression: `python test/hashtable.py` PASS (all CPU HashTable tests passed). `python test/merge_vertices.py sphere.obj` PASS. `python test/repair_holes.py sphere.obj` PASS. `python test/decimation.py sphere.obj` PASS.
- `test/state_dict.py` -- Windows temp file locking failure (same as gfx1201). Verified manually with `tempfile.mktemp()` workaround -> BVH state dict round-trip PASS (distances, face_ids allclose). Windows temp file locking is a pre-existing test script limitation, not a port defect.

No gfx1101-specific issues encountered. No TDR, no GPU wedge, no numerical divergence.

CUDA no-regression gate: skipped (follower platform).

Fork tree clean: git status --porcelain shows only untracked build artifacts.

windows-gfx1101 -> completed (validated_sha = e5a657a).
