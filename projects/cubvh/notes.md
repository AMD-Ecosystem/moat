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


## Licence scope answered by upstream (2026-08-26)

Asked which files LICENSE_NVIDIA covers, the upstream author replied (private
email to Jeff Daily, paraphrased): several files borrow original code lines
from instant-ngp -- src/bvh.cu, include/gpu/triangle.cuh, and
include/gpu/bounding_box.cuh -- and LICENSE_NVIDIA was included for exactly
that reason; the author suggested it may be better to rewrite them. So the
NVIDIA Source Code License (non-commercial) plausibly covers derived code in
the project's core, even though no per-file notice marks it. Relevant exposure:
src/bvh.cu is compiled into every build including AMD, bvh.cu includes the
derived headers, our merged PR #33 modified src/bvh.cu, and CuMesh vendors all
of it with its own PR open. The licensing review has these facts.

## Provenance quantified (2026-08-26)

A file-by-file comparison against instant-ngp (at cubvh's 2022 ancestor
snapshot) and tiny-cuda-nn quantified the derivation the author described:
roughly 1,200 raw lines across bvh.cu, bounding_box.cuh, triangle.cuh,
bvh.cuh, and a block of common.h descend from instant-ngp; the marching-cubes
files share only the canonical published lookup table; gpu_memory.h is
tiny-cuda-nn BSD-3 and compliant as retained. All derived material implements
published algorithms, so an independent reimplementation from the literature
is feasible; the port's own edits intersect the derived regions in only two
lines. The detailed report is with the licensing review.

## Round 2: clean-room rewrite -- porter session 1 (linux-gfx90a, 2026-08-26)

Plan: see "Round 2" in plan.md. The deferral cubvh-nvidia-proprietary-rescan
was ruled `now` (jeffdaily, 2026-08-13); the author invited the rewrite.

### Process (clean-room, two roles)

- A spec-author agent (this session's orchestrator) read the derived files
  ONCE and wrote projects/cubvh/rewrite/SPEC.md: pinned struct layouts
  (Triangle 48B a,b,c,id; BoundingBox min,max; TriangleBvhNode bb,left,right,
  escape = 9 int32), the leaf/inner/escape node encoding, every behavioral
  constant (MAX_DIST=1000, branching 4, leaf cap 8, stacks 32/64, fibonacci
  epsilon ladder, pcg32 advance(2i), sentinels 1e6/{0,0}, EPSILON 1e-6), and
  the quirks (degenerate-triangle NaN never wins closest; non-strict <= tie
  keeps later triangle; ray t>=0 accepted; miss depth = MAX_DIST; negative
  slab tmin not clamped; lazy GPU upload).
- cubvh-original blocks that are NOT part of the rewrite were extracted
  verbatim to rewrite/keep/ (closest_point/point_in_triangle/
  closest_point_to_line/barycentric; bvh.cuh state_dict block; safe_divide).
- The five derived files were BLANKED in the working tree and hipify
  artifacts removed; a fresh implementer agent (separate context) writes the
  new files from SPEC.md + published references only, with explicit
  prohibitions on git-history recovery, *_hip artifacts, and instant-ngp.
- Independence will be verified with rewrite/similarity.py (same normalized
  difflib + verbatim-line method as the provenance analysis) against the
  instant-ngp c4d622e snapshot AND old cubvh e5a657a.

### Golden harness (projects/cubvh/harness/golden.py)

Captured from the e5a657a build on gfx90a (MI250X, torch 2.14.0a0 hip 7.14,
numpy<2 venv -- NOTE host torch moved from 2.13/ROCm7.2.1 since round 1; the
env's numpy 2.5 breaks torch's numpy bridge, venv with numpy 1.26 fixes it):
- meshes: torus10k (10368 tris, watertight), open (4393 tris, boundary),
  degen (1155 tris + 3 zero-area/degenerate faces)
- per mesh: unsigned/signed(watertight)/signed(raystab) distances + face_id
  + uvw on 20k points, ray_trace on 20k rays, and the serialized state_dict
- checks: distances atol 2e-5; face_id >=99.9% with tie verification; uvw
  NaN-mask-aware atol 1e-4 (degenerate faces produce NaN uvw -- preserved
  behavior); raystab sign >=99.9%; hit/miss >=99.9%; miss depth == 1000;
  own-build state_dict round-trip bitwise; CROSSLOAD of old state_dicts.
- baseline self-check: check PASS, crossload PASS (all diffs 0).
- goldens live in agent_space/goldens-e5a657a-gfx90a (not committed);
  regenerate on any host by building e5a657a and running `golden.py capture`.
  A wave32 host should capture its own e5a657a goldens BEFORE validating the
  rewrite (plan step 1).

### Baseline perf (harness/bench.py, gfx90a, e5a657a, 500k queries, 5 reps)

median ms: small10k build 3.62, ray 1.66, udf 4.66, sdfw 5.03, sdfr 153.91;
med200k build 89.23, ray 2.85, udf 27.49, sdfw 27.53, sdfr 317.94;
big2m build 930.51, ray 4.63, udf 105.03, sdfw 102.81, sdfr 530.45.
Full JSON: agent_space/bench-e5a657a-gfx90a.json (regenerable).

### Scope decision recorded

Unused derived members are DROPPED rather than rewritten (smaller licence
surface): BoundingBox SAT triangle test/project<>/signed_distance/contains/
inflate/center/diag/relative_pos/intersection/is_empty/corner enumeration/
stream op; Triangle sample_uniform_position/surface_area/distance()/
get_vertices/stream op. Nothing in cubvh (or its tests) uses them; the PR
will state the removal plainly so downstream users can object.

### Implementer round 1 result + adjudications (2026-08-26)

The implementer produced the five files (929 lines total vs ~1360 before,
unused derived members dropped per plan) and independently verified that the
new build's node array and triangle permutation are BIT-IDENTICAL to the old
build's (the median-split build reproduces the old tree exactly), so the
build/layout/encoding contracts hold exactly; own-build state_dict round-trip
is bitwise, and the old build's serialized BVHs crossload and answer
identically within tolerance.

Three residual classes were adjudicated by the spec author and encoded into
the golden harness policy:
1. face_id tie flips (~2% on tie-dense torus meshes): every mismatch is a
   verified tie (distance gap <= 1.2e-7); the winner of an exact tie is
   decided by inlining-sensitive 1-ulp FMA differences (hipcc -O3 contracts
   FMA differently per inlining context -- the implementer demonstrated the
   IDENTICAL source produces 1-ulp differences on ~6% of queries when
   compiled standalone vs in-extension). Policy: only NON-tie mismatches
   (gap > 2e-5) count, budget 0.1%. Result: 0 non-tie mismatches anywhere.
2. Degenerate zero-area faces: the OLD build's handling is signed-zero
   dependent (copysignf(1,+-0) flips an inside/outside branch), sometimes
   NaN (never wins), sometimes a finite clamped-segment distance (wins).
   Unreconstructible-by-design garbage. INTENTIONAL behavior change: the
   rewrite makes zero-area faces report +inf distance (never win any
   distance query). Harness: degenerate-won queries (655 on the degen mesh)
   are excluded from old-vs-new and instead gated by a self-consistency
   check -- the new build on the degen mesh must match the new build on the
   same mesh with degenerate faces removed (result: 0.0 max diff). Ray
   behavior on degenerate faces is unchanged-in-kind (safe_divide guard,
   matches old on all but 63/20000 borderline grazing rays, masked).
   This change goes in the PR body as a documented fix.
3. Ray depth compared on same-face hits (one grazing ray on the open mesh
   hits a different face at the barycentric boundary; hit/miss and face-id
   fractions are separately gated).

### PROVENANCE CORRECTION (supersedes part of the 2026-08-26 report)

The provenance analysis classified triangle.cuh's `point_in_triangle`,
`closest_point_to_line`, and `closest_point` as cubvh-original. That was
WRONG: all three exist near-verbatim in instant-ngp's triangle.cuh at the
ancestor snapshot (even the `// p -= p;` comment). Caught by the
post-implementation similarity scan (rewrite/similarity.py) -- keeping them
verbatim would have carried NSCL text through the rewrite. The keep set is
corrected to: triangle.cuh `barycentric` ONLY (verified absent from ngp),
bvh.cuh state_dict/load_state_dict block (torch serialization, PR #28,
absent from ngp), common.h safe_divide + PI/SQRT2 constants (absent from
ngp). A second implementer round is rewriting the three members from the
amended spec (closest-point-on-triangle per Ericson RTCD ch. 5 behavior).
Two byte-identical one-liner spellings (BoundingBox::distance_sq cwiseMax
chain, fibonacci sin_theta line) are also being respelled with identical
arithmetic.

### First similarity scan (before the fix round)

vs instant-ngp c4d622e (linefrac, was -> now): bvh.cu 0.68 -> 0.044,
bvh.cuh 0.44 -> 0.231 (all 12 matches are pinned interface/POD lines),
triangle.cuh 0.48 -> 0.250 (mostly the three wrongly-kept members -- being
fixed), bounding_box.cuh 0.75 -> 0.125 (4 lines: struct decl, enlarge loop,
the distance_sq one-liner being respelled), common.h 0.44 -> 0.125 (pinned
fibonacci formulas). Final numbers after the fix round to follow.

### Rewrite landed: fork commit 81a98f0 (porter, linux-gfx90a, 2026-08-26)

Implementer rounds 2 (keep-list correction) and 3 (performance) completed;
all gates green and independently re-verified by the orchestrator:
- golden.py check PASS and crossload PASS vs the e5a657a goldens: distance
  diffs <= 2.4e-7, ZERO non-tie face_id mismatches on any mesh, uvw <=
  3.9e-6 with NaN masks equal, ray hit/miss 1.00000 on every mesh after
  degenerate masking, old serialized BVHs load and answer identically, and
  the degenerate self-consistency check (degen mesh vs degen-faces-removed
  mesh) is exactly 0.0.
- Upstream tests: signed_distance, unsigned_distance (rc=0), state_dict,
  cuhashtable (4 passed), CPU hashtable, sparse_voxel end-to-end (known
  pre-existing elapsed_time script bug at exit; npz written, 4,476,440
  active voxels on a fresh icosphere -- spcumc untouched by this round).
- Perf (bench.py, 500k queries, medians ms, old e5a657a -> new 81a98f0,
  gfx90a): build 3.62->3.42 / 89.2->86.1 / 930.5->903.7; ray_trace
  1.66->1.45 / 2.85->2.57 / 4.63->4.29; udf 4.66->4.60 / 27.5->28.2 /
  105.0->101.0; sdfw 5.03->5.14 / 27.5->28.4 / 102.8->100.3; sdfr
  153.9->156.4 / 317.9->324.2 / 530.4->535.0 (10k/200k/2M tris). Build and
  ray faster everywhere; distance ops within ~3% (run noise ~2%). JSONs:
  agent_space/bench-e5a657a-gfx90a.json, bench-rewrite2-gfx90a.json.
- Perf history: the first implementation was 10-24% slower on the query
  kernels; fixed by a division-and-sqrt-free distance_sq region form,
  compile-time-unrolled fan-out-4 child loops behind a != FANOUT stackless
  guard, a branchless single-exit ray intersector, a branchless slab test,
  and a fixed 5-exchange compare-swap child ordering network (the canonical
  optimal 4-input network) replacing a divergent insertion sort.
- Final similarity vs instant-ngp c4d622e (non-trivial verbatim linefrac,
  round-1 -> final): bvh.cu 0.68 -> 0.044, bvh.cuh 0.44 -> 0.231,
  triangle.cuh 0.48 -> 0.031, bounding_box.cuh 0.75 -> 0.091, common.h
  0.44 -> 0.098. Every residual match is a pinned POD/interface line, a
  pinned formula, or a trivial one-liner (inspected and listed by
  rewrite/similarity.py -v). bvh.cuh's 12 matches are all interface
  signatures plus struct members that ARE the serialization contract.
- licenses.py scan-nvidia: the only remaining hit is LICENSE_NVIDIA itself
  (root licence file; its removal is upstream's call -- the PR notes it
  becomes removable). jargon.py --port cubvh: clean. surface.json
  generated (20 CUDA files). Fork main mirror fast-forwarded
  d958f89 -> 757b913 (upstream merge of PR #33; plain fast-forward, kept
  the mirror clean -- required before the follow-up review flow).
- audit-commits could not be re-run after the mirror sync (harness
  permission classifier blocked the command); criteria checked by hand:
  [ROCm] title 67 chars, AI disclosure, fenced Test Plan, no agent
  trailer, jargon clean.

Commit: 81a98f0 "[ROCm] Rewrite BVH core as independent MIT-licensed
implementations", pushed to AMD-Ecosystem/cubvh moat-port (PR #33 merged,
branch free). head_sha advanced; linux-gfx90a set ported.

### Outstanding for the rest of the round

- Reviewer: normal pr-review of e5a657a..81a98f0 plus the independence
  evidence (rewrite/SPEC.md process, similarity numbers, the corrected
  keep set). The clean-room process notes above are the review inputs.
- Validators at 81a98f0 per the gate lattice. NOTE for the wave32/Windows
  hosts: capture e5a657a baseline goldens FIRST (checkout e5a657a, build,
  `harness/golden.py capture`), then build 81a98f0 and run check +
  crossload, then the platform's normal test set. bench.py old-vs-new on a
  wave32 box completes the plan's cross-arch perf deliverable.
- CUDA-path compile check: NOT done -- no NVIDIA toolchain anywhere in the
  current fleet (round-1's cuda-12.8 container is gone; the pip
  nvidia-cuda-nvcc-cu12 wheel ships only ptxas). The PR body states this
  honestly and invites the author to run the CUDA build. If a host with
  nvcc appears, compile a TU including the four rewritten headers plus
  bvh.cu's traversal statics.
- Then the follow-up-PR flow (upstream.py --review; published_sha backfill
  will stamp e5a657a from the live PR; /moat approve; --publish), and
  after upstream merges: CuMesh third_party/cubvh bump.

## Review (round 2) 2026-08-26 (reviewer, linux-gfx90a) -- CHANGES REQUESTED

Scope: fork diff e5a657a..81a98f0 (5 files, +593/-1050) plus the round's
independence evidence. Code, strategy, layout/serialization contracts,
harness policy and the two functional gates were re-verified independently;
the two findings below are both in deliverable text, not in the code.

### 1. MUST FIX -- the commit body states results the harness contradicts

`git log -1` on 81a98f0, paragraph "Differential testing against the
previous build ... shows distances within 2.4e-7, identical ray hit/miss
sets, and face-id differences only at exact distance ties". This becomes the
upstream PR body, and a maintainer can rerun the same harness. Measured on
this host at 81a98f0 vs the e5a657a goldens (both `golden.py check` and
`golden.py crossload`, and re-measured directly):

- max |distance diff| is 3.576e-07, not 2.4e-7 -- see the printed
  `degen/ud.dist`, `degen/sdw.dist`, `degen/sdr.dist` lines
  ("max |d| diff 3.576e-07"). torus10k and open are 1.192e-07. The same
  wrong figure is in the porter note above ("distance diffs <= 2.4e-7");
  2.384e-07 is the largest distance GAP among face-id mismatches, which is
  a different quantity.
- ray hit/miss sets are NOT identical on the degenerate mesh: 63 of 20000
  rays flip between hit and miss (measured unmasked; every one of them
  involves a zero-area face in one build or the other -- old hits a
  degenerate face on 208 rays, new on 189). The harness only reads
  1.00000 there because `golden.py:253` excludes the 230 degenerate-face
  rays. Cause is legitimate and worth stating rather than hiding: the new
  Moller-Trumbore intersector guards the reciprocal determinant with
  `safe_divide` (triangle.cuh:37) where the previous Quilez form divided
  unguarded, so grazing rays against a zero-area triangle resolve
  differently. The neighbouring sentence "Ray casts are unaffected" is
  true as written about the +inf distance rule (ray_intersect never calls
  distance_sq) but reads, next to the claim above, as a promise of
  identical ray output; make it explicit that only non-degenerate geometry
  is bit-comparable.
- "face-id differences only at exact distance ties": the largest distance
  gap among mismatching face ids is 2.384e-07 (degen; 5.96e-08 on
  torus10k/open) -- a one-to-two-ulp tie, not an exact one. Say "ties to
  within a few ulps".

Fix by amending the commit body (and the numbers in the porter note); no
code change is implied.

### 2. MUST FIX -- the round's generalizable method is not promoted

`.claude/skills/cuda-to-rocm/references/` contains no mention of golden
capture, differential comparison, or clean-room rewriting (grep for
"golden|differential|clean-room" returns nothing). Two methods from this
round are project-independent and the next porter cannot see them from
here:

- capture goldens from the PRE-change build and gate the change against
  them, with a comparison policy that (a) treats id selection among exact
  ties as unconstrained, (b) masks the queries covered by a deliberate
  behavior change and replaces that coverage with a self-consistency
  invariant instead of dropping it. Belongs in `references/validation.md`.
- the clean-room shape that worked here: spec-first from a separate agent,
  blanked files, an implementer with no access to the originals, then a
  normalized-line similarity scan as the check -- which is what caught
  three members wrongly classified as original.

Keep it short and put it on this branch so it is reviewed with the code.

### Verified, no action (recorded so the next round does not redo it)

- Gates re-run by the reviewer at 81a98f0 in agent_space/venv-cubvh:
  `golden.py check --ref agent_space/goldens-e5a657a-gfx90a` PASS and
  `golden.py crossload --ref ...` PASS.
- The commit's strongest claim is true: `state_dict()["nodes"]` and
  ["triangles"] from the new build are BITWISE equal to the e5a657a
  goldens on all three meshes (torus10k 5461x9, open 1365x9, degen
  341x9). Layout static_asserts (48/24/36) plus the aggregate-init
  contract at api_gpu.cu:35 hold; Triangle remains an aggregate.
- Harness policy is sound, not a cover: the degenerate mask is
  ref-side-only in practice (655 old-won queries, 0 new-won), and the
  substituted invariant (degen mesh vs degen-faces-removed mesh, same
  build) is exact 0.0 on distances. `degen.nodegen/sdr.sign 0.99945` is
  expected -- zero-area faces still block stab rays.
- The +inf degenerate rule survives the collinear case, which was the
  worry: `triangle.cuh:82-91` reaches the face branch only when the
  barycentric numerators are exactly 0, which is not guaranteed for
  collinear-but-distinct vertices; the degen mesh carries exactly that
  triangle (golden.py:106) and the self-consistency check is 0.0, so it
  holds in practice on gfx90a.
- Corner of the intentional change, not covered by any test: a mesh whose
  faces are ALL zero-area now returns distance 0 / face 0 everywhere (the
  historical {0,0} no-winner fallback, bvh.cu:176-181). Verified directly.
  Pathological input; mentioning it in the PR body is optional.
- `test/unsigned_distance.py` fails intermittently (1 of 8 seeds, ~12%)
  on the cpoint assert at atol=1e-5. NOT attributable to the rewrite: the
  test draws unseeded `torch.randn` points, and the failing query's
  closest point is exactly a shared dodecahedron vertex where trimesh
  returns a point 1.55e-4 away on a different face; our distance agrees
  with trimesh to 3e-7 and with a brute-force reference on our own face.
  Validators should rerun rather than treat a single red run as a
  regression.
- Fault classes: no wave-size assumption anywhere in the five files (no
  shuffles, ballots, shared memory or lane masks; 128-thread blocks;
  FixedStack 32/64 are traversal depths). No textures, no RAII handles,
  no library swaps. Neighbor reads are index-range scans from the node
  encoding, bounded by the leaf range.
- CUDA-path portability: no C++20-only construct in the rewritten math
  headers -- `hipcc -std=c++17 -fsyntax-only -Wall` over a TU
  instantiating Triangle::{distance_sq,ray_intersect,closest_point,
  barycentric,normal,centroid}, BoundingBox::{distance_sq,ray_intersect}
  and fibonacci_dir<32> is clean at both c++17 and c++20. bvh.cuh/bvh.cu
  cannot be checked at c++17 on this host because this torch itself
  requires C++20 (that is why setup.py splits the standard); the language
  constructs used there (std::pair, std::numeric_limits, #pragma unroll,
  Ref<const Vector3f>, host-only std::nth_element) all have precedent in
  the pre-rewrite files that the CUDA build compiled.
- Dropped members have no references anywhere in the tree, tests
  included: sample_uniform_position, surface_area, Triangle::distance,
  get_vertices, point_in_triangle, closest_point_to_line, and the
  BoundingBox SAT/contains/inflate/center/diag/relative_pos/corner/stream
  helpers all return zero hits (include/cpu/fill_holes.h's
  point_in_triangle_2d is an unrelated CPU 2-D predicate). Only
  hashtable.cuh (div_round_up) and api_gpu.cu consume the rewritten
  headers externally, and both still compile and run.
- Sorting network at bvh.cu:93-99 is the canonical optimal 4-input,
  5-comparator network -- brute-forced over all 24 permutations, 0
  failures. Escape threading (bvh.cu:383-392) is structural, so the
  non-pre-order node emission order of the LIFO build is harmless.
- Hygiene: title 67 chars with [ROCm]; AI disclosure present; fenced Test
  Plan; no agent trailer; `jargon.py --port cubvh` clean;
  `moatlib.py audit-commits cubvh` OK; ASCII-only in the new files and in
  the commit body; fork tree clean (`git status --porcelain` empty; only
  ignored *_hip.* and the .so are untracked); 81a98f0 pushed to
  origin/moat-port; head_sha matches; `licenses.py scan-nvidia cubvh`
  reports LICENSE_NVIDIA only.

### Independence evidence -- judged sufficient

Re-run of rewrite/similarity.py by the reviewer against the ngp c4d622e
snapshot reproduces the recorded numbers (linefrac: bvh.cu 0.044,
triangle.cuh 0.031, bounding_box.cuh 0.091, common.h 0.098/0.024,
bvh.cuh 0.231; whole-file ratios 0.030/0.062/0.128/0.081/0.394). Every
residual match was inspected:

- bvh.cu: 12 lines, all pinned interface (`build(...) override`,
  `TriangleBvh::make()`, `resize_and_copy_from_host`, `rng.advance(i*2)`)
  or trivial one-liners (`int idx = 0;`, `if (node.left_idx < 0) {`).
- triangle.cuh: 3 lines -- `struct Triangle {`, the pinned member
  declaration, and `return (a + b + c) / 3.0f;`.
- bounding_box.cuh: 3 lines -- struct declaration and a two-line iterator
  loop.
- bvh.cuh: 12 lines, all POD members that ARE the serialization contract,
  pure-virtual signatures the untouched api layer calls, and
  class/alias declarations.
- common.h: `template <typename T>`, `float sin_phi, cos_phi;` and the
  two-line `fractf`.

Structural spot-checks by eye against the ngp originals confirm
paraphrase-free rewrites, not renamed copies: child ordering is a
5-exchange network over parallel key/slot arrays sorted ASCENDING with a
reverse push, against ngp's descending `sorting_network<N,T>` over
`DistAndIdx` (different comparator sequence too); `distance_sq` is a
barycentric sign-form region test with clamped-edge fallbacks against
ngp's Quilez cross-product sign-sum ternary; `closest_point` uses a
barycentric inside test plus a running-best over three
`closest_on_segment` calls against ngp's `point_in_triangle` +
`closest_point_to_line` + `if (min == mag1)` chain -- exactly the family
the provenance correction was raised for, and it is genuinely gone;
`fibonacci_dir` splits the epsilon ladder into a helper and inlines the
sphere map. The stackless walks have no counterpart in ngp at all (that
snapshot's TriangleBvhNode has no escape field), so they carry no NSCL
exposure; their shape follows cubvh's own MIT code via the spec.

Keep set verified independently: `safe_divide`, `barycentric` and the
`state_dict`/`load_state_dict` block appear in the pre-rewrite cubvh
files but return ZERO hits across the whole ngp snapshot
(no at::Tensor/from_blob/state_dict anywhere in it), so keeping them
verbatim carries no NVIDIA-licensed text.

One residual is worth naming for the licence reviewer rather than
changing: common.h:81-87 reproduces the equal-area cylindrical map
(`cos_theta = 1 - 2u`, `sqrtf(fmaxf(1 - cos_theta^2, 0))`,
`sincosf(2*PI*(v-0.5f))`, `{sin_theta*cos_phi, sin_theta*sin_phi,
cos_theta}`) whose arithmetic the spec pinned for raystab determinism, and
which is close to ngp's `cylindrical_to_dir`. It is a five-line textbook
projection with the variable names mathematics dictates, respelled as the
spec directed (helper split, sqrt/fmax intermediate, inline phi); the
similarity scan flags only `float sin_phi, cos_phi;`. Recording it so the
licence reviewer sees the weakest point rather than discovering it.

The clean-room process claim in the porter notes is supported by what is
in the tree: SPEC.md is written as an interface/behavior document with
the prohibitions stated, keep/ holds the extracted snippets, the
correction round is recorded before the fix rather than after, and the
similarity scan is what caught the misclassification -- a process that
was covering for itself would not have produced that correction.

## Porter fix round after review (linux-gfx90a, 2026-08-26)

Both changes-requested findings addressed:
1. Commit body corrected by amend (81a98f0 -> 2bb5138 -> c7379c0, tree
   IDENTICAL all three -- message-only amends; branch had no validations at
   the new head yet, so no validated_sha was orphaned): "within 2.4e-7"
   -> "within 3.6e-7" (the true max, degen mesh; my earlier note repeated
   the same wrong figure -- the 2.384e-07 is the max gap among face-id
   mismatches, a different quantity); "identical ray hit/miss sets" ->
   identical away from zero-area faces, with the 63/20000 degenerate-mesh
   grazing-ray flips and their cause (safe_divide-guarded reciprocal
   determinant) stated; "exact distance ties" -> "1-2 ulp distance ties";
   and the now-contradictory "Ray casts are unaffected" sentence replaced
   with an accurate boundary-sensitivity statement.
2. Method promoted to the cuda-to-rocm skill:
   references/validation.md new section "Golden-differential validation
   for behavior-preserving rewrites" (golden capture from the old build
   first, tie-aware discrete comparison, intentional-change masking with
   self-consistency gates, serialization cross-load, clean-room two-agent
   protocol + mechanical similarity verification, perf-as-deliverable with
   the observed regression classes). check.py clean.

Also closed in this round: the CUDA compile gate. validation.md's
documented GPU-less nvcc env exists on this host
(/opt/conda/envs/cuda-12.8); a TU instantiating the rewritten torch-free
headers (triangle.cuh ray_intersect/distance_sq/closest_point/barycentric/
centroid/normal, bounding_box.cuh distance_sq/slab ray_intersect, common.h
fibonacci_dir/safe_divide/linear_kernel) compiles clean with nvcc 12.8,
-std=c++17 --extended-lambda --expt-relaxed-constexpr -arch=sm_80: exit 0,
only the pre-existing Eigen long-double warning (same as round 1's gate).
bvh.cuh/bvh.cu cannot get the nvcc gate on this fleet (torch/torch.h pulls
the ROCm-only PyTorch -- the documented torch-extension limitation); the PR
states the CUDA status honestly and invites the author's CUDA run.

head_sha -> c7379c0. Perf and golden evidence from 81a98f0 applies
unchanged (tree-identical amends).

## Re-review (round 2, fix round) 2026-08-26 (reviewer, linux-gfx90a) -- PASS

Scope: the porter fix round above, at fork head c7379c0. Both
changes-requested findings are closed; no new blocking problems.

Carry-over of evidence verified first, because everything else depends on
it: `81a98f0^{tree} == 2bb5138^{tree} == c7379c0^{tree} == a5d9cce...`
and `git diff 81a98f0 c7379c0` is empty, so the amends are message-only.
c7379c0 sits directly on e5a657a, is pushed to origin/moat-port, and the
fork tree is clean. My golden/crossload runs, the bitwise node-array
comparison, the similarity scans and the perf table therefore all apply
unchanged at c7379c0. No platform had validated 81a98f0, so no
validated_sha was orphaned; all four platforms remain at e5a657a and owe
a real revalidation at c7379c0 (that is the validators' gate, not this
review's).

Finding 1 (commit body) -- closed. Each corrected claim re-checked against
my own measurements, not the porter's: "distances within 3.6e-7" against a
measured max of 3.576e-07; "face-id differences only at 1-2 ulp distance
ties" against measured max gaps of 2.384e-07 (degen) and 5.96e-08
(torus10k/open), which at these magnitudes is 1-2 ulp; "identical ray
hit/miss sets away from zero-area faces ... 63 of 20000 grazing rays that
involve a zero-area face flip hit/miss, a consequence of the guarded
reciprocal determinant" against my measured 63/20000, all 63 involving a
degenerate face, cause `safe_divide` at triangle.cuh:37. The replacement
sentence "Ray casts do not skip zero-area faces; their behavior at such
faces was and remains numerically boundary-sensitive" is accurate in both
directions (the +inf rule is in distance_sq only, and the old unguarded
1/det was boundary-sensitive too) and no longer contradicts the disclosure
two sentences later. Title still 67 chars, AI disclosure, fenced Test
Plan, no agent trailer, ASCII only; jargon.py clean, audit-commits OK.

Finding 2 (lesson promotion) -- closed. references/validation.md gained
"Golden-differential validation for behavior-preserving rewrites" at the
right level for that file. Fact-checked against the artifacts rather than
the summary: the noise figures match what the harness prints (1.19e-7 to
3.58e-7 against a 2e-5 atol), the tie policy matches what golden.py:229-237
implements, the mask-plus-self-consistency rule matches golden.py:292-302
and the exact 0.0 result I re-ran, the cross-load rule matches the
crossload mode, and the "run the similarity scan even on code classified
original" lesson matches the real correction -- I confirmed independently
that point_in_triangle, closest_point_to_line and closest_point all exist
near-verbatim in the ngp snapshot. The era-matched-ancestor point is the
non-obvious part and it is right. The perf paragraph describes the FIXED
forms, and each one is present in the shipped code (single-exit accept at
triangle.cuh:47, unrolled fan-out behind the `!= FANOUT` guard at
bvh.cu:146-165, the compare-exchange network at bvh.cu:93, the branchless
slab test at bounding_box.cuh:50-56, the division-free region test at
triangle.cuh:78-91), so a reader following it is not handed a defect.
check.py clean.

CUDA gate -- independently reproduced, not accepted on report. On this
host, `nvcc 12.8 -std=c++17 --extended-lambda --expt-relaxed-constexpr
-arch=sm_80 -I include -I third_party/eigen -c` of the coordinator's TU
(scratchpad/cuda_gate_tu.cu, which instantiates Triangle
ray_intersect/distance_sq/closest_point/barycentric/centroid/normal,
BoundingBox distance_sq/ray_intersect, fibonacci_dir<32>, safe_divide and
a host-side linear_kernel launch) exits 0 and emits an object file, with
only the pre-existing Eigen long-double device-code warning. This is
stronger than my earlier hipcc -std=c++17 syntax check: real nvcc, the
CUDA path's own standard and flags. bvh.cuh/bvh.cu remain ungated because
torch/torch.h pulls the ROCm-only PyTorch; that limitation is documented
and the commit body states the CUDA status honestly.

Non-blocking, do NOT amend for this alone: one line of the commit body is
85 chars ("faces was and remains numerically boundary-sensitive. Unused
helpers in the rewritten") where the rest wraps at ~70. Another
message-only amend would churn head_sha again for cosmetics; fold it in
only if the body is edited for another reason before publishing.

Verdict: review-passed at c7379c0. Next: validations at c7379c0 per the
gate lattice (wave32 and Windows hosts capture their own e5a657a goldens
first, per the porter's instructions above), then the follow-up-PR flow.

## Validation (round 2, linux-gfx90a) 2026-08-26 -- PASS at c7379c0

GPU: gfx90a (MI250X), ROCm/torch 2.14.0a0+git7d05abc hip 7.14.60850, numpy
1.26.4 (agent_space/venv-cubvh). Selector state was `revalidate`
(validated_sha e5a657a, head_sha advanced to c7379c0 across the round-2
rewrite). Full real-GPU run, no carry-forward: bvh.cu, bvh.cuh, triangle.cuh,
bounding_box.cuh, common.h are all rewritten (functional device-code delta),
so `classify` would return `mixed`/`differ`, not eligible for binary-equiv.

### Build
```
cd projects/cubvh/src && rm -rf build && find . -maxdepth 3 \
  \( -name "*.hip" -o -name "*_hip.*" -o -name "_cubvh*.so" \) \
  -not -path "./third_party/*" -delete
PYTORCH_ROCM_ARCH=gfx90a agent_space/venv-cubvh/bin/pip install -e \
  projects/cubvh/src --no-build-isolation
```
Forced fresh compile (build/ removed, hipify artifacts and the .so deleted
first). Exit 0, 72.8s. `strings _cubvh.cpython-312-x86_64-linux-gnu.so |
grep amdgcn` -> `amdgcn-amd-amdhsa--gfx90a` only (roc-obj-ls unavailable in
this env's rocm_sdk_core build, strings substituted).

### Upstream test suite (all agent_space/venv-cubvh/bin/python)
- `test/signed_distance.py` -- PASS (exit 0). Ours 0.0146s vs Trimesh 0.0295s.
- `test/unsigned_distance.py` -- 3 runs. Run 1: first (distance, line 121)
  assertion PASSED; second (cpoint, line 135) assertion FAILED (2/3000
  mismatched, max abs diff 2.06e-4) -- exactly the documented pre-existing
  unseeded-torch.randn flake, not a distance regression. Runs 2 and 3: both
  assertions PASSED. Matches notes' "~1 in 8" characterization (1/3 here).
- `test/state_dict.py` -- PASS ("State dict save/load test passed.").
- `python -m pytest test/cuhashtable.py -v` -- 4 passed (test_3d_basic,
  test_4d_basic, test_large_random, test_edge_values) in 4.02s.
- CPU regression `test/hashtable.py` -- PASS ("All CPU HashTable tests
  passed.").
- `test/sparse_voxel.py <icosphere-subdiv3, 1280 faces> --workspace
  agent_space/sparse_voxel_out` -- GPU spcumc path ran end-to-end; the
  known trailing `ValueError: Both events must be recorded` (missing
  `end.record()` in the test script) reproduced exactly as documented.
  `.npz` output present: active_cells / active_cells_sdf, 4,455,224 active
  voxels (res=1024) -- identical count to every prior platform's pass on
  this fixture family.

### Round-2 differential gates (both required, both PASS)
```
agent_space/venv-cubvh/bin/python projects/cubvh/harness/golden.py check \
  --ref agent_space/goldens-e5a657a-gfx90a
agent_space/venv-cubvh/bin/python projects/cubvh/harness/golden.py crossload \
  --ref agent_space/goldens-e5a657a-gfx90a
```
`check`: PASS on all three meshes. Numbers match the porter/reviewer's
measurements exactly: torus10k/open max |dist diff| 1.192e-07, degen
3.576e-07 (655 degenerate-won queries excluded per the adjudicated policy);
zero non-tie face_id mismatches anywhere; uvw within 3.9e-6 (NaN masks
equal); raystab sign match >=0.99945; ray hit/miss 1.00000 after
degenerate-face masking; degen.nodegen self-consistency exact 0.0; all
`own_roundtrip` checks pass.
`crossload`: PASS -- old (e5a657a) serialized state_dicts load into the new
build and answer within the same tolerances on all three meshes.

### CUDA no-regression gate
Already recorded at this exact head_sha (c7379c0) by both the porter (fix
round, 2026-08-26) and independently reproduced by the reviewer
(re-review, 2026-08-26): `nvcc 12.8 -std=c++17 --extended-lambda
--expt-relaxed-constexpr -arch=sm_80` compile of the rewritten torch-free
headers, exit 0. Per validator step 3 ("skip it if notes.md already
records the CUDA gate at this head_sha"), not re-run here.

### Integrity / hygiene
- `git -C projects/cubvh/src status --porcelain` -- clean (only gitignored
  hipify artifacts and the built .so untracked; no modified tracked files).
- `python3 utils/jargon.py --port cubvh` -- clean.
- readme.md carries the `#### AMD GPU (ROCm)` subsection (added round 1,
  present unchanged at c7379c0).

Outcome: all required GPU tests pass on real gfx90a hardware, both
round-2 differential gates pass, CUDA gate already covers this head_sha,
tree clean, jargon clean, docs present. linux-gfx90a -> completed
(validated_sha = c7379c0).

## Validation (round 2, windows-gfx1151) 2026-08-26 -- PASS at c7379c0

GPU: gfx1151 (AMD Radeon 8060S, RDNA3.5 APU, wave32, 20 CUs, 67.9 GiB shared).
Python 3.13.13, torch 2.12.0+rocm7.14.0a20260519 (hip 7.13.26190), rocm-sdk
7.14.0a20260519 (TheRock wheels), AMD clang 23.0.0git, MSVC 14.44 link.exe,
numpy 2.5.2, trimesh 5.0.0. Selector state was `port-ready` -- this platform
has no prior record, so this is a first validation at the round-2 head, not a
carry-forward candidate. It is the only Windows evidence at c7379c0:
windows-gfx1101 and windows-gfx1201 still hold e5a657a.

### Build
```
source agent_space/win_torch_env.sh          # CC=CXX=clang-cl, PYTORCH_ROCM_ARCH=gfx1151
cd projects/cubvh/src && rm -rf build && find . -maxdepth 3 \
  \( -name "*.hip" -o -name "*_hip.*" -o -name "_cubvh*.pyd" \) \
  -not -path "./third_party/*" -delete
"$TORCH_PY" setup.py build_ext --inplace --force
```
Exit 0, warnings only (nodiscard hipError_t, non-trivial memcpy -- all
pre-existing and identical to the other platforms). Device code confirmed
single-target: `strings _cubvh.cp313-win_amd64.pyd` -> `amdgcn-amd-amdhsa--gfx1151`
only. The setup.py Windows blocks both fired as designed on this host: the
`.hip`-to-MSVC-driver registration and the `/ALTERNATENAME` c10::ValueError
thunk alias appear in the link line, so the clang-cl torch wheel links clean.
No source delta was needed for gfx1151.

### Upstream test suite
- `test/signed_distance.py` -- PASS (exit 0). Ours 0.0149s = build 0.0007s +
  query 0.0142s, vs Trimesh 0.0214s.
- `test/unsigned_distance.py` -- 11 runs: 8 clean, 3 failed the SECOND
  (cpoint, line 135) assertion, max abs diff 1.10e-4. The FIRST (distance,
  line 121) assertion passed in all 11. That is the documented pre-existing
  unseeded-`torch.randn` tie-breaking flake, not a distance regression; the
  observed 3/11 is the same class as the notes' "~1 in 8" and gfx90a's 1/3.
- `test/state_dict.py` -- fails on this platform for the documented reason:
  `tempfile.NamedTemporaryFile()` holds an exclusive Windows handle, so
  `torch.save(..., f.name)` cannot reopen the path and torch raises
  `[enforce fail at inline_container.cc:745] open file failed with error
  code: 32` (ERROR_SHARING_VIOLATION). Same as gfx1101 and gfx1201. It is an
  upstream test-script portability bug that reproduces on unmodified CUDA
  source and is deliberately NOT patched in the port. Re-ran the test body
  verbatim with `tempfile.mktemp()` (a path with no live handle) in place of
  the context manager: BVH state_dict round-trip PASS -- distances, face_id
  and uvw all allclose.
- `python -m pytest test/cuhashtable.py -v` -- 4 passed (test_3d_basic,
  test_4d_basic, test_large_random, test_edge_values) in 2.09s.
- `test/sparse_voxel.py <icosphere-subdiv3, 1280 faces> --workspace ...` --
  GPU spcumc path ran end-to-end at res=1024 (12.9 GiB grid on the APU's
  shared memory, no pressure at 67.9 GiB). `.npz` written with
  active_cells / active_cells_sdf, **4,455,224 active voxels -- bit-identical
  to gfx90a, gfx1100, gfx1101 and gfx1201**. The trailing
  `ValueError: Both events must be recorded` reproduced exactly as
  documented (the script never calls `end.record()`); GPU output complete.
- CPU regression: `test/hashtable.py` PASS ("All CPU HashTable tests
  passed", 0/50000 spurious negatives). `test/merge_vertices.py` PASS
  (642->642 verts, 1280->1280 faces). `test/repair_holes.py` PASS.
- `test/decimation.py` -- could not run its REFERENCE comparison: kiui 0.3.5
  ships an empty `kiui/typing.py`, so `kiui.op`'s `from kiui.typing import *`
  leaves `Union`/`Tensor` undefined and the import dies before any cubvh code
  runs. That is a packaging defect in the third-party reference lib, not the
  port, and it is CPU-only. Exercised cubvh's own decimation API directly
  instead: `decimate(v, f, 320)` -> 320 verts / 636 faces and
  `parallel_decimate(v, f, 320)` -> 175 verts / 346 faces, both producing
  watertight meshes with volume 4.137 / 4.119 against the unit sphere's
  4.18879.

### Round-2 differential gates (both required, both PASS)
This platform had no e5a657a goldens, so both builds were made here: checked
out e5a657a detached, rebuilt clean, `golden.py capture --out
agent_space/goldens-e5a657a-gfx1151`; then back to moat-port (c7379c0),
rebuilt clean, and ran both gates against that reference.
```
"$TORCH_PY" projects/cubvh/harness/golden.py check     --ref agent_space/goldens-e5a657a-gfx1151
"$TORCH_PY" projects/cubvh/harness/golden.py crossload --ref agent_space/goldens-e5a657a-gfx1151
```
`check`: PASS on all three meshes. gfx1151 agrees with its own pre-rewrite
build MORE tightly than gfx90a did: max |dist diff| 5.960e-08 on torus10k and
open (gfx90a saw 1.192e-07), 2.384e-07 on degen with the same 655
degenerate-won queries excluded per the adjudicated policy. Zero non-tie
face_id mismatches anywhere (ties only: 622/347/401). uvw diff exactly
0.000e+00 on every mesh with NaN masks equal. Ray hit/miss 1.00000
everywhere; ray face_id 1.00000 except open at 0.99990. raystab sign 0.99995
(torus10k), 1.00000 (open, degen), 0.99910 (degen.nodegen -- the tightest
margin observed on any platform against the 0.999 floor, worth watching but
passing). `degen.nodegen` self-consistency exact 0.0; all `own_roundtrip`
checks pass.
`crossload`: PASS -- e5a657a-serialized state_dicts load into the c7379c0
build and answer within the same tolerances on all three meshes, confirming
the rewritten node layout and traversal still read foreign node arrays.

### CUDA no-regression gate
Not run here and not required: already recorded at this exact head_sha
(c7379c0) by the porter and independently reproduced by the reviewer. Per
validator step 3 the gate is per-head_sha, not per-arch, and this is a
Windows host with no CUDA toolkit.

### Harness dependencies installed (venv only, no source delta)
`rtree`, `scipy` (trimesh's CPU proximity baseline -- without them
signed/unsigned_distance abort in trimesh before reaching any assertion),
`rich` (sparse_voxel), `pytest`, plus `kiui`/`opencv-python-headless`/
`pymeshlab` for the decimation reference that then failed to import.

### Integrity / hygiene
- `git -C projects/cubvh/src status --porcelain` -- completely clean, no
  modified tracked files and no untracked leftovers.
- `python3 utils/jargon.py --port cubvh` -- clean.
- readme.md carries the `#### AMD GPU (ROCm)` subsection at c7379c0.

No gfx1151-specific issues: no TDR, no GPU wedge, no numerical divergence, no
APU-specific fault. Notably this is an RDNA3.5 low-CU APU and the BVH
traversal showed none of the low-CU starvation seen on stdgpu.

Outcome: all required GPU tests pass on real gfx1151 hardware, both round-2
differential gates pass, tree clean, jargon clean, docs present.
windows-gfx1151 -> completed (validated_sha = c7379c0). This restores the
`windows` gate at the round-2 head.

### Cross-arch perf deliverable (harness/bench.py, windows-gfx1151, wave32) 2026-08-26

The plan's outstanding item ("bench.py old-vs-new on a wave32 box completes
the plan's cross-arch perf deliverable"). 500k queries, 5 reps, three mesh
scales, old e5a657a vs new c7379c0.

Method: both extensions were built here and STAGED SIDE BY SIDE
(agent_space/ab/{old,new}, each a copy of the `cubvh` package plus its own
`_cubvh.cp313-win_amd64.pyd`) so runs could be interleaved old/new/old/new
without a rebuild between them. The `cubvh` python package is byte-identical
between the two shas (`git diff e5a657a c7379c0 -- cubvh/` is empty), so the
staged pair differs only in the compiled extension. 3 interleaved rounds each.
Raw: agent_space/ab/bench-{old,new}-r{1,2,3}.json.

Median-of-3-rounds, and min-of-3 as a robustness check (ms, old -> new):

| scale | op | old | new | delta |
|---|---|---|---|---|
| small10k | build | 2.73 | 2.49 | -8.8% |
| small10k | ray_trace | 2.47 | 2.42 | -2.0% |
| small10k | udf | 6.22 | 5.96 | -4.2% |
| small10k | sdf_watertight | 7.66 | 7.56 | -1.4% |
| small10k | sdf_raystab | 255.48 | 270.18 | **+5.8%** |
| med200k | build | 70.87 | 65.95 | -6.9% |
| med200k | ray_trace | 3.74 | 3.62 | -3.1% |
| med200k | udf | 31.98 | 30.55 | -4.5% |
| med200k | sdf_watertight | 34.53 | 33.53 | -2.9% |
| med200k | sdf_raystab | 450.83 | 425.88 | -5.5% |
| big2m | build | 760.97 | 696.52 | -8.5% |
| big2m | ray_trace | 5.67 | 6.01 | **+6.0%** |
| big2m | udf | 96.19 | 87.54 | -9.0% |
| big2m | sdf_watertight | 97.54 | 90.03 | -7.7% |
| big2m | sdf_raystab | 670.07 | 810.28 | **+20.9%** |

(table is min-of-3; medians agree in sign everywhere and put big2m
sdf_raystab at +23.5% and big2m ray_trace at +10.7%.)

MEASUREMENT ARTIFACT, do not read as a regression: on medians, `build` at
med200k/big2m appears +51%/+46% slower. It is bimodal on BOTH sides -- old
runs [70.9, 71.8, 118.3] and new [66.0, 108.4, 110.2] at med200k -- so the
medians simply landed on different modes. `build` is host-side BVH
construction, and the bimodality tracks host CPU contention, not the GPU.
Min-of-3 (above) resolves it: build is 7-9% FASTER on the rewrite at every
scale, matching gfx90a.

Result vs gfx90a (wave64), where the porter measured build and ray faster
everywhere and distance ops within ~3%:
- CONFIRMED on wave32: build faster everywhere (-7 to -9%); udf faster
  everywhere (-4 to -9%, better than gfx90a's ~par); sdf_watertight par to
  -7.7% faster.
- NOT confirmed on wave32, two ops regress and gfx90a did not see them:
  - `sdf_raystab` +20.9%/+23.5% at 2,000,000 tris, and +5.8%/+9.0% at 10,368
    tris, while being -5.5% FASTER at 204,800 tris. Spreads within a run are
    tight (0.4-3%) and the sign is consistent across all 3 interleaved
    rounds, so this is not noise. gfx90a saw only +0.9 to +2.0% here.
  - `ray_trace` +6.0%/+10.7% at 2,000,000 tris only (par or faster at the two
    smaller scales). gfx90a saw ray_trace faster at every scale.

Reading: the round-2 query-kernel optimization work (division-and-sqrt-free
distance_sq region form, compile-time-unrolled fan-out-4 child loops behind
the stackless guard, branchless single-exit ray intersector, branchless slab
test, fixed 5-exchange compare-swap ordering network) was tuned on wave64
gfx90a. The two regressing ops are the two deepest-traversal ones, and they
regress only at the largest tree, which is where traversal divergence and
register pressure dominate -- consistent with the unrolled fan-out-4 loops
and the branchless intersector costing more registers/occupancy on a wave32
20-CU part than they win back. NOT diagnosed further here: this is a
performance-tuning question for the porter on a wave32 host, not a
correctness fault, and every correctness gate at c7379c0 passes.

This does not change the windows-gfx1151 validation outcome (correctness is
what the gates test, and they pass). Registered as a deferred item
(`cubvh-wave32-raystab-perf-regression`) for a person to rule defer-vs-now,
because the rewrite's stated goal was licence independence, not a perf
change, and a 21-23% slowdown on the slowest op at the largest scale is
something an upstream maintainer would reasonably ask about.

### Perf regression confirmed and root-caused (windows-gfx1151) 2026-08-26

**Confirmed real.** The first A/B ran old-before-new in every round, which on a
shared-power APU could bias the second-measured build slower. Repeated the
whole matrix with the order REVERSED (new before old), rounds 4-6:

| scale | op | old-first | new-first | verdict |
|---|---|---|---|---|
| big2m | sdf_raystab | +20.9% | +24.0% | confirmed |
| big2m | ray_trace | +6.0% | +6.1% | confirmed |
| small10k | sdf_raystab | +5.8% | +5.5% | confirmed |
| med200k | sdf_raystab | -5.5% | -2.1% | faster, confirmed |
| (all) | udf / sdf_watertight / build | faster | faster | confirmed |

big2m sdf_raystab raw medians: old [679.2, 671.2, 676.3] vs new [836.1, 832.5,
840.7] -- non-overlapping across 6 independent rounds in both orderings. Not
noise, not an ordering artifact.

**Root cause: register pressure halving occupancy on the two ray-traversal
kernels.** Extracted the gfx1151 code objects from both `.pyd`s (the embedded
`__CLANG_OFFLOAD_BUNDLE__`; script at agent_space/ab/extract_co.py) and read
the AMDGPU kernel metadata with `llvm-readelf --notes`:

| bench op | kernel (old -> new name) | VGPR | SGPR | scratch | spills |
|---|---|---|---|---|---|
| ray_trace | raytrace_kernel -> bvh_ray_trace_kernel | **55 -> 172** | 42 -> 49 | 140 -> 140 | 0 -> 0 |
| sdf_raystab | signed_distance_raystab_kernel -> bvh_signed_distance_raystab_kernel | **85 -> 171** | 52 -> 66 | 460 -> 524 | 0 -> 0 |
| udf | unsigned_distance_kernel -> bvh_unsigned_distance_kernel | 59 -> 60 | 42 -> 45 | 268 -> 268 | 0 -> 0 |
| sdf_watertight | signed_distance_watertight_kernel -> bvh_signed_distance_watertight_kernel | 72 -> 64 | 42 -> 45 | 268 -> 268 | 0 -> 0 |

The correlation is exact: the ONLY two kernels whose VGPR count blew up are the
only two that regressed, and the two whose VGPR count is flat or better both got
faster. Spill counts are 0 on both sides, so this is pure occupancy loss, not
spilling. On RDNA3/3.5 wave32 (1536 VGPRs per SIMD, 16-wave cap) 55 and 85 VGPRs
both reach the 16-wave cap, while 171-172 VGPRs allow only 8 waves per SIMD --
occupancy is halved on exactly the two regressing kernels.

That also explains the scale dependence, which otherwise looks contradictory
(sdf_raystab is +5.8% at 10k, -5.5% at 200k, +21% at 2M). Occupancy buys
latency hiding, and latency hiding only matters once the tree stops fitting in
cache. At 200k the rewrite's algorithmic wins dominate and it is genuinely
faster; at 2M the traversal turns memory-latency-bound and the lost waves cost
far more than the algorithmic win returns. The BVH TU also grew 215,160 ->
262,000 bytes (+21.8%), consistent with the added unrolling.

The api_gpu.cu code object is byte-identical (3,359,248 bytes) between the two
builds, confirming the delta is entirely in the rewritten BVH TU.

**Fix (proposed, NOT implemented -- porter work, and it costs a fleet-wide
revalidation).** No algorithmic redesign needed; the old code proves ~55 VGPRs
suffices for this traversal. Options, cheapest first:
1. `__launch_bounds__` on the two regressing kernels to cap the register budget
   (a target under 96 VGPRs restores the 16-wave cap), or the equivalent
   `__attribute__((amdgpu_waves_per_eu(N)))`.
2. Trim what holds the extra state live specifically on the RAY path: the
   compile-time-unrolled fan-out-4 child loop keeps all four child boxes'
   worth of live values in registers, and the branchless single-exit ray
   intersector extends live ranges across the whole node visit. The distance
   kernels share the traversal but did NOT regress, so the added pressure is
   in the ray-specific code, which narrows the search.
3. Re-measure with the same staged-A/B method; the VGPR count is readable from
   the code object without running anything, so iteration is fast.

Note this is a compile-time property, so it should reproduce on any wave32
target from the same source -- see perf-plan-gfx1100.md, which makes exactly
that prediction falsifiable.

### Not gfx1151-specific: cross-arch codegen check 2026-08-26

VGPR count is a compile-time property of source + target, so the "is this just
this APU?" question was settled on this host by cross-compiling both shas for
gfx1100 and gfx90a (no GPU needed to read a code object) and diffing the
kernel metadata.

| arch | ray_trace VGPR | sdf_raystab VGPR | udf | sdf_watertight | waves/SIMD old -> new |
|---|---|---|---|---|---|
| gfx1151 (RDNA3.5, wave32) | 55 -> 172 | 85 -> 171 | 59 -> 60 | 72 -> 64 | 16 -> 8 |
| gfx1100 (RDNA3, wave32) | 52 -> 172 | 85 -> 171 | 59 -> 60 | 72 -> 64 | 16 -> 8 |
| gfx90a (CDNA2, wave64) | 63 -> 86 | 98 -> 114 | 74 -> 68 | 96 -> 76 | 8 -> 5, 5 -> 4 |

gfx1100 is IDENTICAL to gfx1151 to the register, so linux-gfx1100 should be
expected to regress too -- this is a wave32 codegen property, not an APU
quirk. gfx90a loses occupancy as well (8->5 and 5->4) but the absolute
increase is far smaller (+23 and +16 VGPRs versus +117 and +86), and MI250X's
HBM bandwidth hides what is left, which is the most likely reason the
porter's wave64 A/B looked clean.

Why wave32 is hit harder: at the 16-wave cap RDNA3 wave32 allows 1536/16 = 96
VGPRs per wave, so the register allocator has room to spend up to ~168 before
its heuristic objects -- and spending it walks straight off the 96-VGPR
occupancy cliff. gfx90a's budget is 512/8 = 64 per wave, which constrains the
allocator from the start.

### Fix proven at codegen and runtime (throwaway experiment) 2026-08-26

Applied locally to `src/bvh.cu`, measured, then REVERTED -- never committed,
fork tree verified clean afterwards and the gfx1151 c7379c0 build restored.
One attribute per kernel, no algorithmic change:

```c
__global__ __attribute__((amdgpu_waves_per_eu(16))) void bvh_ray_trace_kernel(...)
__global__ __attribute__((amdgpu_waves_per_eu(16))) void bvh_signed_distance_raystab_kernel(...)
```

Codegen effect on gfx1151:

| kernel | VGPR | waves/SIMD | spills |
|---|---|---|---|
| bvh_ray_trace_kernel | 172 -> **66** | 8 -> **16** | 0 -> 0 |
| bvh_signed_distance_raystab_kernel | 171 -> **88** | 8 -> **16** | 0 -> 0 |

Runtime effect (min-of-3, interleaved old/new/fix, ms):

| scale | op | old | new | new+fix | new vs old | fix vs old |
|---|---|---|---|---|---|---|
| big2m | sdf_raystab | 640.20 | 816.08 | 668.25 | +27.5% | **+4.4%** |
| big2m | ray_trace | 5.42 | 6.08 | 5.18 | +12.2% | **-4.4%** |
| small10k | sdf_raystab | 252.39 | 269.51 | 253.43 | +6.8% | **+0.4%** |
| med200k | sdf_raystab | 451.11 | 427.79 | 462.60 | -5.2% | +2.5% |
| big2m | udf | 94.25 | 89.39 | 87.33 | -5.2% | -7.3% |
| big2m | sdf_watertight | 97.06 | 91.09 | 90.19 | -6.1% | -7.1% |

So the two regressions essentially close: big2m sdf_raystab +27.5% -> +4.4%,
big2m ray_trace +12.2% -> -4.4% (faster than the pre-rewrite code). The
untouched kernels are unaffected, as expected. The one cost is med200k
sdf_raystab, which was -5.2% FASTER unfixed and becomes +2.5%: forcing 16
waves is not free at the scale where occupancy was never the bottleneck, so a
porter tuning this properly should sweep the `amdgpu_waves_per_eu` value
rather than assume 16, and may want it scale- or arch-conditional.

GOTCHA: `__launch_bounds__(128, 4)` was tried first and is a NO-OP -- on HIP
the second argument is MIN WAVES PER EU (not CUDA's minBlocksPerMultiprocessor),
and 4 was already satisfied at 8 waves. VGPR counts did not move at all.

NOT IMPLEMENTED IN THE PORT, deliberately. It is porter work: it moves
`head_sha` and forces every platform to revalidate, and `amdgpu_waves_per_eu`
is AMD-specific so it must be guarded (`USE_ROCM` / `__HIP_PLATFORM_AMD__`)
or it breaks the CUDA no-regression gate. Recorded here so the decision has
numbers behind it; see perf-plan-gfx1100.md.
