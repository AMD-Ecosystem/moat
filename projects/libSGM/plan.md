# libSGM -- ROCm/HIP Port Plan (lead: linux-gfx90a)

## Project
- Name: libSGM
- Upstream: https://github.com/fixstars/libSGM (fixstars/libSGM)
- Default branch: master (status.json/upstream.json say "main"; the actual default branch on the clone is `master` -- the fork's moat-port should branch from `master`)
- Clone HEAD: e4c669b ("Merge pull request #80 from EarlMilktea/fix-cmake"), version 3.1.0
- Domain: Semi-Global Matching (SGM) stereo disparity. Pipeline: census transform -> path cost aggregation (4 or 8 directions) -> winner-takes-all (WTA) + uniqueness + subpixel -> median filter -> left/right consistency check -> disparity-range correction.

## Existing AMD support
None. Pure CUDA: `find_package(CUDAToolkit)`, `LANGUAGES CXX CUDA`, `<cuda_runtime.h>`, `cudaMalloc/Free/Memcpy/Memset`, `cudaStream*`. No HIP path, no OpenCL/Vulkan/SYCL alternative, no stale ROCm branch or PR. Grep for hip/rocm/amd across the repo returns nothing.
Decision: **fresh CUDA->HIP port adds value; proceed.** This is a correctness-first mechanical port (Strategy A). No CUTLASS/CuTe/wgmma anywhere (confirmed), so no AMD-native rewrite is warranted; the kernels are hand-written census/DP/WTA code. A later perf pass could revisit wave64 occupancy but is out of scope for the lead bringup.

## Build classification: **cmake** (Strategy A)
Evidence:
- Root `CMakeLists.txt`: `project(libSGM)`, `option(...)`, `CMAKE_CUDA_ARCHITECTURES`, `add_subdirectory(src)`. No torch.
- `src/CMakeLists.txt:7` `project(${PROJECT_NAME} LANGUAGES CXX CUDA)`, `:11` `find_package(CUDAToolkit REQUIRED)`, `:31` `target_link_libraries(... CUDA::cudart ...)`, builds `add_library(sgm STATIC|SHARED)`. No `find_package(Torch)`, no `CUDAExtension`, no setup.py / pyproject.toml anywhere.
ext_type = `cmake` (already set in upstream.json/status.json; keep).

## Port strategy: A (compat header + `enable_language(HIP)` + `.cu` marked LANGUAGE HIP)
Colmap model, minimal footprint. The CUDA runtime surface is tiny (cudaMalloc/Free/Memcpy/Memset/Stream*/GetLastError/GetErrorString/cudaSuccess + the `<<<>>>` launch syntax, which hipcc accepts). The hard work is NOT symbol renaming; it is two device-code semantic classes: the **wave64 warp model** and **missing SIMD video intrinsics**. Concretely:
1. Add `src/cuda_to_hip.h` compat header: on `USE_HIP`/`__HIP_PLATFORM_AMD__` include `<hip/hip_runtime.h>` and alias the handful of `cuda*` runtime symbols used in `device_allocator.cpp`/`device_image.cpp`; else include `<cuda_runtime.h>`. (The `.cu` already include `<cuda_runtime.h>` directly -- include the compat header there instead, or force-include it via `CMAKE_HIP_FLAGS -include`.)
2. CMake: add `option(USE_HIP ... OFF)`; under it `enable_language(HIP)`, set `CMAKE_HIP_ARCHITECTURES` from the cache var defaulting to `gfx90a` ONLY when unset (never a literal, per guide), `set_source_files_properties(<the .cu> PROPERTIES LANGUAGE HIP)`, and set the target's `HIP_ARCHITECTURES` from `${CMAKE_HIP_ARCHITECTURES}`. Gate `find_package(CUDAToolkit)` / `CUDA::cudart` off under HIP (link `hip::device` or nothing -- the HIP runtime is implicit). Do the same in `test/CMakeLists.txt` (it also declares `LANGUAGES CXX CUDA` and `find_package(CUDAToolkit)`).
3. The central change is a **warp-size abstraction** (see Risks). Replace the file-scope `WARP_SIZE = 32u` constant's *roles* carefully -- it is overloaded for both block sizing and shuffle-group width.

## CUDA surface inventory
Kernels (all `__global__`, templated on census width 32/64-bit and MAX_DISPARITY in {64,128,256}):
- `census_transform.cu`: `census_transform_kernel`, `symmetric_census_kernel`. Shared-mem sliding window, neighbor compares, `(f<<1)|(a>b)`. No warp intrinsics. BLOCK_SIZE=128 literal. Edge reads are index-guarded (`0<=x&&x<width`). Low risk.
- `cost_aggregation.cu`: vertical (up2down/down2up), horizontal (left2right/right2left), oblique (4 diagonals) path kernels + `DynamicProgramming<DP_BLOCK_SIZE,SUBGROUP_SIZE>::update`. **Heavily warp-cooperative**: `SHFL_UP/SHFL_DOWN` along the scanline DP, `subgroup_min<SUBGROUP_SIZE>` via `__shfl_xor`, per-subgroup `shfl_mask`. Uses `__ldg`, `__popc`/`__popcll`. This is the wave64 epicenter.
- `winner_takes_all.cu`: `winner_takes_all_kernel`. Full-warp `subgroup_min<WARP_SIZE>` and `subgroup_and<WARP_SIZE>`, `__shfl`/`__shfl_sync` for the right-disparity gather, `__syncwarp()` (CUDA>=9000 path) / `__threadfence_block` fallback. `REDUCTION_PER_THREAD = MAX_DISPARITY/WARP_SIZE` -- the per-lane data layout is a direct function of warp width. Second wave64 epicenter.
- `median_filter.cu`: 3x3 median selection network. Scalar path + vectorized 2x/4x path using **`__vcmpgtu2/4`, `__vminu2/4`, `__vmaxu2/4`** SIMD video intrinsics. `__ldg`-free, reads are guarded. Uses 32-bit packed-pixel reinterpret loads.
- `check_consistency.cu`, `correct_disparity_range.cu`, `cuda_utils.cu` (16<->8 bit casts): simple 2D index-guarded kernels, no warp intrinsics, no risk.

Warp intrinsics: `__shfl`, `__shfl_up`, `__shfl_down`, `__shfl_xor` (CUDA<9000 path) and `_sync` variants (CUDA>=9000 path), `__syncwarp`. **Note: HIP does NOT define `CUDA_VERSION`, so every `#if CUDA_VERSION >= 9000` block takes the ELSE (legacy, non-sync) branch.** That is fine -- HIP provides the bare `__shfl*` (confirmed in `/opt/rocm-7.2.1/include/hip/amd_detail/amd_warp_functions.h`) and `__syncwarp()`. No `__ballot`/`__activemask`/`__popc`-on-ballot.
popcount: `__popc`/`__popcll` -- provided by HIP, fine.
Textures/surfaces: NONE. cuBLAS/cuFFT/cuRAND/cuSPARSE/Thrust/CUB/cooperative-groups: NONE. Pinned/managed memory: none (plain `cudaMalloc`). Streams/events: `cudaStreamCreate/Synchronize/Destroy` (8 streams, one per aggregation path) -- 1:1 hip equivalents.
Host runtime surface (compat-header aliases needed): `cudaMalloc`, `cudaFree`, `cudaMemcpy`, `cudaMemcpyHostToDevice`, `cudaMemcpyDeviceToHost`, `cudaMemset`, `cudaStream_t`, `cudaStreamCreate`, `cudaStreamSynchronize`, `cudaStreamDestroy`, `cudaGetLastError`, `cudaGetErrorString`, `cudaError_t`, `cudaSuccess`.

## Risk list

### R1 (CENTRAL) -- wave64 vs the WARP_SIZE=32 warp model
`src/constants.h:24` `static constexpr unsigned int WARP_SIZE = 32u;` is overloaded across TWO independent roles, and they must be split for a correct wave64 port:

  (a) **Shuffle-group / subgroup partitioning.** In cost aggregation, `SUBGROUP_SIZE = MAX_DISPARITY / DP_BLOCK_SIZE` (vertical/oblique DP_BLOCK_SIZE=16 -> SUBGROUP_SIZE in {4,8,16}; horizontal DP_BLOCK_SIZE=8 -> {8,16,32}). A warp is split into `WARP_SIZE/SUBGROUP_SIZE` groups; `group_id = threadIdx.x % WARP_SIZE / SUBGROUP_SIZE`; `shfl_mask = generate_mask<SUBGROUP_SIZE>() << (group_id*SUBGROUP_SIZE)`. The DP `SHFL_UP/DOWN(..., WARP_SIZE)` shift by 1 within the SUBGROUP boundary (the subgroup occupies contiguous lanes and the boundary lane is special-cased via `lane_id`). On AMD wave64 the *real* wavefront is 64 lanes. The cleanest correct port is to make WARP_SIZE the **true wavefront width** (64 on gfx90a, 32 on RDNA), via the guide's per-arch constant keyed on `__GFX9__`:
        `kWarpSize = 64 on __HIP_PLATFORM_AMD__ && __GFX9__, else 32`.
     With WARP_SIZE=64 the subgroup math still holds (`WARP_SIZE/SUBGROUP_SIZE` groups per 64-lane wavefront, `generate_mask` shifted by `group_id*SUBGROUP_SIZE`) AS LONG AS the shuffle width passed to `__shfl*` is the SUBGROUP_SIZE (true for the horizontal DP_BLOCK_SIZE shuffles) AND the full-warp `WARP_SIZE`-width DP shuffles are reinterpreted on a 64-lane wavefront. CAUTION: `generate_mask<SIZE>` static_asserts `SIZE<=32` and returns a 32-bit mask. On wave64 the legacy bare `__shfl*` (the branch HIP takes) IGNORE the mask argument entirely (HIP's bare `__shfl_up(var,delta,width)` takes no mask), so the 32-bit-mask limitation does not actually corrupt the shuffle -- but the `_sync` masks would be wrong on wave64 if that branch were ever taken. Keep HIP on the non-sync branch (it is, since CUDA_VERSION is undefined) and rely on `width=SUBGROUP_SIZE`-scoped shuffles; verify each subgroup stays within one 64-lane wavefront (it does: SUBGROUP_SIZE<=32 divides 64).

  (b) **WTA data layout coupling.** `winner_takes_all.cu:87` `REDUCTION_PER_THREAD = MAX_DISPARITY / WARP_SIZE`. With WARP_SIZE 32: disp 64->2, 128->4, 256->8 per lane. If WARP_SIZE becomes 64: 64->1, 128->2, 256->4, and the warp now spans 64 lanes covering the same MAX_DISPARITY -- the per-lane gather (`k0 = lane_id*REDUCTION_PER_THREAD`), the full-warp `subgroup_min<WARP_SIZE>`/`subgroup_and<WARP_SIZE>` reduction, and the `__shfl(..., d/REDUCTION_PER_THREAD, WARP_SIZE)` right-disparity gather are ALL parameterized by WARP_SIZE and remain internally consistent if WARP_SIZE == true wavefront width. The `smem_cost_sum[WARPS_PER_BLOCK][ACCUMULATION_INTERVAL][MAX_DISPARITY]` shared array and `BLOCK_SIZE=WARPS_PER_BLOCK*WARP_SIZE` also scale consistently. **This is the slice most likely to break first** -- exact-match WTA tests are the canary.

  (c) **Block-size sizing.** `BLOCK_SIZE = WARP_SIZE*8` (cost agg vertical/oblique), `WARP_SIZE*WARPS_PER_BLOCK` (horizontal, WTA). With WARP_SIZE=64 these blocks DOUBLE (512->1024 etc.). 1024 is the gfx90a max block; `WARP_SIZE*8 = 512` vertical is fine, but check WTA `WARPS_PER_BLOCK=8 -> 8*64=512` (fine) and any block that would exceed 1024. Verify launch bounds and occupancy; if a doubled block overflows, cap WARPS_PER_BLOCK on HIP.

  Decision: make WARP_SIZE the **true per-arch wavefront width** (64 on gfx90a) rather than forcing 32-lane emulation. This keeps all the SUBGROUP_SIZE/REDUCTION_PER_THREAD/mask math internally consistent and is the lowest-risk correct port for CDNA. The follower RDNA arches (gfx1100/gfx1151) then naturally get WARP_SIZE=32, matching the original CUDA layout. **Validate with the bit-exact unit tests** (cost_aggregation_test, winner_takes_all_test, integration_test) which compare GPU output against a warp-size-agnostic CPU reference -- they will catch any subgroup/layout mismatch as a hard mismatch.
  Open sub-risk: confirm the horizontal kernel's `SUBGROUP_SIZE` can reach 32 (DP_BLOCK_SIZE=8, disp=256 -> SUBGROUP_SIZE=32) and that a 32-wide shuffle on a 64-lane wavefront partitions cleanly into two independent 32-lane subgroups (it must, and `group_id` indexes them). Trace this case explicitly during the port.

### R2 -- missing SIMD video intrinsics (median_filter.cu)
`__vcmpgtu2`, `__vcmpgtu4`, `__vminu2`, `__vmaxu2`, `__vminu4`, `__vmaxu4` are used in the vectorized 2x/4x median path. **Confirmed ABSENT from ROCm 7.2.1 HIP headers** (grep of /opt/rocm-7.2.1/include returns none outside CK/cub, which are unrelated). These will not compile under HIP. Fix: provide `USE_HIP`-guarded software emulations (byte/halfword-wise unsigned min/max/compare-greater over a packed uint32) in the compat header or a small device helper, matching the exact CUDA semantics (per-2-byte / per-4-byte lanes, `__vcmpgtu` returns 0xFFFF/0xFF per-lane on greater-than). The median selection network result must be bit-identical to the scalar path (the test compares against a scalar CPU median). Emulation is straightforward and correctness is checkable by the median_filter_test exact compare.

### R3 -- `CUDA_VERSION` undefined under HIP routes to legacy intrinsics
Every `#if CUDA_VERSION >= 9000` (cost_aggregation SHFL macros, device_utility subgroup_min/and, winner_takes_all `__syncwarp` vs `__threadfence_block` and `__shfl_sync` vs `__shfl`) takes the ELSE branch on HIP. This is acceptable (HIP has bare `__shfl*` and `__syncwarp`), BUT the WTA fallback uses `__threadfence_block()` instead of `__syncwarp()` for the inter-lane smem handoff. On wave64 a `__threadfence_block` is a weaker guarantee than the warp-sync the CUDA>=9000 path uses; the inter-lane read of `smem_cost_sum` after the accumulation store may race. Mitigation: on HIP, prefer the `__syncwarp()` path (HIP provides `__syncwarp`), i.e. treat HIP like CUDA>=9000 for the sync choice (define a local `SGM_HAS_SYNCWARP` that is true on HIP). Verify determinism with two identical runs of winner_takes_all_test / integration_test.

### R4 -- `__ldg` on HIP
Used in cost_aggregation (`load_census_with_check`, left/right loads). HIP provides `__ldg` (`hip/amd_detail/hip_ldg.h`). No action; just confirm it is in scope via hip_runtime.

### R5 -- OOB neighbor reads
Census/median/consistency kernels read neighbors but are index-guarded (`0<=x&&x<width`, `x>=RADIUS && x<w-RADIUS`, `load_census_with_check`). The median 4x/2x vectorized path does 32-bit reads at `x_4-4`/`x_4+4` but guards `x_4>=4 && x_4+7<w`. Low risk; AMD's stricter OOB fault is covered by the existing guards. Spot-check the vectorized median border handling against the test (which uses non-multiple widths via the v4/v2 split).

### R6 -- block size > 1024 on wave64 (sub-risk of R1c). Audit every `BLOCK_SIZE` after WARP_SIZE=64: vertical/oblique `WARP_SIZE*8`=512 OK; WTA/horizontal `WARP_SIZE*WARPS_PER_BLOCK` (8 and 4) = 512/256 OK. census BLOCK_SIZE=128 literal (independent of WARP_SIZE) OK. No overflow expected, but confirm at compile (static_assert) / launch.

### R7 -- texture pitch / rule-of-five / managed-memory atomics / fsqrt / fp-contract: NOT APPLICABLE. No textures, no RAII GPU handles beyond the simple DeviceAllocator (plain malloc/free, ref-counted; check its move/copy semantics survive but it is host-side and CUDA-agnostic), no managed memory, no float sqrt in kernels, integer-only cost math (no fp-contract bit-exactness concern). The bit-exact tests are integer comparisons, so the float fault classes do not bite.

## File-by-file change list (porter)
- `src/constants.h`: replace `WARP_SIZE = 32u` with a per-arch constant (64 on `__HIP_PLATFORM_AMD__ && __GFX9__`, else 32). For host-side launch math that needs it as a constexpr, keep the compile-time constant (device pass sees `__GFX9__`); host code here only uses it inside device-compiled `.cu`, so the device-pass constant is correct. Audit that no HOST-only TU consumes WARP_SIZE as a launch param requiring the runtime value (currently the launch configs use `BLOCK_SIZE` derived from WARP_SIZE inside the `.cu`, compiled by hipcc -> device-pass constant available; OK).
- NEW `src/cuda_to_hip.h`: USE_HIP runtime-symbol aliases + include hip_runtime; CUDA passthrough otherwise. Optionally house the SIMD-video-intrinsic emulations and the `SGM_HAS_SYNCWARP` definition here.
- `src/median_filter.cu`: USE_HIP-guarded emulations for `__vcmpgtu2/4`, `__vminu2/4`, `__vmaxu2/4` (or route through compat header).
- `src/cost_aggregation.cu`, `src/winner_takes_all.cu`, `src/device_utility.h`: on HIP force the `_sync`/`__syncwarp` path (define the CUDA>=9000-equivalent guard true on HIP) OR leave on the bare-intrinsic path but switch the WTA inter-lane fence to `__syncwarp` (R3). Replace direct `#include <cuda_runtime.h>` with the compat header (or `-include` it).
- `src/device_allocator.cpp`, `src/device_image.cpp`: include compat header instead of `<cuda_runtime.h>` (these are `.cpp` compiled as CXX; on HIP they still need the hip runtime decls -- include hip_runtime via compat there too, and ensure `<cstring>`/`<cstdlib>` precede it per the gpuRIR lesson if any host memcpy/memset is used; device_image uses `cudaMemset` which is the runtime, fine).
- Root `CMakeLists.txt` + `src/CMakeLists.txt` + `test/CMakeLists.txt`: add `USE_HIP` option, `enable_language(HIP)`, configurable `CMAKE_HIP_ARCHITECTURES` (default gfx90a only when unset), mark `.cu` `LANGUAGE HIP`, gate `find_package(CUDAToolkit)`/`CUDA::cudart` under the non-HIP branch, optionally `-include cuda_to_hip.h` and `-ffp-contract=on` (harmless; integer kernels) in `CMAKE_HIP_FLAGS`. Do NOT touch `.gitlab-ci.yml` or add GHA.
- `test/CMakeLists.txt`: the `find_package(OpenCV REQUIRED)` is vestigial (no test source includes OpenCV -- confirmed by grep). Keep it if OpenCV is installed (it is, opencv4 present), to minimize diff; otherwise gate it. The googletest submodule must be inited (`git submodule update --init`).

## Build commands (gfx90a)
```
cd projects/libSGM/src
git submodule update --init   # googletest, needed for ENABLE_TESTS
cmake -S . -B build-hip \
  -DUSE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DENABLE_TESTS=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-hip -j
```
Same command for followers with `-DCMAKE_HIP_ARCHITECTURES=gfx1100` / `gfx1151` and NO source change (per the configurable-arch lesson).
CPU-only compile smoketest (optional, not a gate): `rocm/dev-ubuntu-24.04:7.2.4-complete` docker.

## Test plan -- REAL GPU GATE
The project ships a gtest suite under `test/` that, for EVERY kernel, computes a warp-size-agnostic CPU reference and compares the GPU output for **bit-exact equality** (`equals()` in `test/test_utility.h` is `==` per element, no tolerance). This is the ideal correctness gate and directly exercises the wave64 risks.
- Build `sgm-test` (above), then run on a real gfx90a GCD: `./build-hip/test/sgm-test`
- Test files / coverage:
  - `census_transform_test.cpp` -- both census variants, all input types.
  - `cost_aggregation_test.cpp` -- 18 params (SGM_32U/64U x disp {64,128,256} x min_disp {0,+16,-16}), 8-path, 320x240, exact compare per path. **Primary wave64 R1(a) gate.**
  - `winner_takes_all_test.cpp` -- **primary wave64 R1(b)/R3 gate** (per-lane layout + inter-lane sync + determinism).
  - `median_filter_test.cpp` -- **primary R2 gate** (SIMD-intrinsic emulation must be bit-exact).
  - `check_consistency_test.cpp`, `correct_disparity_range_test.cpp`, `cast_test.cpp` -- low-risk kernels.
  - `integration_test.cpp` -- full pipeline (census->aggregation->WTA->median->consistency->range) on 311x239, disp 128, 4-path, subpixel, symmetric census; end-to-end exact compare. The disparity-map correctness gate.
- Determinism check (R1/R3): run `sgm-test` twice; the pass/fail and any intermediate dumps must be identical run-to-run. A flaky WTA/aggregation failure fingerprints a wave64 subgroup or sync bug.
- Non-GPU regression set: there are no CPU-only tests to protect (the CPU "reference" funcs live inside the gtest TUs as the gold); just ensure the library still builds for a CUDA target conceptually (the `#else` paths are unchanged). No host-only test regression risk.
- Samples (`stereosgm_image`) require OpenCV + sample images and are an optional secondary visual check; not required for the gate. The gtest exact-compare suite IS the disparity-map-correctness-vs-reference gate the task asks for.

## Disposition
**Proceed with a Strategy-A mechanical ROCm/HIP port.** No skip (no existing AMD support). No AMD-native rewrite (no CUTLASS/tensor-core code; hand-written integer DP/census/WTA). The single substantive engineering item is the wave64 warp-size abstraction (R1) plus the SIMD-video-intrinsic emulation (R2); both are validated bit-exactly by the shipped gtest suite on real gfx90a.

## Open questions
- R1(a)/R1(c): does the disp=256 horizontal case (SUBGROUP_SIZE=32) partition cleanly into two 32-lane subgroups within one 64-lane wavefront, with `group_id` and `shfl_mask` indexing them correctly? Resolve by tracing + the cost_aggregation_test disp-256 params (they exist) on real gfx90a. If it breaks, the fix is to keep WARP_SIZE=64 but ensure the width-scoped shuffles use SUBGROUP_SIZE (they do) and that `subgroup_min<WARP_SIZE>` in WTA reduces across the true 64 lanes (it will with WARP_SIZE=64).
- R3: confirm whether forcing the `__syncwarp` (CUDA>=9000-equivalent) path on HIP is necessary, or the bare `__threadfence_block` fallback is already correct on wave64. Decide via the WTA determinism check.
- Default-branch mismatch: status.json/upstream.json record `main` but the clone's default is `master`. The porter should fork and base `moat-port` off `master`; flag if the recorded branch needs fixing.
