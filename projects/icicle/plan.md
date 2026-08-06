# icicle -- ROCm/HIP port plan (lead platform linux-gfx90a)

## Project
- Name: icicle
- Upstream: https://github.com/ingonyama-zk/icicle (ingonyama-zk/icicle)
- Default branch: main
- Planning sha: 625532a624e5aaa6e9d31a1c92587f1fcc30dc76 (2025-08-06)
- Domain: GPU-accelerated zero-knowledge crypto primitives. v3+ uses a runtime device-abstraction architecture: a frontend (`icicle_device` + per-feature dispatchers) loads/links backends that register named devices ("CPU", "CUDA", "CUDA-PQC", and -- privately -- "METAL"/"VULKAN").

## Existing AMD support
None for the portable GPU code. Findings:
- The famous MSM / NTT / ECNTT / EC + big-integer-field CUDA kernels (the PTX `madc`/`addc` carry-chain field arithmetic that ZK libs are known for) are **closed-source**. The top CMakeLists pulls `CUDA_BACKEND`, `METAL_BACKEND`, `VULKAN_BACKEND` from **private repos** at configure time ("note they are in private repos"). None of that source is in this tree, so it cannot be ported here.
- There is NO HIP backend and NO ROCm device-abstraction entry. AMD is reachable only via the closed METAL/VULKAN backends (not HIP, not in-tree).
- The ONLY open-source GPU code in the repo is `backend/cuda_pqc/` -- a self-contained, license-free CUDA backend implementing **post-quantum crypto (ML-KEM / Kyber-512/768/1024)**: SHA3/SHAKE (Keccak), CBD samplers, NTT-domain sampling, keygen/encaps/decaps. It registers the device `"CUDA-PQC"` via `REGISTER_DEVICE_API` and plugs into the same dispatcher the frontend uses.

Decision: **PROCEED, scoped to a new `hip_pqc` backend** that registers `"HIP-PQC"` alongside `"CUDA-PQC"`/`"CPU"`. This is an ENABLEMENT on the device-abstraction layer (additive, mirrors `backend/cuda_pqc/`), not a fork of upstream's closed kernels. It is a real CUDA->HIP port of GPU kernels (Keccak + samplers) with a genuine wave64 fault class, and it brings AMD GPU acceleration to icicle's PQC path where today AMD has none. Not a skip: "already supports AMD" is false for HIP and false for any in-tree GPU code.

Out of scope (cannot be done from this repo): MSM, NTT, ECNTT, EC/field arithmetic -- those are the closed CUDA backend. The PTX-carry-asm reimplementation cost that the task worried about does NOT arise here: `cuda_pqc`'s field type is `Zq` (Kyber prime q=3329, 16-bit), implemented in **pure portable C++** (`% q`, no inline asm). Verified: zero `asm`/`madc`/`addc` in `backend/cuda_pqc/`.

## Build classification
**Pure CMake -> Strategy A.** Evidence:
- Top `icicle/CMakeLists.txt`: `cmake_minimum_required(VERSION 3.18)`, `project(icicle)`, `enable`-style CUDA via the backend subproject; no `find_package(Torch)`, no `torch.utils.cpp_extension`, no setup.py/pyproject for the C++ core. Rust (`go.mod`/Cargo) and Go are thin FFI bindings over the C++ libs, not the build target.
- `backend/cuda_pqc/CMakeLists.txt`: `project(icicle_cuda_pqc_backend LANGUAGES CUDA CXX)`, `find_package(CUDAToolkit)`, builds `.cu` via nvcc into `icicle_backend_cuda_pqc`. This is the classic standalone-CMake `.cu` pattern.
- ext_type = `cmake`.

## Port strategy
**Strategy A, sibling-backend model (not in-place #ifdef of cuda_pqc).** Mirror `backend/cuda_pqc/` into a new `backend/hip_pqc/` (the amgcl "mirror the backend header" lesson at backend granularity), keeping the CUDA backend byte-for-byte. Rationale:
- The device-abstraction layer is purpose-built for this: a backend is a self-contained DSO/static-lib that registers a device string. Adding `"HIP-PQC"` is the intended extension point and keeps the NVIDIA path at zero regression risk.
- The kernels are header-heavy templates shared between the device-api TU and the ml_kem TU; an in-place HIP retarget would pepper the closed-source-adjacent tree with guards. A sibling dir keeps the diff additive and reviewable.
- Wire-in: a top-level `option(HIP_PQC_BACKEND ...)` paralleling `CUDA_PQC_BACKEND`, `enable_language(HIP)` gated on it, and `set_source_files_properties(<the 2 .cu> PROPERTIES LANGUAGE HIP)`. Arch via `${CMAKE_HIP_ARCHITECTURES}` defaulted to gfx90a only when unset (never a literal -- followers gfx1100/gfx1151 then need no CMake edit).
- Compat header `backend/hip_pqc/include/cuda_to_hip.h` aliasing the cuda* runtime symbols the device-api uses (small set, enumerated below) and the device intrinsics. Force-include it on the HIP TUs (`CMAKE_HIP_FLAGS -include`).

A correctness-first mechanical+wave64 port is the right first step; there is no CUTLASS/MMA here so no AMD-native rewrite question. The wave64 re-derivation of the warp-cooperative Keccak IS the substantive work.

## CUDA surface inventory (backend/cuda_pqc only)
Runtime API (device_api.cu, all 1:1 hip*): cudaSetDevice, cudaGetDeviceCount, cudaMalloc, cudaMallocAsync, cudaFree, cudaFreeAsync, cudaMemGetInfo, cudaMemset, cudaMemsetAsync, cudaMemcpy, cudaMemcpyAsync, cudaMemcpyKind + 3 directions, cudaDeviceSynchronize, cudaStreamCreate/Destroy/Synchronize, cudaStream_t, cudaError_t, cudaSuccess. No events, no pinned/managed memory (`supports_pinned_memory=false`), no textures/surfaces, no cuBLAS/cuFFT/cuRAND/cuSPARSE/Thrust/CUB. All map directly to HIP.
- Kernels: 3 launches, all `<<<config.batch_size, 128, 0, stream>>>` (ml_kem_keygen/encaps/decaps_kernel, templated on Kyber category). 128 threads/block = 4 warps@32 (CUDA) vs 2 waves@64 (gfx90a).
- Device intrinsics present and their ROCm status (all exist in ROCm 7.2.1 headers): `__shfl_sync`, `__ballot_sync`, `__any_sync`, `__reduce_add_sync` (warp-sync family -- REQUIRE 64-bit mask on HIP, see risks), `__ffs`, `__funnelshift_l`, `__ldcs`, `__stcs`, `__signbit`, `__syncwarp`, `ROTL64`. `__constant__` tables (RC, rot_amount). `__shared__` Keccak state indexed per-warp by `threadIdx.x / 32`.
- No inline PTX anywhere. No `__shfl` width args other than implicit-32 via the sync-mask. Field math `Zq` is portable C++.

## Risk list
1. **(HARD, port core) wave64 vs warp32 in the warp-cooperative Keccak + samplers.** This is the whole port. Two designs both hardcode a 32-lane warp:
   - `cuda_sha3_32threads.cuh`: one Keccak state spread over 25 of 32 lanes; cross-lane via per-warp `__shared__` `cs[]` based at `MAX_HASHES_PER_WRAP*25*(threadIdx.x/32)`, guarded `if(lane<25)` + `__syncwarp()`. On wave64 a 64-lane wavefront splits into two `threadIdx.x/32 in {0,1}` halves that index DISTINCT shared slots -- structurally these behave like two independent 32-lane warps, which is what we want; but `__syncwarp()` on HIP syncs all 64 lanes, and the shared array must be sized for the larger (block/32) warp count. Likely OK with a per-32-lane-subgroup treatment, must verify by KAT.
   - `cuda_sample_helpers_5threads.cuh` + `cuda_sample_utils.cuh`: packs 6 Keccak states into 32 lanes (5 lanes each, `MAX_HASHES_PER_WRAP 6 = 32/5`), drives `__ballot_sync(MASK,...)` returning a 32-bit active-mask, `__shfl_sync(MASK, ptr/j, idx)`, `active &= ~((1u<<(idx+5))-1)`, and a 28-lane `__reduce_add_sync(MASK28=0xFFFFFFF, ...)`. ALL of this assumes a 32-wide ballot/reduce. On wave64, ballot is 64-bit and `__reduce_add_sync` reduces over the FULL wavefront (HIP `wfReduce`), so MASK28 no longer means "lanes 0-27" -- the reduction folds in lanes 28-63. **Fix approach (popsift/RXMesh two-32-lane-halves lesson):** treat each wave64 wavefront as two independent 32-lane subgroups -- width-32 shuffles, ballot masked+shifted per half (`(ballot64 >> 32*((tid%warpSize)/32)) & 0xffffffff`), reduce restricted to the lane's own 32-lane half. Keep `threadIdx.x/32`-based shared slotting (already half-aware). Decision: do NOT re-derive a native 64-lane Keccak; preserve the 32-lane group algorithm exactly and make it run twice per wavefront. This matches CUDA bit-for-bit and is RDNA-correct (the upper half is empty on wave32). Validate by NIST KAT, not by inspection.
2. **(compile wall, trivial) 32-bit sync masks fail to compile on HIP.** Confirmed by trial-compile @ gfx90a: `__ballot_sync`/`__shfl_sync`/`__any_sync`/`__reduce_add_sync` with `MASK=0xffffffff`/`MASK28=0xFFFFFFF` all hit `static_assert sizeof(MaskT)==8 "mask must be a 64-bit integer"` in `amd_warp_sync_functions.h`, during the HIP HOST pass. Define a 64-bit full-wave mask under USE_HIP and re-spell the partial masks as 64-bit -- but note (risk 1) the VALUE/semantics, not just the width, must be reworked for the 5-thread reduce.
3. **(semantic) `__ballot_sync` return-type width.** `uint32_t active = __ballot_sync(...)` truncates lanes 32-63 on wave64; the `__ffs(active)` leader-scan and the `1u<<(idx+5)` clearing are 32-bit. Per the per-32-lane-subgroup approach the active mask stays 32-bit but must be the half-relevant 32 bits.
4. **(low) `__reduce_add_sync` is sm_80+ on CUDA but present in ROCm 7.2.1.** No availability gap; only the mask-semantics issue (risk 1).
5. **(low) `ROTL1` uses `__signbit(*reinterpret_cast<double*>(&x))`** -- type-pun + `__signbit`; `__signbit` exists in ROCm. Watch strict-aliasing under clang -O3; verify the rotate-by-1 result against KAT (it is a Keccak theta sub-step, easy to bit-check).
6. **(low) static `__shared__` Keccak state sizing** must use a compile-time warps-per-block upper bound (block 128 / 32 = 4 on CUDA but / wave64 = 2; size to the max so wave32 followers also fit). Per the C10_WARP_SIZE_UPPER_BOUND guidance.
7. **(build) whole-archive/static-link registration.** The backend self-registers via a static initializer; the cuda_pqc CMake uses `-Wl,--whole-archive` to retain it. Mirror exactly for hip_pqc so `REGISTER_DEVICE_API("HIP-PQC", ...)` is not dropped.
8. **(none) PTX carry-chain field arithmetic** -- N/A here (closed backend only). Recorded so the disposition is honest: the large per-field PTX reimplementation cost does NOT apply to this port.
9. **(follower) RDNA wave32**: with the two-32-lane-subgroup design the upper half is simply absent on gfx1100/gfx1151, so it should be wave32-correct unchanged; re-validate on hardware.

## File-by-file change list (all additive; CUDA tree untouched)
- NEW `backend/hip_pqc/` mirrored from `backend/cuda_pqc/`:
  - `CMakeLists.txt` -- `project(... LANGUAGES HIP CXX)` (or enable_language(HIP) gated), `find_package(hip)`, `.cu` marked LANGUAGE HIP, arch from `${CMAKE_HIP_ARCHITECTURES}`, whole-archive interface lib mirrored, `REGISTER_DEVICE_API("HIP-PQC", ...)`.
  - `src/hip_pqc_device_api.cu` -- copy of cuda_pqc_device_api.cu with cuda*->hip* via the compat header; register "HIP-PQC".
  - `src/ml_kem/*.cu/.cuh`, `include/ml_kem/**`, `include/gpu-utils/**` -- copies with the wave64 fixes localized to `cuda_sample_helpers_5threads.cuh`, `cuda_sample_utils.cuh`, `cuda_sha3_5threads.cuh`, `cuda_sha3_32threads.cuh`, `cuda_hash_consts.cuh` (MASK).
  - NEW `include/cuda_to_hip.h` compat header (runtime aliases + 64-bit wave mask + a kWarpSize per-arch constant + the two-32-lane-subgroup ballot/shfl/reduce helpers). Force-included on HIP TUs.
- `icicle/CMakeLists.txt` and `cmake/pqc.cmake` -- add `option(HIP_PQC_BACKEND ...)`, `add_subdirectory(backend/hip_pqc)` under it, append to `PQC_BACKEND_TARGETS`. Guarded so the default/CUDA build is unchanged.
- `backend/hip_pqc/tests/` -- mirror cuda_pqc tests (or just reuse the frontend `tests/test_pqc_api.cpp` pointed at "HIP-PQC"); link hip_pqc.

Note for porter: confirm whether `backend/hip_pqc` can reuse the cuda_pqc tree via shared sources + a per-backend compat-include instead of a full copy, to minimize duplication. A full mirror is the safe default; dedup is a nice-to-have if include paths allow it without touching the cuda_pqc dir.

## Build commands (gfx90a)
HIP toolchain is ROCm 7.2.1 (`/opt/rocm`). From `projects/icicle/src/icicle`:
```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCPU_BACKEND=ON -DHIP_PQC_BACKEND=ON -DCUDA_PQC_BACKEND=OFF \
  -DPQC=ON -DBUILD_TESTS=ON \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a
cmake --build build -j
```
(DISABLE_ALL_FEATURES is implicit for a PQC-only build; do NOT set CURVE/FIELD. Followers: re-run with `-DCMAKE_HIP_ARCHITECTURES=gfx1100` / `gfx1151`, no source change.)
Optional CPU-only compile smoketest in `rocm/dev-ubuntu-24.04:7.2.4-complete` (compile proof only, never the gate).

## Test plan (real GPU gate)
The validatable slice is the ML-KEM correctness suite -- run on a real gfx90a GCD:
- Primary GPU gate: the `cuda_pqc` gtests rebuilt against hip_pqc (ml_kem keygen/encaps/decaps + kyber + sha3 + pke), which check against the bundled NIST KAT vectors in `backend/cuda_pqc/tests/ml_kem/test_data/{ml_kem_512,768,1024}_data`. Bit-exact KAT match IS the correctness proof for the wave64 Keccak/sampler rework. Run via `ctest` (per-case isolation via `gtest_discover_tests`, per the FAISS one-process-per-case lesson), serially on a single GCD (`ctest` not `-jN`, MPPI lesson).
- Secondary: frontend `tests/test_pqc_api.cpp` (`PqcTest.MLkemSharedSecretConsistencyTest` over Kyber512/768/1024, batch 4096) with `icicle_set_device("HIP-PQC")` -- exercises the full device-abstraction dispatch + a large batch (encaps/decaps shared-secret round-trip equality).
- Cross-check available: build "CPU" backend too and compare HIP-PQC output to the CPU backend on identical entropy (the device abstraction makes this a device-string swap).
- Non-GPU regression set (must not break): the CPU-backend tests and the frontend build. The closed CUDA/curve/field tests cannot be built here (no closed backend) and are out of scope -- they are not a regression surface for an additive HIP-PQC backend.

## Open questions
- None blocking. Confirm at porting time whether hip_pqc tests can share the cuda_pqc test sources/data via include paths (dedup) vs a full mirror. Confirm the two-32-lane-subgroup rework passes Kyber-1024 KATs (the 3-bit CBD `samplePolyCBD_3_5threads` path with `__funnelshift_l` and the `0xFFFFFF` 24-lane shuffle is the trickiest; KAT is decisive).
