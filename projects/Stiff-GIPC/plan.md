# Stiff-GIPC: ROCm port plan

Planner run 2026-11-25 on linux-gfx90a (no GPU used). Upstream read at bb2849a7 (main tip, unchanged since 2026-08-31; checked with `git ls-remote` on 2026-11-25).

## Project

- Name: Stiff-GIPC
- Upstream: https://github.com/KemengHuang/Stiff-GIPC, default branch `main`
- Fork: https://github.com/AMD-Ecosystem/Stiff-GIPC (`main` is the mirror, at bb2849a). The port goes on `moat-port`, cut from bb2849a.
- What it is: Kemeng Huang's StiffGIPC simulator (TOG 2025). It is one CMake executable, `gipc`, with FEM soft bodies, cloth, and affine body dynamics (ABD) coupled through IPC contact. It uses a Newton solver with PCG and a MAS preconditioner, all FP64 on the GPU, and a FreeGLUT viewer. It is the same author's successor to GPU_IPC, whose HIP port merged upstream as KemengHuang/GPU_IPC#4 (fork commit 940bdae). About 42% of its lines are shared with GPU_IPC (intake measurement in notes.md).

## Existing AMD support

None. Intake found no HIP/ROCm/AMD work in upstream, its five branches, its 16 forks, AMD-Ecosystem or ROCm (notes.md, "Duplicate effort"). Rechecked 2026-11-25: upstream has one PR in any state (#1, "add vcpkg.json" by MuGdxy, open since 2025-11-18) and one closed issue. Neither is AMD-related.

Decision: port from scratch, reusing the merged GPU_IPC port's design and code where the files descend from GPU_IPC. Upstream already took that port, so the author knows its shape (opt-in `USE_HIP`, pinned 32-lane warp helpers, a HIP-only Jacobi eigensolver, `-ffp-contract=on`). This is a correctness-first mechanical port. No kernel is NVIDIA-tuned (no CUTLASS, tensor cores or inline PTX), so an AMD-native rewrite would buy nothing.

Delivery vehicle: an upstream PR to KemengHuang/Stiff-GIPC. The author merged GPU_IPC#4, and in https://github.com/KemengHuang/GPU_IPC/pull/4#issuecomment-7122416498 he described this repository while explicitly inviting a PR only for libuipc. So a PR here is plausible but was not requested. Upstream merges contributions; it does not link platform forks.

## Build classification

`cmake`, so Strategy A. Recorded with `set-ext-type Stiff-GIPC cmake`.

Evidence, from `CMakeLists.txt` at bb2849a:

- line 15: `project(StiffGIPC LANGUAGES CXX CUDA)`, plus `enable_language(CUDA)` and `find_package(CUDAToolkit REQUIRED)`.
- `add_executable(gipc)` gets every `StiffGIPC/*.cu` and `*.cpp` from `file(GLOB_RECURSE ...)`.
- `target_link_libraries(gipc PRIVATE cusparse cublas cusolver cudart_static)` and `CUDA::cudart`.
- `CUDA_SEPARABLE_COMPILATION ON`, `CUDA_RESOLVE_DEVICE_SYMBOLS ON`, `CUDA_ARCHITECTURES native`, `CUDA_EXTENSIONS ON`.
- CUDA flags: `--use_fast_math -lineinfo --ptxas-options=... --extended-lambda --expt-relaxed-constexpr --default-stream=per-thread`, plus MSVC `-Xcompiler` options.
- `add_subdirectory(MeshProcess)` builds the vendored METIS and GKlib (plain C) and `metis_partition` (plain C++), all with the host compiler and no CUDA.
- No torch and no Python build.

Hard-coded behavior worth knowing: `set(CMAKE_BUILD_TYPE "Release")` is forced, so the HIP build cannot fall into an unoptimized unset build type. On UNIX `CMAKE_CXX_FLAGS` is overwritten with `-O3`.

## Port strategy

Strategy A, in the GPU_IPC shape:

1. **Opt-in `USE_HIP` CMake option.** When it is ON, `project(... LANGUAGES CXX HIP)` and `find_package(hip REQUIRED)` are used, the `.cu` files are marked `LANGUAGE HIP`, and HIP-only properties and flags apply. The CUDA branch stays byte-for-byte what it is now, so the default build is unchanged.
2. **Shim headers for the CUDA runtime and library header names, on the HIP include path only.** Add a top-level `hip/` directory (outside `StiffGIPC/`, so the header glob does not pick it up), added with `target_include_directories(gipc BEFORE PRIVATE hip)` only under `USE_HIP`. It holds `cuda_runtime.h` (includes `<hip/hip_runtime.h>` and carries the cuda->hip name mappings; the list is the GPU_IPC one, which covers every runtime name used here, see the inventory), `cuda_runtime_api.h`, `device_launch_parameters.h`, `vector_types.h`, `cublas_v2.h` (hipBLAS plus the handful of type/function aliases), `cusparse_v2.h` (hipSPARSE likewise), `cooperative_groups.h` and `cooperative_groups/reduce.h` (`<hip/hip_cooperative_groups.h>`).
   - Why shims rather than editing includes as GPU_IPC did: about 30 files include these headers by name (16 `<cuda_runtime.h>`, 5 `<cuda_runtime_api.h>`, 5 quoted `"device_launch_parameters.h"`, 2 `<vector_types.h>`, cooperative groups once, cuBLAS/cuSPARSE once). Shims leave all of those lines byte-identical to upstream. Quoted includes still resolve to the shim, because no same-named file sits beside the includer.
   - **CUB is the exception.** Its 15 includes are in 6 files. Edit those in place to `#if <HIP guard>` `<hipcub/...hpp>` plus `namespace cub = hipcub;` `#else` the existing include, as GPU_IPC's `cuda_cub_wrappers.cu` and `mlbvh.cu` do. Mirroring the `cub/` tree as 11 shim files would add more for the maintainer to review than it saves. The skill's "prefer the specific hipCUB header over the umbrella" applies.
3. **Pinned 32-lane warp helpers** in a new `StiffGIPC/cuda_tools/cuda_warp.h` (namespace `cudatool`). The CUDA branch calls exactly the intrinsics the call sites call today. The HIP branch is GPU_IPC's `details/device_utils.inl` translation: shuffles at width 32, and ballots shifted down to the calling lane's own 32-lane half (`mask >> (__lane_id() & ~31u)`). Call sites change mechanically from `__shfl_down_sync(m, v, d)` to `cudatool::warp_shfl_down(m, v, d)` (mask first, same argument order). They must not be macros named after HIP's own functions (skill: "Do not name compat macros after the warp functions ROCm declares"). Native HIP `__*_sync` is not an option for two reasons. It `static_assert`s a 64-bit mask, so every `0xffffffff` call site fails to compile (checked in `amd_warp_sync_functions.h`, ROCm 10.1). And its default width is 64 on CDNA, which silently changes 32-lane logic.
4. **Fix the two `cuda_tools` compile-phase macros** that key on nvcc-only macros. Clang HIP defines neither `__CUDA_ARCH__` nor `__CUDACC__`; probed with the ROCm 10.1 clang, the host pass defines `__HIPCC__ __HIP__` and the device pass adds `__HIP_DEVICE_COMPILE__`.
   - `cuda_tools/cuda_def.h:13`: `#ifdef __CUDA_ARCH__` / `#define CT_GENERIC __host__ __device__`. Under HIP, `CT_GENERIC` is empty on both passes, so all 188 uses (Eigen interop, buffer views, ABD energy, `evd`) become host-only and every device caller fails. Extending the guard to the device pass alone (`|| __HIP_DEVICE_COMPILE__`) is not enough. Probed: clang checks kernel bodies in the host pass too, and fails with "call to __host__ function from __global__ function". The fix that compiles is `#if defined(__CUDA_ARCH__) || defined(__HIPCC__)`, i.e. HD on both HIP passes and unchanged for nvcc and the host compiler. Probed clean, including an HD template that calls a `__device__`-only function.
   - `cuda_tools/cuda_all.h:12`: `#ifdef __CUDACC__` gates `cuda_cub_wrappers.h`. Extend it to `|| defined(__HIPCC__)` ("compiled by the device compiler" semantics), or every `.cu` that relies on `cuda_all.h` for `cudatool::DeviceReduce` and friends loses them.
   - `cuda_tools/cuda_debug.h:11` (`#ifndef __CUDA_ARCH__`) defines the host error checker in the device pass too. That is harmless unless device code calls `checkCudaErrors`. Leave it unless the compile says otherwise.
5. **Replace the Eigen general eigensolver in device code, HIP only,** with a normalized cyclic Jacobi (GPU_IPC's `selfAdjointJacobi`, generalized to N = 6, 9, 12). It must expose the `eigenvalues()`/`eigenvectors()` accessors the callers read, so the clamping and reconstruction stay upstream's code. Probed with the host's Eigen 3.4.0 under ROCm 10.1 clang: `SelfAdjointEigenSolver<Matrix<double,9,9>>` constructed in a kernel fails with the expected host-only `applyThisOnTheLeft` / `scaleAndAddTo` / `extract` errors. `computeDirect()` on 3x3 and fixed 9x9 and 12x12 products `A*B*A^T` compile clean in device code. Sites:
   - `fem/femEnergy.cu:66` `makePDSNK<Scalar,size>` (constructor = general `compute()`), instantiated at 9 (line 1588), 6 (2098) and 12 (2426).
   - `cuda_tools/cuda_eigen.h:173` `cudatool::eigen::evd<T,N>` (general), reached from `abd_system/.../setup_abd_system_gradient_and_hessian.cu:148` `make_pd` (N = 9), which is device-reachable from line 215.
   - Not affected: `core/GIPC.cu:132` `makePDGeneral` is only instantiated at size 2 and takes the `if constexpr(size <= 3)` `computeDirect` branch. `core/GIPC.cu:160` `makePD` is never instantiated. `fem/femEnergy.cu:1697` is `computeDirect` on a 3x3. The porter confirms these at compile time; a stray instantiation shows up as the same Eigen errors.
6. **Floating point: `-ffp-contract=on`, never `-ffast-math`,** as in GPU_IPC. clang's `-ffast-math` adds finite-math-only and double-precision reassociation, which nvcc's `--use_fast_math` does not do. The CCD and intersection predicates cannot tolerate either.

Everything else (Thrust is absent, there are no textures and no GL interop) needs no source change.

## CUDA surface inventory

Measured on bb2849a. `surface.py generate` floor: 49 CUDA files; markers kernel 192, device_code 542, launch 109, warp_intrinsic 84; library hits cub 32, cublas 9, cusparse 8, cusolver 15.

### Size and layout

`StiffGIPC/`: 31 `.cu`, 10 `.cuh`, 51 `.h`, 8 `.inl`, 2 `.hpp`, 4 `.cpp`; 29.8k lines in `.cu`+`.cpp`. Largest: `core/GIPC.cu` 11368, `fem/femEnergy.cu` 2901, `solver/MASPreconditioner.cu` 2509, `collision/mlbvh.cu` 2432, `math/gpu_eigen_libs.cu` 2051, `app/gl_main.cu` 1828, `io/load_mesh.cpp` 1346, `abd_system/abd_system.cu` 857, `collision/ACCD.cu` 774, `abd_system/.../setup_abd_system_gradient_and_hessian.cu` 770. The new-since-GPU_IPC subsystems are `abd_system/`, `linear_system/` (global matrix, subsystems, preconditioners, PCG, `converter.cu`, `spmv.cu`), `cuda_tools/` and `gipc/utils/parallel_algorithm/`.

### Runtime API (all one-to-one HIP names, mapped in the `cuda_runtime.h` shim)

`cudaMemcpy` 115, `cudaDeviceSynchronize` 51, `cudaMemcpy{HostToDevice,DeviceToHost,DeviceToDevice}` 118, `cudaMemset` 38, `cudaFree` 24, `cudaEvent{Create,Record,Destroy,ElapsedTime}` and `cudaEvent_t` 84, `cudaMalloc` 17, `cudaStream_t` 5, `cudaMemcpyAsync` 4 (all in the commented-out cuSOLVER block in `mlbvh.cu`), `cudaStreamSynchronize` 2, `cudaSetDevice` 2, `cudaDeviceReset` 2, `cudaGetErrorString`/`cudaGetLastError`/`cudaError_t`/`cudaError`/`cudaSuccess`. `cudaStreamCreateWithFlags`/`cudaStreamDestroy`/`cudaStreamNonBlocking` appear only in that same commented-out block. No managed memory (the `CT_MANAGED` macro is defined and unused). No `__constant__` or `cudaMemcpyToSymbol`, no textures or surfaces, no graphics interop (the viewer copies vertices to the host), no `__launch_bounds__`, no inline asm, no `clock64`, no `__half`/bf16, no `__ldg`, no device `malloc`/`new[]`.

### Warp-level code (the main work)

| file | what | count |
|---|---|---|
| `solver/MASPreconditioner.cu` | `__ballot_sync(0xffffffffU, ...)` and nested-mask ballots, `__syncwarp(mask)`, `__shfl_down_sync`, `__shfl_sync`, plus `__popc`/`__ffs`/`__clz` on bank masks | 95 matched lines; about 60 raw `*_sync` calls |
| `core/GIPC.cu` | 32-lane tree reductions `__shfl_down_sync(0xffffffff, x, i)`, two stages (lines 4881-7810) | 46 |
| `gipc/utils/parallel_algorithm/details/fast_segmental_reduce.inl` | `__syncwarp()` with no argument (HIP has this overload; leave it) | 2 |
| `collision/mlbvh.cu` | `__clzll` on 64-bit Morton codes (fine) | 2 |

- **MAS uses `BANKSIZE 16`** (`math/eigen_data.h:17`; GPU_IPC used 32) with `DEFAULT_BLOCKSIZE 256` and `DEFAULT_WARPNUM 16`. Two lane spaces coexist. The bank-local masks (`connectMsk`, `prefixMsk`, `_LanemaskLt(laneId)` with `laneId = idx % BANKSIZE`) are pure data, independent of the hardware width, and need no change. The hardware-lane masks (`warp_mask`, `inRangeMask`, `directReductionMask`, `boundaryMask`, `mark`) come from 32-bit ballots, are indexed by `threadIdx.x & 0x1f` (e.g. line 845 `int warpId = threadIdx.x & 0x1f`), feed `_segmentInterval` (32-bit `__ffs`/`__clz`, `lane == 31`), and gate `__shfl_down_sync(mask, ..., iter)` for `iter < BANKSIZE` or `< 32`. The pinned-32 helpers make each 32-lane half of a wave64 wavefront behave exactly like a CUDA warp; on wave32 they are the identity.
- **`core/GIPC.cu` reductions.** `threadIdx.x >> 5` and `% 32` index 32-element groups, and there is no `warpSize` anywhere in the tree, so GPU_IPC's `warpSize` -> 32 fix has no counterpart here. Each kernel returns early (`if(idx >= number) return;`, and `if(threadIdx.x >= warpNum) return;` before the second stage) and then shuffles across lanes that may have exited. CUDA formally leaves that undefined too. On AMD the read returns 0 from an inactive lane, and GPU_IPC measured the same pattern exact against a CPU sum, including non-power-of-two `warpNum`. Every remaining raw reduction is a sum or a `std::max`, so a 0 is harmless exactly when the reduced quantity is non-negative. **The porter checks each max-reduction's operand sign** (most reduce `1/tempv`, `tempv` or step-size bounds). Any that can be negative gets a mask-from-count guard, as GPU_IPC's `WARP_MASK_FROM_COUNT` did, so it does not read past the live lanes. Upstream's 42e69645 already moved several reductions to `cub::BlockReduce` with `valid_items`.
- **CUB warp collectives** (logical width 32): `linear_system/utils/spmv.cu:33` `cub::WarpReduce<Float,32>::HeadSegmentedReduce`, and `fast_segmental_reduce.inl` (`WarpSize = 32` default, `fast_segmental_reduce.h:11`). hipCUB/rocPRIM support a logical width of 32 below the wave64 hardware width. **Risk in `spmv.cu`:** threads with `global_thread_id >= triplet_count` return before the warp collective, so the tail lanes of the last warp neither flag a head nor hold a value. `fast_segmental_reduce.inl` handles the same case explicitly (`is_head = 1` on invalid lanes). If wave64 or wave32 results disagree in PCG convergence, look here first. The minimal fix is to keep invalid lanes alive with `flags = 1` and a zero value, mirroring upstream's own segmental reduce.

### Libraries

| CUDA | where | ROCm | note |
|---|---|---|---|
| CUB device: `DeviceReduce::{Reduce,Sum,Max}`, `DeviceRadixSort::SortPairs`, `DeviceScan::ExclusiveSum`, `DeviceSelect::Flagged`, `DeviceRunLengthEncode::Encode`, `DeviceMergeSort::SortKeys`, `DeviceSegmentedReduce::Reduce` | `cuda_tools/cuda_cub_wrappers.h` (7 includes) | hipCUB | Radix sort keys are 64-bit hashes over the full bit range (`begin_bit` 0), so the rocPRIM `end_bit = 64, begin_bit != 0` defect does not apply, and no float keys means none of the NaN ones do either. Callers: `GIPC.cu:9801`, `mlbvh.cu:2045`, `converter.cu:202`, `unique_reduce.inl:31` (merge sort with a `__host__ __device__` lambda; clang accepts it without `--extended-lambda`). |
| CUB block/warp: `BlockReduce<double,256>`, `BlockReduce<double/double2, default_threads>`, `WarpReduce<T,32>` | `GIPC.cu` (10), `pcg_solver.cu`, `spmv.cu`, `fast_segmental_reduce.inl` | hipCUB | Each kernel uses one collective per temp-storage instance (the `GIPC_CUB_BLOCK_SUM_AND_STORE` macro, GIPC.cu:119), so the "block collective in a loop needs a barrier" hazard does not arise. The `fast_segmental_reduce` loop reuses warp storage behind `__syncwarp()`, which is correct on both widths. |
| cuBLAS, cuSPARSE | `cuda_tools/cuda_linear_system.h:146-182` (`LinearSystemContext`), headers from `cuda_dense_vector.h:6-7` | hipBLAS, hipSPARSE | Only `Create`/`SetStream`/`Destroy` of handles, owned by `GlobalLinearSystem::m_context`; no BLAS or sparse routine is called. Shim headers alias the four calls and two handle types; link `roc::hipblas roc::hipsparse`. Keeping the handles (rather than stubbing them) preserves upstream's structure. |
| cuSOLVER | linked in CMake; only use (`cusolverDnDsyevd`) commented out in `mlbvh.cu:2317-2430` | none | Do not link it under HIP. |
| cooperative groups | `MASPreconditioner.cu:19-22` includes and `using namespace cooperative_groups;` | `<hip/hip_cooperative_groups.h>` (shim) | No `cg::` call anywhere; only the namespace must exist. |
| Thrust | none in active code (upstream AGENTS.md agrees) | n/a | |

### Atomics and fences

`atomicAdd` 399 (FP64 on `double*` dominates, also float, int, `unsigned`, `unsigned long long`, via `cuda_tools/cuda_atomic.h` and directly), and `atomicOr`/`atomicCAS`/`atomicExch`/`atomicMin`/`atomicMax`/`atomicInc` 22. FP64 atomics are native on gfx90a/gfx942/gfx950 and on RDNA3/4. Keep the default (safe) FP atomics; do not add `-munsafe-fp-atomics`, which changes results on fine-grained memory and is not needed for correctness. There is one `__threadfence()`, in the bottom-up BVH refit (`mlbvh.cu:1249`), which is correct as is on HIP.

### Device memory initialization

`DeviceBuffer<T>` (`cuda_tools/cuda_buffer_view.h`) is the project-wide owner since upstream's 7842c6d memory refactor. Its `resize(n, value)` path memsets, but plain `resize`/`reserve` does not. ROCm does not hand back zeroed memory. Before claiming correctness, the porter checks every counter or accumulator buffer whose first device access is `atomicAdd`/`+=` (collision-pair counters, MAS `prefixSum`-style scratch held in global memory, Hessian triplet counts) for a preceding memset on the same stream. If a run misbehaves, use the skill's fill-pattern differential (memset 0 versus 0xFF in the allocator) rather than an audit.

## Risk list

1. **`CT_GENERIC` and the `__CUDACC__` gate** (certain compile failure; fix known and probed). This is the single largest compile blocker, since 188 uses depend on it.
2. **Raw `*_sync` intrinsics with 32-bit masks** (certain compile failure, about 110 call sites in two files). Mechanical, but the translation has to be the pinned-32 one, not HIP's native forms.
3. **wave64 lane semantics in MAS** (`BANKSIZE 16` inside 32-lane hardware groups). The helpers preserve them. It is the highest-value thing to validate on CDNA, because the MAS preconditioner runs in every scene (`initScene` sets `ipc.pcg_data.P_type = 1`).
4. **Eigen general eigensolver in device code** (certain compile failure at three instantiations plus `evd`). The Jacobi replacement must normalize by `max|a_ij|` first (skill, faults-numerics) and be checked on the host against Eigen before any GPU run.
5. **Exited-lane shuffles in GIPC.cu reductions and the early return before `WarpReduce` in `spmv.cu`.** Both are latent; see the warp section for what to check.
6. **Uninitialized device memory** (see above). GPU_IPC's open "line search wedges after about 30 frames" was never pinned to this, nor ruled out.
7. **Numerics:** `-ffp-contract=on` instead of `--use_fast_math`. Run-to-run drift is expected from FP64 `atomicAdd` ordering, so cross-run and cross-arch comparisons need a tolerance, not bit-equality.
8. **Upstream's unbounded line-search loop** (GPU_IPC's wedge: `while(checkInterset && isIntersected(...)) alpha /= 2`). If a case spins printing "intersection happened", that is upstream behavior seen on AMD, not a port fault, until an NVIDIA baseline says otherwise. Do not "fix" upstream's loop in the port.
9. **Headless GL:** batch mode still calls `glutInit`/`glutCreateWindow` and requests `GLUT_MULTISAMPLE` with 16 samples. Under Xvfb with llvmpipe that has worked for GPU_IPC; if the multisample visual is refused, the fallback is a different Xvfb depth or `-screen` setting, never a source change.
10. **Windows.** The only in-service Windows host is the gfx1151 laptop (windows-gfx1101 and windows-gfx1201 are `[offline]`). GPU_IPC's windows gate was waived by jeff on 2026-11-20 because that laptop bugchecks (CrowdStrike `csagent.sys`, HYPERVISOR_ERROR) under this code family's GPU work. Expect the same. Also expect the all-clang Windows toolchain issues: CMake 3.29+ `-fuse-ld=lld-link` (GPU_IPC's `CMAKE_HIP_USING_LINKER_DEFAULT "-fuse-ld=lld"` fix), the MSVC-only `/NODEFAULTLIB:LIBCMT` and `MSVC_RUNTIME_LIBRARY` blocks under a GNU-style clang, and vendored GKlib under clang. If GKlib breaks, guard the fix to that platform in CMake; take the dependency from vcpkg only if a guarded fix is not small.
11. **Upstream PR #1 (vcpkg.json)** touches the build manifest, not `CMakeLists.txt`. It is a low conflict risk, but re-check it at PR time.

## File-by-file change list

New:

- `hip/cuda_runtime.h`, `hip/cuda_runtime_api.h`, `hip/device_launch_parameters.h`, `hip/vector_types.h`, `hip/cublas_v2.h`, `hip/cusparse_v2.h`, `hip/cooperative_groups.h`, `hip/cooperative_groups/reduce.h`: HIP-only shim headers (strategy point 2). They must be host-includable, because `io/load_mesh.cpp`, `gipc/statistics.cpp`, `gipc/utils/timer.cpp` and `gipc/utils/simple_scene_importer.cpp` reach `cuda_runtime.h` through the host compiler (via `hip::host`). Mark them `#pragma once`, with no `#ifdef USE_HIP` needed since they are only on the HIP path.
- `StiffGIPC/cuda_tools/cuda_warp.h`: `cudatool::warp_sync`, `warp_ballot`, `warp_shfl`, `warp_shfl_down`, as an `#if` HIP / `#else` CUDA pair (strategy point 3). Include it from `cuda_tools/cuda_all.h` or directly in the two users.
- A HIP-only Jacobi eigensolver. Put it beside `evd` in `cuda_tools/cuda_eigen.h` so both `evd` and `femEnergy.cu`'s `makePDSNK` use one implementation (a small `SelfAdjointJacobi<T,N>` exposing `eigenvalues()`/`eigenvectors()`).

Modified:

- `CMakeLists.txt`: add the `USE_HIP` option and branch (`project(... CXX HIP)`, `find_package(hip)`, `LANGUAGE HIP` on `StiffGIPC/*.cu`, `HIP_SEPARABLE_COMPILATION ON` plus `-fgpu-rdc` on compile and link, since cross-TU `__device__` functions exist: `math/gpu_eigen_libs.cuh` declares 123 that `gpu_eigen_libs.cu` defines). Also add `-ffp-contract=on`, `-fgpu-default-stream=per-thread` (the HIP spelling of `--default-stream=per-thread`), the HIP standard 17, `HIP_EXTENSIONS ON`, the `hip/` include dir, `USE_HIP` compile definition, `hip::host roc::hipblas roc::hipsparse` links (no cusolver), and the Windows linker default. Leave the CUDA branch's lines untouched. `MeshProcess/` needs no change, since it is host-only and its `CUDA_*` target properties are inert.
- `StiffGIPC/cuda_tools/cuda_def.h`: the `CT_GENERIC` guard (strategy point 4).
- `StiffGIPC/cuda_tools/cuda_all.h`: the `__CUDACC__` guard (strategy point 4).
- `StiffGIPC/cuda_tools/cuda_eigen.h`: `evd` uses the Jacobi solver under HIP.
- `StiffGIPC/fem/femEnergy.cu`: `makePDSNK` uses the Jacobi solver under HIP.
- `StiffGIPC/solver/MASPreconditioner.cu`: about 60 raw `*_sync` calls -> `cudatool::warp_*`.
- `StiffGIPC/core/GIPC.cu`: 46 `__shfl_down_sync` -> `cudatool::warp_shfl_down`, plus any mask guard the operand-sign check calls for.
- CUB include blocks in `cuda_tools/cuda_cub_wrappers.h`, `gipc/utils/parallel_algorithm/details/fast_segmental_reduce.inl`, `linear_system/utils/spmv.cu`, `linear_system/solver/pcg_solver.cu`, `collision/mlbvh.cu`, and `core/GIPC.cu` (wherever `cub/block/block_reduce.cuh` is included).
- `StiffGIPC/app/gl_main.cu`: only if the "CUDA-capable GPU" message is changed, as GPU_IPC did at its maintainer's request. Optional.
- `README.md`: an "AMD GPUs (ROCm)" build subsection with the configure line and dependencies (the documentation gate), written like GPU_IPC's README addition.

Expected footprint: about 20 files, roughly +500/-150, a little larger than GPU_IPC (+480/-91).

## Build commands

Linux (any arch; `<gfx>` is the executing GPU, e.g. `gfx90a`, `gfx942`, `gfx1100`, `gfx1201`):

```bash
sudo apt install libglew-dev freeglut3-dev libeigen3-dev nlohmann-json3-dev xvfb
cmake -S . -B build -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=<gfx> -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$MAX_JOBS"
```

On a host with only the pip ROCm SDK (no `/opt/rocm`), add `-DCMAKE_PREFIX_PATH="$(rocm-sdk path --cmake)"` and `-DCMAKE_HIP_COMPILER="$(rocm-sdk path --root)/llvm/bin/clang++"`. Wrap both steps in `utils/timeit.sh Stiff-GIPC compile -- ...`. Cap `MAX_JOBS` at the physical core count.

Windows (gfx1151 laptop, TheRock ROCm, all-clang with Ninja, vcpkg deps `eigen3 freeglut glew nlohmann-json` per upstream README):

```bash
cmake -S . -B build -G Ninja -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1151 -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_HIP_COMPILER=clang++ -DCMAKE_TOOLCHAIN_FILE=<vcpkg>/scripts/buildsystems/vcpkg.cmake
cmake --build build
```

CUDA no-regression (required before review is presented, on a host with nvcc, e.g. the mi355 fleet host or `/opt/conda/envs/cuda-12.8`), before and after the port, same command:

```bash
cmake -S . -B build-cuda -DCMAKE_CUDA_ARCHITECTURES=80 -DCMAKE_BUILD_TYPE=Release
cmake --build build-cuda -j"$MAX_JOBS"
```

GPU_IPC needed `-DCMAKE_CXX_FLAGS="-include cstdint"` for an upstream missing include. If this tree hits the same thing, record that it fails identically without the port.

## Test plan

There is no test suite. Upstream deleted its only test (`tests/dynamic_memory_tests.cu`, in 42e69645 "fix global reduction outputs and remove tests") and the CTest wiring with it. Do not resurrect it into the port. Upstream's own documented non-interactive validation path is the batch mode in `app/gl_main.cu:1753-1806` (upstream AGENTS.md, "Running the Program"):

- `GIPC_CASE=1..6` selects `set_case1..6` (box pipe; soft-rigid-cloth coupling; wrecking ball; fixed cloth; twisting mat; large box pipe plus cloth (the default)).
- `GIPC_STEPS=N` hides the window, runs N `IPC_Solver` frames, and exits 0. It exits 3 if any vertex is non-finite.
- `GIPC_DUMP_STATE=<path>` writes the final vertex array as packed `double3`.

It still needs a GL context, so on Linux run it under Xvfb with software GL. The compute stays on the AMD GPU. No xdotool is needed, unlike GPU_IPC.

### Per-platform GPU validation (identical on every Linux platform)

```bash
Xvfb :77 -screen 0 1024x1024x24 & export DISPLAY=:77 LIBGL_ALWAYS_SOFTWARE=1
for c in 1 2 3 4 5 6; do
  GIPC_CASE=$c GIPC_STEPS=<S_c> GIPC_DUMP_STATE=run1_case$c.bin timeout 1800 stdbuf -o0 ./build/gipc > case$c.log 2>&1; echo "case $c rc=$?"
done
```

Pass criteria for each case:

- exit 0;
- no HIP error, and no "Non-finite vertex";
- the Newton/PCG progress lines show convergence (record the iteration counts and per-frame time);
- no endless "intersection happened" spin (risk 8).

Pick `S_c` per case so each finishes in about 2-10 minutes on the platform. Start with 100 and record the chosen value. All six cases are required: case 1 (ABD only), cases 4 and 5 (FEM only), and cases 2, 3 and 6 (coupled) exercise different subsystems (`abd_system/`, `fem/`, both `linear_system` subsystems, MAS, CCD, friction).

Reproducibility and cross-arch agreement:

- Run each case a second time (`run2_case$c.bin`) and compare the dumps: maximum vertex displacement, normalized by the scene's bounding-box diagonal.
- Compare against the other wavefront width's dumps at the same `S_c` once both exist.
- Expect run-to-run noise from FP64 atomics. The port is suspect when the cross-arch difference is orders of magnitude above the run-to-run difference.
- Write the comparison commands and numbers into notes.md, because `agent_space/` is wiped.

Before any GPU run (porter): a host check of the Jacobi solver against `Eigen::SelfAdjointEigenSolver` for N = 6, 9, 12 over 20000 random symmetric matrices each. Report the worst relative eigenvalue and reconstruction error, check PSD after clamping, and check NaN in -> NaN out. Record the numbers in notes.md.

Longer run: case 6 with `GIPC_STEPS` large enough for about 20 minutes (GPU_IPC wedged around frame 30), so the long-horizon behavior is on record.

### Gates

- **wave64:** any `linux-gfx9*` platform, e.g. this host's gfx90a, or gfx942/gfx950. Full six-case run as above. This is the gate that proves the MAS/GIPC lane translation on 64-lane wavefronts.
- **wave32:** a Linux RDNA platform (gfx1100 or gfx1201), with the same six cases. Wave32 runs the HIP helpers as the identity, so it checks the build, the shims, the Jacobi path and the libraries apart from the lane logic.
- **windows:** windows-gfx1151 (laptop), run last, after both Linux gates pass at the same head (standing rule). Build with the Windows command above, then run the same six cases natively (a desktop GL context exists, so no Xvfb is needed; `set GIPC_CASE=...`, `set GIPC_STEPS=...`). If the host bugchecks as it did for GPU_IPC, record the platform `blocked` with the bugcheck evidence and bring jeff the question (see Open questions). Only he can approve a waiver.

### Non-GPU regression set

- The CUDA build compiles and links with nvcc before and after the port, with the same warnings (see the CUDA no-regression commands under Build commands).
- `metis_partition`, METIS and GKlib build unchanged under the HIP configuration's host compiler on Linux and Windows.
- `python3 utils/jargon.py --port Stiff-GIPC` is clean before every push.

### Optional extra evidence

An NVIDIA baseline would settle "port versus upstream" for any long-horizon anomaly at once, which GPU_IPC never had. It means the six cases at the same `S_c` with `GIPC_DUMP_STATE` on a CUDA GPU (a short bounded `srun` on the shared V100 Slurm partition). It is not a gate; do it only if an anomaly needs it.

## Dependencies

- MOAT hard dependencies: none (`depends_on` stays `[]`). Every library is a system package (Eigen 3.4.0, GLEW, FreeGLUT, OpenGL, nlohmann_json; vcpkg on Windows) or ROCm itself (hipCUB/rocPRIM, hipBLAS, hipSPARSE). METIS and GKlib are vendored and built in-tree.
- Optional: none. libuipc is a separate code base (5% verbatim overlap, intake) and nothing here needs it.

## Scope

Full port, with no cut proposed. The project is one executable whose viewer is also the batch-mode host (GLUT is initialized in both modes), plus host-only mesh preprocessing that needs no change. There is no library, app or GUI layer that could be deferred without breaking the deliverable, so no scope question for jeff arises from this plan.

## Open questions

1. **Windows gate (for jeff, later, not blocking the porter):** if the gfx1151 laptop bugchecks under this project's GPU work as it did for GPU_IPC (waiver approved 2026-11-20), the windows gate can only be satisfied by a waiver, which is his decision. The port will still compile and link on Windows, which the porter records either way.
2. **Shim headers versus GPU_IPC-style edited includes:** this plan picks shims (sources byte-identical for about 30 include lines). If the reviewer or jeff prefers consistency with the merged GPU_IPC diff (`#include "cuda_to_hip.h"` in each file), switching is mechanical and changes nothing else.
3. **Message text in `gl_main.cu`:** upstream's "CUDA-capable GPU" message. GPU_IPC changed it under a HIP guard; it is optional here.
