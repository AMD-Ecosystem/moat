# libuipc port plan

Planned 2026-11-25 on linux-gfx90a (planner, no build, no GPU run). Upstream read at `spiriMirror/libuipc` main `6c288d9af54b496bc6232a0d77c3d04fe19c664a` (2026-11-26 +0800); the fork `AMD-Ecosystem/libuipc` `main` is at the same sha, and `moat-port` does not exist yet (the porter creates it from `main`).

## Project

- Name: libuipc (pyuipc on PyPI), C++20 IPC contact simulator with Python bindings.
- Upstream: https://github.com/spiriMirror/libuipc, default branch `main`. The PR goes there (intake notes: KemengHuang/libuipc is the author's personal fork).
- Fork: https://github.com/AMD-Ecosystem/libuipc, port branch `moat-port`.
- Licence Apache-2.0 (intake). The maintainer invited this PR on KemengHuang/GPU_IPC#4.

## Existing AMD support

None (intake searched upstream PRs, issues, branches, forks and AMD orgs; nothing). Decision: port from scratch. Reuse the techniques of our own merged GPU_IPC port (KemengHuang/GPU_IPC#4, same author and algorithm family), not its code: libuipc is a rewrite with its own `cuda_tool` layer.

Upstream takes PRs (the maintainer asked for this one); it does not link platform forks, so an upstream PR is the right vehicle.

Port vs rewrite: mechanical HIP port. There is no CUTLASS/CuTe/tensor-core code; the kernels are FP64 scalar and Eigen fixed-size arithmetic, CUB collectives, and BVH traversal. Those translate directly and CDNA's FP64 throughput is the point of the port. No AMD-native rewrite is planned.

## Build classification

**cmake** (recorded with `set-ext-type libuipc cmake`). Evidence:

- `CMakeLists.txt:1` `cmake_minimum_required(VERSION 3.26)`, `project(uipc ...)`, options `UIPC_WITH_CUDA_BACKEND` (ON), `UIPC_CUDA_ARCHITECTURES` ("native"), `UIPC_WITH_CUDA_LEGACY_COLLISION` (ON).
- `src/backends/cuda/CMakeLists.txt`: `enable_language(CUDA)`, `uipc_add_backend(cuda)` (a SHARED library `uipc_backend_cuda`, loaded at run time by name), `CUDA_SEPARABLE_COMPILATION ON` + `CUDA_RESOLVE_DEVICE_SYMBOLS ON` (RDC, required: `affine_body/utils.cu` device functions are called from other TUs), a generated empty `.cu` "device-link anchor", nvcc `--extended-lambda --expt-relaxed-constexpr --diag-suppress=...`.
- `src/backends/cuda/components.cmake`: 8 OBJECT libraries (`cuda_runtime_objects`, `cuda_affine_body_objects`, `cuda_collision_objects`, `cuda_collision_legacy_objects`, `cuda_contact_objects`, `cuda_fem_objects`, `cuda_linear_system_objects`, `cuda_coupling_objects`) feeding the one backend target, with an inventory check that every `.cu`/`.cpp` belongs to exactly one.
- `apps/tests/backends/cuda/CMakeLists.txt`: `enable_language(CUDA)`, test executable `backend_cuda` with RDC, links `uipc::backend::cuda`; excludes `internals/` on WIN32.
- No torch anywhere. Dependencies are vcpkg CPU libraries (manifest generated at configure by `scripts/gen_vcpkg_json.py`; `--with_cuda_backend` adds nothing). xmake is a parallel build and is out of scope by jeff's 2026-11-24 ruling (the PR body says so).

## Port strategy

**Strategy A, masquerading the existing CUDA backend.** No `hip/` directory, no renamed files, no second backend.

1. A new option `UIPC_WITH_HIP` (OFF by default; meaningful only with `UIPC_WITH_CUDA_BACKEND=ON`) makes the same `src/backends/cuda` targets build with the CMake `HIP` language: `enable_language(HIP)`, `set_source_files_properties(<the .cu files> PROPERTIES LANGUAGE HIP)` in `components.cmake`, the backend and the `backend_cuda` test. The `.cpp` files stay `CXX` (all 4 in the backend are host-only; checked). Architectures come from `CMAKE_HIP_ARCHITECTURES` (auto-detect when unset, de-duplicated, mixed gfx9 + gfx10/11/12 lists refused only if a build-time wave width is ever introduced; this plan introduces none). Never pin an arch in the committed build.
2. The backend keeps its name `cuda` (`uipc_backend_cuda`, `Engine{"cuda"}`), so every user script, test, sample and the Blender integration runs unchanged on AMD. The intended upstream framing: "the CUDA backend can now be compiled with HIP for AMD GPUs". If the maintainer wants a distinct name later, it is a rename on top; it is an open question below, not a blocker.
3. A shim include directory (e.g. `src/backends/cuda/hip_compat/`, prepended `BEFORE` on the HIP path only) supplies `cuda_runtime.h`, `cuda_runtime_api.h`, `cuda.h`, the `cub/...cuh` headers the tree includes (each mapping to the specific `hipcub/...hpp`, never the hipCUB umbrella, plus `namespace cub = hipcub;`), `cuda/atomic` and `cuda/std/functional`. `cuda_runtime.h` carries the runtime-name aliases (only the symbols used; list below) and the warp-function layer. Add `-include cuda_runtime.h` for HIP TUs because nvcc includes it implicitly. The CUDA build never sees the directory.
4. Real source edits only where semantics differ (the risk list): the `__CUDA_ARCH__` gates, ballots and default-width warp collectives, Eigen `compute()` in device code, and the few CUDA-only intrinsics. Each is guarded so the nvcc build compiles exactly what it does today; prefer a fix that is correct on both back ends (width-32 shuffles written unconditionally) over an `#ifdef` pair.
5. RDC: `-fgpu-rdc` as compile option on every HIP OBJECT library and on the backend and test targets, and as link option on the targets that device-link (CMake has no `HIP_SEPARABLE_COMPILATION`; `cuda-to-rocm` build-cmake-link.md "Relocatable device code"). Keep the generated anchor source (mark it HIP too) or drop it on the HIP path if the device link is scheduled without it; the porter decides by building.
6. nvcc-only flags (`--extended-lambda`, `--expt-relaxed-constexpr`, `--diag-suppress`, `-Xcompiler=/Zc:preprocessor`) stay under `$<COMPILE_LANGUAGE:CUDA>` and simply do not apply to HIP. Do not add `-ffast-math` (none exists today); add `-ffp-contract=on` for HIP to match nvcc's contraction (GPU_IPC lesson). The `CUDAToolkit_VERSION_MAJOR` / `${CUDAToolkit_INCLUDE_DIRS}/cccl` / `${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}` lines stay CUDA-only.
7. Large device code: if the shared backend's host image approaches the x86-64 relocation reach or link time explodes, add `--offload-compress` (strategy-a-cmake.md). Expect it to be needed for multi-arch builds; single-arch builds probably are not.

## CUDA surface inventory

Counts over `src/backends/cuda` (200 `.cu`, ~61k lines; ~106k with headers) and `apps/tests/backends/cuda` (72 sources, 152 TEST_CASEs). `surface.json` holds the machine census (297 CUDA files).

| area | what | HIP mapping / action |
|---|---|---|
| kernels | 440 `__global__`, 458 `<<<>>>` launches in 106 files, almost all `k<<<best_grid_dim(n,k), best_block_dim(k), 0, s>>>`; `best_block_dim` caches `cudaOccupancyMaxPotentialBlockSize` (launch.h:43-57) | unchanged spelling; `hipOccupancyMaxPotentialBlockSize` returns wave64-multiple sizes on CDNA (fine) |
| device/host annotation | `UIPC_GENERIC` = `__host__ __device__` only `#ifdef __CUDA_ARCH__` (type_define.h:30), 563 uses in 94 files | add `|| defined(__HIP_DEVICE_COMPILE__)`; same per-pass trick works in clang's split host/device compile |
| other `__CUDA_ARCH__` gates | debug.h:71 (`__trap`), debug.h:92 (device print format), linear_bvh.inl:241 (`__clzll`, host branch asserts false: would trap at run time on HIP), utils/step_displacement.h:31,43 (`__dmul_rn`/`__dadd_rn`, a documented bit-identity contract between line search and swept boxes), fast_segmental_reduce.h:49 (`__reduce_max_sync` on sm80+: HIP correctly takes the shuffle fallback), mas_preconditioner_engine.cu:684 (launch-bounds min blocks) | add `__HIP_DEVICE_COMPILE__` to the first four; leave fast_segmental_reduce.h:49 alone; see risk on launch bounds |
| `__CUDACC__` | type_define.h:14, cuda_tool/stream.h:31: global `arg(std::complex)` shim for Eigen | add `|| defined(__HIPCC__)` only if the HIP compile needs it |
| `CUDART_VERSION` | graph.h:98 (`>=11040` InstantiateWithFlags), :248 (`>=12000` ExecUpdate with ResultInfo), :345 (`>=12040` conditional WHILE nodes), :451 (`>=13000` 6-arg AddNode) | must stay undefined on HIP (all four evaluate 0, giving the legacy 5-arg instantiate, re-instantiate on recapture, and no WHILE path, which matches what HIP 7.16 offers). Never alias `CUDART_VERSION` to `HIP_VERSION` |
| warp intrinsics | `__shfl_down_sync` 48, `__shfl_sync` 16, `__shfl_xor_sync` 10, `__shfl_up_sync` 6, `__ballot_sync` 12, `__activemask` 2, `__syncwarp` 42, 74 literal 32-bit full masks, `__fns` 1 (spmv.cu:252), `__reduce_max_sync` (gated, see above) | project-owned width-32 functions (see risk 1); `__syncwarp(mask)` maps to a wave barrier; `__fns` replaced with a popc/ffs loop |
| CUB device algorithms | all through `cuda_tool/cub.h`: DeviceReduce Sum/Min/Max/Reduce/ArgMin/ArgMax, DeviceScan Inclusive/ExclusiveSum/Scan, DeviceSelect If/Flagged/Unique, DevicePartition If/Flagged, DeviceRadixSort SortKeys/SortPairs (every call site uses `begin_bit = 0`; integer keys), DeviceMergeSort, DeviceRunLengthEncode Encode/NonTrivialRuns | hipCUB has all of them |
| CUB block/warp collectives | BlockReduce `<Float,256>`, `<T,256>`, `<Float,32>`, `<int,128>`, `<int,512>`; BlockScan `<int,1024>`, `<int,512>`; BlockRadixSort `<uint64_t,1024,3,int>`; WarpReduce `<int/Float,32>`, `<float,16>`, **`<double>` default width** (mas_preconditioner_engine.cu:748); **WarpScan `<int>` default width** (spmv.cu:151,294, declared but unused); `ShuffleIndex<32>`, `ShuffleDown<32>` | pin the two default-width ones to 32; the rest are fine |
| libcu++ | `::cuda::std::plus<T>` (fast_segmental_reduce.h, 6 default template args; test fast_segmental_reduce_lean.cu), `::cuda::atomic_thread_fence(acquire, system)` (linear_bvh.inl:189) | either libhipcxx (`hipccl`, bundled with ROCm 10.1+) or, simpler and with no new dependency, the shim's `cuda/std/functional` provides `plus` and `cuda/atomic` maps the fence to `__atomic_thread_fence(__ATOMIC_ACQUIRE)`/`__threadfence_system()`. Prefer the shim; record the choice |
| runtime API | streams (`cudaStream_t` 445), events, `cudaMalloc/Free/Memcpy(Async)/Memset(Async)`, `cudaMallocHost/FreeHost`, `cudaHostAlloc(Mapped)` + `cudaHostGetDevicePointer` (ReadbackBuffer, buffer.h:545-606), `cudaDeviceGetAttribute` (MultiProcessorCount, CanMapHostMemory, UnifiedAddressing), `cudaStreamCreateWithPriority`, `cudaDeviceGetStreamPriorityRange`, `cudaStreamAddCallback`, `cudaGetDeviceProperties` (`name/major/minor/totalGlobalMem`), `cudaDeviceSetLimit(cudaLimitStackSize)` (tests), `cudaStreamLegacy` (tests; `hipStreamLegacy` exists in HIP 7.16) | 1:1 aliases in the shim |
| CUDA graphs | `GraphCapture` (graph.h:136-328): stream capture on a blocking stream, ThreadLocal mode, instantiate, `recapture`; used by the line searcher and fused PCG (mode 1). `GraphWhile` (graph.h:345-578): conditional WHILE nodes plus device `cudaGraphSetConditional`, compiled only at `CUDART_VERSION >= 12040`; `select_graph_mode` degrades mode 2 to 1, and failed captures fall back to plain launches (linear_fused_pcg.cu:57-69, 998-1004, 1051-1058) | capture API aliases (`hipStreamBeginCapture`, `hipGraphInstantiate`, ...); WHILE path compiled out on HIP. HIP has no conditional nodes (none in HIP 7.16 headers) |
| Eigen on device | 109 `.cu` files use fixed-size Eigen. `cuda_tool/eigen/evd.h` uses `computeDirect()` for N<=3 and `SelfAdjointEigenSolver::compute()` for N>3; nvcc accepts the latter only because `--diag-suppress=20011,20012,20014` hides host-from-device calls. N>3 device callers: `make_spd<N>` (utils/make_spd.h) from affine_body_fixed_joint.cu:179 (9), affine_body_prismatic_joint.cu:485 and `_function.h:84` (9), affine_body_revolute_joint.cu:240 (6) and :429 (12), arap_3d.cu:113 (9), hookean_spring_1d.cu:114 (6), kirchhoff_rod_bending.cu:128 (9), strain_plastic_discrete_shell_bending.cu:153 (12), soft_vertex_edge_stitch.cu:105 (9), soft_vertex_stitch.cu:123 (6), soft_vertex_triangle_stitch.cu:114 (9); matrix_utils.cu:56,62,67,79 (9, 12); tests abd_launch_shape.cu:123, warp_joint_hessian.cu:76, simplex_friction_hessian_lift.cu:131,133 | host-only in HIP device code at every size (GPU_IPC finding). See risk 3 |
| other intrinsics | `__fadd_rn`/`__fsub_rn`/`__frsqrt_rn` (svd_impl.h), `__dmul_rn`/`__dadd_rn`, `__clzll`, `__popc`, `__ffs`, `__trap`, `clock64` (tests), device `printf` (77 calls via `UIPC_KERNEL_*`) | all exist in HIP; `__trap` -> `__builtin_trap()` |
| inline PTX | none in the backend. Test only: `asm volatile("mov.u64 %0, %%globaltimer;")` + `__nanosleep` in internals/mas_preconditioner_direct_store.cu:1234-1238 (a stream-hold kernel) | HIP branch with `wall_clock64()` and `__builtin_amdgcn_s_sleep` |
| driver API | `#include <cuda.h>` in svd_impl.h:2 and toolkit include dirs; no `cu*` calls | shim `cuda.h` includes the HIP runtime |
| atomics | 61 `atomicAdd` (mostly double), 34 atomicMin/Max, 8 atomicCAS | native on CDNA and RDNA; FP64 atomics on gfx90a coarse-grained memory are fine; check `-munsafe-fp-atomics` only as a performance option |
| memory | `DeviceVar` allocates uninitialized; `DeviceVector::resize` zero-fills only the exposed range; no managed memory | zero-init is an explicit API contract, but the validator keeps the memset-0/0xFF probe from GPU_IPC in reserve |
| launch geometry | fixed 32-thread blocks for "one-warp" kernels (`spread_block_dim=32`, `JointWarpSize`, `ABDDiagInverseThreads`, `FusedPcgBlockDiagThreads`), 1024-thread blocks with `__launch_bounds__(1024)`, `__launch_bounds__(256,4/5)`, `(96,6)`, `(Wide,1)`, `(32)`; no dynamic shared memory | 1024 is within the AMD limit; 32-thread blocks are half a wave64 (correct, see risk 1; a performance note only) |
| CUDA libraries | CUB only. No Thrust, cuBLAS, cuSPARSE (only comments mention it; removed from the census floor with that reason), cuSOLVER, cuRAND, cuFFT, NCCL, CUTLASS, textures, NVRTC | -- |

## Risk list

1. **Wavefront width (the main correctness risk on wave64).** From the full audit (site table summarized):
   - Breaks on wave64 as written: every `__ballot_sync` (64-bit wave mask truncated to 32 bits, so lanes 32-63 see the lower half): `info_stackless_bvh.inl` `drainLeafQueue` (902-929) and `stacklessOtherWarp_kernel` (959-1094), `info_stackless_bvh_simplex_trajectory_filter.cu` `judge_cover_ee` (1160-1171), `judge_ee_append` (1265-1275), `judge_scatter` (1549-1575, with `__ffs` on the ballot), `fast_segmental_reduce.h` `warp_head_segments` (68-71), `cuda_device/detail/bit_operation.inl` `WARP_BALLOT` (returns `int`) used by spmv.cu:251,255; `fast_segmental_reduce.inl:76` `__shfl_sync(..., 31)` (an absolute source lane: at width 64 lanes 32-63 read lane 31); lane-0 broadcasts `__shfl_sync(base, 0)` in the BVH drain and the trajectory filter; `cub::WarpReduce<double>` default width (mas_preconditioner_engine.cu:748, whose storage and lane math assume 32); `mas_logical_group_mask` (mas_preconditioner_engine.cu:66-90) builds a 32-bit `__syncwarp` mask from `threadIdx.x & 31`, wrong for lanes 32-63.
   - Correct today only because the read pattern stays inside 32 lanes (down-trees from offset 16 read by lane 0, xor <= 16, 32-thread blocks): the BVH `calcMaxBVFromBox` reductions, `toi_min.h`, `axis_max.h`, `abd_linear_subsystem.cu` aggregation, `abd_diag_preconditioner.cu`, `fused_pcg_kernels.h`, `warp_spd.h`/`warp_joint_hessian.h`. Pin them to width 32 anyway.
   - Divergence pattern (a condition uniform per 32 lanes but not per 64, followed by warp ops): spmv.cu:456 early return and :397 break, fast_segmental_reduce.inl:53, the `info_stackless_bvh.inl` `while(true)`/`busy==0` loop, `judge_ee_append`'s `mask==0` return, `judge_scatter`'s `while(kept)`. All of them are correct once ballots return the calling lane's 32-bit half and shuffles are width 32 with sources inside that half (`ds_bpermute` clamps to the 32-lane group, faults-wavefront.md "fixed-cluster warp reduction"). Do not restructure the branches.
   - Fix shape: one set of project-owned device functions (not macros named after the intrinsics; ROCm declares real `__shfl_*_sync` templates, build-headers-macros.md) for ballot (shifted to the lane's half on wave64, identity on wave32), `shfl/shfl_up/shfl_down/shfl_xor` with width 32, and a wave barrier for `__syncwarp`. Where CUDA's own spelling can carry an explicit width 32 on both back ends, prefer that over an `#ifdef` pair. Wave size in device code from `__GFX8__/__GFX9__` guards, never a CMake define. Pin `cub::WarpReduce<double>` and `cub::WarpScan<int>` to 32.
   - Pre-existing on CUDA too, not ours to change but worth a note: spmv.cu `Spmv_rbk_sym_spmv_kernel` WarpReduce inside a grid-stride loop with a non-uniform tail; mas_preconditioner_engine.cu early returns before the warp reduce at 703/714. Note in notes.md if the validator ever sees them; do not fix in the port unless a test fails on AMD because of them.
   - Canary tests: `fast_segmental_reduce_lean.cu` (warp_head_segments kernel), `ccd_launch_shape.cu` ("same bits with one-warp blocks"), `internals/abd_diag_inverse.cu`, `internals/mas_preconditioner_direct_store.cu`, `warp_joint_hessian.cu`, the BVH tests. They must pass on real wave64 (CDNA); `-mwavefrontsize64` on RDNA is useless for whole-app evidence because rocPRIM keys its wave size on the arch (GPU_IPC finding).
2. **`__CUDA_ARCH__`-gated code silently takes host branches on HIP.** `UIPC_GENERIC` first (compile failure everywhere, obvious), then the quiet ones: `common_upper_bits` would assert/return -1 inside the LBVH build (wrong trees), `step_displacement` would lose its `_rn` no-FMA guarantee (the line search's trajectory reuse relies on bit identity; with HIP's default `-ffp-contract=fast` the plain form can fuse). Grep every `__CUDA_ARCH__` and `__CUDACC__` after the edit; the list above is complete at 6c288d9.
3. **Eigen `SelfAdjointEigenSolver::compute()` in device code** (N = 6, 9, 12). On HIP it does not compile in device code at any size (only `computeDirect()` for 2x2/3x3 does). nvcc builds it today under `--diag-suppress`, so the CUDA path must stay on upstream's Eigen call. Fix: in `evd.h`, route N>3 to a device-safe symmetric eigensolver on the HIP device pass only (`__HIP_DEVICE_COMPILE__`), leaving the CUDA path byte-identical. Do not reuse the existing `evd_jacobi` as-is: it is classical Jacobi with an absolute `1e-6` off-diagonal tolerance and no iteration cap (`while(calc_sumDiagOff > 1e-6)`), which may never terminate on badly scaled Hessians. Use a cyclic Jacobi with a relative convergence test, max-coefficient pre-scaling (the GPU_IPC review found the unscaled form overflowing to `inf <= inf` above ~1e154) and a hard sweep cap, as the GPU_IPC port did. Validate it on the host against Eigen over random and clustered spectra for N = 6, 9, 12, and through the existing tests `fixed_size_evd.cu`, `warp_joint_hessian.cu`, `simplex_friction_hessian_lift.cu`, `abd_launch_shape.cu`. The reconstruction `V * D * V^T` on fixed-size matrices compiles in HIP device code (GPU_IPC correction), so `make_spd` itself needs no change. Expect other host-only calls the nvcc `--diag-suppress` list hid; read clang's `note: called by` chain to find the one call site each time.
4. **CUDA graph capture semantics.** The design relies on a launch on the legacy default stream invalidating an active capture (graph.h comments :52-58, :71-78; `FEMLinearSubsystem` `LegacyStreamCheck` at fem_linear_subsystem.cu:540-560 uses a Relaxed-mode capture as a detector). HIP refuses null-stream work against a capturing blocking stream with `hipErrorStreamCaptureImplicit`, which should give the same signal, but this is a runtime behaviour to prove with `cuda_tool_graph.cu`, `graph_capture.cu`, `internals/fused_pcg_*` and `internals/fused_pcg_duplicate_preconditioner.cu`. If a capture invalidates where CUDA's does not, the code falls back to plain launches: correct but slower; record it rather than forcing graphs. `GraphWhile` (mode 2) is compiled out; the `fused_pcg_while_publication.cu` test must still compile and report the unsupported path, not fail.
5. **hipCUB temp-storage contract.** `run_with_temp_storage` (cub.h:76-95) calls first with the cached workspace and expects `cudaErrorInvalidValue` when it is too small, then queries. rocPRIM must return `hipErrorInvalidValue` there and must not launch; prove with `cuda_tool_cub.cu` and a test that grows a workload across calls. rocPRIM asks for at least 4 bytes where CUB asks 1 (libraries.md); the wrapper sizes from the query, so that is fine.
6. **`__launch_bounds__` second argument.** HIP's interpretation differs from CUDA's min-blocks-per-SM; `(256,5)`, `(96,6)`, `(Wide,1)` may force register caps that spill heavily (FP64, 9x9/12x12 Eigen temporaries) or fail to compile. If the compile or performance shows it, guard the second argument on the HIP path; record measurements, not guesses.
7. **RDC device link and the shared backend.** `-fgpu-rdc` with 8 OBJECT libraries into one SHARED target, plus a test executable that links that DSO and also has its own device code. Device symbols do not cross the DSO boundary on either back end, so the test's `.cu` files must not need backend device symbols (true on CUDA today since the CUDA build links the same way). On Windows the backend DLL exports only module entry points, which is why upstream already drops `internals/` there.
8. **Mapped pinned readbacks.** `ReadbackBuffer` lets kernels write `cudaHostAllocMapped` memory directly when `CanMapHostMemory && UnifiedAddressing`. `hipHostMalloc(Mapped)` is coherent by default; check `internals/readback_buffer.cu` and `cuda_tool_readback_buffer.cu` on gfx90a, and do not change flags unless a test shows stale data.
9. **Stack size.** Tests raise `cudaLimitStackSize` to 32 KB for 12x12 work (fixed_size_evd.cu:270, ipc_simplex_normal_contact_assembly_launch.cu:347). HIP scratch is allocated per lane; 9x9/12x12 FP64 temporaries will spill to scratch (GPU_IPC measured 2304 B per thread for two 12x12 arrays). Correctness first; note occupancy if slow.
10. **Bitwise-identity tests.** Many backend tests assert identical bits between two code paths on the same device ("keeps the bits of CUB", "warp inverse bitwise identical", "same bits with one-warp blocks"). Those compare AMD against AMD, so they stay meaningful; a failure means the two paths differ on AMD (often a contraction or reduction-order difference), not that AMD differs from NVIDIA. `-ffp-contract=on` matters here.
11. **Numerics.** FP64 throughout; no fast-math. `svd_impl.h` uses `__frsqrt_rn` (exists in HIP). Watch `-ffp-contract`: CUDA contracts within an expression; match that.
12. **Line search and CCD termination.** Checked: unlike GPU_IPC, both backtracking loops are bounded by `line_search/max_iter` (default 8; advance_ipc.cu:679-751, advance_al.cu:382-400), the additive CCD loops get `max_iter = 1000` from every caller (ccd.inl:231-440), BVH traversal is capped at `2 * numObjs`. A hang on AMD is therefore not the GPU_IPC upstream wedge; treat it as a port defect until shown otherwise.
13. **Windows.** Upstream's Windows build is MSVC + vcpkg (`x64-windows`); HIP needs clang. Use the GNU-driver `clang`/`clang++` for C, CXX and HIP (windows.md: mixing cl.exe CXX with HIP fails "mixes Clang and MSVC"; clang-cl HIP with `-fgpu-rdc` hits the `/Fo` bundler bug and needs the vendored `hip_link_win.py`), vcpkg deps with the dynamic CRT matching, `MSVC_RUNTIME_LIBRARY` handling for HIP (windows.md), `/Zc:preprocessor` and `/FS` stay MSVC-only. Any workaround goes in the project's CMake guarded to Windows+HIP, never README-only. Upstream excludes `internals/` tests on WIN32; keep that.
14. **Python packaging.** `pyuipc` is host-only and loads the backend by name, so it needs no code change. `pyproject.toml` passes `UIPC_CUDA_ARCHITECTURES` to scikit-build; the documented ROCm wheel build becomes `pip install . -Ccmake.define.UIPC_WITH_HIP=ON -Ccmake.define.CMAKE_HIP_ARCHITECTURES=<gfx>` (verify on the validator). `src/pybind/pyuipc/CMakeLists.txt:24-41` records CUDA toolkit metadata for `uipc doctor`; in HIP mode report the HIP version instead of an empty CUDA version (small CMake change). The wheel audit script and the cibuildwheel CUDA pipeline are untouched (upstream release CI; MOAT adds no workflows). The intake noted `python/LICENSE` is GPL-3.0 text; jeff ruled one line in the PR body only if the port touches the Python packaging.

## File-by-file change list

Expected; the porter adjusts by building.

- `CMakeLists.txt`: `option(UIPC_WITH_HIP ...)`, print it in `uipc_show_options` (`cmake/uipc_utils.cmake`).
- `src/backends/cuda/CMakeLists.txt`: HIP branch: `enable_language(HIP)`, de-duplicate `CMAKE_HIP_ARCHITECTURES`, `find_package(hip)`/`hipcub` (from `CMAKE_PREFIX_PATH`, never `/opt/rocm/include`), link `hip::host` and `hip::hipcub`, shim include dir `BEFORE`, `-include cuda_runtime.h`, `-fgpu-rdc` compile and link, `-ffp-contract=on`, `HIP_ARCHITECTURES` from `CMAKE_HIP_ARCHITECTURES`; the CUDA-only lines move inside the else-branch unchanged.
- `src/backends/cuda/components.cmake`: `uipc_add_cuda_component` marks its `.cu` files `LANGUAGE HIP`, uses HIP options/properties under `UIPC_WITH_HIP`.
- `cmake/uipc_utils.cmake`: `uipc_set_target_cuda_architectures` is CUDA-only; add a HIP counterpart or skip on HIP.
- `apps/tests/backends/cuda/CMakeLists.txt`: same HIP branch for `backend_cuda`.
- `apps/tests/CMakeLists.txt`, `apps/examples/CMakeLists.txt`, `apps/tests/sanity_check/CMakeLists.txt`: they key on `UIPC_WITH_CUDA_BACKEND`, which stays ON, so probably no change; confirm executables linking HIP objects get `HIP_ARCHITECTURES` when auto-detecting (strategy-a-cmake.md, qrack note).
- `src/pybind/pyuipc/CMakeLists.txt`: HIP version metadata.
- New `src/backends/cuda/hip_compat/` (name the porter's choice): `cuda_runtime.h`, `cuda_runtime_api.h`, `cuda.h`, `cub/...` shadows, `cuda/atomic`, `cuda/std/functional`, the width-32 warp functions.
- `src/backends/cuda/type_define.h`: `UIPC_GENERIC` gate (+ `__CUDACC__` shim if needed).
- `src/backends/cuda/cuda_tool/debug.h`, `collision_detection/details/linear_bvh.inl`, `utils/step_displacement.h`: `__HIP_DEVICE_COMPILE__` gates.
- `src/backends/cuda/cuda_tool/eigen/evd.h`: HIP device-pass eigensolver for N>3.
- Warp sites: `cuda_device/detail/bit_operation.inl`, `algorithm/fast_segmental_reduce.h`, `algorithm/details/fast_segmental_reduce.inl`, `collision_detection/details/{info_stackless_bvh,info_stackless_bvh_v0,stackless_bvh}.inl`, `collision_detection/filters/{info_stackless_bvh_simplex_trajectory_filter.cu,toi_min.h}`, `global_geometry/axis_max.h`, `linear_system/{spmv.cu,linear_fused_pcg.cu,fused_pcg_kernels.h}`, `finite_element/mas_preconditioner_engine.cu`, `affine_body/{abd_linear_subsystem.cu,abd_diag_preconditioner.cu}`, `utils/{warp_spd.h,warp_joint_hessian.h}`, plus the joint `.cu` files that use 32-bit masks. If the shim's width-32 functions are reached through the CUDA names, many of these need no edit; the ballot sites and the absolute-lane shuffles do.
- Tests with CUDA-only bits: `internals/mas_preconditioner_direct_store.cu` (globaltimer asm, `__nanosleep`), any test copying the kernels above (`fast_segmental_reduce_lean.cu`, `internals/fused_pcg_spmv_pipeline.cu`, `warp_joint_hessian.cu`) gets the same warp fix as the code it mirrors.
- `README.md` / `docs/build_install/{linux,windows}.md`: a ROCm build section with the exact commands (with `-DCMAKE_PREFIX_PATH=<rocm>`; no device-selection env vars in front of commands).

## Build commands

Linux (any arch; the arch is a configure argument only). vcpkg is built inside the job directory.

```bash
git clone https://github.com/microsoft/vcpkg "$W/vcpkg" && "$W/vcpkg/bootstrap-vcpkg.sh" -disableMetrics
export CMAKE_TOOLCHAIN_FILE="$W/vcpkg/scripts/buildsystems/vcpkg.cmake"
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DUIPC_WITH_HIP=ON -DCMAKE_HIP_ARCHITECTURES=<gfx90a|gfx942|gfx1100|...> \
  -DCMAKE_PREFIX_PATH=<rocm prefix: /opt/rocm or the pip SDK root> \
  -DUIPC_BUILD_PYBIND=ON -DUIPC_PYTHON_EXECUTABLE_PATH=<venv python>
cmake --build build -j"$MAX_JOBS"
```

Also configure once without `-DCMAKE_HIP_ARCHITECTURES` (auto-detect path) and once with `-DUIPC_WITH_CUDA_LEGACY_COLLISION=OFF` (the lean option must still build).

Windows (gfx1151 laptop, last per jeff's standing order): GNU-driver clang from the ROCm SDK for C/CXX/HIP, Ninja, vcpkg `x64-windows`, same options; the exact recipe is the porter's to record in notes.md from windows.md.

CUDA no-regression (mandatory before presenting, on a host with nvcc): the same tree configured with the default options (`UIPC_WITH_HIP=OFF`) and `-DUIPC_CUDA_ARCHITECTURES=89`, built fully including tests; compare against upstream base 6c288d9 built the same way. Ideally also preprocess a sample of `.cu` files with nvcc at both shas to show identical device code where the port intends no change.

## Test plan

Test inventory (C++, Catch2, one CTest entry per executable, `TIMEOUT 900`, `WORKING_DIRECTORY` = source root, assets in-tree):

| CTest | executable | cases | needs | label |
|---|---|---|---|---|
| uipc.common | uipc_test_common | 3 | CPU | fast;cpu |
| uipc.core | uipc_test_core | 39 (+1 hidden `.stack_trace`) | CPU | fast;cpu |
| uipc.geometry | uipc_test_geometry | 48 | CPU | fast;cpu |
| uipc.sanity_check | uipc_test_sanity_check | 3 (none + cuda sections) | GPU | gpu |
| uipc.backend_cuda | uipc_test_backend_cuda | 152 (fewer on Windows: no internals/) | GPU | gpu |
| uipc.sim_case | uipc_test_sim_case | 95 scenes, 3-360 frames | GPU | gpu |
| uipc.regression | uipc_test_regression | 1 | GPU | gpu |

Python: 25 pytest files; `-m "cuda and not example"` runs the GPU ones (9 sim_case files, contact FD checks, baraff-witkin anisotropy, external force, profile); the default selection (`not example and not cuda`) is the CPU set. Examples `hello_affine_body` and `wrecking_ball` run the GPU backend.

Per platform, the validator runs, on the exact fork sha:

1. `ctest --test-dir build --output-on-failure --timeout 900 -L fast` (CPU regression set; must match a CPU-only upstream build: all pass).
2. `ctest --test-dir build --output-on-failure --timeout 900 -L gpu` (backend_cuda, sanity_check, sim_case, regression). If `uipc.sim_case` exceeds 900 s as one executable on a slow card, run it with `scripts/run_sim_case_isolated.py` (upstream's per-case runner) rather than raising the timeout, and record per-case results. Record pass/fail counts per executable and the Catch2 case counts (`uipc_test_backend_cuda --list-tests`, `-r compact`).
3. `python -m pytest -m "cuda and not example" python/tests` and `python -m pytest python/tests` (default selection) with the pyuipc built from the same tree.
4. Run `wrecking_ball` (and `hello_affine_body`) to completion; record frames and wall time.
5. Kernel-level proof that the AMD build actually ran on the GPU: `.hip_fatbin` present in `libuipc_backend_cuda.so` for the configured arch, and one run under `AMD_LOG_LEVEL=3` (or rocprofv3 kernel trace) listing backend kernels.
6. A comparison point for the GPU suites: the same test list on an NVIDIA GPU at upstream base (fleet V100 is sm_70 -- check CUDA 12.x still builds for it -- or any CUDA host jeff names), so a test that fails on AMD can be classed as port defect vs upstream failure. If no NVIDIA run is possible, say so in notes; do not assume upstream passes every sim case.

Gates:

- **wave64** (linux-gfx90a or linux-gfx942, CDNA): steps 1-6. The wave64 canaries in risk 1 must pass here; this is the only meaningful evidence for the ballot and width fixes.
- **wave32** (linux-gfx1100 or linux-gfx1201, RDNA): steps 1-5. FP64 is slow on RDNA; record sim_case wall time and use the isolated runner if the 900 s budget is exceeded.
- **windows** (windows-gfx1151, RDNA wave32, last): build with the documented Windows commands, steps 1-3 (backend_cuda without internals/, as upstream), step 4. The GPU_IPC windows gate was waived after the laptop bugchecked under GPU work (`csagent.sys`); if that recurs, the validator records the evidence and the waiver is jeff's decision, not an agent's.

Non-GPU regression set: `uipc.common`, `uipc.core`, `uipc.geometry`, the CPU pytest selection, and a default CUDA configure+build (the CUDA no-regression compile). The `none` backend sections of `uipc.sanity_check` also run on every platform.

## Open questions

1. Backend naming upstream: keep the backend name `cuda` on AMD (planned; zero user-facing change) or introduce a distinct name. Planned default stands unless jeff or the maintainer says otherwise; a rename is a small follow-up.
2. libcu++ route: shim (`plus` + fence, no new dependency; planned) vs libhipcxx (`hipccl`). The porter records which and why.
3. ROCm wheels: the port makes `pip install .` work with HIP config-settings; publishing ROCm wheels (cibuildwheel image, audit script) is upstream release CI and is not planned. Mention in the PR body as a possible follow-up for the maintainer.
4. NVIDIA baseline host for the GPU suites (test plan step 6): which CUDA host to use.
