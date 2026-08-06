# k2 -- ROCm/HIP port plan (lead platform: linux-gfx90a)

## Project
- Name: k2
- Upstream: https://github.com/k2-fsa/k2
- Default branch: master
- Cloned at HEAD e625cb9 ("Support building with torch 2.12.0", #1352)
- Domain: FSA/FST (finite-state automaton/transducer) algorithms for speech (CTC/LF-MMI/lattice rescoring), the C++/CUDA core behind icefall/sherpa/Next-gen Kaldi. PyTorch-integrated via a pybind11 `_k2` module.

## Existing AMD support assessment
**None. From-scratch CUDA->HIP port.**
- No HIP guards in any source: `grep -rIl hip|HIP|rocm|ROCm` over k2/csrc matches 6 files, all false positives ("relationship", "graphical", a comment in ctc_loss.py). Zero `__HIP_PLATFORM_*`, zero `USE_HIP`/`USE_ROCM`.
- No remote AMD/ROCm/HIP branch on origin (`git ls-remote --heads origin | grep -iE hip|rocm|amd` empty).
- README and CMake only know CUDA (`enable_language(CUDA)`, `find_program(nvcc)`, `K2_WITH_CUDA`). AMD is not mentioned anywhere.
**Decision: PROCEED. From-scratch port adds clear value** -- k2 is the foundation of the Next-gen Kaldi speech stack (icefall, sherpa) and currently has no AMD path at all. Not a perf-rewrite case: k2 is reduction/scan/sort/load-balance bound (cub + moderngpu primitives), not GEMM/attention/CUTLASS, so a correctness-first mechanical port + library swaps is the right first pass (no rocWMMA/CK rewrite needed).

## Build classification: CMake + Torch (hybrid) -> port mechanics are Strategy A
- Evidence it is **CMake-driven, not torch-AOT-hipify**:
  - `setup.py` (lines 125-261): `cmake_extension("_k2")` creates an `Extension` with **empty sources**; `BuildExtension.build_extension` just shells out to `cmake ... && make install`. It does **not** use `torch.utils.cpp_extension.CUDAExtension`/`BuildExtension`, so torch never AOT-hipifies the sources. `grep -rInE 'hipify|CUDAExtension|cpp_extension' setup.py cmake/ k2/python` = empty.
  - `CMakeLists.txt:32-54`: `set(languages CXX)`, conditionally `CUDA`, then `project(k2 ${languages})` and `enable_language(CUDA)` (implicitly via the `project()` language list). 120 `.cu` files are compiled by CMake's CUDA toolchain.
  - `find_package(Torch)` (`cmake/torch.cmake:15`) + pybind11 (`cmake/pybind11.cmake`) link torch and build the python module.
- Per PORTING_GUIDE "Build classification": a CMake build that finds Torch is nominally the "pytorch extension" bucket, **but** because torch's hipify is NOT in the build path, **Strategy B's "let torch hipify the sources" does not apply**. The correct mechanism is **Strategy A** (one compat header + `enable_language(HIP)` + mark `.cu` `LANGUAGE HIP`), exactly the RXMesh/cupoch/STRUMPACK pattern (CMake project that happens to consume a ROCm torch). The ROCm torch supplies `c10::cuda::*` already hipified at the torch level; k2 consumes `c10::cuda::getCurrentCUDAStream` (pytorch_context.cu) and that returns the HIP stream.

## Port strategy: Strategy A (compat header + enable_language(HIP)) + library swaps
Rationale: minimal footprint, NVIDIA path byte-for-byte unchanged, and it isolates HIP to the `.cu` TUs. The real work is **not** symbol renaming (CUDA runtime surface is a clean 1:1 hipify) -- it is **replacing two non-portable third-party deps** (moderngpu, cudpp) and **vendoring libhipcxx**. Concretely:
1. Add `K2_WITH_HIP`/`USE_HIP` CMake option that `enable_language(HIP)`, bypasses the entire NVIDIA-arch block (`select_compute_arch.cmake`, the `K2_COMPUTE_ARCH*` gencode loop, all under `if(K2_WITH_CUDA)`), marks the `.cu` sources `LANGUAGE HIP`, sets `HIP_ARCHITECTURES` from `${CMAKE_HIP_ARCHITECTURES}` (default gfx90a only when unset -- never a literal, per the CudaSift/Gpufit lesson), and turns on `HIP_SEPARABLE_COMPILATION` (mirrors the existing `CUDA_SEPARABLE_COMPILATION ON` -- see fault classes, k2 uses RDC).
2. One compat header `k2/csrc/cuda_to_hip.h` (the only file that knows HIP): include `<hip/hip_runtime.h>`, alias the ~30 `cudaXxx` runtime symbols used (see surface inventory) to `hipXxx`, define a full-warp mask and `kWarpSize`. Keep CUDA spelling everywhere else. Force-include it on every HIP TU (`CMAKE_HIP_FLAGS -include .../cuda_to_hip.h`) so defines precede includes regardless of file order (MPPI-Generic lesson).
3. Swap libraries on the HIP path: `cub/cub.cuh` -> hipCUB (`<hipcub/hipcub.hpp>`), curand device API -> hipRAND device API, `<cuda/std/functional>` -> vendored ROCm/libhipcxx.
4. Replace moderngpu and cudpp (the two hard blockers) -- see below.
5. Guard genuinely divergent code with `#if defined(USE_HIP)`; keep guards rare.

## CUDA surface inventory (from k2/csrc: 120 .cu, 42 .cc, 123 .h)
- **Kernels / device code**: 15 `__global__`, 179 `__device__`, but the bulk of parallelism is k2's lambda-launch machinery in `eval.h` (`K2_EVAL`, `EvalDevice`, `eval_lambda`, `eval_lambda_group`) -- portable as-is.
- **CUDA runtime symbols** (~30 unique, all 1:1 hipify): `cudaStream_t`(56), `cudaError_t`(39), `cudaSuccess`(22), `cudaEvent_t`(17) + `cudaEventCreate[WithFlags]/Record/Destroy/Query/Synchronize/ElapsedTime/DisableTiming`, `cudaMemcpy[Async]`+`cudaMemcpyKind`+`...DeviceToHost/HostToDevice/DeviceToDevice`, `cudaMalloc`/`cudaFree`, `cudaMallocHost`(5, pinned -> `hipHostMalloc`), `cudaStreamCreate/Destroy/Synchronize/WaitEvent`, `cudaSetDevice`/`cudaGetDevice`/`cudaGetDeviceCount`/`cudaGetDeviceProperties`/`cudaDeviceProp`/`cudaDeviceSynchronize`, `cudaGetLastError`/`cudaGetErrorString`, `cuda{Runtime,Driver}GetVersion`, error enums (`cudaErrorNotReady`/`...Assert`/`...MemoryAllocation`/`...InitializationError`).
- **CUDA Cooperative Groups** (eval.h, intersect.cu, fsa_utils.cu): `cg::tiled_partition<N>` / `cg::thread_block_tile<N>` with N in {4, 8, and 8 via *2} -- ALL <= 32. Ops used: `g.sync()`(4), `g.thread_rank()`(3), `g.size()`(3), `g.shfl_up`(1), `g.shfl(v,lane)`(intersect). **No `cg::reduce`, no `cg::labeled_partition`, no `cg::ballot/any/all`** -- i.e. exactly the ops the gsplat lesson flagged as MISSING from HIP CG are NOT used here. HIP maps `<cooperative_groups.h>` -> `<hip/hip_cooperative_groups.h>` and these sub-warp tiles are wave64-safe (the tile's shfl uses width=N<=32, stays within a 64-lane wavefront). Expected: zero source change beyond the include mapping.
- **moderngpu** (8 files, 38 `mgpu::` uses): `transform_lbs`(4), `mergesort`(3), `segmented_sort`/`segmented_sort_indices`(3), `load_balance_search`(1), `sorted_search<bounds_lower>`(1), `transform_scan`(1), plus the allocator base `standard_context_t`/`context_t`. **THE central blocker -- see Risk list.**
- **cub** (47 uses, `CUB_WRAPPED_NAMESPACE=k2`): `DeviceScan`(15), `DeviceReduce`(11), `DeviceSegmentedReduce`(10), `DeviceRadixSort`(5), `DeviceHistogram`(2). All have hipCUB equivalents (header drop-in under the `hipcub::` -> wrapped namespace). k2 already wraps cub in namespace `k2` to dodge torch's bundled cub; the same define works for hipCUB.
- **cudpp** (`k2/csrc/cudpp/`, compiled only when `K2_WITH_CUDA`): a vendored 2007 CUDPP segmented-scan providing `SegmentedExclusiveSum`. `segmented_scan_cta.h` is **warp-synchronous** (`WARP_SIZE=32`, `LOG_WARP_SIZE=5`, `lane=idx&(WARP_SIZE-1)`, two-warps-per-block via `warpid`/`warpid2`, `warpSegScan`). **A genuine wave64 fault.** Only 1 production caller (`fsa_utils.cu:2232`) + 1 test.
- **curand** (rand.cu only): `curandStatePhilox4_32_10_t`, `curand_init`, `curand_uniform4`, `curand_uniform2_double` -> hipRAND device API (`hiprandStatePhilox4_32_10_t`, `hiprand_init`, `hiprand_uniform4`, `hiprand_uniform2_double`).
- **Torch coupling**: `pytorch_context.cu` uses `c10::cuda::{getCurrentCUDAStream,set_device,device_count,current_device,CUDACachingAllocator}` -- on a ROCm torch these are the hipified c10 symbols; the `#ifdef K2_WITH_CUDA` block that includes `c10/cuda/*` must also be active on HIP (define `K2_WITH_CUDA` for the HIP build, or broaden the guard to `K2_WITH_CUDA || K2_WITH_HIP`).
- **NOT present** (whole fault-class families are N/A): textures/surfaces (0), `cudaArray`/layered arrays (0), managed memory (0), cuBLAS/cuFFT/cuSPARSE/cuDNN/cusolver/nvrtc/jitify (0), thrust (0 direct uses), `__shfl*`/`__ballot`/`__activemask` in k2's own code (0 -- all such code is inside moderngpu/cudpp which get replaced). `warpSize` runtime read appears once (benchmark.cu, fine).

## Risk list (ordered by severity)
1. **moderngpu is unportable and has no ROCm fork (TOP RISK -- analogous to CUTLASS/cuCollections).** `intrinsics.hxx:6` is `#error "You must compile this file with nvcc. You must."`; `meta.hxx:54` hardcodes `warp_size = 32`; CTA reduce/scan/segsort use inline PTX (`bfe.b32`, `bfi.b32`, `prmt.b32`, `shfl.*` asm) and gate every device path on `__CUDA_ARCH__ >= 300/200` (which is undefined under HIP -> collapses to host fallback). It cannot be hipified. **Mitigation: do NOT port moderngpu.** The API surface k2 uses is small and every primitive maps to hipCUB/rocThrust, which k2 already uses elsewhere (`cub::DeviceScan::ExclusiveSum` is already its native ExclusiveSum). Replacement plan, USE_HIP-guarded, behind k2's own helper sites so call sites stay shaped the same:
   - `load_balance_search` / `sorted_search<bounds_lower>` (both in `utils.cu`, both compute row_ids from row_splits): replace with the row_ids computation. `utils.cu` already contains a non-mgpu fallback branch (`if (1) {...} else {...}` "Will probably just delete this branch") and a CPU reference; the GPU path is exactly an upper-bound of each element index into row_splits -> `rocprim`/`hipcub` upper-bound, or `thrust::upper_bound` (rocThrust), or k2's existing binary-search lambda.
   - `transform_lbs` (array_ops.cu x2, array_ops_inl.h, macros.h): load-balanced segmented transform `f(index, seg, rank)`. Decompose into row_ids (= the replacement above) + a `K2_EVAL` over `ans_size` computing `seg = row_ids[index]`, `rank = index - row_splits[seg]`, then call the existing lambda body. k2 already materializes row_ids everywhere, so this is natural.
   - `mergesort` (array_ops_inl.h x3): keys-only and keys+index_map -> `hipcub::DeviceMergeSort::Sort{Keys,Pairs}` or rocThrust `sort`/`sort_by_key` (index_map seeded by `Range`).
   - `segmented_sort` / `segmented_sort_indices` (ragged_ops_inl.h): per-segment sort with custom comparator -> `hipcub::DeviceSegmentedRadixSort::SortPairs` with `segment.Data()` offsets (k2's comparators are <,> on int/float, expressible as ascending/descending radix; if a true custom comparator is needed, fall back to rocThrust per-segment sort). NOTE the cudaKDTree lesson: hipCUB radix sort with a non-zero begin_bit is broken on gfx90a -- sort full key width.
   - `transform_scan` (macros.h, the segmented-/non-seg exclusive-sum macro): -> `cub::DeviceScan::ExclusiveSum` (k2 already uses this; reuse its `ExclusiveSum` wrapper).
   - `standard_context_t` allocator (moderngpu_allocator.cu): only needed to feed moderngpu; once moderngpu is gone, the allocator wrapper can be dropped on HIP (the replacements take a stream + k2 Context allocate/deallocate directly). Exclude `moderngpu_allocator.cu`/`.h` from the HIP source list, or stub `GetModernGpuAllocator` out.
2. **cudpp warp-synchronous segmented scan = wave64 fault** (`segmented_scan_cta.h`). The unrolled `warpSegScan` assumes 32-lane lockstep and packs two warps per block via `warpid`/`warpid2` (the MPPI warp-synchronous-reduction + popsift two-32-rows fault classes). **Mitigation: replace `SegmentedExclusiveSum` on HIP with hipCUB**, not fix the warp arithmetic -- it is exactly a segmented exclusive scan over a flags array, which `hipcub::DeviceScan`/rocPRIM segmented scan provides natively. Exclude `cudpp/cudpp.cu` from the HIP source list and route `SegmentedExclusiveSum` to a hipCUB implementation. Validate against the existing `RaggedOpsTest.SegmentedExclusiveSum` gtest (int32/float/double) and the `fsa_utils.cu:2232` caller (arc CDF).
3. **libcu++ / `cuda::std` gap.** `cub.h` includes `<cuda/std/functional>` and ROCm ships no `cuda/std`. Mitigation: vendor ROCm/libhipcxx (`git clone --branch amd-develop https://github.com/ROCm/libhipcxx`, header-only, add `-I<clone>/include`) per the rmm/gsplat/libhipcxx lesson; `<cuda/std/*>` passes through unchanged. (Confirmed locally: hipcub/rocprim/hiprand/rocrand are present under /opt/rocm/include; rocthrust + cuda/std are absent -- install rocThrust via the ROCm package if the thrust-based replacements are chosen, and vendor libhipcxx regardless.)
4. **Relocatable device code (RDC) required.** `context` and the test exes set `CUDA_SEPARABLE_COMPILATION ON` because k2 declares `__device__` symbols in headers and defines them across `.cu` TUs (explicit-instantiation). HIP needs `-fgpu-rdc` + `HIP_SEPARABLE_COMPILATION ON` on the lib AND every consumer (RXMesh lesson), else "undefined hidden symbol" at device link. Also mark consumer `.cu` (tests, python module) `LANGUAGE HIP` or CMake silently drops them from the device link (RXMesh: tests linked zero objects, fell back to gtest_main).
5. **CUB wrapped namespace + torch's bundled hipCUB.** k2 sets `CUB_WRAPPED_NAMESPACE=k2`. hipCUB honors a wrapping macro too; verify the equivalent (`THRUST_NS_QUALIFIER` is still meaningful; hipCUB uses its own). The ROCm torch also bundles a hipCUB -- keep k2's wrap so the symbols do not clash, same reason as the CUDA build.
6. **`c10::cuda` activation on HIP.** The torch-allocator/stream bridge is gated on `#ifdef K2_WITH_CUDA`; if `K2_WITH_HIP` is a separate define, those blocks go dark and GetCudaStream returns the invalid stream. Mitigation (AutoDock-GPU lesson): define `K2_WITH_CUDA` FOR the HIP build (one line) so the shared GPU/driver blocks compile unchanged, and gate only genuinely NVIDIA-only includes (`cuda_runtime.h` direct, nvToolsExt) to `K2_WITH_CUDA && !K2_WITH_HIP`. The compat header retargets cuda*->hip*. Far smaller than broadening dozens of guards.
7. **NVTX.** `K2_ENABLE_NVTX` + nvToolsExt has no ROCm equivalent (roctx exists but API differs). Mitigation: set `K2_ENABLE_NVTX OFF` on HIP (it is already auto-OFF when `K2_WITH_CUDA` is OFF, CMakeLists.txt:95-98) and make `NVTX_RANGE`/`K2_FUNC` no-ops on HIP (k2_nvtx is INTERFACE-only; just don't define K2_ENABLE_NVTX). No functional impact.
8. **`-Wall`/`-Wno-*` flags pushed into CMAKE_CUDA_FLAGS** assume nvcc front-end. On the HIP path these are clang flags (mostly compatible) but `--compiler-options`/`-Xptxas`/`-use_fast_math`/`-lineinfo`/`--expt-extended-lambda` (the moderngpu flags, CMakeLists.txt:317) are nvcc-only. Mitigation: the whole flag block is under `if(K2_WITH_CUDA)`; the HIP path must not inherit it. `--expt-extended-lambda` (extended device lambdas) is implicit under hipcc/clang; no flag needed.
9. **pybind11 + HIP LTO** (gpuRIR/Fast-Poisson lesson): if the `_k2` module links via `pybind11_add_module`, HIP LTO can strip `PyInit__k2`. Mitigation: `NO_EXTRAS` on the module / disable IPO on the HIP build. Check `cmake/pybind11.cmake` + `k2/python/csrc/CMakeLists.txt` at port time.
10. **double-precision device code**: rand.cu uses `curand_uniform2_double`, several ops are templated on double. gfx90a supports fp64; just confirm hipRAND double API and that tests using double pass.

## File-by-file change list (all HIP-guarded; CUDA path byte-for-byte unchanged)
- `CMakeLists.txt`: add `option(K2_WITH_HIP)`; when ON -> `enable_language(HIP)`, skip the NVIDIA arch/flag blocks, set `K2_WITH_CUDA`-for-HIP define, add `-DK2_WITH_HIP`; default `K2_ENABLE_NVTX OFF`.
- `cmake/torch.cmake`: skip the `CUDA_VERSION == TORCH_CUDA_VERSION` check on HIP; read `torch.version.hip` instead; do not strip torch CUDA interface options the same way (verify torch_hip target).
- `cmake/moderngpu.cmake`: do not fetch/link moderngpu on HIP (it is unportable).
- `cmake/cub.cmake`: N/A on modern CUDA; on HIP use hipCUB include path (header-only on /opt/rocm).
- new `k2/csrc/cuda_to_hip.h`: the single compat header (runtime symbol aliases, warp mask, kWarpSize, pinned-alloc alias). Force-included on HIP TUs.
- `k2/csrc/cub.h`: on HIP include `<hipcub/hipcub.hpp>` + libhipcxx `<cuda/std/functional>`; drop nvToolsExt.
- `k2/csrc/CMakeLists.txt`: mark `context_srcs` + test `.cu` + benchmark `.cu` `LANGUAGE HIP`; `HIP_SEPARABLE_COMPILATION ON`; on HIP drop `cudpp/cudpp.cu` and `moderngpu_allocator.cu` from the list; link hip::host (NOT hip::device -- the `--offload-arch` propagation trap, cupoch/STRUMPACK lesson); add libhipcxx -I.
- `k2/csrc/utils.cu`: USE_HIP path for `RowSplitsToRowIds` (replace `load_balance_search`/`sorted_search` with row_ids-from-upper-bound).
- `k2/csrc/array_ops.cu` + `array_ops_inl.h`: USE_HIP `transform_lbs` -> row_ids + K2_EVAL; `mergesort` -> hipCUB/rocThrust.
- `k2/csrc/ragged_ops.cu` + `ragged_ops_inl.h`: USE_HIP `segmented_sort[_indices]` -> hipCUB DeviceSegmentedRadixSort / rocThrust; the `ragged_ops.cu:1537` mgpu context use.
- `k2/csrc/macros.h`: USE_HIP `transform_scan` -> existing `ExclusiveSum` (cub).
- new `k2/csrc/cudpp/cudpp_hip.cu` (or guard in cudpp.cu): `SegmentedExclusiveSum` via hipCUB DeviceScan over flags; exclude the warp-sync .cu on HIP.
- `k2/csrc/rand.cu`: curand device API -> hipRAND device API (USE_HIP-guarded includes + symbol aliases, or via compat header).
- `k2/csrc/pytorch_context.cu` / `pytorch_context.h`: ensure the `c10::cuda` block is active on HIP (define K2_WITH_CUDA-for-HIP).
- `k2/csrc/eval.h`, `intersect.cu`, `fsa_utils.cu`: expected NO change (CG sub-warp tiles portable); confirm at build/validate time.
- `k2/python/csrc/CMakeLists.txt`, `cmake/pybind11.cmake`: `NO_EXTRAS`/IPO-off for the HIP `_k2` module if LTO strips PyInit.
- `k2/torch/csrc` (28 files): the libtorch C++ API layer (k2 standalone C++ decoders). Mark `.cu` LANGUAGE HIP; likely no semantic change (same runtime surface). Lower priority than the python path; validate the gtest core first.

## Build commands (gfx90a)
Prereqs (planner-confirmed present): ROCm 7.2.1, torch 2.13.0a0 with hip 7.2.53211 (`torch.cuda.is_available()=True`), hipcub/rocprim/hiprand/rocrand under /opt/rocm/include. To install/vendor: rocThrust (`apt-get install rocthrust` or ROCm meta-pkg, if the thrust-based replacements are used) and libhipcxx (vendor: `git clone --branch amd-develop https://github.com/ROCm/libhipcxx _deps/libhipcxx`).
Configure + build (out-of-source; k2 forbids in-source):
```
export PYTHONPATH=  # use the ROCm torch in the active env
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release \
      -DK2_WITH_HIP=ON -DK2_WITH_CUDA=OFF \
      -DCMAKE_HIP_ARCHITECTURES=gfx90a \
      -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
      -DCMAKE_CXX_STANDARD=17 \
      -DPYTHON_EXECUTABLE=$(which python3) \
      -DK2_ENABLE_TESTS=ON -DK2_ENABLE_BENCHMARK=OFF \
      ..
make -j"$(nproc)"
```
Notes: C++17 is required (torch >= 2.1 and rocPRIM/hipCUB hard-`#error` on < C++17 -- GPUMD lesson; setup.py already bumps to 17 for torch>=2.1). Pass every target arch at once if validating followers from the lead commit (`-DCMAKE_HIP_ARCHITECTURES="gfx90a;gfx1100;gfx1151"`), but lead build uses gfx90a only. Python build path: `K2_CMAKE_ARGS="-DK2_WITH_HIP=ON -DK2_WITH_CUDA=OFF -DCMAKE_HIP_ARCHITECTURES=gfx90a" python3 setup.py install`.

## Test plan
GPU is present (gfx90a, MI250X). Assign one GPU and run serially (MPPI lesson: parallel ctest on one GPU flaps).
- **C++ gtest (primary correctness gate)**: 28 executables `cu_*_test` built by `k2/csrc/CMakeLists.txt` (algorithms, array_ops, array, connect, dtype, fsa_algo, fsa, fsa_utils, hash, host_shim, intersect, log, macros, math, nbest, nvtx, pinned_context, ragged_shape, ragged, ragged_utils, rand, reverse, rm_epsilon, rnnt_decode, tensor_ops, tensor, thread_pool, top_sort, utils). Run: `cd build && HIP_VISIBLE_DEVICES=<id> ctest --output-on-failure` (serial). The highest-signal tests for this port:
  - `cu_ragged_test` -> `RaggedOpsTest.SegmentedExclusiveSum` (int32/float/double): validates the cudpp replacement.
  - `cu_array_ops_test`, `cu_array_test`: ExclusiveSum, mergesort, Cat-with-offsets -> validates the moderngpu mergesort/transform_lbs/transform_scan replacements.
  - `cu_ragged_test` / `cu_ragged_utils_test` (SortSublists): validates segmented_sort replacement.
  - `cu_utils_test`: RowSplitsToRowIds/RowIdsToRowSplits -> validates load_balance_search/sorted_search replacement.
  - `cu_intersect_test`, `cu_fsa_algo_test`, `cu_rnnt_decode_test`: exercise the cooperative-groups sub-warp tiles end-to-end on wave64.
  - `cu_rand_test`: hipRAND device API.
  Each gtest already runs CPU vs CUDA-device internally (k2 tests compare kCpu and kCuda results), so they ARE cross-backend correctness checks -- ideal for a port. Bar: 100% pass on gfx90a, identical to CPU reference within k2's own tolerances.
- **Python tests (integration gate)**: 64 files in `k2/python/tests/` (intersect_dense[_pruned], ctc_loss, mutual_information, rnnt_loss/decode, get_forward/backward_scores, levenshtein, arc_sort, connect, ...). Run after install: `cd k2/python/tests && for t in *_test.py; do HIP_VISIBLE_DEVICES=<id> python3 -m pytest -q $t; done` (or k2's `scripts/` test runner). These exercise the autograd/pybind path and real FSA algorithms on GPU. multi_gpu_test.py needs >1 GPU (skip if single-GPU assigned).
- **Non-GPU regression set (must not regress)**: every gtest also runs its kCpu path; the host C++ in `k2/csrc/host/` is CPU-only. Build with `K2_WITH_HIP=OFF K2_WITH_CUDA=OFF` once to confirm the CPU-only build still works (the `transform()` macro maps `.cu`->`.cc` for the no-CUDA build). Confirm CPU gtests pass unchanged.
- **Determinism**: reductions/scans/sorts via hipCUB are deterministic for a fixed problem; assert run-to-run bitwise identity on a representative ragged op (e.g. ExclusiveSum + SortSublists) across >=2 runs (MPPI/amgcl bar). Segmented sort with equal keys: stability differs between mergesort (mgpu, stable) and radix (hipCUB) -- if a test relies on stable order of equal keys, use a stable replacement (rocThrust stable_sort_by_key) and note it.

## Inter-project dependencies
**None.** k2's real third-party deps are moderngpu (no ROCm fork -> replaced, not ported), cub (-> hipCUB, header-only), pybind11, googletest, and torch (ROCm torch already installed). None are MOAT projects, so no `set-deps` is required. (k2 is itself a base for icefall/sherpa, but those are out of MOAT scope here.)

## Open questions
1. **moderngpu `segmented_sort` stability + custom comparator.** k2 passes `Op()` (a `<`/`>` functor) to segmented_sort_indices. hipCUB DeviceSegmentedRadixSort handles ascending/descending only; if any call site needs a non-order comparator or stable equal-key order, fall back to rocThrust per-segment `stable_sort_by_key`. To resolve at port time by reading every `Op()` passed (grep shows LessThan/GreaterThan int32 in the sites seen). Low risk.
2. **rocThrust vs pure-hipCUB for the replacements.** rocThrust is not currently installed (cuda/std + rocthrust absent locally). Prefer pure hipCUB/rocPRIM where possible (already present) to avoid the rocThrust dependency; install rocThrust only if a custom-comparator sort forces it. Decide during porting.
3. **`transform_lbs` perf.** Decomposing into row_ids + K2_EVAL is correct but may be slightly less cache-optimal than moderngpu's cached segmented load. Correctness-first is fine for the lead port; revisit only if a benchmark regresses materially (k2 benchmarks are optional, K2_ENABLE_BENCHMARK).
4. **k2/torch (libtorch C++ decoders, 28 files) scope.** Validate the python `_k2` + gtest core first; the standalone C++ torch API (k2/torch/bin) is lower priority and can be a follow-up if it needs separate attention. Confirm it is in the default build target set or behind its own option.
5. **CUB wrapped-namespace macro for hipCUB.** Confirm hipCUB's namespace-wrap mechanism matches `CUB_WRAPPED_NAMESPACE=k2` semantics (avoid clashing with the ROCm torch's bundled hipCUB). Verify at first compile.
