# amgcl ROCm/HIP port plan (linux-gfx90a lead)

## Project

ddemidov/amgcl -- header-only C++ algebraic multigrid (AMG) library. Multiple
compute backends: builtin (OpenMP), cuda (cuSPARSE + Thrust), vexcl/viennacl
(OpenCL), eigen, blaze, mkl, hpx. CMake project, NOT a pytorch extension
(no find_package(Torch), no torch.utils.cpp_extension) -> pure-CMake, but the
HIP target is header-only template code, so the build story is just "compile a
.cu/.hip TU that includes the new backend header with the HIP toolchain".

## Existing-AMD assessment (the key planning call)

Grep of the entire source tree for `hip`, `rocsparse`, `rocthrust`, `rocprim`,
`hipsparse` returns ZERO matches (`amgcl/`, `examples/`, `tests/`). There is no
native HIP/rocSPARSE backend upstream. AMD GPUs are reachable today only via the
OpenCL-based vexcl and viennacl backends. Per PORTING_GUIDE "assess existing AMD
support": OpenCL-only support does NOT count as already-supported, and a native
HIP/rocSPARSE backend is the value-add. Disposition: PORT (not already-supported,
not ported-elsewhere).

## What the CUDA backend actually is (port surface)

Two and only two files in the library depend on CUDA
(grep cusparse|cuda_runtime|thrust/ over amgcl/):

1. `amgcl/backend/cuda.hpp` (843 lines) -- THE backend. Structure:
   - Vector ops (axpby, axpbypcz, vmul, inner_product, clear, copy, gather,
     scatter, residual) -- pure Thrust (`thrust::for_each`/`transform`/
     `inner_product`/`gather`/`scatter` + zip-iterator functors). No cuSPARSE.
   - `cuda_matrix<real>` SpMV -- the cuSPARSE generic API:
     `cusparseCreateCsr`, `cusparseCreateDnVec`, `cusparseSpMV_bufferSize`,
     `cusparseSpMV_preprocess` (CUDART>=12040), `cusparseSpMV` with
     `CUSPARSE_SPMV_CSR_ALG1`. (A pre-CUDA-11 Hyb fallback also exists under
     `CUDART_VERSION < 11000`; not relevant on ROCm, drop behind the version
     guard.)
   - `solver::cuda_skyline_lu` -- coarse direct solver; copies to host, pure
     Thrust.
   - `cuda_event`/`cuda_clock` -- timing via `cudaEvent*`.
   - backend `params` carries a `cusparseHandle_t`.

2. `amgcl/relaxation/cusparse_ilu0.hpp` (783 lines) -- the ILU0 smoother
   specialization `ilu0<backend::cuda<real>>`. Uses `cusparseXcsrilu02*`
   (incomplete-LU factorization) + the cuSPARSE generic SpSV
   (`cusparseSpSV_createDescr`/`_bufferSize`/`_analysis`/`_solve`, persistent
   `cusparseSpSVDescr_t`) + `cusparseSpMatSetAttribute` (fill mode / diag type).

The default AMG smoother is `spai0` (relaxation/runtime.hpp:143
`prm.get("type", runtime::relaxation::spai0)`), and `spai0.hpp` has NO cuda/
cusparse/thrust dependency -- it runs through the generic backend ops. So a full
AMG solve with default settings needs ONLY the backend ported. cusparse_ilu0 is
needed only (a) when the user selects the `ilu0` smoother and (b) for the
runtime-relaxation example to compile the `ilu0` case against the HIP backend.
Both files are ported here so the HIP backend is a true peer of the CUDA backend.

## Strategy: hipSPARSE shim (mirror cuda.hpp, minimal diff), not low-level rocSPARSE

The decisive finding: hipSPARSE is installed (ROCm 7.2.1, /opt/rocm/include/
hipsparse) and exposes the EXACT cuSPARSE API surface this code uses, name for
name:
- `hipsparseCreateCsr`, `hipsparseCreateDnVec`, `hipsparseDestroySpMat`,
  `hipsparseDestroyDnVec` (hipsparse-generic-auxiliary.h)
- `hipsparseSpMV_bufferSize`, `hipsparseSpMV_preprocess`, `hipsparseSpMV`,
  `HIPSPARSE_SPMV_CSR_ALG1` (internal/generic/hipsparse_spmv.h,
  hipsparse-generic-types.h)
- `hipsparseSpSVDescr_t`, `hipsparseSpSV_createDescr`/`_destroyDescr`/
  `_bufferSize`/`_analysis`/`_solve`, `HIPSPARSE_SPSV_ALG_DEFAULT`
  (internal/generic/hipsparse_spsv.h) -- the PERSISTENT SpSV descriptor exists,
  matching cuSPARSE; the lower-level rocSPARSE generic SpSV has no persistent
  descriptor (stage-based), so hipSPARSE is the closer match.
- `csrilu02Info_t`, `hipsparseCreateCsrilu02Info`/`DestroyCsrilu02Info`,
  `hipsparseXcsrilu02_bufferSize`/`_analysis`/`_zeroPivot`,
  `hipsparseDcsrilu02`/`Scsrilu02` (internal/precond/hipsparse_csrilu0.h)
- `hipsparseSpMatSetAttribute`, `HIPSPARSE_SPMAT_FILL_MODE`/`_DIAG_TYPE`,
  `HIPSPARSE_FILL_MODE_LOWER`/`_UPPER`, `HIPSPARSE_DIAG_TYPE_UNIT`/`_NON_UNIT`.
- data type is `hipDataType` (`HIP_R_32F`/`HIP_R_64F`, hip/library_types.h),
  replacing `cudaDataType`/`CUDA_R_32F`/`CUDA_R_64F`.

Because hipSPARSE is a 1:1 cuSPARSE clone, the HIP backend can mirror cuda.hpp
line-for-line with a mechanical `cusparse`->`hipsparse`, `CUSPARSE_`->
`HIPSPARSE_`, `cudaDataType`->`hipDataType`, `CUDA_R_*`->`HIP_R_*`,
`cudaEvent*`->`hipEvent*`, `cudaError_t`->`hipError_t` rename. Thrust stays as
`thrust::` (rocThrust is a header-path-compatible drop-in under /opt/rocm/include
-- confirmed by prior MOAT ports, cudaKDTree changelog). This keeps the diff
small, keeps the structure identical to upstream (so it is reviewable and
upstreamable), and avoids re-deriving the lower-level rocSPARSE stage machinery.

### Footprint (minimal, additive, existing backends untouched)

Add two new headers (do NOT edit cuda.hpp / cusparse_ilu0.hpp -- keep the NVIDIA
path byte-identical):
- `amgcl/backend/hip.hpp` -- `namespace amgcl::backend { template<...> struct hip; }`
  mirroring `cuda`, with `hip_matrix`, `hip_skyline_lu`, `hip_event`/`hip_clock`,
  and the spmv/residual/axpby/... `_impl` specializations on
  `thrust::device_vector` (these are structurally identical to cuda.hpp; the
  Thrust specializations are shared in form but must be defined once -- see
  "shared Thrust specializations" risk below).
- `amgcl/relaxation/rocsparse_ilu0.hpp` -- `ilu0<backend::hip<real>>`
  specialization mirroring cusparse_ilu0.hpp via hipSPARSE.

Wire-up (additive):
- Top `CMakeLists.txt`: add a `USE_HIP` option; when ON, `enable_language(HIP)`,
  find hipSPARSE (`find_package(hipsparse)` or just link
  `/opt/rocm/lib/libhipsparse.so` + rocThrust headers), and define an
  INTERFACE `hip_target` (mirrors `cuda_target`) carrying
  `SOLVER_BACKEND_HIP` + the hip libs. Do NOT hardcode the arch: default
  `CMAKE_HIP_ARCHITECTURES` to gfx90a only when unset, read it on the target
  (PORTING_GUIDE lead-arch rule).
- `examples/CMakeLists.txt`: add an `add_hip_example` that copies `<ex>.cpp` ->
  `<ex>.hip` and compiles as HIP with `-DSOLVER_BACKEND_HIP` (mirror
  `add_cuda_example`). Wire `solver` at least.
- `examples/solver.cpp` (and the two other cuda examples if cheap): add an
  `#elif defined(SOLVER_BACKEND_HIP)` arm including `backend/hip.hpp` +
  `relaxation/rocsparse_ilu0.hpp` and `typedef amgcl::backend::hip<double>
  Backend;`, plus the matching arms at the other `SOLVER_BACKEND_CUDA`
  branch points (lines ~251, 297, 322 set up the handle / params).
  All new arms guarded so the CUDA/builtin builds are unchanged.

### Risks / things to get right (recorded, will confirm at build)

1. Shared Thrust `_impl` specializations. cuda.hpp defines
   `spmv_impl`/`axpby_impl`/`inner_product_impl`/... specialized on
   `thrust::device_vector<V>`. hip.hpp's vector type is ALSO
   `thrust::device_vector<V>` (rocThrust), so most of those specializations are
   identical TYPES and would collide (ODR / redefinition) if both headers are
   included in one TU. The CUDA and HIP backends are never used in the same
   build (mutually exclusive `SOLVER_BACKEND_*`), so in practice only one is
   included. Plan: keep hip.hpp self-contained (its own copy of the Thrust
   specializations, like cuda.hpp), and rely on the backends being mutually
   exclusive. If a single TU ever needs both, that is out of scope. Verify no
   accidental double-include in the example.

2. `hipsparseSpMV_preprocess` exists unconditionally in hipSPARSE (7 decls); the
   cuda.hpp call is guarded by `CUDART_VERSION >= 12040`. On HIP there is no
   CUDART_VERSION; gate the preprocess call on a HIP condition (call it always,
   or guard with `#ifdef __HIP_PLATFORM_AMD__`). preprocess is valid/no-harm.

3. SpMV algorithm + determinism. cuSPARSE `CSR_ALG1` -> hipSPARSE
   `HIPSPARSE_SPMV_CSR_ALG1` (= rocSPARSE rowsplit, which is DETERMINISTIC per
   rocsparse_spmv.h alg table). Good: deterministic SpMV supports the
   run-to-run determinism validation requirement.

4. Warp size. The backend has NO custom kernels with hardcoded 32-lane
   assumptions -- all device work is rocSPARSE SpMV/SpSV + Thrust transforms
   (zip-iterator functors are per-element, lane-agnostic). So the wave64 fault
   class does not bite here. (Confirm: no `__shfl`/`__ballot`/`warpSize`/`32`
   in the ported headers.)

5. `int` indices. cuda_matrix stores `thrust::device_vector<int>` ptr/col and
   passes `HIPSPARSE_INDEX_32I`; matches hipSPARSE. Fine for the validation
   problem sizes.

6. C++ standard. rocThrust/rocPRIM/hipCUB hard-#error on < C++17 (GPUMD
   changelog). amgcl requires C++11+ generally; ensure the HIP TU compiles at
   `-std=c++17` (set on the hip_target / example).

## Build (lead, gfx90a, GPU 1)

```
export HIP_VISIBLE_DEVICES=1
cd projects/amgcl/src
cmake -S . -B build-hip -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a \
      -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
      -DCMAKE_BUILD_TYPE=Release -DAMGCL_BUILD_EXAMPLES=ON
cmake --build build-hip --target solver_hip -j
```

(Exact CMake option names confirmed against the tree during porting; amgcl uses
custom BUILD flags -- read top CMakeLists for the example/test toggles.)

## Validation (REAL GPU, gfx90a / GPU 1)

Canonical AMG test: 7-point 3D Poisson on an n^3 grid (SPD), generated by
`tests/sample_problem.hpp` / `examples/make_poisson.py`. Plan:
1. Build `solver_hip` (HIP backend) and `solver` (builtin CPU backend) from the
   same `solver.cpp`.
2. Solve the SAME Poisson system (e.g. n=64 -> 262144 unknowns) with both.
   Default preconditioner = AMG + spai0 smoother, default Krylov (bicgstab/cg).
3. Assert on the HIP run:
   - CONVERGES: solver reports `converged`, iters finite, within max_iter.
   - RESIDUAL: final relative residual <= solver tol (e.g. 1e-8), and matches the
     builtin backend's iteration count / final residual within tolerance
     (same algorithm, so iters should match closely; residual ~ machine eps of
     each other).
   - SOLUTION vs reference: download the HIP solution x_hip and compare to the
     builtin x_cpu elementwise: ||x_hip - x_cpu||_inf / ||x_cpu||_inf <= ~1e-6.
   - DETERMINISM: run the HIP solve twice; x_hip(run1) == x_hip(run2) bitwise (or
     to ~1e-12). Deterministic SpMV (rowsplit) + Thrust reductions should give
     run-to-run stability; if Thrust inner_product reduction order varies, fall
     back to a tight tolerance and note it.
4. If the example harness is awkward to assert on, write a small standalone
   driver (in agent_space, NOT committed to the fork) that includes
   `backend/hip.hpp` + builtin, builds both solvers, and does the compare. The
   fork change stays the backend headers + CMake/example wiring.
5. Also exercise the `ilu0` smoother path (`solver_hip ... precond.relax.type=
   ilu0`) to validate rocsparse_ilu0.hpp factorization + triangular solves
   converge; compare against builtin ilu0.

Compile/link alone is explicitly NOT acceptance -- must show convergence +
residual match + determinism on GPU 1.

## Quirks log -> notes.md; generalizable cuSPARSE->hipSPARSE lessons ->
PORTING_GUIDE changelog.

## Out of scope

- VexCL/ViennaCL/MPI/pyamgcl backends (unchanged).
- Performance tuning of the HIP SpMV (correctness-first mechanical port; the
  rowsplit alg is the deterministic baseline, adaptive could be faster later).
- Block-valued (static_matrix) CUDA paths beyond what solver.cpp exercises.
