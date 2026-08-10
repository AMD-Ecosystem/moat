# gpu4pyscf -- ROCm/HIP port plan (lead platform: linux-gfx90a, MI250X, ROCm 7.2.1)

## Project
- Name: gpu4pyscf
- Upstream: https://github.com/pyscf/gpu4pyscf
- Default branch: master
- Pinned base for this plan: 72087c7 ("Optimize J/K build for molecular code", #773)
- What it is: GPU plugin for PySCF (quantum chemistry). GPU-accelerated electron-repulsion
  integrals (ERIs), direct SCF + density fitting, DFT (LDA/GGA/mGGA/hybrid), analytic gradients
  and Hessians, TDDFT, PCM/SMD solvent, ECP, PBC. ~347k LOC of .cu/.cuh plus a large CuPy-based
  Python layer.

## Existing AMD support: NONE -> fresh CUDA-to-ROCm port (proceed)
- No HIP/ROCm/AMD code anywhere in the tree (grep for hip/HIP/rocm/gfx/USE_HIP/__HIP_PLATFORM
  finds only false positives in unrelated words). No remote branch mentions hip/rocm/amd.
- No OpenCL/Vulkan/SYCL path either. This is a CUDA-only project. A ROCm/HIP port adds real value.
- Disposition: proceed with a correctness-first mechanical port of the kernel libraries +
  a CuPy-ROCm/library-binding port of the Python layer. NOT a skip.

## Build classification: pure-CMake CUDA libraries + CuPy Python plugin (NOT a torch extension)
ext_type = `cmake-cupy-plugin` (set in upstream.json/status.json).

Evidence:
- `setup.py` line 63-89 `CMakeBuildPy(build_py)`: configures `cmake -S gpu4pyscf/lib -B ... -DBUILD_LIBXC=OFF`
  then `cmake --build`. No `torch.utils.cpp_extension`, no `CUDAExtension`, no `find_package(Torch)`
  anywhere. setup.py install_requires (line 141-148): `pyscf`, `cupy-cuda12x`, `gpu4pyscf-libxc-cuda12x`,
  `pyscf-dispersion`, `geometric`. No torch dependency.
- `gpu4pyscf/lib/CMakeLists.txt` line 16: `project (gpu4pyscf C CXX CUDA Fortran)`; builds ~12 SHARED
  libraries (gint, gvhf, gvhf_md, gvhf_rys, gdft, cupy_helper, sem, solvent, pbc, mgrid, mgrid_v2, gecp)
  from `.cu` (+ a handful of `.c`) sources. These `.so` are loaded from Python via ctypes
  (`lib/utils.py:31 load_library`), NOT linked into a Python extension module.
- The Python layer is **CuPy-first**: arrays are `cupy.ndarray`; cuBLAS/cuSOLVER/cuSPARSE/cuFFT/cuTENSOR
  are reached through `cupy_backends.cuda.libs.*` and `cupyx`, with two extra direct `ctypes.CDLL`
  loads (`lib/cublas.py`, `lib/cusolver.py`).

This is therefore **Strategy A** for the CUDA libraries (compat header + `enable_language(HIP)` +
`.cu` LANGUAGE HIP), plus a **CuPy-ROCm environment port** for the Python layer (no hipify; CuPy-ROCm
already maps CuPy's cuBLAS/cuSOLVER/cuSPARSE backends to hipBLAS/rocSOLVER/hipSPARSE).

## CuPy-ROCm dependency assessment (the make-or-break, and it is GOOD on this host)
- **CuPy-ROCm is already installed**: `cupy_rocm_7_0-14.1.0` in the py_3.12 conda env. Its backend
  `cupy_backends/cuda/libs/cusolver.cpython-*.so` links **librocsolver.so / libhipblas.so / librocblas.so**
  (confirmed via ldd). So `from cupy_backends.cuda.libs import cusolver/cublas/cusparse` is backed by
  ROCm libraries with zero source change to gpu4pyscf's call sites. This is exactly the "ROCm-CuPy build"
  analogue the task anticipated, and it exists on the box.
- **It is currently broken only by a numpy ABI mismatch** (cupy 14.1.0 built for numpy 2.x; the env has
  numpy 1.26.4 -> `numpy.core.multiarray failed to import`). This is an environment fix (align numpy to
  the version cupy_rocm_7_0 expects, in a dedicated env), NOT a fundamental blocker. The porter's first
  bringup task is a clean env where `import cupy` succeeds and `cupy.zeros(...).sum()` runs on gfx90a.
- pyscf is not installed in this env (`pip install pyscf>=2.8.0` needed) -- trivial, pure-Python CPU dep
  that also provides the CPU reference path for validation.

Conclusion: gpu4pyscf does NOT need its custom `.cu` libraries rewritten to use CuPy's RawKernel; it needs
(a) its own `.cu` libraries built under HIP (Strategy A), and (b) the CuPy-ROCm runtime for the array/BLAS/
solver layer. Both legs are feasible on this host.

## CUDA surface inventory
Kernels: ~175 `.cu` + ~28 `.cuh` across the 12 libraries; pure `__global__`/`__device__` kernels for
ERIs (Rys quadrature + McMurchie-Davidson), DFT grid/XC evaluation, ECP, PBC lattice sums, multigrid.

- Warp intrinsics: **149 `__shfl_down_sync(mask, val, offset)`** calls, ALL of the 16-lane-subgroup form
  (no `__shfl_xor`, no `__shfl_up`, no explicit-width `__shfl`). Concentrated in **gvhf-md** (the
  McMurchie-Davidson J-engine: unrolled_md_j.cu 42, unrolled_md_j_4dm.cu 90, md_contract_j.cu 8) plus a
  few scattered (ecp/common.cu 5, pbc/ft_ao.cu 1, solvent/pcm.cu 1, gvhf-md/contract_int3c2e.cu 2,
  pbc/int3c2e_create_tasks.cuh 2). **gvhf-rys (the primary ERI engine) uses NO `__shfl`** -- it reduces
  via shared memory + `gout_stride`/`nsq_per_block` striding and `atomicAdd`, which is wave-agnostic.
- `__ballot` x1, `__any/__all/__popc` x4 (small), `warpSize` x8 -- minor.
- Hardcoded `WARP_SIZE 32` (#define) in multigrid/multigrid.cuh, pbc/int3c2e.cuh, pbc/ft_ao.cuh,
  pbc/int3c2e_create_tasks*.cuh; `THREADS 32`/`THREADSX 32` etc in several files. Used BOTH as a
  shared-memory stride/SIMD-group width AND as a lane count -- must be audited per use (see Risks).
- Atomics: **16,377 `atomicAdd`** (double-precision ERI/Fock digestion; supported natively on gfx90a) +
  1 `atomicOr`. No int `atomicMin/atomicMax` (so the cudaKDTree coarse-grained-atomic-drop class does
  NOT apply). No atomicCAS spin-locks (so the RXMesh wave64 deadlock class does NOT apply).
- Textures / surfaces / cudaArray / managed memory / hostRegister: **NONE**. The entire popsift/colmap/
  CudaSift texture fault-class family (pitch, layered arrays, linear filtering, rule-of-five on tex
  handles, 256B pitch) is OUT OF SCOPE here.
- Pinned memory: 3 references (host-staging); maps 1:1 to hipHostMalloc.
- Streams/events: standard `cudaStream_t` / launch on `cupy.cuda.get_current_stream().ptr` passed as
  void* into the .so; maps 1:1 (hipStream_t is the same handle ABI under CuPy-ROCm).
- Separable compilation: gint sets `CUDA_SEPARABLE_COMPILATION ON` (check others) -> needs HIP
  `-fgpu-rdc` + `HIP_SEPARABLE_COMPILATION ON` if device symbols cross TU (RXMesh lesson).

Library dependencies (Python layer, via CuPy-ROCm unless noted):
- **cuBLAS -> hipBLAS/rocBLAS**: via `cupy_backends.cuda.libs.cublas` (CuPy-ROCm-backed, no change).
  PLUS one direct `lib/cublas.py:20 ctypes.CDLL('libcublas.so')` whose handle is **never used** (the file
  only grabs a handle behind a "add modified cublas function here" comment). Hardcoded `libcublas.so` will
  raise on import under ROCm -> make the load lazy/optional or point at `libhipblas.so` under ROCm.
- **cuSOLVER -> hipSOLVER/rocSOLVER**: `lib/cusolver.py` does BOTH `cupy_backends.cuda.libs.cusolver`
  (CuPy-ROCm-backed) AND a direct `find_library('cusolver') -> ctypes.CDLL` calling
  `cusolverDnDsygvd/cusolverDnZhegvd` (generalized dense eigensolver) + `dpotrf/zpotrf` (Cholesky).
  **GOOD NEWS vs raft-lanczos**: the symbols this code needs DO exist in hipSOLVER 7.2.1 -- confirmed in
  `/opt/rocm/lib/libhipsolver.so`: `hipsolverDnDsygvd`, `hipsolverDnDsygvd_bufferSize`, `hipsolverDnChegvd`,
  `hipsolverDn{D,C}potrf`, plus syevd/syevj. These are the DEVICE eigensolvers (the `Dn` API), NOT the
  cuSOLVER `*Host`/LAPACK-on-host variants that raft-lanczos found missing -- so the raft-lanczos gap
  does NOT bite the eigh path here. Port `lib/cusolver.py` by: load `libhipsolver.so` and call
  `hipsolverDnDsygvd*` with the hipSOLVER signature (it takes an extra trailing devInfo arg layout very
  close to cuSOLVER; verify the bufferSize/solve prototypes against hipsolver-dense docs). MAX_EIGH_DIM
  (32-bit lwork overflow guard) stays.
- **cuTENSOR -> NO CuPy-ROCm binding (the real gap)**: `lib/cutensor.py` imports
  `cupy_backends.cuda.libs.cutensor` + `cupyx.cutensor`. **CuPy-ROCm 14.1.0 ships NEITHER** (confirmed:
  `ModuleNotFoundError: cupy_backends.cuda.libs.cutensor`, no `cupyx/cutensor`). ROCm DOES ship
  `/opt/rocm/lib/libhiptensor.so.0.1.70201` but it is v0.1 (early, partial cuTENSOR-2.x coverage) and is
  NOT exposed through CuPy. **Mitigation is built in**: `lib/cutensor.py:136-168` has a designed fallback
  -- when the cutensor import fails it sets `contract_engine='cupy'` and routes `contract()` through
  `cupy.einsum`. So the port is functionally correct WITHOUT cuTENSOR (just slower tensor contractions).
  This is the right correctness-first choice. Do NOT attempt to wire hiptensor v0.1 into CuPy in the first
  pass. Flag a perf follow-up.
- **cuSPARSE -> hipSPARSE**: via `cupy_backends.cuda.libs.cusparse` (CuPy-ROCm-backed). No project .cu uses
  cuSPARSE directly. Generic-API cuSPARSE through CuPy maps cleanly (amgcl lesson: hipSPARSE mirrors
  cuSPARSE name-for-name). Low risk.
- **cuFFT/cuRAND -> hipFFT/hipRAND**: only via CuPy (`cupy.fft`, `cupy.random`), no direct project use.
- **CUTLASS -> NO ROCm PORT (must reimplement; second hard wall)**: `cupy_helper/grouped_gemm.cu` and
  `cupy_helper/grouped_dot.cu` include `cutlass/gemm/device/gemm_grouped.h` etc. (CMake fetches
  NVIDIA/cutlass v3.4.0 under `option(BUILD_CUTLASS ON)`). PORTING_GUIDE is explicit: CUTLASS does not
  port to ROCm. These two functions implement grouped GEMMs over lists of varying-size matrices
  (`grouped_dot`: C=A@B.T; `grouped_gemm`: C=A.T@B, M<128 case) and are called from `dft/numint.py`
  (DFT XC numerical integration, lines 622/635/655/780/792/807) with **no Python fallback**. Options
  (decide in porter, correctness-first): (1) reimplement as a per-group loop of hipBLAS `gemm`/`gemmStridedBatched`
  calls (simplest, correct, modest perf); (2) reimplement against Composable Kernel grouped GEMM
  (`DeviceGroupedGemm` / ck_tile) for perf -- classic CK is acceptable here per the raft note. First pass:
  hipBLAS per-group loop (or a cupy-side replacement of grouped_dot/grouped_gemm by `cupy.matmul` in a
  Python loop) to unblock the DFT path; CK rewrite is a perf follow-up.
- **libxc (GPU) sub-dependency**: DFT XC functionals run on GPU via libxc. The pip wheel
  `gpu4pyscf-libxc-cuda12x` is CUDA-only; setup.py defaults to `-DBUILD_LIBXC=OFF` and uses that wheel.
  The CMake alternative builds `wxj6000/libxc` with `-DENABLE_CUDA=ON` from source. There is NO prebuilt
  ROCm libxc wheel. So **DFT (any xc) needs libxc-GPU itself ported to HIP** -- this is a separate CUDA
  codebase (wxj6000/libxc fork). Scope decision: stage HF (no libxc) first; DFT requires the libxc-HIP
  build and is a second milestone (see Staged strategy). Treat libxc-HIP as a sibling effort, not in the
  gpu4pyscf .so build.

## Risk list (ordered by severity)
1. **wave64 16-lane-subgroup `__shfl_down_sync` (THE primary correctness wall).** gvhf-md kernels do
   `sq_id = tx + 16*ty; lane_id = thread_id%32; group_id = lane_id/16; mask = 0xffff << (group_id*16);`
   then `for(offset=8; offset>0; offset/=2) v += __shfl_down_sync(mask, v, offset);` -- a reduction over
   a 16-lane subgroup, two subgroups packed into a 32-lane CUDA warp. Two failures on gfx90a:
   (a) compile: ROCm 7.x `__shfl_down_sync` static_asserts a **64-bit** mask, so the 32-bit literal
   `0xffff << ...` will not compile (AutoDock-GPU lesson) -- even before correctness.
   (b) semantics: a 64-lane wavefront holds **four** 16-lane groups, but `lane_id = thread_id%32` and the
   16-bit mask only describe two groups in the low 32 lanes -> lanes 32-63 mis-partitioned, wrong J matrix.
   Fix (popsift two-32-rows / RXMesh sub-word-ballot family, generalized to 16): keep the reduction
   WIDTH-LOCAL to 16. Cleanest is a HIP path using width-parameterized `__shfl_down(v, offset, 16)` (HIP's
   non-sync shuffle takes a width and stays within the 16-lane group on any wave size), USE_HIP-guarded,
   leaving the CUDA `__shfl_down_sync(mask,...)` untouched. Recompute `group_id`/subgroup membership from
   the true wave width (4 groups on wave64, 2 on wave32) so all lanes participate. Validate with the
   J-matrix fingerprint test (`test_get_j`/`test_get_jk`, `lib.fp(vj)` to 7 d.p. + max-abs-diff vs CPU
   PySCF to 7 d.p.) AND a fixed-seed run-to-run determinism check.
2. **CUTLASS grouped GEMM (second wall).** grouped_dot/grouped_gemm have no fallback; reimplement
   (hipBLAS per-group loop first, CK later). Gates the DFT numint path; HF does not use them.
3. **libxc-GPU is CUDA-only and unported.** Gates all DFT (xc). HF-only validation avoids it; full DFT
   needs a libxc-HIP build (separate codebase). State the milestone boundary clearly.
4. **cuTENSOR has no CuPy-ROCm binding.** Handled by the built-in cupy.einsum fallback (correctness ok,
   perf hit). Do not block on hiptensor v0.1.
5. **`WARP_SIZE 32` / `THREADS 32` hardcodes used as shared-mem strides.** In multigrid/pbc, `WARP_SIZE`
   indexes shared-memory arrays (`cx[lx*WARP_SIZE]`) -- a STRIDE/tile width, not necessarily a hardware
   lane count. Where it is a pure layout stride with no cross-lane op it can stay 32 (a tile width is wave
   agnostic, cf. fused-ssim lesson), but where the same kernel also does `thread_id % WARP_SIZE` lane
   partitioning it must track the real wave size. Audit each `WARP_SIZE` use site: layout-only -> leave;
   lane-partition -> wave-size-aware. Do not blanket-redefine to 64.
6. **`cusolverDn*` -> `hipsolverDn*` signature drift.** hipSOLVER sygvd/hegvd/potrf exist (confirmed) but
   the ctypes prototypes in cusolver.py (argtypes, devInfo placement, handle from
   `cupy.cuda.device.get_cusolver_handle()` -- which under CuPy-ROCm is a rocSOLVER/hipSOLVER handle) must
   be matched to hipSOLVER's actual ABI. Verify the handle returned by CuPy-ROCm is accepted by a directly
   `dlopen`ed libhipsolver (handle provenance must match the library you call). Prefer routing through
   `cupy_backends.cuda.libs.cusolver` (already ROCm) where it exposes sygvd, falling back to a direct
   libhipsolver load only if needed.
7. **Hardcoded `libcublas.so` / `find_library('cusolver')` in the Python layer** raise on a ROCm box.
   Make these loads ROCm-aware (libhipblas.so / libhipsolver.so) or lazy/guarded. `lib/cublas.py`'s handle
   is dead -- safest to make its load optional.
8. **Build-scale large `.so` (offload-compress).** gvhf-rys has very large unrolled TUs (rys_roots_dat.cu
   50k LOC, unrolled_rys_jk_ip1.cu 34k, unrolled_ejk_ip1.cu 27k LOC) compiled per-arch. Single-arch
   (gfx90a) fatbins are smaller than multi-arch, but the unrolled angular-momentum kernels can still
   produce large objects. If any single `.so` overruns the +/-2 GiB x86-64 relocation reach at link, add
   `--offload-compress` to the HIP compile options (cudf lesson) BEFORE splitting. Build a single target
   arch (no multi-arch fatbin) and check `size -A` on the biggest objects.
9. **`enable_language(HIP)` + Fortran in the project() line.** CMakeLists declares `CUDA Fortran` but the
   port can likely drop Fortran (used only by an optional libxc-from-source build with ENABLE_FORTRAN=OFF
   anyway). Under USE_HIP, gate the project languages to `C CXX HIP` (no CUDA, no Fortran) so cmake does
   not require a CUDA toolkit.
10. **ecp/CMakeLists arch parsing.** `foreach(arch IN LISTS CMAKE_CUDA_ARCHITECTURES) REGEX MATCH [0-9]+`
    will FATAL_ERROR on `gfx90a` (no numeric id). Under USE_HIP, bypass the Blackwell/CUDA-version block
    entirely (it is a CUDA<13.1 nvcc workaround, irrelevant to HIP).
11. **`--ptxas-options=-v` in gint/ecp CMAKE_CUDA_FLAGS** is an nvcc flag; must not leak into the HIP
    compile. Gate the CUDA flag blocks under the non-HIP branch.
12. **fp-contract / fast-math drift.** clang(HIP) defaults to `-ffp-contract=fast`; the J/K fingerprint
    tests compare to 7 d.p. against the CPU PySCF path. If the fingerprints drift ~1 ULP, pin
    `-ffp-contract=on` in CMAKE_HIP_FLAGS (CV-CUDA lesson). The double-precision atomicAdd accumulation is
    order-nondeterministic on ANY GPU, so a max-abs-diff to 7 d.p. vs CPU is the right bar, not bitwise.
13. **Boys/Rys special functions** (`boys_fn`, gamma_inc, rys_roots) use double-precision exp/erf/sqrt.
    gfx90a has fast f64; watch the `__fsqrt_rn` 1-ULP class only if any single-precision sqrt feeds an
    exact compare (the integral path is f64, so low risk). Validate by energy/fingerprint, not bit-exact.

## File-by-file change list (lead bringup)
CMake (Strategy A; minimal-footprint, follow colmap/MPPI lessons):
- `gpu4pyscf/lib/CMakeLists.txt`: add `option(USE_HIP ...)`; under USE_HIP set project languages to
  `C CXX HIP` (drop CUDA/Fortran), `enable_language(HIP)`, default `CMAKE_HIP_ARCHITECTURES` to gfx90a
  ONLY WHEN UNSET (never a literal -- CudaSift/Gpufit lesson, so gfx1100/gfx1151 need only the cache var);
  set `BUILD_CUTLASS OFF` and `BUILD_LIBXC OFF` defaults under HIP; provide a top-level add_library
  override OR per-subdir `set_source_files_properties(<.cu> PROPERTIES LANGUAGE HIP)` to retag .cu.
- One compat header `gpu4pyscf/lib/cuda_to_hip.h` (the only file that knows HIP): alias the cuda* symbols
  the .cu actually use (cudaMalloc/Free/Memcpy*/Memset*/Stream*/Event*/GetDevice/DeviceSynchronize/
  funcSetAttribute/getLastError/etc.), define a 64-bit full-warp mask and a width-aware subgroup helper,
  include <hip/hip_runtime.h>; on CUDA it is a no-op `#include <cuda_runtime.h>`. Force-include it on every
  HIP TU via CMAKE_HIP_FLAGS `-include .../cuda_to_hip.h` (MPPI lesson) so defines precede use.
- Per-subdir CMakeLists (gint, gvhf, gvhf-rys, gvhf-md, gdft, sem, solvent, pbc, multigrid, ecp,
  cupy_helper): gate `--ptxas-options=-v` and CUDA-only flags under non-HIP; map
  `CUDA_SEPARABLE_COMPILATION`/`CUDA_ARCHITECTURES` props to the HIP equivalents (HIP_SEPARABLE_COMPILATION,
  HIP_ARCHITECTURES); fix ecp arch-parse loop under HIP.
- `cupy_helper/CMakeLists.txt`: under USE_HIP drop the cutlass dependency + include dirs; build the two
  grouped-GEMM TUs from their reimplemented (hipBLAS/CK) sources or exclude them and provide a Python-side
  fallback first.

Kernel sources (USE_HIP-guarded, CUDA path byte-identical):
- gvhf-md/unrolled_md_j.cu, unrolled_md_j_4dm.cu, md_contract_j.cu, contract_int3c2e.cu;
  ecp/common.cu; pbc/ft_ao.cu, pbc/int3c2e_create_tasks*.cuh; solvent/pcm.cu:
  replace the 16-lane `__shfl_down_sync(mask,...)` reductions with the width-16 HIP shuffle path and
  wave-aware subgroup partition (the central wave64 fix).
- multigrid/*.cu, pbc/*.cuh: audit `WARP_SIZE` use sites (layout stride vs lane partition); fix only the
  lane-partition uses.
- cupy_helper/grouped_gemm.cu, grouped_dot.cu: reimplement (hipBLAS grouped/strided-batched gemm or CK).

Python layer:
- `lib/cublas.py`: make the `libcublas.so` load ROCm-aware/lazy (handle is unused).
- `lib/cusolver.py`: load `libhipsolver.so` under ROCm and call `hipsolverDn{D,Z}sygvd*`/`hipsolver{D,Z}potrf`
  with verified prototypes; prefer the CuPy-ROCm `cusolver` backend where it exposes sygvd.
- `lib/cutensor.py`: no edit required -- the cupy.einsum fallback triggers automatically when the cutensor
  import fails on ROCm. (Optional: add an explicit ROCm note.)
- Environment (not a code change, but a bringup prerequisite): a conda env where `cupy_rocm_7_0` imports
  (align numpy), pyscf installed, `CUPY_ACCELERATORS` set without cutensor under ROCm.

## Build commands (gfx90a)
Environment bringup (porter, record exact steps in notes.md):
- Use/repair a conda env so `python3 -c "import cupy; cupy.zeros(4).sum()"` runs on gfx90a (fix the
  numpy/cupy_rocm_7_0 ABI mismatch); `pip install 'pyscf>=2.8.0' pyscf-dispersion geometric`.

Configure + build the kernel libraries (Strategy A):
```
cmake -S gpu4pyscf/lib -B build/temp.gpu4pyscf \
  -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DBUILD_CUTLASS=OFF -DBUILD_LIBXC=OFF -DBUILD_SOLVENT=ON
cmake --build build/temp.gpu4pyscf -j 8
```
Then `export PYTHONPATH=$PWD:$PYTHONPATH` (the .so are emitted into gpu4pyscf/lib via
LIBRARY_OUTPUT_DIRECTORY). Followers add only `-DCMAKE_HIP_ARCHITECTURES=gfx1100|gfx1151`.

A CPU-only `rocm/dev-ubuntu-24.04:7.2.4-complete` docker build is a fine manual compile smoke check only;
it is NOT a validation gate (no GPU) and must not be wired into the fork's Actions.

## Test plan
Canonical invocation (from .github/workflows/unittest.yml + run_ci.sh): after building and setting
PYTHONPATH, `pytest -m 'not slow and not benchmark and not special'`. Run **serially** on a single
assigned GCD (one GPU; per MPPI lesson avoid hammering one GPU with -jN).

GPU validation gate (what "validated on gfx90a" means), staged:
- Milestone 1 (HF / integrals, no libxc, no CUTLASS): `scf/tests/test_rhf.py`, `test_uhf.py`,
  `test_scf_jk.py`, `test_scf_j_engine.py`, `scf/tests/test_int2c2e.py`, `test_int4c2e.py`. These assert
  `lib.fp(vj)`/`lib.fp(vk)` against hardcoded references to **7 decimal places** AND `abs(vj-refj).max()`
  vs the CPU PySCF path (`mf.to_cpu().get_jk`) to 7 d.p. PASS here = the ERI engine (gvhf-rys) and the
  wave64-fixed J-engine (gvhf-md) are correct. Add a fixed-seed run-to-run determinism check on vj/vk
  (two runs equal to print precision) to prove the wave64 reduction fix is race-free.
- Milestone 2 (DFT): requires libxc-HIP and the grouped-GEMM reimplementation. `dft/tests/test_rks.py`,
  `test_uks.py`, `test_numint.py`, `test_ao_values.py`, `test_grids.py`. Bar: total SCF/DFT energy matches
  the CPU PySCF reference (and the in-test hardcoded reference) to chemical accuracy / ~1e-7 Hartree.
- Milestone 3 (gradients/Hessian/df/solvent/pbc/tdscf): the remaining ~200 test files, brought online as
  their kernel surfaces pass. df/tests (27) and grad/tests (20) are the next-highest-value sets.

Non-GPU regression set: the project is GPU-only; the "must not regress" baseline is the CPU PySCF
reference path each test compares against (`*.to_cpu()`), which runs on the host pyscf install and must
remain the trusted gold. There is no separate CPU-only unit suite in gpu4pyscf to regress.

Validation is on real gfx90a; lint and a CPU docker build are never the gate.

## Staged strategy + the hardest walls (recommendation)
Realistic scope: this is a LARGE port (12 CUDA libraries, a CuPy-heavy Python layer, two hard external
deps). Do NOT attempt the whole feature matrix at once. Recommended order:

1. **Env + build bringup**: CuPy-ROCm importing on gfx90a; kernel libs compile under HIP with CUTLASS and
   libxc OFF and the CUDA-only CMake flags gated. (Wall: build-system gating, ecp arch parse, possible
   --offload-compress on the big unrolled TUs.)
2. **HF / integral correctness (Milestone 1)** -- the core MOAT deliverable. Fix the wave64 16-lane
   `__shfl_down_sync` reductions in gvhf-md (THE hardest correctness wall) and validate J/K against the
   CPU PySCF fingerprints + determinism. gvhf-rys (shared-mem reduction) should pass with little change.
   Port lib/cusolver.py (hipSOLVER sygvd exists) and the libcublas/libcusolver Python loads.
3. **Grouped GEMM** reimplementation (hipBLAS loop first) to unblock the DFT numint path. (Second wall.)
4. **libxc-HIP** (separate codebase port) to unblock DFT functionals. (Third wall; may be its own MOAT
   project / a sibling fork.)
5. **DFT (Milestone 2)**, then gradients/Hessian/df/solvent (Milestone 3).

A correctness-first mechanical port is the right first pass for the kernels (per the guide). The two
perf-tuned NVIDIA pieces (CUTLASS grouped GEMM; the cuTENSOR contraction engine) get AMD-native treatment
(CK grouped GEMM; possibly hipTENSOR) only as a later perf pass -- correctness via hipBLAS-loop + cupy.einsum
fallback first. Mark the lead platform `planned`; a successful Milestone-1 (HF/integrals) GPU validation is
the minimum bar to call gfx90a a real port, with DFT/libxc as a clearly-scoped follow-on milestone.

## Open questions
- Does the porter scope the first validated deliverable at HF/integrals (Milestone 1) only, deferring
  DFT+libxc-HIP+grouped-GEMM to follow-on? (Recommended: yes -- it is a self-contained, GPU-validatable
  unit and the core of the project; full DFT is a multi-dependency effort.)
- libxc-GPU (wxj6000/libxc fork, CUDA): port it inside this effort or scaffold it as a separate MOAT
  project that gpu4pyscf depends_on? (It is a distinct codebase; a separate project + depends_on is cleaner.)
- Is the CuPy-ROCm `get_cusolver_handle()` handle directly usable by a `dlopen`ed libhipsolver, or must the
  sygvd call go through CuPy's own cusolver backend to keep handle provenance consistent? (Resolve early in
  Milestone 1.)
- grouped GEMM: hipBLAS per-group loop (simple, correct) vs CK DeviceGroupedGemm (perf) for the first pass?
  (Recommended: hipBLAS loop first, CK as a perf follow-up.)
