# RXMesh ROCm/HIP port plan (lead: linux-gfx90a, CDNA2 / wave64)

## Upstream
- owensgroup/RXMesh @ 30a4137 (clone in projects/RXMesh/src, gitignored).
- GPU-accelerated triangle-mesh data structure + processing library. Pure CMake
  (CPM/FetchContent), C++17, header-heavy (~48k LOC under include/), 65 .cu / 70
  .cuh / 126 .h. Static lib `RXMesh` (16 compiled TUs) + googletest unit tests +
  ~25 sample apps (Geodesic, MCF, VertexNormal, ARAP, Remesh, ...).

## Existing AMD support: NONE
- `grep -rniE 'hip[A-Z]|__HIP|USE_HIP|HIP_PLATFORM'` -> 0 hits. The only matches
  for amd/hip/wavefront/gfx are coincidental: "Wavefront OBJ" parser comment
  (cmake/rapidobj.cmake:5), "approximate minimum degree (AMD)" reordering algo
  name (cudss_cholesky_solver.h:79, NDReorder). CUDA-only project.
- Conclusion: a real CUDA->HIP port is warranted. Not skippable.

## Build classification: pure CMake -> Strategy A
- No find_package(Torch) / cpp_extension / torch dep. Root
  `project(RXMesh LANGUAGES C CXX CUDA)`; `.cu` sources; links CUDA::cusparse,
  CUDA::cusolver. -> Strategy A (compat header + `.cu` marked LANGUAGE HIP +
  USE_HIP option + arch from CMAKE_HIP_ARCHITECTURES, never a literal gfx90a).

## CUDA surface
- Warp intrinsics (the wave64 risk, this is a warp-cooperative mesh library):
  - `#define WARP_SIZE 32` (util/macros.h:42), used for `num_warps =
    blockThreads / WARP_SIZE` (cavity_manager_impl.cuh:2849).
  - `__ballot_sync(0xFFFFFFFF, pred)` stored into a **uint32_t**, with `lane_id =
    threadIdx.x % 32` and a per-32-element bitmask word `mask_id = id/32`, at:
    - kernels/dynamic_util.cuh:40 (warp_update_mask, dynamic deletion)
    - kernels/query_dispatcher.cuh:138 (static query participant bitmask)
    On wave64 a ballot returns 64 bits; storing in uint32 truncates lanes 32-63,
    and one 64-lane wavefront spans TWO 32-bit bitmask words. Classic
    two-32-rows-in-one-wavefront fault (popsift changelog). Fix per 32-lane
    sub-group.
  - `lane_id()` / `warp_id()` (kernels/util.cuh:225,232) use **inline PTX**
    `mov.u32 %0,%laneid` / `%warpid` -- CUDA-only asm, hard compile blocker under
    HIP. Replace with HIP-portable equivalents.
  - `__popc(x)` on 32-bit bitmask words (nd_patch.cuh:916, min_deg_patch.cuh:294)
    -- width-independent, safe.
  - `__syncwarp()` (dynamic_util.cuh:39) -- HIP no-op-safe.
- Cooperative groups: `#include <cooperative_groups.h>` x10, 183 cg:: uses. HIP
  ships hip/hip_cooperative_groups.h; alias the header in compat.
- Thrust (66 uses) / CUB (45 uses): rocThrust is a drop-in (/opt/rocm/include/
  thrust same API), hipCUB for cub:: (alias `#define cub hipcub`, headers
  cub/... -> hipcub/...). Needs -std=c++17 (already set).
- Math libs (matrix/diff path only): cuSolver (79), cuSparse (113), cuBLAS (49),
  cuDSS (67). hipSOLVER/hipSPARSE/hipBLAS present in ROCm 7.2.1; cuDSS has no HIP
  equivalent (RX_USE_CUDSS already OFF by default -- keep off).
- cuBQL: linked (`target_link_libraries(RXMesh PUBLIC cuBQL cuBQL_queries)`,
  fetched from NVIDIA/cuBQL) but **never #included in any source** -> drop the
  link for the HIP build, zero source impact.

## Coupling / scoping
- Core compiled TUs (rxmesh_static.cu, query.cu, rxmesh_dynamic.cu, attribute.cu,
  patch*.cu, lp_hashtable.cu, ...) do NOT include matrix/. The cuSolver/cuSparse/
  cuBLAS surface is confined to matrix/*.h + diff/gauss_newton_solver.h, plus the
  error-check wrappers + handle types pulled in unconditionally by
  util/macros.h (includes <cusolverSp.h>,<cusparse.h>) and the version banner in
  util/cuda_query.h.
- polyscope (viz, needs a display) is FetchContent + all uses guarded by
  USE_POLYSCOPE -> configure RX_USE_POLYSCOPE=OFF; those paths compile out.

## Strategy / scope
Strategy A, minimal footprint, CUDA path untouched, all HIP behind
USE_HIP/`__HIP_PLATFORM_AMD__`.

1. Compat header `include/rxmesh/util/cuda_to_hip.h`: on HIP include
   hip_runtime + hip_cooperative_groups + hipBLAS/hipSOLVER/hipSPARSE, alias the
   cuda*/cub spellings the project uses; on CUDA a no-op include of the CUDA
   runtime. Include <cstring>/<cstdlib> before hip_runtime (gpuRIR lesson).
2. CMake: `option(USE_HIP)`, gate `enable_language(HIP)` vs CUDA, mark the .cu
   sources LANGUAGE HIP, set HIP_ARCHITECTURES from ${CMAKE_HIP_ARCHITECTURES}
   (default gfx90a only when unset), drop cuBQL link + CUDA-only nvcc flags
   (-Xcudafe/-Xptxas/-rdc=true/--expt-*) under HIP, swap CUDA::cusparse/cusolver
   -> hip::hipsparse/hipsolver(/hipblas). Force RX_USE_POLYSCOPE off path.
3. Device-code fixes (the real work, all wave64):
   - per-arch warp constant (WARP_SIZE -> 64 on __GFX9__, else 32).
   - lane_id()/warp_id(): drop inline PTX on HIP, use portable form.
   - the two ballot bitmask sites: 64-bit ballot, write each 32-bit bitmask word
     from the lane-0 of its own 32-lane sub-group (treat the 64-lane wavefront as
     two independent 32-lane groups, popsift playbook).
4. Library swaps via the compat header (hipBLAS/hipSOLVER/hipSPARSE), watch
   handle types and a few v2 enum differences. Defer/skip cuDSS (no HIP lib;
   gated off). Clamp any OOB neighbor reads found while porting.

## Validation (GPU 0, HIP_VISIBLE_DEVICES=0, gfx90a)
- Build the RXMesh static lib + RXMesh_test (googletest) with
  -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a -DRX_USE_POLYSCOPE=OFF.
- Run the unit tests that exercise the warp-cooperative core against their
  built-in CPU references: queries (Oriented_VV, MultiQueries, EVDiamond,
  Iterator, BoundaryVertex, higher queries), attributes, for_each, validate,
  dynamic (RandomFlips/RandomCollapse, TriangleRefinement). Prove correctness vs
  the CPU/reference and determinism across repeated runs (the dynamic deletion
  path is exactly where the ballot bug would corrupt the bitmask non-
  deterministically). If the solver tests build cleanly with hipSOLVER/hipSPARSE,
  include them; otherwise the warp-core query/dynamic suite is the validation
  gate and the solver port is recorded as follow-up in notes.md.
- Compile/lint alone is NOT validation (MOAT rule).

## Risks
- Enormous transitive FetchContent (eigen, openmesh, metis, googletest, cuBQL,
  polyscope). Disable what is not needed (polyscope, cuBQL); build the core +
  tests target only.
- hipSOLVER/hipSPARSE API deltas vs cuSolverSp/cuSparse generic API may be large;
  if the solver surface does not port cleanly in scope, gate the matrix solver
  tests out and validate the mesh-query core (RXMesh's actual contribution).

## Outcome (porter, gfx90a) -- see notes.md for full detail

Done + GPU-validated: the static query + core mesh data-structure path. RXMesh
static lib and RXMesh_test build for gfx90a (HIP), and 21 tests pass on GPU 0:
all 8 query ops (x1000 iters vs CPU ref), Oriented_VV, EVDiamond, MultiQueries,
BoundaryVertex, Iterator, LPHashTable, Scan/AtomicMin/AtomicAdd/Tet/
BlockMatrixTranspose, and dynamic PatchScheduler/PatchLock/Validate. Determinism
holds across runs. Real wave64 work landed (ballot_sub_warp_32 for the 32-bit
participation/deletion bitmasks; the `__CUDA_ARCH__`->`__HIP_DEVICE_COMPILE__`
device/host-pointer fix was the key to the query path; inline PTX, `__CUDACC__`,
clang two-phase + attribute-on-instantiation fixes).

BLOCKED: dynamic mesh-EDITING tests (RandomFlips/RandomCollapse/
TriangleRefinement/PatchSlicing) HANG -- slice_patches spins in ShmemMutex::lock
on a garbage `__shared__` mutex pointer under HIP -fgpu-rdc. Two targeted fixes
did not stop it; needs deeper GPU debugging. cusolverSp sparse-solver path
(test_solver/sparse_matrix/jacobian + autodiff) excluded: no hipSOLVER equivalent.
