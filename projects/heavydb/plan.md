# Plan: heavydb (HeavyAI / OmniSciDB GPU SQL database)

## Project
- Name: heavydb
- Upstream: https://github.com/heavyai/heavydb
- Default branch: master
- Clone head at planning: da23f70ea060f308bb9536ec8515a4ef1d07a4bd (2025-06-17)
- Kind: GPU-accelerated columnar SQL analytics database SERVER (not a library). ~111 MB tree.
- Lead platform: linux-gfx90a (MI250X, ROCm 7.2.1).

## TL;DR / disposition: BLOCKED -- not a mechanical MOAT port (upstream compiler-backend scale)

HeavyDB does not ship static GPU kernels for queries. It JIT-COMPILES every SQL query to GPU
code through LLVM: it builds LLVM IR, stamps the NVPTX target, lowers via the LLVM NVPTX backend
to PTX, JIT-links the PTX against a precompiled device-runtime fatbin with the CUDA driver linker,
then loads and launches via the CUDA driver API. A ROCm port is therefore a compiler-backend
retarget (NVPTX -> AMDGPU code object + HIP module load/launch), not a CUDA->HIP symbol
translation. On top of that, the codegen is written against LLVM 14 APIs that are REMOVED in the
ROCm LLVM (22.0.0git) on this host, so the existing path does not even compile -- let alone run --
against the only LLVM that has the AMDGPU target. There is no isolated GPU kernel unit test that
can pass before the whole codegen+runtime path works for a query, so no correctness-first slice
exists for the MOAT validator to gate on.

Recommended action: set linux-gfx90a `blocked` with the reason below and MOVE ON (per CLAUDE.md:
genuinely unclear/upstream-scale strategy). Do NOT silently `triage.py skip` it -- the staged path
below is real engineering that jeff may choose to fund as a standalone project; blocked keeps it
visible. If jeff prefers it off the board entirely, the skip reason is `cant-port`.

## Existing AMD support: NONE (abandoned non-NVIDIA backend was Intel L0, not AMD)

No HIP/ROCm/AMDGPU code path anywhere (the "hip" grep hits are substrings: ship/chip/relationship).
What exists is the inert residue of an abandoned Intel GPU effort:
- `L0Mgr/` -- a Level Zero (Intel oneAPI) device+memory manager, gated `ENABLE_L0` (default OFF;
  the default build compiles `L0MgrNoL0.cpp`, a no-op stub). Built (`add_subdirectory(L0Mgr)`,
  CMakeLists.txt:1104) but referenced nowhere outside `L0Mgr/` -- never wired into DataMgr or the
  executor. Dead/vestigial.
- No SPIR-V or any non-NVPTX codegen path exists (grep for spirv in QueryEngine is empty).
- Upstream history shows the Intel backend as a series of CLOSED, UNMERGED PRs against master
  (#671/#683/#688/#689/#690/#691/#697, branches alexb_intel/*). All merged=false. Only the inert
  L0 memory-manager stub landed.

Decision: abandoned non-NVIDIA backend, abandoned at vendor/upstream scale. A ROCm/HIP port would
add value in principle (no HIP path exists), but the cost is compiler-engineering scale and is not
GPU-validatable in the pipeline today -> blocked, staged path documented.

## Build classification: pure CMake (NOT a pytorch extension) -- but Strategy A does not fit

- `CMakeLists.txt:254-256`: `option(ENABLE_CUDA ... ON)` then `enable_language(CUDA)`;
  `CMAKE_CUDA_ARCHITECTURES` (273-344). No `find_package(Torch)` for the GPU path (Torch is an
  optional CPU ML-inference dep, `HAVE_CUDA_TORCH`, 1083-1087), no torch.utils.cpp_extension, no
  CUDAExtension. By the PORTING_GUIDE rule this is Strategy A territory.
- BUT Strategy A (a cuda_to_hip.h compat header + `.cu` marked LANGUAGE HIP) addresses only the
  8 hand-written `.cu` files. The dominant GPU surface is EMITTED AT RUNTIME by an LLVM backend,
  which a compat header cannot touch. Strategy B does not apply (no torch extension machinery).

## Port strategy: neither A nor B; the real work is a new LLVM-AMDGPU codegen backend

The dominant surface is the runtime query codegen + the CUDA-driver-coupled JIT/runtime. That is a
re-architecture (a second, AMD codegen path inside HeavyDB + an LLVM toolchain reconciliation + a
HIP runtime-compile/module path), not a translation. The 8 static `.cu` files (the only
Strategy-A-shaped surface) are downstream of the unportable codegen and cannot be exercised alone.

## THE CRUX: NVPTX -> AMDGPU JIT retarget (make-or-break, dominant effort) -- VERIFIED

The query-execution codegen path, end to end, with verified citations:

(a) IR build + NVPTX stamp. `QueryEngine/NativeCodegen.cpp`:
  - `llvm_module->setDataLayout("e-p:64:64:64-...")` then
    `setTargetTriple("nvptx64-nvidia-cuda")` (1347-1352); a second NVPTX triple string at 1078.
  - Kernels marked by `nvvm.annotations` "kernel" metadata (1374-1383).
  - `linkModuleWithLibdevice` links `<cuda>/nvvm/libdevice/libdevice.10.bc` (1196);
    `check_module_requires_libdevice` keys on the `__nv_*` prefix (333-338).
  - NVVMReflect activation + `nvvm-reflect-ftz` / `nvptx-f32ftz` fn attrs (1289-1304).
  - `legalize_nvvm_ir` strips NVPTX-illegal IR (stacksave/restore/lifetime/PHI) (1026 ff).
(b) PTX emission. `initializeNVPTXBackend` (1623): `TargetRegistry::lookupTarget("nvptx64")` (1633)
    -> `createTargetMachine("nvptx64-nvidia-cuda", deviceArchToSM(arch), ...)` (1638).
    `generatePTX` (1592): `nvptx_target_machine->addPassesToEmitFile(..., CGFT_AssemblyFile)` (1613).
    The TargetMachine is carried in `struct GPUTarget { llvm::TargetMachine* nvptx_target_machine;
    const CudaMgr* cuda_mgr; ... }` (CodeGenerator.h:106-111) -- there is NO backend abstraction;
    NVPTX is hardcoded into the struct and the codegen body.
(c) Runtime PTX->cubin JIT + module load/launch. `QueryEngine/NvidiaKernel.cpp`:
    `ptx_to_cubin` (132) uses the CUDA driver JIT linker -- `cuLinkCreate` (141),
    `cuLinkAddFile(CU_JIT_INPUT_FATBINARY, cuda_mapd_rt.fatbin)` (62, 98-106, 156-164),
    `cuLinkAddData(PTX)` (175), `cuLinkComplete` (186); then `cuModuleLoadDataEx`/
    `cuModuleGetFunction` (218). Launch in `DeviceKernel.cpp`: `cuLaunchKernel` (73),
    `cuOccupancyMaxPotentialBlockSize` (51), `cuFuncSetAttribute` (38), takes a `CUstream` (130).
    `create_device_kernel` returns ONLY `NvidiaKernel` under `#ifdef HAVE_CUDA`, else nullptr.

The AMD analogue, function by function:
  - triple `amdgcn-amd-amdhsa`, the AMDGPU datalayout, `amdgpu_kernel` calling convention +
    `!amdgpu.*` metadata instead of `nvvm.annotations`, AMDGPU address spaces.
  - swap libdevice (`__nv_*` + libdevice.10.bc) for ROCm device-libs (ocml/ockl + the oclc_*
    control bitcodes), incl. the `__ocml_*`/`__ockl_*` math ABI; drop NVVMReflect/`nvptx-f32ftz`.
  - emit an AMDGPU relocatable code object (not PTX text), then link it against a PREBUILT
    device-runtime code object (the AMD equivalent of cuda_mapd_rt.fatbin) via comgr or lld
    `--whole-archive`, and load via `hipModuleLoadData`/`hipModuleGetFunction`/
    `hipModuleLaunchKernel`. hipRTC is an alternative front for the compile half but does not change
    the link-against-prebuilt-runtime requirement.
  - re-express the `GPUTarget`/`DeviceKernel`/`create_device_kernel` factory to carry an AMD
    TargetMachine and return an `AmdKernel` (this is exactly the multi-backend factory work the
    abandoned Intel L0 path attempted).

Effort: this is a multi-week-to-multi-month compiler-backend project (a real second codegen target
inside HeavyDB + the runtime plumbing). It is the dominant cost and the make-or-break item.

## SECOND WALL (verified, independent of the retarget): LLVM 14 -> ROCm LLVM 22 API gap

HeavyDB pins LLVM 14.0.6 (`scripts/common-functions.sh:345`, `common-functions-rockylinux.sh:398`;
the conda recipe's `llvmdev <11` is stale). The only LLVM on this host with the AMDGPU target is
the ROCm LLVM 22.0.0git (`/opt/rocm/llvm`, `llvm-config --targets-built` -> "AMDGPU X86"; NVPTX is
NOT present). The codegen uses LLVM-14 APIs that are REMOVED in LLVM 22 -- confirmed on disk:
  - `#include <llvm/Transforms/IPO/PassManagerBuilder.h>` (NativeCodegen.cpp:51) -- the header does
    NOT exist under `/opt/rocm/llvm/include`; `llvm::PassManagerBuilder` is gone (legacy PM builder
    removed). Used at 1249, 1354, 1358.
  - `llvm::CGFT_AssemblyFile` (470, 1614) -- removed; LLVM 22 has `enum class CodeGenFileType`
    (`CodeGen.h:117`), so it is now `CodeGenFileType::AssemblyFile`.
  - `llvm::legacy::PassManager` for codegen (464, 503, 1357, 1610, 3293) -- still present but the
    surrounding pass-pipeline setup changed.
  - `StringRef::startswith`/`endswith` (24 sites across QueryEngine) -- renamed `starts_with`/
    `ends_with`.
Net: HeavyDB's QueryEngine will not compile against the ROCm LLVM 22 as-is. Either modernize the
codegen to the LLVM 22 API (the AMDGPU target lives only there), or build a custom LLVM that has
BOTH NVPTX and the full ROCm AMDGPU lowering at HeavyDB's pinned version and matching ROCm's
device-libs ABI -- a project-level toolchain problem either way. (PORTING_GUIDE jitify->hipRTC /
LLVM-codegen lesson family; this is the same class but heavier because the whole query engine, not
one kernel, rides on the in-process LLVM.)

## CUDA surface inventory

1. Runtime LLVM NVVM/NVPTX query codegen -- the crux above (NativeCodegen.cpp, 3579 LOC).
2. PTX->cubin JIT + module load/launch -- the crux (NvidiaKernel.cpp 231 LOC, DeviceKernel.cpp).
3. Device runtime library, TARGET-AGNOSTIC at compile time and the most portable piece:
   - `RuntimeFunctions.cpp` is compiled by the HOST LLVM `clang++` with `-emit-llvm` to a generic
     (NOT GPU-targeted) `RuntimeFunctions.bc` (QueryEngine/CMakeLists.txt:366-376); the codegen
     clones functions from it into each query module and only THEN applies the NVPTX triple. So the
     runtime bitcode itself is not NVPTX-bound; the target-specificity is all in NativeCodegen.cpp.
   - `cuda_mapd_rt.cu` (1427 LOC) is separately nvcc-compiled `-fatbin -rdc=true` into
     `cuda_mapd_rt.fatbin` (QueryEngine/CMakeLists.txt:471-497) and JIT-linked per query. These are
     the `extern "C" __device__` runtime fns the codegen calls (get_thread_index, pos_start_impl,
     agg_*, hashing). Warp size is PARAMETERIZED (`thread_warp_idx(warp_sz)`); warp intrinsics are
     nearly absent (only a /32,%32 bit-array word index ~cuda_mapd_rt.cu:1305). Atomics are the live
     fault surface: atomicCAS / atomicMin / atomicAdd for group-by aggregation (cuda_mapd_rt.cu
     ~713-986). On AMD this needs the fatbin path replaced by an AMDGPU code-object built by the
     same retargeted toolchain.
4. Hand-written Thrust/CUB sort + hash-join kernels (map to rocThrust/hipCUB, mostly drop-in per
   prior ports -- rocThrust is a true Thrust drop-in): ResultSetSortImpl.cu, InPlaceSortImpl.cu,
   TopKSort.cu, SortUtils.cuh, JoinHashTable/Runtime/HashJoinRuntimeGpu.cu (also atomicCAS at
   40/46/75/91/388), DataMgr Thrust allocators.
5. warpcore (ThirdParty/warpcore) -- header-only NVIDIA cooperative-groups GPU hash table, used by
   AggModeHashTableGpu (the SQL MODE() aggregate) via QueryMemoryInitializer / QueryExecutionContext.
   Cooperative groups + `__shfl`/FULL_MASK + `--extended-lambda` (CMakeLists.txt:260). Per
   PORTING_GUIDE, HIP cooperative groups (ROCm 7.2.1) lack cg::reduce/labeled_partition; warpcore is
   heavier than what HIP CG provides, so it needs a HIP port or a portable-hash-table rewrite. It is
   per-query-conditional (only when a query has a kMODE aggregate), so it can be stubbed/disabled for
   an initial slice.
6. CUDA driver/runtime memory + management. CudaMgr/CudaMgr.cpp uses 34 distinct driver symbols
   incl. the VIRTUAL MEMORY MGMT api (cuMemCreate / cuMemMap / cuMemAddressReserve / cuMemSetAccess /
   cuMemExportToShareableHandle / cuMemGetAllocationGranularity), plus cuCtx*/cuModuleLoadDataEx/
   cuMemcpy*/cuMemsetD/cuStreamSynchronize. ~68 distinct cu*/CU* symbols across ~70 files outside
   ThirdParty; `CUstream`/`CUdeviceptr`/`CUcontext` are threaded through executor HEADERS (Execute.h,
   QueryEngine.h, ResultSet.h, QueryMemoryInitializer.h) -- so the driver-API port is NOT isolated to
   `.cu` TUs; it touches host C++ broadly. HIP has hipCtx/hipModule/hipStream and a VMM api
   (hipMemCreate etc.), so the symbols map, but the abstraction is CUDA-only and a third manager +
   factory wiring is required (the abandoned L0 work).

## Risk / fault-class list

FC-1 (blocking) -- LLVM target retarget NVPTX -> AMDGPU (the crux, section above). New compiler
  engineering inside HeavyDB, not symbol aliasing.
FC-2 (blocking) -- LLVM toolchain reconciliation: AMDGPU target must be registered in the SAME
  in-process LLVM the whole server links (`llvm_map_components_to_libnames(... ${LLVM_TARGETS_TO_BUILD}
  ...)`, CMakeLists.txt:751), AND that LLVM is v22 here while HeavyDB's code is LLVM-14-era (does not
  compile against 22; PassManagerBuilder.h absent, CGFT_* renamed). Two coupled toolchain problems.
FC-3 (blocking) -- runtime PTX->cubin JIT has no drop-in HIP analogue; must be re-expressed as
  AMDGPU code-object + comgr/lld link against a prebuilt runtime code object + hipModule* (new
  runtime plumbing, not a rename).
FC-4 (large) -- CUDA-driver-API runtime port across ~70 files incl. executor headers; needs a third
  DeviceKernel/manager backend + factory wiring; CudaMgr uses the VMM api.
FC-5 (large) -- warpcore cooperative-groups hash table: HIP CG lacks cg::reduce/labeled_partition;
  port warpcore to HIP or rewrite AggMode against a portable hash table. Per-query-conditional, so
  deferrable for a first slice.
FC-6 (medium) -- Thrust/CUB sort + hash-join kernels -> rocThrust/hipCUB (mostly drop-in) but sit
  behind the CudaMgr/driver coupling.
FC-7 (correctness, applies once any AMD device code runs):
  - wave64 (PORTING_GUIDE): the static runtime parameterizes warp_sz, but every hand-written
    reduction/word-index (cuda_mapd_rt.cu /32,%32; any TopKSort/HashJoin reduction) must use a
    per-arch wavefront constant (64 on gfx90a), never a literal 32.
  - int atomicMin/atomicMax SILENTLY DROPPED on coarse-grained / hipMallocManaged memory on
    gfx90a/CDNA2 (PORTING_GUIDE, cudaKDTree). The group-by aggregation in cuda_mapd_rt.cu relies on
    atomicMin (and the agg buffers may be managed/coarse-grained), so MIN/MAX aggregates could
    no-op silently. Emulate int min/max with an atomicCAS loop on HIP; micro-test on hipMalloc vs
    hipMallocManaged.
  - hipCUB DeviceRadixSort with a nonzero begin_bit is broken on gfx90a (PORTING_GUIDE, cudaKDTree):
    sort full key width. Relevant to TopKSort/InPlaceSort if they sort key sub-ranges.
FC-8 (validation) -- no isolated GPU kernel unit test; GPU correctness is observed only by running
  a SQL query through the full executor in GPU device mode, which needs FC-1..FC-3 all solved
  simultaneously. No correctness-first slice for the MOAT validator (section below).

## Why there is no tractable scoped gfx90a slice for the MOAT validator

A MOAT port wants a first slice that GPU-validates. HeavyDB has none:
- The hand-written `.cu` runtime (the only Strategy-A-shaped surface) is never executed on its own;
  it is JIT-linked against runtime-generated PTX. With no AMDGPU codegen (FC-1) and no AMD JIT path
  (FC-3) it cannot be exercised by any query.
- The query GPU code is GENERATED, so there is no static kernel to translate-and-test.
- The smallest validatable unit is "one SQL query runs on GPU", which requires the entire
  codegen+JIT+runtime+memory stack on AMD at once.
This is precisely the NVVM/NVPTX-codegen coupling to assess, and it is where the analogous Intel
backend died.

## Dependency situation (heavy server build; obtainable, but multi-hour from source)

Pinned versions on master (scripts/common-functions.sh): LLVM 14.0.6, Apache Arrow 18.1.0,
Thrift 0.20.0, Boost 1.86.0, GDAL 3.10.3, PROJ 9.6.0, TIFF 4.7.0, AWS-SDK 1.11.517, plus
TBB/BLOSC/Snappy/CURL/gperftools/LibArchive/RdKafka/GEOS/LibKML, optional oneDAL/MLPACK/Armadillo,
optional Torch, and Calcite (Java/Maven, the SQL parser/optimizer). Provisioned via
scripts/mapd-deps-*.sh and a conda recipe (scripts/conda/meta.yaml). Obtainable but this is a
from-source SERVER build (hours), and the LLVM pin (14) collides with the ROCm LLVM (22) needed for
AMDGPU -- so the dependency stack is itself part of FC-2, not a side issue.

## Inter-project deps

None. HeavyDB consumes no other MOAT project, and (being blocked) nothing should depend on it.
No `set-deps` action.

## File-by-file change list (for the staged path, NOT a mechanical diff)

Stage 0 (toolchain, no AMD yet): modernize QueryEngine to the ROCm LLVM 22 API so HeavyDB even
  compiles against it -- NativeCodegen.cpp (PassManagerBuilder removal, CGFT_*-> CodeGenFileType::,
  pass-pipeline), 24 startswith/endswith sites, any other LLVM-14-only API. Build the rest of the
  server (Arrow 18 / Thrift / Boost / Calcite) CPU-only first.
Stage 1 (static surface, behind a USE_HIP/ENABLE_HIP option): the 8 `.cu` files + a cuda_to_hip.h
  for the driver symbols in CudaMgr.cpp/DeviceKernel.cpp/CudaAllocator/ThrustAllocator/GpuMemUtils;
  rocThrust/hipCUB for the sort/hash-join kernels; wave64 + atomic-min/max-CAS fixes (FC-7). NOTE:
  not runnable on GPU until Stage 2.
Stage 2 (the crux): add an AMDGPU codegen path in NativeCodegen.cpp/CodeGenerator.h (GPUTarget ->
  carry an AMD TargetMachine; amdgcn triple/datalayout; amdgpu_kernel metadata; ocml/ockl device-
  libs link; AMDGPU code-object emit) + an AmdKernel + create_device_kernel factory branch + the HIP
  module-load/launch path (NvidiaKernel.cpp analogue). Build cuda_mapd_rt as an AMDGPU code object.
Stage 3: warpcore HIP port or AggMode portable-hash-table rewrite (FC-5); re-enable MODE().

## Build commands (current, CUDA reference; there is no working gfx90a configure yet)

There is no buildable gfx90a configuration at planning time (FC-1/FC-2 unsolved). The reference
CUDA configure is roughly:
  bash scripts/mapd-deps-ubuntu.sh   # provisions the dep tarball (LLVM 14, Arrow 18, Thrift, ...)
  cmake -DENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=<sm> ... && make
A gfx90a configure would, after Stage 0-2, look like `-DENABLE_CUDA=OFF -DENABLE_HIP=ON
-DCMAKE_HIP_ARCHITECTURES=gfx90a` against a single LLVM 22 that has the AMDGPU target -- but that
option and that codegen path do not exist yet; creating them is the project.

## Test plan (documented; NOT runnable without the port)

GPU gate: `Tests/ExecuteTest.cpp` (31207 LOC) runs the full SQL query matrix in both device modes;
`skip_tests(ExecutorDeviceType::GPU)` returns true unless `QR::get()->gpusPresent()` (line 222-226),
and the harness compares GPU results against CPU/SQLite references. Validation would be: build with
the AMD GPU backend, start a HeavyDB instance on the gfx90a host, run ExecuteTest in
ExecutorDeviceType::GPU mode asserting GPU == CPU for the matrix, plus CodeGeneratorTest /
BumpAllocatorTest / the CudaMgr-equivalent. Non-GPU regression set (must not regress): the CPU
device half of ExecuteTest and the storage/catalog/parser tests (91 test executables in Tests/).
None is reachable until FC-1..FC-3 are solved, so it cannot be a MOAT gate now.

## Recommendation (scope + staging, honest)

BLOCK linux-gfx90a now (reason below), surface to jeff, and move on. If funded, treat it as a
standalone multi-week-to-multi-month compiler project with the four stages above; the realistic
first milestone is NOT a passing GPU test but "Stage 0 compiles HeavyDB against ROCm LLVM 22 +
Stage 1 static `.cu` ported", followed by the Stage 2 NVPTX->AMDGPU retarget that is the true
make-or-break. A correctness-first mechanical port (the usual MOAT first slice) is not achievable
because the GPU code is generated and there is no isolated kernel test.

Blocked reason (concrete): HeavyDB JIT-compiles every SQL query to GPU code via an in-process LLVM
NVVM->NVPTX pipeline (NativeCodegen.cpp setTargetTriple "nvptx64-nvidia-cuda" / nvvm.annotations /
libdevice.10.bc / lookupTarget("nvptx64") / addPassesToEmitFile CGFT_AssemblyFile) whose PTX is
JIT-linked to cubin by the CUDA driver (NvidiaKernel.cpp cuLink*/cuModuleLoadDataEx) and launched
via the CUDA driver API (DeviceKernel.cpp cuLaunchKernel). A ROCm port requires (1) a new
LLVM-AMDGPU codegen path (amdgcn triple, ocml/ockl device-libs, AMDGPU code-object emit), (2) a
HIP runtime-compile/module-load path replacing the CUDA driver JIT linker, (3) a port of the
CUDA-driver-coupled runtime across ~70 files incl. executor headers + a third DeviceKernel backend,
and (4) a warpcore CG-hash-table port. It also does not compile against the ROCm LLVM 22 on this
host (its codegen uses LLVM-14 APIs removed in 22: PassManagerBuilder.h absent, CGFT_* ->
CodeGenFileType), and that LLVM 22 is the only one here with the AMDGPU target (NVPTX absent). There
is no static GPU kernel and no isolated GPU unit test, so no correctness-first slice exists for the
MOAT validator. The analogous non-NVIDIA backend (Intel SPIR-V + Level Zero, PRs
#671/#683/#688-691/#697) was attempted by Intel and CLOSED UNMERGED; only an inert L0 memory-manager
stub remains in master. This is upstream compiler-backend engineering, not a mechanical MOAT port,
and is not GPU-validatable in the pipeline as-is.

## Open questions (for jeff)

- Fund as a standalone compiler-backend project, or skip (`triage.py skip heavyai/heavydb --reason
  cant-port`)? It is not a per-repo mechanical MOAT port.
- If funded: pin the LLVM story first -- modernize HeavyDB's codegen to ROCm LLVM 22, or build one
  custom LLVM carrying both NVPTX and full ROCm AMDGPU lowering at a fixed version with a matching
  device-libs ABI? This decision gates everything downstream.
