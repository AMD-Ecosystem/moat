# MMseqs2 port plan (linux-gfx90a lead)

## Project
- Name: MMseqs2
- Upstream: https://github.com/soedinglab/MMseqs2
- Default branch: main
- GPU surface: the vendored `lib/libmarv` aligner (a fork/derivative of cudasw4),
  built only when `-DENABLE_CUDA=1`. Exposed to MMseqs2 through `Marv` (lib/libmarv/src/marv.h)
  and driven by `src/commons/GpuUtil.*`, `src/prefiltering/ungappedprefilter.cpp`,
  `src/util/gpuserver.cpp`. The CPU search/clustering toolkit is unaffected by the port.

## DISPOSITION: tractable clean HIP port (Strategy A). Dispatch a porter.
Effort class: medium (not a trivial colmap-style header job, but well below a CK rewrite).
The whole GPU SIMD surface is funneled through one abstraction layer (`MathOps<T>` in
`lib/libmarv/src/mathops.cuh`, ~19 intrinsic sites, plus ~5 in `kernelhelpers.cuh`/`convert.cuh`),
so the port is "supply AMD implementations of a closed set of ~14 SIMD/half2 intrinsics behind the
existing abstraction" + the usual Strategy-A glue, NOT a rewrite of the 9.7k lines of kernels.
This is explicitly NOT a CK/MFMA-deferred case (see Turing+ root cause).

## Existing AMD support
NONE. Confirmed:
- `gh api repos/soedinglab/MMseqs2/forks --paginate`: no fork under ROCm/AMD/GPUOpen orgs; no fork
  with rocm/hip/amd in the name (only a coincidental `sarahalamdari/MMseqs2`, a personal fork).
- No upstream rocm/hip branch; no ROCm-DS-style separately-named AMD project.
- WebSearch ("MMseqs2 ROCm/AMD GPU/HIP"): only NVIDIA GPU coverage (Nature Methods 2025,
  NVIDIA NIM blogs). README states GPU support requires "CUDA-enabled GPUs of the Turing generation
  or newer"; the precompiled GPU binary needs an NVIDIA driver.
Decision: from-scratch HIP port adds clear value. No authoritative or community AMD effort to adopt.

## Build classification: pure CMake (Strategy A). Evidence:
- Top CMakeLists.txt: `set(ENABLE_CUDA 0 CACHE BOOL "Enable CUDA")` (line 22);
  `if (ENABLE_CUDA) ... add_subdirectory(lib/libmarv/src EXCLUDE_FROM_ALL)` (lines 297-301).
- `lib/libmarv/src/CMakeLists.txt`: `project(... LANGUAGES CXX CUDA)`, `add_library(marv ...)`,
  `CUDA_SEPARABLE_COMPILATION ON`, `CUDA_RESOLVE_DEVICE_SYMBOLS ON`, NVCC_FLAGS `-rdc=true
  --extended-lambda --expt-relaxed-constexpr`. Standalone CMake + `.cu`; no Torch, no setup.py.
=> Strategy A (compat header + `enable_language(HIP)`, mark `.cu` LANGUAGE HIP, gate on a HIP option).

## THE DECIDING QUESTION -- why "Turing+"? (root cause)
NOT tensor cores, NOT CUTLASS/CuTe, NOT mandatory DPX, NO inline PTX in the compute path, NO __dp4a.
Verified by grep across lib/libmarv/src: no `wmma`/`nvcuda`/`mma.sync`/`fragment<`, no
`#include <cutlass>`/`<cute/>` (the "cute" hits are substrings of "execute"/"executeCopy"); the only
inline PTX (`ptx_wrappers.cuh`) is an sm120f-and-CUDA>=13.2-guarded Blackwell int8 path that is
compiled out everywhere else and is never the default.

The real reason for Turing+ is the SIMD-within-a-32-bit-word integer path the Smith-Waterman /
gapless score recurrences are built on. `mathops.cuh` `MathOps<T>` uses:
- short2 (16x2) gapless/SW-int: `__vadd2`, `__vmaxs2`, `__vibmax_u16x2`, `__vimax3_s16x2`,
  `__viaddmax_s16x2`, `__viaddmax_s16x2_relu` (the latter four are the NVIDIA "DPX" family:
  hardware on Hopper sm90, emulated by CUDA on Turing/Ampere -- which is exactly why "Turing+",
  not "Hopper-only").
- int (scalar 32-bit) SW path: `__vibmax_s32`, `__vimax3_s32`, `__viaddmax_s32`, `__viaddmax_s32_relu` (DPX).
- u8x4 (8x4) gapless: `__vmaxu4`, `__vadd4`, `__vaddus4`, `__vsubus4` (pre-DPX SIMD video intrinsics).
- half2 gapless: `__hadd2`, `__hmax2`.

Portability verdict on ROCm 7.2.1 (checked /opt/rocm/include and a compile probe):
- ALL of `__vadd2/__vmaxs2/__vmaxu4/__vadd4/__vaddus4/__vsubus4` and the entire DPX family
  (`__vimax3_*`, `__viaddmax_*`, `__vibmax_*`) are MISSING -- HIP provides no SIMD-in-a-word
  integer intrinsics at all.
- `__hadd2` is present, but `__hmax2`/`__hmin2` are MISSING; only scalar `__hmax`/`__hmin` exist.
- `__reduce_max_sync` IS present (used in `kernelhelpers.cuh` under `__CUDA_ARCH__>=800`).
These are all SIMT scalar/packed-integer ops with trivial, well-defined scalar fallbacks
(unpack the 2x16 / 4x8 lanes, do per-lane signed/unsigned saturating add/max/min, repack). No
hardware feature is required; on AMD they lower to ordinary integer ALU + a couple of bit ops.
This is a portable-SIMT port, NOT a DPX/tensor-core-bound CK reimplementation.

The gapped Smith-Waterman kernel is templated on ScoreType in {float, int} (static_assert,
pssmkernels_smithwaterman.cuh:81): the `float` instantiation uses only `max`/`+` (zero intrinsics,
already portable); the `int` instantiation uses the scalar-32 DPX min/max above.

## CUDA-surface inventory + ROCm mapping
- SIMD/half2 intrinsics: emulate the ~14 missing ops inside `MathOps<T>` behind `#if defined(USE_HIP)`
  (or a small `marv_simd_amd.cuh` the abstraction includes). Closed set, ~24 call sites total
  (mathops.cuh 19, kernelhelpers.cuh 4, convert.cuh 1). This is the bulk of the real work.
- Cooperative groups: `cg::tiled_partition<groupsize>` with groupsize in {4,8,16}, `static_assert(groupsize<=32)`,
  `blocksize % groupsize == 0` (pssmkernels_*.cuh). These are LOGICAL sub-warp tiles <=32 lanes and are
  arch-agnostic per PORTING_GUIDE (width-32 logical-warp ops are fine on wave64). cg + cg::reduce are
  supported by rocPRIM-backed HIP cooperative_groups. No full-warp / wave-width assumption in the tiling.
- Warp shuffles: `__shfl_up_sync`/`__shfl_down_sync`/`__shfl_sync` always called with an explicit
  width = `group_size` (<=32) and reductions hardcode offset start 16 (a 32-wide logical reduce).
  Arch-agnostic; HIP has all three. No `__ballot`/`__activemask`/`__popc` in live code (the one
  `__ballot` block in `hpc_helpers/cuda_helpers.cuh` is commented out).
- Thrust: used in cudasw4.cuh, dbbatching.cuh, custom_thrust_allocators.cuh, util.cuh -> rocThrust
  (1:1 under HIP; `thrust::` namespace unchanged).
- Dynamic shared memory: `cudaFuncSetAttribute(..., cudaFuncAttributeMaxDynamicSharedMemorySize, ...)`
  (kernels.cuh, pssmkernels_gapless.cuh/_int8.cuh, pssmkernels_smithwaterman.cuh) -> hipFuncSetAttribute.
  Watch gfx90a max dynamic shared mem per block (64 KB) vs the per-config tile sizes; some large-tile
  configs may exceed it and must be filtered out of the AMD config list (see Risks).
- Streams/events, pinned host memory (`cudaMallocHost`/`cudaHostAlloc`), peer access: 1:1 hip* equivalents.
- IPC: gpuserver shares the padded DB via a memory handle (`Marv::getDbMemoryHandle`,
  `setDbWithAllocation`); maps to hipIpcGetMemHandle/hipIpcOpenMemHandle. Validate the single-process
  `--gpu 1` path first; the gpuserver IPC path is a secondary gate.
- No textures/surfaces, no cuBLAS/cuFFT/cuRAND/cuSPARSE, no cooperative grid launch, no managed memory.
- Compute-capability config selection (`gapless_kernel_config.cuh getOptimalKernelConfigs_gapless`,
  and the SW analogue): a hard `if(ccMajor==7&&ccMinor==5)...else default` ladder keyed on
  `cudaDevAttrComputeCapabilityMajor/Minor`. On AMD this returns junk cc; add an AMD branch that
  selects a conservative config set (start from the sm75 list, then prune by shared-mem/occupancy)
  and feeds the half2 path first for bring-up.

## Risk list
- Wave size: LOW. No wave-width geometry is baked in (tiles <=32 logical lanes; shuffles use explicit
  width<=32; reductions are 32-wide logical). gfx90a is wave64 but each 32-or-less tile lives within one
  wavefront. STILL: when followers reach gfx1100/gfx1151 (wave32) verify the <=32 tiles are intact
  (they should be, since width is explicit). Keep any emulation lane-count tied to the SIMD word
  (2x16, 4x8), never to warpSize.
- DPX semantics exactness: the emulation must reproduce signed/unsigned SATURATION and the relu
  (`__viaddmax_*_relu` = max(a+b, c, 0)) and the "which-lane-is-max" boolean outputs
  (`__vibmax_*` returns per-lane max + a bool of whether a>=b). The SW traceback/end-position uses the
  bool outputs; getting saturation or the >= tie-break wrong corrupts scores/end positions silently.
  Validate against the CPU reference, not just "runs".
- Dynamic shared memory ceiling: gfx90a allows up to 64 KB dynamic shared per block; large-tile configs
  tuned for NVIDIA's 100-227 KB opt-in may exceed it. The AMD config list must drop configs whose
  shared-mem request > device max; a too-large `hipFuncSetAttribute` will fail at launch.
- `-rdc=true` + `CUDA_SEPARABLE_COMPILATION ON`: maps to HIP `-fgpu-rdc` with a device-link step.
  Confirm the HIP RDC link works for this many TUs; if the host image is large, apply
  `--offload-compress` (PORTING_GUIDE large-libraries note). Single target arch for bring-up.
- Missing-return / `__CUDA_ARCH__`-bodyless-kernel UB: the cc-ladder and the `#if __CUDA_ARCH__>=800`
  reduce path must compile to a valid AMD body (no empty `__device__` function that returns garbage on
  AMD). Audit the `ptx_wrappers.cuh` stub pattern (`extern __device__ void ...are_only_available...()`)
  -- ensure the AMD build never instantiates the Blackwell int8 path.
- Float exactness / fp-contract: SW float path may drift vs CPU gold by FMA contraction; pin
  `-ffp-contract=on` per PORTING_GUIDE if a bit-exact score compare fails (scores are integers in the
  default BLOSUM path, so this is likely a non-issue; note for the float SW configs).
- `__hmax2` emulation half-precision: emulate as `make_half2(__hmax(a.x,b.x), __hmax(a.y,b.y))`; confirm
  NaN/-0 selection matches CUDA `__hmax2` (CUDA __hmax propagates the non-NaN operand).

## File-by-file change list (lead, gfx90a)
- `CMakeLists.txt` (top): add `USE_HIP` option; when set, drive libmarv via HIP (keep ENABLE_CUDA path).
- `lib/libmarv/src/CMakeLists.txt`: under USE_HIP, `enable_language(HIP)`, set the `.cu` sources
  LANGUAGE HIP, `HIP_ARCHITECTURES` from `${CMAKE_HIP_ARCHITECTURES}` (no hardcoded gfx90a literal),
  translate NVCC_FLAGS (`--extended-lambda`->default on clang, `--expt-relaxed-constexpr`->n/a,
  `-rdc=true`->`-fgpu-rdc`), link rocThrust, add `--offload-compress` if needed.
- NEW `lib/libmarv/src/cuda_to_hip.h` (compat header): alias cudaXxx->hipXxx for the runtime symbols
  libmarv uses (malloc/free/memcpy/stream/event/funcattr/host-alloc/ipc/peer), included by the .cu/.cuh.
- NEW `lib/libmarv/src/marv_simd_amd.cuh` (or inline in mathops.cuh under USE_HIP): scalar emulations of
  `__vadd2/__vmaxs2/__vibmax_u16x2/__vimax3_s16x2/__viaddmax_s16x2[_relu]`,
  `__vibmax_s32/__vimax3_s32/__viaddmax_s32[_relu]`, `__vmaxu4/__vadd4/__vaddus4/__vsubus4`, `__hmax2`.
- `lib/libmarv/src/mathops.cuh`, `kernelhelpers.cuh`, `convert.cuh`: route the ~24 intrinsic sites to the
  emulations under `#if defined(USE_HIP)`; leave CUDA spelling otherwise.
- `lib/libmarv/src/gapless_kernel_config.cuh` (+ SW config analogue) and the cc-selection in cudasw4.cuh:
  add an AMD branch returning a shared-mem-safe config set.
- `ptx_wrappers.cuh`: ensure the Blackwell int8 path stays disabled on AMD.
- src/ GPU glue (GpuUtil.*, gpuserver.cpp, ungappedprefilter.cpp): expected to need little/none beyond
  the build wiring, since they call through `Marv`. Audit for any cuda* runtime calls and route via the
  compat header.

## Build commands (gfx90a)
Dependencies: ROCm 7.2.1 (have), rocThrust (ships with ROCm), zlib, OpenMP. Install any missing apt pkg.
    cmake -S . -B build-hip -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a \
      -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ -DCMAKE_BUILD_TYPE=Release
    cmake --build build-hip -j16 --target mmseqs
Follower (no source change): `-DCMAKE_HIP_ARCHITECTURES=gfx1100` (or `gfx90a;gfx1100` fat binary).
Multi-arch correctness check: `llvm-objdump --offloading build-hip/.../libmarv* ` shows both code objects.

## Test plan (real GPU)
Bundled data, ZERO external egress required (host is egress-limited ~40-160 KB/s):
- `examples/QUERY.fasta` (300 KB) and `examples/DB.fasta` (11 MB) ship in the repo.
GPU correctness gate (a CPU-vs-GPU consistency diff -- libmarv has no separate gold, but MMseqs2's CPU
ungapped/SW search IS the reference the GPU path must reproduce):
    mmseqs createdb examples/DB.fasta targetDB
    mmseqs makepaddedseqdb targetDB targetDB_padded
    # GPU path (exercises libmarv gapless + Smith-Waterman):
    mmseqs easy-search examples/QUERY.fasta targetDB_padded gpu.m8 tmp_gpu --gpu 1
    # CPU reference, same params:
    mmseqs easy-search examples/QUERY.fasta targetDB cpu.m8 tmp_cpu --gpu 0
    # Compare hit IDs + scores (allowing benign ordering): the GPU score column must match CPU.
Pass = GPU hits/scores match the CPU reference on the bundled QUERY/DB. Also run libmarv's own `align`
executable on `lib/libmarv/allqueries.fasta` (in-repo) as a unit-level SW/gapless smoke if useful.
Non-GPU regression set that must NOT regress: the upstream `util/regression` harness (regression_*.sh)
and `data/` outputs (PAM/VTML matrices, CPU search) -- run a representative subset built WITHOUT
ENABLE_CUDA/USE_HIP to confirm the CPU toolkit is untouched.
Cross-arch consistency (followers): diff gfx1100 GPU output against gfx90a GPU output for the same input
(deterministic), not just "plausible" -- per PORTING_GUIDE wave-size validation rule.

## Open questions
- gfx90a 64 KB dynamic-shared ceiling vs the NVIDIA tile configs: which configs survive? (porter: probe
  `hipDeviceAttributeMaxSharedMemoryPerBlockOptin` and prune.) Likely forces the smaller-tile sm75-like set.
- Does easy-search `--gpu 1` require the gpuserver/IPC path, or does it run in-process for a single search?
  (Determines whether hipIpc must work for the first validation or only later.)
- rocThrust + `-fgpu-rdc` device-link scale for the marv TUs: link clean, or need `--offload-compress`?
- Exact CUDA semantics of `__viaddmax_s16x2_relu` and `__vibmax_*` (saturation + tie-break + bool output)
  to mirror precisely; cross-check against the CUDA math API docs during emulation.
