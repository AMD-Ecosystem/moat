# AutoDock-GPU -- ROCm/HIP port plan

Upstream: ccsb-scripps/AutoDock-GPU (branch `develop`, cloned at 28a6139).
Lead platform: linux-gfx90a (MI250X, CDNA2, wave64).

## Existing AMD support assessment

AutoDock-GPU ALREADY runs on AMD GPUs via its OpenCL backend (`DEVICE=OCLGPU`
or `DEVICE=OPENCL`); the OpenCL kernels live in `device/` and are selected by
`Makefile.OpenCL`. Per PORTING_GUIDE "assess existing AMD support", an existing
OpenCL path does NOT make this skip: a native HIP backend is still valuable
(typically faster than OpenCL on AMD, and shares the maintained CUDA kernel
sources rather than the parallel OpenCL copy). So this port adds a HIP backend
by porting the CUDA path; it does not touch the OpenCL or CPU paths.

Disposition: PORT (add HIP backend). Not already-supported (no HIP path),
not ported-elsewhere.

## Build classification

Pure Makefile project (no CMake, no pytorch/Torch dependency) -> Strategy A
(compat header + compile the existing `.cu` as HIP with hipcc). Confirmed: no
`find_package(Torch)`, no `torch.utils.cpp_extension`, no setup.py torch dep.

How DEVICE selects the backend (top-level `Makefile`):
- `DEVICE=CUDA` (or `DEVICE=GPU` when nvcc + a CUDA test compile succeed) ->
  `include Makefile.Cuda`: compiles `cuda/kernels.cu` with `nvcc` into
  `kernels.o`, then links it with the host `.cpp` (g++) and `-lcurand -lcudart`.
- `DEVICE=OCLGPU` / `DEVICE=OPENCL` (or `DEVICE=GPU` when CUDA is unavailable),
  `DEVICE=CPU` -> `include Makefile.OpenCL`: the OpenCL/CPU path.
- The `cuda/` directory is one TU: `cuda/kernels.cu` `#include`s the other `.cu`
  (`calcenergy.cu`, `calcMergeEneGra.cu`, `auxiliary_genetic.cu`,
  `kernel1..4.cu`, `kernel_ad.cu`, `kernel_adam.cu`).
- Host/device split in `host/src/performdocking.cpp`: CUDA-specific code is under
  `#ifdef USE_CUDA`, OpenCL under `#ifdef USE_OPENCL`. The only CUDA headers are
  in `host/inc/performdocking.h:34-36` (`cuda.h`, `curand.h`,
  `cuda_runtime_api.h`), included on the CUDA path only.

Add a new `DEVICE=HIP` that mirrors the CUDA path through hipcc.

## CUDA surface (what the HIP backend must cover)

Kernels (`cuda/*.cu`, all host-callable launchers + `__global__`):
- kernel1 initpop, kernel2 sum-evals, kernel3 Solis-Wets local search,
  kernel4 GA gen+eval (elitist + tournament/crossover/mutation),
  kernel_ad / kernel_adam ADADELTA/ADAM gradient local search,
  calcenergy / calcMergeEneGra energy+gradient evaluators.
- Runtime/library calls (host): `cudaMalloc/Free/Memcpy/MemcpyToSymbol/
  FromSymbol/GetDeviceProperties/SetDevice/DeviceSynchronize/MemGetInfo/
  DeviceReset/GetLastError`, `curand`-seeded host PRNG init (states are filled on
  host then uploaded; the device PRNG is a hand-rolled LCG in
  `auxiliary_genetic.cu`, no cuRAND device calls). `__constant__ GpuData cData`
  set via `cudaMemcpyToSymbol`.
- Device intrinsics: `__shfl_sync`, `__any_sync`, `__syncthreads`,
  `__threadfence`, `atomicAdd`, `__powf`. No `__ballot`/`__popc`/`__activemask`.
- Two inline-PTX helpers `llitoulli`/`ullitolli` (`asm("mov.b64 ...")`) -- NVIDIA
  PTX, must be replaced for HIP.

## Strategy A implementation

1. `cuda/cuda_to_hip.h` compat header (the ONLY file that knows HIP names).
   Under `USE_HIP`/`__HIP_PLATFORM_AMD__`: `#include <cstring>`/`<cstdlib>`
   BEFORE `<hip/hip_runtime.h>` (changelog gpuRIR: host memcpy/memset must bind
   to libc, not HIP device overloads), then alias the cuda* symbols the project
   uses to hip* (Memcpy, MemcpyToSymbol/FromSymbol, Malloc, Free,
   GetDeviceProperties, SetDevice, GetDevice, GetDeviceCount, DeviceSynchronize,
   DeviceReset, MemGetInfo, GetLastError, GetErrorString, the cudaMemcpyKind
   enums, `cudaError_t`/`cudaSuccess`, `cudaDeviceProp`, the printf/limit symbol
   if used). Authoritative name source: pytorch
   `torch/utils/hipify/cuda_to_hip_mappings.py`. On NVIDIA the header is a plain
   `#include <cuda_runtime.h>` so the CUDA build is byte-for-byte unchanged.
   Include it at the top of `cuda/kernels.cu`, `cuda/GpuData.h`, and gate the
   CUDA headers in `host/inc/performdocking.h` so the HIP build pulls the compat
   header / hipRAND instead of `cuda.h`+`curand.h`.

2. `Makefile.Hip` (new, mirrors `Makefile.Cuda`): `HIPCC = hipcc`, compile
   `cuda/kernels.cu` with `-x hip -DUSE_HIP --offload-arch=$(HIP_ARCH)` into
   `kernels.o`, link host `.cpp` (g++) with `kernels.o` + `-L$(ROCM)/lib
   -lamdhip64 -lhiprand`. Do NOT hardcode the arch: `HIP_ARCH ?= gfx90a`
   (default the lead arch only when unset) so a follower builds gfx1100/gfx1151
   with just `HIP_ARCH=<arch>` and no source/Makefile edit (PORTING_GUIDE +
   CudaSift/Gpufit changelog). Keep `TENSOR` off for HIP (NVIDIA tensor-core
   path; the `wmma_extension` submodule is not needed).

3. Top-level `Makefile`: add a `DEVICE=HIP` branch that sets `DEVICE:=GPU`
   (`-DGPU_DEVICE`, `-DN64WI` default like CUDA) and `include Makefile.Hip`,
   bypassing the nvcc `test_cuda.sh` probe entirely. CUDA/OpenCL/CPU branches
   untouched.

## Fault classes (PORTING_GUIDE) -- expected and how handled

- WARP SIZE / wave64 (the real risk; docking does heavy warp reductions). The
  code is partly warp-size-parameterized: kernels read `cData.warpmask` /
  `cData.warpbits`, but the HOST hardcodes them to 32-lane values
  (`performdocking.cpp:785-786`: `warpmask=31`, `warpbits=5`) and two device
  spots assume 32 lanes. gfx90a is wave64 -> fix natively (warpmask=63,
  warpbits=6), NOT as two 32-lane halves, because the reductions are genuinely
  warp-wide and combine per-warp partials via a block-wide `atomicAdd`, so a true
  64-lane warp is the natural and faster mapping and matches the upstream
  parameterization intent. Concretely:
  - Host `performdocking.cpp:785-786`: derive warpmask/warpbits from the device's
    `props.warpSize` at runtime (query `cudaGetDeviceProperties` for `cData.devid`)
    instead of the literals 31/5. Works for wave64 and wave32 followers with no
    further change.
  - `cuda/kernels.cu` `WARPMINIMUM2` (min-reduction): currently 5 XOR-exchange
    steps (masks 1,2,4,8,16 = 32 lanes). Add a 6th step (mask 32) under wave64
    via a `WARP_SIZE`-gated macro so it reduces all 64 lanes.
  - `cuda/kernels.cu` `REDUCEINTEGERSUM` / `REDUCEFLOATSUM` (sum-reductions): add
    the 6th shuffle step (`tgx ^ 32`) under wave64, and widen the `__shfl_sync` /
    `__any_sync` lane mask from `0xffffffff` (32-bit) to a 64-bit all-ones mask on
    wave64 (HIP masks are 64-bit on a 64-wide wavefront).
  - `cuda/kernel4.cu:93` `int blocks = blockDim.x / 32;` (number of warps for the
    final cross-warp reduction): replace the literal 32 with the per-arch
    `WARP_SIZE` constant.
  - Device-side `WARP_SIZE` constant (kWarpSize playbook): 64 for `__GFX9__`
    (gfx90a/gfx94x), else 32 (RDNA), else 32 (CUDA). `sBestEnergy[32]`/`sBestID[32]`
    static arrays in kernel4 hold one entry per warp; max warps/block =
    256/64 = 4 on wave64, so 32 is a safe upper bound (no resize needed).
- Inline PTX (`asm("mov.b64")` in `llitoulli`/`ullitolli`): replace with a
  portable bit-cast (`memcpy`/union) under HIP; these are pure 64-bit reinterprets.
- Atomics: only `atomicAdd` on int/float (not the silently-dropped int
  atomicMin/Max class from the cudaKDTree changelog), so no CAS emulation needed.
  Watch for any OOB; clamp if a fault appears.
- Library swap: cuRAND host seeding -> hipRAND (`-lhiprand`); device PRNG is a
  hand-rolled LCG, no rocRAND device dependency.
- Textures/surfaces: AutoDock-GPU uses none -> none of the texture fault classes
  apply.

Keep CUDA and OpenCL paths fully working; all HIP divergence sits behind
`USE_HIP` / the `DEVICE=HIP` path and the compat header (minimal footprint).

## Build + validation plan (gfx90a)

- Build: `make DEVICE=HIP NUMWI=64 HIP_ARCH=gfx90a` (NUMWI=64 maps one wavefront
  per block on wave64; also test 128). Pick a free GPU via `rocm-smi`, export
  `HIP_VISIBLE_DEVICES`. Artifacts (`bin/`, `kernels.o`) stay out of git.
- VALIDATE FOR REAL on gfx90a with the shipped `input/1stp` (streptavidin-biotin,
  the canonical AutoDock case) at a FIXED seed (`-seed <fixed>`), e.g.
  `bin/autodock_gpu_64wi -ffile input/1stp/derived/1stp_protein.maps.fld
  -lfile input/1stp/derived/1stp_ligand.pdbqt -nrun 10 -seed 42`. Check the
  result is PHYSICALLY correct: best binding energy (kcal/mol) and best-pose RMSD
  to the crystal ligand match the known-good reference for 1stp within docking
  run-to-run tolerance (1stp best energy is roughly -10 to -11 kcal/mol with
  sub-2A RMSD to the bound pose). Compile/lint is NOT validation. Cross-check
  against the OpenCL backend (`DEVICE=OCLGPU`) on the same input+seed as a
  same-machine reference if a published number is ambiguous.

## Out of scope / deliverables

- Do NOT fork/push the source; leave the port as working-tree edits under
  `projects/AutoDock-GPU/src` (parent delivers). Do NOT add GitHub Actions. Do
  NOT run gen_readme.py / commit README.md. Commit only
  `projects/AutoDock-GPU/{status.json,plan.md,notes.md}` and any PORTING_GUIDE
  changelog line via moatlib.commit_and_push.
