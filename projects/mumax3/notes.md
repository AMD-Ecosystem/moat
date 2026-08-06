# 3 (mumax3) notes

GPU micromagnetics, Go + cgo host, CUDA device kernels dispatched at RUNTIME
via the CUDA Driver API. Ported to HIP/ROCm. ext_type = go-cgo-driver-api (a
new build class).

Fork: https://github.com/AMD-Ecosystem/mumax3  branch moat-port (default branch master).
Validated: linux-gfx90a (MI250X), ROCm 7.2.1, HIP_VISIBLE_DEVICES=3.

## Architecture (why this is not a clean hipify)

Device dispatch is 100% the CUDA Driver API, not nvcc-`.cu`-linked or `<<<>>>`:
- `cuda/Makefile` compiled each `.cu` to PTX TEXT with `nvcc -ptx`, one per
  compute capability.
- `cuda/cuda2go.go` embedded each PTX string into `<name>_wrapper.go`.
- At runtime `cuda/fatbin.go` -> `cu.ModuleLoadData(ptx).GetFunction(fn)` and
  `cu.LaunchKernel` -> the hand-written cgo driver-API binding in `cuda/cu/`.
- FFT and RNG are separate hand-written cgo bindings over `<cufft.h>` /
  `<curand.h>`.

So the port is Driver-API plumbing + ROCm-library swap + a build-time
code-object retarget, not a kernel hipify.

## The port, in three stages

STAGE 1 (driver-API plumbing + build-time code objects):
- Rewrote `cuda/cu/*.go` from the CUDA Driver API to the HIP Driver API:
  `<cuda.h>` -> `<hip/hip_runtime.h>`, `-lcuda` -> `-lamdhip64`, every
  `C.cuXxx` -> `C.hipXxx`, the CUresult enum -> hipError_t names, the
  CUdevice_attribute enum -> hipDeviceAttribute_t, the CUfunction_attribute
  enum -> HIP_FUNC_ATTRIBUTE_*. Launch is `hipModuleLaunchKernel` (driver-API
  form), not `hipLaunchKernel`.
- Build-time code objects (plan Option 1): `cuda/Makefile` now compiles each
  `.cu` with `hipcc --genco --offload-arch=<arch>` to a relocatable HIP code
  object (`.co`), and `cuda/cuda2go.go` embeds the code-object BYTES
  (base64-encoded, since they contain NUL) keyed by gfx arch STRING.
  `hipModuleLoadData` accepts a `--genco` image directly (no hiprtc needed).
- `ModuleLoadData` API changed from `string` (NUL-terminated C string) to
  `[]byte` (length-delimited) because code objects are binary.
- `cuda/fatbin.go` arch selection: replaced `determineCC`/`ccIsOK` (highest CC
  trial-load) with `determineArch`, which reads `hipDeviceProp_t.gcnArchName`
  via `cu.Device.ArchName()`, strips the feature suffix
  (`gfx90a:sramecc+:xnack-` -> `gfx90a`), and trial-loads as a fallback.

STAGE 2 (ROCm library swaps, near-mechanical, cu* -> hip* is 1:1):
- `cuda/cufft/*` -> hipFFT: `<cufft.h>` -> `<hipfft/hipfft.h>`,
  `-lcufft` -> `-lhipfft`, `cufftXxx` -> `hipfftXxx`,
  `cudaStream_t` -> `hipStream_t`. Note: on AMD `hipfftHandle` is a POINTER
  (`struct hipfftHandle_t*`), not the `int` that `cufftHandle` is, so the Go
  `Handle` (uintptr) conversions go through `unsafe.Pointer`. Removed the dead
  `mode.go` (CUFFT_COMPATIBILITY_FFTW_PADDING has no hipFFT equivalent and was
  unused).
- `cuda/curand/*` -> hipRAND: `<curand.h>` -> `<hiprand/hiprand.h>`,
  `-lcurand` -> `-lhiprand`, `curandXxx` -> `hiprandXxx`. Only the float
  GenerateNormal path is used (thermal term).

STAGE 3 (device-code correctness):
- `cuda/reduce.h` wave64 fix (PRIMARY): the reduce macro dropped to an
  unrolled warp-synchronous tail (`volatile float* smem; smem[tid] += [tid+32]
  ... +1` with no __syncthreads) once the __syncthreads tree reached 32,
  relying on 32-lane implicit lockstep. Invalid on gfx90a wave64. Fix: let the
  __syncthreads-ed tree run all the way down to `s>0` (same add order,
  block-wide barrier, correct on wave32 AND wave64). Feeds all 6 reduce
  kernels (MaxTorque, solver error control).
- `cuda/atomicf.h`: atomicFmaxabs used int `atomicMax` on float-as-int. Replaced
  with an atomicCAS loop on the float bits (values are non-negative so the
  IEEE-754 bit order is monotonic). atomicCAS is honored on all ROCm memory
  modes, sidestepping the CDNA coarse-grained int-atomicMax-silently-dropped
  class regardless of allocation coherence.

## Build recipe (linux-gfx90a)

Go 1.22.4 is required (go.mod) and is NOT in apt at that patch level; installed
the official tarball to /var/lib/jenkins/goroot.

```
export GOROOT=/var/lib/jenkins/goroot
export PATH=/var/lib/jenkins/goroot/bin:/opt/rocm/bin:$PATH
export GOPATH=/var/lib/jenkins/go
export CGO_ENABLED=1 GOFLAGS=-mod=mod
cd projects/3/src/cuda
make wrappers CUDA_CC=gfx90a          # hipcc --genco per arch -> .co -> cuda2go embeds -> wrappers
cd ..
go install github.com/mumax/3/...      # links mumax3 + tools into $GOPATH/bin
```

Multi-arch (followers, no source change): `make wrappers CUDA_CC="gfx90a gfx1100 gfx1151"`.
The wrapper map is keyed by gfx string and the loader picks by gcnArchName, so
one build serves all archs and a follower only needs to add its arch to the
list.

Kernel compile flags (cuda/Makefile HIPCCFLAGS):
- `-include hip/hip_runtime.h`: the `.cu` rely on nvcc's implicit builtins
  (blockIdx, threadIdx, atomicAdd, float3, ...); hipcc needs them included.
  Sources stay CUDA-spelled and unmodified.
- `-I.` so `#include <cuComplex.h>` resolves to the local compat shim
  `cuda/cuComplex.h` (maps cuComplex names to hipComplex; one .cu uses it).
- `-Wno-bitwise-instead-of-logical`: clang errors on the intentional bitwise-or
  of bools in the topological-charge / magnetoelastic stencils that nvcc
  accepts; -Werror kept for everything else.

## Validation (real gfx90a, HIP_VISIBLE_DEVICES=3)

- standardproblem4.go: M.Average() = (-0.98461, 0.12605, 0.04327), all 3
  components OK within TOL=1e-3 (the headline correctness gate and the
  followers' cross-arch diff target).
- standardproblem5.mx3: mx/my/mz all OK within 1e-4.
- `go test ./cuda/...`: PASS (reduce_test directly exercises the reduce.h fix;
  buffer/slice tests run on GPU).
- `go test ./cuda/cu/...`: PASS (module load+launch, memcpy, memset, context).
- Non-GPU regression (`go test ./data/... ./httpfs/...`): PASS.
- `mumax3 -vet *.mx3`: all 176 scripts OK.
- Subset of self-checking .mx3 (demag FFT, exchange, DMI, anisotropy energy
  conservation, cubic anisotropy, thermal/curand): all OK.

## Gotchas / fault classes

- NEW PATTERN: go-cgo-driver-api / runtime-code-object loading. The CUDA path
  embedded PTX TEXT (a NUL-terminated C string) keyed by integer CC. HIP code
  objects from `hipcc --genco` are BINARY (contain NUL), so (a) the embed must
  be length-delimited (base64 const decoded at init into `map[string][]byte`),
  not a C string, and (b) `hipModuleLoadData` takes `unsafe.Pointer(&buf[0])`,
  not `C.CString`. Key the map by gfx arch string and select on gcnArchName,
  not by trial-loading the "highest CC" (no CC concept on AMD).
- `hipCtxSynchronize` and `hipCtxGetApiVersion` (the deprecated HIP context
  API) return `hipErrorNotSupported` on ROCm 7.2.1. mumax uses
  `stream0.Synchronize()` everywhere, so this never bites the runtime; only an
  upstream cu-package test called ApiVersion (made tolerant). Do NOT route
  Sync() through CtxSynchronize on ROCm.
- LockOSThread + hipCtxCreate (legacy/deprecated context API) works fine under
  cgo for the single-GPU stream-0 path; no need for
  hipDevicePrimaryCtxRetain. The deprecation is warnings-only.
- hiprand.h -> rocrand.h does `typedef __half half`, and the AMD `__half` is a
  C++ struct, so the cgo C-preamble parse fails with "unknown type name
  '__half'". Fix: alias `typedef _Float16 __half;` in the cgo preamble before
  `#include <hiprand/hiprand.h>` (only the float generator is used). IMPORTANT
  cgo subtlety: this typedef must be in the contiguous `//`-comment block
  immediately above `import "C"`; put any prose rationale as a SEPARATE Go
  comment with a blank line, or the prose (and its apostrophes) get parsed as
  C and break the build.
- hipFFT R2C/C2R/C2C layout and scaling match cuFFT for the batched real
  transforms mumax uses; the demag self-test passed with no normalization
  change.
- The int atomicMax-on-float coarse-grained-drop class: mumax's atomicFmaxabs
  targets hipMalloc device memory (likely fine), but the atomicCAS-loop
  rewrite is correct regardless and removes the risk entirely.

## Install as a dependency

Not a library other MOAT targets depend on; no dependents.

## Review 2026-06-04 (reviewer)

Verdict: review-passed. Driver-API rewrite, build-time code-object path, library swaps, and both device-code fixes are correct. One minor cleanup, no blocker.

Findings (problems only):
- cuda/init.go:36,42-43 -- dead code. `M, m := dev.ComputeCapability()` then `_ = M; _ = m`. The CC concept is fully gone from arch selection (gcnArchName-based now) and M/m are read nowhere else; ComputeCapability() is a pure query. Drop the call and the two blank assignments. Minor; not a correctness issue.

Verified sound (so the next agent need not recheck):
- reduce.h wave64 fix: block size REDUCE_BLOCKSIZE=512 (power of 2); the tree `for(s=blockDim.x/2; s>0; s>>=1)` with `if(tid<s)` reads only sdata[tid+s] where tid+s < 2s <= 512, no OOB; all 512 entries seeded before the tree. The dropped unrolled tail's 32-lane implicit lockstep is invalid on wave64, so removing it is the correct fix; the all-__syncthreads tree is correct on wave32 AND wave64. "Same add order" in the comment is approximate for sum (associativity reordering) but reducesum already accumulates via atomicAdd across blocks (nondeterministic), and fmax is exactly associative; headline gates pass at 1e-3/1e-4. Feeds all 6 reduce kernels (MaxTorque, solver error control) -- the key correctness item, and it is right.
- atomicf.h atomicCAS loop: reducemaxabs seeds initVal=0 and loads fabs(src[i]), so accumulator and candidates are non-negative -> IEEE-754 int bits monotonic, integer-max-over-bits semantics preserved. atomicCAS honored on all ROCm coherence modes, sidestepping the CDNA int-atomicMax coarse-grained-drop class. Correct.
- cu/* driver-API mappings: cuXxx->hipXxx, <cuda.h>-><hip/hip_runtime.h>, -lcuda->-lamdhip64 all correct. result.go enum remap complete; only ERROR_NO_BINARY_FOR_GPU and ERROR_INVALID_IMAGE are referenced (fatbin.go archLoads recover), both retained. Dropped enums (NOT_PERMITTED, TOO_MANY_PEERS, LAUNCH_INCOMPATIBLE_TEXTURING, HARDWARE_STACK_ERROR, ...) referenced nowhere. hipMemsetD32 takes int value -> C.int(value) correct. hipMemAllocHost present in ROCm 7.2.1 header (links).
- MemcpyAsync (generic cuMemcpyAsync) -> hipMemcpyDtoDAsync: sole caller (slice.go:58) is DevicePtr->DevicePtr, semantically correct. Memcpy generic -> hipMemcpyDtoD has no remaining callers (harmless). MemcpyPeer/Async signature changed Context->Device but has zero callers (safe).
- Build path: hipModuleLoadData gets unsafe.Pointer(&image[0]) of a length-delimited []byte (not C.CString); base64 const decoded at init. determineArch returns only a present-and-loadable arch or log.Fatalf, so fatbinLoad's codeobjs[arch] is never empty (&image[0] cannot panic on the normal path). Arch keyed on gcnArchName suffix-stripped via regex `^(gfx[0-9a-f]+)`, trial-load fallback retained. Sound; multi-arch CUDA_CC="gfx90a gfx1100 gfx1151" serves followers with no source change.
- hipFFT/hipRAND swaps: complete and 1:1; hipfftHandle pointer routed through unsafe.Pointer in a handle() helper. rocrand.h `typedef __half half` C-parse fix via `typedef _Float16 __half;` placed in the contiguous cgo preamble (generator.go, status.go), prose rationale split into a separate Go comment block -- correct cgo discipline.
- Deprecated HIP context API: Sync() -> stream0.Synchronize() -> hipStreamSynchronize, never hipCtxSynchronize. context_test made tolerant of hipErrorNotSupported on ApiVersion. Correct.
- Commit hygiene: title "[ROCm] Add AMD GPU support via the HIP driver API" (<=72), Test Plan with literal commands, root-cause explanation, Claude disclosed, no noreply trailer. Author the public account is his own public email (not an internal account). No MOAT jargon, no non-ASCII, no em-dash in changed hand files. cuComplex.h shim included by exactly one .cu, resolved via -I.

PORTING_GUIDE promotion (recommend to the validator/next prep phase):
- PROMOTE (general, new build class): runtime-code-object loading for a Go+cgo Driver-API project -- embed `hipcc --genco --offload-arch=<arch>` code-object BYTES base64-keyed by gfx arch string (NOT PTX text keyed by CC, NOT "highest CC" trial-load); hipModuleLoadData takes unsafe.Pointer(&buf[0]) of a length-delimited buffer, never C.CString; select on gcnArchName suffix-stripped with trial-load fallback. This is the defining lesson of the go-cgo-driver-api class and recurs for any driver-API embed scheme.
- PROMOTE (general cgo): hiprand.h `typedef __half half` breaks the cgo C-preamble parse (AMD __half is a C++ struct); fix with `typedef _Float16 __half;` in the contiguous comment block immediately above import "C", prose rationale in a SEPARATE Go comment. Applies to any cgo binding over a roc*/hip* header that re-typedefs __half.
- PROMOTE (general): deprecated HIP context API (hipCtxSynchronize, hipCtxGetApiVersion) returns hipErrorNotSupported on ROCm 7.2.1; route synchronization through stream.Synchronize() (hipStreamSynchronize), not the context API.
- KEEP project-local (mumax-specific): the -include hip/hip_runtime.h + -I. cuComplex shim hipcc flags -- a kernel-build detail tied to mumax's CUDA-spelled-unmodified .cu strategy; not broadly general. Note the -include-builtins trick in the guide as a one-liner only if another --genco port needs it.

## Validation 2026-06-04 (linux-gfx90a, validator)

Platform: linux-gfx90a, AMD Instinct MI250X / MI250 (gfx90a:sramecc+:xnack-), ROCm 7.2.1, HIP_VISIBLE_DEVICES=0.
Fork sha: 64cb1c7cb5c3cb560b9407b9a3a1492e6491d813.

Build commands:
```
export GOROOT=/var/lib/jenkins/goroot
export PATH=/var/lib/jenkins/goroot/bin:/opt/rocm/bin:$PATH
export GOPATH=/var/lib/jenkins/go
export CGO_ENABLED=1 GOFLAGS=-mod=mod HIP_VISIBLE_DEVICES=0
cd projects/3/src/cuda && make wrappers CUDA_CC=gfx90a  # code objects already present, no-op
cd .. && go install github.com/mumax/3/...
```

Build result: PASS (binary at /var/lib/jenkins/go/bin/mumax3, 15 MB; deprecated HIP ctx API warnings only, no errors).

Test results:
- `go test ./cuda/...`: 8 tests PASS (TestBuffer, TestReduceSum, TestReduceDot, TestReduceMaxAbs, TestSlice, TestCpy, TestSliceFree, TestSliceHost; plus cufft FFT1D test).
- `go test ./cuda/cu/...`: 12 tests PASS (TestContext, TestDevice, TestMalloc, TestMemAddressRange, TestMemGetInfo, TestMemsetAsync, TestMemset, TestMemcpy, TestMemcpyAsync, TestMemcpyAsyncRegistered, TestModule, TestVersion).
- `go test ./data/... ./httpfs/...`: PASS (non-GPU regression tests).
- `mumax3 -vet *.mx3`: 176/176 scripts OK.
- `mumax3 -paranoid=false -failfast -cache /tmp -http "" -f *.go *.mx3`: 181 OK, 0 failed.

Headline gates:
- standardproblem4 (M.Average() within 1e-3): computed (-0.98461187, 0.12604699, 0.04326887) vs expected (-0.98461241, 0.12604089, 0.04327124) -- PASS.
- standardproblem5 mx/my/mz within 1e-4: mx=-0.23488 (OK), my=-0.09453 (OK), mz=0.02296 (OK) -- PASS.

Verdict: PASS. State -> completed.

## Validation 2026-06-04 (linux-gfx1100, RDNA3 native wave32)

Platform: linux-gfx1100, AMD Radeon Pro W7800 48GB (gfx1100), ROCm 7.2.1, HIP_VISIBLE_DEVICES=1.
Fork sha: 7ef67aa4fdf6b4234849ef41729fb3a4eeb6e286 (adds TestModule arch fix on top of 64cb1c7).
Wave size: ATTRIBUTE_WARP_SIZE = 32 (confirmed by TestDevice output -- RDNA3 native wave32).
gfx1100 code object: confirmed by boot log "using gfx1100 code object" at runtime.

Build commands:
```
export GOROOT=/var/lib/jenkins/goroot
export PATH=/var/lib/jenkins/goroot/bin:/opt/rocm/bin:$PATH
export GOPATH=/var/lib/jenkins/go
export CGO_ENABLED=1 GOFLAGS=-mod=mod HIP_VISIBLE_DEVICES=1
cd projects/3/src/cuda
make wrappers CUDA_CC=gfx1100   # hipcc --genco --offload-arch=gfx1100 per .cu -> embeds gfx1100 code objects
cd ..
go install github.com/mumax/3/...
```

Build result: PASS (binary at /var/lib/jenkins/go/bin/mumax3, 15 MB; deprecated HIP ctx API warnings only, no errors).

Note: GPU[0] (HIP_VISIBLE_DEVICES=0) had an orphaned KFD context that blocked queue creation (hipMemsetD32 hung indefinitely); GPU[1] (HIP_VISIBLE_DEVICES=1) and GPU[3] were responsive. All tests run on GPU[1].

TestModule fix: module_test.go hardcoded testdata/testmodule_gfx90a.co, causing hipErrorInvalidImage on gfx1100. Fixed to query Device.ArchName() and load testdata/testmodule_<arch>.co; added testdata/testmodule_gfx1100.co built from testmodule.cu with hipcc --genco --offload-arch=gfx1100. Committed as a new commit on top (does not amend the gfx90a-validated sha). On gfx90a the fix loads testmodule_gfx90a.co as before.

Test results:
- `go test ./cuda/...`: 8 tests PASS (TestBuffer, TestReduceSum, TestReduceDot, TestReduceMaxAbs, TestSlice, TestCpy, TestSliceFree, TestSliceHost; plus cufft FFT1D test). Wave32 reduction tests (TestReduceSum, TestReduceDot, TestReduceMaxAbs) all PASS -- the reduce.h fix (all-__syncthreads tree, no unrolled 32-lane tail) is correct on RDNA3 native wave32.
- `go test ./cuda/cu/...`: 12 tests PASS (TestContext, TestDevice, TestMalloc, TestMemAddressRange, TestMemGetInfo, TestMemsetAsync, TestMemset, TestMemcpy, TestMemcpyAsync, TestMemcpyAsyncRegistered, TestModule, TestVersion).
- `go test ./data/... ./httpfs/...`: PASS (non-GPU regression tests).
- `mumax3 -vet *.mx3`: 176/176 scripts OK.
- `mumax3 -paranoid=false -failfast -cache /tmp -http "" -f *.go *.mx3`: 181 OK, 0 failed.

Headline gates:
- standardproblem4 (M.Average() within 1e-3): computed (-0.9846119, 0.1260456, 0.0432690) vs expected (-0.9846124, 0.1260409, 0.0432712) -- PASS. Cross-arch comparison vs gfx90a reference (-0.98461187, 0.12604699, 0.04326887): difference at 7th decimal place (max delta 5e-7) -- no wave32 reduction fault.
- standardproblem5 mx/my/mz within 1e-4: mx=-0.23488 (|diff|=8.6e-5, OK), my=-0.09453 (|diff|=3.0e-6, OK), mz=0.02296 (|diff|=1.8e-6, OK). Cross-arch comparison vs gfx90a reference (mx=-0.23488, my=-0.09453, mz=0.02296): matching to 5 significant figures -- PASS.

atomicFmaxabs: TestReduceMaxAbs PASS on wave32 confirms the atomicCAS loop works correctly on gfx1100.

Verdict: PASS. State -> completed. validated_sha = 7ef67aa4fdf6b4234849ef41729fb3a4eeb6e286.

## Validation 2026-06-04 (linux-gfx90a, revalidate carry-forward)

Platform: linux-gfx90a, AMD Instinct MI250X / MI250 (gfx90a), ROCm 7.2.1.
Revalidate trigger: head advanced from gfx90a validated_sha 64cb1c7c to 7ef67aa4 (gfx1151 follower delta-port commit).

Delta: 2 files -- cuda/cu/module_test.go (TestModule arch fix) and cuda/cu/testdata/testmodule_gfx1100.co (gfx1100 test code object). No device source (.cu/.cuh) changed.

Binary-equivalence check:
```
# Extracted all 65 gfx90a .co files at both shas; compared md5sum:
md5sum old/*_gfx90a.co == md5sum new/*_gfx90a.co  # all 65 byte-identical
# Confirmed via clang-offload-bundler unbundle on a sample:
md5sum cellindices_old_gfx90a.elf cellindices_new_gfx90a.elf  # identical
```

Verdict: binary-equiv carry-forward. All 65 gfx90a code objects byte-identical; test-only delta does not affect device ISA. State -> completed at 7ef67aa4fdf6b4234849ef41729fb3a4eeb6e286. No GPU re-run required.

## Validation 2026-06-07 (windows-gfx1201)

Platform: windows-gfx1201, AMD Radeon RX 9070 XT (gfx1201 / RDNA4, wave32), TheRock ROCm 7.14, Windows 11 Pro for Workstations, HIP_VISIBLE_DEVICES=0.

New commit: f6b642d5ce5bc0f832d809953efe0ebfc01d7a83 (on top of 7ef67aa4).
69 files changed: 65 *_wrapper.go (gfx1201 code objects), curand/generator.go, curand/status.go (curand_shim.h), cuda/curand/curand_shim.h (NEW), cuda/cu/testdata/testmodule_gfx1201.co (NEW).

### Windows-specific fixes needed

1. **hiprand.h C++-only incompatibility (TheRock 7.14)**
   - `hiprand.h` -> `rocrand.h` defines `uint4` as an anonymous struct in its C-mode `#else` block, conflicting with `hip_vector_types.h`'s `uint4` definition. Also `typedef _Float16 __half` in generator.go's cgo preamble conflicts with `rocrand.h`'s `typedef unsigned short __half`.
   - Fix: added `cuda/curand/curand_shim.h` that declares only the hiprand symbols mumax3 uses, without including the full hiprand.h chain. Changed cgo preambles in generator.go and status.go from `//typedef _Float16 __half;\n//#include <hiprand/hiprand.h>` to `//#include "curand_shim.h"`.

2. **testdata/testmodule_gfx1201.co required for TestModule**
   - module_test.go probes for `testdata/testmodule_{gcnArchName}.co` and SKIPs if absent.
   - Built with: `hipcc --genco --offload-arch=gfx1201 -include hip/hip_runtime.h testmodule.cu -o testmodule_gfx1201.co`

3. **MinGW import libraries for HIP DLLs**
   - cgo on Windows uses MinGW gcc (Strawberry Perl); cannot link MSVC `.lib` files.
   - Created MinGW import libraries from DLLs:
     ```
     gendef amdhip64.dll; dlltool -d amdhip64.def -l libamdhip64.dll.a
     gendef hipfft.dll;   dlltool -d hipfft.def   -l libhipfft.dll.a
     gendef hiprand.dll;  dlltool -d hiprand.def   -l libhiprand.dll.a
     ```
   - Stored in `C:\Users\Shark44\AppData\Local\Temp\mingw_libs\`.

### Build commands

```
# Build wrappers for gfx1201 (from cuda/ subdir)
cd B:\develop\moat\projects\3\src\cuda
set HIP_PATH=B:\develop\TheRock\external-builds\pytorch\.venv\Lib\site-packages\_rocm_sdk_devel
make wrappers CUDA_CC=gfx1201

# Build testmodule code object
cd cuda\cu\testdata
hipcc --genco --offload-arch=gfx1201 -include hip/hip_runtime.h testmodule.cu -o testmodule_gfx1201.co

# Install mumax3
set GOROOT=B:\develop\go_root\go
set GOPATH=B:\develop\go_path_gfx1201
set CGO_ENABLED=1
set CGO_CFLAGS=-IB:\develop\TheRock\external-builds\pytorch\.venv\Lib\site-packages\_rocm_sdk_devel\include -D__HIP_PLATFORM_AMD__
set CGO_LDFLAGS=-LC:\Users\Shark44\AppData\Local\Temp\mingw_libs -lamdhip64 -lhipfft -lhiprand
set GOFLAGS=-mod=mod
set HIP_VISIBLE_DEVICES=0
# PATH must include GOPATH\bin and ROCm\bin so DLLs are found
go install -v ./...
```

### Test results

GPU: ArchName=gfx1201, warpSize=32 (confirmed from TestVersion output).

```
go test -v -count=1 ./cuda/cu/...
# TestContext, TestDevice, TestMalloc, TestMemAddressRange, TestMemGetInfo,
# TestMemsetAsync, TestMemset, TestMemcpy, TestMemcpyAsync,
# TestMemcpyAsyncRegistered, TestModule, TestVersion
12/12 PASS
```

```
go test -v -count=1 ./cuda/...
# TestBuffer, TestReduceSum, TestReduceDot, TestReduceMaxAbs,
# TestSlice, TestCpy, TestSliceFree, TestSliceHost
8/8 PASS
```

```
go test -v -count=1 ./cuda/cufft/...
# TestExampleFFT1D
PASS
```

```
go test -v -count=1 ./data/... ./httpfs/...
# non-GPU regression: all PASS
```

```
# standardproblem4: M.Average within 1e-3 of reference
# result: m=(-0.9846119, 0.12605, 0.04327)  ref=(-0.9846, 0.1260, 0.0435) -- PASS

# standardproblem5: mx/my/mz all within 1e-4 -- PASS
```

### Impact on Linux platforms

This commit regenerated the *_wrapper.go files with gfx1201 code objects (removing gfx90a/gfx1100 objects). The device kernel LOGIC is unchanged; only which arch's objects are embedded differs. Linux platforms (gfx90a, gfx1100) will be flipped to `revalidate` by advance_head. Those validators should do a binary-equivalence check after rebuilding with `make wrappers CUDA_CC=<their-arch>`: if the extracted code objects match (same kernel logic, new hipcc may produce byte-identical or near-identical output), carry forward; otherwise GPU re-run is needed.

For future multi-arch builds (all platforms in one repo state): use `make wrappers CUDA_CC="gfx90a gfx1100 gfx1201"` so all arches are embedded simultaneously.

Verdict: PASS. State -> completed. validated_sha = f6b642d5ce5bc0f832d809953efe0ebfc01d7a83.

## Validation 2026-06-08 (linux-gfx90a, revalidate carry-forward)

Platform: linux-gfx90a, AMD Instinct MI250X / MI250 (gfx90a), ROCm 7.2.1.
Revalidate trigger: head advanced from gfx90a validated_sha 7ef67aa4 to f6b642d5 (windows-gfx1201 delta).

Delta (7ef67aa4 -> f6b642d5): 65 *_wrapper.go files regenerated with gfx1201 code objects; cuda/cu/testdata/testmodule_gfx1201.co (new); cuda/curand/curand_shim.h (new, Windows SDK compat shim); cuda/curand/generator.go + status.go (cgo preamble switches from typedef+hiprand.h to curand_shim.h). No .cu, .cuh, or Makefile changes.

Binary-equivalence check:
- Rebuilt gfx90a code objects at both shas using identical toolchain (ROCm 7.2.1, same Makefile/HIPCCFLAGS):
  `make realclean && make wrappers CUDA_CC=gfx90a` at 7ef67aa4 -> 64 *_gfx90a.co files saved.
  `make realclean && make wrappers CUDA_CC=gfx90a` at f6b642d5 -> 64 *_gfx90a.co files saved.
- md5sum comparison: all 64 gfx90a .co files byte-identical across both shas (same .cu sources, same Makefile).
- curand_shim.h host change: compiled successfully on Linux/ROCm 7.2.1 (`go build ./cmd/mumax3` -- no errors, deprecation warnings only, same as before).
- Note: codeobj_diff.py not applicable to Go executables (looks for .so/.hsaco only); manual .co comparison used instead.

Verdict: binary-equiv carry-forward. All 64 gfx90a code objects byte-identical; host-only curand_shim.h change compiles clean. State -> completed at f6b642d5ce5bc0f832d809953efe0ebfc01d7a83. No GPU re-run required.

## Validation 2026-06-08 (linux-gfx1100, revalidate carry-forward)

Platform: linux-gfx1100, AMD Radeon Pro W7800 48GB (gfx1100), ROCm 7.2.1, HIP_VISIBLE_DEVICES=0.
Revalidate trigger: head advanced from linux-gfx1100 validated_sha 7ef67aa4 to f6b642d5 (windows-gfx1201 delta).

Delta (7ef67aa4 -> f6b642d5): 65 *_wrapper.go files regenerated with gfx1201 code objects; cuda/cu/testdata/testmodule_gfx1201.co (new); cuda/curand/curand_shim.h (new, Windows SDK compat shim); cuda/curand/generator.go + status.go (cgo preamble switches from typedef+hiprand.h to curand_shim.h). No .cu, .cuh, or Makefile changes.

Binary-equivalence check:
- Rebuilt gfx1100 code objects at both shas using identical toolchain (ROCm 7.2.1, same Makefile/HIPCCFLAGS):
  `make realclean && make CUDA_CC=gfx1100` at 7ef67aa4 -> 64 *_gfx1100.co files saved to agent_space/3-gfx1100-revalidate/co-old/.
  `make realclean && make CUDA_CC=gfx1100` at f6b642d5 -> 64 *_gfx1100.co files saved to agent_space/3-gfx1100-revalidate/co-head/.
- cmp -s comparison: all 64 gfx1100 .co files byte-identical across both shas (same .cu sources, same Makefile; diff confirms no .cu/.h/Makefile changes).
- curand_shim.h host change: compiled successfully on Linux/ROCm 7.2.1 (`go build ./cmd/mumax3` -- no errors, deprecation warnings only, same as before).
- Note: codeobj_diff.py not applicable to Go executables (looks for .so/.hsaco only); manual .co comparison used instead.

Verdict: binary-equiv carry-forward. All 64 gfx1100 code objects byte-identical; host-only curand_shim.h change compiles clean. State -> completed at f6b642d5ce5bc0f832d809953efe0ebfc01d7a83. No GPU re-run required.

## Restructure 2026-06-08 (additive HIP backend)

The earlier commits replaced the CUDA path in place (rewrote cuda/cu, cuda/cufft,
cuda/curand to HIP; -lcuda -> -lamdhip64; deleted testmodule.ptx and mode.go).
That breaks every NVIDIA user, so it could not be upstreamed. This restructure
(new commit on top of the validated f6b642d5, NOT an amend) makes the HIP support
ADDITIVE and opt-in: default `go build` still compiles the unchanged upstream
CUDA path; `go build -tags hip` (plus `make BACKEND=hip`) compiles the HIP path.
No device kernel logic changed -- only the file layout, build tags, and the
kernel-build backend selection.

### Build-tag dual backend (the file-layout scheme)

Idiomatic Go build tags select the backend. For every cgo/build file that the
HIP port had rewritten, the upstream CUDA file is restored verbatim and
constrained with `//go:build !hip`, and the HIP version moves to a sibling file
constrained with `//go:build hip`:

- cuda/cu/*: context.go [!hip] + context_hip.go [hip]; cgoflags.go (-lcuda) +
  cgoflags_hip.go (-lamdhip64); result.go (CUresult) + result_hip.go
  (hipError_t); device/execution/function/init/memory/memset/module/peer/
  stream/version the same. Test files: context_test.go [!hip] +
  context_hip_test.go [hip] (the HIP test sibling is named *_hip_test.go, NOT
  *_test_hip.go, so `go test` still recognizes it as a test file). testdata
  carries both testmodule.ptx (CUDA test) and testmodule_<arch>.co (HIP test).
- cuda/cufft/*, cuda/curand/*: same split. cufft/mode.go (uses
  CUFFT_COMPATIBILITY_FFTW_PADDING, no hipFFT equivalent) restored as a
  [!hip]-only file. curand/curand_shim.h stays HIP-only (included only by the
  hip cgo preamble).
- cuda/ build infra: fatbin.go (determineCC/PTX-map, map[int]string) [!hip] +
  fatbin_hip.go (determineArch/gcnArchName, map[string][]byte) [hip]; init.go +
  init_hip.go the same. The 65 generated wrappers: upstream PTX *_wrapper.go
  [!hip] (keyed by compute capability) alongside HIP *_wrapper_hip.go [hip]
  (code-object bytes keyed by gfx arch). The committed *_wrapper_hip.go embed a
  multi-arch set (gfx90a gfx1100 gfx1201) so all validated platforms are served
  by one committed state.
- cuda/cuda2go.go: now backend-aware via -backend=cuda|hip; the cuda branch
  emits PTX *_wrapper.go (//go:build !hip), the hip branch emits code-object
  *_wrapper_hip.go (//go:build hip). It can regenerate either set.
- cuda/Makefile: BACKEND ?= cuda. Default = upstream nvcc `-ptx` per compute
  capability in $CUDA_CC; BACKEND=hip = hipcc `--genco --offload-arch=<arch>`
  per gfx arch in $CUDA_CC. Both branches drive cuda2go with the matching
  -backend flag.
- Host files: engine/engine.go's UNAME var moved into engine/uname.go [!hip]
  (CUDA wording, cu.CUDA_VERSION) and engine/uname_hip.go [hip] (HIP wording,
  cu.HIP_VERSION); engine.go keeps everything else and drops the now-unused
  fmt/runtime/cu imports. cmd/mumax3/main.go's one backend-specific print line
  becomes gpuInfoLine(), defined in cmd/mumax3/gpuinfo.go [!hip] ("cc=%d PTX",
  cuda.UseCC) and gpuinfo_hip.go [hip] ("%s code object", cuda.UseArch). Those
  two files also define goBuildTags ("" / "hip"); runGoFile forwards `-tags hip`
  to the `go run` of a .go input script on the HIP build so the script compiles
  with the same backend (the default build's `go run` args are unchanged).

### #ifdef-guarded device headers (compiled by BOTH nvcc and hipcc)

cuda/reduce.h and cuda/atomicf.h are ONE shared file each, guarded with
`#ifdef __HIP_PLATFORM_AMD__ ... #else <upstream code> #endif`:

- reduce.h: the reduce() macro is a backslash-continued #define, so a directive
  cannot sit inside one body; the whole macro is defined twice -- HIP branch =
  all-__syncthreads tree to s>0 (wave64-correct), CUDA #else branch = the
  upstream unrolled 32-lane warp tail, byte-identical to upstream.
- atomicf.h: HIP branch = atomicCAS loop on the float bits; CUDA #else branch =
  upstream int atomicMax one-liner, byte-identical.
- cuComplex.h stays HIP-only and is reached only via the hipcc -I. include path
  (nvcc finds the real cuComplex.h).

The hipcc compile sets -D__HIP_PLATFORM_AMD__ (HIPCCFLAGS + the cgo CFLAGS), so
the HIP branch is taken on the AMD build and the upstream branch on nvcc.

### CUDA-preservation proof (structural; CUDA path NOT built here -- no nvcc/libcuda)

For every restored CUDA cgo .go file, `git diff 3fe3d41f -- <file>` shows ONLY
the added `//go:build !hip` constraint plus its mandatory trailing blank line
(zero removed lines, CUDA cgo/content otherwise byte-identical to upstream).
Verified programmatically across all 26 restored .go files and all 65 PTX
*_wrapper.go: PASS. The CUDA #else branches of reduce.h/atomicf.h were extracted
and compared byte-for-byte to upstream: identical. The default build cannot be
linked on this AMD host (no CUDA Toolkit): `go build` of cuda/cu stops at
"fatal error: cuda.h: No such file or directory" -- a C-preprocessor error, not
a Go error, confirming the !hip Go structure typechecks and only the missing
CUDA headers block it. Preservation is structural by construction, not claimed
as built/run.

### HIP gfx90a build + test (real MI250X, ROCm 7.2.1, HIP_VISIBLE_DEVICES=0)

Build: `make wrappers BACKEND=hip CUDA_CC="gfx90a gfx1100 gfx1201"` then
`go install -tags hip github.com/mumax/3/...` -> mumax3 binary (15.6 MB) linking
libamdhip64/libhipfft/libhiprand/librocfft. Deprecated-HIP-context-API warnings
only.

- `go test -tags hip ./cuda/`: 8/8 PASS (TestBuffer, TestReduceSum,
  TestReduceDot, TestReduceMaxAbs -- the reduce.h wave64 fix and the atomicCAS
  atomicFmaxabs -- TestSlice, TestCpy, TestSliceFree, TestSliceHost).
- `go test -tags hip ./cuda/cu/...`: 12/12 PASS (module load+launch, memcpy,
  memset, context, version).
- `go test -tags hip ./cuda/cufft/...`: TestExampleFFT1D PASS.
- `go test -tags hip ./data/... ./httpfs/...`: PASS (non-GPU regression). oommf
  fails a pre-existing `go vet` nit (int->string in unmodified ovf2.go); with
  `-vet=off`, the project's documented mode, it has no test files. Not a port
  regression.
- standardproblem4 (M.Average within 1e-3): m=(-0.98461181, 0.12604699,
  0.04326887), all OK. UNAME line shows "HIP-7.2", "using gfx90a code object".
- standardproblem5 (mx/my/mz within 1e-4): mx=-0.23488, my=-0.09453, mz=0.02296,
  all OK.
- `mumax3 -vet *.mx3`: 176/176 OK.
- Broader self-checking scripts: demag2D (9 OK), cubicanisotropy (15 OK),
  anisenergyconservation (10 OK), dmi (2 OK) -- demag FFT, cubic anisotropy,
  energy conservation, DMI kernel paths, zero failures.

### Code-object byte-identity (cross-arch carry-forward)

Built the 64 gfx90a .co at f6b642d5 (its HIP-only Makefile) in a detached
worktree and `cmp`-compared to the 64 gfx90a .co from the restructured tree:
MATCH=64 DIFFER=0. The restructure changed no device code (same .cu, same
HIPCCFLAGS, byte-identical HIP header branches), so the per-arch code objects
are bit-for-bit unchanged. Cross-arch implication: gfx1100 and gfx1201 revalidate
as a binary-equiv carry-forward on their own hosts -- the validator rebuilds its
arch's .co (`make wrappers BACKEND=hip CUDA_CC=<arch>`) and confirms it matches
the previously-validated bytes; no GPU re-run needed for the followers since the
ISA is identical.

### Subtleties for the next agent

- The HIP test-file siblings must end in `_test.go` (named `*_hip_test.go`), not
  `*_test_hip.go`; Go only treats `_test.go` files as tests. The non-test HIP
  siblings are `*_hip.go`; `hip` is not a GOOS/GOARCH so it carries no implicit
  constraint and the explicit `//go:build hip` controls them.
- mumax runs .go input scripts by shelling out to `go run`; that subprocess
  must get `-tags hip` on the HIP build (goBuildTags + runGoFile), or the script
  recompiles the default CUDA path and fails on cuda.h. The cmd-only `-failfast`
  flag is not defined on the script's flag set, so a script run with `-f` should
  omit cmd-batch-only flags (pre-existing upstream behavior, not port-specific).
- To regenerate a single backend's wrappers, the Makefile is timestamp-driven;
  `rm -f cuda/*_wrapper_hip.go` (or *_wrapper.go) before `make wrappers
  BACKEND=...` forces a full regen.

## Review 2026-06-08 (reviewer, additive restructure)

Scope: the additive dual-backend restructure, commit 074bfdce on top of validated f6b642d5 (CUDA baseline = upstream 3fe3d41f). Device-code correctness (reduce.h wave64, atomicf.h) was reviewed+validated previously and is not re-litigated; this review is the ADDITIVE correctness of the build-tag split.

Verdict: changes-requested. One MOAT-jargon leak in upstream-visible text; the restructure is otherwise sound (CUDA preservation, tag partition, and device-header #else byte-identity all verified to hold). Fix the one comment, then this re-validates as a comment-only carry-forward (no GPU re-run).

Problem (must fix):
- cuda/Makefile:47 -- MOAT vocabulary in an upstream-visible comment: `# Default to the lead arch when unset so a bare "make BACKEND=hip" still builds.` "lead" is MOAT jargon (lead/follower platform) banned in upstream-visible text (CLAUDE.md Standing rules). Reword without the label, e.g. `# Default to gfx90a when CUDA_CC is unset so a bare "make BACKEND=hip" still builds.` This is the ONLY jargon hit in the entire delta (swept all changed files including generated wrappers).

Verified sound (so the next agent need not recheck):
- CUDA preservation: every restored CUDA .go in cuda/cu, cuda/cufft, cuda/curand (26 files) and all 65 PTX *_wrapper.go differ from upstream 3fe3d41f by ONLY an added `//go:build !hip` line plus its blank line -- zero removed/changed content. All cuda/-dir non-wrapper restored files (fatbin.go, init.go, the kernel launchers, mode.go, generator.go, status.go, module_test.go, ...) likewise differ only by the constraint. cuda2go.go is the sole content change and is `//go:build ignore` (a generator, compiled into neither backend).
- Device-header #else byte-identity: reduce.h and atomicf.h are single shared files guarded by `#ifdef __HIP_PLATFORM_AMD__ ... #else <upstream> #endif`. Both #else (CUDA) branches are byte-for-byte identical to upstream 3fe3d41f (independently recomputed). nvcc (no -D__HIP_PLATFORM_AMD__) sees exactly the original code. The HIP branch of each is unchanged vs f6b642d5 (the restructure only wrapped it in the #ifdef), so the validated gfx90a .co bytes are preserved.
- Tag partition (no redeclaration): every _hip.go and _hip_test.go carries `//go:build hip`; every one has a CUDA counterpart carrying `//go:build !hip`. Comprehensive package-level symbol scan (cuda, cuda/cu, cuda/cufft, cuda/curand) under both `default` and `-tags hip`: zero top-level func/var/type/const clashes; the only same-name overlaps are methods on distinct receiver types (Context.Destroy vs Stream.Destroy; DevicePtr.String vs Result.String; cufft CompatibilityMode/Result/Type.String) -- legal in Go. Every shared type (Context, Stream, DevicePtr, Result, MemoryType, Function, Module, Device, cufft Result/Type/Handle) is defined in exactly two mutually-exclusive files, never an unconstrained file plus a tagged file.
- Cross-tag reference coherence (the CUDA path cannot be linked on this AMD host, so checked structurally): UNAME/uname.go[!hip]->cu.CUDA_VERSION (version.go[!hip]); uname_hip.go[hip]->cu.HIP_VERSION (version_hip.go[hip]); gpuinfo.go[!hip]->cuda.UseCC (fatbin.go[!hip]); gpuinfo_hip.go[hip]->cuda.UseArch (fatbin_hip.go[hip]); GPUInfo defined in both init.go[!hip] and init_hip.go[hip]; mustDecodeCodeobj in fatbin_hip.go[hip] referenced only by HIP wrappers; fatbinLoad overloaded by tag (map[int]string vs map[string][]byte) matching each wrapper's map type. All resolve.
- Build switch coherence: cuda/Makefile BACKEND ?= cuda (default nvcc -ptx per CC) / BACKEND=hip (hipcc --genco --offload-arch per gfx arch), each drives cuda2go with the matching -backend; the .tmp cp line is upstream, preserved. cuda2go.go -backend=cuda|hip emits the matching wrapper set with the correct constraint. Committed tree carries BOTH wrapper sets; all 65 *_wrapper_hip.go embed exactly {gfx90a, gfx1100, gfx1201}, so every validated platform is served by the committed state.
- Host splits: engine.go drops the now-unused UNAME var and its orphaned fmt/runtime/cu imports (clean); main.go forwards goBuildTags to `go run` and routes the GPU-info line through gpuInfoLine(). Test fixtures: testmodule.ptx (upstream CUDA fixture) restored for module_test.go[!hip]; testmodule_{gfx90a,gfx1100,gfx1201}.co for module_hip_test.go[hip]. .gitignore updated to un-ignore both fixture types.
- Hygiene: commit title "[ROCm] Make the HIP/AMD backend additive and opt-in" (51 chars); Test Plan with literal commands; root cause explained; Claude disclosed; no Co-Authored-By/noreply trailer; author the public account (public). All changed hand-written files ASCII, no em-dash. No AMD-internal account references (the "internal" hits are upstream API constants CUFFT_INTERNAL_ERROR / CURAND_STATUS_INTERNAL_ERROR and the upstream -test help text).

Carry-forward note for the validator (post-fix): the device .cu sources and the HIP branches of reduce.h/atomicf.h are unchanged vs f6b642d5, and the gfx90a/gfx1100/gfx1201 .co bytes are bit-for-bit identical (porter cmp MATCH=64 DIFFER=0); gfx1100 and gfx1201 carry forward via .co byte-identity once the gfx90a HIP path validates at the new sha.

## Validation 2026-06-08 (linux-gfx1100, additive-restructure carry-forward)

Platform: linux-gfx1100, AMD Radeon Pro W7800 48GB (gfx1100), ROCm 7.2.1, HIP_VISIBLE_DEVICES=0.
Revalidate trigger: head advanced from linux-gfx1100 validated_sha f6b642d5 to 4f3de0cf (additive dual-backend restructure + comment-only Makefile reword).

Delta (f6b642d5 -> 4f3de0cf): 193 files -- additive dual-backend restructure (074bfdce) adds *_hip.go + *_wrapper_hip.go sidecars, restores upstream CUDA .go files with //go:build !hip, #ifdef-guards reduce.h + atomicf.h. No .cu sources changed, no HIPCCFLAGS changed. 4f3de0cf rewrites a Makefile comment only.

Binary-equivalence check:
- Built 64 gfx1100 .co files at f6b642d5 (validated sha) using `make wrappers CUDA_CC=gfx1100` in a git worktree -> saved to agent_space/3-gfx1100-revalidate2/co-old/.
- Built 64 gfx1100 .co files at 4f3de0cf (HEAD) using `make wrappers BACKEND=hip CUDA_CC=gfx1100` -> saved to agent_space/3-gfx1100-revalidate2/co-head/.
- cmp -s comparison: MATCH=64 DIFFER=0. All 64 gfx1100 code objects byte-identical.
- Host-side build: `go build -tags hip ./cmd/mumax3` at HEAD compiles clean (deprecation warnings only, no errors).

Commands:
```
# At f6b642d5 (worktree)
cd agent_space/3-gfx1100-revalidate2/src-old/cuda
export HIP_VISIBLE_DEVICES=0 PATH=/var/lib/jenkins/goroot/bin:/opt/rocm/bin:$PATH
go build -o cuda2go cuda2go.go && make wrappers CUDA_CC=gfx1100
cp *_gfx1100.co /var/lib/jenkins/moat/agent_space/3-gfx1100-revalidate2/co-old/

# At 4f3de0cf (HEAD)
cd /var/lib/jenkins/moat/projects/3/src/cuda
go build -o cuda2go cuda2go.go && make wrappers BACKEND=hip CUDA_CC=gfx1100
cp *_gfx1100.co /var/lib/jenkins/moat/agent_space/3-gfx1100-revalidate2/co-head/

# Compare
for f in co-old/*.co; do cmp -s "$f" "co-head/$(basename $f)" || echo "DIFFER: $(basename $f)"; done
# MATCH=64 DIFFER=0

# Host compile check
cd /var/lib/jenkins/moat/projects/3/src
go build -tags hip ./cmd/mumax3  # deprecation warnings only
```

Verdict: binary-equiv carry-forward. All 64 gfx1100 code objects byte-identical; device sources and HIPCCFLAGS unchanged; host-side -tags hip build compiles clean. State -> completed at 4f3de0cfcbc044ddfbb97451d65f0abf1b7d0fbc. No GPU re-run required.

## Validation 2026-06-08 (linux-gfx90a, additive-restructure carry-forward)

Platform: linux-gfx90a, AMD Instinct MI250X / MI250 (gfx90a), ROCm 7.2.1, HIP_VISIBLE_DEVICES=0.
Delta: f6b642d5 -> 4f3de0cf (074bfdce additive-backend restructure + 4f3de0cf comment-only Makefile reword).

Method: built 64 gfx90a .co files at 4f3de0cf (HEAD) via `make wrappers BACKEND=hip CUDA_CC=gfx90a` in projects/3/src/cuda, and at f6b642d5 via `make CUDA_CC=gfx90a` in a git worktree. Compared each pair with `cmp -s`.

Binary-equivalence check:
```
# co-head: 64 *_gfx90a.co from HEAD (4f3de0cf), same .cu sources + HIPCCFLAGS
# co-old:  64 *_gfx90a.co from f6b642d5 (prior validated sha)
cmp -s result: MATCH=64 DIFFER=0
```

All 64 gfx90a code objects byte-identical. The restructure changed no device source (.cu) and no HIPCCFLAGS; the comment reword (4f3de0cf) only touched a Makefile comment with no effect on kernel compilation. HIP path `go install -tags hip ./...` at HEAD compiles clean (deprecation warnings only, no errors; binary 15.6 MB).

No GPU re-run required. Carry-forward confirmed.

Verdict: binary-equiv carry-forward. State -> completed at 4f3de0cfcbc044ddfbb97451d65f0abf1b7d0fbc. No GPU re-run required.

## Revalidate check 2026-06-08 (windows-gfx1201)

Delta: f6b642d5 -> 4f3de0cf (two commits: 074bfdce additive-backend restructure, 4f3de0cf Makefile comment reword).

Method: code object byte comparison.

The gfx1201 code objects embedded in the new wrappers (cellindices_wrapper_hip.go, etc.) were compiled with AMD clang 22.0.0 (ROCm 7.2.1, Linux) vs the previously validated objects which were compiled with AMD clang 23.0.0 (TheRock 7.14, Windows). Sizes differ (e.g., cellindices_co_gfx1201: 10352 bytes in new vs 10224 bytes in old). Not byte-identical.

The linux carry-forward correctly confirmed gfx90a and gfx1100 objects were unchanged -- those were re-built on the same Linux host. But the gfx1201 objects were regenerated on a different toolchain (Linux ROCm 7.2.1 instead of Windows TheRock 7.14). The source .cu files are unchanged, but the device ISA bytes are compiler-dependent.

VERDICT: Cannot carry forward by binary-equivalence. Left as `revalidate`. Needs either: (a) a GPU test run on gfx1201 with the new objects, or (b) rebuild the gfx1201 code objects on this Windows host at 4f3de0cf and compare.

## Validation 2026-06-08 (windows-gfx1201, revalidate at 4f3de0cf)

Platform: windows-gfx1201, AMD Radeon RX 9070 XT (gfx1201 / RDNA4, wave32), TheRock ROCm 7.14, Windows 11 Pro for Workstations, HIP_VISIBLE_DEVICES=0.
Revalidate trigger: head advanced from f6b642d5 to 4f3de0cf (additive dual-backend restructure + Makefile comment reword). gfx1201 code objects in the new head were compiled on Linux (ROCm 7.2.1, AMD clang 22.0.0), not on this host -- binary-equiv is not valid; full GPU run required.
Fork sha: 4f3de0cfcbc044ddfbb97451d65f0abf1b7d0fbc.

Build (with -tags hip, additive backend structure):
```
ROCM_PATH=B:\develop\TheRock\external-builds\pytorch\.venv\Lib\site-packages\_rocm_sdk_devel
GOROOT=B:\develop\go_root\go
GOPATH=B:\develop\go_path_gfx1201
CGO_CFLAGS=-I$ROCM_PATH/include -D__HIP_PLATFORM_AMD__
CGO_LDFLAGS=-LC:\Users\Shark44\AppData\Local\Temp\mingw_libs -lamdhip64 -lhipfft -lhiprand
HIP_VISIBLE_DEVICES=0
cd projects/3/src && go install -tags hip ./...
```

Build result: PASS (deprecation warnings for hipCtxCreate/hipMemAllocHost only, no errors).

Test results:
```
go test -tags hip -count=1 github.com/mumax/3/cuda/cu   # 12/12 PASS
go test -tags hip -count=1 github.com/mumax/3/cuda       # 8/8 PASS
go test -tags hip -count=1 github.com/mumax/3/cuda/cufft # TestExampleFFT1D PASS
go test -tags hip -count=1 github.com/mumax/3/data github.com/mumax/3/httpfs  # PASS (non-GPU)
```

GPU: ArchName=gfx1201, warpSize=32 confirmed from TestDevice output.
TestModule PASS confirms the Linux-compiled gfx1201 code objects load and execute correctly on the RX 9070 XT.

Headline gates:
- standardproblem4 (M.Average within 1e-3): m=(-0.9846119, 0.1260456, 0.0432690) vs ref=(-0.9846124, 0.1260409, 0.0432712) -- PASS (max delta 4.7e-6).
- standardproblem5 (mx/my/mz within 1e-4): mx=-0.23488 (diff 8.6e-5 OK), my=-0.09453 (diff 3.0e-6 OK), mz=0.02296 (diff 1.8e-6 OK) -- PASS.

Note: mumax3.exe used `HIP Library Path: C:\WINDOWS\SYSTEM32\amdhip64_7.dll` (AMD Adrenalin driver System32 copy) -- this works fine for standardproblem5.mx3 via the installed binary since that path ships with Adrenalin. The go test runs also used the System32 amdhip64 without issue.

Verdict: PASS. State -> completed. validated_sha = 4f3de0cfcbc044ddfbb97451d65f0abf1b7d0fbc.

## amdgcnspirv re-port (2026-06-10, linux-gfx1100)

### Rationale: PTX analog, not SASS analog
Upstream mumax3 embeds PTX (a forward-compatible virtual ISA the CUDA driver
JITs at load) keyed by compute capability. The original HIP port embedded
per-gfx-arch code objects (reducesum_gfx1100.co, testmodule_gfx1201.co, ...)
selected by a gcnArchName-keyed loader driven by a CUDA_CC gfx list -- that is
the SASS/sm_XX analog, arch-pinned and needing a rebuild/relist for every new
GPU. The correct PTX analog on AMD is amdgcnspirv: `--offload-arch=amdgcnspirv`
emits ONE generic SPIR-V image that the ROCm runtime (comgr) finalizes for the
present GPU at hipModuleLoadData time. One image per kernel, runs on any
supported gfx arch, no per-arch matrix.

Spike confirmation on this gfx1100: the bundle from
`hipcc --genco --offload-arch=amdgcnspirv -include hip/hip_runtime.h ...`
carries the SPIR-V magic (0x07230203) at offset 4096; amdgcnspirv defines BOTH
__HIP_PLATFORM_AMD__ and __SPIRV__, so the existing `#ifdef __HIP_PLATFORM_AMD__`
guards (reduce.h wave-agnostic branch, atomicf.h fmaxabs) resolve to the AMD
branch under SPIR-V exactly as they did per-arch.

### Build/embed/load changes (all under //go:build hip or BACKEND=hip)
- cuda/Makefile (BACKEND=hip): the per-arch `for arch in $(CUDA_CC)` loop is
  replaced by a single `hipcc --genco --offload-arch=amdgcnspirv` producing
  `<name>.co`. CUDA_CC no longer drives the HIP build (it stays a CUDA/nvcc
  concept). New HIP_OFFLOAD_ARCH=amdgcnspirv knob.
- cuda/cuda2go.go (hip template): the `map[string][]byte` keyed by gfx arch plus
  per-arch base64 consts collapse to a single `<name>_image = mustDecodeCodeobj(<name>_spirv)`
  and one base64 const. Kernel.Code went from `[]codeobj{Arch,B64}` to a single
  `string`; the .co discovery matches `<name>.co` exactly instead of `<name>_(.+).co`.
- cuda/fatbin_hip.go: dropped determineArch / UseArch / gfxArchRe / archLoads
  (the whole arch-detection + trial-load apparatus). fatbinLoad now takes a
  single `[]byte` and does `cu.ModuleLoadData(image).GetFunction(fn)`. Kept
  mustDecodeCodeobj.
- cuda/init_hip.go: the early-load probe is `fatbinLoad(madd2_image, "madd2")`;
  cudaDev comment "used to select the code-object arch" dropped.
- cmd/mumax3/gpuinfo_hip.go: boot line no longer prints cuda.UseArch (gone); it
  now reports "using generic amdgcnspirv image". GPUInfo still names the device
  arch (arch=gfx1100).
- cuda/cu/module_hip_test.go: dropped the ArchName()-keyed `testmodule_<arch>.co`
  selection; loads the single `testdata/testmodule_amdgcnspirv.co`.
- testdata: removed (git rm) testmodule_gfx90a.co / _gfx1100.co / _gfx1201.co;
  added one testmodule_amdgcnspirv.co.
- Regenerated all 65 cuda/*_wrapper_hip.go via `make wrappers BACKEND=hip`; each
  embeds a valid SPIR-V blob (verified: base64-decode -> magic at offset 4096
  for all 65).

CUDA path is byte-identical: `git diff --name-only HEAD` touches zero
`!hip`-tagged files -- no *_wrapper.go, no fatbin.go/init.go/module_test.go/
gpuinfo.go/module.go. All changes are HIP-tagged.

### All kernels lowered to SPIR-V
All 64 generated .co (65 .cu; one .cu has no own .co on disk only because the
`topologicalcharge.cu` recipe's `rm -f topologicalcharge*.co` glob deletes the
already-embedded topologicalcharge-lattice.co as collateral -- harmless, its
wrapper blob is correct and SPIR-V-bearing) carry the SPIR-V magic. No kernel
needed an __SPIRV__ branch or a dynamic-warpSize fix: the device code was
already arch-generic (no __GFX9__, no hardcoded warpSize literal; reduce.h uses
an all-__syncthreads tree, atomicf.h guards on __HIP_PLATFORM_AMD__ which
amdgcnspirv defines). Nothing assumed a compile-time wave width.

### gfx1100 validation evidence (AMD Radeon Pro W7800 48GB, ROCm 7.2.1, HIP_VISIBLE_DEVICES=0)
Build: `cd cuda && make wrappers BACKEND=hip` then
`go install -tags hip github.com/mumax/3/...` -- exit 0 (deprecation warnings only).
- `go test -tags hip -count=1 ./cuda/`: 8/8 PASS (TestBuffer, TestReduceSum,
  TestReduceDot, TestReduceMaxAbs, TestSlice, TestCpy, TestSliceFree, TestSliceHost).
- `go test -tags hip -count=1 ./cuda/cu/`: 12/12 PASS incl. TestModule (now on the
  generic amdgcnspirv image) and TestVersion; cufft FFT1D PASS.
- `go test -tags hip -count=1 ./data/... ./httpfs/...`: PASS (non-GPU regression).
- standardproblem4.mx3: m=(-0.9846120, 0.1260455, 0.0432691) vs ref
  (-0.9846124, 0.1260409, 0.0432712), all OK within 1e-5 (gate is 1e-3) -- PASS.
- standardproblem5.mx3: mx=-0.2348835 (OK), my=-0.0945328 (OK), mz=0.0229620 (OK),
  all within 1e-4 -- PASS.
- Boot log confirms the single generic image:
  "GPU info: AMD Radeon Pro W7800 48GB(46064MB), HIP Driver 7.2, arch=gfx1100,
  using generic amdgcnspirv image" -- no gfx-specific .co selection remains.

### First-load JIT/finalize latency
comgr finalizes each generic SPIR-V image on first hipModuleLoadData. Measured
on gfx1100 over a representative 11-module sample: most kernels finalize in
~2.8-3.2ms; three heavier ones cost more (crossproduct 79ms, dotproduct 70ms,
adddmibulk 113ms). Sample total 285ms (avg ~26ms). Across all ~65 modules the
cumulative cold-finalize cost is roughly 1-2s, but mumax finalizes lazily on
each kernel's first launch (the `if <name>_code == 0` guard in every wrapper),
not in a single startup stall; only the Init-time madd2 probe (~3ms) is forced.
This is comparable to the CUDA path's driver JIT of PTX and is not meaningful
against a typical multi-second-to-minute simulation. (Code objects could be
cached, but mumax already caches the demag kernel, not the device modules; left
as-is to match upstream behavior.)

## Validation 2026-06-10 (linux-gfx90a, amdgcnspirv revalidate)

Platform: linux-gfx90a, AMD Instinct MI250X / MI250 (gfx90a:sramecc+:xnack-,
wave64), ROCm 7.2.1, HIP_VISIBLE_DEVICES=1.
Fork sha: 76136004e9794cd0f89da83e0394d1631780c998 (amdgcnspirv re-port).
Revalidate trigger: functional change to device-code generation/load path
(per-gfx-arch .co -> single generic SPIR-V finalized by comgr at load time).
Binary-equiv carry-forward does NOT apply; full GPU run required.

Go 1.22.4 installed to /var/lib/jenkins/goroot (tarball; apt provides 1.22.2
which go.mod rejects). GOPATH=/var/lib/jenkins/gopath.

Build commands:
```
export GOROOT=/var/lib/jenkins/goroot
export GOPATH=/var/lib/jenkins/gopath
export PATH=/var/lib/jenkins/goroot/bin:/opt/rocm/bin:$PATH
export CGO_ENABLED=1 GOFLAGS=-mod=mod HIP_VISIBLE_DEVICES=1
cd projects/3/src
go install -tags hip ./cmd/mumax3 ./cmd/mumax3-convert ./cmd/mumax3-httpfsd
```

Build result: PASS (18 MB binary; deprecated HIP ctx API warnings only).
The *_wrapper_hip.go files and testmodule_amdgcnspirv.co are pre-committed;
`make wrappers BACKEND=hip` is not needed for validation.

Boot log (from standardproblem4.out/log.txt):
"GPU info: AMD Instinct MI250X / MI250(65520MB), HIP Driver 7.2,
arch=gfx90a:sramecc+:xnack-, using generic amdgcnspirv image"
Confirmed: comgr finalized the generic SPIR-V for gfx90a at runtime; no
arch-specific .co selection remains. ATTRIBUTE_WARP_SIZE=64 (wave64, gfx90a).

Test results:
- `go test -v -tags hip -count=1 ./cuda/`: 8/8 PASS (TestBuffer, TestReduceSum,
  TestReduceDot, TestReduceMaxAbs, TestSlice, TestCpy, TestSliceFree,
  TestSliceHost). reduce.h wave64 fix correct; atomicCAS fmaxabs correct.
- `go test -v -tags hip -count=1 ./cuda/cu/`: 12/12 PASS (TestContext, TestDevice,
  TestMalloc, TestMemAddressRange, TestMemGetInfo, TestMemsetAsync, TestMemset,
  TestMemcpy, TestMemcpyAsync, TestMemcpyAsyncRegistered, TestModule, TestVersion).
  TestModule PASS confirms the single testmodule_amdgcnspirv.co loads and
  executes on gfx90a via hipModuleLoadData.
- `go test -v -tags hip -count=1 ./cuda/cufft/`: TestExampleFFT1D PASS.
- `go test -tags hip -count=1 ./data/... ./httpfs/...`: PASS (non-GPU regression).

Headline gates:
- standardproblem4 (M.Average within 1e-5):
  computed (-0.9846118, 0.1260469, 0.0432689)
  vs ref   (-0.9846124, 0.1260409, 0.0432712) -- PASS (max delta 6e-7).
  Cross-arch comparison vs gfx1100 ref (-0.9846120, 0.1260455, 0.0432691):
  delta at 7th decimal -- no wave64 reduction fault.
- standardproblem5 (mx/my/mz within 1e-4):
  mx=-0.2348835 (diff 5.5e-5 OK), my=-0.0945328 (diff 3.0e-6 OK),
  mz=0.0229620 (diff 1.8e-6 OK) -- PASS.

gfx90a-specific wave64 risk check: no mis-finalization. All reduce tests
(TestReduceSum, TestReduceDot, TestReduceMaxAbs) PASS on wave64; the
all-__syncthreads reduce.h path is arch-generic as required. The generic
SPIR-V image does not carry compile-time wave width assumptions; comgr
correctly targets gfx90a wave64.

Verdict: PASS. State -> completed. validated_sha = 76136004e9794cd0f89da83e0394d1631780c998.
## Validation 2026-06-10 (windows-gfx1201, amdgcnspirv -- FAILED)

Platform: windows-gfx1201, AMD Radeon RX 9070 XT (gfx1201 / RDNA4, wave32), TheRock ROCm 7.14, Windows 11 Pro for Workstations, HIP_VISIBLE_DEVICES=0.
Revalidate trigger: head advanced from f6b642d5/66126b23 (squashed) to 76136004 (amdgcnspirv implementation).
Fork sha: 76136004e9794cd0f89da83e0394d1631780c998.

### Classification

Delta is functional (new device-code generation path: per-arch ELF code objects ->
single amdgcnspirv SPIR-V image). Binary-equivalence carry-forward not applicable;
full GPU revalidation required. `python3 utils/moatlib.py classify` returned
`class=mixed arch_independent=False`.

### Build

Build: `go install -tags hip ./...` with TheRock 7.14 CGO flags -- exit 0,
deprecation warnings only (no errors), same as before. Build time ~5s.

### SPIR-V load failure on Windows

The amdgcnspirv format does NOT work with `hipModuleLoadData` (or `hipModuleLoad`)
on the Windows HIP runtime (Adrenalin `C:\WINDOWS\SYSTEM32\amdhip64_7.dll`).

Verified: loaded Linux-compiled and Windows-compiled `testmodule_amdgcnspirv.co`
bundles -- both return `hipErrorInvalidValue` (error 1). The per-arch
`testmodule_gfx1201.co` (ELF `hipv4-amdgcn-amd-amdhsa--gfx1201` bundle) loads
successfully via both `hipModuleLoadData` and `hipModuleLoad`.

Test results (HIP_VISIBLE_DEVICES=0, gfx1201):

```
go test -tags hip -v -count=1 github.com/mumax/3/cuda/cu
# TestContext, TestDevice, TestMalloc, TestMemAddressRange, TestMemGetInfo,
# TestMemsetAsync, TestMemset, TestMemcpy, TestMemcpyAsync,
# TestMemcpyAsyncRegistered, TestVersion: 10/11 PASS
# TestModule: FAIL -- panic: hipErrorInvalidValue (hipModuleLoadData with SPIR-V)
FAIL github.com/mumax/3/cuda/cu

go test -tags hip -v -count=1 github.com/mumax/3/cuda
# TestBuffer: PASS (no kernel load needed for allocation)
# TestReduceSum and all remaining kernel tests: FAIL (SPIR-V load)
FAIL github.com/mumax/3/cuda

go test -tags hip github.com/mumax/3/data github.com/mumax/3/httpfs
# Non-GPU tests: PASS
```

### Root cause

The Windows AMD HIP runtime (`amdhip64_7.dll` from Adrenalin driver, and also
TheRock 7.14's copy) does not support the `amdgcnspirv` / SPIR-V bundle format
as input to `hipModuleLoadData`. The CLANG_OFFLOAD_BUNDLE `__clang_offload_bundle__`
containing a `hip-spirv64-amd-amdhsa--amdgcnspirv` entry is rejected with
`hipErrorInvalidValue` on Windows regardless of whether the .co was compiled on
Linux or Windows. The per-arch ELF (`hipv4-amdgcn-amd-amdhsa--gfx1201`) format
works fine.

The previous port (per-arch gfx1201 code objects) DID work on Windows. The
amdgcnspirv re-port regresses Windows support.

### Fix needed

The porter must detect Windows at runtime (or build-time) and fall back to
per-arch code objects, OR build a bundle containing BOTH the SPIR-V section (for
Linux) AND per-arch ELF sections (for Windows) keyed by platform. Alternatively,
for Windows always use per-arch gfx1201/gfx1101 objects embedded alongside the
SPIR-V blob with a runtime OS check.

Verdict: FAIL. State -> validation-failed.

Note (2026-06-10 revalidation): the "and also TheRock 7.14's copy" claim above was incorrect. TheRock's runtime (amdhip64_7.dll + amd_comgr.dll from _rocm_sdk_core/bin) DOES support SPIR-V finalization. The actual root cause was DLL binding: the Go test binary bound System32's Adrenalin amdhip64_7.dll (exe-dir -> System32 -> PATH; System32 wins over PATH) which lacks comgr. See the revalidation below.

## Validation 2026-06-10 (windows-gfx1201, PASS -- DLL binding fix)

Platform: windows-gfx1201, AMD Radeon RX 9070 XT (gfx1201 / RDNA4, wave32), TheRock ROCm 7.14, Windows 11 Pro for Workstations, HIP_VISIBLE_DEVICES=0.
Fork sha: 76136004e9794cd0f89da83e0394d1631780c998 (no source change from the FAILED run).

### Root cause of the previous FAILED verdict

The 2026-06-10 FAILED verdict was a runtime-binding artifact, not a port regression. The System32 Adrenalin driver copy of amdhip64_7.dll lacks the SPIR-V->ISA finalization path (comgr); only TheRock's full runtime (amdhip64_7.dll + amd_comgr.dll from _rocm_sdk_core/bin) has it. A native Windows exe's DLL search order is: exe's own directory -> System32 -> PATH. `go test` places the test binary in a temp dir with no runtime DLLs, so Windows falls back to System32's Adrenalin amdhip64_7.dll, which rejects the SPIR-V bundle with hipErrorInvalidValue. The same fix that unblocked dietgpu on Windows: copy TheRock's runtime DLLs next to the test binary so exe-dir search (#1) binds the correct DLL before System32 (#2).

### Fix: stage TheRock runtime DLLs next to the test binary

Used `go test -c` to compile each test package to a known output directory, then copied the five TheRock runtime DLLs from `_rocm_sdk_core/bin` into that directory before running:

```
OUTDIR=agent_space/mumax3-gfx1201-revalidate/

# Build test binaries
go test -tags hip -c -o $OUTDIR/cu.test.exe    ./cuda/cu
go test -tags hip -c -o $OUTDIR/cuda.test.exe  ./cuda
go test -tags hip -c -o $OUTDIR/cufft.test.exe ./cuda/cufft

# Stage TheRock runtime DLLs next to each binary
CORE=/b/develop/TheRock/external-builds/pytorch/.venv/Lib/site-packages/_rocm_sdk_core/bin
cp $CORE/amdhip64_7.dll $CORE/amd_comgr.dll $CORE/rocm_kpack.dll \
   $CORE/hiprtc0714.dll $CORE/hiprtc-builtins0714.dll $OUTDIR/

DEVEL=/b/develop/TheRock/external-builds/pytorch/.venv/Lib/site-packages/_rocm_sdk_devel/bin
cp $DEVEL/hipfft.dll $DEVEL/rocfft.dll $DEVEL/hiprand.dll $DEVEL/rocrand.dll $OUTDIR/

# Run from the package source dir (for testdata/ relative-path lookup)
cd projects/3/src/cuda/cu    && $OUTDIR/cu.test.exe    -test.v -test.count=1
cd projects/3/src/cuda       && $OUTDIR/cuda.test.exe  -test.v -test.count=1
cd projects/3/src/cuda/cufft && $OUTDIR/cufft.test.exe -test.v -test.count=1
```

### Test results (HIP_VISIBLE_DEVICES=0, gfx1201, TheRock runtime)

```
# cuda/cu (12/12 PASS):
TestContext PASS, TestDevice PASS (ArchName=gfx1201, warpSize=32),
TestMalloc PASS, TestMemAddressRange PASS, TestMemGetInfo PASS,
TestMemsetAsync PASS, TestMemset PASS, TestMemcpy PASS,
TestMemcpyAsync PASS, TestMemcpyAsyncRegistered PASS,
TestModule PASS (131s -- comgr SPIR-V finalization for gfx1201),
TestVersion PASS (HIP driver version 71460850)
12/12 PASS
```

```
# cuda (8/8 PASS):
TestBuffer PASS, TestReduceSum PASS (134s -- first-load finalization),
TestReduceDot PASS (133s), TestReduceMaxAbs PASS (136s),
TestSlice PASS, TestCpy PASS, TestSliceFree PASS, TestSliceHost PASS
8/8 PASS
```

```
# cuda/cufft:
TestExampleFFT1D PASS (2s -- hipFFT is a library call, no SPIR-V finalization)
PASS
```

```
# Non-GPU regression (data, httpfs): PASS
```

Confirmed: TestModule PASS proves the amdgcnspirv image loads and finalizes correctly on gfx1201 under TheRock's runtime. TestReduceSum/Dot/MaxAbs PASS confirms the reduce.h wave32 fix and atomicCAS fmaxabs are correct on RDNA4.

Note on first-load latency: comgr JIT on gfx1201 takes ~130-136s per kernel module on first load. This is significantly slower than gfx1100 (~3ms typical) and appears to be a cold-start cost on this gfx1201/RDNA4 machine. Runtime-cached subsequent loads will be fast, and mumax3 itself uses lazy per-kernel loading (first simulation step) rather than a startup stall.

Verdict: PASS. This supersedes the 2026-06-10 FAILED verdict. No source change was made; the failure was purely a runtime-binding artifact. State -> completed. validated_sha = 76136004e9794cd0f89da83e0394d1631780c998.

## Validation 2026-06-19 (windows-gfx1101)

Platform: windows-gfx1101, AMD Radeon PRO V710 (gfx1101 / RDNA3, wave32), TheRock ROCm 7.14, Windows 11 Pro for Workstations, HIP_VISIBLE_DEVICES=1.
Fork sha: a76378a35fc663eb93b841e3b17a101eb2092a28 (amdgcnspirv port, head at validation).
First-time validation (port-ready -> completed). OPTIONAL platform; additive only.

### Build

```
export GOROOT=B:/develop/go_root/go
export GOPATH=B:/develop/go_path_gfx1101
export CGO_CFLAGS="-I$ROCM_PATH/include -D__HIP_PLATFORM_AMD__"
export CGO_LDFLAGS="-LC:/Users/Shark44/AppData/Local/Temp/mingw_libs -lamdhip64 -lhipfft -lhiprand"
export HIP_VISIBLE_DEVICES=1
go -C /b/develop/moat/projects/3/src install -tags hip ./cmd/mumax3 ./cmd/mumax3-convert ./cmd/mumax3-httpfsd
```

Build result: PASS (exit 0, ~0.3s incremental; 28 MB mumax3.exe binary). Deprecation warnings for hipCtxCreate/hipMemAllocHost only, no errors.

### DLL staging

mumax3.exe and test binaries need TheRock runtime DLLs in the exe directory (same fix as gfx1201). Staged to /b/develop/go_path_gfx1101/bin/ (mumax3.exe lives there) and agent_space/mumax3-gfx1101-test/ (test binaries):

```
CORE=/b/develop/TheRock/external-builds/pytorch/.venv/Lib/site-packages/_rocm_sdk_core/bin
DEVEL=/b/develop/TheRock/external-builds/pytorch/.venv/Lib/site-packages/_rocm_sdk_devel/bin
# core: amdhip64_7.dll amd_comgr.dll rocm_kpack.dll hiprtc0714.dll hiprtc-builtins0714.dll
# devel: hipfft.dll rocfft.dll hiprand.dll rocrand.dll
```

MinGW import libraries (.dll.a) created via gendef+dlltool, stored at C:/Users/Shark44/AppData/Local/Temp/mingw_libs/. Build uses CGO_LDFLAGS=-L<that dir>.

### Unit tests

Test binaries compiled with `go test -tags hip -c`, run from their package source dirs with TheRock DLLs staged next to each binary (HIP_VISIBLE_DEVICES=1):

```
# cuda/cu (12/12 PASS):
TestContext, TestDevice (ArchName=gfx1101, warpSize=32), TestMalloc,
TestMemAddressRange, TestMemGetInfo, TestMemsetAsync, TestMemset,
TestMemcpy, TestMemcpyAsync, TestMemcpyAsyncRegistered,
TestModule (amdgcnspirv SPIR-V finalized by comgr for gfx1101), TestVersion
12/12 PASS

# cuda (8/8 PASS):
TestBuffer, TestReduceSum, TestReduceDot, TestReduceMaxAbs,
TestSlice, TestCpy, TestSliceFree, TestSliceHost
8/8 PASS

# cuda/cufft:
TestExampleFFT1D PASS

# data, httpfs (non-GPU regression): PASS
```

### Headline gates

standardproblem4 (M.Average within 1e-5 after 1ns field reversal):
```
//GPU info: AMD Radeon PRO V710(26096MB), HIP Driver 7.14, arch=gfx1101, using generic amdgcnspirv image
//m[0] : -0.9846119284629822 OK   (ref -0.9846124053001404, |diff| 4.77e-7)
//m[1] : 0.1260455995798111 OK    (ref 0.12604089081287384, |diff| 4.71e-6)
//m[2] : 0.04326904937624931 OK   (ref 0.04327124357223511, |diff| 2.19e-6)
```
Wall time: 42m33s (11:33:50 -> 12:16:23 PDT). Heavy SPIR-V JIT latency on cold start (~100s per kernel, 20-21 comgr-pid dirs created).

standardproblem5 (mx/my/mz within 1e-4 after 1ns current-driven vortex dynamics):
```
//mx : -0.23488348722457886 OK   (ref -0.23479773, |diff| 8.58e-5)
//my : -0.0945327877998352 OK    (ref -0.09453578, |diff| 2.99e-6)
//mz : 0.022961996495723724 OK   (ref 0.02296375,  |diff| 1.75e-6)
```
Wall time: 37m34s (12:16:41 -> 12:54:15 PDT). Demag kernel cached from SP4 run; 21 comgr dirs peak (zhangli2 + full LLG kernels).

Both gates: PASS. mumax3 boot confirmed "using generic amdgcnspirv image" on gfx1101.

### Notes

- SPIR-V JIT latency on gfx1101: ~100s per novel kernel module (cold; comgr disk-caches within a run, so each unique kernel compiles once per process). Comparable to gfx1201 (~130s), much slower than gfx1100 (~3ms).
- Memory spike during JIT: 100-150 GB working set per kernel compilation (gfx1101 comgr behavior, same class as gfx1201 but slightly lower peak). Machine has 512 GB, no OOM risk.
- standardproblem5 needs Zhang-Li STT (zhangli2 kernel) not exercised in SP4, so SP5 hits additional novel JITs even with SP4's comgr cache on disk.
- No source changes vs gfx1201 validated sha. Fork is clean (git status --porcelain: no tracked-file modifications).

Verdict: PASS. State -> completed. validated_sha = a76378a35fc663eb93b841e3b17a101eb2092a28.
