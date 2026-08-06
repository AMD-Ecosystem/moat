# Port plan: mumax3 (mumax/3)

## Project
- Name: 3 (mumax3)
- Upstream: https://github.com/mumax/3
- Default branch: main
- Domain: GPU-accelerated micromagnetics (finite-difference LLG solver). Go + cgo host, CUDA device kernels.

## DISPOSITION
INVOLVED-BUT-TRACTABLE -- proceed to porter, but scope it as a HARD port, not a clean hipify.
The effort is dominated by rewriting THREE hand-written cgo bindings (CUDA Driver API, cuFFT, cuRAND) to their HIP/ROCm equivalents and replacing a runtime-PTX-string loading scheme with per-arch HIP code objects. The .cu kernels themselves are simple and few of them carry real fault-class risk (one wave-size reduction tail). The risk is concentrated in the build/runtime plumbing and the Go+cgo+ROCm toolchain bringup, not in kernel correctness. Effort class: HARD (multi-stage). Recommendation: dispatch the porter; if Stage 1 (driver-API cgo bringup + a single kernel launching on gfx90a) does not converge, the platform should go blocked with the bringup findings rather than thrash.

## Existing AMD support
- NO HIP/ROCm port of mumax3 exists. Confirmed via: web search (no rocm.docs/AMD/GPUOpen result), `gh api repos/mumax/3/forks` (no fork under ROCm/AMD/GPUOpen or with rocm/hip/amd in the name), and `git ls-remote` (no rocm/hip/amd branch upstream).
- Related projects, NONE of which is a ROCm port (do not count as already-supported):
  - mumax+ (https://github.com/mumax/plus) -- a newer C++/CUDA rewrite with a Python UI. Still CUDA/NVIDIA-only. Not AMD.
  - uMagNUS / OpenCL micromagnetics -- OpenCL-based alternatives exist in the field. Per the authoritative-vs-community + "OpenCL is not a HIP path" rule, an OpenCL alternative does NOT make a HIP port redundant.
- Judgment: net-new HIP port targeting ROCm adds value. No authoritative AMD effort to adopt or improve. Port from scratch our way.

## Build classification: NEITHER cmake nor torch-extension -- Go + cgo + runtime-PTX + CUDA Driver API
ext_type = go-cgo-driver-api (a third class; record as such).

Evidence (the DECIDING build-model finding):
- `Makefile` -> `cudakernels:` -> `cd cuda && make`. `cuda/Makefile` rule `%_wrapper.go: %.cu cuda2go` runs `nvcc ... -arch=compute_$cc -code=sm_$cc $< -o <name>_$cc.ptx` with the `-ptx` flag (line 50,70-71). It emits PTX TEXT, one per compute capability in `$CUDA_CC`, NOT a linked .o/.a/fatbin.
- `cuda/cuda2go.go` then embeds each PTX file's text as a Go string constant into `<name>_wrapper.go` (the large 20KB-840KB generated Go files), keyed by CC in a `<name>_map map[int]string` (see `madd2_wrapper.go` lines 73-88).
- At RUNTIME: `cuda/fatbin.go` `fatbinLoad(sm, fn)` -> `cu.ModuleLoadData(sm[cc]).GetFunction(fn)` -> `cuda/cu/module.go` `C.cuModuleLoadData` + `C.cuModuleGetFunction` (CUDA Driver API). Launch: `cuda/cu/execution.go` `C.cuLaunchKernel`. So device dispatch is 100% the CUDA Driver API, NOT the runtime API and NOT `<<<>>>`.
- `cuda/cu/` is a hand-written cgo binding over `#include <cuda.h>` linking `-lcuda` (cgoflags.go). Full driver surface used (enumerated): cuInit, cuCtxCreate/Destroy/SetCurrent/GetCurrent/GetDevice/Synchronize/ApiVersion, cuDeviceGet/GetCount/GetName/TotalMem/GetAttribute/CanAccessPeer, cuDriverGetVersion, cuMemAlloc/Free/AllocHost/FreeHost/GetInfo/GetAddressRange/HostRegister/HostUnregister, cuMemcpy{,Async,DtoD,DtoDAsync,HtoD,HtoDAsync,DtoH,DtoHAsync,Peer,PeerAsync}, cuMemsetD8/D32{,Async}, cuModuleLoad/LoadData/GetFunction, cuLaunchKernel, cuFuncGetAttribute, cuStreamCreate/Destroy/Query/Synchronize, cuCtx/Device PeerAccess, cuPointerGetAttribute.
- FFT: `cuda/cufft/` hand-written cgo over `#include <cufft.h>` linking `-lcufft`. Used: cufftPlan1d/2d/3d/PlanMany, cufftExec{C2C,C2R,R2C,Z2Z,Z2D,D2Z}, cufftSetStream, cufftDestroy, the cufft{Real,Complex,DoubleReal,DoubleComplex,Handle,Type} types. Consumed by the demag convolution (`conv_demag.go`, fft3dr2c.go, fft3dc2r.go, conv_kernmul.go) -- the magnetostatic field is an FFT convolution, so FFT is on the hot path and used every timestep.
- RAND: `cuda/curand/` hand-written cgo over `#include <curand.h>` linking `-lcurand`. Used: curandCreateGenerator, curandSetPseudoRandomGeneratorSeed, curandGenerateNormal (thermal/temperature term only).
- Go is the host language (go.mod, go 1.22.4). `go install` links everything; cgo provides the C bridge to libcuda/libcufft/libcurand.

This is exactly the "runtime-PTX + Driver-API via cgo" design the task flagged as a substantial rework. It is NOT plain nvcc-compiled .cu linked into a lib.

## Port strategy (custom; neither Strategy A nor B applies cleanly)
The colmap compat-header model (A) and torch-hipify (B) do not apply -- there is no CMake and no torch. The port is a HIP Driver-API + ROCm-libraries swap inside the Go cgo layer plus a kernel-build retarget. Plan in stages so a blocker is isolated.

Two build-time options for getting device code objects from the .cu kernels; choose Option 1 (preferred):
- Option 1 (build-time code objects, RECOMMENDED): change `cuda/Makefile` to compile each `.cu` with `hipcc --genco --offload-arch=gfx90a[,gfx1100,gfx1151] name.cu -o name_<arch>.co` (a relocatable HIP code object), and change `cuda2go.go` to embed the code-object BYTES (not PTX text) keyed by arch string instead of integer CC. `hipModuleLoadData` accepts a code-object image directly. This mirrors the existing scheme (embed-then-ModuleLoadData) most closely and keeps the runtime loader logic intact, only swapping the key from CC-int to gfx-arch-string. The .cu sources stay CUDA-spelled and are accepted by hipcc (HIP is a near-superset of the CUDA C the kernels use: __global__, __device__, float3, atomicAdd/atomicMax, __syncthreads; no warp intrinsics).
- Option 2 (hiprtc runtime compile): ship .cu/.hip source strings and compile at startup with hiprtc (`/opt/rocm/include/hip/hiprtc.h` present). More faithful to "runtime compile" but adds a startup cost and a runtime compiler dependency. Defer unless Option 1's embedded-arch-selection is awkward.

`determineCC()` / `ccIsOK()` (fatbin.go) currently pick the highest CC the driver accepts by trial `cuModuleLoadData`. Replace with an arch-string selector keyed on `hipDeviceProp_t.gcnArchName` (strip the feature suffix, e.g. "gfx90a:sramecc+:xnack-" -> "gfx90a"); keep the trial-load fallback for robustness.

## CUDA surface inventory
- Kernels: 65 `.cu` files, each one or a few `__global__` kernels, all simple elementwise / stencil / reduction micromagnetics ops (madd, exchange, dmi, anisotropy, torque, demag copy/pad/kernmul, shift, region decode, reductions, topological charge, hopfindex). Device helpers in float3.h, amul.h, stencil.h, constants.h, sum.h.
- Warp intrinsics: NONE. `grep` for `__shfl*`, `__ballot`, `__activemask`, `__syncwarp`, `warpSize`, `cg::`, `tiled_partition` finds zero hits in any .cu/.h. The only 32-related construct is the reduce.h warp-tail (see risk).
- Textures/surfaces: NONE.
- Driver API: see build-model section. All have 1:1 HIP Driver-API equivalents: cuInit->hipInit, cuCtxCreate->hipCtxCreate (or modern hipDevicePrimaryCtx), cuModuleLoadData->hipModuleLoadData, cuModuleGetFunction->hipModuleGetFunction, cuLaunchKernel->hipModuleLaunchKernel, cuMemAlloc->hipMalloc/hipMemAlloc, cuMemcpy*->hipMemcpy*, cuMemsetD8/D32->hipMemsetD8/D32, cuStream*->hipStream*, cuMemHostRegister->hipHostRegister, cuMemAllocHost->hipHostMalloc, cuPointerGetAttribute->hipPointerGetAttribute, cuFuncGetAttribute->hipFuncGetAttribute, peer-access 1:1.
- cuFFT -> hipFFT. hipFFT exposes the cufft* API names 1:1 (hipfftPlan1d/2d/3d/PlanMany, hipfftExecR2C/C2R/C2C/D2Z/Z2D/Z2Z, hipfftSetStream, hipfftDestroy, hipfftReal/Complex/DoubleReal/DoubleComplex/Handle/Type). The cgo binding rewrite is near-mechanical: swap `#include <cufft.h>`->`<hipfft/hipfft.h>`, `-lcufft`->`-lhipfft`, `cufftXxx`->`hipfftXxx` in cuda/cufft/*.go. /opt/rocm/lib/libhipfft.so present (ROCm 7.2.1, rocFFT backend).
- cuRAND -> hipRAND. hiprand exposes hiprandCreateGenerator/SetPseudoRandomGeneratorSeed/GenerateNormal/hiprandRngType_t 1:1. Swap `<curand.h>`->`<hiprand/hiprand.h>`, `-lcurand`->`-lhiprand`. /opt/rocm/lib/libhiprand.so present.
- cuBLAS/cuSPARSE/cuDNN/Thrust/CUB: NONE used.
- Pinned/managed memory: pinned via cuMemAllocHost/cuMemHostRegister -> hipHostMalloc/hipHostRegister. No managed memory.
- Streams/events: a single global stream0 (cu.Stream(0)) plus stream create/destroy/sync. No events.

## Risk list
1. (PRIMARY device-code fix) reduce.h warp-synchronous tail assumes a 32-lane warp. The reduce macro does `for (s=blockDim.x/2; s>32; s>>=1){ ... __syncthreads(); }` then an UNROLLED `if (tid<32){ volatile float* smem=sdata; smem[tid]=op(smem[tid],smem[tid+32]); ...+16,+8,+4,+2,+1; }` with NO __syncthreads between the unrolled steps -- it relies on implicit 32-lane warp lockstep. Block size is 512 (util/reduce.go). On gfx90a (wave64) the wavefront is 64 lanes, so the 32-lane implicit-lockstep assumption is invalid and `volatile` does not order writes across the wavefront's two 32-lane halves the way CUDA's 32-lane warp did; result can be wrong/nondeterministic. Affects all 6 reduce kernels (reducesum, reducedot, reducemaxabs, reducemaxvecnorm2, reducemaxdiff, reducemaxvecdiff2) -- which feed MaxTorque/solver error control, so a wrong reduction corrupts adaptive timestepping and the ExpectV gates. FIX: make the tail warpSize-generic -- either continue the `__syncthreads()` loop all the way down to s>0 (simplest, correct on wave32 and wave64), or use a proper wave-collective. Do NOT hardcode 64. Validate on both wave widths.
2. atomicf.h atomicFmaxabs does `atomicMax((int*)a, intbits)` on a float-as-int. PORTING_GUIDE: int atomicMax is SILENTLY DROPPED on coarse-grained/managed memory on gfx90a. Here the targets are cuMemAlloc/hipMalloc DEVICE memory (fine-grained path is the concern only for managed), so likely OK, but VERIFY -- if reducemaxabs results are stale/zero, this is the cause; emulate with atomicCAS loop.
3. cuMemcpyAsync from pageable host memory: CUDA stages synchronously, HIP may be truly async (PORTING_GUIDE). mumax3 uses async copies on stream0 with sync points; audit that host buffers outlive the async copy or are pinned. Low risk (mumax pins its staging buffers) but watch slice.go/buffer.go host transfers.
4. Driver-context model: mumax uses the legacy `cuCtxCreate` primary-context pattern + `runtime.LockOSThread`. HIP's context API (hipCtxCreate) is deprecated-but-present; prefer mapping to hipCtxCreate first (minimal diff), consider hipDevicePrimaryCtxRetain if the deprecated path misbehaves. cgo + LockOSThread interaction with the HIP runtime needs a smoke test early.
5. cuLaunchKernel arg marshalling (execution.go) passes a packed argp array; hipModuleLaunchKernel takes the same `void** kernelParams` form -- 1:1, but the struct field packing/alignment in each *_args_t (generated) must match what hipcc-compiled kernels expect (same ABI as the .cu signature; should be identical since the kernel signatures are unchanged).
6. Code-object arch selection must NOT bake a single wave width: build a multi-arch set (gfx90a;gfx1100;gfx1151) so followers need no source change, and key the runtime loader on gcnArchName. A single-arch embed forces followers to rebuild and churns head_sha.
7. PTX-text -> code-object-bytes change in cuda2go.go: the embedded strings become binary; ensure Go string/[]byte embedding handles NUL bytes (use a byte slice or base64, not a C-string). cuModuleLoadData took a C.CString (NUL-terminated) -- that path must change to a length-delimited buffer for binary code objects.
8. Toolchain: Go is NOT installed on this host. The porter must `apt install golang-go` (or fetch a Go 1.22 tarball; go.mod requires 1.22.4) before any build. CGO_ENABLED=1 with CC=gcc; cgo must find hip/hipfft/hiprand headers (-I/opt/rocm/include) and libs (-L/opt/rocm/lib -lamdhip64 -lhipfft -lhiprand). amdhip64 is the HIP runtime+driver-API lib (replaces -lcuda).

## File-by-file change list (planner estimate; porter refines)
- cuda/cu/*.go -- rewrite the Driver-API cgo binding to HIP: cgoflags.go (-I/opt/rocm/include, -L/opt/rocm/lib -lamdhip64), module.go/execution.go/memory.go/memset.go/context.go/device.go/stream.go/init.go/peer.go/function.go/version.go/result.go: `#include <cuda.h>`->`#include <hip/hip_runtime.h>`, every `C.cuXxx`->`C.hipXxx`, CUresult/CUdevice/CUmodule/CUfunction/CUstream/CUcontext/CUdeviceptr -> hip equivalents, the error enum (result.go) remapped to hipError_t names. This is the bulk of the work.
- cuda/cufft/*.go -- swap to hipFFT (mechanical name+include+lib swap).
- cuda/curand/*.go -- swap to hipRAND (mechanical name+include+lib swap).
- cuda/Makefile -- replace the nvcc `-ptx` per-CC loop with hipcc `--genco --offload-arch=<arch>` per-arch loop producing code objects.
- cuda/cuda2go.go -- emit per-arch code-object bytes (length-delimited) instead of per-CC PTX text; key the map by arch string.
- cuda/fatbin.go + cuda/init.go -- replace determineCC/ccIsOK with gcnArchName-based selection; update the early test-load.
- cuda/reduce.h -- fix the warp-sync reduction tail to be warpSize-generic (PRIMARY correctness fix).
- cuda/atomicf.h -- only if risk 2 materializes (atomicCAS-loop emulation).
- Top-level Makefile -- ensure CUDA_CC -> arch list flows to cuda/Makefile; keep the NVIDIA path intact behind a backend toggle (e.g. an env/flag or a build tag) so upstream's CUDA build is not broken (important for the eventual upstream PR -- this must be additive, not a replacement).
- Note: mumax3 has NO existing build-tag/backend abstraction; the cleanest upstreamable design is a Go build tag (e.g. `//go:build hip` vs the default cuda) selecting cu_cuda.go vs cu_hip.go, OR a separate cu/ implementation chosen at build time. The porter should pick a minimal-footprint additive scheme and record it; a wholesale replacement of the cu/ package is acceptable for the MOAT fork but should be structured so upstream can take it as an opt-in backend.

## Build commands (gfx90a)
Prereqs (porter installs): `apt-get install -y golang-go` (or a Go 1.22.4 tarball); ROCm 7.2.1 already at /opt/rocm.
```
export GOPATH=$HOME/go PATH=$PATH:$GOPATH/bin
export CGO_ENABLED=1
export CUDA_CC="gfx90a"            # repurposed as the HIP arch list after Makefile change
cd projects/3/src
make cudakernels                  # hipcc --genco per-arch -> code objects -> cuda2go embeds -> go install ./cuda
go install -v github.com/mumax/3/...
cd cmd/mumax3 && make              # links the mumax3 binary
```
Multi-arch (for followers, no source change): `CUDA_CC="gfx90a gfx1100 gfx1151"`.

## Test plan (real GPU)
- GPU correctness gate (self-contained, no large downloads): the muMAG standard-problem scripts in test/ embed hard reference values with tolerances. Best gate:
  - `test/standardproblem4.go` -- ExpectV("m", M.Average(), Vector(-0.9846..., 0.1260..., 0.0433...), TOL=1e-3). A reference-comparing pass/fail in-script. Run: `$GOPATH/bin/mumax3 test/standardproblem4.mx3` and the `-f` go-script form via test/run.bash.
  - test/standardproblem5*.mx3, plus the dynamics/energy-conservation .mx3 scripts (176 total). Many call ExpectV/Expect with tolerances internally, so a green run is a real correctness check, not just "it ran".
- Full suite: `cd test && ./run.bash` runs `mumax3 -vet *.mx3` then `mumax3 -paranoid=false -failfast -cache /tmp -http "" -f *.go *.mx3` -- exercises every kernel path on the GPU and self-checks. This is the validator's GPU gate.
- Go unit tests: `go test -vet=off github.com/mumax/3/...` includes cuda/*_test.go (buffer_test, reduce_test, slice_test) which run on the GPU -- the reduce_test directly exercises the reduce.h fix.
- Non-GPU regression set: the pure-Go packages (data, util, draw, dump, httpfs, oommf I/O) have CPU tests; these must not regress. Run the same `go test ./...` and confirm non-cuda packages pass unchanged.
- Cross-arch (followers): standard-problem outputs are deterministic to tolerance, so the gfx1100/gfx1151 follower diffs its M.Average() against the gfx90a result for the same script (catches a wave32 reduce regression that a loose "ran ok" gate would miss).

## Open questions
1. Backend selection design for upstreamability: Go build tags (cu_cuda.go/cu_hip.go) vs a separate package vs an env-switch. Porter picks; affects whether the PR is cleanly additive. Lean toward build tags.
2. Code-object embed vs hiprtc: Option 1 (embed --genco code objects) is the plan; confirm hipModuleLoadData accepts a relocatable --genco object directly on ROCm 7.2.1 (vs needing a fully-linked bundle). Fallback: hiprtc (Option 2).
3. hipCtxCreate (deprecated) vs hipDevicePrimaryCtxRetain under cgo+LockOSThread -- verify the legacy context path works with stream0 = stream 0 before doing the full binding.
4. atomicFmaxabs int-atomicMax on hipMalloc device memory -- confirm not silently dropped on gfx90a (risk 2); micro-test early.
5. cuFFT R2C/C2R plan layout + scaling conventions: rocFFT matches cuFFT for these batched real transforms, but the demag self-test (conv_selftest.go) is the place a layout/normalization mismatch would surface -- run it first among GPU tests.
