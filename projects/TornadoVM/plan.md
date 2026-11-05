# TornadoVM port plan

## Project

- Name: TornadoVM
- Upstream: https://github.com/beehive-lab/TornadoVM
- Recorded default branch: `master` (the release branch). **Development and every pull request target `develop`** (CONTRIBUTING.md: "Create a Pull Request (PR) to the `develop` branch"; the last 15 PRs all based on `develop`). `develop` is 40 commits ahead of `master` at planning time.
- Planned against: `develop` @ `f8e3ac907b62` (2026-11-03, merge of #1316). The read-only clone in `projects/TornadoVM/src` is `develop`.
- Fork: https://github.com/AMD-Ecosystem/TornadoVM. It currently carries only `master`. Per jeff's standing rule the fork's GitHub default branch becomes `moat-port` once that branch exists; `status.json` `fork_default_branch` stays the upstream mirror.
- Licence: since #1266 (merged 2026-10-23) the whole tree is Apache-2.0 ("All of TornadoVM -- the API, runtime, compiler and drivers -- is licensed under Apache 2.0"; only `LICENSE_APACHE2` remains). Intake's per-module GPL-2.0-with-Classpath-exception reading is superseded; it was already cleared, and the new state is strictly simpler. `status.json` `license_spdx` still carries the old OR expression; correcting it is a records question for jeff (open question 7), not a planner edit.

## What has changed since intake

Intake described the CUDA slice as a PTX-emitting Graal backend (`tornado-drivers/ptx`). That backend is gone. Upstream 8.0.0 replaced the vendored Graal compiler with **TC** (`tornado-compiler/`, pure Java), which prints **CUDA C++ source** and hands it to **NVRTC** through Java FFM bindings (`java.lang.foreign`) to `libcuda`/`nvrtc`. `tornado-drivers/ptx` and `tornado-drivers/spirv` now hold one stub class each.

This turns the port from "write an AMDGPU code generator" into something far more tractable: TC's CUDA C++ is very close to HIP C++, hipRTC compiles HIP C++ at runtime, and the HIP module API mirrors the CUDA driver API almost call-for-call. TC's own README states the extension model: "A new target is a dialect (types, address spaces, intrinsic spellings, preamble) plus `TargetCaps`. Passes and lowering are shared."

## Existing AMD support

- Category: **OpenCL-only** (upstream's own default OpenCL backend runs on AMD GPUs; README backend table: "OpenCL (default) | NVIDIA / AMD / Intel GPUs"). Authoritative and maintained, but it is not a HIP path.
- No HIP/ROCm effort exists anywhere: upstream search for `HIP`/`ROCm` finds no issue or PR; the tree has no `rocm|hipModule|hiprtc|amdgpu|gfx9|gfx1` hit. No AMD-official or community fork.
- Decision (jeff, intake queue issue #5, 2026-08-07): adopt for a **native HIP backend**, explicitly not for enablement. Per `assess-existing-support.md`, "already supports AMD via OpenCL" does not mean skip.
- Delivery vehicle: upstream merges new backends into the main tree (Metal arrived as a peer backend, `tornado-drivers/metal`, with its own FFM port), and does not link platform forks. An upstream PR to `develop` is the right vehicle in principle. Because this is a whole new backend that the maintainers would own, the pull request should be preceded by a short upstream issue asking whether a HIP backend is wanted (upstream-visible write -- jeff's call, open question 6).
- Port vs rewrite: TC's tensor-core path (`mma.sync`, `ldmatrix`, `cp.async` inline PTX in `tc-codegen/.../Helpers.java`), the CUDA Tile path (`nvcc -tilecubin`) and the CUTLASS shim have no mechanical HIP translation; their AMD equivalents (MFMA/WMMA builtins, Composable Kernel, hipBLASLt) would be AMD-native rewrites. The plan is a **correctness-first native backend** for the SIMT path; the tensor-core and tile paths are refused cleanly on HIP devices (the same way TC already refuses them below sm_80), with an AMD-native MFMA/WMMA lowering proposed as a later, separate round (open question 3).

## Build classification

**Neither A nor B: a third class -- Java FFM bindings to the vendor driver API plus a runtime source compiler** (the cuda-to-rocm "runtime PTX + driver API" class, with CUDA C++/NVRTC in place of PTX).

Evidence:
- Build is Maven (`pom.xml`, `./mvnw`), driven by `Makefile` -> `bin/compile` with `BACKEND=<opencl|cuda|metal>` mapped to Maven profiles `<backend>-backend` (`Makefile:13 SUPPORTED_BACKENDS := opencl cuda metal`, `bin/compile process_backends_as_mvn_profiles`, `tornado-drivers/pom.xml` profiles).
- No `.cu` is compiled into the backend. The CUDA backend (`tornado-drivers/cuda`, 86 Java files, ~17.5k lines) binds `libcuda.so.1`/`nvcuda.dll` in `ffm/CUDADriverAPI.java:102` and NVRTC in `ffm/NVRTCAPI.java` through `Linker.downcallHandle`; kernels are TC output compiled at run time in `ffm/CUDACompiler.java`.
- No torch anywhere. The only CMake/C++ in the tree is three optional JNI shims in `tornado-native-libs/{cudf,cudnn,cutlass}-jni` (the 3 components `surface.py` found).

`ext_type` set to `java-ffm-driver-api` (the Java analogue of mumax3's `go-cgo-driver-api`).

## Port strategy

A **new peer backend, `hip`**, alongside `opencl`, `cuda` and `metal`, built with `make BACKEND=hip`, plus a **HIP dialect in TC**. The CUDA backend and the CUDA dialect stay byte-identical.

1. **TC dialect** `Dialect.HIP`: a thin dialect over the shared C core that reuses the CUDA dialect's spellings and overrides only what differs (preamble headers, bf16/fp8 type names, warp-sync mask spelling, no device launch, no tile prefix). Intrinsic lookup for `Dialect.HIP` falls back to the `Dialect.CUDA` spelling when no HIP-specific one exists, so the intrinsic table does not double. `TargetCaps` for a HIP device: `mma`/`cp.async` false, `deviceLaunch` false, `warpSize` from the device, `computeCapability` NOT derived from `hipDeviceAttributeComputeCapabilityMajor/Minor` (see risk R2).
2. **Driver module** `tornado-drivers/hip`: derived from `tornado-drivers/cuda` (the upstream pattern: every backend is a full peer module), with its own FFM layer:
   - `ffm/HIPDriverAPI.java` binding `amdhip64` (module API: `hipModuleLoadDataEx`, `hipModuleGetFunction`, `hipModuleLaunchKernel`, streams, events, memory, graphs);
   - `ffm/HIPRTCAPI.java` binding `hiprtc` (`hiprtcCreateProgram`, `hiprtcCompileProgram`, `hiprtcGetCode[Size]`, `hiprtcGetProgramLog[Size]`, `hiprtcVersion`);
   - `ffm/ROCTXAPI.java` (optional, no-op when `roctx64` is absent, as `NVTXAPI` is for NVTX);
   - `ffm/HIPCompiler.java` replacing `CUDACompiler` (no PTX/cubin split, no PTX-JIT arch fallback, no `-rdc`/`libcudadevrt` link).
   Dropped from the copy, not ported: `CUDATileCompiler`/`CUDATileScheduler`, device-runtime linking, the NVML power handler (replaced by an empty handler, or `amd_smi` if cheap -- see open question 5).
3. **Wiring**: `TornadoVMBackendType.HIP` (tornado-api), `TornadoBackends`, `TornadoOptions`, the `hip-backend` Maven profile in `tornado-drivers/pom.xml` and `tornado-assembly/pom.xml`, `Makefile` `SUPPORTED_BACKENDS`, `bin/compile`, `bin/tornadovm-installer` (`__SUPPORTED_BACKENDS__`), `bin/post_installation.py`, `bin/update_paths.py`, `tornado-assembly/src/bin/tornado.py`, `gen-tornado-argfile-template.py`, `test-native.sh`/`.cmd`, and `tc emit --target hip` (`tc-runtime/tools/Tc.java`).
4. **Tests**: every unit test that today runs only on CUDA by excluding OpenCL and Metal (`assertNotBackend(OPENCL)` + `assertNotBackend(METAL)`, ~127/92 call sites) must either pass on HIP or be gated, by capability where one exists (`TargetCaps::mma`, device launch) and by `assertNotBackend(HIP)` with a reason otherwise. The TC lit tests gain `hip` RUN lines checked by hipRTC when present (skipped when absent, as `nvcc` RUN lines are).

Why a peer backend rather than teaching the CUDA backend to load `amdhip64`: (a) it keeps the CUDA backend -- upstream's flagship -- untouched, so the NVIDIA no-regression gate is trivially a pure passthrough; (b) the backend type `CUDA` gates every NVIDIA library provider (`CuBlasLibraryProvider.canHandle` checks `getTornadoVMBackend() == CUDA`), so an AMD device posing as CUDA would route cuBLAS/cuDNN calls to a library that does not exist; (c) it matches how upstream added Metal; (d) users and `tornado --devices` see "HIP", not a misleading "CUDA". The cost is duplicated host code (~10-12k lines after the drops); that is upstream's own style for backends and is the decision to confirm (open question 1).

Library tasks (cuBLAS, cuFFT, ...) are separate modules under `tornado-native-libs/`. Each NVIDIA library gets its ROCm counterpart, per jeff's standing rule, as a HIP branch of the existing provider (`canHandle` accepts `HIP` devices and binds the ROCm library) so user code calling `CuBlas.sgemm` in a TaskGraph runs unchanged on AMD. Which ones are in this round is open question 2; see the scope table.

## Scope (component accounting, mirrored in surface.json)

| Component | Where | Proposed | ROCm target |
|---|---|---|---|
| TC HIP dialect + TargetCaps | `tornado-compiler/tc-codegen`, `tc-ir`, `tc-tornado` | **in** | hipRTC |
| HIP driver backend | new `tornado-drivers/hip` | **in** | HIP module API, hipRTC |
| CUDA Graphs (`withCUDAGraph`) | CUDA backend `CUDACommandQueue` | **in** | hipGraph (stream capture) |
| NVTX ranges | `ffm/NVTXAPI.java` | **in** (optional lib) | roctx |
| Pinned / registered host memory | `mm/CUDAPinnedMemoryRegistry.java` | **in** | `hipHostRegister`, `hipHostMalloc` |
| Prebuilt-task path + `examples/generated/*.cu` | `tornado-assembly/src/examples/generated`, `unittests/prebuilt` | **in** | HIP-source prebuilt kernels (`add.hip`, ...) |
| Build/installer/launcher wiring | `Makefile`, `bin/*`, `tornado-assembly` | **in** | -- |
| `tornado-selftest` on a HIP device | `tc-selftest` | **in** | -- |
| Unit-test gating for HIP | `tornado-unittests` | **in** | -- |
| `tc emit --target hip`, TC lit tests | `tc-runtime`, `tc-codegen/src/test/lit` | **in** | hipRTC as the RUN-line checker |
| Docs (README backend table, install docs, `docs/source`) | `README.md`, `INSTALL_FROM_SOURCE.md`, `docs/` | **in** | -- |
| Virtual devices (offline compile from a JSON device) | CUDA `virtual/*` | **in** if small, else proposed out | -- |
| cuBLAS / cuBLASLt library tasks | `tornado-native-libs/tornado-cublas` | proposed **in**, pending jeff | hipBLAS / hipBLASLt |
| cuFFT | `tornado-cufft` | proposed **in**, pending jeff | hipFFT |
| cuRAND | `tornado-curand` | proposed **in**, pending jeff | hipRAND |
| cuSPARSE | `tornado-cusparse` | proposed **in**, pending jeff | hipSPARSE |
| NCCL (+ its `libcudart` use) | `tornado-nccl` | proposed **in**, pending jeff | RCCL + HIP runtime |
| cuDNN library tasks | `tornado-cudnn` | proposed **later round**, pending jeff | MIOpen (API differs substantially) |
| cuDNN SDPA JNI shim | `tornado-native-libs/cudnn-jni` | proposed **out**, pending jeff | no drop-in; MIOpen/CK fmha rewrite |
| cuDF (+ `cudf-jni`) | `tornado-cudf`, `cudf-jni` | proposed **out**, pending jeff | ROCm-DS hipDF -- availability on this stack unverified |
| cuVS | `tornado-cuvs` | proposed **out**, pending jeff | ROCm-DS vector search -- unverified |
| CUTLASS (+ `cutlass-jni`) | `tornado-cutlass`, `cutlass-jni` | proposed **out**, pending jeff | Composable Kernel / hipBLASLt rewrite |
| Tensor-core `mma.sync` / `ldmatrix` / `cp.async` intrinsics | TC `Helpers.java`, `Intrinsics.java` | refused on HIP this round; MFMA/WMMA lowering **proposed later**, pending jeff | MFMA (CDNA), WMMA (RDNA3+) |
| CUDA Tile (`TileContext`, `nvcc -tilecubin`) | `CUDATileCompiler`, TC `TileSpellings` | proposed **out**, pending jeff | no HIP equivalent |
| Device-side launch (dynamic parallelism) | TC `deviceLaunch`, `CUDACompiler -rdc` | **out** (HIP has no device-side kernel launch) | none |
| NVML power metrics | `drivers-common/ffm/NVMLAPI.java` | empty handler this round; amd-smi proposed later | amd-smi |
| CLBlast, MLX | `tornado-clblast`, `tornado-mlx` | not part of the port (OpenCL / Apple; CLBlast already reaches AMD via OpenCL) | -- |
| OpenCL, Metal backends | `tornado-drivers/opencl`, `metal` | untouched; must not regress | -- |

Every "proposed out" or "later" row is a cut jeff has not ruled on. None is assumed; they are open questions 2-4. Until he rules, surface.json lists them as components with no `scoped_out` entry, so the surface gate keeps them visible.

## CUDA surface inventory

There are no CUDA kernels to translate. The surface is three layers.

**1. Driver API via FFM** (`tornado-drivers/cuda/.../ffm/CUDADriverAPI.java`, 790 lines), every entry point and its HIP equivalent:

| CUDA | HIP | note |
|---|---|---|
| `cuInit`, `cuDriverGetVersion` | `hipInit`, `hipDriverGetVersion` | |
| `cuGetErrorName/String` | `hipGetErrorName/String` | return `const char*` directly, no out-param |
| `cuDeviceGet/GetCount/GetName/TotalMem_v2` | `hipDeviceGet/hipGetDeviceCount/hipDeviceGetName/hipDeviceTotalMem` | |
| `cuDeviceGetAttribute` | `hipDeviceGetAttribute` | **enum values differ**; remap every `CU_DEVICE_ATTRIBUTE_*` |
| `cuDevicePrimaryCtxRetain/Release/SetFlags`, `cuCtxSetCurrent` | `hipSetDevice` (+ `hipSetDeviceFlags`) | ctx API is deprecated on HIP |
| `cuCtxSynchronize` | `hipDeviceSynchronize` | `hipCtxSynchronize` returns `hipErrorNotSupported` on ROCm 7.2.1 |
| `cuCtxSetLimit(DEV_RUNTIME_PENDING_LAUNCH_COUNT)` | none | device launch is out |
| `cuStreamCreate/Destroy/Synchronize/WaitEvent` | `hipStreamCreateWithFlags/...` | |
| `cuEventCreate/Destroy/Record/Query/Synchronize/ElapsedTime` | `hipEvent*` | `hipErrorNotReady` = 600 also |
| `cuMemAlloc/Free_v2`, `cuMemsetD8`, `cuMemcpyDtoD` | `hipMalloc/hipFree/hipMemsetD8/hipMemcpyDtoD` | allocations are not zeroed (risk R6) |
| `cuMemcpyHtoDAsync/DtoHAsync` | `hipMemcpyHtoDAsync/DtoHAsync` | pageable-source staging differs (R7) |
| `cuMemAllocHost/FreeHost`, `cuMemHostRegister/Unregister` | `hipHostMalloc/hipHostFree`, `hipHostRegister/Unregister` | flag values differ |
| `cuModuleLoadDataEx/GetFunction/Unload` | `hipModuleLoadDataEx/GetFunction/Unload` | JIT option enums differ; pass the code-object buffer, not a C string |
| `cuLaunchKernel` | `hipModuleLaunchKernel` | same shape |
| `cuOccupancyMaxPotentialBlockSize` | `hipModuleOccupancyMaxPotentialBlockSize` | |
| `cuLinkCreate/AddData/AddFile/Complete/Destroy` | not needed | used only for `libcudadevrt` (device launch) |
| `cuStreamBeginCapture_v2/EndCapture/IsCapturing` | `hipStreamBeginCapture/EndCapture/IsCapturing` | capture-mode enum |
| `cuGraphInstantiateWithFlags/Launch/Destroy/ExecDestroy` | `hipGraph*` | |
| `cuGraphExecUpdate_v2` | `hipGraphExecUpdate` | **different signature** (error node + result out-params, no info struct) |

**2. NVRTC via FFM** (`ffm/NVRTCAPI.java`, `ffm/CUDACompiler.java`): `nvrtcCreateProgram/CompileProgram/GetProgramLog[Size]/GetCUBIN[Size]/GetPTX[Size]/GetSupportedArchs/Version/DestroyProgram` -> `hiprtcCreateProgram/CompileProgram/GetProgramLog[Size]/GetCode[Size]/Version/DestroyProgram`. Option translation is needed (R5). NVRTC library discovery by toolkit version (`nvrtc64_<MM>_0.dll`, `libnvrtc.so.<N>`) becomes `hiprtc<MMmm>.dll` / `libhiprtc.so.<N>` discovery under `HIP_PATH`/`ROCM_PATH`.

**3. Generated device code** (TC `CudaDialect`, `Intrinsics`, `Helpers`), compiled by hipRTC:
- thread indexing, `__syncthreads`, `atomicAdd/Sub/Exch`, `__shared__`, `__float_as_int` family, `__float2int_rz` family, `__dp4a`: direct HIP equivalents.
- `#include <cuda_fp16.h>` / `<cuda_bf16.h>` / `<cuda_fp8.h>`, `__nv_bfloat16`, `__hadd_rn/__hsub_rn/__hmul_rn`: `hip/hip_fp16.h`, `hip/hip_bf16.h`, `__hip_bfloat16`; verify each `_rn` half op exists in the hipRTC headers (R9).
- `__shfl_down_sync/__shfl_sync/__any_sync/__all_sync/__ballot_sync(0xffffffffu, ...)`: HIP's `_sync` builtins require a 64-bit mask type; ballot returns 64 bits on wave64 (R1).
- `tornado.simd.sum` helper: hardcoded 32-lane butterfly (`for lane = 16`).
- tensor cores: inline PTX `mma.sync`, `ldmatrix`, `movmatrix`, `cp.async` -- no HIP equivalent; refused via TargetCaps.
- device launch: `<<<...>>>` with `cudaStreamTailLaunch`/`cudaStreamFireAndForget` -- refused via TargetCaps.
- tiles: `__tile_global__`, CUDA Tile C++ through `nvcc -tilecubin` -- refused.

**4. CUDA library surface** (`tornado-native-libs`, all FFM except the three JNI shims): cuBLAS/cuBLASLt, cuFFT, cuRAND, cuSPARSE, cuDNN (+ `cudnn-jni` SDPA, cuDNN frontend), cuDF (+ `cudf-jni`), cuVS, CUTLASS (+ `cutlass-jni`, a real `.cu`), NCCL plus `libcudart`. Mapped in the scope table above.

**5. Misc**: NVML power (`drivers-common/ffm/NVMLAPI.java`, also used by OpenCL's NVIDIA handler -- leave it, add nothing to drivers-common unless amd-smi is in scope); `scheduler/CUDAAMDScheduler.java` already exists in the CUDA backend (inherited from OpenCL) and is the natural default scheduler for the HIP backend.

## Risk list

- **R1 Warp size 32 vs 64.** `Legality.java:64` refuses warp operations (TC-E4015) when `target.warpSize() != 32`. On wave64 (gfx90a, gfx942) HIP devices KernelContext warp intrinsics are therefore refused, exactly as they are today through OpenCL on the same hardware; on wave32 RDNA (gfx1100, gfx1151) they are legal. Keep that policy (the Java API defines 32-lane semantics and `ballot` returns `int`); do not widen it this round. On wave32 the HIP dialect must spell the mask as a 64-bit constant or use the non-`_sync` forms, because HIP's `__shfl_*_sync` static-asserts a 64-bit mask. `warpSize` comes from `hipDeviceAttributeWarpSize`, never a constant. Any shared-code change must be correct at both widths.
- **R2 Compute-capability collision.** `hipDeviceAttributeComputeCapabilityMajor/Minor` reports e.g. 9.0 for gfx90a and 11.5 for gfx1151, which TC reads as "sm_90"/"sm_115" and would pass the `>= 80` gate for `mma.sync`/`cp.async` (TC-E4017) and the FP8 gate, emitting inline PTX to hipRTC. HIP TargetCaps must carry `computeCapability = 0` (or a HIP-specific flag) and key all tensor-core/async-copy legality on the dialect.
- **R3 Device arch for hipRTC.** hipRTC needs `--offload-arch=<gcnArchName>`; `gcnArchName` lives in `hipDeviceProp_t`, whose layout changes across ROCm majors (fragile through FFM). Prefer hipRTC's default (the current device's arch) or a stable query; record whichever is chosen. Strip feature suffixes (`gfx90a:sramecc+:xnack-`) only where a bare name is required.
- **R4 Library discovery on Windows.** The HIP runtime DLL (`amdhip64_7.dll`) and hipRTC (`hiprtc0702.dll`-style versioned name) must be loaded by absolute path from `HIP_PATH\bin`/`ROCM_PATH\bin` before any bare name: on the gfx1151 host the System32 `amdhip64_7.dll` is broken and wins any PATH search. hipRTC on a TheRock install also needs `HIP_DEVICE_LIB_PATH` for the bitcode libraries. `FFMSupport.loadLibrary` already takes a candidate list; mirror `NVRTCAPI.loadNvrtc`'s toolkit-root-first order.
- **R5 Compiler-option translation.** Upstream's flags are NVRTC spellings: `--fmad=false` (strict FP and `withStrictFloatingPoint`, `CudaCompileProfile.REPRO`), `--use_fast_math --extra-device-vectorization` (`FAST`), `--gpu-architecture`, `--include-path`, `-rdc`. hipRTC/clang wants `-ffp-contract=off`, `-ffast-math`, `--offload-arch`, `-I`. clang's HIP default is `-ffp-contract=fast` and contracts across expressions, so the HIP default must be `-ffp-contract=off` wherever NVRTC's would have kept Java semantics, or the `strict-fp` and `java` selftest groups will fail. `CompilerFlags` merging logic is NVRTC-keyed; a HIP flag set must not be fed through the NVRTC dedupe rules blindly.
- **R6 Fresh allocations are not zeroed** on ROCm. TC's "`new` arrays zeroed in loops" check and any runtime path relying on zeroed `cuMemAlloc` memory must zero explicitly.
- **R7 Async copies from pageable memory.** `hipMemcpyHtoDAsync` from a pageable buffer that is freed or reused before the stream syncs is a race HIP does not stage around. The pinned-memory registry (#1305 just fixed a lifetime bug there on CUDA) must keep the same lifetime discipline.
- **R8 Rule of five on handles.** Streams, events, modules, graphs and graph-execs destroyed twice are tolerated by the CUDA driver, not by HIP. The event pool and code cache must destroy exactly once.
- **R9 Half/bf16/fp8 spellings in hipRTC headers.** `__hadd_rn` and friends, `__double2bfloat16`, `__half2ll_rz`: confirm each exists in the hipRTC-bundled headers for the validating ROCm; fall back to plain operators on `__half` with `-ffp-contract=off` where an `_rn` form is missing. FP8 types differ (OCP vs FNUZ on gfx942) but FP8 is only used by the refused MMA path this round.
- **R10 OOB reads.** Generated kernels index with Java bounds semantics, but helpers that read one element past an array (neighbour reads, vector loads at the tail) fault on AMD where CUDA is lenient. Watch the `vectortypes` and `images` suites.
- **R11 Graph update semantics.** `hipGraphExecUpdate`'s different signature and stricter update rules; on failure fall back to re-instantiate (which the CUDA code already does for `CU_GRAPH_EXEC_UPDATE_ERROR*`).
- **R12 Unit-test gating.** Tests excluded only for OPENCL+METAL will run on HIP. Without explicit gating, every CUDA-library, tile, MMA and device-launch test fails on HIP and buries real signal. The porter adds HIP gating before judging pass rates.
- **R13 Upstream velocity.** `develop` moves several merges a day (40 ahead of `master` within days of 8.0.0), mostly by one maintainer, often touching the CUDA backend and TC. The port branch must merge `develop` regularly (never rebase) and re-diff the CUDA backend for changes that need a HIP twin (cuda-to-rocm `upstream-sync.md`: a clean auto-merge lands the NVIDIA half only).
- **R14 Windows APU host.** gfx1151 is a 20-CU APU; long `tornado-test` full runs are heavy. Run the quick pass and the selftest first; exclude known-hanging tests before a full run.

## File-by-file change list

New (fork, `moat-port`):
- `tornado-drivers/hip/pom.xml`, `src/main/java/module-info.java`, `src/main/resources/META-INF/services/uk.ac.manchester.tornado.runtime.TornadoBackendProvider`.
- `tornado-drivers/hip/src/main/java/uk/ac/manchester/tornado/drivers/hip/**`: one class per CUDA backend class kept (backend impl, platform, device, context, device context, command queue + table, event + pool, code cache, program, kernel, installed code, grid info, target device, memory manager and wrappers, schedulers, runtime device/buffer provider, enums, exceptions), plus `ffm/HIPDriverAPI.java`, `ffm/HIPRTCAPI.java`, `ffm/HIPCompiler.java`, `ffm/HIPHandles.java`, `ffm/ROCTXAPI.java`, `power/HIPEmptyPowerMetricHandler.java`.
- `tornado-compiler/tc-codegen/.../HipDialect.java`; HIP lit tests under `tc-codegen/src/test/lit/HIP/`.
- `tornado-assembly/src/examples/generated/add.hip`, `add_uncompressed.hip` (prebuilt-task sources).
- HIP native-API tests mirroring `drivers.cuda.tests.TestCUDAJITCompiler`/`TestCUDATornadoCompiler`.

Modified:
- `tornado-compiler/tc-ir/.../target/Dialect` (add `HIP`), `TargetCaps` (HIP construction path), `tc-tornado/.../Legality.java` (MMA/cp.async/device-launch legality keyed on dialect), `tc-codegen/.../Intrinsics.java` (HIP fallback to CUDA spelling + mask override), `CEmitter.java` (dialect switches that test `Dialect.CUDA`, e.g. FREM at `:848`), `Naming.java` (HIP reserved names), `tc-runtime/tools/Tc.java` (`--target hip`), `tc-selftest` (HIP target line).
- `tornado-drivers/drivers-common/.../tc/TcHook` / `TcTargetDevice` (HIP caps), `tornado-drivers/pom.xml` (`hip-backend` profile).
- `tornado-api/.../enums/TornadoVMBackendType.java`, `tornado-runtime/.../enums/TornadoBackends.java`, `TornadoOptions.java` (`tornado.hip.*` options), `drivers-common/.../logging/Logger.java` if it lists backends.
- `Makefile`, `Makefile.mak`, `bin/compile`, `bin/tornadovm-installer`, `bin/installer_config.py`, `bin/post_installation.py`, `bin/update_paths.py`, `bin/intellij_build.py`, `tornado-assembly/pom.xml`, `tornado-assembly/src/bin/tornado.py`, `gen-tornado-argfile-template.py`, `test-native.sh`, `test-native.cmd`, `scripts/build-release-sdks.py` (only if release SDK naming must know the backend).
- `tornado-unittests/**`: HIP gating on CUDA-only tests (capability-based where possible); `prebuilt/PrebuiltTests.java` choose `.hip` source on HIP.
- `README.md`, `INSTALL_FROM_SOURCE.md`, `docs/source/*` (backend table, HIP install prerequisites).
- Per in-scope library module (pending jeff): `tornado-native-libs/tornado-<lib>/.../provider/*Provider.java` `canHandle` + a ROCm FFM binding class (`HipBlasNativeLib` etc.) selected for HIP devices.

Untouched: `tornado-drivers/cuda/**`, `CudaDialect.java`, OpenCL and Metal backends, `.github/workflows/**` (no fork Actions).

## Build commands

Prerequisites: JDK 21 (or 22+ with `jdk22plus`), Maven via `./mvnw`, Python 3 venv with `tqdm psutil`, GCC >= 13 on Linux / VS 2022 tools on Windows, ROCm >= 6.4 with hipRTC (`HIP_PATH`/`ROCM_PATH` set). No GPU arch is baked into the build: kernels are compiled at run time for the device present, so the same SDK serves gfx90a, gfx942, gfx1100 and gfx1151.

Linux:
```bash
python3 -m venv .venv && . .venv/bin/activate && pip install tqdm psutil
make jdk21 BACKEND=hip            # or: make jdk22plus BACKEND=hip
source setvars.sh
tornado --devices
```

Windows (gfx1151):
```bat
python -m venv .venv
.venv\Scripts\activate.bat
.\bin\windowsMicrosoftStudioTools2022.cmd
python bin\tornadovm-installer --jdk jdk21 --backend hip
```

No-regression builds: `make jdk21 BACKEND=cuda` (compiles on a GPU-less host; the CUDA backend is pure Java) and `make jdk21 BACKEND=opencl`.

## Test plan

Headless, every host (must not regress):
```bash
make test-nogpu JDK=jdk21     # tornado-api, tornado-runtime, all tc-* modules incl. 76+ lit tests
```
plus the new HIP lit tests, whose `RUN:` lines compile the emitted HIP through hipRTC when ROCm is present.

GPU, on each AMD platform, with a HIP-only SDK (build HIP alone: upstream warns that with two backends installed the tests may silently run on the other backend's device):
```bash
tornado --devices                                   # the AMD GPU listed as a HIP device
tornado-selftest --json selftest-hip.json           # backend, core, types, java, strict-fp groups PASS; device/tensor-cores SKIP or UNSUPPORTED
make fast-tests                                     # tornado-test --verbose --quickPass + HeapFail#test03 + test-native.sh
make tests                                          # full tornado-test --verbose
tornado-test -V uk.ac.manchester.tornado.unittests.prebuilt.PrebuiltTests
tornado-test -V --fast uk.ac.manchester.tornado.unittests.tc  # TC runtime tests
```
plus, for each library module ruled in scope: `tornado-test -V uk.ac.manchester.tornado.unittests.<cublas|cufft|curand|cusparse|nccl>` (and its examples/benchmarks once for smoke).

Oracle and baseline: the same AMD device is reachable through upstream's OpenCL backend. Build an OpenCL-only SDK on the same host and run the same `tornado-selftest` and `make tests`; record both pass/fail lists. The HIP backend must pass every test the OpenCL backend passes on that device, other than ones with a recorded backend-specific reason. Tests that HIP passes and OpenCL fails are evidence for the PR.

Required gates: wave64 (gfx90a or gfx942: warp ops refused cleanly, everything else passes), wave32 (gfx1100: warp intrinsics run), windows (gfx1151: Windows library discovery, `tornado.bat`, `test-native.cmd`). Per jeff's rule the Linux wave64 and wave32 hosts validate first; the Windows laptop goes last at the same head. NVIDIA no-regression: the CUDA backend is not edited, so the gate is `make jdk21 BACKEND=cuda` compiling plus `make test-nogpu`; a V100 `make fast-tests` run is optional extra evidence.

## Open questions (for jeff)

1. **Backend shape.** Planned: a new peer `hip` backend (`make BACKEND=hip`, `TornadoVMBackendType.HIP`) duplicating and adapting ~10-12k lines of the CUDA backend's host code, leaving the CUDA backend untouched. The alternative -- a vendor switch inside the CUDA backend's FFM layer -- is a smaller diff but refactors upstream's flagship backend, makes AMD devices report as CUDA, and needs every CUDA library provider taught to tell them apart. Confirm the peer-backend shape.
2. **Library tasks this round** (proposed, pending jeff): include the direct analogues -- cuBLAS/cuBLASLt -> hipBLAS/hipBLASLt, cuFFT -> hipFFT, cuRAND -> hipRAND, cuSPARSE -> hipSPARSE, NCCL -> RCCL -- as HIP branches of the existing providers; move cuDNN -> MIOpen to a later round (different API). Or: core backend first and all library tasks as a second round (smaller first PR; upstream review is easier). Which?
3. **Tensor cores** (proposed, pending jeff): refuse `mma.sync`/`ldmatrix`/`cp.async` on HIP devices this round (clean UNSUPPORTED, as TC already does below sm_80) and plan an AMD-native MFMA (CDNA) / WMMA (RDNA3+) lowering of the same `KernelContext` MMA API as a later round. Approve the deferral?
4. **No-equivalent / reimplementation components** (proposed out, pending jeff): CUDA Tile (`TileContext`; no HIP equivalent), device-side launch (HIP has no device-side kernel launch -- a platform limit, not a choice), CUTLASS + `cutlass-jni` (CK/hipBLASLt rewrite), cuDNN SDPA `cudnn-jni` (MIOpen/CK fmha rewrite), cuDF + `cudf-jni` and cuVS (ROCm-DS equivalents not verified on this stack). Approve scoping these out, each with its recorded reason?
5. **Power metrics**: empty handler on HIP this round (NVML has no use on AMD), or bind amd-smi now?
6. **Upstream contact before the PR**: open a short issue on beehive-lab/TornadoVM asking whether a HIP backend is wanted, before the porter invests the full effort? (Upstream-visible write; jeff's approval and wording.)
7. **Base branch and records**: the port is planned against `develop` (upstream's PR target), not the recorded default `master`. The fork has no `develop` mirror yet, and `moat-port` should be cut from upstream `develop`. Confirm, and whether `status.json` `fork_default_branch`/`license_spdx` (now plain Apache-2.0) should be corrected through moatlib.
8. **Virtual devices** (offline compile against a JSON device description): port for HIP if it falls out of the copy cheaply, else scope out -- porter's call, or ruling needed?

## Rulings (jeff, 2026-11-04)

a. A new peer `hip` backend (tornado-drivers/hip, `make BACKEND=hip`) plus a TC HIP dialect. The CUDA backend stays byte-identical.
b. Libraries now, all of them: hipBLAS/hipBLASLt, hipFFT, hipRAND, hipSPARSE, RCCL and MIOpen (no round-2 deferral).
c. Tensor-core intrinsics now: lower the `KernelContext` MMA API (`mma.sync`/`ldmatrix`/`cp.async`) to AMD-native MFMA on CDNA and WMMA on RDNA3+, not a clean refusal. Refuse only on hardware with neither, with a reason.
d. Out: CUDA Tile (`TileContext`, `nvcc -tilecubin`) and device-side kernel launch (HIP has no dynamic parallelism). In: the CUTLASS / cutlass-jni path, mapped to Composable Kernel. cuDF / cudf-jni maps to AMD-Ecosystem/hipDF and cuVS to AMD-Ecosystem/hipVS (both exist in the org). The cuDNN SDPA shim may have no target yet (no hipDNN): try MIOpen or CK attention first; if neither fits, stop on that item and record the gap for jeff rather than dropping it silently.
e. Power metrics: bind amd-smi now (FFM, mirroring the NVML handler).
f. No upstream issue first; the PR is the proposal.
g. Done 2026-11-04: upstream `develop` (f8e3ac907) pushed to the fork as `develop`; `fork_default_branch` set to `develop` (`set-base-branch`). Cut `moat-port` from the fork's `develop`. `license_spdx` still reads the old OR expression; moatlib has no setter for it, so it needs a records fix (Apache-2.0 since #1266) before PR prep.
h. Virtual devices: port them if cheap; otherwise record why not in notes.md.


## Ruling 2026-11-04 (jeff): 32-lane warps on wave64 (supersedes plan item R1)
Keep the exact 32-lane warp emulation on 64-lane CDNA wavefronts; refusing warp operations would make every warp-based program fail on Instinct GPUs. Document the limitation in the HIP backend docs: the two 32-lane halves of a wave64 do not make independent forward progress, so a kernel where one half spin-waits on a value written by the other half can hang where it would not on NVIDIA. Where that pattern can be detected, fail with a clear error instead of hanging.

## Ruling 2026-11-04 (jeff): gfx1250 tensor-core MMA
Implement the KernelContext MMA API on gfx1250 WMMA now (replacing the refusal) and validate it on the fleet's gfx1250 host (alias mi450, one shared GPU: keep the validation job short and run only when the GPU is idle).

## Ruling 2026-11-04 (jeff): SDPA stays on Composable Kernel
Keep cuDNN SDPA on the CK FMHA ck-hip module (MIOpen attention lacks FP16 on gfx950 and ignores the causal mask). Register a deferral to re-evaluate hipDNN for SDPA once it is released and its attention covers gfx950, RDNA and Windows.

## Ruling 2026-11-04 (jeff): CK pin
Keep the pinned CK development commit (CK release tags lack the gfx11 FMHA kernels). Document the reason next to the pin in the ck-hip CMake and in the README. Deferral tornadovm-ck-release-tag tracks moving to the first CK release tag that has them.
