# jitllm notes

## Intake screen (2026-11-17, windows-gfx1151)

Requested by jeff. Upstream https://github.com/beehive-lab/jitllm, default branch `main`, screened at `8ccbd7ecae3e49d0d361ed19a8dae4f91fd41c1c` (2026-11-13; repo pushed 2026-11-17, not archived, 296 stars). Shallow screening clone: `agent_space/jitllm-screen`.

### What it is

A pure-Java LLM inference and serving engine (GGUF models: Llama, Qwen2/3/3.5, Gemma3/4, Mistral, Phi3, DeepSeek, Granite, ...) whose GPU work is JIT-compiled by TornadoVM. The repository has no `.cu`/`.cuh`/`.c`/`.cpp` sources and no submodules: everything GPU-specific is Java kernels written against TornadoVM's `KernelContext` intrinsics, plus calls into TornadoVM's own vendor-library modules. So this is not a hipify port. It is backend enablement on top of the TornadoVM HIP backend.

### Licence

`python3 utils/licenses.py check beehive-lab/jitllm`: MIT, tier 1, cleared to contribute. LICENSE file read: MIT, "Copyright (c) 2025 Beehive lab". `python3 utils/licenses.py scan-nvidia agent_space/jitllm-screen`: no NVIDIA proprietary licence text. No submodules or vendored native code.

### Duplicate effort

- AMD-Ecosystem/jitllm and ROCm/jitllm: 404. `gh search repos jitllm`: only beehive-lab and unrelated/demo repos (mikepapadim demos, an unrelated `jitllm/jitllm` org).
- Forks (`repos/beehive-lab/jitllm/forks`): personal forks and GPULlama3.java-era forks (the project's former name); none AMD/ROCm.
- Upstream PRs and issues matching `hip OR rocm OR amd` (all states): none.
- Branches matching hip/rocm/amd: none. `fix/non-nvidia-hardware` exists but is a stale 2025-11 GPULlama3-era scheduler/model-loader branch for the OpenCL path, not a HIP effort.
- Docs grep: README says "NVIDIA (CUDA), Intel & AMD (OpenCL), Apple Silicon (Metal)". AMD GPUs therefore run today through TornadoVM's OpenCL backend, but only on the generic scalar kernel families (see capability gating below): no tensor-core MMA prefill, no shuffle-reduced GEMVs, no dp4a packed path, no vendor libraries, no graph capture, and OpenCL withholds PACKED_HALF2_MATH for an FP16 parity failure. That is functional reach, not mature AMD support, so not `already-supported`.
- Maintainer demand: mikepapadim on beehive-lab/TornadoVM#1327: "We are looking forward to test it with jitLLM." jeff committed publicly to trying jitllm on HIP and posting numbers: https://github.com/beehive-lab/TornadoVM/pull/1327#issuecomment-6928038516

### Dependency: TornadoVM (hard)

`depends_on: [TornadoVM]` (MOAT project `TornadoVM`). The HIP backend merged upstream as beehive-lab/TornadoVM#1327, merged 2026-11-17T13:24:28Z, merge commit `808e42633df104a92e37088dde116063691b7b79`; upstream `develop` now contains `tornado-native-libs/ck-hip` and `cudf-hip`. That PR carried the full HIP port: `TornadoVMBackendType.HIP`, `hip-backend` in `etc/tornado.backend`, MMA intrinsic lowering (4x4 MFMA on CDNA, WMMA on RDNA3/4; gfx1250 refused), 32-lane warp-op semantics on wave64, `cp.async` as a synchronous copy, and `Hip*LibraryProvider` siblings with the same `libraryName()` as the CUDA providers (hipBLAS/hipBLASLt for tornado.cublas, MIOpen + CK attention for tornado.cudnn incl. SDPA, etc.). Per projects/TornadoVM/notes.md: TestCuBlas 27/27, TestCuBlasLt 10/10, TestCuDnn 18/18 (SDPA included) on gfx950. Windows HIP: ck-hip is skipped under the Visual Studio generator, so cuDNN SDPA reports unsupported there (jitllm already falls back to JIT attention when the SDPA probe says no).

jitllm pins `tornadovm.release.version` 8.1.0 in `pom.xml` and builds against `$TORNADOVM_HOME` (SDK modules `tornado.api,tornado.runtime,tornado.cublas,tornado.cudnn,tornado.mlx`). A TornadoVM SDK built from develop with `BACKEND=hip` (or a later release) is what the port builds and runs against.

### How the NVIDIA native libraries are bound

Through TornadoVM, not directly. `src/main/java/org/beehive/jitllm/backend/tornado/ops/nativelib/nvidia/` holds thin `Implementation` classes (`CublasMatMul`, `CublasFp16MatMul`, `CublasDequantizingMatMul`, `CublasLayerProjections`, `CudnnLayerAttention`, 27-35 lines each) that only answer `accepts(step, target)` via `Target.cublas()`; the real work is TornadoVM `LibraryTaskDescriptor`s (e.g. `CuDnn.sdpaForward`) run by TornadoVM's library providers. No JNI, no `System.loadLibrary`, no FFM downcalls in jitllm (its `java.lang.foreign` uses are `MemorySegment` buffer handling). `NativePrefillSupport.java` is the single decision point and is CUDA-specific in exactly these places:

- `Probe.CUBLAS = LibraryProbe.loads(CuBlasLibraryProvider::isAvailable)` and `Probe.CUDNN = ...CuDnnLibraryProvider::isAvailable`: static probes of the CUDA providers. On a HIP-only host these report false, so the native path never engages even though `LibraryRegistry.findProvider` would pick the HIP provider. Needs a provider-agnostic probe (e.g. via `LibraryRegistry.findProvider(name, device)` or the HIP providers' `isAvailable`).
- `cublasLtResolvable()` / `resolvedCublasLt()` look for `libcublasLt.so.12` on `LD_LIBRARY_PATH` and `/etc/ld.so.cache` (guards a cuDNN abort()); `cublasTooOld()` checks Blackwell compute capability vs cuBLAS soname. Must not gate a HIP device.
- `probeSdpaThroughTornado` already goes through `LibraryRegistry.findProvider(CuDnn.LIBRARY_NAME, device)` + `provider.prepare(...)`, so it is backend-neutral and should pick the MIOpen/CK provider on HIP.
- `describe()` strings name cuBLAS/cuDNN.

### Capability and backend gating (the actual port surface)

- `backend/tornado/device/TornadoDevices.java` `capabilitiesOf(type, platformName, deviceInfo)` (~line 182): `TENSOR_CORE_MMA` and `INT8_TENSOR_CORE_MMA` (CUDA, sm >= 80), `SHUFFLE_REDUCED_FP16_GEMV` (CUDA), `PACKED_INTEGER_DOT` (CUDA, sm >= 61), `SINGLE_PASS_RMS` (platform name contains nvidia/cuda) are granted only to `TornadoVMBackendType.CUDA`. `backendId()` maps an unknown type to `BackendId.of(type.name())`, so a HIP device becomes `BackendId("hip")` with only SPLIT_KV_ATTENTION + PACKED_HALF2_MATH: it would run the scalar families. HIP grants need per-arch rules (matrix cores on gfx9 CDNA / gfx11 / gfx12; not gfx1250 until TornadoVM supports it) and CPU-parity evidence for each grant, including the shuffle reductions on wave64.
- `TensorCoreSupport.java`: reads TENSOR_CORE_MMA; javadoc says MMA lowers only on CUDA.
- `CudaGraphDefault.java`: graph capture defaults on only for `BackendId.CUDA`. TornadoVM HIP supports graph capture (hipGraph), so this is a choice to make and measure.
- `ops/Target.java` `cublas()`: native libraries policy && TENSOR_CORE_MMA && `NativePrefillSupport.cublasAvailable()`.
- About 20 files compare against `BackendId.CUDA`/compute capability: `NativeLibrarySupport.java` (5), `TornadoDevices.java` (3), `ModelLoader.java` (3), `DeepSeek2PlanProvider.java` (2), and one each in `CudaGraphDefault`, `Fp16KeyValueSupport`, `Qwen35FFNLayers`, `NativePrefillSupport`, `WeightPrepacking`, and the Llama/Qwen2/Qwen3/Qwen35/Gemma4/Granite/Mistral/Phi3/Devstral/DeepSeekR1DistillQwen plan providers.
- Kernel intrinsics in use (counts across src/main): Metal `simdgroupMatrix*` (thousands; Metal-only families, stay Metal-gated), `simdShuffleDown` 470, CUDA-style `mma`/`mmaLoadA/B`/`mmaStore`/`mmaFragment` and `asyncCopy*` in 12 files (the `*MMA*` prefill layers and `Qwen35MMAKernels`). All are TornadoVM intrinsics that the HIP backend lowers; no kernel source needs translation, but the MMA families have only ever been validated on NVIDIA. `kernels/CuDnnPrefillAttentionKernels.java` is portable layout-adapter JIT kernels around the TornadoVM SDPA call.

### Launcher

Both launchers refuse a HIP-only SDK with "Unsupported backend(s)":

- `jitllm` (Python, repo root): `class Backend(Enum)` OPENCL/CUDA/METAL (lines 26-29), `BACKEND_NAME_MAP` {opencl-backend, cuda-backend, metal-backend} (lines 33-37), `BACKEND_PRIORITY_PROPERTY` (line ~44, `tornado.<b>.priority`), `BACKEND_PRIORITY` (line 53), `detect_installed_backends()` (lines 89-110), `--opencl/--cuda/--metal` flags (line ~744).
- `jitllm4j` (Java single-file launcher): `enum Backend { OPENCL, CUDA, METAL }` (line 8), `BACKEND_PRIORITY` (line 13), name map in `detectInstalledBackends` (lines ~292-296).

Needed: `hip-backend` -> HIP, `tornado.hip.priority`, a `--hip` flag, and the argfile/module wiring. Also `.github/actions/run-inference/action.yml` and the docs (`docs/architecture/models-and-backends.md`, `development.md`) list opencl/cuda/metal. Windows has `set_paths.cmd`, so the windows gate is plausible on the TornadoVM HIP Windows build.

### Scripts

`scripts/manual_cuda_half2_*.sh` and `scripts/profile_cuda_baseline_matrix.sh` are NVIDIA benchmarking helpers; out of scope for the port beyond an optional HIP equivalent for the numbers jeff promised on #1327.

### Viability and recommendation

Viable: yes. Recommendation: fork. The port is Java backend enablement in jitllm (launcher mapping, HIP capability grants, provider-agnostic native-library probes, graph-capture default), resting entirely on the merged TornadoVM HIP backend; no CUDA source translation. Risk is correctness and performance of the MMA and shuffle families on MFMA/WMMA via TornadoVM's emulated m16n8 fragment layouts, which jitllm's CPU-parity gates (419 test classes in `src/test`) are positioned to catch. The maintainers asked for it, and jeff promised #1327 HIP numbers.
