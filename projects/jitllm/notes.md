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

## Planning (2026-11-20, linux-gfx942, planner; no GPU used)

### Rulings needed from jeff before a porter starts

The plan ports the whole project. No app, GUI or library layer is deferred. Two questions still need jeff's answer. Nobody could be asked during this job, so neither is assumed.

**Q1. How should a HIP device that reports version 9.0 be treated for the matrix-core (MMA) grants?**

jitllm can see the GPU architecture only through TornadoVM's device description, `Device version : HIP <major>.<minor>` (`HIPDevice.java:211-215` in TornadoVM develop). That string is the HIP compute capability. It tells gfx942 (9.4), gfx950 (9.5), gfx11 (11.x) and gfx1200/1201 (12.0) apart. It cannot tell gfx906, which has no matrix cores, from gfx908 or gfx90a, which have MFMA: all three report 9.0. `HIPTargetDevice.getArchitecture()` returns the real `gfx90a`, but it lives in the HIP driver module, not in the API jitllm compiles against. A wrong grant fails the kernel compile (TC-E4017), and under TornadoVM's default bailout that becomes a silent fall-back to sequential Java. So the rule has to be exact.

- (a) Grant MMA on 9.4, 9.5, 11.x and 12.0 only. A 9.0 device (gfx90a/MI200, gfx908) runs the scalar families, as it does on OpenCL today. This changes jitllm only. MI200 gets no tensor-core prefill.
- (b) First add an `Architecture : gfxNNN` line to TornadoVM's `HIPDevice.getDeviceInfo()` (a one-line TornadoVM change and a follow-up upstream PR, since #1327 has merged), then key jitllm on the gfx name. This is exact on every arch, including gfx90a. It adds a TornadoVM round and makes this port depend on it landing upstream.
- (c) Probe at start-up: compile a one-tile MMA kernel and treat TC-E4017 as "no matrix cores". This is exact and needs no TornadoVM change. It costs a compile on first use, and it is new mechanism that jitllm's maintainers may not want.

Planner's recommendation: (a) now, and (b) as a later TornadoVM follow-up if MI200 matters. The plan is written so that (a) is one line of the rule table and (b) or (c) replaces only the parser.

**Q2. DeepSeek2 on HIP: enable it, or keep it CUDA-only?**

`DeepSeek2PlanProvider` refuses every non-CUDA backend: "every deepseek2 kernel is CUDA's: they reduce with warp shuffles". Warp shuffles are correct on HIP (TornadoVM half-wave lowering), so nothing technical forces the refusal. However, DeepSeek2 has no test fixture in jitllm. `verification.md` says DeepSeek-R1-Distill-Qwen and Devstral are not gated, and no `*AccelTest` loads a deepseek2 model. So the upstream suite cannot judge a HIP enablement.

- (a) Enable it on HIP. The validator does a one-off CPU-reference check: the same DeepSeek-V2-Lite GGUF (about 10 to 16 GB), the same prompt, greedy decoding, CPU against HIP, token and top-5 agreement recorded in notes. This evidence lives outside the suite, like upstream's own CUDA enablement.
- (b) Keep DeepSeek2 CUDA-only. Record it in `surface.json` as scoped out, with "no fixture to verify a HIP enablement" as the reason, and say so in the PR body.

Planner's recommendation: (a), because the refusal's stated reason does not hold on HIP and the check is cheap on an MI300X. `surface.json` lists `family:deepseek2` under `owed` until jeff rules.

### Non-blocking items (no ruling needed to start)

- **N1. Native image / Leyden AOT on HIP.** `docs/native-image.md` builds against a jdk22plus SDK, and TornadoVM compiles kernels at run time in both modes. Whether TornadoVM's HIP driver (FFM) works under SubstrateVM is untested. The porter or validator tries it once on Linux. If it fails for a TornadoVM reason, record the error and bring jeff the scope question. Do not scope it out unasked. (Owed in `surface.json`.)
- **N2. Release coordinates.** `-P release` pins TornadoVM 8.1.0, which predates HIP. A HIP user needs a development SDK until a TornadoVM release carries HIP. The PR body says so; the profile is unchanged (scoped out).
- **N3. Benchmark numbers for TornadoVM#1327.** jeff committed publicly to posting jitllm-on-HIP numbers. The validator records them in this file (plan Test plan 6). Posting them is jeff's action.
- **N4. Planner scope calls jeff may overturn.** These are scoped out with reasons in `surface.json`: upstream CI workflows (no AMD runners; fork Actions disabled), the NVIDIA nsys/ncu profiling scripts, the Class C CI benchmark gate (the HIP tuple is record-only by design), and the Metal-only kernels. These are developer tooling, not product layers.

### What the planning found (beyond intake)

- Build type recorded as `java-tornadovm` (`moatlib.py set-ext-type`). It is a Maven build against `$TORNADOVM_HOME`, with no native compile in jitllm. The TornadoVM Java modules `tornado.cublas`, `tornado.cudnn` and `tornado.mlx` are built for every backend (`tornado-native-libs/pom.xml:28-38`). Only the `ck-hip`/`cudf-hip` shims are behind `-Phip-backend`. So a `BACKEND=hip` SDK satisfies jitllm's `--add-modules` with no pom change.
- TornadoVM develop at `3e260b2c` (2026-11-20) has the following.
  - `TornadoVMBackendType.HIP`.
  - `tornado.hip.priority` (`TornadoOptions.java:156`).
  - `hip-backend` written to `etc/tornado.backend` by `bin/compile`.
  - Platform name `AMD HIP`.
  - TC legality:
    - TC-E4015 (warp ops) is accepted on HIP wave64 (`TargetCaps.warps32()`).
    - TC-E4017 refuses MMA without matrix cores.
    - TC-E4019 refuses a local-memory spin-wait on wave64.
  - `TargetCaps.matrixCores()`: MFMA for gfx908/90a/940/941/942/950, WMMA for gfx11, gfx1200/1201 and gfx1250.
  - Int8 MMA via `__builtin_amdgcn_mfma_i32_4x4x4i8` and WMMA iu8.
  - `tc_dp4a`.
  - hipGraph capture behind the same `withCUDAGraph()`.
  - `HipBlasLibraryProvider`, `HipBlasLtLibraryProvider` and `MIOpenLibraryProvider`, each with a static `isAvailable()`.
  - `LibraryRegistry.findProvider(String, TornadoXPUDevice)`.
- jitllm gates on the CUDA backend in exactly these places: `TornadoDevices.capabilitiesOf`; `NativeLibrarySupport` (5 sites); `NativePrefillSupport` (static CUDA-provider probes, the `libcublasLt.so.12` loader check, the Blackwell check); `nativeLibraryWeights` in 10 plan providers; `Fp16KeyValueSupport`; `WeightPrepacking`; `Qwen35FFNLayers`; `DeepSeek2PlanProvider`; `CudaGraphDefault`. `ModelLoader.java:125` labels a CPU-resolved load as CUDA. That is a pre-existing, documented choice; it is read-only for this port.
- The test surface: 476 test files, 261 `*AccelTest` classes (Class B, `-Paccel-tests`). The Class A tests that pin backend behaviour and need HIP cases are listed in plan.md. Class B classes that `assumeTrue(BackendId.CUDA...)`: `Qwen35F16BatchedPrefillParityAccelTest`, `Qwen35PackedDecodeParityAccelTest`, and `Qwen35DecodeAttentionWarpNumericsAccelTest` (via `TensorCoreSupport`).
- Fixtures (`GoldenFixture.java`): Llama-3.2-1B (F16/Q8_0/Q4_0), Qwen2.5-0.5B, Qwen3-0.6B (many quants), Qwen3-1.7B, granite-3.2-2b, Phi-3-mini, Mistral-7B Q8_0, gemma-4-E2B (BF16/Q8_0/Q4_0), Qwen3.8-27B Q4_0 (16 GB). They are resolved from `$JITLLM_TEST_MODELS` or `~/.jitllm/test-models/`, with SHA-256 pinned.
- Scratch clones (container-local, not records): `projects/jitllm/src` (upstream, depth 1) and `agent_space/tvm-develop` (TornadoVM develop, depth 1).
