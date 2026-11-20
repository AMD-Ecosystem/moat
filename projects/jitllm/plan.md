# jitllm port plan

Planned 2026-11-20 on linux-gfx942 (planner, no GPU used). Upstream snapshot `8ccbd7ecae3e49d0d361ed19a8dae4f91fd41c1c` (2026-11-13). TornadoVM read at upstream `develop` `3e260b2c7629534eb36ffccdaa78241ecd905653` (2026-11-20), which carries the merged HIP backend (beehive-lab/TornadoVM#1327).

**Two rulings from jeff are needed before a porter starts. They are listed at the top of the planning section in `notes.md` (Q1 and Q2).** This plan is written so that each answer changes one bounded item. It does not assume either answer.

## Project

- Name: jitllm
- Upstream: https://github.com/beehive-lab/jitllm, default branch `main`
- Fork: https://github.com/AMD-Ecosystem/jitllm. Port branch `moat-port`, cut from `main`.
- What it is: a pure-Java LLM inference and serving engine for GGUF models. TornadoVM JIT-compiles its GPU kernels. The repository contains no C, C++ or CUDA source.
- Hard dependency: MOAT project `TornadoVM`. Its HIP backend merged upstream on 2026-11-17 and has been validated on gfx950, gfx1201 and windows-gfx1151 (`moatlib.py show TornadoVM`).

## Existing AMD support

**Classification: improvable (OpenCL-only).** Upstream's README lists "Intel & AMD (OpenCL)". On AMD GPUs jitllm runs today through TornadoVM's OpenCL backend, using only the generic scalar kernel families:

- no tensor-core MMA prefill;
- no shuffle-reduced GEMVs and no dp4a packed path (both are withheld on OpenCL, which miscompiles `simdShuffleDown`, per `TornadoDevices.java`);
- no FP16 KV cache on non-NVIDIA OpenCL;
- no vendor libraries and no graph capture.

There is no AMD fork, PR, issue or branch (intake, 2026-11-17). The upstream maintainers asked for HIP testing on TornadoVM#1327 ("We are looking forward to test it with jitLLM"). An upstream PR is therefore the right vehicle; upstream does not link platform forks.

**Port versus rewrite.** There are no CUDA kernels to translate. Every GPU kernel is a Java method written against TornadoVM `KernelContext` intrinsics:

- `mma*` and `asyncCopy*` in 12 files (the `*MMA*` prefill layers and `Qwen35MMAKernels`);
- `simdShuffleDown` (470 uses);
- `dp4a`.

TornadoVM's HIP backend lowers all of them through its TC compiler:

- MFMA (`v_mfma_*_4x4x4*`) on CDNA and WMMA on gfx11/gfx12, using an emulated m16n8 fragment layout;
- 32-lane warp ops on wave64, run as two half-wavefronts;
- `cp.async` as a synchronous copy.

The vendor-library tasks (`tornado.cublas`, `tornado.cudnn`) resolve to `HipBlasLibraryProvider`/`HipBlasLtLibraryProvider` and `MIOpenLibraryProvider`, with CK FMHA for SDPA, under the same `libraryName()`. So the port is a mechanical enablement: tell jitllm that a HIP device exists, and what it may run.

The emulated m16n8 layout over 4x4 MFMA is unlikely to match an AMD-native tile, so MMA prefill speed may trail what an MFMA-native kernel could reach. Tuning that layout is a TornadoVM matter and out of scope here. This port is correctness-first, and its benchmark numbers (`notes.md`) are the baseline for any later TornadoVM work.

## Build classification

**`java-tornadovm`: Maven, JDK 21 or newer, compiled against a TornadoVM SDK.** It is neither Strategy A nor Strategy B: there is nothing to hipify, nothing to compile with hipcc, and no ROCm library is linked by jitllm itself. Evidence:

- `pom.xml:61-69`: `tornadovm.sdk.modules=${env.TORNADOVM_HOME}/share/java/tornado`, `--add-modules tornado.api,tornado.runtime,tornado.cublas,tornado.cudnn,tornado.mlx`. TornadoVM builds all three library modules unconditionally (`tornado-native-libs/pom.xml:28-38`), so a `BACKEND=hip` SDK satisfies the module path. Only the native shims (`ck-hip`, `cudf-hip`) are under the `hip-backend` profile.
- `pom.xml:226-258`: the `require-tornadovm-sdk` enforcer checks only that `TORNADOVM_HOME` exists and that its JDK line matches. It does not check the backend.
- `pom.xml:406-472` (`accel-tests` profile): surefire runs `**/*AccelTest.java` with `@$TORNADOVM_HOME/tornado-argfile`, `-Dtornado.recover.bailout=False`, `-Dtornado.cuda.priority=100 -Dtornado.opencl.priority=0`, and one fork per class.
- `find src -name '*.java'`: 1185 files (476 under `src/test`). There are no `.cu`, `.c`, `.cpp` or `.h` files and no submodules.
- `utils/surface.py generate jitllm` finds 0 CUDA files and 0 components. Like mumax3, the whole surface is hand-enumerated below and in `surface.json`.

## Port strategy

Backend enablement in Java and in the two launchers. The existing CUDA behaviour stays byte-for-byte identical in what it selects:

1. **A distinct HIP backend identity.** Add `BackendId.HIP = of("hip")` and map `TornadoVMBackendType.HIP` to it in `TornadoDevices.backendId` (today it falls through to `BackendId.of("HIP")`, which is the same value but unnamed). Never fold HIP into `BackendId.CUDA`: cache keys, `nativeLibraryWeights`, and every `BackendId.CUDA.equals` gate must keep meaning NVIDIA.
2. **Per-architecture HIP capability grants** in `TornadoDevices.capabilitiesOf`. The architecture is read from the device description. A TornadoVM HIP device reports `Device version   : HIP <major>.<minor>`: the gfx major and minor that HIP gives as its compute capability (`HIPDevice.java:211-215`). Parse that with its own regex. Do not reuse `cudaComputeCapability`, whose regex requires `CUDA`, and never compare the value to sm numbers: AMD's 9.4 and NVIDIA's sm_94 are different things.

   Matrix-core families, mirroring TornadoVM's `TargetCaps.matrixCores()`:

   | HIP version | gfx | matrix cores | in this rule |
   |---|---|---|---|
   | 9.4 | gfx940/941/942 | MFMA | yes |
   | 9.5 | gfx950 | MFMA | yes |
   | 11.x | gfx11 | WMMA | yes |
   | 12.0 | gfx1200/1201 | WMMA | yes |
   | 12.5 | gfx1250 | WMMA | only with gfx1250 evidence |
   | 9.0 | gfx906/908/90a | ambiguous | see Q1 |
   | 10.x | gfx10 | none | no |

   9.0 is ambiguous: gfx906 has no matrix cores, while gfx908 and gfx90a do. How to treat 9.0 is ruling **Q1**.

   On HIP, a wrong MMA grant fails to compile (TornadoVM refuses with TC-E4017). With TornadoVM's default `recover.bailout=true`, that silently falls back to sequential Java. So the rule must be exact, not optimistic.

   Grants, each recorded only with the CPU-parity evidence listed in the test plan:

   - `TENSOR_CORE_MMA` and `INT8_TENSOR_CORE_MMA`: matrix-core archs per the table. Int8 lowers to `v_mfma_i32_4x4x4i8` on CDNA and to WMMA iu8 on gfx11/12.
   - `SHUFFLE_REDUCED_FP16_GEMV`: all HIP devices. It is a correctness grant; the preference remains in `Fp16GemvReductionPolicy`.
   - `PACKED_INTEGER_DOT`: all HIP devices. `dp4a` lowers to `tc_dp4a`, and the packed kernels' shuffle reduction is correct on HIP, unlike on OpenCL.
   - `SINGLE_PASS_RMS`: HIP. Today the grant keys on an NVIDIA platform name; add an explicit HIP condition, do not match on a name.
   - `SPLIT_KV_ATTENTION` and `PACKED_HALF2_MATH`: already granted to every non-OpenCL backend; keep them.
   - Leave `WARP_SHUFFLE` ungranted, as upstream deliberately does on CUDA. Leave `SUBGROUP_SHUFFLE_32` Metal-only.

   Any grant that fails parity on one family is withheld on that family, with the evidence, rather than loosened. See risk R3.
3. **Provider-agnostic vendor-library probes** in `NativePrefillSupport`/`NativeLibrarySupport`:
   - `Probe.CUBLAS`/`Probe.CUDNN` statically probe `CuBlasLibraryProvider::isAvailable`/`CuDnnLibraryProvider::isAvailable`. On a HIP target, probe the HIP providers (`HipBlasLibraryProvider`/`HipBlasLtLibraryProvider`, `MIOpenLibraryProvider`), or ask `LibraryRegistry.findProvider(CuBlas.LIBRARY_NAME, device)`, which already picks the HIP provider. The second is preferred because `probeSdpaThroughTornado` already does it.
   - `cublasLtResolvable()` (the `libcublasLt.so.12` loader-path guard against a cuDNN abort) and `cublasTooOld()` (Blackwell versus cuBLAS soname) apply only to a CUDA target. They must neither gate nor be evaluated for HIP.
   - `NativeLibrarySupport.unsupported`: change "CUDA and Metal only" to admit HIP, with the same tensor-core requirement as CUDA.
   - The install notice and `describe()` name hipBLAS/hipBLASLt and MIOpen on HIP.
   - Keep the `ops/nativelib/nvidia/` package and class names (`CublasMatMul` and the rest). They describe TornadoVM's `tornado.cublas`/`tornado.cudnn` library names, which HIP shares. Renaming them would be churn upstream did not ask for.
4. **Per-family native-library weights.** The 11 plan providers return `TornadoSupportSets.F16_AND_NATIVE_QUANTIZED` (or a family-specific set) only for `BackendId.CUDA`. Admit HIP through one shared predicate (for example in `TornadoSupportSets`), not 11 copies of `|| HIP`.
5. **Remaining backend gates**, each enabled on HIP only with the matching test evidence:
   - `Fp16KeyValueSupport.unsupported` `verifiedBackend` (the FP16 KV cache; the `GpuFp16Kv*AccelTest` classes);
   - `WeightPrepacking` line 155 (int8 prepack: `PACKED_INTEGER_DOT` and `INT8_TENSOR_CORE_MMA` on CUDA);
   - `Qwen35FFNLayers` line 1072 (split-KV decode attention for Qwen3.5);
   - `DeepSeek2PlanProvider` lines 61/72 (all layers CUDA-only "because they reduce with warp shuffles"). DeepSeek2 has no fixture; this is ruling **Q2**.
6. **Graph capture default** (`CudaGraphDefault`): TornadoVM's `withCUDAGraph()` captures a hipGraph on HIP (`HIPDeviceContext`). Default it on for HIP only after the benchmark shows it helps and the native-library combination survives it (risk R6). `--cuda-graphs`/`--no-cuda-graphs` keep their names: they are user-facing flags, and `-Djitllm.cudaGraphs` is the property.
7. **Launchers**:
   - `jitllm` (Python): `Backend.HIP`, `"hip-backend": Backend.HIP` in `BACKEND_NAME_MAP`, `tornado.hip.priority` in `BACKEND_PRIORITY_PROPERTY` (TornadoVM reads it: `TornadoOptions.java:156`), a place in `BACKEND_PRIORITY` (after CUDA, before OpenCL), and a `--hip` flag beside `--opencl/--cuda/--metal`.
   - `jitllm4j` (Java single-file launcher): the same four changes.
   - `scripts/tornadovm-dev.sh --backend hip`, and HIP detection when `--backend` is omitted (`rocminfo`/`amd-smi`, or `hipInfo` on Windows) before the OpenCL fallback.
8. **Test harness**: the `accel-tests` argLine pins `tornado.cuda.priority=100 tornado.opencl.priority=0`. Add `-Dtornado.hip.priority`, so that an `opencl,hip` SDK runs the Class B suite on HIP. Or document `-Dtornado.enabled.backends=hip`, if the porter finds that cleaner, but the suite must be runnable on HIP with one documented command. Extend the Class A unit tests that pin backend behaviour (`TornadoDevicesTest`, `NativeLibrarySupportTest`, `Fp16KeyValueSupportTest`, `BatchPrefillSupportTest`, `PlanSupportTest`, `DeviceResolversTest`, `BackendSpiTypesTest`, `ModelLoaderProviderBackendTest`, and the launcher tests in `scripts/tests/`) with HIP cases. Accel tests that `assumeTrue(BackendId.CUDA.equals(backend))` for a CUDA-specific reason (`Qwen35F16BatchedPrefillParityAccelTest`, `Qwen35PackedDecodeParityAccelTest`) get HIP admitted when the gate they mirror is opened for HIP. Their skip messages say "CUDA only (warp shuffles)", which is now wrong for HIP.
9. **Docs**: `README.md` backend line, `docs/architecture/models-and-backends.md`, `docs/architecture/development.md` (building a HIP SDK: `make jdk21 BACKEND=hip` in TornadoVM, or `scripts/tornadovm-dev.sh setup --backend hip`), and `docs/architecture/verification.md` (what is gated on HIP). The capability javadoc in `TornadoDevices.capabilitiesOf` and `TensorCoreSupport` ("MMA lowers only on CUDA") must be corrected.

## CUDA surface inventory

| surface | where | HIP mapping | risk |
|---|---|---|---|
| MMA m16n8k16 fp16/bf16 + ldmatrix + cp.async | 12 `*MMA*` layer/kernel files under `backend/tornado/kernels` and `layers` | TornadoVM TC: MFMA 4x4 (CDNA), WMMA (gfx11/12); cp.async synchronous | R2, R3 |
| MMA m16n8k32 s8 | int8 prefill (`INT8_TENSOR_CORE_MMA`), `WeightPrepacking` | `v_mfma_i32_4x4x4i8` / WMMA iu8 | R2 |
| `simdShuffleDown` (470) | shuffle-reduced GEMVs, packed decode, split-KV, Qwen35 attention, DeepSeek2 | 32-lane logical warps; wave64 = two half-waves (`TargetCaps.warps32`) | R1 |
| `dp4a` | packed-integer GEMV families | `tc_dp4a` | low |
| cuBLAS / cuBLASLt library tasks | `ops/nativelib/nvidia/Cublas*`, `Target.cublas()` | `HipBlasLibraryProvider` / `HipBlasLtLibraryProvider` (same `libraryName`) | R5 |
| cuDNN SDPA | `CudnnLayerAttention`, `CuDnnPrefillAttentionKernels`, `probeSdpaThroughTornado` | `MIOpenLibraryProvider` + CK FMHA (fp16, head dims 32/64/128/256) | R5 |
| CUDA graphs | `CudaGraphDefault`, `PipelineTransport.executeGraphs` | hipGraph via the same `withCUDAGraph()` | R6 |
| NCCL | `src/nccl/java/.../NcclTransport.java` (`-Pnccl`) | `tornado.nccl` picks RCCL for HIP devices | multi-GPU only |
| compute-capability gates | `TornadoDevices` (sm >= 80, >= 61), `cublasTooOld` (sm 100) | HIP: gfx version table above; Blackwell check CUDA-only | R4 |
| CUDA-only `BackendId` gates | about 20 files (list in `notes.md` intake) | `BackendId.HIP` admitted per evidence | R7 |
| Metal `simdgroupMatrix*` families | Metal kernels | unchanged, stay Metal-gated | none |

No textures, no device memory management beyond TornadoVM's, no JNI, no `System.loadLibrary`, no FFM downcalls in jitllm.

## Risk list

- **R1 wave64 shuffle semantics.** On CDNA, a 32-lane logical warp is half of a 64-lane wavefront. Kernels whose local-memory sizing or reduction tree assumes a 32-wide hardware SIMD are correct only if TornadoVM's half-wave lowering covers them. TornadoVM refuses (TC-E4019) a loop waiting on a local-memory write from its own wavefront, so that class fails to compile rather than hanging. The parity tests on gfx942 or gfx950 (wave64) and on gfx1100/1151/1201 (wave32) are the check; both widths are required.
- **R2 MMA via emulated fragments.** jitllm's MMA families have been validated only on NVIDIA. TornadoVM's TC tests cover the intrinsic, not jitllm's tiles. TornadoVM review item 9 (MMA helper `#else return c;`, which silently computes D = C for an arch the macros do not list) means that an arch admitted by the Java rule but missing from TornadoVM's macro routing gives wrong numbers without an error. Grant only the archs the table lists, and check on each new family that a parity test exercising MMA fails when the MMA is deliberately mis-granted (a negative control, once per family).
- **R3 gfx11/gfx12 WMMA drift over long k.** It is documented across every library: WMMA rounds per k-chunk. Projection k is the hidden size, 896 to 5120. If MMA-family parity fails only on RDNA, record the per-element error magnitude and withhold `TENSOR_CORE_MMA` on that arch family. Do not add a HIP-gated tolerance without jeff's ruling, because upstream's verification doc treats tolerances as fixed.
- **R4 compute-capability collision.** HIP 9.0 is gfx906, gfx908 and gfx90a (Q1). HIP 12.5 is gfx1250. The NVIDIA-only Blackwell check must not see an AMD number: `targetCudaComputeCapability()` already returns -1 for a non-CUDA backend; keep it so.
- **R5 vendor-library coverage gaps.**
  - CK FMHA has fp16 instances for head dims 32/64/128/256 only. Any layer whose head size falls outside that set (check `Gemma4Configuration.headSize()` and the other families' configurations) must reach the JIT attention through the existing SDPA probe. The porter must confirm the probe declines such a shape rather than throwing.
  - On Windows, TornadoVM's `ck-hip` builds only under its Visual Studio path with CMake 4 and Ninja. If skipped, SDPA reports unsupported (fallback, not failure).
  - hipBLASLt FP8 is unsupported on gfx942 (TornadoVM TestCuBlasLt 9 + 1). That does not matter today: `grep -ri 'e4m3\|fp8' src/main/java` finds nothing, so jitllm issues no FP8 GEMMs.
- **R6 graph capture with library handles.** Creating a hipBLASLt or MIOpen handle while any thread captures a graph exits the process. jitllm runs the native prefill outside the captured decode graph, but the validator must run `--with-native-libraries` together with graphs on, and the pipeline (multi-stage) mode, before graphs default on for HIP.
- **R7 a gate opened without evidence.** Each `BackendId.CUDA` gate encodes "measured on CUDA". Opening one for HIP without its parity test is the "claimed success over a subset" failure. `surface.json` lists each gate as its own component for that reason.
- **R8 device memory on one shared GPU.** The Class B suite sets `-Dtornado.device.memory=20GB`, forks a JVM per class, and loads models up to 7B (Mistral Q8_0), plus Qwen3.8-27B Q4_0 (16 GB file) for the Qwen3.5 classes. MI300X/MI355X have room. A 16 GB RDNA card (gfx1201, some gfx1100 boards) and the gfx1151 APU need the fixture subset that fits; record which classes skipped for memory and why.
- **R9 release coordinates.** `-P release` pins TornadoVM 8.1.0 from Maven Central, which predates the HIP backend. A HIP user needs a development SDK (`TORNADOVM_HOME`) until a TornadoVM release carries HIP. That is upstream's release decision; the PR body states it and changes nothing in the release profile.

## File-by-file change list

All paths are relative to the fork root. **(E)** means the change is enabled only with the evidence named in the test plan.

| file | change |
|---|---|
| `src/main/java/org/beehive/jitllm/runtime/backend/BackendId.java` | `HIP` constant |
| `src/main/java/org/beehive/jitllm/backend/tornado/device/TornadoDevices.java` | `backendId` HIP case; `hipVersion(deviceInfo)` parser; HIP grants per the table (E); javadoc |
| `src/main/java/org/beehive/jitllm/backend/tornado/TensorCoreSupport.java` | javadoc only |
| `src/main/java/org/beehive/jitllm/backend/tornado/ops/nativelib/nvidia/NativePrefillSupport.java` | provider-agnostic probes; CUDA-only loader and Blackwell checks; `describe()` |
| `src/main/java/org/beehive/jitllm/backend/tornado/NativeLibrarySupport.java` | admit HIP; HIP install notice; skip the CUDA-only checks on HIP |
| `src/main/java/org/beehive/jitllm/backend/tornado/ops/Target.java` | `cublas()` reads the agnostic probe (name kept) |
| `src/main/java/org/beehive/jitllm/backend/tornado/lowering/TornadoSupportSets.java` (or the shared helper the porter picks) | one "vendor BLAS backend" predicate |
| `src/main/java/org/beehive/jitllm/backend/tornado/plan/{Llama,Qwen2,Qwen3,Qwen35,Gemma4,Granite,Mistral,Phi3,Devstral,DeepSeekR1DistillQwen}PlanProvider.java` | `nativeLibraryWeights` via the predicate (E) |
| `src/main/java/org/beehive/jitllm/backend/tornado/plan/DeepSeek2PlanProvider.java` | per Q2 |
| `src/main/java/org/beehive/jitllm/backend/tornado/Fp16KeyValueSupport.java` | `verifiedBackend` admits HIP (E) |
| `src/main/java/org/beehive/jitllm/backend/tornado/plan/prepacking/WeightPrepacking.java` | admit HIP (E) |
| `src/main/java/org/beehive/jitllm/backend/tornado/layers/Qwen35FFNLayers.java` | split-KV decode admits HIP (E) |
| `src/main/java/org/beehive/jitllm/backend/tornado/CudaGraphDefault.java` | HIP default per measurement (E) |
| `src/main/java/org/beehive/jitllm/model/loader/ModelLoader.java` | read only: confirm the CPU-to-CUDA fallback label (line 125) does not mislabel a HIP run |
| `jitllm`, `jitllm4j` | HIP backend, priority property, `--hip` |
| `scripts/tornadovm-dev.sh` | `--backend hip`, HIP auto-detect |
| `pom.xml` | `accel-tests` argLine: HIP priority |
| `src/test/...` (Class A tests named above), `scripts/tests/test_launcher_{flags,commands}.py`, `scripts/tests/test_tornadovm_dev.py` | HIP cases |
| `src/test/.../golden/Qwen35F16BatchedPrefillParityAccelTest.java`, `Qwen35PackedDecodeParityAccelTest.java`, `kernels/Qwen35DecodeAttentionWarpNumericsAccelTest.java` | admit HIP where the mirrored gate opens; fix the skip message |
| `README.md`, `docs/architecture/{models-and-backends,development,verification}.md` | HIP backend documentation |

There are no new files beyond tests, unless the porter's shared predicate needs one.

## Build commands

The TornadoVM SDK is built once per host from upstream `develop` (or the release that first carries HIP), in the job's own directory:

```
git clone --depth 1 --branch develop https://github.com/beehive-lab/TornadoVM
cd TornadoVM
TORNADO_CK_GPU_TARGETS=<gfx> make jdk21 BACKEND=hip   # opencl,hip if the OpenCL comparison is wanted
source setvars.sh                                     # exports TORNADOVM_HOME
```

`TORNADO_CK_GPU_TARGETS=<gfx>` keeps the CK FMHA build to this host's arch (2.5 min for nine targets at 128 jobs). On Windows: `nmake /f Makefile.mak jdk21 BACKEND=hip` from a VS 2022 developer prompt with TheRock ROCm, CMake 4 and Ninja (TornadoVM notes).

jitllm itself:

```
./mvnw -Pjdk21 clean install -DskipTests     # or: make install
```

There is no arch flag; kernels compile at run time through hipRTC for the device present.

## Test plan

All Class B runs use `ulimit -c 0`, one pinned GPU (`HIP_VISIBLE_DEVICES`), and `-Dtornado.recover.bailout=False`, which the profile already sets. That flag is what turns a refused or failed HIP kernel into an error instead of a silent Java fallback.

1. **Class A (no GPU), must not regress:** `./mvnw -Pjdk21 test` and `make test-scripts`. Run them on the unmodified upstream first to get the baseline count, then on the port, with the new HIP unit cases added.
2. **Device and launcher smoke:** `./jitllm --hip -m Llama-3.2-1B-Instruct-Q8_0.gguf -p "..." --instruct`. Assert that the resolved backend prints HIP and that the output is coherent. `tornado --devices` lists the HIP device. Run `jitllm4j` the same way.
3. **Class B on HIP:** `./mvnw -Pjdk21,accel-tests verify` with `JITLLM_TEST_MODELS` pointing at the fixtures (the `GoldenFixture.java` list; Hugging Face GGUFs; SHA-256 pinned in the test resources). The required classes, by the capability whose grant they justify:
   - baseline (no new grants): `GoldenLogitsAccelTest` (drops to parity tolerance off the pinned tuple, as designed), `*CpuGpuParityAccelTest` (Llama, Qwen2, Qwen3, Qwen35, Granite, Phi3, Mistral, Gemma4, Qwen2MoE), `CompiledProgramIdentityAccelTest`, `Lowered*ParityAccelTest`.
   - `TENSOR_CORE_MMA` / `INT8_TENSOR_CORE_MMA`: `*BatchedPrefillParityAccelTest`, `Qwen35Mma*ParityAccelTest`, `NarrowChunk*AccelTest`, `BatchedEngineAccelTest`, `WeightPrepackingPolicyAccelTest`, plus the negative control from R2.
   - `SHUFFLE_REDUCED_FP16_GEMV`, `PACKED_INTEGER_DOT`, `SINGLE_PASS_RMS`: the F16 and Q8_0 `*CpuGpuParityAccelTest` and `Lowered*ParityAccelTest` with the grant on, `Qwen35PackedDecodeParityAccelTest`, `Qwen35DecodeAttentionWarpNumericsAccelTest`, `Fp16DeterminismProbe` (the repeated-capture `bit_exact` check that once found a racy RMS reduction).
   - FP16 KV: every `GpuFp16Kv*AccelTest`, the `*Fp16KvSequenceReset*AccelTest` classes, `EngineFp16KvAccelTest`.
   - native libraries: `NativeLibrariesAccelTest`, `Gemma4Q8_0NativePrefillParityAccelTest`, `Gemma4Q4_0NativePrefillParityAccelTest`, `TaskGraphChainQwen3NativeAccelTest`.
   - graphs: the parity set above re-run with `-Djitllm.cudaGraphs=true`, and once with native libraries plus graphs (R6).

   Report pass/fail/skip per class with the skip reason. A skip is never counted as a pass. The upstream suite is the GPU test; no new test harness is needed, only HIP cases in the existing unit tests.
4. **Both wavefront widths and Windows:** the full set on one wave64 platform (gfx942 or gfx950) and one wave32 platform (gfx1100, gfx1151 or gfx1201). On windows-gfx1151, steps 1 to 3 within what fits the APU's memory (R8).
5. **Multi-GPU (`-Pnccl`), where a host can give the job two GPUs:** `docs/architecture/multi-gpu.md`'s pipeline run with `NcclTransport` on RCCL, checked against the single-GPU output. This is evidence only, not a gate: other jobs share the fleet hosts.
6. **Benchmarks, for notes.md and the numbers jeff promised on TornadoVM#1327:** decode tok/s and prefill tok/s for Llama-3.2-1B and Qwen3-1.7B in F16 and Q8_0 on gfx942/gfx950 and one RDNA card. Use the `docs/architecture/verification.md` procedure (3 warm-up runs, median of 5), with and without native libraries and graphs, and the OpenCL backend on the same card as the "before" figure. They are not posted anywhere by an agent.
7. **CUDA no-regression:** jitllm has no native CUDA compile, so the check is the following.
   - On the mi355 fleet host, build jitllm against a `BACKEND=cuda` TornadoVM SDK and run Class A, which pins `capabilitiesOf` for CUDA inputs.
   - Have a unit test assert that CUDA grants are unchanged for representative `deviceInfo` strings (sm_61, sm_80, sm_90, sm_100).
   - Record in `notes.md` that the CUDA selection paths are byte-identical (`git diff` shows only additive HIP branches).
   - An NVIDIA GPU run of the Class B suite, if the fleet's NVIDIA host is available, is additive evidence.

## Open questions

The two rulings are Q1 and Q2 at the top of the `notes.md` planning section. The non-blocking items there (native image, release coordinates, benchmark publication) need no ruling to start.
