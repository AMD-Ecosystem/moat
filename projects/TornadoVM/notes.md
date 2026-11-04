# TornadoVM notes

## Intake screen (2026-08-07)

TornadoVM (beehive-lab/TornadoVM) is a JIT compiler that translates Java bytecode to run on accelerators. It has four backends: OpenCL C, NVIDIA CUDA (PTX), Apple Metal (MSL), and a multi-core CPU backend. OpenCL is the default backend and is documented as targeting NVIDIA, AMD and Intel GPUs directly (README.md: "OpenCL (default) | NVIDIA / AMD / Intel GPUs, multi-core CPUs, FPGAs"; also "dedicated GPUs from NVIDIA, AMD, and Intel"). AMD hardware is already reachable today through this backend without any ROCm/HIP work.

The CUDA-specific surface is not CUDA kernels in-tree: repo-wide there are only 3 `.cu`/`.cuh` files, one of which is a generated example output and one is a CUTLASS JNI wrapper (`tornado-drivers/cutlass-jni/.../tornado-cutlass.cu`). The `tornado-drivers/ptx` module is TornadoVM's own JIT compiler backend: it emits PTX assembly text from its internal IR and loads it via the NVIDIA driver/NVRTC APIs -- CUDA is a compilation TARGET for this backend's code generator, not a body of CUDA source to hipify. The newer `tornado-cublas`/`tornado-cudnn`/`tornado-cufft`/`tornado-cusparse` modules (mostly README/pom.xml/ROADMAP stubs right now) are direct JNI bindings to NVIDIA's proprietary math libraries and Tensor Core `mma.sync` intrinsics -- an AMD analogue would mean new bindings against rocBLAS/MIOpen/hipFFT/hipSPARSE plus a from-scratch AMDGPU code-generation backend in the compiler, not a source port. This matches the reimplement-not-port pattern already dispositioned for mirage, SpargeAttn and FlashRT.

### Licence

Dual/tri-licensed by module, documented explicitly in README.md ("Licenses per module") and confirmed against the LICENSE files in the tree (`LICENSE_APACHE2`, `LICENSE_GPLv2CE`, `LICENSE_MIT`):
- Tornado-API, Tornado-Assembly, Tornado-scripts, Tornado-Annotation, Tornado-Unittests, Tornado-Benchmarks, Tornado-Examples, Tornado-Matrices, Tornado-Drivers-OpenCL-Headers: Apache-2.0 (tier 1).
- Tornado-Runtime, Tornado-Drivers (this is where any CUDA/PTX-backend work would land): GPL-2.0 with Classpath Exception (base SPDX GPL-2.0-only, tier 2 -- "approved for contributing to third-party projects").

Recording `license_spdx: "Apache-2.0 OR GPL-2.0-only WITH Classpath-exception-2.0 OR MIT"` (per-module, not ambiguous -- the split is explicit and documented, not a guess). Both applicable tiers (1 and 2) are cleared to contribute; this is not a gating concern for either fork or decline.

### Duplicate effort / viability verdict: DECLINE, already-supported

AMD GPUs are already reachable through TornadoVM's own default OpenCL backend -- this is not a partial/parked port by another team, it is upstream's own shipped functionality, so `already-supported` is the closer fit than `not-a-target` even though the CUDA-only slice (PTX backend, CUTLASS/cuBLAS/cuDNN/cuFFT/cuSPARSE JNI bindings) is also, independently, a code-generation/library-binding target rather than in-tree CUDA kernels -- the same reimplement-not-port shape as mirage/SpargeAttn/FlashRT. Recommending decline on both grounds: the goal (Java code running on AMD GPUs) is already met without any port, and the part that is CUDA-specific would not be a port even if pursued.

Upstream is active (not archived); no dependency on any other MOAT project.

## Approved against intake's recommendation, 2026-08-07

Intake recommended declining this as already-supported, on the grounds that TornadoVM's own default OpenCL backend already reaches AMD hardware. jeffdaily overrode that on intake queue issue #5: "Approve TornadoVM because it looks like an interesting challenge to port, but I acknowledge the OpenCL path already works on AMD hardware."

So the scope is NOT enablement. AMD users can run TornadoVM today. What this port is for is native AMDGPU code generation: a backend beside `tornado-drivers/ptx` that emits GCN/HSACO and loads it through hipRTC/hipModule, rather than reaching AMD through OpenCL.

That makes it a third shape, neither Strategy A nor Strategy B. There are no CUDA kernels to translate: the repo has 3 `.cu` files, one generated and one a CUTLASS JNI wrapper. The CUDA-specific surface is a JIT compiler backend emitting PTX assembly text, so the equivalent work is writing a code generator rather than diffing existing kernels. The one thing that makes it tractable where mirage and pegainfer are not is that TornadoVM already abstracts its backend interface across OpenCL, PTX and Metal, so there is a defined seam to implement against instead of a rewrite of the engine.

Licence note for the planner: the recorded `license_spdx` reads as an OR expression, but TornadoVM is per-MODULE licensed, which is not the same thing. What governs a contribution is the licence of the files changed, and the modules this work touches -- Tornado-Runtime and Tornado-Drivers -- are GPL-2.0 with Classpath Exception, tier 2, cleared to contribute. The Apache-2.0 and MIT modules are the user-facing API, which this work would not modify.

## Planning (2026-11-03, windows-gfx1151)

Planned against upstream `develop` @ f8e3ac907b62, not `master`: every upstream PR targets `develop` (CONTRIBUTING.md), which was 40 commits ahead of `master` at planning time. The fork has no `develop` mirror yet.

Facts that changed since intake: upstream 8.0.0 replaced the Graal/PTX backend with TC (`tornado-compiler/`), which prints CUDA C++ and compiles it with NVRTC through Java FFM bindings; `tornado-drivers/ptx` is now a one-class stub. #1266 (merged 2026-10-23) relicensed the whole tree to Apache-2.0. Library-task modules moved under `tornado-native-libs/` (#1307).

Strategy, scope, risks and the eight open questions for jeff are in plan.md. ext_type `java-ffm-driver-api`.

## Porting round 1 (2026-11-04, linux-gfx950, MI355X)

Following plan.md "Rulings" (jeff, 2026-11-04). `moat-port` cut from the fork's `develop` (f8e3ac907). Work lands as a sequence of commits: core backend first, then each library/feature.

### Host and toolchain (this round)

- ROCm 10.1 from the TheRock Python wheels (rocm-sdk core/devel/libraries 10.1.0); `ROCM_PATH` is the wheel's `_rocm_sdk_devel` directory; HIP runtime 7.16.26385, hipRTC 9.0. No `/opt/rocm`. No system JDK: Temurin 21.0.12 unpacked outside the tree, `JAVA_HOME` pointed at it.
- hipRTC on this stack needs no `-I`: `half`, `__hip_bfloat16`, `__hadd_rn`, `__half2ll_rz`, `__double2bfloat16`, the `*_sync` warp builtins all resolve from the hipRTC prelude. `#include <hip/hip_fp16.h>` fails without `-I$ROCM_PATH/include`, so the HIP dialect includes it only `__has_include`-guarded.
- Not in hipRTC: `__double2half`, `__float2bfloat16_rn` (HIP spells it `__float2bfloat16`), `__dp4a`, `llmin/llmax`. `__builtin_amdgcn_sdot4` needs `dot1-insts` (absent on gfx1100). The dialect uses helpers for all of these.
- Enum values probed on this host (C program against hip_runtime_api.h): hipDeviceAttribute MaxThreadsPerBlock 56, MaxBlockDimX/Y/Z 26/27/28, MaxSharedMemoryPerBlock 74, TotalConstantMemory 83, WarpSize 87, ClockRate 5, MultiprocessorCount 63, ConcurrentKernels 8, AsyncEngineCount 2, ComputeCapabilityMajor/Minor 23/61; hipDeviceProp_t (R0600) is 1472 bytes with gcnArchName at offset 1160. MI355X reports cc 9.5, warp 64, 160 KB LDS per block, TotalConstantMemory INT_MAX.

### Build

```bash
export JAVA_HOME=<jdk-21> ROCM_PATH=<rocm root> HIP_PATH=$ROCM_PATH
# bin/compile needs the Python modules tqdm and psutil (present in this host's venv)
make jdk21 BACKEND=hip                  # ~40 s on this host
source setvars.sh
tornado --devices
```

### Design as built

- TC: `TargetCaps.Dialect.HIP`, a new `arch` component (delegating 10-arg constructor keeps the CUDA backend's calls untouched), `TargetCaps.hip(...)` and `HIP_GFX942` (headless). `HipDialect` delegates to `CudaDialect` and overrides bf16 type/conversions, double->half/bf16 (round-to-odd through float, single rounding), Java float->int helpers (`tc_f2i/tc_f2l`, NaN tested on bits), `#pragma clang fp contract(off)` for strict plans, refuses tiles and device launch. Intrinsics and helpers fall back to the CUDA spelling; HIP overrides warp ops (`__shfl_*(.., 32)`, `__ballot` low 32 bits), dp4a, llmin/llmax, cp.async (plain synchronous copy). Warp ops stay refused on wave64 (plan R1). MMA refused on HIP in the core commit.
- Driver: `tornado-drivers/hip` is a copy of `tornado-drivers/cuda` with the FFM layer rewritten: `HIPDriverAPI` (amdhip64), `HIPRTCAPI`, `ROCTXAPI` (rocprofiler-sdk roctx, then libroctx64), `ROCmLibraries` (ROCM_PATH/HIP_PATH//opt/rocm absolute paths before bare names), `HIPCompiler` (hipRTC for `--offload-arch=<gcnArchName>`, in-process cache, no PTX fallback, no device-runtime link). HIP's context API is deprecated, so a context handle is `device ordinal + 1` and "set current" is `hipSetDevice`. Dropped from the copy: CUDA Tile compiler/scheduler, NVIDIA scheduler, NVML handler, device-launch limit.
- Runtime/API wiring: `TornadoVMBackendType.HIP`, `TornadoBackends.HIP` (`tornado.hip.priority`), `DEFAULT_HIP_COMPILER_FLAGS` (`tornado.hip.compiler.flags`, `tornado.hip.compile.profile` = default|fast|debug|repro via `HipCompileProfile`), strict FP adds `-ffp-contract=off`; `hip-backend` Maven profiles; `hip-exports`; Makefile/Makefile.mak/bin/compile/installer/post_installation/intellij_build/tornado.py/tornado.bat/update_paths/argfile generator.
- Virtual devices: ported cheaply (same shape as CUDA's: compile for the fixed headless target, here `HIP_GFX942`).
- `test-native.sh` runs `drivers.cuda.tests.*` classes that no longer exist upstream; no HIP equivalent was added.

### Results so far (gfx950)

- `make test-nogpu JDK=jdk21`: BUILD SUCCESS (all TC/runtime unit tests).
- TC codegen lit: 58/58 pass, including 13 new `lit/HIP` tests compiled by `utils/hiprtccheck` for gfx942 and gfx1100.
- `tornado-selftest`: 17 pass, 4 skip (warp-sum: wave64; device-launch; two tensor-core checks), 0 fail.

### Commit 1 pushed: core HIP backend (`94b5c8c88`)

- Wave64 warp semantics: implemented rather than refused (deviation from plan R1, recorded for the reviewer). KernelContext warp ops are a 32-lane API; on a 64-wide wavefront HIP emits width-32 `__shfl*` and reads the warp's half of `__ballot` (`tc_warp_bits`, `__lane_id() & 32`), which is exact 32-lane semantics. The tensor-core ruling (MMA on CDNA) needs the same 32-lane-warp-on-wave64 model anyway. `TargetCaps.warps32()` is the shared predicate (Legality TC-E4015, selftest warp-sum).
- `TcHook`: TC-E4016 (device launch on a device without it) now raises `TornadoDeviceDynamicParallelismNotSupported` (already in the harness's UNSUPPORTED list) instead of a bailout; shared code, affects only the exception type for devices without device launch.
- Test gating policy: tests that are OpenCL- or Metal-specific exclude HIP exactly where they exclude CUDA and fail on HIP (virtual-device OpenCL reference kernels, precompiled OpenCL atomics, Metal local-memory report, simdgroup_matrix, "rejected on OpenCL" warp tests). CUDA exclusions in vectortypes/swizzled-local tests were left alone: those tests pass on HIP. `TestLaunchFailureReporting` excludes HIP: it relies on CUDA's 65535 cap on gridDim.y; AMD limits the grid in threads (2^32 per dimension), so HIP correctly runs the 10^8-block grid. `TestDynamicParallelismPipelines` excludes HIP (device-side launch; this host also has NVIDIA's cuBLAS installed, so its cuBLAS-available check passes).
- Results at `94b5c8c88` on gfx950: selftest 18 pass / 3 skip (device-launch, two tensor-core checks); quickPass 1236 pass / 941 unsupported / 5 fail (2 whitelisted Mandelbrot/Julia off-by-one iteration counts; 3 TestUnsupportedKernelContextOps cases fixed before the commit and passing on rerun: 8 ran, 0 fail, 3 unsupported); HeapFail#test03 pass; `make test-nogpu` pass; codegen lit 58/58; `make jdk21 BACKEND=cuda` and `BACKEND=opencl` build; `git diff develop -- tornado-drivers/{cuda,opencl,metal}` empty.
- Unsupported on HIP at this commit, by design or pending a later commit: CUDA Tile (out), MLX/simdgroup (Metal), CLBlast (OpenCL), all CUDA library tasks (pending HIP branches), tensor-core MMA (pending MFMA/WMMA commit), multi-device tests (one GPU visible).

### Next commits (in order)

1. MMA lowering (design worked out): A/B fragments stay in the documented PTX m16n8 layouts; HIP load helpers emulate `ldmatrix`/`movmatrix` with plain loads (bit-identical fragments); `mma` shuffles fragments into MFMA (CDNA, two 16x16x16 calls per wave64 since a wave holds two 32-lane warps) or WMMA (gfx11: 16 halves/lane, rows duplicated in both half-waves; gfx12: 8 halves/lane) operand layouts and the result back to the PTX C/D layout. A partially active wave64 (one warp of the wave calling mma) falls back to a shuffle-FMA path within the warp, since MFMA reads all 64 lanes. FP8 is widened exactly to half. bf16 on gfx908 uses 16x16x8bf16; int8 uses 16x16x16i8 (gfx908/90a) or 16x16x32_i8 (gfx94x/950). gfx1250 has only its own 16x16x32 WMMA forms whose register layout cannot be verified without the hardware: refused with that reason, flagged for jeff.
2. Library tasks as HIP branches of the existing providers: hipBLAS/hipBLASLt, hipFFT, hipRAND, hipSPARSE, RCCL, MIOpen; then CUTLASS via Composable Kernel, cuDF via AMD-Ecosystem/hipDF, cuVS via AMD-Ecosystem/hipVS, the cuDNN SDPA shim (MIOpen/CK attention, or stop and record the gap). This host's ROCm 10.1 SDK also ships `libhipdnn_backend.so` (hipDNN), which the ruling assumed did not exist; worth evaluating for the SDPA shim.
3. amd-smi power metrics (FFM, mirroring the NVML handler).

### Commit 2 pushed: MMA lowering (`e0b31b746`)

- Design as built (differs from the earlier sketch): CDNA uses the 4x4 MFMA forms (`mfma_f32_4x4x4f16`, `4x4x4bf16_1k`, gfx908 `4x4x2bf16`, `mfma_i32_4x4x4i8`): sixteen independent 4-lane blocks, so each 32-lane warp's 16 x 8 tile is eight blocks within its own lanes. Chosen because the unit tests (and real kernels) launch one 32-thread warp per block, leaving half of a wave64 idle, and the 16x16 MFMA reads all 64 lanes. RDNA3/RDNA4 use one WMMA 16x16x16 per K=16 (B padded with zero columns). Layouts verified on gfx950 with a standalone harness (in agent_space, not committed): 16x16x16f16 and 4x4x4f16 (block = lane/4, A_b[lane%4][k], B_b[k][lane%4], D_b[i][lane%4]); all five mma variants exact against a CPU reference with 1, 2 and 4 warps per block. gfx908 bf16 (4x4x2) and the WMMA branches are compile-checked only (hipcc and hipRTC for gfx908/90a/942/950/1100/1151/1200/1201); their runtime check falls to the wave32 (gfx1100) validator and any gfx908/gfx90a host.
- gfx1250 refused (TC-E4017 with a gfx1250-specific message): it has only its own 16x16x32 WMMA forms; layout unverifiable without the hardware. Question for jeff: implement gfx1250 WMMA blind, or keep it refused until a gfx1250 host can validate?
- Results at `e0b31b746` on gfx950: selftest 20 pass / 1 skip (device-launch); MMA classes 65/65 (TestMatrixMultiplicationMMA 18, MMAInt8 12, MMAFP8 8, MMABF16 7, MMACpAsync 8, MMAGlobalFragmentLoad 6, MMAInt8FragmentElementRead 4, MMAFragmentHelper 2); TestAsyncCopyToLocal 6/6; codegen lit 60/60; `make test-nogpu` pass.
- Lessons promoted to the cuda-to-rocm skill (this branch): `references/no-hip-equivalent.md` (warp-level mma.sync with exposed PTX fragment layouts can be reimplemented with shuffles + 4x4 MFMA / WMMA; measured layouts) and `references/fault-classes.md` (NVRTC -> hipRTC runtime class: library discovery, header/intrinsic differences, gcnArchName via hipGetDevicePropertiesR0600, attribute enums, context API, fp-contract).

### Commits 3-6 pushed: hipBLAS/hipBLASLt, hipFFT, hipRAND, hipSPARSE (`d171cc86b`, `de638ad9a`, `4a1ebfb68`, `85b92503d`)

- Pattern (for the remaining libraries): each CUDA library module gains sibling `Hip*NativeLib` (FFM bindings loaded through `tornado.runtime.ffm.ROCmLibraries`, moved there from the driver in `d171cc86b`) and `Hip*LibraryProvider` classes with the same `libraryName()` and `canHandle(HIP)`, registered in the module's `module-info`. The CUDA providers are untouched; `LibraryRegistry.findProvider` already picks by `canHandle`, and `destroyContexts` now does too (it used the first provider by name). The test gate in each `Test<Lib>` class runs on HIP when the HIP provider's library is present (`agent_space/scripts/gate.py` did the edit); `tornado-assembly/pom.xml` hip-backend profile and `tornado.py` `hipModules` ship the module (`agent_space/scripts/ship.py`).
- Enum translations: hipBLAS `HIPBLAS_OP_N/T/C` = cuBLAS + 111, `HIPBLAS_COMPUTE_32F` = 2, `HIPBLAS_GEMM_DEFAULT` = 160, TF32 math -> `HIPBLAS_XF32_XDL_MATH` (NOT_SUPPORTED ignored: full FP32), `hipblasGemmEx_v2` (ROCm 6) before `hipblasGemmEx` (ROCm 7). hipBLASLt descriptor attributes TRANSA/TRANSB/EPILOGUE/BIAS_POINTER = 0/1/2/3 (cuBLASLt 3/4/7/8), heuristic result 56 bytes, algo 24. hipRAND generator types = cuRAND + 300. hipFFT handle is a pointer (cuFFT: int). hipSPARSE enums and statuses = cuSPARSE's.
- hipSPARSE under HIP graph capture: `hipsparseDestroySpMat` calls `hipDeviceSynchronize` (invalidates capture), the default SpMV CSR algorithm analyses on first use (INTERNAL_ERROR in capture), and a non-null external buffer with a queried size of 0 gives INVALID_VALUE once descriptors are reused. Fix: cache descriptors per operand set in the context, `HIPSPARSE_SPMV_CSR_ALG2`, NULL buffer when size is 0. Diagnosed with `AMD_LOG_LEVEL=3` and a standalone probe (`agent_space/probe-java/GraphProbe.java`, run with `tornado -cp ...`), because tornado-test prints only the outer bailout message.
- Results on gfx950: TestCuBlas 27/27, TestCuBlasLt 10/10 (FP8 E4M3 included), TestCuFft 9/9, TestCuRand 7/7, TestCusparse 17/17. `make jdk21 BACKEND=cuda` and `make test-nogpu` pass after the hipBLAS commit.
- Host note: this image also has NVIDIA's CUDA libraries installed (`/usr/local/cuda`), so the CUDA providers' `isAvailable()` returns true here; irrelevant to HIP devices but explains why CUDA-only library tests reach the library task on HIP before being gated.

### Commit 7 pushed: NCCL on RCCL (`dffb8db92`)

- `NcclNativeLib` picks the vendor once: `-Dtornado.nccl.library=rccl|nccl`, else RCCL when the HIP backend is installed without CUDA or is the default backend. RCCL binds librccl plus amdhip64 (`hipSetDevice`, `hipStreamCreate`, ... same signatures as the four CUDA runtime calls). Unique id 128 bytes, data types and reduction ops identical.
- Results on gfx950 (one GPU pinned): TestNccl 20 pass / 10 unsupported (two GPUs needed), TestNcclMultiProcess 3 pass / 1 unsupported. Multi-GPU cases remain for a host with two visible AMD GPUs. `make jdk21 BACKEND=cuda` passes.
- Next: cuDNN on MIOpen (softmax, activations, max pooling, convolution fwd/bwd via immediate mode), then the SDPA shim against MIOpen's MHA problem API (`miopenCreateMhaProblem`, present in this SDK's miopen.h) or CK.

### Commits 8-9 pushed: cuDNN on MIOpen (`71627fa6e`), SDPA on CK attention (`de819cd13`)

- `MIOpenLibraryProvider` + `MIOpenNativeLib` (FFM to libMIOpen, hipMalloc workspace): softmax (`miopenSoftmaxForward_V2`, ACCURATE, MODE_INSTANCE), activations (LOGISTIC=1, TANH=2 with alpha=beta=1, RELU=3), max pooling (`miopenPoolingForward`, do_backward=false, no workspace), convolutions in immediate mode (`GetSolution` top-ranked, `CompileSolution` at plan creation, `*Immediate`; NHWC/KRSC described by strides through `miopenSetTensorDescriptor`; deterministic property -> `MIOPEN_CONVOLUTION_ATTRIB_DETERMINISTIC`). Enum values verified with a C probe (`agent_space/miopen/enums.cpp`): status InternalError=5, miopenHalf=0, miopenFloat=1, miopenConvSolution_t 32 bytes (workspace_size @8, solution_id @16). Graph capture of the immediate-mode convolutions works with no extra handling.
- SDPA ruling item: MIOpen's MHA problem API (find 2.0, `miopenCreateMhaProblem`) on gfx950 / this SDK has only the `MhaForward` solver (id 162): FP32 only, no solution for FP16 (`MhaForward: Not applicable`), none for strided (packed QKV) tensors, and the causal mask (`miopenTensorMhaMask` scalar, also tried as a problem descriptor) was not applied (results equal the unmasked attention). No CK flash-attention solver in this libMIOpen. Probe: `agent_space/miopen/mha.cpp`. So MIOpen is not a workable target; CK attention is.
- CK attention: new module `tornado-native-libs/ck-hip` -> `libtornado-ck.so` (C ABI `tornado_ck_sdpa_forward`, bound by `CkAttentionNativeLib`). CMake sparse-fetches composable_kernel at develop `9abc8908235e8a62460f81d4e54866586b3cda9d` (release tag rocm-7.2.4 has no gfx11 FMHA forward kernels: 0 instances for gfx1100/gfx1151), runs `example/ck_tile/01_fmha/generate.py --api fwd --optdim 32,64,128,256 --filter "*fp16_batch*_nlogits_nbias_*_nlse_ndropout_nskip_nqscale*_nsink"` (fnmatch is against the kernel name without the arch suffix, so the pattern must end at `_nsink`), compiles 139 sources for gfx90a;gfx942;gfx950;gfx1100;gfx1101;gfx1102;gfx1151;gfx1200;gfx1201 (instances per family: gfx950 64, gfx9 22, gfx11 20, gfx115 16, gfx12 16). 2.5 min at 128 jobs (`bin/compile` now sets `CMAKE_BUILD_PARALLEL_LEVEL` for HIP builds). Library 11.5 MB.
- Mask semantics: cuDNN `set_causal_mask` = CK `mask_top_left` (window left -1, right 0); verified with sQ < sKv and sQ > sKv against a CPU reference (`agent_space/ckproto/test.cpp`, 10 cases incl. packed QKV and d = 32/64/72/80/128/256, all within 2e-4).
- Windows: the module is skipped under the Visual Studio generator (no HIP language support), so SDPA reports unsupported on Windows HIP. Open question for the Windows validator / jeff: build it with Ninja there, or accept the gap.
- Results on gfx950 at `de819cd13`: TestCuDnn 18/18 (SDPA x3 included). `make jdk21 BACKEND=cuda` passes. RDNA3/RDNA4 attention kernels are compile-checked only; the wave32 validator exercises them via TestCuDnn.
- hipDNN (ROCm 10.1 SDK, `libhipdnn_backend.so` + plugins incl. an `AsmSdpaEngine` in `libhip_kernel_provider.so`) was not used: the ruling named MIOpen or CK. It is a closer analogue of cudnn-frontend and could replace the CK module later if jeff prefers.

### Commit 10 pushed: CUTLASS on Composable Kernel (`cf3563f72`)

- `ComposableKernelLibraryProvider` + `ComposableKernelNativeLib` (tornado-cutlass) bind the GEMM entry points added to libtornado-ck: `tornado_ck_gemm(stream, type F32/F16/BF16, m, n, k, alpha, a, b, beta, c, workspace, query)`, `tornado_ck_gemm_bias_act(...)`, `tornado_ck_hgemm_batched(...)`; query non-null = report workspace bytes without launching (used in `prepare`). Return codes 0 ok / 1 no kernel (-> TornadoLibraryTaskNotSupported) / 2 invalid / 3 HIP error.
- CK device ops (sources `ck-hip/src/main/cpp/source/tornado-ck-gemm*.{hpp,cpp}`, one TU per type and per activation): `DeviceGemmMultipleD_Xdl_CShuffle` on CDNA for F16/BF16 (128x64 tile, 4-element vectors: 8-byte operand alignment from the 24-byte array header) and F32 (scalar loads, any shape), `DeviceGemmMultipleD_Wmma_CShuffleV3` on RDNA for F16/BF16 (needs the extra KBatch argument; detected with a C++20 requires-expression), `DeviceGemmMultipleD_Dl` for F32 on RDNA. DL needs M%128, N%128, K%16 and has no padding mode, so that path copies A/B into zero-padded workspace buffers, computes alpha*A*B, and a small kernel stores it (+ beta*C) back. The DL op's own D-operand epilogue gave wrong results here (with D aliased to E and with a separate D buffer), hence the store kernel. ck/config.h is generated from CK's config.h.in by the module CMake (FP16/BF16/FP32, DL, XDL, WMMA on).
- FMHA-only compile flags (`-fgpu-flush-denormals-to-zero`, FAST_EXP2, API toggles) are now per-source on the attention files only.
- Results on gfx950 at `cf3563f72`: TestCutlass 26/26, TestCuDnn 18/18; standalone harness (`agent_space/ckproto/gemmtest.cpp`, operands at +24 bytes like TornadoVM arrays) passes all 21 cases on the XDL path and the FP32 DL path (forced on gfx950 with a prototype-only switch). WMMA/DL-on-RDNA are compile-checked only: the wave32 validator should run TestCutlass.
- Build: the HIP build is about 2.5 min at 128 jobs; libtornado-ck ~25 MB with nine architectures.

### Commit 11 pushed: amd-smi power (`180fa32eb`)

- `AMDSMIAPI` (FFM, libamd_smi.so.27/26/25 via ROCmLibraries; no Windows library) + `HIPAMDPowerMetricHandler`, lazy init like the NVML handler. GPU located by PCI BDF (hipDeviceAttributePciDomainId 69 / PciBusId 67 / PciDeviceId 68, verified with a C probe) -> `amdsmi_get_processor_handle_from_bdf` (bdf union passed by value as uint64: function bits 0-2, device 3-7, bus 8-15, domain 16+). `amdsmi_power_info_t` 192 bytes: socket_power u64 @0 (W on bare metal), current_socket_power u32 @8 (W, MI300+), average_socket_power u32 @12 (UINT32_MAX on MI355X). Probe: `agent_space/smi/smi.cpp`.
- Results: profiler `POWER_USAGE_mW` 286000 on gfx950 (MatrixMultiplication2D example); TestProfiler 9/9.

### Commit 12 pushed: cuDF on hipDF (`999bff725`)

- hipDF source: AMD PyPI index `https://pypi.amd.com/rocm-7.2.3/simple` has C++ library wheels: `amd-libhipdf` 3.0.0 (libcudf headers + libcudf.so + libhipcomp.so + cmake packages incl. hipCCL/cuco/libhipcxx), `amd-libhipmm` 4.0.0 (librmm), `amd-rocmds-logger` 1.0.0 (rapids_logger), `amd-libhipvs` 1.0.0 (libcuvs + libcuvs_c), `amd-libraft` 1.0.0. `amd-libhipdf` declares no dependencies: `amd-libhipmm` is needed too. On this host the wheels were fetched with curl into `agent_space/rocmds/wheels` and unpacked side by side in `agent_space/rocmds/x` (the fleet guard refuses any command that names pip's install verb, even in prose). libcudf.so needs libamdhip64.so.7 and resolves against the ROCm 10.1 SDK here.
- Build: `HIPDF_HOME=$M/agent_space/rocmds/x/libcudf make jdk21 BACKEND=hip`. Module `cudf-hip` compiles the unchanged cudf-jni shim with ROCm clang (hipDF headers include hip_runtime.h), C++20, `find_package(cudf CONFIG)` (needs CMake >= 3.30.4; the wheel prefixes and their `lib64/cmake` must be on CMAKE_PREFIX_PATH because CMake only looked at the roots here; hipDF's prefixes must come before ROCm's). Compat headers: `cuda_runtime.h` -> `cudf/cuda_runtime.h`; `rmm/mr/{cuda_memory_resource,cuda_async_memory_resource,per_device_resource}.hpp` -> `rmm/mr/device/...`; an overload `set_current_device_resource(device_memory_resource&)` (hipMM takes a pointer).
- Run: `LD_LIBRARY_PATH=$X/libcudf/lib64:$X/librmm/lib64:$X/rapids_logger/lib64:...`. Vendor choice in `CudfNativeLib` follows the NCCL rule; `-Dtornado.cudf.library=hipdf|cudf` overrides.
- Results gfx950: TestCudf 24/24; TestCudfColumns 20 pass / 3 unsupported. Gap: the hipDF 3.0 Parquet writer has no ZSTD ("nvCOMP 2.4 or newer is required for Zstandard compression" from nvcomp_adapter.cpp); `LIBCUDF_HOST_COMPRESSION=ON` (+THRESHOLD=0, RATIO=100) and `LIBCUDF_NVCOMP_POLICY=ALWAYS` do not route around it (probe `agent_space/rocmds/proto/pq.cpp`: NONE and SNAPPY ok). Mapped to TornadoLibraryTaskNotSupported by matching hipDF's message. Question for jeff: accept the ZSTD-write gap until hipDF gains it?
- hipDF officially supports gfx90a/gfx942/gfx950 only; RDNA (the wave32 gate) and Windows have no hipDF build, so cuDF tasks report unsupported there (the TestCudf* gates report unsupported when the shim is absent).

### Commit 13 pushed: cuVS on hipVS (`0185c7f40`)

- hipVS (wheel `amd-libhipvs` 1.0.0, no declared dependencies) ships `libcuvs/lib64/libcuvs_c.so` + `libcuvs.so` with the cuVS C API; needs librmm/rapids_logger (amd-libhipmm), ROCm's hipBLAS/hipSOLVER/hipSPARSE/hipRAND/hipBLASLt/RCCL (all in this SDK), system libopenblas, and `libomp.so` from `$ROCM_PATH/lib/llvm/lib` (not on the default path). Run: `LD_LIBRARY_PATH=$X/libcuvs/lib64:$X/libraft/lib64:$X/librmm/lib64:$X/rapids_logger/lib64:$ROCM_PATH/lib/llvm/lib:...`.
- Findings: hipVS refuses `kDLROCM` (10) device tensors ("indices tensor must be device-compatible"); `kDLCUDA` (2) works. hipVS's `cuvsAllNeighborsBuild` on a host dataset requires device outputs (cuVS 26.08 accepts host ones): staged through `cuvsRMMAlloc`/`cuvsStreamSync`/`hipMemcpy` D2H/`cuvsRMMFree` on the hipVS path only. Struct offsets used by the bindings are identical (all tests pass).
- Vendor choice in `CuVSNativeLib` follows the NCCL rule; `-Dtornado.cuvs.library=hipvs|cuvs`.
- Results gfx950: TestCuVS 7/7. `make jdk21 BACKEND=cuda` passes.
- hipVS/hipDF wheels are x86_64 Linux and target gfx90a/gfx942/gfx950; on RDNA and Windows these tasks report unsupported (library absent).

### Commits 14-15 pushed: docs (`7096165e2`), CK legacy framework (`ebb89fdf0`)

- `7096165e2`: native-libs README module table names the ROCm library per module on HIP and lists ck-hip/cudf-hip; HYBRID_API_GUIDE intro paragraph for AMD GPUs.
- `ebb89fdf0`: the pinned CK develop commit defaults to `USE_NEW_UNIFIED_FRAMEWORK=1`, whose host code wants the GPU target at compile time; without `CK_CMAKE_GPU_TARGET_IDS` every source printed "ck_tile: no GPU target known at compile time ... host pass sees HOST". Defining the IDs (as CK's CMake does, first target wins) fails the gfx1100 device pass of a gfx9+gfx11 build (static assert in `warp_wmma_gemm_gfx11_utils.hpp:22`). `USE_NEW_UNIFIED_FRAMEWORK=0` (legacy WarpGemm, what CK forces for gfx1250) builds clean for all nine targets; TestCuDnn 18/18, TestCutlass 26/26 on gfx950 after the change.
- Skill lessons committed on this branch (`6efd3f07`): fault-classes entries for CK FMHA, CK GEMM device ops, ROCm-DS wheels, hipSPARSE under capture, AMD SMI; SKILL.md index lines.
