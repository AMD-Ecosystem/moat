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
