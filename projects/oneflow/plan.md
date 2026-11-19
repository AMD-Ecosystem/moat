# oneflow -- ROCm/HIP port plan (linux-gfx90a lead)

## Project
- Name: oneflow
- Upstream: https://github.com/Oneflow-Inc/oneflow
- Default branch: master
- Clone HEAD analyzed: 25c8978 ("support npu multinomial (#10668)"), shallow depth-1
- Lead platform: linux-gfx90a (MI250X, CDNA2, wave64), ROCm 7.2.1

## Existing AMD support: removed-from-master ROCm fossils -> finish/revive (NOT auto-skip, NOT GPUMD-style live-rot)
oneflow historically carried its own ROCm/HIP backend, but current master has had it **completely removed**. Findings (read-only):
- No HIP build plumbing anywhere: zero `enable_language(HIP)`, `find_package(hip)`, hipcc, `HIP_ARCHITECTURES`, rocBLAS/MIOpen/RCCL/hipFFT, or hipify in `cmake/`, `CMakeLists.txt`, `external/`, `docker/`, `ci/`. The only `cmake/cuda.cmake` path is `find_package(CUDAToolkit REQUIRED)` + `enable_language(CUDA)`.
- No ROCm device type: `oneflow/core/common/device_type.proto` enumerates kCPU, kCUDA, kMockDevice, kMeta, kMLU, kNPU, kXPU -- **no kROCM**. The execution-provider tree `oneflow/core/ep/` has `cpu/` and `cuda/` only; there is no `ep/rocm/`.
- README/docs/pip wheels mention only CUDA (cu118); GitHub commit search for rocm/hip returns nothing on master.
- Surviving ROCm fossils (inert, prove prior support, harmless to keep): `oneflow/core/kernel/util/numeric_limits.cuh` (header comment "very specific to ROCm HIP ... to support HIP compilation", copied from PyTorch's ROCm-aware NumericLimits.cuh); one `#if defined(__HIP_DEVICE_COMPILE__)` guard in `oneflow/core/ep/include/primitive/fast_integer_math.h`; the DLPack `kDLROCM=10` device enum in `oneflow/api/python/dlpack/dlpack.h` (interop standard, unrelated to backend).

Decision: **proceed** with a from-scratch HIP enablement of the CUDA backend. This is a port (the CUDA code has no live HIP path), but the surviving fossils + the clean `ep/` device abstraction mean the framework was *designed* to host a second GPU backend. It is **not** a GPUMD-style "build the bit-rotted makefile.hip and fix the rot" -- there is no makefile.hip to build; the HIP backend must be reconstructed. It is **not** an auto-skip (no ROCm support exists on master). Per-op math kernels are mostly mechanical; the perf-critical fused/attention kernels are CUTLASS-based and out of the first slice (see below).

## Build classification: pure CMake (Strategy A), NOT a torch extension
Evidence:
- `CMakeLists.txt` is a standalone `project(oneflow C CXX)` with `option(BUILD_CUDA "" ON)`; `cmake/cuda.cmake` does `find_package(CUDAToolkit REQUIRED)` + `enable_language(CUDA)` + `set_source_files_properties(... LANGUAGE CUDA)` via the glob.
- **No** `find_package(Torch)`, no `torch.utils.cpp_extension`, no `CUDAExtension` anywhere. `python/setup.py` packages the CMake-built artifacts; it does not drive a torch BuildExtension.
- `.cu`/`.cuh` are gathered by `file(GLOB_RECURSE)` in `cmake/oneflow.cmake` (lines 41-84) and compiled by the CUDA language into ONE monolithic shared lib `oneflow` (`oneflow_add_library(oneflow SHARED ${of_all_obj_cc})`, line 224); the Python C-extension `oneflow_internal` (pybind11, line 410) links that lib.

So: Strategy A (compat header + `LANGUAGE HIP` on the existing `.cu`), driven by oneflow's own `oneflow_add_library` wrapper as the retag hook. PyTorch is involved only as the **test oracle** (the autotest suite imports torch), never as a build dependency.

## Port strategy: Strategy A, colmap/MPPI model, applied at framework scale
1. One CUDA->HIP compat header (e.g. `oneflow/core/device/cuda_to_hip.h`) aliasing the ~114 `cudaXxx` symbols oneflow actually uses (`cudaMalloc`/`cudaMemcpyAsync`/`cudaStream_t`/`cudaError_t`/`cudaEvent_t`/`cudaGetErrorString` etc.), the cuRAND/cuBLAS/cuFFT/cuSOLVER spellings, and the warp-mask helper. On NVIDIA it is a no-op `#include <cuda_runtime.h>`. Force-include it on every HIP TU via `CMAKE_HIP_FLAGS -include` (MPPI lesson) so defines precede each file's includes. Generate the alias block from pytorch hipify `cuda_to_hip_mappings.py` filtered to oneflow's grep'd symbol surface + a hand tail (cuRAND/cuBLAS-Lt enums hipify omits).
2. In CMake, add `option(USE_HIP)`; under it `enable_language(HIP)`, set `CMAKE_HIP_ARCHITECTURES` from the cache var (default the lead arch only when unset -- never a literal "gfx90a", CudaSift/Gpufit lesson) and retag every `.cu`/`.cuh` `LANGUAGE HIP` inside the `oneflow_add_library` wrapper (which already wraps every target). Map the math libs: cuBLAS->hipBLAS, cuBLASLt->hipBLASLt, cuRAND->hipRAND, cuFFT->hipFFT, cuSOLVER->hipSOLVER, cuDNN->MIOpen, NCCL->RCCL, CUB->hipCUB, Thrust->rocThrust (header drop-ins on /opt/rocm/include).
3. Add a `kROCM` path the minimal way: oneflow's whole host backend is gated on `#ifdef WITH_CUDA` (271 sites) and the device code on `WITH_CUDA`/`__CUDA_ARCH__`. Rather than introduce a brand-new `kROCM` DeviceType (which would touch the proto enum + every registry + thousands of `DeviceType::kCUDA` switch arms), **define `WITH_CUDA` FOR the HIP build** and let `kCUDA` mean "the GPU backend" (AutoDock-GPU lesson: define USE_CUDA for HIP so the shared driver/alloc/launch blocks compile unchanged); gate only the genuinely NVIDIA-only includes (`<cuda.h>`, `<nvml.h>`, nvjpeg/npp) behind `defined(WITH_CUDA) && !defined(USE_HIP)`. The compat header retargets the `cuda*` runtime symbols to `hip*`. This keeps `kCUDA` as the device the Python `device="cuda"` maps to (so torch's ROCm `torch.cuda` and oneflow's `cuda` line up in autotest) with a far smaller diff than a new device type. Revisit a true `kROCM` enum only if a downstream check requires it.

Port-vs-rewrite for perf kernels: the first slice is **mechanical, correctness-first**. The CUTLASS/CuTe and TensorRT-flash fused kernels (attention/GLU/conv) are excluded from the slice (CUTLASS does not port to ROCm -- guide); a later AMD-native (CK/ck_tile) pass is where they would be reimplemented, out of scope here.

## CUDA surface inventory
Scale (whole `oneflow/` tree): 220 `.cu` + 18 `.cuh` (~52k LOC), 166 files with `__global__`, 114 unique `cudaXxx` symbols, 215 `cudaStream_t` / 262 `cudaError_t` refs, 522 `WITH_CUDA` refs (271 `#ifdef WITH_CUDA`), 137 `__CUDA_ARCH__` sites.
- Kernels / launches: pervasive; `kCudaThreadsNumPerBlock=512`, `OF_CUDA_CHECK`, `CudaCurrentDeviceGuard` in `oneflow/core/device/cuda_util.h`.
- Warp intrinsics: 27 sites. Hot path: `oneflow/core/cuda/{softmax,layer_norm,rms_norm}.cuh` each hardcode `constexpr int kWarpSize = 32` and do `__shfl_xor_sync(0xffffffff, val, mask)` butterfly reductions (templated on `thread_group_width = kWarpSize`). Also `oneflow/core/device/cuda_util.h:141 const int32_t kCudaWarpSize = 32;`. `__shfl*` also in `layer_norm_gpu_kernel.cu`, `fused_matmul_bias_add_relu_dropout.cu`, `data_shuffle_kernel.cu`, `core/embedding/lru_cache.cu`.
- Atomics: `oneflow/core/cuda/atomic.cuh` -- CAS-loop emulation for half/bf16/binary-op atomics (`atomicCAS` on `unsigned short`/`unsigned int`/`unsigned long long`), `atomicAdd` for half/bf16. Uses device memory (hipMalloc), not managed.
- Libraries: cuBLAS (26 files), cuDNN (26), NCCL (76 -- comms/boxing, mostly host .cpp), cuFFT (5), cuRAND (23), cuSOLVER (5), CUB (61), Thrust (1), nvjpeg (3), npp (1), CUTLASS (10, all `#ifdef WITH_CUTLASS`-guarded). cuSPARSE: 0.
- Textures/surfaces: none found in the core slice (oneflow is a DL framework, not imaging) -- the colmap/popsift texture/pitch/layered fault classes do not apply here.
- Streams/events/pinned/managed: `cudaStream_t`/`cudaEvent_t` throughout `ep/cuda/cuda_stream.*`, `cuda_event.*`; `NumaAwareCudaMallocHost` (pinned). Managed memory: not in the core slice.

## Scoped first GPU-validatable slice (validation-scope, NOT build-scope)
KEY CONSTRAINT: oneflow links ALL `.cu` + `.cpp` into one monolithic `liboneflow.so` (`oneflow_add_library(oneflow SHARED ${of_all_obj_cc})`); there is no per-op sub-target. Unlike cupoch (separable CORE_ONLY), you **cannot** build only a kernel subset -- the whole lib must compile and link under HIP before any Python op runs. So the slice is a **validation** scope, not a compile scope:
- Compile scope (what must build under HIP): the whole `liboneflow.so` with `-DWITH_CUTLASS=OFF -DWITH_MLIR=OFF -DBUILD_RDMA=OFF -DWITH_ONEDNN=ON -DBUILD_CPP_API=OFF` and CUTLASS/flash/conv-cutlass kernels elided (already `#ifdef WITH_CUTLASS`). nvjpeg/npp (image decode, 4 files) and NCCL multi-GPU (single-GPU validation) gated out where they need NVIDIA-only headers.
- Validation scope (what is GPU-checked): the `oneflow/core/ep/cuda/primitive/` core tensor-op layer (30 `.cu`: elementwise unary/binary, broadcast binary math/logical/comparison, add, cast, fill, tensor_fill, constant_pad, copy_nd, permute, softmax, softmax_backward, where, broadcast_matmul) plus the hot reduction headers `core/cuda/{softmax,layer_norm,rms_norm}.cuh`. These are exercised by named Python autotests: `test_add`, `test_cast`, `test_where`, `test_masked_fill`, `test_softmax`, `test_layer_norm`, `test_matmul`, `test_broadcast_ops`, `test_permute`, plus activation/elementwise tests. Each runs the op on oneflow `device="cuda"` (-> HIP) AND on PyTorch as reference, forward + backward, and asserts agreement.

## Risk list
- **wave64 warp size (PRIMARY).** `kWarpSize=32`/`kCudaWarpSize=32` are hardcoded; CDNA2 is 64. The butterfly reductions are templated on `thread_group_width`, so the value flows into shfl widths. Fix per guide: a build-time/per-arch `kWarpSize` (64 on `__GFX9__`, else 32; runtime `prop.warpSize` on host) and replace the literal `kWarpSize=32` defaults. Static shared arrays sized by warp count need a compile-time upper bound.
- **32-bit shuffle mask will NOT COMPILE (AutoDock-GPU lesson, line 176).** `__shfl_xor_sync(0xffffffff, ...)` -- ROCm 7.x `__shfl_*_sync` static_assert `sizeof(mask)==8`. The literal `0xffffffff` fails during hipcc's HOST pass of the `.cu`. Define a USE_HIP full-warp mask `0xffffffffffffffffULL` in the compat header (keyed on USE_HIP, not on wave width) and route the reductions through it. This bites softmax/layer_norm/rms_norm immediately.
- **`__CUDA_ARCH__` undefined on HIP (cudaKDTree/MPPI/raft/RXMesh family, 137 sites).** Every `#ifdef __CUDA_ARCH__` device-vs-host selection silently takes the HOST branch during HIP device compilation (HIP uses `__HIP_DEVICE_COMPILE__`), risking host-fn-from-device calls or, worse, a host pointer dereferenced in a kernel -> runtime fault. Mechanically rewrite to `#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)` (and the `#ifndef` complement). Numeric `#if __CUDA_ARCH__ >= NNN` intrinsic gates get `&& !defined(USE_HIP)` to take the portable fallback unless an AMD intrinsic is wired. `fast_integer_math.h` already uses the correct `||__HIP_DEVICE_COMPILE__` form -- follow that pattern. (Same `OF_NUMERICS_FUNC` macro in numeric_limits.cuh is `#if defined(__CUDACC__)`-gated; hipcc defines `__HIPCC__` not `__CUDACC__`, so add `||defined(__HIPCC__)` -- the cudf/cudaKDTree `__CUDACC__`-empty-macro family.)
- **Library enum/handle deltas.** hipBLAS v2 enums, hipBLASLt API surface, hipFFT/hipRAND status-enum coverage not 1:1 (MPPI lesson -- map only codes that exist, USE_HIP-guard orphan cases), MIOpen is NOT a drop-in for cuDNN (different descriptor/algo-find API; the `ep/cuda` conv/pooling/normalization primitives that call cuDNN may need real adaptation -- keep cuDNN-heavy ops OUT of the first slice if MIOpen mapping balloons).
- **Monolith build cost / latent dead code.** Enabling previously-dead `#ifdef __CUDA_ARCH__` device code on HIP can surface bugs never executed under CUDA (RXMesh lesson). The whole-lib link means a single unportable TU blocks all validation -- expect to `list(FILTER ... EXCLUDE)` the deepest offenders (nvjpeg/npp, cuSOLVER-only, CUTLASS) off the validated path and document covered-vs-deferred (cupoch lesson).
- **TREAT_WARNINGS_AS_ERRORS=ON.** hipcc/clang emits different warnings than nvcc (e.g. nodiscard on hipEvent timing -- amgcl lesson); will need `-DTREAT_WARNINGS_AS_ERRORS=OFF` for bringup or targeted suppressions.
- **atomicMin/Max-on-coarse-memory (cudaKDTree) -- LOW risk here:** oneflow uses device memory, not hipMallocManaged, so the silent-drop class likely does not apply; verify if any embedding/unique kernel uses managed memory.

## File-by-file change list (lead bringup)
- `cmake/oneflow.cmake` -- add `USE_HIP` handling in the `oneflow_add_library` wrapper (retag `.cu`/`.cuh` `LANGUAGE HIP`, set `HIP_ARCHITECTURES` from cache var); gate the CUDA-language path under `if(NOT USE_HIP)`.
- `cmake/cuda.cmake` -- under `USE_HIP`: skip `find_package(CUDAToolkit)`/`enable_language(CUDA)`; `find_package(hip hipblas hipblaslt hiprand hipfft hipsolver miopen rccl)`; build VENDOR_CUDA_LIBRARIES from the hip:: targets; force-include the compat header via `CMAKE_HIP_FLAGS`.
- New `oneflow/core/device/cuda_to_hip.h` -- the compat header (symbol aliases, full-warp 64-bit mask, `__HIPCC__`->`__CUDACC__` and device-pass `__CUDA_ARCH__` shims as needed, `WITH_CUDA` defined-for-HIP shared-block enablement, libc-before-hip include order per gpuRIR).
- `oneflow/core/device/cuda_util.h` -- per-arch `kCudaWarpSize`; guard nvjpeg/nvml-only bits.
- `oneflow/core/cuda/{softmax,layer_norm,rms_norm}.cuh` -- per-arch `kWarpSize`, route `__shfl_xor_sync` through the compat full-warp mask.
- `oneflow/core/cuda/atomic.cuh` -- verify CAS-loop atomics compile under HIP (half/bf16 atomicAdd exist on gfx90a); USE_HIP-guard any NVIDIA-intrinsic atomic.
- `cmake/third_party/{cutlass,flash_attention,trt_flash_attention,nccl,cub}.cmake` and the CUTLASS `set_source_files_properties` block in oneflow.cmake -- gate behind `if(NOT USE_HIP)` for the first slice (or map cub->hipCUB).
- Mechanical `__CUDA_ARCH__`/`__CUDACC__` guard rewrites across `oneflow/core` and `oneflow/user/kernels` for files on the slice's compile path (driven by build errors).

## Build commands (gfx90a)
There is no ROCm cmake cache shipped; start from the CUDA cache and override. Approximate first-bringup configure (porter will iterate):

```
cd projects/oneflow/src && mkdir -p build && cd build
cmake .. \
  -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DBUILD_CUDA=ON \                      # WITH_CUDA path == the GPU backend; HIP retargets it
  -DWITH_CUTLASS=OFF -DWITH_MLIR=OFF -DBUILD_RDMA=OFF -DBUILD_CPP_API=OFF \
  -DBUILD_TESTING=OFF -DTREAT_WARNINGS_AS_ERRORS=OFF \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
  -DPython3_EXECUTABLE=$(which python3)
cmake --build . -j$(nproc)              # produces liboneflow.so + oneflow_internal*.so into python/oneflow/
```

Local CPU-only compile smoketest (manual only, never a gate): `rocm/dev-ubuntu-24.04:7.2.4-complete`.
gfx1100/gfx1151 followers: same commit, only `-DCMAKE_HIP_ARCHITECTURES=<arch>` changes (arch read from the cache var, no source edit) -- but note RDNA is wave32, so the warp-size fixes above must be correct for both 32 and 64.

## Test plan
- GPU correctness gate (the real validation): oneflow's Python autotest, which runs each op on oneflow `device="cuda"` (HIP) and on PyTorch as the reference oracle, forward + backward, and asserts numerical agreement. Requires a ROCm-build PyTorch importable in the env (test-time oracle, not a MOAT build dep). Build oneflow, set `PYTHONPATH` to the built `python/`, then run the slice's tests:
  - `python -m pytest python/oneflow/test/modules/test_add.py test_cast.py test_where.py test_masked_fill.py test_softmax.py test_layer_norm.py test_matmul.py test_broadcast_ops.py test_permute.py test_activation.py -v`
  - Run SERIALLY on a single assigned GPU (MPPI lesson: `pytest` not `-n`, parallel procs hammering one GPU cause false transient failures).
  - autotest internally randomizes shapes/dtypes and compares vs torch within tolerance; treat a clean pass of the per-op modules above as the slice's validation. Capture `AMD_LOG_LEVEL=3` once to confirm device kernels actually dispatch on gfx90a.
- Determinism: oneflow ops are deterministic for elementwise/softmax; assert run-to-run agreement where the op is not atomically-reduced. layer_norm/softmax butterfly reductions are fixed-order -> bit-stable; flag any atomic-fan-out reduction (data_shuffle/embedding) as tolerance-not-bitwise.
- Non-GPU regression set (must not break): the CPU autotest path (`device="cpu"`) of the same modules, and the C++ gtests gated by `BUILD_TESTING` (run a CPU subset). The NVIDIA build path must stay byte-identical (Strategy A: only `.cu` see HIP, gated by `USE_HIP`).

## Inter-project MOAT deps
None. oneflow vendors all 32 third-party libraries via its own `cmake/third_party/*` (including its own CUTLASS fork at `CMakeLists.txt:214`); it does not consume rmm/raft/cudf or any other MOAT target as a build dependency. No `set-deps` needed. (PyTorch-ROCm is a test-time oracle for the autotest suite, satisfied by the environment, not an inter-project build dep.)

## Open questions
- MIOpen-vs-cuDNN adaptation depth: the `ep/cuda` conv/pool/batchnorm/softmax primitives that call cuDNN may need real API rework (MIOpen is not a drop-in). Decision: keep cuDNN-heavy ops OUT of the first validated slice; the slice's softmax is the hand-written `core/cuda/softmax.cuh` reduction, not the cuDNN softmax. Revisit cuDNN->MIOpen as a second slice.
- Whether to introduce a real `kROCM` DeviceType later, or permanently let `kCUDA`+USE_HIP denote the GPU backend (smaller diff, mirrors how torch-ROCm keeps `torch.cuda`). First slice uses the latter.
- hipBLASLt coverage for `cublas_fused_mlp_grad`/`fused_matmul_bias` grouped-GEMM kernels (cuBLASLt-heavy) -- likely defer to a later slice with the fused kernels.
- Monolith compile time on the bringup host (220 `.cu` under hipcc) -- may need `-j` tuning / ccache; record in notes.md once measured.
