# Port plan: aihwkit (IBM Analog Hardware Acceleration Kit)

## Project
- Name: aihwkit
- Upstream: https://github.com/IBM/aihwkit
- Default branch: main
- Lead platform: linux-gfx90a (CDNA2, wave64)
- Domain: PyTorch library simulating analog in-memory computing (resistive crossbar / RPU tiles). C++/CUDA backend "RPUCuda" exposed to Python via a pybind11 module (`rpu_base`).

## Existing AMD support
NONE. From-scratch HIP port.
- Web search ("aihwkit ROCm", "aihwkit AMD GPU HIP", "IBM aihwkit RPUCuda AMD MI300 gfx9"): no ROCm/HIP port, no separately-named AMD project (no ROCm-DS analogue). IBM ported Qiskit to ROCm, not aihwkit. setup.py classifier is `Environment :: GPU :: NVIDIA CUDA`; only a CUDA.Dockerfile ships.
- `gh api repos/IBM/aihwkit/forks`: no fork under ROCm/AMD/GPUOpen orgs and none with rocm/hip/amd in the name. The two name-substring hits (ZhipingWoods/aihwkit, PJLAB-CHIP/AnalogAIold) are false positives -- plain mirrors, no AMD work.
- Upstream branches: no rocm/hip/amd branch. No AMD issues/PRs found.
- Decision: clean from-scratch HIP port targeting ROCm. No authoritative or community base to adopt.

## Disposition
TRACTABLE -- proceed with a Strategy-A mechanical HIP port. Effort class: MEDIUM (37 `.cu` files, ~10k LOC of device code, but a clean and small external-API surface: cuBLAS + cuRAND + CUB only, NO textures/surfaces, NO CUTLASS, NO managed memory). The single real semantic risk is one warp-size-coupled bit-packed data format in `bit_line_maker.cu`. Dispatch a porter.

NOT a CK reimplementation: the only CUTLASS references in the tree are three CODE COMMENTS in `forward_backward_pass.cu` ("eventually we might want to use cutlass", "CUTLASS will help") -- aspirational, no CUTLASS/CuTe code, no wgmma/mma_sync, no tensor-core path. The matvec is plain cuBLAS GEMM/GEMV plus hand-written analog-noise-injection kernels. A mechanical HIP port is correct and sufficient; no AMD-native GEMM rewrite is needed for correctness.

## Build classification: cmake (Strategy A) -- evidence
This is a PyTorch library but it does NOT use `torch.utils.cpp_extension`. The CUDA is compiled by a standalone CMake/nvcc path; Torch is found only for headers/linkage.
- `setup.py:12` -- `from skbuild import setup` (scikit-build invokes CMake; no `CUDAExtension`/`BuildExtension`).
- `CMakeLists.txt:9-11` -- `project(aihwkit C CXX)`, `option(USE_CUDA ...)`; `cmake/dependencies_cuda.cmake:9` -- `enable_language(CUDA)`, `:12` -- `find_package(CUDAToolkit)`. Native CMake CUDA language, not torch's extension machinery.
- `CMakeLists.txt:108,168,228` -- `set_property(TARGET ... PROPERTY CUDA_ARCHITECTURES ${RPU_CUDA_ARCHITECTURES})` with default `"75;80;89"` (`:26`). Arch is a CMake property, the Strategy-A pattern.
- `CMakeLists.txt:102` -- `target_link_libraries(RPU_GPU RPU_CPU cublas curand ...)`: links CUDA libs by bare name directly.
- Torch is linked only as `torch_cuda`/`c10_cuda`/`torch_python` (CMakeLists.txt:104,159-161; src/aihwkit/simulator/CMakeLists.txt:9,13). No `find_package(Torch)` that drives extension compilation; torch supplies tensor-buffer headers (RPU_TORCH_CUDA_BUFFERS) and the pybind link.
- `src/aihwkit/simulator/CMakeLists.txt:8` -- `pybind11_add_module(rpu_base MODULE ...)` linking the CMake-built `RPU_GPU` static lib (`:13`). The Python extension is a CMake pybind target, not a torch CUDAExtension.

Conclusion: standalone CMake + nvcc, Torch-for-linkage-only. Use Strategy A (USE_HIP CMake option + a single compat header), NOT Strategy B (no torch hipify involved). Set ext_type = `cmake`.

## Port strategy: A (compat header + enable_language(HIP))
Mirror the compat-header approach and the existing `USE_CUDA` switch with a parallel `USE_HIP`.
1. Add one compat header (e.g. `src/rpucuda/cuda/cuda_to_hip.h`) included once from `cuda_util.h` (which already centralizes `cublas_v2.h`, `cuda_runtime.h`, `curand.h`, `curand_kernel.h` at lines 18-23). On HIP it includes `<hip/hip_runtime.h>`, `<hipblas/hipblas.h>`, `<hiprand/hiprand_kernel.h>`, `<hipcub/hipcub.hpp>` and aliases the cuda* spellings the project uses to hip*; on CUDA it is a no-op. Keep `<cstring>/<cstdlib>` included BEFORE the HIP runtime (gpuRIR lesson: host memcpy/memset can resolve to HIP __device__ overloads otherwise).
2. In CMake, add `option(USE_HIP ...)`; when set, `enable_language(HIP)`, mark the glob'd `${RPU_GPU_SRCS}` `LANGUAGE HIP`, set `HIP_ARCHITECTURES` from `${CMAKE_HIP_ARCHITECTURES}` (default gfx90a only when unset -- never hardcode the literal, so followers need no source edit), and link `hipblas hiprand` instead of `cublas curand`. Keep `RPU_USE_CUDA` defined on the HIP build (the `#ifdef RPU_USE_CUDA` blocks in rpu_base.cpp gate the CudaAnalogTile pybind exposure and the whole device tile code; the compat header makes "CUDA-spelled" code compile under HIP).
3. Disable IPO/LTO for the HIP build of the pybind module (gpuRIR lesson: HIP link does not finalize LTO -> empty PyInit_ -> ImportError). Check whether scikit-build/CMake enables INTERPROCEDURAL_OPTIMIZATION; force it OFF for HIP.
4. Guard the few genuinely divergent spots with `#if defined(USE_HIP)`; keep rare.

Because RNG state and BLM use CUDA-library types in headers shared with host code (`curandState_t` in cuda_util.h, vector_device.h, transfer_device.h, etc.), the compat header must alias those device-RNG and cuBLAS type/enum names project-wide -- the aliasing surface is larger than colmap's but still mechanical.

## CUDA surface inventory
- Kernels: ~hundreds of `__global__`/`__device__` across 37 `.cu` (RPU tile forward/backward matvec with injected analog noise, pulsed stochastic weight update, noise/weight management, transfer devices). All standard HIP-portable.
- cuBLAS (cuda_math_util.cu, cuda_util.h/.cu, rpucuda_pulsed_device.cu): Sgemm/Dgemm/Hgemm, Sgemv/Dgemv, Sger/Dger, Sscal/Dscal, Scopy/Dcopy, Snrm2/Dnrm2; cublasHandle_t, cublasSetStream, cublasSet/GetPointerMode (HOST/DEVICE), cublasSetPointerMode. -> hipBLAS, 1:1. WATCH: hipBLAS v2 enum spellings (CUBLAS_POINTER_MODE_DEVICE -> HIPBLAS_POINTER_MODE_DEVICE, op-N/T enums); device-pointer-mode path (getBlasDeviceHandle / cublasHandle_t* device_handle_ in cuda_util.h) -- verify hipBLAS supports the same device-pointer scalar mode.
- cuRAND: BOTH APIs. Host generator `curandGenerator_t rng_` (cuda_util.h:403) and device per-thread state `curandState_t` / `curandStateXORWOW` with `curand_init`, `curand_normal`, `curand_uniform`, `curandSetStream`, `curandSetup` (cuda_util.h:500). -> hipRAND, 1:1 (hiprand_kernel.h device API + host generator). WATCH: XORWOW generator state layout and stream-ordered seeding; analog noise is the core of the simulation, so RNG correctness is gated by the CPU-vs-CUDA test (see Test plan). Bitstream values will NOT match CUDA bit-for-bit (different RNG impl) -- tests compare statistically / against CPU tile, not against a fixed CUDA seed (confirm tolerance-based asserts).
- CUB (rpu_cub.h wraps `<cub/cub.cuh>` in namespace RPU::cub; used in noise_manager.cu, maximizer.cu, update_management_helper.cu, weight_clipper_cuda.cu, mixedprec, cuda_buffer.h): DeviceReduce/DeviceScan-class and `cub::BlockScan<T, thread_block_size>` (block-width templated, NOT warp-width). -> hipCUB. rpu_cub.h must include `<hipcub/hipcub.hpp>` and use the hipCUB namespace under HIP; the CUB_NS_PREFIX wrapping needs a hipCUB-compatible spelling. WATCH the wave64 TempStorage-reuse race (CV-CUDA lesson) on any back-to-back block-collective sharing TempStorage.
- Thrust: none.
- Warp intrinsics: concentrated in bit_line_maker.cu -- `__shfl_up_sync`, `__shfl_sync`, `__ballot_sync` (all with mask `0xFFFFFFFF`), `warpSize`; `__popc` in pwu_kernel.h (operates on a 32-bit packed K-word, logical not wavefront), `__popcll` in cuda_math_util.cu (operates on a uint64 per-thread flicker_state, NOT a ballot -- arch-independent, fine).
- FP16: cuda_fp16_util.h + RPU_USE_FP16/RPU_BFLOAT_AS_FP16 (experimental, OFF by default; CMakeLists comments "only supported for A100+"). Port path: hip_fp16 / hip_bfloat16. Defer: build with FP16 OFF (default) for the lead port; flag RPU_USE_FP16 as a follow-up.
- Atomics: atomicAdd, atomicCAS, atomicOr, plus a project `atomicMaxFP` helper (CAS-based float max). Standard; verify atomicMaxFP CAS loop compiles/behaves under HIP.
- Pinned memory: cudaHostAlloc in io_manager.cu -> hipHostAlloc/hipHostMalloc, 1:1.
- Textures/surfaces: NONE. (Entire popsift/CV-CUDA texture-fault-class set does not apply.)
- Managed memory: none.
- Streams/events: standard (per-context stream + cublas/curand SetStream). 1:1.
- CUTLASS/CuTe/wgmma/tensor-core: NONE (only aspirational comments). No CK rewrite needed.

## Risk list
1. WARP-SIZE-COUPLED SERIALIZED DATA FORMAT (PRIMARY RISK -- dietgpu class). bit_line_maker.cu encodes the stochastic pulse train as a bit-packed format: `nK32 = (Kplus1+31)/32` 32-bit words, `laneId = threadIdx.x & 0x1f`, `__ballot_sync(0xFFFFFFFF, stoch_value < value)` whose 32-bit result IS one word of the `CudaArray<uint32_t> dev_counts` consumed downstream by the pulsed-weight-updater kernels (pwu_kernel.h reads it back word-by-word with `>>5`/`&0x1f`/`__popc`). The producer and consumer must agree on a 32-bit-per-word layout regardless of wavefront width.
   - On wave64, `__ballot_sync(0xFFFFFFFF, pred)` returns a 64-bit value; the low 32 bits are lanes 0-31, high 32 bits lanes 32-63. With `laneId = threadIdx.x & 0x1f` there are TWO logical 32-lane groups per wavefront, each with a `laneId==0` leader -> the wave64 leader/broadcast/ballot logic must be done PER 32-lane group (the popsift two-rows-per-wavefront lesson), not over the whole 64-lane wavefront. The serialized word stays 32 bits (do NOT widen to 64; pwu_kernel consumes uint32 words), so a wave32 device and a wave64 device must produce the SAME 32-bit-word stream.
   - Concretely: keep the logical-warp width = 32 for the ballot (mask the correct 32-lane group, capture only that group's 32 bits, the group's lane-0 is the leader). The launch geometry `numwarpsperblock = RPU_THREADS_PER_BLOCK_UPDATE / 32` (bit_line_maker.cu:641) ALREADY assumes 32-lane logical warps; on wave64 two such logical warps share a wavefront. Make every ballot/shfl explicitly width-32 (`__ballot`/`__shfl` over the 32-lane subgroup) and lane = `threadIdx.x & 0x1f` so the format is wave-width-independent. Verify on BOTH wave64 (gfx90a) and wave32 (gfx1100) -- the dev_counts stream must be byte-identical across widths for the same RNG-disabled input, or the pulsed update will read a garbage pulse train (the bug fingerprint: pulsed-update tests pass on CPU/CUDA, fail or non-deterministic on AMD).
   - The `0xFFFFFFFF` literal masks must NOT be widened to 64-bit (that would make the format 64-coupled); they must select the active 32-lane group. The `warpSize` references (bit_line_maker.cu:575,1166) in launch-geometry comments need checking against the actual `numwarpsperblock` math.
2. hipBLAS v2 enum/handle differences (pointer-mode, op enums, device-pointer-mode handle path). Mechanical but verify the device-handle scalar path.
3. CUB block-collective TempStorage reuse race on wave64 (add __syncthreads between reused-TempStorage block collectives). Audit update_management_helper.cu BlockScan and noise_manager/maximizer reduces.
4. IPO/LTO + HIP -> broken pybind PyInit (gpuRIR). Disable IPO for the HIP build.
5. host memcpy/memset resolving to HIP __device__ overloads inside .cu (include cstring/cstdlib before hip_runtime in compat header).
6. RNG non-bit-matching: hipRAND XORWOW != cuRAND XORWOW bitstream. Confirm the test suite uses tolerance/statistical asserts (CPU-vs-CUDA-tile), not fixed-seed bit-equality. If any test pins a CUDA RNG bitstream, that test is not a valid cross-arch gate -- note it.
7. atomicMaxFP CAS-based float-max helper: verify HIP semantics (NaN handling) match.
8. f32 math exactness (CV-CUDA __fsqrt_rn / ffp-contract lessons): aihwkit asserts are tolerance-based (analog noise sim), so bit-exactness is unlikely to gate, but if any tile test tightens tolerance, pin -ffp-contract=on for HIP.

## File-by-file change sketch
- src/rpucuda/cuda/cuda_to_hip.h (NEW): the only HIP-aware file; cuda*->hip* aliases for the runtime, cuBLAS, cuRAND (host+device), CUB namespace; include order cstring/cstdlib before hip_runtime.
- src/rpucuda/cuda/cuda_util.h: include cuda_to_hip.h at top (before the cublas_v2.h/cuda.h/curand*.h block at lines 18-23), guarded so CUDA build is unchanged.
- src/rpucuda/cuda/rpu_cub.h: under USE_HIP include `<hipcub/hipcub.hpp>` and use hipCUB namespace (the CUB_NS_PREFIX wrapping needs a hipCUB spelling).
- src/rpucuda/cuda/bit_line_maker.cu: make the ballot/shfl/leader logic width-32-logical-warp explicit and wave-width-independent (PRIMARY fix; risk 1). Possibly the only .cu needing real logic edits.
- src/rpucuda/cuda/cuda_math_util.cu: hipBLAS enum/handle spelling fixes (pointer mode, op enums); confirm via compat aliases where possible.
- src/rpucuda/cuda/update_management_helper.cu, noise_manager.cu, maximizer.cu, weight_clipper_cuda.cu: add __syncthreads around reused block-collective TempStorage if needed (risk 3).
- src/rpucuda/cuda/io_manager.cu: cudaHostAlloc -> hip via compat alias.
- CMakeLists.txt: add `option(USE_HIP ...)`; HIP branch mirroring the USE_CUDA branch (enable_language(HIP), LANGUAGE HIP on RPU_GPU_SRCS, HIP_ARCHITECTURES from CMAKE_HIP_ARCHITECTURES, link hipblas hiprand, keep RPU_USE_CUDA defined, disable IPO). Apply the same to the BUILD_TEST and AIHWKIT_EXTENSION_OPS_GPU branches.
- cmake/dependencies_cuda.cmake: add a USE_HIP path (or a sibling dependencies_hip.cmake) -- enable_language(HIP), find hipBLAS/hipRAND/hipCUB, add_compile_definitions(RPU_USE_CUDA) so device code stays gated on.
- src/aihwkit/simulator/CMakeLists.txt + src/aihwkit/extension/CMakeLists.txt: link RPU_GPU/hip libs under USE_HIP; ensure pybind module IPO off.
- setup.py: scikit-build passes -DUSE_HIP=ON via env/cmake args (USE_CUDA already reads $ENV{USE_CUDA}); add a parallel USE_HIP env hook.

## Build commands (gfx90a)
Prereqs present on host: ROCm 7.2 (/opt/rocm), hipcc, hipBLAS, hipRAND, hipCUB, gfx90a GPU. Need a ROCm PyTorch in the env (torch.version.hip set) for torch_cuda/c10_cuda linkage and the pinned scikit-build torch include path.

Direct CMake (bring-up / C++ gtests):
```
cd projects/aihwkit/src
cmake -S . -B build_hip -GNinja \
  -DUSE_HIP=ON -DUSE_CUDA=OFF \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_PREFIX_PATH="$(python -c 'import torch,os;print(os.path.dirname(torch.__file__))')/share/cmake" \
  -DRPU_BLAS=OpenBLAS -DBUILD_TEST=ON -DRPU_USE_TORCH_BUFFERS=OFF
cmake --build build_hip -j16
```
Multi-arch sanity (warp-size class): add `-DCMAKE_HIP_ARCHITECTURES="gfx90a;gfx1100"` and confirm both code objects via `llvm-objdump --offloading build_hip/.../librpucuda*.so`.

Python package (the real install the pytest suite imports):
```
cd projects/aihwkit/src
USE_HIP=ON USE_CUDA=0 CMAKE_HIP_ARCHITECTURES=gfx90a \
  pip install -v -e . \
  --config-settings=cmake.args="-DUSE_HIP=ON;-DCMAKE_HIP_ARCHITECTURES=gfx90a;-DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++"
```
(Exact scikit-build flag plumbing -- skbuild vs setuptools_scm config-settings spelling -- is an open question for the porter; the Makefile `make build_inplace_cuda` target is the CUDA reference to mirror.)

## Test plan
The pytest suite is the validation gate, and it is an IDEAL cross-arch correctness gate: tile tests are parametrized over CPU and CUDA tiles (tests/helpers/tiles.py: `use_cuda` flag + `CudaAnalogTile` via `getattr(tiles, "CudaAnalogTile", None)`; `.cuda()` conversion in simulator/tiles/base.py), so the SAME analog-tile forward/backward/update test runs on the C++/CPU tile and the HIP/CUDA tile and compares against the same reference. With `RPU_USE_CUDA` kept defined on the HIP build, `CudaAnalogTile` is exposed and `.cuda()` lands on the gfx90a device.

GPU-gating tests (run on real gfx90a; these exercise RPUCuda tiles on device vs CPU/C++):
- tests/test_simulator_tiles.py -- core analog tile forward/backward/update, CPU vs CUDA.
- tests/test_specific_tiles.py -- per-device-model tiles (constant/linear/exp/pow-step, softbounds, transfer, mixedprec) on CPU vs CUDA; exercises bit_line_maker + pwu update path (the warp-size risk).
- tests/test_bindings_tiles.py -- direct C++-binding tile tests, CPU vs CUDA.
- tests/test_torch_tiles.py, tests/test_inference_tiles.py, tests/test_quantized_tile.py -- torch-backed and inference tiles on device.
- tests/test_layers*.py (linear/convolution/mapped/rnn) -- analog layers on CUDA tiles.
- tests/test_extension.py -- only if BUILD_EXTENSION=ON (defer; extension is optional).

Commands:
```
cd projects/aihwkit/src
pip install -r requirements-dev.txt   # pytest + dev tools
# GPU gate (device tile correctness, the warp-size-sensitive update path):
pytest -v -s tests/test_simulator_tiles.py tests/test_specific_tiles.py tests/test_bindings_tiles.py
# Broader analog-layer GPU coverage:
pytest -v -s tests/test_torch_tiles.py tests/test_inference_tiles.py tests/test_layers_linear.py tests/test_layers_convolution.py
# Full suite (also guards non-GPU regressions):
make pytest
```
Non-GPU regression set (must not regress): the same files run their CPU-tile parametrizations, plus pure-Python tests (test_presets, test_rpu_configurations, test_conversions, test_quant_conversion, test_utils, test_export, test_optimizers). The cloud/experiment runner tests (test_cloud_runner, test_client, test_experiment_runners, test_localrunner_infer) may need network/credentials -- treat failures there as environment, not port regressions; baseline them on the unmodified CPU build first.

Dataset needs: none for the tile/layer GPU gate (tests synthesize small tensors). Some example/notebook flows pull datasets, but they are not in the pytest gate; host egress is slow, so avoid dataset-dependent tests.

Cross-arch gate for followers (gfx1100 wave32): re-run test_specific_tiles.py and confirm the pulsed-update results match gfx90a for the same input (the warp-size data-format risk is exactly what diverges between wave64 and wave32 if the BLM fix is wrong). Where RNG is involved, compare tolerance-based tile outputs, not RNG bitstreams.

## Open questions
1. scikit-build config-settings spelling to pass -DUSE_HIP / arch through `pip install -e .` (skbuild vs scikit-build-core; the Makefile CUDA target is the reference). Bring-up can use direct CMake first.
2. Does hipBLAS support the device-pointer scalar mode the code uses (getBlasDeviceHandle, cublasHandle_t* device_handle_)? Verify the device-pointer-mode GEMV/scal path.
3. RPU_USE_TORCH_BUFFERS=ON requires torch CUDA buffers (RPU_TORCH_CUDA_BUFFERS, --expt-relaxed-constexpr). On a ROCm torch the buffers are HIP tensors; confirm the torch buffer path compiles under HIP (start with RPU_USE_TORCH_BUFFERS=OFF for bring-up, then turn on). Note BUILD_TEST forces it OFF anyway.
4. Does any tile test assert against a FIXED cuRAND bitstream (would be a false cross-arch gate)? Inspect the tolerance of the CPU-vs-CUDA comparisons before trusting them as the warp-size gate.
5. FP16/bfloat16 path (RPU_USE_FP16): deferred (off by default). hip_fp16/hip_bfloat16 port is a follow-up if a tile test needs half precision.
6. Whether scikit-build/CMake enables IPO that breaks the HIP pybind module (gpuRIR) -- confirm and force IPO off for HIP.
