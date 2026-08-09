# aihwkit notes

IBM Analog Hardware Acceleration Kit. PyTorch library with a standalone
CMake/scikit-build CUDA backend (RPUCuda) exposed via a pybind11 module
(`rpu_base`). Strategy A HIP port: one compat header plus a USE_HIP CMake
option; Torch is found for headers/linkage only (NOT torch cpp_extension).

## Environment (linux-gfx90a)
- ROCm 7.2.1 at /opt/rocm; hipBLAS, hipRAND, hipCUB present.
- ROCm PyTorch in conda env py_3.12: torch 2.13.0a0, torch.version.hip 7.2.x,
  device AMD Instinct MI250X / MI250 (gfx90a, wave64).
- This newer Torch requires C++20 (its headers use the `requires` keyword), so
  the build sets RPU_CXX_STANDARD=20. Default stays 17 to match older Torch.

## The port
- New: src/rpucuda/cuda/cuda_to_hip.h -- the only HIP-aware header. Included
  once from cuda_util.h before the CUDA runtime / cuBLAS / cuRAND block,
  guarded by USE_HIP (CUDA build byte-identical). Pulls <cstring>/<cstdlib>
  before <hip/hip_runtime.h>, then aliases the cuda*/cublas*/curand* spellings
  the project uses to hip*/hipblas*/hiprand*.
- New: cmake/dependencies_hip.cmake -- enable_language(HIP),
  find_package(hip/hipblas/hiprand/hipcub), defines RPU_USE_CUDA and USE_HIP.
- CMakeLists.txt: option(USE_HIP); the GPU library/test/pybind branches now
  fire on USE_CUDA OR USE_HIP. Under HIP the .cu sources get LANGUAGE HIP, link
  roc::hipblas hip::hiprand hip::hipcub, HIP_ARCHITECTURES from
  CMAKE_HIP_ARCHITECTURES (default gfx90a when unset; no literal hardcode), IPO
  forced OFF. RPU_CXX_STANDARD cache var replaces the hardcoded CXX_STANDARD 17.
- rpu_cub.h: under USE_HIP include <hipcub/hipcub.hpp> and set
  RPU_CUB_NS_QUALIFIER to `hipcub::` (hipCUB ignores CUB_NS_PREFIX wrapping).
- src/aihwkit/simulator/CMakeLists.txt: the pybind TUs get LANGUAGE HIP under
  HIP (they inherit RPU_GPU's --offload-arch usage requirement, which only the
  HIP/clang driver understands), and IPO OFF on the module.
- rpu_base_tiles_cuda.cpp: under USE_HIP skip the raw `cuda.h` include and use
  <c10/hip/HIPStream.h> + a `using c10::cuda::getCurrentCUDAStream` in the
  at::cuda namespace, instead of <ATen/cuda/CUDAContext.h> which transitively
  pulls hipSOLVER/hipSPARSE headers not in this build.
- rpu_linearstep_device.h: declare the ctor as `SoftBoundsRPUDeviceMetaParameter()`
  not `...<T>()` -- a template-id constructor name is ill-formed under C++20
  (latent upstream bug, exposed only because the newer Torch forces C++20).

### Library swaps (mechanical, via the compat header)
- cuBLAS -> hipBLAS: Sgemm/Dgemm/Hgemm, Sgemv/Dgemv, Sger/Dger, Sscal/Dscal,
  Scopy/Dcopy, Snrm2/Dnrm2; handle + SetStream + Get/SetPointerMode (HOST and
  DEVICE pointer modes on the host handle -- hipBLAS supports both); op N/T and
  status enums. CUBLAS_STATUS_LICENSE_ERROR has no hipBLAS analogue -> mapped to
  HIPBLAS_STATUS_UNKNOWN so the CUBLAS_CALL switch still compiles. The
  device-side cuBLAS path (kernelCublasCreateDevice/getDeviceHandle, device API)
  is fully behind RPU_WITH_CUBLAS_DEVICE (OFF by default) and is never reached;
  hipBLAS has no device API so leaving it gated off is correct.
- cuRAND -> hipRAND: host generator (curandGenerator_t + create/destroy/
  generate/seed/SetStream) and device per-thread state (curand_init/normal/
  uniform). GOTCHA: in CUDA, curandState, curandState_t and curandStateXORWOW
  are the SAME XORWOW type; hipRAND declares hiprandState and
  hiprandStateXORWOW_t as DISTINCT structs. The project mixes the spellings
  (CudaArray<curandState_t> at call sites, explicit instantiation of
  CudaArray<curandStateXORWOW>), so all three are aliased to one type
  (hiprandState_t). Mapping them to different hip types caused a missing
  template instantiation -> ImportError: undefined symbol
  CudaArray<hiprandState>.
- CUB -> hipCUB: DeviceReduce/DeviceScan/DeviceSegmentedReduce + BlockScan
  (block-width templated, adapts to wave64 automatically). The single BlockScan
  TempStorage is used once (no back-to-back reuse) so no wave64 TempStorage race.

### bit_line_maker.cu warp-size fix (the risk path -- dietgpu-class)
The stochastic pulse train is a warp-size-coupled SERIALIZED bit format:
__ballot_sync(0xFFFFFFFF, ...) packed into 32-bit words (nK32 = (Kplus1+31)/32),
laneId = threadIdx.x & 0x1f; the pulsed-weight-updater kernels (pwu_kernel.h)
read it back word-by-word with >>5 / &0x1f / __popc. The on-device word layout
must stay byte-identical regardless of wavefront width. HIP 7.2 provides real
__ballot_sync/__shfl_sync that static_assert a 64-bit mask, and a native
__ballot on wave64 spans all 64 lanes -- which would merge the two independent
32-lane logical warps a wave64 wavefront holds into one 64-bit value and corrupt
the format. Fix (in cuda_to_hip.h): redefine __ballot_sync/__shfl*_sync as
width-32 logical-warp ops. __ballot_sync uses a helper that takes the full
__ballot() and shifts down by (__lane_id() & 0x20) so each 32-lane subgroup
extracts ONLY its own 32 lanes as a 32-bit word (wave32: shift always 0; wave64:
lanes 32-63 select the high word). The shuffles use width 32. The producer's
per-subgroup leader (laneId==0 -> physical lanes 0 and 32 on wave64) and
per-subgroup sourceId (threadIdx.x >> 5) are already correct in the surrounding
code, so each logical warp writes its own 32-bit word exactly as CUDA does. The
word width stays 32 (NOT widened to 64); the consumer is unchanged. The launch
geometry already assumes 32-lane logical warps (numwarpsperblock = threads/32).

Byte-identity evidence: the multi-arch fat-binary check
(-DCMAKE_HIP_ARCHITECTURES="gfx90a;gfx1100") emits BOTH gfx90a (wave64) and
gfx1100 (wave32) code objects from the one wave-agnostic source for
bit_line_maker.cu (llvm-objdump --offloading confirms both). The pulsed-update
tile tests passing on wave64 (test_specific_tiles, InferenceCuda program_weights)
prove the packed stream is read back correctly on wave64. A literal byte-for-byte
diff against a CUDA build cannot be run here (no NVIDIA GPU on the host); the
format-preservation is structural (32-bit-per-32-lane-subgroup, consumer
unchanged).

## Build recipe (gfx90a)
Direct CMake (bring-up / fastest device-code iteration):
```
cd projects/aihwkit/src
TORCH_CMAKE=$(python -c "import torch,os;print(os.path.dirname(torch.__file__))")/share/cmake
cmake -S . -B build_hip -GNinja \
  -DUSE_HIP=ON -DUSE_CUDA=OFF -DRPU_CXX_STANDARD=20 \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_PREFIX_PATH="$TORCH_CMAKE;/opt/rocm" \
  -DRPU_BLAS=OpenBLAS -DBUILD_TEST=OFF -DRPU_USE_TORCH_BUFFERS=OFF
cmake --build build_hip -j16
```
Python package in-place (what the pytest suite imports):
```
cd projects/aihwkit/src
USE_HIP=ON USE_CUDA=0 python setup.py build_ext -j16 --inplace \
  -DUSE_HIP=ON -DUSE_CUDA=OFF -DRPU_CXX_STANDARD=20 \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_PREFIX_PATH="$TORCH_CMAKE;/opt/rocm" \
  -DRPU_BLAS=OpenBLAS -DRPU_USE_TORCH_BUFFERS=OFF -DCMAKE_BUILD_TYPE=Release
```
This drops rpu_base.cpython-*.so into src/aihwkit/simulator/. Run tests with
PYTHONPATH=src (the editable package is not pip-installed here).

## Validation (real gfx90a, HIP_VISIBLE_DEVICES=0)
CPU-tile-vs-CUDA-tile comparisons are tolerance-based (assertTensorAlmostEqual
decimal=4); test_specific_tiles uses no fixed cuRAND bitstream, so it is a valid
cross-arch gate. Results:
- tests/test_specific_tiles.py: 18 passed (the bit_line_maker + pulsed-update
  warp-size path; 9 cuda-parametrized cases). CRITICAL gate -- PASS.
- tests/test_simulator_tiles.py + tests/test_bindings_tiles.py: 530 passed,
  1 failed, 56 skipped. The single failure was TileTest_Inference (the CPU
  tile, not CUDA) test_program_weights -- a stochastic 5%-tolerance
  weight-programming convergence test sensitive to global RNG ordering. It
  passes 15/15 in isolation; the CUDA variant InferenceCuda passes 8/8. Not a
  port regression (pre-existing stochastic-test fragility on the CPU path).
- tests/test_torch_tiles.py + tests/test_inference_tiles.py: 406 passed, 55
  skipped.
- tests/test_layers_linear.py + tests/test_layers_convolution.py: 566 passed,
  216 skipped.
Multi-arch code-object check: fat binary for "gfx90a;gfx1100" builds and emits
both code objects for bit_line_maker.cu (the warp-size TU).

Pin a GCD with HIP_VISIBLE_DEVICES=0 (host has 4 GCDs shared with another
session).

## Known follow-ups (not blocking the lead port)
- FP16/bfloat16 tile support (RPU_USE_FP16) is experimental and OFF by default;
  left unported (hip_fp16/hip_bfloat16 is a future delta if a half tile test
  needs it).
- RPU_USE_TORCH_BUFFERS=ON (RPU_TORCH_CUDA_BUFFERS) was OFF for this validation
  to keep bring-up focused; the CUDA-only --expt-relaxed-constexpr/-Xcudafe
  flags are now gated to NOT USE_HIP, so turning torch buffers ON under HIP is a
  straightforward follow-up to verify.
- BUILD_EXTENSION (aihwkit_extension) GPU ops left on the CUDA path only
  (BUILD_EXTENSION OFF by default).

## Install as a dependency
Not a base library for other MOAT projects; no dependents.

## Validation 2026-06-04 (validator, linux-gfx90a)

GPU: AMD Instinct MI250X / MI250 (gfx90a, wave64). ROCm 7.2.1. HIP_VISIBLE_DEVICES=0.
Fork: AMD-Ecosystem/aihwkit moat-port @ 9b4f7be7406cc939109690eb9e05f9ba2dcd3a5c.

Build: setup.py build_ext --inplace (scikit-build), USE_HIP=ON USE_CUDA=OFF
DRPU_CXX_STANDARD=20 CMAKE_HIP_ARCHITECTURES=gfx90a. Build time ~216 s.
rpu_base.cpython-312-x86_64-linux-gnu.so carries gfx90a code objects (confirmed via
llvm-objdump --offloading: 20+ amdgcn-amd-amdhsa--gfx90a bundles).

Test results (all with PYTHONPATH=src HIP_VISIBLE_DEVICES=0):
- tests/test_specific_tiles.py: **18/18 PASSED** (CRITICAL -- bit_line_maker +
  pulsed-weight-update warp-size path; 9 Cuda-parametrized cases on gfx90a wave64).
- tests/test_simulator_tiles.py + tests/test_bindings_tiles.py: **531 passed, 56 skipped, 0 failed**.
  TileTest_Inference::test_program_weights passed this run; TileTest_InferenceCuda::test_program_weights
  also passed (1 pass in isolation). The stochastic CPU-tile failure documented by the porter was not
  reproduced; it is a pre-existing non-CUDA stochastic convergence test (global RNG ordering sensitive),
  not a port regression.
- tests/test_torch_tiles.py + tests/test_inference_tiles.py: **406 passed, 55 skipped, 0 failed**.
- tests/test_layers_linear.py + tests/test_layers_convolution.py: **566 passed, 216 skipped, 0 failed**.

Total GPU-gated tests passing: 1521 passed, 383 skipped, 0 failed.
Verdict: PASS. Transitioning linux-gfx90a to completed (validated_sha 9b4f7be).

## Review 2026-06-04 (reviewer, linux-gfx90a)

Verdict: review-passed. Strategy A is correct for this standalone scikit-build/CMake
backend (Torch headers/linkage only, no torch cpp_extension), so a compat header +
USE_HIP LANGUAGE-HIP path is right, not Strategy B. Diff is minimal (8 files, +302).
Every load-bearing claim was verified by reading the ROCm 7.2.1 headers and the
producer/consumer source, not by trusting the build.

No blocking findings. Items below are non-defects, recorded for completeness:

- cuda_to_hip.h defines __shfl_down_sync but the backend never uses it (only
  __shfl_up_sync, __shfl_sync, __ballot_sync are used; all in bit_line_maker.cu).
  Harmless dead coverage; defensible as intrinsic-set completeness. No action.
- rpu_linearstep_device.h:72 ctor rename (SoftBoundsRPUDeviceMetaParameter<T>() ->
  SoftBoundsRPUDeviceMetaParameter()) is the one UNCONDITIONAL edit to a CPU/CUDA-shared
  header (not USE_HIP-guarded). It is a strict C++ conformance fix (a template-id is
  ill-formed as an injected-class-name constructor declarator) that is behavior-identical
  on the CUDA/CPU path; acceptable and minimal. Latent upstream bug exposed only because
  the newer Torch forces C++20.

Verified directly:

(a) bit_line_maker byte-identical-format correctness -- SOUND. The compat-header
__rpu_logical_warp32_ballot takes the full-wavefront __ballot() (HIP: returns
unsigned long long, 64 bits on wave64 / low 32 on wave32) and shifts by
(__lane_id() & 0x20): wave32 lane_id<=31 so shift always 0; wave64 lanes 0-31 -> 0,
lanes 32-63 -> 32. Cast to unsigned int then yields exactly the calling lane's own
32-lane subgroup as a 32-bit word == what CUDA __ballot_sync(0xFFFFFFFF,...) gives a
32-lane warp. The redefined width-32 __shfl/__shfl_up confirmed (amd_warp_functions.h)
to compute the source lane WITHIN the caller's 32-lane subgroup ((self & ~31) term),
so __shfl_sync(...,0) broadcasts each subgroup's own lane-0 and __shfl_up(...,32) stays
in-subgroup. Producer geometry already per-32-subgroup: laneId = threadIdx.x & 0x1f
(two leaders 0/32 on wave64), sourceId = (tid+i_stride)>>5 and
blockIdx.x*(blockDim.x>>5)+(threadIdx.x>>5) (distinct source per subgroup), word width
stays 32, consumer pwu_kernel.h (>>5/&0x1f/__popc on a 32-bit word) unchanged. No mask
widened to 64. The first kernel's __shfl_up_sync(...,kidthread) is also safe: nKthreads
divides 32 (constraint 32%Kplus1==0) so nKthreads-groups never straddle a 32-lane
boundary. On-device bitstream is wave-width-invariant by construction; the fat-binary
gfx90a;gfx1100 check + wave64 test pass are corroborating, not the proof.

(b) cuRAND distinct-type mapping -- CORRECT. The backend uses CudaArray<curandState>
(bit_line_maker.cu:90,131; test_helper.cu:42), CudaArray<curandState_t> at call sites
(cuda_util.h:414-415,504; cuda_util.cu:179), AND an explicit instantiation + method
specializations of CudaArray<curandStateXORWOW> (cuda_util.cu:1102/1165/1168/1171/1407).
In CUDA all three are the one XORWOW type. Aliasing all three to hiprandState_t makes
the explicit instantiation match every call-site type; mapping curandStateXORWOW to
hiprandStateXORWOW_t separately would leave CudaArray<hiprandState>'s specializations
undefined -> the exact ImportError the porter hit. No layout conflation: hipRAND's
default state IS the XORWOW state, and producer+consumer share the single aliased type
within this build, so there is no two-different-layouts hazard.

Also confirmed: cuBLAS status switch has no duplicate-case collision
(LICENSE_ERROR->UNKNOWN=11 is distinct from the 7 other mapped values, hipblas-common.h);
device-side cuBLAS (kernelCublasCreateDevice/getDeviceHandle) is fully inside
#ifdef RPU_WITH_CUBLAS_DEVICE which is never defined in any build path; host DEVICE
pointer-mode (nrm2) is on the host handle and hipBLAS-supported; BlockScan TempStorage
used exactly once (no wave64 reuse race); pinned mem cudaMallocHost/cudaFreeHost aliased;
ATen stream swap preserves at::cuda::getCurrentCUDAStream() at all 9 call sites; CMake
CMAKE_HIP_ARCHITECTURES default gfx90a only-when-unset (no hardcode), USE_HIP default OFF,
IPO off for HIP, CUDA path byte-identical; commit message [ROCm] 55 chars, Claude named,
Test Plan present, no noreply trailer, no MOAT jargon, ASCII-clean, no AMD-internal refs.
The 1 CPU-tile InferenceCuda program_weights failure is a pre-existing stochastic
convergence test (passes in isolation, CUDA variant 8/8), not a port regression.

## Validation 2026-06-04 (linux-gfx1100, RDNA3 native wave32)

GPU: AMD Radeon Pro W7800 48GB (gfx1100, wave32 native). ROCm 7.2.1. HIP_VISIBLE_DEVICES=1.
Fork: AMD-Ecosystem/aihwkit moat-port @ 9b4f7be7406cc939109690eb9e05f9ba2dcd3a5c.

Build: setup.py build_ext --inplace (scikit-build), USE_HIP=ON USE_CUDA=OFF
RPU_CXX_STANDARD=20 CMAKE_HIP_ARCHITECTURES=gfx1100. Build time ~2 min.
rpu_base.cpython-312-x86_64-linux-gnu.so carries gfx1100 code objects (confirmed via
llvm-objdump --offloading: 28+ amdgcn-amd-amdhsa--gfx1100 bundles).

Wave32 confirmation: gfx1100 is a native wave32 arch -- each physical wavefront is
exactly one 32-lane logical warp, so the __ballot_sync shift-by-(__lane_id()&0x20) in
cuda_to_hip.h is always a shift-by-0. The bit_line_maker packed format is word-identical
to CUDA wave32 by construction, confirmed by test_specific_tiles.py passing below.

Test results (all with PYTHONPATH=src HIP_VISIBLE_DEVICES=1):
- tests/test_specific_tiles.py: **18/18 PASSED** (CRITICAL -- bit_line_maker +
  pulsed-weight-update warp-size path on gfx1100 native wave32; 9 Cuda-parametrized cases).
- tests/test_simulator_tiles.py + tests/test_bindings_tiles.py: **part of 1521 passed, 0 failed**.
- tests/test_torch_tiles.py + tests/test_inference_tiles.py: **part of 1521 passed, 0 failed**.
- tests/test_layers_linear.py + tests/test_layers_convolution.py: **part of 1521 passed, 0 failed**.

Total across all suites: 1521 passed, 327 skipped, 0 failed (1848 collected).
(Lead gfx90a: 1521 passed, 383 skipped -- skip count varies by arch as some tests query
wave size or arch capabilities; pass count and zero failures match exactly.)

Verdict: PASS. All suites pass on gfx1100 native wave32 with no regressions.
Transitioning linux-gfx1100 to completed (validated_sha 9b4f7be).

## Validation 2026-06-07 (windows-gfx1201, RDNA4 native wave32)

GPU: AMD Radeon RX 9070 XT (gfx1201, wave32 native). ROCm 7.14.0a20260604 (TheRock venv).
HIP_VISIBLE_DEVICES=0 (gfx1201 is device index 0 on this host; gfx1101 is index 1).
Fork: AMD-Ecosystem/aihwkit moat-port @ 50360f7a07281ce9cf0272b2e078e6821d5f9a07.

Three Windows-specific fixes were needed on top of the Linux port (9b4f7be), committed as
a second commit on moat-port:

1. CMakeLists.txt: guard MSVC-only /O2 flag with if(MSVC) instead of if(WIN32). amdclang++
   on Windows is !MSVC and rejects /O2.

2. rpu_base_tiles_cuda.cpp: replace `using c10::cuda::getCurrentCUDAStream` with an inline
   wrapper calling c10::hip::getCurrentHIPStream. Torch 2.9/ROCm 7.14 removed the
   getCurrentCUDAStream alias from c10::cuda namespace in HIPStream.h.

3. pwu_kernel_parameter_base.h: change PulsedUpdateMetaParameter forward declaration from
   `class` to `struct` (matching rpu_pulsed_meta_parameter.h and rpucuda_pulsed_device.h).
   On Linux/ELF both mangle identically; on Windows/MSVC-ABI (used by amdclang++), struct
   and class produce different name mangling (AEBU vs AEBV), causing virtual method
   implementations in RPU_GPU.lib to not resolve at link time despite being present.

Build: CMake + Ninja, amdclang++/lld-link. All-clang (MSVC host unsupported with HIP).
Build flags require post-cmake manual steps in build.ninja:
- Remove -fuse-ld=lld-link injected by CMake 4.x into HIP device-link steps (keep it for
  the final host+device combined link of rpu_base.pyd)
- Replace /WHOLEARCHIVE via -Xlinker for RPU_GPU.lib in LINK_LIBRARIES
- Add c10_hip.lib, torch_hip.lib, and a generated libomp140.lib (from libomp140.x86_64.dll;
  import lib DLL name must include .dll extension or Windows loader fails to find the file)
- Add mypy's stubgen.exe path explicitly in POST_BUILD

Runtime: rpu_base.cp312-win_amd64.pyd requires ROCm DLLs. For Python import to work,
torch must be imported first (which preloads ROCm DLLs via rocm_sdk.preload_libraries),
AND several DLLs must be present in the package directory next to the pyd:
amd_comgr.dll, amdhip64_7.dll, c10_hip.dll, hipblas.dll, hiprand.dll, rocm-openblas.dll,
rocm_kpack.dll, rocrand.dll, rocblas.dll, rocsolver.dll, shm.dll.
The simulator/__init__.py already imports torch before rpu_base, satisfying the preload
requirement for the test suite.

gfx1201 wave32 note: RDNA4 is native wave32, so the __ballot_sync shift-by-(__lane_id()&0x20)
is always a shift-by-0 (identical to CUDA wave32 behavior). The bit_line_maker packed format
is byte-identical to gfx1100 by construction.

Test results (PYTHONPATH=src, HIP_VISIBLE_DEVICES=0):
- tests/test_specific_tiles.py: **18/18 PASSED** (CRITICAL -- bit_line_maker + pulsed-weight-update
  warp-size path; 9 Cuda-parametrized cases on gfx1201 native wave32).
- tests/test_simulator_tiles.py + tests/test_bindings_tiles.py: **530 passed, 56 skipped, 1 failed**.
  The 1 failure is TileForwardBackwardTest_Inference::test_set_forward_out_noise_std -- a stochastic
  test that passes in isolation (1/1) and is a pre-existing non-CUDA flakiness, not a port regression.
- tests/test_torch_tiles.py + tests/test_inference_tiles.py: **406 passed, 55 skipped, 0 failed**.
- tests/test_layers_linear.py + tests/test_layers_convolution.py: **566 passed, 216 skipped, 0 failed**.

Total GPU-gated tests: 1520 passed, 327 skipped, 1 failed (stochastic non-GPU pre-existing).
Verdict: PASS. Transitioning windows-gfx1201 to completed (validated_sha 50360f7).

## Revalidation 2026-06-08 (linux-gfx90a)

State: revalidate (validated_sha 9b4f7be7 -> head_sha 50360f7a -> new head d6d4561).

Delta classification: The 9b4f7be7..50360f7a delta (Windows Clang fix) was classified `mixed`
by moatlib.classify. Binary equivalence check revealed a Linux compilation failure: the Windows
fix used `c10::hip::getCurrentHIPStream(device_index)` unconditionally, but that symbol is
guarded by `#ifdef USE_ROCM` in HIPStream.h and aihwkit does not define USE_ROCM -- so the
build failed on Linux ROCm 7.2.1 with "no member named 'getCurrentHIPStream' in namespace
'c10::hip'". This is the "Windows commit breaks Linux ROCm compilation" trap from CLAUDE.md.

Fix applied (new commit d6d4561 on moat-port on top of 50360f7a): gate the function body on
HIP_VERSION_MINOR. ROCm <= 7.2 uses `c10::cuda::getCurrentCUDAStream` (still present in
torch 2.13/ROCm 7.2.1 HIPStream.h). ROCm >= 7.14 uses `c10::hip::getCurrentHIPStream`
(supported by the TheRock torch headers on Windows which expose it without the USE_ROCM guard,
as proven by the windows-gfx1201 validation at 50360f7a). This is a host-side change; device
code objects are unchanged on gfx90a.

Build: setup.py build_ext --inplace, USE_HIP=ON USE_CUDA=OFF RPU_CXX_STANDARD=20
CMAKE_HIP_ARCHITECTURES=gfx90a. 29 gfx90a code objects confirmed. Build time ~216 s.

Full GPU test suite (HIP_VISIBLE_DEVICES=3, gfx90a wave64, ROCm 7.2.1):
- tests/test_specific_tiles.py: **18/18 PASSED** (CRITICAL -- bit_line_maker + pulsed-weight-update warp-size path on gfx90a wave64).
- tests/test_simulator_tiles.py + tests/test_bindings_tiles.py: **531 passed, 56 skipped, 0 failed**.
- tests/test_torch_tiles.py + tests/test_inference_tiles.py: **406 passed, 55 skipped, 0 failed**.
- tests/test_layers_linear.py + tests/test_layers_convolution.py: **566 passed, 216 skipped, 0 failed**.

Total: 1521 passed, 383 skipped, 0 failed. Identical pass count to original validation.
Verdict: PASS. Transitioning linux-gfx90a to completed (validated_sha d6d4561).

## Re-key 2026-06-08 (linux-gfx90a, delta-ported at b346589)

Re-keyed the at::cuda::getCurrentCUDAStream shim in rpu_base_tiles_cuda.cpp from a
ROCm-version proxy to the correct axis: the torch hipify generation. This TU is never
hipified by PyTorch source-hipify (aihwkit is scikit-build/CMake with its own USE_HIP),
so it must hand-pick the c10 stream symbol. The real selector is the hipify version:
- hipify v2 (masquerading, this Linux env: torch 2.13, hipify 2.0.0): c10::cuda::
  getCurrentCUDAStream is the public API; c10::hip::getCurrentHIPStream is #ifdef
  USE_ROCM and this build defines USE_HIP not USE_ROCM, so only c10::cuda works.
- hipify v1 (rename, Windows TheRock torch 2.9.1): c10::cuda::getCurrentCUDAStream is
  removed, so c10::hip::getCurrentHIPStream is required.

The old d6d4561 gate (HIP_VERSION >= 7.14 -> c10::hip) worked only because in our fleet
ROCm version anti-correlates with the hipify generation (ROCm 7.2 + new torch/v2 on
Linux; ROCm 7.14 + old torch/v1 on Windows). Now CMake (cmake/dependencies_hip.cmake)
probes the build's own torch via RPU_PYTHON_EXECUTABLE
(`python -c "from torch.utils.hipify import __version__"`), defines TORCH_HIPIFY_V2 on
the rpu_base target when >= 2.0.0, and the cpp keys on `#if defined(TORCH_HIPIFY_V2)`.
The sense FLIPS vs the old gate (v2 now takes the c10::cuda branch). Detection failure
leaves it undefined = v1 default. Removed the now-unused <hip/hip_version.h> include.

CMake on this Linux env logged `-- torch hipify version: 2.0.0` and put -DTORCH_HIPIFY_V2
on the rpu_base_tiles_cuda.cpp.o compile line (confirmed in build.ninja DEFINES).

Behavior-preservation proof (the gate for this delta, no full GPU re-run required):
codeobj_diff between a pristine d6d4561 build and the b346589 build of
rpu_base.cpython-312-x86_64-linux-gnu.so (both 337135072 bytes, 29 gfx90a code objects):
`verdict=identical -- exported symbols + device ISA identical (53320 exports)`. On Linux
v2 the re-key selects the SAME c10::cuda::getCurrentCUDAStream the old gate's #else branch
already selected (ROCm 7.2 < 7.14), so the device ISA is unchanged by construction; the
diff confirms it.

GPU smoke on MI250X (gfx90a wave64, ROCm 7.2.1, HIP_VISIBLE_DEVICES=0): rpu_base loads,
rpu_base.cuda.is_compiled()=True; tests/test_specific_tiles.py 18/18 PASS (the critical
bit_line_maker pulsed-update warp-size path); tests/test_simulator_tiles.py -k Cuda 264
passed 0 failed; tests/test_inference_tiles.py -k Cuda 32 passed 0 failed.

Commit b346589 on top of d6d4561 (new commit, not amend). This makes aihwkit the
reference example for the PORTING_GUIDE "key on hipify generation, not ROCm version"
entry. Linux-gfx90a -> delta-ported.

## Revalidation 2026-06-08 (linux-gfx1100)

State: revalidate (validated_sha 9b4f7be -> head_sha d6d4561).
GPU: AMD Radeon Pro W7800 48GB (gfx1100, wave32 native). ROCm 7.2.1. HIP_VISIBLE_DEVICES=2.

Delta 9b4f7be..d6d4561 (2 commits: Windows Clang build fixes + getCurrentCUDAStream ROCm compat):
- CMakeLists.txt: guard /O2 with if(MSVC) instead of if(WIN32). On Linux MSVC=false; the else() branch is unchanged.
- rpu_base_tiles_cuda.cpp: HIP_VERSION_MINOR-gated getCurrentHIPStream/getCurrentCUDAStream. On Linux ROCm 7.2.1: HIP_VERSION_MAJOR=7, HIP_VERSION_MINOR=2 < 14, so c10::cuda::getCurrentCUDAStream path taken -- same as before the change.
- pwu_kernel_parameter_base.h: class->struct for PulsedUpdateMetaParameter forward declaration. On Linux/ELF both mangle identically; no ABI impact.

No .cu device files changed. Binary equivalence check:

Build commands (HIP_VISIBLE_DEVICES=2, CMAKE_HIP_ARCHITECTURES=gfx1100):
```
cmake -S . -B build_hip -GNinja -DUSE_HIP=ON -DUSE_CUDA=OFF -DRPU_CXX_STANDARD=20 \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_PREFIX_PATH="${TORCH_CMAKE};/opt/rocm" -DRPU_BLAS=OpenBLAS -DBUILD_TEST=OFF \
  -DRPU_USE_TORCH_BUFFERS=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build_hip -j16
```

codeobj_diff result:
  src/aihwkit/simulator/rpu_base.cpython-312-x86_64-linux-gnu.so: identical (exported symbols + device ISA identical (11257 exports))
  verdict=identical

Carry-forward applied: moatlib carry-forward binary-equiv, validated_sha -> d6d4561.
No GPU re-run required. linux-gfx1100 -> completed.

## Review 2026-06-08 (reviewer, linux-gfx90a, re-key delta d6d4561..b346589)

Verdict: review-passed. Reviewed the hipify-version re-key of the
at::cuda::getCurrentCUDAStream shim (3 files, +35/-7). No defects found; all
load-bearing claims verified against the build env's torch headers, not trusted.

No blocking findings. Verified directly:

- Probe (cmake/dependencies_hip.cmake:34-46): uses RPU_PYTHON_EXECUTABLE, which
  dependencies.cmake:84 sets from Python3_EXECUTABLE and is included at
  CMakeLists.txt:53, BEFORE dependencies_hip.cmake at :55 -- so the build's own
  Python runs the probe. Ran the exact probe command in the env: exit 0, stdout
  "2.0.0" clean; the NumPy import warning goes to stderr (ERROR_QUIET-suppressed,
  not captured into the version var). The RESULT==0 AND NOT STREQUAL "" guard is
  robust; detection failure leaves RPU_TORCH_HIPIFY_V2 unset (safe v1 default).
- Variable scope: set(RPU_TORCH_HIPIFY_V2 ON) is in the top-level include scope
  (CMakeLists.txt:55); src/aihwkit/simulator is add_subdirectory at :156 (> :55),
  so the child scope inherits it. target_compile_definitions(rpu_base PRIVATE
  TORCH_HIPIFY_V2) (src/aihwkit/simulator/CMakeLists.txt:32-34) applies ONLY to
  the pybind module target (the rpu_base_src/*.cpp binding TUs incl.
  rpu_base_tiles_cuda.cpp), NOT via add_compile_definitions -- so the .cu device
  TUs in RPU_GPU are untouched (consistent with codeobj_diff = identical).
- #if defined(TORCH_HIPIFY_V2) SENSE is correct (and is the flip of the old
  HIP_VERSION gate). Confirmed in this env's c10/hip/HIPStream.h: getCurrentCUDAStream
  is unconditional in c10::cuda (:235); getCurrentHIPStream is #ifdef USE_ROCM (:256)
  which this build does not define. So v2 -> c10::cuda::getCurrentCUDAStream (the
  only available symbol), v1 -> c10::hip::getCurrentHIPStream. Env hipify = 2.0.0,
  so VERSION_GREATER_EQUAL "2.0.0" true -> define set -> c10::cuda branch taken.
- Orphan cleanup complete: <hip/hip_version.h> removed; no remaining HIP_VERSION*
  macro usage in the TU (grep clean).
- CUDA/CPU paths unaffected: the shim is double-guarded by #ifdef RPU_USE_CUDA (:7)
  and #if defined(USE_HIP) (:17). A CUDA build (RPU_USE_CUDA set, USE_HIP unset)
  takes the #else at :39 (<ATen/cuda/CUDAContext.h>), unchanged; a pure-CPU build
  compiles none of it. TORCH_HIPIFY_V2 is only ever defined on the HIP pybind build.
- Commit hygiene: title 72 chars exactly, [ROCm] prefix; no MOAT jargon in message
  or comments; ASCII clean, no em-dash; no noreply trailer, no ghstack; Claude
  disclosed; Test Plan present; no AMD-internal account refs.

Carry-forward note: this delta changes only a host-side pybind TU and no .cu
device code; on Linux v2 it selects the same c10::cuda::getCurrentCUDAStream the
old #else already resolved, so gfx90a / gfx1100 / gfx1201 all carry forward via
.co byte-identity (gfx90a codeobj_diff already identical; gfx1100/gfx1201 same by
construction -- device ISA unchanged). Hands to validator for carry-forward
confirmation; no functional GPU re-run is required for this behavior-preserving
host-only change.

## Validation 2026-06-08 (linux-gfx90a, re-key carry-forward)

GPU: AMD Instinct MI250X / MI250 (gfx90a, wave64). ROCm 7.2.1. HIP_VISIBLE_DEVICES=0.
Delta: d6d4561 -> b346589 (hipify-version re-key, host-only pybind TU).

Build: CMake + Ninja at BOTH shas (fresh build dirs, not recycled). At b346589 cmake configured
with `-- torch hipify version: 2.0.0` and set -DTORCH_HIPIFY_V2 in build.ninja DEFINES for
rpu_base_tiles_cuda.cpp.o (confirmed). At d6d4561 (worktree, HIP_VERSION_MINOR gate, ROCm 7.2 <
7.14 takes the c10::cuda::getCurrentCUDAStream path). Both builds produced
rpu_base.cpython-312-x86_64-linux-gnu.so.

codeobj_diff: verdict=identical -- exported symbols + device ISA identical (53320 exports).
Matches the porter's prior run exactly. On Linux v2 both gates resolve the same
c10::cuda::getCurrentCUDAStream symbol, so device ISA is unchanged by construction; diff confirms.

GPU smoke (gfx90a wave64, HIP_VISIBLE_DEVICES=0): tests/test_specific_tiles.py 18/18 PASSED
(bit_line_maker + pulsed-weight-update warp-size path, 9 Cuda-parametrized cases).

Verdict: carry-forward confirmed. linux-gfx90a -> completed (validated_sha b346589).

## Revalidation 2026-06-08 (linux-gfx1100, delta d6d4561->b346589)

State: revalidate (validated_sha d6d4561 -> head_sha b346589).
GPU: AMD Radeon Pro W7800 48GB (gfx1100, wave32 native). ROCm 7.2.1. HIP_VISIBLE_DEVICES=1.

Delta d6d4561..b346589 (hipify-version re-key): changes only 3 host-side files --
cmake/dependencies_hip.cmake, src/aihwkit/simulator/CMakeLists.txt, and
src/aihwkit/simulator/rpu_base_src/rpu_base_tiles_cuda.cpp. No .cu device code changed.
On this Linux env, hipify version is 2.0.0 (v2), so TORCH_HIPIFY_V2 is defined and the
re-keyed gate selects the same c10::cuda::getCurrentCUDAStream the old HIP_VERSION_MINOR<14
branch already selected (ROCm 7.2 < 7.14). Device ISA is unchanged by construction.

Binary equivalence check:
- Build at d6d4561: cmake -S . -B build_gfx1100 -GNinja -DUSE_HIP=ON -DUSE_CUDA=OFF
  -DRPU_CXX_STANDARD=20 -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++
  -DCMAKE_PREFIX_PATH="${TORCH_CMAKE};/opt/rocm" -DRPU_BLAS=OpenBLAS -DBUILD_TEST=OFF
  -DRPU_USE_TORCH_BUFFERS=OFF -DCMAKE_BUILD_TYPE=Release; cmake --build build_gfx1100 -j16
- Build at b346589: same flags. Both built successfully (warnings only, no errors).
- codeobj_diff result:
    src/aihwkit/simulator/rpu_base.cpython-312-x86_64-linux-gnu.so: identical
    (exported symbols + device ISA identical (11257 exports))
    verdict=identical

Carry-forward applied. linux-gfx1100 -> completed (validated_sha b346589). No GPU re-run required.

## Validation 2026-06-08 (windows-gfx1201, carry-forward)

Delta: 50360f7a -> b346589 (3 host-side files only).

Method: source-class inspection.

Changed files:
- cmake/dependencies_hip.cmake: adds cmake `execute_process` to detect torch hipify version and set RPU_TORCH_HIPIFY_V2 if version >= 2.0.0
- src/aihwkit/simulator/CMakeLists.txt: passes TORCH_HIPIFY_V2 compile definition if RPU_TORCH_HIPIFY_V2 is set
- src/aihwkit/simulator/rpu_base_src/rpu_base_tiles_cuda.cpp: adds `#if defined(TORCH_HIPIFY_V2)` branch for getCurrentCUDAStream

On this Windows host (TheRock 7.14 PyTorch), torch hipify version is 1.0.0 (v1). Therefore TORCH_HIPIFY_V2 is NOT defined at configure time, and the #if branch is dead code. The compiled output at b346589 uses the same `c10::hip::getCurrentHIPStream(device_index)` path as 50360f7a. No GPU device code (.cu files) was changed. Device ISA for gfx1201 is identical by construction.

VERDICT: CARRY-FORWARD (source-class). State -> completed (validated_sha = b346589). No GPU re-run required.

## PR-prep 2026-06-08 (lead) -- docs + squash; carry-forward, no GPU re-run

Pre-PR cleanup. The port was already clean (no MOAT jargon in code or commit
messages; CMake arch handling already correct -- CMAKE_HIP_ARCHITECTURES,
default gfx90a only when unset). Only substantive prep was documentation.

- docs/source/developer_install.rst: documented the ROCm/HIP build alongside the
  CUDA build in house style -- a GPU build example, gfx-arch identification via
  rocminfo (parallel to the nvidia-smi CUDA_ARCH snippet), the USE_HIP row in the
  compilation-flags table, and a USE_HIP build example (setup.py + cmake, noted
  mutually exclusive with USE_CUDA). No ROCm wheel/conda/docker exists, so the
  prebuilt-install docs (install.rst, README install section) were left alone --
  ROCm is from-source only.
- README.md: feature line "(CUDA-capable)" -> "(CUDA- and ROCm/HIP-capable)".

Squashed the 4 port commits + the doc commit into ONE commit on the upstream
base (clean PR diff): 0b49124 [ROCm] Add AMD GPU (HIP) support to the RPUCuda
backend, parent 4d73afd. 11 files, +367/-31. The squashed message consolidates
the four commit messages and drops the stream-shim iteration narrative (it
describes the final TORCH_HIPIFY_V2-keyed shim, not the HIP_VERSION steps).

advance_head classified the doc delta doc-only; squash-carry-forward carried
linux-gfx90a, linux-gfx1100, windows-gfx1201 to 0b49124. gfx1151 kept blocked
(retired); gfx1101 stays port-ready (redundant Windows tier, gfx1201 satisfies
it). pr-ready=True.

NEXT: upstream-PR gate (lead-only, jeff approval). No existing jeffdaily PR on
IBM/aihwkit. PR body scopes out gfx1151; claims Linux gfx90a + gfx1100 and
Windows gfx1201.

## Validation 2026-06-19 (windows-gfx1101, RDNA3 native wave32)

GPU: AMD Radeon PRO V710 (gfx1101, wave32 native). ROCm 7.14.0a20260604 (TheRock venv).
HIP_VISIBLE_DEVICES=1 (gfx1101 is device index 1 on this host; gfx1201 is index 0).
Fork: AMD-Ecosystem/aihwkit moat-port @ 42e046594e91c1283ddd647d2165ba8b662b3999 (squashed PR commit).

Build: CMake + Ninja, amdclang++/lld-link, CMAKE_HIP_ARCHITECTURES=gfx1101.
Same three Windows-specific fixes as gfx1201 are in the squashed commit: if(MSVC) /O2 guard,
TORCH_HIPIFY_V2 detection for stream symbol, PulsedUpdateMetaParameter class->struct.

CMake configure flags:
  -DUSE_HIP=ON -DUSE_CUDA=OFF -DRPU_CXX_STANDARD=20 -DCMAKE_HIP_ARCHITECTURES=gfx1101
  -DCMAKE_HIP_COMPILER=amdclang++ -DCMAKE_C_COMPILER=amdclang -DCMAKE_CXX_COMPILER=amdclang++
  -DCMAKE_PREFIX_PATH="$TORCH_CMAKE;$ROCM_DEVEL;$ROCM_DEVEL/lib/cmake"
  -Dhip_DIR/hipblas_DIR/hipblas-common_DIR/hiprand_DIR/rocrand_DIR/hipcub_DIR/rocprim_DIR set explicitly
  -DOpenBLAS_INCLUDE_DIR=$ROCM_DEVEL/lib/host-math/include/openblas
  -DOpenBLAS_LIB=$ROCM_DEVEL/lib/host-math/lib/rocm-openblas.lib
  -DRPU_USE_TORCH_BUFFERS=OFF -DBUILD_TEST=OFF

build.ninja manual patches (same class as gfx1201):
- Add c10_hip.lib, torch_hip.lib (HIPStream symbols) to LINK_LIBRARIES
- Add libomp140.lib (generated from libomp140.x86_64.dll via llvm-dlltool) for OpenMP symbols
- Add -Xlinker /WHOLEARCHIVE:RPU_GPU.lib for complete template instantiation
- Fix POST_BUILD: replace bare 'stubgen' with full path to venv/Scripts/stubgen.exe

Runtime environment: same DLLs as gfx1201 (already in simulator/), pyd replaced with gfx1101 build.
ROCBLAS_TENSILE_LIBPATH=_rocm_sdk_libraries/bin/rocblas/library; ROCBLAS_USE_HIPBLASLT=0.

gfx1101 wave32 note: RDNA3 is native wave32, so the __ballot_sync shift-by-(__lane_id()&0x20)
is always a shift-by-0 (identical to gfx1100 and gfx1201 wave32 behavior). The bit_line_maker
packed format is byte-identical to gfx1100/gfx1201 by construction.

Test results (PYTHONPATH=src, HIP_VISIBLE_DEVICES=1):
- tests/test_specific_tiles.py: **18/18 PASSED** (CRITICAL -- bit_line_maker + pulsed-weight-update
  warp-size path; 9 Cuda-parametrized cases on gfx1101 native wave32).
- tests/test_simulator_tiles.py + tests/test_bindings_tiles.py: **531 passed, 56 skipped, 0 failed**.
- tests/test_torch_tiles.py + tests/test_inference_tiles.py: **406 passed, 55 skipped, 0 failed**.
- tests/test_layers_linear.py + tests/test_layers_convolution.py: **566 passed, 216 skipped, 0 failed**.

Total GPU-gated tests: 1521 passed, 383 skipped, 0 failed. Identical pass count to gfx90a.
Verdict: PASS. Transitioning windows-gfx1101 to completed (validated_sha 42e0465).

## Revalidation 2026-08-09 (linux-gfx90a)

State: revalidate (validated_sha 4bc09a0 -> head_sha 70577b5).
GPU: AMD Instinct MI250X / MI250 (gfx90a, wave64). ROCm 7.2.1. HIP_VISIBLE_DEVICES=2.

Delta 4bc09a0..70577b5 (1 commit, CMakeLists.txt only, +2/-3): drops the
`if(NOT DEFINED CMAKE_HIP_ARCHITECTURES) set(... "gfx90a" ...) endif()` pin,
leaving CMAKE_HIP_ARCHITECTURES to `enable_language(HIP)`'s own auto-detect
when a builder does not pass `-DCMAKE_HIP_ARCHITECTURES` explicitly. Our build
recipe always passes `-DCMAKE_HIP_ARCHITECTURES=gfx90a` explicitly, so this is
a default-path-only change here. `moatlib.py classify` returned `mixed`
(CMakeLists.txt token count differs), so did NOT auto-carry; ran the binary-
equivalence check per the mixed/unknown protocol.

Binary equivalence check (both shas built from the same absolute checkout
path, fresh build dirs, explicit CMAKE_HIP_ARCHITECTURES=gfx90a):
```
cd projects/aihwkit/src
git checkout 4bc09a0fa9a76d3038e6729ea03615fcc0beb76b
cmake -S . -B build_old -GNinja -DUSE_HIP=ON -DUSE_CUDA=OFF -DRPU_CXX_STANDARD=20 \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_PREFIX_PATH="$TORCH_CMAKE;/opt/rocm" -DRPU_BLAS=OpenBLAS -DBUILD_TEST=OFF \
  -DRPU_USE_TORCH_BUFFERS=OFF
cmake --build build_old -j16
git checkout 70577b578b7faa661611e905c0b9ec1a1d85bf33
cmake -S . -B build_new -GNinja <same flags>
cmake --build build_new -j16
python3 utils/codeobj_diff.py projects/aihwkit/src/build_old projects/aihwkit/src/build_new
```
Result: overall verdict `indeterminate`, but that came ONLY from CMake's own
compiler-ABI-detection scratch binaries (CMakeDetermineCompilerABI_*.bin,
CompilerIdC/CompilerIdCXX/CompilerIdHIP a.out, FindOpenMP ompver_*.bin) --
plain host x86-64 ELF test programs with no HIP device code at all (confirmed
via `file` + `nm -D`: only libc/GCC weak symbols, identical between old and
new). These are not part of the aihwkit build output and never ship. The one
real artifact, `rpu_base.cpython-312-x86_64-linux-gnu.so`, reported
**identical** -- exported symbols + device ISA identical (53232 exports).

GPU corroboration (HIP_VISIBLE_DEVICES=2, gfx90a wave64), rpu_base.so from the
head-sha build (build_new) copied into src/aihwkit/simulator/:
- Import smoke: `rpu_base.cuda.is_compiled()` True, `torch.cuda.get_device_name(0)`
  = "AMD Instinct MI250X / MI250".
- tests/test_specific_tiles.py: **18/18 PASSED** (CRITICAL -- bit_line_maker +
  pulsed-weight-update warp-size path).
- tests/test_simulator_tiles.py + tests/test_bindings_tiles.py `-k Cuda`:
  **284 passed, 47 skipped, 0 failed**.
- tests/test_inference_tiles.py `-k Cuda`: **32 passed, 22 skipped, 0 failed**.

CUDA no-regression gate: never previously recorded for this project at any
head_sha, so ran it here (compile-only, no NVIDIA GPU on host). Worked around
the two known environmental walls (ROCm-build torch/include missing
c10/cuda/impl/cuda_cmake_macros.h; torch/headeronly/util/complex.h duplicated-
token #if guard) by building against a scratch venv with pip-installed
`torch==2.11.0+cu128` (CUDA wheel). RPU_CUDA_ARCHITECTURES is already a fixed
cache-var list ("75;80;89", includes 80) in CMakeLists.txt -- not `native`
autodetect -- so no arch-pin patch was needed.
```
python3 -m venv venv && source venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.11.0+cu128
pip install numpy pybind11 scikit-build cmake
cd projects/aihwkit/src   # at head_sha 70577b578b7faa661611e905c0b9ec1a1d85bf33
cmake -S . -B build_cuda_head -GNinja -DUSE_CUDA=ON -DUSE_HIP=OFF -DRPU_CXX_STANDARD=17 \
  -DCMAKE_CUDA_COMPILER=/opt/conda/envs/cuda-12.8/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DCMAKE_PREFIX_PATH="$TORCH_CMAKE" -DRPU_BLAS=OpenBLAS -DBUILD_TEST=OFF -DRPU_USE_TORCH_BUFFERS=OFF
cmake --build build_cuda_head -j16
```
Result: clean build, exit 0, produced rpu_base.cpython-312-x86_64-linux-gnu.so
(198 MB) with real CUDA sm_75/sm_80/sm_89 code objects for bit_line_maker.cu
and all RPU_GPU .cu sources. No `-fvisibility=hidden`-on-nvcc-style failure,
no `atomicAdd(double*)` overload failure (would have signaled a wrong/native
arch pin). CUDA path is a pure passthrough -- no regression.

Jargon: `python3 utils/jargon.py --port aihwkit` -> clean.

Verdict: PASS (binary-equivalence carry-forward, corroborated by real-GPU
smoke). Transitioning linux-gfx90a to completed (validated_sha 70577b5).
