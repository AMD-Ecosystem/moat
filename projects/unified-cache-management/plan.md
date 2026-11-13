# Port plan: unified-cache-management (UCM)

## Project
- Name: unified-cache-management (UCM, "uc-manager")
- Upstream: https://github.com/ModelEngine-Group/unified-cache-management (ModelEngine-Group, Huawei)
- Default branch: develop
- What it is: a KVCache offload/sparse-attention framework for vLLM. Mostly Python; a thin C++/CUDA layer does (a) host<->device KV block transfer kernels and (b) a Hamming-distance scoring kernel for sparse retrieval.

## Disposition: CLEAN PORT (Strategy A, pure CMake) -- worth dispatching a porter
Effort class: clean-hipify with a small targeted PTX-to-portable-intrinsic rewrite. NOT a CK/ck_tile reimplementation, NOT a skip.

Rationale / evidence that this is tractable hipify-able work and NOT a CUTLASS reimplementation:
- The ENTIRE compiled GPU surface is 4 files. `find . -name '*.cu' -o -name '*.cuh'`:
  - ucm/shared/trans/cuda/cuda_sm_kernel.cu  (KV block copy kernel)
  - ucm/store/nfsstore/device/cuda/cuda_device.cu  (H2D/D2H batch copy + CudaDevice runtime wrapper)
  - ucm/sparse/gsa_on_device/csrc/cuda/ham_dist/paged_ham_dist_mla.cu  (Hamming-score kernel)
  - ucm/sparse/gsa_on_device/csrc/cuda/ham_dist/cp_async.cuh  (FlashInfer cp.async helper, header)
- No CUTLASS, no CuTe, no `mma`/`wgmma`, no warp specialization, no tensor cores anywhere. `grep -rn 'cutlass\|cute\|wgmma\|mma.sync'` -> nothing.
- The kernels are naive/pointwise: the copy kernels are grid-stride 16/32-byte vectorized memcpy; the Hamming kernel is XOR + `__popc`/`__popcll` + shared-memory popcount accumulation + `half` reduce. All have direct HIP equivalents.
- No `__shfl*`/`__ballot`/`__activemask`/`warpSize`/hardcoded-32 in the device code (`grep` clean), so the wave64/wave32 fault class largely does NOT apply to correctness (block size 256/512, no warp collectives). This is unusually low risk.
- No existing HIP/ROCm/AMD path exists (`grep -rn -i 'hip\|rocm\|gfx\|__HIP\|amdgpu' ucm` finds only unrelated hits in patches/docs), so this adds real value; not already-supported.
- A sibling vendor backend already proves the portable pattern: ucm/store/nfsstore/device/maca/maca_device.cu (MetaX, a different non-AMD vendor) implements the identical copy kernel using portable `__ldcs`/`__stcg`/`__stcs` intrinsics instead of raw PTX. HIP provides `__ldcs`/`__stcg`/`__stcs` (cache-streaming load/store builtins), so the maca file is essentially the template for the AMD path.

## Existing AMD support
None. Backends present: simu (CPU stub), ascend (Huawei NPU), musa (Moore Threads), maca (MetaX), cuda (NVIDIA). musa/maca are non-AMD CUDA-like vendors. There is NO hip/rocm backend. Decision: proceed with a new ROCm/HIP backend modeled on the cuda backend (and the portable maca intrinsic pattern).

## Build classification: pure CMake (Strategy A)
Evidence:
- Top-level build is CMake: CMakeLists.txt (`project(... LANGUAGES CXX)`, `enable_testing()`, `add_subdirectory(ucm)`).
- setup.py wraps CMake via a custom `CMakeExtension`/`CMakeBuild(build_ext)` (setup.py:108-199); it is NOT a torch `CUDAExtension`/`BuildExtension`. It selects the backend by passing `-DRUNTIME_ENVIRONMENT=<cuda|ascend|musa|maca|simu>` from the `PLATFORM` env var (setup.py:174-186).
- The CUDA language is brought up inside the per-backend CMakeLists via `enable_language(CUDA)` (ucm/shared/trans/cuda/CMakeLists.txt, ucm/store/nfsstore/device/cuda/CMakeLists.txt, ucm/sparse/gsa_on_device/csrc/cuda/ham_dist/CMakeLists.txt), with a hardcoded nvcc path and `CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90`.
- NOTE the sparse `hamming` module DOES link libtorch (it finds torch via python and links torch/c10/torch_cpu/torch_python, ham_dist/CMakeLists.txt) and includes `<torch/script.h>`. It is still built by the project's own CMake, not torch's extension machinery, so it stays Strategy A; we just build it against a ROCm torch and let our compat header + LANGUAGE HIP handle the .cu. (The copy/store kernels do NOT depend on torch.)

This is the compat-header approach: add one compat header, gate the language on a `USE_HIP`/new RUNTIME_ENVIRONMENT value, mark the existing `.cu` LANGUAGE HIP, leave the NVIDIA path untouched.

## Port strategy (Strategy A, plus a new RUNTIME_ENVIRONMENT=rocm backend)
Two viable shapes; recommend (1):
1. Add a new backend value `RUNTIME_ENVIRONMENT=rocm` (PLATFORM=rocm in setup.py) that mirrors the `cuda` dirs. Create sibling rocm/ subdirs OR reuse the cuda/ dirs guarded by the runtime value. Because the cuda sources are almost entirely portable, the lightest footprint is to REUSE the cuda/*.cu sources, marked `LANGUAGE HIP`, behind a `cuda_to_hip.h` compat header, and add `elseif(RUNTIME_ENVIRONMENT STREQUAL "rocm")` arms in the three backend-selecting CMakeLists (ucm/shared/trans/CMakeLists.txt, ucm/store/nfsstore/device/CMakeLists.txt, ucm/sparse/gsa_on_device/CMakeLists.txt) that `enable_language(HIP)` and set `HIP_ARCHITECTURES` from `${CMAKE_HIP_ARCHITECTURES}` (default gfx90a only when unset -- never a literal; see guide).
2. (Alternative) Mirror cuda/ -> rocm/ as new files (amgcl backend-mirror model). More files but zero risk to the NVIDIA path. Pick this only if the in-place LANGUAGE-HIP retag proves intrusive to the per-target CMakeLists.

Recommend (1) with a single `cuda_to_hip.h` that aliases the cuda* runtime symbols actually used (see inventory) and includes `<hip/hip_runtime.h>`; on non-HIP it is a no-op include of `<cuda_runtime.h>`.

## CUDA surface inventory (map each to ROCm/HIP)
Runtime API (cuda_device.cu): cudaMalloc/Free/Memcpy/MemcpyAsync, cudaMallocHost/cudaFreeHost, cudaSetDevice, cudaStreamCreate/StreamSynchronize/StreamAddCallback, cudaGetLastError/GetErrorString, cudaError_t/cudaSuccess/cudaStream_t, cudaMemcpyHostToDevice/DeviceToHost. -> all 1:1 hip* (hipMalloc, hipHostMalloc, hipFreeHost/hipHostFree, hipStreamAddCallback, hipStream_t, hipSuccess, ...). hipMallocHost is deprecated-aliased to hipHostMalloc; use the alias or hipHostMalloc. cudaStreamAddCallback -> hipStreamAddCallback (signature matches).
Kernels:
- cuda_sm_kernel.cu CudaCopyKernel (3 overloads): grid-stride copy, 32-byte unit via INLINE PTX `ld.global.cs.v4.b32` / `st.volatile.global.v4.b32`. PTX does NOT exist on AMD. RISK item #1 -- rewrite (see risks).
- cuda_device.cu H2DKernel/D2HKernel: 16-byte unit via INLINE PTX `ld.global.cs.v2.u64` / `st.global.cg.v2.u64` / `st.volatile.global.v2.u64`. Same PTX issue. RISK item #1.
- paged_ham_dist_mla.cu HammingScoreContiKernel / HammingScoreKernel: `__popc`, `__popcll`, `half`, `__hlt`, `__syncthreads`, `extern __shared__`, `min`, `INFINITY`. All available in HIP. The `--use_fast_math`, `-U__CUDA_NO_HALF_OPERATORS__` etc flags have hip equivalents (hipcc accepts -ffast-math; the half-operator unmacros are nvcc-only and simply unneeded on HIP). torch `half`/`at::Half`, c10::cuda::getCurrentCUDAStream -> on a ROCm torch these resolve to the HIP-backed stream automatically (torch keeps the "cuda" spelling on ROCm).
- cp_async.cuh: PTX cp.async + `__cvta_generic_to_shared`. ALREADY has a non-PTX fallback gated on `FLASHINFER_CP_ASYNC_ENABLED`, which is only defined for `__CUDACC_VER_MAJOR__>=11 && __CUDA_ARCH__>=800`. On HIP neither macro is defined, so it compiles the portable `*(uint4*)smem = *(uint4*)gmem` path. LIKELY NO CHANGE NEEDED; verify the macro guard does not accidentally trip under hipcc.
Libraries: none of cuBLAS/cuFFT/cuRAND/cuSPARSE/Thrust/CUB. The only external GPU lib is cudart (-> amdhip64 via HIP) and (for hamming) libtorch.
Textures/surfaces/managed memory: none. Pinned host memory: cudaMallocHost (-> hipHostMalloc). Streams/events: streams + a stream callback only (no events).

## Risk list
1. INLINE PTX in the two copy kernels (cuda_sm_kernel.cu, cuda_device.cu). `ld.global.cs.v4.b32`, `st.volatile.global.v4.b32`, `ld.global.cs.v2.u64`, `st.global.cg.v2.u64`. PTX is NVIDIA-only; will not compile under hipcc. FIX: replace the asm with portable vectorized loads/stores. The maca backend already shows the intended portable form: `__ldcs((const uint4*)src)` + `__stcg((uint4*)dst)` (cache-streaming) for the 32B path, and the 16B path similarly. HIP provides `__ldcs`/`__stcg`/`__stcs`. Simplest correct fallback if an intrinsic is missing on a given ROCm: plain `*(uint4*)dst = *(const uint4*)src` (the cp_async.cuh fallback already does exactly this). Guard the PTX block `#if defined(__CUDA_ARCH__)` (or the runtime-env macro) and provide the `__ldcs/__stcg` path for HIP. The `volatile` store semantics are about visibility ordering for the KV-transfer; on HIP use `__stcg`/`__stcs` (streaming, bypasses L1) to match intent. Validate by the trans gtest round-trip + a byte-exact compare.
2. Warp size: LOW risk. No warp collectives or hardcoded 32 in device code. Block dims are 256 (copy) and 512 (hamming) with `__syncthreads()` only. Still build the fat binary `gfx90a;gfx1100` at validation to confirm both code objects emit, per guide.
3. The hamming module links libtorch and includes <torch/script.h>: must build against a ROCm-enabled torch in the env. The CMake greps torch via the active python; ensure the venv has a rocm torch. Half operator unmacros (`-U__CUDA_NO_HALF_OPERATORS__`) are nvcc-only flags -- drop them (or guard) for the HIP compile or hipcc will warn/ignore.
4. `c10::cuda::getCurrentCUDAStream` / `at::Half` in paged_ham_dist_mla.cu: on ROCm torch these exist under the same names (torch retains the cuda spelling on HIP); no change expected, but it is a hipify-via-torch-headers detail to verify at compile.
5. CMake hardcodes nvcc path and `CMAKE_CUDA_ARCHITECTURES`; the rocm arm must NOT hardcode a literal gfx arch (set from `${CMAKE_HIP_ARCHITECTURES}`, default gfx90a only when unset) so followers need no source edit (guide lesson, CudaSift/Gpufit).
6. `-Wall -Werror` is set globally (CMakeLists.txt:33 FLAGS_PUBLIC). hipcc may emit new warnings (deprecated hipMallocHost, unused) that become errors. Watch for `-Werror` breaks; fix warnings rather than dropping -Werror.
7. `-fstack-protector-strong -D_FORTIFY_SOURCE=2 -Wl,-z,relro,-z,now` in release flags: confirm clang/hipcc accepts these on the HIP TUs (they are host-linker/codegen flags; usually fine, but the HIP device pass may warn).

## File-by-file change list (porter)
- ADD ucm/shared/trans/cuda_to_hip.h (or a shared ucm-level compat header): alias the cuda* runtime symbols listed above to hip*, include <cstring>/<cstdlib> BEFORE <hip/hip_runtime.h> (gpuRIR lesson), no-op include cuda_runtime.h on NVIDIA.
- EDIT ucm/shared/trans/cuda/cuda_sm_kernel.cu: guard the PTX `CudaCopyUnit` with `#if defined(__CUDA_ARCH__)`; add a HIP/`#else` branch using `__ldcs`/`__stcg` (or plain uint4 copy). Include the compat header.
- EDIT ucm/store/nfsstore/device/cuda/cuda_device.cu: same PTX guard for H2DUnit/D2HUnit; route the cuda* runtime calls through the compat header (or include it so the aliases apply).
- VERIFY ucm/sparse/gsa_on_device/csrc/cuda/ham_dist/paged_ham_dist_mla.cu + cp_async.cuh compile under hipcc; expected change: none or trivial (drop nvcc-only half unmacro flags in the rocm CMake arm).
- EDIT ucm/shared/trans/CMakeLists.txt: add `elseif(RUNTIME_ENVIRONMENT STREQUAL "rocm")` -> add_subdirectory(cuda) reused under HIP, OR a new rocm/ arm. Mirror the gdr-source exclusion logic.
- EDIT ucm/store/nfsstore/device/CMakeLists.txt: add a `rocm` arm creating the `storedevice` target via HIP.
- EDIT ucm/sparse/gsa_on_device/CMakeLists.txt: add a `rocm` arm building csrc/cuda/ham_dist (+ hash_retrieval, which is pure C++/pybind, no GPU) under HIP.
- ADD per-backend HIP CMake (either new rocm/CMakeLists.txt files or `if(RUNTIME_ENVIRONMENT STREQUAL "rocm")` inside the cuda/CMakeLists.txt): `enable_language(HIP)`, `set_source_files_properties(<.cu> PROPERTIES LANGUAGE HIP)`, `HIP_ARCHITECTURES` from `${CMAKE_HIP_ARCHITECTURES}` default gfx90a-when-unset, link amdhip64/torch as appropriate, force-include the compat header on the HIP TUs.
- EDIT setup.py: add `case "rocm": cmake_args += ["-DRUNTIME_ENVIRONMENT=rocm"]` (setup.py:174). Keep PLATFORM warning text in sync (optional).

## Build commands (gfx90a)
Configure the C++ store/trans surface (no torch needed) directly with CMake for fast iteration and unit tests:
```
cmake -S projects/unified-cache-management/src -B build_rocm \
  -DRUNTIME_ENVIRONMENT=rocm -DBUILD_UCM_STORE=ON -DBUILD_UNIT_TESTS=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ -DCMAKE_BUILD_TYPE=Release
cmake --build build_rocm -j16
```
Fat-binary check for the warp-size class:
```
cmake -S ... -B build_multi -DCMAKE_HIP_ARCHITECTURES="gfx90a;gfx1100" ... && cmake --build build_multi -j16
llvm-objdump --offloading build_multi/.../*.so   # expect gfx90a AND gfx1100
```
For the sparse hamming module, build via setup.py against a ROCm torch venv:
```
PLATFORM=rocm ENABLE_SPARSE=true pip install -e projects/unified-cache-management/src   # ROCm torch active
```

## Test plan
GPU-validatable correctness slices (the real gate):
1. C++ gtests (Strategy-A native, no torch): build with `-DBUILD_UNIT_TESTS=ON`, run `ctest` SERIALLY (one assigned GPU -- guide MPPI lesson). Key targets:
   - ucm/shared/test/case/trans/trans_test.cc -- exercises the copy kernel (cuda_sm_kernel) via Device::HostToDeviceAsync/DeviceToHostAsync round-trip and asserts byte-exact readback (UCTransUnitTest.CopyDataWithCE). This is a HARD correctness gate for the PTX rewrite.
   - ucm/store/test/case/* -- posix/cache/fake/pcstore store tests; the cache_trans_* cases exercise the storedevice (cuda_device.cu) H2D/D2H batch copy with readback compares.
2. Hamming kernel: ucm/sparse/test/gsa/test_cuda_hamming_mla.py and test_cuda_hamming_gqa.py drive `hamming.hamming_score(...)` on real GPU tensors. They are currently print/derive scripts, not asserting. PLAN: turn each into a pass/fail check by computing a CPU torch reference (XOR of int views + bitcount via `torch` then the same sink/recent/min-over-kv reduction) and asserting bit-exact/allclose against the kernel output. This gives a deterministic cross-checked correctness gate for the Hamming kernel. (test_hamming_mla.py / test_hamming_gqa.py are Ascend-NPU-only -- they import torch_npu/vllm_ascend; do NOT use those on AMD.)
Determinism: the copy kernels are pure memcpy (deterministic); the hamming kernel sums popcounts in a fixed per-thread order into shared mem then reduces -- integer adds, fully deterministic. Two-run bit-identical check is a valid extra gate.
Non-GPU regression set (must not regress): the top-level test/ pytest suite is vLLM end-to-end integration (requires model servers + pynvml) and is out of scope for a unit GPU run; do not regress the C++ gtests on the NVIDIA path (they are the only self-contained suite). Leave the simu/ascend/musa/maca backends untouched (additive rocm arm only) so their builds are unchanged.

## Open questions
- Does the host need `__ldcs`/`__stcg` to be cache-streaming for KV-transfer correctness, or is plain uint4 copy sufficient? Functionally a plain copy is correct; streaming is a perf hint. Start with the portable intrinsics (match maca), fall back to plain copy if an intrinsic is unavailable on the target ROCm.
- The sparse `hamming` build links full libtorch; confirm the validation host has a ROCm torch in the active env. If not readily available, the C++ store/trans gtests alone (no torch) are a sufficient first GPU-validation gate, with the hamming module a second pass.
- `cudaStreamAddCallback` is deprecated on newer CUDA; hipStreamAddCallback exists. Confirm it is present and not nodiscard-warning under -Werror on the ROCm in use.
