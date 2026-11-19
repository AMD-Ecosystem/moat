# qrack port plan (linux-gfx90a lead)

## Project
- Name: qrack
- Upstream: https://github.com/unitaryfoundation/qrack
- Default branch: main (cloned @ 49d2efc)
- Domain: GPU-accelerated universal quantum-computer simulator (state-vector / Schrodinger-style), C++11. GPU backends are OpenCL (primary, mature) and CUDA (optional, `ENABLE_CUDA`).

## DISPOSITION
Clean HIP port of the self-contained CUDA backend (Strategy A, compat-header). Effort class: LOW-to-MODERATE. No CK/AMD-native rewrite needed (correctness-first mechanical port is the right and sufficient first step; see value judgment). Set `planned`. Dispatch a porter on GCD3.

## Existing AMD support + value judgment
- No native ROCm/HIP backend exists. No rocm/hip/amd branch upstream; no fork under ROCm/AMD/GPUOpen orgs or with rocm/hip/amd in the name (checked `gh api repos/unitaryfoundation/qrack/forks` and `/branches`); zero `rocm`/`hip`/`__HIP_`/`hipMalloc`/`hip_runtime` references anywhere in the source tree.
- AMD GPUs today are reachable ONLY via qrack's OpenCL backend (AMD OpenCL ICD). That is the existing AMD path. There is no HIP path.
- Web search found AMD/ROCm quantum-sim work for OTHER simulators (qsim HIP backend, Qiskit-Aer ROCm Thrust backend, PennyLane-Lightning-Kokkos, Qibo) but NONE for qrack. None is authoritative-for-qrack; treat as ecosystem context only, no code to inherit.
- Value question (honest): per PORTING_GUIDE rule, "AMD supported only via OpenCL with no HIP path -> a ROCm/HIP port of the CUDA code is still valuable." Qrack's own README claims OpenCL often outperforms its CUDA path, so the perf upside of a native ROCm/HIP backend over OpenCL-on-AMD is NOT guaranteed and may be modest. The defensible value is: (1) a first-class ROCm/HIP build for users who standardize on ROCm tooling (profilers, rocBLAS-adjacent stacks, containers) rather than the AMD OpenCL ICD; (2) the CUDA backend currently only runs on NVIDIA -- a HIP port makes that exact code path run on AMD; (3) the port is cheap and self-contained (see below), so value/effort is favorable. This is NOT a skip: OpenCL-on-AMD existing does not, by the guide, justify skipping a HIP port. But it should be framed to upstream as "native ROCm/HIP backend" not "first AMD support."

## Build classification: pure CMake (Strategy A)
Evidence:
- Top-level `CMakeLists.txt`; no `find_package(Torch)`, no `torch.utils.cpp_extension`, no setup.py torch dependency. Not a pytorch extension.
- Dedicated `cmake/CUDA.cmake`: `option(ENABLE_CUDA ... OFF)`, `enable_language(CUDA)`, `target_compile_definitions(qrack PUBLIC ENABLE_CUDA=1)`, and adds exactly three `.cu` TUs to the `qrack` target:
  - `src/common/cudaengine.cu`
  - `src/common/qengine.cu`
  - `src/qengine/cuda.cu`
- Arch handling already parameterized: `QRACK_CUDA_ARCHITECTURES` (auto-detect if unset) -> `set_target_properties(qrack PROPERTIES CUDA_ARCHITECTURES ...)`. Clean analogue for `CMAKE_HIP_ARCHITECTURES`.
- The CUDA backend is a SELF-CONTAINED set of 5 files (3 `.cu` + 2 headers `include/common/cuda_kernels.cuh`, `include/common/cudaengine.cuh`), mirroring the OpenCL backend (`oclengine.cpp` + `*.cl`). Host C++ selects the backend via `ENABLE_CUDA`/`ENABLE_OPENCL` macros; the `.cu`/.cuh are not entangled with host code. This is the colmap-model ideal for Strategy A.

## Port strategy: A (compat header + LANGUAGE HIP), rationale
- Add one `include/common/cuda_to_hip.h` (NVIDIA: no-op `#include <cuda_runtime.h>`; HIP: `#include <hip/hip_runtime.h>` + the ~20 `cudaXxx -> hipXxx` aliases this backend actually uses, plus `cuda_fp16.h -> hip_fp16.h` and `__half2`/`make_half2` aliases). Include it at the top of the 3 `.cu` and 2 `.cuh`.
- In `cmake/CUDA.cmake` add `option(USE_HIP ... OFF)`; when set, `enable_language(HIP)`, mark the 3 `.cu` `LANGUAGE HIP`, set `HIP_ARCHITECTURES` from `CMAKE_HIP_ARCHITECTURES` (default gfx90a only when unset; never hardcode a literal that overrides the followers). Keep `enable_language(CUDA)` on the `else` path so the NVIDIA build is byte-identical.
- Map `-use_fast_math` (nvcc) to the HIP equivalent and pin `-ffp-contract=on` (see risks).
- Keep all guards rare; the diff should be the compat header + ~30 lines of CMake.

## CUDA surface inventory
Kernels (all in `src/common/qengine.cu`, declared in `include/common/cuda_kernels.cuh`): ~60 `__global__` state-vector kernels -- 1-/2-qubit gate application (`apply2x2*`), Pauli/phase/invert, uniformly-controlled, compose/decompose/dispose, probability/parity/expectation reductions (`prob*`, `probmask*`, `probparity`, `expperm`, `forcemparity`), normalization (`nrmlze*`, `updatenorm`), `approxcompare`, measurement (`applym*`), shuffle, ALU (inc/dec/mul/div/modexp, indexed LDA/ADC/SBC, hash, BCD) under `ENABLE_ALU`/`ENABLE_BCD`.
- Reductions: shared-memory tree reduction via `SUM_LOCAL` macro (`extern __shared__ qCudaReal1 lBuffer[]`, halving loop with `__syncthreads()` at EVERY level, `locID==0` writes `sumBuffer[blockIdx.x]`). WAVE-SIZE SAFE: no warp-shuffle, no `__ballot`, no hardcoded 32, syncs every level (no last-warp barrier elision). A second `extern __shared__ qCudaCmplx lCmplxBuffer[]` reduction (qengine.cu:1087) follows the same pattern.
- Warp intrinsics: NONE. No `__shfl*`, `__ballot`, `__activemask`, `__popc`, `warpSize` in device code, no hardcoded 32, no cooperative groups.
- Atomics: NONE in device code.
- Libraries: NONE -- no cuBLAS/cuFFT/cuRAND/cuSPARSE/Thrust/CUB/CUTLASS. (RNG is host-side, `cmake/Random.cmake`.) No library swaps required.
- fp16: `__half2`/`make_half2` + `cuda_fp16.h`, used ONLY when `FPPOW < 5` (half precision). Default FPPOW=5 (float) does NOT touch fp16. HIP has `hip_fp16.h` + `__half2`/`make_half2`; alias in the compat header for completeness but it is off the default validation path.
- Runtime API used (all 1:1 to HIP): `cudaMalloc/Free`, `cudaMemcpy[Async]` (+ kinds), `cudaMemsetAsync`, `cudaStreamCreate/Destroy/Synchronize`, `cudaDeviceSynchronize`, `cudaGetDeviceCount/Properties`, `cudaDeviceProp`, `cudaEvent_t` (EventVec), `cudaHostRegister/Unregister` (+`cudaHostRegisterDefault`) for pinned host memory, `cudaLaunchHostFunc` (host callback). All have direct `hipXxx` equivalents.
- Launch geometry: kernels are grid-stride loops over `MAXI_ARG` with block dim = `nrmGroupSize`, derived at RUNTIME from `properties.warpSize` (`GetPreferredSizeMultiple`/`GetMaxWorkGroupSize`), clamped and rounded to a power of two. On gfx90a this is 64 (vs 32 on NVIDIA/RDNA); `localBuffSize` (dynamic shared mem) tracks the same group size, so the reduction buffer auto-sizes. No source change needed for wave64.

## Risk list
- Wave size: LOW. No warp-level primitives anywhere; reductions are block-level `__syncthreads`-fenced trees that are wave-agnostic, and block dim comes from a runtime `warpSize` query (64 on gfx90a, 32 on RDNA) -- not a hardcoded constant. The dynamic-shared-mem reduction buffer auto-sizes to the (runtime) block dim. gfx1100/gfx1151 followers should re-run but no warp-size source delta is anticipated.
- Rule-of-five on resource handles: LOW-MODERATE. `CUDADeviceContext` (`include/common/cudaengine.cuh`) holds two `cudaStream_t` (default-init to 0 -- good) and destroys them in `~CUDADeviceContext` (good), but declares no deleted/explicit copy/move. It is held only via `DeviceContextPtr = shared_ptr<CUDADeviceContext>` (cudaengine.cuh:33), so it is never copied in practice -> no double-`hipStreamDestroy`. Confirm no value-copy path during the port; add `= delete` copy/move if any is found (CUDA tolerates double-destroy of a stream, AMD may fault -- the colmap CuTexObj class).
- `-use_fast_math` + clang(HIP) default `-ffp-contract=fast`: LOW. Pin `-ffp-contract=on` (expression-only) on the HIP path to match nvcc FMA formation and reduce float drift. Qrack unit tests compare against tolerances / analytic probabilities (not bit-exact KATs), so the residual risk is small, but set it to avoid borderline tolerance failures.
- OOB neighbor reads / textures / surfaces / layered arrays / texture pitch: NOT APPLICABLE. No textures, surfaces, or stencil/neighbor gathers in this backend.
- `__smid()`-indexed pools: NOT APPLICABLE (no `__smid()` use).
- `-Werror all-warnings` (nvcc) in the Release CUDA opts: the HIP/clang path must NOT inherit `-Werror all-warnings` verbatim (that is an nvcc spelling) -- map to a sane clang warning posture or drop `-Werror` on the HIP path to avoid spurious build breaks from HIP-header warnings.
- `--cudart=shared` / `--ptxas-options` / `-Xcompiler`: nvcc-only flags; gate them behind the CUDA (non-HIP) branch so the HIP compile never sees them.
- Block size = warpSize (=64 on gfx90a): functionally correct; a porter perf note only. Occupancy of a 64-thread block on CDNA is fine; no correctness impact.

## File-by-file change list
- ADD `include/common/cuda_to_hip.h` -- the only HIP-aware file: runtime aliases + fp16 + `__half2`/`make_half2`.
- EDIT `include/common/cuda_kernels.cuh` -- include the compat header (replace the bare `cuda_fp16.h` include with the guarded one).
- EDIT `include/common/cudaengine.cuh` -- include the compat header (for `cudaStream_t`, `cudaEvent_t`, `cudaDeviceProp`); consider `= delete` copy/move on `CUDADeviceContext` if a copy path exists.
- EDIT `src/common/cudaengine.cu`, `src/common/qengine.cu`, `src/qengine/cuda.cu` -- include the compat header at top (after `<cstring>/<cstdlib>` if any host memcpy/memset is used inside a HIP-compiled TU; see gpuRIR lesson) ; no kernel-body changes expected.
- EDIT `cmake/CUDA.cmake` -- add `USE_HIP` option, `enable_language(HIP)` branch, `LANGUAGE HIP` on the 3 `.cu`, `HIP_ARCHITECTURES` from `CMAKE_HIP_ARCHITECTURES`, HIP-appropriate compile opts (`-ffp-contract=on`, fast-math equivalent, drop nvcc-only flags), keep the CUDA branch intact.
- EDIT top-level `CMakeLists.txt` lines ~195-201 / ~302-323 -- ensure the `QRACK_CUDA_COMPILE_OPTS` (with `-use_fast_math -Werror all-warnings --ptxas-options --cudart=shared -Xcompiler`) are applied only on the CUDA path; provide a HIP opt set on the USE_HIP path.

## Build commands (gfx90a)
```
cd /var/lib/jenkins/moat/projects/qrack/src   # (porter works in the fork clone, not here)
cmake -S . -B _build \
  -DENABLE_CUDA=ON -DENABLE_OPENCL=OFF \
  -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DENABLE_TESTS=ON
cmake --build _build -j16 --target unittest
```
Multi-arch correctness check for the warp-size class (build only): `-DCMAKE_HIP_ARCHITECTURES="gfx90a;gfx1100"` then `llvm-objdump --offloading _build/libqrack.so | grep -E 'gfx90a|gfx1100'` -- expect both code objects.

## Test plan (real GPU, self-contained, egress-free)
- The GPU is selected at COMPILE time: with `ENABLE_OPENCL=OFF` + `ENABLE_CUDA=ON`, `test/test_main.cpp` sets `QRACK_GPU_*` to `CUDAEngine`/`QEngineCUDA`/`QINTERFACE_CUDA`, so all GPU-backed tests exercise the CUDA (post-port: HIP) path and compare to the CPU/analytic reference. No external datasets.
- Primary gate: `_build/unittest` (Catch2; `test/tests.cpp`). Runs quantum circuits and checks probabilities/state against analytic expectations on the GPU backend.
- Target the GPU layers explicitly via the `--proc-*-cuda` / GPU CLI opts in `test_main.cpp` (e.g. `--layer-qengine` plus the CUDA/QEngine selectors) so the run hits `QEngineCUDA`, not a CPU fallback. Porter to enumerate the exact opt names from `test_main.cpp` (`Opt(...)` list around lines 92+) and record the precise command in notes.md.
- Non-GPU regression set that must not regress: the CPU engine tests (QEngineCPU / QUnit-over-CPU) in the same `unittest` binary; run the full `unittest` and confirm the CPU layers still pass. (The port touches only `.cu`/CMake; CPU paths should be unaffected.)
- `_build/benchmarks` (`test/benchmarks.cpp`) is optional perf/smoke, not a correctness gate.

## Open questions
- Exact `--proc-*`/`--layer-*` CLI flag combination that forces `QEngineCUDA` for the unit tests (porter resolves from `test_main.cpp`).
- Does any code path value-copy `CUDADeviceContext` (would require `= delete` move/copy)? Inspect `CUDAEngine`/`InitCUDAResult` usage.
- HIP equivalent posture for `-use_fast_math` (qrack relies on fast intrinsics in `qCudaDot`/normalization); confirm tolerance tests still pass with `-ffp-contract=on` + HIP fast-math.
- `cudaLaunchHostFunc` semantics under ROCm (`hipLaunchHostFunc`) for the params-queue host callback path -- verify behavior matches.
- Whether to also validate the `FPPOW<5` (half-precision, `__half2`) build, or scope the lead gate to default float precision and note half as follow-up.
