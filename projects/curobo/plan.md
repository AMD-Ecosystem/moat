# curobo port plan

Planner run 2026-11-20 on linux-gfx942 (analysis only; no GPU used, nothing built). Upstream analysed at `NVlabs/curobo` main `0f04ee18066822ec0c6751fa085d7b033734c4da` (2026-11-19, after tag v0.9.0). The intake section of `notes.md` covers licence and duplicate effort; this plan does not repeat them.

**Three rulings from jeff are needed before a porter starts.** They are stated with options at the top of the planner section in `notes.md` (Q1 runtime backend, Q2 `cpp_runtime/` scope, Q3 the Warp dependency). This plan is written for the recommended answers and says where a different answer changes it.

## Project

- Name: curobo (NVlabs/curobo, "cuRobo": GPU kinematics, dynamics, IK, trajectory optimisation, motion planning, MPC, collision, volumetric mapping)
- Upstream: https://github.com/NVlabs/curobo, default branch `main`
- Fork: https://github.com/AMD-Ecosystem/curobo, port branch `moat-port`
- Python package `nvidia-curobo` (setuptools, `pyproject.toml`), plus a separate C++ deployment runtime in `cpp_runtime/` (CMake, three projects)

## Existing AMD support

None authoritative. The only AMD effort is the community wrapper https://github.com/ZiguanWang/curobo_rocm_wrapper: patches over v0.8.0 (and over the old non-commercial v0.7.8), gfx1201 only, built on the v0.8.0 pybind extension that upstream has since deleted. Judgment: community, non-authoritative, and targeting a build path that no longer exists; a hint at most (it confirms the 32-bit mask and wave-size classes), never code to inherit. Decision: port.

Delivery vehicle: NVIDIA-owned, accepts DCO-signed contributions (CONTRIBUTING.md), does not link platform forks. Intake expects fork-as-deliverable; whether to offer an upstream PR is a later moat-checkup decision, not a planning blocker.

Performance kernels: none are CUTLASS/CuTe/wgmma. The only tensor-core code is one `nvcuda::wmma` tf32 kernel (`lbfgs_compact_step_kernel.cuh`) that upstream no longer dispatches (see surface). Decision: mechanical, correctness-first port everywhere; no AMD-native rewrite is called for.

## Build classification

Neither Strategy A nor B: **runtime-compiled kernels from Python** (`ext_type` recorded as `python-nvrtc-runtime`). Evidence:

- `pyproject.toml` has no compiled extension; package data ships the kernel sources: `[tool.setuptools.package-data] "curobo._src.curobolib.kernels" = ["**/*.cu", "**/*.cuh", "**/*.h"]` ("Include kernel source files for cuda.core runtime compilation").
- `curobo/_src/curobolib/backends/cuda_core_backend/kernel_cache.py`: `Program(source_code, code_type="c++", options=ProgramOptions(std="c++17", arch=f"sm_{device.arch}", ftz=True, fma=True, prec_div=False, prec_sqrt=False, ...)).compile("cubin", name_expressions=(kernel_name,))`, then `mod.get_kernel(kernel_name)`. Kernel names are template expressions specialised per call (`kinematics_forward_kernel<num_links, compute_com, n_joints, n_tool_frames>`), so the set of instantiations is open-ended and only known at run time.
- Launch: `cuda.core.launch(stream, LaunchConfig(grid, block, shmem_size), kernel, *args)` on a cuda.core stream wrapping `torch.cuda.current_stream().cuda_stream`; errors via `cuda.bindings.runtime.cudaGetLastError`; dynamic shared memory opt-in via `cudaFuncSetAttribute(..., cudaFuncAttributeMaxDynamicSharedMemorySize, ...)` (optimization.py, trajectory_seed_fitter.py, geometry.py).
- v0.8.0's `setup.py` + pybind torch extension were deleted in v0.9.0. `cuda.core` has no AMD implementation, and AMD's `hip-python` (now `rocm-bindings-hip`/`-core` 10.0) exposes only the low-level `hip`/`hiprtc` APIs, not a cuda.core equivalent.

`cpp_runtime/` is a separate pure-CMake build (`project(... LANGUAGES CXX CUDA)`, `find_package(CUDAToolkit 12.3)`, `CMAKE_CUDA_ARCHITECTURES`), i.e. Strategy A, if it is in scope (Q2).

## Port strategy

### Host side: a hipRTC launch path behind the cuda.core surface curobo uses (Q1, recommended)

curobo uses a small slice of cuda.core: `Device` (arch, set_current, create_stream from a foreign handle), `Program`/`ProgramOptions` -> compile -> `get_kernel`, `LaunchConfig`, `launch`, plus three `cuda.bindings.runtime` calls. The port adds one module that provides exactly that slice on ROCm and makes the backend import its primitives from one place:

1. New `backends/cuda_core_backend/hip_runtime.py` (name the porter's choice): hipRTC compile (`hiprtcCreateProgram`, `hiprtcAddNameExpression`, `hiprtcCompileProgram`, `hiprtcGetLoweredName`, `hiprtcGetCode`, the log on failure), `hipModuleLoadData` + `hipModuleGetFunction`, `hipModuleLaunchKernel` on the torch stream handle, `hipGetLastError`, `hipStreamSynchronize`, and a dynamic-shared-memory check that validates against the device limit (AMD has no opt-in above the per-block LDS size; `hipFuncSetAttribute` on a module function is not the CUDA call).
2. A selector (`torch.version.hip` is set) so `kernel_cache.py`, `launch_helper.py`, the `*_config.py` modules and `trajectory/timeopt/timeopt_solver.py` take `LaunchConfig`/`launch`/the attribute call from cuda.core on CUDA and from the HIP module on ROCm. On CUDA every line keeps its current behaviour; the cuda.core import stays lazy as today (`backends/__init__.py` proxy), and its "install the CUDA extra" error gets a ROCm counterpart.
3. Argument marshalling must match cuda.core's: Python `int` -> 64-bit (`intptr_t`; pointers come in as `tensor.data_ptr()` ints), `numpy.float32`/`numpy.int32` -> their own width, `bool` -> 1 byte. Upstream already wraps every float scalar in `np.float32`, so the same rules give correct parameters on both back ends. (This is why `torch.cuda._compile_kernel` is not usable: it packs Python ints as C `int`, truncating every pointer, and it is a private API.)
4. Binding: **ctypes against the HIP runtime and hipRTC libraries of the ROCm that torch itself loaded** (resolve from torch's library directory, then `ROCM_PATH`/`HIP_PATH`, then the pip ROCm SDK; on Windows scan for `hiprtc0*.dll` and `amdhip64_*.dll`). This adds no dependency and guarantees one HIP runtime in the process (the skill's "same ROCm in one process" rule). Read `gcnArchName` through `hipGetDevicePropertiesR0600` (or from `torch.cuda.get_device_properties().gcnArchName`) and pass `--offload-arch=<gcnArchName>`. The alternative is AMD's `rocm-bindings-hip` (`hip-python`) wheel: official bindings, but a new dependency whose version must track the ROCm major torch was built against. Either is acceptable; jeff rules (Q1).
5. Compile options (no `-ffast-math`: it implies finite-math under clang and curobo uses large sentinels; see faults-numerics): `-std=c++17`, `-O3`, `--offload-arch=<gfx>`, `-fgpu-flush-denormals-to-zero` (ftz), `-ffp-contract=fast` (fma; clang's HIP default), `-fno-hip-fp32-correctly-rounded-divide-sqrt` (prec_div/prec_sqrt false), `-I` the kernel include dirs as absolute paths, and `-D` a platform macro the shared headers test (the JIT inherits no host define). Debug (`curobo._src.runtime.debug_cuda_compile`): `-O0 -g`. Never `-fgpu-rdc` (0-byte code objects). NUL-terminate the log buffer.
6. Kernel cache key: `self.arch` becomes the gfx name on ROCm so cache entries cannot collide with an `sm_` string (AMD compute-capability numbers collide with NVIDIA's).
7. `_read_sources` prepends an `int8_t`..`uint64_t` typedef block for NVRTC. Under hipRTC those names may already be declared (`uint64_t` as `__hip_internal::uint64_t`); on HIP prepend nothing and `#include <cstdint>`-equivalent behaviour comes from hipRTC. Verify with a compile, do not guess.

Rejected alternatives: (a) reinstating a torch `CUDAExtension` for ROCm only: v0.9.0 kernels are template-specialised at run time on robot size (`num_links`, `n_joints`, `max_horizon`, probe counts, ...), so an ahead-of-time extension would need a fixed instantiation list or a dispatch rewrite, and it diverges from upstream's single build model; (b) a full port of `cuda.core` to HIP: far larger than the slice curobo uses and not ours to own.

### Device side: logical 32-lane warps, pinned in one header

The kernels are designed around 32-lane warps on purpose, host and device together: `curobo::common::warpSize = 32` (every one of the 35 device `warpSize` uses is this namespaced constant, not the built-in), 32-bit lane masks, `threadIdx.x & 31`, and host launch geometry that sizes blocks and shared scratch in units of 32 (`output_threads_per_batch = 32`, `DynamicsLaunchCfg.WARP_SIZE = 32`, `TrajectoryKernelCfg WARP_SIZE = 32`, `block=32` for the timeopt factor/correct kernels, `(threads_per_block // 32)` scratch slots, 1..32 line-search probes, 32 timing controls). Re-deriving all of that for 64 would touch host configs, shared-memory sizing and numerics across the whole package.

Instead keep the logical warp at 32 on every AMD target. Width-32 logical warp operations are arch-agnostic (skill: faults-wavefront, "Width-32 LOGICAL warp ops"); on wave64 a wavefront simply holds two independent 32-lane groups. Concretely:

- New `kernels/common/warp_ops.cuh` with project-owned inline functions for each warp primitive the kernels use: shuffle (idx/up/down/xor), ballot, activemask, match_any, any/all, syncwarp. On CUDA each is the native `*_sync` intrinsic, byte-for-byte the same code. On HIP: shuffles call `__shfl*(v, x, 32)` (width pinned); ballot/activemask return the caller's 32-lane half as a 32-bit mask, `(unsigned)(__ballot(p) >> (__lane_id() & 32))`; match_any returns the 32-lane peer mask restricted to that half; syncwarp is a wavefront barrier with fences. Do NOT `#define` the CUDA intrinsic names and do NOT rely on ROCm's own `*_sync` defaults (64-bit wavefront masks, width = `warpSize` = 64 on CDNA). Skill entries: faults-wavefront "A compat layer for CUDA's `*_sync`..." and build-headers-macros "Do not name compat macros after the warp functions ROCm declares".
- Mechanically switch the call sites to those functions (about 120 sites in 24 files, list below). Masks stay 32-bit logical masks, so `__popc`/`__ffs` on them stay correct.
- The platform test inside the header must work under hipRTC, where the host build's defines are absent: key it on `__HIPCC_RTC__`/`__HIP_DEVICE_COMPILE__` *and* the explicit `-D` the host passes (point 5 above).
- Sweep each kernel for the wave-size hazards the census lists (below) after the switch, especially the peer-election code (`rnea_helpers.cuh`, `self_collision_helper.cuh`: `__match_any_sync` + `__popc` + `__ffs - 1` as a source lane, which must be a lane index within the 32-lane group, not the wavefront) and any `threadIdx.x / 32` that indexes shared scratch sized from the host's `// 32`.

Other device fixes:

- `third_party/helper_math.h` (CUDA Samples): wrap the vector operator block in `#if !defined(__HIP_PLATFORM_AMD__)`-equivalent (HIP_vector_type already defines them; the free functions are ambiguous), and extend the `#ifndef __CUDACC__` host-fallback guard (skill: build-headers-macros, helper_math entry). Under hipRTC the platform macro must be the one the host passes.
- `optimization/pose_refiner/pose_refiner_lm_kernel.cuh` includes `<cuda_runtime.h>`: guard it (`__has_include` or the platform macro); hipRTC provides the runtime declarations itself.
- `common/cuda_precisions.h` includes `<torch/extension.h>`/`<c10/cuda/CUDAGuard.h>`: verify it is unreachable from any runtime-compiled kernel (it appears to be a v0.8 leftover); leave it untouched if so.
- `optimization/lbfgs/lbfgs_compact_step_kernel.cuh` uses `nvcuda::wmma` with `precision::tf32` and a hi/lo split to recover fp32 accuracy. Upstream says "tensor-core compact is no longer dispatched" (`curobo/_src/optim/gradient/lbfgs.py:102`); it is reachable only from `LBFGSCompactCu` in `curobo/tests/_src/optim/gradient/test_lbfgs.py`. On HIP give the tile product a plain fp32 FMA path in the same tiling (at least as accurate as the tf32 hi/lo emulation; gfx950 has no xf32 MFMA, so rocWMMA xf32 is not portable). The test compares it against the two-loop step, so it remains a real check.
- Dynamic shared memory: the backend sizes it from `shared_memory_per_block_optin`; on AMD that is the LDS size (64 KB on gfx90a/gfx942/RDNA, 160 KB on gfx950). Confirm what torch reports on ROCm and that every launcher that would exceed it takes its existing fallback (`use_shared_buffers_actual`, `_validate_and_configure_shared_memory`) rather than launching nothing.

### Warp (NVIDIA Warp) dependency (Q3)

About 80 modules import `warp`; heavy use of the tile API (`tile_sum` 96, `tile_map` 91, `tile_atomic_add` 90, `launch_tiled` 11, plus `tile_matmul`, `tile_cholesky`, `tile_cholesky_solve`, `tile_scan_inclusive`, `tile_argmin`), `wp.Mesh` BVH queries, `wp.from_torch` (659), `wp.stream_from_torch`/`ScopedStream`, and Warp launches inside torch CUDA-graph capture. No `wp.Texture`/`Volume` use, so AMD Warp's known texture gap (AMD-Ecosystem/warp#20) does not apply. curobo pins nothing (`warp-lang>=0.10.0`) but its notices assess warp-lang 1.16.0; AMD Warp is 1.13.0+amd0 on `amd-integration` and 1.14.0 on the `release/1.14.0.amd0` line that MOAT is porting as `warp-1.14` (stage `porting`, no platform validated yet). The porter must establish which AMD Warp build runs curobo's Warp kernels on each gate (tile reductions and tile linear algebra on wave64 in particular) and record it under `## Install as a dependency`-style instructions in notes. Whether to record `depends_on: [warp-1.14]` (which makes the selector wait for that port) is Q3.

### `cpp_runtime/` (Q2, recommended: defer to a second round)

~35k lines of C++/CUDA: robot runtime, calibration runtime, mapper runtime, 26 CTest targets and a benchmark. It uses the CUDA runtime API broadly (memcpy/memset, streams, graph capture and graph node APIs, `cudaStreamGetCaptureInfo_v2`, `cudaStreamUpdateCaptureDependencies`, profiler), the driver API (`cuModuleLoad`/`cuModuleGetFunction`/`cuLaunchKernel`/`cuMemsetD32Async`) to run **cubins exported from Python** (`cpp_runtime/robot/tools/export_*.py` via cuda.core, and Warp AOT modules via `wp.compile_aot_module`, whose `.cubin` files the C++ side loads with a hand-written Warp kernel ABI in `robot/src/warp_abi.h`), `nvrtcGetSupportedArchs`, CUB `DeviceScan`/`DeviceReduce` (mapper), and 65 `<<<>>>` launches. Porting it means Strategy A for the CMake projects, a code-object export path in the Python exporters, and AMD Warp AOT support for code objects, which may not exist yet in AMD Warp. The Python package is complete and useful without it (cpp_runtime is for deploying a configuration already validated in Python). Recommendation: phase 1 = the Python package; `cpp_runtime/` + exporters in a follow-up round once the Python port is validated and AMD Warp AOT is known. This is a partial-port cut and needs jeff's ruling.

## CUDA surface inventory

Kernel sources: 47 headers in `curobo/_src/curobolib/kernels/` (~630 KB; there are no `.cu` translation units, every kernel is a header compiled by NVRTC), plus 15 files in `cpp_runtime/`. `surface.json` lists them; component ids below in brackets.

| area | files | hazards found | HIP mapping |
|---|---|---|---|
| runtime compile + launch [backend:cuda_core-runtime-compile, python:timeopt_solver-direct-launch] | kernel_cache.py, launch_helper.py, 9 `*_config.py`/launch modules, context.py, timeopt_solver.py | cuda.core/cuda.bindings only; `sm_` arch strings | hipRTC + `hipModule*` via ctypes (Q1) |
| warp primitives [kernels:common and every group] | 24 files; top: self_collision_helper.cuh 32, timeopt_solver_kernel.cuh 20, lbfgs_step_kernel.cuh 18, block_warp_reductions.cuh 17, lbfgs_vector_free_step_kernel.cuh 16, timeopt_common.cuh 13, lbfgs_compact_step_kernel.cuh 10, rnea_helpers.cuh 10 | `__shfl*_sync` 67, `__syncwarp` 18, `fullMask`(0xffffffffu) 16 + literal `0xffffffff` 9, `__ballot_sync` 6, `__activemask` 3, `__match_any_sync` 2, `__any/__all_sync` 1, `__popc` 2/`__ffs` 5 on 32-bit masks, `& 31` 13, `% 32` 20, `/ 32` 31, `constexpr int warp_size = 32` (bspline_kernel.cuh) | `warp_ops.cuh`, logical width 32 |
| tensor cores [kernels:optimization-lbfgs] | lbfgs_compact_step_kernel.cuh | `nvcuda::wmma` 16x16x8 tf32, `__float_to_tf32` | fp32 FMA path on HIP (test-only kernel) |
| atomics | dynamics, kinematics, self-collision, lbfgs, timeopt | float `atomicAdd` (shared and global), `atomicMax` on int bit patterns, packed 64-bit max-contact entries | native; check 64-bit atomics on LDS and that accumulators are zeroed before first use |
| dynamic shared memory | 15 `extern __shared__`; opt-in in optimization.py, trajectory_seed_fitter.py, geometry.py | sizes from `shared_memory_per_block_optin` | validate against AMD LDS limits |
| launch bounds | lbfgs_step_kernel.cuh, lbfgs_compact_step_kernel.cuh | `__launch_bounds__(OP_THREADS)` | fine (helps AMD register allocation) |
| headers | helper_math.h, pose_refiner_lm_kernel.cuh, cuda_precisions.h | Samples operators; `<cuda_runtime.h>`; torch headers | guards as above |
| textures, PTX/asm, clock64, cooperative groups, `__ldg`, CUB/Thrust (Python path) | none | | |
| cpp_runtime [library:*, test:curobo_*, benchmark:curobo_benchmark, tool:export-cpp-runtime] | 15 files | runtime + driver API, graph-node APIs, CUB, NVRTC arch query, cubin loading, Warp AOT ABI | Strategy A + hipCUB + code objects (Q2) |

Host-side Python items with no CUDA symbol: `torch.cuda.*` streams/events/graphs (work on ROCm torch as-is), `torch.backends.cudnn.allow_tf32` in `seed_ik_solver.py` (harmless on ROCm), `multi_processor_count` small-GPU launch policy (`runtime._SMALL_GPU_MAX_SM_COUNT = 32`; CU counts are comparable, no change).

## Risk list

1. **Wave64 logical-warp correctness** (gfx90a/gfx942/gfx950). Every warp primitive must go through the pinned header; one missed `__shfl_down_sync` with the default width or one 64-bit ballot stored in `unsigned` breaks CDNA only, silently. Validate on wave64 and wave32. Peer election (`__match_any_sync` + `__ffs`) is the subtlest site.
2. **Partial warps.** Host configs launch blocks that are not multiples of 64 (32-thread blocks for timeopt factor/correct, `ceil_div(n, 32) * 32` rounding). A 32-thread block on wave64 is one half-populated wavefront: correct with pinned width-32 operations, wasteful but acceptable for correctness-first. Check that no reduction reads lanes 32..63 of such a wavefront.
3. **hipRTC vs NVRTC.** Missing predeclared types, typedef collisions, `__launch_bounds__` placement, `-D` platform macro absent from the JIT, include paths, name-expression lowering of template kernel names with `bool` arguments (`true`/`false` spelled in the expression), and error logs swallowed by the caller. Read the full log on a failed compile.
4. **Argument packing.** Must match cuda.core exactly (point 3). A wrong width is a silent garbage parameter, not an error.
5. **Graph capture.** curobo captures torch CUDA graphs that contain hipRTC-module launches and Warp launches on the torch stream; `hipModuleLaunchKernel` is capturable, and `torch.cuda.synchronize()` after a module load (kernel_cache.py) must not run while capturing (it already does not for cached kernels; check first-use inside capture).
6. **Fresh allocations not zeroed.** Any `torch.empty` buffer fed to an `atomicAdd` accumulator without a `zero_()` would be latent on CUDA too, but surfaces differently; sweep the launchers that accumulate (dynamics backward, kinematics backward `grad_q`, self-collision).
7. **Numerics.** ftz + contraction differ in detail between NVRTC and clang. Upstream tests carry tolerances; any ROCm-only loosening is gated on ROCm, never on CUDA.
8. **AMD Warp coverage** of tile reductions/linear algebra and graph capture at the version curobo needs (Q3). A Warp failure is a dependency defect to register, not a curobo port bug.
9. **Shared-memory limits** (64 KB LDS): launchers that pick a shared-buffer variant from the opt-in limit must fall back, not skip the launch.
10. **Upstream velocity.** Very active (several commits a day). Expect sync rounds; keep the diff small and confined to the backend selector, `hip_runtime.py`, `warp_ops.cuh`, mechanical call-site edits, two header guards, the wmma fallback, packaging and docs.

## File-by-file change list (phase 1, Python package)

- `curobo/_src/curobolib/backends/cuda_core_backend/hip_runtime.py` (new): ctypes hipRTC + HIP module runtime implementing the cuda.core slice (compile/get_kernel, LaunchConfig, launch, stream from torch handle, last error, stream sync, shared-memory attribute check).
- `.../kernel_cache.py`: choose compiler by `torch.version.hip`; arch = gcnArchName; HIP option list; no NVRTC typedef prelude on HIP; no `cuda.pathfinder` lookup on HIP.
- `.../launch_helper.py`, `optimization.py`, `trajectory_seed_fitter.py`, `geometry.py`, every `*_config.py` that imports `LaunchConfig`, `trajectory_timeopt.py`, `pose_refiner.py`, and `curobo/_src/trajectory/timeopt/timeopt_solver.py`: import `LaunchConfig`/`launch`/attribute helper through the selector instead of `cuda.core`/`cuda.bindings` directly.
- `curobo/_src/curobolib/backends/__init__.py`: ROCm-aware import error message.
- `curobo/_src/curobolib/kernels/common/warp_ops.cuh` (new) and the 24 kernel headers in the census: call-site switch to the pinned functions; `curobo_constants.h` stays (its `warpSize = 32` and `fullMask` are the logical warp).
- `kernels/third_party/helper_math.h`, `kernels/optimization/pose_refiner/pose_refiner_lm_kernel.cuh`: guards.
- `kernels/optimization/lbfgs/lbfgs_compact_step_kernel.cuh`: HIP fp32 tile path.
- `pyproject.toml`: a ROCm install route (an extra with no CUDA packages, or document `pip install .` without an extra plus a ROCm torch and AMD Warp); no CUDA extra changes.
- `docs/getting-started/installation.rst`: ROCm section (torch ROCm wheel, AMD Warp, install command, supported GPUs).
- Tests: no test is weakened or skipped. If a test hard-codes CUDA-only behaviour (an `sm_` string, cuda.core objects), add the HIP branch beside it.

## Build commands

Nothing is compiled ahead of time; "building" is installing the package and letting the first kernel use compile it. Per gate:

```bash
# venv with a ROCm PyTorch matching the host ROCm (fleet hosts: the image torch or a job venv)
python -m venv .venv && . .venv/bin/activate
# ROCm torch: index per host ROCm version (pip SDK hosts: the TheRock index)
pip install --index-url <rocm-torch-index> torch
# AMD Warp: build or install from AMD-Ecosystem/warp (branch per Q3), e.g.
git clone -b <amd-warp-branch> https://github.com/AMD-Ecosystem/warp && (cd warp && python build_lib.py && pip install -e .)
# curobo (fork, moat-port), no CUDA extra
pip install -e ".[dev]"
python -c "import torch, warp, curobo; print(torch.version.hip, warp.__version__)"
```

Windows: same with TheRock wheels (`torch` with the `[device-<gfx>]` extra, hipRTC DLL resolution per point 4).

## Test plan

GPU suite (the gate), from `pyproject.toml` `[tool.pytest.ini_options]` (`testpaths = ["curobo/tests"]`, `-n 4 --dist loadscope` by default):

```bash
python -m pytest curobo/tests -n 4 --dist loadscope -ra 2>&1 | tee pytest-full.log
```

- Must pass, in this order of signal: `curobo/tests/_src/curobolib` (direct kernel tests: dynamics launch config, cuda_ops), `robot/kinematics`, `robot/dynamics`, `optim/gradient/test_lbfgs.py` (includes the compact-step kernel), `cost` (self-collision), `trajectory` (+ `timeopt`), `solver`, `motion`, `perception/mapper` (PBA, pose refiner, Warp-heavy), `test_examples.py` (getting_started end-to-end).
- Record pass/fail/skip counts per directory and the reason for every skip; a skip that only happens on ROCm is a finding.
- Run once with `-n 1` on each gate as well (graph capture + JIT under one process), and once with `curobo._src.runtime.cuda_graphs = False` if a graph test fails, to separate capture faults from kernel faults.
- Prove kernels ran: the first-use compile log (`Compiling kernel: ...`) and a nonzero count of distinct compiled kernels in the cache.
- Both wavefront widths: one wave64 (gfx90a or gfx942/gfx950) and one wave32 (gfx11/gfx12) run at the same head; Windows on gfx1151 last.
- Benchmarks (`benchmark/ik_benchmark.py`, `inverse_dynamics_kernel_benchmark.py`) as performance evidence only.
- Non-GPU regression set: the whole suite is collected on CPU-only machines as well (`pytest --collect-only` must still work without cuda.core or hipRTC importable, because `backends/__init__.py` loads backends lazily); config/parser/types tests that need no device must keep passing.
- CUDA no-regression: the CUDA path is Python plus NVRTC-compiled headers, so the check is (a) the kernel headers still compile under NVRTC for an `sm_` target with the edited `warp_ops.cuh` (run NVRTC through cuda.core on the nvcc fleet host; no GPU needed for `Program.compile`), and (b) the Python import paths on a CUDA torch are unchanged. Record the command and result in notes.

## Open questions

Q1-Q3 are in `notes.md` (planner section, top) with options; they block the porter. Smaller items the porter settles and records:

- Whether `cuda_precisions.h` is reachable from any runtime compile (expected: no).
- Exact hipRTC behaviour for the integer typedef prelude on the host ROCm (10.x).
- What `shared_memory_per_block_optin` torch reports on each AMD target.
- The `amd`/`AMD-AGI` org duplicate check intake could not do (SSO) is still open; no evidence of an internal port surfaced in planning either.
