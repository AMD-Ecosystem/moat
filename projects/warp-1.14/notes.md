# warp-1.14 notes

## Intake (2026-11-18)

Release line of NVIDIA/warp for upstream's `release-1.14`. The screen, the take-up ruling (2026-11-10) and the base ruling (2026-11-11) are in the development-line project's notes (`git show origin/port/warp:projects/warp/notes.md`); they bind this line and are not repeated here. Jeff Daily directed the start on 2026-11-18. First line, so no `depends_on`; `warp-1.15` will depend on this one.

Recorded from those rulings (`--by jeffdaily`): `set-fork-branch release/1.14.0.amd0`, `set-delivery fork-release` (tag `v1.14.0+rocm.0`), `set-fork-url https://github.com/AMD-Ecosystem/warp`. `mirror-branch` created `release-1.14` and `release/1.14.0.amd0` in the fork, both at upstream's 3601b2d07708 (one commit past tag `v1.14.0`: "Run docs deploy without project sync").

### Licence
- Apache-2.0, tier 1 (`set-license-spdx`, evidence LICENSE.md on `release-1.14`).
- `licenses.py scan-nvidia warp-1.14`: 4 hits, all licence notices, no proprietary source file: `README.md` (the "Building from Source" paragraph naming the NVIDIA Software License Agreement for libmathdx), `licenses/cuda-LICENSE.txt`, `licenses/libmathdx-LICENSE.txt`, `licenses/nanovdb-LICENSE.txt`. The in-tree `warp/native/nanovdb/` headers are Apache-2.0 (OpenVDB project); `licenses/nanovdb-LICENSE.txt` reproduces an older NVIDIA notice.
- Consequence for the wheel: upstream's CUDA build downloads NVIDIA libmathdx and links it statically, and its wheels redistribute CUDA runtime pieces under `licenses/cuda-LICENSE.txt`. A `+rocm` wheel must contain neither; the planner accounts for what replaces or disables the libmathdx-backed features (tile Cholesky/solve/FFT/matmul) and for the licence files the ROCm wheel ships.

### Fork clone and Git LFS
The repository uses Git LFS (`*.usd`, `*.png`, `*.gif`, `*.nvdb`, `*.grid`; 67 files), so a fresh clone carries LFS's own `pre-push` hook and `protect-fork` refuses to replace it, leaving the pristine mirror unguarded. Per clone: `mv .git/hooks/pre-push .git/hooks/pre-push.lfs-original`, then `python3 utils/moatlib.py protect-fork warp-1.14`. The port is not expected to add LFS-tracked files; if a commit ever does, run `git lfs push origin release/1.14.0.amd0` before the push. `protect-fork` chaining an existing LFS hook is a tooling gap for the trunk.

## Planning (2026-11-18, windows-gfx1151 host, planner)

Plan written to `plan.md` from source at `3601b2d07708`; no build or GPU run on this host. `surface.json` generated (`python utils/surface.py generate warp-1.14`: 41 CUDA files, 11 components) and extended to 58 components covering the runtime compiler, driver API, wave width, libmathdx replacement, graphs, textures, packaging, the test groups and the three platforms. `ext_type` set to `python-build-nvrtc`. Eight open questions for jeff are in the plan (Q1-Q8); none is ruled.

## Inherited GitHub Actions in the fork (2026-11-18)

Creating the mirror `release-1.14` matched the `release-*` push trigger of upstream's workflows, which ran on upstream's own commit 3601b2d0 and failed for reasons unrelated to the port: CI (both Windows CUDA jobs: the `windows-2025` runner has Visual Studio 18, which nvcc 12.9/13.0 refuses) and Deploy Sphinx documentation to Pages (`-Werror` on a `_POSIX_C_SOURCE` redefinition from the runner's Python 3.12.13 headers). CodeQL passed.

Jeff Daily disabled the inherited workflows on 2026-11-18 with `gh workflow disable` (write access suffices; he has no admin rights on the fork): CI (222324845), Deploy Sphinx documentation to Pages (222324849), Build Warp C++ Test Environment Image (331229363). Left active: the AMD Warp team's security scans, Black Duck SCA, Copilot code review, and CodeQL.

Still open for the porter: `draft-release.yml` triggers on `v*` tags and is not registered in the fork yet, so it cannot be disabled ahead of time. The release tag `v1.14.0+rocm.0` sits on a port-branch commit, so the port branch must stop it there: guard its job with `if: github.repository == 'NVIDIA/warp'` (the idiom upstream uses in `ci.yml:26` and `build-warp-cpp-test-env-image.yml:22`). Do this before sign-off.

## Porting session 1 (2026-11-18, linux-gfx950 MI355X, porter): Phase 0 complete

Fork head after this session: `f1719cce` on `release/1.14.0.amd0` (pushed; `advance-head` recorded). Stage left at `porting` (lock held by this host) for the next session, as directed.

### Environment
- ROCm: TheRock wheels `rocm[devel]==10.2.0` (`rocm-sdk-core`, `rocm-sdk-devel` 10.2.0) installed into the job venv from `https://stable.repo.amd.com/rocm/whl-next/`; root from `rocm-sdk path --root` (`.../site-packages/_rocm_sdk_devel`). HIP 7.17.26434, ROCm clang 24 (`lib/llvm/bin/clang++`). This host has no system ROCm (`/opt/rocm` absent).
- CUDA check: `/usr/local/cuda` nvcc 12.8, no NVIDIA GPU on the host.
- Host compiler g++ 13.3; Python 3.11 job venv (numpy 1.26.4). warp-clang's packman LLVM download worked.
- `ruff==0.15.9` and `clang-format==22.1.2` (the versions pinned in `.pre-commit-config.yaml`) installed into the job venv for formatting.
- Git LFS: `sudo apt-get install git-lfs` (3.4.1), then in the clone `git lfs install --local --skip-repo` (filters only, no hooks, so the protect-fork pre-push hook stays) and `git lfs pull`; needed for `warp/tests/assets/*.nvdb` (test_volume reads them). `git status` stays clean.

### Commits (release-1.14..f1719cce)
1. `cf4c370e` Stop release tags from triggering workflows in this fork: `if: github.repository == 'NVIDIA/warp'` on draft-release's job, `tags: v*` removed from ci.yml and codeql.yml. `grep -n -A3 "tags:" .github/workflows/*.yml` now shows only draft-release (guarded) and sphinx.yml. Note for Jeff: sphinx.yml ("Deploy Sphinx documentation to Pages") also has a tag trigger, `'v[0-9]*.[0-9]*.[0-9]*'`, which matches `v1.14.0+rocm.0`; it is one of the workflows Jeff disabled in the fork on 2026-11-18, so it cannot run, and the plan limited the change to the three files, so it was left as is.
2. `06d642ef` [ROCm] version `1.14.0+rocm.0` in VERSION.md, warp/config.py, warp/native/version.h. The pre-commit `check_version_consistency.py` regex rejected a PEP 440 local label; it gained `(\+[a-z0-9]+(\.[a-z0-9]+)*)?`.
3. `df7718a1` [ROCm] NanoVDB: `defined(__CUDACC__) || defined(__HIP__)` (NanoVDB's own idiom) on the 9 gates that hide device helpers (PNanoVDB.h:80,278; NanoVDB.h:1226; GridHandle.h:529; NodeManager.h:319; math/Math.h:519,1286; util/cuda/Util.h:239,319).
4. `3f731039` [ROCm] HIP build of warp.so (build driver, compat header and shims, driver wrappers, device predicate, new exports). Details below.
5. `bf6bd5ba` [ROCm] Python: `Runtime.is_hip`, `Runtime.hip_version`, `Device.is_hip`, `Device.arch_name`; greeting/diagnostics show HIP and the gfx target; test_devices/test_diagnostics assert the AMD values on AMD GPUs.
6. `f1719cce` [ROCm] clang-format of the HIP-only code (all hunks inside `#if WP_ENABLE_HIP` in cuda_util.cpp, plus two shims).

### Design as built (Phase 0)
- Backend selection (Q6 ruling): `build_lib.py` gains `--rocm-path` and `--hip-arch`. With neither toolkit named it runs `find_cuda_sdk()` and the new `find_rocm_sdk()` (WARP_ROCM_PATH, ROCM_PATH, HIP_PATH on Windows, `python -m rocm_sdk path --root`, /opt/rocm); both found is an error naming `--cuda-path`/`--rocm-path`; naming both is an error. ROCm rejects `--clang-build-toolchain`, `--use-dynamic-cuda`, `--libmathdx-path`; `--use-libmathdx` (default on) is forced off with a message. `--hip-arch` without ROCm is an error.
- Default `--offload-arch` list (ruling): read from `<rocm>/share/therock/dist_info.json` key `dist_amdgpu_targets` (TheRock 10.2 Linux: 25 targets, gfx908 ... gfx1250); fallback to `offload-arch`/`amdgpu-arch` (GPUs present) with a warning; otherwise an error asking for `--hip-arch`. Phase 0 builds used `--hip-arch gfx950`; the full-list build time/size is Phase 7's measurement.
- `build_dll.py` HIP branch (Linux): `.cu` compiled by `<rocm>/lib/llvm/bin/clang++ -x hip --rocm-path=... --offload-arch=<each> -fno-gpu-rdc -std=c++17 -DWP_ENABLE_HIP=1 -I native/hip -include native/hip_compat.h -DWP_DISABLE_CUBQL -O3 -fPIC ...`; host `.cpp` by g++ with `-DWP_ENABLE_HIP=1 -D__HIP_PLATFORM_AMD__ -DWP_DISABLE_CUBQL -I native/hip -I <rocm>/include` (the -Werror host flags unchanged); link `-lamdhip64 -lhiprtc -Wl,-rpath,<rocm>/lib`; `WP_ENABLE_CUDA_COMPATIBILITY=0`. Windows HIP raises "not supported yet" (Phase 6).
- Shim headers (`warp/native/hip/`, on the HIP include path only): cuda.h, cuda_runtime.h, cuda_runtime_api.h, cudaTypedefs.h, nvrtc.h, cuda_fp16.h, cub/cub.cuh, cub/device/{device_radix_sort,device_run_length_encode,device_scan,device_reduce,device_segmented_radix_sort}.cuh, cub/block/block_reduce.cuh, hip_cub.h (`namespace cub = hipcub;`). They pull in `warp/native/hip_compat.h`: 228 `#define`s generated from `hipify-perl -experimental` over every CUDA identifier used in warp/native (*.cu/*.cpp/*.h) and the NanoVDB CUDA headers, curated (driver entry points excluded, CUBIN/PTX getters excluded), plus `CUcontext` = `WpHipContext*`, `cuuint64_t`, `CUoccupancyB2DSize`, `cudaMallocHost` (3-arg hipHostMalloc), `CU_EVENT_WAIT_*`. CUDA_VERSION/CUDART_VERSION are left undefined (hipCUB/rocPRIM read them as NVIDIA backend; NanoVDB's `CUDART_VERSION < 13000` path then takes the int-device API, which matches HIP).
- `cuda_util.cpp`: `#if WP_ENABLE_HIP` block replaces the dlopen/cuGetProcAddress section and the `*_f` wrappers with direct HIP calls. Contexts are emulated (`WpHipContext {device, is_primary}`): primary handles created at init per device; `cuCtxGetCurrent_f` follows `hipGetDevice()` (so a device set by other HIP code, e.g. PyTorch, is honoured); `cuCtxCreate_f` allocates a non-primary handle sharing the device; push/pop via a thread-local stack; `cuStreamGetCtx_f` via `hipStreamGetDevice`; `cuMemcpyPeerAsync_f` maps contexts to devices; `cuFuncSetAttribute_f(MAX_DYNAMIC_SHARED_SIZE_BYTES)` succeeds iff value <= `hipDeviceAttributeMaxSharedMemoryPerBlock` (AMD has no opt-in); occupancy callback variant returns hipErrorNotSupported (Warp passes NULL). `hipError_t` is `[[nodiscard]]` in C++17 and the host build is -Werror, so `check_cu_result`'s ignored `pfn_cuGetErrorString` call goes through a void-returning wrapper.
- `warp.cu`: nvPTXCompiler only on CUDA; device init records `gcnArchName` and `warpSize`; `is_ipc_supported` = discrete && !Windows (HIP has no IPC-event attribute); `is_capture_pause_supported()` helper (HIP false until Phase 3); `wp_cuda_toolkit_version()` = 0 on HIP, new `wp_hip_runtime_version()` = HIP_VERSION; `wp_nvrtc_supported_arch_count()` = 0 on HIP; PTX module load refused on HIP; **`wp_cuda_compile_program` on HIP is a stub that prints "Runtime kernel compilation is not available in this build" and returns -1 (Phase 1 replaces it with the hipRTC compile)**. New exports `wp_is_hip_enabled`, `wp_hip_runtime_version`, `wp_cuda_device_get_arch_name`, `wp_cuda_device_get_warp_size` (warp.h, warp.cpp CPU stubs; warp.map exports `wp_*`).
- Device predicate, by what each gate protects (Q2 ruling): `__HIP_DEVICE_COMPILE__` added to builtin.h presence gates over portable device code (fmin/fmax 705-737, block_dim 1940, tid debug print 1951, generic/int64/f64 atomic_add host gates, atomic_min/max/cas/exch/and/or/xor), bvh.h (shared stack, bvh_load_node), svd.h (rsqrt), crt.h (WP_API and host helper decls). `CUDA_CALLABLE` keyed on `__CUDACC__ || __HIPCC__`; apic_internal.h `__CUDACC__` gates gain `__HIPCC__`. Left CUDA-only on purpose (inline PTX): builtin.h:241 approx_rcp/div (HIP takes exact division), :337 half conversions (HIP takes the existing `__clang__` `_Float16` branch, as ruled), :452/:471 bf16 arithmetic (generic float round-trip, as ruled). intersect.h, texture.h (Phase 5) and all tile*.h gates are untouched (Phase 1 / Phase 4).
- Atomics interim: fp16/bf16 atomic_add and bf16 atomic_min/max on HIP use `wp_hip_atomic_update_u16` (32-bit atomicCAS on the containing aligned word, other half untouched). Correct but CAS-based; Phase 1 replaces atomic_add with the packed hardware atomics and fp32/fp64 `unsafeAtomicAdd` per the amended Q2 ruling, with the allocator/memory-kind evidence and gfx942/wave32 measurements it requires.
- **Interim, Phase 2 removes it: HIP builds define `WP_DISABLE_CUBQL`** (host and device) because the vendored cuBQL GPU builder does not compile under HIP. Backport check: `git apply --check --directory=warp/native -p1` of NVIDIA/cuBQL#34 (`git diff e82f1dc^1 e82f1dc -- cuBQL`) fails on builder/cuda.h, math/math.h, queries/triangleData/math/rayTriangleIntersections.h, traversal/rayQueries.h; #35 (`git diff d1bfc3c^1 d1bfc3c -- cuBQL`) fails on bvh.h and math/common.h. 18 upstream commits lie between the vendored 5a651b3 and #34's base, so the hunks need manual application (ruling: one commit per PR, non-applying hunks listed with reasons). Side effect: the CPU cuBQL builder is also off in HIP builds, so 3 CPU tests that hard-code `bvh_constructor="cubql"` fail (below).

### Build and test results (gfx950, HIP_VISIBLE_DEVICES=0)
- HIP build: `python build_lib.py --rocm-path "$(rocm-sdk path --root)" --hip-arch gfx950 -j 128` -> "Warp build succeeded" (warp.so .cu build 19.4 s wall at 64 workers, link 0.24 s; full script ~21 s). Remaining compiler output is only `-Wunused-value` "ignoring return value of type 'hipError_t' declared with 'nodiscard'" warnings on upstream's deliberate ignores (warp.cu 10 sites, apic.cu 9, NanoVDB 86).
- `python -c "import warp as wp; wp.init(); print(wp.get_cuda_devices())"` -> `['cuda:0']`; greeting `HIP 7.17`, `"cuda:0" : "AMD Instinct MI355X" (288 GiB, gfx950:sramecc+:xnack-, mempool enabled)`; `arch`=95, `arch_name`=`gfx950:sramecc+:xnack-`, sm_count 256, IPC True, mempool True. `print_diagnostics()` HIP section: HIP 7.17, hipRTC 9.0 (what `hiprtcVersion` returns).
- `python -m warp.tests -s autodetect -p test_version.py` 4/4 OK; `-p test_devices.py` 13/13 OK; `-p test_diagnostics.py` 14/14 OK (after the two AMD-value assertions; before them: 1 failure each, `get_cuda_supported_archs()` empty and arch regex `^sm_\d+$`).
- CPU regression, GPUs hidden (`HIP_VISIBLE_DEVICES=-1 CUDA_VISIBLE_DEVICES=-1 python -m warp.tests -s autodetect -k cpu -m 32`): pristine `release-1.14` CPU-only build (`build_lib.py --no-cuda`) 2264 tests OK (skipped=29); port HIP build 2264 tests, errors=3, skipped=29. The 3 are test_mesh_query_ray_cpu, test_mesh_refit_cpu, test_mesh_query_ray_edge_cpu ("Failed to create mesh" for `bvh_constructor="cubql"`), caused by the interim WP_DISABLE_CUBQL; they must pass again after Phase 2.
- With the GPU visible, `-k cpu` (2251 tests) has 5 failures/29 errors: tests that launch on the default device `cuda:0` and hit the runtime-compile stub ("CUDA kernel build failed with error code 18446744073709551615"), plus the same 3 mesh tests. Expected until Phase 1.
- CUDA no-regression (nvcc 12.8, separate worktree at `bf6bd5ba`, identical CUDA expansion to `f1719cce` since the last commit only formats HIP-only code): `python build_lib.py --cuda-path /usr/local/cuda --no-use-libmathdx -j 128` -> "Warp build succeeded"; import with no NVIDIA driver reports "CUDA Toolkit 12.8, CUDA driver not available (NVRTC compilation available)". The libmathdx variant was not run this session.
- `python3 utils/jargon.py --port warp-1.14` clean; `ruff check`/`ruff format --check` clean on changed Python except `tools/pre-commit-hooks/check_version_consistency.py`, whose `"-v", "--verbose"` line is upstream's own formatting (not touched).

### Next step (Phase 1, start here)
1. hipRTC branch of `wp_cuda_compile_program` (replace the HIP stub): target from the device's `arch_name` (Python must pass it; the int `arch` collides with NVIDIA numbers), code object output (`.hsaco`, `hiprtcGetCode`), option translation per plan risk 2 (`--offload-arch=`, `-ffp-contract`, `-ffast-math`, no PCH/LTO/`--device-as-default-execution-space`), `--define-macro=WP_TILE_WARP_SIZE=<warpSize>` and `__CUDA_ARCH__=1` for user snippets (Q2), hipRTC include paths found by Warp itself (ruling 5). Python: `get_cuda_output_arch`/arch suffix bypass, output extension and kernel-cache key on the full gcnArchName, PCH off, `is_cubin_supported` meaning on HIP.
2. Device predicate for hipRTC: `__CUDACC__`/`__CUDACC_RTC__` gates in crt.h:233 (cuda_crt.h exclusion), builtin.h:62/66, tile.h:17-67, volume.h:14-23 need their hipRTC spelling; first probe which macros hipRTC defines (`__HIPCC__`, `__HIPCC_RTC__`, `__HIP__`). `wp_hip_atomic_update_u16` and lambdas must compile under hipRTC.
3. Q2 atomics: packed fp16/bf16 hardware add with the allocator evidence, fp32/fp64 `unsafeAtomicAdd` after the memory-kind audit, measure vs CAS on gfx942/gfx950 (and a wave32 part later); record numbers.
4. Then the Phase 1 test list in plan.md.

## Lesson for cuda-to-rocm: hipError_t is [[nodiscard]], so a -Werror host build breaks on ignored results CUDA allows
Reference file: `references/fault-classes.md` (headers, includes and build). HIP declares `hipError_t` as `[[nodiscard]]` whenever the code is compiled as C++17 or later (`__HIP_NODISCARD` in hip_runtime_api.h, with no opt-out macro), while CUDA's `cudaError_t` and `CUresult` carry no such attribute. Code that deliberately drops an error code (clearing a sticky error with `cudaGetLastError();`, or calling an error-string getter for its out-parameter) compiles silently with nvcc or g++ against CUDA headers and warns under HIP; in a host build that uses `-Werror` (Warp's g++ flags) the warning becomes an error. Fix the few host-side sites (cast to `(void)` or route through a void-returning wrapper) instead of adding `-Wno-unused-result`, which would also hide real ignored errors; in device-side translation units without `-Werror` the warnings are harmless noise. (warp-1.14)

## Porting session 2 (2026-11-18, linux-gfx950 MI355X, porter): Phase 1 complete

Fork head `fda7eb27` (pushed, `advance-head` recorded). Commits after `f1719cce`:
- `2f0d2552` docs workflow: removed the `'v[0-9]*.[0-9]*.[0-9]*'` tag trigger from sphinx.yml (Jeff Daily, 2026-11-18: no v* tag may start any workflow in the fork, even a disabled one). `grep -n -A3 "tags:" .github/workflows/*.yml` now shows only draft-release, whose job is guarded.
- `75874a38` [ROCm] runtime compile with hipRTC (native + Python).
- `fda7eb27` [ROCm] test adjustments (below).

### Runtime compiler (as built)
- hipRTC predefined macros, measured with a hiprtcCompileProgram probe (`agent_space/probe/rtc_probe.cpp`, gfx950 and gfx1100 alike): `__HIPCC__`, `__HIPCC_RTC__`, `__HIP__`, `__HIP_DEVICE_COMPILE__`, `__HIP_PLATFORM_AMD__`, `__AMDGCN__`, `__clang__` (and `__GFX9__` on gfx9). NOT defined: `__CUDACC__`, `__CUDACC_RTC__`, `__CUDA__`, `__CUDA_ARCH__`, any wavefront-size macro. hipRTC is device-only.
- New export `wp_hip_compile_program(src, name, target, include_dir, debug, opt, verbose, verify_fp, fast_math, fuse_fp, lineinfo, output_path)`; `wp_cuda_compile_program` on ROCm builds is an error stub. Python `build_cuda(..., hip_target=device.arch_name)` calls it on `runtime.is_hip`; output name `<module>.<target with ':'->'_'>.hsaco`; `get_cuda_output_format()` = "cubin" and `is_cubin_supported` = True for AMD devices; loading uses the existing non-PTX path (`hipModuleLoadDataEx`).
- Options: `--offload-arch=<gcnArchName>`, `-I<warp/native>`, `-std=c++17`, `-O<0..3>`, `-DNDEBUG`/`-D_DEBUG -g`, `-gline-tables-only` for lineinfo, `-D/-UWP_VERIFY_FP`, `-DWP_ENABLE_MATHDX=0`, `-ffast-math`, `-ffp-contract=on|off` (fuse_fp). No PCH, no LTO. hipRTC found its builtin headers itself from the TheRock wheel (no environment variables).
- Source wrapping: `#pragma clang force_cuda_host_device begin` + `#define __CUDA_ARCH__ 1` + `#line 1` + source + `end`. The pragma replaces NVRTC's `--device-as-default-execution-space` (without it tile_radix_sort's CPU sort and tile_matmul's `__syncthreads` fail as host functions calling device functions). `__CUDA_ARCH__` is defined in the source text, after hipRTC's pre-included runtime header, so that header never sees it (this settles the Q2 "builtin headers must not react" check by construction).
- Q2 audit of Warp's presence gates: HIP-first (`__HIP_DEVICE_COMPILE__` tested before `__CUDA_ARCH__`) at builtin.h approx_rcp/div (now an AMD branch: `__builtin_amdgcn_rcpf`/`__builtin_amdgcn_rcp`, v_rcp_f32/v_rcp_f64; test_fast_math tolerances pass), half conversions (Clang `_Float16`), bf16 arithmetic (generic, `!HIP &&` added to the 900/800 gates), texture.h (9 sites; device textures are Phase 5). Numeric atomic gates were already HIP-first. tile*.h presence gates gained the HIP device predicate (device code paths); tile.h float4/unroll/thread-index gates gained hipRTC; `#pragma unroll` inside a macro only works with nvcc, so `_Pragma` is used for every Clang HIP compile. tile_matmul's four entry points gained `CUDA_CALLABLE` (needed by the ROCm AOT compile too).
- `tile_mask_t` (64-bit on AMD device code, `unsigned int` on CUDA) introduced in tile_reduce.h and used by tile_radix_sort's `warp_scan_inclusive` (the only non-template ballot site, which hipRTC rejected: HIP's `__ballot_sync` static_asserts a 64-bit mask). The 32-lane logic itself is unchanged: Phase 4 makes it correct on wave64.
- crt.h under hipRTC: neither cuda_crt.h nor the host CRT declarations; `assert()` maps to `__assert_fail`. `wp.breakpoint()` maps to `__builtin_debugtrap()` on AMD GPUs.
- Q7 runtime switch: `wp.config.llvm_cuda` with an AMD device raises in `Module._compile` before the cache lookup ("... not supported on AMD GPUs; the HIP runtime compiler ... is already Clang-based"); options are resolved when the kernel is defined, so the new test sets the flag before defining its kernel.

### Q2 user snippets (probe `agent_space/probe/snip.py`, gfx950 wave64)
A `func_native` snippet with `#if defined(__CUDA_ARCH__)` took the device branch and these compiled and returned exact expected values: `nextafterf`, `__frsqrt_rn`, `__popcll`, `__ffsll`, `__clzll`, `__syncwarp`, `__threadfence`, `__shared__` + `__syncthreads`. Wave32 still to be checked on a Radeon host. New test `test_snippet.py::test_cuda_arch_device_branch` (nextafterf in the device branch).

### Q2 atomics: measurements and an item stopped for Jeff
ISA of small hipRTC kernels (`agent_space/probe/atom.cu`, `atom2.cu`; `llvm-objdump -d` of the code object), ROCm 10.2:

| target | atomicAdd f32 | atomicAdd f64 | unsafeAtomicAdd f32 | unsafe pk f16 / bf16 | atomicMin f32 | atomicMin/Max f64 | safe atomicAdd(__half2) |
|---|---|---|---|---|---|---|---|
| gfx950, gfx942 | hw add_f32 | hw add_f64 | hw | hw pk_add_f16 / pk_add_bf16 | CAS | hw min/max_f64 | CAS |
| gfx90a | CAS | CAS | hw | CAS | CAS | CAS | CAS |
| gfx908, gfx1030 | CAS | CAS | CAS | CAS | CAS | CAS | CAS |
| gfx1100, gfx1151 | CAS | CAS | hw | CAS | CAS | CAS | CAS |
| gfx1201 | hw add_f32 | CAS | hw | hw pk_add_f16 / pk_add_bf16 | hw min_num_f32 | CAS | CAS |

- fp32/fp64 `atomic_add` stays HIP's `atomicAdd`, which is already the hardware instruction wherever the backend considers it valid for every memory kind (gfx942, gfx950; f32 on gfx1201).
- float/double `atomic_min`/`atomic_max` now use HIP's `atomicMin`/`atomicMax` (minNum semantics, same as Warp's device `fminf`/`fmaxf`): hardware f64 on gfx942/gfx950, f32 on gfx1201, CAS elsewhere. test_atomic 164/164.
- **Stopped, needs Jeff's call: the switch to `unsafeAtomicAdd` (fp32/fp64 on gfx90a/gfx11) and to the packed fp16/bf16 hardware add.** The amended ruling makes both conditional on Warp arrays being device memory (and, for the packed add, on the neighbour half always being mapped device memory). That cannot be established: Warp launches GPU kernels on pinned CPU arrays when the device has UVA (test_unified_memory "cuda_launch_reads/writes_pinned_cpu_array_when_uva_supported"; AMD devices report `is_uva` True), the unsafe forms are not valid on that fine-grained host memory, and generated kernel code cannot tell array kinds apart. The safe `atomicAdd(__half2*)` is a CAS loop on every target, so the packed hardware add needs the unsafe form. So fp16/bf16 atomic_add stays the 32-bit CAS on the containing word (correct; test_fp16 6/6, test_bf16 48/48) and fp32/fp64 stay `atomicAdd`. Options for Jeff: (a) keep as is; (b) adopt the unsafe forms on all AMD targets and document that atomics on pinned CPU arrays from AMD GPU kernels are unsupported; (c) add a module option for it. Timings against the CAS forms were not taken, because the switch was not made.

### test_atomic_cas: hang measured, explained
Upstream's test spins every thread of a launch on one lock (`while atomic_cas(lock,0,0,1) == 1: pass`). On the MI355X it never finishes (`test_cas_2d_float32_cuda_0` stopped after more than 5 min). Probes (`agent_space/probe/cas*.py`): CAS itself is correct (64 threads, exactly 1 winner); the portable "do the critical section in the acquiring iteration" form also hangs with 2 threads, for `while done == 0` and for `while True ... break`, but works for 100 threads once the loop has a second exit (`and it < 2000000000`). This is consistent with the compiler sinking the lock release out of the branch into the loop-exit block, which on hardware without independent thread scheduling runs only after every lane has left the loop; HIP documents intra-wavefront locks as unsupported. Not a ROCm defect and not a port defect, so no deferral. The test now uses `get_test_devices_with_independent_thread_scheduling()` (CPU and NVIDIA GPUs); 24/24 CPU cases pass. Release notes should state that spin locks contended inside one wavefront hang on AMD GPUs.

### Phase 1 test results (gfx950, HIP_VISIBLE_DEVICES=0, `python -m warp.tests -s autodetect -p <file> -m 32`)
test_codegen 117 OK; test_snippet 12 OK; test_arithmetic 126 OK; test_math 16 OK; test_fp16 6 OK; test_bf16 48 OK; test_atomic 164 OK; test_atomic_bitwise 6 OK; test_atomic_cas 24 OK (GPU cases filtered, above); test_array 138 OK; test_copy 14 OK; test_launch 32 OK; test_fast_math 12 run, OK, 1 skipped (PTX content check, AMD produces no PTX); test_grad 41 OK (test_gradient_slice_3d_1d needed atol=1e-5 on AMD: ROCm device `powf(19, 1.0)` = 18.999998, `powf(21, 2.0)` = 441.00003, within the library's 1 ulp, CUDA stays exact); test_kernel_cache 5 OK; test_module_hashing 8 OK; test_print 10 OK; test_mat*.py (matrix/, 7 files) 601 OK; test_vec* 531 OK; test_quat* 173 OK; test_spatial* 207 OK; test_rand 10 OK; test_noise 14 OK; test_clang_cuda 7 run: 6 NVIDIA-only skipped, test_llvm_cuda_unsupported OK.
- CPU regression with GPUs hidden after Phase 1: 2264 tests, errors=3, skipped=29 (the same 3 cubql mesh tests as Phase 0; upstream 2264 OK).
- CUDA no-regression at `fda7eb27`: `build_lib.py --cuda-path /usr/local/cuda --no-use-libmathdx` succeeded (nvcc 12.8); offline NVRTC compile from that build of a module with tile_load/tile_sum/tile_matmul, fp16/bf16 atomic_add, f64 atomic_min/max and div_approx produced `sm86.cubin` and `sm90.ptx` (`agent_space/nvrtc_check/nvrtc_offline.py`), so the shared headers still compile under NVRTC.
- Kernel cache note: the test runner clears the shared kernel cache at start; a probe compiling at the same moment failed once with error 4294967295 and passed on rerun. Run probes when no test runner is active.

## Porting session 2, continued (2026-11-18, linux-gfx950 MI355X, porter): Phase 2 complete

Fork head `d46141ae` (pushed, `advance-head` recorded). Commits after `fda7eb27`:
- `9674c7c7` [ROCm] backport NVIDIA/cuBQL#34 (merge e82f1dc826, Ingo Wald) into the vendored cuBQL. Applied as-is: builder/cpu/spatialMedian.h, builder/cuda/builder_common.h, builder/cuda/sm_builder.h, math/affine.h, math/box.h, math/common.h, math/vec.h, hunk 1 of builder/cuda.h and of math/math.h. Adapted: math/math.h hunk 2 (host `__int_as_float` fallbacks also excluded under `__HIPCC__`; its CUBQL_INF part does not apply). Not applied (snapshot code differs, hunks only reformat or comment out debug output): builder/cuda.h hunk 2, rayTriangleIntersections.h hunk 1, traversal/rayQueries.h hunks 1-7. This commit was first made without the `[ROCm]` prefix that `audit-commits` requires (the added lines name `__HIPCC__`); since it had never been pushed or validated, it was redone with `git reset --soft HEAD~1` before the push.
- `2eec289e` [ROCm] backport NVIDIA/cuBQL#35 (merge d1bfc3c2e3, Jeff Daily). Applied as-is: builder/cuda.h, math/constants.h, math/vec.h, queries/pointData/findClosest.h, traversal/shrinkingRadiusQuery.h. Applied by hand with unchanged content (context differs): bvh.h (two `__HIPCC__` gates; its whitespace-only line not taken), math/common.h (the cudaXxx->hipXxx alias block). CMake/README/samples files of the PR do not apply.
- `f869ac29` (no prefix, shared-code fix) value-initialize tile reduction locals returned or copied from every thread (see lesson below).
- `632a2edf` [ROCm] tile collectives on 64-wide wavefronts (pulled forward from Phase 4, because sparse, linear solvers and fem use tile reductions): `WP_TILE_WARP_SIZE` 64 for gfx9 device code, 32 otherwise; `tile_mask_t`/`WP_TILE_FULL_MASK`/`tile_ffs`; tile_scan over WP_TILE_WARP_SIZE lanes; radix sort kept in 32-thread groups (explicit width 32, per-group half of the ballot via `__lane_id() & 32`, 32-based group count `WP_TILE_RADIX_SORT_GROUP_SIZE`); `WP_TILE_WARP_THREADS` = min(block_dim, warp size) for the tier-2 axis reduction (block_dim 32 on wave64 had reduced 104 of 200 elements); `WP_TILE_SHARED_ARRAY`/`WP_TILE_SHARED_VAR` raw storage for class-typed `__shared__` declarations (Clang rejects their implicit initialization).
- `01b066f2` [ROCm] WP_DISABLE_CUBQL removed from the ROCm build; hip_cub.h now `namespace cub { using namespace hipcub; }` (cuBQL declares the same), `cudaMallocHost` -> `hipHostMalloc` object-like (C++ overload defaults flags; matches cuBQL's mapping, no redefinition warning).
- `d46141ae` [ROCm] cuBQL SAH builder sorts all 64 key bits under `__HIPCC__` (rocPRIM defect below). Warp builds cuBQL BVHs with `enableSAH()`, so only sah_builder.h is used; sm_builder.h, elh_builder.h and rebinMortonBuilder.h carry the same [32,64) sort and are untouched.

### rocPRIM radix sort defect (registered: deferred `rocprim-radix-sort-u64-end-bit-64`, kind rocm-bug-report, not filed)
Standalone reproducer (`agent_space/probe/sortbits/s2.hip`, ROCm clang `-x hip --offload-arch=gfx950 -O2`, links amdhip64): `hipcub::DeviceRadixSort::SortKeys` on `uint64_t` keys, random values in bits [b,e), stream 0. With e == 64 and b in {1,4,8,16,24,31,32,33,40}: output not ordered by the selected bits for n >= 1025 (n = 1024 correct), still a permutation. Correct: [0,64), [0,63), [1,63), [16,48), [8,40), [32,56), [32,63) for every n up to 1e6. rocPRIM 4.8 (`ROCPRIM_VERSION 400800`), ROCm 10.2 wheels. Effect in Warp: cuBQL GPU BVH with leaf size > 1 wrong; test_mesh_query_ray_count_intersections (needs usd-core) leaf 4: 1337/5000 counts wrong; leaf 1 and 2 and all other constructors exact. Filing upstream (rocPRIM, and a note to cuBQL whose HIP path is affected) is Jeff's decision.

### The tile_sum bug (diagnosis record)
tile_sum over a block of two or more wavefronts returned one wavefront's partial (128 ones -> 64; correct at `-O0`, garbage at `-O1`, wrong at `-O2/-O3`; adding a printf hid it). Ruled out by experiment: kernel cache (probes now use a private `WARP_CACHE_PATH` per run, `agent_space/wppy.sh`), the shared-array macros, the shuffle union punning, dynamic LDS size (16 bytes requested, as on CUDA), and a raw func_native version of the same combine (correct). LLVM IR of the generated module (hipRTC `-fgpu-rdc` bitcode, `llvm-dis`) showed `llvm.assume(icmp eq workitem.id.x, 0)` at loop entry; minimal kernels located it in tile_sum; cause was `block_combine_thread_results` returning `T block_sum;` assigned only on thread 0 (Clang `noundef` return). Fixed by value-initialization (commit `f869ac29`); afterwards tile_sum correct for block 64..1024 at every -O level.

### Phase 2 test results (gfx950, `-s autodetect -p <file> -m 32`, usd-core installed in the job venv)
test_runlength_encode 16 OK; test_array_reduce 16 OK; test_sparse 47 OK; test_linear_solvers 34 OK; test_coloring 3 OK; test_bvh 13 OK; test_grouped_bvh 11 OK; test_hash_grid 23 OK; test_marching_cubes 17 OK; test_mesh*.py 49 OK (0 skipped with usd-core; 18 skipped "Requires usd-core" without it); test_volume 59 OK; test_volume_write 5 OK; test_apic 40 OK; test_apic_mesh 6 OK; test_fem_*.py 107 run, 2 failures, both "Unexpected output" (Warp's init greeting captured by a check_output test that is the first to initialize Warp in its worker): test_gimp_quadrature and test_cube_shape_functions. The pristine release-1.14 CPU-only build fails the same way when test_fem_quadrature.py / test_fem_shape.py run alone (`HIP_VISIBLE_DEVICES=-1`): upstream test-isolation behaviour, not a port defect, test left unchanged. Also: test_tile_reduce 62 OK, test_tile_sort 2 OK (Phase 4 still owns the full tile set).

### CPU-only regression at the end of Phase 2 (Jeff Daily's requirement)
`HIP_VISIBLE_DEVICES=-1 CUDA_VISIBLE_DEVICES=-1 python -m warp.tests -s autodetect -k cpu -m 32`:
- port (`d46141ae` HIP build, cuBQL enabled): **Ran 2264 tests, OK (skipped=16)**
- upstream release-1.14 CPU-only build, same environment (usd-core installed): **Ran 2264 tests, OK (skipped=16)**
The 3 cubql mesh errors of Phase 0/1 are gone. (The earlier upstream baseline of 2264 OK, skipped=29 was taken before usd-core was installed.)

### CUDA no-regression at `d46141ae`
`build_lib.py --cuda-path /usr/local/cuda --no-use-libmathdx` (nvcc 12.8) succeeded with "cuBQL: enabled"; offline NVRTC compile of the tile/atomic module (`agent_space/nvrtc_check/nvrtc_offline.py`) produced sm86.cubin and sm90.ptx. `audit-commits` OK, `jargon.py --port` clean.

### Next step (Phase 3, start here)
Streams, graphs, mempools, multi-GPU, interop on wave64 (plan Phase 3). Known items found so far:
1. `is_capture_pause_supported()` in warp.cu returns false on HIP; HIP 7.17 has `hipStreamBeginCaptureToGraph`, so pause/resume (used by capturable_tmp_alloc and capture_if/while staging) can be enabled on HIP by moving `wp_cuda_graph_pause_capture`/`resume_capture` out of the `CUDA_VERSION >= 12040` block (conditional-node functions stay unsupported, Q4).
2. Q4: `is_conditional_graph_supported()`/`assert_conditional_graph_support()` (context.py:~9987) must answer from an explicit HIP capability with a HIP-worded error, not from `toolkit_version is None`.
3. `unittest_utils.is_cuda_graph_module_load_supported()` keys on `wp.get_cuda_driver_version() >= (12, 3)`, which is None on HIP, so graph-module-load tests are skipped on AMD GPUs; needs a HIP answer (measure whether hipModuleLoad during capture works).
4. HIP_VISIBLE_DEVICES pins this job to one GPU; multi-GPU/peer/IPC tests need a session with more than one device visible (the host has more).
5. DLPack `kDLROCM` for HIP arrays (dlpack.py), torch-ROCm interop (needs a ROCm torch wheel matching ROCm 10.2 in the job venv).
Then the Phase 3 test list in plan.md.

## Lesson for cuda-to-rocm: an uninitialized value returned from a device function lets Clang assume the thread index
Reference file: `references/fault-classes.md` (floating point / types and dispatch, or a new "Clang optimizes undefined behaviour nvcc tolerates" entry next to the MSVC-lenient one). Block-reduction helpers often assign the result only on thread 0 (`T sum; if (threadIdx.x == 0) sum = ...; return sum;`) and every thread returns it. nvcc/NVVM tolerates this. Clang marks return values `noundef`, so returning an uninitialized value on any path is immediate undefined behaviour, and the optimizer may conclude the other paths never run: on AMD GPUs it emitted `llvm.assume(threadIdx.x == 0)`, then folded `partials[threadIdx.x / warpSize]` to `partials[0]`, so a two-wavefront block sum returned one wavefront's partial. Symptoms: correct at -O0, wrong at -O1 and above, and the bug disappears when a printf is added. Find it by compiling to LLVM IR (hipRTC `-fgpu-rdc` + `llvm-dis`, or `-emit-llvm`) and grepping for `llvm.assume` on `workitem.id`. Fix: value-initialize (`T sum {};`) every local that is returned or copied while unassigned on some threads. (warp-1.14)

## Lesson for cuda-to-rocm: rocPRIM radix sort of 64-bit keys with end_bit 64 and a non-zero begin_bit
Reference file: `references/fault-classes.md` (library swaps, next to the hipCUB/rocPRIM entries). With rocPRIM 4.8 (ROCm 10.2), `hipcub::DeviceRadixSort::SortKeys` on 64-bit keys returns keys not ordered by the selected bits when `end_bit == 64` and `begin_bit > 0`, for more than 1024 keys; `[0,64)` and any range ending below bit 64 are correct. Code that sorts packed `{payload, key}` words by their high half (`begin_bit = 32, end_bit = 64`, common in BVH builders) silently mis-groups. Workaround when the key is the high half: sort all 64 bits. Check this in any port that passes a sub-range ending at bit 64. (warp-1.14)

## Porting session 3 (2026-11-18, linux-gfx950 MI355X, porter): Phase 3 complete (single-GPU part)

Fork head `8d13d596` (pushed, `advance-head` recorded). Commits after `d46141ae`:
- `c225c180` [ROCm] graph capture pause/resume on HIP: `wp_cuda_graph_pause_capture`/`resume_capture` moved verbatim out of the `CUDA_VERSION >= 12040` block under `#if CUDA_VERSION >= 12040 || WP_ENABLE_HIP` (HIP 7.17 has hipStreamBeginCaptureToGraph; conditional-node functions keep their stubs, Q4); `is_capture_pause_supported()` true on HIP. Q4 per ruling: `is_conditional_graph_supported()` returns False and `assert_conditional_graph_support()` raises "Conditional graph nodes are not supported on AMD GPUs: HIP has no conditional graph node API", both from `runtime.is_hip`, not from version comparisons. Module load during capture: measured working on HIP in global, thread-local and relaxed modes (`agent_space/probe/capload.py`: kernel first loaded inside the capture, graph replayed 3x, correct), so `capture_begin` uses `force_module_load=False` on HIP as for CUDA driver 12.3+, and `unittest_utils.is_cuda_graph_module_load_supported()` returns True for HIP devices; test_map's graph-capture test uses that helper instead of asserting a CUDA driver version (it had asserted `driver_version is not None`).
- `4f5bcf21` (shared fix) zero-initialize the IPC event handle so a failed `hipIpcGetEventHandle` (event without interprocess=True; HIP returns hipErrorInvalidConfiguration) reads as all-zero and triggers Warp's existing warning.
- `9e90beb3` [ROCm] occupancy: HIP's `hipModuleOccupancyMaxPotentialBlockSize` suggests 1024 for a `__launch_bounds__(128)` kernel whose `HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK` is 128 (standalone probe `agent_space/probe/occ/occ.cpp`); the wrapper passes that attribute as blockSizeLimit (gives 128, grid 4096). Registered deferred `hip-module-occupancy-ignores-launch-bounds`.
- `a7166b39` (shared fix) cuBQL refit runs on Warp's current stream instead of stream 0. Measured: a cubql mesh refit inside `wp.ScopedCapture` segfaulted in `hipMallocAsync` on the null stream (rocgdb backtrace: libamdhip64 hipMallocAsync <- warp.so <- wp_mesh_refit_device); CUDA would invalidate a global capture there instead. Registered deferred `hip-mallocasync-null-stream-during-capture-segfault`. After the change the in-capture refit matches the no-capture refit and the CPU (note: the probe's 2.0 expectation was wrong for cubql: cubql's mesh_query_point returns 2.828 on this grid on the CPU as well; sah/median/lbvh give 2.0). BVH rebuild inside a capture measured correct for sah, median, lbvh (20000 boxes); `wp.Bvh` offers no cubql constructor upstream; Mesh has refit only.
- `cfac4f9b` [ROCm] DLPack: AMD GPU arrays export `kDLROCM`, pinned arrays of a ROCm build `kDLROCMHost`; both accepted on import; DLPack's legacy default stream is 0 for ROCm (1 for CUDA), applied to `__dlpack__`'s default and to the null-stream value `from_dlpack` passes to ROCm producers (passing CUDA's 1 on HIP would be an invalid stream pointer).
- `8d13d596` [ROCm] tests: test_capture_mode binds `hipFree` from the loaded HIP runtime on AMD GPUs (it loaded NVIDIA's libcudart from /usr/local/cuda and got error 35); HIP behaves as the tests expect (allowed under relaxed, rejected and capture invalidated under thread-local). test_cuda_arch_suffix registers on NVIDIA GPUs only (a/f suffixes are NVIDIA feature sets; on AMD the integer arch only collides with NVIDIA numbers; on AMD the suffix is ignored when compiling).

### Environment changes this session (job venv only)
- `torch[device-gfx950]==2.15.0+rocm10.2.0` from `https://stable.repo.amd.com/rocm/whl-next/` (with `--extra-index-url https://pypi.org/simple`). Without the `device-gfx950` extra torch's own kernels fail with hipErrorInvalidKernelFile ("Could not find device type specific kpack code objects", AMD_LOG_LEVEL=1).
- The venv had 25 stale `rocm-sdk-device-gfx*==10.1.0` packages from its creation; gfx950/gfx942/gfx90a upgraded to 10.2.0, the rest uninstalled, then `rocm-sdk init`.
- ROCm PyTorch's runtime aborts on `HIP_VISIBLE_DEVICES=-1` and on `CUDA_VISIBLE_DEVICES=-1` ("Sequence '-1' ... not recognized"); the CPU-only runner `agent_space/wptest_nogpu.sh` now hides GPUs with `HIP_VISIBLE_DEVICES=99 CUDA_VISIBLE_DEVICES=99`.

### Phase 3 test results (gfx950, one visible GPU, `-s autodetect -p <file> -m 32`)
test_graph 35 OK (3 skipped: multi-GPU/peer/mempool access, memory-usage flakiness); test_array_fill_capture 3 OK; test_capture_mode 2 OK; test_conditional_captures 32 run, OK, 16 skipped "Conditional graph nodes not supported" (existing upstream branch, Q4); test_conditional 50 OK; test_async 2373 OK; test_streams 24 OK (2 skipped multi-GPU); test_mempool 7 OK (2 skipped); test_multigpu 5 run, all skipped (one GPU); test_peer 4 OK (2 skipped); test_ipc 4 OK; test_pinned 1 OK; test_unified_memory 27 OK (8 skipped: 2 "cuda:0 cannot access CPU memory" (MI355X reports no pageable-memory access), rmm not installed, others device-count); test_occupancy 6 OK; test_allocator 16 OK (3 skipped); test_allocation_tracker 34 OK; test_dlpack 16 OK (8 without torch); test_torch 34 OK; test_cuda_arch_suffix 10 registered, all skipped on AMD; test_map 45 OK; test_apic 52 OK; test_apic_mesh 12 OK; test_reload 14 OK; test_mesh* 49 OK; test_sparse 47 OK; test_linear_solvers 34 OK. test_jax/test_paddle: frameworks not installed, nothing registered.

### CPU-only regression after Phase 3
Port (`8d13d596`), GPUs hidden: Ran 2282 tests (torch now installed adds CPU interop tests), 1 failure: `test_core.example_torch_cpu`, an artefact of the hiding values (ROCm torch aborts on -1); rerun with `HIP_VISIBLE_DEVICES=99 CUDA_VISIBLE_DEVICES=99` it passes on the port and on pristine. Pristine release-1.14 CPU-only full run in this environment took 819 s and reported 27 errors in test_constant_precision (`'Adjoint' object has no attribute 'return_var'`) plus the same artefact; that file passes 23/23 on pristine and on the port when run alone, so the pristine full-run errors are not attributable to the port (not investigated further). Phase 2's like-for-like comparison (2264/2264 OK, skipped 16 each) predates the torch install.

### CUDA no-regression at `8d13d596`
`build_lib.py --cuda-path /usr/local/cuda --no-use-libmathdx` (nvcc 12.8) succeeded; offline NVRTC compile of the tile/atomic module OK. `audit-commits` OK, `jargon.py --port` clean.

### Not done on this host (needs more than one visible GPU)
This job is pinned to `HIP_VISIBLE_DEVICES=0` (kept, per fleet rules). Plan Phase 3 items left: test_multigpu, multi-device test_peer/test_mempool/test_streams/test_graph cases, cross-device IPC, and plan risk 11 (graph mempool capture on ordinals > 0). These need a session dispatched with at least two GPUs on a wave64 host.

### Next step (Phase 4, start here)
1. Wave64 tile layer: the collectives were made wave64-correct in Phase 2 (`632a2edf`); run all of `tile/*` (22 files incl. test_tile_sort's 32-element case, test_tile_mathdx through the scalar paths, test_tile_shared_memory, test_tile_load/view, matmul/cholesky/solve/fft no-mathdx variants), test_block_dim_dispatch, test_template_launch_bounds on gfx950, and fix what fails. Check block_dim < 64 cases everywhere (tile_scan with num_warps 0, radix sort groups).
2. Q3 ruling: wave64 `tile_matmul` scalar-GEMM benchmark (16/32/64 tiles; fp32/fp64/fp16; forward and adjoint; the tile tests' block_dim values) on gfx950 here; the plan names gfx942 and a wave32 Radeon for the numbers that go to Jeff, which this host cannot provide. Record the script and raw timings; make no matrix-core decision.
3. Wave32: the Phase 1-3 sets and tile/* on a gfx1100/gfx1201 host (separate session).
4. Multi-GPU Phase 3 leftovers on a session with two or more GPUs.
5. Open for Jeff: unsafe atomics (Phase 1 notes), filing the three registered ROCm defects (rocPRIM sort, HIP occupancy, hipMallocAsync null-stream crash).

## Porting session 4 (2026-11-19, linux-gfx950 MI355X, porter): Phase 4 steps 1-2 on gfx950

Fork head unchanged at `8d13d596` (`release/1.14.0.amd0`): the whole tile set passed without a source change, so there is no new commit, no `advance-head` and no new nvcc run (the CUDA no-regression check at `8d13d596` from session 3 still applies). Stage stays `porting`, lock held by this host.

### Step 1: wave64 tile test set on gfx950 (all pass, no fork change)
Command per file (job venv, `HIP_VISIBLE_DEVICES=0`, `ulimit -c 0`, warp.so from `build_lib.py --rocm-path "$(rocm-sdk path --root)" --hip-arch gfx950`):
```
cd projects/warp-1.14/src
for f in $(cd warp/tests/tile && ls test_tile*.py) test_block_dim_dispatch.py test_template_launch_bounds.py; do
  python -m warp.tests -s autodetect -p "$f" -m 32
done
```
`-p` is a file-name pattern that `unittest` discovery applies in every subdirectory, so `test_tile_sort.py` finds `warp/tests/tile/test_tile_sort.py` (a directory-style pattern such as `tile/*` matches nothing).

| file | ran | result | file | ran | result |
|---|---|---|---|---|---|
| test_tile | 126 | OK | test_tile_matmul_no_mathdx | 22 | OK |
| test_tile_atomic_bitwise | 3 | OK | test_tile_mlp | 3 | OK, 1 skipped (a) |
| test_tile_cholesky | 33 | OK | test_tile_reduce | 62 | OK |
| test_tile_cholesky_no_mathdx | 16 | OK | test_tile_shared_memory | 30 | OK |
| test_tile_empty | 20 | OK | test_tile_solve | 5 | OK, 1 skipped (b) |
| test_tile_fft | 20 | OK, 8 skipped (c) | test_tile_solve_no_mathdx | 28 | OK |
| test_tile_fft_no_mathdx | 16 | OK | test_tile_sort | 2 | OK |
| test_tile_func_arg | 27 | OK | test_tile_stack | 40 | OK |
| test_tile_fused_ops | 36 | OK | test_tile_view | 16 | OK |
| test_tile_large_offsets | 1 | OK | test_block_dim_dispatch | 2 | OK |
| test_tile_load | 64 | OK | test_template_launch_bounds | 36 | OK |
| test_tile_mathdx | 6 | OK (scalar paths) | | | |
| test_tile_matmul | 22 | OK | | | |

Skips are upstream's own no-libmathdx rules, not port skips: (a) "Skipping test on CUDA device without MathDx (tolerance)" (test_multi_layer_nn); (b) "MathDx is not enabled" (singular-matrix solve); (c) "Warp was not built with MathDx support" for the 2D/3D `tile_fft`/`tile_ifft` vec2f/vec2d cases (the Q1 gap). Total 636 run, 10 skipped, 0 failures.

Coverage the plan asked for, checked rather than assumed: `git diff origin/release-1.14 -- warp/tests/tile warp/tests/test_block_dim_dispatch.py warp/tests/test_template_launch_bounds.py` is empty, so every file runs at upstream's sizes. test_tile_sort sweeps `block_dim` 2^5..2^9 against lengths 2^i+1 up to 2049 (int32/int64/uint64/float32), so the 32-thread case the earlier AMD tree skipped runs and passes. block_dim < 64 (fewer threads than one gfx950 wavefront) is exercised widely: 59 `block_dim=32` launches across tile/*, plus `block_dim=1` cases; all pass on the Phase 2 wave64 collectives (`632a2edf`).

### Step 2: Q3 `tile_matmul` benchmark, gfx950 (raw numbers for Jeff; no decision taken)
- What it times: the cooperative scalar GEMM (`partitioned_gemm::scalar_matmul`, `tile_matmul.h:164`), the only GPU GEMM path in the HIP build. Square MxM tiles, M in {16, 32, 64}; fp16, fp32, fp64; `block_dim` in {32, 64, 128, 256} (the matmul tests use 32 and 64; other tile tests 128); 16384 blocks (64 per CU on the 256-CU MI355X). "fwd": 32 accumulating `tile_matmul(a, b, c)` per block after one load, so loads and stores are amortized; GFLOP/s = 2*M^3*32*blocks/time. "adj": one `C = A @ B` per block, timed `tape.backward()` (plus the three `fill_`/`zero_` memsets that seed `C.grad` and clear `A.grad`/`B.grad`); GFLOP/s counts the two adjoint GEMMs only (the backward kernel also replays the forward GEMM, and `adj_tile_load` adds into the gradient arrays with `atomic_add`, `tile.h:3033`). Each row checks block 0 against NumPy (forward) and adj_A against `ones @ B0^T` (backward).
- Command, three runs: `python bench_tile_matmul.py --blocks 16384 --iters 50 --csv gfx950_run<n>.csv` (wrapped in `utils/timeit.sh warp-1.14 bench`), merged with `summarize.py`; both scripts are below. Run-to-run spread is 0-3%. An earlier pair of runs had a benchmark bug (the backward consumes `C.grad`, so replays after the first ran with a zero seed; all adj rows failed the check); those numbers were discarded and the script fixed before the three recorded runs.
- fp64 64x64 adjoint cannot launch on gfx950: the backward kernel needs 196608 bytes of shared memory and the device allows 163840 per block ("Failed to configure kernel dynamic shared memory ... 196608 bytes, but maximum available is 163840"; Warp then reports `hipErrorInvalidValue` on launch, and the gradient check catches the untouched gradient). That is a hardware limit (A, B, C and their three adjoints at 32 KiB each), not a port defect; its adj columns below are meaningless. On an NVIDIA part with 227 KiB opt-in shared memory the same kernel would launch.
- fp16 adjoint is 3-4x slower than fp32 at 16x16 and 32x32. A likely contributor is the gradient accumulation: fp16 `atomic_add` on HIP is the 32-bit CAS loop (Phase 1 notes, the unsafe-atomics question left for Jeff). Not measured further; recorded so the ruling can take it into account.
- No NVIDIA GPU on this host (`nvidia-smi` absent, no `/dev/nvidia*`), so there are no libmathdx/no-libmathdx reference numbers.

gfx950 (MI355X, HIP 7.17, ROCm 10.2.0 wheels), fork `8d13d596`, no libmathdx:

| dtype | tile | block_dim | fwd ms (min-max) | fwd GFLOP/s (best) | adj ms (min-max) | adj GFLOP/s (best) | check |
|---|---|---|---|---|---|---|---|
| float16 | 16x16 | 32 | 0.316-0.316 | 13602 | 0.287-0.287 | 936 | ok |
| float16 | 16x16 | 64 | 0.185-0.188 | 23241 | 0.334-0.335 | 803 | ok |
| float16 | 16x16 | 128 | 0.153-0.155 | 27985 | 0.313-0.314 | 857 | ok |
| float16 | 16x16 | 256 | 0.216-0.219 | 19877 | 0.308-0.308 | 872 | ok |
| float16 | 32x32 | 32 | 1.352-1.356 | 25415 | 1.475-1.486 | 1456 | ok |
| float16 | 32x32 | 64 | 0.645-0.648 | 53296 | 1.661-1.677 | 1293 | ok |
| float16 | 32x32 | 128 | 0.699-0.705 | 49149 | 2.050-2.113 | 1047 | ok |
| float16 | 32x32 | 256 | 0.890-0.899 | 38589 | 1.857-1.891 | 1157 | ok |
| float16 | 64x64 | 32 | 12.068-12.125 | 22777 | 12.879-12.969 | 1334 | ok |
| float16 | 64x64 | 64 | 6.462-6.480 | 42536 | 9.750-9.834 | 1762 | ok |
| float16 | 64x64 | 128 | 5.120-5.200 | 53684 | 6.889-6.909 | 2494 | ok |
| float16 | 64x64 | 256 | 4.644-4.649 | 59196 | 5.623-5.647 | 3056 | ok |
| float32 | 16x16 | 32 | 0.416-0.418 | 10313 | 0.155-0.159 | 1730 | ok |
| float32 | 16x16 | 64 | 0.344-0.345 | 12479 | 0.087-0.088 | 3072 | ok |
| float32 | 16x16 | 128 | 0.218-0.220 | 19696 | 0.082-0.083 | 3260 | ok |
| float32 | 16x16 | 256 | 0.270-0.272 | 15890 | 0.084-0.085 | 3182 | ok |
| float32 | 32x32 | 32 | 2.173-2.177 | 15811 | 1.115-1.116 | 1927 | ok |
| float32 | 32x32 | 64 | 1.401-1.405 | 24518 | 0.701-0.708 | 3064 | ok |
| float32 | 32x32 | 128 | 1.264-1.270 | 27181 | 0.681-0.698 | 3153 | ok |
| float32 | 32x32 | 256 | 1.470-1.474 | 23369 | 0.663-0.674 | 3237 | ok |
| float32 | 64x64 | 32 | 14.575-14.672 | 18860 | 19.338-19.426 | 888 | ok |
| float32 | 64x64 | 64 | 7.881-7.954 | 34877 | 9.341-9.483 | 1839 | ok |
| float32 | 64x64 | 128 | 7.124-7.126 | 38585 | 5.404-5.414 | 3179 | ok |
| float32 | 64x64 | 256 | 8.018-8.099 | 34281 | 3.404-3.472 | 5047 | ok |
| float64 | 16x16 | 32 | 0.523-0.524 | 8215 | 0.198-0.199 | 1353 | ok |
| float64 | 16x16 | 64 | 0.402-0.403 | 10696 | 0.120-0.121 | 2228 | ok |
| float64 | 16x16 | 128 | 0.386-0.391 | 11139 | 0.190-0.191 | 1409 | ok |
| float64 | 16x16 | 256 | 0.471-0.476 | 9119 | 0.190-0.191 | 1410 | ok |
| float64 | 32x32 | 32 | 3.110-3.114 | 11048 | 2.230-2.232 | 963 | ok |
| float64 | 32x32 | 64 | 2.247-2.257 | 15290 | 1.376-1.382 | 1561 | ok |
| float64 | 32x32 | 128 | 2.009-2.011 | 17106 | 1.089-1.106 | 1971 | ok |
| float64 | 32x32 | 256 | 2.821-2.827 | 12180 | 1.046-1.055 | 2053 | ok |
| float64 | 64x64 | 32 | 76.567-76.793 | 3590 | (no launch) | - | ADJ-FAILED (smem) |
| float64 | 64x64 | 64 | 40.331-40.375 | 6816 | (no launch) | - | ADJ-FAILED (smem) |
| float64 | 64x64 | 128 | 22.716-22.785 | 12100 | (no launch) | - | ADJ-FAILED (smem) |
| float64 | 64x64 | 256 | 20.865-20.954 | 13174 | (no launch) | - | ADJ-FAILED (smem) |

The Q3 ruling's hardware context, checked on this host with the ROCm 10.2 compiler (`<rocm>/lib/llvm/bin/clang -x hip --cuda-device-only -nogpulib -c`, probes in `agent_space/bench/mma/`), not assumed:
- MFMA: `__builtin_amdgcn_mfma_f32_16x16x4f32`, `mfma_f64_16x16x4f64`, `mfma_f32_16x16x16f16`, `mfma_f32_16x16x16bf16_1k` all compile for gfx90a, gfx942 and gfx950. So the Instinct matrix cores take fp16, bf16, fp32 and fp64 inputs, on 16x16 fragments per wavefront (32x32 and 4x4 variants also exist).
- WMMA: gfx1100 (`wmma_f32_16x16x16_f16_w32`) and gfx1201 (`..._w32_gfx12`) compile; the compiler's WMMA set on those targets is f16, bf16, int8/int4, plus fp8/bf8 on gfx12, and no fp32 or fp64 input form. The fp32-input `__builtin_amdgcn_wmma_f32_16x16x4_f32` is refused for gfx1100 and gfx1201 ("needs target feature gfx1250-insts,wavefrontsize32") and compiles for gfx1250. So "WMMA takes fp16/bf16/int8 only" holds for RDNA3/RDNA4, but not for gfx1250. Fragments are 16x16 on every target.
- cuBLASDx is **not** a SIMT-only GEMM, contrary to the ruling's working assumption: NVIDIA's documentation (https://docs.nvidia.com/cuda/cublasdx/index.html) lists "Static database dispatching to appropriate FMA / MMA instruction configuration", "LDSM / STSM MMA loading instructions", and from 0.5.0 "WGMMA / 1SM UTCMMA" instructions. That is, it selects tensor-core MMA where its database says so (the page does not list which precisions). So a CUDA build with libmathdx can run `tile_matmul` on tensor cores, where this HIP build always uses the scalar path.

`bench_tile_matmul.py` (as run; copy into a file to reproduce):
```python
# tile_matmul throughput: square MxM tiles, one tile GEMM chain per block.
# Usage: python bench_tile_matmul.py [--device cuda:0] [--mathdx on|off] [--csv out.csv]
# fwd:  REPS accumulating tile_matmul calls per block (C += A @ B), loads/stores amortized.
# adj:  one C = A @ B per block, timed tape.backward (adj_A = adj_C B^T, adj_B = A^T adj_C);
#       adj_gflops counts those 2 GEMMs (the backward kernel also replays the forward GEMM).
import argparse
import csv
import sys

import numpy as np

import warp as wp

p = argparse.ArgumentParser()
p.add_argument("--device", default="cuda:0")
p.add_argument("--mathdx", choices=["on", "off"], default="off")
p.add_argument("--csv", default=None)
p.add_argument("--blocks", type=int, default=4096)
p.add_argument("--reps", type=int, default=32)
p.add_argument("--iters", type=int, default=20)
args = p.parse_args()

wp.init()
wp.config.enable_mathdx_gemm = args.mathdx == "on"
dev = wp.get_device(args.device)

SIZES = (16, 32, 64)
DTYPES = (wp.float16, wp.float32, wp.float64)
BLOCK_DIMS = (32, 64, 128, 256)
REPS = args.reps


def make_kernels(M, dtype):
    @wp.kernel(enable_backward=False, module="unique")
    def fwd(A: wp.array3d(dtype=dtype), B: wp.array3d(dtype=dtype), C: wp.array3d(dtype=dtype)):
        i = wp.tid()
        a = wp.tile_load(A[i], shape=(M, M))
        b = wp.tile_load(B[i], shape=(M, M))
        c = wp.tile_zeros(shape=(M, M), dtype=dtype)
        for _r in range(REPS):
            wp.tile_matmul(a, b, c)
        wp.tile_store(C[i], c)

    @wp.kernel(module="unique")
    def one(A: wp.array3d(dtype=dtype), B: wp.array3d(dtype=dtype), C: wp.array3d(dtype=dtype)):
        i = wp.tid()
        a = wp.tile_load(A[i], shape=(M, M))
        b = wp.tile_load(B[i], shape=(M, M))
        c = wp.tile_matmul(a, b)
        wp.tile_store(C[i], c)

    return fwd, one


def timed(fn, iters):
    fn()  # warm-up, includes module load
    wp.synchronize_device(dev)
    with wp.ScopedDevice(dev):
        s = wp.Event(enable_timing=True)
        e = wp.Event(enable_timing=True)
        wp.record_event(s)
        for _ in range(iters):
            fn()
        wp.record_event(e)
        wp.synchronize_device(dev)
    return wp.get_event_elapsed_time(s, e) / iters  # ms


rows = []
rng = np.random.default_rng(0)
print(f"device={dev.name} arch={dev.arch} mathdx={args.mathdx} blocks={args.blocks} reps={REPS}", flush=True)
print("dtype,M,block_dim,fwd_ms,fwd_gflops,adj_ms,adj_gflops,check", flush=True)
for dtype in DTYPES:
    for M in SIZES:
        fwd, one = make_kernels(M, dtype)
        npt = wp.dtype_to_numpy(dtype)
        A = wp.array(rng.uniform(-1, 1, (args.blocks, M, M)).astype(npt), dtype=dtype, device=dev, requires_grad=True)
        B = wp.array(rng.uniform(-1, 1, (args.blocks, M, M)).astype(npt), dtype=dtype, device=dev, requires_grad=True)
        C = wp.zeros((args.blocks, M, M), dtype=dtype, device=dev, requires_grad=True)
        for bd in BLOCK_DIMS:
            def run_fwd():
                wp.launch_tiled(fwd, dim=[args.blocks], inputs=[A, B, C], block_dim=bd, device=dev)

            fwd_ms = timed(run_fwd, args.iters)
            # correctness on block 0: C = REPS * A0 @ B0
            ref = REPS * (A.numpy()[0].astype(np.float64) @ B.numpy()[0].astype(np.float64))
            got = C.numpy()[0].astype(np.float64)
            tol = {wp.float16: 5e-2, wp.float32: 1e-4, wp.float64: 1e-10}[dtype]
            ok = np.allclose(got, ref, rtol=tol, atol=tol * REPS * M)

            tape = wp.Tape()
            with tape:
                wp.launch_tiled(one, dim=[args.blocks], inputs=[A, B, C], block_dim=bd, device=dev)
            def run_adj():
                # the backward pass consumes (zeroes) adj_C, so seed it on every replay
                C.grad.fill_(1.0)
                A.grad.zero_()
                B.grad.zero_()
                tape.backward()

            adj_ms = timed(run_adj, args.iters)
            # adj_A[0] = adj_C @ B0^T with adj_C = ones; a failed launch leaves it zero
            gref = np.ones((M, M)) @ B.numpy()[0].astype(np.float64).T
            gok = np.allclose(A.grad.numpy()[0].astype(np.float64), gref, rtol=tol, atol=tol * M)
            fl = 2.0 * M * M * M * args.blocks
            check = "ok" if ok and gok else ("MISMATCH" if not ok else "ADJ-FAILED")
            r = (dtype.__name__, M, bd, fwd_ms, fl * REPS / fwd_ms / 1e6, adj_ms, 2 * fl / adj_ms / 1e6, check)
            rows.append(r)
            print("%s,%d,%d,%.4f,%.1f,%.4f,%.1f,%s" % r, flush=True)

if args.csv:
    with open(args.csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dtype", "M", "block_dim", "fwd_ms", "fwd_gflops", "adj_ms", "adj_gflops", "check"])
        w.writerows(rows)
sys.exit(0 if all(r[-1] == "ok" for r in rows) else 1)
```
The script exits 1 whenever any row is not `ok`, so on gfx950 it always exits 1 because of the four fp64 64x64 rows; read the CSV, not the exit code.

`summarize.py`:
```python
# Merge bench_tile_matmul.py CSVs from repeated runs into a Markdown table (min and max per row).
# Usage: python summarize.py run1.csv run2.csv ...
import csv
import sys
from collections import defaultdict

rows = defaultdict(list)
order = []
for path in sys.argv[1:]:
    with open(path) as f:
        for r in csv.DictReader(f):
            key = (r["dtype"], int(r["M"]), int(r["block_dim"]))
            if key not in rows:
                order.append(key)
            rows[key].append(r)

print("| dtype | tile | block_dim | fwd ms (min-max) | fwd GFLOP/s (best) | adj ms (min-max) | adj GFLOP/s (best) | check |")
print("|---|---|---|---|---|---|---|---|")
for key in order:
    rs = rows[key]
    fwd = [float(r["fwd_ms"]) for r in rs]
    adj = [float(r["adj_ms"]) for r in rs]
    fg = max(float(r["fwd_gflops"]) for r in rs)
    ag = max(float(r["adj_gflops"]) for r in rs)
    checks = sorted({r["check"] for r in rs})
    print(
        f"| {key[0]} | {key[1]}x{key[1]} | {key[2]} | {min(fwd):.3f}-{max(fwd):.3f} | {fg:.0f} "
        f"| {min(adj):.3f}-{max(adj):.3f} | {ag:.0f} | {'/'.join(checks)} |"
    )
```

### Next step (handoff: what each remaining host runs, from these notes alone)
The project stays in `porting` with the work lock held by linux-gfx950. The remaining Phase 4 work runs on other hardware, and the plan treats it as porting, not validation. So a session on another host needs the lock moved to it first. That is a person's decision (AGENTS.md: takeover is never automatic), or this host releases it when Jeff says so. Nothing below needs a source change unless a test fails; if one does, that host fixes it on `release/1.14.0.amd0`, compiles the CUDA path with nvcc on the mi355 host, and records `advance-head`.

**Common setup on every host** (Linux, TheRock ROCm 10.2 wheels, a job-private venv with Python 3.11; replace `<gfx>` with the host's target, e.g. gfx942, gfx1100 or gfx1201):
```
cd <moat clone> && git checkout port/warp-1.14 && git pull --ff-only && bash utils/orient.sh
git clone https://github.com/AMD-Ecosystem/warp projects/warp-1.14/src   # skip if present (fork_url in status.json)
python3 utils/moatlib.py protect-fork warp-1.14
cd projects/warp-1.14/src
git fetch origin && git checkout release/1.14.0.amd0 && git pull --ff-only
git rev-parse HEAD                            # expect 8d13d596e6787841adccfc47d75406ef6bcf460a (or a later recorded head_sha)
git lfs install --local --skip-repo && git lfs pull
python -m pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ --extra-index-url https://pypi.org/simple \
    "rocm[devel,device-<gfx>]==10.2.0" "torch[device-<gfx>]==2.15.0+rocm10.2.0" numpy usd-core
rocm-sdk init
python build_lib.py --rocm-path "$(rocm-sdk path --root)" --hip-arch <gfx> -j "$MAX_JOBS"
ulimit -c 0
python -c "import warp as wp; wp.init()"      # greeting must show HIP 7.17 and the <gfx> device
```
Without the `device-<gfx>` extra torch's own kernels fail with hipErrorInvalidKernelFile (session 3). For CPU-only runs hide GPUs with `HIP_VISIBLE_DEVICES=99 CUDA_VISIBLE_DEVICES=99`, never `-1` (ROCm torch aborts on it). Copy `bench_tile_matmul.py` and `summarize.py` from the blocks above into a scratch directory. Every command below runs from `projects/warp-1.14/src` with one GPU pinned unless stated.

The tile set, used by **A** and **B**:
```
for f in $(cd warp/tests/tile && ls test_tile*.py) test_block_dim_dispatch.py test_template_launch_bounds.py; do
  python -m warp.tests -s autodetect -p "$f" -m 32
done
```
Expected on gfx950: 636 run, 10 skipped (the three upstream no-libmathdx skips listed in step 1), 0 failures. Any other skip or a failure is a finding.

The benchmark, used by **A** and **B**:
```
for n in 1 2 3; do python <scratch>/bench_tile_matmul.py --blocks 16384 --iters 50 --csv <host>_run$n.csv; done
python <scratch>/summarize.py <host>_run1.csv <host>_run2.csv <host>_run3.csv
```
Record the table, the device name and CU count, and the run-to-run spread. Expect `ADJ-FAILED` only where the backward kernel's shared memory exceeds the device's per-block limit (the Warp UserWarning names the byte counts; 64 KiB-LDS parts will lose more rows than gfx950); anything else is a finding. No matrix-core decision on any host: the numbers go to Jeff.

**A. gfx942 host (MI300X/MI325X), wave64 (plan's Phase 4 primary target):**
1. Common setup with `<gfx>` = gfx942.
2. The tile set above; record the per-file table as in step 1.
3. The benchmark above (the Q3 numbers the plan asks for on gfx942).
4. Optionally (useful for the Q2 ruling, not required by Phase 4): the Phase 1 atomic timings the notes deferred; skip unless Jeff asks.

**B. wave32 Radeon host (gfx1100/gfx1101 or gfx1201), closes the wave32 gate:**
1. Common setup with `<gfx>` = the host's target. Confirm `python -c "import warp as wp; wp.init(); print(wp.get_device('cuda:0').arch_name)"` names the gfx11/gfx12 part. `WP_TILE_WARP_SIZE` is chosen at compile time from the target (`tile_reduce.h:15`: 64 when `__GFX9__`, else 32), so wave32 needs no build switch; a gfx10+ part compiled in wave64 mode would break that assumption, which is why this run matters.
2. The tile set above (the first time the tile collectives run at wavefront 32).
3. The benchmark above (the Q3 wave32 numbers).
4. The Phase 1-3 sets at wavefront 32, one `-p` pattern per line:
```
for p in test_codegen.py test_snippet.py test_arithmetic.py test_math.py test_fp16.py test_bf16.py test_atomic.py \
    test_atomic_bitwise.py test_atomic_cas.py test_array.py test_copy.py test_launch.py test_fast_math.py test_grad.py \
    test_kernel_cache.py test_module_hashing.py test_print.py 'test_mat*.py' 'test_vec*.py' 'test_quat*.py' \
    'test_spatial*.py' test_rand.py test_noise.py test_clang_cuda.py \
    test_runlength_encode.py test_array_reduce.py test_sparse.py test_linear_solvers.py test_coloring.py \
    test_bvh.py test_grouped_bvh.py test_hash_grid.py test_marching_cubes.py 'test_mesh*.py' test_volume.py \
    test_volume_write.py test_apic.py test_apic_mesh.py 'test_fem*.py' \
    test_graph.py test_array_fill_capture.py test_capture_mode.py test_conditional_captures.py test_conditional.py \
    test_async.py test_streams.py test_mempool.py test_pinned.py test_unified_memory.py test_occupancy.py \
    test_allocator.py test_allocation_tracker.py test_dlpack.py test_torch.py test_map.py test_reload.py \
    test_cuda_arch_suffix.py; do
  python -m warp.tests -s autodetect -p "$p" -m 32
done
```
   Compare with the gfx950 counts in the Phase 1, 2 and 3 result sections above (same files, same patterns); skips should differ only where a wave64-only path or a device property differs, and each difference needs a reason in the notes. Known wave32 differences to expect: hardware atomics differ per the Phase 1 ISA table (gfx1100 CAS for f32/f64 add; gfx1201 hardware f32 add), and `test_unified_memory`'s pageable-memory cases depend on the part.
5. The CPU-only suite with GPUs hidden (`HIP_VISIBLE_DEVICES=99 CUDA_VISIBLE_DEVICES=99 python -m warp.tests -s autodetect -k cpu -m 32`): expect 2282 run with torch installed, 0 failures (session 3).

**C. Two-GPU session on a wave64 host (gfx950 or gfx942), Phase 3 leftovers and plan risk 11:**
1. Dispatch with two GPUs, e.g. `HIP_VISIBLE_DEVICES=0,1` (keep whatever pin the dispatcher sets; it must expose two devices of the same arch). Common setup with that `<gfx>`; confirm `wp.get_cuda_devices()` lists `cuda:0` and `cuda:1`.
2. With two identical devices Warp's default test mode (`unique_or_2x`, `unittest_utils.py:28`) adds `cuda:1` to every device test, so the multi-GPU cases and the non-primary-ordinal captures run without editing tests:
```
for p in test_multigpu.py test_peer.py test_mempool.py test_streams.py test_graph.py test_ipc.py test_unified_memory.py \
    test_allocator.py test_async.py test_sparse.py test_fem_integrate.py test_array.py test_copy.py; do
  python -m warp.tests -s autodetect -p "$p" -m 32
done
```
   Risk 11 (the earlier AMD tree restricted the in-capture mempool tests of `test_sparse.py` and `fem/test_fem_integrate.py` to ordinal 0, hiding a fault on non-primary devices): the `_cuda_1` variants of the capture tests in those two files must pass. A failure there is the fault the plan asks to reproduce; fix it or register it with `utils/deferred.py add rocm-bug-report --project warp-1.14`.
3. Expected from gfx950 single-GPU runs: test_multigpu 5 registered (all skipped there), test_peer 2 skipped, test_mempool 2 skipped, test_streams 2 skipped, test_graph 3 skipped. With two GPUs the device-count skips should disappear; peer-access skips may remain if the two GPUs report no peer access (record `wp.is_peer_access_supported("cuda:0", "cuda:1")`).

**Still open for Jeff (unchanged):** unsafe atomics (Phase 1 notes; the fp16 adjoint numbers above are relevant), the Q3 matrix-core ruling once A and B have run, filing the three registered ROCm defects (rocPRIM radix sort, HIP occupancy launch bounds, hipMallocAsync null stream during capture).

## Ruling on the unsafe atomic forms (2026-11-18, Jeff Daily)

RULED: option (a), keep the safe forms. fp32/fp64 `atomic_add` stay HIP's `atomicAdd`, fp16/bf16 `atomic_add` stay the 32-bit `atomicCAS` on the containing word; correct for every memory kind, slow where the measurement table says CAS. `unsafeAtomicAdd` and the packed fp16/bf16 hardware add are not adopted in this line, and no module option is added. This closes the item the porter stopped on in the Q2 atomics section and supersedes the "prefer hardware atomics" part of the amended Q2 ruling in plan.md where the two conflict. Release notes say that fp16/bf16 atomic adds, and fp32/fp64 atomic adds on targets outside gfx942/gfx950 (fp32 also gfx1201), are compare-and-swap loops.

For reference, what AMD's `release/1.13.0.amd0` branch does (read from its `warp/native/builtin.h`, not built or run): fp16 `atomic_add` calls `unsafeAtomicAdd(__half*, ...)` unconditionally on HIP, with no memory-kind condition; fp32 and fp64 use plain `atomicAdd`; float/double min and max keep upstream's `atomicCAS` loops; bf16 min and max use a 16-bit `atomicCAS` loop; bf16 `atomic_add` has no HIP branch and that tree never defines `__CUDA_ARCH__`, so on a HIP device compile it takes the host branch, a plain non-atomic read-add-write.

## Ruling on test_atomic_cas (2026-11-18, Jeff Daily), and work it adds for the porter

RULED: keep the narrowing of the spin-lock test to devices with independent thread scheduling, and add a small AMD-capable test of `atomic_cas` itself (many threads compete once on one cell, exactly one wins; also `atomic_exch`), run on every device, so the operations are covered on AMD GPUs.

Who depends on the spin-lock behaviour (asked by Jeff; `git grep atomic_cas|atomic_exch` on release-1.14 outside tests and docs): no library code. `atomic_cas`/`atomic_exch` are user-facing builtins added in eaa3cf3d ("Expose atomic cas and and atomic exch GH-767"); the only callers in the tree are `warp/tests/test_atomic_cas.py` and two examples, neither registered in the test suite: `warp/examples/core/example_spin_lock.py` (the same `while atomic_cas(lock,0,0,1) == 1: pass` spin lock, so it is expected to hang on AMD GPUs) and `warp/examples/core/example_work_queue.py` (a lock-free queue: CAS retry loops where no thread waits on another to release, so it is expected to make progress; not yet run).

Porter, next session: (1) add the AMD-capable `atomic_cas`/`atomic_exch` test; (2) run `example_work_queue.py` on an AMD GPU under a timeout and record the result; (3) make `example_spin_lock.py` end with a clear message on a GPU without independent thread scheduling instead of hanging, using the same capability the test uses; (4) release notes: spin locks contended inside one wavefront hang on AMD GPUs, and `example_spin_lock.py` is the in-tree case.

## Evidence session (2026-11-19, linux-gfx942, 2x MI300X, evidence only): handoff items A and C

Requested by Jeff Daily. The work lock stays with linux-gfx950. This session changed nothing in the fork, stage, lock, head or platform records. The clone was read-only at the recorded `head_sha` `8d13d596`, and local experiments in it were reverted (`git status --porcelain` clean at the end). Where a fix is proposed below, it is a diff for the lock holder to judge.

### Setup (deviations from the common setup block)
- Host: 2x "AMD Instinct MI300X HF" (192 GiB each, `gfx942:sramecc+:xnack-`, 304 CUs), dispatched with `HIP_VISIBLE_DEVICES=0,1`. Items A ran with `HIP_VISIBLE_DEVICES=0` (one GPU, as on gfx950) and item C with `0,1`. No system ROCm on the loader path.
- The job venv came with TheRock 10.1.0 wheels. Installed into it: `rocm[devel,device-gfx942]==10.2.0`, `torch[device-gfx942]==2.15.0+rocm10.2.0`, `numpy==1.26.4` and `usd-core` (index `https://stable.repo.amd.com/rocm/whl-next/`, extra index PyPI), then `rocm-sdk init`. The stale `rocm-sdk-device-gfx*==10.1.0` packages remain visible through the job's read-only `venv-base-links` path (pip cannot uninstall them there). Warp uses none of the ROCm math libraries, so they do not matter here. HIP 7.17.26434, torch 2.15.0+rocm10.2.0, Python 3.11.17.
- `git-lfs` is not installed on this host, so `git lfs pull` was skipped. None of the files run here read LFS assets.
- Build: `python build_lib.py --rocm-path "$(rocm-sdk path --root)" --hip-arch gfx942 -j 112` (112 = physical cores) -> "Warp build succeeded", 30.5 s at 56 workers. Greeting: `HIP 7.17`, both devices listed, "CUDA peer access: Supported fully (all-directional)". `wp.is_peer_access_supported("cuda:0", "cuda:1")` = True.
- Every run used a private `WARP_CACHE_PATH` under `agent_space/`, because the test runner clears its kernel cache at start and the default `~/.cache/warp` is shared with other jobs on the host. All GPU runs were under `ulimit -c 0`.

### A.2: tile set on gfx942 (one GPU): 636 run, 10 skipped, 4 failures
Same loop as step 1 of session 4 (`-p <file> -m 32`, from `projects/warp-1.14/src`).
## Evidence session (2026-11-19, linux-gfx1100, Radeon Pro W7800, evidence only): handoff item B (wave32)

Requested by Jeff Daily. The work lock stays with linux-gfx950. This session changed nothing in the fork, stage, lock, head or platform records. The clone was read-only at the recorded `head_sha` `8d13d596` (`git status --porcelain` clean apart from untracked test output). Local experiments ran in a separate `git worktree` of the clone, outside `projects/warp-1.14/src`. That way the warp.so the tests used was never rebuilt mid-run. This is the first run of the port on a wave32 GPU.

### Setup (deviations from the common setup block)
- Device: "AMD Radeon Pro W7800 48GB", `gfx1100` (no target features in `arch_name`), Warp `arch` 110, `sm_count` 35 (HIP's `multiProcessorCount`, which counts WGPs on RDNA3; the part has 70 CUs), warp size 32 (`wp_cuda_device_get_warp_size`), 64 KiB LDS per workgroup (`hipDeviceAttributeMaxSharedMemoryPerBlock` = 65536). Dispatcher pin `HIP_VISIBLE_DEVICES=0` kept. The host has four such GPUs.
- The job venv came with TheRock 10.1.0 wheels. Installed into it: `rocm[devel,device-gfx1100]==10.2.0`, `torch[device-gfx1100]==2.15.0+rocm10.2.0`, `numpy==1.26.4`, `usd-core`, then `rocm-sdk init`. As on gfx942, the stale `rocm-sdk-device-gfx*==10.1.0` entries stay visible through the read-only `venv-base-links` path. The venv's own 10.2 `rocm_sdk`/`_rocm_sdk_core` shadow them (checked: `rocm_sdk.__file__` is in the venv, HIP 7.17.26434).
- Trap seen here (it cost one experiment): `venv-base-links` also puts the image's `rocm-sdk` 10.1 console script on the path when the venv's `bin` is not first on `PATH`. In `cd x && PATH=$FLEET_VENV/bin:$PATH python build_lib.py --rocm-path "$(rocm-sdk path --root)"`, the `$(...)` runs before the `PATH=` prefix applies. It returned `/opt/python-3.11-venv/.../_rocm_sdk_devel` (10.1) and the build linked the 10.1 runtime. `export PATH=$FLEET_VENV/bin:$PATH` first. Check the result with `readelf -d warp/bin/warp.so | grep RUNPATH`. The build used for all results below has the venv's 10.2 RUNPATH.
- `git-lfs` 3.4.1 installed with apt, then `git lfs install --local --skip-repo && git lfs pull` (test_volume reads the `.nvdb` assets).
- Build: `export PATH=$FLEET_VENV/bin:$PATH; python build_lib.py --rocm-path "$(rocm-sdk path --root)" --hip-arch gfx1100 -j 32` (32 = physical cores) -> "Warp build succeeded", 33 s (timeit). Greeting: `HIP 7.17`, `"cuda:0" : "AMD Radeon Pro W7800 48GB" (45 GiB, gfx1100, mempool enabled)`.
- Every run used a private `WARP_CACHE_PATH` under `agent_space/`. All GPU runs were under `ulimit -c 0`. Runner per file: `python -m warp.tests -s autodetect -p "<pattern>" -m 32` from `projects/warp-1.14/src`.

### B.2: tile set at wavefront 32: 636 run, 10 skipped, 4 failures (the same 4 as gfx942, not wave-related)

| file | ran | result | file | ran | result |
|---|---|---|---|---|---|
| test_tile | 126 | OK | test_tile_matmul_no_mathdx | 22 | OK |
| test_tile_atomic_bitwise | 3 | OK | test_tile_mlp | 3 | OK, 1 skipped (a) |
| test_tile_cholesky | 33 | OK | test_tile_reduce | 62 | **FAILED (failures=2)** |
| test_tile_cholesky_no_mathdx | 16 | OK | test_tile_shared_memory | 30 | **FAILED (failures=1)** |
| test_tile_empty | 20 | OK | test_tile_solve | 5 | OK, 1 skipped (b) |
| test_tile_fft | 20 | OK, 8 skipped (c) | test_tile_solve_no_mathdx | 28 | OK |
| test_tile_fft_no_mathdx | 16 | OK | test_tile_sort | 2 | OK |
| test_tile_func_arg | 27 | OK | test_tile_stack | 40 | OK |
| test_tile_fused_ops | 36 | OK | test_tile_view | 16 | **FAILED (failures=1)** |
| test_tile_large_offsets | 1 | OK | test_block_dim_dispatch | 2 | OK |
| test_tile_load | 64 | OK | test_template_launch_bounds | 36 | OK |
| test_tile_mathdx | 6 | OK | | | |
| test_tile_matmul | 22 | OK | | | |

The skips are the same three upstream no-libmathdx skips as on gfx950: (a), (b), (c).

The four failures: `test_tile_reduce_axis_tier2_cuda_0`, `test_tile_reduce_axis_tier2_single_warp_cuda_0`, `test_tile_shared_mem_large_cuda_0`, `test_tile_assign_2d_cuda_0`. In each one, the output or gradient array stays all zeros, and the log has exactly one `Warp CUDA error 1: invalid argument (in function wp_cuda_launch_kernel, warp.cu:5466)` per failing test.

Cause, measured: each failing kernel asks for exactly 65536 bytes of dynamic shared memory. Per module meta: tier2 sum-axis2 backward 65536, assign_2d backward 65536, shared_mem_large forward 65536. On top of that it carries static LDS. `llvm-readobj --notes` on the cached code objects gives `group_segment_fixed_size` of 1024 (shared_mem_large forward), 392 and 520 (tier2 sum-axis2 backward at the two block_dims) and 128 (assign_2d backward). gfx942 allows 65536 bytes of LDS per workgroup in total (`hipDeviceAttributeMaxSharedMemoryPerBlock`), so 65536 + static > limit, and `hipModuleLaunchKernel` returns `hipErrorInvalidValue`. The 1024 static bytes in shared_mem_large are Warp's own per-thread allocator offsets, `__shared__ unsigned int smem_base[WP_TILE_BLOCK_DIM]` (`tile.h:1143`, 256 x 4 B). Upstream has the same array on CUDA, where the opt-in limit (up to 227 KiB) leaves room for it. gfx950 has 160 KiB, which is why these four pass there. This is the same hardware-capacity class as the fp64 64x64 adjoint row on gfx950, not a miscompile. Upstream's test comment says as much: "set dimensions that require 64kb for the forward kernel ... checks that we can configure shared memory > 48kb default".

Port defect found while diagnosing it: Warp printed no "Failed to configure kernel dynamic shared memory" warning for these kernels. On gfx950 it does print one for the fp64 adjoint. The HIP emulation of `cuFuncSetAttribute_f(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES)` (`cuda_util.cpp:410-428`) compares the requested dynamic size alone against the per-block limit. CUDA's `cuFuncSetAttribute` fails when dynamic + static exceeds the limit, so the emulation lets these kernels through to a launch failure. Proposed fix, tested as an uncommitted local experiment: with it, the shared_mem_large probe prints `Warp UserWarning: Failed to configure kernel dynamic shared memory ... for 65536 bytes, but maximum available is 65536` before the launch error. The kernel still cannot run, by hardware limit.
```diff
--- a/warp/native/cuda_util.cpp
+++ b/warp/native/cuda_util.cpp
@@ -409,19 +409,20 @@ CUresult cuModuleGetGlobal_f(CUdeviceptr* dptr, size_t* bytes, CUmodule hmod, co
 
 CUresult cuFuncSetAttribute_f(CUfunction hfunc, CUfunction_attribute attrib, int value)
 {
-    (void)hfunc;
-
     // AMD GPUs have no opt-in for dynamic shared memory: a kernel may use up to the
-    // per-block limit without configuration
+    // per-block limit, shared with its static shared memory, without configuration
     if (attrib == CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES) {
         hipDevice_t device;
         int max_smem_bytes = 0;
+        int static_smem_bytes = 0;
         hipError_t result = hipGetDevice(&device);
         if (result == hipSuccess)
             result = hipDeviceGetAttribute(&max_smem_bytes, hipDeviceAttributeMaxSharedMemoryPerBlock, device);
+        if (result == hipSuccess)
+            result = hipFuncGetAttribute(&static_smem_bytes, HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, hfunc);
         if (result != hipSuccess)
             return result;
-        return value <= max_smem_bytes ? hipSuccess : hipErrorInvalidValue;
+        return value <= max_smem_bytes - static_smem_bytes ? hipSuccess : hipErrorInvalidValue;
     }
 
     return hipErrorNotSupported;
```
Open for the lock holder or Jeff, not decided here: the four tests cannot pass on any 64 KiB-LDS part (gfx90a and gfx942 among the wave64 targets; RDNA parts too). Options include a capability-based skip when the kernel's dynamic + static shared memory exceeds `device.max_shared_memory_per_block`, or keeping them as known failures on 64 KiB parts. No test was changed.

### A.3: Q3 `tile_matmul` benchmark on gfx942 (raw numbers for Jeff; no decision taken)
Device "AMD Instinct MI300X HF", 304 CUs, HIP 7.17, ROCm 10.2.0 wheels, fork `8d13d596`, no libmathdx, `HIP_VISIBLE_DEVICES=0`. Scripts: `bench_tile_matmul.py` and `summarize.py` exactly as in session 4, extracted verbatim from these notes. Command, wrapped in `utils/timeit.sh warp-1.14 bench` (160 s for the three runs):
```
for n in 1 2 3; do python bench_tile_matmul.py --blocks 16384 --iters 50 --csv gfx942_run$n.csv; done
python summarize.py gfx942_run1.csv gfx942_run2.csv gfx942_run3.csv
```
16384 blocks is the same count as on gfx950, for comparability. On this 304-CU part that is about 54 blocks per CU, against 64 on gfx950.

| dtype | tile | block_dim | fwd ms (min-max) | fwd GFLOP/s (best) | adj ms (min-max) | adj GFLOP/s (best) | check |
|---|---|---|---|---|---|---|---|
| float16 | 16x16 | 32 | 0.324-0.325 | 13261 | 0.340-0.347 | 790 | ok |
| float16 | 16x16 | 64 | 0.190-0.194 | 22593 | 0.350-0.370 | 768 | ok |
| float16 | 16x16 | 128 | 0.183-0.187 | 23468 | 0.301-0.323 | 891 | ok |
| float16 | 16x16 | 256 | 0.239-0.242 | 17998 | 0.292-0.313 | 918 | ok |
| float16 | 32x32 | 32 | 2.126-2.130 | 16159 | 2.190-2.195 | 980 | ok |
| float16 | 32x32 | 64 | 0.861-0.869 | 39898 | 1.940-1.955 | 1107 | ok |
| float16 | 32x32 | 128 | 0.938-0.963 | 36642 | 2.077-2.094 | 1034 | ok |
| float16 | 32x32 | 256 | 1.121-1.141 | 30664 | 1.831-1.842 | 1173 | ok |
| float16 | 64x64 | 32 | 25.694-25.699 | 10698 | 29.794-29.832 | 577 | ok |
| float16 | 64x64 | 64 | 13.584-13.592 | 20235 | 19.028-19.040 | 903 | ok |
| float16 | 64x64 | 128 | 9.159-9.198 | 30010 | 11.452-11.457 | 1500 | ok |
| float16 | 64x64 | 256 | 6.317-6.352 | 43517 | 7.818-7.829 | 2197 | ok |
| float32 | 16x16 | 32 | 0.416-0.417 | 10316 | 0.253-0.269 | 1061 | ok |
| float32 | 16x16 | 64 | 0.341-0.345 | 12582 | 0.124-0.138 | 2161 | ok |
| float32 | 16x16 | 128 | 0.231-0.233 | 18616 | 0.092-0.099 | 2908 | ok |
| float32 | 16x16 | 256 | 0.276-0.280 | 15544 | 0.091-0.099 | 2941 | ok |
| float32 | 32x32 | 32 | 3.125-3.141 | 10995 | 2.236-2.256 | 960 | ok |
| float32 | 32x32 | 64 | 1.543-1.559 | 22270 | 1.275-1.467 | 1684 | ok |
| float32 | 32x32 | 128 | 1.421-1.435 | 24175 | 1.132-1.142 | 1897 | ok |
| float32 | 32x32 | 256 | 1.579-1.604 | 21766 | 0.883-0.896 | 2433 | ok |
| float32 | 64x64 | 32 | 38.033-38.039 | 7227 | (no launch) | - | ADJ-FAILED (smem) |
| float32 | 64x64 | 64 | 18.730-18.734 | 14676 | (no launch) | - | ADJ-FAILED (smem) |
| float32 | 64x64 | 128 | 13.437-13.448 | 20456 | (no launch) | - | ADJ-FAILED (smem) |
| float32 | 64x64 | 256 | 9.886-9.950 | 27805 | (no launch) | - | ADJ-FAILED (smem) |
| float64 | 16x16 | 32 | 0.588-0.589 | 7305 | 0.321-0.330 | 835 | ok |
| float64 | 16x16 | 64 | 0.442-0.450 | 9724 | 0.166-0.330 | 1613 | ok |
| float64 | 16x16 | 128 | 0.437-0.449 | 9819 | 0.209-0.214 | 1281 | ok |
| float64 | 16x16 | 256 | 0.527-0.536 | 8143 | 0.209-0.213 | 1286 | ok |
| float64 | 32x32 | 32 | 5.952-5.958 | 5773 | 4.579-4.584 | 469 | ok |
| float64 | 32x32 | 64 | 3.241-3.244 | 10602 | 2.725-2.740 | 788 | ok |
| float64 | 32x32 | 128 | 2.708-2.729 | 12687 | 1.919-1.942 | 1119 | ok |
| float64 | 32x32 | 256 | 3.087-3.113 | 11132 | 1.511-1.526 | 1421 | ok |
Skips: the three upstream no-libmathdx skips (a), (b), (c) of session 4, nothing else.

The four failures are exactly the gfx942 four (`test_tile_reduce_axis_tier2_cuda_0`, `test_tile_reduce_axis_tier2_single_warp_cuda_0`, `test_tile_shared_mem_large_cuda_0`, `test_tile_assign_2d_cuda_0`), with the same signature: one `Warp CUDA error 1: invalid argument (in function wp_cuda_launch_kernel, warp.cu:5466)` per test, and an all-zero output or gradient. The cause is the one the gfx942 section measured: the kernel asks for 65536 bytes of dynamic LDS, and the 64 KiB per-workgroup limit leaves no room for its static LDS. Measured here: `test_tile_shared_mem_large`'s forward kernel has `.group_segment_fixed_size: 1024` (`llvm-readelf --notes` on the cached `.hsaco`). Taken from the hipRTC LLVM IR (`-fgpu-rdc` bitcode through `llvm-dis`), those 1024 bytes are the only static LDS global: `tile_shared_storage_t::get_smem_base()::smem_base`, upstream's per-thread allocator offset table `__shared__ unsigned int smem_base[WP_TILE_BLOCK_DIM]` (`tile.h:1143`, 4 B x block_dim 256). It is upstream code and is present on CUDA too. So the same four tests would also fail on an NVIDIA part whose per-block limit is 64 KiB. The probe `agent_space/probe/smem_large.py` with tile `64x120` (61440 dynamic bytes) passes; `64x128` (65536) fails.

Wave32 itself is not the problem. The same kernels at half size, forward and backward at wavefront 32, all correct (`agent_space/probe/half_lds.py`):

| kernel | block_dim | backward dyn. LDS | forward | gradient |
|---|---|---|---|---|
| tile_sum axis 2, (8,8,128) as in the test | 64 / 32 | 65536 | ok / ok | launch fails / launch fails |
| tile_sum axis 2, (8,8,64) | 64 / 32 | 32768 | ok / ok | ok / ok |
| tile_assign, (16,32,8) as in the test | 32 | 65536 | ok | launch fails |
| tile_assign, (16,32,4) | 32 | 32768 | ok | ok |

The `cuFuncSetAttribute_f` emulation diff proposed in the gfx942 section was confirmed here too, as an uncommitted experiment in a separate worktree built against the venv's 10.2. With it, all four kernels print Warp's own "Failed to configure kernel dynamic shared memory ... 65536 bytes, but maximum available is 65536" UserWarning before the (still failing) launch, the way CUDA does. The tests still fail, as expected. My version was the same change plus a `&& hfunc` guard on the `hipFuncGetAttribute` call; either is fine. One wording point for whoever lands it: the warning's "maximum available" comes from `device.max_shared_memory_per_block` (65536) and does not subtract the kernel's static bytes, so the message reads "65536 bytes, but maximum available is 65536". The message is upstream's Python and is the same on CUDA, so it is noted here and not proposed.

Side observation, not followed up: the build that accidentally linked the image's ROCm 10.1 runtime (setup trap above) did not reject the oversize launch. `hipModuleLaunchKernel` returned hipSuccess, the dispatch went out with `group_seg_size=66560`, and the queue aborted with `HSA_STATUS_ERROR_INVALID_ALLOCATION`, which Warp then reports as error 700 on the next copy. The 10.2 runtime returns `hipErrorInvalidValue` for the same launch, from Warp and from a standalone `hipModuleLoad` + `hipModuleLaunchKernel` reproducer (`agent_space/probe/ldsval/`), with or without a prior `hipFuncGetAttribute`. The wheel pins ROCm 10.2, so this is recorded only.

### B.3: Q3 `tile_matmul` benchmark on gfx1100 (raw numbers for Jeff; no decision taken)
Device "AMD Radeon Pro W7800 48GB" (gfx1100, 70 CUs / 35 WGPs, wave32), HIP 7.17, ROCm 10.2.0 wheels, fork `8d13d596`, no libmathdx, `HIP_VISIBLE_DEVICES=0`. Scripts `bench_tile_matmul.py` and `summarize.py` extracted verbatim from session 4. Command, each run wrapped in `utils/timeit.sh warp-1.14 bench` (161 s for the first run, which includes the kernel compiles, then 54 s and 53 s):
```
for n in 1 2 3; do python bench_tile_matmul.py --blocks 16384 --iters 50 --csv gfx1100_run$n.csv; done
python summarize.py gfx1100_run1.csv gfx1100_run2.csv gfx1100_run3.csv
```
16384 blocks as on gfx950/gfx942, for comparability (about 234 blocks per CU on this part).

| dtype | tile | block_dim | fwd ms (min-max) | fwd GFLOP/s (best) | adj ms (min-max) | adj GFLOP/s (best) | check |
|---|---|---|---|---|---|---|---|
| float16 | 16x16 | 32 | 1.074-1.123 | 4000 | 0.295-0.300 | 909 | ok |
| float16 | 16x16 | 64 | 1.066-1.164 | 4027 | 0.283-0.288 | 950 | ok |
| float16 | 16x16 | 128 | 0.911-0.989 | 4714 | 0.257-0.268 | 1044 | ok |
| float16 | 16x16 | 256 | 1.274-1.381 | 3371 | 0.299-0.307 | 899 | ok |
| float16 | 32x32 | 32 | 3.545-3.729 | 9691 | 4.299-4.424 | 499 | ok |
| float16 | 32x32 | 64 | 3.983-4.240 | 8627 | 2.395-2.478 | 897 | ok |
| float16 | 32x32 | 128 | 4.918-5.313 | 6987 | 1.864-1.943 | 1152 | ok |
| float16 | 32x32 | 256 | 6.316-6.816 | 5440 | 1.818-1.960 | 1181 | ok |
| float16 | 64x64 | 32 | 32.239-33.227 | 8526 | 60.545-60.710 | 284 | ok |
| float16 | 64x64 | 64 | 26.391-27.793 | 10416 | 33.613-33.716 | 511 | ok |
| float16 | 64x64 | 128 | 26.454-28.234 | 10391 | 16.997-17.437 | 1011 | ok |
| float16 | 64x64 | 256 | 16.794-18.081 | 16368 | 13.297-13.819 | 1292 | ok |
| float32 | 16x16 | 32 | 1.761-1.900 | 2439 | 0.409-0.431 | 657 | ok |
| float32 | 16x16 | 64 | 1.924-2.027 | 2233 | 0.339-0.345 | 792 | ok |
| float32 | 16x16 | 128 | 1.223-1.287 | 3512 | 0.383-0.407 | 700 | ok |
| float32 | 16x16 | 256 | 1.560-1.620 | 2753 | 0.419-0.461 | 641 | ok |
| float32 | 32x32 | 32 | 7.470-7.936 | 4600 | 6.469-6.569 | 332 | ok |
| float32 | 32x32 | 64 | 7.901-8.492 | 4349 | 3.280-3.450 | 655 | ok |
| float32 | 32x32 | 128 | 7.585-8.291 | 4530 | 2.534-2.674 | 847 | ok |
| float32 | 32x32 | 256 | 9.065-9.886 | 3791 | 2.688-3.051 | 799 | ok |
| float32 | 64x64 | 32 | 68.598-69.710 | 4007 | (no launch) | - | ADJ-FAILED (smem) |
| float32 | 64x64 | 64 | 42.993-44.342 | 6394 | (no launch) | - | ADJ-FAILED (smem) |
| float32 | 64x64 | 128 | 38.867-41.488 | 7072 | (no launch) | - | ADJ-FAILED (smem) |
| float32 | 64x64 | 256 | 48.129-50.039 | 5711 | (no launch) | - | ADJ-FAILED (smem) |
| float64 | 16x16 | 32 | 6.602-6.851 | 651 | 1.387-1.450 | 194 | ok |
| float64 | 16x16 | 64 | 6.515-6.805 | 659 | 0.989-1.012 | 271 | ok |
| float64 | 16x16 | 128 | 6.638-6.949 | 647 | 1.280-1.356 | 210 | ok |
| float64 | 16x16 | 256 | 6.589-6.907 | 652 | 1.242-1.330 | 216 | ok |
| float64 | 32x32 | 32 | 56.860-57.329 | 604 | 23.149-23.249 | 93 | ok |
| float64 | 32x32 | 64 | 49.601-50.236 | 693 | 13.046-13.252 | 165 | ok |
| float64 | 32x32 | 128 | 45.203-46.083 | 760 | 8.844-9.041 | 243 | ok |
| float64 | 32x32 | 256 | 46.405-47.247 | 740 | 7.912-8.057 | 271 | ok |
| float64 | 64x64 | 32 | (no launch) | - | (no launch) | - | MISMATCH (smem) |
| float64 | 64x64 | 64 | (no launch) | - | (no launch) | - | MISMATCH (smem) |
| float64 | 64x64 | 128 | (no launch) | - | (no launch) | - | MISMATCH (smem) |
| float64 | 64x64 | 256 | (no launch) | - | (no launch) | - | MISMATCH (smem) |

- LDS-limited rows (64 KiB per workgroup on gfx942), with the Warp UserWarnings that name the bytes. fp32 64x64 adjoint: the backward kernel needs 98304 bytes, so it did not launch. Its timed 0.25 ms (67 TFLOP/s reported by the script) is the three memsets alone. fp64 64x64: the forward kernel needs 98304 and the backward 196608, so neither launched. The forward then fails the check (`MISMATCH`), and its 0.014 ms "timings" (about 20 PFLOP/s reported) are launch failures. Those columns are meaningless and are replaced by "(no launch)" above; the raw values are in the CSV below. As expected from the handoff, this part loses one row group more than gfx950 (fp32 64x64 adjoint).
- Run-to-run spread over the 56 timing cells of the 28 all-ok rows: median 1.0%. Above 5%: fp64 16x16 bd64 adj 98% (runs 1-2 at 0.169/0.166 ms, run 3 at 0.330 ms: one outlier run), fp32 32x32 bd64 adj 15%, fp32 16x16 bd64 adj 11%, fp32 16x16 bd256 adj 8%, fp16 16x16 bd128 adj 7%, fp16 16x16 bd256 adj 7%. All are sub-0.4 ms adjoint cells, where the three seeding memsets are a large share.
- gfx942 vs gfx950 at the same settings (best forward GFLOP/s, ratio gfx942/gfx950): fp16 64x64 bd256 0.74, fp32 32x32 bd128 0.89, fp32 64x64 bd128 0.53, fp64 32x32 bd128 0.74. Numbers only; no interpretation is drawn here.
- As on gfx950, fp16 adjoint is much slower than fp32 at 16x16 and 32x32 (fp16 atomic_add is the 32-bit CAS loop, Phase 1 notes).
- No NVIDIA GPU on this host, so no libmathdx reference numbers.

Raw per-run timings, ms (`gfx942_run<n>.csv` merged; checks identical across runs):
```
dtype,M,block_dim,fwd_ms_r1,fwd_ms_r2,fwd_ms_r3,adj_ms_r1,adj_ms_r2,adj_ms_r3,check
float16,16,32,0.3239,0.3251,0.3243,0.3473,0.3397,0.3407,ok
float16,16,64,0.1945,0.1901,0.1903,0.3695,0.3497,0.3503,ok
float16,16,128,0.1874,0.1862,0.1830,0.3235,0.3013,0.3013,ok
float16,16,256,0.2419,0.2386,0.2421,0.3127,0.2927,0.2925,ok
float16,32,32,2.1264,2.1299,2.1290,2.1946,2.1904,2.1938,ok
float16,32,64,0.8612,0.8641,0.8691,1.9550,1.9399,1.9470,ok
float16,32,128,0.9377,0.9601,0.9632,2.0938,2.0769,2.0804,ok
float16,32,256,1.1205,1.1407,1.1293,1.8424,1.8327,1.8313,ok
float16,64,32,25.6960,25.6989,25.6941,29.7936,29.8107,29.8318,ok
float16,64,64,13.5915,13.5904,13.5842,19.0395,19.0280,19.0316,ok
float16,64,128,9.1594,9.1970,9.1979,11.4572,11.4573,11.4522,ok
float16,64,256,6.3166,6.3444,6.3523,7.8291,7.8182,7.8188,ok
float32,16,32,0.4163,0.4169,0.4173,0.2692,0.2541,0.2529,ok
float32,16,64,0.3448,0.3414,0.3415,0.1381,0.1242,0.1259,ok
float32,16,128,0.2333,0.2314,0.2307,0.0985,0.0923,0.0929,ok
float32,16,256,0.2803,0.2771,0.2763,0.0986,0.0913,0.0935,ok
float32,32,32,3.1414,3.1305,3.1249,2.2560,2.2362,2.2443,ok
float32,32,64,1.5429,1.5563,1.5588,1.2902,1.4674,1.2749,ok
float32,32,128,1.4213,1.4347,1.4316,1.1382,1.1422,1.1318,ok
float32,32,256,1.5786,1.5964,1.6038,0.8898,0.8825,0.8958,ok
float32,64,32,38.0388,38.0330,38.0333,0.2576,0.2509,0.2508,ADJ-FAILED
float32,64,64,18.7341,18.7326,18.7296,0.2602,0.2549,0.2539,ADJ-FAILED
float32,64,128,13.4477,13.4372,13.4421,0.2603,0.2547,0.2552,ADJ-FAILED
float32,64,256,9.8860,9.9363,9.9502,0.2609,0.2563,0.2576,ADJ-FAILED
float64,16,32,0.5882,0.5891,0.5880,0.3300,0.3213,0.3231,ok
float64,16,64,0.4417,0.4498,0.4493,0.1693,0.1664,0.3299,ok
float64,16,128,0.4374,0.4493,0.4490,0.2141,0.2113,0.2095,ok
float64,16,256,0.5275,0.5365,0.5354,0.2128,0.2087,0.2099,ok
float64,32,32,5.9578,5.9534,5.9521,4.5795,4.5832,4.5844,ok
float64,32,64,3.2444,3.2409,3.2442,2.7309,2.7254,2.7401,ok
float64,32,128,2.7084,2.7288,2.7254,1.9290,1.9191,1.9417,ok
float64,32,256,3.0867,3.1087,3.1134,1.5162,1.5113,1.5264,ok
float64,64,32,0.0151,0.0144,0.0143,0.3937,0.3830,0.3853,MISMATCH
float64,64,64,0.0144,0.0150,0.0138,0.3938,0.3833,0.3841,MISMATCH
float64,64,128,0.0144,0.0140,0.0137,0.3922,0.3832,0.3811,MISMATCH
float64,64,256,0.0144,0.0136,0.0137,0.3914,0.3820,0.3843,MISMATCH
```

### C: two-GPU session (gfx942, `HIP_VISIBLE_DEVICES=0,1`, default `unique_or_2x` mode adds `cuda:1`)
Command: the loop of handoff item C.2 (13 files, `-p <file> -m 32`), wrapped in `utils/timeit.sh warp-1.14 test` (420 s).

| file | ran | result | failing / notable |
|---|---|---|---|
| test_multigpu | 5 | OK, 2 skipped | pingpong x2: "Peer-to-Peer transfers not supported" (IOMMU heuristic, below) |
| test_peer | 5 | OK, 1 skipped | "Requires devices without peer access support" (expected) |
| test_mempool | 10 | **errors=1**, 1 skipped | `test_mempool_access` (finding C2) |
| test_streams | 42 | **failures=2**, 2 skipped | `test_stream_priority_timings_cuda_0/_1` (finding C4); 2 mgpu graph tests skipped by the IOMMU heuristic |
| test_graph | 61 | **errors=1**, 4 skipped | `test_cuda_graph_alloc_transient_stream_cuda_1` (finding C1); skips: 2 IOMMU heuristic, 2 upstream "flakiness in memory usage measurement" |
| test_ipc | 8 | OK | |
| test_unified_memory | 40 | **errors=1**, 8 skipped | `..._device_can_access_uses_mempool_state_when_target_mempools_enabled` (C2); skips: 4 IOMMU heuristic, 4 "cuda:N cannot access CPU memory" |
| test_allocator | 16 | OK, 1 skipped | rmm not installed |
| test_async | 3533 | OK | |
| test_sparse | 71 | **errors=1** | `test_capturability_cuda_1` (C1, plan risk 11) |
| test_fem_integrate | 25 | **failures=1, errors=1** | `test_capturability_cuda_1` (C1, plan risk 11); `test_integrate_gradient_cuda_0` (C5, collateral of C1) |
| test_array | 207 | OK | |
| test_copy | 30 | OK | |

Single-GPU control runs (same build, one GPU visible) of test_streams, test_fem_integrate, test_graph and test_sparse, with physical GPU 0 alone and physical GPU 1 alone. GPU0: streams 24 run, failures=1 (priority timings), 2 skipped; fem_integrate 17 OK; graph 35 OK (3 skipped); sparse 47 OK. GPU1 alone: identical. So the capture failures need a non-primary ordinal, not a particular physical GPU. The priority-timing failure is a gfx942 property.

**C1. Plan risk 11 reproduced: a ROCm defect in graph instantiation for host<->graph-allocated copies. Not a port defect.** All three `_cuda_1` capture failures are `Warp CUDA error 801: operation not supported (in function wp_cuda_graph_create_exec, warp.cu:3493)`, that is, `hipGraphInstantiateWithFlags` returning `hipErrorNotSupported`. Each also reproduces when run alone (`-k <name> -m 1`). Narrowed with Warp (`cap_red.py`): on `cuda:1`, a capture containing a kernel, `wp.zeros`/`wp.ones`/`wp.empty` allocations and a side stream instantiates fine. Adding a copy from a pinned CPU array into an array allocated in the capture fails, on the main stream or a side stream. The sparse/fem captures contain no such host-to-device copy, but they do contain `hipMemcpyAsync(..., hipMemcpyDeviceToHost)` from in-capture allocations into pinned memory (HIP API trace, `AMD_LOG_LEVEL=3 AMD_LOG_MASK=1`). The `Failed hsa_amd_vmem_set_access ... status 4099` lines that HIP logs at every in-capture `hipMallocAsync` also appear on device 0 and in passing graphs, so they are noise. Standalone HIP reproducer (no Warp), results with `HIP_VISIBLE_DEVICES=0,1`, one case per process:

| capture device | pinned buffer allocated with current device | copy | copy endpoint allocated in the capture (`hipMallocAsync`) | result |
|---|---|---|---|---|
| 0 | 0 | H2D or D2H | yes | OK |
| 0 | 1 | H2D | yes | **segfault in the process** |
| 0 | 1 | D2H | yes | **instantiate 801** |
| 1 | 0 or 1 | H2D | yes | **instantiate 801** |
| 1 | 0 | D2H | yes | **instantiate 801** |
| 1 | 1 | D2H | yes | OK |
| any | any | H2D or D2H | no (plain `hipMalloc` endpoint) | OK |

Retaining a user object or freeing the buffer inside the capture changes nothing. Plain-HIP graphs on device 1 with in-capture alloc/free, kernels and D2D copies all instantiate and run correctly (`cap1.hip`). So HIP appears to pick the wrong device for a memcpy node between pinned host memory and graph-allocated (virtual, not yet mapped) memory, unless every device involved is the same. That is inferred from the matrix, not from the runtime source. On device 0 a mismatch can crash the process. Reproducer (ROCm clang from the TheRock wheels: `clang++ -x hip --rocm-path=$R --offload-arch=gfx942 -O2 h2d2.hip -o h2d2 -L$R/lib -lamdhip64 -Wl,-rpath,$R/lib`; run `./h2d2 <dev> <pinned_dev> <async_dst> <userobj> <free> [d2h]`, e.g. `./h2d2 1 1 1 0 0` -> 801, `./h2d2 0 1 1 0 0` -> segfault):
```cpp
// Device `dev`: capture { hipMallocAsync(dst) ; hipMemcpyAsync(dst <- pinned host, HostToDevice) ; [hipFreeAsync] },
// optionally retaining a user object, pinned buffer allocated while `alloc_dev` was current.
#include <hip/hip_runtime.h>
#include <cstdio>
#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) printf("  %s -> %d %s\n", #x, e, hipGetErrorString(e)); } while (0)
static void dtor(void*) {}
static int g_d2h = 0;
static void run(int dev, int alloc_dev, int async_dst, int uobj, int free_in) {
    const size_t n = 1024;
    CK(hipSetDevice(alloc_dev));
    float* h; CK(hipHostMalloc((void**)&h, n * 4, 0));
    for (size_t i = 0; i < n; ++i) h[i] = 42.f;
    CK(hipSetDevice(dev));
    hipStream_t s; CK(hipStreamCreate(&s));
    float* out; CK(hipMalloc(&out, n * 4));
    hipGraph_t g = nullptr; hipGraphExec_t ge = nullptr;
    CK(hipStreamBeginCapture(s, hipStreamCaptureModeGlobal));
    float* d = out;
    if (async_dst) CK(hipMallocAsync((void**)&d, n * 4, s));
    if (g_d2h) { CK(hipMemcpyAsync(d, out, n * 4, hipMemcpyDeviceToDevice, s)); CK(hipMemcpyAsync(h, d, n * 4, hipMemcpyDeviceToHost, s)); }
    else CK(hipMemcpyAsync(d, h, n * 4, hipMemcpyHostToDevice, s));
    if (async_dst) { CK(hipMemcpyAsync(out, d, n * 4, hipMemcpyDeviceToDevice, s)); if (free_in) CK(hipFreeAsync(d, s)); }
    CK(hipStreamEndCapture(s, &g));
    if (uobj) { hipUserObject_t u; CK(hipUserObjectCreate(&u, nullptr, dtor, 1, hipUserObjectNoDestructorSync)); CK(hipGraphRetainUserObject(g, u, 1, hipGraphUserObjectMove)); }
    hipError_t e = hipGraphInstantiateWithFlags(&ge, g, hipGraphInstantiateFlagAutoFreeOnLaunch);
    printf("dev %d pinned-on %d async_dst %d userobj %d free %d: instantiate -> %d %s", dev, alloc_dev, async_dst, uobj, free_in, e, hipGetErrorString(e));
    if (e == hipSuccess) { CK(hipGraphLaunch(ge, s)); CK(hipStreamSynchronize(s)); float v = 0; CK(hipMemcpy(&v, out, 4, hipMemcpyDeviceToHost)); printf(", out[0]=%g", v); CK(hipGraphExecDestroy(ge)); }
    printf("\n");
    if (g) CK(hipGraphDestroy(g));
    CK(hipFree(out)); CK(hipStreamDestroy(s)); CK(hipHostFree(h));
}
#include <cstdlib>
int main(int argc, char** argv) { setvbuf(stdout, nullptr, _IONBF, 0);
    g_d2h = argc > 6 ? atoi(argv[6]) : 0;
    run(atoi(argv[1]), atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), atoi(argv[5]));
    return 0;
}
```
This is the fault the earlier AMD tree hid by restricting the capture tests to ordinal 0. Warp-side mitigation is limited. `wp_alloc_pinned` (`warp.cu:747`) calls `hipHostMalloc` with whatever device is current, but even matching devices fail for H2D on device 1, so allocator changes cannot fix it. Proposed registration for the lock holder (not run here, because this session writes no project records other than these notes): `python3 utils/deferred.py add rocm-bug-report --project warp-1.14` with id `hip-graph-memcpy-host-graph-alloc-device-mismatch` and the matrix and reproducer above.

**C2. ROCm defect: `hipMemPoolSetAccess` cannot revoke peer access.** `test_mempool_access` and `test_unified_memory..._device_can_access_uses_mempool_state_when_target_mempools_enabled` fail with `Warp CUDA error 101: invalid device ordinal (in function wp_cuda_set_mempool_access_enabled, warp.cu:3059)` -> "Failed to disable memory pool access from device cuda:1 to device cuda:0". Standalone (`poolacc.hip`): on device 0's default pool, `hipMemPoolSetAccess({device 1, ProtReadWrite})` succeeds and `hipMemPoolGetAccess` then reports flags 3. `hipMemPoolSetAccess({device 1, ProtNone})` returns `hipErrorInvalidDevice` (101), with or without an outstanding allocation, and access stays 3. CUDA documents `ProtNone` as revoking access. Warp's call (`warp.cu:3024-3066`) is correct, so this is not a port defect. Once a mempool peer grant is made it cannot be undone, so the tests' restore step fails too. Reproducer:
```cpp
#include <hip/hip_runtime.h>
#include <cstdio>
#define CK(x) do { hipError_t e = (x); printf("%-50s -> %d %s\n", #x, e, hipGetErrorString(e)); } while (0)
int main() {
    hipMemPool_t pool; CK(hipDeviceGetDefaultMemPool(&pool, 0));
    hipMemAccessDesc d; d.location.type = hipMemLocationTypeDevice; d.location.id = 1;
    d.flags = hipMemAccessFlagsProtReadWrite; CK(hipMemPoolSetAccess(pool, &d, 1));   // 0
    d.flags = hipMemAccessFlagsProtNone;      CK(hipMemPoolSetAccess(pool, &d, 1));   // 101 hipErrorInvalidDevice
    hipMemAccessFlags f; hipMemLocation loc = d.location; CK(hipMemPoolGetAccess(&f, pool, &loc)); printf("flags %d\n", (int)f);  // 3
}
```
Proposed registration (not run): `deferred.py add rocm-bug-report --project warp-1.14`, id `hip-mempool-set-access-cannot-revoke`.

**C3. HIP limitation: one graph cannot hold kernel nodes for two devices.** This was found only with the IOMMU heuristic overridden (next paragraph). `test_stream_arg_graph_mgpu` and `test_stream_scope_graph_mgpu` capture on a `cuda:0` stream, fork to a `cuda:1` stream and launch a kernel on each device. Instantiation returns 801, and the follow-up `wp_cuda_context_synchronize` reports 900 (still capturing). Standalone (`mdev.hip`, peer access enabled both ways, default allocators): fork/join plus a kernel on device 0 instantiates, with or without a peer copy from device 1. Adding a kernel on device 1 in the same capture -> `hipGraphInstantiate` 801. CUDA supports multi-device graphs. Proposed registration (not run): id `hip-graph-multi-device-kernel-nodes`, kind rocm-bug-report, or a documented limitation if HIP declares it one.

IOMMU heuristic: Warp's `check_p2p()` (`warp/_src/utils.py:1747`) returns False on Linux whenever `/sys/class/iommu` or `/sys/kernel/iommu_groups` is non-empty. This host boots with `amd_iommu=on iommu=pt` (passthrough), so 8 tests skip with "Peer-to-Peer transfers not supported", although HIP reports full peer access. That is upstream behaviour and was left alone. With `check_p2p()` forced to return True (local edit, reverted): test_multigpu 5/5 OK (both pingpong tests pass), `test_graph_launch_array_access_mode_checked_peer_access_cuda_capture` OK, `..._mempool_access_cuda_capture` error (C2), unified_memory "Slices of CUDA mempool allocations..." OK, the other three peer/mempool cases error (C2: 5 revoke failures in that file), and test_streams' two mgpu graph tests error (C3).

**C4. `test_stream_priority_timings` fails on gfx942 (one or two GPUs). Not a port defect.** 5 of 5 single-GPU reruns fail by 0.5-1% (for example 1.1212 vs 1.1184 ms). A probe with the test's workload (`prio.py`, 2 x 128 Mi-float D2D copies per stream, 5 repetitions per cell) shows that issue order decides the result and priority has no measurable effect. With the low-priority stream's copies issued first, the high-priority stream is slower 5/5, at priority -1 and also at equal priority 0. With the high-priority stream issued first, it wins 5/5 at either priority. Stream priorities are applied (`Stream.priority` reads 0 and -1), but they do not affect these same-device copies on gfx942. The test passed on gfx950 in session 3. Whether to keep, skip or rework it is for the lock holder or Jeff.

**C5. `test_fem_integrate.test_integrate_gradient_cuda_0`: collateral of C1.** It fails 3/3 in full-file two-GPU runs (202/256 gradient entries wrong, max abs diff 0.0267) and passes 5/5 alone. Pairwise: `-k capturability_cuda_1 -k integrate_gradient_cuda_0` fails, `-k capturability_cuda_0 -k integrate_gradient_cuda_0` passes. So the failed `cuda:1` capture leaves state that corrupts a later `cuda:0` computation in the same process. Likely candidates are temporaries cached during the never-instantiated capture; not traced further, because it should disappear with C1. Worth rechecking once C1 is fixed or worked around.

### Not done in this session
- Handoff item A.4 (optional atomic timings): not requested.
- Item B (wave32) and the gfx950 two-GPU variant of C: other hosts.

### Next step for the lock holder
1. Decide on the `cuFuncSetAttribute_f` static-LDS fix above (shared HIP shim code, not CUDA). If adopted: commit it, compile the CUDA path with nvcc, and `advance-head`.
2. Decide how the four 64 KiB-LDS tile tests are handled on 64 KiB parts (A.2).
3. Register C1, C2 and C3 with `deferred.py` (ids proposed above) and bring them to Jeff for filing. Decide whether test_mempool_access, the unified-memory mempool test and the `_cuda_1` capture tests get capability skips naming those registrations.
4. C4: priority-timing test on gfx942.
5. The Q3 gfx942 numbers above go to Jeff together with the wave32 numbers from item B.
- LDS-limited rows are the same as on gfx942 (64 KiB per workgroup), with Warp's UserWarnings naming the bytes: fp32 64x64 adjoint needs 98304 (backward), and fp64 64x64 needs 98304 (forward) and 196608 (backward). Their timed values are memsets or launch failures (the raw CSV below keeps them: about 1.19 ms fp32 adj, about 0.02 ms fp64 fwd). They are meaningless and shown as "(no launch)".
- Run-to-run spread over the 56 timing cells of the 28 all-ok rows: median 4.6%, maximum 14% (fp32 32x32 bd256 adj). 26 cells are above 5%. The pattern is systematic, not noise: run 1 is the fastest in almost every forward cell, and runs 2 and 3 are 3-9% slower. That fits clocks dropping as this workstation card warms up under back-to-back runs. Clocks were not recorded (no `rocm-smi` in this container). The "best" columns therefore come from run 1.
- fp16 forward is faster than fp32 forward at every size here (best 16368 vs 7072 GFLOP/s at 64x64). fp64 is 6-14x slower than fp32. fp16 adjoint is again slower than fp32 at 32x32 bd32 (fp16 `atomic_add` is the CAS loop, ruled to stay, see the unsafe-atomics ruling above). Numbers only; no interpretation is drawn here.
- No NVIDIA GPU on this host, so no libmathdx reference numbers.

Raw per-run timings, ms (`gfx1100_run<n>.csv` merged; checks identical across runs):
```
dtype,M,block_dim,fwd_ms_r1,fwd_ms_r2,fwd_ms_r3,adj_ms_r1,adj_ms_r2,adj_ms_r3,check
float16,16,32,1.0736,1.1152,1.1232,0.2954,0.2980,0.3004,ok
float16,16,64,1.0665,1.1639,1.1596,0.2841,0.2826,0.2876,ok
float16,16,128,0.9111,0.9706,0.9887,0.2572,0.2637,0.2675,ok
float16,16,256,1.2742,1.3464,1.3806,0.2987,0.3029,0.3073,ok
float16,32,32,3.5454,3.7013,3.7293,4.2994,4.3821,4.4236,ok
float16,32,64,3.9828,4.2204,4.2405,2.3945,2.4780,2.4720,ok
float16,32,128,4.9177,5.2821,5.3128,1.8641,1.9432,1.9410,ok
float16,32,256,6.3164,6.8108,6.8161,1.8179,1.9603,1.9526,ok
float16,64,32,32.2393,33.1857,33.2275,60.5488,60.7100,60.5453,ok
float16,64,64,26.3912,27.7929,27.3382,33.6135,33.7163,33.6232,ok
float16,64,128,26.4536,28.2339,27.3993,16.9968,17.4369,17.1895,ok
float16,64,256,16.7938,18.0810,17.4252,13.2974,13.8193,13.5415,ok
float32,16,32,1.7610,1.9001,1.8533,0.4087,0.4306,0.4109,ok
float32,16,64,1.9238,2.0268,1.9680,0.3390,0.3454,0.3450,ok
float32,16,128,1.2230,1.2867,1.2411,0.3834,0.4075,0.3945,ok
float32,16,256,1.5603,1.6199,1.5623,0.4186,0.4607,0.4509,ok
float32,32,32,7.4703,7.9364,7.7204,6.4692,6.5691,6.5138,ok
float32,32,64,7.9014,8.4922,8.2303,3.2805,3.4502,3.3850,ok
float32,32,128,7.5852,8.2912,7.9819,2.5344,2.6738,2.6386,ok
float32,32,256,9.0645,9.8857,9.5215,2.6883,3.0513,2.9950,ok
float32,64,32,68.5983,69.7095,68.8262,1.2021,1.2059,1.2036,ADJ-FAILED
float32,64,64,43.0511,44.3421,42.9927,1.1828,1.1927,1.1921,ADJ-FAILED
float32,64,128,41.4885,40.0710,38.8672,1.1821,1.1924,1.1919,ADJ-FAILED
float32,64,256,48.1286,50.0389,48.6761,1.1810,1.1928,1.1916,ADJ-FAILED
float64,16,32,6.6023,6.8514,6.6832,1.3867,1.4502,1.4318,ok
float64,16,64,6.5154,6.8051,6.6362,0.9893,1.0124,1.0040,ok
float64,16,128,6.6380,6.9486,6.8079,1.2803,1.3560,1.3319,ok
float64,16,256,6.5893,6.9068,6.7651,1.2425,1.3297,1.3156,ok
float64,32,32,56.8597,57.3286,56.9156,23.1491,23.2490,23.1692,ok
float64,32,64,49.6009,50.2361,49.6633,13.0458,13.2523,13.1529,ok
float64,32,128,45.2028,46.0826,45.3639,8.8441,9.0414,8.9385,ok
float64,32,256,46.4047,47.2472,46.5232,7.9123,8.0572,7.9521,ok
float64,64,32,0.0211,0.0196,0.0192,2.5090,2.5072,2.5090,MISMATCH
float64,64,64,0.0199,0.0180,0.0192,2.4874,2.4877,2.4876,MISMATCH
float64,64,128,0.0182,0.0186,0.0232,2.4866,2.4876,2.4876,MISMATCH
float64,64,256,0.0177,0.0187,0.0190,2.4866,2.4884,2.4865,MISMATCH
```

### B.4: Phase 1-3 test files at wavefront 32: all pass except the 2 known fem isolation failures
Same patterns as the handoff list, one `-p` per line, `-m 32`, one GPU visible. Compared with the gfx950 counts of sessions 2 and 3:

| file / pattern | ran | result | same as gfx950 |
|---|---|---|---|
| test_codegen | 117 | OK | yes |
| test_snippet | 12 | OK | yes |
| test_arithmetic | 126 | OK | yes |
| test_math | 16 | OK | yes |
| test_fp16 | 6 | OK | yes |
| test_bf16 | 48 | OK | yes |
| test_atomic | 164 | OK | yes |
| test_atomic_bitwise | 6 | OK | yes |
| test_atomic_cas | 24 | OK | yes (GPU cases filtered by capability, as ruled) |
| test_array | 139 | OK | gfx950 Phase 1 recorded 138; test_array.py is unchanged since then (`git log fda7eb27..8d13d596 -- warp/tests/test_array.py` empty); all pass |
| test_copy | 14 | OK | yes |
| test_launch | 32 | OK | yes |
| test_fast_math | 12 | OK, 1 skipped | yes (PTX content check) |
| test_grad | 41 | OK | yes |
| test_kernel_cache | 5 | OK | yes |
| test_module_hashing | 8 | OK | yes |
| test_print | 10 | OK | yes |
| test_mat*.py | 601 | OK | yes |
| test_vec*.py | 531 | OK | yes |
| test_quat*.py | 173 | OK | yes |
| test_spatial*.py | 207 | OK | yes |
| test_rand | 10 | OK | yes |
| test_noise | 14 | OK | yes |
| test_clang_cuda | 7 | OK, 6 skipped | yes (NVIDIA-only) |
| test_runlength_encode | 16 | OK | yes |
| test_array_reduce | 16 | OK | yes |
| test_sparse | 47 | OK | yes |
| test_linear_solvers | 34 | OK | yes |
| test_coloring | 3 | OK | yes |
| test_bvh | 13 | OK | yes |
| test_grouped_bvh | 11 | OK | yes |
| test_hash_grid | 23 | OK | yes |
| test_marching_cubes | 17 | OK | yes |
| test_mesh*.py | 49 | OK | yes (usd-core installed, 0 skipped) |
| test_volume | 59 | OK | yes |
| test_volume_write | 5 | OK | yes |
| test_apic | 52 | OK | yes (session 3 count) |
| test_apic_mesh | 12 | OK | yes (session 3 count) |
| test_fem*.py | 107 | 2 failures | yes: test_gimp_quadrature and test_cube_shape_functions, "Unexpected output" = Warp's init greeting captured by the first check_output test in its worker, the upstream isolation behaviour recorded in session 2 |
| test_graph | 35 | OK, 3 skipped | yes |
| test_array_fill_capture | 3 | OK | yes |
| test_capture_mode | 2 | OK | yes |
| test_conditional_captures | 32 | OK, 16 skipped | yes (Q4) |
| test_conditional | 50 | OK | yes |
| test_async | 2373 | OK | yes |
| test_streams | 24 | OK, 2 skipped | yes |
| test_mempool | 7 | OK, 2 skipped | yes |
| test_pinned | 1 | OK | yes |
| test_unified_memory | 27 | OK, 8 skipped | yes: 2 "cuda:0 cannot access CPU memory" (the W7800 also reports no pageable-memory access), 6 multi-device mempool/peer cases |
| test_occupancy | 6 | OK | yes |
| test_allocator | 16 | OK, 3 skipped | yes |
| test_allocation_tracker | 34 | OK | yes |
| test_dlpack | 16 | OK | yes |
| test_torch | 34 | OK | yes |
| test_map | 45 | OK | yes |
| test_reload | 14 | OK | yes |
| test_cuda_arch_suffix | 10 | all skipped | yes (NVIDIA-only registration) |

Every skip reason across the tile and Phase 1-3 runs (`grep -h "skipped '"` over all logs): Conditional graph nodes not supported 16, Warp was not built with MathDx support 8, MathDx is not enabled 1, Skipping test on CUDA device without MathDx (tolerance) 1, No suitable devices to run the test 17, Requires at least two CUDA devices 2, Multi-GPU not available 2, mempool/peer-access device-pair requirements 10, cuda:0 cannot access CPU memory 2, flakiness in memory usage measurement 1, rmm not installed 1. None names wavefront size or AMD.

### B.1 / Q2: user-snippet intrinsics under hipRTC at wavefront 32
Probe `agent_space/probe/snip32.py` (full source below). Each case is a `wp.func_native` snippet whose device code sits under `#if defined(__CUDA_ARCH__)`, with a sentinel in the `#else` branch, so the host branch would be caught. Each case compiles in its own module, launches 256 threads with `block_dim=64` (two 32-lane wavefronts per block), and is compared against NumPy.

| case | result | detail |
|---|---|---|
| `nextafterf` | ok | exact, 256 values |
| `__frsqrt_rn` | **not correctly rounded** | max 1 ulp; see below |
| `__popcll` | ok | exact |
| `__ffsll` | ok | exact |
| `__clzll` | ok | exact |
| `__syncwarp` + `__shared__` (lane `^ 1` exchange) | ok | exact |
| `__shared__` + `__syncthreads` across two wavefronts | ok | exact |
| `__threadfence` | ok | exact |
| `warpSize` | ok | 32 |
| `__activemask()` | ok | 0xFFFFFFFF (32 lanes) |
| `__ballot_sync(~0ull, pred)` | ok | 0xAAAAAAAA |
| `__ballot_sync(0xffffffff, pred)` (CUDA spelling) | **compile error** | hipRTC: `static assertion failed ... 'sizeof(unsigned int) == 8': The mask must be a 64-bit integer` (`hiprtc_runtime.h:7981`) |
| `__shfl_xor_sync(0xffffffff, v, 1, 32)` (CUDA spelling) | **compile error** | same static assertion (`hiprtc_runtime.h:7862`) |

All eight intrinsics the Q2 ruling lists compile and take the device branch at wavefront 32. Seven of them give exact results. The exception is `__frsqrt_rn`: HIP defines it as `__builtin_amdgcn_rsqf(x)` (`lib/llvm/lib/clang/24/include/__clang_hip_math.h:289`), the hardware `v_rsq_f32` approximation. CUDA's `__frsqrt_rn` is the IEEE round-to-nearest reciprocal square root. Measured over 2^20 inputs spread over 2^-60..2^60 (`agent_space/probe/frsqrt.py`), compared with `1/sqrt` computed in float64 and rounded to float32: `__frsqrt_rn` max 1 ulp, 11.0% not correctly rounded, bit-identical to `rsqrtf`. The portable `1.0f/sqrtf(x)` is worse (1 ulp, 27.0%). This is a HIP header semantic, the same on every AMD target, wave64 included. Session 2's gfx950 probe reported `__frsqrt_rn` exact, probably on inputs whose result is exact (an inference: that probe is not in the repository). It maps to the native AMD operation, as the ruling asked, and is within 1 ulp. Newton's uses (sdf_contact, support_function) look tolerance-based, but that was not checked. Whether Warp's hipRTC prelude should provide a correctly rounded `__frsqrt_rn` (it cannot simply redefine clang's inline, so it would need a different spelling or a header-order change) is open for the lock holder or Jeff. Nothing proposed here.

Not on the ruling's list, but relevant to downstream snippets: HIP refuses a 32-bit lane mask in `__ballot_sync`, `__shfl*_sync` and the other `*_sync` intrinsics at compile time, on every AMD target, wave32 included. A snippet copied from CUDA with `0xffffffff` masks therefore fails to build instead of misbehaving, and needs `~0ull` (or `__activemask()`) on HIP. Newton's steel solver, recorded under Q2 as hard-coding 32-lane warps, is the kind of code that hits this. On wave32 its `tid & 31` lane logic is correct; only the mask spelling breaks.

`snip32.py`:
```python
# Q2 user-snippet probe: CUDA intrinsics in `#if defined(__CUDA_ARCH__)` device branches of wp.func_native,
# launched with block_dim 64 (two 32-lane wavefronts on gfx11). Each case compiles in its own module.
import sys
import numpy as np
import warp as wp

wp.init()
dev = wp.get_device("cuda:0")
print(f"device {dev.name} arch_name={dev.arch_name}")
N = 256
BD = 64
results = []


def run(name, snippet, check, dtype=wp.uint64, n=N):
    try:
        f = wp.func_native(snippet)

        def native(out: wp.array(dtype=dtype), tid: int): ...

        native.__name__ = f"snip{len(results)}"
        native.__qualname__ = native.__name__
        nf = f(native)

        @wp.kernel(module="unique")
        def k(out: wp.array(dtype=dtype)):
            tid = wp.tid()
            nf(out, tid)

        out = wp.zeros(n, dtype=dtype, device=dev)
        wp.launch(k, dim=n, inputs=[out], block_dim=BD, device=dev)
        got = out.numpy()
        ok, detail = check(got)
        results.append((name, "ok" if ok else "WRONG", detail))
    except Exception as e:  # compile failure or launch error
        results.append((name, "COMPILE/RUN ERROR", str(e).splitlines()[0][:160]))


def eq(expected):
    def c(got):
        bad = np.nonzero(got != expected)[0]
        return (len(bad) == 0, "exact" if len(bad) == 0 else f"{len(bad)} mismatches, first idx {bad[0]}: got {got[bad[0]]} want {expected[bad[0]]}")
    return c


tids = np.arange(N, dtype=np.uint64)

# the value the device branch must take; -1 sentinel marks the host branch
run("nextafterf", """
#if defined(__CUDA_ARCH__)
    out[tid] = nextafterf(1.0f + tid, 1.0e9f);
#else
    out[tid] = -1.0f;
#endif
""", eq(np.nextafter(np.float32(1.0) + np.arange(N, dtype=np.float32), np.float32(1e9))), dtype=wp.float32)

xs = (np.arange(N, dtype=np.float32) * 0.37 + 0.01).astype(np.float32)
rs = (1.0 / np.sqrt(xs.astype(np.float64))).astype(np.float32)  # correctly rounded reference
run("__frsqrt_rn", """
#if defined(__CUDA_ARCH__)
    float x = (float)tid * 0.37f + 0.01f;
    out[tid] = __frsqrt_rn(x);
#else
    out[tid] = -1.0f;
#endif
""", lambda got: (lambda d: (d.max() == 0, f"max ulp diff {d.max()}, {np.count_nonzero(d)} of {N} not correctly rounded"))(np.abs(got.view(np.int32).astype(np.int64) - rs.view(np.int32).astype(np.int64))), dtype=wp.float32)

vals = (np.uint64(0x9E3779B97F4A7C15) * (tids + np.uint64(1))) >> (tids % np.uint64(64))
SEED = """unsigned long long v = (0x9E3779B97F4A7C15ull * ((unsigned long long)tid + 1ull)) >> (tid % 64);"""
run("__popcll", f"""
#if defined(__CUDA_ARCH__)
    {SEED}
    out[tid] = (unsigned long long)__popcll(v);
#else
    out[tid] = 999;
#endif
""", eq(np.array([bin(int(v)).count("1") for v in vals], dtype=np.uint64)))


def ffs(v):
    v = int(v)
    return 0 if v == 0 else (v & -v).bit_length()


def clz(v):
    return 64 - int(v).bit_length()


run("__ffsll", f"""
#if defined(__CUDA_ARCH__)
    {SEED}
    out[tid] = (unsigned long long)__ffsll((long long)(v << (tid % 61)));
#else
    out[tid] = 999;
#endif
""", eq(np.array([ffs((int(v) << (i % 61)) & (2**64 - 1)) for i, v in enumerate(vals)], dtype=np.uint64)))
run("__clzll", f"""
#if defined(__CUDA_ARCH__)
    {SEED}
    out[tid] = (unsigned long long)__clzll((long long)v);
#else
    out[tid] = 999;
#endif
""", eq(np.array([clz(v) for v in vals], dtype=np.uint64)))

# warp-synchronous exchange through shared memory between neighbouring lanes (tid & 31 style)
run("__syncwarp+__shared__", """
#if defined(__CUDA_ARCH__)
    __shared__ unsigned long long s[64];
    int lane = threadIdx.x;
    s[lane] = (unsigned long long)tid * 3ull;
    __syncwarp();
    out[tid] = s[lane ^ 1];
#else
    out[tid] = 999;
#endif
""", eq((tids ^ np.uint64(1)) * np.uint64(3)))

run("__shared__+__syncthreads (2 wavefronts)", """
#if defined(__CUDA_ARCH__)
    __shared__ unsigned long long s[64];
    s[threadIdx.x] = (unsigned long long)tid;
    __syncthreads();
    unsigned long long sum = 0;
    for (int i = 0; i < 64; ++i) sum += s[i];
    out[tid] = sum;
#else
    out[tid] = 999;
#endif
""", eq(np.repeat([tids[b:b + 64].sum() for b in range(0, N, 64)], 64).astype(np.uint64)))

run("__threadfence", """
#if defined(__CUDA_ARCH__)
    out[tid] = (unsigned long long)tid + 7ull;
    __threadfence();
    out[tid] += 1ull;
#else
    out[tid] = 999;
#endif
""", eq(tids + np.uint64(8)))

run("warpSize", """
#if defined(__CUDA_ARCH__)
    out[tid] = (unsigned long long)warpSize;
#else
    out[tid] = 999;
#endif
""", eq(np.full(N, 32, dtype=np.uint64)))

# Beyond the plan's list: CUDA-style 32-bit-mask warp intrinsics as Newton-like snippets write them
run("__shfl_xor_sync(0xffffffff,..,32)", """
#if defined(__CUDA_ARCH__)
    out[tid] = (unsigned long long)__shfl_xor_sync(0xffffffff, (int)tid, 1, 32);
#else
    out[tid] = 999;
#endif
""", eq(tids ^ np.uint64(1)))
run("__ballot_sync(0xffffffff,..)", """
#if defined(__CUDA_ARCH__)
    out[tid] = (unsigned long long)__ballot_sync(0xffffffff, (tid & 1) == 1);
#else
    out[tid] = 999;
#endif
""", eq(np.full(N, 0xAAAAAAAA, dtype=np.uint64)))
run("__ballot_sync(~0ull,..)", """
#if defined(__CUDA_ARCH__)
    out[tid] = (unsigned long long)__ballot_sync(~0ull, (tid & 1) == 1);
#else
    out[tid] = 999;
#endif
""", eq(np.full(N, 0xAAAAAAAA, dtype=np.uint64)))
run("__activemask", """
#if defined(__CUDA_ARCH__)
    out[tid] = (unsigned long long)__activemask();
#else
    out[tid] = 999;
#endif
""", eq(np.full(N, 0xFFFFFFFF, dtype=np.uint64)))

print("| case | result | detail |")
print("|---|---|---|")
for r in results:
    print(f"| `{r[0]}` | {r[1]} | {r[2]} |")
```

### B.5: CPU-only suite with GPUs hidden
`HIP_VISIBLE_DEVICES=99 CUDA_VISIBLE_DEVICES=99 python -m warp.tests -s autodetect -k cpu -m 32` (wrapped in `utils/timeit.sh warp-1.14 test`; the greeting confirms "HIP runtime or AMD GPU devices not available", only `cpu` listed): **Ran 2282 tests, OK (skipped=15)**. That matches the session 3 expectation (2282 with torch installed, 0 failures). Skips: "No suitable devices to run the test" 10, "Requires CUDA" 2, "Requires both CUDA and CPU devices" 2, "CUDA not available" 1. The `example_torch_cpu` artefact of session 3 does not appear with the `99` hiding values.

### Summary for the lock holder and Jeff
- Wave32 found no wavefront-size defect. The tile collectives (reductions, scans, radix sort in 32-thread groups, the tier-2 axis reduction with `WP_TILE_WARP_THREADS`), all of tile/*, and the Phase 1-3 sets give the gfx950 results at wavefront 32. The exceptions are the four 64 KiB-LDS tile tests (also failing on gfx942, a hardware limit, with the shim diagnostic diff proposed in the gfx942 section and confirmed here) and the two known fem isolation failures. The CPU-only suite is green (2282 OK).
- Q2: the eight listed intrinsics work at wave32. `__frsqrt_rn` is 1-ulp approximate on HIP (all AMD targets). CUDA-spelled 32-bit `*_sync` masks do not compile under HIP.
- Q3: gfx1100 numbers above, for Jeff's matrix-core ruling. No decision taken.
- Nothing was committed to the fork; no record other than this notes section and the telemetry lines changed.
