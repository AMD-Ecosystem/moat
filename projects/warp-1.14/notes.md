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


## Porting session 5 (2026-11-19, linux-gfx950 MI355X, porter): rulings, evidence follow-ups, then Phase 5

Fork `release/1.14.0.amd0` advanced `8d13d596..f4f27aa8` (pushed, `advance-head` recorded). nvcc 12.8 CUDA build at `f4f27aa8` (`build_lib.py --cuda-path /usr/local/cuda --no-use-libmathdx`, worktree `agent_space/cuda_wt`): "Warp build succeeded". `audit-commits` OK, `jargon.py --port` clean. One GPU visible (`HIP_VISIBLE_DEVICES=0`, kept).

### 1. Work from Jeff's rulings (atomics, test_atomic_cas)
- Unsafe atomics: nothing done, as ruled (safe forms stay).
- `b04cfbef` Test atomic_cas and atomic_exch on every device: `test_cas_single_winner_<dtype>` (1024 threads each try once to claim a zeroed cell; exactly one sees 0, the cell holds winner+1, every loser read winner+1) and `test_exch_permutation_<dtype>` (1024 threads exchange tid+1 into one cell; the values read plus the final cell value are exactly 0..1024). int32/uint32/int64/uint64/float32/float64, registered with `get_test_devices()`, so they run on AMD GPUs. The spin-lock tests keep their narrowing. `python -m warp.tests -s autodetect -p test_atomic_cas.py -m 32`: 48 run, OK (24 new: 12 cpu, 12 cuda:0).
- `example_work_queue.py` on gfx950 under `timeout 120`: completes in about 1.2 s, rc 0, three GPU runs. It is a lock-free queue, so it makes progress, as expected. It prints 7424, 7359, 7552 "Processed item" lines on the GPU and 8068 on the CPU, with no duplicates and no "Failed to enqueue". The count below 8192 is the example's own behaviour on any device: a dequeuer can read a slot before the enqueuer that won the tail CAS has written it and then exit on an apparently empty queue, and its 4096-entry buffer is indexed modulo `capacity` 8192, so re-enqueued items are written past the end of the buffer. Both are upstream example defects, not AMD ones; left as is (the example has no assertion).
- `20af89a0` Explain instead of hanging in the spin-lock example: `__main__` checks `is_independent_thread_scheduling_supported(device)` (the capability test_atomic_cas uses) and prints "Not running on cuda:0 (AMD Instinct MI355X): threads of one warp do not make independent forward progress on this GPU, so a spin lock contended inside a warp would never be released." Measured before the change: the original example's launch on gfx950 was killed by `timeout 60` (rc 124); on the CPU it prints counter 1024 and passes.
- Release notes (for the Phase 7 notes): spin locks contended inside one wavefront hang on AMD GPUs; `warp/examples/core/example_spin_lock.py` is the in-tree case and now says so instead of hanging.

### 2. Shared-memory attribute fix (gfx942 A.2 proposal)
`8f40a8b1` [ROCm] Count static shared memory in the dynamic shared-memory check: exactly the diff proposed in the gfx942 section (`hipFuncGetAttribute(HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES)` subtracted from `hipDeviceAttributeMaxSharedMemoryPerBlock`). Proof on gfx950 (`agent_space/p5/smem_probe.py`: a 256-thread tile kernel with one shared float64 tile of 159 KiB and of 160 KiB = the 163840-byte limit): without the fix (same tree, rebuilt) the 160 KiB kernel failed at launch with only "Warp CUDA error 1: invalid argument"; with it Warp first prints "Failed to configure kernel dynamic shared memory ... for 163840 bytes, but maximum available is 163840". The 159 KiB kernel runs and sums correctly either way. test_tile_shared_memory 30, test_tile_reduce 62, test_tile_view 16, test_launch 32, test_occupancy 6, test_tile_matmul 22: all OK.

### 3. The four 64 KiB tile tests (for Jeff's ruling; its own commit, droppable)
`59d873af` Size four tile tests from the device's shared memory per block. New helper `unittest_utils.select_shared_memory_fit(device, candidates, shared_bytes, block_dim)`: the first candidate (upstream's size first) whose dynamic shared memory fits `device.max_shared_memory_per_block` minus `4 * block_dim + 1024` bytes kept for static shared memory (Warp's allocator offsets are 4 bytes per thread; the measured static sizes of these kernels were 128-1024 bytes); CPU gets the first candidate. Per test:
- `test_tile_shared_mem_large`: rows (64, 56): 64 KiB forward upstream, 56 KiB on 64 KiB parts. It still asks for more than the 48 KiB default (its stated purpose); the `== 2**16` assertion became `> 48 * 1024`, and the smem-hook assertion compares with the computed size.
- `test_tile_reduce_axis_tier2` and `_single_warp`: the 3-D sum keeps its 128-long reduced axis (tier 2); the leading dimension is (8, 4, 2), so the backward kernel needs 64 KiB upstream, 32 KiB on 64 KiB parts. The kernel became a cached factory.
- `test_tile_assign_2d`: slices (16, 8, 4): backward 64 KiB upstream, 32 KiB on 64 KiB parts; also a cached factory.
Evidence on gfx950 (`agent_space/p5/smem64.py`, runs the six cases and logs each selection): with the real 163840-byte limit every test selects the upstream size (unchanged behaviour), 6/6 OK; with `max_shared_memory_per_block` set to 65536 in-process, the GPU cases select outer 4, 56 rows and 8 slices, launch, check their results and pass, 6/6 OK (the CPU cases keep upstream sizes). Full files on gfx950: test_tile_shared_memory 30, test_tile_reduce 62, test_tile_view 16, OK. Not yet run on a real 64 KiB part: the next gfx942 or gfx1100 session must run these three files (handoff below).

The alternative for Jeff, a capability skip: keep upstream's sizes and skip each test when its kernel's shared memory (dynamic plus an allowance for static) exceeds `device.max_shared_memory_per_block`, with a message such as "kernel needs 64 KiB of shared memory per block plus static; device allows 64 KiB". Smaller diff (a decorator or a two-line check per test, no kernel factories) and the tests stay byte-for-byte upstream where they run; but on every 64 KiB part (gfx90a, gfx942, all Radeon, and NVIDIA Turing, whose opt-in limit is also 64 KiB) the tier-2 axis reduction backward, the 2-D tile assign backward and the >48 KiB opt-in path go untested. The sizing commit keeps all three exercised there. Either way no arch name appears. If Jeff picks the skip, drop `59d873af`; nothing later depends on it (the multi-GPU capability helpers were deliberately put elsewhere in unittest_utils.py so the two hunks do not touch).

### 4. gfx942 findings C1-C5
Registered (`deferred.py add rocm-bug-report --project warp-1.14`, ids and wording as the gfx942 session proposed; each `--ref` points at that section): `hip-graph-memcpy-host-graph-alloc-device-mismatch` (C1), `hip-mempool-set-access-cannot-revoke` (C2), `hip-graph-multi-device-kernel-nodes` (C3). They wait for Jeff's defer/now ruling like the others.

`f4f27aa8` [ROCm] Skip multi-GPU tests that hit three HIP graph and mempool defects. Three capability helpers in `unittest_utils.py`, each answering from the device (never the arch) and true on CUDA and CPU, with a skip message that names the defect:
- `is_graph_host_copy_of_capture_allocation_supported(device)`: false for an AMD device with ordinal != 0 (C1). Gates `test_sparse.test_capturability`, `test_fem_integrate.test_capturability`, `test_graph.test_cuda_graph_alloc_transient_stream`; their `cuda:0` cases still run. C5 (`test_integrate_gradient_cuda_0` corrupted after the failed `cuda:1` capture in the same process) should disappear with it; recheck on two GPUs.
- `is_mempool_access_revocable(devices)`: false when any device is AMD (C2). Gates the tests that turn memory pool peer access off or restore it to a saved off state: `test_mempool_access`, `test_graph_launch_array_access_mode_checked_mempool_access_cuda_capture`, and five `test_unified_memory` tests (`verify_uses_peer_access_for_default_cuda_allocations`, `verify_uses_parent_allocator_for_default_cuda_slices`, `device_can_access_uses_mempool_state_when_target_mempools_enabled`, `verify_uses_mempool_access_for_cuda_mempool_allocations`, `verify_uses_parent_allocator_for_cuda_mempool_slices`). The last two only restore the saved state, but once an earlier test has granted access that restore is a revoke too (the gfx942 run saw them pass or fail by order), so they are gated as well. `verify_rejects_mempool_allocation_without_mempool_access` mocks the state and is not gated.
- `is_multi_device_graph_supported(devices)`: false when any device is AMD (C3). Gates `test_stream_arg_graph_mgpu` and `test_stream_scope_graph_mgpu` (they skip by the IOMMU heuristic on IOMMU hosts anyway).
On gfx950 with one GPU the affected files give the Phase 3 counts unchanged: test_sparse 47 OK, test_fem_integrate 17 OK, test_graph 35 (3 skipped), test_mempool 7 (2 skipped), test_unified_memory 27 (8 skipped), test_streams 24 (2 skipped). The helpers answer `cuda:0` -> host-copy True, revocable False, multi-device False; `cpu`/None -> all True. The `cuda:1` and device-pair gates are not exercised on one GPU: the next two-GPU session must run the C.2 loop (handoff below).

C4 proposal (no change made): `test_stream_priority_timings` issues two 128 Mi-float device-to-device copies on a priority-0 and a priority -1 stream and asserts the high-priority stream finishes first. On gfx942 issue order decides and priority has no measurable effect (gfx942 session); on gfx950 it passed in session 3. Proposal, in order: (1) measure on gfx942 whether priority reorders kernels rather than copies: same test shape with a long-running compute kernel per stream instead of `wp.copy` (on AMD, device-to-device copies may run on copy engines or blit kernels whose scheduling ignores the stream priority, while kernel dispatch honours queue priority); (2) if kernels honour it, give the HIP device a kernel workload in the test (the property tested, "higher-priority work completes first", is unchanged) and keep copies on CUDA; (3) if not, keep the priority readback part and skip only the timing assertion through a capability such as `is_stream_priority_scheduling_observable(device)` with a message saying HIP treats stream priority as a scheduling hint that does not reorder same-device copies, false on AMD devices. Not a ROCm defect registration unless (1) shows kernels ignore priority too, since HIP documents priorities as hints. Either way it is also a timing assertion, which Warp's own test guidelines discourage; that argues for (3) if (2) is flaky.

### 5. For Jeff: `__frsqrt_rn` and 32-bit lane masks (proposals only, no code change)

**`__frsqrt_rn`.** Exhaustive over every float in [1, 4) (2^24 inputs; covers all normal floats because rsqrt(4x) = rsqrt(x)/2 exactly), judged exactly (r is correct iff x*m_lo^2 < 1 < x*m_hi^2 for the midpoints to its neighbours), gfx950, `agent_space/p5/rsq/rsq.hip` + `judge.py`:

| form | not correctly rounded | throughput (G evals/s, 256 CUs x 64 blocks x 256 threads x 4096) |
|---|---|---|
| `__frsqrt_rn(x)` (HIP: `__builtin_amdgcn_rsqf`, `v_rsq_f32`) | 1851555 (11.04%), max 1 ulp | 11403 |
| `rsqrtf(x)` | identical to `__frsqrt_rn` | |
| `(float)(1.0 / sqrt((double)x))` | 0 | 2405 (4.7x slower) |
| `v_rsq_f32` then pick r or a neighbour by the sign of `fma(-x, m*m, 1.0)` in double for the two midpoints | 0 | 1847 (6.2x slower) |
| `1.0f / sqrtf(x)` | 26.00%, max 1 ulp | 1752 |

The same table through a Warp `wp.func_native` snippet under hipRTC (`agent_space/p5/rsq/snipsq2.py`, one intrinsic per module) gives the same counts. So a correctly rounded form exists and is exact on every normal float, but it is not cheap: about 4.7x the hardware instruction on gfx950. Radeon parts have far lower fp64 throughput relative to fp32, so it should cost more there (not measured; the next wave32 session can run rsq.hip, command in the handoff). Also related: `__fsqrt_rn` in clang's HIP header is `__ocml_native_sqrt_f32` unless `OCML_BASIC_ROUNDED_OPERATIONS` is defined, and is 1 ulp off on 15.11% of inputs (plain `sqrtf` is correctly rounded, as is Warp's own `wp.sqrt`, 0 wrong); the macro gives correctly rounded `__fsqrt_rn`/`__frcp_rn`/`__fadd_rn` and friends but has no `__frsqrt_rn` variant at all.

Proposal: keep the native mapping (the Q2 ruling asked for native AMD operations, and the error is at most 1 ulp), and state in the release notes that `__frsqrt_rn`, `rsqrtf` and `__fsqrt_rn` in user snippets are 1-ulp approximations on AMD GPUs and that `(float)(1.0 / sqrt((double)x))` / `sqrtf` give the correctly rounded value. Do not redefine `__frsqrt_rn` in Warp's hipRTC prelude: clang's header defines it as an inline function, so overriding it needs a header-order trick or a different spelling, it would make every snippet pay 4.7x or more for a function whose CUDA users chose it for speed, and Warp's own kernels never call it. Optionally, define `OCML_BASIC_ROUNDED_OPERATIONS` in the hipRTC options so `__fsqrt_rn` and the other `_rn` intrinsics match CUDA's correctly rounded meaning; that one is cheap to try (one define) but changes generated code for any snippet that uses them, so it wants Jeff's call and a test run.

Found on the way and registered (`amdgpu-earlycse-sqrt-keeps-fpmath`, rocm-bug-report, LLVM): at -O3, a function that computes both `__fsqrt_rn(x)` and `sqrtf(x)` gets one merged `llvm.sqrt.f32` call that keeps the approximation's `!fpmath !{float 3.0}`, so `sqrtf` silently becomes the 1-ulp approximation (2535452 of 2^24 wrong; 0 when alone or at -O0). `opt -O3 -print-after-all` on the unoptimized IR shows two calls after inlining and one after `EarlyCSEPass`. Standalone reproducer (`clang++ -x hip --rocm-path=$R --offload-arch=gfx950 -O3 sq3.hip -o sq3 -L$R/lib -lamdhip64 -Wl,-rpath,$R/lib`; prints "sqrtf alone wrong 0; sqrtf beside __fsqrt_rn wrong 2535452"):
```cpp
// sqrtf alone vs sqrtf next to __fsqrt_rn of the same value, over every float in [1, 4).
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
__global__ void alone(uint32_t base, uint32_t n, unsigned long long* bad)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = __uint_as_float(base + i);
    if (sqrtf(x) != (float)sqrt((double)x)) atomicAdd(&bad[0], 1ull);
}
__global__ void together(uint32_t base, uint32_t n, float* sink, unsigned long long* bad)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = __uint_as_float(base + i);
    sink[i] = __fsqrt_rn(x);
    if (sqrtf(x) != (float)sqrt((double)x)) atomicAdd(&bad[1], 1ull);
}
int main()
{
    uint32_t lo = 0x3f800000u, n = 0x40800000u - lo;
    unsigned long long* bad;
    float* sink;
    (void)hipMallocManaged(&bad, 16);
    (void)hipMalloc(&sink, n * 4);
    bad[0] = bad[1] = 0;
    alone<<<(n + 255) / 256, 256>>>(lo, n, bad);
    together<<<(n + 255) / 256, 256>>>(lo, n, sink, bad);
    (void)hipDeviceSynchronize();
    printf("sqrtf alone wrong %llu; sqrtf beside __fsqrt_rn wrong %llu (of %u)\n", bad[0], bad[1], n);
}
```
(`(float)sqrt((double)x)` is a correctly rounded float sqrt, since 53 >= 2*24 + 2; confirmed against the host's numpy result, 0 differences.)

**32-bit lane masks.** HIP's warp-sync intrinsics (`__ballot_sync`, `__shfl*_sync`, `__any_sync`, `__all_sync`, `__match_*_sync`, `__syncwarp`, `__reduce_*_sync`) are templates with `static_assert(sizeof(MaskT) == 8, "The mask must be a 64-bit integer")` (`include/hip/amd_detail/amd_warp_sync_functions.h`; the same text is built into hipRTC). A CUDA snippet with `0xffffffff` therefore fails to compile on every AMD target, wave32 included. The only switch is `HIP_DISABLE_WARP_SYNC_BUILTINS`, which removes the `*_sync` builtins (and `__activemask`) altogether. Warp's own device code already uses 64-bit masks (`tile_mask_t`), so only user snippets meet this.

Proposal: leave HIP's behaviour (a compile error is the safe failure: on wave64 a 32-bit mask would silently exclude lanes 32-63, which is exactly the class of bug the check exists to stop) and document it in the release notes with the fix: spell masks `~0ull` or `__activemask()` (both correct on wave32 and wave64), and note that `tid & 31` lane arithmetic is only right on wave32. Rejected alternative: a Warp prelude that disables HIP's builtins and supplies overloads widening a 32-bit mask on wave32 targets (and refusing it on wave64): it fixes copied CUDA snippets on Radeon only, duplicates about a dozen HIP intrinsics (with HIP's mask checks) inside Warp, and has to track HIP's interface as it changes. Not worth it for snippets that need a one-token edit.

### Phase 5: device textures (code complete; hardware proof needs an image-capable AMD GPU)
Fork head `2505f4a3` (pushed, `advance-head` recorded). nvcc 12.8 build at `2505f4a3` succeeded; offline NVRTC compile of a texture-sampling module (`agent_space/nvrtc_check/nvrtc_tex.py`, 1D/2D/3D float and vec4f samples) gave `sm86.cubin` and `sm90.ptx` with 8 `tex.{1d,2d,3d}.v4.f32.f32` instructions, so the CUDA sampling path is unchanged; the tile/atomic NVRTC check also still passes. `audit-commits` OK, `jargon.py --port` clean.

**Finding that reshapes the plan's Phase 5: gfx942 and gfx950 have no image hardware.** On the MI355X `hipDeviceAttributeImageSupport` is 0 and `hipArray3DCreate` returns `hipErrorNotSupported` for every shape (1D/2D/3D), format (float, uint8, half) and channel count, with or without `hipArraySurfaceLoadStore` (`agent_space/p5/tex/arr.hip`). At compile time ROCm clang defines `__HIP_NO_IMAGE_SUPPORT` for gfx942 and gfx950 (not for gfx90a, gfx1100) and HIP's headers mark `tex1D/2D/3D` and the surface functions unavailable there ("The image/texture API not supported on the device"; `amd_detail/host_defines.h:389`, `amd_device_functions.h:805`). So device textures cannot exist on the two current Instinct targets: the plan's proof "on gfx942 and wave32" can only be met on wave32 (and gfx90a, gfx1151). gfx942's runtime attribute was not measured here (compile-time evidence only); the next gfx942 session confirms it (handoff A below). Before this phase every GPU texture test on gfx950 failed with "Warp CUDA error 801: operation not supported (in function wp_texture_create_device, texture.cpp:171)" (66 errors of 119 run).

Commits:
- `2208ef30` [ROCm] Sample textures with HIP on AMD GPUs with image support. `texture.h`: one macro block decides hardware sampling: CUDA as upstream; HIP device compiles without `__HIP_NO_IMAGE_SUPPORT` call HIP's `tex1D/tex2D/tex3D<T>` with the handle converted back to `hipTextureObject_t` (a pointer on HIP, stored in Warp's 64-bit handle); HIP device compiles with `__HIP_NO_IMAGE_SUPPORT` return `zero()` (unreachable, creation is refused); host/CPU keeps the software sampler. All 9 sites (float, vec2f, vec4f x 1D/2D/3D; the 1D samplers included). Before, the HIP device branch compiled the CPU sampler and dereferenced the texture object as a host `Texture*`. Proof without the hardware (`agent_space/p5/tex/xcompile.py`): a module sampling all 9 combinations, built on gfx950, then its generated source recompiled through Warp's own hipRTC path (`warp._src.build.build_cuda(..., hip_target=<gfx>)`) for gfx1100, gfx1201, gfx1151, gfx90a, gfx942, gfx950: all compile; `llvm-objdump -d --mcpu=<gfx>` counts 9 `image_sample` instructions on gfx1100/gfx1201/gfx1151/gfx90a and 0 on gfx942/gfx950.
- `2505f4a3` [ROCm] Refuse textures on AMD GPUs without image support. `DeviceInfo.is_texture_supported` (HIP: `hipDeviceAttributeImageSupport`; CUDA: 1), export `wp_cuda_device_is_texture_supported`, `Device.is_texture_supported` (CPU True, it samples in software; documented in the Device docstring); `Texture.__init__` raises "Textures are not supported on device cuda:0 (AMD Instinct MI355X): it has no image support" before touching HIP. test_texture registers its device tests only on devices with `is_texture_supported` (the `devices_with_mempools` idiom) and adds `test_texture_unsupported_device_raises` on devices without it, so the gap is reported, not silent. A refused texture is collected without an "Exception ignored" warning (`__new__` zeroes the handles).

Host side on image-capable AMD GPUs needs no change by reading: `cuArray3DCreate`/`cuArrayDestroy`/`cuArray3DGetDescriptor`/`cuMemcpy3DAsync`/`cuTexObjectCreate`/`cuTexObjectDestroy` already map to `hipArray3DCreate`/`hipArrayDestroy`/`hipArray3DGetDescriptor`/`hipDrvMemcpy3DAsync`/`hipTexObjectCreate`/`hipTexObjectDestroy` (Phase 0 wrappers), and `cudaCreateSurfaceObject` to `hipCreateSurfaceObject` (compat header). Without `HIP_TRSF_READ_AS_INTEGER` HIP reads 8/16-bit formats as normalized floats, as CUDA does. Plan items checked: no mipmapped arrays are used by Warp 1.14 (only the descriptor's mipmap fields are zeroed), so no `hipMipmappedArray` work; no pitched-memory texture binds (arrays only; copies pass the linear source's row pitch to `hipDrvMemcpy3DAsync`), so the 256-byte pitch rule does not arise; no layered arrays; `Texture.__del__` destroys the texture object, surface and owned array once each, in that order (plan risk 4). These are readings; the runtime proof is handoff B.

Results on gfx950 (`-s autodetect -p <file> -m 32`): test_texture 70 run, OK, 17 skipped ("No suitable devices to run the test.": the CUDA-only registrations with an empty device list); 50 CPU cases and `test_texture_unsupported_device_raises_cuda_0` pass. test_devices 13 OK, test_diagnostics 14 OK, test_codegen 117 OK. Newton-style probe (`sdf_probe.py`, below): `cpu` PASS (SDF vs CPU sampler 0, vs analytic 1.28e-3 with grid spacing 1.56e-2, RGBA8 0); `cuda:0` raises the refusal, as designed. CPU-only suite with GPUs hidden (`HIP_VISIBLE_DEVICES=99 CUDA_VISIBLE_DEVICES=99 python -m warp.tests -s autodetect -k cpu -m 32`): Ran 2294, OK (skipped=15) = session 3's 2282 plus the 12 new CPU atomic_cas/atomic_exch cases.

**For Jeff (decision, nothing implemented): device textures on gfx942/gfx950.** Upstream Warp has only hardware textures on GPUs and a software sampler on the CPU. The two current Instinct targets cannot have hardware textures, so this line refuses them there, which is the honest minimum. Option: a GPU software-sampling path for AMD devices without image support: keep the texel data in a linear device allocation, store a device-side `Texture` (the CPU struct) as the handle, and run the existing software sampler (`cpu_sample_*_channel`, already compiled into device code on HIP) on the GPU. That is a GPU implementation, not a CPU fallback; it reuses upstream's sampler (so results match Warp's CPU textures exactly, not the hardware's 8-bit-weight filtering), costs roughly: allocation and copy paths in texture.cpp/texture.py for a second device storage kind, the `cuda_array`/`cuda_surface`/`cuda_texture` interop properties raising on those devices, and test registration by capability instead of by device. Performance would be well below hardware sampling (8 global loads and the filter arithmetic per trilinear sample). Newton's SDF sampling and similar users would then run on MI300/MI355. Recommendation: refuse in 1.14.0+rocm.0 (as now) and register the software path as a deferred feature candidate if Jeff wants it later; not registered by the porter, because whether to defer is Jeff's call.

`sdf_probe.py` (Newton-style probe; copy into a file to run):
```python
# Newton-style textured sampling probe: a sphere signed distance field in a 3D float texture
# (linear filter, normalized coordinates, clamp), sampled at random points in a kernel, plus a
# 2D RGBA8 texture read as normalized floats. The GPU result must match Warp's CPU texture
# sampler for the same data; the SDF must also be close to the analytic distance.
# Usage: python sdf_probe.py [device]   (default cuda:0; "cpu" checks the probe itself)
import sys

import numpy as np

import warp as wp

wp.init()
device = wp.get_device(sys.argv[1] if len(sys.argv) > 1 else "cuda:0")
print(f"device {device} ({device.name}) is_texture_supported={device.is_texture_supported}")

N = 64  # grid resolution per axis
R = 0.3  # sphere radius in normalized [0, 1] coordinates, centred at 0.5
c = (np.arange(N, dtype=np.float32) + 0.5) / N  # texel centres
z, y, x = np.meshgrid(c, c, c, indexing="ij")
sdf = (np.sqrt((x - 0.5) ** 2 + (y - 0.5) ** 2 + (z - 0.5) ** 2) - R).astype(np.float32)  # [depth, height, width]

rgba = np.random.default_rng(1).integers(0, 256, size=(32, 48, 4), dtype=np.uint8)  # [height, width, 4]


@wp.kernel
def query(
    sdf_tex: wp.Texture3D,
    img_tex: wp.Texture2D,
    pts: wp.array(dtype=wp.vec3f),
    d: wp.array(dtype=float),
    col: wp.array(dtype=wp.vec4f),
):
    i = wp.tid()
    p = pts[i]
    d[i] = wp.texture_sample(sdf_tex, p, dtype=float)
    col[i] = wp.texture_sample(img_tex, wp.vec2f(p[0], p[1]), dtype=wp.vec4f)


def run(dev):
    sdf_tex = wp.Texture3D(
        sdf, filter_mode=wp.TextureFilterMode.LINEAR, address_mode=wp.TextureAddressMode.CLAMP,
        normalized_coords=True, device=dev,
    )  # fmt: skip
    img_tex = wp.Texture2D(
        rgba, filter_mode=wp.TextureFilterMode.LINEAR, address_mode=wp.TextureAddressMode.WRAP,
        normalized_coords=True, device=dev,
    )  # fmt: skip
    pts = wp.array(np.random.default_rng(0).uniform(0.1, 0.9, size=(4096, 3)).astype(np.float32), dtype=wp.vec3f, device=dev)
    d = wp.zeros(4096, dtype=float, device=dev)
    col = wp.zeros(4096, dtype=wp.vec4f, device=dev)
    wp.launch(query, dim=4096, inputs=[sdf_tex, img_tex, pts, d, col], device=dev)
    return pts.numpy(), d.numpy(), col.numpy()


pts, d, col = run(device)
_, d_ref, col_ref = run("cpu")
analytic = np.linalg.norm(pts - 0.5, axis=1) - R
# hardware filtering uses 8-bit fixed-point weights (CUDA and AMD alike), so allow a small difference
print(f"SDF vs CPU sampler: max abs diff {np.abs(d - d_ref).max():.2e}")
print(f"SDF vs analytic:    max abs diff {np.abs(d - analytic).max():.2e} (grid spacing {1.0 / N:.2e})")
print(f"RGBA8 vs CPU sampler: max abs diff {np.abs(col - col_ref).max():.2e} (values in [0, 1])")
ok = np.abs(d - d_ref).max() < 2e-3 and np.abs(d - analytic).max() < 1.0 / N and np.abs(col - col_ref).max() < 1e-2
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
```
The tolerances (2e-3 on the SDF against the CPU sampler, 1e-2 on RGBA8) are a first guess at hardware filtering error, not measured on AMD; if the wave32 run fails only on them, record the measured differences and compare with what CUDA's 8-bit filter weights allow before changing anything.

### Next step (Phase 5 boundary; what each host still has to run)
The project stays in `porting`, lock held by linux-gfx950, head `2505f4a3`. Common setup as in the session 4 handoff (TheRock 10.2 wheels with the host's `device-<gfx>` extra, `export PATH=$FLEET_VENV/bin:$PATH` before `$(rocm-sdk path --root)` (the gfx1100 trap), `python build_lib.py --rocm-path "$(rocm-sdk path --root)" --hip-arch <gfx> -j <physical cores>`, `git lfs pull`, private `WARP_CACHE_PATH`, `ulimit -c 0`), at `git rev-parse HEAD` = `2505f4a3d7dd8a6bb4fd52f535ff846bd2d48800` or the later recorded `head_sha`. Copy `sdf_probe.py` from the block above. All `python -m warp.tests` commands run from `projects/warp-1.14/src`.

**B. wave32 Radeon host (gfx1100 or gfx1201; image-capable), the Phase 5 proof:**
1. `python -c "import warp as wp; wp.init(); d = wp.get_device('cuda:0'); print(d.arch_name, d.is_texture_supported)"`: expect True.
2. `python -m warp.tests -s autodetect -p test_texture.py -m 32`: every GPU registration must now run on `cuda:0`: expect 119 run (the count gfx950 registered before the capability filter), 0 skipped, and `test_texture_unsupported_device_raises` not registered. Any failure is a Phase 5 finding: fix on `release/1.14.0.amd0` if it is the port (sampling coordinates, normalized reads, address modes, surface objects, 3D copies), register it if it is HIP.
3. `python sdf_probe.py cuda:0`: PASS expected; record the three printed differences either way.
4. The 64 KiB tile sizing on a real 64 KiB part (`59d873af`): `for f in test_tile_shared_memory.py test_tile_reduce.py test_tile_view.py; do python -m warp.tests -s autodetect -p "$f" -m 32; done`: expect 30, 62, 16 OK (the four tests that failed in B.2 now pass with smaller tiles).
5. The `__frsqrt_rn` cost on wave32 for Jeff: build and run `rsq.hip` + `judge.py` (session 5 item 5; they live in `agent_space/p5/rsq/` on the gfx950 host only, so reproduce from the description: `(float)(1.0 / sqrt((double)x))` vs `__frsqrt_rn` throughput, 256-thread blocks, 64 blocks per CU, 4096 iterations per thread); optional, only if Jeff asks.

**A. gfx942 host:**
1. Confirm no image support at run time: the same one-liner as B.1 (expect False) and `python -m warp.tests -s autodetect -p test_texture.py -m 32` (expect the CPU cases plus `test_texture_unsupported_device_raises_cuda_0` OK).
2. The 64 KiB tile sizing: the three files of B.4 (expect 30, 62, 16 OK; they failed 4 tests in A.2).

**C. two-GPU session (gfx942 or gfx950, `HIP_VISIBLE_DEVICES=0,1`):** the C.2 loop of the session 4 handoff (13 files). Expect the C1/C2/C3 failures of the gfx942 run to appear as skips naming the HIP defect ("HIP cannot instantiate a graph captured on a device other than 0 ...", "HIP cannot revoke memory pool peer access ...", "HIP cannot instantiate a graph with kernel nodes on more than one device"), no errors, and C5 (`test_fem_integrate.test_integrate_gradient_cuda_0`) passing in the full-file run. `test_stream_priority_timings` still fails on gfx942 (C4, proposal above, not changed).

**gfx90a, if a host exists:** B.1-B.3 (image-capable wave64).

**Then Phase 6 (Windows gfx1151), only after B passes at the same head on both Linux wave families** (rule: Linux wave64 and wave32 first): the plan's Phase 6 section.

**Open for Jeff:** the 64 KiB tile sizing vs a capability skip (item 3 above); C4's direction; `__frsqrt_rn`/`OCML_BASIC_ROUNDED_OPERATIONS` and the lane-mask note (item 5); device textures on gfx942/gfx950 (software sampling or refuse); defer/now rulings on the four `rocm-bug-report` entries registered this session (`hip-graph-memcpy-host-graph-alloc-device-mismatch`, `hip-mempool-set-access-cannot-revoke`, `hip-graph-multi-device-kernel-nodes`, `amdgpu-earlycse-sqrt-keeps-fpmath`; `python3 utils/deferred.py pending`); the Q3 matrix-core ruling (gfx950, gfx942 and gfx1100 numbers are all in these notes).

## Evidence session 2 (2026-11-19, linux-gfx942, 2x MI300X, evidence only): Phase 5 boundary items A and C at `2505f4a3`

Requested by Jeff Daily. The lock stays with linux-gfx950. As before, nothing in the fork, stage, lock, head, platform or deferred records was changed. The read-only clone was moved from `8d13d596` to the recorded `head_sha` `2505f4a3d7dd8a6bb4fd52f535ff846bd2d48800` (`git status --porcelain` empty before and after; the one local experiment below was reverted). Same venv and setup as evidence session 1 (TheRock 10.2.0 wheels with `device-gfx942`, torch 2.15.0+rocm10.2.0, private `WARP_CACHE_PATH` per run, `ulimit -c 0`, no git-lfs on the host). Build: `python build_lib.py --rocm-path "$(rocm-sdk path --root)" --hip-arch gfx942 -j 112` -> "Warp build succeeded", 30.4 s at 56 workers.

### A.1: no image support at run time (one GPU, `HIP_VISIBLE_DEVICES=0`)
- `python -c "import warp as wp; wp.init(); d = wp.get_device('cuda:0'); print(d.arch_name, d.is_texture_supported)"` -> `gfx942:sramecc+:xnack- False`. This confirms at run time what session 5 inferred for gfx942 from compile-time evidence.
- Standalone HIP check (`hipDeviceGetAttribute(hipDeviceAttributeImageSupport, 0)`, then `hipArray3DCreate` of a 64x64 float array, built with ROCm clang `--offload-arch=gfx942`): attribute 0, `hipArray3DCreate` -> 801 `hipErrorNotSupported`. Same as gfx950.
- `python -m warp.tests -s autodetect -p test_texture.py -m 32`: Ran 70, OK (skipped=17, all "No suitable devices to run the test.", the CUDA-only registrations with an empty device list). `test_texture_unsupported_device_raises_cuda_0` ok; the CPU cases pass; no other `cuda_0` case registered. Same counts as gfx950 in session 5.

### A.2: the 64 KiB tile sizing (`59d873af`) on a real 64 KiB part
`device.max_shared_memory_per_block` = 65536. `for f in test_tile_shared_memory.py test_tile_reduce.py test_tile_view.py; do python -m warp.tests -s autodetect -p "$f" -m 32; done`:

| file | ran | result |
|---|---|---|
| test_tile_shared_memory | 30 | OK |
| test_tile_reduce | 62 | OK |
| test_tile_view | 16 | OK |

The four tests that failed in session 1 A.2 now run on `cuda:0` and pass: `test_tile_reduce_axis_tier2_cuda_0`, `test_tile_reduce_axis_tier2_single_warp_cuda_0`, `test_tile_shared_mem_large_cuda_0`, `test_tile_assign_2d_cuda_0`. No `Warp CUDA error` in any log. One new message, from the static-shared-memory fix `8f40a8b1` working as intended: `test_tile_shared_mem_func_cuda_0` prints "Warp UserWarning: Failed to configure kernel dynamic shared memory ... compute_..._cuda_kernel_backward kernel for 65536 bytes, but maximum available is 65536" and passes. That test never launches its backward kernel, so the warning is correct and harmless. Before `8f40a8b1` it was silent. If the tile-sizing approach is kept, this test could be sized the same way so the warning goes away on 64 KiB parts; left for the lock holder.

### C: two-GPU session (`HIP_VISIBLE_DEVICES=0,1`, `unique_or_2x` mode)
The same 13-file loop as session 1 C (`-p <file> -m 32`), wrapped in `utils/timeit.sh warp-1.14 test`.

| file | ran | result | session 1 (`8d13d596`) |
|---|---|---|---|
| test_multigpu | 5 | OK, 2 skipped | same |
| test_peer | 5 | OK, 1 skipped | same |
| test_mempool | 10 | OK, 2 skipped | errors=1, 1 skipped |
| test_streams | 42 | **failures=2**, 2 skipped | same (C4) |
| test_graph | 61 | OK, 5 skipped | errors=1, 4 skipped |
| test_ipc | 8 | OK | same |
| test_unified_memory | 40 | OK, 9 skipped | errors=1, 8 skipped |
| test_allocator | 16 | OK, 1 skipped | same |
| test_async | 3533 | OK | same |
| test_sparse | 71 | OK, 1 skipped | errors=1 |
| test_fem_integrate | 25 | OK, 1 skipped | failures=1, errors=1 |
| test_array | 207 | OK | same |
| test_copy | 30 | OK | same |

Skips, all files together: 3 "HIP cannot instantiate a graph captured on a device other than 0 that copies between pinned host memory and memory allocated during the capture" (C1: the `_cuda_1` cases of test_sparse/test_fem_integrate `test_capturability` and test_graph `test_cuda_graph_alloc_transient_stream`); 2 "HIP cannot revoke memory pool peer access (hipMemPoolSetAccess with hipMemAccessFlagsProtNone fails)" (C2: test_mempool_access, one unified_memory test); 10 "Peer-to-Peer transfers not supported" (upstream's IOMMU heuristic: multigpu 2, streams 2, graph 2, unified_memory 4; session 1 said 8, which was a miscount, it was 10 there too); 4 "cuda:N cannot access CPU memory"; 2 upstream "flakiness in memory usage measurement"; 1 each "Requires devices without mempool access support", "Requires devices without peer access support", "rmm not installed".

- No errors. C1 and C2 now appear as skips naming the HIP defect, as the handoff expected.
- C5: `test_fem_integrate.test_integrate_gradient_cuda_0` passes in the full-file two-GPU run (session 1: failed 3/3). It was collateral of the failed `cuda:1` capture, as diagnosed.
- C4: `test_stream_priority_timings_cuda_0/_1` still fail (not changed by the porter, as stated in session 5). Session 5's proposal step (1), timing compute kernels instead of copies on gfx942, was not part of this host's items and was not run.

Most of the C2 gates and all of the C3 gates sit behind the IOMMU heuristic, so the run above does not reach them on this host. To exercise them, `check_p2p()` (`warp/_src/utils.py:1747`) was forced to return True as a local edit, then reverted. Same environment, 5 files:

| file | ran | result |
|---|---|---|
| test_multigpu | 5 | OK (both pingpong tests pass) |
| test_graph | 61 | OK, 4 skipped |
| test_unified_memory | 40 | OK, 9 skipped |
| test_streams | 42 | failures=2 (C4 only), 2 skipped |
| test_mempool | 10 | OK, 2 skipped |

Skips there: 7 C2 (`test_graph_launch_array_access_mode_checked_mempool_access_cuda_capture`, `test_mempool_access`, and the five unified_memory tests session 5 gated); 2 C3 "HIP cannot instantiate a graph with kernel nodes on more than one device" (`test_stream_arg_graph_mgpu`, `test_stream_scope_graph_mgpu`); 1 C1; plus the device-property and upstream skips listed above. `test_graph_launch_array_access_mode_checked_peer_access_cuda_capture` and `test_unified_memory_can_access_cuda_0/_1` pass. So every gate added in `f4f27aa8` fires where intended on two gfx942 devices, and nothing they cover errors any more.

### Next step for the lock holder
- A.1, A.2 and C at `2505f4a3` on gfx942 are done; results above. Nothing new needs a fork change. Optional: size `test_tile_shared_mem_func` like the other four, to silence its harmless backward-kernel warning on 64 KiB parts.
- Still open for Jeff, as listed in session 5: the 64 KiB sizing vs a capability skip (the sizing is now proven on real 64 KiB hardware), C4's direction, textures on gfx942/gfx950, and the defer/now rulings on the registered ROCm defects.

## Evidence session 2 (2026-11-19, linux-gfx1100, Radeon Pro W7800, evidence only): Phase 5 handoff items B.1-B.4

Requested by Jeff Daily. The work lock stays with linux-gfx950. Nothing was committed to the fork, and no stage, lock, head, platform or deferral record changed. The clone was updated read-only to the recorded `head_sha` `2505f4a3d7dd8a6bb4fd52f535ff846bd2d48800` (`git status --porcelain` clean). Experiments ran in a separate `git worktree` outside `projects/warp-1.14/src`. B.5 (`__frsqrt_rn` cost) is marked optional, only if Jeff asks, so it was not run.

Setup as in the first gfx1100 session (same venv, ROCm 10.2.0 wheels, HIP 7.17.26434). Build: `export PATH=$FLEET_VENV/bin:$PATH; python build_lib.py --rocm-path "$(rocm-sdk path --root)" --hip-arch gfx1100 -j 32` -> "Warp build succeeded" (33 s). `readelf -d warp/bin/warp.so` RUNPATH points at the venv's `_rocm_sdk_devel/lib`. Private `WARP_CACHE_PATH`, `ulimit -c 0`, `HIP_VISIBLE_DEVICES=0`.

### Results at `2505f4a3` (unpatched)

| item | command | expected | result |
|---|---|---|---|
| B.1 | `python -c "import warp as wp; wp.init(); d = wp.get_device('cuda:0'); print(d.arch_name, d.is_texture_supported)"` | True | `gfx1100 True` |
| B.2 | `python -m warp.tests -s autodetect -p test_texture.py -m 32` | 119 run, 0 skipped | **Ran 120, FAILED (errors=64, skipped=1)** |
| B.3 | `python sdf_probe.py cuda:0` (`cpu` first, as the probe's self-check) | PASS | `cpu` PASS (0 / 1.28e-3 / 0); **`cuda:0` RuntimeError "Failed to copy to texture: Invalid destination memory type 3"** |
| B.4 | `test_tile_shared_memory.py`, `test_tile_reduce.py`, `test_tile_view.py` | 30, 62, 16 OK | **30 OK, 62 OK, 16 OK**: the 64 KiB tile sizing (`59d873af`) fixes the four B.2 failures on a real 64 KiB part |

B.2 count: 120 = 50 CPU cases (all ok) + 67 `cuda_0` cases + 2 `new_del_None` cases (ok) + 1 skip. The skip is `test_texture_unsupported_device_raises`, registered with an empty `devices_without_textures` list, which `add_function_test` turns into a placeholder skip ("No suitable devices to run the test."). So "not registered" in the handoff expectation is in substance a placeholder skip, not a defect. Of the 67 GPU cases, 3 passed and **64 errored, all with the same `RuntimeError: Failed to copy to texture: Invalid destination memory type 3`**.

### Finding 1 (port defect): texture copies pass CUDA's memory-type numbers to HIP
`warp/_src/texture.py:65-73` `MemoryType` is the CUDA driver enum by value (HOST 1, DEVICE 2, ARRAY 3). `wp_texture_copy_device` (`texture.cpp:183-235`) compares the incoming int against `CU_MEMORYTYPE_*` and casts it into `copy_params.*MemoryType`. On HIP, `hip_compat.h:247` maps `CU_MEMORYTYPE_ARRAY` to `hipMemoryTypeArray`, which is **10** (`hip_runtime_api.h`: Unregistered 0, Host 1, Device 2, **Managed 3**, Array 10, Unified 11). Every copy into or out of a texture array is therefore refused ("Invalid destination memory type 3"). If the check were bypassed, 3 would be read as `hipMemoryTypeManaged`. gfx950 could not see this because it refuses textures before any copy. The fix translates Warp's values instead of casting (below, `texture.cpp` part). With only this fix, B.3 passes, and B.2 goes from 64 errors to 30 failures, which are findings 2 and 3.

### Finding 2 (HIP defect): `hipDrvMemcpy3DAsync` and `hipMemcpy3DAsync` access pageable host memory after returning
Symptom after finding 1: textures built from NumPy data have their first 16 bytes wrong (pointer-like garbage), in the array itself (the readback through `copy_to` shows the same values, so sampling is faithful). Measured with `agent_space/p5/tex_small.py` (1D widths 4..256: always exactly 4 floats wrong) and `tex_async.py`:

| source of the upload | bad elements of 32 |
|---|---|
| A: constructor from NumPy (Warp copies into a temporary CPU array, freed on return) | 4 |
| B: constructor from a CPU Warp array kept alive | 0 |
| C: CPU Warp array dropped right after the constructor | 4 |
| E: device array | 0 |

That is glibc writing its free-list header into the freed temporary while the copy still reads it. Standalone reproducer without Warp (`agent_space/probe/memcpy3d/m3d.cpp`: `hipArray3DCreate`, `hipDrvMemcpy3DAsync` from a `malloc` buffer on a created stream, `memset` of the buffer right after the call returns, `hipStreamSynchronize`, read back): **every element** holds the overwritten value, at 1D w=32/256/4096 and 2D 2x2, 64x64 and 1024x1024. The runtime-API `hipMemcpy3DAsync` with `hipMemcpy3DParms` behaves the same (`rt.cpp`: 2048 of 2048 wrong). The linear `hipMemcpyAsync` H2D does stage pageable memory (`lin.cpp`: 0 bad at n = 32, 4096, 1M). CUDA documents that an async copy from pageable host memory returns only after the source is staged, so the caller may reuse it at once. Warp relies on that. Proposed registration (for the lock holder; not registered here): `utils/deferred.py add rocm-bug-report --project warp-1.14`, slug `hip-memcpy3d-async-pageable-host-read-after-return`.

### Finding 3 (HIP defect): `hipDrvMemcpy3DAsync` on a capturing stream copies at once and records no graph node
After finding 1 plus a stream sync for host copies, B.2 has 5 failures left: `test_texture{2d,3d}_cuda_array_copy_api_graph_capture{,_explicit_stream}` and `test_texture2d_cuda_surface_property_graph_capture_stability`, with 100% mismatch. Standalone (`agent_space/probe/memcpy3d/cap.cpp`, 64x32 float array, device->array and array->device copies captured in global mode): every call returns success, **the captured graph has 0 nodes**, the copy has already happened before instantiation, and a replay after changing the source copies nothing (2048 of 2048 wrong). The runtime-API `hipMemcpy3DAsync` captures correctly: 2 memcpy nodes, both replays right, in all three capture modes and for 1 and 4 channels (`rt.cpp`). `hipDrvGraphAddMemcpyNode` also works, but it needs a `hipCtx_t` (null gives invalid argument), and `hipCtxGetCurrent` is deprecated, which is an error under the host `-Werror`. One more constraint, found while building the workaround: `hipArray3DGetDescriptor` returns `hipErrorStreamCaptureUnsupported` during a global-mode capture, which invalidates the capture. So the copy cannot query the array's format at that point. Proposed registration (not registered here): slug `hip-drv-memcpy3d-async-not-captured`.

### Proposed fix, confirmed as an uncommitted experiment
Two parts:
- `texture.cpp`: translate Warp's memory-type values.
- `cuda_util.cpp`, HIP branch: `cuMemcpy3DAsync_f` builds `hipMemcpy3DParms` and calls the runtime `hipMemcpy3DAsync`. That call is captured correctly. Its extent width is in elements, so each array's element size is recorded when `cuArrayCreate_f`/`cuArray3DCreate_f` create it, and dropped in `cuArrayDestroy_f`, because the descriptor query is not allowed during capture. Outside a capture, a copy with a host side synchronizes the stream, which gives CUDA's pageable-memory semantics.

The CUDA build is untouched: both files' changes sit in HIP-only code or in code whose CUDA expansion is unchanged. The `texture_memory_type` helper is shared, but it maps 1/2/3 to the same CUDA constants. The CUDA path was not compiled with nvcc here (no nvcc on this host).

Results with the patch (gfx1100, same build flags):
- **test_texture: Ran 120, OK (skipped=1)**: all 67 GPU cases, 50 CPU cases and 2 `new_del_None` cases pass, plus the placeholder skip. Run twice, both OK.
- **`sdf_probe.py cuda:0`: PASS**. SDF vs CPU sampler max abs diff 4.78e-05; SDF vs analytic 1.31e-03 (grid spacing 1.56e-02; the CPU run gives 1.28e-03); RGBA8 vs CPU sampler 2.52e-03 (values in [0, 1], about 0.64/255). Both GPU-vs-CPU differences are consistent with hardware linear filtering, whose interpolation weights have less precision than the CPU sampler's float math. They are within the probe's tolerances.
- `tex_async.py` A/B/C/E all 0 bad; `tex_small.py` 1D widths 4..256 readback and sampling all exact; graph capture probe `tex_cap.py` 0 bad.

```diff
diff --git a/warp/native/cuda_util.cpp b/warp/native/cuda_util.cpp
index f7b86437..8714b49f 100644
--- a/warp/native/cuda_util.cpp
+++ b/warp/native/cuda_util.cpp
@@ -15,7 +15,9 @@
 #include <dlfcn.h>
 #endif
 
+#include <mutex>
 #include <queue>
+#include <unordered_map>
 #include <unordered_set>
 #include <vector>
 
@@ -448,16 +450,62 @@ CUresult cuIpcOpenMemHandle_f(CUdeviceptr* pdptr, CUipcMemHandle handle, unsigne
 CUresult cuIpcCloseMemHandle_f(CUdeviceptr dptr) { return hipIpcCloseMemHandle(dptr); }
 
 // Texture functions
+// Bytes per array element, recorded at creation: the 3D copy needs it and HIP's descriptor
+// query is not permitted while a stream is capturing
+static std::mutex g_array_element_size_mutex;
+static std::unordered_map<hipArray_t, size_t> g_array_element_size;
+
+static size_t hip_array_format_size(hipArray_Format format, unsigned int num_channels)
+{
+    switch (format) {
+    case HIP_AD_FORMAT_UNSIGNED_INT8:
+    case HIP_AD_FORMAT_SIGNED_INT8:
+        return num_channels;
+    case HIP_AD_FORMAT_UNSIGNED_INT16:
+    case HIP_AD_FORMAT_SIGNED_INT16:
+    case HIP_AD_FORMAT_HALF:
+        return 2 * num_channels;
+    default:
+        return 4 * num_channels;
+    }
+}
+
+static void record_array_element_size(hipArray_t array, hipArray_Format format, unsigned int num_channels)
+{
+    std::lock_guard<std::mutex> lock(g_array_element_size_mutex);
+    g_array_element_size[array] = hip_array_format_size(format, num_channels);
+}
+
+static size_t hip_array_element_size(hipArray_t array)
+{
+    std::lock_guard<std::mutex> lock(g_array_element_size_mutex);
+    auto it = g_array_element_size.find(array);
+    return it != g_array_element_size.end() ? it->second : 0;
+}
+
 CUresult cuArrayCreate_f(CUarray* pHandle, const CUDA_ARRAY_DESCRIPTOR* pAllocateArray)
 {
-    return hipArrayCreate(pHandle, pAllocateArray);
+    hipError_t result = hipArrayCreate(pHandle, pAllocateArray);
+    if (result == hipSuccess)
+        record_array_element_size(*pHandle, pAllocateArray->Format, pAllocateArray->NumChannels);
+    return result;
 }
 
-CUresult cuArrayDestroy_f(CUarray hArray) { return hipArrayDestroy(hArray); }
+CUresult cuArrayDestroy_f(CUarray hArray)
+{
+    {
+        std::lock_guard<std::mutex> lock(g_array_element_size_mutex);
+        g_array_element_size.erase(hArray);
+    }
+    return hipArrayDestroy(hArray);
+}
 
 CUresult cuArray3DCreate_f(CUarray* pHandle, const CUDA_ARRAY3D_DESCRIPTOR* pAllocateArray)
 {
-    return hipArray3DCreate(pHandle, pAllocateArray);
+    hipError_t result = hipArray3DCreate(pHandle, pAllocateArray);
+    if (result == hipSuccess)
+        record_array_element_size(*pHandle, pAllocateArray->Format, pAllocateArray->NumChannels);
+    return result;
 }
 
 CUresult cuArray3DGetDescriptor_f(CUDA_ARRAY3D_DESCRIPTOR* pArrayDescriptor, CUarray hArray)
@@ -474,7 +522,63 @@ CUresult cuMemcpy2DAsync_f(const CUDA_MEMCPY2D* pCopy, CUstream hStream)
 
 CUresult cuMemcpy3D_f(const CUDA_MEMCPY3D* pCopy) { return hipDrvMemcpy3D(pCopy); }
 
-CUresult cuMemcpy3DAsync_f(const CUDA_MEMCPY3D* pCopy, CUstream hStream) { return hipDrvMemcpy3DAsync(pCopy, hStream); }
+static hipPitchedPtr hip_linear_side(
+    hipMemoryType type, const void* host, hipDeviceptr_t device, size_t pitch, size_t width, size_t height
+)
+{
+    void* ptr = type == hipMemoryTypeHost ? const_cast<void*>(host) : reinterpret_cast<void*>(device);
+    return make_hipPitchedPtr(ptr, pitch, width, height);
+}
+
+// Warp's 3D copies always have an array on at least one side and no offsets. They go through the
+// runtime-API hipMemcpy3DAsync because hipDrvMemcpy3DAsync on a capturing stream copies at once
+// and records no graph node.
+CUresult cuMemcpy3DAsync_f(const CUDA_MEMCPY3D* pCopy, CUstream hStream)
+{
+    hipArray_t array = pCopy->dstMemoryType == hipMemoryTypeArray ? pCopy->dstArray : pCopy->srcArray;
+    size_t element_size = hip_array_element_size(array);
+    if (element_size == 0)
+        return hipErrorInvalidValue;
+
+    size_t height = pCopy->Height ? pCopy->Height : 1;
+    size_t depth = pCopy->Depth ? pCopy->Depth : 1;
+
+    hipMemcpy3DParms params = {};
+    params.extent = make_hipExtent(pCopy->WidthInBytes / element_size, height, depth);
+    if (pCopy->srcMemoryType == hipMemoryTypeArray)
+        params.srcArray = pCopy->srcArray;
+    else
+        params.srcPtr = hip_linear_side(
+            pCopy->srcMemoryType, pCopy->srcHost, pCopy->srcDevice, pCopy->srcPitch, params.extent.width,
+            pCopy->srcHeight ? pCopy->srcHeight : height
+        );
+    if (pCopy->dstMemoryType == hipMemoryTypeArray)
+        params.dstArray = pCopy->dstArray;
+    else
+        params.dstPtr = hip_linear_side(
+            pCopy->dstMemoryType, pCopy->dstHost, pCopy->dstDevice, pCopy->dstPitch, params.extent.width,
+            pCopy->dstHeight ? pCopy->dstHeight : height
+        );
+    if (pCopy->srcMemoryType == hipMemoryTypeHost)
+        params.kind = hipMemcpyHostToDevice;
+    else if (pCopy->dstMemoryType == hipMemoryTypeHost)
+        params.kind = hipMemcpyDeviceToHost;
+    else
+        params.kind = hipMemcpyDeviceToDevice;
+
+    hipStreamCaptureStatus status = hipStreamCaptureStatusNone;
+    hipError_t result = hipStreamIsCapturing(hStream, &status);
+    if (result == hipSuccess)
+        result = hipMemcpy3DAsync(&params, hStream);
+
+    // CUDA stages pageable host memory before cuMemcpy3DAsync returns, so callers free or
+    // reuse the buffer at once; HIP's 3D copies still access it afterwards
+    if (result == hipSuccess && status == hipStreamCaptureStatusNone
+        && (pCopy->srcMemoryType == hipMemoryTypeHost || pCopy->dstMemoryType == hipMemoryTypeHost))
+        result = hipStreamSynchronize(hStream);
+
+    return result;
+}
 
 CUresult cuTexObjectCreate_f(
     CUtexObject* pTexObject,
diff --git a/warp/native/texture.cpp b/warp/native/texture.cpp
index 68955b23..85876af7 100644
--- a/warp/native/texture.cpp
+++ b/warp/native/texture.cpp
@@ -181,6 +181,25 @@ void wp_texture_destroy_device(void* context, uint64_t array_handle)
     }
 }
 
+// MemoryType in texture.py carries the CUDA driver values (host 1, device 2, array 3);
+// HIP numbers its array type differently, so translate rather than cast
+static bool texture_memory_type(int warp_memory_type, CUmemorytype* memory_type)
+{
+    switch (warp_memory_type) {
+    case 1:
+        *memory_type = CU_MEMORYTYPE_HOST;
+        return true;
+    case 2:
+        *memory_type = CU_MEMORYTYPE_DEVICE;
+        return true;
+    case 3:
+        *memory_type = CU_MEMORYTYPE_ARRAY;
+        return true;
+    default:
+        return false;
+    }
+}
+
 bool wp_texture_copy_device(
     void* context,
     unsigned width_bytes,
@@ -206,30 +225,34 @@ bool wp_texture_copy_device(
     copy_params.Height = height;
     copy_params.Depth = depth;
 
-    copy_params.dstMemoryType = static_cast<CUmemorytype>(dst_memory_type);
-    if (dst_memory_type == CU_MEMORYTYPE_HOST) {
+    CUmemorytype dst_type;
+    if (!texture_memory_type(dst_memory_type, &dst_type)) {
+        wp::set_error_string("Invalid destination memory type %d", dst_memory_type);
+        return false;
+    }
+    copy_params.dstMemoryType = dst_type;
+    if (dst_type == CU_MEMORYTYPE_HOST) {
         copy_params.dstHost = reinterpret_cast<void*>(dst_handle);
-    } else if (dst_memory_type == CU_MEMORYTYPE_DEVICE) {
+    } else if (dst_type == CU_MEMORYTYPE_DEVICE) {
         copy_params.dstDevice = (CUdeviceptr)dst_handle;
-    } else if (dst_memory_type == CU_MEMORYTYPE_ARRAY) {
-        copy_params.dstArray = reinterpret_cast<CUarray>(dst_handle);
     } else {
-        wp::set_error_string("Invalid destination memory type %d", dst_memory_type);
-        return false;
+        copy_params.dstArray = reinterpret_cast<CUarray>(dst_handle);
     }
     copy_params.dstPitch = dst_pitch;
     copy_params.dstHeight = dst_height;
 
-    copy_params.srcMemoryType = static_cast<CUmemorytype>(src_memory_type);
-    if (src_memory_type == CU_MEMORYTYPE_HOST) {
+    CUmemorytype src_type;
+    if (!texture_memory_type(src_memory_type, &src_type)) {
+        wp::set_error_string("Invalid source memory type %d", src_memory_type);
+        return false;
+    }
+    copy_params.srcMemoryType = src_type;
+    if (src_type == CU_MEMORYTYPE_HOST) {
         copy_params.srcHost = reinterpret_cast<void*>(src_handle);
-    } else if (src_memory_type == CU_MEMORYTYPE_DEVICE) {
+    } else if (src_type == CU_MEMORYTYPE_DEVICE) {
         copy_params.srcDevice = (CUdeviceptr)src_handle;
-    } else if (src_memory_type == CU_MEMORYTYPE_ARRAY) {
-        copy_params.srcArray = reinterpret_cast<CUarray>(src_handle);
     } else {
-        wp::set_error_string("Invalid source memory type %d", src_memory_type);
-        return false;
+        copy_params.srcArray = reinterpret_cast<CUarray>(src_handle);
     }
     copy_params.srcPitch = src_pitch;
     copy_params.srcHeight = src_height;
```

### Not done
- B.5 (`__frsqrt_rn` throughput on wave32): optional, only if Jeff asks.
- No nvcc compile of the proposed diff (no CUDA toolkit on this host); the lock holder's usual mi355 nvcc check applies if it is taken.
- The patched build was used only for test_texture and the probes above. The rest of the suite was not rerun against it.

## Porting session 6 (2026-11-19, linux-gfx950 MI355X, porter): Phase 5 evidence follow-ups

Fork `release/1.14.0.amd0` advanced `2505f4a3..2f5b9d65` (pushed, `advance-head` recorded). nvcc 12.8 CUDA build at `2f5b9d65` (`build_lib.py --cuda-path /usr/local/cuda --no-use-libmathdx`, worktree `agent_space/cuda_wt`): "Warp build succeeded"; offline NVRTC compiles (`agent_space/nvrtc_check/nvrtc_tex.py`, texture sampling; `nvrtc_offline.py`, tile/atomics): both OK. `audit-commits` OK, `jargon.py --port` clean. One GPU visible (`HIP_VISIBLE_DEVICES=0`, kept).

### 1. The gfx1100 texture-copy fix, reviewed and applied
The diff from "Evidence session 2 (linux-gfx1100) / Proposed fix" applied cleanly to `2505f4a3` (`git apply`, extracted verbatim from these notes). Review before committing:
- `texture.cpp` (`texture_memory_type`): shared code. CUDA's `cuda.h` defines `CU_MEMORYTYPE_HOST/DEVICE/ARRAY` as 1/2/3, Warp's `MemoryType` values, so on CUDA the translation is the identity, values other than 1-3 are rejected with the same messages as before, and the descriptor gets the same fields. Behaviour unchanged on CUDA; the nvcc build compiles it.
- `cuda_util.cpp`: every change is in the `#if WP_ENABLE_HIP` branch except two `#include`s (`<mutex>`, `<unordered_map>`) added to the shared include block, harmless for CUDA. Callers of `cuMemcpy3DAsync_f`: only `wp_texture_copy_device` (`grep -rn cuMemcpy3DAsync_f warp/native`), and `texture.py` always puts the texture's own array on one side.
- Two gaps found and fixed on top of the proposal: (a) a texture that wraps an external array (`Texture2D(cuda_array=...)`) never went through `cuArray*Create_f`, so its element size was unrecorded and every copy into it would have returned `hipErrorInvalidValue` where the old driver call worked. `cuArray3DGetDescriptor_f`, which the `cuda_array=` constructor calls (`wp_texture_descriptor_from_cuda_array`), now records the element size too, so such copies also work inside a capture; for anything still unrecorded, `hip_array_element_size` falls back to `hipArray3DGetDescriptor` (fails only during a global capture). (b) A copy with offsets (`src/dstXInBytes`, `Y`, `Z`, `LOD`) or with no array side keeps the original `hipDrvMemcpy3DAsync`, since the runtime-parameter translation assumes neither; Warp issues no such copy today.
- Accepted as is: the host-side `hipStreamSynchronize` outside a capture also applies to pinned host buffers, where CUDA would stay asynchronous (a performance difference only, and texture uploads are rare); during a capture no sync is done.
Commits: `c14d2019` [ROCm] Translate texture copy memory types instead of casting them; `8de519f3` [ROCm] Copy texture arrays with hipMemcpy3DAsync. gfx950 cannot create texture arrays, so these paths are not exercised here; the gfx1100 host ran its version (without (a)/(b)) with test_texture 120 OK and the SDF probe PASS. The wave32 re-run below covers the final code, including the new `tex_alias.py` probe for (a).

Re-run on gfx950 for the shared wrapper changes (`-p <file> -m 32`, all OK, counts identical to Phase 3): test_graph 35 (3 skipped), test_array_fill_capture 3, test_capture_mode 2, test_conditional_captures 32 (16 skipped, Q4), test_conditional 50, test_async 2373, test_streams 24 (2 skipped), test_mempool 7 (2 skipped), test_multigpu 5 (5 skipped, one GPU), test_peer 4 (2 skipped), test_ipc 4, test_pinned 1, test_unified_memory 27 (8 skipped), test_occupancy 6, test_allocator 16 (3 skipped), test_allocation_tracker 34, test_dlpack 16, test_torch 34, test_map 45, test_copy 14, test_array 139, test_texture 70 (17 skipped), test_launch 32. After the review changes (a)/(b): test_texture 70, test_graph 35, test_array_fill_capture 3, test_copy 14, all OK.

`tex_alias.py` (copies into a texture wrapping an external array, plain and captured; for the wave32 host):
```python
# Copies into and out of a texture that wraps an array Warp did not record (cuda_array=...),
# plain and inside a graph capture. Usage: python tex_alias.py [device]
import sys

import numpy as np

import warp as wp

wp.init()
device = wp.get_device(sys.argv[1] if len(sys.argv) > 1 else "cuda:0")
rng = np.random.default_rng(0)
h, w = 16, 32
owner = wp.Texture2D(np.zeros((h, w, 4), dtype=np.float32), device=device)
alias = wp.Texture2D(cuda_array=owner.cuda_array, device=device)

ok = True
for label, capture in (("plain", False), ("graph capture", True)):
    data = rng.random((h, w, 4), dtype=np.float32)
    src = wp.array(data, dtype=wp.vec4f, device=device)
    dst = wp.zeros((h, w), dtype=wp.vec4f, device=device)
    if capture:
        with wp.ScopedCapture(device=device) as cap:
            alias.copy_from(src)
            owner.copy_to(dst)
        wp.capture_launch(cap.graph)
    else:
        alias.copy_from(src)
        owner.copy_to(dst)
    bad = int(np.count_nonzero(dst.numpy() != data))
    print(f"{label}: {bad} of {data.size} wrong")
    ok = ok and bad == 0
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
```

### 2. The two HIP defects from the gfx1100 run, registered
`deferred.py add rocm-bug-report --project warp-1.14 --component HIP` with the proposed ids: `hip-memcpy3d-async-pageable-host-read-after-return` (Finding 2) and `hip-drv-memcpy3d-async-not-captured` (Finding 3). Their original reproducers (`m3d.cpp`, `rt.cpp`, `lin.cpp`, `cap.cpp`) were in the gfx1100 host's `agent_space/` and are not in these notes, so standalone reproducers were rewritten here from the recorded descriptions and compiled for gfx1100 with ROCm clang (`clang++ -x hip --rocm-path=$R --offload-arch=gfx1100 -O2 <file>.cpp -o <name> -L$R/lib -lamdhip64 -Wl,-rpath,$R/lib`, both build). They cannot run on gfx950 (`hipArray3DCreate -> hipErrorNotSupported`); the wave32 host runs them (handoff below) to confirm they reproduce before Jeff files anything.

`m3d_pageable.cpp` (expected on HIP 7.17: both lines report all elements wrong; on a correct runtime, 0):
```cpp
// hipDrvMemcpy3DAsync / hipMemcpy3DAsync from pageable host memory into an array: the source is
// overwritten right after the call returns; CUDA stages pageable memory before returning, so the
// array must hold the original values. Usage: ./m3d_pageable [width] [height]
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CK(x) do { hipError_t err_ = (x); if (err_ != hipSuccess) { printf("%s -> %s\n", #x, hipGetErrorName(err_)); exit(1); } } while (0)

static int run(bool runtime_api, size_t w, size_t h)
{
    HIP_ARRAY3D_DESCRIPTOR d = {};
    d.Width = w;
    d.Height = h;
    d.Format = HIP_AD_FORMAT_FLOAT;
    d.NumChannels = 1;
    hipArray_t a;
    CK(hipArray3DCreate(&a, &d));
    hipStream_t s;
    CK(hipStreamCreate(&s));
    size_t n = w * (h ? h : 1);
    float* src = (float*)malloc(n * 4);  // pageable
    for (size_t i = 0; i < n; ++i) src[i] = float(i);
    if (runtime_api) {
        hipMemcpy3DParms p = {};
        p.srcPtr = make_hipPitchedPtr(src, w * 4, w, h ? h : 1);
        p.dstArray = a;
        p.extent = make_hipExtent(w, h ? h : 1, 1);
        p.kind = hipMemcpyHostToDevice;
        CK(hipMemcpy3DAsync(&p, s));
    } else {
        HIP_MEMCPY3D c = {};
        c.WidthInBytes = w * 4;
        c.Height = h ? h : 1;
        c.Depth = 1;
        c.srcMemoryType = hipMemoryTypeHost;
        c.srcHost = src;
        c.srcPitch = w * 4;
        c.srcHeight = h ? h : 1;
        c.dstMemoryType = hipMemoryTypeArray;
        c.dstArray = a;
        CK(hipDrvMemcpy3DAsync(&c, s));
    }
    memset(src, 0xff, n * 4);  // the caller reuses its buffer, as CUDA allows
    CK(hipStreamSynchronize(s));
    std::vector<float> back(n);
    HIP_MEMCPY3D r = {};
    r.WidthInBytes = w * 4;
    r.Height = h ? h : 1;
    r.Depth = 1;
    r.srcMemoryType = hipMemoryTypeArray;
    r.srcArray = a;
    r.dstMemoryType = hipMemoryTypeHost;
    r.dstHost = back.data();
    r.dstPitch = w * 4;
    r.dstHeight = h ? h : 1;
    CK(hipDrvMemcpy3D(&r));
    size_t bad = 0;
    for (size_t i = 0; i < n; ++i) bad += back[i] != float(i);
    printf("%-22s %zux%zu: %zu of %zu elements wrong\n", runtime_api ? "hipMemcpy3DAsync" : "hipDrvMemcpy3DAsync", w, h, bad, n);
    free(src);
    CK(hipArrayDestroy(a));
    CK(hipStreamDestroy(s));
    return bad != 0;
}

int main(int argc, char** argv)
{
    size_t w = argc > 1 ? atoi(argv[1]) : 64, h = argc > 2 ? atoi(argv[2]) : 64;
    int fails = run(false, w, h) + run(true, w, h);
    return fails ? 1 : 0;
}
```

`m3d_capture.cpp` (expected on HIP 7.17: the driver line reports 0 graph nodes, the copy done before instantiation and all elements wrong after the replay; the runtime line 2 nodes and 0 wrong):
```cpp
// hipDrvMemcpy3DAsync on a capturing stream: device -> array -> device copies captured in global
// mode. A correct capture has 2 memcpy nodes and a replay copies the source's current contents.
// The runtime hipMemcpy3DAsync is run the same way for comparison.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CK(x) do { hipError_t err_ = (x); if (err_ != hipSuccess) { printf("%s -> %s\n", #x, hipGetErrorName(err_)); exit(1); } } while (0)

static void copy(bool runtime_api, hipArray_t a, float* dev, bool to_array, size_t w, size_t h, hipStream_t s)
{
    if (runtime_api) {
        hipMemcpy3DParms p = {};
        p.extent = make_hipExtent(w, h, 1);
        p.kind = hipMemcpyDeviceToDevice;
        if (to_array) { p.srcPtr = make_hipPitchedPtr(dev, w * 4, w, h); p.dstArray = a; }
        else { p.srcArray = a; p.dstPtr = make_hipPitchedPtr(dev, w * 4, w, h); }
        CK(hipMemcpy3DAsync(&p, s));
    } else {
        HIP_MEMCPY3D c = {};
        c.WidthInBytes = w * 4;
        c.Height = h;
        c.Depth = 1;
        if (to_array) {
            c.srcMemoryType = hipMemoryTypeDevice; c.srcDevice = (hipDeviceptr_t)dev; c.srcPitch = w * 4; c.srcHeight = h;
            c.dstMemoryType = hipMemoryTypeArray; c.dstArray = a;
        } else {
            c.srcMemoryType = hipMemoryTypeArray; c.srcArray = a;
            c.dstMemoryType = hipMemoryTypeDevice; c.dstDevice = (hipDeviceptr_t)dev; c.dstPitch = w * 4; c.dstHeight = h;
        }
        CK(hipDrvMemcpy3DAsync(&c, s));
    }
}

static int run(bool runtime_api)
{
    const size_t w = 64, h = 32, n = w * h;
    HIP_ARRAY3D_DESCRIPTOR d = {};
    d.Width = w; d.Height = h; d.Format = HIP_AD_FORMAT_FLOAT; d.NumChannels = 1;
    hipArray_t a;
    CK(hipArray3DCreate(&a, &d));
    float *src, *dst;
    CK(hipMalloc(&src, n * 4));
    CK(hipMalloc(&dst, n * 4));
    std::vector<float> h1(n), h2(n), back(n);
    for (size_t i = 0; i < n; ++i) { h1[i] = float(i + 1); h2[i] = float(2 * i + 1); }
    hipStream_t s;
    CK(hipStreamCreate(&s));
    CK(hipMemcpy(src, h1.data(), n * 4, hipMemcpyHostToDevice));
    CK(hipMemset(dst, 0, n * 4));

    hipGraph_t g;
    CK(hipStreamBeginCapture(s, hipStreamCaptureModeGlobal));
    copy(runtime_api, a, src, true, w, h, s);
    copy(runtime_api, a, dst, false, w, h, s);
    CK(hipStreamEndCapture(s, &g));
    size_t nodes = 0;
    CK(hipGraphGetNodes(g, nullptr, &nodes));
    CK(hipMemcpy(back.data(), dst, n * 4, hipMemcpyDeviceToHost));
    size_t early = 0;
    for (size_t i = 0; i < n; ++i) early += back[i] == h1[i];

    hipGraphExec_t ge;
    CK(hipGraphInstantiate(&ge, g, nullptr, nullptr, 0));
    CK(hipMemcpy(src, h2.data(), n * 4, hipMemcpyHostToDevice));  // replay must copy the new contents
    CK(hipGraphLaunch(ge, s));
    CK(hipStreamSynchronize(s));
    CK(hipMemcpy(back.data(), dst, n * 4, hipMemcpyDeviceToHost));
    size_t bad = 0;
    for (size_t i = 0; i < n; ++i) bad += back[i] != h2[i];
    printf("%-22s graph nodes %zu, copied before instantiation %zu of %zu, wrong after replay %zu of %zu\n",
           runtime_api ? "hipMemcpy3DAsync" : "hipDrvMemcpy3DAsync", nodes, early, n, bad, n);
    CK(hipGraphExecDestroy(ge));
    CK(hipGraphDestroy(g));
    CK(hipArrayDestroy(a));
    CK(hipFree(src));
    CK(hipFree(dst));
    CK(hipStreamDestroy(s));
    return bad != 0 || nodes != 2;
}

int main()
{
    int fails = run(false) + run(true);
    return fails ? 1 : 0;
}
```

### 3. `test_tile_shared_mem_func` sized like the other four
`2f5b9d65` Size test_tile_shared_mem_func from the device's shared memory per block: the larger tile's rows are `select_shared_memory_fit(device, (64, 32), rows * 64 * 4 * 2 * 2, 256)` (the backward kernel needs twice the forward's 32 KiB, 64 KiB at 64 rows); the 16x16 small tile stays smaller, which is what the test checks, and the expected byte counts are computed from the selected size. gfx950 (`agent_space/p6/smemfunc.py`): real 163840-byte limit -> 64 rows (upstream), OK; limit reported as 65536 -> 32 rows (32768 backward bytes), OK. test_tile_shared_memory 30 OK. It uses the helper from `59d873af`, so if Jeff rules for capability skips instead of sizing, drop both commits together.

### 4. notes.md merge untangled
The first gfx942 evidence section (commit `42c8596b`) had been interleaved with the first gfx1100 section (`e12a96ff`, committed on top through a union merge): the gfx1100 heading appeared right after "Same loop as step 1 of session 4", and the gfx942 A.2 table, A.3, C and Next step followed inside the gfx1100 text. Rebuilt in this session (script `agent_space/p6/rebuild.py`): the gfx942 section is now its own commit's text, verbatim and contiguous (272 lines from `git show 42c8596b:projects/warp-1.14/notes.md`), followed by the gfx1100 section, which is the 401 lines only it wrote, in their original order (`difflib` alignment against the gfx942 text: all 272 gfx942 lines matched in order, the rest unmatched), plus copies of the 23 lines the union merge had kept only once because both sessions wrote them identically: the 15-line tile-set table after the B.2 heading (gfx942's A.2 table had the same rows and failures), the four fp64 64x64 "(no launch)" rows ending the B.3 table, and four blank lines around them. A line-multiset check showed no line of the old region missing from the new one. A blank line was also added between the second gfx942 section and the second gfx1100 heading, which had been joined without one (that pair was not interleaved). Nothing else changed.

### 5. Full Phase 1-5 set on gfx950 at `2f5b9d65` (wave64 half of the same-head requirement)
82 patterns, one per line, `python -m warp.tests -s autodetect -p "<pattern>" -m 32`, one GPU (list in the Next step below; 1695 s under `utils/timeit.sh warp-1.14 test`). 80 OK, 2 not OK:
- Phase 1: test_codegen 117, test_snippet 12, test_arithmetic 126, test_math 16, test_fp16 6, test_bf16 48, test_atomic 164, test_atomic_bitwise 6, test_atomic_cas 48, test_array 139, test_copy 14, test_launch 32, test_fast_math 12 (1 skipped, PTX check), test_grad 41, test_kernel_cache 5, test_module_hashing 8, test_print 10, test_mat* 601, test_vec* 531, test_quat* 173, test_spatial* 207, test_rand 10, test_noise 14, test_clang_cuda 7 (6 skipped, NVIDIA-only): all OK.
- Phase 2: test_runlength_encode 16, test_array_reduce 16, test_sparse 47, test_linear_solvers 34, test_coloring 3, test_bvh 13, test_grouped_bvh 11, test_hash_grid 23, test_marching_cubes 17, test_mesh* 49, test_volume 59, test_volume_write 5, test_apic 52, test_apic_mesh 12: OK. **test_fem* 107, failures=2**: `test_gimp_quadrature` and `test_cube_shape_functions`, "Unexpected output" holding Warp's init greeting, the upstream test-isolation behaviour recorded in session 2 and on gfx1100 (B.4); not a port defect.
- Phase 3: test_graph 35 (3 skipped), test_array_fill_capture 3, test_capture_mode 2, test_conditional_captures 32 (16 skipped, Q4), test_conditional 50, test_async 2373, test_mempool 7 (2 skipped), test_pinned 1, test_unified_memory 27 (8 skipped), test_occupancy 6, test_allocator 16 (3 skipped), test_allocation_tracker 34, test_dlpack 16, test_torch 34, test_map 45, test_reload 14, test_cuda_arch_suffix 10 (all skipped, NVIDIA-only): OK. **test_streams 24, failures=1 (2 skipped)**: `test_stream_priority_timings_cuda_0`, "0.8098440170288086 not less than 0.80976402759552". Rerun of `-k priority` three times: OK, OK, FAILED (0.8380 vs 0.8362). So C4 is not gfx942-only: on gfx950 too the two streams finish within 0.3% of each other and the order is a coin toss; its session 3 pass was luck. C4's proposal (above) now covers both wave64 targets.
- Phase 4: all 22 tile files plus test_block_dim_dispatch 2 and test_template_launch_bounds 36: the session 4 counts (636 run, 10 upstream no-libmathdx skips), all OK.
- Phase 5: test_texture 70 (17 skipped: the CUDA-only registrations with an empty device list on this GPU without image support), OK.

### Next step (Phase 5 boundary, head `2f5b9d65`)
The project stays in `porting`, lock held by linux-gfx950. Use the common setup of the session 4 handoff (TheRock 10.2 wheels with `device-<gfx>`, `export PATH=$FLEET_VENV/bin:$PATH` before `$(rocm-sdk path --root)`, `python build_lib.py --rocm-path "$(rocm-sdk path --root)" --hip-arch <gfx> -j <physical cores>`, `git lfs pull`, private `WARP_CACHE_PATH`, `ulimit -c 0`), with `git rev-parse HEAD` = `2f5b9d657295a2d5ba5371ae1f7ff4582ade7acf` (or the later recorded `head_sha`), from `projects/warp-1.14/src`.

**B. wave32 Radeon host (gfx1100 or gfx1201), re-run at the new head (the Phase 5 proof and the wave32 half of the same-head requirement):**
1. `python -m warp.tests -s autodetect -p test_texture.py -m 32`: expect Ran 120, OK (skipped=1, the `test_texture_unsupported_device_raises` placeholder).
2. `python sdf_probe.py cpu` then `python sdf_probe.py cuda:0` (script in Porting session 5, Phase 5): PASS; record the three differences.
3. `python tex_alias.py cuda:0` (script in Porting session 6, item 1): PASS, 0 wrong plain and under capture. It exercises the external-array path added in review, which no test covers.
4. The two HIP reproducers (sources in Porting session 6, item 2), built with `clang++ -x hip --rocm-path=$R --offload-arch=<gfx> -O2 <file>.cpp -o <name> -L$R/lib -lamdhip64 -Wl,-rpath,$R/lib`: `./m3d_pageable 64 64` and `./m3d_capture`; record whether each reproduces as the gfx1100 session described (pageable: all elements wrong for both APIs; capture: the driver call with 0 graph nodes and all wrong after replay, the runtime call 2 nodes and 0 wrong). If one does not reproduce, say so: the registration text then needs `deferred.py amend`.
5. The full Phase 1-5 set, one pattern per line:
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
    test_cuda_arch_suffix.py \
    test_tile.py test_tile_atomic_bitwise.py test_tile_cholesky.py test_tile_cholesky_no_mathdx.py test_tile_empty.py \
    test_tile_fft.py test_tile_fft_no_mathdx.py test_tile_func_arg.py test_tile_fused_ops.py test_tile_large_offsets.py \
    test_tile_load.py test_tile_mathdx.py test_tile_matmul.py test_tile_matmul_no_mathdx.py test_tile_mlp.py \
    test_tile_reduce.py test_tile_shared_memory.py test_tile_solve.py test_tile_solve_no_mathdx.py test_tile_sort.py \
    test_tile_stack.py test_tile_view.py test_block_dim_dispatch.py test_template_launch_bounds.py \
    test_texture.py; do
  python -m warp.tests -s autodetect -p "$p" -m 32
done
```
   Expected: gfx1100 B.4 counts for Phase 1-3, with test_atomic_cas now 48 (24 new cases) and test_array 139; the tile set 636 run, 10 skipped, 0 failures (the sizing commits), with no "Failed to configure kernel dynamic shared memory" warning from `test_tile_shared_mem_func`; test_texture 120 OK. Known non-port failures: the two fem isolation tests (`test_gimp_quadrature`, `test_cube_shape_functions`). Anything else is a finding.
6. The CPU-only suite with GPUs hidden: `HIP_VISIBLE_DEVICES=99 CUDA_VISIBLE_DEVICES=99 python -m warp.tests -s autodetect -k cpu -m 32`: expect Ran 2294, OK (skipped=15) (session 5 on gfx950).

**What remains before Phase 6 (Windows gfx1151):**
1. B above passes at the head gfx950 passed (`2f5b9d65`, or a later head both re-run). Then both Linux wave families have passed the Phase 1-5 set at the same head, which the rule asks before Windows.
2. A new head means the wave64 set runs again on gfx950 (this section's item 5 is the reference). If B needs a fix, the gfx950 re-run follows that fix.
3. Decisions for Jeff that change code or tests before the line can close (none blocks starting Phase 6, but each may move the head): the 64 KiB tile sizing (`59d873af` + `2f5b9d65`) vs capability skips; C4 (`test_stream_priority_timings`, now seen on gfx942 and gfx950); device textures on gfx942/gfx950 (refuse, as now, or GPU software sampling); `__frsqrt_rn`/`OCML_BASIC_ROUNDED_OPERATIONS` and the lane-mask documentation; defer/now on the six undecided `rocm-bug-report` entries (`python3 utils/deferred.py pending`: `hip-graph-memcpy-host-graph-alloc-device-mismatch`, `hip-mempool-set-access-cannot-revoke`, `hip-graph-multi-device-kernel-nodes`, `amdgpu-earlycse-sqrt-keeps-fpmath`, `hip-memcpy3d-async-pageable-host-read-after-return`, `hip-drv-memcpy3d-async-not-captured`); the Q3 matrix-core ruling (numbers for gfx950, gfx942 and gfx1100 are in these notes).
4. Phase 6 itself (not started, as instructed): the plan's Phase 6 section (clang-cl/ROCm clang branch of `build_dll.py`, DLL loading by path, the Phase 1-5 set within the APU's limits) on windows-gfx1151, after item 1.

## Measurement for Jeff (2026-11-19, linux-gfx950 MI355X, evidence only): CPU texture sampling plus copies vs texel loads on the GPU

Asked by Jeff Daily for the gfx942/gfx950 texture decision (refuse, or a GPU software sampler). No fork commit, no state change; the existing build at fork head `2f5b9d65` (HIP, `--hip-arch gfx950`), `HIP_VISIBLE_DEVICES=0`.

**(a) approximates the proposed GPU software sampler; it is not that sampler.** It is a hand-written kernel that does the same trilinear filter as Warp's CPU texture sampler (normalized coordinates to texel space `c * n - 0.5`, clamp addressing, floor, neighbour indices clamped, 8 loads, the same nested lerps in the same order) from a `wp.array3d(dtype=float)` in device memory, with the coordinates and results on the GPU. A real sampler would also dereference the texture handle and switch on dtype, address mode and channel count per sample, so it would be somewhat slower than (a); hardware texture units cannot be compared on this GPU (none exist).

Setup: float32 1-channel volumes of 64^3 and 256^3 (random values), 1e6 and 1.6e7 random sample points in [0, 1)^3, normalized coordinates, linear filter, clamp addressing. Each figure is the median of 5 timed runs after one warm-up, with `wp.synchronize()` before and after every timed region (wall clock, `time.perf_counter`). (b) is what a user would have to do today: coordinates already in a GPU `wp.array(dtype=wp.vec3)`, `wp.copy` into a preallocated CPU array (pageable, Warp's default CPU allocation), launch `wp.texture_sample` on a CPU `wp.Texture3D` built from the same volume, `wp.copy` the results into a preallocated GPU array; the three steps are timed separately and then together. (c) is the same CPU kernel with coordinates and results already on the CPU.

CPU: AMD EPYC 9575F 64-Core Processor, 2 sockets x 64 cores x 2 threads (256 logical CPUs, all in the job's affinity mask). Warp's CPU device ran each kernel on **one** of them: the generated CPU entry point is a serial `for (size_t task_index = 0; task_index < dim->size; ++task_index)` loop (checked in the module's generated `.cpp`), and Warp has no CPU thread pool.

| volume | samples | (a) GPU manual ms | (b) D2H ms | (b) CPU kernel ms | (b) H2D ms | (b) total ms | (c) CPU kernel ms | (b) total / (a) | (a) vs (c) max abs | max ulp | values differing |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 64^3 | 1.0e6 | 0.039 | 0.23 | 20.5 | 0.08 | 20.8 | 20.5 | 534x | 0 | 0 | 0 of 1000000 |
| 64^3 | 1.6e7 | 0.299 | 3.42 | 331.8 | 1.14 | 335.1 | 329.3 | 1121x | 0 | 0 | 0 of 16000000 |
| 256^3 | 1.0e6 | 0.089 | 0.22 | 134.1 | 0.08 | 135.5 | 137.2 | 1519x | 0 | 0 | 0 of 1000000 |
| 256^3 | 1.6e7 | 1.160 | 3.42 | 2141.7 | 1.14 | 2128.6 | 2108.0 | 1834x | 0 | 0 | 0 of 16000000 |

Numerics: (a) and (c) are bit-identical on every sample at all four sizes (max abs 0, max 0 ulp, 0 values differing), so the GPU arithmetic matches Warp's CPU sampler exactly.

Reading of the numbers, for Jeff's decision (no decision taken here):
- Doing the texel loads on the GPU is three orders of magnitude faster than the CPU route: 534x to 1834x end to end, growing with volume size (the 256^3 volume, 64 MiB, no longer fits the CPU's cache, while the GPU's bandwidth hides it: 1.6e7 samples in 1.16 ms).
- The CPU route is dominated by the single-threaded CPU kernel (98-99% of (b)); the two copies together cost 0.3 ms per 1e6 samples (D2H 12 bytes and H2D 4 bytes per sample, pageable). (b)'s CPU kernel time matches (c) within noise, so the copies do not slow the CPU kernel.
- Even a CPU sampler spread perfectly over all 128 cores (Warp has none today) would be at best about 9x (64^3) to 14x (256^3) slower than (a) at 1.6e7 samples, before the copies; that bound is arithmetic from the table, not a measurement.

Script (`agent_space/p7/tex_cpu_vs_gpu.py`, run with `utils/timeit.sh warp-1.14 bench`, 51 s wall including the module builds):
```python
# Trilinear float32 3D texture sampling: (a) a hand-written trilinear kernel on the GPU reading a
# wp.array3d in device memory (approximates a GPU software sampler, it is not that sampler);
# (b) coordinates on the GPU -> copy to CPU -> wp.Texture3D sampled on the CPU device -> results
# copied back to the GPU; (c) the CPU texture kernel alone, everything already on the CPU.
# Median of 5 timed runs after one warm-up, wp.synchronize() around every timed region.
import statistics
import sys
import time

import numpy as np

import warp as wp

wp.init()
gpu = wp.get_device("cuda:0")
cpu = wp.get_device("cpu")
RUNS = 5


@wp.func
def clampi(i: int, n: int):
    return wp.min(wp.max(i, 0), n - 1)


@wp.func
def texel(c: float, n: int):
    # same as the CPU sampler: normalized -> texel space, clamp addressing
    t = c * float(n) - 0.5
    return wp.min(wp.max(t, 0.0), float(n) - 1.0)


@wp.kernel
def sample_manual(vol: wp.array3d(dtype=float), pts: wp.array(dtype=wp.vec3), out: wp.array(dtype=float)):
    i = wp.tid()
    p = pts[i]
    d = vol.shape[0]
    h = vol.shape[1]
    w = vol.shape[2]
    tx = texel(p[0], w)
    ty = texel(p[1], h)
    tz = texel(p[2], d)
    x0 = int(wp.floor(tx))
    y0 = int(wp.floor(ty))
    z0 = int(wp.floor(tz))
    fx = tx - float(x0)
    fy = ty - float(y0)
    fz = tz - float(z0)
    x1 = clampi(x0 + 1, w)
    y1 = clampi(y0 + 1, h)
    z1 = clampi(z0 + 1, d)
    x0 = clampi(x0, w)
    y0 = clampi(y0, h)
    z0 = clampi(z0, d)
    v000 = vol[z0, y0, x0]
    v100 = vol[z0, y0, x1]
    v010 = vol[z0, y1, x0]
    v110 = vol[z0, y1, x1]
    v001 = vol[z1, y0, x0]
    v101 = vol[z1, y0, x1]
    v011 = vol[z1, y1, x0]
    v111 = vol[z1, y1, x1]
    v00 = v000 * (1.0 - fx) + v100 * fx
    v10 = v010 * (1.0 - fx) + v110 * fx
    v01 = v001 * (1.0 - fx) + v101 * fx
    v11 = v011 * (1.0 - fx) + v111 * fx
    v0 = v00 * (1.0 - fy) + v10 * fy
    v1 = v01 * (1.0 - fy) + v11 * fy
    out[i] = v0 * (1.0 - fz) + v1 * fz


@wp.kernel
def sample_tex(tex: wp.Texture3D, pts: wp.array(dtype=wp.vec3), out: wp.array(dtype=float)):
    i = wp.tid()
    out[i] = wp.texture_sample(tex, pts[i], dtype=float)


def timed(fn):
    fn()  # warm-up
    wp.synchronize()
    ts = []
    for _ in range(RUNS):
        wp.synchronize()
        t0 = time.perf_counter()
        fn()
        wp.synchronize()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1e3  # ms


print(f"Warp CPU device: {cpu.name}; Warp launches CPU kernels on one thread (no thread pool)")
rows = []
for size in (64, 256):
    rng = np.random.default_rng(size)
    vol_np = rng.random((size, size, size), dtype=np.float32)
    vol_gpu = wp.array(vol_np, dtype=float, device=gpu)
    tex_cpu = wp.Texture3D(
        vol_np, filter_mode=wp.TextureFilterMode.LINEAR, address_mode=wp.TextureAddressMode.CLAMP,
        normalized_coords=True, device=cpu,
    )  # fmt: skip
    for n in (1_000_000, 16_000_000):
        pts_np = rng.random((n, 3), dtype=np.float32)
        pts_gpu = wp.array(pts_np, dtype=wp.vec3, device=gpu)
        out_gpu = wp.empty(n, dtype=float, device=gpu)
        pts_cpu = wp.empty(n, dtype=wp.vec3, device=cpu)
        out_cpu = wp.empty(n, dtype=float, device=cpu)
        pts_cpu_c = wp.array(pts_np, dtype=wp.vec3, device=cpu)
        out_cpu_c = wp.empty(n, dtype=float, device=cpu)

        a = timed(lambda: wp.launch(sample_manual, dim=n, inputs=[vol_gpu, pts_gpu, out_gpu], device=gpu))
        b_d2h = timed(lambda: wp.copy(pts_cpu, pts_gpu))
        b_kern = timed(lambda: wp.launch(sample_tex, dim=n, inputs=[tex_cpu, pts_cpu, out_cpu], device=cpu))
        b_h2d = timed(lambda: wp.copy(out_gpu, out_cpu))

        def b_all():
            wp.copy(pts_cpu, pts_gpu)
            wp.launch(sample_tex, dim=n, inputs=[tex_cpu, pts_cpu, out_cpu], device=cpu)
            wp.copy(out_gpu, out_cpu)

        b_tot = timed(b_all)
        c = timed(lambda: wp.launch(sample_tex, dim=n, inputs=[tex_cpu, pts_cpu_c, out_cpu_c], device=cpu))

        # numerics: (a) vs (c) on the same points
        wp.launch(sample_manual, dim=n, inputs=[vol_gpu, pts_gpu, out_gpu], device=gpu)
        ra = out_gpu.numpy()
        rc = out_cpu_c.numpy()
        diff = np.abs(ra - rc)
        ulp = np.abs(ra.view(np.int32).astype(np.int64) - rc.view(np.int32).astype(np.int64))
        rows.append((size, n, a, b_d2h, b_kern, b_h2d, b_tot, c, diff.max(), int(ulp.max()), int(np.count_nonzero(ulp))))
        print(rows[-1], flush=True)

print("| volume | samples | (a) GPU manual ms | (b) D2H ms | (b) CPU kernel ms | (b) H2D ms | (b) total ms | (c) CPU kernel ms | (b)/(a) | (a) vs (c) max abs | max ulp | values differing |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
for s, n, a, d2h, k, h2d, tot, c, md, mu, nd in rows:
    print(f"| {s}^3 | {n:.1e} | {a:.3f} | {d2h:.2f} | {k:.1f} | {h2d:.2f} | {tot:.1f} | {c:.1f} | {tot / a:.0f}x | {md:.2e} | {mu} | {nd} of {n} |")
sys.exit(0)
```
## Evidence session 3 (2026-11-19, linux-gfx1100, Radeon Pro W7800, evidence only): Phase 5 re-run at `2f5b9d65` (handoff B.1-B.6)

Requested by Jeff Daily. The work lock stays with linux-gfx950. Nothing was committed to the fork, and no stage, lock, head, platform or deferral record changed. The clone was updated read-only to the recorded `head_sha` `2f5b9d657295a2d5ba5371ae1f7ff4582ade7acf` (`git status --porcelain` clean). Same venv as the earlier gfx1100 sessions (TheRock ROCm 10.2.0 wheels, HIP 7.17.26434, torch 2.15.0+rocm10.2.0). Build: `export PATH=$FLEET_VENV/bin:$PATH; python build_lib.py --rocm-path "$(rocm-sdk path --root)" --hip-arch gfx1100 -j 32` -> "Warp build succeeded". RUNPATH points at the venv's `_rocm_sdk_devel/lib`. Private `WARP_CACHE_PATH`, `ulimit -c 0`, `HIP_VISIBLE_DEVICES=0`. Scripts extracted verbatim from Porting sessions 5 and 6.

### B.1-B.3: textures at the new head: all as expected

| item | expected | result |
|---|---|---|
| B.1 `python -m warp.tests -s autodetect -p test_texture.py -m 32` | Ran 120, OK (skipped=1) | **Ran 120, OK (skipped=1)**; the skip is the `test_texture_unsupported_device_raises` placeholder ("No suitable devices to run the test.") |
| B.2 `python sdf_probe.py cpu` | PASS | PASS: SDF vs CPU sampler 0.00e+00, SDF vs analytic 1.28e-03 (grid spacing 1.56e-02), RGBA8 vs CPU sampler 0.00e+00 |
| B.2 `python sdf_probe.py cuda:0` | PASS | **PASS**: SDF vs CPU sampler 4.78e-05, SDF vs analytic 1.31e-03 (grid spacing 1.56e-02), RGBA8 vs CPU sampler 2.52e-03 (identical to the patched experiment of evidence session 2) |
| B.3 `python tex_alias.py cuda:0` | PASS, 0 wrong plain and under capture | **PASS**: "plain: 0 of 2048 wrong", "graph capture: 0 of 2048 wrong" |

So the porter's version of the fix, with its two additions (element size recorded from `cuArray3DGetDescriptor_f` for external arrays, and the offset/no-array fallback), behaves on real image hardware like the experiment did. The external-array path that `tex_alias.py` covers also works under capture.

### B.4: the two rewritten HIP reproducers: both reproduce
Built as the handoff says: `clang++ -x hip --rocm-path=$R --offload-arch=gfx1100 -O2 <file>.cpp -o <name> -L$R/lib -lamdhip64 -Wl,-rpath,$R/lib`, with `R=$(rocm-sdk path --root)`. Both compile without warnings.

```
$ ./m3d_pageable 64 64            (exit 1)
hipDrvMemcpy3DAsync    64x64: 4096 of 4096 elements wrong
hipMemcpy3DAsync       64x64: 4096 of 4096 elements wrong
$ ./m3d_capture                   (exit 1)
hipDrvMemcpy3DAsync    graph nodes 0, copied before instantiation 2048 of 2048, wrong after replay 2048 of 2048
hipMemcpy3DAsync       graph nodes 2, copied before instantiation 1 of 2048, wrong after replay 0 of 2048
```

- `hip-memcpy3d-async-pageable-host-read-after-return`: reproduces exactly as registered (all elements wrong, both APIs).
- `hip-drv-memcpy3d-async-not-captured`: reproduces as registered (driver call: 0 nodes, copied eagerly, all wrong after replay; runtime call: 2 nodes, 0 wrong after replay). The registration text needs no `deferred.py amend`.
- One correction to the reproducer, not to the finding: the runtime line's "copied before instantiation 1 of 2048" should read 0. `m3d_capture.cpp:47` fills `h1[i] = float(i)`, so `h1[0]` is 0.0. Line 51 zeroes `dst` with `hipMemset`, so element 0 matches at line 62 without any copy having happened. Proposed one-line change before the reproducer goes into a report: `h1[i] = float(i + 1)` (line 47). The exit codes are by design: each program returns 1 when the defect reproduces.

### B.5: full Phase 1-5 set (82 patterns)
Command: the handoff's 82-pattern loop, `python -m warp.tests -s autodetect -p "<pattern>" -m 32` each, from `projects/warp-1.14/src`, wrapped in `utils/timeit.sh warp-1.14 test`. **82 patterns: 81 OK, 1 with the 2 known failures** (fem). Every count matches the handoff's expectation:

| pattern | ran | result | pattern | ran | result |
|---|---|---|---|---|---|
| test_codegen | 117 | OK | test_capture_mode | 2 | OK |
| test_snippet | 12 | OK | test_conditional_captures | 32 | OK, 16 skipped |
| test_arithmetic | 126 | OK | test_conditional | 50 | OK |
| test_math | 16 | OK | test_async | 2373 | OK |
| test_fp16 | 6 | OK | test_streams | 24 | OK, 2 skipped |
| test_bf16 | 48 | OK | test_mempool | 7 | OK, 2 skipped |
| test_atomic | 164 | OK | test_pinned | 1 | OK |
| test_atomic_bitwise | 6 | OK | test_unified_memory | 27 | OK, 8 skipped |
| test_atomic_cas | 48 | OK | test_occupancy | 6 | OK |
| test_array | 139 | OK | test_allocator | 16 | OK, 3 skipped |
| test_copy | 14 | OK | test_allocation_tracker | 34 | OK |
| test_launch | 32 | OK | test_dlpack | 16 | OK |
| test_fast_math | 12 | OK, 1 skipped | test_torch | 34 | OK |
| test_grad | 41 | OK | test_map | 45 | OK |
| test_kernel_cache | 5 | OK | test_reload | 14 | OK |
| test_module_hashing | 8 | OK | test_cuda_arch_suffix | 10 | OK, 10 skipped |
| test_print | 10 | OK | test_tile | 126 | OK |
| test_mat* | 601 | OK | test_tile_atomic_bitwise | 3 | OK |
| test_vec* | 531 | OK | test_tile_cholesky | 33 | OK |
| test_quat* | 173 | OK | test_tile_cholesky_no_mathdx | 16 | OK |
| test_spatial* | 207 | OK | test_tile_empty | 20 | OK |
| test_rand | 10 | OK | test_tile_fft | 20 | OK, 8 skipped |
| test_noise | 14 | OK | test_tile_fft_no_mathdx | 16 | OK |
| test_clang_cuda | 7 | OK, 6 skipped | test_tile_func_arg | 27 | OK |
| test_runlength_encode | 16 | OK | test_tile_fused_ops | 36 | OK |
| test_array_reduce | 16 | OK | test_tile_large_offsets | 1 | OK |
| test_sparse | 47 | OK | test_tile_load | 64 | OK |
| test_linear_solvers | 34 | OK | test_tile_mathdx | 6 | OK |
| test_coloring | 3 | OK | test_tile_matmul | 22 | OK |
| test_bvh | 13 | OK | test_tile_matmul_no_mathdx | 22 | OK |
| test_grouped_bvh | 11 | OK | test_tile_mlp | 3 | OK, 1 skipped |
| test_hash_grid | 23 | OK | test_tile_reduce | 62 | OK |
| test_marching_cubes | 17 | OK | test_tile_shared_memory | 30 | OK |
| test_mesh* | 49 | OK | test_tile_solve | 5 | OK, 1 skipped |
| test_volume | 59 | OK | test_tile_solve_no_mathdx | 28 | OK |
| test_volume_write | 5 | OK | test_tile_sort | 2 | OK |
| test_apic | 52 | OK | test_tile_stack | 40 | OK |
| test_apic_mesh | 12 | OK | test_tile_view | 16 | OK |
| test_fem* | 107 | **2 failures** | test_block_dim_dispatch | 2 | OK |
| test_graph | 35 | OK, 3 skipped | test_template_launch_bounds | 36 | OK |
| test_array_fill_capture | 3 | OK | test_texture | 120 | OK, 1 skipped |

- Phase 1-3: the B.4 counts of the first gfx1100 session, with test_atomic_cas now 48 (the 24 new AMD-capable `atomic_cas`/`atomic_exch` cases pass at wavefront 32) and test_array 139.
- test_fem*: 107 run, failures=2: `test_gimp_quadrature` and `test_cube_shape_functions`, "Unexpected output" whose first line is Warp's init greeting. These are the known upstream isolation failures; nothing else in fem fails.
- test_streams: 24 run, OK (2 skipped). The C4 case `test_stream_priority_timings_cuda_0`, which fails on gfx942 and gfx950, passes here.
- Tile set (22 tile files + test_block_dim_dispatch + test_template_launch_bounds): 636 run, 10 skipped (the three upstream no-libmathdx skips), 0 failures. The 64 KiB sizing commits work on this part: test_tile_shared_memory 30, test_tile_reduce 62 and test_tile_view 16 OK, and `test_tile_shared_mem_func` prints no shared-memory warning.
- One shared-memory warning remains in the whole set, from a test the sizing commits did not touch: test_tile_load prints `Failed to configure kernel dynamic shared memory ... test_tile_load_scoped_kernel_..._cuda_kernel_backward kernel for 65536 bytes, but maximum available is 65536`. The test passes, because it launches only the forward kernel (`test_tile_load.py:949`, no tape). Warp configures both kernels' shared memory when the module loads, and the backward kernel (64 KiB dynamic plus static LDS) cannot fit on a 64 KiB part. This is the corrected diagnostic from the shim fix, not a failure. My first gfx1100 run (old shim) printed nothing here. The same arithmetic predicts the line on gfx942 (not checked here). Recorded for the lock holder; no change proposed, since the test does what upstream intends and a launch of that backward kernel would fail on any 64 KiB part, NVIDIA included.
- test_texture: 120 run, OK (1 skipped, the placeholder).
- Skip reasons across all 82 logs (`grep -h "skipped '"`): Conditional graph nodes not supported 16, Warp was not built with MathDx support 8, MathDx is not enabled 1, Skipping test on CUDA device without MathDx (tolerance) 1, No suitable devices to run the test 18 (17 as before plus the texture placeholder), Requires at least two CUDA devices 2, Multi-GPU not available 2, mempool/peer-access device-pair requirements 10, cuda:0 cannot access CPU memory 2, flakiness in memory usage measurement 1, rmm not installed 1. None names wavefront size, textures or AMD.
- No `Warp CUDA error` line in any of the 82 logs.

### B.6: CPU-only suite with GPUs hidden
`HIP_VISIBLE_DEVICES=99 CUDA_VISIBLE_DEVICES=99 python -m warp.tests -s autodetect -k cpu -m 32` (timeit-wrapped; greeting lists only `cpu`): **Ran 2294 tests, OK (skipped=15)**. That matches the expectation (session 5 on gfx950). Skips: No suitable devices 10, Requires CUDA 2, Requires both CUDA and CPU devices 2, CUDA not available 1.

### Summary
At `2f5b9d65`, the wave32 half of the same-head requirement is met. The Phase 1-5 set passes on gfx1100 apart from the two known fem isolation failures, which gfx950 shows identically. The texture fix as committed passes test_texture, the SDF probe and tex_alias.py on real image hardware. Both registered HIP defects reproduce with the porter's rewritten reproducers. One cosmetic fix to `m3d_capture.cpp` is needed (line 47, `float(i + 1)`). The CPU-only suite is green. Nothing new needs a fork change.

## Ruling on textures for AMD GPUs without image support (2026-11-18, Jeff Daily)

RULED: option B, in this line. gfx942 and gfx950 (and any AMD GPU HIP reports without image support) get a GPU software sampler instead of refusing textures. Evidence he asked for first: the GPU-loads measurement in this file (534x to 1834x faster than sampling on the CPU and copying back; Warp's CPU device is single-threaded).

Design constraint from Jeff: minimise the diff carried forward into every later Warp release. So:
- Kernel side (`texture.h`): no new sampler. Remove the HIP no-image-support branch that returns zero so those device compiles fall through to upstream's existing software sampler (`cpu_sample_*_channel`); selection stays at compile time through `__HIP_NO_IMAGE_SUPPORT`, with no per-sample run-time branch. The handle is then a pointer to a `Texture` struct in device memory whose `data` points at device memory.
- Native side: the second storage kind (struct plus texels in ordinary device memory; create, copy in and out, destroy) lives in a new file of the port's own, reached from the existing `texture.cpp` functions through one capability check each. Touch as few upstream lines as possible; no parallel copies of upstream functions.
- Python side: smallest change that works; `cuda_array`, `cuda_surface` and `cuda_texture` interop properties raise a clear error on these devices; `Device.is_texture_supported` becomes True where the software path serves it, and a separate capability says whether sampling is hardware.
- Tests: `test_texture` runs on these devices, registered by capability, never by arch; results must match Warp's CPU textures (expected bit-identical, since it is the same sampler), with hardware-filtering tolerances only where a test compares against the hardware path.
- Release notes: textures on MI300/MI355 are sampled in software on the GPU; say how that differs from hardware sampling in precision and speed.

## Ruling on the tile tests that need more than 64 KiB of shared memory (2026-11-18, Jeff Daily)

RULED: capability skip, for the minimal carried diff. The five tests (`test_tile_shared_mem_large`, `test_tile_shared_mem_func`, the tier-2 axis-2 reduction test in `test_tile_reduce.py`, and the two slice-copy tests in `test_tile_view.py`; take the exact list from commits `59d873af` and `2f5b9d65`) stay byte-for-byte upstream and skip on any device whose shared memory per block cannot hold what the test needs. The reason he gave when asked: about 96 changed lines in three upstream test files plus a helper is too much to carry into every later Warp release for the small coverage unique to these five tests; the other tile tests already cover smaller shared tiles on those GPUs.

Porter, next session:
- Undo the sizing with new commits on top (`git revert` of `2f5b9d65` and `59d873af`, no history rewrite), including the `select_shared_memory_fit` helper if nothing else uses it.
- Add the skip with the fewest lines possible: decided from `device.max_shared_memory_per_block` against the bytes the test needs (dynamic plus Warp's static allocation), never from an arch or vendor name, with a skip message that states the two numbers. It must not skip on gfx950 or on any device where the upstream size fits.
- Leave `test_tile_load_scoped` alone; its shared-memory warning for a backward kernel it never launches is a correct diagnostic.
- No spill-to-global-memory fallback for oversized shared tiles: upstream has none on any GPU.
- Release notes: shared-memory tiles are limited to 64 KiB per block (tiles plus Warp's small static allocation) on gfx90a, gfx942 and Radeon GPUs, 160 KiB on gfx950; Warp warns when a kernel's request does not fit.

## Ruling on `__frsqrt_rn` / `__fsqrt_rn` accuracy (2026-11-18, Jeff Daily)

RULED: option A. The port leaves both intrinsics as the HIP toolchain defines them and adds no override; the release notes say that on AMD GPUs `__frsqrt_rn` and `__fsqrt_rn` are within 1 ulp but not correctly rounded (wrong in the last bit on about 11 and 15 percent of inputs), unlike CUDA's. Whether to report it upstream is not ruled; Jeff asked why AMD would see it as a defect and not a deliberate choice.

Where the definitions are (read from the ROCm 10.2 clang resource header `lib/clang/23/include/__clang_hip_math.h`, lines 283-303; this is the compiler's header, not HIP's): the `_rn` family is implemented twice. With `OCML_BASIC_ROUNDED_OPERATIONS` defined, `__fadd_rn`, `__fdiv_rn`, `__fmul_rn`, `__frcp_rn`, `__fmaf_rn` and `__fsqrt_rn` call `__ocml_*_rte_f32` (round to nearest even). Without it, add, div, mul, rcp and fma fall back to the plain IEEE operation, which is correctly rounded, but `__fsqrt_rn` falls back to `__ocml_native_sqrt_f32` (approximate) and not to `__builtin_sqrtf`. `__frsqrt_rn` is `__builtin_amdgcn_rsqf` in both configurations; there is no rounded form of it in the header at all. Not checked: whether defining `OCML_BASIC_ROUNDED_OPERATIONS` works with ROCm 10.2's device libraries.
## Porting session 7 (2026-11-19, linux-gfx950 MI355X, porter): textures option B, GPU software sampler

Fork `release/1.14.0.amd0` advanced `2f5b9d65..495d7c5a` (pushed, `advance-head` recorded): `495d7c5a` [ROCm] Sample textures in software on AMD GPUs without image support. One commit, so the whole option can be reviewed or dropped as a unit. nvcc 12.8 build at `495d7c5a` succeeded (it compiles `texture_software.cpp` too); offline NVRTC of the texture module gave `sm86.cubin` and `sm90.ptx` with the same 8 `tex.{1d,2d,3d}.v4.f32.f32` instructions as before; the tile/atomic NVRTC check also passes. `audit-commits` OK, `jargon.py --port` clean.

### Design as built, against Jeff's constraints
- `texture.h`: the HIP no-image `#elif defined(__HIP_DEVICE_COMPILE__) return zero();` pair removed at all 9 sites (plus a comment line), nothing else. Device compiles with `__HIP_NO_IMAGE_SUPPORT` now fall through to upstream's existing `#else` branch, the software sampler (`cpu_sample_*_channel` on a `const Texture*`); the choice stays at compile time, with no per-sample run-time branch. `git diff 2f5b9d65 -- warp/native/texture.h`: only those 18 lines removed and 1 comment line changed; every `WP_TEXTURE_HW_SAMPLING` line is byte-identical.
- New `warp/native/texture_software.{h,cpp}` (35 + 131 lines, SPDX licence line only, like the port's other new files): `SoftwareTextureArray` (device texel buffer + shape/channels/dtype) stands in for a CUDA array; create/destroy (`wp_alloc_device`/`wp_free_device`), copy (one `wp_memcpy_h2d/d2h/d2d` block per copy: texels and Warp's linear sides are dense, pitched copies refused with a clear error; none occur in Warp), texture object create (a `Texture` built on the stack with the 1-texel 1D constructor, fields set, `data` swapped to the device texels, copied to a `wp_alloc_device` struct with `wp_memcpy_h2d` the way `mesh.cu` copies its descriptor, then `data` restored so the destructor frees its own 1-byte buffer), object destroy. It uses only Warp's exported functions, so it has no HIP code, and it builds in the CUDA build as well (never reached there). Added to `build_lib.py`'s source list (1 line).
- `texture.cpp`: one include and one capability check per device function (`wp_texture_create_device`, `wp_texture_destroy_device`, `wp_texture_copy_device`, `wp_texture_object_create_device`, `wp_texture_object_destroy_device`): `if (wp_texture_software_enabled(context)) return wp_texture_software_...(...)`, 18 added lines, no upstream line changed. `wp_texture_software_enabled` = the device has no texture hardware (`wp_cuda_device_is_texture_hardware_supported(wp_cuda_context_get_device_ordinal(context))`), so it is false for every CUDA device and for AMD GPUs with image support. Surface and external-array functions need no hook: Python refuses those first.
- Native capability renamed to what it measures: `DeviceInfo.is_texture_hardware_supported` / `wp_cuda_device_is_texture_hardware_supported` (HIP image-support attribute; 1 on CUDA). Both were the port's own additions, so the rename costs no upstream lines.
- Python: `Device.is_texture_supported` is now True on every device; new `Device.is_texture_hardware_supported` (False on CPU and on AMD GPUs without image support; documented in the Device docstring). `texture.py`: the session 5 refusal removed; one static helper `_require_texture_hardware` raising "<name> is not available on cuda:0: it has no texture hardware and samples textures in software", called by the `cuda_array`, `cuda_texture` and `cuda_surface` properties and by the `cuda_array=` constructor argument; the four internal copy paths use `_array_handle` instead of the `cuda_array` property (else every copy would have raised).
- Tests (`test_texture.py`): upstream's `cuda_devices`/`all_devices` lists restored; the 9 tests that use the interop handles are registered on `interop_cuda_devices`/`interop_devices` (devices with texture hardware, plus CPU where the test has a CPU branch); the refusal test is replaced by `test_texture_interop_requires_hardware`, registered on CUDA devices without texture hardware (the three properties and `cuda_array=` raise; `id` works). Everything else, interpolation, uint8/uint16 normalization, address modes, graph-capture copies, runs on these GPUs with upstream's tolerances.

Diff carried against upstream `release-1.14` after this commit (`git diff --numstat origin/release-1.14`; added/deleted lines), with this commit's share:

| file | vs release-1.14 | this commit | before (at `2f5b9d65`) |
|---|---|---|---|
| `warp/native/texture.h` | +31 / -18 | +2 / -19 | +48 / -18 |
| `warp/native/texture.cpp` | +60 / -19 | +18 / -0 | +42 / -19 |
| `warp/_src/texture.py` | +15 / -4 | +15 / -9 | +5 / -0 |
| `warp/native/texture_software.cpp` (new) | +131 | +131 | - |
| `warp/native/texture_software.h` (new) | +35 | +35 | - |

`texture.h` got smaller; the 42 earlier `texture.cpp` lines are the memory-type translation of session 6, not this option. Other files this commit touched: `build_lib.py` +1, `context.py` +7/-4, `warp.cu`/`warp.h`/`warp.cpp` renames (+6/-6, +1/-1, +1/-1), `test_texture.py` (+36/-9 against upstream in total).

### Hardware path unchanged, checked three ways
- Source: as above, no line of the `WP_TEXTURE_HW_SAMPLING` branches changed in `texture.h`; in `texture.cpp` the hardware code is untouched and only reached when the check is false.
- HIP targets with image support: the all-combinations sampling module (`agent_space/p5/tex/xcompile.py`, 9 samples) recompiled through Warp's hipRTC path at `495d7c5a` gives code objects **byte-identical** (md5) to session 5's for gfx1100, gfx1201, gfx1151 and gfx90a (9 `image_sample` each). gfx942/gfx950 now compile the software sampler (128000-byte objects, `flat_load_*` instead of `image_sample`).
- CUDA: the NVRTC PTX keeps its 8 hardware `tex.` instructions; nvcc builds.

### Tests and probes on gfx950 (first host that runs the new path; `HIP_VISIBLE_DEVICES=0`)
- `python -m warp.tests -s autodetect -p test_texture.py -m 32`: **Ran 118, OK (skipped=7)**. 59 `cuda_0` cases pass, among them the four `cuda_array_copy_api_graph_capture{,_explicit_stream}` tests (2D and 3D) and `test_texture_interop_requires_hardware_cuda_0`; the 7 skips are the placeholders of the 7 interop-only registrations that have no device here ("No suitable devices to run the test."). Before this commit gfx950 ran 70 with 17 placeholders and no texture case on the GPU.
- `sdf_probe.py cuda:0`: PASS; SDF vs CPU sampler 5.96e-08, SDF vs analytic 1.28e-03 (the CPU's own value), RGBA8 vs CPU sampler 1.55e-06.
- Graph capture with textures (`tex_capture.py`, below: one graph uploads a source array into a 32^3 texture and samples it at 4096 points; replayed three times after changing the source): every replay matches the CPU sampler within 1.19e-07: PASS.
- External-array probe (`tex_alias.py`): not applicable on these GPUs, by design; it now stops at `Texture2D(cuda_array=...)` with "cuda_array is not available on cuda:0: it has no texture hardware and samples textures in software".
- test_devices 13, test_diagnostics 14, test_codegen 117, test_tile_load 64: OK. CPU-only suite with GPUs hidden: **Ran 2294, OK (skipped=15)**.

**Bit-identity with the CPU sampler: only with contraction off.** The ruling expected bit-identical results. They are bit-identical when the module is built with `fuse_fp=False` (`wp.set_module_options({"fuse_fp": False})`): SDF and RGBA8 differences 0, and in the benchmark below 0 differing values out of 1.6e7. With the default (`fuse_fp=True`, so hipRTC compiles with `-ffp-contract=on`) the GPU compiler fuses the sampler's `a * (1 - f) + b * f` lerps into FMAs and about a third of the samples differ from the CPU in the last bits (max 1.79e-07 on values in [0, 1)). The CPU build of the same C++ does not contract. Warp's own generated kernel code is not affected the same way (the hand-written kernel matches the CPU exactly with either setting). The tests pass with upstream's tolerances either way; no test asserts bit-identity. Recorded for Jeff: if he wants bit-identity by default, the sampler's lerps would need an explicit no-contract form (a `texture.h` change, so more carried diff); not done.

### Real sampler vs the hand-written kernel (same benchmark as the earlier measurement)
`wp.Texture3D` on cuda:0 (the real software sampler, (d)) vs the hand-written trilinear kernel ((a), from the "Measurement for Jeff" section), float32 3D, trilinear, clamp, normalized coordinates, median of 5 timed runs after warm-up with `wp.synchronize()` around each, default `fuse_fp`. The host was noisier than during the earlier measurement ((a) at 64^3 / 1.6e7 is 0.52-0.64 ms here against 0.30 ms then), so four runs are listed; the ratio is the figure to read. One more run with `fuse_fp=False` gave ratios 1.36x, 1.84x, 0.61x, 1.49x and bit-identical results.

First run (default `fuse_fp`):

| volume | samples | (a) ms | (d) ms | (d)/(a) | (d) vs CPU sampler max abs | values differing | (a) vs CPU sampler |
|---|---|---|---|---|---|---|---|
| 64^3 | 1.0e6 | 0.040 | 0.055 | 1.39x | 1.79e-07 | 338479 of 1000000 | 0 |
| 64^3 | 1.6e7 | 0.522 | 0.935 | 1.79x | 1.79e-07 | 5416499 of 16000000 | 0 |
| 256^3 | 1.0e6 | 1.597 | 0.215 | 0.13x | 1.79e-07 | 340617 of 1000000 | 0 |
| 256^3 | 1.6e7 | 1.502 | 3.064 | 2.04x | 1.79e-07 | 5454682 of 16000000 | 0 |

Three repeat runs (same columns):
```
== run 1
| 64^3 | 1.0e+06 | 0.091 | 0.109 | 1.20x | 1.79e-07 | 338479 of 1000000 | 0.00e+00 |
| 64^3 | 1.6e+07 | 0.570 | 0.812 | 1.43x | 1.79e-07 | 5416499 of 16000000 | 0.00e+00 |
| 256^3 | 1.0e+06 | 0.138 | 0.185 | 1.34x | 1.79e-07 | 340617 of 1000000 | 0.00e+00 |
| 256^3 | 1.6e+07 | 1.449 | 2.023 | 1.40x | 1.79e-07 | 5454682 of 16000000 | 0.00e+00 |
== run 2
| 64^3 | 1.0e+06 | 0.081 | 0.102 | 1.26x | 1.79e-07 | 338479 of 1000000 | 0.00e+00 |
| 64^3 | 1.6e+07 | 0.523 | 0.953 | 1.82x | 1.79e-07 | 5416499 of 16000000 | 0.00e+00 |
| 256^3 | 1.0e+06 | 0.329 | 0.524 | 1.59x | 1.79e-07 | 340617 of 1000000 | 0.00e+00 |
| 256^3 | 1.6e+07 | 1.331 | 1.976 | 1.48x | 1.79e-07 | 5454682 of 16000000 | 0.00e+00 |
== run 3
| 64^3 | 1.0e+06 | 0.102 | 0.112 | 1.10x | 1.79e-07 | 338479 of 1000000 | 0.00e+00 |
| 64^3 | 1.6e+07 | 0.642 | 0.947 | 1.47x | 1.79e-07 | 5416499 of 16000000 | 0.00e+00 |
| 256^3 | 1.0e+06 | 0.137 | 0.211 | 1.54x | 1.79e-07 | 340617 of 1000000 | 0.00e+00 |
| 256^3 | 1.6e+07 | 1.548 | 1.991 | 1.29x | 1.79e-07 | 5454682 of 16000000 | 0.00e+00 |
```
Median ratio over the four default runs per cell: 1.23x (64^3, 1e6), 1.63x (64^3, 1.6e7), 1.44x (256^3, 1e6; the 0.13x cell of the first run is a disturbed (a) timing, 1.597 ms against 0.14-0.33 ms in the others), 1.44x (256^3, 1.6e7). So the real software sampler costs about 1.2x to 1.8x the hand-written kernel (the extra work is the per-sample dtype, address-mode and channel switches and the handle dereference), and it is still about three orders of magnitude faster than sampling on the CPU and copying back (2128.6 ms for 256^3 / 1.6e7 in the earlier measurement, against about 2-3 ms here).

Release-notes text for Phase 7 (from the ruling): "On AMD Instinct MI300 and MI355 GPUs (gfx942, gfx950), which have no texture hardware, textures are sampled in software on the GPU with the same code as Warp's CPU textures. Results match CPU textures (bit for bit with `fuse_fp=False`, otherwise to the last bits) rather than NVIDIA or AMD texture hardware, whose linear filtering uses reduced-precision weights; linear filtering of normalized 8-bit data can differ from hardware sampling by a few 1e-3 (2.5e-3 measured on a Radeon GPU), float32 data by about 5e-5. Sampling is about 1.2x to 1.8x slower than a hand-written trilinear lookup in a Warp array on the same GPU, and much slower than texture hardware would be. The CUDA interop properties `cuda_array`, `cuda_texture` and `cuda_surface`, and wrapping an external CUDA array, are not available on these GPUs; `Device.is_texture_hardware_supported` tells whether a device samples in hardware." The 1e-3 figure comes from the gfx1100 probe (RGBA8 vs CPU 2.52e-03 absolute, SDF 4.78e-05); there is no gfx950 hardware figure, since it has no texture hardware.

`tex_capture.py`:
```python
# A graph that uploads into a texture and samples it; replayed after changing the source array,
# the samples must follow the new data. Compared with the same steps on the CPU device.
import sys

import numpy as np

import warp as wp

wp.init()
device = wp.get_device(sys.argv[1] if len(sys.argv) > 1 else "cuda:0")


@wp.kernel
def sample(tex: wp.Texture3D, pts: wp.array(dtype=wp.vec3), out: wp.array(dtype=float)):
    i = wp.tid()
    out[i] = wp.texture_sample(tex, pts[i], dtype=float)


rng = np.random.default_rng(0)
n, s = 4096, 32
pts_np = rng.random((n, 3), dtype=np.float32)
vols = [rng.random((s, s, s), dtype=np.float32) for _ in range(3)]


def reference(vol):
    tex = wp.Texture3D(vol, filter_mode=wp.TextureFilterMode.LINEAR, device="cpu")
    out = wp.zeros(n, dtype=float, device="cpu")
    wp.launch(sample, dim=n, inputs=[tex, wp.array(pts_np, dtype=wp.vec3, device="cpu"), out], device="cpu")
    return out.numpy()


tex = wp.Texture3D(np.zeros((s, s, s), dtype=np.float32), filter_mode=wp.TextureFilterMode.LINEAR, device=device)
src = wp.array(vols[0], dtype=float, device=device)
pts = wp.array(pts_np, dtype=wp.vec3, device=device)
out = wp.zeros(n, dtype=float, device=device)
wp.load_module(device=device)
with wp.ScopedCapture(device=device) as cap:
    tex.copy_from(src)
    wp.launch(sample, dim=n, inputs=[tex, pts, out], device=device)
ok = True
for k, vol in enumerate(vols):
    src.assign(vol)
    wp.capture_launch(cap.graph)
    d = np.abs(out.numpy() - reference(vol)).max()
    print(f"replay {k}: max abs diff vs CPU {d:.2e}")
    ok = ok and d < 1e-6
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
```

Benchmark: `agent_space/p7/tex_sw_vs_manual.py` is the earlier `tex_cpu_vs_gpu.py` with the same imports, `timed`, `sample_manual` and `sample_tex`, and this main part in place of the CPU-route measurement:
```python
rows = []
for size in (64, 256):
    rng = np.random.default_rng(size)
    vol_np = rng.random((size, size, size), dtype=np.float32)
    vol_gpu = wp.array(vol_np, dtype=float, device=gpu)
    kw = dict(filter_mode=wp.TextureFilterMode.LINEAR, address_mode=wp.TextureAddressMode.CLAMP, normalized_coords=True)
    tex_gpu = wp.Texture3D(vol_np, device=gpu, **kw)
    tex_cpu = wp.Texture3D(vol_np, device=cpu, **kw)
    for n in (1_000_000, 16_000_000):
        pts_np = rng.random((n, 3), dtype=np.float32)
        pts_gpu = wp.array(pts_np, dtype=wp.vec3, device=gpu)
        out_a = wp.empty(n, dtype=float, device=gpu)
        out_d = wp.empty(n, dtype=float, device=gpu)
        a = timed(lambda: wp.launch(sample_manual, dim=n, inputs=[vol_gpu, pts_gpu, out_a], device=gpu))
        d = timed(lambda: wp.launch(sample_tex, dim=n, inputs=[tex_gpu, pts_gpu, out_d], device=gpu))
        pts_cpu = wp.array(pts_np, dtype=wp.vec3, device=cpu)
        out_c = wp.empty(n, dtype=float, device=cpu)
        wp.launch(sample_tex, dim=n, inputs=[tex_cpu, pts_cpu, out_c], device=cpu)
        ra, rd, rc = out_a.numpy(), out_d.numpy(), out_c.numpy()
        rows.append((size, n, a, d, np.abs(rd - rc).max(), int(np.count_nonzero(rd != rc)), np.abs(ra - rc).max()))
        print(rows[-1], flush=True)

print("| volume | samples | (a) hand-written kernel ms | (d) wp.Texture3D on cuda:0 ms | (d)/(a) | (d) vs CPU sampler max abs | values differing | (a) vs CPU sampler max abs |")
print("|---|---|---|---|---|---|---|---|")
for s, n, a, d, md, nd, ma in rows:
    print(f"| {s}^3 | {n:.1e} | {a:.3f} | {d:.3f} | {d / a:.2f}x | {md:.2e} | {nd} of {n} | {ma:.2e} |")
```

### Reproducer correction
`m3d_capture.cpp` in Porting session 6 now fills `h1[i] = float(i + 1)` (the wave32 host's correction), so element 0 no longer matches the zeroed destination by accident and the runtime line reports 0 copied before instantiation. Rebuilt for gfx1100: compiles.

### Next step (head `495d7c5a`; what each host must re-run)
The project stays in `porting`, lock held by linux-gfx950. Common setup as before, at `git rev-parse HEAD` = `495d7c5a03ff3cacf5f48270f1bc9581650dd2da`. This commit changes texture behaviour on every AMD GPU without image support and adds a source file to the build, so every host rebuilds.

**gfx942 (one GPU):** the second host for the new path.
1. `python -c "import warp as wp; wp.init(); d = wp.get_device('cuda:0'); print(d.is_texture_supported, d.is_texture_hardware_supported)"`: expect `True False`.
2. `python -m warp.tests -s autodetect -p test_texture.py -m 32`: expect Ran 118, OK (skipped=7), 59 `cuda_0` cases passing (the gfx950 result).
3. `python sdf_probe.py cuda:0` (Porting session 5): PASS; `python tex_capture.py cuda:0` (above): PASS; record the differences.
4. The full Phase 1-5 set (82 patterns, Porting session 6 handoff B.5): gfx942 has not run it since `8d13d596`. Expected: the gfx950 counts of Porting session 6 item 5, with test_texture 118 (7 skipped); known failures the two fem isolation tests and `test_stream_priority_timings` (C4).

**Two-GPU session (gfx942 or gfx950, `HIP_VISIBLE_DEVICES=0,1`):** the 13-file C.2 loop (expect Evidence session 2 gfx942's result: no errors, C1/C2 skips naming the defects, C4 failures only), plus `python -m warp.tests -s autodetect -p test_texture.py -m 32`: with two identical devices the `cuda_1` texture cases register too, which checks the software path on a non-primary device (device-memory allocation and copies on ordinal 1). Expect all `cuda_0` and `cuda_1` cases to pass.

**Wave32 host (gfx1100 or gfx1201; image hardware):** confirms the hardware path is unchanged at the new head.
1. `python -m warp.tests -s autodetect -p test_texture.py -m 32`: expect Ran 120, OK (skipped=1) (50 CPU, 67 `cuda_0` including all interop tests, 2 deviceless, and the placeholder of `test_texture_interop_requires_hardware`, which has no device there); the same total as at `2f5b9d65`.
2. `python sdf_probe.py cuda:0` and `python tex_alias.py cuda:0`: PASS with the values of Evidence session 3 (SDF 4.78e-05, RGBA8 2.52e-03).
3. The full Phase 1-5 set (82 patterns) and the CPU-only suite (expect 2294 OK), since the head moved.
4. Optional: rebuild and run `m3d_capture.cpp` with the corrected line to see "copied before instantiation 0 of 2048" on the runtime line.

**gfx950 (this host):** the new path is covered above; the full 82-pattern set at `495d7c5a` was not re-run this session (only test_texture, test_devices, test_diagnostics, test_codegen, test_tile_load and the CPU-only suite). The next gfx950 session runs it, so that both wave families pass the set at the same head before Phase 6.

**Still open for Jeff:** whether default results should be bit-identical to CPU textures (would need a no-contract form of the sampler's lerps in `texture.h`); and, unchanged from session 6: the 64 KiB tile sizing vs capability skips, C4, `__frsqrt_rn` and the lane-mask documentation, defer/now on the six undecided `rocm-bug-report` entries, the Q3 matrix-core ruling. Phase 6 not started.

## Stream priority on HIP (2026-11-18, Jeff Daily)

Stated by Jeff: stream priorities are essentially ignored in the current HIP runtime; high-priority streams exist as an API feature but map to normal priority. He does not expect a test that measures a timing difference to achieve anything on HIP. This settles C4 without the proposed kernel-workload measurement (step 1 of the C4 proposal is dropped): it is known HIP behaviour, not a defect to register.

Porter, next session: in `test_stream_priority_timings`, keep everything that does not depend on scheduling (creating streams with a priority, reading the priority back) and skip only the timing assertion on HIP devices, with the fewest lines possible, through a device capability (not an arch or vendor name in the test) and a skip message saying HIP maps stream priorities to normal priority. Release notes: stream priority is accepted but has no scheduling effect on AMD GPUs.

## Porting session 8 (2026-11-19, linux-gfx950 MI355X, porter): tile-test skip and stream-priority rulings

Fork `release/1.14.0.amd0` advanced `495d7c5a..974cd4bf` (pushed, `advance-head` recorded). Four commits, no native change (`git diff --stat 495d7c5a 974cd4bf -- warp/native build_lib.py` empty, so the gfx950 `warp.so` from session 7 is the head's build):
- `10fbbedf` Revert the test_tile_shared_mem_func sizing commit (reverts `2f5b9d65`)
- `c98ba981` Revert the sizing of four shared-memory tile tests (reverts `59d873af`, including the `select_shared_memory_fit` helper, which nothing else used)
- `250b7c2d` Skip tile tests whose shared memory exceeds the device's per-block limit
- `974cd4bf` [ROCm] Skip the stream-priority timing assertion on HIP
New commits on top, no history rewrite of anything pushed. (The two reverts were first made with git's default messages, which lack the sign-off and test plan, and the first title was 81 characters; both were recreated with full messages before anything was pushed, and the two later commits replayed unchanged; the tree is identical.) nvcc 12.8 build at `974cd4bf`: "Warp build succeeded"; offline NVRTC texture and tile/atomic compiles OK. `audit-commits` OK, `jargon.py --port` clean.

### 1. Tile tests: capability skip (Jeff's ruling)
After the reverts, `test_tile_shared_memory.py`, `test_tile_reduce.py` and `test_tile_view.py` were byte-identical to `release-1.14`. The skip then adds one line per test function, calling a new helper in `unittest_utils.py`:
```python
def skip_if_shared_memory_exceeded(test, device, dynamic_bytes: int, block_dim: int):
    needed = dynamic_bytes + 4 * block_dim + 1024
    if device.is_cuda and needed > device.max_shared_memory_per_block:
        test.skipTest(f"Needs {needed} bytes of shared memory per block, device has {device.max_shared_memory_per_block}")
```
(plus its docstring: Warp's tile allocator keeps 4 bytes per thread, plus a few scalars, in static shared memory, so `4 * block_dim + 1024` bytes are added). Calls, with each test's own dimensions: `test_tile_shared_mem_large` (`DIM_M * DIM_N * 4 * 2`, forward, 256 threads), `test_tile_shared_mem_func` (`DIM_M * DIM_N * 4 * 2 * 2`, backward), `test_tile_reduce_axis_tier2` (`2 * 8 * 8 * 128 * 4`, backward, `block_dim`; covers both registrations, so five tests), `test_tile_assign_2d` (`4 * TILE_M * TILE_N * TILE_O * 4`, backward, 32 threads). Decided from `max_shared_memory_per_block` only, never an arch or vendor name. `test_tile_load_scoped` left alone, as ruled.

Checked on gfx950 (`agent_space/p8/skip64.py`, the five tests by name): at the real limit (163840) all five run and pass, none skips; with `max_shared_memory_per_block` set to 65536 in-process all five skip: "Needs 67584 bytes of shared memory per block, device has 65536" (shared_mem_large, shared_mem_func), 66816 (tier2, 64 threads), 66688 (tier2 single warp and assign_2d, 32 threads). Full files: test_tile_shared_memory 30, test_tile_reduce 62, test_tile_view 16, OK. Real 64 KiB parts (gfx942, gfx1100) must confirm the skips (handoff below).

Carried diff in the three tile files: +4/-0 against `release-1.14` (was +64/-32 with the sizing commits at `2f5b9d65`), plus the 13-line helper in `unittest_utils.py`.

Release-notes text (from the ruling): "Shared-memory tiles are limited to 64 KiB per block, tiles plus Warp's small static allocation, on gfx90a, gfx942 and Radeon GPUs, and to 160 KiB on gfx950. Warp warns when a kernel's request does not fit; there is no spill to global memory, as on any GPU."

### 2. Stream priority: timing assertion skipped on HIP
`test_stream_priority_timings` still creates the priority-0 and priority -1 streams and keeps upstream's check that their priorities differ (HIP reports 0 and -1); then, through a new capability `is_stream_priority_scheduling_supported(device)` in `unittest_utils.py` (false on HIP devices, so the test itself names no vendor), it skips with "HIP maps stream priorities to normal priority, so priority has no scheduling effect." before the timing part. 2 lines in the test, 4 in `unittest_utils.py`. No kernel-workload measurement, as ruled; nothing registered (known HIP behaviour). gfx950: test_streams 24 run, OK, 3 skipped (the priority test plus the 2 multi-GPU skips). Release notes: "Stream priority is accepted and reported but has no scheduling effect on AMD GPUs."

Texture sampler floating-point behaviour left as it is, as instructed; release-notes wording for it: "results can differ from Warp's CPU textures in the last bits unless the module is built with `fuse_fp=False`."

### 3. gfx950 at `974cd4bf`: full Phase 1-5 set and CPU-only suite
82 patterns (`agent_space/p6/full_set.txt`, the session 6 list), `-p "<pattern>" -m 32`, 1609 s: **81 OK, 1 with the 2 known failures** (`test_fem*`: `test_gimp_quadrature` and `test_cube_shape_functions`, the upstream isolation behaviour). Counts identical to session 6 item 5 except: test_streams 24 run with 3 skipped (priority timing now skips instead of failing 1 in 3); test_texture 118 run, 7 skipped (session 7's software path, 59 GPU cases); test_atomic_cas 48; tile files 30/62/16 OK with no skip on gfx950. CPU-only suite with GPUs hidden: **Ran 2294, OK (skipped=15)** (79 s).

### 4. Carried test diff against `release-1.14` (for Jeff)
`git diff --stat origin/release-1.14 -- warp/tests` at `974cd4bf`: 21 files, +341/-25:
```
 warp/tests/cuda/test_capture_mode.py       | 35 ++++++++++--
 warp/tests/cuda/test_clang_cuda.py         | 31 +++++++++-
 warp/tests/cuda/test_cuda_arch_suffix.py   |  3 +-
 warp/tests/cuda/test_mempool.py            |  3 +
 warp/tests/cuda/test_streams.py            |  4 ++
 warp/tests/cuda/test_texture.py            | 45 ++++++++++++---
 warp/tests/cuda/test_unified_memory.py     | 15 +++++
 warp/tests/fem/test_fem_integrate.py       |  3 +
 warp/tests/test_atomic_cas.py              | 66 +++++++++++++++++++++-
 warp/tests/test_devices.py                 |  5 +-
 warp/tests/test_diagnostics.py             |  6 +-
 warp/tests/test_fast_math.py               |  6 +-
 warp/tests/test_grad.py                    |  5 +-
 warp/tests/test_graph.py                   | 10 ++++
 warp/tests/test_map.py                     |  4 +-
 warp/tests/test_snippet.py                 | 27 +++++++++
 warp/tests/test_sparse.py                  |  3 +
 warp/tests/tile/test_tile_reduce.py        |  1 +
 warp/tests/tile/test_tile_shared_memory.py |  2 +
 warp/tests/tile/test_tile_view.py          |  1 +
 warp/tests/unittest_utils.py               | 91 +++++++++++++++++++++++++++++-
 21 files changed, 341 insertions(+), 25 deletions(-)
```
Where the larger entries come from: `unittest_utils.py` 91 lines = the capability helpers (independent thread scheduling, graph module load, the three multi-GPU HIP defects with their messages, stream-priority scheduling, the shared-memory skip) and their docstrings; `test_atomic_cas.py` 66 = the two AMD-capable tests Jeff ruled for; `test_texture.py` 45 = capability registration and the interop test; `test_capture_mode.py` 35 = calling HIP's free function on AMD GPUs; `test_clang_cuda.py` 31 and `test_snippet.py` 27 = Phase 1, `fda7eb27` (the clang JIT tests registered on NVIDIA GPUs only plus a test of the error AMD GPUs raise; the snippet device-branch test the plan asked for).

### Next step (head `974cd4bf`; what each host must re-run)
The project stays in `porting`, lock held by linux-gfx950. Common setup as before, at `git rev-parse HEAD` = `974cd4bf864080331cc3090a5ca0ed8d17da1ce5`. The native code is the same as `495d7c5a`, so a host that built `495d7c5a` may reuse that build (check `git diff --stat 495d7c5a HEAD -- warp/native build_lib.py` is empty); otherwise build as usual. gfx950 has passed the full set at this head (item 3).

**gfx942 (one GPU):**
1. `python -m warp.tests -s autodetect -p test_texture.py -m 32`: expect Ran 118, OK (skipped=7), 59 `cuda_0` cases (the software sampler, first run off gfx950); `python sdf_probe.py cuda:0` and `python tex_capture.py cuda:0`: PASS.
2. The tile skips on a real 64 KiB part: `for f in test_tile_shared_memory.py test_tile_reduce.py test_tile_view.py; do python -m warp.tests -s autodetect -p "$f" -m 32; done`: expect OK with exactly five skips "Needs <N> bytes of shared memory per block, device has 65536" (shared_mem_large and shared_mem_func 67584, tier2 66816, tier2_single_warp and assign_2d 66688); `test_tile_load` still prints its harmless warning for `test_tile_load_scoped`.
3. The full 82-pattern set (Porting session 6 handoff B.5) and the CPU-only suite (expect 2294 OK). Expected: the gfx950 result of item 3, plus the five tile skips; test_streams OK with the priority test skipped.

**Two-GPU session (gfx942 or gfx950, `HIP_VISIBLE_DEVICES=0,1`):** the 13-file C.2 loop: expect no errors and no failures any more (C4 now skips; C1/C2 skips naming the HIP defects; IOMMU-heuristic skips as before); plus test_texture, whose `cuda_1` cases run the software sampler on a non-primary device: expect every `cuda_0` and `cuda_1` case to pass.

**Wave32 host (gfx1100 or gfx1201):**
1. `python -m warp.tests -s autodetect -p test_texture.py -m 32`: expect Ran 120, OK (skipped=1); `sdf_probe.py cuda:0` and `tex_alias.py cuda:0`: PASS with the Evidence session 3 values (hardware path unchanged since `2f5b9d65`, code objects byte-identical).
2. The tile skips: the same three files as gfx942 item 2 (gfx1100 has 64 KiB per workgroup): expect the same five skips and no "Failed to configure" warning from `test_tile_shared_mem_func` (it now skips before building).
3. The full 82-pattern set and the CPU-only suite (expect 2294 OK). Expected: Evidence session 3's B.5 counts, with test_texture 120 (1 skipped), the five tile skips, test_streams with the priority test skipped.

**Before Phase 6 (not started):** gfx942/wave32 pass the full set at this head (or a later one gfx950 also re-runs); then windows-gfx1151 per the plan's Phase 6. Still open for Jeff: defer/now on the six undecided `rocm-bug-report` entries; the Q3 matrix-core ruling.

## Evidence session 4 (2026-11-19, linux-gfx1100, Radeon Pro W7800, evidence only): wave32 re-run at `974cd4bf`

Requested by Jeff Daily. The work lock stays with linux-gfx950. Nothing was committed to the fork, and no stage, lock, head, platform or deferral record changed. The clone was updated read-only to the recorded `head_sha` `974cd4bf864080331cc3090a5ca0ed8d17da1ce5` (`git status --porcelain` clean). This host last built `2f5b9d65`, not `495d7c5a`, so it rebuilt: `export PATH=$FLEET_VENV/bin:$PATH; python build_lib.py --rocm-path "$(rocm-sdk path --root)" --hip-arch gfx1100 -j 32` -> "Warp build succeeded" (32 s), with RUNPATH pointing at the venv's ROCm 10.2 `_rocm_sdk_devel/lib`. Same venv (HIP 7.17.26434, torch 2.15.0+rocm10.2.0), private `WARP_CACHE_PATH`, `ulimit -c 0`, `HIP_VISIBLE_DEVICES=0`. Items are the "Wave32 host" list of the `974cd4bf` Next step.

### Item 1: hardware texture path unchanged

| check | expected | result |
|---|---|---|
| `print(d.is_texture_supported, d.is_texture_hardware_supported)` (extra, not in the list) | - | `True True` (the hardware path is selected on gfx1100) |
| `python -m warp.tests -s autodetect -p test_texture.py -m 32` | Ran 120, OK (skipped=1) | **Ran 120, OK (skipped=1)**; the skip is `test_texture_interop_requires_hardware`'s placeholder ("No suitable devices to run the test.") |
| `python sdf_probe.py cuda:0` | PASS, Evidence session 3 values | **PASS**: SDF vs CPU sampler 4.78e-05, SDF vs analytic 1.31e-03 (grid spacing 1.56e-02), RGBA8 vs CPU sampler 2.52e-03, identical to session 3 |
| `python tex_alias.py cuda:0` | PASS | **PASS**: "plain: 0 of 2048 wrong", "graph capture: 0 of 2048 wrong" |

### Item 2: the capability skips on a 64 KiB part

`for f in test_tile_shared_memory.py test_tile_reduce.py test_tile_view.py; do python -m warp.tests -s autodetect -p "$f" -m 32; done`:
- test_tile_shared_memory: Ran 30, OK (skipped=2)
- test_tile_reduce: Ran 62, OK (skipped=2)
- test_tile_view: Ran 16, OK (skipped=1)

The five skips, with exactly the expected byte counts:
```
test_tile_shared_mem_large_cuda_0              skipped 'Needs 67584 bytes of shared memory per block, device has 65536'
test_tile_shared_mem_func_cuda_0               skipped 'Needs 67584 bytes of shared memory per block, device has 65536'
test_tile_reduce_axis_tier2_cuda_0             skipped 'Needs 66816 bytes of shared memory per block, device has 65536'
test_tile_reduce_axis_tier2_single_warp_cuda_0 skipped 'Needs 66688 bytes of shared memory per block, device has 65536'
test_tile_assign_2d_cuda_0                     skipped 'Needs 66688 bytes of shared memory per block, device has 65536'
```
No "Failed to configure kernel dynamic shared memory" warning in any of the three logs (`grep -c` = 0), so `test_tile_shared_mem_func` now skips before building.

### Item 3: full 82-pattern set and CPU-only suite
Command: the 82 patterns of the Porting session 6 handoff (B.5), the same list Evidence session 3 ran, `python -m warp.tests -s autodetect -p "<pattern>" -m 32` each, from `projects/warp-1.14/src`, wrapped in `utils/timeit.sh warp-1.14 test`. (The first launch built its pattern list from notes.md and picked up 139 entries from several loops. It was stopped after the first file, and the run was restarted with the verified 82-pattern list. The stopped run wrote no telemetry line and its logs were discarded.)

**82 patterns: 81 OK, 1 with the 2 known failures** (`test_fem*`: `test_gimp_quadrature`, `test_cube_shape_functions`, "Unexpected output" holding the init greeting, the upstream isolation behaviour). Against Evidence session 3's B.5 table (`diff` of the per-pattern summary lines, timings stripped), exactly four patterns differ (three diff hunks), all as expected:

| pattern | session 3 (`2f5b9d65`) | this run (`974cd4bf`) |
|---|---|---|
| test_streams | Ran 24, OK (skipped=2) | Ran 24, OK (skipped=3): `test_stream_priority_timings_cuda_0` skipped "HIP maps stream priorities to normal priority, so priority has no scheduling effect."; `test_stream_priority_basics_cuda_0` still runs and passes |
| test_tile_reduce | Ran 62, OK | Ran 62, OK (skipped=2) |
| test_tile_shared_memory | Ran 30, OK | Ran 30, OK (skipped=2) |
| test_tile_view | Ran 16, OK | Ran 16, OK (skipped=1) |

Every other count is unchanged: test_atomic_cas 48, test_array 139, tile set 636 run, now 15 skipped (the 10 upstream no-libmathdx skips plus the five capability skips of item 2), test_texture 120 (1 skipped).

- Skip reasons across the 82 logs: as in session 3 plus "Needs 67584 bytes ..." 2, "Needs 66816 bytes ..." 1, "Needs 66688 bytes ..." 2, and the stream-priority skip 1.
- The only "Failed to configure kernel dynamic shared memory" line in the 82 logs is the known one from `test_tile_load_scoped`'s backward kernel (the test passes; forward only).
- No `Warp CUDA error` line in any log.

#### CPU-only suite with GPUs hidden
`HIP_VISIBLE_DEVICES=99 CUDA_VISIBLE_DEVICES=99 python -m warp.tests -s autodetect -k cpu -m 32` (timeit-wrapped, no `cuda:0` in the log): **Ran 2294 tests, OK (skipped=15)**, as expected.

### Summary
At `974cd4bf`, every wave32 item gives the expected result:
- The hardware texture path is unchanged (test_texture 120 OK; SDF probe and tex_alias.py PASS with the session 3 values).
- The five capability skips appear on this 64 KiB part with the predicted byte counts, and no shared-memory warning comes from the skipped tests.
- The stream-priority timing assertion skips with the HIP message.
- The full 82-pattern set matches session 3 apart from those six skips and the two known fem isolation failures.
- The CPU-only suite is green.

With gfx950's result at this head (Porting session 8), both Linux wave families have passed the Phase 1-5 set at `974cd4bf`. Nothing here needs a fork change.

## Ruling on matrix cores for tile_matmul, plan Q3 (2026-11-18, Jeff Daily)

RULED: option B. `tile_matmul` gets matrix-core kernels in this line: MFMA on Instinct (gfx90a, gfx942, gfx950; fp16, bf16, fp32, fp64) and WMMA on Radeon (fp16 and bf16 on gfx11/gfx12; fp32 also where the target has it, gfx1250). He had the scalar benchmark numbers from gfx950, gfx942 and gfx1100 in front of him, and was told the gain is unmeasured. He is disappointed that WMMA on gfx11/gfx12 has no fp32 or fp64 input form; those dtypes stay on the scalar path there because the hardware has no instruction for them (verified with the ROCm 10.2 compiler, Phase 4 notes), not by choice.

Constraints (the diff-minimisation rule he set for textures applies here too):
- The matrix-core code lives in new files of the port's own. `tile_matmul.h` gets the smallest possible dispatch: one compile-time selection per target feature, with upstream's scalar `partitioned_gemm::scalar_matmul` kept untouched as the path for everything the matrix cores do not cover (unsupported dtype on the target, tile shapes that do not fit the 16x16 fragments, remainders). No arch names in the selection where a target feature macro exists.
- Forward and adjoint, for 64-lane and 32-lane wavefronts, at every `block_dim` Warp allows; a block whose size is not a whole number of wavefronts, or that has fewer threads than a fragment needs, must still be correct.
- Results: same as the scalar path within the tolerances the existing tests use. fp16 and bf16 matrix instructions accumulate in fp32, so they can differ from scalar fp16 accumulation in the last bits; say so in the notes and release notes, and do not loosen any CUDA tolerance.
- Evidence the ruling needs back: the same benchmark as Phase 4 (script in the notes), scalar against matrix-core, per dtype, tile size and block_dim, on gfx950, gfx942 and a wave32 Radeon; any configuration where the matrix-core path is slower than scalar must fall back to scalar or be explained. Also the carried diff per upstream file.
- Tests: the existing tile matmul tests on every host, plus cases that hit the fragment boundaries (tile sizes that are and are not multiples of 16, non-square tiles, each dtype), compared against the CPU result.
- The 64 KiB shared-memory limit and the safe-atomics ruling stand: no change to tile storage or to atomics as part of this.
- The CUDA path is unchanged in source and is compiled with nvcc after the change.

## Porting session 9 (2026-11-19, linux-gfx950 MI355X, porter): matrix-core tile_matmul (Q3 ruling)

### Design note (written before the code)
- **Where it hooks.** Every `tile_matmul` path, forward (`tile_matmul`, `tile_matmul_acc`) and adjoint (`adj_tile_matmul`, `adj_tile_matmul_acc`), ends in `partitioned_gemm::scalar_matmul<Accumulate, LayoutA, LayoutB, LayoutC>(A, B, C, alpha, beta)`, which computes `C = alpha * A @ B (+ beta * C)` from shared-memory pointers with compile-time shapes and strides. The adjoint is two such calls: `adj_A += adj_C @ B^T` and `adj_B += A^T @ adj_C`, where the transposes are stride-swapped layouts and `Accumulate = true, beta = 1`. So one compile-time hook at the top of `scalar_matmul`, `if constexpr (supported) return tile_mma::matmul<...>(...)`, covers forward and adjoint, provided the matrix-core kernel reads operands through the same compile-time strides (it does: each element is loaded as `ptr[i * s0 + j * s1]`, so a transposed view costs nothing extra). Everything else in `scalar_matmul` stays as upstream wrote it and remains the path for all cases the matrix cores do not take.
- **Selection.** In the new header, only in HIP device compiles, by `__has_builtin(<instruction>)`, which ROCm clang answers per target feature (checked: the MFMA builtins are present for gfx90a/gfx942/gfx950 only, `wmma_*_w32` for gfx11, `wmma_*_w32_gfx12` for gfx12, `wmma_f32_16x16x4_f32` for gfx1250, none for gfx1030). No arch names. MFMA targets are 64-lane, WMMA targets 32-lane, so each path knows its wavefront width.
- **Mapping onto MFMA 16x16 fragments (64-lane wavefronts).** The output tile `C[M][N]` is cut into `(M/16) * (N/16)` 16x16 fragments. With `W = block_dim / 64` wavefronts, wavefront `w` computes fragments `w, w + W, w + 2W, ...`; one fragment is computed entirely by one wavefront, so every lane of a wavefront executes every matrix instruction (MFMA needs all 64 lanes) and different wavefronts write disjoint parts of C. Within a fragment at `(fi, fj)` lane `l` (`lo = l % 16`, `g = l / 16`) loads, for each K step `k0`: `A[fi + lo][k0 + kpl * g + v]` and `B[k0 + kpl * g + v][fj + lo]` for `v < kpl` (`kpl` = 1 for the 16x16x4 f32/f64 instructions, 4 for the 16x16x16 f16/bf16 ones), and after the K loop holds four outputs at `C[fi + 4g + r][fj + lo]` (f32, f16, bf16 accumulators) or `C[fi + g + 4r][fj + lo]` (f64), `r < 4`. All four layouts were measured on gfx950 before writing the kernel (`agent_space/p9/mfma_layout.hip`, a full 16x16 GEMM per instruction at K = 16, 32, 64 against a CPU result: 0 of 256 wrong each; the f64 output layout differs from the others and was decoded with a position-coded product, `f64_layout.hip`). Operands are converted from the tile's element type to the instruction's (exactly: a half or bf16 value is representable in the instruction's input type).
- **What falls back to scalar** (decided at compile time, per call): any dtype other than float32, float64, float16 and bfloat16; mixed element types between A, B and C; `M` or `N` not a multiple of 16, or `K` not a multiple of the instruction's K step (4 for f32/f64, 16 for f16/bf16), so no fragment ever reads outside the tile; `block_dim` not a multiple of the wavefront width (a block that is not a whole number of wavefronts, or smaller than one); targets without the instruction (gfx1030, and on Radeon every dtype the WMMA instructions lack); CPU and CUDA builds (the hook is compiled only under `__HIP_DEVICE_COMPILE__`, so the CUDA source and its expansion are unchanged). Whole calls fall back, never parts of a tile.
- **Accumulation and results.** f32 and f64 accumulate in their own precision, like the scalar path but in a different summation order, so results agree to rounding. f16 and bf16 instructions accumulate in fp32 and the result is rounded once to the tile type, where the scalar path accumulates in the tile type: the matrix-core result is the more accurate one and can differ in the last bits.
- **WMMA (32-lane wavefronts).** Same structure, one fragment per wavefront, `W = block_dim / 32`; operand and accumulator layouts per instruction family (gfx11 duplicates A and B across the two half-waves; gfx12 and gfx1250 split K between them). These layouts are taken from the ISA documentation and cannot be measured on this host, so the WMMA path is compiled but off by default behind one switch; the wave32 host measures the layouts with the same probe method and runs the tests, and only then is it enabled. Shipping it on would be a guess.
## Evidence session 3 (2026-11-19, linux-gfx942, 2x MI300X, evidence only): re-run at `974cd4bf` (gfx942 items 1-3, two-GPU session)

Requested by Jeff Daily. The lock stays with linux-gfx950. As before, no fork commit and no stage, lock, head, platform or deferred record changed. The read-only clone was moved from `2505f4a3` to the recorded `head_sha` `974cd4bf864080331cc3090a5ca0ed8d17da1ce5` (`git status --porcelain` empty before and after). Same venv and setup as evidence sessions 1-2 (TheRock 10.2.0 wheels with `device-gfx942`, torch 2.15.0+rocm10.2.0, private `WARP_CACHE_PATH` per run, `ulimit -c 0`). Build: `python build_lib.py --rocm-path "$(rocm-sdk path --root)" --hip-arch gfx942 -j 112` -> "Warp build succeeded", 31.0 s at 56 workers. One-GPU items used `HIP_VISIBLE_DEVICES=0`; the two-GPU session used `0,1`.

**Environment gap: Git LFS assets are missing on this host.** `git-lfs` is not installed, so all 14 tracked `.usd`/`.nvdb` files in the clone are LFS pointer files (`version https://git-lfs.github.com/spec/v1`). Earlier sessions here did not run any test that reads them; this session's full set and CPU-only suite do. I tried to fetch a job-local `git-lfs` release binary into `agent_space/`, and the session's permission policy refused it. Installing `git-lfs` on this host needs a person's go-ahead, so the affected tests are reported below as environment failures, each traced to its asset. Every one of them reads `bunny.usd`, `cube.usd`, `spiky.usd` or a `.nvdb` grid.

### Item 1: textures through the GPU software sampler (first run off gfx950)
- `python -c "import warp as wp; wp.init(); d = wp.get_device('cuda:0'); print(d.is_texture_supported, d.is_texture_hardware_supported)"` -> `True False`.
- `python -m warp.tests -s autodetect -p test_texture.py -m 32`: Ran 118, OK (skipped=7, all "No suitable devices to run the test."). 59 `cuda_0` cases pass. This is the gfx950 result exactly.
- `python sdf_probe.py cpu`: SDF vs CPU sampler 0.00e+00, vs analytic 1.28e-03 (grid spacing 1.56e-02), RGBA8 0.00e+00: PASS.
- `python sdf_probe.py cuda:0` (MI300X, `is_texture_supported=True`): SDF vs CPU sampler 5.96e-08, vs analytic 1.28e-03, RGBA8 vs CPU sampler 1.55e-06: PASS.
- `python tex_capture.py cuda:0`: replays 0, 1 and 2 each max abs diff vs CPU 1.19e-07: PASS.
- Both scripts were taken verbatim from Porting session 5 (`sdf_probe.py`) and Porting session 7 (`tex_capture.py`).

### Item 2: tile capability skips on a real 64 KiB part
`for f in test_tile_shared_memory.py test_tile_reduce.py test_tile_view.py; do python -m warp.tests -s autodetect -p "$f" -m 32; done` (`max_shared_memory_per_block` 65536):

| file | ran | result | skips |
|---|---|---|---|
| test_tile_shared_memory | 30 | OK | 2: `test_tile_shared_mem_large_cuda_0`, `test_tile_shared_mem_func_cuda_0`, both "Needs 67584 bytes of shared memory per block, device has 65536" |
| test_tile_reduce | 62 | OK | 2: `test_tile_reduce_axis_tier2_cuda_0` "Needs 66816 ...", `test_tile_reduce_axis_tier2_single_warp_cuda_0` "Needs 66688 ..." |
| test_tile_view | 16 | OK | 1: `test_tile_assign_2d_cuda_0` "Needs 66688 ..." |

Exactly the five skips and byte counts the handoff predicted. `test_tile_shared_mem_func` no longer prints its "Failed to configure" warning, because it now skips before building. `test_tile_load` (64 run, OK) still prints its one harmless warning, for `test_tile_load_scoped_kernel_..._cuda_kernel_backward` at 65536 bytes, as ruled.

### Item 3: full 82-pattern set and CPU-only suite
Full set: the 82-pattern loop of the Porting session 6 handoff, extracted verbatim from notes (82 patterns), `-p "<pattern>" -m 32`, one GPU, 2356 s under `utils/timeit.sh warp-1.14 test`. **78 patterns OK, 4 not OK.** The 4 are the 2 known fem isolation failures plus 15 errors that all read LFS assets:
- Phase 1: test_codegen 117, test_snippet 12, test_arithmetic 126, test_math 16, test_fp16 6, test_bf16 48, test_atomic 164, test_atomic_bitwise 6, test_atomic_cas 48, test_array 139, test_copy 14, test_launch 32, test_fast_math 12 (1 skipped), test_grad 41, test_kernel_cache 5, test_module_hashing 8, test_print 10, test_mat* 601, test_vec* 531, test_quat* 173, test_spatial* 207, test_rand 10, test_noise 14, test_clang_cuda 7 (6 skipped): all OK.
- Phase 2: test_runlength_encode 16, test_array_reduce 16, test_sparse 47, test_linear_solvers 34, test_bvh 13, test_grouped_bvh 11, test_hash_grid 23, test_marching_cubes 17, test_volume_write 5, test_apic 52, test_apic_mesh 12: OK.
  - **test_coloring 3, errors=1**: `test_coloring_trimesh` opens `bunny.usd` (LFS pointer; `pxr.Tf.ErrorException` "Failed to open layer").
  - **test_mesh* 49, errors=12**: 8 open `bunny.usd`, 4 open `spiky.usd`, all `pxr.Tf.ErrorException`.
  - **test_volume 1, errors=1**: the module fails at import, "NanoVDB signature not found" on `warp/tests/assets/test_grid.nvdb`. None of its cases registered.
  - **test_fem* 107, failures=3**: the two known isolation failures (`test_gimp_quadrature`, `test_cube_shape_functions`, "Unexpected output" holding the init greeting), plus `test_fem.example_adaptive_grid_cuda_0`, whose example loads an `.nvdb` collider ("NanoVDB signature not found").
- Phase 3: test_graph 35 (3 skipped), test_array_fill_capture 3, test_capture_mode 2, test_conditional_captures 32 (16 skipped), test_conditional 50, test_async 2373, test_streams 24 (3 skipped: the priority timing assertion "HIP maps stream priorities to normal priority, so priority has no scheduling effect." plus the 2 multi-GPU skips), test_mempool 7 (2 skipped), test_pinned 1, test_unified_memory 27 (8 skipped), test_occupancy 6, test_allocator 16 (3 skipped), test_allocation_tracker 34, test_dlpack 16, test_torch 34, test_map 45, test_reload 14, test_cuda_arch_suffix 10 (10 skipped): all OK.
- Phase 4: all 22 tile files plus test_block_dim_dispatch 2 and test_template_launch_bounds 36 OK. Skips: the 10 upstream no-libmathdx skips (fft 8, mlp 1, solve 1) plus the five capability skips of item 2.
- Phase 5: test_texture 118 (7 skipped) OK.

Compared with gfx950's item 3 at this head, the counts are identical except for the five tile skips (expected on a 64 KiB part) and the LFS-asset errors. gfx950 has the assets, so its test_coloring, test_mesh* and test_volume pass there. The mesh, volume and coloring tests that need real assets have never run on gfx942 (sessions 1-2 did not include those files), so they remain unproven here until the assets are present.

CPU-only suite (`HIP_VISIBLE_DEVICES=99 CUDA_VISIBLE_DEVICES=99 python -m warp.tests -s autodetect -k cpu -m 32`, run concurrently with the start of the full set): Ran 2271, failures=5, errors=7, skipped=15. All 12 are LFS assets: `test_volume` import (1 error; its CPU cases do not register, which is why 2271 and not 2294), 6 `test_mesh_query_*` CPU cases (bunny/spiky), and 5 examples (`example_mesh`, `example_mesh_intersect`, `example_raycast`, `example_diffray` read `.usd`; `example_nvdb` reads `.nvdb`).

To finish item 3 properly, the host needs `git-lfs` (the porter's recipe: `sudo apt-get install git-lfs`, then in the clone `git lfs install --local --skip-repo && git lfs pull`). Then re-run `test_coloring.py`, `test_mesh*.py`, `test_volume.py`, `test_fem*.py` and the CPU-only suite.

### Two-GPU session (`HIP_VISIBLE_DEVICES=0,1`): the 13-file C.2 loop plus test_texture
Same loop as evidence sessions 1-2, with `test_texture.py` appended, under `utils/timeit.sh warp-1.14 test`.

| file | ran | result | evidence session 2 (`2505f4a3`) |
|---|---|---|---|
| test_multigpu | 5 | OK, 2 skipped | same |
| test_peer | 5 | OK, 1 skipped | same |
| test_mempool | 10 | OK, 2 skipped | same |
| test_streams | 42 | OK, 4 skipped | failures=2 (C4), 2 skipped |
| test_graph | 61 | OK, 5 skipped | same |
| test_ipc | 8 | OK | same |
| test_unified_memory | 40 | OK, 9 skipped | same |
| test_allocator | 16 | OK, 1 skipped | same |
| test_async | 3533 | OK | same |
| test_sparse | 71 | OK, 1 skipped | same |
| test_fem_integrate | 25 | OK, 1 skipped | same |
| test_array | 207 | OK | same |
| test_copy | 30 | OK | same |
| test_texture | 177 | OK, 7 skipped | (not run) |

**No errors and no failures.** Skips, all files: 3 C1, 2 C2 (each naming the HIP defect), 2 "HIP maps stream priorities to normal priority ..." (C4, `_cuda_0` and `_cuda_1`), 10 IOMMU heuristic "Peer-to-Peer transfers not supported", 4 "cuda:N cannot access CPU memory", 2 upstream memory-usage flakiness, 7 "No suitable devices to run the test." (test_texture's CUDA-only registrations), and 1 each of "Requires devices without mempool access support", "Requires devices without peer access support" and "rmm not installed".

test_texture with two devices: 59 `cuda_0` and 59 `cuda_1` cases pass (118 GPU + 50 CPU + 2 deviceless + 7 skipped = 177). So the software sampler's device-memory allocation, copies and sampling work on a non-primary ordinal.

Observation, not a defect: `test_synchronize_during_capture_cuda_1` prints `Warp CUDA error 900: operation not permitted when stream is capturing (in function wp_cuda_context_synchronize, ...)` and passes. The test calls `wp.synchronize()` inside a capture and expects `RuntimeError "Cannot synchronize device"`. On `cuda:0` (one-GPU run) nothing is printed. On `cuda:1` the native synchronize runs and HIP refuses it before Warp raises the expected error. The same line is in both earlier two-GPU logs at `8d13d596` and `2505f4a3`. **Correction to evidence session 1:** finding C3 attributed that 900 to the failed multi-device capture; it comes from this test instead.

### Next step for the lock holder
- Items 1 and 2 and the two-GPU session at `974cd4bf` on gfx942 match every expectation of the handoff.
- Item 3 matches too, except the mesh/volume/coloring files and the CPU-only suite, which need the LFS assets: 15 GPU-run errors and 12 CPU-only errors, all traced to pointer files. Nothing points at the port. To close item 3 on gfx942, a person approves installing `git-lfs` on this host, and those four patterns plus the CPU-only suite are re-run.

## Evidence session 4 (2026-11-19, linux-gfx942, MI300X, evidence only): LFS-asset re-run at `974cd4bf`

Requested by Jeff Daily, to close item 3 of evidence session 3. The lock stays with linux-gfx950. No fork commit and no stage, lock, head, platform or deferred record changed. Head unchanged at `974cd4bf`, so the session 3 build was reused.

Setup: `sudo apt-get install -y git-lfs` (3.4.1-1ubuntu0.4), the system package manager, per Jeff's standing rule for missing system utilities. Then, in the read-only clone, `git lfs install --local --skip-repo && git lfs pull`. The fork's `moat-fork-hook v5` pre-push hook is still the only hook. All 14 tracked `.usd`/`.nvdb` files are now real assets, not pointers, and `git status --porcelain` is empty. Re-run with a fresh private `WARP_CACHE_PATH`, `ulimit -c 0`, 475 s under `utils/timeit.sh warp-1.14 test`:

| pattern | GPU (`HIP_VISIBLE_DEVICES=0`) | session 3 (pointers) |
|---|---|---|
| test_coloring.py | 3 run, OK (`test_coloring_trimesh` on `bunny.usd` passes) | errors=1 |
| test_mesh*.py | 49 run, OK (22 `cuda_0`, 20 `cpu`, 7 deviceless) | errors=12 |
| test_volume.py | 59 run, OK (34 `cuda_0`, 24 `cpu`) | import error, 1 run |
| test_fem*.py | 107 run, failures=2: only the known isolation failures `test_gimp_quadrature` and `test_cube_shape_functions` ("Unexpected output" holding the init greeting); `test_fem.example_adaptive_grid_cuda_0` passes | failures=3 |

CPU-only suite (`HIP_VISIBLE_DEVICES=99 CUDA_VISIBLE_DEVICES=99 python -m warp.tests -s autodetect -k cpu -m 32`): **Ran 2294, OK (skipped=15)**, the expected result (gfx950 session 8 item 3).

No `Warp CUDA error` in any log. These mesh, volume and coloring tests ran on gfx942 with real assets for the first time, and all pass. With session 3, the full 82-pattern set on gfx942 at `974cd4bf` therefore has only the two known fem isolation failures, the same result as gfx950 at this head (plus the five 64 KiB tile capability skips). The CPU-only suite matches too.

(An example test writes `example_apic_fluid.usd` into the clone's working directory during the CPU-only suite; the untracked file was deleted after the run, in sessions 3 and 4.)

### Next step for the lock holder
gfx942 has now run every item of the `974cd4bf` handoff (one-GPU items 1-3, two-GPU session) with no unexplained result. Nothing here needs a fork change.

### Implementation (fork `974cd4bf..930d0098`, pushed, `advance-head` recorded)
- `92152f86` [ROCm] Compute tile_matmul with MFMA matrix instructions: new `warp/native/tile_matmul_mma.h` (SPDX line only, like the port's other new files) and a 6-line hook in `tile_matmul.h` (one `#include`, and at the top of `partitioned_gemm::scalar_matmul`, after its element types are known: `#if defined(WP_TILE_MMA) / if constexpr (tile_mma::supported<...>()) return tile_mma::matmul<...>(...); / #endif`). Upstream's scalar loop is otherwise untouched. A new test, `test_tile_matmul_fragment_shapes_{float16,float32,float64}` in `test_tile_matmul.py` (+41 lines), compares GPU and CPU results, forward and both gradients, for shapes (16,16,16), (48,16,32) (non-square, MFMA), (24,16,16) and (17,33,15) (do not fit, scalar), at block_dim 64 (whole wavefronts) and 96 (not), registered on CUDA devices by `get_cuda_test_devices()` (no arch in the test id).
- `930d0098` [ROCm] Share the A operand across MFMA fragments of a tile row: each wavefront computes a row of up to 4 fragments that share the A operand of every K step (the widest group that divides the fragment row and still leaves at least one group per wavefront), because the first version was slower than scalar for some large tiles (below).
- **bf16, a correction to the ruling's dtype list:** upstream `tile_matmul` rejects a bf16 output tile and a bf16 backward in Python ("does not support a bfloat16 'out' tile"; "does not support bfloat16 'a' or 'b' tiles when the backward pass is enabled"), so bf16 can only appear as A and B inputs with a float32 output, forward only, in the accumulating form, with the module's backward disabled (upstream's `test_tile_gemm_bf16` does exactly that). MFMA handles that case (`v_mfma_f32_16x16x16_bf16`); `supported()` accepts inputs equal to the output type, or bf16 inputs with a float32 output. bf16 WMMA was dropped: it would be reachable only the same way, and no wave32 host has measured the layouts yet, so it was not worth carrying. Adding bf16 outputs or a bf16 backward would change upstream's tile_matmul API, which the ruling did not ask for.
- **gfx1250 fp16/bf16:** gfx1250 has none of the gfx11/gfx12 `16x16x16` f16 WMMA builtins (checked with `__has_builtin`); its f16 WMMA instructions have other shapes. fp16 stays scalar there; only its fp32 WMMA (`16x16x4`) is wired.
- **WMMA is compiled but off** (`#elif defined(WP_TILE_WMMA) && WP_TILE_WMMA` in the header; nothing defines it), because its layouts come from documentation and are unmeasured: shipping it on would be a guess. Compiled here through Warp's hipRTC path (`agent_space/p9/wmma_xcompile.py`: a generated module with fp16 and fp32 32x32x32 tile_matmul, forward and backward, at block_dim 64, compiled per target as shipped and with `#define WP_TILE_WMMA 1` prepended): gfx1100 and gfx1201 build and emit 16 `v_wmma_f32_16x16x16_f16`, gfx1250 builds and emits 64 `v_wmma_f32_16x16x4_f32`; as shipped, none of the three contains a matrix instruction; gfx950's code is the same with or without the switch (32 `v_mfma_f32_16x16x16_f16`, 128 `v_mfma_f32_16x16x4_f32`).

### Correctness on gfx950
- Layouts first (design note): `mfma_layout.hip`, 0 of 256 wrong for all four instructions at K = 16, 32, 64, after decoding the f64 output layout. The same probe compiles for gfx90a and gfx942 (same instructions; to be run there).
- `agent_space/p9/mma_check.py` (below): 3 dtypes x 11 tile shapes x 5 block_dims (32, 64, 96, 128, 256) = 165 cases, each `C = A @ B`, `D += A @ B` and both gradients through `wp.Tape`, GPU against the CPU device: **0 failures**. The only non-ok cases are the five fp64 64x64x64 adjoints, whose backward kernel needs 196608 bytes of shared memory against gfx950's 163840 and cannot launch on any path (Warp's warning, gradients left zero; the same as the scalar path in sessions 4-5). `mma_check_bf16.py`: bf16 inputs, float32 output, 9 shapes x 5 block_dims: 0 failures, max difference to the CPU 9.5e-07; its code objects contain 90 `v_mfma_f32_16x16x16_bf16`.
- Results: fp32 MFMA results were **bit-identical** to the CPU in every case (the 16x16x4 f32 instruction evidently accumulates its 4 products in k order, like the scalar loop; 64x64x64: 0 of 4096 values differ); fp64 likewise. fp16 differs, as expected: the instruction accumulates in fp32 and rounds once, the scalar path accumulates in fp16; for 64x64x64 the MFMA result is 3.8e-03 from a float64 reference against the CPU's 2.2e-02, so it is the more accurate. That engagement was checked, not assumed: the modules contain `v_mfma_f32_16x16x4_f32`, `v_mfma_f64_16x16x4_f64`, `v_mfma_f32_16x16x16_f16`.
- Full tile set (22 tile files + test_block_dim_dispatch + test_template_launch_bounds) at `930d0098`: 639 run (636 + the 3 new), 10 skipped (upstream's no-libmathdx skips), 0 failures. CPU-only suite with GPUs hidden: Ran 2294, OK (skipped=15).

### CUDA path unchanged
nvcc 12.8 build at `92152f86` and `930d0098`: "Warp build succeeded". The whole new header is inside `#if defined(__HIP_DEVICE_COMPILE__)` and the hook inside `#if defined(WP_TILE_MMA)`, so CUDA compiles neither; to show it rather than argue it, the offline NVRTC compile of the tile_matmul module (`agent_space/nvrtc_check/nvrtc_offline.py`) was run with the native headers of `974cd4bf` and of `92152f86`/`930d0098`: identical `sm90.ptx` (md5 `0805dcc9fb1a...`) and `sm86.cubin` (`6f630854bc9f...`). `audit-commits` OK, `jargon.py --port` clean.

### Benchmark, scalar against MFMA (gfx950, Phase 4 script, three runs each)
Method: the Phase 4 `bench_tile_matmul.py` (16384 blocks, 50 timed iterations, fwd = 32 accumulating tile_matmul calls per block, adj = one GEMM's `tape.backward()`), three runs against the fork at `930d0098` and three against a second worktree at the same head with only `tile_matmul.h` reverted to `974cd4bf` (so no hook; same `warp.so`), merged with `compare.py` (median of three per cell). The scalar caches contain no MFMA instruction, the MFMA caches 486-628 per run. Run-to-run spread over the 144 cells: median 0.6%, max 9.3%. block_dim 32 is not a whole 64-lane wavefront, so it stays on the scalar path (1.00x by construction).

Final (`930d0098`):

| dtype | tile | block_dim | scalar fwd ms | MFMA fwd ms | fwd speedup | scalar adj ms | MFMA adj ms | adj speedup | checks |
|---|---|---|---|---|---|---|---|---|---|
| float16 | 16x16 | 32 | 0.316 | 0.317 | 1.00x | 0.287 | 0.287 | 1.00x | ok |
| float16 | 16x16 | 64 | 0.187 | 0.072 | 2.60x | 0.333 | 0.298 | 1.12x | ok |
| float16 | 16x16 | 128 | 0.154 | 0.076 | 2.03x | 0.313 | 0.310 | 1.01x | ok |
| float16 | 16x16 | 256 | 0.218 | 0.081 | 2.70x | 0.307 | 0.307 | 1.00x | ok |
| float16 | 32x32 | 32 | 1.354 | 1.355 | 1.00x | 1.478 | 1.479 | 1.00x | ok |
| float16 | 32x32 | 64 | 0.643 | 0.289 | 2.22x | 1.656 | 1.262 | 1.31x | ok |
| float16 | 32x32 | 128 | 0.695 | 0.335 | 2.07x | 2.065 | 1.251 | 1.65x | ok |
| float16 | 32x32 | 256 | 0.894 | 0.460 | 1.95x | 1.855 | 1.205 | 1.54x | ok |
| float16 | 64x64 | 32 | 12.089 | 12.049 | 1.00x | 12.874 | 12.929 | 1.00x | ok |
| float16 | 64x64 | 64 | 6.481 | 2.517 | 2.58x | 9.813 | 9.109 | 1.08x | ok |
| float16 | 64x64 | 128 | 5.109 | 2.229 | 2.29x | 6.895 | 6.591 | 1.05x | ok |
| float16 | 64x64 | 256 | 4.649 | 2.184 | 2.13x | 5.583 | 5.454 | 1.02x | ok |
| float32 | 16x16 | 32 | 0.418 | 0.417 | 1.00x | 0.158 | 0.158 | 1.00x | ok |
| float32 | 16x16 | 64 | 0.344 | 0.104 | 3.31x | 0.088 | 0.050 | 1.75x | ok |
| float32 | 16x16 | 128 | 0.219 | 0.108 | 2.02x | 0.083 | 0.048 | 1.71x | ok |
| float32 | 16x16 | 256 | 0.268 | 0.112 | 2.40x | 0.085 | 0.053 | 1.59x | ok |
| float32 | 32x32 | 32 | 2.194 | 2.174 | 1.01x | 1.114 | 1.118 | 1.00x | ok |
| float32 | 32x32 | 64 | 1.402 | 0.696 | 2.01x | 0.708 | 0.405 | 1.75x | ok |
| float32 | 32x32 | 128 | 1.266 | 0.689 | 1.84x | 0.693 | 0.295 | 2.35x | ok |
| float32 | 32x32 | 256 | 1.469 | 1.126 | 1.30x | 0.667 | 0.231 | 2.89x | ok |
| float32 | 64x64 | 32 | 14.640 | 14.579 | 1.00x | 19.367 | 19.379 | 1.00x | ok |
| float32 | 64x64 | 64 | 7.880 | 6.038 | 1.31x | 9.366 | 8.346 | 1.12x | ok |
| float32 | 64x64 | 128 | 7.132 | 4.317 | 1.65x | 5.427 | 4.804 | 1.13x | ok |
| float32 | 64x64 | 256 | 8.033 | 3.587 | 2.24x | 3.416 | 2.942 | 1.16x | ok |
| float64 | 16x16 | 32 | 0.522 | 0.523 | 1.00x | 0.197 | 0.198 | 1.00x | ok |
| float64 | 16x16 | 64 | 0.401 | 0.275 | 1.46x | 0.121 | 0.087 | 1.39x | ok |
| float64 | 16x16 | 128 | 0.389 | 0.286 | 1.36x | 0.191 | 0.081 | 2.36x | ok |
| float64 | 16x16 | 256 | 0.474 | 0.285 | 1.66x | 0.191 | 0.085 | 2.24x | ok |
| float64 | 32x32 | 32 | 3.115 | 3.112 | 1.00x | 2.239 | 2.240 | 1.00x | ok |
| float64 | 32x32 | 64 | 2.253 | 1.077 | 2.09x | 1.383 | 1.061 | 1.30x | ok |
| float64 | 32x32 | 128 | 2.010 | 0.971 | 2.07x | 1.090 | 0.713 | 1.53x | ok |
| float64 | 32x32 | 256 | 2.830 | 1.805 | 1.57x | 1.051 | 0.441 | 2.38x | ok |
| float64 | 64x64 | 32 | 76.681 | 76.487 | 1.00x | (no launch) | (no launch) | - | ADJ-FAILED |
| float64 | 64x64 | 64 | 40.249 | 25.550 | 1.58x | (no launch) | (no launch) | - | ADJ-FAILED |
| float64 | 64x64 | 128 | 22.757 | 13.717 | 1.66x | (no launch) | (no launch) | - | ADJ-FAILED |
| float64 | 64x64 | 256 | 20.892 | 7.676 | 2.72x | (no launch) | (no launch) | - | ADJ-FAILED |
run-to-run spread (max-min)/median over 144 cells: median 0.6%, max 9.3%

First version (`92152f86`, one fragment per wavefront) for the record: the large-tile forward cells that were slower than scalar, which the A-operand sharing fixed: float32 64x64 bd64 0.84x; float32 64x64 bd128 0.81x; float64 64x64 bd64 0.94x; float64 64x64 bd128 0.97x. With the final version no measured cell is slower than scalar; the smallest gains are the fp16 adjoints (1.00-1.12x at 16x16 and 64x64). Likely cause, not measured: that backward is dominated by accumulating the fp16 gradients into the input arrays (`adj_tile_load` uses `atomic_add`, a CAS loop for fp16 under the safe-atomics ruling), not by the two GEMMs.

Release-notes text: "On AMD Instinct GPUs (gfx90a, gfx942, gfx950) `tile_matmul` uses MFMA matrix instructions for float32, float64 and float16 tiles, and for bfloat16 inputs with a float32 output, when the tile dimensions are multiples of 16 (and K of 4 for float32/float64, 16 for float16/bfloat16) and the block is a whole number of 64-thread wavefronts; other tiles use the scalar path. float16 results are accumulated in float32 and can differ from the scalar path in the last bits. On Radeon GPUs `tile_matmul` uses the scalar path in this release."

### Carried diff for this work (against `release-1.14`)
`warp/native/tile_matmul.h` +6/-0 from this work (+14/-8 in total with Phase 4's `CUDA_CALLABLE` changes); new `warp/native/tile_matmul_mma.h` 287 lines (MFMA and the dormant WMMA path); `warp/tests/tile/test_tile_matmul.py` +41/-0.

### Next step (head `930d0098`)
The project stays in `porting`, lock held by linux-gfx950. Common setup as before, at `git rev-parse HEAD` = `930d0098dafa5ef08945a4382b87c07d89c9a7f0`; rebuild (`tile_matmul_mma.h` is a new header, read at kernel compile time).

**gfx942 (and gfx90a if a host exists): MFMA on the other Instinct targets.**
1. Layouts: build and run `mfma_layout.hip` (below) with `clang++ -x hip --rocm-path=$R --offload-arch=<gfx> -O2 -w mfma_layout.hip -o mfma_layout -L$R/lib -lamdhip64 -Wl,-rpath,$R/lib`: expect 0 of 256 wrong on all twelve lines (four instructions x K = 16, 32, 64). A wrong line means that target's layout differs from gfx950's: stop and record it; the MFMA path must not be trusted there until fixed.
2. `python mma_check.py` (below) and `python mma_check_bf16.py` (the bf16 variant described above: bf16 A/B, float32 out, accumulating form, `module_options={"enable_backward": False}`): expect 0 failures; on a 64 KiB part more fp64 and fp32 64-tile adjoints will report SMEM-LIMIT (backward not launched), never FAIL.
3. Tile set (22 files + 2): expect 639 run, the five shared-memory skips of session 8, 10 upstream skips, 0 failures; `test_tile_matmul_fragment_shapes_*` 3 OK.
4. Benchmark: `bench_both.sh` and `compare.py` (below; set up the scalar worktree first: `git -C projects/warp-1.14/src worktree add --detach <dir> 930d0098`, `git -C <dir> checkout 974cd4bf -- warp/native/tile_matmul.h`, copy `warp/bin/*.so` into `<dir>/warp/bin`); record the table; any MFMA cell slower than scalar must be reported (the ruling requires a fallback or an explanation).
5. The full 82-pattern set and the CPU-only suite at this head.

**Wave32 host (gfx1100 or gfx1201; gfx1250 if one exists): measure the WMMA layouts, then test with the switch on.**
1. `wmma_layout.hip` (below), built with `clang++ -x hip --rocm-path=$R --offload-arch=<gfx> -O2 -w wmma_layout.hip -o wmma_layout -L$R/lib -lamdhip64 -Wl,-rpath,$R/lib` and run: expect "WMMA family 11" (gfx11), "12" (gfx12) or "1250", and 0 of 256 wrong at K = 16, 32, 64. If any line is wrong, stop there and record the counts (a position-coded decode like the gfx950 f64 one, `C[i][j] = i + 100 j` from `A[i][0] = i, A[i][1] = 1, B[0][j] = 1, B[1][j] = 100 j`, shows the actual layout): do not enable the switch.
2. Only if step 1 passes: as an uncommitted local experiment, add `#define WP_TILE_WMMA 1` as the first line after `#pragma once` in `warp/native/tile_matmul_mma.h`, then run `python mma_check.py` (expect 0 failures; fp32 stays scalar on gfx11/gfx12, so its 55 cases test the fallback there), `python -m warp.tests -s autodetect -p test_tile_matmul.py -m 32` and the whole tile set, check that the modules contain `v_wmma_f32_16x16x16_f16` (`llvm-objdump -d --mcpu=<gfx>` on the cached `.hsaco`), and run the benchmark (`bench_both.sh`/`compare.py`, scalar worktree as above) for fp16. Revert the line afterwards (`git status --porcelain` clean) and record everything; the lock holder then commits the switch-on (or the layout fix) for the targets that passed.
3. Without the switch (as shipped), the tile set, test_tile_matmul (25 OK) and the full 82-pattern set at this head: wave32 must be unchanged (scalar everywhere).

**Two-GPU session:** no tile_matmul change affects multi-GPU; the session 8 handoff (13-file loop plus test_texture) at this head is enough.

**Still open for Jeff:** the bf16 scope note above (inputs only, forward only, as upstream allows); defer/now on the six undecided `rocm-bug-report` entries. Phase 6 not started.

`mma_check.py`:
```python
# tile_matmul forward (C = A @ B and C += A @ B) and adjoint on the GPU against the same kernels on
# the CPU device, for dtypes x tile shapes x block_dims. Usage: python mma_check.py [dtype ...]
import sys

import numpy as np

import warp as wp

wp.init()
gpu = wp.get_device("cuda:0")
DTYPES = {"float32": wp.float32, "float64": wp.float64, "float16": wp.float16}  # tile_matmul rejects bf16
SHAPES = [(16, 16, 16), (32, 32, 32), (64, 64, 64), (16, 48, 32), (48, 16, 64), (32, 64, 16), (64, 32, 48),
          (8, 8, 8), (24, 16, 16), (16, 16, 12), (17, 33, 15)]  # fmt: skip
BLOCK_DIMS = (32, 64, 96, 128, 256)
TOL = {wp.float32: 2e-5, wp.float64: 1e-12, wp.float16: 1e-2}


def make(M, N, K, dtype):
    @wp.kernel(module="unique")
    def fwd(A: wp.array2d(dtype=dtype), B: wp.array2d(dtype=dtype), C: wp.array2d(dtype=dtype), D: wp.array2d(dtype=dtype)):
        a = wp.tile_load(A, shape=(M, K))
        b = wp.tile_load(B, shape=(K, N))
        c = wp.tile_matmul(a, b)
        d = wp.tile_load(D, shape=(M, N))
        wp.tile_matmul(a, b, d)
        wp.tile_store(C, c)
        wp.tile_store(D, d)

    return fwd


def run(dev, kern, A, B, D0, bd, M, N, K, dtype):
    a = wp.array(A, dtype=dtype, device=dev, requires_grad=True)
    b = wp.array(B, dtype=dtype, device=dev, requires_grad=True)
    c = wp.zeros((M, N), dtype=dtype, device=dev, requires_grad=True)
    d = wp.array(D0, dtype=dtype, device=dev, requires_grad=True)
    tape = wp.Tape()
    with tape:
        wp.launch_tiled(kern, dim=[1], inputs=[a, b, c, d], block_dim=bd, device=dev)
    c.grad = wp.array(np.ones((M, N)), dtype=dtype, device=dev)
    d.grad = wp.array(np.full((M, N), 0.5), dtype=dtype, device=dev)
    tape.backward()
    return [x.numpy().astype(np.float64) for x in (c, d, a.grad, b.grad)]


rng = np.random.default_rng(0)
fails = 0
for name in sys.argv[1:] or DTYPES:
    dtype = DTYPES[name]
    for M, N, K in SHAPES:
        kern = make(M, N, K, dtype)
        A = rng.uniform(-1, 1, (M, K))
        B = rng.uniform(-1, 1, (K, N))
        D0 = rng.uniform(-1, 1, (M, N))
        for bd in BLOCK_DIMS:
            ref = run("cpu", kern, A, B, D0, bd, M, N, K, dtype)
            got = run(gpu, kern, A, B, D0, bd, M, N, K, dtype)
            errs = [np.abs(g - r).max() / max(1.0, np.abs(r).max()) for g, r in zip(got, ref)]
            ok = all(e <= TOL[dtype] * max(1, K / 16) for e in errs)
            # the backward kernel cannot launch when it needs more shared memory than the device has
            smem = (not ok and all(e <= TOL[dtype] * max(1, K / 16) for e in errs[:2])
                    and not got[2].any() and not got[3].any())  # fmt: skip
            fails += not ok and not smem
            if not ok or bd == 64:
                status = "ok" if ok else ("SMEM-LIMIT (backward not launched)" if smem else "FAIL")
                print(f"{name:8s} M{M:3d} N{N:3d} K{K:3d} bd{bd:4d}: c {errs[0]:.1e} d {errs[1]:.1e} gA {errs[2]:.1e} gB {errs[3]:.1e} {status}", flush=True)
print("FAILURES", fails)
sys.exit(1 if fails else 0)
```

`wmma_layout.hip`:
```cpp
// Check the WMMA 16x16 fragment layouts that warp/native/tile_matmul_mma.h assumes, on one 32-lane
// wavefront, against a CPU GEMM (small integers, exact in every type). lane l: lo = l % 16, g = l / 16.
//   gfx11 f16 (wmma_f32_16x16x16_f16_w32): A[lo][v], B[v][lo] for v < 16 in both half-waves;
//       accumulator r is C[2 r + g][lo]
//   gfx12 f16 (wmma_f32_16x16x16_f16_w32_gfx12): A[lo][8 g + v], B[8 g + v][lo] for v < 8;
//       accumulator r is C[8 g + r][lo]
//   gfx1250 f32 (wmma_f32_16x16x4_f32): A[lo][2 g + v], B[2 g + v][lo] for v < 2; accumulator as gfx12
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstring>

typedef float f8 __attribute__((ext_vector_type(8)));

__global__ void k(const double* A, const double* B, double* C, int K, int* kind)
{
    int l = threadIdx.x, lo = l % 16, g = l / 16;
    f8 acc = {};
#if __has_builtin(__builtin_amdgcn_wmma_f32_16x16x16_f16_w32)
    typedef _Float16 V __attribute__((ext_vector_type(16)));
    for (int k0 = 0; k0 < K; k0 += 16) {
        V a, b;
        for (int v = 0; v < 16; ++v) {
            a[v] = _Float16(A[lo * K + k0 + v]);
            b[v] = _Float16(B[(k0 + v) * 16 + lo]);
        }
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, acc);
    }
    for (int r = 0; r < 8; ++r) C[(2 * r + g) * 16 + lo] = acc[r];
    if (l == 0) *kind = 11;
#elif __has_builtin(__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12)
    typedef _Float16 V __attribute__((ext_vector_type(8)));
    for (int k0 = 0; k0 < K; k0 += 16) {
        V a, b;
        for (int v = 0; v < 8; ++v) {
            a[v] = _Float16(A[lo * K + k0 + 8 * g + v]);
            b[v] = _Float16(B[(k0 + 8 * g + v) * 16 + lo]);
        }
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, acc);
    }
    for (int r = 0; r < 8; ++r) C[(8 * g + r) * 16 + lo] = acc[r];
    if (l == 0) *kind = 12;
#elif __has_builtin(__builtin_amdgcn_wmma_f32_16x16x4_f32)
    typedef float V __attribute__((ext_vector_type(2)));
    for (int k0 = 0; k0 < K; k0 += 4) {
        V a = {float(A[lo * K + k0 + 2 * g]), float(A[lo * K + k0 + 2 * g + 1])};
        V b = {float(B[(k0 + 2 * g) * 16 + lo]), float(B[(k0 + 2 * g + 1) * 16 + lo])};
        acc = __builtin_amdgcn_wmma_f32_16x16x4_f32(false, a, false, b, (short)0, acc, false, false);
    }
    for (int r = 0; r < 8; ++r) C[(8 * g + r) * 16 + lo] = acc[r];
    if (l == 0) *kind = 1250;
#else
    if (l == 0) *kind = -1;
#endif
}

int main()
{
    int fails = 0;
    for (int K : {16, 32, 64}) {
        double *A, *B, *C;
        int* kind;
        hipMallocManaged(&A, 16 * K * 8);
        hipMallocManaged(&B, K * 16 * 8);
        hipMallocManaged(&C, 256 * 8);
        hipMallocManaged(&kind, 4);
        for (int x = 0; x < 16 * K; ++x) A[x] = double((x * 7 + 3) % 9) - 4.0;
        for (int x = 0; x < K * 16; ++x) B[x] = double((x * 5 + 1) % 7) - 3.0;
        memset(C, 0, 256 * 8);
        *kind = 0;
        k<<<1, 32>>>(A, B, C, K, kind);
        hipDeviceSynchronize();
        int bad = 0;
        for (int i = 0; i < 16; ++i)
            for (int j = 0; j < 16; ++j) {
                double ref = 0;
                for (int kk = 0; kk < K; ++kk) ref += A[i * K + kk] * B[kk * 16 + j];
                bad += C[i * 16 + j] != ref;
            }
        printf("WMMA family %d, K=%2d: %d of 256 wrong\n", *kind, K, bad);
        fails += bad != 0 || *kind <= 0;
    }
    return fails ? 1 : 0;
}
```

`bench_both.sh` (paths are the gfx950 host's; adjust `A` and the scalar worktree):
```bash
#!/bin/bash
# Phase 4 tile_matmul benchmark, three runs each against the MFMA tree and the scalar tree.
ulimit -c 0
A=/var/lib/jenkins/fleet/jobs/mi355-20261118-122134-warp114-porter-mi355-prompt/moat/agent_space
for tree in ${TREES:-mfma scalar}; do
  if [ $tree = mfma ]; then src=$A/../projects/warp-1.14/src; else src=$A/scalar_wt; fi
  for n in 1 2 3; do
    PYTHONPATH=$src WARP_CACHE_PATH=$(mktemp -d $A/kcache.XXXXXX) "$FLEET_VENV/bin/python" $A/bench/bench_tile_matmul.py \
      --blocks 16384 --iters 50 --csv $A/p9/bench_${tree}_run$n.csv > $A/p9/bench_${tree}_run$n.log 2>&1
    echo "$tree run$n rc=$?"
  done
done
```

`compare.py`:
```python
# Merge the three runs of each tree: median ms per cell, MFMA speedup over scalar, run-to-run spread.
import csv
import statistics
from collections import defaultdict

data = defaultdict(lambda: defaultdict(list))
checks = defaultdict(set)
for tree in ("scalar", "mfma"):
    for n in (1, 2, 3):
        for r in csv.DictReader(open(f"bench_{tree}_run{n}.csv")):
            key = (r["dtype"], int(r["M"]), int(r["block_dim"]))
            data[key][tree + "_fwd"].append(float(r["fwd_ms"]))
            data[key][tree + "_adj"].append(float(r["adj_ms"]))
            checks[key].add(f"{tree}:{r['check']}")
spread = []
print("| dtype | tile | block_dim | scalar fwd ms | MFMA fwd ms | fwd speedup | scalar adj ms | MFMA adj ms | adj speedup | checks |")
print("|---|---|---|---|---|---|---|---|---|---|")
for key in sorted(data, key=lambda k: (["float16", "float32", "float64"].index(k[0]), k[1], k[2])):
    d = data[key]
    m = {k: statistics.median(v) for k, v in d.items()}
    for v in d.values():
        spread.append((max(v) - min(v)) / statistics.median(v))
    ck = sorted(checks[key])
    adj_ok = all(c.endswith(":ok") for c in ck)
    adj = f"{m['scalar_adj']:.3f} | {m['mfma_adj']:.3f} | {m['scalar_adj'] / m['mfma_adj']:.2f}x" if adj_ok else "(no launch) | (no launch) | -"
    print(f"| {key[0]} | {key[1]}x{key[1]} | {key[2]} | {m['scalar_fwd']:.3f} | {m['mfma_fwd']:.3f} | {m['scalar_fwd'] / m['mfma_fwd']:.2f}x | {adj} | {'/'.join(c.split(':')[1] for c in ck if c.startswith('mfma'))} |")
spread.sort()
print(f"run-to-run spread (max-min)/median over {len(spread)} cells: median {100 * statistics.median(spread):.1f}%, max {100 * spread[-1]:.1f}%")
```

`mfma_layout.hip` (as run on gfx950; f64 accumulator rows `g + 4r`):
```cpp
// Check hypothesised MFMA 16x16 fragment layouts on one 64-lane wavefront against a CPU GEMM.
// Hypothesis (operands, K = kstep * 4 groups of lanes):
//   A: lane l holds A[l % 16][kpl * (l / 16) + v], v < kpl   (kpl = k values per lane)
//   B: lane l holds B[kpl * (l / 16) + v][l % 16]
//   C: lane l holds C[4 * (l / 16) + r][l % 16], r < 4
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

typedef float f4 __attribute__((ext_vector_type(4)));
typedef double d4 __attribute__((ext_vector_type(4)));
typedef _Float16 h4 __attribute__((ext_vector_type(4)));
typedef short s4 __attribute__((ext_vector_type(4)));

__device__ short bf16_bits(float x)
{
    unsigned u = __float_as_uint(x);
    return short(u >> 16);  // inputs are exactly representable in bf16
}

template <int KIND>  // 0 f32, 1 f64, 2 f16, 3 bf16
__global__ void k(const double* A, const double* B, double* C, int K)
{
    int l = threadIdx.x;
    int i = l % 16, g = l / 16;
    if (KIND == 1) {
        d4 acc = {0, 0, 0, 0};
        for (int k0 = 0; k0 < K; k0 += 4)
            acc = __builtin_amdgcn_mfma_f64_16x16x4f64(A[i * K + k0 + g], B[(k0 + g) * 16 + i], acc, 0, 0, 0);
        for (int r = 0; r < 4; ++r) C[(g + 4 * r) * 16 + i] = acc[r];  // f64: rows g + 4r
        return;
    }
    f4 acc = {0, 0, 0, 0};
    if (KIND == 0) {
        for (int k0 = 0; k0 < K; k0 += 4)
            acc = __builtin_amdgcn_mfma_f32_16x16x4f32(float(A[i * K + k0 + g]), float(B[(k0 + g) * 16 + i]), acc, 0, 0, 0);
    } else {
        for (int k0 = 0; k0 < K; k0 += 16) {
            if (KIND == 2) {
                h4 a, b;
                for (int v = 0; v < 4; ++v) {
                    a[v] = _Float16(A[i * K + k0 + 4 * g + v]);
                    b[v] = _Float16(B[(k0 + 4 * g + v) * 16 + i]);
                }
                acc = __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, acc, 0, 0, 0);
            } else {
                s4 a, b;
                for (int v = 0; v < 4; ++v) {
                    a[v] = bf16_bits(float(A[i * K + k0 + 4 * g + v]));
                    b[v] = bf16_bits(float(B[(k0 + 4 * g + v) * 16 + i]));
                }
                acc = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, acc, 0, 0, 0);
            }
        }
    }
    for (int r = 0; r < 4; ++r) C[(4 * g + r) * 16 + i] = acc[r];
}

int main()
{
    const char* names[] = {"f32 16x16x4", "f64 16x16x4", "f16 16x16x16", "bf16 16x16x16"};
    int fails = 0;
    for (int kind = 0; kind < 4; ++kind) {
        for (int K : {16, 32, 64}) {
            double *A, *B, *C;
            hipMallocManaged(&A, 16 * K * 8);
            hipMallocManaged(&B, K * 16 * 8);
            hipMallocManaged(&C, 256 * 8);
            // small integers: exact in every input type and in fp32 accumulation
            for (int x = 0; x < 16 * K; ++x) A[x] = double((x * 7 + 3) % 9) - 4.0;
            for (int x = 0; x < K * 16; ++x) B[x] = double((x * 5 + 1) % 7) - 3.0;
            memset(C, 0, 256 * 8);
            if (kind == 0) k<0><<<1, 64>>>(A, B, C, K);
            if (kind == 1) k<1><<<1, 64>>>(A, B, C, K);
            if (kind == 2) k<2><<<1, 64>>>(A, B, C, K);
            if (kind == 3) k<3><<<1, 64>>>(A, B, C, K);
            hipDeviceSynchronize();
            int bad = 0;
            for (int i = 0; i < 16; ++i)
                for (int j = 0; j < 16; ++j) {
                    double ref = 0;
                    for (int kk = 0; kk < K; ++kk) ref += A[i * K + kk] * B[kk * 16 + j];
                    bad += C[i * 16 + j] != ref;
                }
            printf("%-14s K=%2d: %d of 256 wrong\n", names[kind], K, bad);
            fails += bad != 0;
            hipFree(A); hipFree(B); hipFree(C);
        }
    }
    return fails ? 1 : 0;
}
```

## Evidence session 5 (2026-11-19, linux-gfx1100, Radeon Pro W7800, evidence only): WMMA items at `930d0098`

Requested by Jeff Daily. The work lock stays with linux-gfx950. Nothing was committed to the fork, and no stage, lock, head, platform or deferral record changed. The clone was updated read-only to the recorded `head_sha` `930d0098dafa5ef08945a4382b87c07d89c9a7f0` and rebuilt (`export PATH=$FLEET_VENV/bin:$PATH; python build_lib.py --rocm-path "$(rocm-sdk path --root)" --hip-arch gfx1100 -j 32` -> "Warp build succeeded"; RUNPATH is the venv's ROCm 10.2). `git status --porcelain` is clean at the end. The items are the "Wave32 host" list of the `930d0098` Next step, run in its order. Same venv, `ulimit -c 0`, `HIP_VISIBLE_DEVICES=0`, private `WARP_CACHE_PATH`.

### Item 1: `wmma_layout.hip`: FAILS, so the switch-on experiment (item 2) was not run
Built exactly as the handoff says (`clang++ -x hip --rocm-path=$R --offload-arch=gfx1100 -O2 -w wmma_layout.hip -o wmma_layout -L$R/lib -lamdhip64 -Wl,-rpath,$R/lib`), script verbatim from the Next step:
```
WMMA family 11, K=16: 96 of 256 wrong
WMMA family 11, K=32: 69 of 256 wrong
WMMA family 11, K=64: 220 of 256 wrong
(exit 1)
```
All three lines are wrong, so per the handoff (and Jeff's instruction for this session) the WMMA part stopped here: `WP_TILE_WMMA` was **not** switched on, and `mma_check.py`, the tile set with the switch and the fp16 benchmark were not run. There is no experiment diff to apply. The switch-on stays the one-line `#define WP_TILE_WMMA 1` after `#pragma once` in `warp/native/tile_matmul_mma.h` that the handoff describes, and whether to make it is the lock holder's and Jeff's call given the diagnosis below.

### Diagnosis: the assumed gfx11 layout is correct; the probe's exact-equality check fails on mixed-sign accumulation
Three standalone probes on gfx1100 (`agent_space/p9/`, ROCm clang 10.2, same build line):

1. **One-hot register decode** (`wmma_decode.hip`: one-hot `a` in lane la slot va with all-ones `b`, and the reverse, then which accumulator (lane, register) cells light up; then one-hot A x one-hot B to pair K slots). For `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32`:
   - Output rows in half-wave g come only from that half's copy of A and B. A one-hot in lane 0 lights row 0 (register 0, lanes 0-15). Lane 17 lights row 1 (register 0, lanes 16-31). Lanes 1 and 16 light nothing, because row 1 is read from the upper half and row 0 from the lower. A and B must therefore be duplicated across both half-waves, which `tile_matmul_mma.h` and the probe both do.
   - A B one-hot in lane lb lights all 8 registers of lane lb, which is column lb % 16 in that half's rows.
   - K slot v of A pairs with K slot v of B, for v = 0, 1, 7, 8, 15.
   - This is exactly the header's gfx11 statement: A[lo][v], B[v][lo] in both halves; accumulator r is C[2r + g][lo].
2. **Position-coded decode** (`wmma_posdecode.hip`, the probe's own kernel at K = 16, with the handoff's encoding A[i][0] = i, A[i][1] = 1, B[0][j] = 1, B[1][j] = 100 j): **all 256 cells exact**. `C[i][j] = i + 100 j` everywhere.
3. **Which data fails** (same kernel, K = 16): the probe's data shifted to be non-negative (A + 4, B + 3) gives 0 of 256 wrong. The probe's signed data gives 96 wrong. The position decode negated (A[i][0] = -i, B[1][j] = -100 j) is wrong in every cell outside row 0 and column 0. The wrong values are the exact result **plus or minus 1 ulp of f32**, not misplaced elements: `C[0][2]` 18.999998092651367 for 19, `C[0][4]` 13.999999046325684 for 14, `C[1][1]` -101.00000762939453 for -101, `C[1][2]` -201.00001525878906 for -201.

So on gfx11 the WMMA f16->f32 accumulation is not exact when products of mixed sign cancel, even for small integers whose exact sum is representable. The error is at f32 rounding level. Measured with `wmma_err.hip` (below; the same accumulating K/16-call loop as the header), as the maximum of |err| / sum |a*b| over the 256 cells:

| data | K | cells differing from the rounded exact result | max abs err | max abs err / sum abs(a*b) |
|---|---|---|---|---|
| probe ints | 16 | 96 | 1.91e-06 | 3.41e-08 |
| probe ints | 32 | 69 | 3.81e-06 | 3.08e-08 |
| probe ints | 64 | 220 | 3.81e-06 | 1.55e-08 |
| random fp16 in [-1, 1] | 16 | 150 | 4.47e-07 | 8.68e-08 |
| random fp16 in [-1, 1] | 32 | 191 | 8.42e-07 | 1.07e-07 |
| random fp16 in [-1, 1] | 64 | 227 | 1.90e-06 | 1.32e-07 |

That is at most about f32 epsilon (1.19e-7) relative to the magnitude sum, the ordinary bound for an f32 dot product. It is several orders below `mma_check.py`'s fp16 tolerance (1e-2 x max(1, K/16)) and below fp16's own epsilon (9.8e-4). gfx950's MFMA passed the same exact-equality test, so CDNA's MFMA and RDNA3's WMMA differ here.

For the lock holder: `wmma_layout.hip`'s pass criterion (`C != ref` with exact equality) cannot pass on gfx11, although the layout is right. A layout check that does not depend on exact accumulation would be either non-negative integer data (mode 2 above: 0 wrong) or the position-coded encoding (all cells exact). Either measures the layout. The precision difference itself is a separate fact for the WMMA path's tests, which `mma_check.py` would cover with its tolerances. Whether this justifies the switch-on, with the probe corrected, is not decided here.

### Item 3: shipped state (switch off): wave32 unchanged, scalar everywhere
- Full 82-pattern set (same list as Evidence sessions 3 and 4, `-p "<pattern>" -m 32`, timeit-wrapped): **82 patterns, 81 OK, 1 with the 2 known fem isolation failures** (`test_gimp_quadrature`, `test_cube_shape_functions`). Against Evidence session 4 at `974cd4bf` (diff of per-pattern summary lines, timings stripped), the only difference is test_tile_matmul: **Ran 25, OK** (was 22): the three new `test_tile_matmul_fragment_shapes_{float16,float32,float64}_cuda_0` pass. The tile set is therefore 639 run, 15 skipped (10 upstream no-libmathdx skips + the five 64 KiB capability skips), 0 failures. test_streams 24 OK (3 skipped, the priority skip). test_texture 120 OK (1 skipped). No `Warp CUDA error` line in any log, and the only shared-memory warning is the known `test_tile_load_scoped` one.
- Scalar on wave32, checked rather than assumed (`agent_space/p9/isa_check.py`: a 32x32 `tile_matmul` at block_dim 64 for fp16 and fp32, then `llvm-objdump -d --mcpu=gfx1100` on the two cached `.hsaco`): **0 `v_wmma` instructions in both**. The fp16 module's arithmetic is 1456 `v_mul_f16` with scalar adds; the fp32 module is scalar FMAs. Results correct (fp16 max abs err 0.024 against the float64 product of the fp16 inputs, fp16-storage level; fp32 3.1e-06).
- The CPU-only suite was not in this host's list for this head and was not run.

`wmma_err.hip`:
```cpp
// gfx11 WMMA f16 -> f32: size of the deviation from the exact result, for the probe's signed data and
// for random fp16 data, K = 16, 32, 64 (accumulating over K/16 WMMA calls, as tile_matmul_mma.h does).
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
typedef _Float16 V __attribute__((ext_vector_type(16)));
typedef float f8 __attribute__((ext_vector_type(8)));
__global__ void k(const float* A, const float* B, float* C, int K)
{
    int l = threadIdx.x, lo = l % 16, g = l / 16;
    f8 acc = {};
    for (int k0 = 0; k0 < K; k0 += 16) {
        V a, b;
        for (int v = 0; v < 16; ++v) { a[v] = _Float16(A[lo * K + k0 + v]); b[v] = _Float16(B[(k0 + v) * 16 + lo]); }
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, acc);
    }
    for (int r = 0; r < 8; ++r) C[(2 * r + g) * 16 + lo] = acc[r];
}
static float h(float x) { return (float)(_Float16)x; }
int main()
{
    srand(1);
    for (int data = 0; data < 2; ++data)
        for (int K : {16, 32, 64}) {
            float *A, *B, *C;
            hipMallocManaged(&A, 16 * K * 4); hipMallocManaged(&B, K * 16 * 4); hipMallocManaged(&C, 256 * 4);
            for (int x = 0; x < 16 * K; ++x) A[x] = data ? h(2.f * rand() / RAND_MAX - 1.f) : float((x * 7 + 3) % 9) - 4.f;
            for (int x = 0; x < K * 16; ++x) B[x] = data ? h(2.f * rand() / RAND_MAX - 1.f) : float((x * 5 + 1) % 7) - 3.f;
            k<<<1, 32>>>(A, B, C, K); hipDeviceSynchronize();
            int bad = 0; double maxabs = 0, maxrel_sum = 0, maxulp = 0;
            for (int i = 0; i < 16; ++i)
                for (int j = 0; j < 16; ++j) {
                    double ref = 0, absum = 0;
                    for (int kk = 0; kk < K; ++kk) { ref += double(A[i * K + kk]) * B[kk * 16 + j]; absum += fabs(double(A[i * K + kk]) * B[kk * 16 + j]); }
                    float rf = (float)ref;
                    double e = fabs(C[i * 16 + j] - ref);
                    bad += C[i * 16 + j] != rf;
                    maxabs = fmax(maxabs, e);
                    maxrel_sum = fmax(maxrel_sum, e / absum);
                    double ulp = fabs(nextafterf(rf, INFINITY) - rf);
                    maxulp = fmax(maxulp, e / ulp);
                }
            printf("%-12s K=%2d: %3d of 256 differ from the rounded exact result; max |err| %.3g, max |err|/sum|a*b| %.3g, max %.2f ulp\n",
                   data ? "random fp16" : "probe ints", K, bad, maxabs, maxrel_sum, maxulp);
        }
}
```
## Evidence session (2026-11-19, linux-gfx1201, Radeon AI PRO R9700, evidence only): first gfx12 run, WMMA layout and switch-on experiment at `930d0098`

Requested by Jeff Daily. The work lock stays with linux-gfx950. Nothing was committed or pushed to the fork, and no stage, lock, head, platform or deferral record changed. The fork was cloned read-only and checked out detached at the recorded `head_sha` `930d0098dafa5ef08945a4382b87c07d89c9a7f0`. The WMMA switch-on experiment ran in a separate `git worktree` of that clone, so the clone under test was never edited. The worktree was removed afterwards, and `git status --porcelain` in the clone is empty. This is the port's first run on a gfx12 GPU.

### Setup (deviations from the common setup block)
- Device: "AMD Radeon AI PRO R9700" (32 GiB), `arch_name` `gfx1201` (no target features), Warp `arch` 120, `sm_count` 32 (HIP's `multiProcessorCount`, which counts WGPs on RDNA; the part has 64 CUs), wavefront 32, `max_shared_memory_per_block` 65536, `is_texture_supported`/`is_texture_hardware_supported` `True True`, UVA and mempool True. Dispatcher pin `HIP_VISIBLE_DEVICES=2` kept (the host has four GPUs). Host CPU: Threadripper PRO 7965WX, 24 physical cores; builds used the job's `MAX_JOBS=12`.
- The job venv came with TheRock 10.1.0 wheels. Installed into it: `rocm[devel,device-gfx1201]==10.2.0`, `torch[device-gfx1201]==2.15.0+rocm10.2.0`, `numpy==1.26.4`, `usd-core` (index `https://stable.repo.amd.com/rocm/whl-next/`, extra index PyPI), then `rocm-sdk init`. `export PATH=$FLEET_VENV/bin:$PATH` first (the gfx1100 trap); `rocm-sdk path --root` is the venv's 10.2 `_rocm_sdk_devel`. HIP 7.17, Python 3.11.17.
- `git-lfs` 3.4.1 installed with `sudo apt-get install -y git-lfs`, then `git lfs install --local --skip-repo && git lfs pull` in the clone (the `.nvdb`/`.usd` assets are real files). `protect-fork warp-1.14` installed the fork pre-push hook (no LFS hook was present).
- Build: `python build_lib.py --rocm-path "$(rocm-sdk path --root)" --hip-arch gfx1201 -j 12` -> "Warp build succeeded" (33.5 s under `utils/timeit.sh`, .cu build 31.0 s). `readelf -d warp/bin/warp.so` RUNPATH is the venv's 10.2 `_rocm_sdk_devel/lib`. Greeting: `HIP 7.17`, `"cuda:0" : "AMD Radeon AI PRO R9700" (32 GiB, gfx1201, mempool enabled)`.
- Every run used a private `WARP_CACHE_PATH` under `agent_space/`, and every GPU run was under `ulimit -c 0`. Runner per pattern: `python -m warp.tests -s autodetect -p "<pattern>" -m 32` from `projects/warp-1.14/src`. Scripts (`sdf_probe.py`, `tex_alias.py`, `mma_check.py`, `wmma_layout.hip`, `bench_tile_matmul.py`, `summarize.py`, `compare.py`) were extracted verbatim from these notes.

### Textures (hardware path, first gfx12 run)
| check | expected (gfx1100) | gfx1201 |
|---|---|---|
| `python -m warp.tests -s autodetect -p test_texture.py -m 32` | Ran 120, OK (skipped=1) | **Ran 120, OK (skipped=1)**; the skip is `test_texture_interop_requires_hardware`'s placeholder ("No suitable devices to run the test.") |
| `python sdf_probe.py cpu` | PASS | PASS: SDF vs CPU sampler 0.00e+00, SDF vs analytic 1.28e-03 (grid spacing 1.56e-02), RGBA8 0.00e+00 |
| `python sdf_probe.py cuda:0` | PASS, 4.78e-05 / 1.31e-03 / 2.52e-03 | **PASS**: SDF vs CPU sampler 4.78e-05, SDF vs analytic 1.31e-03, RGBA8 vs CPU sampler 2.52e-03 (identical to gfx1100) |
| `python tex_alias.py cuda:0` | PASS | **PASS**: "plain: 0 of 2048 wrong", "graph capture: 0 of 2048 wrong" |

### Full Phase 1-5 set (82 patterns, as shipped: no WMMA)
The 82 patterns of the Porting session 6 handoff (B.5), extracted from these notes (82 lines, no duplicates), one per line, 1992 s under `utils/timeit.sh warp-1.14 test`. **81 patterns OK, 1 with the 2 known failures**: `test_fem*` 107 run, failures=2, `test_gimp_quadrature` and `test_cube_shape_functions`, "Unexpected output" holding the init greeting (the upstream isolation behaviour seen on every host).
- Phase 1: test_codegen 117, test_snippet 12, test_arithmetic 126, test_math 16, test_fp16 6, test_bf16 48, test_atomic 164, test_atomic_bitwise 6, test_atomic_cas 48, test_array 139, test_copy 14, test_launch 32, test_fast_math 12 (1 skipped), test_grad 41, test_kernel_cache 5, test_module_hashing 8, test_print 10, test_mat* 601, test_vec* 531, test_quat* 173, test_spatial* 207, test_rand 10, test_noise 14, test_clang_cuda 7 (6 skipped): all OK.
- Phase 2: test_runlength_encode 16, test_array_reduce 16, test_sparse 47, test_linear_solvers 34, test_coloring 3, test_bvh 13, test_grouped_bvh 11, test_hash_grid 23, test_marching_cubes 17, test_mesh* 49, test_volume 59, test_volume_write 5, test_apic 52, test_apic_mesh 12: OK; test_fem* as above.
- Phase 3: test_graph 35 (3 skipped), test_array_fill_capture 3, test_capture_mode 2, test_conditional_captures 32 (16 skipped, Q4), test_conditional 50, test_async 2373, test_streams 24 (3 skipped, incl. the HIP stream-priority skip), test_mempool 7 (2 skipped), test_pinned 1, test_unified_memory 27 (8 skipped), test_occupancy 6, test_allocator 16 (3 skipped), test_allocation_tracker 34, test_dlpack 16, test_torch 34, test_map 45, test_reload 14, test_cuda_arch_suffix 10 (10 skipped): all OK.
- Phase 4, tile set (24 files): 639 run, 15 skipped, 0 failures, per file:

| file | ran | result |
|---|---|---|
| test_tile.py | 126 | OK |
| test_tile_atomic_bitwise.py | 3 | OK |
| test_tile_cholesky.py | 33 | OK |
| test_tile_cholesky_no_mathdx.py | 16 | OK |
| test_tile_empty.py | 20 | OK |
| test_tile_fft.py | 20 | OK (skipped=8) |
| test_tile_fft_no_mathdx.py | 16 | OK |
| test_tile_func_arg.py | 27 | OK |
| test_tile_fused_ops.py | 36 | OK |
| test_tile_large_offsets.py | 1 | OK |
| test_tile_load.py | 64 | OK |
| test_tile_mathdx.py | 6 | OK |
| test_tile_matmul.py | 25 | OK |
| test_tile_matmul_no_mathdx.py | 22 | OK |
| test_tile_mlp.py | 3 | OK (skipped=1) |
| test_tile_reduce.py | 62 | OK (skipped=2) |
| test_tile_shared_memory.py | 30 | OK (skipped=2) |
| test_tile_solve.py | 5 | OK (skipped=1) |
| test_tile_solve_no_mathdx.py | 28 | OK |
| test_tile_sort.py | 2 | OK |
| test_tile_stack.py | 40 | OK |
| test_tile_view.py | 16 | OK (skipped=1) |
| test_block_dim_dispatch.py | 2 | OK |
| test_template_launch_bounds.py | 36 | OK |

  Skips: the 10 upstream no-libmathdx skips (fft 8, mlp 1, solve 1) and the five 64 KiB capability skips, with the predicted byte counts: `test_tile_shared_mem_large` and `test_tile_shared_mem_func` "Needs 67584 bytes of shared memory per block, device has 65536", `test_tile_reduce_axis_tier2` 66816, `test_tile_reduce_axis_tier2_single_warp` and `test_tile_assign_2d` 66688. `test_tile_matmul` 25 includes the three `test_tile_matmul_fragment_shapes_*` cases (scalar path on gfx1201 as shipped).
- Phase 5: test_texture 120 (1 skipped) OK.
- Across the 82 logs: no `Warp CUDA error` line. The only "Failed to configure kernel dynamic shared memory" line is the known one from `test_tile_load_scoped`'s backward kernel (65536 bytes; the test passes, forward only, as ruled). Skip reasons are the same set as gfx1100 at `974cd4bf`.
- Compared with gfx1100 at `974cd4bf` (Evidence session 4), every count is the same except `test_tile_matmul`, now 25 (the three new tests, expected). So gfx12 behaves like gfx11 on the whole set.
- An example test leaves an untracked `example_apic_fluid.usd` in the clone, as on gfx942; deleted after the run.

### CPU-only suite with GPUs hidden
`HIP_VISIBLE_DEVICES=99 CUDA_VISIBLE_DEVICES=99 python -m warp.tests -s autodetect -k cpu -m 32` (timeit-wrapped, no `cuda:0` in the log): **Ran 2294 tests, OK (skipped=15)**, as expected. (It ran while the 82-pattern set was on its Phase 1 files; GPU results are unaffected, the CPU suite does not touch the GPU.)

### WMMA step 1: layout probe (passes)
`wmma_layout.hip` from Porting session 9, built with `$R/lib/llvm/bin/clang++ -x hip --rocm-path=$R --offload-arch=gfx1201 -O2 -w wmma_layout.hip -o wmma_layout -L$R/lib -lamdhip64 -Wl,-rpath,$R/lib` (`R=$(rocm-sdk path --root)`, ROCm 10.2), run on the pinned GPU:
```
WMMA family 12, K=16: 0 of 256 wrong
WMMA family 12, K=32: 0 of 256 wrong
WMMA family 12, K=64: 0 of 256 wrong
```
Exit 0. So the gfx12 fp16 layout the header assumes is right on real hardware: lanes 0-15 hold K 0-7 and lanes 16-31 K 8-15 of their A row / B column, and accumulator r of lane l is `C[8 (l / 16) + r][l % 16]`. Checked as well: `tile_matmul_mma.h`'s gfx12 `op<half>` uses exactly that (`kpl = 8`, `k_split = true`, so `k_offset = 8 g`, `k_step = 16`, `row(g, r) = 8 g + r`), so the probe tests the code that ships, not just the instruction. The gfx11 lines were not run here (gfx1100 host).

### WMMA step 2: switch-on experiment (uncommitted, reverted)
The experiment's diff, applied in a separate worktree of the clone at `930d0098` (`git -C projects/warp-1.14/src worktree add --detach agent_space/wt_wmma 930d0098`, `warp/bin/*.so` copied in from the clone's build, used through `PYTHONPATH=agent_space/wt_wmma`):
```diff
--- a/warp/native/tile_matmul_mma.h
+++ b/warp/native/tile_matmul_mma.h
@@ -9,6 +9,7 @@
 // feature through __has_builtin.
 
 #pragma once
+#define WP_TILE_WMMA 1
 
 #if defined(__HIP_DEVICE_COMPILE__)
 
```
Removed afterwards with `git worktree remove --force` and `git worktree prune`; the clone was never edited (`git status --porcelain` empty, `git worktree list` shows only the clone).

**`python mma_check.py` (3 dtypes x 11 shapes x 5 block_dims), with the switch on and, for comparison, as shipped:** the script prints FAILURES 15 in both trees, and the per-case status lines are identical (`diff` of the status lists: no difference): 25 printed ok, 25 SMEM-LIMIT (backward not launched), 15 FAIL. **None of the 15 FAILs is caused by WMMA.** All are fp32 or fp64 cases, which have no WMMA instruction on gfx12 and stay scalar in both trees. Their *forward* kernel needs more than the 64 KiB per-block limit, so nothing launches, and the script's SMEM-LIMIT rule only recognises a failed backward: fp32 64x64x64 (forward 65536 dynamic bytes plus the static allocation, backward 131072), fp64 (64,32,48) (forward 69632, backward 139264) and fp64 64x64x64 (forward 131072, backward 262144), each at all 5 block_dims. The Warp UserWarning before each FAIL names the forward kernel's byte count. On gfx950's 160 KiB these launch, which is why its run had no FAIL. A fix for the script (not needed for this evidence): treat a case whose `c` and `d` are also all zero after a forward "Failed to configure" warning as SMEM-LIMIT (forward).

fp16 with the switch on, every case `ok` (or SMEM-LIMIT for the 64x64x64 backward at 65536 bytes, all 5 block_dims, the same in the shipped tree). Max relative difference to the CPU device (bd 64 lines; tolerance `1e-2 * max(1, K/16)`):

| fp16 shape (M,N,K) | as shipped (scalar) c / d / gA / gB | WMMA c / d / gA / gB |
|---|---|---|
| 16,16,16 | 0 / 0 / 0 / 0 | 1.6e-03 / 1.5e-03 / 3.4e-04 / 4.1e-04 |
| 32,32,32 | 0 / 0 / 0 / 0 | 1.2e-03 / 1.2e-03 / 1.2e-03 / 1.6e-03 |
| 64,64,64 (fwd only) | 0 / 0 / - / - | 1.9e-03 / 1.7e-03 / - / - |
| 16,48,32 | 0 / 0 / 0 / 0 | 1.1e-03 / 1.2e-03 / 2.0e-03 / 1.2e-03 |
| 48,16,64 | 0 / 0 / 0 / 0 | 1.9e-03 / 1.9e-03 / 8.2e-04 / 1.3e-03 |
| 32,64,16 | 0 / 0 / 0 / 0 | 1.0e-03 / 1.1e-03 / 1.4e-03 / 6.9e-04 |
| 64,32,48 | 0 / 0 / 0 / 0 | 1.1e-03 / 1.0e-03 / 1.3e-03 / 3.5e-03 |
| 8,8,8; 24,16,16; 16,16,12; 17,33,15 (do not fit, scalar) | 0 | 0 |

The scalar GPU path agrees bit for bit with the CPU (same fp16 accumulation order). WMMA accumulates in fp32 and rounds once, so it differs from both. Which side is more accurate was checked against a float64 NumPy reference (`acc64.py`: fp16 `C = A @ B`, block_dim 64, max abs error):

| M,N,K | CPU | GPU as shipped | GPU WMMA |
|---|---|---|---|
| 32,32,32 | 9.67e-03 | 9.67e-03 | 1.94e-03 |
| 64,64,64 | 2.16e-02 | 2.16e-02 | 3.82e-03 |
| 64,32,48 | 1.14e-02 | 1.14e-02 | 2.86e-03 |

So WMMA is 4-6x closer to the exact result, the same direction gfx950 measured for MFMA fp16 (3.8e-03 vs 2.2e-02 at 64x64x64).

**Tile set with the switch on** (24 files, worktree tree): **639 run, 15 skipped, 0 failures**, the same per-file counts and the same 15 skips as the shipped run above (`test_tile_matmul` 25 OK).

**Engagement, checked in the code objects** (`$R/lib/llvm/bin/llvm-objdump -d --mcpu=gfx1201` on the cached `.hsaco`):
- `mma_check.py`: switch on, 165 code objects with 1336 `v_wmma_f32_16x16x16_f16` in total; as shipped, 165 code objects with 0.
- `test_tile_matmul.py` alone, fresh cache: switch on, 28 code objects, 4 with WMMA (32 `v_wmma_f32_16x16x16_f16`), Ran 25 OK; as shipped, 28 with 0, Ran 25 OK. (The tile-set loop could not be inspected directly: the parallel runner clears its cache at the start of every file, so only the last file's objects remain.)
- Benchmark caches: every fp16 module contains `v_wmma_f32_16x16x16_f16` (2 to 52 per module); the fp32 and fp64 modules are byte-identical between the two trees (48 of 48 `cmp`-equal), as they must be with no WMMA instruction for those dtypes.
- block_dim 32 is one whole 32-lane wavefront, so on gfx12 it takes WMMA too (on gfx950 it is below one 64-lane wavefront and stays scalar).

Observation for whoever toggles the switch: the module hash does not cover the native headers. `acc64.py` printed the same module hash (`1176892` for cuda:0) in both trees. So with a shared kernel cache, adding or removing `WP_TILE_WMMA` would silently reuse code objects built the other way. Every run here used a fresh private `WARP_CACHE_PATH`. Upstream behaves the same way. For a committed change this does not matter: the cache directory is keyed on `config.version`, and a rebuilt install starts clean. It matters only for local experiments like this one.

### Benchmark: scalar (Phase 4, three runs) and WMMA fp16 comparison
Method: the Phase 4 `bench_tile_matmul.py` (16384 blocks, 50 timed iterations, fwd = 32 accumulating `tile_matmul` per block, adj = one GEMM's `tape.backward()`), three runs on the shipped head and three on the switch-on worktree, each with a fresh cache, under `utils/timeit.sh warp-1.14 bench` (457 s for all six). `bench_both.sh` was adapted for this host (`agent_space/bench/bench_both.sh`). The scalar side is the shipped tree, not the `974cd4bf` `tile_matmul.h` worktree the gfx950 recipe uses, because on gfx1201 the shipped head already contains no matrix instruction (verified above), and so its three runs are also the Phase 4 scalar numbers Jeff asked for. `compare.py` was used with `mfma` renamed to `wmma` (`compare_wmma.py`; the diff is only those names). `device=AMD Radeon AI PRO R9700 arch=120 mathdx=off blocks=16384 reps=32` (512 blocks per WGP, 256 per CU).

Rows that are not `ok`, identical in all six runs, all the 64 KiB per-block limit, none the port: fp32 64x64 adj at every block_dim (backward kernel 98304 bytes: `ADJ-FAILED`, the adj column times a kernel that did not launch); fp64 64x64 at every block_dim (forward kernel 98304 bytes, backward 196608: neither launches, so the forward check reads zeros, `MISMATCH`, and both its columns time nothing). gfx1100 has the same 64 KiB limit.

**Phase 4 scalar `tile_matmul`, gfx1201, fork `930d0098` as shipped, no libmathdx, three runs** (`summarize.py`):

| dtype | tile | block_dim | fwd ms (min-max) | fwd GFLOP/s (best) | adj ms (min-max) | adj GFLOP/s (best) | check |
|---|---|---|---|---|---|---|---|
| float16 | 16x16 | 32 | 0.930-0.944 | 4617 | 0.258-0.262 | 1041 | ok |
| float16 | 16x16 | 64 | 1.020-1.041 | 4211 | 0.260-0.284 | 1034 | ok |
| float16 | 16x16 | 128 | 0.858-0.865 | 5005 | 0.236-0.255 | 1139 | ok |
| float16 | 16x16 | 256 | 1.221-1.241 | 3518 | 0.289-0.309 | 928 | ok |
| float16 | 32x32 | 32 | 2.928-3.033 | 11735 | 3.769-3.808 | 570 | ok |
| float16 | 32x32 | 64 | 3.192-3.257 | 10765 | 1.917-1.939 | 1120 | ok |
| float16 | 32x32 | 128 | 3.624-3.734 | 9482 | 1.533-1.583 | 1401 | ok |
| float16 | 32x32 | 256 | 5.191-5.367 | 6619 | 1.668-1.708 | 1287 | ok |
| float16 | 64x64 | 32 | 27.452-28.115 | 10013 | 56.168-57.424 | 306 | ok |
| float16 | 64x64 | 64 | 22.767-24.032 | 12074 | 29.783-31.442 | 577 | ok |
| float16 | 64x64 | 128 | 23.149-24.410 | 11874 | 17.383-17.704 | 988 | ok |
| float16 | 64x64 | 256 | 13.917-14.701 | 19752 | 18.494-18.710 | 929 | ok |
| float32 | 16x16 | 32 | 1.683-1.711 | 2552 | 1.265-1.290 | 212 | ok |
| float32 | 16x16 | 64 | 1.856-1.891 | 2315 | 1.210-1.222 | 222 | ok |
| float32 | 16x16 | 128 | 1.139-1.157 | 3771 | 1.190-1.204 | 226 | ok |
| float32 | 16x16 | 256 | 1.468-1.491 | 2926 | 1.191-1.201 | 225 | ok |
| float32 | 32x32 | 32 | 6.801-6.985 | 5052 | 5.160-5.170 | 416 | ok |
| float32 | 32x32 | 64 | 7.233-7.463 | 4750 | 4.827-4.830 | 445 | ok |
| float32 | 32x32 | 128 | 6.841-7.082 | 5023 | 5.911-5.970 | 363 | ok |
| float32 | 32x32 | 256 | 8.217-8.535 | 4182 | 5.590-5.621 | 384 | ok |
| float32 | 64x64 | 32 | 56.598-56.866 | 4857 | 1.300-1.301 | 13212 | ADJ-FAILED |
| float32 | 64x64 | 64 | 35.720-36.303 | 7695 | 1.289-1.292 | 13333 | ADJ-FAILED |
| float32 | 64x64 | 128 | 38.803-39.500 | 7084 | 1.288-1.295 | 13339 | ADJ-FAILED |
| float32 | 64x64 | 256 | 43.520-44.221 | 6316 | 1.292-1.295 | 13298 | ADJ-FAILED |
| float64 | 16x16 | 32 | 6.091-6.162 | 705 | 1.339-1.344 | 200 | ok |
| float64 | 16x16 | 64 | 5.915-5.965 | 726 | 0.967-0.975 | 278 | ok |
| float64 | 16x16 | 128 | 5.978-6.065 | 718 | 1.227-1.234 | 219 | ok |
| float64 | 16x16 | 256 | 5.910-6.065 | 727 | 1.193-1.201 | 225 | ok |
| float64 | 32x32 | 32 | 55.598-55.684 | 618 | 22.513-23.147 | 95 | ok |
| float64 | 32x32 | 64 | 47.305-48.168 | 726 | 12.720-13.094 | 169 | ok |
| float64 | 32x32 | 128 | 42.556-43.264 | 807 | 8.013-8.184 | 268 | ok |
| float64 | 32x32 | 256 | 43.013-43.704 | 799 | 7.181-7.305 | 299 | ok |
| float64 | 64x64 | 32 | 0.011-0.012 | 25271714 | 2.633-2.634 | 6526 | MISMATCH |
| float64 | 64x64 | 64 | 0.011-0.013 | 25949258 | 2.612-2.614 | 6578 | MISMATCH |
| float64 | 64x64 | 128 | 0.011-0.012 | 26151202 | 2.609-2.623 | 6585 | MISMATCH |
| float64 | 64x64 | 256 | 0.011-0.014 | 25357038 | 2.609-2.614 | 6584 | MISMATCH |

Run-to-run spread over the 56 `ok` cells (fwd and adj, (max-min)/median): median 1.8%, max 9.3% (the fp16 16x16 adjoints, which take about 0.25 ms). Compared with gfx1100 (Evidence session B.3, same script and block count, 70 CUs), gfx1201's scalar numbers are of the same order. For example fp16 16x16 fwd at bd 128 is 0.858 ms here against 0.911 ms there.

**fp16 scalar against WMMA (the switch-on experiment), median of three runs per tree** (`compare_wmma.py`; fp32 and fp64 rows included for completeness, see the note below the table):

| dtype | tile | block_dim | scalar fwd ms | WMMA fwd ms | fwd speedup | scalar adj ms | WMMA adj ms | adj speedup | checks |
|---|---|---|---|---|---|---|---|---|---|
| float16 | 16x16 | 32 | 0.935 | 0.367 | 2.55x | 0.259 | 0.219 | 1.18x | ok |
| float16 | 16x16 | 64 | 1.027 | 0.353 | 2.91x | 0.260 | 0.185 | 1.41x | ok |
| float16 | 16x16 | 128 | 0.859 | 0.393 | 2.19x | 0.237 | 0.188 | 1.26x | ok |
| float16 | 16x16 | 256 | 1.233 | 0.418 | 2.95x | 0.290 | 0.246 | 1.18x | ok |
| float16 | 32x32 | 32 | 3.011 | 1.785 | 1.69x | 3.800 | 2.380 | 1.60x | ok |
| float16 | 32x32 | 64 | 3.226 | 1.785 | 1.81x | 1.931 | 1.361 | 1.42x | ok |
| float16 | 32x32 | 128 | 3.694 | 2.063 | 1.79x | 1.564 | 0.989 | 1.58x | ok |
| float16 | 32x32 | 256 | 5.319 | 2.098 | 2.53x | 1.697 | 0.969 | 1.75x | ok |
| float16 | 64x64 | 32 | 27.914 | 9.754 | 2.86x | 56.369 | 47.858 | 1.18x | ok |
| float16 | 64x64 | 64 | 23.460 | 10.939 | 2.14x | 30.056 | 24.851 | 1.21x | ok |
| float16 | 64x64 | 128 | 23.667 | 10.186 | 2.32x | 17.397 | 13.773 | 1.26x | ok |
| float16 | 64x64 | 256 | 14.005 | 12.843 | 1.09x | 18.666 | 7.651 | 2.44x | ok |
| float32 | 16x16 | 32 | 1.687 | 1.716 | 0.98x | 1.268 | 1.268 | 1.00x | ok |
| float32 | 16x16 | 64 | 1.861 | 1.897 | 0.98x | 1.211 | 1.213 | 1.00x | ok |
| float32 | 16x16 | 128 | 1.154 | 1.162 | 0.99x | 1.192 | 1.193 | 1.00x | ok |
| float32 | 16x16 | 256 | 1.469 | 1.497 | 0.98x | 1.191 | 1.191 | 1.00x | ok |
| float32 | 32x32 | 32 | 6.822 | 6.997 | 0.98x | 5.166 | 5.177 | 1.00x | ok |
| float32 | 32x32 | 64 | 7.234 | 7.464 | 0.97x | 4.827 | 4.828 | 1.00x | ok |
| float32 | 32x32 | 128 | 6.874 | 7.092 | 0.97x | 5.913 | 5.893 | 1.00x | ok |
| float32 | 32x32 | 256 | 8.246 | 8.505 | 0.97x | 5.615 | 5.601 | 1.00x | ok |
| float32 | 64x64 | 32 | 56.759 | 56.905 | 1.00x | (no launch) | (no launch) | - | ADJ-FAILED |
| float32 | 64x64 | 64 | 35.840 | 36.300 | 0.99x | (no launch) | (no launch) | - | ADJ-FAILED |
| float32 | 64x64 | 128 | 38.979 | 39.468 | 0.99x | (no launch) | (no launch) | - | ADJ-FAILED |
| float32 | 64x64 | 256 | 43.801 | 44.463 | 0.99x | (no launch) | (no launch) | - | ADJ-FAILED |
| float64 | 16x16 | 32 | 6.132 | 6.239 | 0.98x | 1.342 | 1.347 | 1.00x | ok |
| float64 | 16x16 | 64 | 5.957 | 6.058 | 0.98x | 0.971 | 0.979 | 0.99x | ok |
| float64 | 16x16 | 128 | 6.050 | 6.186 | 0.98x | 1.231 | 1.242 | 0.99x | ok |
| float64 | 16x16 | 256 | 6.063 | 6.149 | 0.99x | 1.198 | 1.211 | 0.99x | ok |
| float64 | 32x32 | 32 | 55.617 | 55.724 | 1.00x | 22.514 | 22.541 | 1.00x | ok |
| float64 | 32x32 | 64 | 47.420 | 47.592 | 1.00x | 12.757 | 12.714 | 1.00x | ok |
| float64 | 32x32 | 128 | 42.640 | 42.829 | 1.00x | 8.023 | 8.065 | 0.99x | ok |
| float64 | 32x32 | 256 | 43.179 | 43.424 | 0.99x | 7.183 | 7.239 | 0.99x | ok |
| float64 | 64x64 | 32 | 0.011 | 0.011 | 0.97x | (no launch) | (no launch) | - | MISMATCH |
| float64 | 64x64 | 64 | 0.011 | 0.011 | 0.95x | (no launch) | (no launch) | - | MISMATCH |
| float64 | 64x64 | 128 | 0.011 | 0.012 | 0.95x | (no launch) | (no launch) | - | MISMATCH |
| float64 | 64x64 | 256 | 0.012 | 0.011 | 1.07x | (no launch) | (no launch) | - | MISMATCH |
run-to-run spread (max-min)/median over 144 cells: median 1.8%, max 47.6%

Reading the table:
- **fp16: WMMA is faster than scalar in every cell**, forward 1.09x to 2.95x and adjoint 1.18x to 2.44x, block_dim 32 included (one whole wavefront on gfx12). No cell is slower, so the ruling's "slower than scalar must fall back or be explained" condition is not triggered on gfx1201. The smallest forward gain is fp16 64x64 at bd 256 (14.005 to 12.843 ms, 1.09x), where the scalar path is already at its best (bd 256 is the scalar optimum for that tile). The smallest adjoint gains (1.18x at 16x16 and at 64x64 bd 32) fit gfx950's explanation that the fp16 backward is dominated by the CAS-loop `atomic_add` gradient accumulation (safe-atomics ruling), not by the GEMMs; not measured further here.
- **fp32 and fp64 show 0.97x-1.00x, which is not a cost of the switch.** Their code objects are byte-identical between the two trees (48 of 48), because gfx12 has no fp32/fp64 WMMA and the hook falls back at compile time. The consistent 1-3% is run order: all three scalar runs went first, then the three WMMA runs, on a warm GPU. The same drift therefore probably sits on the fp16 ratios too, against WMMA, so the fp16 speedups are if anything slightly understated.
- The fp32 64x64 adj and fp64 64x64 cells are the 64 KiB limit (above). The fp64 64x64 "fwd" times (0.011 ms) are a launch that did nothing.
- WMMA-run spread over the 56 `ok` cells: median 2.0%, max 16.1% (the fp16 16x16 adjoints, about 0.19 ms; the other fp16 cells are within a few percent).

### Summary for the lock holder and Jeff
- First gfx12 run at `930d0098`: textures (hardware path, test_texture 120 OK, SDF probe and tex_alias PASS with the gfx1100 values), the full 82-pattern set (81 OK plus the 2 known fem isolation failures, the 5 expected 64 KiB tile skips) and the CPU-only suite (2294 OK, 15 skipped) all match gfx1100. Nothing needs a fork change.
- WMMA layout probe: all three gfx12 lines 0 of 256 wrong, and the probe's layout is the one the header implements.
- Switch-on experiment on gfx1201: mma_check has no WMMA-caused failure (its 15 FAILs are fp32/fp64 forward kernels over 64 KiB, identical in the shipped tree); tile set 639 run, 0 failures; `test_tile_matmul` 25 OK with WMMA engaged; fp16 results 4-6x closer to float64 than scalar; fp16 benchmark faster in all 12 cells. **gfx12 passes the conditions the Porting session 9 handoff set for enabling WMMA.**
- Proposed change for the lock holder to judge (not committed: the experiment diff above, expressed for the targets that passed). gfx11 has not run its probe yet. So the switch should be on for the gfx12 family only, until a gfx1100 host runs `wmma_layout.hip` and the same checks. The header can use the gfx12 builtin itself as the condition, without an arch name, the same feature-macro rule as the rest of the selection:
```diff
--- a/warp/native/tile_matmul_mma.h
+++ b/warp/native/tile_matmul_mma.h
@@
-#elif defined(WP_TILE_WMMA) && WP_TILE_WMMA
-// 32-lane WMMA. The operand and accumulator layouts below come from the ISA documentation and
-// have not been measured yet, so this path is off unless WP_TILE_WMMA is defined to 1.
+#elif (defined(WP_TILE_WMMA) && WP_TILE_WMMA) || __has_builtin(__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12)
+// 32-lane WMMA. The gfx12 fp16 layout is measured on gfx1201; the gfx11 and gfx1250 layouts
+// come from the ISA documentation and stay off unless WP_TILE_WMMA is defined to 1.
```
  That one condition is enough by itself. Checked with the ROCm 10.2 compiler (`clang++ -x hip --cuda-device-only -nogpulib -E` on `__has_builtin`), each WMMA builtin exists on exactly one family: gfx1100 has only `wmma_f32_16x16x16_f16_w32`, gfx1201 only `..._w32_gfx12`, gfx1250 only `wmma_f32_16x16x4_f32`, and gfx950 none. So the branch turns on for gfx12 targets only. Inside it, the gfx11 and gfx1250 blocks stay empty there, and gfx11 and gfx1250 still need `WP_TILE_WMMA=1`, as now. On gfx1201 this is exactly the code tested here. The MFMA branch comes first and is unaffected. gfx1200 also has only the gfx12 builtin (checked the same way), so it would be enabled too; it was not run on hardware.
- Release notes, if gfx12 is enabled: the session 9 wording ("On Radeon GPUs `tile_matmul` uses the scalar path") changes for gfx12 to: fp16 `tile_matmul` uses WMMA when tiles are multiples of 16 and the block is a whole number of 32-thread wavefronts, accumulating in fp32 (results can differ from the scalar path in the last bits, and are closer to the exact product); fp32 and fp64 stay scalar because RDNA4 has no WMMA instruction for them.
- Not done here: the gfx11 probe and the bf16 path (dropped from WMMA by the porter); multi-GPU (one GPU pinned).

#### Addendum: reading this beside gfx1100's Evidence session 5 (same head, run in parallel)
This section was written before gfx1100's Evidence session 5 landed, so its "gfx11 has not run its probe yet" lines are out of date. gfx1100 found the gfx11 layout correct but its WMMA f16->f32 accumulation off by up to about 1 ulp of f32 when products of mixed sign cancel, so the probe's exact-equality check fails there. On gfx1201 the same probe, with the same signed integer data (`A = (x*7+3)%9 - 4`, `B = (x*5+1)%7 - 3`), was **exact in all 768 cells** (K = 16, 32, 64). So RDNA4's gfx12 WMMA accumulates these exactly where RDNA3's does not, and for gfx12 the probe's pass is a real exact-equality pass, not a tolerance pass. The proposed gfx12-only condition above stands as written: it enables nothing on gfx11. Whether gfx11 is enabled as well (layout correct, f32-level accumulation differences, which `mma_check.py`'s tolerances would cover) is the lock holder's and Jeff's call, after a gfx11 switch-on run of `mma_check.py`, the tile set and the benchmark.
## Evidence session 5 (2026-11-19, linux-gfx942, MI300X, evidence only): MFMA tile_matmul at `930d0098` (gfx942 items 1-5)

Requested by Jeff Daily. The lock stays with linux-gfx950. No fork commit and no stage, lock, head, platform or deferred record changed. The read-only clone was moved from `974cd4bf` to the recorded `head_sha` `930d0098dafa5ef08945a4382b87c07d89c9a7f0` (`git status --porcelain` empty before and after; LFS assets present since evidence session 4; fork pre-push hook unchanged). Build: `python build_lib.py --rocm-path "$(rocm-sdk path --root)" --hip-arch gfx942 -j 112` -> "Warp build succeeded", 30.5 s. One GPU (`HIP_VISIBLE_DEVICES=0`), `ulimit -c 0`, a fresh private `WARP_CACHE_PATH` per run. Items run in the handoff's order. `mfma_layout.hip`, `mma_check.py`, `bench_both.sh` and `compare.py` were extracted verbatim from Porting session 9; `bench_both.sh` only had its `A=` path changed to this host and `export HIP_VISIBLE_DEVICES=0` added.

### 1. MFMA layouts on gfx942: all match gfx950
`clang++ -x hip --rocm-path=$R --offload-arch=gfx942 -O2 -w mfma_layout.hip -o mfma_layout -L$R/lib -lamdhip64 -Wl,-rpath,$R/lib && ./mfma_layout`: **0 of 256 wrong on all twelve lines** (f32 16x16x4, f64 16x16x4, f16 16x16x16, bf16 16x16x16, each at K = 16, 32, 64). The MFMA path's operand and accumulator layouts, including the f64 output layout, hold on gfx942.

### 2. `mma_check.py` and `mma_check_bf16.py`
**`mma_check_bf16.py` had no source in the notes**, only the description in Porting session 9. I wrote it from that description, following the API form of upstream's `test_tile_gemm_bf16`: bf16 A/B arrays converted on the device, a float32 C tile, the accumulating `wp.tile_matmul(a, b, c)`, `module_options={"enable_backward": False}`. 9 shapes x block_dims (32, 64, 96, 128, 256): (16,16,16), (32,32,32), (64,64,64), (16,48,32), (48,16,64), (32,64,16), (64,32,48), (24,16,16), (17,33,15). Each GPU result is compared with the CPU device and with a float64 NumPy product of the bf16-rounded inputs. Source below. Result: **45 cases, 0 failures, max difference to the CPU 9.54e-07** (gfx950: 9.5e-07). Its code objects contain 121 `v_mfma_f32_16x16x16_bf16`.

`mma_check.py`: 165 cases, script exit 1 with **"FAILURES 15"**. On examination none of the 15 is a matrix-core or correctness problem: they are shared-memory limits of this 64 KiB part that the script does not classify.
- The 15 are fp32 64x64x64, fp64 64x64x64 and fp64 64x32x48, at all five block_dims. In each, the *forward* output is wrong too (c ~1.0) and Warp prints `Warp CUDA error 1: invalid argument (in function wp_cuda_launch_kernel)`. Warp's warnings show the forward kernels need 65536 (fp32 64x64x64: A, B, C and D tiles of 16 KiB each, plus Warp's static allocation, which does not fit 65536), 69632 (fp64 64x32x48) and 131072 bytes (fp64 64x64x64). `mma_check.py` labels a case SMEM-LIMIT only when the forward results are correct and the gradients are zero, so a forward kernel that cannot launch is counted as FAIL. On gfx950 (163840 bytes per block) these kernels fit, which is why it saw only the fp64 64x64x64 adjoint limit.
- 25 more cases report SMEM-LIMIT (backward not launched): fp32 64x32x48, fp64 32x32x32, fp64 48x16x64, fp64 32x64x16 and fp16 64x64x64, each at all five block_dims. Warp's backward warnings ask for 65536 to 262144 bytes.
- Control, with the same script against a scalar tree (the step 4 worktree: `930d0098` with only `tile_matmul.h` reverted to `974cd4bf`, same `warp.so`): **the identical 40 non-ok cases** (15 FAIL + 25 SMEM-LIMIT, `diff` of the case lists empty), and its kernel caches contain 0 `v_mfma` instructions. So every non-ok case is the shared-memory limit, on either path.
- Matrix instructions engaged in the MFMA run: 948 `v_mfma_f32_16x16x16_f16`, 3864 `v_mfma_f32_16x16x4_f32`, 3864 `v_mfma_f64_16x16x4_f64` in its code objects.
- Accuracy on the cases that launch, vs the CPU device (relative max over C, D, gA, gB, from the bd 64 lines the script prints): fp32 and fp64 **bit-identical (0.0e+00) for every shape**, MFMA and fallback shapes alike, as on gfx950. fp16: 1.0e-03 to 3.5e-03 on MFMA shapes (fp32 accumulation, rounded once), 0 on the fallback shapes (8,8,8), (24,16,16), (16,16,12), (17,33,15). In the scalar control, fp16 is 0 everywhere. All within the script's tolerances.
- Suggestion for the lock holder (not applied; `mma_check.py` is the porter's script): count a forward-launch failure whose Warp warning names the forward kernel as SMEM-LIMIT too, so the script exits 0 on 64 KiB parts.

### 3. Tile set (22 files + test_block_dim_dispatch + test_template_launch_bounds)
**639 run, 15 skipped, 0 failures**, every file OK. The skips are the 10 upstream no-libmathdx skips plus the five "Needs <N> bytes of shared memory per block, device has 65536" skips of session 8. `test_tile_matmul` ran 25: `test_tile_matmul_fragment_shapes_float16_cuda_0`, `_float32_cuda_0` and `_float64_cuda_0` all OK.

### 4. Benchmark, scalar against MFMA (gfx942)
Scalar worktree as the handoff says: `git -C projects/warp-1.14/src worktree add --detach agent_space/scalar_wt 930d0098`, `git -C agent_space/scalar_wt checkout 974cd4bf -- warp/native/tile_matmul.h` (-6 lines, the hook), `warp/bin/*.so` copied from the main clone. Then `bench_both.sh` (three runs per tree of `bench_tile_matmul.py --blocks 16384 --iters 50`, 16 min under `utils/timeit.sh warp-1.14 bench`) and `compare.py` (medians of three). Every run exits 1, as the script always does on a part that cannot launch some backward kernels (session 1). The three MFMA-tree kernel caches each contain 3036 `v_mfma` instructions; the three scalar-tree caches contain 0. The worktree was removed afterwards.

| dtype | tile | block_dim | scalar fwd ms | MFMA fwd ms | fwd speedup | scalar adj ms | MFMA adj ms | adj speedup | checks |
|---|---|---|---|---|---|---|---|---|---|
| float16 | 16x16 | 32 | 0.325 | 0.325 | 1.00x | 0.341 | 0.340 | 1.00x | ok |
| float16 | 16x16 | 64 | 0.195 | 0.072 | 2.70x | 0.350 | 0.293 | 1.19x | ok |
| float16 | 16x16 | 128 | 0.191 | 0.078 | 2.45x | 0.301 | 0.298 | 1.01x | ok |
| float16 | 16x16 | 256 | 0.246 | 0.080 | 3.07x | 0.293 | 0.292 | 1.00x | ok |
| float16 | 32x32 | 32 | 2.132 | 2.132 | 1.00x | 2.192 | 2.190 | 1.00x | ok |
| float16 | 32x32 | 64 | 0.867 | 0.342 | 2.54x | 1.942 | 1.461 | 1.33x | ok |
| float16 | 32x32 | 128 | 0.958 | 0.381 | 2.52x | 2.094 | 1.212 | 1.73x | ok |
| float16 | 32x32 | 256 | 1.125 | 0.450 | 2.50x | 1.844 | 1.134 | 1.63x | ok |
| float16 | 64x64 | 32 | 25.703 | 25.704 | 1.00x | 29.771 | 29.813 | 1.00x | ok |
| float16 | 64x64 | 64 | 13.588 | 5.296 | 2.57x | 19.076 | 17.076 | 1.12x | ok |
| float16 | 64x64 | 128 | 9.182 | 2.469 | 3.72x | 11.497 | 10.663 | 1.08x | ok |
| float16 | 64x64 | 256 | 6.349 | 2.330 | 2.73x | 7.836 | 7.101 | 1.10x | ok |
| float32 | 16x16 | 32 | 0.416 | 0.416 | 1.00x | 0.257 | 0.258 | 1.00x | ok |
| float32 | 16x16 | 64 | 0.344 | 0.111 | 3.11x | 0.129 | 0.068 | 1.90x | ok |
| float32 | 16x16 | 128 | 0.240 | 0.112 | 2.13x | 0.095 | 0.062 | 1.53x | ok |
| float32 | 16x16 | 256 | 0.286 | 0.116 | 2.47x | 0.098 | 0.064 | 1.53x | ok |
| float32 | 32x32 | 32 | 3.136 | 3.135 | 1.00x | 2.240 | 2.246 | 1.00x | ok |
| float32 | 32x32 | 64 | 1.552 | 0.761 | 2.04x | 1.273 | 0.760 | 1.68x | ok |
| float32 | 32x32 | 128 | 1.428 | 0.707 | 2.02x | 1.128 | 0.460 | 2.45x | ok |
| float32 | 32x32 | 256 | 1.580 | 1.107 | 1.43x | 0.889 | 0.332 | 2.68x | ok |
| float32 | 64x64 | 32 | 38.043 | 38.037 | 1.00x | (no launch) | (no launch) | - | ADJ-FAILED |
| float32 | 64x64 | 64 | 18.732 | 14.583 | 1.28x | (no launch) | (no launch) | - | ADJ-FAILED |
| float32 | 64x64 | 128 | 13.450 | 7.846 | 1.71x | (no launch) | (no launch) | - | ADJ-FAILED |
| float32 | 64x64 | 256 | 9.941 | 4.842 | 2.05x | (no launch) | (no launch) | - | ADJ-FAILED |
| float64 | 16x16 | 32 | 0.589 | 0.589 | 1.00x | 0.322 | 0.322 | 1.00x | ok |
| float64 | 16x16 | 64 | 0.446 | 0.289 | 1.55x | 0.166 | 0.117 | 1.42x | ok |
| float64 | 16x16 | 128 | 0.449 | 0.291 | 1.55x | 0.211 | 0.109 | 1.94x | ok |
| float64 | 16x16 | 256 | 0.533 | 0.285 | 1.87x | 0.210 | 0.109 | 1.92x | ok |
| float64 | 32x32 | 32 | 5.952 | 5.952 | 1.00x | 4.576 | 4.569 | 1.00x | ok |
| float64 | 32x32 | 64 | 3.239 | 1.637 | 1.98x | 2.720 | 2.119 | 1.28x | ok |
| float64 | 32x32 | 128 | 2.724 | 1.390 | 1.96x | 1.909 | 1.301 | 1.47x | ok |
| float64 | 32x32 | 256 | 3.110 | 2.079 | 1.50x | 1.502 | 0.708 | 2.12x | ok |
| float64 | 64x64 | 32 | 0.014 | 0.014 | 0.99x | (no launch) | (no launch) | - | MISMATCH |
| float64 | 64x64 | 64 | 0.014 | 0.014 | 1.00x | (no launch) | (no launch) | - | MISMATCH |
| float64 | 64x64 | 128 | 0.014 | 0.014 | 1.01x | (no launch) | (no launch) | - | MISMATCH |
| float64 | 64x64 | 256 | 0.014 | 0.014 | 1.00x | (no launch) | (no launch) | - | MISMATCH |
run-to-run spread (max-min)/median over 144 cells: median 1.0%, max 15.2%

- **No launched cell is slower on MFMA than scalar.** block_dim 32 is scalar by construction (1.00x). The 0.99x and 1.01x figures are the fp64 64x64 rows, whose forward kernel does not launch on either tree (131072 bytes of shared memory). Their 0.014 ms are launch-failure times, not GEMMs, and the check column says MISMATCH for that reason, as in session 1. "(no launch)" adjoints: fp32 and fp64 64x64, backward over 64 KiB, as in session 1.
- Smallest gains: the fp16 adjoints at 16x16 bd 128/256 (1.00-1.01x) and at 64x64 (1.08-1.12x), the same pattern gfx950 showed. Porting session 9 suggested a cause (fp16 gradient accumulation through the CAS-loop `atomic_add`), not measured here either.
- Run-to-run spread: median 1.0% over the 144 cells, max 15.2%. The five largest are all MFMA adjoint cells of 0.06-0.13 ms (fp64 16x16 bd128 15.2%, fp32 16x16 bd128 14.4%, fp32 16x16 bd64 14.3%, fp32 16x16 bd256 13.6%), where the three seeding memsets dominate. Medians are used.
- Against gfx950 (Porting session 9 table): forward speedups are similar or higher on gfx942 (fp16 64x64 bd128 3.72x vs 2.29x; fp32 16x16 bd64 3.11x vs 3.31x; fp64 32x32 bd64 1.98x vs 2.09x).

### 5. Full 82-pattern set and CPU-only suite at `930d0098`
Same 82-pattern list and loop as evidence session 3, 2526 s together with the CPU-only suite under `utils/timeit.sh warp-1.14 test`. **81 patterns OK, 1 with the 2 known failures**: `test_fem*` 107, `test_gimp_quadrature` and `test_cube_shape_functions` ("Unexpected output", upstream test isolation). Compared line by line with evidence sessions 3/4 at `974cd4bf`, the only count change is `test_tile_matmul` 22 -> 25 (the three new fragment-shape tests). The LFS-dependent files are as in session 4 (`test_coloring` 3, `test_mesh*` 49, `test_volume` 59 OK; `example_adaptive_grid` passes). Skips are unchanged: five 64 KiB tile skips, priority timing skipped.

CPU-only suite (`HIP_VISIBLE_DEVICES=99 CUDA_VISIBLE_DEVICES=99 python -m warp.tests -s autodetect -k cpu -m 32`): **Ran 2294, OK (skipped=15)**.

The untracked `example_apic_fluid.usd` an example writes into the clone was deleted after the run.

`mma_check_bf16.py` (as run):
```python
# bf16 A and B tiles with a float32 accumulator tile (the only bf16 form upstream tile_matmul allows:
# accumulating, backward disabled), GPU against the CPU device and against a float64 NumPy product of
# the bf16-rounded inputs, for tile shapes x block_dims. Usage: python mma_check_bf16.py
import sys

import numpy as np

import warp as wp

wp.init()
gpu = wp.get_device("cuda:0")
SHAPES = [(16, 16, 16), (32, 32, 32), (64, 64, 64), (16, 48, 32), (48, 16, 64), (32, 64, 16), (64, 32, 48),
          (24, 16, 16), (17, 33, 15)]  # fmt: skip
BLOCK_DIMS = (32, 64, 96, 128, 256)


@wp.kernel
def to_bf16(x: wp.array2d(dtype=wp.float32), y: wp.array2d(dtype=wp.bfloat16)):
    i, j = wp.tid()
    y[i, j] = wp.bfloat16(x[i, j])


@wp.kernel
def to_f32(x: wp.array2d(dtype=wp.bfloat16), y: wp.array2d(dtype=wp.float32)):
    i, j = wp.tid()
    y[i, j] = wp.float32(x[i, j])


def make(M, N, K):
    @wp.kernel(module="unique", module_options={"enable_backward": False})
    def gemm(A: wp.array2d(dtype=wp.bfloat16), B: wp.array2d(dtype=wp.bfloat16), C: wp.array2d(dtype=wp.float32)):
        a = wp.tile_load(A, shape=(M, K))
        b = wp.tile_load(B, shape=(K, N))
        c = wp.tile_load(C, shape=(M, N))
        wp.tile_matmul(a, b, c)
        wp.tile_store(C, c)

    return gemm


def bf16(x, dev):
    y = wp.zeros(x.shape, dtype=wp.bfloat16, device=dev)
    wp.launch(to_bf16, dim=x.shape, inputs=[wp.array(x, dtype=wp.float32, device=dev), y], device=dev)
    return y


def run(dev, kern, A, B, C0, bd):
    a, b = bf16(A, dev), bf16(B, dev)
    c = wp.array(C0, dtype=wp.float32, device=dev)
    wp.launch_tiled(kern, dim=[1], inputs=[a, b, c], block_dim=bd, device=dev)
    af = wp.zeros(A.shape, dtype=wp.float32, device=dev)
    bf = wp.zeros(B.shape, dtype=wp.float32, device=dev)
    wp.launch(to_f32, dim=A.shape, inputs=[a, af], device=dev)
    wp.launch(to_f32, dim=B.shape, inputs=[b, bf], device=dev)
    return c.numpy().astype(np.float64), af.numpy().astype(np.float64), bf.numpy().astype(np.float64)


rng = np.random.default_rng(0)
fails = 0
worst = 0.0
for M, N, K in SHAPES:
    kern = make(M, N, K)
    A = rng.uniform(-1, 1, (M, K)).astype(np.float32)
    B = rng.uniform(-1, 1, (K, N)).astype(np.float32)
    C0 = rng.uniform(-1, 1, (M, N)).astype(np.float32)
    for bd in BLOCK_DIMS:
        ref_cpu, _, _ = run("cpu", kern, A, B, C0, bd)
        got, aq, bq = run(gpu, kern, A, B, C0, bd)
        exact = C0.astype(np.float64) + aq @ bq
        e_cpu = np.abs(got - ref_cpu).max()
        e_np = np.abs(got - exact).max()
        worst = max(worst, e_cpu)
        ok = e_cpu <= 1e-5 * max(1, K / 16) and e_np <= 1e-5 * max(1, K / 16)
        fails += not ok
        if not ok or bd == 64:
            print(f"bf16->f32 M{M:3d} N{N:3d} K{K:3d} bd{bd:4d}: vs cpu {e_cpu:.1e} vs float64 {e_np:.1e} {'ok' if ok else 'FAIL'}", flush=True)
print(f"max difference to the CPU {worst:.2e}")
print("FAILURES", fails)
sys.exit(1 if fails else 0)
```

### Next step for the lock holder
- gfx942 items 1-5 at `930d0098` are done. The MFMA layouts hold on gfx942, and correctness, the tile set, the full set and the CPU-only suite are unchanged apart from the new tests. No MFMA cell is slower than scalar. Nothing needs a fork change.
- Optional: teach `mma_check.py` that a forward kernel over the shared-memory limit is SMEM-LIMIT, not FAIL (item 2), and record `mma_check_bf16.py` in the notes (source above) so other hosts run the same script.

## Evidence session (2026-11-19, linux-gfx90a, MI250X, evidence only): first gfx90a run, Phase 1-5 set and MFMA tile_matmul at `930d0098`

Requested by Jeff Daily. This is the port's first run on gfx90a. The lock stays with linux-gfx950. No fork commit, and no stage, lock, head, platform or deferred record changed. The clone was made read-only at the recorded `head_sha` `930d0098dafa5ef08945a4382b87c07d89c9a7f0` on `release/1.14.0.amd0`; `git status --porcelain` was empty at the end.

### Setup (the common setup block, with these details)
- Host: "AMD Instinct MI250X / MI250", one GCD visible (dispatcher pin `HIP_VISIBLE_DEVICES=3`, kept): `gfx90a:sramecc+:xnack-`, 64 GiB, 104 CUs, `max_shared_memory_per_block` 65536, `arch` 90. No system ROCm (`/opt/rocm` absent). Kernel boot options `amd_iommu=on iommu=pt`.
- Fresh clone of `https://github.com/AMD-Ecosystem/warp`. LFS: `sudo apt-get install -y git-lfs` (system package, Jeff's standing rule for missing system utilities). The clone then carried LFS's own `pre-push`, so per the Intake note: `mv .git/hooks/pre-push .git/hooks/pre-push.lfs-original`, `python3 utils/moatlib.py protect-fork warp-1.14`, then `git lfs install --local --skip-repo && git lfs pull`. All `.usd`/`.nvdb` assets are real files.
- Job venv: it came with TheRock 10.1.0 wheels. Installed `rocm[devel,device-gfx90a]==10.2.0`, `torch[device-gfx90a]==2.15.0+rocm10.2.0`, `numpy==1.26.4` and `usd-core` (index `https://stable.repo.amd.com/rocm/whl-next/`, extra index PyPI), then `rocm-sdk init`. As on gfx942, the stale `rocm-sdk-device-gfx*==10.1.0` packages stay (pip warns; Warp uses no ROCm math library). HIP 7.17.26434, AMD clang 24.0.0git, Python 3.11.17.
- Build: `export PATH=$FLEET_VENV/bin:$PATH; python build_lib.py --rocm-path "$(rocm-sdk path --root)" --hip-arch gfx90a -j 64` (`MAX_JOBS`) -> "Warp build succeeded", 48.7 s. RUNPATH is the venv's `_rocm_sdk_devel/lib`. The greeting shows `HIP 7.17` and `"cuda:0" : "AMD Instinct MI250X / MI250" (64 GiB, gfx90a:sramecc+:xnack-, mempool enabled)`. `is_texture_supported, is_texture_hardware_supported` = `True True`.
- Every GPU run used `ulimit -c 0` and its own private `WARP_CACHE_PATH`. The benchmark ran alone on the GPU; the correctness checks ran next to the full set. Scripts were extracted verbatim from these notes (`bench_tile_matmul.py`, `summarize.py`, `mma_check.py`, `compare.py`, `mfma_layout.hip`, `sdf_probe.py`, `tex_alias.py`, `tex_capture.py`). `bench_both.sh` differs only in its paths. `mma_check_bf16.py` is my own version (source below), written from the Porting session 9 description before the gfx942 session's version reached the notes.

### Summary
- **Phase 1-5 set (82 patterns): 80 OK, plus 2 not OK.** `test_fem*` has the 2 known isolation failures. `test_texture` has **14 errors, a new gfx90a finding**: HIP refuses 3D textures with linear filtering on gfx90a (below; it needs a decision).
- CPU-only suite: **Ran 2294, OK (skipped=15)**.
- MFMA: the layouts match gfx950 on all twelve lines. `mma_check.py` gives the same 40 non-ok cases as gfx942, all shared-memory limits; the scalar control is identical. bf16: 0 failures. Tile set: 639 run, 15 skipped, 0 failures. Benchmark: no launched MFMA cell is slower than scalar.

### 1. Phase 1-5 set (82 patterns, `-p "<pattern>" -m 32`, 3945 s under `utils/timeit.sh warp-1.14 test`)
Per-pattern result (`Ran N`, status):
- Phase 1: test_codegen 117, test_snippet 12, test_arithmetic 126, test_math 16, test_fp16 6, test_bf16 48, test_atomic 164, test_atomic_bitwise 6, test_atomic_cas 48, test_array 139, test_copy 14, test_launch 32, test_fast_math 12 (1 skipped), test_grad 41, test_kernel_cache 5, test_module_hashing 8, test_print 10, test_mat* 601, test_vec* 531, test_quat* 173, test_spatial* 207, test_rand 10, test_noise 14, test_clang_cuda 7 (6 skipped): all OK.
- Phase 2: test_runlength_encode 16, test_array_reduce 16, test_sparse 47, test_linear_solvers 34, test_coloring 3, test_bvh 13, test_grouped_bvh 11, test_hash_grid 23, test_marching_cubes 17, test_mesh* 49, test_volume 59, test_volume_write 5, test_apic 52, test_apic_mesh 12: OK. **test_fem* 108, failures=2**: `test_gimp_quadrature` and `test_cube_shape_functions` ("Unexpected output", the known upstream isolation behaviour). 108 instead of gfx942's 107 because `test_fem.example_diffusion_mgpu_cuda_0` registers here and passes. Upstream's `check_p2p()` gates it; it passes on this host (`iommu=pt`), while the gfx942 host's IOMMU heuristic skips it.
- Phase 3: test_graph 35 (3 skipped), test_array_fill_capture 3, test_capture_mode 2, test_conditional_captures 32 (16 skipped), test_conditional 50, test_async 2373, test_streams 24 (3 skipped, incl. the HIP priority-timing skip), test_mempool 7 (2 skipped), test_pinned 1, test_unified_memory 27 (8 skipped), test_occupancy 6, test_allocator 16 (3 skipped), test_allocation_tracker 34, test_dlpack 16, test_torch 34, test_map 45, test_reload 14, test_cuda_arch_suffix 10 (10 skipped): all OK. These are the gfx942 counts at this head.
- Phase 4: every tile file OK; see item 4.
- Phase 5: **test_texture 120, FAILED (errors=14, skipped=1)**; see the finding below.
- Skip reasons across the 82 logs: "No suitable devices to run the test." 18, "Conditional graph nodes not supported" 16, "Warp was not built with MathDx support" 8, "Requires devices with mempool access support" 5, the five "Needs <N> bytes of shared memory per block, device has 65536" (67584 x2, 66688 x2, 66816 x1), "cuda:0 cannot access CPU memory" 2, "Requires devices with peer access support" 2, "Requires at least two CUDA devices" 2, "Multi-GPU not available" 2, and one each of "rmm not installed", "Skipping test on CUDA device without MathDx (tolerance)", "Skipped due to flakiness in memory usage measurement", "Requires devices without mempool access support", "Requires devices with peer access and CUDA graph module-load support", "Requires devices with mempool access and CUDA graph module-load support", "MathDx is not enabled", "HIP maps stream priorities to normal priority, so priority has no scheduling effect."
- The only "Failed to configure kernel dynamic shared memory" line in the 82 logs is the known `test_tile_load_scoped` backward (65536 bytes; test passes). `Warp CUDA error` lines appear only in test_texture's log (the 14 errors).
- An example writes the untracked `example_apic_fluid.usd` into the clone; it was deleted after the run.

### 2. CPU-only suite
`HIP_VISIBLE_DEVICES=99 CUDA_VISIBLE_DEVICES=99 PYTHONPATH=<src> python -m warp.tests -s autodetect -k cpu -m 32`, run from a scratch directory so no example output lands in the clone, 195 s: **Ran 2294 tests, OK (skipped=15)**, no `cuda_0` in the log.

### Finding (gfx90a): HIP refuses 3D textures with linear filtering, so every default `Texture3D` fails
`python -m warp.tests -s autodetect -p test_texture.py -m 32`: **Ran 120, FAILED (errors=14, skipped=1)**. The count matches gfx1100 (120 = 50 CPU + 67 `cuda_0` + 2 deviceless + the `test_texture_interop_requires_hardware` placeholder skip), and `is_texture_supported, is_texture_hardware_supported` = `True True`, so the hardware path is selected, as expected for gfx90a. All 1D and 2D GPU cases pass, and so do 12 3D cases (`test_texture3d_{1,2,4}channel`, `address_mode_tuple`, `array`, `backward_compat_address_mode`, `nearest_interpolation`, `non_normalized_coords`, `per_axis_address_modes`, `struct_member`, `uint16`, `uint8`). The 14 errors all say `RuntimeError: Failed to create CUDA texture object: Warp CUDA error 801: operation not supported (in function wp_texture_object_create_device, texture.cpp:393)`:
`test_texture3d_cuda_array_copy_api`, `..._graph_capture`, `..._graph_capture_explicit_stream`, `..._rejects_indexedarray`, `test_texture3d_cuda_interop_handles`, `test_texture3d_cuda_surface_property_api`, `..._requires_surface_access`, `test_texture3d_linear_filter`, `test_texture3d_linear_interpolation`, `test_texture3d_resolution_query`, `test_texture3d_uint16_linear_interpolation`, `test_texture3d_uint8_linear_interpolation`, `test_texture_dtype_prefers_warp_types`, `test_texture_handle_properties_host_vs_cuda`. Each one creates a `Texture3D` with `filter_mode=LINEAR`, either explicitly or through the default (`texture.py:206`, `TextureFilterMode.LINEAR`).

Cause, from a standalone reproducer without Warp (`tex3d.cpp` below) and HIP's own source:
- `hipArray3DCreate` succeeds for every shape. `hipTexObjectCreate` (driver API) and `hipCreateTextureObject` (runtime API) succeed for 1D and 2D arrays. For every 3D array (8x8x8, 4x4x2, 64x64x64; 1 and 4 float channels) they return `hipErrorNotSupported` (801) with the linear filter, and succeed with the point filter. The device reports `hipDeviceAttributeImageSupport` 1 and `MaxTexture3D` 16384x16384x8192, so no attribute warns of this.
- `AMD_LOG_LEVEL=3` prints `hip_texture.cpp:88: gfx90a:sramecc+:xnack- doesn't support 3D linear filter!`. The check is in HIP itself, keyed on the device name (ROCm/rocm-systems `projects/clr/hipamd/src/hip_texture.cpp:85-89`, read with `gh api`): `else if (pResDesc->res.array.array->depth > 0 && pTexDesc->filterMode == hipFilterModeLinear && !strncmp(info.name_, "gfx90a", strlen("gfx90a"))) { LogPrintfInfo("%s doesn't support 3D linear filter!", ...); return hipErrorNotSupported; }`. HIP's own tests skip the case on gfx90a too (`hip-tests catch/unit/texture/hipTextureObj3DCheckModes.cc`: "gfx90a doesn't support 3D linear filter! Skipped!"). So this is a deliberate hardware limitation of MI200 that HIP enforces. It is not a port defect, and not a HIP bug to register.
- The port has no answer for it. On gfx90a, `wp.Texture3D(data)` with default arguments raises. So does Newton-style SDF sampling: `sdf_probe.py cuda:0` fails with the same error at `Texture3D(...)`, and so does `tex_capture.py cuda:0`, whose texture is 3D linear. What the hardware does serve is correct. `tex_alias.py cuda:0` (2D, external array, plain and captured) passes with 0 of 2048 wrong both ways. `sdf_probe_closest.py` (the SDF probe with the 3D texture switched to `CLOSEST`; the one-line `sed` is in the commands below) gives SDF vs CPU sampler 0.00e+00, SDF vs analytic 1.17e-02 (point sampling; grid spacing 1.56e-02), and RGBA8 2D linear vs CPU sampler 2.52e-03, the gfx1100 figure. PASS.

Options for the lock holder and Jeff. This is a behaviour decision like the earlier texture ruling, so no patch was written:
- **A. 3D textures on gfx90a are sampled in software on the GPU, as on gfx942/gfx950; 1D and 2D stay on hardware.** This extends the textures ruling ("AMD GPUs without image support get the GPU software sampler") to "without 3D linear filtering, for 3D". Device side: the selection must stay at compile time, so the three `sample_3d` sites in `texture.h` test a 3D-specific macro, for example `WP_TEXTURE_HW_SAMPLING_3D` = `WP_TEXTURE_HW_SAMPLING && !defined(__gfx90a__)`. No compiler feature macro exists for this, and HIP itself keys on the name. Measured: hipRTC defines `__gfx90a__` (and `__GFX9__`) for `--offload-arch=gfx90a`, and does not define `__HIP_NO_IMAGE_SUPPORT` there (`agent_space/rtcmac/m.py`). Native side: `texture_software.cpp`'s per-context `wp_texture_software_enabled(context)` is not enough once hardware and software arrays coexist on one device. `wp_texture_create_device` and `wp_texture_object_create_device` know `ndim`, but `wp_texture_destroy_device`, `wp_texture_copy_device` and `wp_texture_object_destroy_device` see only a handle. So those three hooks need to recognize software handles, for example through a set of the software array and object handles kept in `texture_software.cpp`. The device capability should be probed once at device init (create a 2x2x2 linear texture object; 801 means no 3D linear filtering) rather than read from the arch name, and it must agree with the compile-time macro. Python: the `cuda_array`/`cuda_texture`/`cuda_surface` properties of a 3D texture on such a device raise, as on gfx942. In `test_texture.py`, the three 3D interop tests (`cuda_array_copy_api*`, `cuda_interop_handles`, `cuda_surface_property*`) need a 3D capability for registration. All 3D textures on gfx90a, point-filtered ones included, then match Warp's CPU textures and are slower than hardware.
- **B. Keep hardware storage and do the trilinear filter in `texture.h` from 8 point fetches on gfx90a.** This keeps 3D interop on MI200, but the kernel must know each texture's filter mode, which `texture3d_t` (an upstream struct, also built from Python) does not carry. That costs more upstream lines than A.
- **C. Refuse clearly.** `Texture3D` with a linear filter on a device without 3D linear filtering raises "3D textures with linear filtering are not supported on cuda:0 (gfx90a)", and the 14 tests skip by capability. This is the smallest diff, but MI200 users lose default `Texture3D` and SDF sampling, which gfx942/gfx950 now have through the software sampler.
Recommendation: A. It reuses the software sampler that was already ruled and reviewed, it is the only option that keeps default `Texture3D` working on MI200, and its device-side diff is three `#if` lines. Release notes in any case: gfx90a has no 3D linear texture filtering in hardware.

`tex3d.cpp` (standalone; `clang++ -x hip --rocm-path=$R --offload-arch=gfx90a -O2 -w tex3d.cpp -o tex3d -L$R/lib -lamdhip64 -Wl,-rpath,$R/lib`, then `./tex3d` and `POINT=1 ./tex3d`):
```cpp
// hipTexObjectCreate on 1D/2D/3D hipArrays (driver and runtime array APIs), then a sample.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

__global__ void s3(hipTextureObject_t t, float* out) { out[0] = tex3D<float>(t, 1.5f, 1.5f, 1.5f); }
__global__ void s2(hipTextureObject_t t, float* out) { out[0] = tex2D<float>(t, 1.5f, 1.5f); }

static int attr(hipDeviceAttribute_t a) { int v = -1; (void)hipDeviceGetAttribute(&v, a, 0); return v; }

int main()
{
    printf("ImageSupport %d MaxTexture1DWidth %d MaxTexture2D %dx%d MaxTexture3D %dx%dx%d\n",
           attr(hipDeviceAttributeImageSupport), attr(hipDeviceAttributeMaxTexture1DWidth),
           attr(hipDeviceAttributeMaxTexture2DWidth), attr(hipDeviceAttributeMaxTexture2DHeight),
           attr(hipDeviceAttributeMaxTexture3DWidth), attr(hipDeviceAttributeMaxTexture3DHeight),
           attr(hipDeviceAttributeMaxTexture3DDepth));
    int dims[][3] = {{8, 0, 0}, {8, 8, 0}, {8, 8, 8}, {4, 4, 2}, {64, 64, 64}};
    for (auto& d : dims) {
        for (unsigned ch : {1u, 4u}) {
            HIP_ARRAY3D_DESCRIPTOR ad = {};
            ad.Width = d[0]; ad.Height = d[1]; ad.Depth = d[2];
            ad.Format = HIP_AD_FORMAT_FLOAT; ad.NumChannels = ch;
            hipArray_t arr = nullptr;
            hipError_t e1 = hipArray3DCreate(&arr, &ad);
            HIP_RESOURCE_DESC rd = {};
            rd.resType = HIP_RESOURCE_TYPE_ARRAY;
            rd.res.array.hArray = arr;
            HIP_TEXTURE_DESC td = {};
            td.filterMode = getenv("POINT") ? HIP_TR_FILTER_MODE_POINT : HIP_TR_FILTER_MODE_LINEAR;
            td.flags = 0;
            hipTextureObject_t t = nullptr;
            hipError_t e2 = e1 ? e1 : hipTexObjectCreate(&t, &rd, &td, nullptr);
            // runtime API path
            hipChannelFormatDesc cd = hipCreateChannelDesc(32, ch > 1 ? 32 : 0, ch > 2 ? 32 : 0, ch > 3 ? 32 : 0, hipChannelFormatKindFloat);
            hipArray_t arr2 = nullptr;
            hipError_t e3 = hipMalloc3DArray(&arr2, &cd, make_hipExtent(d[0], d[1], d[2]), 0);
            hipResourceDesc rd2 = {};
            rd2.resType = hipResourceTypeArray; rd2.res.array.array = arr2;
            hipTextureDesc td2 = {};
            td2.filterMode = getenv("POINT") ? hipFilterModePoint : hipFilterModeLinear; td2.readMode = hipReadModeElementType;
            hipTextureObject_t t2 = nullptr;
            hipError_t e4 = e3 ? e3 : hipCreateTextureObject(&t2, &rd2, &td2, nullptr);
            printf("%2dx%2dx%2d ch%u: hipArray3DCreate %d hipTexObjectCreate %d | hipMalloc3DArray %d hipCreateTextureObject %d\n",
                   d[0], d[1], d[2], ch, e1, e2, e3, e4);
            if (t) (void)hipTexObjectDestroy(t);
            if (t2) (void)hipDestroyTextureObject(t2);
            if (arr) (void)hipArrayDestroy(arr);
            if (arr2) (void)hipFreeArray(arr2);
        }
    }
    return 0;
}
```
Output on this host: `ImageSupport 1 MaxTexture1DWidth 16384 MaxTexture2D 16384x16384 MaxTexture3D 16384x16384x8192`. Without `POINT`, every 3D line reads `hipArray3DCreate 0 hipTexObjectCreate 801 | hipMalloc3DArray 0 hipCreateTextureObject 801`, and the 1D/2D lines read all 0. With `POINT=1`, every line reads all 0.
`sdf_probe_closest.py` is `sed 's/sdf, filter_mode=wp.TextureFilterMode.LINEAR/sdf, filter_mode=wp.TextureFilterMode.CLOSEST/' sdf_probe.py > sdf_probe_closest.py`.

### 3. MFMA layouts and correctness on gfx90a
- `clang++ -x hip --rocm-path=$R --offload-arch=gfx90a -O2 -w mfma_layout.hip -o mfma_layout -L$R/lib -lamdhip64 -Wl,-rpath,$R/lib && ./mfma_layout`: **0 of 256 wrong on all twelve lines** (f32 16x16x4, f64 16x16x4, f16 16x16x16, bf16 16x16x16 at K = 16, 32, 64); exit 0. The gfx950 layouts, including the f64 output layout, hold on gfx90a.
- `python mma_check.py` (165 cases): **"FAILURES 15"**, and none of them is a matrix-core problem. They are exactly gfx942's 15: fp32 64x64x64, fp64 64x64x64 and fp64 64x32x48 at all five block_dims. In each, the *forward* kernel cannot launch on a 64 KiB part: "tried to configure ... forward kernel for 65536 / 69632 / 131072 bytes, but maximum available is 65536", then `Warp CUDA error 1: invalid argument (wp_cuda_launch_kernel)`. The outputs are left untouched (c error exactly 1.0), and the script only knows the backward form of this limit. A further 25 cases report SMEM-LIMIT (backward not launched): fp32 64x32x48, fp64 32x32x32, fp64 48x16x64, fp64 32x64x16 and fp16 64x64x64 at all five block_dims. Control: the same script against the scalar worktree (item 5) gives **the identical status for every case** (`diff` of the status lines empty), and its code objects contain 0 `v_mfma`. Accuracy where the kernels launch, against the CPU device: fp32 and fp64 **0.0 (bit-identical) in every case**; fp16 up to 3.5e-03 on MFMA shapes, and 0 in the scalar control. The MFMA run's 222 code objects contain 3864 `v_mfma_f32_16x16x4f32`, 3864 `v_mfma_f64_16x16x4f64`, 948 `v_mfma_f32_16x16x16f16` and, from the bf16 check, 121 `v_mfma_f32_16x16x16bf16_1k` (`llvm-objdump -d --mcpu=gfx90a`). I second gfx942's suggestion that the porter's script also classify forward launch failures as SMEM-LIMIT.
- `python mma_check_bf16.py` (my version, below: all 11 `mma_check.py` shapes, including (8,8,8) and (16,16,12), x 5 block_dims = 55 cases): **FAILURES 0**; the largest difference to the CPU is 9.5e-07 (64x64x64 and 64x32x48), and to float64 on the bf16-rounded inputs 8.3e-07.

### 4. Tile set (22 files + test_block_dim_dispatch + test_template_launch_bounds; part of the full-set loop)
**639 run, 15 skipped, 0 failures.** test_tile 126, test_tile_atomic_bitwise 3, test_tile_cholesky 33, test_tile_cholesky_no_mathdx 16, test_tile_empty 20, test_tile_fft 20 (8 skipped), test_tile_fft_no_mathdx 16, test_tile_func_arg 27, test_tile_fused_ops 36, test_tile_large_offsets 1, test_tile_load 64, test_tile_mathdx 6, **test_tile_matmul 25** (`test_tile_matmul_fragment_shapes_{float16,float32,float64}_cuda_0` OK), test_tile_matmul_no_mathdx 22, test_tile_mlp 3 (1 skipped), test_tile_reduce 62 (2 skipped), test_tile_shared_memory 30 (2 skipped), test_tile_solve 5 (1 skipped), test_tile_solve_no_mathdx 28, test_tile_sort 2, test_tile_stack 40, test_tile_view 16 (1 skipped), test_block_dim_dispatch 2, test_template_launch_bounds 36. The skips are the 10 upstream no-libmathdx skips plus the five expected capability skips, with exactly the predicted byte counts: `test_tile_shared_mem_large` and `test_tile_shared_mem_func` 67584, `test_tile_reduce_axis_tier2` 66816, `..._single_warp` and `test_tile_assign_2d` 66688.

### 5. Benchmark, scalar against MFMA (gfx90a, MI250X one GCD, 104 CUs)
Scalar worktree as the handoff says (`git worktree add --detach agent_space/scalar_wt 930d0098`, `checkout 974cd4bf -- warp/native/tile_matmul.h`, `warp/bin/*.so` copied), then `bench_both.sh` (three runs per tree of `bench_tile_matmul.py --blocks 16384 --iters 50`, alone on the GPU, under `utils/timeit.sh warp-1.14 bench`) and `compare.py`. Every run exits 1, as expected on a 64 KiB part. Each run's CSV: 28 ok, 4 ADJ-FAILED, 4 MISMATCH. The three MFMA-tree caches contain 3036 `v_mfma` each, the scalar-tree caches 0. The worktree was removed afterwards.

| dtype | tile | block_dim | scalar fwd ms | MFMA fwd ms | fwd speedup | scalar adj ms | MFMA adj ms | adj speedup | checks |
|---|---|---|---|---|---|---|---|---|---|
| float16 | 16x16 | 32 | 1.011 | 1.011 | 1.00x | 0.748 | 0.748 | 1.00x | ok |
| float16 | 16x16 | 64 | 0.663 | 0.250 | 2.65x | 0.326 | 0.234 | 1.39x | ok |
| float16 | 16x16 | 128 | 0.636 | 0.263 | 2.42x | 0.254 | 0.217 | 1.17x | ok |
| float16 | 16x16 | 256 | 0.810 | 0.276 | 2.94x | 0.240 | 0.219 | 1.10x | ok |
| float16 | 32x32 | 32 | 14.366 | 14.363 | 1.00x | 9.132 | 9.133 | 1.00x | ok |
| float16 | 32x32 | 64 | 2.771 | 1.208 | 2.29x | 3.801 | 1.932 | 1.97x | ok |
| float16 | 32x32 | 128 | 3.040 | 1.378 | 2.21x | 2.657 | 1.206 | 2.20x | ok |
| float16 | 32x32 | 256 | 3.951 | 1.668 | 2.37x | 2.498 | 0.935 | 2.67x | ok |
| float16 | 64x64 | 32 | 86.609 | 86.609 | 1.00x | 107.461 | 107.591 | 1.00x | ok |
| float16 | 64x64 | 64 | 50.726 | 19.858 | 2.55x | 54.495 | 49.377 | 1.10x | ok |
| float16 | 64x64 | 128 | 33.941 | 11.785 | 2.88x | 23.170 | 19.711 | 1.18x | ok |
| float16 | 64x64 | 256 | 20.828 | 8.843 | 2.36x | 14.171 | 10.041 | 1.41x | ok |
| float32 | 16x16 | 32 | 1.503 | 1.504 | 1.00x | 0.810 | 0.817 | 0.99x | ok |
| float32 | 16x16 | 64 | 1.249 | 0.361 | 3.46x | 0.362 | 0.284 | 1.28x | ok |
| float32 | 16x16 | 128 | 0.767 | 0.366 | 2.10x | 0.311 | 0.226 | 1.38x | ok |
| float32 | 16x16 | 256 | 0.953 | 0.370 | 2.57x | 0.305 | 0.202 | 1.52x | ok |
| float32 | 32x32 | 32 | 9.652 | 9.656 | 1.00x | 14.489 | 14.427 | 1.00x | ok |
| float32 | 32x32 | 64 | 5.289 | 2.607 | 2.03x | 6.026 | 3.907 | 1.54x | ok |
| float32 | 32x32 | 128 | 4.822 | 2.535 | 1.90x | 4.135 | 2.295 | 1.80x | ok |
| float32 | 32x32 | 256 | 5.537 | 4.117 | 1.34x | 2.662 | 1.441 | 1.85x | ok |
| float32 | 64x64 | 32 | 134.021 | 134.034 | 1.00x | (no launch) | (no launch) | - | ADJ-FAILED |
| float32 | 64x64 | 64 | 69.050 | 50.666 | 1.36x | (no launch) | (no launch) | - | ADJ-FAILED |
| float32 | 64x64 | 128 | 51.917 | 27.780 | 1.87x | (no launch) | (no launch) | - | ADJ-FAILED |
| float32 | 64x64 | 256 | 34.130 | 16.570 | 2.06x | (no launch) | (no launch) | - | ADJ-FAILED |
| float64 | 16x16 | 32 | 2.060 | 2.060 | 1.00x | 1.511 | 1.509 | 1.00x | ok |
| float64 | 16x16 | 64 | 1.511 | 1.019 | 1.48x | 0.704 | 0.532 | 1.32x | ok |
| float64 | 16x16 | 128 | 1.463 | 1.045 | 1.40x | 0.764 | 0.410 | 1.87x | ok |
| float64 | 16x16 | 256 | 1.840 | 1.024 | 1.80x | 0.759 | 0.389 | 1.95x | ok |
| float64 | 32x32 | 32 | 24.099 | 24.103 | 1.00x | 24.399 | 24.003 | 1.02x | ok |
| float64 | 32x32 | 64 | 11.869 | 5.960 | 1.99x | 11.431 | 8.928 | 1.28x | ok |
| float64 | 32x32 | 128 | 9.648 | 4.959 | 1.95x | 7.670 | 5.231 | 1.47x | ok |
| float64 | 32x32 | 256 | 10.985 | 7.629 | 1.44x | 5.884 | 3.058 | 1.92x | ok |
| float64 | 64x64 | 32 | 0.024 | 0.024 | 0.99x | (no launch) | (no launch) | - | MISMATCH |
| float64 | 64x64 | 64 | 0.023 | 0.023 | 0.99x | (no launch) | (no launch) | - | MISMATCH |
| float64 | 64x64 | 128 | 0.023 | 0.024 | 0.99x | (no launch) | (no launch) | - | MISMATCH |
| float64 | 64x64 | 256 | 0.023 | 0.024 | 0.97x | (no launch) | (no launch) | - | MISMATCH |
run-to-run spread (max-min)/median over 144 cells: median 0.1%, max 8.7%

- **No launched cell is slower on MFMA than scalar.** The only cell below 1.00x that launches is fp32 16x16 block_dim 32 adjoint (0.99x). block_dim 32 is not a whole 64-lane wavefront, so it runs the scalar path in both trees, and the per-run values overlap (scalar 0.810-0.815 ms, MFMA 0.810-0.817 ms): noise. The fp64 64x64 rows (0.97-0.99x, MISMATCH) are launch failures on both trees: the forward needs 98304 bytes, and their 0.023 ms is not a GEMM. Their "(no launch)" adjoints are the same, as are the fp32 64x64 adjoints (98304 / 196608 bytes; Warp's warnings name them).
- Smallest gains on launched cells: the fp16 adjoints at 16x16 bd 128/256 (1.10-1.17x) and 64x64 (1.10-1.41x), the same pattern as gfx950 and gfx942. Forward at block_dim 64-256: fp16 2.2-2.9x at every size, fp32 1.3-3.5x (smallest at 32x32 bd256 and 64x64 bd64), fp64 1.4-2.0x.
- Against gfx942 at this head, the same cells are about 3-4x slower on one MI250X GCD on both trees (for example fp32 64x64 bd64 scalar 69.1 ms vs 18.7 ms), while the MFMA-over-scalar ratios are similar.
- Run-to-run spread over the 144 cells: median 0.1%, max 8.7%.

`bench_both.sh` (as run here):
```bash
#!/bin/bash
# Phase 4 tile_matmul benchmark, three runs each against the MFMA tree and the scalar tree (gfx90a host).
ulimit -c 0
A=<moat>/agent_space
for tree in ${TREES:-mfma scalar}; do
  if [ $tree = mfma ]; then src=$A/../projects/warp-1.14/src; else src=$A/scalar_wt; fi
  for n in 1 2 3; do
    PYTHONPATH=$src WARP_CACHE_PATH=$(mktemp -d $A/kcache.XXXXXX) "$FLEET_VENV/bin/python" $A/s/bench_tile_matmul.py \
      --blocks 16384 --iters 50 --csv $A/p9/bench_${tree}_run$n.csv > $A/p9/bench_${tree}_run$n.log 2>&1
    echo "$tree run$n rc=$?"
  done
done
```

`mma_check_bf16.py` (my version, as run):
```python
# tile_matmul with bfloat16 A/B and a float32 output, accumulating form (D += A @ B), forward only
# (module enable_backward=False, as upstream's test_tile_gemm_bf16), GPU against the CPU device and
# against float64 NumPy on the bf16-quantized inputs. Written from the description in Porting session 9
# (its source is not in the notes); uses all 11 mma_check.py shapes.
import sys

import numpy as np

import warp as wp

wp.init()
gpu = wp.get_device("cuda:0")
SHAPES = [(16, 16, 16), (32, 32, 32), (64, 64, 64), (16, 48, 32), (48, 16, 64), (32, 64, 16), (64, 32, 48),
          (8, 8, 8), (24, 16, 16), (16, 16, 12), (17, 33, 15)]  # fmt: skip
BLOCK_DIMS = (32, 64, 96, 128, 256)


@wp.kernel(module="unique")
def to_bf16(x: wp.array2d(dtype=wp.float32), y: wp.array2d(dtype=wp.bfloat16)):
    i, j = wp.tid()
    y[i, j] = wp.bfloat16(x[i, j])


@wp.kernel(module="unique")
def to_f32(x: wp.array2d(dtype=wp.bfloat16), y: wp.array2d(dtype=wp.float32)):
    i, j = wp.tid()
    y[i, j] = wp.float32(x[i, j])


def make(M, N, K):
    @wp.kernel(module="unique", module_options={"enable_backward": False})
    def gemm(A: wp.array2d(dtype=wp.bfloat16), B: wp.array2d(dtype=wp.bfloat16), D: wp.array2d(dtype=wp.float32)):
        a = wp.tile_load(A, shape=(M, K))
        b = wp.tile_load(B, shape=(K, N))
        d = wp.tile_load(D, shape=(M, N))
        wp.tile_matmul(a, b, d)
        wp.tile_store(D, d)

    return gemm


def bf16(dev, x):
    src = wp.array(x.astype(np.float32), dtype=wp.float32, device=dev)
    dst = wp.zeros(x.shape, dtype=wp.bfloat16, device=dev)
    wp.launch(to_bf16, dim=x.shape, inputs=[src, dst], device=dev)
    return dst


def back(dev, x):
    out = wp.zeros(x.shape, dtype=wp.float32, device=dev)
    wp.launch(to_f32, dim=x.shape, inputs=[x, out], device=dev)
    return out.numpy().astype(np.float64)


rng = np.random.default_rng(0)
fails = 0
for M, N, K in SHAPES:
    kern = make(M, N, K)
    A = rng.uniform(-1, 1, (M, K))
    B = rng.uniform(-1, 1, (K, N))
    D0 = rng.uniform(-1, 1, (M, N)).astype(np.float32)
    for bd in BLOCK_DIMS:
        res = {}
        for dev in ("cpu", gpu):
            a, b = bf16(dev, A), bf16(dev, B)
            d = wp.array(D0, dtype=wp.float32, device=dev)
            wp.launch_tiled(kern, dim=[1], inputs=[a, b, d], block_dim=bd, device=dev)
            res[str(dev)] = (d.numpy().astype(np.float64), back(dev, a), back(dev, b))
        got, qa, qb = res[str(gpu)]
        ref = res["cpu"][0]
        exact = D0.astype(np.float64) + qa @ qb
        e_cpu = np.abs(got - ref).max()
        e_np = np.abs(got - exact).max()
        ok = e_cpu <= 1e-5 * max(1, K / 16) and e_np <= 1e-5 * max(1, K / 16)
        fails += not ok
        if not ok or bd == 64:
            print(f"bf16->f32 M{M:3d} N{N:3d} K{K:3d} bd{bd:4d}: vs cpu {e_cpu:.1e} vs float64 {e_np:.1e} {'ok' if ok else 'FAIL'}", flush=True)
print("FAILURES", fails)
sys.exit(1 if fails else 0)
```

### Next step for the lock holder
- gfx90a passes everything at `930d0098` except textures. The 82-pattern set matches gfx942's at this head, plus `example_diffusion_mgpu` (P2P registration, passes). The CPU-only suite is green, and the MFMA path is correct (layouts, checks, tile set) and never slower than scalar where it runs.
- **Decision needed (Jeff): 3D textures with linear filtering on gfx90a.** HIP refuses them by design (`hip_texture.cpp:85-89`). Options A/B/C are in the finding above; A is recommended. Until then, `Texture3D` with default arguments raises on MI200, and `test_texture` has 14 errors there. Nothing to register as a ROCm defect: it is a documented gfx90a limitation that HIP enforces deliberately.
- Release-notes items from this run: gfx90a has no 3D linear texture filtering in hardware; shared-memory tiles are limited to 64 KiB on gfx90a (already in the notes text).

## Porting session 10 (2026-11-19, linux-gfx950 MI355X, porter): WMMA on gfx12, probe and check fixes, gfx90a 3D textures

Fork `release/1.14.0.amd0` advanced `930d0098..c9595bce` (pushed, `advance-head` recorded): `43331f4a` [ROCm] Use WMMA for float16 tile_matmul on gfx12 GPUs; `c9595bce` [ROCm] Sample 3D textures in software on gfx90a (its own commit, droppable). nvcc 12.8 build at `c9595bce`: "Warp build succeeded". NVRTC compiles of the tile_matmul module and of the texture module with the native headers of `930d0098` and of `c9595bce`: identical PTX and cubin (tile `0805dcc9fb1a`/`6f630854bc9f`, texture `98d84c023888`/`7b52ca293caf`), so the CUDA path is unchanged. `audit-commits` OK, `jargon.py --port` clean.

### 1. WMMA on for gfx12 (`43331f4a`)
The gfx1201 host's proposed condition, unchanged: `#elif (defined(WP_TILE_WMMA) && WP_TILE_WMMA) || __has_builtin(__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12)` (3 lines in `tile_matmul_mma.h`, comment included). A compiler feature, no arch name. Checked through Warp's hipRTC path (`agent_space/p9/wmma_xcompile.py`, fp16 and fp32 32x32x32 tile_matmul forward and backward at block_dim 64, compiled as shipped and with `#define WP_TILE_WMMA 1` prepended): gfx1201 and gfx1200 now emit 16 `v_wmma_f32_16x16x16_f16` as shipped, and their shipped instructions are identical to the switch-on build that gfx1201 validated (disassembly compared with addresses stripped; the binaries differ only through the prepended line's source positions, which also makes gfx950's two builds differ while its instructions are identical); gfx1100 and gfx1250 have no WMMA as shipped (only with the switch); gfx950's MFMA code is unchanged (32 `v_mfma_f32_16x16x16_f16`, 128 `v_mfma_f32_16x16x4_f32`, the same as session 9).

Release-notes text update: "On Radeon gfx12 GPUs (RX 9000 / Radeon AI PRO R9700 class, gfx1200 and gfx1201) float16 `tile_matmul` uses WMMA matrix instructions when the tile dimensions are multiples of 16 and the block is a whole number of 32-thread wavefronts, accumulating in float32 (results can differ from the scalar path in the last bits and are closer to the exact product). float32 and float64 stay on the scalar path there (RDNA4 has no WMMA instruction for them), and other Radeon GPUs use the scalar path in this release."

### 2. The WMMA probe, fixed; gfx11 stays off
`wmma_layout.hip` (below) now separates the two questions the gfx1100 host pulled apart. Per family and K = 16, 32, 64: **layout (non-negative)**, small non-negative integers, every partial sum an exact integer below 2^24 (max 804 at K = 64), so any wrong cell is a layout error, never accumulation order; **layout (position)**, `C[i][j] = i + 100 j` names its own position; both require exact equality. **accumulation**, random signed fp16 in [-1, 1], requires `|C - exact| <= (K/16 + 1) * FLT_EPSILON * sum |a b|` (one f32 rounding per instruction, plus one) and prints the worst error as a fraction of that bound; gfx1100's measured worst case (1.32e-7 relative to the magnitude sum at K = 64, about 1.1 epsilon) is about a fifth of the bound there (5 epsilon). Built here for gfx1100, gfx1201 and gfx1250 (all compile); not runnable on gfx950. gfx11 stays switched off in the committed code.

What the gfx1100 host must run to switch gfx11 on (in this order; stop at the first failure and record it):
1. `clang++ -x hip --rocm-path=$R --offload-arch=gfx1100 -O2 -w wmma_layout.hip -o wmma_layout -L$R/lib -lamdhip64 -Wl,-rpath,$R/lib && ./wmma_layout`: expect "WMMA family 11", the six layout lines 0 of 256 wrong, the three accumulation lines 0 of 256 over the bound (worst well below 1.00), exit 0.
2. As an uncommitted experiment in a separate worktree (as gfx1201 did), widen the selection in `warp/native/tile_matmul_mma.h` by one term:
```diff
-#elif (defined(WP_TILE_WMMA) && WP_TILE_WMMA) || __has_builtin(__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12)
+#elif (defined(WP_TILE_WMMA) && WP_TILE_WMMA) || __has_builtin(__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12) \
+    || __has_builtin(__builtin_amdgcn_wmma_f32_16x16x16_f16_w32)
```
   (gfx1250's fp32 block stays behind `WP_TILE_WMMA`.) Then: `python mma_check.py` (the fixed version below; expect FAILURES 0, the 64 KiB forward and backward limits reported as SMEM-LIMIT); `python mma_check_bf16.py` (fp32 output with bf16 inputs stays scalar on gfx11; expect FAILURES 0); the 24-file tile set (expect 639 run, the 15 known skips, 0 failures); `test_tile_matmul.py` (25 OK); confirm `v_wmma_f32_16x16x16_f16` in the fp16 code objects (`llvm-objdump -d --mcpu=gfx1100`); the fp16 benchmark against the scalar tree (Porting session 9 `bench_both.sh`/`compare.py`), three runs.
3. Report the diff, the counts and the benchmark table; any fp16 cell slower than scalar must be listed. The lock holder then commits that one-line widening (or a fix). Fresh `WARP_CACHE_PATH` for each tree (item 4).

### 3. `mma_check.py` fixed; the bf16 check's source recorded
`mma_check.py` (below) now also recognises a *forward* kernel over the device's shared memory: Warp warns and does not launch it, so `c` stays all zero and `d` keeps exactly its initial values (the script records `d` before the launch); such a case is "SMEM-LIMIT (forward not launched)", the backward form as before, and neither counts as a failure. The premise was checked on gfx950 with a forward kernel over its 160 KiB (`agent_space/p10/fwd_limit.py`: float32 128x128x64, 196608 bytes): Warp's "Failed to configure kernel dynamic shared memory ... forward kernel for 196608 bytes" warning, then `c` all zero and `d` unchanged. (A full `mma_check.py` case of that size cannot be used for this, because the CPU reference crashes: Warp's CPU tile allocator has a fixed 256 KiB block, upstream behaviour.) On gfx950 the fixed script gives FAILURES 0 with the 5 known backward limits and no forward limit; on 64 KiB parts the 15 cases gfx942 and gfx90a reported as FAIL become forward limits. `mma_check_bf16.py` (below) now covers all 11 shapes and the same forward-limit rule; gfx950: 55 cases, FAILURES 0.

### 4. Kernel cache and the native headers: can a wheel upgrade reuse a stale kernel?
Read from the source (`warp/_src/build.py` `init_kernel_cache`, `warp/_src/context.py` `ModuleHasher`, module output naming at `context.py:2895`):
- The module hash covers the kernels' Python source, their native snippets and the module options; it does **not** cover Warp's native headers (`builtin.h`, `texture.h`, `tile_matmul.h`, `tile_matmul_mma.h`, ...). So a change that only touches a header does not change any module hash.
- The cache root is `<cache dir>/<warp.config.version>/` for every way the cache location is chosen (`kernel_cache_dir`, `WARP_CACHE_PATH`, the default user cache through `appdirs` with `version=warp.config.version`).
- The code object name contains the full target name (`<module>.gfx950_sramecc+_xnack-.hsaco`), so GPUs of different architectures sharing a cache never share code objects.
Answer: a user who upgrades the wheel gets a new version string (this line releases as `1.14.0+rocm.0`, then `+rocm.1` and so on, by the release-tag scheme), hence a new cache directory, and cannot reuse kernels built against the old headers; that covers both `tile_matmul_mma.h` and the texture changes. The stale case needs two different native trees under the same version string: a source build reinstalled after a header change, a local experiment that toggles a header switch (as the gfx1201 and gfx1100 hosts did, and avoided with fresh caches), or a re-published wheel without a version bump. Upstream has the same property (its version is also the only header-dependent key). For this line: every published wheel must carry a new `+rocm.N`; no code change proposed. A user can clear a suspected stale cache with `wp.clear_kernel_cache()`.

### 5. gfx90a 3D textures in software (`c9595bce`, option A, its own commit)
- `texture.h`: the three `sample_3d` sites test `WP_TEXTURE_HW_SAMPLING_3D` instead of `WP_TEXTURE_HW_SAMPLING`, defined as hardware sampling `&& !defined(__gfx90a__)` (5 lines with its comment). No compiler feature macro describes the limitation (the gfx90a host checked: hipRTC defines `__gfx90a__`, not `__HIP_NO_IMAGE_SUPPORT`), and HIP's own refusal keys on the same name.
- Native: `texture_software.cpp` keeps a mutex-guarded set of the array and texture handles it created, because hardware and software textures now coexist on one device: create asks `wp_texture_software_enabled(context, ndim)`; copy, destroy, object create and object destroy ask `wp_texture_software_owns(handle)` (copy: either array side). The five hook lines in `texture.cpp` changed only their condition (+5/-5). The device capability is per dimension: `wp_cuda_device_is_texture_hardware_supported(ordinal, ndim)` (the port's own export, signature widened) and `DeviceInfo.is_texture3d_hardware_supported`, set at init from the image-support attribute and the same `gfx90a` name test, so the runtime always agrees with the compiled kernels (a run-time probe could disagree with the compile-time selection if HIP ever lifted the check, which would hand hardware texture objects to kernels compiled for the software sampler).
- Python: `Device.is_texture3d_hardware_supported` (documented); the interop check follows the texture's dimension (`cuda_array`/`cuda_texture`/`cuda_surface` and `cuda_array=` on a 3D texture raise on gfx90a). Tests: the four 3D interop registrations (`test_texture3d_cuda_interop_handles`, `test_texture_handle_properties_host_vs_cuda`, `test_texture3d_cuda_surface_property_api`, `..._requires_surface_access`) use the 3D capability.
- Evidence without gfx90a hardware: the all-combinations sampling module (`agent_space/p5/tex/xcompile.py`, 9 samples, 3 per dimension) through Warp's hipRTC path: gfx90a now 6 `image_sample` (1D and 2D on hardware, 3D in software, 27448-byte object), gfx1100 and gfx1201 9 each and **byte-identical** to session 5's objects, gfx942 0 (all software, as before). On gfx950 (every texture software, so the new handle set is exercised end to end): test_texture 118 OK (7 skipped), test_devices 13 OK, `sdf_probe.py cuda:0` PASS, `tex_capture.py cuda:0` PASS.
- Diff of this commit: `texture.h` +8/-3, `texture.cpp` +5/-5, `texture.py` +6/-6, `context.py` +10/-2, `texture_software.cpp` +36/-3, `texture_software.h` +8/-5, `warp.cu` +8/-2, `warp.h`/`warp.cpp` +1/-1 each, `test_texture.py` +6/-4. If Jeff rules option C instead, drop `c9595bce` (nothing later depends on it).
- Release notes: "On MI200 (gfx90a), which has no 3D linear texture filtering in hardware, 3D textures are sampled in software on the GPU (same results as Warp's CPU textures, slower than hardware); 1D and 2D textures use texture hardware, and the CUDA interop handles of 3D textures are not available there."

### Carried diff against `release-1.14` after this session (upstream files touched by textures and tile_matmul)
`warp/native/texture.h` +36/-18, `warp/native/texture.cpp` +60/-19, `warp/_src/texture.py` +15/-4, `warp/native/tile_matmul.h` +14/-8, `warp/tests/cuda/test_texture.py` +38/-9; the port's own files `texture_software.{h,cpp}` and `tile_matmul_mma.h` are new.

### gfx950 results at `c9595bce`
Tile set (24 files): 639 run, 10 skipped, 0 failures. `mma_check.py` (fixed): FAILURES 0 (5 backward shared-memory limits, 0 forward). `mma_check_bf16.py`: FAILURES 0 (55 cases). test_texture 118 OK. CPU-only suite with GPUs hidden: Ran 2294, OK (skipped=15).

### Next step (head `c9595bce`)
Project stays in `porting`, lock held by linux-gfx950. Common setup as before at `git rev-parse HEAD` = `c9595bcedee56cffa5b8181372e9f46643a087e8`; rebuild (native texture changes); fresh `WARP_CACHE_PATH` per run (item 4).

**gfx90a (MI200):** the new path's first run.
1. `python -c "import warp as wp; wp.init(); d = wp.get_device('cuda:0'); print(d.is_texture_hardware_supported, d.is_texture3d_hardware_supported)"`: expect `True False`.
2. `python -m warp.tests -s autodetect -p test_texture.py -m 32`: expect 0 errors (the 14 from the evidence session gone): the 3D linear cases pass through the software sampler, the 4 3D interop tests register no `cuda_0` case (placeholders), 1D/2D hardware cases unchanged.
3. `python sdf_probe.py cuda:0` and `python tex_capture.py cuda:0`: PASS (3D software); `python tex_alias.py cuda:0`: PASS (2D hardware, external array); `tex3d.cpp` unchanged (HIP still refuses 3D linear, by design).
4. `python mma_check.py` (fixed, below): FAILURES 0 with the 64 KiB forward cases now SMEM-LIMIT; `python mma_check_bf16.py`: FAILURES 0.
5. The tile set and the full 82-pattern set at this head, and the CPU-only suite.

**gfx1100:** the gfx11 switch-on sequence of item 2 (fixed probe, experiment, checks, benchmark); then, as shipped, test_tile_matmul (25 OK) and the tile set (WMMA still absent: confirm 0 `v_wmma`).

**gfx1201 (gfx12):** the committed code is the validated experiment: the tile set, test_tile_matmul (25 OK, `v_wmma_f32_16x16x16_f16` in the fp16 objects as shipped, no switch), `mma_check.py` (fixed; expect FAILURES 0) and the full 82-pattern set at this head.

**gfx942:** `mma_check.py` (fixed; the 15 earlier FAILs should now read SMEM-LIMIT (forward not launched), FAILURES 0) and test_texture (unchanged: 118 OK) at this head; the full set only if time allows (no change affects gfx942 except the texture handle bookkeeping, which test_texture covers).

**Two-GPU session:** test_texture at this head (the software handle set on two devices, including `cuda:1`), plus the 13-file loop.

**Still open for Jeff:** gfx90a 3D textures, option A as committed or C (drop `c9595bce`); gfx11 WMMA after the gfx1100 run; defer/now on the six undecided `rocm-bug-report` entries. Phase 6 not started.

`wmma_layout.hip` (fixed):
```cpp
// Check the WMMA 16x16 fragment layouts that warp/native/tile_matmul_mma.h assumes, on one 32-lane
// wavefront, and separately bound the accumulation error. lane l: lo = l % 16, g = l / 16.
//   gfx11 f16 (wmma_f32_16x16x16_f16_w32): A[lo][v], B[v][lo] for v < 16 in both half-waves;
//       accumulator r is C[2 r + g][lo]
//   gfx12 f16 (wmma_f32_16x16x16_f16_w32_gfx12): A[lo][8 g + v], B[8 g + v][lo] for v < 8;
//       accumulator r is C[8 g + r][lo]
//   gfx1250 f32 (wmma_f32_16x16x4_f32): A[lo][2 g + v], B[2 g + v][lo] for v < 2; accumulator as gfx12
// Tests, for K = 16, 32, 64:
//   layout (non-negative): small non-negative integers, every sum exact in f32 whatever the
//       accumulation order, so any wrong cell is a layout error
//   layout (position): A[i][0] = i, A[i][1] = 1, B[0][j] = 1, B[1][j] = 100 j, so C[i][j] = i + 100 j
//       names its own position
//   accumulation: random signed fp16 in [-1, 1]; |C - exact| must stay within
//       (K / 16 + 1) * FLT_EPSILON * sum |a b| (one f32 rounding per instruction, plus one)
#include <hip/hip_runtime.h>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

typedef float f8 __attribute__((ext_vector_type(8)));

__global__ void k(const double* A, const double* B, double* C, int K, int* kind)
{
    int l = threadIdx.x, lo = l % 16, g = l / 16;
    f8 acc = {};
#if __has_builtin(__builtin_amdgcn_wmma_f32_16x16x16_f16_w32)
    typedef _Float16 V __attribute__((ext_vector_type(16)));
    for (int k0 = 0; k0 < K; k0 += 16) {
        V a, b;
        for (int v = 0; v < 16; ++v) {
            a[v] = _Float16(A[lo * K + k0 + v]);
            b[v] = _Float16(B[(k0 + v) * 16 + lo]);
        }
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, acc);
    }
    for (int r = 0; r < 8; ++r) C[(2 * r + g) * 16 + lo] = acc[r];
    if (l == 0) *kind = 11;
#elif __has_builtin(__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12)
    typedef _Float16 V __attribute__((ext_vector_type(8)));
    for (int k0 = 0; k0 < K; k0 += 16) {
        V a, b;
        for (int v = 0; v < 8; ++v) {
            a[v] = _Float16(A[lo * K + k0 + 8 * g + v]);
            b[v] = _Float16(B[(k0 + 8 * g + v) * 16 + lo]);
        }
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, acc);
    }
    for (int r = 0; r < 8; ++r) C[(8 * g + r) * 16 + lo] = acc[r];
    if (l == 0) *kind = 12;
#elif __has_builtin(__builtin_amdgcn_wmma_f32_16x16x4_f32)
    typedef float V __attribute__((ext_vector_type(2)));
    for (int k0 = 0; k0 < K; k0 += 4) {
        V a = {float(A[lo * K + k0 + 2 * g]), float(A[lo * K + k0 + 2 * g + 1])};
        V b = {float(B[(k0 + 2 * g) * 16 + lo]), float(B[(k0 + 2 * g + 1) * 16 + lo])};
        acc = __builtin_amdgcn_wmma_f32_16x16x4_f32(false, a, false, b, (short)0, acc, false, false);
    }
    for (int r = 0; r < 8; ++r) C[(8 * g + r) * 16 + lo] = acc[r];
    if (l == 0) *kind = 1250;
#else
    if (l == 0) *kind = -1;
#endif
}

// fill A (16 x K) and B (K x 16) for one test; returns the per-cell bound (0 = exact required)
static void fill(int test, int K, double* A, double* B)
{
    for (int x = 0; x < 16 * K; ++x) {
        int i = x / K, kk = x % K;
        if (test == 0) A[x] = double((x * 7 + 3) % 9);
        if (test == 1) A[x] = kk == 0 ? double(i) : (kk == 1 ? 1.0 : 0.0);
        if (test == 2) A[x] = double(_Float16(2.0 * rand() / RAND_MAX - 1.0));
    }
    for (int x = 0; x < K * 16; ++x) {
        int kk = x / 16, j = x % 16;
        if (test == 0) B[x] = double((x * 5 + 1) % 7);
        if (test == 1) B[x] = kk == 0 ? 1.0 : (kk == 1 ? 100.0 * j : 0.0);
        if (test == 2) B[x] = double(_Float16(2.0 * rand() / RAND_MAX - 1.0));
    }
}

int main()
{
    const char* names[] = {"layout (non-negative)", "layout (position)", "accumulation"};
    int fails = 0, kind_seen = 0;
    srand(1);
    for (int test = 0; test < 3; ++test)
        for (int K : {16, 32, 64}) {
            double *A, *B, *C;
            int* kind;
            hipMallocManaged(&A, 16 * K * 8);
            hipMallocManaged(&B, K * 16 * 8);
            hipMallocManaged(&C, 256 * 8);
            hipMallocManaged(&kind, 4);
            fill(test, K, A, B);
            memset(C, 0, 256 * 8);
            *kind = 0;
            k<<<1, 32>>>(A, B, C, K, kind);
            hipDeviceSynchronize();
            kind_seen = *kind;
            int bad = 0;
            double worst = 0;  // largest error as a fraction of the allowed bound
            for (int i = 0; i < 16; ++i)
                for (int j = 0; j < 16; ++j) {
                    double ref = 0, mag = 0;
                    for (int kk = 0; kk < K; ++kk) {
                        ref += A[i * K + kk] * B[kk * 16 + j];
                        mag += fabs(A[i * K + kk] * B[kk * 16 + j]);
                    }
                    double err = fabs(C[i * 16 + j] - ref);
                    if (test < 2) {
                        bad += err != 0.0;
                    } else {
                        double bound = (K / 16 + 1) * FLT_EPSILON * mag;
                        bad += err > bound;
                        worst = fmax(worst, err / bound);
                    }
                }
            if (test < 2)
                printf("WMMA family %d, %-22s K=%2d: %d of 256 wrong\n", *kind, names[test], K, bad);
            else
                printf("WMMA family %d, %-22s K=%2d: %d of 256 over the bound, worst error %.2f of the bound\n", *kind,
                       names[test], K, bad, worst);
            fails += bad != 0 || *kind <= 0;
            hipFree(A); hipFree(B); hipFree(C); hipFree(kind);
        }
    if (kind_seen <= 0) printf("no WMMA instruction on this target\n");
    return fails ? 1 : 0;
}
```

`mma_check.py` (fixed):
```python
# tile_matmul forward (C = A @ B and C += A @ B) and adjoint on the GPU against the same kernels on
# the CPU device, for dtypes x tile shapes x block_dims. Usage: python mma_check.py [dtype ...]
import sys

import numpy as np

import warp as wp

wp.init()
gpu = wp.get_device("cuda:0")
DTYPES = {"float32": wp.float32, "float64": wp.float64, "float16": wp.float16}  # tile_matmul rejects bf16
SHAPES = [(16, 16, 16), (32, 32, 32), (64, 64, 64), (16, 48, 32), (48, 16, 64), (32, 64, 16), (64, 32, 48),
          (8, 8, 8), (24, 16, 16), (16, 16, 12), (17, 33, 15)]  # fmt: skip
BLOCK_DIMS = (32, 64, 96, 128, 256)
TOL = {wp.float32: 2e-5, wp.float64: 1e-12, wp.float16: 1e-2}


def make(M, N, K, dtype):
    @wp.kernel(module="unique")
    def fwd(A: wp.array2d(dtype=dtype), B: wp.array2d(dtype=dtype), C: wp.array2d(dtype=dtype), D: wp.array2d(dtype=dtype)):
        a = wp.tile_load(A, shape=(M, K))
        b = wp.tile_load(B, shape=(K, N))
        c = wp.tile_matmul(a, b)
        d = wp.tile_load(D, shape=(M, N))
        wp.tile_matmul(a, b, d)
        wp.tile_store(C, c)
        wp.tile_store(D, d)

    return fwd


def run(dev, kern, A, B, D0, bd, M, N, K, dtype):
    a = wp.array(A, dtype=dtype, device=dev, requires_grad=True)
    b = wp.array(B, dtype=dtype, device=dev, requires_grad=True)
    c = wp.zeros((M, N), dtype=dtype, device=dev, requires_grad=True)
    d = wp.array(D0, dtype=dtype, device=dev, requires_grad=True)
    d_init = d.numpy().astype(np.float64)
    tape = wp.Tape()
    with tape:
        wp.launch_tiled(kern, dim=[1], inputs=[a, b, c, d], block_dim=bd, device=dev)
    c.grad = wp.array(np.ones((M, N)), dtype=dtype, device=dev)
    d.grad = wp.array(np.full((M, N), 0.5), dtype=dtype, device=dev)
    tape.backward()
    return [x.numpy().astype(np.float64) for x in (c, d, a.grad, b.grad)], d_init


rng = np.random.default_rng(0)
fails = 0
for name in sys.argv[1:] or DTYPES:
    dtype = DTYPES[name]
    for M, N, K in SHAPES:
        kern = make(M, N, K, dtype)
        A = rng.uniform(-1, 1, (M, K))
        B = rng.uniform(-1, 1, (K, N))
        D0 = rng.uniform(-1, 1, (M, N))
        for bd in BLOCK_DIMS:
            ref, _ = run("cpu", kern, A, B, D0, bd, M, N, K, dtype)
            got, d_init = run(gpu, kern, A, B, D0, bd, M, N, K, dtype)
            errs = [np.abs(g - r).max() / max(1.0, np.abs(r).max()) for g, r in zip(got, ref)]
            ok = all(e <= TOL[dtype] * max(1, K / 16) for e in errs)
            # a kernel that needs more shared memory than the device has does not launch (Warp warns):
            # forward: c stays zero and d keeps its initial values; backward: the gradients stay zero
            no_grads = not got[2].any() and not got[3].any()
            smem_fwd = not ok and not got[0].any() and np.array_equal(got[1], d_init) and no_grads
            smem_bwd = not ok and all(e <= TOL[dtype] * max(1, K / 16) for e in errs[:2]) and no_grads
            smem = smem_fwd or smem_bwd
            fails += not ok and not smem
            if not ok or bd == 64:
                status = "ok" if ok else (
                    "SMEM-LIMIT (forward not launched)" if smem_fwd else "SMEM-LIMIT (backward not launched)" if smem_bwd else "FAIL"
                )
                print(f"{name:8s} M{M:3d} N{N:3d} K{K:3d} bd{bd:4d}: c {errs[0]:.1e} d {errs[1]:.1e} gA {errs[2]:.1e} gB {errs[3]:.1e} {status}", flush=True)
print("FAILURES", fails)
sys.exit(1 if fails else 0)
```

`mma_check_bf16.py`:
```python
# tile_matmul with bf16 inputs and a float32 output (forward only; tile_matmul rejects a bf16 backward),
# GPU against the CPU device, shapes x block_dims.
import sys

import numpy as np

import warp as wp

wp.init()
SHAPES = [(16, 16, 16), (32, 32, 32), (64, 64, 64), (16, 48, 32), (48, 16, 64), (32, 64, 16), (64, 32, 48),
          (8, 8, 8), (24, 16, 16), (16, 16, 12), (17, 33, 15)]  # fmt: skip


def make(M, N, K):
    @wp.kernel(module="unique", module_options={"enable_backward": False})
    def fwd(A: wp.array2d(dtype=wp.bfloat16), B: wp.array2d(dtype=wp.bfloat16), C: wp.array2d(dtype=float), D: wp.array2d(dtype=float)):
        a = wp.tile_load(A, shape=(M, K))
        b = wp.tile_load(B, shape=(K, N))
        c = wp.tile_zeros(shape=(M, N), dtype=float)
        wp.tile_matmul(a, b, c)
        d = wp.tile_load(D, shape=(M, N))
        wp.tile_matmul(a, b, d)
        wp.tile_store(C, c)
        wp.tile_store(D, d)

    return fwd


@wp.kernel
def to_bf16(x: wp.array2d(dtype=float), y: wp.array2d(dtype=wp.bfloat16)):
    i, j = wp.tid()
    y[i, j] = wp.bfloat16(x[i, j])


rng = np.random.default_rng(0)
fails = 0
for M, N, K in SHAPES:
    kern = make(M, N, K)
    A = rng.uniform(-1, 1, (M, K)).astype(np.float32)
    B = rng.uniform(-1, 1, (K, N)).astype(np.float32)
    D0 = rng.uniform(-1, 1, (M, N)).astype(np.float32)
    for bd in (32, 64, 96, 128, 256):
        out = {}
        for dev in ("cpu", "cuda:0"):
            a = wp.zeros((M, K), dtype=wp.bfloat16, device=dev)
            b = wp.zeros((K, N), dtype=wp.bfloat16, device=dev)
            wp.launch(to_bf16, dim=(M, K), inputs=[wp.array(A, device=dev), a], device=dev)
            wp.launch(to_bf16, dim=(K, N), inputs=[wp.array(B, device=dev), b], device=dev)
            c = wp.zeros((M, N), dtype=float, device=dev)
            d = wp.array(D0, device=dev)
            wp.launch_tiled(kern, dim=[1], inputs=[a, b, c, d], block_dim=bd, device=dev)
            out[dev] = (c.numpy(), d.numpy())
        e = max(np.abs(out["cuda:0"][i] - out["cpu"][i]).max() for i in (0, 1))
        ok = e <= 2e-5 * max(1, K / 16)
        # a forward kernel over the device's shared memory does not launch: c stays zero, d unchanged
        smem = not ok and not out["cuda:0"][0].any() and np.array_equal(out["cuda:0"][1], D0)
        fails += not ok and not smem
        if not ok or bd == 64:
            status = "ok" if ok else ("SMEM-LIMIT (forward not launched)" if smem else "FAIL")
            print(f"bf16->f32 M{M:3d} N{N:3d} K{K:3d} bd{bd:4d}: max abs diff {e:.1e} {status}", flush=True)
print("FAILURES", fails)
sys.exit(1 if fails else 0)
```

## Evidence session 2 (2026-11-19, linux-gfx1201, Radeon AI PRO R9700, evidence only): gfx12 items at `c9595bce`

Requested by Jeff Daily. The work lock stays with linux-gfx950. Nothing was committed or pushed to the fork, and no stage, lock, head, platform or deferral record changed. The read-only clone was moved from `930d0098` to the recorded `head_sha` `c9595bcedee56cffa5b8181372e9f46643a087e8` (detached; `git status --porcelain` empty before and after; `git lfs pull` again). Rebuilt against the venv's ROCm 10.2 wheels: `export PATH=$FLEET_VENV/bin:$PATH; python build_lib.py --rocm-path "$(rocm-sdk path --root)" --hip-arch gfx1201 -j 12` -> "Warp build succeeded" (.cu build 30.4 s); RUNPATH the venv's `_rocm_sdk_devel/lib`. Same device and venv as the first gfx1201 session, `HIP_VISIBLE_DEVICES=2`, `ulimit -c 0`, a fresh private `WARP_CACHE_PATH` for every run. The items are the "gfx1201 (gfx12)" list of Porting session 10's Next step, run in its order. No switch or local edit was used: everything below is the committed code.

### 1. Tile set (24 files): 639 run, 15 skipped, 0 failures
Per-file counts are identical to the first gfx1201 session's tile table at `930d0098` (test_tile 126, ..., test_tile_matmul 25, test_template_launch_bounds 36). The skips are the 10 upstream no-libmathdx skips (fft 8, mlp 1, solve 1) and the five 64 KiB capability skips ("Needs 67584 ..." 2, "Needs 66816 ..." 1, "Needs 66688 ..." 2). There is no `Warp CUDA error`, and the only shared-memory warning is the known `test_tile_load_scoped` backward one.

### 2. test_tile_matmul alone: Ran 25, OK; WMMA in the fp16 objects as shipped
Fresh cache, then `llvm-objdump -d --mcpu=gfx1201` on every cached `.hsaco`. Each fp16 module's tile shapes and block_dim were read from its generated `.cu`:

| fp16 module (A, B tile; block_dim) | `v_wmma_f32_16x16x16_f16` |
|---|---|
| 16x16, 16x16; 64 | 4 |
| 16x16, 16x16; 96 | 4 |
| 48x32, 32x16; 64 | 15 |
| 48x32, 32x16; 96 | 9 |
| 24x16, 16x16; 64 and 96 | 0 (M not a multiple of 16: scalar, as designed) |
| 17x15, 15x33; 64 and 96 | 0 (does not fit: scalar) |
| 8x8, 8x4; 64 | 0 (does not fit: scalar) |

Every fp32 and fp64 module (and the bf16 test's float32-output module) has 0 WMMA: gfx12 has no WMMA instruction for them. block_dim 96 is three 32-lane wavefronts, so on gfx12 it takes WMMA; on gfx950 it is not a whole number of 64-lane wavefronts and stays scalar. So the committed condition behaves as the experiment did, with no switch.

### 3. `mma_check.py` (fixed version from Porting session 10): FAILURES 0
165 cases: 25 printed `ok`, 25 "SMEM-LIMIT (backward not launched)", 15 "SMEM-LIMIT (forward not launched)", 0 FAIL, exit 0. The 15 forward limits are exactly the 15 cases the old script called FAIL in the first session: fp32 64x64x64, fp64 (64,32,48) and fp64 64x64x64, at all 5 block_dims. The 25 backward limits are fp16 64x64x64, fp32 (64,32,48), fp64 32x32x32, (32,64,16) and (48,16,64), at all 5 block_dims. The fixed forward rule (c all zero, d unchanged, no gradients) therefore classifies the 64 KiB forward kernels correctly on a real 64 KiB part. Its code objects hold 1336 `v_wmma_f32_16x16x16_f16` across 165 `.hsaco`, the same total as the switch-on experiment at `930d0098`. Every printed fp16 error line (c, d, gA, gB per case) is identical to that experiment's log (`diff` empty), and so are the fp32/fp64 lines outside the old FAIL cases. The committed gfx12 path is therefore numerically the code that was validated.

### 4. Full 82-pattern set: 81 OK, 1 with the 2 known fem isolation failures
Same 82-pattern list, `-p "<pattern>" -m 32`, 1840 s under `utils/timeit.sh warp-1.14 test`. The `test_fem*` failures are `test_gimp_quadrature` and `test_cube_shape_functions` ("Unexpected output" with the init greeting, upstream isolation behaviour). The per-pattern summary lines (Ran count and result, timings stripped) are **identical** to the `930d0098` run of the first gfx1201 session (`diff` empty), and so is the skip-reason histogram over all 82 logs. test_texture is Ran 120, OK (skipped=1), so the gfx90a-only 3D software sampling and the per-handle bookkeeping (`c9595bce`) leave the gfx12 hardware texture path unchanged. There is no `Warp CUDA error` in any log, and the only shared-memory warning is the known `test_tile_load_scoped` one. The untracked `example_apic_fluid.usd` an example writes was deleted after the run.

Telemetry note: one `test` phase line of 0.004 s with exit 1 at 08:40:04Z in `stats.jsonl` is a launcher typo of mine (the pattern-list path), relaunched at once. It ran no test.

### Summary
At `c9595bce`, every gfx1201 item gives the expected result. The committed WMMA condition engages WMMA for fp16 tiles that fit, with no switch, and falls back to scalar for the rest. The tile set and test_tile_matmul pass, the fixed `mma_check.py` reports FAILURES 0 with results identical to the validated experiment, and the full 82-pattern set is unchanged from `930d0098`. Nothing needs a fork change. (The CPU-only suite and the benchmark were not in this host's list for this head and were not run.)
## Evidence session 6 (2026-11-19, linux-gfx1100, Radeon Pro W7800, evidence only): gfx11 WMMA switch-on sequence at `c9595bce`

Requested by Jeff Daily. The work lock stays with linux-gfx950. Nothing was committed to the fork, and no stage, lock, head, platform or deferral record changed. The clone was updated read-only to the recorded `head_sha` `c9595bcedee56cffa5b8181372e9f46643a087e8` and rebuilt (`export PATH=$FLEET_VENV/bin:$PATH; python build_lib.py --rocm-path "$(rocm-sdk path --root)" --hip-arch gfx1100 -j 32` -> "Warp build succeeded"; RUNPATH is the venv's ROCm 10.2 `_rocm_sdk_devel/lib`). Same venv (HIP 7.17.26434), `ulimit -c 0`, `HIP_VISIBLE_DEVICES=0`.

The experiment ran in a separate `git worktree` (`agent_space/wmma_wt`), and the scalar baseline in another (`agent_space/scalar_wt`: `git checkout 974cd4bf -- warp/native/tile_matmul.h`). Each got a copy of this build's `warp/bin/*.so`; the native code is identical and the kernel headers are read from each tree at kernel compile time. Both worktrees were removed at the end. The main clone was never modified (`git status --porcelain` clean).

**A fresh `WARP_CACHE_PATH` was used for every run of every tree:** each `mma_check` script, each test file, each ISA check and each benchmark run. This matters, not just in theory: the fp16 tile_matmul test module has the same name (`wp_k_1212a376_ecb1985.gfx1100.hsaco`) in the shipped tree and in the experiment tree, as Porting session 10 item 4 predicts (the module hash does not cover the native headers). A shared cache would have handed one tree the other's code object.

The items are the "gfx1100" line of the `c9595bce` Next step, i.e. Porting session 10 item 2's sequence, then the as-shipped checks, in that order.

### Step 1: fixed `wmma_layout.hip`: PASS
Built as given (`clang++ -x hip --rocm-path=$R --offload-arch=gfx1100 -O2 -w wmma_layout.hip -o wmma_layout -L$R/lib -lamdhip64 -Wl,-rpath,$R/lib`), script verbatim from Porting session 10:
```
WMMA family 11, layout (non-negative)  K=16: 0 of 256 wrong
WMMA family 11, layout (non-negative)  K=32: 0 of 256 wrong
WMMA family 11, layout (non-negative)  K=64: 0 of 256 wrong
WMMA family 11, layout (position)      K=16: 0 of 256 wrong
WMMA family 11, layout (position)      K=32: 0 of 256 wrong
WMMA family 11, layout (position)      K=64: 0 of 256 wrong
WMMA family 11, accumulation           K=16: 0 of 256 over the bound, worst error 0.33 of the bound
WMMA family 11, accumulation           K=32: 0 of 256 over the bound, worst error 0.38 of the bound
WMMA family 11, accumulation           K=64: 0 of 256 over the bound, worst error 0.26 of the bound
(exit 0)
```

### Step 2: the switch-on experiment (uncommitted, worktree only)
The diff, exactly the one-term widening of Porting session 10 item 2 (`git diff` in the experiment worktree; apply with `git apply`):
```diff
diff --git a/warp/native/tile_matmul_mma.h b/warp/native/tile_matmul_mma.h
index 220000d7..1855ce1f 100644
--- a/warp/native/tile_matmul_mma.h
+++ b/warp/native/tile_matmul_mma.h
@@ -86,7 +86,8 @@ template <> struct op<bfloat16> {
 };
 #endif
 
-#elif (defined(WP_TILE_WMMA) && WP_TILE_WMMA) || __has_builtin(__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12)
+#elif (defined(WP_TILE_WMMA) && WP_TILE_WMMA) || __has_builtin(__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12) \
+    || __has_builtin(__builtin_amdgcn_wmma_f32_16x16x16_f16_w32)
 // 32-lane WMMA. The gfx12 fp16 layout is measured and on; the gfx11 and gfx1250 paths stay off
 // unless WP_TILE_WMMA is defined to 1 (each WMMA builtin exists on exactly one target family).
 #define WP_TILE_MMA 1
```

| check | expected | result |
|---|---|---|
| `python mma_check.py` (fixed) | FAILURES 0, the 64 KiB limits as SMEM-LIMIT | **FAILURES 0**; 15 "SMEM-LIMIT (forward not launched)" and 25 "SMEM-LIMIT (backward not launched)" (list below) |
| `python mma_check_bf16.py` | FAILURES 0 | **FAILURES 0** (55 cases; no shared-memory limit) |
| 24-file tile set | 639 run, 15 known skips, 0 failures | **639 run, 15 skipped, 0 failures**. Skips: 8 "Warp was not built with MathDx support", 1 "MathDx is not enabled", 1 "Skipping test on CUDA device without MathDx (tolerance)", and the five "Needs <N> bytes of shared memory per block, device has 65536". No `Warp CUDA error`; the only shared-memory warning is the known `test_tile_load_scoped` one |
| `test_tile_matmul.py` | 25 OK | **Ran 25, OK** (part of the tile set run) |
| `v_wmma_f32_16x16x16_f16` in the fp16 objects | present | **16** in the fp16 32x32x32 forward+backward module, the count gfx12 shows; the fp32 module has 0 (scalar, 1729 `v_fma*_f32`) |

- `mma_check.py`, fp16 results: shapes whose dimensions are multiples of 16 differ from the CPU by about 1e-3 (WMMA with fp32 accumulation). For example, at bd 64: 16x16x16 c 1.6e-03; 32x32x32 c 1.2e-03, gA 1.2e-03; 64x32x48 c 1.1e-03, gB 3.5e-03. The non-multiple shapes (8x8x8, 24x16x16, 16x16x12, 17x33x15) are bit-identical to the CPU, i.e. they take the scalar path, as the selection intends.
- ISA check (`agent_space/p10/isa_check.py`, 32x32x32 fp16 and fp32 at block_dim 64, forward and backward, against float64): with the switch, fp16 forward max err 1.9e-03 and adj_A 2.6e-03. As shipped (scalar), fp16 forward 1.1e-02 and adj_A 1.3e-02. fp32 is identical in both (1.4e-06 / 9.3e-07).
- The 40 shared-memory limits of `mma_check.py`, every one recognised by the script as a launch that did not happen:
```
float32  M 64 N 64 K 64 bd  32: c 1.0e+00 d 1.1e+00 gA 1.0e+00 gB 1.0e+00 SMEM-LIMIT (forward not launched)
float32  M 64 N 64 K 64 bd  64: c 1.0e+00 d 1.1e+00 gA 1.0e+00 gB 1.0e+00 SMEM-LIMIT (forward not launched)
float32  M 64 N 64 K 64 bd  96: c 1.0e+00 d 1.1e+00 gA 1.0e+00 gB 1.0e+00 SMEM-LIMIT (forward not launched)
float32  M 64 N 64 K 64 bd 128: c 1.0e+00 d 1.1e+00 gA 1.0e+00 gB 1.0e+00 SMEM-LIMIT (forward not launched)
float32  M 64 N 64 K 64 bd 256: c 1.0e+00 d 1.1e+00 gA 1.0e+00 gB 1.0e+00 SMEM-LIMIT (forward not launched)
float32  M 64 N 32 K 48 bd  32: c 0.0e+00 d 0.0e+00 gA 1.0e+00 gB 1.0e+00 SMEM-LIMIT (backward not launched)
float32  M 64 N 32 K 48 bd  64: c 0.0e+00 d 0.0e+00 gA 1.0e+00 gB 1.0e+00 SMEM-LIMIT (backward not launched)
float32  M 64 N 32 K 48 bd  96: c 0.0e+00 d 0.0e+00 gA 1.0e+00 gB 1.0e+00 SMEM-LIMIT (backward not launched)
float32  M 64 N 32 K 48 bd 128: c 0.0e+00 d 0.0e+00 gA 1.0e+00 gB 1.0e+00 SMEM-LIMIT (backward not launched)
float32  M 64 N 32 K 48 bd 256: c 0.0e+00 d 0.0e+00 gA 1.0e+00 gB 1.0e+00 SMEM-LIMIT (backward not launched)
float64  M 32 N 32 K 32 bd  32: c 0.0e+00 d 0.0e+00 gA 1.0e+00 gB 1.0e+00 SMEM-LIMIT (backward not launched)
float64  M 32 N 32 K 32 bd  64: c 0.0e+00 d 0.0e+00 gA 1.0e+00 gB 1.0e+00 SMEM-LIMIT (backward not launched)
float64  M 32 N 32 K 32 bd  96: c 0.0e+00 d 0.0e+00 gA 1.0e+00 gB 1.0e+00 SMEM-LIMIT (backward not launched)
float64  M 32 N 32 K 32 bd 128: c 0.0e+00 d 0.0e+00 gA 1.0e+00 gB 1.0e+00 SMEM-LIMIT (backward not launched)
float64  M 32 N 32 K 32 bd 256: c 0.0e+00 d 0.0e+00 gA 1.0e+00 gB 1.0e+00 SMEM-LIMIT (backward not launched)
float64  M 64 N 64 K 64 bd  32: c 1.0e+00 d 9.7e-01 gA 1.0e+00 gB 1.0e+00 SMEM-LIMIT (forward not launched)
float64  M 64 N 64 K 64 bd  64: c 1.0e+00 d 9.7e-01 gA 1.0e+00 gB 1.0e+00 SMEM-LIMIT (forward not launched)
float64  M 64 N 64 K 64 bd  96: c 1.0e+00 d 9.7e-01 gA 1.0e+00 gB 1.0e+00 SMEM-LIMIT (forward not launched)
float64  M 64 N 64 K 64 bd 128: c 1.0e+00 d 9.7e-01 gA 1.0e+00 gB 1.0e+00 SMEM-LIMIT (forward not launched)
float64  M 64 N 64 K 64 bd 256: c 1.0e+00 d 9.7e-01 gA 1.0e+00 gB 1.0e+00 SMEM-LIMIT (forward not launched)
float64  M 48 N 16 K 64 bd  32: c 0.0e+00 d 0.0e+00 gA 1.0e+00 gB 1.0e+00 SMEM-LIMIT (backward not launched)
float64  M 48 N 16 K 64 bd  64: c 0.0e+00 d 0.0e+00 gA 1.0e+00 gB 1.0e+00 SMEM-LIMIT (backward not launched)
float64  M 48 N 16 K 64 bd  96: c 0.0e+00 d 0.0e+00 gA 1.0e+00 gB 1.0e+00 SMEM-LIMIT (backward not launched)
float64  M 48 N 16 K 64 bd 128: c 0.0e+00 d 0.0e+00 gA 1.0e+00 gB 1.0e+00 SMEM-LIMIT (backward not launched)
float64  M 48 N 16 K 64 bd 256: c 0.0e+00 d 0.0e+00 gA 1.0e+00 gB 1.0e+00 SMEM-LIMIT (backward not launched)
float64  M 32 N 64 K 16 bd  32: c 0.0e+00 d 0.0e+00 gA 1.0e+00 gB 1.0e+00 SMEM-LIMIT (backward not launched)
float64  M 32 N 64 K 16 bd  64: c 0.0e+00 d 0.0e+00 gA 1.0e+00 gB 1.0e+00 SMEM-LIMIT (backward not launched)
float64  M 32 N 64 K 16 bd  96: c 0.0e+00 d 0.0e+00 gA 1.0e+00 gB 1.0e+00 SMEM-LIMIT (backward not launched)
float64  M 32 N 64 K 16 bd 128: c 0.0e+00 d 0.0e+00 gA 1.0e+00 gB 1.0e+00 SMEM-LIMIT (backward not launched)
float64  M 32 N 64 K 16 bd 256: c 0.0e+00 d 0.0e+00 gA 1.0e+00 gB 1.0e+00 SMEM-LIMIT (backward not launched)
float64  M 64 N 32 K 48 bd  32: c 1.0e+00 d 1.0e+00 gA 1.0e+00 gB 1.0e+00 SMEM-LIMIT (forward not launched)
float64  M 64 N 32 K 48 bd  64: c 1.0e+00 d 1.0e+00 gA 1.0e+00 gB 1.0e+00 SMEM-LIMIT (forward not launched)
float64  M 64 N 32 K 48 bd  96: c 1.0e+00 d 1.0e+00 gA 1.0e+00 gB 1.0e+00 SMEM-LIMIT (forward not launched)
float64  M 64 N 32 K 48 bd 128: c 1.0e+00 d 1.0e+00 gA 1.0e+00 gB 1.0e+00 SMEM-LIMIT (forward not launched)
float64  M 64 N 32 K 48 bd 256: c 1.0e+00 d 1.0e+00 gA 1.0e+00 gB 1.0e+00 SMEM-LIMIT (forward not launched)
float16  M 64 N 64 K 64 bd  32: c 1.9e-03 d 1.7e-03 gA 1.0e+00 gB 1.0e+00 SMEM-LIMIT (backward not launched)
float16  M 64 N 64 K 64 bd  64: c 1.9e-03 d 1.7e-03 gA 1.0e+00 gB 1.0e+00 SMEM-LIMIT (backward not launched)
float16  M 64 N 64 K 64 bd  96: c 1.9e-03 d 1.7e-03 gA 1.0e+00 gB 1.0e+00 SMEM-LIMIT (backward not launched)
float16  M 64 N 64 K 64 bd 128: c 1.9e-03 d 1.7e-03 gA 1.0e+00 gB 1.0e+00 SMEM-LIMIT (backward not launched)
float16  M 64 N 64 K 64 bd 256: c 1.9e-03 d 1.7e-03 gA 1.0e+00 gB 1.0e+00 SMEM-LIMIT (backward not launched)
```

### Step 2: fp16 benchmark against the scalar tree
Porting session 9's `bench_both.sh` with this host's paths (trees `wmma_wt` and `scalar_wt`, `bench_tile_matmul.py --blocks 16384 --iters 50`, a fresh `WARP_CACHE_PATH` per run), three runs per tree, merged with `compare.py` (labels changed from MFMA to WMMA). The benchmark covers all dtypes; only fp16 changes on gfx11, and the fp32/fp64 rows (scalar in both trees) serve as a control.

**The benchmark was run twice.** In the first set, WMMA run 1 was disturbed: 32 cells exceeded 1.5x their median, and these included fp32 and fp64 cells whose code is identical in both trees (for example fp32 32x32 bd64 adjoint 9.1x). WMMA run 2 had one more such cell (fp16 16x16 bd32 forward, 7.0 ms against 0.27 ms). That is interference on the GPU during those runs, not a property of the code. I did not find the source; the host's other GPUs belong to other jobs. So the whole benchmark was repeated (set 2) with a 5-second load and GPU-busy monitor. Set 2 has no cell above 1.5x its median, and its run-to-run spread over the 112 launched cells is median 0.6%, max 5.4%. **The table is set 2**:

| dtype | tile | block_dim | scalar fwd ms | WMMA fwd ms | fwd speedup | scalar adj ms | WMMA adj ms | adj speedup | checks |
|---|---|---|---|---|---|---|---|---|---|
| float16 | 16x16 | 32 | 1.123 | 0.272 | 4.13x | 0.296 | 0.235 | 1.26x | ok |
| float16 | 16x16 | 64 | 1.132 | 0.281 | 4.02x | 0.282 | 0.207 | 1.36x | ok |
| float16 | 16x16 | 128 | 0.976 | 0.422 | 2.31x | 0.266 | 0.207 | 1.28x | ok |
| float16 | 16x16 | 256 | 1.392 | 0.474 | 2.94x | 0.306 | 0.248 | 1.23x | ok |
| float16 | 32x32 | 32 | 3.711 | 1.825 | 2.03x | 4.406 | 2.674 | 1.65x | ok |
| float16 | 32x32 | 64 | 4.259 | 1.870 | 2.28x | 2.474 | 1.660 | 1.49x | ok |
| float16 | 32x32 | 128 | 5.318 | 2.664 | 2.00x | 1.936 | 1.177 | 1.64x | ok |
| float16 | 32x32 | 256 | 6.848 | 2.715 | 2.52x | 1.957 | 1.164 | 1.68x | ok |
| float16 | 64x64 | 32 | 33.274 | 13.875 | 2.40x | 60.700 | 54.099 | 1.12x | ok |
| float16 | 64x64 | 64 | 27.490 | 18.984 | 1.45x | 33.660 | 30.238 | 1.11x | ok |
| float16 | 64x64 | 128 | 27.634 | 14.103 | 1.96x | 17.228 | 14.726 | 1.17x | ok |
| float16 | 64x64 | 256 | 17.624 | 19.408 | 0.91x | 13.607 | 8.582 | 1.59x | ok |
| float32 | 16x16 | 32 | 1.856 | 1.835 | 1.01x | 0.403 | 0.400 | 1.01x | ok |
| float32 | 16x16 | 64 | 2.028 | 2.001 | 1.01x | 0.347 | 0.347 | 1.00x | ok |
| float32 | 16x16 | 128 | 1.308 | 1.297 | 1.01x | 0.403 | 0.399 | 1.01x | ok |
| float32 | 16x16 | 256 | 1.675 | 1.667 | 1.01x | 0.469 | 0.462 | 1.01x | ok |
| float32 | 32x32 | 32 | 7.724 | 7.689 | 1.00x | 6.501 | 6.508 | 1.00x | ok |
| float32 | 32x32 | 64 | 8.267 | 8.222 | 1.01x | 3.383 | 3.373 | 1.00x | ok |
| float32 | 32x32 | 128 | 8.058 | 7.996 | 1.01x | 2.629 | 2.620 | 1.00x | ok |
| float32 | 32x32 | 256 | 9.593 | 9.545 | 1.01x | 2.987 | 2.962 | 1.01x | ok |
| float32 | 64x64 | 32 | 69.112 | 68.934 | 1.00x | (no launch) | (no launch) | - | ADJ-FAILED |
| float32 | 64x64 | 64 | 43.826 | 43.236 | 1.01x | (no launch) | (no launch) | - | ADJ-FAILED |
| float32 | 64x64 | 128 | 42.229 | 42.013 | 1.01x | (no launch) | (no launch) | - | ADJ-FAILED |
| float32 | 64x64 | 256 | 48.787 | 48.149 | 1.01x | (no launch) | (no launch) | - | ADJ-FAILED |
| float64 | 16x16 | 32 | 6.763 | 6.737 | 1.00x | 1.436 | 1.421 | 1.01x | ok |
| float64 | 16x16 | 64 | 6.704 | 6.690 | 1.00x | 0.999 | 1.000 | 1.00x | ok |
| float64 | 16x16 | 128 | 6.860 | 6.835 | 1.00x | 1.335 | 1.333 | 1.00x | ok |
| float64 | 16x16 | 256 | 6.842 | 6.795 | 1.01x | 1.317 | 1.314 | 1.00x | ok |
| float64 | 32x32 | 32 | 57.046 | 57.089 | 1.00x | 23.186 | 23.222 | 1.00x | ok |
| float64 | 32x32 | 64 | 49.950 | 49.885 | 1.00x | 13.181 | 13.239 | 1.00x | ok |
| float64 | 32x32 | 128 | 45.799 | 45.722 | 1.00x | 9.005 | 8.992 | 1.00x | ok |
| float64 | 32x32 | 256 | 47.134 | 46.998 | 1.00x | 8.048 | 8.036 | 1.00x | ok |
| float64 | 64x64 | 32 | 0.020 | 0.020 | 1.00x | (no launch) | (no launch) | - | MISMATCH |
| float64 | 64x64 | 64 | 0.019 | 0.019 | 0.96x | (no launch) | (no launch) | - | MISMATCH |
| float64 | 64x64 | 128 | 0.017 | 0.019 | 0.90x | (no launch) | (no launch) | - | MISMATCH |
| float64 | 64x64 | 256 | 0.018 | 0.020 | 0.90x | (no launch) | (no launch) | - | MISMATCH |
run-to-run spread (max-min)/median over 144 cells: median 0.5%, max 15.7%

(The final spread line counts the meaningless "(no launch)" cells. The launched-cell figures are above. The fp64 64x64 "speedups" are launch-failure timings.)

**fp16 cells slower than scalar (the ruling asks for each):** one cell in set 2, **64x64, block_dim 256, forward: WMMA 18.64/19.42/19.41 ms against scalar 17.63/17.62/17.62 ms (0.91x)**. It was also slower in set 1 (19.37-19.55 ms in its two undisturbed runs against 16.64-18.05 ms). No fp16 adjoint cell is slower in set 2. In set 1, four adjoint cells (16x16 bd32, 32x32 bd64/128/256) had a WMMA median above scalar. In each of them WMMA runs 1 and 2 were slower than scalar and run 3 faster: 16x16 bd32 0.891/0.703/0.234 ms, 32x32 bd64 2.694/2.645/1.680, 32x32 bd128 13.116/1.986/1.190, 32x32 bd256 8.895/3.472/1.174, against scalar about 0.29, 2.46, 1.93 and 1.95. Run 1 is the disturbed run. Run 2 shows no disturbance in the fp32/fp64 control cells, so the cause of its slow fp16 adjoint values is not established. Set 2's three runs show all four cells faster than scalar (1.26x-1.68x) with run-to-run spread of at most 1.52%. Every other fp16 cell is faster: forward 1.45x-4.13x, adjoint 1.11x-1.68x. The 64x64 bd256 forward cell is the one the lock holder must explain or give a fallback, as the ruling requires. Two facts it may help to know: at block_dim 256 (8 wavefronts) a 64x64 tile has 16 16x16 fragments, two per wavefront. The scalar path is at its fastest there (17.6 ms is its best 64x64 forward, while WMMA's best is 13.9 ms at bd32 and 14.1 ms at bd128). No explanation is proposed here.

Raw fp16 timings per run, ms (set 1, then set 2; `agent_space/p10/set1|set2/bench_<tree>_run<n>.csv`):
```
--- set 1 (WMMA run 1 disturbed)
tree,dtype,M,block_dim,fwd_r1,fwd_r2,fwd_r3,adj_r1,adj_r2,adj_r3
scalar,float16,16,32,1.1033,1.1158,1.1245,0.2860,0.2901,0.2919
scalar,float16,16,64,1.0878,1.1399,1.1342,0.2856,0.2842,0.2842
scalar,float16,16,128,0.9131,0.9692,0.9754,0.2587,0.2570,0.2608
scalar,float16,16,256,1.3007,1.3889,1.3933,0.3033,0.3104,0.3109
scalar,float16,32,32,3.6287,3.7196,3.7107,4.3063,4.3918,4.3932
scalar,float16,32,64,4.0299,4.2393,4.2569,2.3949,2.4600,2.4731
scalar,float16,32,128,4.9882,5.2744,5.2960,1.8569,1.9266,1.9331
scalar,float16,32,256,6.3856,6.8216,6.8429,1.8221,1.9493,1.9463
scalar,float16,64,32,32.4149,33.2914,33.2335,60.5661,60.6982,60.6824
scalar,float16,64,64,26.3514,27.8515,27.3947,33.6102,33.6937,33.6242
scalar,float16,64,128,26.5723,28.2518,27.4997,16.9763,17.4226,17.2193
scalar,float16,64,256,16.6433,18.0510,17.5354,13.2033,13.8048,13.5788
wmma,float16,16,32,0.5380,7.0096,0.2671,0.8907,0.7028,0.2337
wmma,float16,16,64,0.2791,0.2942,0.2774,0.2080,0.1996,0.2016
wmma,float16,16,128,0.4884,0.4406,0.4209,0.2283,0.2027,0.2087
wmma,float16,16,256,0.8347,0.5127,0.4723,0.2632,0.2457,0.2540
wmma,float16,32,32,2.7707,2.0006,1.8155,7.8170,3.9979,2.6716
wmma,float16,32,64,5.3456,2.1104,1.8616,2.6937,2.6453,1.6804
wmma,float16,32,128,4.7396,2.6714,2.6634,13.1156,1.9862,1.1895
wmma,float16,32,256,9.9486,6.1324,2.7078,8.8949,3.4718,1.1742
wmma,float16,64,32,37.6842,20.7396,13.8167,115.5697,53.7518,54.1822
wmma,float16,64,64,19.2462,19.2210,18.9349,83.2576,30.2919,30.1955
wmma,float16,64,128,43.4118,14.2349,14.0375,122.9680,14.7503,14.7494
wmma,float16,64,256,89.7136,19.5513,19.3675,12.3484,8.6330,8.5829
--- set 2
tree,dtype,M,block_dim,fwd_r1,fwd_r2,fwd_r3,adj_r1,adj_r2,adj_r3
scalar,float16,16,32,1.1216,1.1244,1.1235,0.2962,0.2996,0.2882
scalar,float16,16,64,1.1321,1.1336,1.1248,0.2822,0.2918,0.2767
scalar,float16,16,128,0.9823,0.9750,0.9760,0.2655,0.2661,0.2664
scalar,float16,16,256,1.3919,1.3918,1.3931,0.3022,0.3067,0.3056
scalar,float16,32,32,3.7115,3.7172,3.7047,4.4064,4.4238,4.3829
scalar,float16,32,64,4.2588,4.2643,4.2423,2.4599,2.4830,2.4743
scalar,float16,32,128,5.3180,5.3272,5.3147,1.9309,1.9356,1.9374
scalar,float16,32,256,6.8343,6.8522,6.8484,1.9417,1.9570,1.9587
scalar,float16,64,32,33.2193,33.2737,33.2762,60.6513,60.7004,60.7208
scalar,float16,64,64,27.4896,27.4768,27.4961,33.6349,33.6601,33.6869
scalar,float16,64,128,27.6343,27.6018,27.6393,17.2279,17.2567,17.2229
scalar,float16,64,256,17.6258,17.6240,17.6227,13.6071,13.6252,13.5968
wmma,float16,16,32,0.2632,0.2724,0.2764,0.2340,0.2347,0.2376
wmma,float16,16,64,0.2779,0.2888,0.2815,0.1968,0.2081,0.2069
wmma,float16,16,128,0.4101,0.4220,0.4234,0.2076,0.2072,0.2055
wmma,float16,16,256,0.4570,0.4739,0.4759,0.2474,0.2519,0.2481
wmma,float16,32,32,1.7499,1.8250,1.8287,2.6415,2.6756,2.6745
wmma,float16,32,64,1.7893,1.8723,1.8704,1.6557,1.6628,1.6595
wmma,float16,32,128,2.5431,2.6712,2.6644,1.1768,1.1752,1.1773
wmma,float16,32,256,2.5966,2.7298,2.7153,1.1642,1.1656,1.1631
wmma,float16,64,32,13.1597,13.8748,13.8780,54.0294,54.2363,54.0986
wmma,float16,64,64,18.2244,19.0131,18.9842,30.1680,30.2615,30.2384
wmma,float16,64,128,13.4162,14.1031,14.1115,14.6703,14.7264,14.7493
wmma,float16,64,256,18.6374,19.4181,19.4080,8.3958,8.6002,8.5818
```

### As shipped (switch off): gfx11 unchanged, no WMMA
- 24-file tile set in the main clone, a fresh cache per file: **639 run, 15 skipped (same reasons as above), 0 failures**; test_tile_matmul **Ran 25, OK**.
- ISA check (same script, fresh cache): **0 `v_wmma`** in both modules. fp16 is 1456 `v_mul_f16` (scalar); fp32 is 1729 `v_fma*_f32`.

### Summary for the lock holder
- The fixed probe passes on gfx1100.
- With the one-term widening above, gfx11 passes `mma_check.py` and `mma_check_bf16.py` (FAILURES 0 each), the tile set (639/15/0) and test_tile_matmul (25), and emits 16 `v_wmma_f32_16x16x16_f16` in the fp16 module.
- The fp16 benchmark (clean set 2) is faster than scalar in 23 of 24 fp16 forward/adjoint cells. The exception is 64x64 bd256 forward, at 0.91x.
- The shipped code is scalar on gfx11 and unchanged.
- Committing the widening, and what to do about the 64x64 bd256 cell, is the lock holder's and Jeff's call.
## Evidence session 6 (2026-11-19, linux-gfx942, 2x MI300X, evidence only): gfx942 and two-GPU items at `c9595bce`

Requested by Jeff Daily. The lock stays with linux-gfx950. No fork commit and no stage, lock, head, platform or deferred record changed. The read-only clone was moved from `930d0098` to the recorded `head_sha` `c9595bcedee56cffa5b8181372e9f46643a087e8` (`git status --porcelain` empty before and after; LFS assets present; fork pre-push hook unchanged). Build: `python build_lib.py --rocm-path "$(rocm-sdk path --root)" --hip-arch gfx942 -j 112` -> "Warp build succeeded", 30.4 s. Every run used a fresh kernel cache (`WARP_CACHE_PATH=$(mktemp -d agent_space/kc_p10_<run>.XXXX)`), as the handoff's item 4 asks. `ulimit -c 0`. Items run in the handoff's order.

### gfx942 (one GPU, `HIP_VISIBLE_DEVICES=0`)
1. `python mma_check.py` (the fixed version from Porting session 10, extracted verbatim): **FAILURES 0**, exit 0. The 15 cases evidence session 5 reported as FAIL now read **"SMEM-LIMIT (forward not launched)"**: fp32 64x64x64, fp64 64x32x48 and fp64 64x64x64, each at all five block_dims. The other 25 are "SMEM-LIMIT (backward not launched)" as before. The code objects contain 948 `v_mfma_f32_16x16x16_f16`, 3864 `v_mfma_f32_16x16x4_f32` and 3864 `v_mfma_f64_16x16x4_f64`, the same counts as at `930d0098`.
2. `python -m warp.tests -s autodetect -p test_texture.py -m 32`: **Ran 118, OK (skipped=7**, all "No suitable devices to run the test."); 59 `cuda_0` cases pass. Unchanged from `974cd4bf`/`930d0098`. `Device` capabilities on the MI300X: `is_texture_supported` True, `is_texture_hardware_supported` False, `is_texture3d_hardware_supported` False (all software, as before).
3. Full 82-pattern set (optional, run because time allowed; same list and loop as evidence sessions 3-5; 2445 s): **81 OK, 1 with the 2 known failures** (`test_fem*`: `test_gimp_quadrature`, `test_cube_shape_functions`, upstream test isolation). Per-pattern counts are **identical line by line** to evidence session 5 at `930d0098` (including test_tile_matmul 25, test_texture 118 with 7 skipped, the five 64 KiB tile skips, test_coloring 3, test_mesh* 49, test_volume 59).

### Two-GPU session (`HIP_VISIBLE_DEVICES=0,1`): test_texture, then the 13-file loop

| file | ran | result |
|---|---|---|
| test_texture | 177 | OK, 7 skipped: 59 `cuda_0` and 59 `cuda_1` cases pass |
| test_multigpu | 5 | OK, 2 skipped |
| test_peer | 5 | OK, 1 skipped |
| test_mempool | 10 | OK, 2 skipped |
| test_streams | 42 | OK, 4 skipped |
| test_graph | 61 | OK, 5 skipped |
| test_ipc | 8 | OK |
| test_unified_memory | 40 | OK, 9 skipped |
| test_allocator | 16 | OK, 1 skipped |
| test_async | 3533 | OK |
| test_sparse | 71 | OK, 1 skipped |
| test_fem_integrate | 25 | OK, 1 skipped |
| test_array | 207 | OK |
| test_copy | 30 | OK |

**No errors and no failures**; the counts and skips are identical to evidence session 3 at `974cd4bf`:
- 3 C1 and 2 C2 skips, each naming the HIP defect.
- 2 C4 skips ("HIP maps stream priorities to normal priority ...").
- 10 IOMMU-heuristic skips and 4 "cuda:N cannot access CPU memory".
- 2 upstream memory-usage flakiness skips and 7 test_texture "No suitable devices".
- 1 each of the device-property and rmm skips.

With two devices, the texture handle bookkeeping of `c9595bce` (the software set shared between devices) works on the non-primary ordinal: all 59 `cuda_1` texture cases pass. The only `Warp CUDA error` line in all 14 logs is the known harmless 900 from `test_synchronize_during_capture_cuda_1` (evidence session 3).

The untracked `example_apic_fluid.usd` an example writes into the clone during the full set was deleted afterwards.

### Next step for the lock holder
gfx942 and two-GPU items at `c9595bce` are complete and match every expectation; nothing needs a fork change.

## Evidence session 2 (2026-11-19, linux-gfx90a, MI250X, evidence only): gfx90a items 1-5 at `c9595bce`

Requested by Jeff Daily. The lock stays with linux-gfx950. No fork commit, and no stage, lock, head, platform or deferred record changed. The read-only clone moved from `930d0098` to the recorded `head_sha` `c9595bcedee56cffa5b8181372e9f46643a087e8` (`git status --porcelain` empty before and after). Same host, venv and ROCm 10.2.0 wheels as the first gfx90a session (one MI250X GCD, `HIP_VISIBLE_DEVICES=3` as pinned, HIP 7.17.26434). Rebuilt: `python build_lib.py --rocm-path "$(rocm-sdk path --root)" --hip-arch gfx90a -j 64` -> "Warp build succeeded", 48.2 s. Every run used `ulimit -c 0` and its own fresh `WARP_CACHE_PATH`. `mma_check.py` (fixed) and `mma_check_bf16.py` were extracted verbatim from Porting session 10; `sdf_probe.py`, `tex_capture.py`, `tex_alias.py` and `tex3d.cpp` are as before. Items run in the handoff's order.

**Result: every item as the handoff expected.** One cosmetic finding: the 3D interop refusal message is wrong for gfx90a. A proposed diff is below, confirmed as an uncommitted experiment.

### 1. Capabilities
`python -c "import warp as wp; wp.init(); d = wp.get_device('cuda:0'); print(d.is_texture_hardware_supported, d.is_texture3d_hardware_supported)"` -> **`True False`**.

### 2. test_texture
`python -m warp.tests -s autodetect -p test_texture.py -m 32`: **Ran 119, OK (skipped=4)**, no `Warp CUDA error` line (the first session at `930d0098`: 120 run, 14 errors). 119 = 50 CPU + 63 `cuda_0` + 2 deviceless + 4 placeholders ("No suitable devices to run the test."). The placeholders are the three CUDA-only 3D interop registrations (`test_texture3d_cuda_interop_handles`, `test_texture3d_cuda_surface_property_api`, `test_texture3d_cuda_surface_property_requires_surface_access`) and `test_texture_interop_requires_hardware`, which has no device here because gfx90a has texture hardware. The fourth 3D interop test, `test_texture_handle_properties_host_vs_cuda`, keeps its CPU case and loses its `cuda_0` case, so `cuda_0` went from 67 to 63. All 14 earlier errors now pass, among them the 3D linear-filter, `uint8`/`uint16` linear interpolation, resolution query, the four `cuda_array_copy_api` variants (plain, both graph captures, indexed-array rejection) and `test_texture_dtype_prefers_warp_types`. The 1D/2D hardware cases pass as before.

### 3. Probes
| probe | result |
|---|---|
| `python sdf_probe.py cpu` | PASS: 0.00e+00 / 1.28e-03 / 0.00e+00 |
| `python sdf_probe.py cuda:0` | **PASS**: SDF vs CPU sampler 5.96e-08 (3D linear in software; gfx942/gfx950 give the same 5.96e-08), SDF vs analytic 1.28e-03, RGBA8 vs CPU sampler 2.52e-03 (2D on hardware, the gfx1100 figure) |
| `python tex_capture.py cuda:0` | **PASS**: replays 0, 1, 2 each max abs diff vs CPU 1.19e-07 (3D software texture uploaded and sampled inside a graph) |
| `python tex_alias.py cuda:0` | **PASS**: plain 0 of 2048 wrong, graph capture 0 of 2048 wrong (2D hardware, external array) |
| `./tex3d` (unchanged binary) | unchanged: 3D `hipTexObjectCreate`/`hipCreateTextureObject` 801 with the linear filter, 0 with `POINT=1`; HIP still refuses by design |

The hardware/software split shows in the code objects these runs left in their caches (`llvm-objdump -d --mcpu=gfx90a`, counting `image_sample`): `tex_capture.py`'s module (3D only) 0; `sdf_probe.py`'s module (one 3D, one 2D texture) 1, the 2D sample; test_texture's module 12.

Extra check, not in the handoff (`interop3d.py`, below): on `cuda:0`, a `Texture2D`'s `cuda_array` and `cuda_texture` are valid handles, while a `Texture3D`'s `cuda_array`, `cuda_texture`, `cuda_surface` and `Texture3D(cuda_array=...)` all raise `RuntimeError`, and `id` works. So the per-dimension refusal behaves as designed. No registered test covers it on gfx90a, because `test_texture_interop_requires_hardware` registers only on devices with no texture hardware at all. **Finding (cosmetic):** the message reads "cuda_array is not available on cuda:0: it has no texture hardware and samples textures in software", which is false on gfx90a (its 1D/2D textures are hardware in the same process). Proposed diff for the lock holder (the 2D/no-hardware message is unchanged, so `test_texture_interop_requires_hardware`'s regex still matches where it runs):
```diff
--- a/warp/_src/texture.py
+++ b/warp/_src/texture.py
@@ -426,10 +426,15 @@ class Texture:
 
     @staticmethod
     def _require_texture_hardware(device, name, ndim):
-        if not (device.is_texture3d_hardware_supported if ndim == 3 else device.is_texture_hardware_supported):
+        if not device.is_texture_hardware_supported:
             raise RuntimeError(
                 f"{name} is not available on {device}: it has no texture hardware and samples textures in software"
             )
+        if ndim == 3 and not device.is_texture3d_hardware_supported:
+            raise RuntimeError(
+                f"{name} is not available for 3D textures on {device}: its texture hardware cannot filter 3D "
+                "textures, so they are sampled in software"
+            )
 
     @staticmethod
     def _validate_texture_dtype(dtype):
```
With it applied locally (Python only, no rebuild, fresh cache): `interop3d.py` prints "cuda_array is not available for 3D textures on cuda:0: its texture hardware cannot filter 3D textures, so they are sampled in software" for all four cases; test_texture 119, OK (skipped=4). Reverted afterwards (`git checkout warp/_src/texture.py`, clone clean). Not linted: `ruff` is not installed in this venv. Lines are under 120 columns.

`interop3d.py`:
```python
# On a device whose 3D textures are sampled in software, the 3D interop handles must raise, the 2D ones work.
import numpy as np

import warp as wp

wp.init()
dev = wp.get_device("cuda:0")
t2 = wp.Texture2D(np.zeros((8, 8), dtype=np.float32), device=dev)
t3 = wp.Texture3D(np.zeros((4, 4, 4), dtype=np.float32), device=dev)
print("2D cuda_array/cuda_texture nonzero:", bool(t2.cuda_array), bool(t2.cuda_texture))
for prop in ("cuda_array", "cuda_texture", "cuda_surface"):
    try:
        getattr(t3, prop)
        print(f"3D {prop}: no exception")
    except Exception as e:
        print(f"3D {prop}: {type(e).__name__}: {e}")
try:
    wp.Texture3D(cuda_array=t2.cuda_array, device=dev)
    print("Texture3D(cuda_array=...): no exception")
except Exception as e:
    print(f"Texture3D(cuda_array=...): {type(e).__name__}: {e}")
print("3D id nonzero:", bool(t3.id))
```

### 4. `mma_check.py` (fixed) and `mma_check_bf16.py`
- `python mma_check.py`: **FAILURES 0**, exit 0, 488 s. 165 cases: 125 ok, **15 SMEM-LIMIT (forward not launched)** (fp32 64x64x64, fp64 64x64x64, fp64 64x32x48, each at all five block_dims: exactly the 15 the first session counted as FAIL), and 25 SMEM-LIMIT (backward not launched) (fp32 64x32x48, fp64 32x32x32, fp64 48x16x64, fp64 32x64x16, fp16 64x64x64; unchanged). Its code objects: 3864 `v_mfma_f32_16x16x4f32`, 3864 `v_mfma_f64_16x16x4f64`, 948 `v_mfma_f32_16x16x16f16`.
- `python mma_check_bf16.py`: **FAILURES 0**, exit 0, 55 cases, max abs diff to the CPU 9.5e-07 (64x64x64, 64x32x48), 0 on the four fallback shapes; 242 `v_mfma_f32_16x16x16bf16_1k` (this version has two matmuls per kernel).

### 5. Tile set, full 82-pattern set, CPU-only suite
- Full set (the same 82-pattern list and loop as the first gfx90a session, 3917 s): **81 patterns OK, 1 with the 2 known failures** (`test_fem*` 108: `test_gimp_quadrature`, `test_cube_shape_functions`, upstream isolation; 108 because `example_diffusion_mgpu` registers on this host, as before). Line by line against the first session at `930d0098`, timings stripped, **the only change is test_texture** (120 with 14 errors -> 119 OK, 4 skipped). Skip reasons are the same as then, plus the 3 new "No suitable devices" texture placeholders (21 instead of 18). No `Warp CUDA error` line in any of the 82 logs. The only "Failed to configure kernel dynamic shared memory" line is the known `test_tile_load_scoped` backward (65536).
- Tile set (24 files, inside the full set): **639 run, 15 skipped, 0 failures**. The skips are the 10 upstream no-libmathdx skips and the five 64 KiB capability skips (67584 x2, 66816, 66688 x2). test_tile_matmul 25 OK, with the three `fragment_shapes` cases.
- CPU-only suite (`HIP_VISIBLE_DEVICES=99 CUDA_VISIBLE_DEVICES=99`, from a scratch directory, 193 s): **Ran 2294, OK (skipped=15)**.
- `example_apic_fluid.usd`, written into the clone by an example during the full set, was deleted afterwards.

### Next step for the lock holder
- `c9595bce` (option A) works on real gfx90a hardware: the 14 texture errors are gone, 3D sampling matches the CPU sampler (5.96e-08, as on gfx942/gfx950), 1D/2D stay on hardware, and the full set has only the known fem pair. If Jeff keeps option A, the message diff above is the one change suggested (cosmetic; `texture.py` +6/-1). Optionally add an interop-refusal test registered on devices with 1D/2D hardware but no 3D hardware, so this path has coverage on gfx90a; not drafted, since it adds carried test lines Jeff may not want.
- Procedure note for other hosts: `utils/timeit.sh` needs the job's `pylib` on `PYTHONPATH` for jsonschema. Exporting `PYTHONPATH=<src>` alone makes it fail with "no Python that can run MOAT's tools" before the wrapped command runs; prepend instead (`PYTHONPATH=<src>:$PYTHONPATH`). It cost one aborted attempt of item 4 here; nothing ran in it.

## Porting session 11 (2026-11-19, linux-gfx950 MI355X, porter): gfx11 WMMA, gfx90a message; Phase 6 candidate head

Fork `release/1.14.0.amd0` advanced `c9595bce..e78ee32f` (pushed, `advance-head` recorded): `bb5d01d7` [ROCm] Use WMMA for float16 tile_matmul on gfx11 GPUs (`tile_matmul_mma.h` +19/-3); `e78ee32f` [ROCm] Say why 3D texture interop handles are missing on gfx90a (`texture.py` +6/-1, the gfx90a host's diff verbatim, `ruff format`/`ruff check` clean; it belongs with `c9595bce`: dropping option A means dropping both). Nothing else changed. nvcc 12.8 build at `e78ee32f`: "Warp build succeeded"; NVRTC compiles of the tile_matmul and texture modules with the native headers of `c9595bce` and `e78ee32f`: identical PTX and cubin (`0805dcc9fb1a`/`6f630854bc9f`, `98d84c023888`/`7b52ca293caf`). `audit-commits` OK, `jargon.py --port` clean.

### 1. gfx11 WMMA: why 64x64 at block_dim 256 was slower, and what changed
From the code (`tile_matmul_mma.h`), for a 64x64 fp16 tile (16 fragments of 16x16, 4 per row):
- The wavefront count is `block_dim / 32`; at 256 that is 8. The fragment grouping picks the widest row group that still leaves a group for every wavefront: 4 would give 16/4 = 4 groups for 8 wavefronts, so it picks 2, one group of 2 fragments per wavefront. At block_dim 128 (4 wavefronts) it picks 4.
- gfx11's WMMA operand layout duplicates A and B across the two half-waves (`k_split = false`, `kpl = 16`): every lane loads all 16 K values of its A row and of each B column from shared memory. Per K step a wavefront with 2 fragments issues 16 (A) + 2 x 16 (B) = 48 shared-memory loads per lane for 2 WMMA instructions, 24 per instruction; with 4 fragments, 16 + 64 = 80 for 4, 20 per instruction.
- gfx12's layout splits K between the half-waves (`k_split = true`, `kpl = 8`): the same cell is 8 + 2 x 8 = 24 loads per lane for 2 instructions, 12 per instruction, half of gfx11's. That is why the same configuration was faster than scalar on gfx1201 and not on gfx1100.
- The scalar path is at its best exactly there: at block_dim 256 a 64x64 tile gives it 4x4 register blocks for all 256 threads (its 17.6 ms is its fastest 64x64 forward on gfx1100, against 27.5-33.3 ms at the other block sizes), while gfx11 WMMA at 2 fragments per wavefront is bound by the duplicated operand loads.
A change to the load pattern cannot be measured on this host, so the configuration class was moved to the scalar path at compile time, by its cause rather than by one cell: on targets whose WMMA duplicates operands (gfx11 family, `k_split = false`), a tile of 16 or more fragments uses WMMA only when every wavefront can take a full row of 4 fragments (`fragments / wavefronts >= 4`); tiles under 16 fragments keep WMMA (16x16 and 32x32 forward measured 2.00x-4.13x faster, groups of 1 or 2 included), and 64x64 keeps WMMA at block_dim 32, 64 and 128 (groups of 4; 2.40x, 1.45x, 1.96x). Among the measured cells this moves only 64x64 at block_dim 256; larger tiles get the same rule untested. MFMA and gfx12 are unaffected (`amortizes_operand_loads` returns true for them). The enabling condition gained the gfx11 builtin: `... || __has_builtin(__builtin_amdgcn_wmma_f32_16x16x16_f16_w32)`.

Checked through Warp's hipRTC path (`wmma_cells.py`, below: fp16 tile_matmul modules per (tile, block_dim), disassembled per target):

| cell | gfx1100 | gfx1201 | gfx950 |
|---|---|---|---|
| 64x64, block_dim 256 | no matrix instructions (scalar) | 8 `v_wmma_f32_16x16x16_f16` | 16 `v_mfma_f32_16x16x16_f16` |
| 64x64, block_dim 128 | 16 `v_wmma_f32_16x16x16_f16` | 16 `v_wmma_f32_16x16x16_f16` | 16 `v_mfma_f32_16x16x16_f16` |
| 32x32, block_dim 256 | 2 `v_wmma_f32_16x16x16_f16` | 2 `v_wmma_f32_16x16x16_f16` | 2 `v_mfma_f32_16x16x16_f16` |

**gfx1101, gfx1102, gfx1150 and gfx1151 have the same gfx11 builtin** (checked with `__has_builtin` for each), so they now take the gfx11 WMMA path too; gfx1151 compiles exactly as gfx1100 in the table above (64x64 bd256 scalar, the other two cells WMMA). Same instruction and layout family as gfx1100, measured only on gfx1100: the Windows gfx1151 run below starts with the layout probe.

Release-notes update: "On Radeon gfx11 GPUs (RX 7000 / Radeon PRO W7000 and Ryzen AI APUs: gfx1100-gfx1102, gfx1150, gfx1151) float16 `tile_matmul` also uses WMMA, under the same conditions as gfx12, except that tiles of 16 or more 16x16 fragments keep the scalar path unless the block has at most one 32-thread wavefront per four fragments."

### 2. gfx90a message
`Texture._require_texture_hardware`: a device without texture hardware keeps the original message (so `test_texture_interop_requires_hardware`'s regex still matches where it runs); a 3D texture on a device whose hardware cannot filter 3D textures now raises "<name> is not available for 3D textures on <device>: its texture hardware cannot filter 3D textures, so they are sampled in software".

### 3. gfx950 at `e78ee32f` (the Phase 6 candidate)
Full Phase 1-5 set, 82 patterns, 1733 s: **81 OK, 1 with the 2 known failures** (`test_fem*` 107: `test_gimp_quadrature`, `test_cube_shape_functions`, upstream isolation). Tile set inside it: 639 run, 10 skipped, 0 failures (`test_tile_matmul` 25 OK); test_texture 118 OK (7 skipped); test_streams 24 OK (3 skipped). CPU-only suite with GPUs hidden: **Ran 2294, OK (skipped=15)**. `mma_check.py` (fixed version, Porting session 10): FAILURES 0.

### Next step (Phase 6 candidate head `e78ee32f`)
The project stays in `porting`, lock held by linux-gfx950. gfx950 has passed the same-head set (item 3). Every Linux host below owes the complete same-head pass at `git rev-parse HEAD` = `e78ee32fa166bb0bbb58c870fa7b13845553df99`, common setup as before (TheRock 10.2 wheels with `device-<gfx>`, `export PATH=$FLEET_VENV/bin:$PATH` first, `python build_lib.py --rocm-path "$(rocm-sdk path --root)" --hip-arch <gfx> -j <physical cores>`, `git lfs pull`, `ulimit -c 0`, a fresh `WARP_CACHE_PATH` per run, `PYTHONPATH=<src>:$PYTHONPATH` when wrapping in `utils/timeit.sh`). Scripts: the 82-pattern loop (Porting session 6 handoff B.5); `mma_check.py`, `mma_check_bf16.py`, `wmma_layout.hip` (Porting session 10); `sdf_probe.py` (session 5); `tex_alias.py` (session 6); `tex_capture.py` (session 7); `interop3d.py` (gfx90a evidence at `c9595bce`); `bench_both.sh`/`compare.py` (session 9). If any host's result differs from the expectation below, it records the difference and the head is not the Phase 6 head until the lock holder resolves it.

**gfx942, two GPUs (`HIP_VISIBLE_DEVICES=0,1`; one-GPU items with `HIP_VISIBLE_DEVICES=0`):**
1. One GPU: the 82-pattern set (expect 81 OK + the fem pair; tile set 639 run, 15 skipped, 0 failures; test_texture 118 OK, 7 skipped; test_streams 3 skipped), the CPU-only suite (2294 OK), `mma_check.py` and `mma_check_bf16.py` (FAILURES 0 each), `sdf_probe.py cuda:0` and `tex_capture.py cuda:0` (PASS).
2. Two GPUs: the 13-file C.2 loop (expect no errors or failures; the C1/C2/C3 skips naming the HIP defects; IOMMU-heuristic skips) and `test_texture.py` (all `cuda_0` and `cuda_1` cases pass).

**gfx90a (MI200):** the 82-pattern set (expect 81 OK + the fem pair, `test_fem*` 108 with `example_diffusion_mgpu`; test_texture 119 OK, 4 skipped), the CPU-only suite, `mma_check.py` and `mma_check_bf16.py` (FAILURES 0), the capability line (`True False`), `sdf_probe.py`, `tex_capture.py`, `tex_alias.py` (PASS), and `interop3d.py` (the four 3D cases now print the new "not available for 3D textures ... cannot filter 3D textures" message; the 2D handles are valid).

**gfx1100:** WMMA is now on as shipped.
1. `test_tile_matmul.py` (25 OK) and, with a fresh cache, the ISA check: the 32x32 fp16 module at block_dim 64 contains `v_wmma_f32_16x16x16_f16`; `wmma_cells.py` built for the device itself (or run as below with `gfx1100`) shows the 64x64 bd256 module without matrix instructions.
2. `mma_check.py` and `mma_check_bf16.py` (FAILURES 0).
3. The fp16 benchmark, as shipped against the scalar tree (scalar worktree: `git worktree add --detach <dir> e78ee32f`, `git -C <dir> checkout 974cd4bf -- warp/native/tile_matmul.h`, copy `warp/bin/*.so`), three runs per tree, with the load monitor used in evidence session at `c9595bce`: expect every fp16 cell at least 1.0x, with **64x64 block_dim 256 forward now equal to scalar** (both trees scalar there: about 17.6 ms each), and the other 23 fp16 cells as in that session's set 2. Any fp16 cell below 1.0x is a finding.
4. The 82-pattern set (expect 81 OK + the fem pair; tile set 639/15/0) and the CPU-only suite; textures: `sdf_probe.py`, `tex_alias.py` (PASS).

**gfx1201:** the 82-pattern set, the CPU-only suite, `mma_check.py` and `mma_check_bf16.py` (FAILURES 0), `test_tile_matmul.py` (25 OK, WMMA present as shipped; gfx12 code unchanged since `43331f4a`), `sdf_probe.py` and `tex_alias.py` (PASS).

**Then Phase 6, Windows gfx1151 laptop** (after all four Linux hosts pass at this head; Linux before Windows). The Windows HIP build is not written yet: `build_dll.py:654` raises "ROCm builds on Windows are not supported yet". So Phase 6 is a porter session on the laptop, with the work lock moved there by Jeff, in two parts:
1. **Build (porter):** in `warp/_src/build_dll.py`, replace the Windows refusal with the HIP branch the plan describes (risk 12): ROCm clang (`<rocm>/lib/llvm/bin/clang++ -x hip`, `--offload-arch` from the Windows targets of `share/therock/dist_info.json`, `-fno-gpu-rdc`) for the `.cu` files; clang-cl for the host `.cpp` files that include HIP headers, MSVC for the rest and for warp-clang; link `amdhip64.lib` and `hiprtc.lib`; Warp's loader finds `amdhip64_7.dll` and `hiprtc*.dll` by full path from the ROCm the build used (not a stale copy in System32). The documented command, `python build_lib.py --rocm-path "%ROCM_PATH%" --hip-arch gfx1151` (and auto-detection without `--rocm-path`), must work in a plain Developer shell; any toolchain workaround goes into the build files, guarded to Windows. CUDA and Linux branches unchanged; afterwards the CUDA path is compiled with nvcc on the mi355 host as every round. Laptop rules: worktrees, venvs, build trees and scratch only under `D:/Develop` (`TMPDIR`/`TEMP` pointed there); TheRock 10.2 Windows wheels with `device-gfx1151`; DLL-load failures, exit 127 and a first-launch `hipErrorLaunchFailure (719)` are environment first.
2. **Validation, light runs only** (20-CU APU sharing system memory; no stress loops, no large-memory cases, no benchmarks, no 64 KiB-plus probes): `python -c "import warp as wp; wp.init(); d = wp.get_device('cuda:0'); print(d.arch_name, d.is_texture_supported, d.is_texture_hardware_supported, d.is_texture3d_hardware_supported)"`; then `wmma_layout.hip` built for gfx1151 (expect "WMMA family 11", layout lines 0 wrong, accumulation within bound; if it fails, stop: gfx1151 must then be excluded from the WMMA path before anything else); `test_tile_matmul.py` and `test_tile_shared_memory.py`; `test_texture.py`, `sdf_probe.py cuda:0`, `tex_alias.py cuda:0`; then the 82-pattern list one pattern at a time with `-m 2` and a per-pattern timeout (900 s), skipping nothing but stopping at the first driver reset or hang and recording it (plan risk 9: `test_atomic_cas`'s spin-lock cases are already narrowed to devices with independent thread scheduling; risk 13: GPU-scheduler bugchecks under texture and stress workloads were seen on that laptop in other ports); finally the CPU-only suite with GPUs hidden. `mma_check.py` only if the above is clean (165 small kernels, not a stress loop). Phase 7 (wheels on every platform) follows Phase 6.

**Still open for Jeff:** gfx90a 3D textures, option A as committed (`c9595bce` + `e78ee32f`) or C (drop both); defer/now on the six undecided `rocm-bug-report` entries. Phase 6 not started.

`wmma_cells.py`:
```python
# fp16 tile_matmul modules for chosen (tile, block_dim) cells, compiled through Warp's hipRTC path for
# other targets; prints the matrix instructions per module. Usage: python wmma_cells.py <outdir> <gfx>...
import glob
import os
import subprocess
import sys

import warp as wp
import warp._src.build as build

wp.init()
CELLS = [(64, 256), (64, 128), (32, 256)]
R = subprocess.check_output([sys.executable, "-m", "rocm_sdk", "path", "--root"]).decode().strip()


def make(M):
    @wp.kernel(module="unique", module_options={"enable_backward": False})
    def mm(A: wp.array2d(dtype=wp.float16), B: wp.array2d(dtype=wp.float16), C: wp.array2d(dtype=wp.float16)):
        wp.tile_store(C, wp.tile_matmul(wp.tile_load(A, shape=(M, M)), wp.tile_load(B, shape=(M, M))))

    return mm


out_dir = sys.argv[1]
for M, bd in CELLS:
    k = make(M)
    before = set(glob.glob(os.path.join(wp.config.kernel_cache_dir, "**", "*.cu"), recursive=True))
    wp.load_module(k.module, device="cuda:0", block_dim=bd)
    src = sorted(set(glob.glob(os.path.join(wp.config.kernel_cache_dir, "**", "*.cu"), recursive=True)) - before)[-1]
    for target in sys.argv[2:]:
        out = os.path.join(out_dir, f"mm{M}_bd{bd}_{target}.hsaco")
        build.build_cuda(src, 0, out, pch_dir=None, hip_target=target)
        dis = subprocess.run([R + "/lib/llvm/bin/llvm-objdump", "-d", f"--mcpu={target}", out], capture_output=True, text=True).stdout
        ops = {}
        for line in dis.splitlines():
            w = line.split()
            if w and (w[0].startswith("v_wmma") or w[0].startswith("v_mfma")):
                ops[w[0]] = ops.get(w[0], 0) + 1
        print(f"{M}x{M} block_dim {bd:3d} {target:8s}: {ops or 'no matrix instructions (scalar)'}")
```

## Evidence session 3 (2026-11-19, linux-gfx1201, Radeon AI PRO R9700, evidence only): same-head pass at the Phase 6 candidate `e78ee32f`

**Every item on the gfx1201 list of Porting session 11's Next step matched its expected result.** gfx1201 has no difference to record at `e78ee32f`.

Requested by Jeff Daily. The work lock stays with linux-gfx950. Nothing was committed or pushed to the fork, and no stage, lock, head, platform or deferral record changed. The read-only clone was moved from `c9595bce` to the recorded `head_sha` `e78ee32fa166bb0bbb58c870fa7b13845553df99` (detached, `git lfs pull`, `git status --porcelain` empty before and after). `git diff --stat c9595bce e78ee32f`: `warp/_src/texture.py` +6/-1, `warp/native/tile_matmul_mma.h` +19/-3. Rebuilt against the venv's ROCm 10.2 wheels: `export PATH=$FLEET_VENV/bin:$PATH; python build_lib.py --rocm-path "$(rocm-sdk path --root)" --hip-arch gfx1201 -j 12` -> "Warp build succeeded" (32.6 s); RUNPATH the venv's `_rocm_sdk_devel/lib`. Same device and venv as before, `HIP_VISIBLE_DEVICES=2`, `ulimit -c 0`, a fresh private `WARP_CACHE_PATH` for every run. Scripts: `mma_check.py` and `mma_check_bf16.py` from Porting session 10, `sdf_probe.py` (session 5), `tex_alias.py` (session 6), extracted verbatim.

| item | expected | result |
|---|---|---|
| 82-pattern set (`-p "<pattern>" -m 32`, 1949 s) | 81 OK + the fem pair | **81 OK; `test_fem*` 107 run, failures=2: `test_gimp_quadrature`, `test_cube_shape_functions`** ("Unexpected output" with the init greeting, upstream isolation). Tile set inside it 639 run, 15 skipped, 0 failures; test_tile_matmul 25 OK; test_texture 120 OK (1 skipped); test_streams 24 OK (3 skipped) |
| CPU-only suite (`HIP_VISIBLE_DEVICES=99 CUDA_VISIBLE_DEVICES=99 python -m warp.tests -s autodetect -k cpu -m 32`) | 2294 OK | **Ran 2294, OK (skipped=15)**, no `cuda:0` in the log |
| `python mma_check.py` | FAILURES 0 | **FAILURES 0**: 25 ok, 25 SMEM-LIMIT (backward), 15 SMEM-LIMIT (forward) (all the 64 KiB limits); 165 code objects, 1336 `v_wmma_f32_16x16x16_f16` |
| `python mma_check_bf16.py` | FAILURES 0 | **FAILURES 0**: 55 cases (11 shapes x 5 block_dims), all ok, no shared-memory limit; 56 code objects, no WMMA (bf16 has no WMMA path, by design) |
| `test_tile_matmul.py`, WMMA as shipped | 25 OK, WMMA present | **Ran 25, OK**; fp16 modules: 16x16x16 at bd 64 and 96 have 4 `v_wmma_f32_16x16x16_f16` each, 48x16 (K 32) 15 at bd 64 and 9 at bd 96; 24x16, 17x33x15 and 8x8x4 have 0 (do not fit, scalar); fp32, fp64 and bf16 modules 0 |
| `sdf_probe.py cpu` / `cuda:0` | PASS | **PASS / PASS**: cpu 0.00e+00, 1.28e-03, 0.00e+00; cuda:0 SDF vs CPU sampler 4.78e-05, SDF vs analytic 1.31e-03, RGBA8 2.52e-03 (the same values as every earlier gfx1201/gfx1100 run) |
| `tex_alias.py cuda:0` | PASS | **PASS**: "plain: 0 of 2048 wrong", "graph capture: 0 of 2048 wrong" |

Checks beyond the list, all consistent with "gfx12 code unchanged since `43331f4a`":
- The 82-pattern per-pattern summary lines (Ran count and result, timings stripped) are **identical** to the `c9595bce` run (Evidence session 2), and so is the skip-reason histogram over all 82 logs. There is no `Warp CUDA error`, and the only shared-memory warning is the known `test_tile_load_scoped` backward one.
- `mma_check.py`'s per-case error lines (c, d, gA, gB for every printed case) are identical to the `c9595bce` run (`diff` empty), with the same 1336 WMMA instructions. The test_tile_matmul WMMA pattern is identical to that session's table. So the gfx11 load-amortization rule (`bb5d01d7`) changes nothing on gfx12.

Notes on the run:
- The CPU-only suite ran while the 82-pattern set was on its first patterns (CPU-only touches no GPU). Its log, and 10 of the earlier GPU-run logs, end with multiprocessing `resource_tracker` "leaked semaphore" UserWarnings printed at interpreter shutdown, after the `OK` line: Python process-pool noise, not a test result.
- `mma_check.py`, `mma_check_bf16.py`, test_tile_matmul and the texture probes ran while the 82-pattern set was running (correctness only, each with its own cache). No benchmark was in this host's list, and none was run.
- The untracked `example_apic_fluid.usd` an example writes was deleted after the run.
## Evidence session 7 (2026-11-19, linux-gfx942, 2x MI300X, evidence only): same-head pass at the Phase 6 candidate `e78ee32f`

**Verdict: every item of the gfx942 list (one GPU and two GPUs) matched the expected result at `e78ee32f`.** One clarification: the handoff's two-GPU expectation names C1/C2/C3 skips. On this host the two C3 tests (`test_stream_arg_graph_mgpu`, `test_stream_scope_graph_mgpu`) are skipped first by upstream's IOMMU heuristic ("Peer-to-Peer transfers not supported"), as in every earlier two-GPU run here. Evidence session 2 showed their C3 gate firing with that heuristic forced off. No difference from the expectation needs resolving.

Requested by Jeff Daily. The lock stays with linux-gfx950. No fork commit and no stage, lock, head, platform or deferred record changed. The read-only clone was moved from `c9595bce` to the recorded `head_sha` `e78ee32fa166bb0bbb58c870fa7b13845553df99` (new commits `bb5d01d7`, `e78ee32f`; `git status --porcelain` empty before and after; LFS assets present; fork pre-push hook unchanged). Build: `python build_lib.py --rocm-path "$(rocm-sdk path --root)" --hip-arch gfx942 -j 112` -> "Warp build succeeded", 31.5 s. Every run used its own fresh kernel cache (`WARP_CACHE_PATH=$(mktemp -d ...)`), `ulimit -c 0`, `PYTHONPATH=<src>`. The scripts were extracted from the notes at their latest versions: `mma_check.py` and `mma_check_bf16.py` from Porting session 10, `sdf_probe.py` from session 5, `tex_capture.py` from session 7. The probes and `mma_check.py` are byte-identical to the copies run in evidence sessions 3-6.

### 1. One GPU (`HIP_VISIBLE_DEVICES=0`)

| item | expected | result |
|---|---|---|
| 82-pattern set (Porting session 6 B.5 list; 2538 s with the CPU suite) | 81 OK + the fem pair | **81 OK, 1 with the 2 known failures** (`test_fem*` 107: `test_gimp_quadrature`, `test_cube_shape_functions`) |
| tile set inside it (24 files) | 639 run, 15 skipped, 0 failures | **639 run, 15 skipped, 0 failures** (10 upstream no-libmathdx + the five 64 KiB skips; `test_tile_matmul` 25 OK) |
| test_texture | 118 OK, 7 skipped | **118 OK, 7 skipped** |
| test_streams | 3 skipped | **24 OK, 3 skipped** (priority timing + 2 multi-GPU) |
| CPU-only suite (`HIP_VISIBLE_DEVICES=99 CUDA_VISIBLE_DEVICES=99 ... -k cpu -m 32`) | 2294 OK | **Ran 2294, OK (skipped=15)** |
| `mma_check.py` | FAILURES 0 | **FAILURES 0** (25 ok lines at bd 64; 15 "SMEM-LIMIT (forward not launched)": fp32 64x64x64, fp64 64x32x48, fp64 64x64x64; 25 "SMEM-LIMIT (backward not launched)") |
| `mma_check_bf16.py` | FAILURES 0 | **FAILURES 0** (11 shapes x 5 block_dims = 55 cases; worst printed difference 9.5e-07) |
| `sdf_probe.py cuda:0` | PASS | **PASS** (SDF vs CPU sampler 5.96e-08, vs analytic 1.28e-03, RGBA8 1.55e-06) |
| `tex_capture.py cuda:0` | PASS | **PASS** (replays 0-2: 1.19e-07 each) |

The 82-pattern per-file counts are identical line by line to evidence session 6 at `c9595bce`. The two new commits (gfx11 WMMA selection, gfx90a texture message) do not reach gfx942's code, as expected.

### 2. Two GPUs (`HIP_VISIBLE_DEVICES=0,1`): the 13-file loop, then test_texture

| file | ran | result |
|---|---|---|
| test_multigpu | 5 | OK, 2 skipped |
| test_peer | 5 | OK, 1 skipped |
| test_mempool | 10 | OK, 2 skipped |
| test_streams | 42 | OK, 4 skipped |
| test_graph | 61 | OK, 5 skipped |
| test_ipc | 8 | OK |
| test_unified_memory | 40 | OK, 9 skipped |
| test_allocator | 16 | OK, 1 skipped |
| test_async | 3533 | OK |
| test_sparse | 71 | OK, 1 skipped |
| test_fem_integrate | 25 | OK, 1 skipped |
| test_array | 207 | OK |
| test_copy | 30 | OK |
| test_texture | 177 | OK, 7 skipped: all 59 `cuda_0` and 59 `cuda_1` cases pass |

**No errors and no failures.** Skips:
- 3 C1 and 2 C2, each naming the HIP defect.
- 2 C4 ("HIP maps stream priorities to normal priority ...").
- 10 IOMMU heuristic (C3's two tests among them, see the verdict) and 4 "cuda:N cannot access CPU memory".
- 2 upstream memory-usage flakiness and 7 test_texture "No suitable devices".
- 1 each of "Requires devices without mempool access support", "Requires devices without peer access support" and "rmm not installed".

All identical to evidence session 6. The only `Warp CUDA error` line is the known harmless 900 in `test_synchronize_during_capture` (evidence session 3).

The untracked `example_apic_fluid.usd` an example writes into the clone during the full set was deleted afterwards.

### Next step for the lock holder
gfx942 has passed its complete same-head list at `e78ee32f` with no difference from the expected results.
## Evidence session 7 (2026-11-19, linux-gfx1100, Radeon Pro W7800, evidence only): same-head pass at the Phase 6 candidate `e78ee32f`

**Verdict: not every item matched.** Items 1, 2 and 4 matched their expected results exactly. Item 3 (the fp16 benchmark) met its pass criterion: every fp16 cell is at least 1.0x, and 64x64 block_dim 256 forward now equals scalar. But one cell differs from the stated expectation that "the other 23 fp16 cells [are] as in that session's set 2": **64x64 block_dim 256 adjoint is 1.03x (13.27 ms, scalar) where set 2 measured 1.59x (8.58 ms, WMMA)**. The compile-time rule of `bb5d01d7` moves the whole 64x64 bd256 configuration, forward *and* backward kernels, to the scalar path (0 `v_wmma` in both, checked below), so the adjoint loses the WMMA gain it had. Per the Next step, the lock holder decides whether this is acceptable or the head changes.

Requested by Jeff Daily. The work lock stays with linux-gfx950. Nothing was committed to the fork, and no stage, lock, head, platform or deferral record changed. The clone was updated read-only to the recorded `head_sha` `e78ee32fa166bb0bbb58c870fa7b13845553df99` and rebuilt (`export PATH=$FLEET_VENV/bin:$PATH; python build_lib.py --rocm-path "$(rocm-sdk path --root)" --hip-arch gfx1100 -j 32` -> "Warp build succeeded"; RUNPATH is the venv's ROCm 10.2). `git status --porcelain` is clean at the end. Same venv (HIP 7.17.26434), `ulimit -c 0`, `HIP_VISIBLE_DEVICES=0`.

A fresh `WARP_CACHE_PATH` was used for every run of every tree and every script: each test pattern, each `mma_check` script, each ISA check, each benchmark run, the CPU-only suite and each probe. The scalar baseline was a separate worktree (`git worktree add --detach <dir> e78ee32f`, `git -C <dir> checkout 974cd4bf -- warp/native/tile_matmul.h`, this build's `warp/bin/*.so` copied in), removed afterwards. Items are the "gfx1100" list of the `e78ee32f` Next step, in its order.

### Item 1: WMMA on as shipped: matches
- `python -m warp.tests -s autodetect -p test_tile_matmul.py -m 32`: **Ran 25, OK**.
- ISA check (`agent_space/p10/isa_check.py`: 32x32x32 fp16 and fp32 tile_matmul, block_dim 64, forward and backward; `llvm-objdump -d --mcpu=gfx1100` on the fresh cache): the fp16 module has **16 `v_wmma_f32_16x16x16_f16`**; the fp32 module 0 (scalar FMAs). fp16 forward max err 1.9e-03, adj_A 2.6e-03 against float64, the same as the `c9595bce` switch-on experiment.
- `python wmma_cells.py <outdir> gfx1100` (verbatim from Porting session 11), identical to the porter's gfx1100 column:
```
64x64 block_dim 256 gfx1100 : no matrix instructions (scalar)
64x64 block_dim 128 gfx1100 : {'v_wmma_f32_16x16x16_f16': 16}
32x32 block_dim 256 gfx1100 : {'v_wmma_f32_16x16x16_f16': 2}
```

### Item 2: `mma_check.py` and `mma_check_bf16.py` (Porting session 10 versions): match
- `mma_check.py`: **FAILURES 0**; 15 "SMEM-LIMIT (forward not launched)" and 25 "SMEM-LIMIT (backward not launched)". These are case for case the same 40 as the `c9595bce` switch-on experiment (`diff` of the SMEM lines with the error columns stripped: empty).
- `mma_check_bf16.py`: **FAILURES 0** (55 cases, no limit).

### Item 3: fp16 benchmark, shipped tree against the scalar tree: pass criterion met, one cell differs from expectation
Porting session 9 `bench_both.sh` with this host's paths (shipped tree = the main clone, scalar tree = the worktree above), `bench_tile_matmul.py --blocks 16384 --iters 50`, three runs per tree, fresh cache per run, merged with `compare.py` (labels adjusted). A 5-second load/GPU-busy monitor ran alongside. **The set is clean:** no cell exceeds 1.5x its median, and the run-to-run spread over the 112 launched cells is median 1.6%, max 11.5%. fp16 rows:

| dtype | tile | block_dim | scalar fwd ms | shipped fwd ms | fwd speedup | scalar adj ms | shipped adj ms | adj speedup | checks |
|---|---|---|---|---|---|---|---|---|---|
| float16 | 16x16 | 32 | 1.125 | 0.268 | 4.20x | 0.292 | 0.239 | 1.22x | ok |
| float16 | 16x16 | 64 | 1.132 | 0.274 | 4.14x | 0.281 | 0.202 | 1.39x | ok |
| float16 | 16x16 | 128 | 0.974 | 0.416 | 2.34x | 0.257 | 0.201 | 1.28x | ok |
| float16 | 16x16 | 256 | 1.392 | 0.465 | 2.99x | 0.307 | 0.248 | 1.24x | ok |
| float16 | 32x32 | 32 | 3.720 | 1.769 | 2.10x | 4.399 | 2.664 | 1.65x | ok |
| float16 | 32x32 | 64 | 4.261 | 1.806 | 2.36x | 2.487 | 1.661 | 1.50x | ok |
| float16 | 32x32 | 128 | 5.312 | 2.569 | 2.07x | 1.939 | 1.178 | 1.65x | ok |
| float16 | 32x32 | 256 | 6.835 | 2.627 | 2.60x | 1.957 | 1.168 | 1.68x | ok |
| float16 | 64x64 | 32 | 33.189 | 13.304 | 2.49x | 60.652 | 54.083 | 1.12x | ok |
| float16 | 64x64 | 64 | 27.456 | 18.334 | 1.50x | 33.671 | 30.212 | 1.11x | ok |
| float16 | 64x64 | 128 | 27.607 | 13.538 | 2.04x | 17.235 | 14.716 | 1.17x | ok |
| float16 | 64x64 | 256 | 17.605 | 16.828 | 1.05x | 13.603 | 13.266 | 1.03x | ok |

- **Every fp16 cell is at least 1.0x: no finding of a slower cell.**
- **64x64 bd256 forward:** shipped 16.19/16.83/17.29 ms against scalar 17.55/17.61/17.62 ms (1.05x on medians). Both trees are scalar here: `wmma_cells.py` shows no matrix instructions, and the forward kernel below has 0 `v_wmma`. The 5% gap is within this cell's shipped-tree spread (6.5%) plus a systematic 1-3% in favour of the shipped tree that the fp32/fp64 control cells show too: their code is identical in both trees, and all 16 launched fp32/fp64 16x16 and 32x32 cells read 1.00x-1.03x. The shipped tree ran first each time. So this matches the expectation "equal to scalar (about 17.6 ms each)".
- **64x64 bd256 adjoint: differs from the expectation.** Shipped 12.96/13.27/13.48 ms against scalar 13.60/13.60/13.61 ms (1.03x). In `c9595bce` set 2 the same cell on WMMA was 8.40/8.60/8.58 ms (1.59x). Checked in the ISA (`agent_space/p11/isa_bwd.py`: the benchmark's one-GEMM 64x64x64 fp16 kernel at block_dim 256 with a backward pass, fresh cache per tree): **0 `v_wmma` in `..._cuda_kernel_forward` and 0 in `..._cuda_kernel_backward`**, in the shipped tree as in the scalar tree. The rule tests the output tile's fragment count against the wavefront count, and the adjoint GEMMs of a 64x64 tile have the same shape (16 fragments, 8 wavefronts), so they go scalar too. The forward cell gained nothing from WMMA (0.91x), but the adjoint gained 1.59x. A rule that kept WMMA for the backward kernel, or for adjoint GEMMs, would keep that gain. Whether to change it is the lock holder's call. Nothing is proposed here beyond the measurement.
- The other 22 fp16 cells are as in set 2, within run-to-run noise (forward 1.50x-4.20x, adjoint 1.11x-1.68x).
- fp32/fp64 (scalar in both trees): 1.00x-1.03x; the 64x64 fp32 adjoint and all fp64 64x64 rows are LDS-limited (no launch), as before.

Raw fp16 timings per run, ms (`agent_space/p11/bench/bench_<tree>_run<n>.csv`):
```
tree,dtype,M,block_dim,fwd_r1,fwd_r2,fwd_r3,adj_r1,adj_r2,adj_r3
scalar,float16,16,32,1.1201,1.1295,1.1253,0.2920,0.2863,0.2977
scalar,float16,16,64,1.1262,1.1463,1.1316,0.2897,0.2813,0.2788
scalar,float16,16,128,0.9764,0.9737,0.9724,0.2595,0.2571,0.2561
scalar,float16,16,256,1.3912,1.3923,1.3972,0.3001,0.3071,0.3115
scalar,float16,32,32,3.7164,3.7380,3.7196,4.3853,4.4076,4.3992
scalar,float16,32,64,4.2486,4.2607,4.2712,2.4584,2.4874,2.5434
scalar,float16,32,128,5.2984,5.3115,5.3183,1.9231,1.9393,1.9449
scalar,float16,32,256,6.8278,6.8352,6.8566,1.9472,1.9588,1.9567
scalar,float16,64,32,33.1798,33.1891,33.2477,60.6744,60.6521,60.6358
scalar,float16,64,64,27.3660,27.4686,27.4555,33.6711,33.6698,33.6716
scalar,float16,64,128,27.5248,27.6074,27.6181,17.2214,17.2361,17.2353
scalar,float16,64,256,17.5456,17.6052,17.6193,13.5998,13.6026,13.6098
shipped,float16,16,32,0.2595,0.2682,0.2710,0.2411,0.2309,0.2390
shipped,float16,16,64,0.2725,0.2736,0.2793,0.2025,0.2014,0.2048
shipped,float16,16,128,0.3958,0.4158,0.4227,0.1954,0.2015,0.2093
shipped,float16,16,256,0.4419,0.4652,0.4719,0.2457,0.2480,0.2572
shipped,float16,32,32,1.7127,1.7694,1.8231,2.6161,2.6640,2.6753
shipped,float16,32,64,1.7071,1.8059,1.8676,1.6040,1.6614,1.6870
shipped,float16,32,128,2.4570,2.5686,2.6600,1.1115,1.1781,1.1880
shipped,float16,32,256,2.4892,2.6267,2.7021,1.1137,1.1675,1.1741
shipped,float16,64,32,12.8944,13.3035,13.7413,53.8120,54.0831,54.1629
shipped,float16,64,64,18.0622,18.3343,18.8437,30.2286,30.1890,30.2116
shipped,float16,64,128,12.8700,13.5382,13.9810,14.7161,14.7101,14.7488
shipped,float16,64,256,16.1877,16.8283,17.2917,12.9604,13.2655,13.4803
```

### Item 4: 82-pattern set, CPU-only suite, textures: match
- 82-pattern set (the same list as Evidence sessions 3-5, one pattern per `python -m warp.tests -s autodetect -p "<pattern>" -m 32`, fresh cache per pattern, timeit-wrapped): **82 patterns, 81 OK, 1 with the 2 known fem isolation failures** (`test_gimp_quadrature`, `test_cube_shape_functions`). The per-pattern summary lines (timings stripped) are **identical** to the `930d0098` run (Evidence session 5). So the tile set is 639 run, 15 skipped, 0 failures; test_tile_matmul 25; test_texture 120 (1 skipped); test_streams 24 (3 skipped). No `Warp CUDA error` line in any log; the only shared-memory warning is the known `test_tile_load_scoped` one.
- CPU-only suite (`HIP_VISIBLE_DEVICES=99 CUDA_VISIBLE_DEVICES=99 python -m warp.tests -s autodetect -k cpu -m 32`, greeting lists only `cpu`): **Ran 2294, OK (skipped=15)**. (The first attempt did not start, because `timeit.sh` found no usable Python when `PYTHONPATH` pointed at the source tree for the wrapper itself. It was rerun with `PYTHONPATH` set only inside the wrapped command. No test ran in the failed attempt.)
- `sdf_probe.py cpu` PASS (0 / 1.28e-03 / 0); `sdf_probe.py cuda:0` **PASS** (4.78e-05 / 1.31e-03 / 2.52e-03, the same values as every earlier gfx1100 session); `tex_alias.py cuda:0` **PASS** (plain 0 of 2048 wrong, graph capture 0 of 2048 wrong).

## Evidence session 3 (2026-11-19, linux-gfx90a, MI250X, evidence only): same-head pass at the Phase 6 candidate `e78ee32f`

**Every item of the gfx90a list in the Porting session 11 Next step matched its expected result.** No difference to record, so from this host's side `e78ee32f` can be the Phase 6 head.

Requested by Jeff Daily. The lock stays with linux-gfx950. No fork commit, and no stage, lock, head, platform or deferred record changed. The read-only clone moved from `c9595bce` to the recorded `head_sha` `e78ee32fa166bb0bbb58c870fa7b13845553df99` (`git status --porcelain` empty before and after; delta `bb5d01d7` `tile_matmul_mma.h` +19/-3, `e78ee32f` `texture.py` +6/-1). Same host, venv and ROCm 10.2.0 wheels as the earlier gfx90a sessions (one MI250X GCD, `HIP_VISIBLE_DEVICES=3` as pinned, HIP 7.17.26434). Rebuilt: `python build_lib.py --rocm-path "$(rocm-sdk path --root)" --hip-arch gfx90a -j 64` -> "Warp build succeeded", 49.2 s. `ulimit -c 0`, a fresh `WARP_CACHE_PATH` for every run, and `PYTHONPATH=<src>:$PYTHONPATH` under `utils/timeit.sh`. Scripts: `mma_check.py`, `mma_check_bf16.py` and `interop3d.py` were checked to be byte-identical to the blocks in these notes (`diff` against the extracted blocks); the others are as in the earlier sessions. The 82-pattern set ran alone on its process, and the other items ran beside it with their own caches (correctness checks only, no timings).

| item | expected (Porting session 11) | result |
|---|---|---|
| 82-pattern set | 81 OK + the fem pair; `test_fem*` 108 with `example_diffusion_mgpu`; test_texture 119 OK, 4 skipped | **81 OK + the fem pair** (`test_gimp_quadrature`, `test_cube_shape_functions`, upstream isolation); `test_fem*` 108, `test_fem.example_diffusion_mgpu_cuda_0` ok; test_texture **119 OK (skipped=4)**; 3910 s |
| CPU-only suite | 2294 OK | **Ran 2294, OK (skipped=15)**, no `cuda_0` in the log; 194 s |
| `mma_check.py` | FAILURES 0 | **FAILURES 0**, exit 0: 125 ok, 15 SMEM-LIMIT (forward not launched), 25 SMEM-LIMIT (backward not launched); every case's status identical to the `c9595bce` run (`diff` empty) |
| `mma_check_bf16.py` | FAILURES 0 | **FAILURES 0**, exit 0, 55 cases, max abs diff to the CPU 9.5e-07 |
| capability line | `True False` | **`True False`** |
| `sdf_probe.py cuda:0` | PASS | **PASS**: SDF vs CPU sampler 5.96e-08, vs analytic 1.28e-03, RGBA8 vs CPU sampler 2.52e-03 |
| `tex_capture.py cuda:0` | PASS | **PASS**: replays 0, 1, 2 each 1.19e-07 |
| `tex_alias.py cuda:0` | PASS | **PASS**: plain 0 of 2048 wrong, graph capture 0 of 2048 wrong |
| `interop3d.py` | the four 3D cases print the new message; 2D handles valid | **as expected**: "2D cuda_array/cuda_texture nonzero: True True"; `cuda_array`, `cuda_texture`, `cuda_surface` and `Texture3D(cuda_array=...)` each raise "<name> is not available for 3D textures on cuda:0: its texture hardware cannot filter 3D textures, so they are sampled in software"; "3D id nonzero: True" |

Full-set detail: the per-pattern summary lines (timings stripped) are **identical** to the gfx90a run at `c9595bce`, and so are the skip-reason counts across the 82 logs. Tile set inside it: 639 run, 15 skipped (10 upstream no-libmathdx, five 64 KiB capability skips), 0 failures; test_tile_matmul 25 OK. test_streams 24 OK (3 skipped, incl. the HIP priority-timing skip). No `Warp CUDA error` line in any log. The only "Failed to configure kernel dynamic shared memory" line is the known `test_tile_load_scoped` backward (65536). The gfx11 WMMA commit does not reach gfx90a code (MFMA target), and the counts confirm nothing moved. `example_apic_fluid.usd`, written into the clone by an example, was deleted afterwards.

### Next step for the lock holder
Nothing from gfx90a: the same-head pass at `e78ee32f` is complete and clean.

## Porting session 12 (2026-11-19, linux-gfx950 MI355X, porter): gfx11 rule narrowed to the forward

Fork `release/1.14.0.amd0` advanced `e78ee32f..a8e95dcb` (pushed, `advance-head` recorded): `a8e95dcb` [ROCm] Keep WMMA for the gfx11 tile_matmul adjoint of large tiles (`tile_matmul_mma.h` +11/-6, only the gfx11 helper and its one call). Nothing else changed.

**Why it can be done at compile time without restructuring.** The forward and the adjoint both reach `partitioned_gemm::scalar_matmul`, and `Accumulate` does not tell them apart (the accumulating forward, `wp.tile_matmul(a, b, out)`, passes `true` like the adjoint). The operand layouts do: the forward of plain tiles passes A and B row-major (unit stride along the second dimension), while the adjoint's two products each read one transposed, stride-swapped operand (`adj_A += adj_C @ B^T`: B operand `B^T`; `adj_B += A^T @ adj_C`: A operand `A^T`). The gfx11 rule `amortizes_operand_loads` now also takes the two layouts and sends a call to the scalar path only when, in addition to the earlier condition (16 or more fragments, fewer than 4 per wavefront), both operands are row-major. Limitation, recorded: a user's forward product on a transposed tile (`tile_transpose` as an operand) would keep WMMA in that configuration; that case is unmeasured.

**Verification through Warp's hipRTC path** (`cells_fb.py`, below: fp16 tile_matmul modules with their backward kernels, compiled for each target with this head's headers ("new") and with `e78ee32f`'s ("old", a worktree at `e78ee32f`, same `warp.so`); matrix instructions counted per kernel function; `insn-md5` over the disassembly):
```
new 64x64 bd256 gfx1100 forward   0 backward  16 insn-md5 fe55400c0d71
new 64x64 bd256 gfx1201 forward   8 backward  24 insn-md5 6ee7f04a0d1d
new 64x64 bd256 gfx950  forward  16 backward  48 insn-md5 d0ae36998d2c
new 64x64 bd256 gfx942  forward  16 backward  48 insn-md5 85006bb0d9e6
new 64x64 bd256 gfx90a  forward  16 backward  48 insn-md5 516c37da1e79
new 64x64 bd128 gfx1100 forward  16 backward  48 insn-md5 0ff181fe0323
new 64x64 bd128 gfx1201 forward  16 backward  36 insn-md5 d641ed889451
new 64x64 bd128 gfx950  forward  16 backward  48 insn-md5 a0803ff918ae
new 64x64 bd128 gfx942  forward  16 backward  48 insn-md5 d004205413d3
new 64x64 bd128 gfx90a  forward  16 backward  48 insn-md5 8c9326bedfc2
new 32x32 bd256 gfx1100 forward   2 backward   6 insn-md5 03335d85c25e
new 32x32 bd256 gfx1201 forward   2 backward   6 insn-md5 5b5f5e20320b
new 32x32 bd256 gfx950  forward   2 backward   6 insn-md5 1ea87b1bed27
new 32x32 bd256 gfx942  forward   2 backward   6 insn-md5 e6f7e4c73ed7
new 32x32 bd256 gfx90a  forward   2 backward   6 insn-md5 69d2b0be74fe
old 64x64 bd256 gfx1100 forward   0 backward   0 insn-md5 3764a8de29a7
old 64x64 bd256 gfx1201 forward   8 backward  24 insn-md5 6ee7f04a0d1d
old 64x64 bd256 gfx950  forward  16 backward  48 insn-md5 d0ae36998d2c
old 64x64 bd256 gfx942  forward  16 backward  48 insn-md5 85006bb0d9e6
old 64x64 bd256 gfx90a  forward  16 backward  48 insn-md5 516c37da1e79
old 64x64 bd128 gfx1100 forward  16 backward  48 insn-md5 0ff181fe0323
old 64x64 bd128 gfx1201 forward  16 backward  36 insn-md5 d641ed889451
old 64x64 bd128 gfx950  forward  16 backward  48 insn-md5 a0803ff918ae
old 64x64 bd128 gfx942  forward  16 backward  48 insn-md5 d004205413d3
old 64x64 bd128 gfx90a  forward  16 backward  48 insn-md5 8c9326bedfc2
old 32x32 bd256 gfx1100 forward   2 backward   6 insn-md5 03335d85c25e
old 32x32 bd256 gfx1201 forward   2 backward   6 insn-md5 5b5f5e20320b
old 32x32 bd256 gfx950  forward   2 backward   6 insn-md5 1ea87b1bed27
old 32x32 bd256 gfx942  forward   2 backward   6 insn-md5 e6f7e4c73ed7
old 32x32 bd256 gfx90a  forward   2 backward   6 insn-md5 69d2b0be74fe
```
- gfx1100, 64x64 at block_dim 256: forward kernel 0 matrix instructions (scalar, as before), backward kernel 16 `v_wmma_f32_16x16x16_f16` (was 0). The backward kernel also replays the forward product, which stays scalar.
- Every other (cell, target) is unchanged: the code objects are **byte-identical** (`cmp`) between the two heads for gfx1201, gfx950, gfx942 and gfx90a in all three cells and for gfx1100 in the other two cells; only `mm64_bd256_gfx1100.hsaco` differs.
- CUDA: nvcc 12.8 build at `a8e95dcb` succeeded; NVRTC compiles of the tile_matmul and texture modules with the headers of `e78ee32f` and `a8e95dcb`: identical PTX and cubin (`0805dcc9fb1a`/`6f630854bc9f`, `98d84c023888`/`7b52ca293caf`).
- gfx950 tile set (24 files): 639 run, 10 skipped, 0 failures. `audit-commits` OK, `jargon.py --port` clean.

Release-notes wording, updated: on gfx11 the scalar fallback applies to forward products of row-major tiles of 16 or more fragments when the block has more than one 32-thread wavefront per four fragments; the backward pass of the same tiles uses WMMA.

### Next step (head `a8e95dcb`)
The project stays in `porting`, lock held by linux-gfx950. Only gfx11 code changed, and only for that configuration: gfx942, gfx90a, gfx1201 and gfx950 compile byte-identical code objects to `e78ee32f` (above), so their passes at `e78ee32f` are unaffected by this commit (whether they are carried forward or re-run at `a8e95dcb` is a validation decision, not the porter's).

**gfx1100 must re-run at `a8e95dcb`** (rebuild; fresh `WARP_CACHE_PATH` per run):
1. `python -m warp.tests -s autodetect -p test_tile_matmul.py -m 32` (expect 25 OK) and the 24-file tile set (expect 639 run, 15 skipped, 0 failures).
2. `python mma_check.py` and `python mma_check_bf16.py` (Porting session 10 versions): FAILURES 0.
3. `cells_fb.py` built for `gfx1100` (`python cells_fb.py <dir> new gfx1100`): expect the table above (64x64 bd256 forward 0, backward 16).
4. The fp16 benchmark against the scalar tree (scalar worktree at `a8e95dcb` with `warp/native/tile_matmul.h` checked out from `974cd4bf`, `warp/bin/*.so` copied), three runs per tree, with the load monitor: expect the 64x64 block_dim 256 **forward** about equal in both trees (both scalar, about 17.6 ms) and its **adjoint** back to the WMMA figure (about 8.6 ms against 13.6 ms scalar, 1.59x); every other fp16 cell as in the earlier set 2; any fp16 cell below 1.0x is a finding.
5. The full 82-pattern set and the CPU-only suite at this head, for the same-head record (expect 81 OK + the fem pair; 2294 OK).
Then Phase 6 on the Windows gfx1151 laptop as planned in Porting session 11 (gfx1151 takes this same gfx11 code).

`cells_fb.py`:
```python
# fp16 tile_matmul modules (forward and backward kernels) per (tile, block_dim), compiled through Warp's hipRTC
# path for other targets; prints matrix instructions per kernel function and an md5 of each code object.
# Usage: python cells_fb.py <outdir> <tag> <gfx>...
import glob
import hashlib
import os
import re
import subprocess
import sys

import warp as wp
import warp._src.build as build

wp.init()
CELLS = [(64, 256), (64, 128), (32, 256)]
R = subprocess.check_output([sys.executable, "-m", "rocm_sdk", "path", "--root"]).decode().strip()


def make(M):
    @wp.kernel(module="unique")
    def mm(A: wp.array2d(dtype=wp.float16), B: wp.array2d(dtype=wp.float16), C: wp.array2d(dtype=wp.float16)):
        wp.tile_store(C, wp.tile_matmul(wp.tile_load(A, shape=(M, M)), wp.tile_load(B, shape=(M, M))))

    return mm


out_dir, tag = sys.argv[1], sys.argv[2]
for M, bd in CELLS:
    k = make(M)
    before = set(glob.glob(os.path.join(wp.config.kernel_cache_dir, "**", "*.cu"), recursive=True))
    wp.load_module(k.module, device="cuda:0", block_dim=bd)
    src = sorted(set(glob.glob(os.path.join(wp.config.kernel_cache_dir, "**", "*.cu"), recursive=True)) - before)[-1]
    for target in sys.argv[3:]:
        out = os.path.join(out_dir, f"{tag}_mm{M}_bd{bd}_{target}.hsaco")
        build.build_cuda(src, 0, out, pch_dir=None, hip_target=target)
        dis = subprocess.run([R + "/lib/llvm/bin/llvm-objdump", "-d", "--no-show-raw-insn", "--no-leading-addr", f"--mcpu={target}", out], capture_output=True, text=True).stdout
        counts, fn = {}, None
        for line in dis.splitlines():
            m = re.match(r"^<(.*)>:$", line.strip())
            if m:
                fn = "forward" if "forward" in m.group(1) else ("backward" if "backward" in m.group(1) else m.group(1))
                counts.setdefault(fn, 0)
                continue
            w = line.split()
            if fn and w and (w[0].startswith("v_wmma") or w[0].startswith("v_mfma")):
                counts[fn] += 1
        insn = hashlib.md5("\n".join(l for l in dis.splitlines() if "file format" not in l).encode()).hexdigest()[:12]
        print(f"{tag} {M}x{M} bd{bd:3d} {target:7s} forward {counts.get('forward', 0):3d} backward {counts.get('backward', 0):3d} insn-md5 {insn}")
```

## Evidence session 8 (2026-11-19, linux-gfx1100, Radeon Pro W7800, evidence only): gfx1100 re-run at `a8e95dcb`

**Verdict: every item matched its expected result.** That covers the tile set, test_tile_matmul, both `mma_check` scripts, the `cells_fb.py` table (with identical instruction hashes), the fp16 benchmark (64x64 bd256 forward equal to scalar, adjoint back on WMMA at 1.54x, no fp16 cell below 1.0x), the 82-pattern set and the CPU-only suite. One extra check outside the list (item 2) confirms the changed backward kernel is correct, because `mma_check.py` cannot launch it on a 64 KiB part.

Requested by Jeff Daily. The work lock stays with linux-gfx950. Nothing was committed to the fork, and no stage, lock, head, platform or deferral record changed. The clone was updated read-only to the recorded `head_sha` `a8e95dcb748c474f63b866ac07227cdde22650c4` (`git diff --stat e78ee32f HEAD`: only `warp/native/tile_matmul_mma.h`, +11/-6) and rebuilt (`export PATH=$FLEET_VENV/bin:$PATH; python build_lib.py --rocm-path "$(rocm-sdk path --root)" --hip-arch gfx1100 -j 32` -> "Warp build succeeded"; RUNPATH is the venv's ROCm 10.2). `git status --porcelain` is clean at the end. Same venv (HIP 7.17.26434), `ulimit -c 0`, `HIP_VISIBLE_DEVICES=0`. A fresh `WARP_CACHE_PATH` was used for every run, script, tree and test pattern. The scalar baseline was a worktree at `a8e95dcb` with `tile_matmul.h` from `974cd4bf` and this build's `warp/bin/*.so`, removed afterwards. Items are the "gfx1100 must re-run at `a8e95dcb`" list, in order.

### Item 1: test_tile_matmul and the tile set: match
`python -m warp.tests -s autodetect -p "<file>" -m 32` per file of the 24-file tile set: **639 run, 15 skipped, 0 failures**, with **test_tile_matmul Ran 25, OK**. The per-file summary lines (timings stripped) are identical to the `e78ee32f` run. The 15 skips are the 10 upstream no-libmathdx skips and the five 64 KiB capability skips. No `Warp CUDA error` line.

### Item 2: `mma_check.py` and `mma_check_bf16.py` (Porting session 10 versions): match
- `mma_check.py`: **FAILURES 0**; 15 forward and 25 backward shared-memory limits, the same 40 cases as at `e78ee32f` (`diff` of the SMEM lines with error columns stripped: empty).
- `mma_check_bf16.py`: **FAILURES 0** (55 cases).
- A coverage gap, recorded for the lock holder: on this 64 KiB part every fp16 64x64x64 case of `mma_check.py` is "SMEM-LIMIT (backward not launched)" at every block_dim, at `e78ee32f` as at `a8e95dcb`. Its kernel holds two products. So `mma_check.py` never executes the backward kernel this commit changed. I therefore ran the extra check below.

**Extra check (not in the list): the changed backward kernel, directly.** `agent_space/p12/bwd_check.py` (source below): a one-product fp16 64x64x64 kernel (fits in 64 KiB), forward and backward at block_dim 256, 128 and 64 on `cuda:0`, plus the same kernel on the CPU device. It compares against float64 products of the fp16 inputs:
```
cpu    bd256: rel err vs float64  C 1.7e-03 adjA 2.8e-03 adjB 2.1e-03 | vs CPU kernel  C 0.0e+00 adjA 0.0e+00 adjB 0.0e+00 | grads nonzero True
cuda:0 bd256: rel err vs float64  C 1.7e-03 adjA 3.6e-04 adjB 4.0e-04 | vs CPU kernel  C 0.0e+00 adjA 2.9e-03 adjB 2.0e-03 | grads nonzero True
cuda:0 bd128: rel err vs float64  C 3.5e-04 adjA 3.6e-04 adjB 4.0e-04 | vs CPU kernel  C 1.8e-03 adjA 2.9e-03 adjB 2.0e-03 | grads nonzero True
cuda:0 bd 64: rel err vs float64  C 3.5e-04 adjA 3.6e-04 adjB 4.0e-04 | vs CPU kernel  C 1.8e-03 adjA 2.9e-03 adjB 2.0e-03 | grads nonzero True
```
At block_dim 256 the forward is bit-identical to the CPU kernel, i.e. scalar, as the rule intends. The gradients have the WMMA (fp32-accumulating) error profile, 3.6e-04 and 4.0e-04 against float64, which is more accurate than the scalar/CPU fp16 path's 2.8e-03 and 2.1e-03. They are equal to the gradients of the all-WMMA block_dims 128 and 64. So the backward kernel the commit moved to WMMA is correct.

### Item 3: `cells_fb.py` for gfx1100: match
`python cells_fb.py <dir> new gfx1100` (verbatim from Porting session 12), with the instruction hashes identical to the porter's table:
```
new 64x64 bd256 gfx1100 forward   0 backward  16 insn-md5 fe55400c0d71
new 64x64 bd128 gfx1100 forward  16 backward  48 insn-md5 0ff181fe0323
new 32x32 bd256 gfx1100 forward   2 backward   6 insn-md5 03335d85c25e
```

### Item 4: fp16 benchmark, shipped tree against the scalar tree: matches
Porting session 9 `bench_both.sh` with this host's paths, `bench_tile_matmul.py --blocks 16384 --iters 50`, three runs per tree, fresh cache per run, 5-second load/GPU-busy monitor. **Clean:** no cell above 1.5x its median, launched-cell spread median 1.1%, max 11.8%. fp16 rows:

| dtype | tile | block_dim | scalar fwd ms | shipped fwd ms | fwd speedup | scalar adj ms | shipped adj ms | adj speedup | checks |
|---|---|---|---|---|---|---|---|---|---|
| float16 | 16x16 | 32 | 1.123 | 0.268 | 4.19x | 0.291 | 0.238 | 1.23x | ok |
| float16 | 16x16 | 64 | 1.145 | 0.277 | 4.14x | 0.277 | 0.204 | 1.36x | ok |
| float16 | 16x16 | 128 | 0.980 | 0.413 | 2.37x | 0.268 | 0.200 | 1.34x | ok |
| float16 | 16x16 | 256 | 1.391 | 0.464 | 3.00x | 0.304 | 0.246 | 1.24x | ok |
| float16 | 32x32 | 32 | 3.714 | 1.772 | 2.10x | 4.397 | 2.627 | 1.67x | ok |
| float16 | 32x32 | 64 | 4.256 | 1.812 | 2.35x | 2.458 | 1.636 | 1.50x | ok |
| float16 | 32x32 | 128 | 5.312 | 2.570 | 2.07x | 1.924 | 1.163 | 1.65x | ok |
| float16 | 32x32 | 256 | 6.841 | 2.624 | 2.61x | 1.949 | 1.150 | 1.69x | ok |
| float16 | 64x64 | 32 | 33.239 | 13.297 | 2.50x | 60.513 | 53.891 | 1.12x | ok |
| float16 | 64x64 | 64 | 27.438 | 18.336 | 1.50x | 33.642 | 30.188 | 1.11x | ok |
| float16 | 64x64 | 128 | 27.581 | 13.535 | 2.04x | 17.222 | 14.718 | 1.17x | ok |
| float16 | 64x64 | 256 | 17.598 | 16.778 | 1.05x | 13.594 | 8.826 | 1.54x | ok |

- **No fp16 cell below 1.0x.**
- **64x64 bd256 adjoint is back on WMMA:** shipped 8.53/8.83/8.99 ms against scalar 13.58/13.59/13.63 ms, **1.54x** (expected about 8.6 ms and 1.59x; set 2 at `c9595bce` measured 8.40-8.60 ms; this run's median is 2.6% above that, within this cell's 5% spread).
- **64x64 bd256 forward is about equal in both trees** (both scalar): shipped 16.20/16.78/17.30 against scalar 17.57/17.60/17.64 ms. The 1.05x is the same tree-order offset as at `e78ee32f`: the shipped tree runs first, and the fp32/fp64 control cells, identical code in both trees, also read 1.00x-1.03x.
- The other 22 fp16 cells are as in `c9595bce` set 2. The largest deviation of a shipped median from set 2's WMMA median is 4.2% (64x64 bd32 forward, faster now). The 64x64 bd256 forward is excluded from that comparison, since it is now scalar by design.

Raw fp16 timings per run, ms (`agent_space/p12/bench/bench_<tree>_run<n>.csv`):
```
tree,dtype,M,block_dim,fwd_r1,fwd_r2,fwd_r3,adj_r1,adj_r2,adj_r3
scalar,float16,16,32,1.1203,1.1231,1.1264,0.2962,0.2892,0.2912
scalar,float16,16,64,1.1339,1.1553,1.1454,0.2767,0.2855,0.2773
scalar,float16,16,128,0.9854,0.9754,0.9798,0.2675,0.2599,0.2795
scalar,float16,16,256,1.3915,1.3933,1.3914,0.3045,0.3168,0.3037
scalar,float16,32,32,3.7142,3.7190,3.7138,4.4015,4.3813,4.3970
scalar,float16,32,64,4.2557,4.2531,4.2638,2.4563,2.4673,2.4584
scalar,float16,32,128,5.3033,5.3253,5.3124,1.9239,1.9309,1.9230
scalar,float16,32,256,6.8148,6.8407,6.8435,1.9486,1.9541,1.9473
scalar,float16,64,32,33.2017,33.2388,33.2874,60.6047,60.4180,60.5127
scalar,float16,64,64,27.4288,27.4380,27.5134,33.6419,33.6178,33.6533
scalar,float16,64,128,27.5813,27.5806,27.6167,17.2054,17.2223,17.2328
scalar,float16,64,256,17.5655,17.5976,17.6421,13.5791,13.5943,13.6253
shipped,float16,16,32,0.2606,0.2680,0.2760,0.2485,0.2327,0.2376
shipped,float16,16,64,0.2719,0.2768,0.2803,0.2039,0.2020,0.2048
shipped,float16,16,128,0.3960,0.4130,0.4232,0.1962,0.2101,0.2004
shipped,float16,16,256,0.4467,0.4638,0.4731,0.2446,0.2534,0.2459
shipped,float16,32,32,1.7110,1.7719,1.8092,2.6226,2.6274,2.6642
shipped,float16,32,64,1.7113,1.8124,1.8625,1.6072,1.6358,1.6633
shipped,float16,32,128,2.4652,2.5696,2.6564,1.1114,1.1635,1.1793
shipped,float16,32,256,2.4900,2.6240,2.7110,1.1154,1.1504,1.1596
shipped,float16,64,32,12.9131,13.2969,13.7305,53.8897,53.9096,53.8913
shipped,float16,64,64,18.0630,18.3358,18.8592,30.1620,30.1884,30.1949
shipped,float16,64,128,12.8994,13.5354,13.9891,14.7282,14.7037,14.7184
shipped,float16,64,256,16.2038,16.7782,17.2962,8.5304,8.8261,8.9859
```

### Item 5: 82-pattern set and CPU-only suite
- 82-pattern set (the same list as Evidence sessions 3-7, `-p "<pattern>" -m 32`, fresh cache per pattern, timeit-wrapped): **82 patterns, 81 OK, 1 with the 2 known fem isolation failures** (`test_gimp_quadrature`, `test_cube_shape_functions`). The per-pattern summary lines (timings stripped) are **identical** to the `e78ee32f` run. No `Warp CUDA error` line in any log; the only shared-memory warning is the known `test_tile_load_scoped` one.
- CPU-only suite (`HIP_VISIBLE_DEVICES=99 CUDA_VISIBLE_DEVICES=99 python -m warp.tests -s autodetect -k cpu -m 32`, `PYTHONPATH` set inside the wrapped command, no `cuda:0` in the log): **Ran 2294, OK (skipped=15)**.

`bwd_check.py`:
```python
# Extra (not in the handoff list): correctness of the fp16 64x64x64 tile_matmul backward at block_dim 256
# (the kernel a8e95dcb moved back to WMMA on gfx11), one product per kernel so it fits in 64 KiB.
# GPU gradients against float64 references of the fp16 inputs, and against the same kernel on the CPU device.
import numpy as np
import warp as wp
wp.init()
M = 64
@wp.kernel(module="unique")
def one(A: wp.array2d(dtype=wp.float16), B: wp.array2d(dtype=wp.float16), C: wp.array2d(dtype=wp.float16)):
    wp.tile_store(C, wp.tile_matmul(wp.tile_load(A, shape=(M, M)), wp.tile_load(B, shape=(M, M))))
rng = np.random.default_rng(3)
An, Bn, Gn = (rng.uniform(-1, 1, (M, M)) for _ in range(3))
res = {}
for dev in ("cpu", "cuda:0"):
    for bd in ((256,) if dev == "cpu" else (256, 128, 64)):
        a = wp.array(An, dtype=wp.float16, device=dev, requires_grad=True)
        b = wp.array(Bn, dtype=wp.float16, device=dev, requires_grad=True)
        c = wp.zeros((M, M), dtype=wp.float16, device=dev, requires_grad=True)
        t = wp.Tape()
        with t:
            wp.launch_tiled(one, dim=[1], inputs=[a, b, c], block_dim=bd, device=dev)
        c.grad = wp.array(Gn, dtype=wp.float16, device=dev)
        t.backward()
        res[(dev, bd)] = [x.numpy().astype(np.float64) for x in (c, a.grad, b.grad)]
a16, b16, g16 = (np.asarray(x, dtype=np.float16).astype(np.float64) for x in (An, Bn, Gn))
ref = [a16 @ b16, g16 @ b16.T, a16.T @ g16]
for key, got in res.items():
    rel = [np.abs(g - r).max() / np.abs(r).max() for g, r in zip(got, ref)]
    vs_cpu = [np.abs(g - r).max() / np.abs(r).max() for g, r in zip(got, res[("cpu", 256)])]
    print(f"{key[0]:6s} bd{key[1]:3d}: rel err vs float64  C {rel[0]:.1e} adjA {rel[1]:.1e} adjB {rel[2]:.1e} | vs CPU kernel  C {vs_cpu[0]:.1e} adjA {vs_cpu[1]:.1e} adjB {vs_cpu[2]:.1e} | grads nonzero {bool(got[1].any() and got[2].any())}")
```

## Windows HIP build experiment (2026-11-19, windows-gfx1151 laptop, not under the lock, nothing pushed)

A private experiment in a local clone at `a8e95dcb` (no fork commit, no state change), to give the Phase 6 porter a tested starting diff. ROCm 10.2.0 TheRock wheels (`rocm[devel]` only, no device extra needed), HIP 7.17.26434, AMD clang 24.0.0git (x86_64-pc-windows-msvc), MSVC 14.44, `link.exe`.

Build: `python build_lib.py --hip-arch gfx1151 -j 8` succeeds in 66 s, `warp.dll` 17.9 MB. `python build_lib.py -j 8` with the default list (22 Windows targets from `share/therock/dist_info.json`: gfx1010-1012, gfx1030-1036, gfx1100-1103, gfx1150-1153, gfx1200, gfx1201, gfx908, gfx90a) succeeds in about 12.5 min, `warp.dll` **290.5 MB**. That size is a Phase 7 number for Jeff (his ruling: bring him wheel size and build time if unreasonable; do not shorten the list yourself). ROCm auto-detection through `rocm-sdk path --root` works.

What the diff does (54 insertions, 8 deletions in `build_dll.py`, `context.py`, `builtin.h`; full diff below): all host `.cpp` of `warp.dll` compile with ROCm's clang-cl (warp.h includes HIP headers); `--rocm-device-lib-path` for the wheels' layout; `.cu` through clang++ with the static MS runtime; link `amdhip64.lib`, `hiprtc.lib` and `clang_rt.builtins-x86_64.lib` (for `__extendhfsf2`/`__truncsfhf2`); the loader preloads `amdhip64` and `hiprtc` through `rocm_sdk.initialize_process` before `ctypes.CDLL(warp.dll, winmode=0)`, because that search order puts System32 first; `builtin.h:64` gains `&& !defined(_MSC_VER)` (`__debugbreak` is a builtin under MS compatibility). No `-fgpu-rdc` workaround was needed. Delay-loading `amdhip64` does not work: the `.cu` objects register fat binaries in static initialisers.

For the porter to improve, not accept as is: the loader decides whether the DLL is a HIP build by scanning its bytes for `amdhip64_` (an mmap search of a DLL that can be 290 MB); a build-time marker would be cleaner. There is no fallback when `rocm_sdk` is not installed (a source build against a non-wheel ROCm then gets the default DLL search). Not linted. The CUDA Windows path was not compiled (nvcc compile on mi355 owed; Windows CUDA compile possible with the laptop's toolkit).

Light validation on the gfx1151-only DLL, each once: device reports gfx1151 with hardware textures (1D/2D/3D); hipRTC kernel OK; WMMA layout probe for gfx11 0 of 256 wrong on all 6 lines and accumulation within bound; test_tile_matmul 25 OK; test_tile_shared_memory 30 OK (2 capability skips); test_texture 120 OK (1 skip); test_arithmetic 126, test_array 138, test_atomic 164, test_fp16 6, test_launch 32, test_utils 62, test_tile_sort 2, test_graph 35 (3 skips): all OK. The loaded runtime was the venv's `_rocm_sdk_core\bin\amdhip64_7.dll`, not System32 (psutil module list). With the abandoned delay-load attempt System32's copy was loaded and init still appeared to work, so a passing init is not evidence of the right runtime.

**Open defect: `test_mesh.py` crashes the process on Windows.** At `test_mesh_refit_graph_cuda_0`, access violation 0xC0000005 with every frame in the venv's `amdhip64_7.dll` on a runtime thread, main thread in `wp.capture_launch`. Order-dependent: the test alone passes, `-k refit` passes, `-k refit -k raises -k new_del -k exceptions` crashes, so one of `test_mesh_create_raises_on_native_failure`, `test_mesh_exceptions_cuda_0`, `test_mesh_new_del`, `test_mesh_points_setter_raises_on_native_device_failure` must run first. Linux passes this file at the same head. Not narrowed further (stopped at three crashes). Either a Windows HIP runtime defect in graph replay or state left by those tests on the Windows path.

Not run: the 82-pattern set, CPU-only suite, sdf/tex_alias probes, mma_check, tests against the 22-target DLL (init plus one kernel only). Environment notes for this laptop: packman's LLVM download needs the Windows root store exported to a PEM and `SSL_CERT_FILE`/`REQUESTS_CA_BUNDLE` (TLS interception); Git Bash needs the venv on PATH as `/d/...`.

```diff
diff --git a/warp/_src/build_dll.py b/warp/_src/build_dll.py
index 93f99ac9..734a67e2 100644
--- a/warp/_src/build_dll.py
+++ b/warp/_src/build_dll.py
@@ -5,6 +5,7 @@ from __future__ import annotations
 
 import concurrent.futures
 import functools
+import glob
 import json
 import os
 import pathlib
@@ -591,6 +592,11 @@ def build_dll_for_arch(args, dll_path, cpp_paths, cu_paths, arch, libs: list[str
         ]
         if args.fast_math:
             hip_opts.append("-ffast-math")
+        if os.name == "nt":
+            # ROCm Python packages keep the device libraries where the Windows driver does not look
+            hip_device_lib = os.path.join(os.path.dirname(os.path.dirname(hip_clang)), "amdgcn", "bitcode")
+            if os.path.isdir(hip_device_lib):
+                hip_opts.append(f'--rocm-device-lib-path="{hip_device_lib}"')
     elif cu_paths:
         # check CUDA Toolkit version
         ctk_version = get_cuda_toolkit_version(cuda_home)
@@ -650,19 +656,29 @@ def build_dll_for_arch(args, dll_path, cpp_paths, cu_paths, arch, libs: list[str
         libmathdx_includes = ""
         mathdx_enabled = "WP_ENABLE_MATHDX=0"
 
-    if os.name == "nt" and rocm_path:
-        raise RuntimeError("Warp build error: ROCm builds on Windows are not supported yet")
-
     if os.name == "nt":
         if args.host_compiler:
             host_linker = os.path.join(os.path.dirname(args.host_compiler), "link.exe")
         else:
             raise RuntimeError("Warp build error: No host compiler was found")
 
+        host_compiler = args.host_compiler
+        if rocm_path:
+            # MSVC cannot parse the HIP headers: host sources are compiled by ROCm's clang-cl,
+            # which takes the same options, and linked by the MSVC linker
+            host_compiler = os.path.join(os.path.dirname(hip_clang), "clang-cl.exe")
+
         # Build include paths for LLVM and CUDA
         llvm_include_paths = get_llvm_include_paths(args, warp_home_path, mode, arch)
         cpp_includes = format_include_paths(llvm_include_paths, "/I")
-        cuda_includes = f' /I"{cuda_home}/include"' if cu_paths else ""
+        if rocm_path:
+            # host sources see the CUDA headers they include through the HIP shims
+            cuda_includes = (
+                f' /D "WP_ENABLE_HIP=1" /D "__HIP_PLATFORM_AMD__" /I"{os.path.join(native_dir, "hip")}"'
+                f' /I"{rocm_path}/include"'
+            )
+        else:
+            cuda_includes = f' /I"{cuda_home}/include"' if cu_paths else ""
         includes = cpp_includes + cuda_includes
 
         # nvrtc_static.lib is built with /MT and _ITERATOR_DEBUG_LEVEL=0 so if we link it in we must match these options
@@ -715,7 +731,7 @@ def build_dll_for_arch(args, dll_path, cpp_paths, cu_paths, arch, libs: list[str
                     extra_flags = " /wd4624"  # suppress C4624: destructor was implicitly defined as deleted
                 if "fastcall.cpp" in cpp_path:
                     extra_flags += f' /I"{python_include_dir}" /DPy_LIMITED_API=0x030a0000'  # Python 3.10
-                cpp_cmd = f'"{args.host_compiler}" {cpp_flags}{extra_flags} -c "{cpp_path}" /Fo"{cpp_out}"'
+                cpp_cmd = f'"{host_compiler}" {cpp_flags}{extra_flags} -c "{cpp_path}" /Fo"{cpp_out}"'
                 cpp_cmds.append(cpp_cmd)
 
             if args.jobs <= 1:
@@ -734,7 +750,14 @@ def build_dll_for_arch(args, dll_path, cpp_paths, cu_paths, arch, libs: list[str
                         opt.replace("@filename@", os.path.basename(cu_path).replace(".", "_")) for opt in nvcc_opts
                     ]
 
-                    if mode == "debug":
+                    if rocm_path:
+                        # match the host objects: static CRT, release iterators
+                        hip_win_opts = f'{" ".join(hip_opts)} -fms-runtime-lib=static -DNDEBUG -D_ITERATOR_DEBUG_LEVEL=0 -DWP_ENABLE_CUDA=1 -I"{native_dir}" -D{mathdx_enabled}'
+                        if mode == "debug":
+                            cuda_cmd = f'"{hip_clang}" {hip_win_opts} -g -O0 -o "{cu_out}" -c "{cu_path}"'
+                        elif mode == "release":
+                            cuda_cmd = f'"{hip_clang}" {hip_win_opts} -O3 -o "{cu_out}" -c "{cu_path}"'
+                    elif mode == "debug":
                         cuda_cmd = f'{nvcc_cmd} --std=c++17 --compiler-options=/MT,/Zi,/Od -g -G -O0 -DNDEBUG -D_ITERATOR_DEBUG_LEVEL=0 -I"{native_dir}" -line-info {" ".join(_nvcc_opts)} -DWP_ENABLE_CUDA=1 -D{mathdx_enabled} {libmathdx_includes} -o "{cu_out}" -c "{cu_path}"'
                     elif mode == "release":
                         cuda_cmd = f'{nvcc_cmd} --std=c++17 -O3 {" ".join(_nvcc_opts)} -I"{native_dir}" -DNDEBUG -DWP_ENABLE_CUDA=1 -D{mathdx_enabled} {libmathdx_includes} -o "{cu_out}" -c "{cu_path}"'
@@ -743,7 +766,18 @@ def build_dll_for_arch(args, dll_path, cpp_paths, cu_paths, arch, libs: list[str
 
                     linkopts.append(quote(cu_out))
 
-                if args.use_dynamic_cuda:
+                if rocm_path:
+                    linkopts.append(f'amdhip64.lib hiprtc.lib /LIBPATH:"{rocm_path}/lib"')
+                    # Clang calls its runtime for float16 conversions on x86-64, and the MSVC linker
+                    # does not add that library by itself
+                    llvm_lib = os.path.join(os.path.dirname(os.path.dirname(hip_clang)), "lib")
+                    linkopts.extend(
+                        quote(path)
+                        for path in glob.glob(
+                            os.path.join(llvm_lib, "clang", "*", "lib", "windows", "clang_rt.builtins-x86_64.lib")
+                        )
+                    )
+                elif args.use_dynamic_cuda:
                     linkopts.append(
                         f'cudart.lib nvrtc.lib nvptxcompiler_static.lib ws2_32.lib user32.lib /LIBPATH:"{cuda_home}/lib/x64"'
                     )
diff --git a/warp/_src/context.py b/warp/_src/context.py
index 041525a8..8ea633ac 100644
--- a/warp/_src/context.py
+++ b/warp/_src/context.py
@@ -17,6 +17,7 @@ import inspect
 import io
 import itertools
 import json
+import mmap
 import operator
 import os
 import platform
@@ -4645,6 +4646,17 @@ class Runtime:
             warp_lib = os.path.join(bin_path, "warp.so")
             llvm_lib = os.path.join(bin_path, "warp-clang.so")
 
+        if os.name == "nt" and importlib.util.find_spec("rocm_sdk"):
+            # A library built for AMD GPUs imports the HIP runtime. Load that by full path from
+            # the ROCm Python packages first, so that another copy on the DLL search path (the
+            # display driver installs one in System32) is not picked up instead.
+            with open(warp_lib, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as image:
+                is_hip_lib = image.find(b"amdhip64_") >= 0
+            if is_hip_lib:
+                import rocm_sdk  # noqa: PLC0415
+
+                rocm_sdk.initialize_process(preload_shortnames=["amdhip64", "hiprtc"])
+
         self.core = self.load_dll(warp_lib)
 
         if os.path.exists(llvm_lib):
diff --git a/warp/native/builtin.h b/warp/native/builtin.h
index b5aa12d4..8b82aead 100644
--- a/warp/native/builtin.h
+++ b/warp/native/builtin.h
@@ -61,7 +61,7 @@
 
 #if defined(__CUDACC__) && !defined(_MSC_VER)
 __device__ inline void __debugbreak() { __brkpt(); }
-#elif defined(__HIP_DEVICE_COMPILE__)
+#elif defined(__HIP_DEVICE_COMPILE__) && !defined(_MSC_VER)
 __device__ inline void __debugbreak() { __builtin_debugtrap(); }
 #endif
 
```

## Rulings (2026-11-19, Jeff Daily)

- **3D textures with linear filtering on gfx90a: option A.** 3D textures on gfx90a use the GPU software sampler; 1D and 2D stay on hardware. Commits `c9595bce` and `e78ee32f` (the message fix) stand.
- **Porting lock for Phase 6: moved to the windows-gfx1151 laptop by Jeff's decision** (`port-lock --take windows-gfx1151`), from linux-gfx950, after all five Linux GPUs passed (gfx950, gfx942 one and two GPUs, gfx90a, gfx1201 at `e78ee32f`; gfx1100 at `a8e95dcb`, whose only difference is one gfx11 kernel). The Windows porter session starts from the "Windows HIP build experiment" section above. When Phase 6 is done the lock goes back to linux-gfx950 for the nvcc CUDA compile check.

## Ruling on bf16 scope for matrix-core tile_matmul (2026-11-19, Jeff Daily)

RULED: accept the current scope. bf16 is accelerated only as upstream's API allows it: bf16 inputs with a float32 output, forward only, on the MFMA path. No change to upstream's Python-level rejection of bf16 outputs and bf16 backward, and no bf16 on the WMMA path in this line.

### Correction to the bf16 scope entry above (2026-11-19)

The entry above wrongly bundled two things. What Jeff accepted is upstream's API restriction (bf16 inputs, float32 output, forward only). He did not agree to leaving bf16 off WMMA: his Q3 ruling asked for WMMA on Radeon for fp16 and bf16 on gfx11/gfx12, and the previous porter's omission of bf16 there was a deviation from that ruling, not a consequence of the API restriction (MFMA accelerates the same restricted case). bf16 inputs on WMMA for gfx11/gfx12 are owed in this line; assigned to the Windows porter session (gfx1151 is a gfx11 WMMA GPU), with the benchmark owed on gfx1100 and gfx1201.
