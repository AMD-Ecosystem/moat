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
