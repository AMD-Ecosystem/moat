# warp-1.14 notes

Older round sections (porting sessions 4 to 9 and the evidence sessions between them, 2026-11-19) are archived verbatim in `notes-archive.md`.

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

Promoted to cuda-to-rocm/references/build-headers-macros.md in bb61ae96.
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

Promoted to cuda-to-rocm/references/faults-wavefront.md in bb61ae96.
Reference file: `references/fault-classes.md` (floating point / types and dispatch, or a new "Clang optimizes undefined behaviour nvcc tolerates" entry next to the MSVC-lenient one). Block-reduction helpers often assign the result only on thread 0 (`T sum; if (threadIdx.x == 0) sum = ...; return sum;`) and every thread returns it. nvcc/NVVM tolerates this. Clang marks return values `noundef`, so returning an uninitialized value on any path is immediate undefined behaviour, and the optimizer may conclude the other paths never run: on AMD GPUs it emitted `llvm.assume(threadIdx.x == 0)`, then folded `partials[threadIdx.x / warpSize]` to `partials[0]`, so a two-wavefront block sum returned one wavefront's partial. Symptoms: correct at -O0, wrong at -O1 and above, and the bug disappears when a printf is added. Find it by compiling to LLVM IR (hipRTC `-fgpu-rdc` + `llvm-dis`, or `-emit-llvm`) and grepping for `llvm.assume` on `workitem.id`. Fix: value-initialize (`T sum {};`) every local that is returned or copied while unassigned on some threads. (warp-1.14)

## Lesson for cuda-to-rocm: rocPRIM radix sort of 64-bit keys with end_bit 64 and a non-zero begin_bit

Promoted to cuda-to-rocm/references/libraries.md in bb61ae96.
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

## Ruling on the unsafe atomic forms (2026-11-18, Jeff Daily)

RULED: option (a), keep the safe forms. fp32/fp64 `atomic_add` stay HIP's `atomicAdd`, fp16/bf16 `atomic_add` stay the 32-bit `atomicCAS` on the containing word; correct for every memory kind, slow where the measurement table says CAS. `unsafeAtomicAdd` and the packed fp16/bf16 hardware add are not adopted in this line, and no module option is added. This closes the item the porter stopped on in the Q2 atomics section and supersedes the "prefer hardware atomics" part of the amended Q2 ruling in plan.md where the two conflict. Release notes say that fp16/bf16 atomic adds, and fp32/fp64 atomic adds on targets outside gfx942/gfx950 (fp32 also gfx1201), are compare-and-swap loops.

For reference, what AMD's `release/1.13.0.amd0` branch does (read from its `warp/native/builtin.h`, not built or run): fp16 `atomic_add` calls `unsafeAtomicAdd(__half*, ...)` unconditionally on HIP, with no memory-kind condition; fp32 and fp64 use plain `atomicAdd`; float/double min and max keep upstream's `atomicCAS` loops; bf16 min and max use a 16-bit `atomicCAS` loop; bf16 `atomic_add` has no HIP branch and that tree never defines `__CUDA_ARCH__`, so on a HIP device compile it takes the host branch, a plain non-atomic read-add-write.

## Ruling on test_atomic_cas (2026-11-18, Jeff Daily), and work it adds for the porter

RULED: keep the narrowing of the spin-lock test to devices with independent thread scheduling, and add a small AMD-capable test of `atomic_cas` itself (many threads compete once on one cell, exactly one wins; also `atomic_exch`), run on every device, so the operations are covered on AMD GPUs.

Who depends on the spin-lock behaviour (asked by Jeff; `git grep atomic_cas|atomic_exch` on release-1.14 outside tests and docs): no library code. `atomic_cas`/`atomic_exch` are user-facing builtins added in eaa3cf3d ("Expose atomic cas and and atomic exch GH-767"); the only callers in the tree are `warp/tests/test_atomic_cas.py` and two examples, neither registered in the test suite: `warp/examples/core/example_spin_lock.py` (the same `while atomic_cas(lock,0,0,1) == 1: pass` spin lock, so it is expected to hang on AMD GPUs) and `warp/examples/core/example_work_queue.py` (a lock-free queue: CAS retry loops where no thread waits on another to release, so it is expected to make progress; not yet run).

Porter, next session: (1) add the AMD-capable `atomic_cas`/`atomic_exch` test; (2) run `example_work_queue.py` on an AMD GPU under a timeout and record the result; (3) make `example_spin_lock.py` end with a clear message on a GPU without independent thread scheduling instead of hanging, using the same capability the test uses; (4) release notes: spin locks contended inside one wavefront hang on AMD GPUs, and `example_spin_lock.py` is the in-tree case.

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
## Stream priority on HIP (2026-11-18, Jeff Daily)

Stated by Jeff: stream priorities are essentially ignored in the current HIP runtime; high-priority streams exist as an API feature but map to normal priority. He does not expect a test that measures a timing difference to achieve anything on HIP. This settles C4 without the proposed kernel-workload measurement (step 1 of the C4 proposal is dropped): it is known HIP behaviour, not a defect to register.

Porter, next session: in `test_stream_priority_timings`, keep everything that does not depend on scheduling (creating streams with a priority, reading the priority back) and skip only the timing assertion on HIP devices, with the fewest lines possible, through a device capability (not an arch or vendor name in the test) and a skip message saying HIP maps stream priorities to normal priority. Release notes: stream priority is accepted but has no scheduling effect on AMD GPUs.

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

## Porting session 13 (2026-11-19, windows-gfx1151 laptop, porter): Phase 6, Windows HIP build

Lock held by windows-gfx1151 (moved by Jeff Daily). Fork `release/1.14.0.amd0` advanced `a8e95dcb..6761183c` (pushed, `advance-head` recorded after each push). Environment: Windows 11, Radeon 8060S (gfx1151, 20 CUs, shared memory), VS 2022 Build Tools MSVC 14.44 (its environment variables already set in the shell, so `build_lib.py` skips vcvars), Python 3.12.10 venv at `agent_space/warp-win/venv` with `rocm[devel]==10.2.0` (HIP 7.17.26434, AMD clang 24), `ruff==0.15.9` and `clang-format==22.1.2` (the pre-commit pins) installed there this session. `TMPDIR`/`TEMP`/`TMP`, pip cache, packman root and every `WARP_CACHE_PATH` under `D:/Develop/moat/agent_space/warp-win/`; `SSL_CERT_FILE`/`REQUESTS_CA_BUNDLE` = the exported Windows root store (`win-ca.pem`) for packman's LLVM download. `utils/timeit.sh` needs `MOAT_PYTHON` set to the system Python 3.12 (the venv lacks jsonschema) and runs the build as `bash -c "cd <src> && <venv python> build_lib.py ..."` because it changes directory.

### Commit 1: `13e4e46e` [ROCm] Build warp.dll for AMD GPUs on Windows
The experiment's diff (Windows HIP build experiment section), with the two changes asked for:
- **No byte scan.** `build_dll.py` (Windows branch, after the link) writes the ROCm root it used to `warp/bin/warp.dll.rocm` and removes that file on any non-HIP Windows build (checked: a CUDA build in a tree holding a stale `warp.dll.rocm` removed it). `context.py` checks `os.path.exists(warp_lib + ".rocm")` (Windows only) and calls the new module-level `load_windows_hip_runtime()`. `import mmap` gone.
- **Defined behaviour without `rocm_sdk`.** `load_windows_hip_runtime`: `rocm_sdk` importable -> `rocm_sdk.initialize_process(preload_shortnames=["amdhip64", "hiprtc"])` as before; otherwise it loads `amdhip64_<n>.dll` and `hiprtc<nnnn>.dll` (regex, newest name) by full path from `<recorded root>/bin` with `ctypes.CDLL`; if either is missing it raises `RuntimeError: Warp was built for AMD GPUs, but the HIP runtime was not found in '<root>\bin'. Install the ROCm Python packages or the ROCm this build used.`
- Checked (`agent_space/warp-win/p6/loadcheck.py`, psutil module list + one kernel on gfx1151): with `rocm_sdk` -> `_rocm_sdk_core\bin\amdhip64_7.dll`, `hiprtc0717.dll`, `amd_comgr.dll`; with `rocm_sdk` hidden (`importlib.util.find_spec` patched to return None for it) -> loaded from the recorded `_rocm_sdk_devel\bin` (psutil shows the `_rocm_sdk_core` path because the two `bin` copies are hard links, `fsutil hardlink list`); marker pointing at a missing root -> the ROCm RuntimeError. No System32 copy loaded in any case.
- Packaging gap for Phase 7: `setup.py` collects `warp/bin/*` by library extension only, so `warp.dll.rocm` is not in a wheel yet. Phase 7 must add it to the Windows wheel (one line in `get_warp_libraries`/package data), otherwise an installed ROCm wheel loads the System32 runtime.
- Comment wording changed from the experiment ("where the Windows driver does not look" -> "where Clang does not look for them on Windows"); everything else in `build_dll.py`/`builtin.h` as tested in the experiment. ruff check/format clean (0.15.9).
- Build: `python build_lib.py --hip-arch gfx1151 -j 8` (ROCm auto-detected through `rocm-sdk path --root`) -> "Warp build succeeded", `warp.dll` 17,916,928 bytes. test_tile_matmul 25 OK, test_texture 120 OK (1 skip) at this commit.

### Windows CUDA compile (no NVIDIA GPU; compile and link only)
Separate worktree at `13e4e46e` (`agent_space/warp-win/cuda-wt`), conda-forge CUDA 12.8 toolkit (`C:\Users\jdaily\cuda-win\cuda128\Library`, nvcc V12.8.93), MSVC 14.44: `LIB="<toolkit>\Library\lib;$LIB" python build_lib.py --cuda-path <toolkit>\Library --use-dynamic-cuda --no-use-libmathdx -j 8` -> **"Warp build succeeded"** (all 11 .cu through nvcc, link OK). `--use-dynamic-cuda` because the conda-forge toolkit ships no `nvrtc_static.lib`/`nvrtc-builtins_static.lib`, and `LIB` because it keeps import libraries in `lib`, not `lib/x64`; both are properties of this toolkit, not of the port. The post-build cache clear failed to load `warp.dll` (cudart64_12/nvrtc DLLs not on PATH); with `Library\bin` on PATH, `import warp; wp.init()` loads it and reports CUDA Toolkit (12, 8), 0 CUDA devices. nvcc 12.8 accepts MSVC 14.44 here, so the upstream VS 18 refusal noted at intake did not arise and `release-1.14` was not built for comparison. The libmathdx variant and the static-CUDA variant were not built on Windows (no static NVRTC in this toolkit); the Linux nvcc check on mi355 is still owed after Phase 6.

### Commit 2: `6761183c` [ROCm] Use WMMA for bf16 tile_matmul inputs on gfx11 and gfx12
Added on Jeff's request today (he accepted upstream's restricted bf16 scope: bf16 inputs, float32 output, forward only). `tile_matmul_mma.h` +49/-2: `op<bfloat16>` for gfx11 (`__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32`, 16 x `short` operands, gfx11 fp16 layout) and gfx12 (`__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12`, 8 x `short`, gfx12 fp16 layout), each under `__has_builtin(...) && !defined(WP_NO_BFLOAT16)`. `supported()` and `amortizes_operand_loads()` unchanged, so the gfx11 forward exclusion (16+ fragments, under 4 per wavefront, both operands row-major: 64x64 at block_dim 256) applies to bf16 exactly as to fp16. `tile_matmul.h` and the Python-level bf16 rules untouched.
- **Layout probe first** (`agent_space/warp-win/p6/bf16/wmma_layout_bf16.hip`, below; the fp16 probe adapted: position code `C[i][j] = i + 16 j` because 100 j is not exact in bf16; inputs rounded to bf16 on the host): gfx1151 -> "bf16 WMMA family 11", all six layout lines 0 of 256 wrong (K = 16, 32, 64), accumulation 0 over the bound, worst 0.21/0.15/0.07 of the bound. Built also for gfx1201 and gfx1100 (compile only). **The gfx12 bf16 layout is assumed from gfx12 fp16, not measured**: gfx1201 must run this probe.
- `mma_check_bf16.py` (Porting session 10 version) on gfx1151, fresh cache: 55 cases, **FAILURES 0**, max difference 1.9e-6 (64x64x64). `isa_count.py` over its cache: `v_wmma_f32_16x16x16_bf16` in 34 code objects, exactly the 7 shapes with 16-multiple M, N, K at all 5 block_dims except 64x64x64 at block_dim 256 (scalar, as intended); the 4 other shapes scalar.
- test_tile_matmul (fresh cache): Ran 25, OK.
- `cells_bf16.py` (below): bf16 and fp16 modules (16x16 bd64, 32x32 bd64, 64x64 bd128, 64x64 bd256) compiled through hipRTC for gfx1151, gfx1100, gfx1201, gfx950, gfx942, gfx90a with this commit's headers ("new") and `13e4e46e`'s ("old", worktree with the same `warp.dll`): 38 of 48 code objects byte-identical, including every MFMA object (gfx950, gfx942, gfx90a, bf16 and fp16) and every fp16 object. The 10 that differ are the gfx11/gfx12 bf16 ones that gained WMMA (16 instructions at 64x64 bd128; gfx1201 64x64 bd256 now 8, as its fp16 twin); gfx1151 and gfx1100 64x64 bd256 bf16 stay scalar and byte-identical.
- **bf16 WMMA speed is unmeasured** (no benchmarks on this laptop). It mirrors the fp16 selection; gfx1100 and gfx1201 must benchmark it against the scalar tree (Next step).

### The `test_mesh.py` crash: not order-dependent, intermittent, cause not established (test unmodified)
Budget was 6 crashing runs; 3 used. Script `agent_space/warp-win/p6/mesh_seq.py` (below) replays test_mesh.py's GPU steps in one process with a print after each graph launch: `exc` = `test_mesh_exceptions(cuda:0)`, `graph` = `test_mesh_refit_graph(cuda:0)` (for sah, median, lbvh: new Mesh, capture `mesh.refit()`, 10 x `wp.capture_launch`, sync), `graphkeep`/`graphkg`/`graphkm` = the same but keeping the previous constructor's mesh and graph / graph only / mesh only alive, `trace` = also print around every native graph launch, graph destroy and free.

| run | steps | result |
|---|---|---|
| 1 | exc graph | **crash** (access violation in `amdhip64_7.dll` worker thread) at median replay 6 |
| 2 | exc gc graph | pass (`gc.collect()` found 0 objects) |
| 3 | exc graph (repeat of 1) | **crash** at median replay 5-6 |
| 4 | trace exc graph | pass (host slowed by the tracing prints) |
| 5 | exc graph:median (median only) | pass |
| 6 | exc graphkeep | pass |
| 7 | exc graphkg | pass |
| 8 | exc graphkm | pass |
| 9 | graph (no exception test first) | **crash** at median replay 7 |

What this establishes:
- **The experiment's order dependence was a red herring.** Run 9 crashes with no other test before it, so none of the four tests the experiment named is needed (three of them never touch the GPU; `test_mesh_exceptions` only allocates and frees small arrays). `-k refit` passing in the experiment was timing.
- The crash needs a previous refit graph (sah) to have been replayed and destroyed (its Mesh freed, its graph and graph exec destroyed) before the next mesh's graph (median) is captured and replayed back to back; it happens on the 5th-7th replay, on a runtime worker thread, while the main thread is inside `hipGraphLaunch`. Faulting address in all three: `amdhip64_7.dll + 0x160C6C` (the export-relative names faulthandler prints are meaningless; the DLL is stripped).
- It is timing-dependent: slowing the host between launches (run 4) or a different preceding call (run 2) avoids it, and single runs that keep the previous mesh and/or graph alive passed (runs 6-8; one run each, so "keeping them alive avoids it" is suggestive, not proven).
- The refit graph contains kernels, a stream-ordered allocation and free around a hipCUB inclusive scan (`scan_device` allocates its temp storage inside the capture, so the graph gets mem-alloc/mem-free nodes), a memset and the BVH refit kernel, and Warp instantiates graphs with `hipGraphInstantiateFlagAutoFreeOnLaunch`. The code involved (`wp_cuda_graph_*`, `wp_alloc/free_device_async`, the capture bookkeeping) is upstream's, unchanged by the port, and the port's HIP wrappers on this path are identical on Linux, where test_mesh passes on all five GPUs. No port code runs on this path only on Windows (the Windows commits touch only the build and the DLL loader).
- Standalone reproducer attempt (`agent_space/warp-win/p6/graphrepro/graph_memnode.hip`: the same node sequence with a hipCUB scan, AutoFreeOnLaunch, 6 rounds x 10 launches, graph destroyed and buffers freed between rounds): **does not reproduce** (PASS). So the evidence leans towards the Windows HIP runtime (graph replay after a destroyed graph with memory nodes) but does not prove it, and nothing was registered. `test_mesh.py` is left unmodified; `test_mesh_refit_graph_cuda_0` can crash the worker on this laptop.
- Next diagnostic step if someone takes it up (not in this budget): a HIP-only replay of Warp's exact call sequence via `WARP_DEBUG`/API tracing (`AMD_LOG_LEVEL=4` on one crashing run), or the reproducer with Warp's real kernels and its non-null-stream capture plus `hipFreeAsync(ptr, nullptr)` on the null stream for the freed mesh.

### Default-target `warp.dll` size (numbers for Jeff; nothing changed)
`agent_space/warp-win/p6/fatbin_sizes.py` (below) parses the 11 clang offload bundles (one per `.cu`) in the experiment's 22-target `warp.dll` (290.5 MB, `python build_lib.py -j 8`, about 12.5 min) and in a gfx1151-only one (17.9 MB):
- Device code: 285.2 MB of the 290.5 MB; **12.7-13.2 MB per target** (gfx1200/1201 13.16, gfx11xx 13.08-13.09, gfx908 13.02, gfx101x 12.88, gfx1031-1036 12.86, gfx1030 and gfx90a 12.67). Host code and data: about 5 MB (17.9 - 13.1 for the single-target DLL). So size is linear in the target count: about 5 MB + 13 MB x targets.
- Compressibility: each target's code objects deflate (zlib level 6) to about **1.07 MB** (12x). The whole 22-target DLL zipped (deflate 6, as a wheel stores it) is **24.4 MB**; the single-target DLL zips to 2.1 MB. The bundles are stored uncompressed (no `CCOB` compressed-bundle marker); clang's `--offload-compress` would store them compressed inside the DLL (installed size near the zipped size) at some load-time cost, unmeasured. Not tried: the target list and the compression switch are Jeff's decision.

### Scripts from this session

`loadcheck.py`:
```python
# Usage: python loadcheck.py [no-rocm-sdk] [marker-override-root]
# Initializes Warp, runs one kernel on cuda:0 and prints which HIP runtime files the process loaded.
import importlib.util
import os
import sys

import psutil

if "no-rocm-sdk" in sys.argv[1:]:
    _orig = importlib.util.find_spec
    importlib.util.find_spec = lambda name, *a, **k: None if name == "rocm_sdk" else _orig(name, *a, **k)

import warp as wp

wp.init()
print("rocm_sdk imported:", "rocm_sdk" in sys.modules)
for m in psutil.Process().memory_maps():
    b = os.path.basename(m.path).lower()
    if b.startswith(("amdhip", "hiprtc", "amd_comgr")):
        print("loaded:", m.path)


@wp.kernel
def k(a: wp.array(dtype=float)):
    i = wp.tid()
    a[i] = float(i) * 2.0


a = wp.zeros(8, dtype=float, device="cuda:0")
wp.launch(k, dim=8, inputs=[a], device="cuda:0")
print(wp.get_device("cuda:0").arch_name, a.numpy().tolist())
```

`mesh_seq.py`:
```python
# Replays test_mesh.py's GPU steps in order with prints, to find which preceding step makes the
# refit graph replay crash. Usage: python mesh_seq.py <step>...  steps: exc, refit, graph
import faulthandler
import sys
import unittest

faulthandler.enable()
import warp as wp
from warp.tests.geometry import test_mesh as tm

wp.config.log_level = wp.LOG_WARNING
wp.init()
dev = wp.get_device("cuda:0")
tc = unittest.TestCase()
kept = []
if "trace" in sys.argv:
    import gc

    gc.callbacks.append(lambda phase, info: print("  gc", phase, info, flush=True))
    core = wp._src.context.runtime.core
    for name in ("wp_cuda_graph_destroy", "wp_cuda_graph_exec_destroy", "wp_mesh_destroy_device", "wp_free_device_async", "wp_cuda_graph_launch"):
        fn = getattr(core, name)

        def wrap(*a, _fn=fn, _name=name):
            print("  call", _name, a[1:] if _name != "wp_cuda_graph_launch" else "", flush=True)
            r = _fn(*a)
            print("  ret", _name, flush=True)
            return r

        setattr(core, name, wrap)
for step in [a for a in sys.argv[1:] if a != "trace"]:
    if step == "gc":
        import gc

        print("collected", gc.collect(), flush=True)
    elif step == "nogc":
        import gc

        gc.disable()
    elif step == "exc":
        tm.test_mesh_exceptions(tc, dev)
    elif step == "refit":
        tm.test_mesh_refit(tc, dev)
    elif step.startswith("graph"):
        ctors = step.split(":")[1].split(",") if ":" in step else ["sah", "median", "lbvh"]
        for c in ctors:
            points = wp.array(tm.POINT_POSITIONS, dtype=wp.vec3, device=dev)
            indices = wp.array(tm.RIGHT_HANDED_FACE_VERTEX_INDICES, dtype=int, device=dev)
            mesh = wp.Mesh(points=points, indices=indices, bvh_constructor=c)
            wp.capture_begin(dev, force_module_load=False)
            try:
                mesh.refit()
            finally:
                graph = wp.capture_end(dev)
            print("captured", c, flush=True)
            for i in range(10):
                wp.capture_launch(graph)
                print(" launch", c, i, flush=True)
            wp.synchronize_device(dev)
            print("done", c, flush=True)
            if step.startswith("graphkeep"):
                kept.append((mesh, graph))
            elif step.startswith("graphkg"):
                kept.append(graph)
            elif step.startswith("graphkm"):
                kept.append(mesh)
    wp.synchronize_device(dev)
    print("step ok:", step, flush=True)
```

`graphrepro/graph_memnode.hip`:
```cpp
// A graph with a stream-ordered allocation inside it (kernel, hipMallocAsync, kernels using the
// allocation (a hipCUB inclusive scan), hipFreeAsync, memset, kernel), instantiated with hipGraphInstantiateFlagAutoFreeOnLaunch
// and launched back to back; then destroyed and the next round captured with fresh buffers.
// Usage: graph_memnode [rounds] [launches]
#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>
#include <cstdio>
#include <cstdlib>
#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { printf("%s:%d %s -> %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e)); exit(1); } } while (0)

__global__ void fill(float* a, int n, float v) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) a[i] = v * i; }
__global__ void scan_like(const float* a, float* t, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) t[i] = a[i] + (i ? a[i - 1] : 0.f); }
__global__ void reduce_one(const float* t, float* out, int n) { if (threadIdx.x == 0 && blockIdx.x == 0) out[0] = t[n - 1] / n; }
__global__ void bounds(const float* a, float* lo, float* hi, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) { lo[i] = a[i] - 1; hi[i] = a[i] + 1; } }
__global__ void refit(int* counts, const float* lo, float* nodes, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) { atomicAdd(&counts[i % 7], 1); nodes[i] = lo[i]; } }

int main(int argc, char** argv) {
    int rounds = argc > 1 ? atoi(argv[1]) : 3, launches = argc > 2 ? atoi(argv[2]) : 10;
    const int n = 12;
    CK(hipSetDevice(0));
    hipStream_t s;
    CK(hipStreamCreateWithPriority(&s, hipStreamDefault, 0));
    for (int r = 0; r < rounds; ++r) {
        float *pts, *lo, *hi, *out, *nodes; int* counts;
        CK(hipMallocAsync((void**)&pts, n * 4, s)); CK(hipMallocAsync((void**)&lo, n * 4, s));
        CK(hipMallocAsync((void**)&hi, n * 4, s)); CK(hipMallocAsync((void**)&out, 64, s));
        CK(hipMallocAsync((void**)&nodes, 2 * n * 4, s)); CK(hipMallocAsync((void**)&counts, 2 * n * 4, s));
        fill<<<1, 64, 0, s>>>(pts, n, 1.f + r);
        CK(hipStreamSynchronize(s));

        hipGraph_t g; hipGraphExec_t ge;
        CK(hipStreamBeginCapture(s, hipStreamCaptureModeThreadLocal));
        fill<<<1, 64, 0, s>>>(lo, n, 0.5f);
        size_t tsz = 0;
        CK(hipcub::DeviceScan::InclusiveSum(nullptr, tsz, lo, lo, n, s));
        void* tmp;
        CK(hipMallocAsync(&tmp, tsz, s));
        CK(hipcub::DeviceScan::InclusiveSum(tmp, tsz, lo, lo, n, s));
        CK(hipFreeAsync(tmp, s));
        reduce_one<<<1, 64, 0, s>>>(lo, out, n);
        bounds<<<1, 64, 0, s>>>(pts, lo, hi, n);
        CK(hipMemsetAsync(counts, 0, 2 * n * 4, s));
        refit<<<1, 64, 0, s>>>(counts, lo, nodes, n);
        CK(hipStreamEndCapture(s, &g));
        CK(hipGraphInstantiateWithFlags(&ge, g, hipGraphInstantiateFlagAutoFreeOnLaunch));
        for (int i = 0; i < launches; ++i) CK(hipGraphLaunch(ge, s));
        CK(hipStreamSynchronize(s));
        CK(hipGraphDestroy(g));
        CK(hipGraphExecDestroy(ge));
        float h[2]; CK(hipMemcpy(h, nodes, 8, hipMemcpyDeviceToHost));
        CK(hipFreeAsync(pts, nullptr)); CK(hipFreeAsync(lo, nullptr)); CK(hipFreeAsync(hi, nullptr));
        CK(hipFreeAsync(out, nullptr)); CK(hipFreeAsync(nodes, nullptr)); CK(hipFreeAsync(counts, nullptr));
        CK(hipDeviceSynchronize());
        printf("round %d ok (%g %g)\n", r, h[0], h[1]); fflush(stdout);
    }
    printf("PASS\n");
    return 0;
}
```

`bf16/wmma_layout_bf16.hip`:
```cpp
// Check the bf16 WMMA 16x16 fragment layouts on one 32-lane wavefront, before tile_matmul_mma.h
// relies on them, and separately bound the accumulation error. lane l: lo = l % 16, g = l / 16.
// Assumed (the fp16 layouts of the same family):
//   gfx11 (wmma_f32_16x16x16_bf16_w32): A[lo][v], B[v][lo] for v < 16 in both half-waves;
//       accumulator r is C[2 r + g][lo]
//   gfx12 (wmma_f32_16x16x16_bf16_w32_gfx12): A[lo][8 g + v], B[8 g + v][lo] for v < 8;
//       accumulator r is C[8 g + r][lo]
// Tests, for K = 16, 32, 64:
//   layout (non-negative): small non-negative integers (exact in bf16), every sum exact in f32
//       whatever the accumulation order, so any wrong cell is a layout error
//   layout (position): A[i][0] = i, A[i][1] = 1, B[0][j] = 1, B[1][j] = 16 j (exact in bf16), so
//       C[i][j] = i + 16 j names its own position
//   accumulation: random signed bf16 in [-1, 1]; |C - exact| must stay within
//       (K / 16 + 1) * FLT_EPSILON * sum |a b| (one f32 rounding per instruction, plus one)
#include <hip/hip_runtime.h>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

typedef float f8 __attribute__((ext_vector_type(8)));

// the values used are exact in bf16, so truncating the f32 bit pattern is exact
__device__ short bf16_bits(double x) { return short(__float_as_uint(float(x)) >> 16); }

static double round_bf16(double x)
{
    float f = float(x);
    unsigned u;
    memcpy(&u, &f, 4);
    u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000u;  // round to nearest even
    memcpy(&f, &u, 4);
    return f;
}

__global__ void k(const double* A, const double* B, double* C, int K, int* kind)
{
    int l = threadIdx.x, lo = l % 16, g = l / 16;
    f8 acc = {};
#if __has_builtin(__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32)
    typedef short V __attribute__((ext_vector_type(16)));
    for (int k0 = 0; k0 < K; k0 += 16) {
        V a, b;
        for (int v = 0; v < 16; ++v) {
            a[v] = bf16_bits(A[lo * K + k0 + v]);
            b[v] = bf16_bits(B[(k0 + v) * 16 + lo]);
        }
        acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, acc);
    }
    for (int r = 0; r < 8; ++r) C[(2 * r + g) * 16 + lo] = acc[r];
    if (l == 0) *kind = 11;
#elif __has_builtin(__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12)
    typedef short V __attribute__((ext_vector_type(8)));
    for (int k0 = 0; k0 < K; k0 += 16) {
        V a, b;
        for (int v = 0; v < 8; ++v) {
            a[v] = bf16_bits(A[lo * K + k0 + 8 * g + v]);
            b[v] = bf16_bits(B[(k0 + 8 * g + v) * 16 + lo]);
        }
        acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, acc);
    }
    for (int r = 0; r < 8; ++r) C[(8 * g + r) * 16 + lo] = acc[r];
    if (l == 0) *kind = 12;
#else
    if (l == 0) *kind = -1;
#endif
}

static void fill(int test, int K, double* A, double* B)
{
    for (int x = 0; x < 16 * K; ++x) {
        int i = x / K, kk = x % K;
        if (test == 0) A[x] = double((x * 7 + 3) % 9);
        if (test == 1) A[x] = kk == 0 ? double(i) : (kk == 1 ? 1.0 : 0.0);
        if (test == 2) A[x] = round_bf16(2.0 * rand() / RAND_MAX - 1.0);
    }
    for (int x = 0; x < K * 16; ++x) {
        int kk = x / 16, j = x % 16;
        if (test == 0) B[x] = double((x * 5 + 1) % 7);
        if (test == 1) B[x] = kk == 0 ? 1.0 : (kk == 1 ? 16.0 * j : 0.0);
        if (test == 2) B[x] = round_bf16(2.0 * rand() / RAND_MAX - 1.0);
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
            double worst = 0;
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
                printf("bf16 WMMA family %d, %-22s K=%2d: %d of 256 wrong\n", *kind, names[test], K, bad);
            else
                printf("bf16 WMMA family %d, %-22s K=%2d: %d of 256 over the bound, worst error %.2f of the bound\n",
                       *kind, names[test], K, bad, worst);
            fails += bad != 0 || *kind <= 0;
            hipFree(A); hipFree(B); hipFree(C); hipFree(kind);
        }
    if (kind_seen <= 0) printf("no bf16 WMMA instruction on this target\n");
    return fails ? 1 : 0;
}
```

`isa_count.py`:
```python
# For every code object in a Warp kernel cache: block_dim, the first two tile_load shapes (A and B),
# and the count of each matrix instruction. Usage: python isa_count.py <cache_dir> <gfx>
import glob
import os
import re
import subprocess
import sys

R = subprocess.check_output([sys.executable, "-m", "rocm_sdk", "path", "--root"]).decode().strip()
rows = []
for hs in glob.glob(os.path.join(sys.argv[1], "**", "*.hsaco"), recursive=True):
    src = open(hs.rsplit(".", 2)[0] + ".cu").read()
    bd = re.search(r"#define WP_TILE_BLOCK_DIM (\d+)", src).group(1)
    loads = re.findall(r"tile_load<wp::(\w+), \w+, \w+, (\d+), (\d+)>", src)
    if len(loads) < 2:
        continue
    (ta, m, k), (tb, k2, n) = loads[0], loads[1]
    dis = subprocess.run([os.path.join(R, "lib", "llvm", "bin", "llvm-objdump"), "-d", f"--mcpu={sys.argv[2]}", hs], capture_output=True, text=True).stdout
    ops = {}
    for line in dis.splitlines():
        w = line.split()
        if w and (w[0].startswith("v_wmma") or w[0].startswith("v_mfma")):
            ops[w[0]] = ops.get(w[0], 0) + 1
    rows.append((ta, int(m), int(n), int(k), int(bd), ops))
for r in sorted(rows, key=lambda r: r[:5]):
    print(f"{r[0]:9s} M{r[1]:3d} N{r[2]:3d} K{r[3]:3d} bd{r[4]:4d}: {r[5] or 'scalar'}")
```

`cells_bf16.py`:
```python
# bf16-input/float32-output and fp16 tile_matmul modules (forward only) per (tile, block_dim), compiled
# through Warp's hipRTC path for other targets; prints matrix instructions per module and an md5 of each
# code object. Usage: python cells_bf16.py <outdir> <tag> <gfx>...
import glob
import hashlib
import os
import subprocess
import sys

import warp as wp
import warp._src.build as build

wp.init()
CELLS = [(16, 64), (32, 64), (64, 128), (64, 256)]
R = subprocess.check_output([sys.executable, "-m", "rocm_sdk", "path", "--root"]).decode().strip()


def make(M, dt):
    if dt == "bf16":

        @wp.kernel(module="unique", module_options={"enable_backward": False})
        def mm(A: wp.array2d(dtype=wp.bfloat16), B: wp.array2d(dtype=wp.bfloat16), C: wp.array2d(dtype=float)):
            c = wp.tile_zeros(shape=(M, M), dtype=float)
            wp.tile_matmul(wp.tile_load(A, shape=(M, M)), wp.tile_load(B, shape=(M, M)), c)
            wp.tile_store(C, c)

    else:

        @wp.kernel(module="unique", module_options={"enable_backward": False})
        def mm(A: wp.array2d(dtype=wp.float16), B: wp.array2d(dtype=wp.float16), C: wp.array2d(dtype=wp.float16)):
            wp.tile_store(C, wp.tile_matmul(wp.tile_load(A, shape=(M, M)), wp.tile_load(B, shape=(M, M))))

    return mm


out_dir, tag = sys.argv[1], sys.argv[2]
os.makedirs(out_dir, exist_ok=True)
for dt in ("bf16", "fp16"):
    for M, bd in CELLS:
        k = make(M, dt)
        before = set(glob.glob(os.path.join(wp.config.kernel_cache_dir, "**", "*.cu"), recursive=True))
        wp.load_module(k.module, device="cuda:0", block_dim=bd)
        src = sorted(set(glob.glob(os.path.join(wp.config.kernel_cache_dir, "**", "*.cu"), recursive=True)) - before)[-1]
        for target in sys.argv[3:]:
            out = os.path.join(out_dir, f"{tag}_{dt}_mm{M}_bd{bd}_{target}.hsaco")
            build.build_cuda(src, 0, out, pch_dir=None, hip_target=target)
            dis = subprocess.run([R + "/lib/llvm/bin/llvm-objdump", "-d", f"--mcpu={target}", out], capture_output=True, text=True).stdout
            ops = {}
            for line in dis.splitlines():
                w = line.split()
                if w and (w[0].startswith("v_wmma") or w[0].startswith("v_mfma")):
                    ops[w[0]] = ops.get(w[0], 0) + 1
            md5 = hashlib.md5(open(out, "rb").read()).hexdigest()[:12]
            print(f"{tag} {dt} {M}x{M} bd{bd:3d} {target:7s} md5 {md5} {ops or 'scalar'}", flush=True)
```

`fatbin_sizes.py`:
```python
# Sum the device code objects embedded in a HIP DLL per offload target, by parsing every
# __CLANG_OFFLOAD_BUNDLE__ header in the file (one bundle per .cu translation unit).
# Usage: python fatbin_sizes.py <dll>
import collections
import struct
import sys
import zlib

data = open(sys.argv[1], "rb").read()
magic = b"__CLANG_OFFLOAD_BUNDLE__"
per = collections.Counter()
per_z = collections.Counter()
bundles = 0
pos = data.find(magic)
compressed = data.count(b"CCOB")
while pos >= 0:
    n = struct.unpack_from("<Q", data, pos + 24)[0]
    off = pos + 32
    for _ in range(n):
        o, size, tl = struct.unpack_from("<QQQ", data, off)
        triple = data[off + 24 : off + 24 + tl].decode()
        off += 24 + tl
        tgt = triple.split("-")[-1] if "amdgcn" in triple else triple
        per[tgt] += size
        per_z[tgt] += len(zlib.compress(data[pos + o : pos + o + size], 6))
    bundles += 1
    pos = data.find(magic, pos + 1)
total = sum(per.values())
print(f"file {len(data) / 1e6:.1f} MB, bundles {bundles}, CCOB markers {compressed}, device code {total / 1e6:.1f} MB")
for t, s in sorted(per.items(), key=lambda kv: -kv[1]):
    print(f"  {t:28s} {s / 1e6:7.2f} MB  (zlib -6: {per_z[t] / 1e6:6.2f} MB)")
```

### Commits 3 and 4 (after the validation loop)
- `9cc2ab02` [ROCm] Document building for AMD GPUs: `docs/user_guide/installation.rst` +15, a paragraph and code block after the CUDA build options in "Building from Source" (ROCm Python packages install line, `python build_lib.py`, detection order, both-toolkits error, default and `--hip-arch` target lists, the Windows toolchain, no MathDx). The README defers build steps to the docs, so it is unchanged.
- `fda2f89a` [ROCm] Fix device assert in debug kernels compiled on Windows: `crt.h` +7. Found by the loop: test_map's three `TestMapDebug` cases failed with "HIP kernel build failed with error code 6", hipRTC log `builtin.h:2032:26: error: use of undeclared identifier '__assert_fail'`. The HIP headers (`hip/amd_detail/hip_assert.h`) declare `__assert_fail` only off Windows; on Windows they declare `_wassert(const wchar_t*, const wchar_t*, unsigned)`, which only traps. The hipRTC `assert` now has an `#elif defined(_WIN32)` branch: `printf` of the expression and location, then `__builtin_trap()`. Inside `#if defined(__HIPCC_RTC__)`, so the DLL build, CUDA and Linux are unaffected. After the fix (fresh cache): test_map Ran 45, OK.
- Both pushed; `advance-head` to `fda2f89a`. jargon clean, audit-commits OK, the four commits carry no CR.

### Validation at `6761183c` (gfx1151-only `warp.dll`, rebuilt at that head; `fda2f89a` re-checked for test_map only)
- Capability line: `gfx1151 True True True` (textures supported, hardware, 3D hardware). Loaded runtime: `_rocm_sdk_core\bin\amdhip64_7.dll`, `hiprtc0717.dll`, `amd_comgr.dll` (psutil), never System32.
- gfx11 fp16 WMMA layout probe (`wmma_layout.hip`, Porting session 10 version): "WMMA family 11", six layout lines 0 of 256 wrong, accumulation 0 over the bound (worst 0.40/0.39/0.21 of the bound).
- `sdf_probe.py cuda:0`: PASS (SDF vs CPU sampler 4.78e-05, vs analytic 1.31e-03, RGBA8 2.52e-03). `tex_alias.py cuda:0`: PASS (plain 0 of 2048 wrong, graph capture 0 of 2048 wrong).
- **82-pattern set**, `agent_space/warp-win/p6/run82.sh`: one pattern at a time, `python -m warp.tests -s autodetect -p "<pattern>.py" -m 2`, 900 s timeout each, under `utils/timeit.sh warp-1.14 test`, stopping on a timeout or a device-lost/launch-failure line. Two restarts, neither a GPU event: after pattern 19 (my first stop regex matched ordinary test names; fixed, resumed at 20) and after pattern 39 (the fem timeout, below; resumed at 40). No driver reset, no device-lost line, no GPU hang. A quoting slip made all patterns share one kernel cache instead of one per pattern; it was one build, so the headers were the same throughout. Result: **78 patterns OK, 4 not OK** (test_map fixed since):

| pattern | ran | result vs gfx1100 | pattern | ran | result vs gfx1100 |
|---|---|---|---|---|---|
| test_codegen | 117 | OK = | test_capture_mode | 2 | OK = |
| test_snippet | 12 | OK = | test_conditional_captures | 32 | OK, 16 skipped = |
| test_arithmetic | 126 | OK = | test_conditional | 50 | OK = |
| test_math | 16 | OK = | test_async | 2373 | OK = |
| test_fp16 | 6 | OK = | test_streams | 24 | **1 failure**, 3 skipped (b) |
| test_bf16 | 48 | OK, 5 skipped (env: ml_dtypes) | test_mempool | 7 | OK, 2 skipped = |
| test_atomic | 164 | OK = | test_pinned | 1 | OK = |
| test_atomic_bitwise | 6 | OK = | test_unified_memory | 27 | OK, 8 skipped = |
| test_atomic_cas | 48 | OK = | test_occupancy | 6 | OK = |
| test_array | 138 | OK (139: one torch test not registered) | test_allocator | 16 | OK, 3 skipped = |
| test_copy | 14 | OK = | test_allocation_tracker | 34 | OK = |
| test_launch | 32 | OK = | test_dlpack | 8 | OK (16: torch half not registered) |
| test_fast_math | 12 | OK, 1 skipped = | test_torch | 0 | (34: torch not installed) |
| test_grad | 41 | OK = | test_map | 45 | **3 errors**, fixed in `fda2f89a` (45 OK) |
| test_kernel_cache | 5 | OK = | test_reload | 14 | OK = |
| test_module_hashing | 8 | OK = | test_cuda_arch_suffix | 10 | OK, 10 skipped = |
| test_print | 10 | OK, 8 skipped (upstream Windows stdout skip) | test_tile | 126 | OK = |
| test_mat* | 601 | OK = | test_tile_atomic_bitwise | 3 | OK = |
| test_vec* | 531 | OK = | test_tile_cholesky | 33 | OK = |
| test_quat* | 173 | OK = | test_tile_cholesky_no_mathdx | 16 | OK = |
| test_spatial* | 207 | OK = | test_tile_empty | 20 | OK = |
| test_rand | 10 | OK = | test_tile_fft | 20 | OK, 8 skipped = |
| test_noise | 14 | OK = | test_tile_fft_no_mathdx | 16 | OK = |
| test_clang_cuda | 7 | OK, 6 skipped = | test_tile_func_arg | 27 | OK = |
| test_runlength_encode | 16 | OK = | test_tile_fused_ops | 36 | OK = |
| test_array_reduce | 16 | OK = | test_tile_large_offsets | 1 | OK = |
| test_sparse | 47 | OK = | test_tile_load | 64 | OK = |
| test_linear_solvers | 34 | OK = | test_tile_mathdx | 6 | OK = |
| test_coloring | 3 | OK, 1 skipped (env: usd-core) | test_tile_matmul | 25 | OK = |
| test_bvh | 13 | OK = | test_tile_matmul_no_mathdx | 22 | OK = |
| test_grouped_bvh | 11 | **1 error** (a) | test_tile_mlp | 0 | (3: torch not installed) |
| test_hash_grid | 23 | OK = | test_tile_reduce | 62 | OK, 2 skipped (64 KiB capability) = |
| test_marching_cubes | 17 | OK = | test_tile_shared_memory | 30 | OK, 2 skipped (64 KiB) = |
| test_mesh* | 49 | OK, 18 skipped (env: usd-core) | test_tile_solve | 5 | OK, 1 skipped = |
| test_volume | 59 | OK = | test_tile_solve_no_mathdx | 28 | OK = |
| test_volume_write | 5 | OK = | test_tile_sort | 2 | OK = |
| test_apic | 52 | OK = | test_tile_stack | 40 | OK = |
| test_apic_mesh | 12 | OK = | test_tile_view | 16 | OK, 1 skipped (64 KiB) = |
| test_fem* | 108 | **99 ok, then the worker pool broke** (c) | test_block_dim_dispatch | 2 | OK = |
| test_graph | 35 | OK, 3 skipped = | test_template_launch_bounds | 36 | OK = |
| test_array_fill_capture | 3 | OK = | test_texture | 120 | OK, 1 skipped = |

"=" means the same counts as gfx1100 (Evidence session 2 B.5 table, with the later tile_matmul 25 and the 64 KiB capability skips). Tile set (22 tile files + 2): 636 run, 14 skipped, 0 failures, which is gfx1100's 639/15 minus test_tile_mlp (torch). This laptop's venv has no torch, usd-core, ml_dtypes or rmm (the Linux hosts had torch and usd-core); every skip reason above was read from the logs.

Non-passing items:
- **(a) test_grouped_bvh: `test_grouped_capture_bvh_rebuild_cuda_0` ERROR**, `RuntimeError: Graph launch error: Warp CUDA error 1: invalid argument (in function wp_cuda_graph_launch, warp.cu:4411)`; the next test then prints the same sticky error from `radix_sort_pairs_device` (`sort.cu:84`) and passes. That test alone with `AMD_LOG_LEVEL=1`: **passes** (Ran 1, OK; the log has only PAL "Virtual address present, but not mapped yet" lines and trap-handler setup warnings). So it is order- or timing-dependent, like the test_mesh crash (replay of a graph holding a BVH rebuild with stream-ordered allocations inside the capture). Not narrowed further. Linux passes this file on every host.
- **(b) test_streams: `test_event_external_cuda_0` FAIL** (`c` all 0, expected 20). Shown to be a race in the upstream test, exposed by Windows submission timing (`agent_space/warp-win/p6/event_ext.py`): the arrays are zeroed by `wp.zeros` on the device's stream, then the two graphs run on two new `wp.Stream()`s with nothing ordering them after the zeroing, and `c.numpy()` reads on the device stream without waiting for `stream2`. Only the first such round in a process fails (`a`, `b`, `c` all 0 even after `wp.synchronize()`: the memsets landed after the graph launches); a second identical round passes; the same first round passes when `wp.synchronize()` follows the `wp.zeros` (mode `zeros-synced`). A plain graph launched 10 times on a new stream as the first graph work of the process gave `a[0] = 1` (9 of 10 increments overwritten by the late memset); launched on the device stream it gave 10. A standalone HIP program with a synchronous memset (`graph_newstream.hip`) passes, consistent with that explanation. Nothing orders the two streams on CUDA either (two blocking non-null streams), so this is not a port defect; the test is left unchanged.
- **(c) test_fem*: the process pool broke.** 108 tests in 10 suites across 2 processes; 99 reported ok; then "Warning: Process pool broken during parallel execution. Switching to isolated single-process fallback." while the workers were in `test_mesh_multi_env_cpu` and `test_cube_shape_functions`. The runner's isolated fallback then failed on Windows for every suite ("Failed to create isolated process ...: [WinError 2] The system cannot find the file specified", a Windows problem of upstream's test runner), so all 108 were marked crashed. The pattern ended at 899 s and the 900 s timeout fired during the runner's cache cleanup (rc 124): not a GPU hang. Which test killed the worker is unknown (no faulthandler output). `test_cube_shape_functions` is one of the two known isolation failures on Linux ("Unexpected output"); whether it or a GPU test crashed the worker here is open. fem was not re-run (each test once on this laptop).
- test_map: fixed (above). test_mesh*: all 49 passed this time, `test_mesh_refit_graph_cuda_0` included (consistent with an intermittent crash).
- **CPU-only suite** (GPU hidden; run alongside the second half of the loop because it does not touch the GPU): `HIP_VISIBLE_DEVICES=99 CUDA_VISIBLE_DEVICES=99 python -m warp.tests -s autodetect -k cpu -m 4` -> **Ran 2276, OK (skipped=34)**, no `cuda:0` in the log, 477 s. Linux: 2294 OK (skipped=15). The 18 fewer tests are torch and ml_dtypes registrations; the extra skips are "Requires usd-core" 13, the upstream Windows stdout skip 5, and one more whose reason line wraps.
- Not run on this laptop: `mma_check.py` (fp16/fp32/fp64, 165 kernels; the bf16 check ran), tests on the 22-target DLL beyond init (experiment), examples, benchmarks (forbidden here).

`event_ext.py` (final form):
```python
# test_streams.test_event_external, with the read done (a) as the test does it and (b) after waiting for
# stream2 explicitly. Usage: python event_ext.py
import numpy as np

import warp as wp
from warp.tests.cuda.test_streams import inc, sum

wp.init()
with wp.ScopedDevice("cuda:0"):
    for mode in __import__("sys").argv[1:]:
        if mode.startswith("noevent"):
            n = 1_000_000
            a = wp.zeros(n, dtype=float)
            with wp.ScopedCapture() as cap:
                wp.launch(inc, dim=n, inputs=[a])
            st = wp.Stream() if mode == "noevent-stream" else None
            for _ in range(10):
                wp.capture_launch(cap.graph, stream=st)
            wp.synchronize()
            print(mode, "a[0] =", a.numpy()[0], "expected 10; c[0] n/a", flush=True)
            continue
        event = wp.Event()
        n = 1_000_000
        a, b, c = (wp.zeros(n, dtype=float) for _ in range(3))
        if mode == "zeros-synced":
            wp.synchronize()
        with wp.ScopedCapture() as capture1:
            wp.launch(inc, dim=n, inputs=[a])
            wp.launch(inc, dim=n, inputs=[b])
            wp.record_event(event, external=True)
        with wp.ScopedCapture() as capture2:
            wp.wait_event(event, external=True)
            wp.launch(sum, dim=n, inputs=[a, b, c])
        stream1, stream2 = wp.Stream(), wp.Stream()
        for _ in range(10):
            wp.capture_launch(capture1.graph, stream=stream1)
            wp.capture_launch(capture2.graph, stream=stream2)
        if mode == "sync-stream2":
            wp.synchronize_stream(stream2)
        got = c.numpy()
        wp.synchronize()
        print(mode, "a[0] =", a.numpy()[0], "b[0] =", b.numpy()[0], "c[0] =", got[0], "expected 20; after a device sync c[0] =", c.numpy()[0], flush=True)
```

## Lesson for cuda-to-rocm: on Windows, HIP device code has no `__assert_fail`
Reference file: `references/windows.md`. Code that supplies its own device `assert` for a runtime compiler (hipRTC has no `<assert.h>`) often expands it to `__assert_fail(expr, file, line, func)`, the glibc name that HIP's `hip/amd_detail/hip_assert.h` defines as a printing device function. That header declares `__assert_fail` only when `_WIN32` is not defined; on Windows it declares `_wassert(const wchar_t*, const wchar_t*, unsigned)` instead, which traps without printing. Debug-mode kernels then fail to compile only on Windows ("use of undeclared identifier '__assert_fail'"), while release builds (`NDEBUG`) hide it. Give the Windows branch its own expansion (`printf`, then `__builtin_trap()`), or call `_wassert` with wide strings. (warp-1.14, Warp's `crt.h`)

## Lesson for cuda-to-rocm: Windows HIP submission timing exposes missing cross-stream ordering
Reference file: `references/diagnosis.md`. On Windows the HIP runtime can submit work on one stream noticeably later than work issued afterwards on another stream. Tests that zero or fill buffers on one stream and then launch kernels or graphs on a freshly created second stream, with no event or synchronization between them, pass on Linux and on CUDA by timing and fail on Windows with results that look like "the kernel never ran" (the late fill overwrites the output). Before suspecting the runtime, add a `synchronize` after the fill: if the failure disappears, the test has a race. Warp's `test_event_external` is an example: only the first round in a process failed. (warp-1.14)

### Next step (head `fda2f89a`; written by the windows-gfx1151 porter)
The project stays in `porting`, lock held by windows-gfx1151, as directed: Phase 7 (wheels), the Linux nvcc check and the hosts below still follow, and the lock goes back to linux-gfx950 by Jeff's decision. Fork commits this session, `a8e95dcb..fda2f89a`: `13e4e46e` Windows HIP build (Windows-only code in `build_dll.py` and the DLL loader, plus a `builtin.h` guard that is a no-op off MSVC); `6761183c` bf16 WMMA (gfx11/gfx12 device code only); `9cc2ab02` docs; `fda2f89a` Windows hipRTC debug assert (Windows-only).

**Open on this platform (windows-gfx1151):**
1. The three graph-related Windows findings: the intermittent `test_mesh_refit_graph` access violation, the order-dependent `test_grouped_capture_bvh_rebuild` "invalid argument" from `hipGraphLaunch`, and the fem worker crash (unknown test). The first two share a shape (replay of graphs that hold stream-ordered allocations around a hipCUB/rocPRIM call, after other graphs were destroyed); the simple standalone reproducer did not fail. Suggested next step: one crashing run each under `AMD_LOG_LEVEL=3` and faulthandler, then a reproducer with Warp's real call sequence; then either a port fix or a `rocm-bug-report` registration plus a capability-named skip.
2. fem* once more in a single process (`-m 1`, or the file list run one at a time) to find which test kills the worker; `mma_check.py` (165 kernels).
3. `test_event_external`: upstream test race, not a port defect; no action unless Jeff wants a test-only fix carried.

**gfx1100 (wave32, gfx11 bf16 WMMA; must re-run at `fda2f89a`; only `6761183c` affects Linux):**
1. `wmma_layout_bf16.hip` (above) built for gfx1100 (`clang++ -x hip --rocm-path=$R --offload-arch=gfx1100 -O2 -w wmma_layout_bf16.hip -o wmma_layout_bf16 -L$R/lib -lamdhip64 -Wl,-rpath,$R/lib`): expect "bf16 WMMA family 11", six layout lines 0 of 256 wrong, accumulation 0 over the bound. If any line is wrong, stop and record.
2. `mma_check_bf16.py` and `mma_check.py` (Porting session 10 versions): FAILURES 0 each. `isa_count.py <cache> gfx1100` on the `mma_check_bf16.py` cache: `v_wmma_f32_16x16x16_bf16` in 34 code objects, 64x64x64 at block_dim 256 scalar (as the gfx1151 table above).
3. test_tile_matmul (25 OK) and the 24-file tile set (639 run, 15 skipped, 0 failures).
4. **bf16 benchmark against the scalar tree** (unmeasured anywhere so far): the Phase 4 `bench_tile_matmul.py` has no bf16 mode, so add a bf16-input/float32-output forward kernel next to the fp16 one (same tiles 16/32/64, same block_dims as the fp16 cells; forward only, tile_matmul rejects a bf16 backward); three runs per tree with the load monitor; scalar tree = a worktree at `fda2f89a` with `warp/native/tile_matmul_mma.h` checked out from `a8e95dcb` (bf16 scalar on gfx11/gfx12; same `warp.so`). Expect bf16 like the fp16 forward cells; **any bf16 cell below 1.0x is a finding**; 64x64 at block_dim 256 is scalar in both trees by construction.
5. The 82-pattern set and the CPU-only suite (expect 81 OK + the fem pair; 2294 OK).

**gfx1201 (wave32, gfx12 bf16 WMMA, layout assumed from gfx12 fp16, never measured):**
1. `wmma_layout_bf16.hip` built for gfx1201: expect "bf16 WMMA family 12", all layout lines 0 wrong, accumulation within the bound. **If it fails, the gfx12 `op<bfloat16>` in `tile_matmul_mma.h` must be removed before anything else** (porter's job; the validator records the counts).
2. `mma_check_bf16.py`, `mma_check.py`: FAILURES 0; `isa_count.py <cache> gfx1201`: bf16 WMMA in all 35 cases with 16-multiple dimensions (gfx12 has no scalar exclusion; `cells_bf16.py` showed 8 instructions at 64x64 block_dim 256).
3. test_tile_matmul, the tile set, the bf16 benchmark as for gfx1100, the 82-pattern set, the CPU-only suite.

**gfx950, gfx942, gfx90a:** no device-code change for them (every MFMA code object byte-identical between `13e4e46e` and `6761183c` in `cells_bf16.py`; `13e4e46e` and `fda2f89a` change only Windows code; `9cc2ab02` is documentation). Whether their `e78ee32f` passes carry forward is a validation decision.

**linux-gfx950 (after the lock returns):** the nvcc 12.8 CUDA compile at the Phase 6 head, as every round (the Windows CUDA compile above passed at `13e4e46e`; `crt.h`'s change is inside `__HIPCC_RTC__`).

**Phase 7 (wheels):** `setup.py` must ship `warp/bin/warp.dll.rocm` in a Windows ROCm wheel, or the installed wheel's loader cannot tell it is a HIP build and binds the System32 runtime. A wheel built on CI records the build machine's ROCm root; harmless, because the loader prefers `rocm_sdk`, which the wheel depends on.

**For Jeff (no decision taken):** the default Windows target list costs about 13 MB uncompressed and 1.07 MB zipped per target (22 targets: 290.5 MB DLL, 24.4 MB zipped); `--offload-compress` is the untried alternative to a shorter list.

## Porting session 14 (2026-11-19, windows-gfx1151 laptop, porter): Windows graph-replay findings

Bounded diagnosis round inside the windows-gfx1151 `porting` lock (lock and stage untouched). Fork `release/1.14.0.amd0` advanced `fda2f89a..de4e07a3` (pushed, `advance-head` recorded). Environment as session 13 (`agent_space/warp-win/p6env.sh`, ROCm 10.2 wheels, HIP 7.17.26434). Scripts in `agent_space/warp-win/p14/`, logs in `agent_space/warp-win/logs/` (`p14_*`, `p14fem/`, `p14fix/`). No driver reset, device-lost line or display glitch at any point.

### Items 2 and 3: one HIP runtime race, Windows only (verdict: runtime defect; port workaround committed)
**Cause (read from the HIP source, confirmed by logs and a standalone reproducer).** `GraphExec::Run` (clr `hipamd/src/hip_graph_internal.cpp`; TheRock's April checkout and rocm-systems `develop` today, `GraphExecClassic::Run`, have the same code) does, for a graph instantiated with `hipGraphInstantiateFlagAutoFreeOnLaunch` (Warp instantiates every graph so):
```cpp
parentGraph->FreeAllMemory(launch_stream);   // on the launching thread
parentGraph->memalloc_nodes_ = 0;
if (!AMD_DIRECT_DISPATCH) launch_stream->finish();
...
if (repeatLaunch_ && GetMemAllocNodeCount() > 0) return hipErrorInvalidValue;
```
`AMD_DIRECT_DISPATCH` defaults to `IS_LINUX` (`rocclr/utils/flags.hpp`), so on Windows it is off (the AMD_LOG_LEVEL=3 log prints `Direct Dispatch: 0`) and the graph's commands, including the mem-alloc node (`VirtualMemAllocNode::submit`, increments `memalloc_nodes_`) and mem-free node (`VirtualMemFreeNode::submit`, returns the allocation to the pool and decrements it), run on the runtime's worker thread. When the next `hipGraphLaunch` of the same exec arrives while the previous launch has executed its alloc but its free node still waits behind a running kernel, `FreeAllMemory` on the launching thread frees that allocation first; the worker then frees memory the pool no longer owns (`:1:hip_graph_internal.hpp:3823: Memory didn't belong to any pool!`) and `memalloc_nodes_` (a plain `uint32_t` written from two threads) underflows. Outcomes: an access violation in the worker thread at `amdhip64_7.dll + 0x160C6C` (item 3), or `hipErrorInvalidValue` from the repeat-launch check (item 2). The comment in that block already names the race for the unmap, but the wait comes after the free. On Linux the runtime dispatches directly on the calling thread, which is why every Linux GPU passes.

**Standalone reproducer** `p14/graph_autofree_race.hip` (below; `clang++ -x hip --rocm-path=$R --rocm-device-lib-path=$R/lib/llvm/amdgcn/bitcode --offload-arch=gfx1151 -O2 -w graph_autofree_race.hip -o graph_autofree_race.exe -L$R/lib -lamdhip64`, `R=$(python -m rocm_sdk path --root)`, TheRock `amdhip64_7.dll` next to the exe, md5 identical to the venv's): capture `hipMallocAsync` -> 2 ms spin kernel -> small kernel -> `hipFreeAsync`, instantiate with AutoFreeOnLaunch, launch 10 times back to back.
- `graph_autofree_race 10 2000`: **access violation at `amdhip64_7.dll + 0x160C6C`** after launch 1 (same address as the Warp crash), twice in two runs (once plain, once under `AMD_LOG_LEVEL=3`, log `logs/p14_repro_log3.txt`: launch 1's `hipGraphLaunch` logs `Pool FreeMem` while launch 0's `spin` kernel runs, then `Memory didn't belong to any pool!`, launch 1 takes 17240 us (the `finish()`), crash during launch 2).
- `graph_autofree_race 10 2000 sync` (hipStreamSynchronize before each launch): PASS, 0 of 10.
- `graph_autofree_race 10 200` (short kernel, free node done before the next launch): PASS. The session 13 reproducer `graph_memnode.hip` used tiny kernels, hence no failure.

**Warp-level evidence.**
- Item 3, crashing run 4 of 6 (1 of this session's 3): `AMD_LOG_LEVEL=3 PYTHONFAULTHANDLER=1 python p6/mesh_seq.py graph` on `fda2f89a` -> access violation at `amdhip64_7.dll + 0x160C6C` during sah replay 8 (main thread in `capture_launch`, context.py:10603); one launch earlier the log has `Pool FreeMem` from the launching thread while the rocPRIM scan of the previous replay runs, then `Memory didn't belong to any pool!` (`logs/p14_meshbase_log3_1.log`). It crashed on the first constructor (sah) with no earlier graph destroyed: session 13's "needs a destroyed previous graph" was timing. The other two crashing runs were not needed.
- Item 2: the grouped BVH capture holds 9 mem-alloc and 9 mem-free nodes, 20 kernels, 5 memcpy, 1 memset (`p14/gb_burst.py 1 dot`, `capture_debug_dot_print`), so it is in the same family. Failing runs used: **0 of 6**; it did not fail in four tries on `fda2f89a` (`python -m unittest -v warp.tests.geometry.test_grouped_bvh` with `AMD_LOG_LEVEL=1` twice, the runner `-p test_grouped_bvh.py -m 2` once, `gb_burst.py 5` back-to-back replays once), because the test's host-side `assign` between launches usually lets the previous launch finish. So the smallest failing prefix and an AMD_LOG_LEVEL=3 log of item 2 itself were not obtained; the attribution rests on the mechanism (the only path in `hipGraphLaunch` that returns `hipErrorInvalidValue` for a valid exec and stream is the repeat-launch `memalloc_nodes_` check, which the race corrupts) and on the fix below removing the window.
- No known report covers it: deferred.json's HIP graph entries (rocm-systems 16184 null-stream hipMallocAsync during capture, 16188 graph memcpy on device != 0, and the others) are different defects; `gh search issues --repo ROCm/rocm-systems AutoFreeOnLaunch` and "hipGraphLaunch invalid" found nothing.

**Port fix `de4e07a3` [ROCm] Wait for the previous graph launch on Windows**: `warp/native/hip_compat.h` +18. Under `#if defined(_WIN32)`, `cudaGraphLaunch` maps to an inline `wp_hip_graph_launch` that calls `hipStreamSynchronize(stream)` before `hipGraphLaunch` when `hipStreamIsCapturing` reports the stream is not capturing; elsewhere it stays `hipGraphLaunch`. Covers both call sites (`warp.cu` `wp_cuda_graph_launch`, `apic.cu` `wp_apic_launch`) without touching them. Cost: on Windows the runtime already blocks in `launch_stream->finish()` inside every AutoFreeOnLaunch launch of a non-empty graph, so moving the wait ahead of the free adds about one extra synchronize call per launch; Linux HIP and CUDA builds unchanged (the file is HIP-only, the branch Windows-only). clang-format 22.1.2 clean, jargon clean, commit CR-free.
- Built in a separate worktree (`agent_space/warp-win/fixwt`, `python build_lib.py --hip-arch gfx1151 -j 8`, "Warp build succeeded"), tested with `PYTHONPATH` pointing at it:
  - `mesh_seq.py graph` 3 runs: **3 of 3 pass** (30 replays each); the same command on `fda2f89a` crashed in session 13 runs 1, 3 and 9 and in this session's one run.
  - `p14/fix_pats.sh` (one pattern at a time, `-m 2`): test_grouped_bvh 11 OK; test_mesh* 49 OK (18 skipped, usd-core); test_bvh 13 OK; test_graph 35 OK (3 skipped); test_array_fill_capture 3 OK; test_capture_mode 2 OK; test_conditional_captures 32 OK (16 skipped); test_mempool 7 OK (2 skipped); test_allocator 16 OK (3 skipped); test_allocation_tracker 34 OK; test_sparse 47 OK; test_apic 52 OK; test_streams 24 run, 1 failure = the known `test_event_external` upstream race (session 13 item b), 3 skipped. Same counts as session 13's table.
- The fork clone's own `warp/bin/warp.dll` is still the `fda2f89a` build (the unfixed runs above and the fem run below used it); rebuild before further testing in that tree.
- **Proposed if Jeff prefers no workaround:** a skip would be capability-named, e.g. "graph memory nodes replayed back to back race in the HIP runtime without direct dispatch" on `test_mesh_refit_graph` and `test_grouped_capture_bvh_rebuild`; not added. Registered `hip-graph-autofree-relaunch-races-worker-memfree` (rocm-bug-report, HIP, unruled).

### Item 1: test_fem* (verdict: the same runtime race; fixed by `de4e07a3`; the exact test that killed session 13's pool worker is not named)
`p14/fem_files.sh` (below): each fem file in its own single process, `python -X faulthandler -m unittest -v warp.tests.fem.<file>`, `PYTHONFAULTHANDLER=1`, 900 s timeout each, on `fda2f89a`, fresh kernel cache:

| file | ran | result |
|---|---|---|
| test_fem_shape | 4 | 1 failure: `test_cube_shape_functions`, "Unexpected output" holding the init greeting (known upstream isolation failure) |
| test_fem_quadrature | 5 | 1 failure: `test_gimp_quadrature`, same (known) |
| test_fem_geometry | 22 | OK (1 skipped) |
| test_fem_field | 6 | OK |
| test_fem_linalg | 5 | OK |
| test_fem_integrate | 17 | OK |
| test_fem_fp64 | 16 | OK |
| test_fem_multi_env | 10 | OK (1 skipped) |
| test_fem_examples | 23 | **1 failure: `test_fem.example_diffusion_mgpu_cuda_0`**, the example subprocess died with 3221225477 (0xC0000005): access violation at `amdhip64_7.dll + 0x160C6C` on the runtime worker thread, main thread in `capture_launch` (context.py:10603) from the capturable CG loop (`optim/linear.py:1796 _run_solver_loop`, called from `examples/fem/utils.py:349 bsr_cg`) |

108 tests, the session 13 total. No in-process crash in single-process mode. `python -m warp.examples.fem.example_diffusion_mgpu --device cuda:0 --headless`: on `fda2f89a` 1 more run, crashed again at the same address (2 of 2); with `de4e07a3` (fixwt) **2 of 2 pass**. With `de4e07a3`: `python -X faulthandler -m unittest warp.tests.fem.test_fem_examples warp.tests.fem.test_fem_integrate warp.tests.fem.test_fem_multi_env` -> **Ran 50, OK (skipped=1)**.
- Session 13's pool break: the example tests run their example as a subprocess, so the example crash alone fails one test rather than killing a worker. The pool worker that died was then in `test_mesh_multi_env_cpu` or `test_cube_shape_functions`, neither of which launches a graph; since the race crashes the runtime's worker thread asynchronously, after the Python call returned, a crash from an earlier in-process GPU graph test (the capturable CG solvers in test_fem_integrate / test_fem_multi_env and others replay graphs with memory nodes) landing during the next test fits, but this was not reproduced. The leaning: the same race, removed by the fix. Not re-run as the 2-worker pattern this round (budget).

### Scripts from this session

`graph_autofree_race.hip`:
```cpp
// A graph holding a stream-ordered allocation (mem alloc node -> long kernel using it -> mem free
// node), instantiated with hipGraphInstantiateFlagAutoFreeOnLaunch and launched back to back.
// With the alloc executed and the free node still waiting for the kernel, the next hipGraphLaunch
// frees the graph's memory and resets the graph's outstanding-allocation count on the calling
// thread while the runtime's worker thread still runs the previous launch's free node.
// Usage: graph_autofree_race [launches] [spin_us] [sync]
//   sync: hipStreamSynchronize before every launch (the workaround)
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { printf("%s:%d %s -> %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e)); return 1; } } while (0)

__global__ void spin(float* buf, int n, long long cycles)
{
    long long t0 = wall_clock64();
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float v = 0.f;
    while (wall_clock64() - t0 < cycles) v += 1.f;
    if (i < n) buf[i] = v;
}
__global__ void reduce(const float* buf, float* out, int n)
{
    if (blockIdx.x == 0 && threadIdx.x == 0) out[0] = buf[n - 1] > 0.f ? 1.f : 0.f;
}

int main(int argc, char** argv)
{
    int launches = argc > 1 ? atoi(argv[1]) : 10;
    int spin_us = argc > 2 ? atoi(argv[2]) : 2000;
    bool sync = argc > 3 && !strcmp(argv[3], "sync");
    const int n = 1 << 16;
    int rate_khz = 0;
    CK(hipSetDevice(0));
    CK(hipDeviceGetAttribute(&rate_khz, hipDeviceAttributeWallClockRate, 0));
    long long cycles = (long long)rate_khz * spin_us / 1000;
    hipStream_t s;
    CK(hipStreamCreate(&s));
    float* out;
    CK(hipMalloc(&out, 4));

    hipGraph_t g;
    hipGraphExec_t ge;
    CK(hipStreamBeginCapture(s, hipStreamCaptureModeGlobal));
    float* tmp;
    CK(hipMallocAsync((void**)&tmp, n * 4, s));
    spin<<<n / 256, 256, 0, s>>>(tmp, n, cycles);
    reduce<<<1, 64, 0, s>>>(tmp, out, n);
    CK(hipFreeAsync(tmp, s));
    CK(hipStreamEndCapture(s, &g));
    CK(hipGraphInstantiateWithFlags(&ge, g, hipGraphInstantiateFlagAutoFreeOnLaunch));
    CK(hipGraphUpload(ge, s));

    int bad = 0;
    for (int i = 0; i < launches; ++i) {
        if (sync) CK(hipStreamSynchronize(s));
        hipError_t e = hipGraphLaunch(ge, s);
        printf("launch %d: %s\n", i, hipGetErrorString(e));
        fflush(stdout);
        if (e != hipSuccess) { ++bad; (void)hipGetLastError(); }
    }
    CK(hipStreamSynchronize(s));
    CK(hipGraphExecDestroy(ge));
    CK(hipGraphDestroy(g));
    printf("%s: %d of %d launches failed (spin %d us, %s)\n", bad ? "FAIL" : "PASS", bad, launches, spin_us,
           sync ? "sync before launch" : "back to back");
    return bad ? 1 : 0;
}
```

`fem_files.sh`:
```bash
#!/bin/bash
# fem test files one at a time, each in a single process (unittest -v, faulthandler), 900 s timeout each.
source /d/Develop/moat/agent_space/warp-win/p6env.sh
L=$LOGS/p14fem; mkdir -p $L
export PYTHONFAULTHANDLER=1
export WARP_CACHE_PATH='D:\Develop\moat\agent_space\warp-win\wpc14fem'
cd /d/Develop/moat/agent_space/warp-win/p14
for f in ${FILES:-test_fem_shape test_fem_quadrature test_fem_geometry test_fem_field test_fem_linalg test_fem_integrate test_fem_fp64 test_fem_multi_env test_fem_examples}; do
  s=$(date +%s)
  timeout 900 $VPY -X faulthandler -m unittest -v warp.tests.fem.$f > "$L/$f.log" 2>&1
  rc=$?
  echo "$f rc=$rc $(( $(date +%s) - s ))s $(grep -E '^Ran [0-9]+ tests|^OK|^FAILED' $L/$f.log | tr '\n' ' ')" >> $L/summary.txt
  if grep -qE "hipErrorDeviceLost|[Dd]evice lost|HSA_STATUS_ERROR" "$L/$f.log"; then echo "DRIVER-ERROR at $f" >> $L/summary.txt; break; fi
done
echo FINISHED >> $L/summary.txt
```

`gb_burst.py`:
```python
# test_grouped_bvh's capture (copies + grouped Bvh.rebuild() + query kernel), then the graph replayed
# back to back without host work between launches, to widen the window between one launch's graph
# memory-free node and the next launch. Usage: python gb_burst.py [launches]
import faulthandler
import sys

import numpy as np

faulthandler.enable()
import warp as wp
from warp.tests.geometry import test_grouped_bvh as tg

wp.init()
launches = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 5
with wp.ScopedDevice("cuda:0"):
    rng = np.random.default_rng(123)
    n, groups = 100000, 100
    lo_np, hi_np = tg.get_random_aabbs(n, np.zeros(3), 2, 0.01, rng)
    lo = wp.array(lo_np, dtype=wp.vec3)
    hi = wp.array(hi_np, dtype=wp.vec3)
    g = wp.array(np.arange(n, dtype=np.int32) % groups, dtype=int)
    bvh = wp.Bvh(lo, hi, groups=g)
    lo2, hi2 = wp.zeros_like(lo), wp.zeros_like(lo)
    lo2_np, hi2_np = tg.get_random_aabbs(n, np.zeros(3), 2, 0.01, rng)
    lo2.assign(lo2_np)
    hi2.assign(hi2_np)
    tlo_np, thi_np = tg.get_random_aabbs(10000, np.zeros(3), 2, 0.05, rng)
    tlo, thi = wp.array(tlo_np, dtype=wp.vec3), wp.array(thi_np, dtype=wp.vec3)
    tg_ = wp.array(np.arange(n, dtype=np.int32) % groups, dtype=int)
    counts, sums = wp.empty(10000, dtype=int), wp.empty(10000, dtype=int)
    wp.load_module(tg, device="cuda:0")
    with wp.ScopedCapture(force_module_load=False) as cap:
        wp.copy(lo, lo2)
        wp.copy(hi, hi2)
        bvh.rebuild()
        wp.launch(tg.compute_num_contact_with_checksums, dim=10000, inputs=[tlo, thi, tg_, bvh.id], outputs=[counts, sums])
    bad = 0
    for i in range(launches):
        try:
            wp.capture_launch(cap.graph)
            print("launch", i, "ok", flush=True)
        except RuntimeError as e:
            bad += 1
            print("launch", i, "FAILED:", e, flush=True)
    wp.synchronize_device()
    print("FAIL" if bad else "PASS", bad, "of", launches, flush=True)
if "dot" in sys.argv:
    wp.capture_debug_dot_print(cap.graph, "gb_graph.dot")
    txt = open("gb_graph.dot").read()
    import re
    print("node types:", {t: txt.count(t) for t in ("MEM_ALLOC", "MEM_FREE", "KERNEL", "MEMCPY", "MEMSET")})
```

`fix_pats.sh` is `run82.sh` with `PYTHONPATH` pointed at the fixwt worktree, its own kernel cache per pattern and the 13 patterns listed above. `gb_seq.py` (runs test_grouped_bvh functions in a chosen order with a print per graph launch) was written for the prefix search but not needed once the mechanism was known.

## Lesson for cuda-to-rocm: Windows HIP graph relaunch races its own memory-free nodes
Reference file: `references/windows.md` (or `references/diagnosis.md`). On Windows the HIP runtime does not dispatch directly: graph commands, including the mem-alloc and mem-free nodes a stream capture creates for `hipMallocAsync`/`hipFreeAsync`, run on a runtime worker thread. Relaunching a graph instantiated with `hipGraphInstantiateFlagAutoFreeOnLaunch` frees the graph's memory on the calling thread before it waits for the previous launch, so when the previous launch's free node is still queued behind a running kernel the worker frees memory twice. Symptoms, only on Windows and only for back-to-back replays: `hipGraphLaunch` returning `hipErrorInvalidValue` for a valid graph, or an access violation inside `amdhip64_7.dll` on a thread that is not yours (the same offset every time), with `Memory didn't belong to any pool!` at `AMD_LOG_LEVEL=1`. Workaround: on Windows, `hipStreamSynchronize` the stream right after each `hipGraphLaunch`, when the stream is not capturing. Waiting before each launch closes the relaunch race but leaves a second failure open: on a graph's first launch, a copy the caller issues right after the launch can find the memory the graph allocates invalid (`hipErrorInvalidValue`) until the launch has run its alloc nodes; a reproducer that copies from that memory after the first launch fails 5 of 5 runs with the wait before the launch and 0 of 5 with the wait after it. The runtime already blocks inside such a launch, so the wait costs little. A reproducer needs a kernel long enough (milliseconds) to keep the free node pending; tiny kernels hide it. (warp-1.14)

### Next step (head `de4e07a3`; written by the windows-gfx1151 porter, session 14)
Supersedes session 13's "Open on this platform" items 1 and 2; everything else in session 13's Next step stands, with `de4e07a3` added: it changes only Windows code (`hip_compat.h` under `_WIN32`), so Linux evidence and device code are unaffected; the nvcc check is unaffected (HIP-only header).

**Open on this platform (windows-gfx1151):**
1. Rebuild the fork clone's `warp.dll` at `de4e07a3` (`python build_lib.py --hip-arch gfx1151`) and re-run the 82-pattern set (`p6/run82.sh`, `-m 2`): expect test_grouped_bvh 11 OK, test_mesh* 49 OK, test_fem* 108 with only the two known isolation failures, test_map 45 OK, test_streams with the known `test_event_external` race; any pool break or `0x160C6C` crash is a finding.
2. `mma_check.py` (165 kernels), still not run here.
3. `test_event_external`: upstream test race (session 13), no action unless Jeff wants a test-only change carried.

**For Jeff:** the workaround is in every later Warp release line's diff (18 lines, Windows-only) unless the runtime is fixed; the runtime defect is registered as `hip-graph-autofree-relaunch-races-worker-memfree` (unruled; reproducer above, also present in rocm-systems `develop`).
## Evidence session (2026-11-19, linux-gfx1201, evidence only): bf16 WMMA at `fda2f89a`

Requested by Jeff Daily. **Every gfx1201 item in Porting session 13's Next step matched its expected result, with one exception: the CPU-only suite had 1 intermittent failure, in an upstream CPU test that the port does not touch.** The gfx12 bf16 operand layout is now measured and matches the layout `tile_matmul_mma.h` assumes. bf16 WMMA beats the scalar tree in all 12 cells, by 2.20x to 8.98x. The porting lock stays with windows-gfx1151. Nothing was committed or pushed to the fork, and no stage, lock, head or platform record changed.

### Setup
- Device "AMD Radeon AI PRO R9700" (gfx1201, wave32, 64 KiB shared memory per block). The dispatcher pin `HIP_VISIBLE_DEVICES=3` was kept. Every GPU run was under `ulimit -c 0` with a private `WARP_CACHE_PATH`.
- The job venv (`$FLEET_VENV`) came with TheRock 10.1.0 wheels. Installed into it, as the earlier gfx1201 sessions did: `pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ --extra-index-url https://pypi.org/simple "rocm[devel,device-gfx1201]==10.2.0" "torch[device-gfx1201]==2.15.0+rocm10.2.0" numpy==1.26.4 usd-core`, then `rocm-sdk init`. `PATH=$FLEET_VENV/bin:$PATH`. `R=$(rocm-sdk path --root)` is the venv's 10.2 `_rocm_sdk_devel`.
- The fork was cloned read-only into `projects/warp-1.14/src` and checked out detached at `fda2f89a145836d8d51ce6360439d28217fce7a5`, followed by `git lfs pull`. `git status --porcelain` was empty after the run: the untracked `example_apic_fluid.usd` that an example writes was deleted. `protect-fork` found a non-MOAT pre-push hook (git-lfs) and left it in place. Nothing was pushed.
- Build: `python build_lib.py --rocm-path $R --hip-arch gfx1201 -j 6` -> "Warp build succeeded" (52 s under `utils/timeit.sh`). The RUNPATH of `warp.so` is the venv's 10.2 `_rocm_sdk_devel/lib`.
- Scripts were extracted verbatim from these notes: `wmma_layout_bf16.hip`, `isa_count.py` and `cells_bf16.py` from Porting session 13; `mma_check.py` and `mma_check_bf16.py` from Porting session 10 (the fixed versions); `bench_tile_matmul.py` and `compare.py` from Porting sessions 4 and 9.

### 1. bf16 WMMA layout probe (first gfx12 measurement): PASS
`$R/lib/llvm/bin/clang++ -x hip --rocm-path=$R --offload-arch=gfx1201 -O2 -w wmma_layout_bf16.hip -o wmma_layout_bf16 -L$R/lib -lamdhip64 -Wl,-rpath,$R/lib && ./wmma_layout_bf16` -> exit 0:
```
bf16 WMMA family 12, layout (non-negative)  K=16: 0 of 256 wrong
bf16 WMMA family 12, layout (non-negative)  K=32: 0 of 256 wrong
bf16 WMMA family 12, layout (non-negative)  K=64: 0 of 256 wrong
bf16 WMMA family 12, layout (position)      K=16: 0 of 256 wrong
bf16 WMMA family 12, layout (position)      K=32: 0 of 256 wrong
bf16 WMMA family 12, layout (position)      K=64: 0 of 256 wrong
bf16 WMMA family 12, accumulation           K=16: 0 of 256 over the bound, worst error 0.18 of the bound
bf16 WMMA family 12, accumulation           K=32: 0 of 256 over the bound, worst error 0.08 of the bound
bf16 WMMA family 12, accumulation           K=64: 0 of 256 over the bound, worst error 0.06 of the bound
```
The device assembly holds `v_wmma_f32_16x16x16_bf16`. So the gfx12 bf16 layout (A[lo][8g+v], B[8g+v][lo], accumulator r = C[8g+r][lo]) is the gfx12 fp16 layout, as `6761183c` assumed, and the gfx12 `op<bfloat16>` stays.

### 2. `mma_check_bf16.py`, `mma_check.py`, `isa_count.py`: as expected
- `python mma_check_bf16.py` -> **FAILURES 0**, exit 0. It ran 55 cases (11 shapes x 5 block_dims) with no shared-memory limit. Max abs difference against the CPU at bd 64: 1.2e-7 (16^3), 4.8e-7 (32^3), 1.9e-6 (64^3), 4.8e-7 (16x48x32), 9.5e-7 (48x16x64), 2.4e-7 (32x64x16), 9.5e-7 (64x32x48), and 0 for the four scalar shapes.
- `python isa_count.py <that cache> gfx1201`: **`v_wmma_f32_16x16x16_bf16` appears in all 35 code objects whose M, N and K are multiples of 16** (7 shapes x 5 block_dims), **64x64x64 at block_dim 256 included (16 instructions, which is 2 tile_matmul calls x 8, matching `cells_bf16.py`'s 8)**. gfx12 has no scalar exclusion. The 4 other shapes (8^3, 16x16x12, 24x16x16, 17x33x15) are scalar at every block_dim.
- `python mma_check.py` -> **FAILURES 0**, exit 0, which is session 3's result at `e78ee32f`: 25 ok, 25 "SMEM-LIMIT (backward not launched)", 15 "SMEM-LIMIT (forward not launched)" (all 64 KiB limits). Across its 165 code objects there are 1336 `v_wmma_f32_16x16x16_f16`, the same total as every earlier gfx1201 run.

### 3a. test_tile_matmul alone: Ran 25, OK
This used a fresh cache, followed by `isa_count.py <cache> gfx1201`. The fp16 WMMA pattern is identical to sessions 2 and 3: 16x16x16 has 4 instructions at bd 64 and 96; 48x16 (K 32) has 15 at bd 64 and 9 at bd 96; 24x16, 17x33x15 and 8x8x4 are scalar. The test's only bf16 module (8x4x8) is scalar, because it does not fit a fragment. fp32 and fp64 have no WMMA.

### 3b. bf16 benchmark against the scalar tree: no cell below 1.0x
- Trees: "WMMA" = the clone at `fda2f89a`. "scalar" = `git -C <clone> worktree add --detach bf16/scalar_wt fda2f89a`, then `git -C bf16/scalar_wt checkout a8e95dcb -- warp/native/tile_matmul_mma.h` (this reverts exactly `6761183c`'s +49/-2; `git diff --stat HEAD` showed that one file), then `warp/bin/*.so` copied from the clone. bf16 is the only dtype whose code differs between the two trees. The worktree was removed afterwards.
- Script: the Phase 4 `bench_tile_matmul.py` with a bf16 mode (`bench_tile_matmul_bf16.py`; the diff is below). The bf16 kernel has bf16 A and B tiles, a float32 C tile, 32 accumulating `tile_matmul` calls per block and `enable_backward: False`. It runs forward only, because tile_matmul rejects a bf16 backward. Each cell checks block 0 against an fp64 reference computed from the bf16-rounded inputs. Tiles are 16/32/64 at block_dim 32/64/128/256, the same cells as fp16.
- Command (`bench_both.sh`, below): three runs per tree, alternating WMMA and scalar, each with a fresh cache: `python bench_tile_matmul_bf16.py --blocks 16384 --iters 50 --dtypes bfloat16,float16,float32` (306.9 s for all six under `utils/timeit.sh warp-1.14 bench`). The fp16 and fp32 rows are controls with identical code in both trees.
- GPU idle check before the run: `amd-smi process` reported no process on any of the four GPUs. A 5-second monitor ran alongside and reported exactly one GPU process (the benchmark) in all 58 samples, with host load average at most 3.57. (Its GFX% column was mis-parsed and is unusable, and its per-run start markers were overwritten: the monitor's `>` and the markers' `>>` shared one file.) **Clean:** no value is above 1.5x its median. Run-to-run spread over 120 series has a median of 1.4% and a max of 14.8%; the max is the sub-0.3 ms fp16 16x16 adjoint cells, where run 1 is slower.
- Every run exits 1, from the known fp32 64x64 `ADJ-FAILED` rows: the backward kernel needs 98304 bytes against 65536, so it does not launch. That holds in all six runs and in both trees, and is not bf16. Every bf16 and fp16 cell is `ok` in every run.

**bf16 input / float32 output, forward, median of 3 runs (ms):**

| tile | block_dim | scalar fwd ms | WMMA fwd ms | speedup | check |
|---|---|---|---|---|---|
| 16x16 | 32 | 0.967 | 0.339 | 2.85x | ok |
| 16x16 | 64 | 1.090 | 0.349 | 3.12x | ok |
| 16x16 | 128 | 0.862 | 0.391 | 2.20x | ok |
| 16x16 | 256 | 1.289 | 0.406 | 3.18x | ok |
| 32x32 | 32 | 3.466 | 1.481 | 2.34x | ok |
| 32x32 | 64 | 4.111 | 1.774 | 2.32x | ok |
| 32x32 | 128 | 4.660 | 2.051 | 2.27x | ok |
| 32x32 | 256 | 6.812 | 2.093 | 3.25x | ok |
| 64x64 | 32 | 63.354 | 7.056 | 8.98x | ok |
| 64x64 | 64 | 35.363 | 8.354 | 4.23x | ok |
| 64x64 | 128 | 30.557 | 10.609 | 2.88x | ok |
| 64x64 | 256 | 30.372 | 13.136 | 2.31x | ok |

- **Every bf16 cell is faster with WMMA: 2.20x to 8.98x.** The 64x64 bd 256 cell takes WMMA here and is 2.31x faster (gfx12 has no scalar exclusion; only gfx11 leaves that cell scalar).
- The bf16 WMMA times are close to the fp16 WMMA forward times measured in the same runs (fp16 forward: 16x16 0.35-0.42 ms, 32x32 1.79-2.11 ms, 64x64 9.7-12.7 ms). The bf16 *scalar* path is slower than fp16 scalar was (64x64 bd 32: 63.4 ms against 27.5-28.1 ms in the first gfx1201 session at `930d0098`), which is why the bf16 64x64 speedups are larger than fp16's.
- Controls: in every fp16 and fp32 cell (forward and adjoint), the WMMA and scalar trees agree within 0.98x-1.02x. That is expected, because their code is identical.

### 3c. Full 82-pattern set and CPU-only suite
- `run_full.sh` (below) runs the 82 patterns of the Porting session 6 handoff (B.5), extracted from these notes (82 lines, no duplicates), one at a time with `python -m warp.tests -s autodetect -p "<pattern>" -m 32` and a private cache each, from the clone. The CPU-only suite (`HIP_VISIBLE_DEVICES=99 CUDA_VISIBLE_DEVICES=99 python -m warp.tests -s autodetect -k cpu -m 32`) ran alongside. Total 1942 s under `utils/timeit.sh warp-1.14 test`.
- **82 patterns: 81 OK, 1 with the 2 known failures** (`test_fem*` 107 run, failures=2: `test_gimp_quadrature`, `test_cube_shape_functions`, the upstream isolation behaviour). The per-pattern summary lines (pattern, rc, Ran count, result; timings stripped) are **identical** to Evidence session 3's run at `e78ee32f` on this host. Inside the set: the tile set (24 files) ran 639 with 15 skipped and 0 failures; test_tile_matmul 25 OK; test_texture 120 OK (1 skipped); test_streams 24 OK (3 skipped); test_torch 34 OK; test_dlpack 16 OK. There is no `Warp CUDA error` in any log, and the only shared-memory warning is the known `test_tile_load_scoped` backward one.
- **CPU-only suite: Ran 2294, FAILED (failures=1, skipped=15)**, against 2294 OK in every earlier run. The failure is `test_cpu_precompiled_headers.TestCpuPrecompiledHeaders.test_basic`: `AssertionError: 0 not greater than 0 : No .pch file generated despite use_precompiled_headers=True`. The kernel itself ran and gave correct values; only the `.pch` file was missing from the per-thread `wp_pch_` temp directory. Re-run alone twice (`HIP_VISIBLE_DEVICES=99 CUDA_VISIBLE_DEVICES=99 python -m warp.tests -s autodetect -p test_cpu_precompiled_headers.py -m 1`, fresh cache each) -> Ran 3, OK both times. The port's diff against `release-1.14` touches no CPU PCH code: the only added `pch` line is the `pch_dir` parameter of the hipRTC build function. So this is an intermittent upstream CPU test under parallel load (32 CPU workers next to the GPU set), not a port defect and not a bf16 effect. No log line names `cuda:0`.

### Summary
At `fda2f89a`, the gfx12 bf16 WMMA path is correct on hardware: the layout probe passes and `mma_check_bf16.py` reports FAILURES 0, with WMMA in all 35 eligible code objects. It is also faster than scalar in every benchmark cell (2.20x-8.98x). The rest of the gfx1201 list reproduces session 3's `e78ee32f` results, except for the one intermittent upstream CPU PCH failure above. Nothing on gfx1201 needs a fork change.

### Scripts from this session

`bench_both.sh`:
```bash
#!/bin/bash
# bf16 tile_matmul benchmark, three runs each against the bf16-WMMA tree and the scalar tree,
# with a 5-second GPU-busy and load monitor.
ulimit -c 0
A=/var/lib/jenkins/fleet/jobs/gfx1201-20261119-101222-warp114-bf16-evidence-gfx1201-prompt/moat/agent_space
B=$A/bf16
mkdir -p $B/runs
(while true; do echo "$(date -u +%T) load $(cut -d' ' -f1 /proc/loadavg) $(amd-smi monitor -u 2>/dev/null | awk 'NR>1{printf "gpu%s=%s%% ", $1, $4}') procs=$(amd-smi process 2>/dev/null | grep -c PID)"; sleep 5; done) > $B/runs/monitor.log 2>&1 &
MON=$!
for n in 1 2 3; do
  for tree in wmma scalar; do
    if [ $tree = wmma ]; then src=$A/wt-warp-1.14/projects/warp-1.14/src; else src=$B/scalar_wt; fi
    echo "$(date -u +%T) start $tree run$n" >> $B/runs/monitor.log
    PYTHONPATH=$src WARP_CACHE_PATH=$(mktemp -d $B/kc.XXXXXX) "$FLEET_VENV/bin/python" $B/bench_tile_matmul_bf16.py \
      --blocks 16384 --iters 50 --dtypes bfloat16,float16,float32 --csv $B/runs/bench_${tree}_run$n.csv > $B/runs/bench_${tree}_run$n.log 2>&1
    echo "$tree run$n rc=$?"
  done
done
kill $MON
```

`bench_tile_matmul_bf16.py` = the Phase 4 `bench_tile_matmul.py` plus this diff:
```diff
--- bench_tile_matmul.py
+++ bench_tile_matmul_bf16.py
@@ -1,5 +1,6 @@
 # tile_matmul throughput: square MxM tiles, one tile GEMM chain per block.
-# Usage: python bench_tile_matmul.py [--device cuda:0] [--mathdx on|off] [--csv out.csv]
+# Usage: python bench_tile_matmul_bf16.py [--device cuda:0] [--mathdx on|off] [--csv out.csv] [--dtypes bfloat16,float16]
+# bfloat16: bf16 A and B with a float32 C (the only bf16 form tile_matmul accepts), forward only.
 # fwd:  REPS accumulating tile_matmul calls per block (C += A @ B), loads/stores amortized.
 # adj:  one C = A @ B per block, timed tape.backward (adj_A = adj_C B^T, adj_B = A^T adj_C);
 #       adj_gflops counts those 2 GEMMs (the backward kernel also replays the forward GEMM).
@@ -18,6 +19,7 @@
 p.add_argument("--blocks", type=int, default=4096)
 p.add_argument("--reps", type=int, default=32)
 p.add_argument("--iters", type=int, default=20)
+p.add_argument("--dtypes", default="bfloat16,float16,float32,float64")
 args = p.parse_args()
 
 wp.init()
@@ -25,12 +27,15 @@
 dev = wp.get_device(args.device)
 
 SIZES = (16, 32, 64)
-DTYPES = (wp.float16, wp.float32, wp.float64)
+DTYPES = tuple(getattr(wp, d) for d in args.dtypes.split(","))
 BLOCK_DIMS = (32, 64, 128, 256)
 REPS = args.reps
 
 
 def make_kernels(M, dtype):
+    if dtype == wp.bfloat16:
+        return make_bf16(M), None
+
     @wp.kernel(enable_backward=False, module="unique")
     def fwd(A: wp.array3d(dtype=dtype), B: wp.array3d(dtype=dtype), C: wp.array3d(dtype=dtype)):
         i = wp.tid()
@@ -52,6 +57,57 @@
     return fwd, one
 
 
+def make_bf16(M):
+    @wp.kernel(module="unique", module_options={"enable_backward": False})
+    def fwd(A: wp.array3d(dtype=wp.bfloat16), B: wp.array3d(dtype=wp.bfloat16), C: wp.array3d(dtype=float)):
+        i = wp.tid()
+        a = wp.tile_load(A[i], shape=(M, M))
+        b = wp.tile_load(B[i], shape=(M, M))
+        c = wp.tile_zeros(shape=(M, M), dtype=float)
+        for _r in range(REPS):
+            wp.tile_matmul(a, b, c)
+        wp.tile_store(C[i], c)
+
+    return fwd
+
+
+@wp.kernel
+def to_bf16(x: wp.array3d(dtype=float), y: wp.array3d(dtype=wp.bfloat16)):
+    i, j, k = wp.tid()
+    y[i, j, k] = wp.bfloat16(x[i, j, k])
+
+
+@wp.kernel
+def from_bf16(y: wp.array3d(dtype=wp.bfloat16), x: wp.array3d(dtype=float)):
+    i, j, k = wp.tid()
+    x[i, j, k] = float(y[i, j, k])
+
+
+def bench_bf16(M, fwd):
+    shape = (args.blocks, M, M)
+    A = wp.empty(shape, dtype=wp.bfloat16, device=dev)
+    B = wp.empty(shape, dtype=wp.bfloat16, device=dev)
+    C = wp.zeros(shape, dtype=float, device=dev)
+    for x, y in ((A, rng), (B, rng)):
+        wp.launch(to_bf16, dim=shape, inputs=[wp.array(y.uniform(-1, 1, shape).astype(np.float32), device=dev), x], device=dev)
+    # reference from the bf16-rounded inputs of block 0
+    Af = wp.empty(shape, dtype=float, device=dev)
+    Bf = wp.empty(shape, dtype=float, device=dev)
+    wp.launch(from_bf16, dim=shape, inputs=[A, Af], device=dev)
+    wp.launch(from_bf16, dim=shape, inputs=[B, Bf], device=dev)
+    ref = REPS * (Af.numpy()[0].astype(np.float64) @ Bf.numpy()[0].astype(np.float64))
+    for bd in BLOCK_DIMS:
+        def run_fwd():
+            wp.launch_tiled(fwd, dim=[args.blocks], inputs=[A, B, C], block_dim=bd, device=dev)
+
+        fwd_ms = timed(run_fwd, args.iters)
+        ok = np.allclose(C.numpy()[0].astype(np.float64), ref, rtol=1e-4, atol=1e-4 * REPS * M)
+        fl = 2.0 * M * M * M * args.blocks
+        r = ("bfloat16", M, bd, fwd_ms, fl * REPS / fwd_ms / 1e6, float("nan"), float("nan"), "ok" if ok else "MISMATCH")
+        rows.append(r)
+        print("%s,%d,%d,%.4f,%.1f,%.4f,%.1f,%s" % r, flush=True)
+
+
 def timed(fn, iters):
     fn()  # warm-up, includes module load
     wp.synchronize_device(dev)
@@ -73,6 +129,9 @@
 for dtype in DTYPES:
     for M in SIZES:
         fwd, one = make_kernels(M, dtype)
+        if dtype == wp.bfloat16:
+            bench_bf16(M, fwd)
+            continue
         npt = wp.dtype_to_numpy(dtype)
         A = wp.array(rng.uniform(-1, 1, (args.blocks, M, M)).astype(npt), dtype=dtype, device=dev, requires_grad=True)
         B = wp.array(rng.uniform(-1, 1, (args.blocks, M, M)).astype(npt), dtype=dtype, device=dev, requires_grad=True)
```

`compare_bf16.py` (run from the directory holding `runs/`):
```python
# Merge the three runs of each tree: median ms per cell, WMMA speedup over scalar, run-to-run spread,
# and cells above 1.5x their median (interference). bf16 rows are forward only.
import csv
import math
import statistics
from collections import defaultdict

data = defaultdict(lambda: defaultdict(list))
checks = defaultdict(set)
for tree in ("scalar", "wmma"):
    for n in (1, 2, 3):
        for r in csv.DictReader(open(f"runs/bench_{tree}_run{n}.csv")):
            key = (r["dtype"], int(r["M"]), int(r["block_dim"]))
            data[key][tree + "_fwd"].append(float(r["fwd_ms"]))
            if not math.isnan(float(r["adj_ms"])):
                data[key][tree + "_adj"].append(float(r["adj_ms"]))
            checks[key].add(f"{tree}:{r['check']}")
spread, outliers = [], []
print("| dtype | tile | block_dim | scalar fwd ms | WMMA fwd ms | fwd speedup | scalar adj ms | WMMA adj ms | adj speedup | checks |")
print("|---|---|---|---|---|---|---|---|---|---|")
order = ["bfloat16", "float16", "float32", "float64"]
for key in sorted(data, key=lambda k: (order.index(k[0]), k[1], k[2])):
    d = data[key]
    m = {k: statistics.median(v) for k, v in d.items()}
    for name, v in d.items():
        spread.append((max(v) - min(v)) / statistics.median(v))
        outliers += [(key, name, x) for x in v if x > 1.5 * statistics.median(v)]
    ck = sorted(checks[key])
    if "scalar_adj" in m and all(c.endswith(":ok") for c in ck):
        adj = f"{m['scalar_adj']:.3f} | {m['wmma_adj']:.3f} | {m['scalar_adj'] / m['wmma_adj']:.2f}x"
    elif "scalar_adj" in m:
        adj = "(no launch) | (no launch) | -"
    else:
        adj = "- | - | -"
    print(f"| {key[0]} | {key[1]}x{key[1]} | {key[2]} | {m['scalar_fwd']:.3f} | {m['wmma_fwd']:.3f} | {m['scalar_fwd'] / m['wmma_fwd']:.2f}x | {adj} | {'/'.join(sorted({c.split(':')[1] for c in ck}))} |")
spread.sort()
print(f"run-to-run spread (max-min)/median over {len(spread)} series: median {100 * statistics.median(spread):.1f}%, max {100 * spread[-1]:.1f}%")
print(f"values above 1.5x their median: {len(outliers)} {outliers[:10]}")
```

`run_full.sh` (`full_set.txt` = the 82 patterns of the B.5 loop, one per line):
```bash
#!/bin/bash
# 82-pattern set, one pattern at a time, private cache each; CPU-only suite alongside (it touches no GPU).
ulimit -c 0
B=/var/lib/jenkins/fleet/jobs/gfx1201-20261119-101222-warp114-bf16-evidence-gfx1201-prompt/moat/agent_space/bf16
S=$B/../wt-warp-1.14/projects/warp-1.14/src
export PATH=$FLEET_VENV/bin:$PATH
mkdir -p $B/full
cd $S
(HIP_VISIBLE_DEVICES=99 CUDA_VISIBLE_DEVICES=99 WARP_CACHE_PATH=$(mktemp -d $B/kc.XXXXXX) python -m warp.tests -s autodetect -k cpu -m 32 > $B/full/cpu_only.log 2>&1; echo "cpu-only rc=$?" >> $B/full/cpu_only.log) &
while read -r p; do
  f=$(echo "$p" | tr -d '*')
  WARP_CACHE_PATH=$(mktemp -d $B/kc.XXXXXX) python -m warp.tests -s autodetect -p "$p" -m 32 > $B/full/$f.log 2>&1
  echo "$p rc=$?"
done < $B/full_set.txt
wait
```

### Head moved during this session
While these runs were in progress, the porter advanced the head to `de4e07a3` ("[ROCm] Wait for the previous graph launch on Windows"). The fork delta `fda2f89a..de4e07a3` is `warp/native/hip_compat.h` +18. All of it is inside `#if defined(_WIN32)`, and the Linux branch keeps `#define cudaGraphLaunch hipGraphLaunch` unchanged. So the Linux build at `de4e07a3` is the same as at `fda2f89a`, and no kernel-cache header (`tile_matmul_mma.h`, `crt.h`, `builtin.h`) changed. This session did not run anything at `de4e07a3`. Whether these results carry forward to it is the validator's call (`carry-forward`); no record was changed here.

## Evidence session (2026-11-19, linux-gfx1100, evidence only): bf16 WMMA at `fda2f89a`

Requested by Jeff Daily. **Every gfx1100 item in Porting session 13's Next step matched its expected result.** The gfx11 bf16 layout probe passes on gfx1100. `mma_check_bf16.py` and `mma_check.py` both report FAILURES 0, with bf16 WMMA in exactly the expected 34 code objects. The tile set gives 639 run, 15 skipped, 0 failures. **bf16 WMMA beats the scalar tree in all 11 cells where it runs, by 2.21x to 5.89x. 64x64 at block_dim 256 is 1.03x, because both trees are scalar there by construction.** The 82-pattern set gives 81 OK plus the two known fem isolation failures, and the CPU-only suite gives 2294 OK. The porting lock stays with windows-gfx1151. Nothing was committed or pushed to the fork, and no stage, lock, head, platform or deferral record changed. The fork head moved to `de4e07a3` during this session. That delta is `hip_compat.h` +18, entirely inside `#if defined(_WIN32)`, so it does not change the Linux build. The evidence here is at `fda2f89a`, as requested.

### Setup
- Device "AMD Radeon Pro W7800 48GB" (gfx1100, wave32, 64 KiB shared memory per block). The dispatcher pin `HIP_VISIBLE_DEVICES=0` was kept; that is PCI bus 0x43. Every GPU run was under `ulimit -c 0` with its own fresh `WARP_CACHE_PATH`.
- The job venv came with TheRock 10.1.0 wheels. Installed into `$FLEET_VENV`, as the earlier gfx1100 sessions did: `"$FLEET_VENV/bin/python" -m pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ --extra-index-url https://pypi.org/simple "rocm[devel,device-gfx1100]==10.2.0" "torch[device-gfx1100]==2.15.0+rocm10.2.0" numpy==1.26.4 usd-core`, then `rocm-sdk init`. `PATH=$FLEET_VENV/bin:$PATH`. This gives HIP 7.17.26434 and torch 2.15.0+rocm10.2.0. `R=$(rocm-sdk path --root)` is the venv's 10.2 `_rocm_sdk_devel`.
- The fork was cloned read-only into `agent_space/warp-src` (`release/1.14.0.amd0`) and checked out detached at `fda2f89a145836d8d51ce6360439d28217fce7a5`. LFS assets were present. At the end, `git status --porcelain` shows only the untracked `example_apic_fluid.usd` written by an example. Nothing was pushed.
- Build: `python build_lib.py --rocm-path "$(rocm-sdk path --root)" --hip-arch gfx1100 -j 32` -> "Warp build succeeded" (32.9 s). The RUNPATH of `warp.so` is the venv's 10.2 `_rocm_sdk_devel/lib`.
- Scripts were extracted verbatim from these notes: `wmma_layout_bf16.hip` and `isa_count.py` from Porting session 13; `mma_check.py` and `mma_check_bf16.py` from Porting session 10 (the fixed versions); `bench_tile_matmul.py` from Porting session 4.
- Telemetry: the `utils/timeit.sh` lines for this session (compile 32.9 s; mma_check_bf16 86.7 s; mma_check 344.1 s; tile set 724.9 s; one 6 s line for an aborted benchmark attempt; then the benchmark, the 82-pattern set and the CPU-only suite) were **lost**. I reverted the uncommitted `stats.jsonl` with `git checkout` before pulling. That was my error. They were not re-created by hand. From the log mtimes: the benchmark took about 8.5 min for all six runs, the 82-pattern set about 42 min, and the CPU-only suite 151 s of test time.

### 1. bf16 WMMA layout probe: PASS
`$R/lib/llvm/bin/clang++ -x hip --rocm-path=$R --offload-arch=gfx1100 -O2 -w wmma_layout_bf16.hip -o wmma_layout_bf16 -L$R/lib -lamdhip64 -Wl,-rpath,$R/lib && ./wmma_layout_bf16` -> exit 0:
```
bf16 WMMA family 11, layout (non-negative)  K=16: 0 of 256 wrong
bf16 WMMA family 11, layout (non-negative)  K=32: 0 of 256 wrong
bf16 WMMA family 11, layout (non-negative)  K=64: 0 of 256 wrong
bf16 WMMA family 11, layout (position)      K=16: 0 of 256 wrong
bf16 WMMA family 11, layout (position)      K=32: 0 of 256 wrong
bf16 WMMA family 11, layout (position)      K=64: 0 of 256 wrong
bf16 WMMA family 11, accumulation           K=16: 0 of 256 over the bound, worst error 0.27 of the bound
bf16 WMMA family 11, accumulation           K=32: 0 of 256 over the bound, worst error 0.13 of the bound
bf16 WMMA family 11, accumulation           K=64: 0 of 256 over the bound, worst error 0.08 of the bound
```
gfx1151 measured 0.21/0.15/0.07 of the bound.

### 2. `mma_check_bf16.py`, `mma_check.py`, `isa_count.py`: as expected
- `python mma_check_bf16.py` -> **FAILURES 0**, exit 0. It ran 55 cases with no shared-memory limit. Max abs difference against the CPU at bd 64: 1.2e-7 (16^3), 9.5e-7 (32^3), 1.9e-6 (64^3), 4.8e-7 (16x48x32), 9.5e-7 (48x16x64), 3.6e-7 (32x64x16), 9.5e-7 (64x32x48), and 0 for the four scalar shapes.
- `python isa_count.py <that cache> gfx1100`: **`v_wmma_f32_16x16x16_bf16` appears in 34 of the 55 code objects**. That is exactly the 7 shapes with 16-multiple M, N and K at all 5 block_dims, **except 64x64x64 at block_dim 256, which is scalar**. The 4 other shapes are scalar at every block_dim. This is the same as the gfx1151 table. Instruction counts per object: 16^3 2; 16x48x32 4 (8 at bd 64); 32^3 16/8/8/4/4; 32x64x16 16/8/8/4/2; 48x16x64 8/16/8/8/8; 64x32x48 12/12/24/12/6; 64^3 32/32/64/32/scalar (bd 32/64/96/128/256).
- `python mma_check.py` -> **FAILURES 0**, exit 0. It reports 15 "SMEM-LIMIT (forward not launched)" and 25 "SMEM-LIMIT (backward not launched)", the same 40 cases as in Evidence session 8 at `a8e95dcb`.

### 3. test_tile_matmul and the tile set: match
`python -m warp.tests -s autodetect -p "<file>" -m 32`, one run per file of the 24-file tile set (`warp/tests/tile/test_tile*.py` plus `test_block_dim_dispatch.py` and `test_template_launch_bounds.py`), each with a fresh cache. Result: **639 run, 15 skipped, 0 failures**, including **test_tile_matmul Ran 25, OK**. The 15 skips are as before: test_tile_fft 8, test_tile_mlp 1, test_tile_reduce 2, test_tile_shared_memory 2, test_tile_solve 1 and test_tile_view 1. No log has a `Warp CUDA error` line. The only shared-memory warning is the known `test_tile_load_scoped` backward one (65536 bytes).

### 4. bf16 benchmark against the scalar tree: no bf16 cell below 1.0x
- Trees: "shipped" = the clone at `fda2f89a`. "scalar" = `git worktree add --detach scalar_wt fda2f89a`, then `git -C scalar_wt checkout a8e95dcb -- warp/native/tile_matmul_mma.h` (`git diff --cached --stat`: that one file, +2/-49, which reverts `6761183c`), then the clone's `warp/bin/*.so` copied in. The worktree was removed afterwards. Only the bf16 code differs between the trees. fp16, fp32 and fp64 are controls with identical code in both: fp16 is on WMMA in both, fp32 and fp64 are scalar in both.
- Script: `bench_tile_matmul_bf16.py`, the Phase 4 `bench_tile_matmul.py` plus a bf16 mode (diff below). The bf16 kernel loads bf16 A and B tiles, accumulates into a float32 C tile with 32 `tile_matmul` calls per block, and runs with `module_options={"enable_backward": False}`. It is forward only, because tile_matmul rejects a bf16 backward. Each cell checks block 0 against an fp64 reference computed from the bf16-rounded inputs (rtol 1e-4). Tiles are 16/32/64 at block_dim 32/64/128/256, the same cells as fp16. The first attempt passed `enable_backward=False` to `@wp.kernel` instead. Codegen refused it ("does not support bfloat16 'a' or 'b' tiles when the backward pass is enabled"), so I stopped it after the first run's first module, fixed the script and ran it again. The aborted attempt produced no data.
- Command: `bench_both.sh` (below), `--blocks 16384 --iters 50`, three runs of the shipped tree and then three of the scalar tree, each with a fresh cache.
- GPU idle check before the run: `gpu_busy_percent` was 0 for card1 (bus 0x43, this job's GPU). Another job's GPU, card4, was at 100% throughout, as it had been since the start of the session. `monitor.sh` (below) logged the host load average and every card's busy percent every 5 s during the six runs (120 samples). Load average was at most 2.00. card1 was busy only while the benchmark ran.
- **Clean:** no value is above 1.5x its median. Over the 144 launched series, the run-to-run spread (max-min)/median has a median of 4.4% and a maximum of 11.5% (fp32 32x32 bd 256 adjoint, shipped). Run 1 is the fastest of three in most cells, the warm-up pattern this card has shown since the first session. Over all 168 series, including the non-launched fp64 64x64 forwards, the maximum spread is 22.7%.
- Every run exits 1. That comes from the known 64 KiB rows in both trees, all six runs: fp32 64x64 `ADJ-FAILED` (the backward kernel does not launch) and fp64 64x64 `MISMATCH` (the forward kernel does not launch; about 0.02 ms). These are the same as the fp32/fp64 rows in Evidence session 6 at `c9595bce`. Every bf16 and fp16 cell is `ok` in every run.

**bf16 input / float32 output, forward, median of 3 runs (ms):**

| tile | block_dim | scalar fwd ms | shipped (WMMA) fwd ms | speedup | check |
|---|---|---|---|---|---|
| 16x16 | 32 | 1.150 | 0.254 | 4.53x | ok |
| 16x16 | 64 | 1.286 | 0.276 | 4.66x | ok |
| 16x16 | 128 | 1.017 | 0.378 | 2.69x | ok |
| 16x16 | 256 | 1.528 | 0.444 | 3.44x | ok |
| 32x32 | 32 | 4.369 | 1.733 | 2.52x | ok |
| 32x32 | 64 | 5.082 | 2.085 | 2.44x | ok |
| 32x32 | 128 | 6.358 | 2.882 | 2.21x | ok |
| 32x32 | 256 | 8.513 | 2.926 | 2.91x | ok |
| 64x64 | 32 | 69.603 | 11.820 | 5.89x | ok |
| 64x64 | 64 | 44.503 | 11.702 | 3.80x | ok |
| 64x64 | 128 | 33.375 | 13.459 | 2.48x | ok |
| 64x64 | 256 | 35.735 | 34.579 | 1.03x (both scalar) | ok |

- **No bf16 cell is below 1.0x.** The 11 WMMA cells are 2.21x to 5.89x faster. The 64x64 bd 256 cell is scalar in both trees, as `isa_count.py` shows above, and reads 1.03x. That is within its run spread (shipped 33.41/34.58/35.23 ms, scalar 34.04/36.01/35.74 ms), and it is the same tree-order offset that the identical fp16 cells show (1.00x-1.02x).
- The bf16 WMMA forward times are close to the fp16 WMMA forward times from the same runs: 16x16 0.25-0.44 ms against 0.29-0.48; 32x32 1.73-2.93 against 1.81-2.69; 64x64 bd 32-128 11.7-13.5 against 13.7-18.8. The bf16 scalar path is much slower than the fp16 WMMA path. For example, 64x64 bd 32 takes 69.6 ms, and the fp16 *scalar* time for that cell at `a8e95dcb` was 33.2 ms. That is why the 64x64 bf16 speedups are larger than fp16's were.
- **Controls**, shipped against scalar (identical code): fp16 forward 1.00x-1.02x, fp16 adjoint 0.99x-1.01x, fp32 0.98x-1.02x. fp64 is 0.99x-1.00x in the launched cells. The fp16 shipped medians agree with Evidence session 8 within its spread; for example, 64x64 bd 256 forward is 17.31 ms against 16.78 ms, and its adjoint is 8.96 ms against 8.83 ms.

Full merged table (`compare_bf16.py`):

| dtype | tile | block_dim | scalar fwd ms | shipped fwd ms | fwd speedup | scalar adj ms | shipped adj ms | adj speedup | checks |
|---|---|---|---|---|---|---|---|---|---|
| bfloat16 | 16x16 | 32 | 1.150 | 0.254 | 4.53x | n/a | n/a | - | ok |
| bfloat16 | 16x16 | 64 | 1.286 | 0.276 | 4.66x | n/a | n/a | - | ok |
| bfloat16 | 16x16 | 128 | 1.017 | 0.378 | 2.69x | n/a | n/a | - | ok |
| bfloat16 | 16x16 | 256 | 1.528 | 0.444 | 3.44x | n/a | n/a | - | ok |
| bfloat16 | 32x32 | 32 | 4.369 | 1.733 | 2.52x | n/a | n/a | - | ok |
| bfloat16 | 32x32 | 64 | 5.082 | 2.085 | 2.44x | n/a | n/a | - | ok |
| bfloat16 | 32x32 | 128 | 6.358 | 2.882 | 2.21x | n/a | n/a | - | ok |
| bfloat16 | 32x32 | 256 | 8.513 | 2.926 | 2.91x | n/a | n/a | - | ok |
| bfloat16 | 64x64 | 32 | 69.603 | 11.820 | 5.89x | n/a | n/a | - | ok |
| bfloat16 | 64x64 | 64 | 44.503 | 11.702 | 3.80x | n/a | n/a | - | ok |
| bfloat16 | 64x64 | 128 | 33.375 | 13.459 | 2.48x | n/a | n/a | - | ok |
| bfloat16 | 64x64 | 256 | 35.735 | 34.579 | 1.03x | n/a | n/a | - | ok |
| float16 | 16x16 | 32 | 0.292 | 0.285 | 1.02x | 0.235 | 0.233 | 1.01x | ok |
| float16 | 16x16 | 64 | 0.287 | 0.287 | 1.00x | 0.207 | 0.209 | 0.99x | ok |
| float16 | 16x16 | 128 | 0.429 | 0.427 | 1.00x | 0.208 | 0.209 | 1.00x | ok |
| float16 | 16x16 | 256 | 0.479 | 0.475 | 1.01x | 0.252 | 0.256 | 0.99x | ok |
| float16 | 32x32 | 32 | 1.824 | 1.809 | 1.01x | 2.685 | 2.658 | 1.01x | ok |
| float16 | 32x32 | 64 | 1.877 | 1.857 | 1.01x | 1.686 | 1.674 | 1.01x | ok |
| float16 | 32x32 | 128 | 2.676 | 2.640 | 1.01x | 1.193 | 1.179 | 1.01x | ok |
| float16 | 32x32 | 256 | 2.723 | 2.691 | 1.01x | 1.179 | 1.165 | 1.01x | ok |
| float16 | 64x64 | 32 | 13.903 | 13.660 | 1.02x | 54.133 | 53.998 | 1.00x | ok |
| float16 | 64x64 | 64 | 18.995 | 18.820 | 1.01x | 30.239 | 30.225 | 1.00x | ok |
| float16 | 64x64 | 128 | 14.075 | 13.930 | 1.01x | 14.747 | 14.726 | 1.00x | ok |
| float16 | 64x64 | 256 | 17.431 | 17.307 | 1.01x | 8.998 | 8.961 | 1.00x | ok |
| float32 | 16x16 | 32 | 1.838 | 1.832 | 1.00x | 0.404 | 0.395 | 1.02x | ok |
| float32 | 16x16 | 64 | 2.003 | 1.994 | 1.00x | 0.353 | 0.346 | 1.02x | ok |
| float32 | 16x16 | 128 | 1.300 | 1.296 | 1.00x | 0.403 | 0.400 | 1.01x | ok |
| float32 | 16x16 | 256 | 1.664 | 1.664 | 1.00x | 0.459 | 0.467 | 0.98x | ok |
| float32 | 32x32 | 32 | 7.700 | 7.663 | 1.00x | 6.516 | 6.508 | 1.00x | ok |
| float32 | 32x32 | 64 | 8.216 | 8.218 | 1.00x | 3.367 | 3.376 | 1.00x | ok |
| float32 | 32x32 | 128 | 7.973 | 7.992 | 1.00x | 2.617 | 2.624 | 1.00x | ok |
| float32 | 32x32 | 256 | 9.505 | 9.529 | 1.00x | 2.971 | 2.979 | 1.00x | ok |
| float32 | 64x64 | 32 | 68.867 | 68.895 | 1.00x | (no launch) | (no launch) | - | ADJ-FAILED |
| float32 | 64x64 | 64 | 43.588 | 43.606 | 1.00x | (no launch) | (no launch) | - | ADJ-FAILED |
| float32 | 64x64 | 128 | 41.881 | 41.994 | 1.00x | (no launch) | (no launch) | - | ADJ-FAILED |
| float32 | 64x64 | 256 | 48.593 | 48.129 | 1.01x | (no launch) | (no launch) | - | ADJ-FAILED |
| float64 | 16x16 | 32 | 6.714 | 6.733 | 1.00x | 1.420 | 1.432 | 0.99x | ok |
| float64 | 16x16 | 64 | 6.645 | 6.689 | 0.99x | 0.993 | 1.007 | 0.99x | ok |
| float64 | 16x16 | 128 | 6.805 | 6.818 | 1.00x | 1.326 | 1.338 | 0.99x | ok |
| float64 | 16x16 | 256 | 6.770 | 6.791 | 1.00x | 1.313 | 1.313 | 1.00x | ok |
| float64 | 32x32 | 32 | 56.989 | 56.992 | 1.00x | 23.219 | 23.235 | 1.00x | ok |
| float64 | 32x32 | 64 | 49.838 | 49.870 | 1.00x | 13.219 | 13.224 | 1.00x | ok |
| float64 | 32x32 | 128 | 45.592 | 45.581 | 1.00x | 8.948 | 8.968 | 1.00x | ok |
| float64 | 32x32 | 256 | 46.882 | 46.806 | 1.00x | 8.004 | 8.008 | 1.00x | ok |
| float64 | 64x64 | 32 | 0.020 | 0.021 | 0.96x | (no launch) | (no launch) | - | MISMATCH |
| float64 | 64x64 | 64 | 0.019 | 0.017 | 1.16x | (no launch) | (no launch) | - | MISMATCH |
| float64 | 64x64 | 128 | 0.020 | 0.019 | 1.03x | (no launch) | (no launch) | - | MISMATCH |
| float64 | 64x64 | 256 | 0.019 | 0.020 | 0.95x | (no launch) | (no launch) | - | MISMATCH |
run-to-run spread (max-min)/median over 168 cells: median 4.3%, max 22.7%; cells above 1.5x median: 0

Raw bf16 forward timings per run, ms (`bench_<tree>_run<n>.csv`):
```
tree,dtype,M,block_dim,fwd_r1,fwd_r2,fwd_r3
scalar,bfloat16,16,32,1.1424,1.1498,1.1606
scalar,bfloat16,16,64,1.2399,1.2858,1.2959
scalar,bfloat16,16,128,0.9666,1.0189,1.0170
scalar,bfloat16,16,256,1.4241,1.5275,1.5353
scalar,bfloat16,32,32,4.3580,4.3693,4.4465
scalar,bfloat16,32,64,4.9709,5.0825,5.1888
scalar,bfloat16,32,128,6.2122,6.3577,6.5024
scalar,bfloat16,32,256,8.1289,8.5130,8.6934
scalar,bfloat16,64,32,69.2900,69.6030,69.6595
scalar,bfloat16,64,64,42.8238,44.5440,44.5028
scalar,bfloat16,64,128,32.1624,33.5842,33.3753
scalar,bfloat16,64,256,34.0359,36.0114,35.7351
shipped,bfloat16,16,32,0.2495,0.2537,0.2566
shipped,bfloat16,16,64,0.2708,0.2757,0.2775
shipped,bfloat16,16,128,0.3673,0.3784,0.3816
shipped,bfloat16,16,256,0.4325,0.4437,0.4479
shipped,bfloat16,32,32,1.6740,1.7329,1.7684
shipped,bfloat16,32,64,2.0115,2.0850,2.1228
shipped,bfloat16,32,128,2.7649,2.8817,2.9413
shipped,bfloat16,32,256,2.8058,2.9262,2.9950
shipped,bfloat16,64,32,11.2988,11.8199,12.1108
shipped,bfloat16,64,64,10.9296,11.7212,11.7018
shipped,bfloat16,64,128,12.8223,13.4590,13.8491
shipped,bfloat16,64,256,33.4074,34.5786,35.2329
```

### 5. 82-pattern set and CPU-only suite: match
- 82-pattern set: the Porting session 6 handoff B.5 list (`run82.sh`, below), `python -m warp.tests -s autodetect -p "<pattern>" -m 32` per pattern with a fresh cache each, from the clone, under `utils/timeit.sh warp-1.14 test`. Result: **82 patterns, 81 OK, 1 with the 2 known fem isolation failures**. `test_fem*` ran 107: `test_gimp_quadrature` and `test_cube_shape_functions` failed with "Unexpected output", the upstream isolation behaviour. The counts match the gfx1100 column of Porting session 13's table, and torch is installed here, so the torch-dependent files also run: test_array 139, test_dlpack 16, test_torch 34 and test_tile_mlp 3 (1 skipped). Also test_bf16 48 OK with no skip, test_mesh* 49 OK with no skip, test_coloring 3 OK, test_print 10 OK, test_streams 24 (3 skipped), test_texture 120 (1 skipped), test_tile_matmul 25 OK, and the tile files as in item 3. No log has a `Warp CUDA error` line. The only shared-memory warning is the known `test_tile_load_scoped` one.
- CPU-only suite: `HIP_VISIBLE_DEVICES=99 CUDA_VISIBLE_DEVICES=99 python -m warp.tests -s autodetect -k cpu -m 32`, with a fresh cache, run alongside the 82-pattern set (it does not touch the GPU). Result: **Ran 2294, OK (skipped=15)**. The log has no `cuda:0`.

### Scripts from this session
`bench_tile_matmul_bf16.py` is `bench_tile_matmul.py` (Porting session 4) with this diff:
```diff
@@ -1,5 +1,6 @@
 # tile_matmul throughput: square MxM tiles, one tile GEMM chain per block.
-# Usage: python bench_tile_matmul.py [--device cuda:0] [--mathdx on|off] [--csv out.csv]
+# Usage: python bench_tile_matmul_bf16.py [--device cuda:0] [--mathdx on|off] [--csv out.csv]
+# bfloat16 rows: bf16 A and B, float32 C, forward only (tile_matmul has no bf16 backward), so adj is n/a.
 # fwd:  REPS accumulating tile_matmul calls per block (C += A @ B), loads/stores amortized.
 # adj:  one C = A @ B per block, timed tape.backward (adj_A = adj_C B^T, adj_B = A^T adj_C);
 #       adj_gflops counts those 2 GEMMs (the backward kernel also replays the forward GEMM).
@@ -25,7 +26,7 @@
 dev = wp.get_device(args.device)
 
 SIZES = (16, 32, 64)
-DTYPES = (wp.float16, wp.float32, wp.float64)
+DTYPES = (wp.bfloat16, wp.float16, wp.float32, wp.float64)
 BLOCK_DIMS = (32, 64, 128, 256)
 REPS = args.reps
 
@@ -52,6 +53,32 @@
     return fwd, one
 
 
+def make_bf16_kernel(M):
+    @wp.kernel(module="unique", module_options={"enable_backward": False})
+    def fwd(A: wp.array3d(dtype=wp.bfloat16), B: wp.array3d(dtype=wp.bfloat16), C: wp.array3d(dtype=float)):
+        i = wp.tid()
+        a = wp.tile_load(A[i], shape=(M, M))
+        b = wp.tile_load(B[i], shape=(M, M))
+        c = wp.tile_zeros(shape=(M, M), dtype=float)
+        for _r in range(REPS):
+            wp.tile_matmul(a, b, c)
+        wp.tile_store(C[i], c)
+
+    return fwd
+
+
+@wp.kernel
+def to_bf16(x: wp.array3d(dtype=float), y: wp.array3d(dtype=wp.bfloat16)):
+    i, j, k = wp.tid()
+    y[i, j, k] = wp.bfloat16(x[i, j, k])
+
+
+def round_bf16(x):
+    u = x.astype(np.float32).view(np.uint32)
+    u = (u + np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1))) & np.uint32(0xFFFF0000)
+    return u.view(np.float32)
+
+
 def timed(fn, iters):
     fn()  # warm-up, includes module load
     wp.synchronize_device(dev)
@@ -72,6 +99,27 @@
 print("dtype,M,block_dim,fwd_ms,fwd_gflops,adj_ms,adj_gflops,check", flush=True)
 for dtype in DTYPES:
     for M in SIZES:
+        if dtype == wp.bfloat16:
+            fwd = make_bf16_kernel(M)
+            An = round_bf16(rng.uniform(-1, 1, (args.blocks, M, M)))
+            Bn = round_bf16(rng.uniform(-1, 1, (args.blocks, M, M)))
+            A = wp.zeros((args.blocks, M, M), dtype=wp.bfloat16, device=dev)
+            B = wp.zeros((args.blocks, M, M), dtype=wp.bfloat16, device=dev)
+            wp.launch(to_bf16, dim=An.shape, inputs=[wp.array(An, device=dev), A], device=dev)
+            wp.launch(to_bf16, dim=Bn.shape, inputs=[wp.array(Bn, device=dev), B], device=dev)
+            C = wp.zeros((args.blocks, M, M), dtype=float, device=dev)
+            for bd in BLOCK_DIMS:
+                def run_fwd():
+                    wp.launch_tiled(fwd, dim=[args.blocks], inputs=[A, B, C], block_dim=bd, device=dev)
+
+                fwd_ms = timed(run_fwd, args.iters)
+                ref = REPS * (An[0].astype(np.float64) @ Bn[0].astype(np.float64))
+                ok = np.allclose(C.numpy()[0].astype(np.float64), ref, rtol=1e-4, atol=1e-4 * REPS * M)
+                fl = 2.0 * M * M * M * args.blocks
+                r = ("bfloat16", M, bd, fwd_ms, fl * REPS / fwd_ms / 1e6, float("nan"), float("nan"), "ok" if ok else "MISMATCH")
+                rows.append(r)
+                print("%s,%d,%d,%.4f,%.1f,%.4f,%.1f,%s" % r, flush=True)
+            continue
         fwd, one = make_kernels(M, dtype)
         npt = wp.dtype_to_numpy(dtype)
         A = wp.array(rng.uniform(-1, 1, (args.blocks, M, M)).astype(npt), dtype=dtype, device=dev, requires_grad=True)
```

`bench_both.sh` (the `A` path is this job's `agent_space`):
```bash
#!/bin/bash
# bf16/fp16/fp32/fp64 tile_matmul benchmark, three runs each against the shipped tree and the scalar tree (gfx1100).
ulimit -c 0
A=<job>/moat/agent_space
mkdir -p $A/bf16/bench
for tree in ${TREES:-shipped scalar}; do
  if [ $tree = shipped ]; then src=$A/warp-src; else src=$A/scalar_wt; fi
  for n in 1 2 3; do
    PYTHONPATH=$src WARP_CACHE_PATH=$(mktemp -d $A/bf16/kc/bench.XXXXXX) "$FLEET_VENV/bin/python" $A/bf16/bench_tile_matmul_bf16.py \
      --blocks 16384 --iters 50 --csv $A/bf16/bench/bench_${tree}_run$n.csv > $A/bf16/bench/bench_${tree}_run$n.log 2>&1
    echo "$tree run$n rc=$?"
  done
done
```

`compare_bf16.py` (run in the directory that holds the six CSVs):
```python
# Merge the three runs of each tree (bench_<tree>_run<n>.csv in the current directory): median ms per cell,
# shipped speedup over scalar, run-to-run spread, cells above 1.5x their median. bfloat16 rows have no adjoint.
import csv
import math
import statistics
from collections import defaultdict

data = defaultdict(lambda: defaultdict(list))
checks = defaultdict(set)
for tree in ("scalar", "shipped"):
    for n in (1, 2, 3):
        for r in csv.DictReader(open(f"bench_{tree}_run{n}.csv")):
            key = (r["dtype"], int(r["M"]), int(r["block_dim"]))
            data[key][tree + "_fwd"].append(float(r["fwd_ms"]))
            if not math.isnan(float(r["adj_ms"])):
                data[key][tree + "_adj"].append(float(r["adj_ms"]))
            checks[key].add(f"{tree}:{r['check']}")
spread, outliers = [], 0
order = ["bfloat16", "float16", "float32", "float64"]
print("| dtype | tile | block_dim | scalar fwd ms | shipped fwd ms | fwd speedup | scalar adj ms | shipped adj ms | adj speedup | checks |")
print("|---|---|---|---|---|---|---|---|---|---|")
for key in sorted(data, key=lambda k: (order.index(k[0]), k[1], k[2])):
    d = data[key]
    m = {k: statistics.median(v) for k, v in d.items()}
    for v in d.values():
        med = statistics.median(v)
        spread.append((max(v) - min(v)) / med)
        outliers += sum(x > 1.5 * med for x in v)
    ck = sorted(checks[key])
    if key[0] == "bfloat16":
        adj = "n/a | n/a | -"
    elif all(c.endswith(":ok") for c in ck):
        adj = f"{m['scalar_adj']:.3f} | {m['shipped_adj']:.3f} | {m['scalar_adj'] / m['shipped_adj']:.2f}x"
    else:
        adj = "(no launch) | (no launch) | -"
    print(f"| {key[0]} | {key[1]}x{key[1]} | {key[2]} | {m['scalar_fwd']:.3f} | {m['shipped_fwd']:.3f} | {m['scalar_fwd'] / m['shipped_fwd']:.2f}x | {adj} | {'/'.join(sorted(set(c.split(':')[1] for c in ck)))} |")
spread.sort()
print(f"run-to-run spread (max-min)/median over {len(spread)} cells: median {100 * statistics.median(spread):.1f}%, max {100 * spread[-1]:.1f}%; cells above 1.5x median: {outliers}")
```

`monitor.sh`:
```bash
#!/bin/bash
# Every 5 s: load average and gpu_busy_percent of every amdgpu card. Usage: monitor.sh <log>
while true; do
  echo "$(date +%T) load $(cut -d' ' -f1 /proc/loadavg) busy $(for c in /sys/class/drm/card[0-9]/device/gpu_busy_percent; do printf '%s=%s ' $(basename $(dirname $(dirname $c))) $(cat $c 2>/dev/null); done)" >> "$1"
  sleep 5
done
```

`run82.sh` (paths shortened to `<A>` = this job's `agent_space`):
```bash
#!/bin/bash
# The 82-pattern Phase 1-5 set (Porting session 6 handoff B.5), one pattern per run, fresh kernel cache each.
ulimit -c 0
cd <A>/warp-src
n=0
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
  n=$((n+1)); log=<A>/bf16/p82/$(printf %02d $n)_$(echo "$p" | tr -d '*').log
  WARP_CACHE_PATH=$(mktemp -d <A>/bf16/kc/p82.XXXXXX) python -m warp.tests -s autodetect -p "$p" -m 32 > "$log" 2>&1
  echo "$p rc=$? $(grep -E '^Ran |^OK|^FAILED' "$log" | tr '\n' ' ')"
done
echo "patterns: $n"
```

The tile-set loop (item 3) is the session 4 loop, with a fresh `WARP_CACHE_PATH=$(mktemp -d ...)` for each file.

## Porting session 15 (2026-11-19, windows-gfx1151 laptop, porter): the remaining graph failures after de4e07a3

Bounded follow-up inside the windows-gfx1151 `porting` lock (lock and stage untouched). Fork `release/1.14.0.amd0` advanced `de4e07a3..6b639456` (pushed, `advance-head` recorded). Environment as session 14 (`agent_space/warp-win/p6env.sh`, ROCm 10.2 wheels, HIP 7.17.26434). Scripts in `agent_space/warp-win/p15/`, logs in `agent_space/warp-win/logs/p15/` (single-test runs and AMD_LOG_LEVEL logs) and `logs/p15v/` (verification). No driver reset, device-lost line or display glitch at any point.

Input: the 82-pattern rerun at `de4e07a3` (`logs/p15pat/summary.txt`): 79 OK; not OK were test_streams (`test_event_external`, the known upstream test race), test_graph (`test_cuda_graph_alloc_transient_stream_cuda_0`: launch error, invalid argument in `wp_cuda_graph_launch`, warp.cu:4411) and test_map (`test_graph_capture_cuda_0`: invalid argument in `wp_memcpy_d2h`, warp.cu:1086; `test_indexedarrays_cuda_0`: invalid argument in `wp_array_copy_device`, warp.cu:1901).

### Verdicts
- **Cross-stream relaunch hypothesis: does not hold.** `test_cuda_graph_alloc_transient_stream` (test_graph.py:443) captures `foo` (zeros `a`, ones `b`, 100 `accum_kernel` launches on a temporary `wp.Stream()`) and `bar` (empty `c`, copy from a pinned host array, on another temporary stream; both temporaries get the same handle in the HIP log because the first is destroyed before the second is created) and launches the graph **once**. No exec is ever relaunched, so no relaunch race can apply, on one stream or several.
- **New runtime defect (Windows, no direct dispatch): a graph allocation is invisible to pointer validation while its alloc node executes on the worker thread.** `GraphMemAllocNode::CreateCommand` (launching thread) adds the reserved VA to `amd::MemObjMap`; `VirtualMemAllocNode::submit` (worker thread, clr `hip_graph_internal.hpp`, rocm-systems `develop` today) on the first launch calls `RemoveMemObj(va)`, then `graph_->AllocateMemory` (milliseconds for 256 MiB) and only then maps and registers the sub-object. Any call on another thread that validates a pointer into that allocation inside the window returns `hipErrorInvalidValue`. On Linux the node runs on the calling thread inside `hipGraphLaunch`, so the window never overlaps application calls.
  - test_graph transient `.numpy()` failure and test_map `test_graph_capture`: the `.numpy()` right after `wp.capture_launch` issues `hipMemcpyAsync` D2H of a graph-allocated array. `logs/p15/ts_log3_3.log` (de4e07a3, AMD_LOG_LEVEL=3): `hipGraphLaunch` returns at 720941 us, the worker's `Pool AllocMem` for the first alloc node logs at 737987 us, `hipMemcpyAsync(..., 000001CA14000000 (= a), 268435456, D2H)` returns `hipErrorInvalidValue` at 738372 us, and the fill of `a` runs on the worker only at 766107 us. Standalone, with thread ids (`logs/p15/repro_d2h_log4.txt`): main tid 0x6429c `hipMemcpyAsync` at +24 us after the launch, worker tid 0x018d5 `Pool AllocMem` at +33 us, the copy returns `hipErrorInvalidValue` at +39 us.
  - test_map `test_indexedarrays`: collateral. It runs right after `test_graph_capture` in the same worker process; the failed `hipMemcpyAsync` leaves the thread's last error set (Warp's `check_cuda(result)` does not clear it), and the next `cudaGetLastError()` (in `wp_array_copy_device` after the copy kernel) returns it. Same semantics as CUDA. With the fix, all 45 test_map tests pass.
  - Not caused by `de4e07a3`: at `fda2f89a` (no workaround; built in `agent_space/warp-win/fixwt` with the uncommitted fix reverted, `python build_lib.py --hip-arch gfx1151 -j 8`, 1 min 39 s), the transient test alone failed in run 4 of 4 with the same `wp_memcpy_d2h` invalid argument (`logs/p15/ts_fda_4.log`). At `de4e07a3` alone: 1 of 3 plain runs failed (`ts_alone_2`), 1 of 3 AMD_LOG_LEVEL=3 runs (`ts_log3_3`). Session 13's and 14's passes were timing.
- **Still open, not fixed: a rarer first-launch failure inside `hipGraphLaunch` itself** (the p15pat test_graph traceback). `logs/p15/fix_log3_3.log` (fixed build): 44 us into the launch `:1:hip_graph_internal.cpp:3158: Command creation for node id(3) failed!`, `Failed to launch nodes!`, `hipGraphLaunch: Returned hipErrorInvalidValue`, before either worker logged its first `Pool AllocMem`. The graph runs on 2 parallel streams (`max_streams: 2`); in a passing AMD_LOG_LEVEL=4 run (`fix_log4_4.log`) worker 0x46e75 allocates `c` (bar's branch, a root of the graph) concurrently with worker 0x7225c allocating `a`, so the leaning is the same window hit from inside the launch: the launching thread creating the command for the memcpy node into `c` (validates `c`) while the other worker is inside `c`'s alloc node. Not proven: the standalone reproducer's `node` mode (a second allocation plus an H2D memcpy node on a forked root branch) and `memset` mode (memset node right after the alloc) passed 5 of 5 each, and 4 AMD_LOG_LEVEL=4 runs of the Warp test did not fail. A wait outside `hipGraphLaunch` cannot close a window inside it, so no Warp-side workaround is proposed for it. Frequency with the fix: 1 of 5 runs of the three tests together (`fix3_5`, which also failed the following `test_graph_capture` with error 901 "operation failed due to a previous error during capture" in `wp_alloc_device_async`, collateral of the failed launch), 1 of 3 AMD_LOG_LEVEL=3 runs of the transient test alone; 0 in the verification run below.

Failing-run budget: transient test 6 failing runs (ts_alone_2, ts_log3_3, ts_fda_4, fix3_5, fix_log3_3, plus the p15pat input), test_map 1 (fix3_5).

### Port fix 6b639456 [ROCm] Wait for each graph launch to finish on Windows
`warp/native/hip_compat.h`, Windows branch only: `wp_hip_graph_launch` now calls `hipGraphLaunch` first and then `hipStreamSynchronize(stream)` when the stream is not capturing (if `hipStreamIsCapturing` fails, it launches without waiting, as upstream would). Replaces `de4e07a3`'s wait-before-launch. Why after rather than per-exec bookkeeping or `hipDeviceSynchronize`: a finished launch cannot race the next launch of the same exec on any stream (so the relaunch race stays closed, including the cross-stream case the session asked about), and it is the only point outside the runtime where the alloc nodes are known to have executed; a per-exec "first launch" set would need the exec's destroy hooked and would still miss the remap path on relaunch (`remap_needed`, same RemoveMemObj-free window when another graph took the physical memory); `hipDeviceSynchronize` waits on unrelated streams and is the call most likely to invalidate a global-mode capture running on another stream. Cost: the host no longer overlaps work with a running graph on Windows; consecutive launches were already serialized by the runtime's own `launch_stream->finish()`. Linux HIP and CUDA builds unchanged. clang-format 22.1.2 clean, jargon clean.

Standalone reproducer `p15/graph_alloc_first_launch.hip` (built like session 14's, `R=$(python -m rocm_sdk path --root)`, `clang++ -x hip --rocm-path=$R --rocm-device-lib-path=$R/lib/llvm/amdgcn/bitcode --offload-arch=gfx1151 -O2 -w graph_alloc_first_launch.hip -o graph_alloc_first_launch.exe -L$R/lib -lamdhip64`, TheRock DLLs next to the exe):
- `graph_alloc_first_launch 5 256 d2h`: **FAIL 5 of 5** (`copy invalid argument` every iteration); `5 1 d2h`: FAIL 5 of 5.
- `5 256 sync` (hipStreamSynchronize after the launch, the fix): PASS 0 of 5.
- `5 256 node`, `5 256 memset`: PASS 0 of 5 (the in-launch variant did not reproduce).

Verification at 6b639456's code (`p15/run_verify.sh`: one pattern at a time, `python -m warp.tests -s autodetect -p "<pat>.py" -m 2`, fresh kernel cache per pattern, 900 s timeout; `logs/p15v/summary.txt`; this run used the build before the last edit, which only changed what happens when `hipStreamIsCapturing` itself fails): test_graph 35 OK (3 skipped); test_map 45 OK; test_mesh* 49 OK (18 skipped); test_grouped_bvh 11 OK; test_capture_mode 2 OK; test_array_fill_capture 3 OK; test_conditional_captures 32 OK (16 skipped); test_conditional 50 OK; test_streams 24 run, 1 failure = the known `test_event_external` upstream race (3 skipped); test_mempool 7 OK (2 skipped); test_apic 52 OK; test_apic_mesh 12 OK; test_fem* 108 OK (2 skipped), 579 s (p15pat: 344 s; fresh cache both times, so part of the gap may be the lost host/GPU overlap; not measured further, no benchmarks on this host). After the last edit, rebuilt (`python build_lib.py --hip-arch gfx1151 -j 8`, succeeded) and `p15/run_final.sh` (`logs/p15final/summary.txt`, same runner settings): test_graph 35 OK, test_map 45 OK, test_capture_mode 2 OK, test_conditional_captures 32 OK, test_mesh* 49 OK, test_grouped_bvh 11 OK. Also before the last edit: the three failing tests together (`python -m unittest warp.tests.test_graph.TestGraph.test_cuda_graph_alloc_transient_stream_cuda_0 warp.tests.test_map.TestMap.test_graph_capture_cuda_0 warp.tests.test_map.TestMap.test_indexedarrays_cuda_0`) 4 of 5 runs OK, run 5 the in-launch failure above. Session 14's `graph_autofree_race 10 2000 sync` mode (wait before each launch) is equivalent to the new wait after each launch, as nothing runs between launches; not rerun.

Registered `hip-graph-alloc-node-va-unregistered-on-worker` (rocm-bug-report, HIP, unruled) for the new defect. `hip-graph-autofree-relaunch-races-worker-memfree` was not amended (an amend clears Jeff's ruling); its last sentence ("Warp now synchronizes before hipGraphLaunch") is now stale: Warp synchronizes after it.

```cpp
// First launch of a graph holding a stream-ordered allocation that outlives the graph (alloc node,
// no free node), instantiated with hipGraphInstantiateFlagAutoFreeOnLaunch. Without direct
// dispatch the alloc node runs on the runtime's worker thread, which removes the allocation's
// virtual address from the runtime's pointer table, allocates and maps physical memory, and only
// then registers it again. A pointer-validating call made on another thread inside that window
// fails with hipErrorInvalidValue.
// Usage: graph_alloc_first_launch [iters] [MiB] [mode]
//   mode d2h  : hipMemcpyAsync D2H from the allocation right after hipGraphLaunch returns (default)
//   mode node : the graph also allocates a second buffer on a forked capture stream, in parallel
//               with the first allocation, and copies host data into it, so the launching thread
//               validates that pointer inside hipGraphLaunch while another worker thread allocates
//   mode sync : as d2h, with hipStreamSynchronize after hipGraphLaunch (the workaround)
//   mode memset : the graph memsets the allocation right after allocating it, so the launching
//               thread validates the pointer inside hipGraphLaunch while the worker allocates
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { printf("%s:%d %s -> %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e)); return 1; } } while (0)

__global__ void fill(float* buf, size_t n, float v)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) buf[i] = v;
}

int main(int argc, char** argv)
{
    int iters = argc > 1 ? atoi(argv[1]) : 5;
    size_t mib = argc > 2 ? (size_t)atoi(argv[2]) : 256;
    const char* mode = argc > 3 ? argv[3] : "d2h";
    bool node = !strcmp(mode, "node"), sync = !strcmp(mode, "sync"), mset = !strcmp(mode, "memset");
    size_t bytes = mib << 20, n = bytes / 4;
    CK(hipSetDevice(0));
    hipStream_t s, s2;
    CK(hipStreamCreate(&s));
    CK(hipStreamCreate(&s2));
    hipEvent_t fork, join;
    CK(hipEventCreateWithFlags(&fork, hipEventDisableTiming));
    CK(hipEventCreateWithFlags(&join, hipEventDisableTiming));
    float* host;
    CK(hipHostMalloc((void**)&host, bytes, 0));
    for (size_t i = 0; i < n; ++i) host[i] = 3.f;

    int bad = 0;
    for (int it = 0; it < iters; ++it) {
        hipGraph_t g;
        hipGraphExec_t ge;
        float *a, *c = nullptr;
        CK(hipStreamBeginCapture(s, hipStreamCaptureModeGlobal));
        if (node) {
            CK(hipEventRecord(fork, s));
            CK(hipStreamWaitEvent(s2, fork, 0));
        }
        CK(hipMallocAsync((void**)&a, bytes, s));
        if (mset) CK(hipMemsetAsync(a, 0, bytes, s));
        fill<<<(unsigned)((n + 255) / 256), 256, 0, s>>>(a, n, 1.f);
        if (node) {
            CK(hipMallocAsync((void**)&c, bytes, s2));
            CK(hipMemcpyAsync(c, host, bytes, hipMemcpyHostToDevice, s2));
            CK(hipEventRecord(join, s2));
            CK(hipStreamWaitEvent(s, join, 0));
        }
        CK(hipStreamEndCapture(s, &g));
        CK(hipGraphInstantiateWithFlags(&ge, g, hipGraphInstantiateFlagAutoFreeOnLaunch));

        hipError_t el = hipGraphLaunch(ge, s), ec = hipSuccess;
        if (el == hipSuccess && sync) el = hipStreamSynchronize(s);
        if (el == hipSuccess && !node && !mset) ec = hipMemcpyAsync(host, a, 4096, hipMemcpyDeviceToHost, s);
        printf("iter %d: launch %s, copy %s\n", it, hipGetErrorString(el), hipGetErrorString(ec));
        fflush(stdout);
        if (el != hipSuccess || ec != hipSuccess) { ++bad; (void)hipGetLastError(); }
        CK(hipStreamSynchronize(s));
        if (!node && !mset && ec == hipSuccess && host[0] != 1.f) { printf("iter %d: wrong value %f\n", it, host[0]); ++bad; }
        for (size_t i = 0; i < 1024; ++i) host[i] = 3.f;
        CK(hipFreeAsync(a, s));
        if (c) CK(hipFreeAsync(c, s));
        CK(hipStreamSynchronize(s));
        CK(hipGraphExecDestroy(ge));
        CK(hipGraphDestroy(g));
    }
    printf("%s: %d of %d iterations failed (%zu MiB, mode %s)\n", bad ? "FAIL" : "PASS", bad, iters, mib, mode);
    return bad ? 1 : 0;
}
```

### Proposed skip (not added; Jeff's decision)
If the in-launch failure is to be kept out of release runs before the runtime is fixed: a capability-named skip on `test_cuda_graph_alloc_transient_stream`, "first launch of a multi-stream graph with memory nodes can fail in hipGraphLaunch on HIP without direct dispatch", Windows HIP only.

### Next step
- Jeff: the runtime bug report gains a second defect (the alloc-node window, deterministic in the standalone reproducer, `graph_alloc_first_launch 5 1 d2h`), either as a second report or a second section of the same one.
- The in-launch variant stays open as a known intermittent Windows failure of one test (about 1 in 3 to 5 runs alone); decide skip versus ship-as-known.
- Then continue session 13's Next step (review, then the Windows validation record) at the new head.

## Rulings on the Windows graph-launch workaround (2026-11-19, Jeff Daily)

- Keep the Windows-only workaround in `hip_compat.h` and report the runtime bug (ruled on `de4e07a3`, wait before launch). The relaunch race is filed as https://github.com/ROCm/rocm-systems/issues/16265.
- After porting session 15 changed the form to launch-then-wait (`6b639456`, which also covers the first-launch allocation window): keep launch-then-wait. Accepted cost: on Windows HIP a graph launch blocks until the graph finishes; not measured. This goes in the release notes.
- A Linux check on gfx950 (plan `hip-graph-autofree-relaunch-direct-dispatch-check-mi355.md`) showed the runtime forces direct dispatch on for Linux even with `AMD_DIRECT_DISPATCH=0`, and the relaunch reproducer passed 30 of 30 runs on ROCm 10.1 and 10.2, so the `_WIN32` guard is the right scope.
- Still open for Jeff: the intermittent in-launch failure of `test_cuda_graph_alloc_transient_stream`, whether to report `hip-graph-alloc-node-va-unregistered-on-worker`, and the upstream `test_event_external` race.


## Porting session 16 (2026-11-19, windows-gfx1151 laptop, porter)

Bounded follow-up to session 15 under Jeff's two-part ruling of today (one more round to reproduce the in-launch failure standalone, then a capability-worded skip of `test_cuda_graph_alloc_transient_stream` on Windows HIP). Lock and stage untouched (`porting`, windows-gfx1151). Fork `release/1.14.0.amd0` advanced `6b639456..9d9a6161` (pushed, `advance-head` recorded). Environment as session 15 (`agent_space/warp-win/p6env.sh`). Scripts and reproducer in `agent_space/warp-win/p16/`, logs in `agent_space/warp-win/logs/p16/` (reproducer runs, `summary.txt`), `logs/p16v/` and `logs/p16v2/` (test_graph verification). No driver reset, device-lost line or display glitch; two Warp test processes hung (below) and were killed with `taskkill /T /F`.

### Part 1: the in-launch failure, identified and reproduced standalone
- **What node id(3) is (from reading the source and the failing log).** clr `hip_graph_internal.cpp` (rocm-systems clone `agent_space/rocm-sys` at 8a0a4db, 2026-10-30; its line numbers run about 6 below the installed runtime's): `Graph::RunNodes` walks `GetTopoOrder()`, and `RunOneNode` logs `Command creation for node id(%d) failed!` with `current_id_ + 1`, where `current_id_` counts the nodes already submitted, so id(3) is the third node in launch order. `Graph::TopologicalOrder` is Kahn's algorithm (FIFO queue, roots in vertex order). The test's graph (`logs/p15/fix_log3_3.log`) has two roots, `alloc a` (foo, first temporary stream) and `alloc c` (bar, second temporary stream), so the order is [alloc a, alloc c, memset a (from `wp.zeros`), memcpy H2D into c, ...]. Consistent with the log: after the failed launch the workers logged exactly two `Pool AllocMem` lines (the two enqueued alloc nodes) during Warp's `hipDeviceSynchronize`. **id(3) is the memset of `a`, not the memcpy into `c`** (session 15's leaning named the memcpy).
- **Why the memset and not the memcpy (reading).** `GraphMemsetNode::CreateCommand` (hip_graph_internal.hpp, class at line 3115) calls `getMemoryObjectForCurrentDevice(memsetParams_.dst)` and returns `hipErrorInvalidValue` when it is null, with no lock. `GraphMemcpyNode1D::CreateCommand` (from line 2535) takes `GraphNode::WorkerThreadLock_` around the same lookups when `!AMD_DIRECT_DISPATCH`, and `VirtualMemAllocNode::submit` holds that lock from `RemoveMemObj(va)` through `AllocateMemory` and `VirtualMapCommand::submit` (the re-registration). The memcpy waits the window out; the memset does not. Also from reading: `GraphExecClassic::Run` with `hipGraphInstantiateFlagAutoFreeOnLaunch` calls `ReleaseCachedMapping` on every alloc node before each launch, which clears `mapped_`, so every launch of a Warp graph (Warp always instantiates with AutoFreeOnLaunch) takes the first-launch path (`RemoveMemObj`, allocate, map), not only the first. Not measured.
- **Standalone reproducer** `p16/graph_memset_first_launch.hip` (source below; built as in session 15: `R=$(python -m rocm_sdk path --root)`, `"$R/lib/llvm/bin/clang++" -x hip --rocm-path=$R --rocm-device-lib-path=$R/lib/llvm/amdgcn/bitcode --offload-arch=gfx1151 -O2 -w graph_memset_first_launch.hip -o graph_memset_first_launch.exe -L$R/lib -lamdhip64`, TheRock DLLs next to the exe). One launch per process via `p16/runv.sh <tag> <MiB> <mode>` (at most 20 process starts, stop at the second failure, one GPU process at a time):

| variant | runs | launch failures |
|---|---|---|
| `two`, 1 MiB (two root branches: alloc + memset + kernel; alloc + H2D memcpy from pinned memory) | 11 | **2** (runs 7 and 11, `launch invalid argument`) |
| `two`, 1 MiB, `AMD_LOG_LEVEL=1` (to capture the node id) | 20 | 0 |
| `nomemset`, 1 MiB (control: memset node removed, kernel and memcpy node kept) | 21 (20 + 1 manual) | 0 |

  Reproduced standalone at 1 MiB, and the control without the memset never failed. 2 of 11 against 0 of 21 is suggestive, not statistically strong (about p = 0.02 at a 2-in-11 rate). The logged variant did not fail, so the node id is not confirmed in a standalone log; the identification rests on the source and the Warp log above. Session 15's single-root `memset` mode (alloc then memset on one stream, 5 graphs in one process) passed 5 of 5; the second root's alloc node probably widens the gap between enqueueing `alloc a` and creating the memset command, giving the worker time to enter the window (not measured).
- **Same defect on a single-stream graph, through Warp.** During the part 2 verification `test_graph_alloc_cuda_0` (test_graph.py:292: capture `wp.zeros(128)` (alloc 512 B + memset) and two kernels; launch, then replay) failed in 2 of 6 test_graph runs with `Warp CUDA error 1: invalid argument (in function wp_cuda_graph_launch, warp.cu:4411)`. Which of its two launches failed is not known (the process was killed before unittest printed the traceback). Both times the next test, `test_graph_launch_array_access_mode_checked_cuda_capture_cuda_0`, then hung (no progress for 5 and 3 minutes; killed). Session 15 and the other runs here passed it (p15v, p15final, p16v runs 1-2, p16v2 runs 1 and 3), so it is intermittent like the transient test. Not skipped: the ruling names only the transient test.

```cpp
// One first launch per process of a graph with two root branches on separate capture streams:
//   branch 1: hipMallocAsync(a), hipMemsetAsync(a), fill kernel on a [, alloc b, kernels, free b]
//   branch 2: hipMallocAsync(c), hipMemcpyAsync(c <- pinned host)
// joined into the origin stream; no free of a or c inside the graph; instantiated with
// hipGraphInstantiateFlagAutoFreeOnLaunch. The executable runs on 2 parallel streams.
// Kahn order is [alloc a, alloc c, memset a, memcpy c, ...]: the launching thread creates the
// memset command for a after enqueueing alloc a (worker thread) and alloc c.
// Usage: graph_memset_first_launch [MiB] [mode]
//   mode two  : the structure above (default)
//   mode warp : branch 1 also allocates b, fills it, runs 100 accumulate kernels and frees b,
//               like Warp's test_cuda_graph_alloc_transient_stream
//   mode nomemset : as two without the memset node (control: the kernel and the memcpy node remain)
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { printf("%s:%d %s -> %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e)); return 2; } } while (0)

__global__ void fill(float* buf, size_t n, float v)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) buf[i] = v;
}

__global__ void accum(float* a, const float* b, size_t n)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] += b[i];
}

int main(int argc, char** argv)
{
    size_t mib = argc > 1 ? (size_t)atoi(argv[1]) : 1;
    const char* mode = argc > 2 ? argv[2] : "two";
    bool warp = !strcmp(mode, "warp"), nomemset = !strcmp(mode, "nomemset");
    size_t bytes = mib << 20, n = bytes / 4;
    unsigned blocks = (unsigned)((n + 255) / 256);
    CK(hipSetDevice(0));
    hipStream_t s, s1, s2;
    CK(hipStreamCreate(&s));
    CK(hipStreamCreate(&s1));
    CK(hipStreamCreate(&s2));
    hipEvent_t f1, j1, j2;
    CK(hipEventCreateWithFlags(&f1, hipEventDisableTiming));
    CK(hipEventCreateWithFlags(&j1, hipEventDisableTiming));
    CK(hipEventCreateWithFlags(&j2, hipEventDisableTiming));
    float* host;
    CK(hipHostMalloc((void**)&host, bytes, 0));
    for (size_t i = 0; i < n; ++i) host[i] = 42.f;

    hipGraph_t g;
    hipGraphExec_t ge;
    float *a, *b, *c;
    CK(hipStreamBeginCapture(s, hipStreamCaptureModeGlobal));
    CK(hipEventRecord(f1, s));
    CK(hipStreamWaitEvent(s1, f1, 0));
    CK(hipStreamWaitEvent(s2, f1, 0));
    CK(hipMallocAsync((void**)&a, bytes, s1));
    if (!nomemset) CK(hipMemsetAsync(a, 0, bytes, s1));
    if (warp) {
        CK(hipMallocAsync((void**)&b, bytes, s1));
        fill<<<blocks, 256, 0, s1>>>(b, n, 1.f);
        for (int k = 0; k < 100; ++k) accum<<<blocks, 256, 0, s1>>>(a, b, n);
        CK(hipFreeAsync(b, s1));
    } else {
        fill<<<blocks, 256, 0, s1>>>(a, n, 1.f);
    }
    CK(hipMallocAsync((void**)&c, bytes, s2));
    CK(hipMemcpyAsync(c, host, bytes, hipMemcpyHostToDevice, s2));
    CK(hipEventRecord(j1, s1));
    CK(hipEventRecord(j2, s2));
    CK(hipStreamWaitEvent(s, j1, 0));
    CK(hipStreamWaitEvent(s, j2, 0));
    CK(hipStreamEndCapture(s, &g));
    size_t nn = 0;
    CK(hipGraphGetNodes(g, nullptr, &nn));
    size_t nr = 0;
    CK(hipGraphGetRootNodes(g, nullptr, &nr));
    CK(hipGraphInstantiateWithFlags(&ge, g, hipGraphInstantiateFlagAutoFreeOnLaunch));

    hipError_t el = hipGraphLaunch(ge, s);
    printf("nodes %zu roots %zu: launch %s\n", nn, nr, hipGetErrorString(el));
    fflush(stdout);
    (void)hipGetLastError();
    CK(hipStreamSynchronize(s));
    int bad = el != hipSuccess;
    if (!bad) {
        float va = 0.f, vc = 0.f;
        CK(hipMemcpy(&va, a, 4, hipMemcpyDeviceToHost));
        CK(hipMemcpy(&vc, c, 4, hipMemcpyDeviceToHost));
        float want = warp ? 100.f : 1.f;
        if (va != want || vc != 42.f) { printf("wrong values a=%f c=%f\n", va, vc); bad = 1; }
    }
    CK(hipGraphExecDestroy(ge));
    CK(hipGraphDestroy(g));
    CK(hipFreeAsync(a, s));
    CK(hipFreeAsync(c, s));
    CK(hipStreamSynchronize(s));
    printf("%s (%zu MiB, mode %s)\n", bad ? "FAIL" : "PASS", mib, mode);
    return bad;
}
```

### Part 2: skip 9d9a6161 [ROCm] Skip transient-stream graph test on Windows HIP
`warp/tests/unittest_utils.py` gains, beside the other HIP capability helpers (`is_graph_host_copy_of_capture_allocation_supported`, `is_multi_device_graph_supported`, `is_mempool_access_revocable`), the reason constant `GRAPH_LAUNCH_WITH_MEMSET_OF_CAPTURE_ALLOCATION_UNRELIABLE` = "HIP without direct dispatch can fail to launch a graph that memsets memory allocated during the capture (hipGraphLaunch returns hipErrorInvalidValue)" and `is_graph_launch_with_memset_of_capture_allocation_reliable(device)`, which returns `not (sys.platform == "win32" and any(d.is_hip for d in _devices(device)))`. `test_cuda_graph_alloc_transient_stream` skips when it is false (after its existing host-copy check). Warp has no device property for direct dispatch and adding one needs a native change, so the platform test sits inside the helper, as the existing helpers hold their `is_hip` tests. The wording dropped "first launch" and "multi-stream" after the evidence above (AutoFreeOnLaunch remaps on every launch; test_graph_alloc is single-stream). 20 added lines; ruff 0.15.9 (the pre-commit pin) check and format clean; jargon clean. If Jeff rules test_graph_alloc in too, it is the same two lines in that test.

What other platforms see (reading the condition): Linux HIP has `sys.platform == "linux"`, so the helper returns True and the test runs as before; on CUDA builds `is_hip` is False for every device, so it runs; the test is CUDA-only, so CPU is not involved.

Verification (`p16/run_graph3.sh` and `run_graph3b.sh`, as run82.sh: `python -m warp.tests -s autodetect -p "test_graph.py" -m 2`, fresh kernel cache per run): p16v (identifiers and wording before the final edit, same condition) runs 1-2 `Ran 35 tests OK (skipped=4)`, run 3 test_graph_alloc ERROR then hang; p16v2 (final code) run 1 OK (skipped=4), run 2 test_graph_alloc ERROR then hang, run 3 OK (skipped=4). The transient test reported `skipped 'HIP without direct dispatch can fail to launch a graph ...'` in every run (skips 3 -> 4 against session 15's 35 OK (3 skipped)). So 4 of 6 runs OK, not the expected 6 of 6; the 2 failures are the same runtime defect in a test the skip does not cover.

Deferred entry `hip-graph-alloc-node-va-unregistered-on-worker` amended (it was unruled): the in-launch variant is the memset node's unlocked lookup, reproduced, also hit on a single-stream graph, and by reading on every AutoFreeOnLaunch launch; proposed runtime fix: take `WorkerThreadLock_` in `GraphMemsetNode::CreateCommand` as the memcpy node does, and/or keep the VA registered while the alloc node maps.

Draft text for the bug report, in-launch section: "Without direct dispatch (Windows), hipGraphLaunch intermittently returns hipErrorInvalidValue, logging 'Command creation for node id(N) failed!', when a memset node follows a memory allocation node. VirtualMemAllocNode::submit runs on the worker thread and, holding GraphNode::WorkerThreadLock_, removes the reserved VA from amd::MemObjMap, allocates and maps physical memory, then registers it again; with hipGraphInstantiateFlagAutoFreeOnLaunch this happens on every launch, because ReleaseCachedMapping clears mapped_. Meanwhile the launching thread creates the commands of later nodes. GraphMemcpyNode1D::CreateCommand takes WorkerThreadLock_ around its getMemoryObjectForCurrentDevice lookups, but GraphMemsetNode::CreateCommand does not, so its lookup can miss the allocation and it returns hipErrorInvalidValue. Reproducer graph_memset_first_launch.hip (two root branches captured on separate streams: hipMallocAsync + hipMemsetAsync + kernel; hipMallocAsync + host-to-device hipMemcpyAsync from pinned memory; 1 MiB; AutoFreeOnLaunch; one launch per process): the launch failed in 2 of 11 process runs on a Radeon 8060S (gfx1151), HIP 7.17.26434, Windows 11; with the memset node removed, 0 of 21. After such a failed launch, later work in the same process can hang. Suggested fix: take WorkerThreadLock_ in GraphMemsetNode::CreateCommand as the memcpy node does, or keep the VA registered (replace the MemObjMap entry atomically) while the allocation node maps physical memory."

### Next step
- Jeff: whether to skip `test_graph_alloc` on Windows HIP too (same defect, 2 of 6 test_graph runs here, and the test after it hung both times), with the same helper, or ship it as known.
- Jeff: report `hip-graph-alloc-node-va-unregistered-on-worker` (both parts; draft above), and the upstream `test_event_external` race (still open).
- Then continue session 13's Next step (review, then the Windows validation record) at 9d9a6161.

## Porting session 17 (2026-11-19, windows-gfx1151 laptop, porter)

Follow-up to session 16 under Jeff's ruling of today: on Windows HIP, while a stream is capturing, fill with a kernel instead of `hipMemsetAsync`, so no memset node enters the graph; if it works, the `9d9a6161` skip comes out. Lock and stage untouched (`porting`, windows-gfx1151). Fork `release/1.14.0.amd0` advanced `9d9a6161..3d74932e` (pushed, `advance-head` recorded). Environment as session 15 (`agent_space/warp-win/p6env.sh`). Scripts in `agent_space/warp-win/p17/`, logs in `agent_space/warp-win/logs/p17/` (builds, reproducer), `logs/p17v/` and `logs/p17v2/` (test_graph x6 on each build), `logs/p17r/` (the pattern set). No driver reset, device-lost line, hang or display glitch.

### Premise check (clr source, rocm-systems clone `agent_space/rocm-sys` at 8a0a4db; rocclr files read with `git show HEAD:` because the clone is sparse)
- **The premise is only half right.** A kernel node's command creation does look its pointer arguments up in `amd::MemObjMap` on the launching thread, without `WorkerThreadLock_`: `GraphKernelNode::CreateCommand` (hipamd/src/hip_graph_internal.hpp:1999) -> `ihipLaunchKernelCommand` (hip_module.cpp:689) -> `ihipCreateKernelCommand` -> `NDRangeKernelCommand::captureHIPArgsAndValidate` (hip_module.cpp:664; rocclr/platform/command.cpp:702) -> `KernelParameters::captureHIPArgs` (rocclr/platform/kernel.cpp:93), which for every `T_POINTER` argument does `if (!AMD_DIRECT_DISPATCH) memArg = amd::MemObjMap::FindMemObj(ptr)` (kernel.cpp:106-112). **But a miss is not an error**: it stores `nullptr` in the command's memory-object slot and returns true, so command creation succeeds and the launch does not fail. The memset node, by contrast, returns `hipErrorInvalidValue` on a miss (hip_graph_internal.hpp:3194/3204).
- **What a kernel-node miss costs (reading only, not observed).** At submit, PAL's `VirtualGPU::submitKernelInternal` (rocclr/device/pal/palvirtual.cpp, about 3834-3853) skips a null memory object: no `addVmMemory` (residency list) and no `memoryDependency().validate` (the intra-queue RAW/WAW barrier tracking, enabled with `GPU_NUM_MEM_DEPENDENCY` = 256 for this ASIC family, palsettings.cpp:221), except that when the device reports fine-grained system SVM a null pointer adds a conservative barrier. So a miss can in principle drop a dependency barrier between two kernels on one queue. Every Warp kernel that takes a capture allocation already passes through this same lookup (the kernels after `wp.zeros` in a capture always did), so the fill kernel adds no new kind of exposure; it removes the one node type that turns the miss into a launch failure. No wrong values seen in any run below.
- **Other node types whose command creation looks pointers up without the lock (would fail the same way on a miss):** memset (`GraphMemsetNode::CreateCommand`, :3186, both the 1D and 2D lookups); memcpy 2D/3D (`GraphMemcpyNode::CreateCommand`, :2284 -> `ihipMemcpy3DCommand` -> `ihipGetMemcpyParam3DCommand`, hip_memory.cpp:2345, which dispatches to the HtoD/DtoH/DtoD/array command builders that resolve device pointers); driver-API memcpy (`GraphDrvMemcpyNode`, :3904, same builder); memcpy from/to symbol when not delegated to the 1D path (`GraphMemcpyNodeFromSymbol` :2843 lookups at :2862-2863, `GraphMemcpyNodeToSymbol` :2996 lookups at :3015-3016, both `return hipErrorInvalidValue` on a miss). Locked: `GraphMemcpyNode1D::CreateCommand` (:2536, lock at :2549 when `!AMD_DIRECT_DISPATCH`). Memory free nodes use `FindVirtualMemObj` (:3858), a different map, unlocked; not examined further. Event record/wait, empty, host and external-semaphore nodes take no device pointer.
- **What Warp captures after a capture-time allocation:** kernels (unlocked lookup, benign miss as above), `wp_memset_device` (memset node: the failure), 1D `cudaMemcpyAsync` H2D/D2H/D2D/Default (locked), `cudaFreeAsync` (free node). Warp's native code calls no 2D/3D or symbol memcpy on a capture path (`grep -rn "cudaMemcpy2D\|cudaMemcpy3D\|cudaMemcpyToSymbol" warp/native`: only cuBQL's macro table and nanovdb's DeviceBuffer, neither on a capture path). So the change covers only the memset. `wp_memtile_device`'s 1-byte branch calls synchronous `cudaMemset` (null stream), which does not add a node to Warp's capture stream; left alone.

### The change: 19b6a5ad [ROCm] Fill with a kernel instead of a memset node on Windows captures
`warp/native/warp.cu`, +20/-0, all under `#if WP_ENABLE_HIP && defined(_WIN32)`: a forward declaration of the existing `memtile_value_kernel` template before `wp_memset_device`, and in `wp_memset_device` one branch at the single `cudaMemsetAsync` call site: when `wp_cuda_stream_is_capturing(cuda_stream)`, and `n > 0`, launch on `cuda_stream` the existing `memset_kernel` (int words, pattern `byte * 0x01010101u`) when `dest` and `n` are 4-byte aligned, otherwise `memtile_value_kernel<uint8_t>` (one thread per byte); `check_cuda(cudaGetLastError())` after the launch. Zero-length fill launches nothing (returns true, as `hipMemsetAsync(.., 0, ..)` does). Handles any memset value (memset semantics, low byte), not only zero: the defect is the node type, and the branch is no simpler for zero only. Outside a capture the call is the unchanged `cudaMemsetAsync`. APIC recording after it is untouched. clang-format 22.1.2 clean (CR-stripped diff against the formatted file is empty).

Other paths, from the diff: the Linux HIP build compiles `warp.cu` without `_WIN32`, so its preprocessed source, host and device, is identical to before (the only new tokens sit inside the guard); the CUDA build has `WP_ENABLE_HIP` = 0, so its translation unit is unchanged as well. The Linux nvcc check still runs later on another host.

### Reproducer variant (`p17/graph_memset_first_launch.hip`, mode `kernel`)
Session 16's program with a fourth mode, same graph otherwise (built with the same command, TheRock DLLs next to the exe; `p17/runv.sh kernel 1 kernel`, 20 process starts, one at a time):

```diff
+//   mode kernel : as two with the memset replaced by a zero-fill kernel (same graph otherwise)
+__global__ void zero_bytes(unsigned char* buf, size_t n)
+{
+    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
+    if (i < n) buf[i] = 0;
+}
-    bool warp = !strcmp(mode, "warp"), nomemset = !strcmp(mode, "nomemset");
+    bool warp = !strcmp(mode, "warp"), nomemset = !strcmp(mode, "nomemset"), kfill = !strcmp(mode, "kernel");
-    if (!nomemset) CK(hipMemsetAsync(a, 0, bytes, s1));
+    if (kfill) {
+        zero_bytes<<<(unsigned)((bytes + 255) / 256), 256, 0, s1>>>((unsigned char*)a, bytes);
+        CK(hipGetLastError());
+    } else if (!nomemset) {
+        CK(hipMemsetAsync(a, 0, bytes, s1));
+    }
```

| variant | runs | launch failures |
|---|---|---|
| `two` (session 16, memset node) | 11 | 2 |
| `nomemset` (session 16) | 21 | 0 |
| `kernel`, 1 MiB (this session) | 20 | **0** (20 PASS lines, values checked) |

### Verification
- Builds: `python build_lib.py --hip-arch gfx1151 -j 8` (wrapped in `utils/timeit.sh warp-1.14 compile -- bash -c "cd <src> && <venv python> build_lib.py ..."`), "Warp build succeeded", twice: build 1 with a first draft (it launched a zero-size grid for an unaligned zero-length fill and computed the word pattern in signed int), build 2 with the committed code. For value 0 and n > 0 the two drafts execute identically.
- Graph shape check (`p17/node_types.py`, hipGraphGetNodes/hipGraphNodeGetType through ctypes): a capture of `wp.zeros(131, int8)` plus `a.zero_()` on a pre-filled array yields node types `[10, 0, 0]` (alloc, kernel, kernel): no memset node.
- Zero-fill values (`p17/zero_fill_check.py`, compared with NumPy on the CPU): for 1, 3, 128 and 1000003 bytes, `int8` `zero_()` in a capture on a 7-filled array, `zero_()` of `b[1:]` (odd address) leaving `b[0]` = 7, `wp.zeros` allocated in the capture followed by an add-one kernel, and `zero_()` of a 3-byte `vector(3, int8)` array (1, 1, 42, 333334 elements): all 16 checks OK on build 1 and on build 2.
- test_graph with the `9d9a6161` skip removed (working tree), six runs each as run82.sh (`python -m warp.tests -s autodetect -p "test_graph.py" -m 2`, fresh kernel cache per run, 900 s watchdog that kills the process tree): build 1 (`logs/p17v`) 6 of 6 `Ran 35 tests OK (skipped=3)`; build 2 (`logs/p17v2`) 6 of 6 `Ran 35 tests OK (skipped=3)`. `test_cuda_graph_alloc_transient_stream_cuda_0 ... ok` and `test_graph_alloc_cuda_0 ... ok` in all 12. Session 16 for comparison: 2 of 6 runs failed test_graph_alloc and then hung.
- Pattern set on build 2, once each (`p17/run_rest.sh`, fresh cache per pattern, 1800 s watchdog; `logs/p17r/summary.txt`): test_graph 35 OK (3 skipped); test_map 45 OK; test_capture_mode 2 OK; test_array_fill_capture 3 OK; test_conditional_captures 32 OK (16 skipped); test_conditional 50 OK; test_mesh* 49 OK (18 skipped); test_grouped_bvh 11 OK; test_array 138 OK; test_copy 14 OK; test_streams 24 run, 1 failure = the known upstream `test_event_external` race (3 skipped); test_mempool 7 OK (2 skipped); test_apic* 64 OK; test_fem* 108 OK (2 skipped), 307 s (session 15: 579 s, not investigated, no benchmarks on this host). The transient test ran (no skip) in every test_graph run.

### Skip removal: 3d74932e [ROCm] Run the transient-stream graph test on Windows HIP again
`git show 9d9a6161 -- warp/tests | git apply -R` (the helper `is_graph_launch_with_memset_of_capture_allocation_reliable`, the reason `GRAPH_LAUNCH_WITH_MEMSET_OF_CAPTURE_ALLOCATION_UNRELIABLE` and its use in `test_cuda_graph_alloc_transient_stream`), committed as a new commit; ruff 0.15.9 check and format clean (the files return to their pre-`9d9a6161` content).

### Next step
- Review, then the Windows validation record, at the new head (session 13's Next step). Every completed platform now reads `revalidate`; the Linux platforms need only a rebuild-and-run, the change being Windows-only by construction, so a carry-forward on a binary-equivalence build is a candidate there (a person or the validator decides).
- CUDA nvcc compile check on the mi355 host before presenting (the CUDA translation unit is unchanged by construction, but the rule is to compile it).
- Jeff: the session 16 bug report draft for `hip-graph-alloc-node-va-unregistered-on-worker` still stands; add the kernel-node finding above (unlocked `FindMemObj` in `captureHIPArgs`, a miss silently drops residency and dependency tracking on PAL) as a second, lower-severity point.
- Jeff: the upstream `test_event_external` race (still open).

## Porting session 18 (2026-11-19, windows-gfx1151 laptop): test_event_external ordering fix and the full Windows pass

- Ruling (Jeff Daily, 2026-11-19): carry a one-line test fix for the upstream `test_event_external` race. Fork commit `ea013807` "Order the zero fills before the graphs in test_event_external": `wp.synchronize_device()` after the three `wp.zeros` calls, with a comment. Test file only; no native or library change. ruff 0.15.9 clean.
- Ruling (Jeff Daily, 2026-11-19): the capture-time fill workaround (`19b6a5ad`) stays and the alloc-node pointer window is reported; filed as https://github.com/ROCm/rocm-systems/issues/16277.
- Full 82-pattern set on gfx1151 (`agent_space/warp-win/p6/run82c.sh`, one pattern at a time, `-m 2`, fresh cache per pattern, DLL built at `3d74932e`, working tree = `3d74932e` plus the `ea013807` test edit): **82 of 82 patterns OK** (`logs/p18pat/summary.txt`), including test_streams 24 OK (3 skipped), test_graph 35 OK (3 skipped), test_map 45 OK, test_fem* 108 OK (2 skipped, 199 s), test_mesh* 49 OK. No driver reset or hang.
- Next: lock back to linux-gfx950 for the nvcc check and Phase 7 (wheels); the Linux hosts' passes are at `fda2f89a`, and the commits since are Windows-guarded (`de4e07a3`, `6b639456`, `19b6a5ad`), test-only (`9d9a6161` and its reversal `3d74932e`, `ea013807`).

## Release goal for this line (stated by Jeff Daily, 2026-11-19)

The deliverable is a 1.14 release on `AMD-Ecosystem/warp` that mirrors upstream's `v1.14.0` release: a GitHub Release carrying a similar set of wheels, built for ROCm 10.2, for manylinux and Windows. The wheels that get uploaded are validated first (the built artifacts themselves, not only a source build at the same commit). The release is not created until the wheels are ready, so the tag, the release and the wheels go out together.

Facts this sets against (checked 2026-11-19):
- Upstream `v1.14.0` assets: `warp_lang-1.14.0+cu12` and `+cu13` wheels for `manylinux_2_28_x86_64`, `manylinux_2_34_aarch64` and `win_amd64` (111 to 140 MB each), plus `+cpu` for `macosx_11_0_arm64` (25 MB). The ROCm counterparts are `manylinux_2_28_x86_64` and `win_amd64`; ROCm has no aarch64 or macOS target.
- `upstream.py --finish-release --apply` tags the approved commit, records the delivery, closes the review PR and freezes the branch. It does not create a GitHub Release or upload assets, so the release creation is a separate GitHub write that Jeff approves and runs.
- `AMD-Ecosystem/warp` has no GitHub Releases today.
- Consequences for the remaining order of work: the wheels are built from the exact final approved commit; each gate platform validates by installing the wheel file that will be uploaded (recorded by sha256); a head move after wheel validation means rebuilding and revalidating the wheels; a published wheel is never replaced under the same version (the kernel cache key does not include native headers), a respin is `+rocm.1`.

## Ruling on version naming (2026-11-19, Jeff Daily)

Switch from `+rocm.N` to `+rocm10.2.N`: the wheel version is `1.14.0+rocm10.2.0`, the tag `v1.14.0+rocm10.2.0` (recorded with `set-delivery --tag-format "v{version}+rocm10.2.{rev}"`). `10.2` is the ROCm minor the wheel is built for and pinned to; the last number is this fork's build counter for that Warp version and ROCm, not a ROCm patch version (AMD's PyTorch wheels use the same shape with the other meaning; the release notes must say so). A fix in the release branch with the same ROCm bumps the counter (`1.14.0+rocm10.2.1`); an upstream patch release resets it (`1.14.1+rocm10.2.0`); a build for a later ROCm changes the middle (`1.14.0+rocm10.3.0`). PEP 440 post-releases (`1.14.0.post1`) are not used: they change the public version, which is upstream's to assign. Every place in the fork that says `1.14.0+rocm.0` (`VERSION.md`, `warp/config.py`, `version.h`, docs, wheel names in the plan) changes with this; the branch name `release/1.14.0.amd0` stays. Jeff also approved staging the built wheels as assets of a draft GitHub Release on the fork (the create command and text are shown to him first).

## Ruling on performance work before the release (2026-11-19, Jeff Daily)

After the comparison against AMD's `release/1.13.0.amd0` on gfx942 (plan `warp-amd113-vs-moat114-perf-comparison-mi300.md`): one release, right the first time, no planned follow-up release for performance. Priorities in his words: minimise the diff, release soon, performance, but he is patient and wants the best quality; "these perf wins seem worth it". So the measurable gaps are closed in this line before release, each with the smallest change that gets the win: GPU lbvh build (1.8x to 2.25x), `mesh_query_ray` (1.06x to 1.44x), `tile_cholesky` at n=16 (1.07x to 1.22x), `mesh.refit()` on small meshes (about 50 us fixed), `example_dem` (1.07x), and the fp32 64x64 block_dim 64 `tile_matmul` cell (1.20x, contradicting the earlier MFMA benchmark). The last three need attribution first; the same fleet job is doing that as a follow-up turn. AMD's code is taken with credit to its authors (`6f0ae752` and `664cc4c5` rtmadduri; `431e0533` Rishi Madduri with Zhihui Du; the ray traversal from `527ba15c` deepsek).

## Porting session 19 (2026-11-19, linux-gfx950 MI355X, porter): CUDA check at `ea013807`, Phase 7 (Linux wheel, C++ examples), size and build-time numbers

Lock moved back to linux-gfx950 by Jeff for this round; `set-state warp-1.14 linux-gfx950 porting` re-stamped it to this job (WARNING printed, the earlier holder had stopped). Fork `release/1.14.0.amd0` advanced `ea013807..9882f005` (pushed, `advance-head` recorded); stage left at `porting`. Environment: fleet job venv (Python 3.11) with TheRock `rocm[devel]==10.2.0` installed from `https://stable.repo.amd.com/rocm/whl-next/` (rocm, rocm-sdk-core, rocm-sdk-devel 10.2.0; HIP 7.17, AMD clang 24.0.0git bdc35841); root `$(python -m rocm_sdk path --root)` = `.../site-packages/_rocm_sdk_devel`. No system ROCm (`/opt/rocm` absent), `LD_LIBRARY_PATH` empty. nvcc 12.8 at `/usr/local/cuda`, no NVIDIA GPU. `HIP_VISIBLE_DEVICES=1` (dispatcher pin, kept) and `ulimit -c 0` for every GPU run. ruff 0.15.9 and clang-format 22.1.2 (the pre-commit pins) in the job venv. Scratch under `agent_space/` (cuda_wt, wt_single, wt_comp worktrees; logs; sizes/).

Setup notes for this host: the fresh clone got Git LFS hooks from the global LFS config (`.git/hooks/pre-push` etc.), which made `protect-fork` refuse ("NON-moat pre-push hook"); removed the four LFS hooks, re-ran `protect-fork` (installed), then `git lfs install --local --skip-repo` and `git lfs pull` as in session 1. `jargon.py --port` needs a local `release-1.14` branch in the clone (`git branch release-1.14 origin/release-1.14`).

### Part 1: CUDA no-regression at `ea013807` (nvcc 12.8)

- Build: worktree `agent_space/cuda_wt` at `ea013807`, `python build_lib.py --cuda-path /usr/local/cuda --no-use-libmathdx -j 64` (wrapped in `utils/timeit.sh`) -> "Warp build succeeded", warp.so 199,765,024 bytes. `import warp; warp.init()` -> "CUDA Toolkit 12.8, CUDA driver not available (NVRTC compilation available)".
- Textual check, stronger than reading the diff: every `warp/native/*.cu` compiled with nvcc for sm_90 with the build's flags and `--keep` (`agent_space/cuda_pp.sh <repo> <rev> <outdir>`, below), comparing the preprocessed device (`.cpp1.ii`) and host (`.cpp4.ii`) translation units, line markers and build paths removed. `a8e95dcb` (the last nvcc-checked head) vs `ea013807`: 22 of 22 identical except `warp.cu`, whose two differ only in the `__LINE__` numbers inside `check_cuda`/`check_cu` expansions (identical after normalising them). So the Windows-guarded `hip_compat.h`/`warp.cu` code, the bf16 WMMA header, `builtin.h`'s `!defined(_MSC_VER)` guard and `crt.h`'s `__HIPCC_RTC__` change do not reach any CUDA translation unit.
- NVRTC: `agent_space/nvrtc/nvrtc_offline.py` (below; a module with tile_load/tile_matmul/tile_sum/tile_atomic_add, f16/bf16 atomic_add, f64 atomic_min/max, div_approx and an assert, generated by Warp's codegen with no GPU and compiled through the CUDA build's NVRTC) with the native headers of `a8e95dcb` and `ea013807`: release sm_90 PTX and sm_86 CUBIN identical; the debug variants differ only in embedded header paths and `__LINE__` values (crt.h gained 7 lines).
- No break, nothing to fix for CUDA at `ea013807`.

After this round's commits (cuda_wt = `ea013807` plus the working diff, native files identical to `9882f005`): nvcc build "Warp build succeeded"; the 22 preprocessed CUDA translation units identical to `ea013807` (warp.cu: line numbers only); NVRTC release PTX/CUBIN identical with the new `tile.h`; the libmathdx variant also built (`python build_lib.py --cuda-path /usr/local/cuda -j 64`, packman fetched libmathdx 0.3.1 from developer.nvidia.com, `-DWP_ENABLE_MATHDX=1`, "Warp build succeeded", warp.so 286,811,784 bytes). A CUDA build imports no ROCm module and maps no HIP library even with `rocm_sdk` installed (checked with `sys.modules` and `/proc/self/maps`).

### Part 2: Phase 7

#### Fork commits (`ea013807..9882f005`)

1. `63fa49bf` [ROCm] Launch at most 2^32 - 1 threads per grid dimension on HIP (`warp.cu` +5). Found by the full default suite: `test_large_launch_large_kernel` (2^33 threads) and `test_large_launch_very_large_kernel` (~2^39) returned 0 with "Warp CUDA error 1: invalid argument (in function wp_cuda_launch_kernel)". HIP refuses a grid of more than 2^32 - 1 threads in a dimension (`hipModuleLaunchKernel` -> `hipErrorInvalidValue`); CUDA only limits blocks to 2^31 - 1. Warp's kernels are grid-stride loops (`codegen.py` 4604/4623, the mechanism `max_blocks` relies on), so the HIP branch caps `max_blocks` at `0xFFFFFFFF / block_dim`. After: test_large 10 OK (157 s), test_launch 32 OK. HIP preprocessed check (`agent_space/hip_pp.sh`, below): of the 22 HIP translation units (11 .cu, gfx950 device and host passes) only warp.cu's two change, by exactly this line.
2. `91a7f149` [ROCm] Fix three tests that assumed a CUDA build on AMD GPUs. Found by the full suite, none in the 82-pattern set used so far: `test_fixedarray.test_capture_if` compared `runtime.toolkit_version < (12, 4)` (None on HIP -> TypeError; now `wp.is_conditional_graph_supported()`, the same two version checks on CUDA, False on HIP); `test_logger.test_cuda_build_honors_deprecated_verbose_flag_at_call_time` stubs the runtime without `is_hip` (stub now has `is_hip=False`, so it keeps testing the CUDA path it mocks); `aot/test_module_aot.test_aot_cache_skip` collected binaries by suffix `.o/.cubin/.ptx` (now also `.hsaco`). After: 20 OK, 34 OK, 23 OK (3 skipped).
3. `6b87ddfd` [ROCm] Package ROCm wheels with their HIP runtime dependency (`setup.py` +40, `build_dll.py` +14/-8, `context.py` +37/-8, `installation.rst` +8):
   - `build_dll.py`: the `.rocm` marker that session 13 wrote on Windows only is now written at the end of `build_dll_for_arch` on every OS (`warp.so.rocm`, `warp.dll.rocm`), removed by any non-ROCm build; content: line 1 the ROCm root (as before), line 2 the ROCm version read from `<root>/.info/version` (`10.2.0`).
   - `context.py`: `load_windows_hip_runtime` became `load_hip_runtime` plus `read_rocm_marker`, called whenever the marker exists. `rocm_sdk` importable -> `rocm_sdk.initialize_process(preload_shortnames=["amd_comgr", "amdhip64", "hiprtc"], check_version="<major>.<minor>.*")` (the plan's three libraries; Windows gains amd_comgr, which has a Windows DLL pattern in `rocm_sdk/_dist_info.py`, and absent libraries are skipped). No `rocm_sdk`: Windows loads by full path from the recorded root as in session 13; Linux returns and lets the loader use warp.so's RUNPATH / `LD_LIBRARY_PATH` (a system-ROCm source build). `load_dll` gains a branch for a ROCm build that fails to load: "Warp was built for AMD GPUs and needs the HIP runtime. Install the ROCm Python packages with `pip install "rocm==10.2.*" --extra-index-url https://stable.repo.amd.com/rocm/whl-next/` or add the ROCm this build used to the library search path." An old one-line Windows marker still reads (version None, no check).
   - `setup.py`: `get_warp_libraries` also ships `<library>.rocm` when present; with a marker the wheel gets `Requires-Dist: rocm>=X.Y,<X.(Y+1)` from the marker's version (10.2.0 -> `rocm<10.3,>=10.2`; a marker without a version is a build error) and `license_files` without `cuda-LICENSE.txt`, `libmathdx-LICENSE.txt` and `nvrtc-LICENSE.txt` (the plan named the first two; the NVRTC licence covers statically linked NVRTC, which a ROCm wheel does not contain, and the plan's own wheel check forbids any `nvrtc*` file). `pyproject.toml` is unchanged: making `dependencies` dynamic would stale `uv.lock`, so `BinaryDistribution.parse_config_files` appends the requirement after the static ones, to both `install_requires` and `metadata.install_requires` (setuptools 79 and 84 both fill the metadata inside `parse_config_files`; the first attempt set only `install_requires` and the wheel had no rocm requirement), and only when there is one: assigning it unconditionally added `Dynamic: requires-dist` to the CUDA wheel's METADATA.
   - Docs: installation guide, after the GitHub Releases table: the ROCm wheels (Linux x86-64, Windows x86-64, ROCm 10.2) are on the AMD-Ecosystem/warp releases page, with `pip install https://github.com/AMD-Ecosystem/warp/releases/download/v1.14.0+rocm.0/warp_lang-1.14.0+rocm.0-py3-none-manylinux_2_28_x86_64.whl --extra-index-url https://stable.repo.amd.com/rocm/whl-next/`. The URL assumes the release tag `v1.14.0+rocm.0` and these asset names; it must match the GitHub Release when it is created.
   - Version: already `1.14.0+rocm.0` (session 1); the wheel and `warp.__version__` both report it.
4. `9882f005` [ROCm] Build and run the C++ examples with HIP (`aot.h` +46/-2, `tile.h` +2/-2, examples: 4 CMakeLists, 2 compile scripts, 02's main.cu, README, test_examples.sh). The examples had no HIP support at all (no diff from upstream before this round), so this is the plan's C++-examples item:
   - Each CMake project: `option(WARP_USE_HIP ... OFF)`, `project(... ${WARP_GPU_LANGUAGE} CXX)`, `main.cu` set to `LANGUAGE HIP`, `find_package(CUDAToolkit)`/CUDA link targets only for CUDA (CMake links the HIP runtime implicitly), `hip_std_17` for 01, nvcc-only flags only for CUDA; 00 globs `generated/wp___main__.gfx*.hsaco` on HIP. CMake detects the HIP architectures from the GPUs present (it listed gfx950 eight times, one per GPU on the host, harmless). `test_examples.sh --hip` passes `-DWARP_USE_HIP=true` and skips the nvcc check.
   - `aot.h`: HIP compilation (`__HIP__`) counts as CUDA; includes `<hip/hip_runtime.h>` and maps the 37 CUDA runtime/driver names the examples use (CUcontext -> hipCtx_t, cuLaunchKernel -> hipModuleLaunchKernel, cuGetErrorString -> hipDrvGetErrorString, ...), so main.cu keeps CUDA spelling. `aot.h` is used only by these examples (`grep` finds no other includer). hipDevicePrimaryCtxRetain/hipCtxSetCurrent print deprecation warnings.
   - `tile.h`: `defined(__HIP__)` joins the "float4 already available" condition; the examples' compile hit `fatal error: 'cuda_runtime.h' file not found` at tile.h:24. In the library's HIP build the include resolved to the shim, already force-included, so the 22 HIP preprocessed units are unchanged (checked above); CUDA/NVRTC unaffected (checked above).
   - 02: on HIP passes `wp_cuda_device_get_primary_context(0)` to `wp_apic_load_graph`. With the example's hipCtx_t, Warp (whose HIP contexts are its own `WpHipContext` handles) printed 22 x "Warp CUDA error 101: invalid device ordinal (in function ContextGuard, cuda_util.h:279)" although the smoke test passed.

#### Builds

- Release build (25 targets from `share/therock/dist_info.json`: gfx1010-1012, gfx1030-1036, gfx1100-1103, gfx1200, gfx1201, gfx1150-1153, gfx908, gfx90a, gfx942, gfx950, gfx1250): `python build_lib.py --rocm-path "$(python -m rocm_sdk path --root)" -j 64`. Without `--rocm-path` the build stops with the both-toolkits error on this host (nvcc present), as ruled (Q6). 320 s wall (the .cu phase 318 s; 11 clang processes, one per .cu, each compiling the 25 targets one after another); rebuilt after commit 1: 330 s, warp.so 330,582,856 bytes, of which 324,391,592 device code (12.6 to 13.5 MB per target, gfx1250 largest).
- Wheel: `python -m build --wheel -C--build-option=-Plinux-x86_64` -> `warp_lang-1.14.0+rocm.0-py3-none-manylinux_2_28_x86_64.whl`, 58,613,145 bytes (from the committed tree at `9882f005`: same file list, METADATA and warp.so md5 as the wheel tested below; sha256 differs by timestamps only).

#### Wheel checks (plan "Build commands")

- `unzip -l`: `warp/bin/warp.so` 330,582,856, `warp/bin/warp-clang.so` 67,136,896, `warp/bin/warp.so.rocm` 142; no libmathdx, nvJitLink, nvrtc or cudart file (`grep -iE "libmathdx|nvjitlink|nvrtc|cudart"` empty); licences: LICENSE.md plus 12 files, no cuda/nvrtc/libmathdx. METADATA: `Version: 1.14.0+rocm.0`, `Requires-Dist: numpy`, `Requires-Dist: rocm<10.3,>=10.2`; WHEEL tag `py3-none-manylinux_2_28_x86_64`.
- Clean venv in the job directory (`/usr/bin/python3.11 -m venv <job>/wheeltest2`, no base-env links): `pip install <wheel> pillow --extra-index-url https://stable.repo.amd.com/rocm/whl-next/` -> rocm 10.2.0, rocm-sdk-core 10.2.0, numpy 2.4.6, pillow 13.0.0 (pillow is for the two examples tests that need it, below; the plain wheel install pulled only rocm and rocm-sdk-core for ROCm, 12 s). From outside the source tree: `python -c "import warp; print(warp.__version__)"` -> `1.14.0+rocm.0`; `agent_space/smoke/smoke.py` (one hipRTC kernel and `wp.utils.array_sum`, prints the HIP libraries in `/proc/self/maps`) -> sum 1000.0, `libamd_comgr.so.3`, `libamdhip64.so.7`, `libhiprtc.so.7` all from the venv's `_rocm_sdk_core/lib`.
- `python -m warp.tests -s autodetect` from the installed wheel, cwd outside the tree, GPU otherwise idle: **Ran 7556 tests in 563.6 s, OK (skipped=116)**. Skips: usd-core 27, "No suitable devices" 24, conditional graph nodes 16 (Q4), MathDx 8+1, two CUDA devices 7, ml_dtypes 5, mempool access 5+1+1, peer access 3+1+1, CPU memory access 2, NVRTC two archs 2, multi-GPU 2, rmm 1, memory-usage flakiness 1, "Requires CUDA" 1, NVRTC 1, stream priority 1. The two known fem isolation failures did not appear: they need the fem file to be first in its worker, which the full suite's scheduling did not produce.
- The first full run (wheel before commits 1-2, no pillow; 7556 tests, 462 s) had 7 failures and 2 errors: test_large x2 and the three tests above (fixed), the two Pillow-less examples tests (`example_particle_repulsion`, `example_tile_mlp`: "No module named 'PIL'", the examples extra), and two `test_async` graph-copy cases (`test_copy_s2c_d2d_SrcPoolOn_DstPoolOff_NoStream_NoGrad_Graph_AccessBoth`, `test_copy_s2f_d2d_SrcPoolOff_DstPoolOff_NoStream_NoGrad_Graph_AccessSrcDst`: the first 125,184 of 1,000,000 elements still 0). That run overlapped my launch-time measurements on the same GPU; `test_async.py` alone right after: 2373 OK, and the second full run passed it. Not reproduced; recorded in case it returns.
- Source build without `rocm_sdk` (`agent_space/smoke/no_rocm_sdk.py`: `importlib.util.find_spec` patched to return None for rocm_sdk, then smoke.py): loads, libraries from `_rocm_sdk_devel/lib` through warp.so's RUNPATH (`$ORIGIN:<root>/lib`), `rocm_sdk imported: False`. With the RUNPATH also removed (copy of the installed package, `patchelf --set-rpath '$ORIGIN'`), the import fails with the message quoted above.

#### C++ examples (CMake/CTest), gfx950

- `cd warp/examples/cpp && bash test_examples.sh --hip` with `ROCM_PATH=<root>`, `HIPCXX=<root>/lib/llvm/bin/clang++`, `PYTHONPATH=<src>`, and a PATH holding only cmake/ctest and a `python3` wrapper for the job venv (with `uv` on PATH the script would run `uv run`, which creates a project venv): kernels generated, 00 and 01 built, then the build stopped in 02's GLAD generator ("No module named 'jinja2'"): GLAD's `find_package(Python)` took `/usr/bin/python3.12` despite `VIRTUAL_ENV` and `Python_ROOT_DIR`. An upstream build dependency on the host, not the port.
- Same steps by hand: `cmake -DWARP_NATIVE_DIR=<src>/warp/native -DWARP_USE_HIP=ON -DPython_EXECUTABLE=<venv python> ..`, `cmake --build . -j 16`, `ctest --output-on-failure --verbose` -> **100% tests passed, 0 tests failed out of 4** (cubin_launch_test 0.24 s "All results correct", source_include_test 0.23 s, apic_visualization_smoke 0.25 s "smoke OK (10 graph launches)", apic_visualization_cpu_smoke 0.16 s), no Warp error output after the 02 fix. jinja2 installed into the job venv for this.
- CUDA configuration (default, nvcc 12.8): kernels for 00/01 compiled offline for sm_86 through NVRTC (`agent_space/nvrtc/gen_example.py`, below), then `cmake -DWARP_NATIVE_DIR=... -DPython_EXECUTABLE=... ..` and `cmake --build .`: all four targets built (00, 01, 02 with nvcc; 03 host only). Not run (no NVIDIA GPU).
- The interactive (GUI) mode of 02/03 was not run (no display), as upstream's CTest does not.

#### CUDA and CPU wheels unchanged

Same CUDA binaries (cuda_wt, nvcc, no libmathdx) packaged once with `ea013807`'s setup.py and once with the new one: identical file list (551 entries, names and sizes; no `.rocm` file) and byte-identical METADATA (no rocm requirement, all licences). A CPU-only build writes no marker either (`build_dll_for_arch` removes it whenever no ROCm path is in use), so it takes the same path.

#### What the Windows validator must check (not buildable here)

1. After `python build_lib.py --hip-arch gfx1151` (or the default list): `warp/bin/warp.dll.rocm` has two lines, the ROCm root and `10.2.0`. The version comes from `<root>/.info/version`, present in the Linux 10.2 SDK wheel, not checked in the Windows one. If line 2 is empty, the wheel build stops with "warp/bin/warp.dll.rocm does not record the ROCm version": that needs a porter fix (for example reading the version from `rocm_sdk`), not a workaround.
2. `python -m build --wheel -C--build-option=-Pwindows-x86_64`: `unzip -l` shows `warp/bin/warp.dll`, `warp/bin/warp-clang.dll` **and `warp/bin/warp.dll.rocm`**; METADATA `Requires-Dist: rocm<10.3,>=10.2`; no cuda/nvrtc/libmathdx licence.
3. Install that wheel into a clean venv with `--extra-index-url https://stable.repo.amd.com/rocm/whl-next/`; session 13's `loadcheck.py`: `amdhip64_7.dll`, `hiprtc0717.dll` and now `amd_comgr*.dll` from the venv's `_rocm_sdk_core\bin`, nothing from System32.
4. The default suite from the installed wheel, outside the tree.
5. The C++ examples with `-DWARP_USE_HIP=ON` on Windows (HIP language in CMake with the ROCm Clang, `aot.h`'s mappings, 02's post-build copy of warp.dll): never built on Windows.
6. The default Windows target list comes from the Windows SDK's `dist_info.json` (22 targets in session 13).

### Part 3: numbers for Jeff (target list and compression; nothing decided, nothing committed)

Linux, MI355X host, ROCm 10.2 wheels, `-j 64`. "Build" is `build_lib.py` wall time (warp-clang's LLVM already cached; the .cu phase is nearly all of it). Sizes in bytes. "Installed" is `du -sb` of `warp/` after `pip install --no-deps --target` of that wheel (includes warp-clang.so 67.1 MB and about 28 MB of Python, tests, examples and .pyc, the same in every row). The compressed rows and `--offload-jobs` were built in a scratch worktree (`agent_space/wt_comp`, an extra-options hook in build_dll.py that is not committed).

| Build | Build wall | warp.so | warp.so zipped (in wheel) | Wheel | Installed warp/ |
|---|---|---|---|---|---|
| 25 targets, as committed | 320 s (330 s rebuild) | 330,582,856 | 27,956,402 | 58,613,145 | 425,949,591 |
| gfx950 only (`--hip-arch gfx950`) | 22 s | 18,533,192 | 2,428,056 | 33,084,724 | 113,899,906 |
| 25 targets, `--offload-compress` (zstd level 3, clang's default) | 323 s | 10,574,664 | 6,266,832 | 36,923,545 | 105,941,644 |
| 25 targets, `--offload-compress --offload-compression-level=9` | 320 s | 10,177,352 | 5,870,843 | 36,527,557 | 105,544,332 |
| 25 targets, `--offload-compress --offload-compression-level=19` | 346 s | 9,309,000 | 5,034,905 | 35,691,618 | 104,675,980 |
| 25 targets, `--offload-jobs=16`, uncompressed | 39 s | 330,582,856 | (same) | 58,614,842 | (same) |

For reference: the CUDA wheel of the same tree (nvcc 12.8, no libmathdx) is 94,614,295 bytes with a 199.8 MB warp.so; upstream's v1.14.0 wheels are 111 to 140 MB. Windows (session 13): 22 targets 290.5 MB warp.dll (285.2 MB device code), 24.4 MB zipped; one target 17.9 MB, 2.1 MB zipped.

Compression facts:
- ROCm 10.2's clang accepts `--offload-compress` and `--offload-compression-level=<n>` for HIP (`clang++ --help-hidden`); the bundler uses zstd (`OFFLOAD_BUNDLER_VERBOSE=1`: "Compressed bundle format version: 3", "Compression method used: zstd", "Compression level: 3", per-bundle ratios 2.1% to 15.5%). Compression itself takes milliseconds per bundle; build time does not change measurably (the 320-346 s spread is noise: the 344-346 s builds ran beside the second test suite).
- The library holds 11 compressed bundles (`CCOB` magic 11 times, `__CLANG_OFFLOAD_BUNDLE__` 0; the uncompressed build the reverse). The ROCm 10.2 HIP runtime loads them: smoke.py (hipRTC kernel plus `array_sum` from reduce.cu) passes on the level-3 build. The full suite was not run on a compressed build.
- Why the gain is so large: the 25 per-target code objects of one translation unit are compressed together, and they are largely alike (zstd reaches 2-16% of the original, against deflate's 8.5% on the whole uncompressed warp.so). The compressed library hardly zips further, so the wheel gain (58.6 -> 36.9 MB) is smaller than the installed gain (426 -> 106 MB).
- Load-time cost (`agent_space/smoke/launch_time.py` and `launch_time2.sh`, below; 5 fresh processes per row, medians, ms): the HIP runtime hands code objects to comgr, which caches unbundled code objects (default `~/.cache/comgr`; 5.4 GB on this shared host). With an empty comgr cache per run ("cold") and with a warmed one ("warm"):

| Library | import | wp.init | first fill (warp.cu) | first array_sum (reduce.cu) | first radix sort (sort.cu) |
|---|---|---|---|---|---|
| 25 targets uncompressed, cold | 140.3 | 276.2 | 11.7 | 10.3 | 13.0 |
| 25 targets compressed (level 3), cold | 140.9 | 274.8 | 13.9 | 47.5 | 61.4 |
| gfx950 only, cold | 140.0 | 277.2 | 11.8 | 9.6 | 12.2 |
| 25 targets uncompressed, warm | 139.8 | 137.8 | 11.9 | 11.9 | 13.7 |
| 25 targets compressed (level 3), warm | 139.3 | 138.9 | 12.3 | 13.4 | 16.1 |
| gfx950 only, warm | 139.4 | 137.3 | 11.7 | 11.3 | 13.6 |

  So decompression costs about 35-50 ms the first time a process uses kernels from each compressed library translation unit while the comgr cache is cold (once per machine and cache, then it is cached: about 1.5-2.5 ms warm); 11 translation units, so at most a few hundred ms in total, spread over first use. The cold cache costs every build about 138 ms in `wp.init` regardless of compression. Measured with the page cache warm (dropping it needs root); levels 9 and 19 were not timed.
- Not measured on Windows: whether the Windows HIP runtime loads compressed bundles, and the Windows sizes and timings.

Build-time fact found on the way: clang's `--offload-jobs=<n>` compiles the targets of one translation unit in parallel. Full 25-target build 39 s instead of 320 s (`--offload-jobs=16`). The code is the same: per bundle entry, sizes match and `llvm-objdump -d` of the gfx950 code object of warp.cu is identical; the bytes differ only in clang's `__hip_cuid_<hash>` symbol, which is derived from the compile command.

### Questions for Jeff (no decision taken)

1. **Default target list and compression.** Options: (a) keep the 25 targets uncompressed, as committed: Linux wheel 58.6 MB, installed 426 MB (warp.so 330.6 MB); no load cost. (b) 25 targets with `--offload-compress` at clang's default level 3: wheel 36.9 MB, installed 106 MB (warp.so 10.6 MB); cost 35-50 ms per library translation unit at first use with a cold comgr cache, about 2 ms warm; build time unchanged; Windows loading of compressed bundles must be verified first. (b') the same at level 19: warp.so 9.3 MB, wheel 35.7 MB, about 6% smaller than level 3; build time within noise here. (c) a shorter list: about 13 MB per target uncompressed, 1.1 MB per target in the wheel (27.96 MB for 25 targets). (b) or (b') is a one-line change in build_dll.py's HIP compile options.
2. **`--offload-jobs` for HIP builds**: 320 s -> 39 s for the 25-target build, identical code. Options: add it to build_dll.py's HIP compile options (value from `-j`, or a fixed number), or leave builds serial per translation unit.
3. **`runtime.toolkit_version` on HIP is None** (and `driver_version`), as in a CPU-only build. Code that assumes a CUDA device implies a version and compares it with a tuple raises TypeError on AMD GPUs; `test_fixedarray` did (fixed in the test with the capability API), and downstream code doing `toolkit_version < (12, 4)` would too. Options: (a) keep None and leave such code to fail; (b) report (0, 0) on HIP so version comparisons read as "older than any CUDA" and take their fallback paths; (c) keep None and document it in the release notes.
4. **ROCm wheel classifiers**: METADATA keeps upstream's static `Environment :: GPU :: NVIDIA CUDA` and `:: 12` classifiers. Options: leave (pyproject.toml untouched, smallest diff), or have setup.py replace them for a ROCm wheel.

## Lesson for cuda-to-rocm: HIP limits a grid to 2^32 - 1 threads per dimension, not 2^31 - 1 blocks
Reference file: `references/fault-classes.md` (launch configuration). CUDA limits gridDim.x to 2^31 - 1 blocks, so code that caps a one-dimensional launch at 2^31 - 1 blocks (common with grid-stride loops) can launch up to 2^31 - 1 times blockDim threads. HIP describes the grid to the hardware in threads, and `hipModuleLaunchKernel`/`hipLaunchKernel` return `hipErrorInvalidValue` when gridDim.x * blockDim.x exceeds 2^32 - 1. The symptom is a launch error and untouched outputs only for very large problems (more than about 4.3 billion elements), which tests rarely reach. For grid-stride kernels cap the block count at `0xFFFFFFFF / blockDim.x` on HIP; kernels that need one thread per element must split the launch. (warp-1.14, `wp_cuda_launch_kernel`)

## Lesson for cuda-to-rocm: compressed offload bundles for multi-target HIP libraries
Reference file: `references/libraries.md` (the pip-installed ROCm SDK / shipping wheels) or a build reference. A HIP library built for every target a ROCm release supports embeds one code object per target per translation unit (here 25 targets, 324 MB of device code). `--offload-compress` stores each translation unit's bundle zstd-compressed (`--offload-compression-level=<n>`); because the per-target code objects are similar, bundles shrink to 2-16% (a 330 MB library to 10.6 MB), at no measurable build cost, and the HIP runtime of ROCm 10.2 loads them. The cost is decompression the first time a process uses kernels from a bundle, about 35-50 ms per bundle here, and only while comgr's code-object cache (`~/.cache/comgr`, `AMD_COMGR_CACHE_DIR`) is cold: measure first-launch time with a fresh `AMD_COMGR_CACHE_DIR` per run, or the cache hides it. Separately, `--offload-jobs=<n>` compiles the targets of one translation unit in parallel (here 320 s -> 39 s); its output differs only in the `__hip_cuid_*` symbol. (warp-1.14)

## Lesson for cuda-to-rocm: a ROCm wheel's runtime dependency with setuptools and a static pyproject.toml
Reference file: `references/libraries.md` (the pip-installed ROCm SDK). To add `Requires-Dist: rocm>=X.Y,<X.(Y+1)` to a ROCm build's wheel without making `dependencies` dynamic (which changes how lock files see the project), append the requirement after setuptools has applied pyproject.toml, in a `Distribution.parse_config_files` override: set both `self.install_requires` and `self.metadata.install_requires` (setuptools copies the former into the latter inside `parse_config_files`, and Requires-Dist is written from the metadata), and only when there is something to add, because any reassignment marks the field `Dynamic: requires-dist` in the CUDA wheel too. Pin the version from what the build used (here `<rocm>/.info/version`, written to a marker next to the library) rather than hard-coding it. (warp-1.14)

### Scripts from this session

`agent_space/cuda_pp.sh`:
```
#!/bin/bash
# Compile every warp/native/*.cu of git revision $2 of repository $1 with nvcc for sm_90, with the flags
# build_dll.py passes, keeping the preprocessed device (.cpp1.ii) and host (.cpp4.ii) translation units;
# writes them to directory $3 with line markers and blank lines dropped.
set -e
repo=$1; rev=$2; out=$3
mkdir -p "$out/tree" "$out/keep"
git -C "$repo" archive "$rev" warp/native | tar -x -C "$out/tree"
for f in "$out"/tree/warp/native/*.cu; do
  b=$(basename "$f" .cu)
  mkdir -p "$out/keep/$b"
  /usr/local/cuda/bin/nvcc --std=c++17 -O3 --compiler-options -fPIC,-fvisibility=hidden,-fvisibility-inlines-hidden,-D_GLIBCXX_USE_CXX11_ABI=0 \
    -gencode=arch=compute_90,code=sm_90 -t0 --extended-lambda -diag-suppress=221 -DNDEBUG -DWP_ENABLE_CUDA=1 \
    -I"$out/tree/warp/native" -DWP_ENABLE_MATHDX=0 --keep --keep-dir "$out/keep/$b" -o "$out/keep/$b/$b.o" -c "$f" > /dev/null 2>&1 &
done
wait
for d in "$out"/keep/*/; do
  b=$(basename "$d")
  for k in cpp1 cpp4; do grep -v '^#' "$d"/$b.$k.ii | grep -v '^[[:space:]]*$' > "$out/$b.$k.i"; done
done
```
Compared with `sed "s#<outdir>/##g"` on each file, then `sed -E 's#("tree/warp/native/[a-z_]+\.(cu|h)", )[0-9]+#\1L#g'` to drop `__LINE__` values. A working tree is archived through `git stash create` (a commit object, nothing added to the stash list).

`agent_space/hip_pp.sh` (HIP counterpart; reads the clang commands from a build log):
```
#!/bin/bash
# Preprocess the 11 Warp .cu files with the HIP flags from a build log ($1), the native tree rewritten from the
# build's ($2) to another ($3), for gfx950 device and host passes, into $4 (line markers and blank lines dropped).
log=$1; from=$2; to=$3; out=$4
mkdir -p "$out"
grep -E '^"[^"]*clang\+\+" -x hip' "$log" | while read -r cmd; do
  src=$(echo "$cmd" | grep -oE '\-c "[^"]+\.cu"' | sed 's/-c "//; s/"$//')
  b=$(basename "$src" .cu)
  c=$(echo "$cmd" | sed -E 's/--offload-arch=[a-z0-9]+//g; s/ -o "[^"]+"//; s/ -c "[^"]+"//')
  c=${c//$from/$to}
  for pass in host device; do
    eval "$c --offload-arch=gfx950 --cuda-$pass-only -E \"${src//$from/$to}\"" 2>/dev/null | grep -v '^#' | grep -v '^[[:space:]]*$' > "$out/$b.$pass.i"
  done
done
```
(`$2` must be the literal path in the log; a first run with an unnormalised path compared the tree with itself.)

`agent_space/nvrtc/nvrtc_offline.py`:
```
# Generate CUDA source for a module of tile and atomic kernels with Warp's codegen (no GPU needed) and compile it
# with the CUDA build's NVRTC for sm_90 (PTX) and sm_86 (CUBIN), release and debug, once per native header tree.
# Usage: python nvrtc_offline.py <outdir> <native_dir>...
import hashlib, os, sys
import warp as wp
import warp._src.context as ctx
import warp._src.build as build

wp.init()
TILE = 16

@wp.kernel
def k_tile(a: wp.array2d(dtype=wp.float32), b: wp.array2d(dtype=wp.float32), c: wp.array2d(dtype=wp.float32)):
    i, j = wp.tid()
    ta = wp.tile_load(a, shape=(TILE, TILE))
    tb = wp.tile_load(b, shape=(TILE, TILE))
    tc = wp.tile_zeros(shape=(TILE, TILE), dtype=wp.float32)
    wp.tile_matmul(ta, tb, tc)
    s = wp.tile_sum(tc)
    wp.tile_store(c, tc)
    wp.tile_atomic_add(c, tc * s[0])

@wp.kernel
def k_atomic(h: wp.array(dtype=wp.float16), bf: wp.array(dtype=wp.bfloat16), d: wp.array(dtype=wp.float64), f: wp.array(dtype=wp.float32)):
    i = wp.tid()
    wp.atomic_add(h, 0, wp.float16(1.0))
    wp.atomic_add(bf, 0, wp.bfloat16(1.0))
    wp.atomic_min(d, 0, wp.float64(i))
    wp.atomic_max(d, 1, wp.float64(i))
    f[i] = wp.div_approx(f[i], 3.0) + wp.float32(h[0]) + wp.float32(bf[0])
    assert f[i] >= 0.0

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
mod = k_tile.module
opts = dict(mod.options); opts["block_dim"] = 64; opts["output_arch"] = 90
src = ctx.ModuleBuilder(mod, opts).codegen("cuda")
src_path = os.path.join(out, "module.cu")
with open(src_path, "w") as f:
    f.write(src)
for native in sys.argv[2:]:
    build.warp_home = os.path.dirname(native.rstrip("/"))
    tag = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(native.rstrip("/")))))
    for arch, ext in ((90, "ptx"), (86, "cubin")):
        for cfg in ("release", "debug"):
            p = os.path.join(out, f"{tag}.sm{arch}.{cfg}.{ext}")
            build.build_cuda(src_path, arch, p, pch_dir=None, config=cfg, use_precompiled_headers=False)
            print(tag, f"sm{arch}", cfg, ext, hashlib.md5(open(p, "rb").read()).hexdigest()[:12], os.path.getsize(p))
```

`agent_space/nvrtc/gen_example.py`:
```
# Define the kernels of a C++ example's compile_kernel.py in __main__ and compile them ahead of time for a CUDA arch
# through NVRTC (no GPU needed), writing the example's generated/ directory.
# Usage: python gen_example.py <compile_kernel.py> <arch> <use_ptx 0|1>
import os
import sys

import warp as wp

path, arch, use_ptx = sys.argv[1], int(sys.argv[2]), sys.argv[3] == "1"
src = open(path).read().split('if __name__ == "__main__":')[0]
exec(compile(src, path, "exec"))
out = os.path.join(os.path.dirname(path), "generated")
os.makedirs(out, exist_ok=True)
wp.compile_aot_module("__main__", arch=arch, module_dir=out, use_ptx=use_ptx, strip_hash=True)
print(sorted(os.listdir(out)))
```

`agent_space/smoke/smoke.py`:
```
# One hipRTC-compiled kernel and one precompiled-library kernel (array sum through the AOT library) on cuda:0;
# prints the Warp version and which HIP runtime files the process mapped.
import warp as wp

wp.init()

@wp.kernel
def add_one(a: wp.array(dtype=float)):
    i = wp.tid()
    a[i] = a[i] + 1.0

a = wp.zeros(1000, dtype=float, device="cuda:0")
wp.launch(add_one, dim=a.shape, inputs=[a], device="cuda:0")
s = wp.zeros(1, dtype=float, device="cuda:0")
wp.utils.array_sum(a, out=s)
print("version", wp.__version__, "sum", s.numpy()[0])
libs = sorted({l.split()[-1] for l in open("/proc/self/maps") if any(k in l for k in ("amdhip64", "hiprtc", "amd_comgr"))})
print("\n".join(libs))
```

`agent_space/smoke/launch_time.py`:
```
# Time, in one fresh process: import warp, wp.init(), and the first launch of a precompiled kernel from three
# different translation units of the Warp library (array fill: warp.cu; array sum: reduce.cu; radix sort: sort.cu),
# each followed by a device synchronize. Prints one CSV line of milliseconds.
import time

t0 = time.perf_counter()
import warp as wp  # noqa: E402

t1 = time.perf_counter()
wp.config.log_level = wp.LOG_WARNING
wp.init()
t2 = time.perf_counter()
d = "cuda:0"
a = wp.empty(1 << 20, dtype=float, device=d)
a.fill_(1.0)
wp.synchronize_device(d)
t3 = time.perf_counter()
s = wp.zeros(1, dtype=float, device=d)
wp.utils.array_sum(a, out=s)
wp.synchronize_device(d)
t4 = time.perf_counter()
k = wp.zeros(2 << 20, dtype=int, device=d)
v = wp.zeros(2 << 20, dtype=int, device=d)
wp.utils.radix_sort_pairs(k, v, 1 << 20)
wp.synchronize_device(d)
t5 = time.perf_counter()
a.fill_(2.0)
wp.synchronize_device(d)
t6 = time.perf_counter()
assert s.numpy()[0] == float(1 << 20)
ms = lambda x, y: f"{(y - x) * 1000:.1f}"  # noqa: E731
print(",".join([ms(t0, t1), ms(t1, t2), ms(t2, t3), ms(t3, t4), ms(t4, t5), ms(t5, t6)]))
```

`agent_space/smoke/launch_time2.sh`:
```
#!/bin/bash
# Run launch_time.py N times against the Warp tree $1, each with a fresh empty comgr cache directory (cold) when $3 is
# "cold", or with one shared comgr cache directory warmed by a first run (warm) otherwise. Prints runs and medians.
# Usage: launch_time2.sh <tree> <N> cold|warm
tree=$1; n=$2; mode=$3; here=$(dirname "$(readlink -f "$0")")
work=$(mktemp -d -p "$here")
cd /tmp
if [ "$mode" = warm ]; then
  AMD_COMGR_CACHE_DIR=$work/cc PYTHONPATH="$tree" WARP_CACHE_PATH="$here/cache_lt" "$FLEET_VENV/bin/python" "$here/launch_time.py" > /dev/null
fi
for i in $(seq "$n"); do
  if [ "$mode" = cold ]; then cc=$work/cc$i; else cc=$work/cc; fi
  AMD_COMGR_CACHE_DIR=$cc PYTHONPATH="$tree" WARP_CACHE_PATH="$here/cache_lt" "$FLEET_VENV/bin/python" "$here/launch_time.py"
done | tee "$work/runs.csv" | \
  "$FLEET_VENV/bin/python" -c "import sys,statistics as st; r=[list(map(float,l.split(','))) for l in sys.stdin if l.strip()]; print('median ms import,init,fill(warp.cu),sum(reduce.cu),sort(sort.cu),fill again:', ','.join(f'{st.median(c):.1f}' for c in zip(*r)))"
du -sh "$work" | sed 's/^/comgr cache size (all runs): /'
```

`agent_space/smoke/bundles.py`:
```
# List every clang offload bundle entry in a library: (bundle index, entry id, size, md5), parsing each
# __CLANG_OFFLOAD_BUNDLE__ header (magic, entry count, then offset/size/id-length/id per entry).
# Usage: python bundles.py <library>
import hashlib, struct, sys
data = open(sys.argv[1], "rb").read()
magic = b"__CLANG_OFFLOAD_BUNDLE__"
pos, n = data.find(magic), 0
while pos >= 0:
    count = struct.unpack_from("<Q", data, pos + 24)[0]
    p = pos + 32
    for _ in range(count):
        off, size, idlen = struct.unpack_from("<QQQ", data, p)
        ident = data[p + 24 : p + 24 + idlen].decode()
        p += 24 + idlen
        print(n, ident, size, hashlib.md5(data[pos + off : pos + off + size]).hexdigest())
    n += 1
    pos = data.find(magic, pos + 1)
```

The scratch build hook (wt_comp only, not committed): in `build_dll.py`'s `hip_opts`, after `"-fno-gpu-rdc",` the line `*os.environ.get("WARP_EXTRA_HIP_OPTS", "").split(),`, then `WARP_EXTRA_HIP_OPTS="--offload-compress --offload-compression-level=19" python build_lib.py --rocm-path "$R" -j 64`.

### Next step (head `9882f005`)

- Not ready for `ported`, and not transitioned (as directed): the release notes text is still owed (it must name the hardware-run and compile-only targets, the `tile_fft` gap, conditional graph nodes, the Windows graph-launch wait, and whatever Jeff rules below); Jeff's ruling on the target list and compression (question 1) is pending and changes the release build; the Windows wheel has not been built (its packaging path is written but untested); the C++ examples have run only on gfx950.
- Jeff: questions 1-4 above.
- Also before `ported`: the performance work Jeff ruled the same day ("Ruling on performance work before the release", just above this session: lbvh build, `mesh_query_ray`, `tile_cholesky` n=16, small-mesh `refit`, `example_dem`, the fp32 64x64 block_dim 64 `tile_matmul` cell). It lands on top of `9882f005`; the wheel numbers above are unaffected unless it changes the build options.
- Windows (windows-gfx1151): the six checks in "What the Windows validator must check". The `setup.py`/loader change is platform-correct by construction but unexercised on Windows; a failure at check 1 or 2 is a porter fix.
- wave32 Linux (gfx1100 or gfx1201): the wheel and the C++ examples (`bash test_examples.sh --hip`, or the manual cmake/ctest run with `-DPython_EXECUTABLE`), as Phase 7 asks; and every Linux platform revalidates at `9882f005`. Device-code change since `ea013807`: only the HIP launch cap in `wp_cuda_launch_kernel` (host code); `tile.h`/`aot.h` change no library translation unit.
- After Jeff's ruling on question 1: if compression is adopted, the full suite on a compressed build on Linux and Windows, and the Windows sizes.

## Porting session 20 (2026-11-19, linux-gfx950 MI355X, porter): version rename and the performance work (in progress)

Lock held by linux-gfx950 since session 19 (orient warned of a re-stamp; the earlier holder, job `...140431-warp114-phase7...`, had stopped). Fork `release/1.14.0.amd0` from `9882f005`. Environment: fleet job venv (Python 3.11) with `rocm[devel]==10.2.0` from `https://stable.repo.amd.com/rocm/whl-next/` (root `$(python -m rocm_sdk path --root)`); the image's ROCm 10.1 (`/opt/python-3.11-venv/.../_rocm_sdk_devel`) for AMD's 1.13 port only; nvcc 12.8; `HIP_VISIBLE_DEVICES=1` kept; `ulimit -c 0` for every GPU run. Fork clone copied from session 19's job (`cp -a`), fetched, `protect-fork` re-run (the copied hook pointed at the old job's checkout).

### Item 1: version rename (`36612c9c`, pushed, `advance-head` recorded)

`[ROCm] Version the AMD GPU build as 1.14.0+rocm10.2.0`: `VERSION.md`, `warp/config.py`, `warp/native/version.h`, and `docs/user_guide/installation.rst` (package name pattern `warp_lang-{ver}+rocm10.2.{rev}-...` and the download URL `.../download/v1.14.0+rocm10.2.0/warp_lang-1.14.0+rocm10.2.0-py3-none-manylinux_2_28_x86_64.whl`). No test mentions the version. `git grep -n "rocm\.0"` empty; `tools/pre-commit-hooks/check_version_consistency.py` passes (its regex accepts the local segment `rocm10.2.0`); `warp.__version__` = `1.14.0+rocm10.2.0`, `packaging` parses local `rocm10.2.0`. `version.h` is included only by `warp.cpp` (host C++), so no `.cu` translation unit changes. To verify when the GitHub Release exists: the download URL with a literal `+` resolves.

## Ruling on the wheel target list and compression (2026-11-19, Jeff Daily)

All targets ROCm 10.2 supports on the OS (25 on Linux, 22 on Windows), with the device code bundles compressed: `--offload-compress` at clang's default zstd level 3. Session 19's numbers: Linux warp.so 330.6 MB -> 10.6 MB, wheel 58.6 -> 36.9 MB, installed 426 -> 106 MB; first-use cost 35 to 50 ms per precompiled translation unit with a cold comgr cache, about 2 ms warm. Windows check on the gfx1151 laptop (scratch worktree `agent_space/warp-win/compwt` at `9882f005` plus the one option, nothing committed): `python build_lib.py --hip-arch gfx1151 -j 8` -> warp.dll 5,489,152 bytes (17.9 MB uncompressed), 11 `CCOB` bundles and 0 uncompressed; `agent_space/warp-win/comp_smoke.py` passes (a hipRTC kernel, `array_sum`, `radix_sort_pairs`, a `wp.Mesh` build and refit). Owed: the full suite on a compressed build on both OSes (it becomes the validated configuration), and the 22-target Windows size.

## Ruling on parallel per-target compilation (2026-11-19, Jeff Daily)

Add clang's `--offload-jobs` to the HIP compile options, with the total bounded so the build does not oversubscribe the machine, the Windows build in particular: the number of source files compiled at once times the per-file offload jobs must not exceed the build's `-j` value (which itself defaults to the machine's cores). The porter picks the split (for example offload jobs = max(1, j // concurrent files)) and measures the 22-target Windows build on the laptop with it. Session 19 measured 320 s -> 39 s on Linux with `--offload-jobs=16`.

## Ruling on toolkit_version and ROCm version attributes (2026-11-19, Jeff Daily)

`runtime.toolkit_version` and `runtime.driver_version` stay `None` on a HIP build, documented in the release notes. Add `runtime.rocm_version` and `runtime.hip_version`. Design to implement (the coordinator's reading of the CUDA pair, toolkit = build time and driver = run time; Jeff asked for the two attributes, not this split): `rocm_version` is the ROCm release the core library was built with, as `(major, minor)` (10, 2), from the build (the same value the `.rocm` marker records); `hip_version` is the version of the HIP runtime loaded in the process, as `(major, minor)` (7, 17), from `hipRuntimeGetVersion` or the equivalent. Both are `None` on CUDA and CPU builds. The `wp.init()` banner on a HIP build names them where a CUDA build names its toolkit and driver, if that is not already the case. Smallest diff; one test.

## Ruling on wheel classifiers (2026-11-19, Jeff Daily)

A ROCm wheel drops the two `Environment :: GPU :: NVIDIA CUDA` classifiers (in `setup.py`, beside the ROCm dependency logic; `pyproject.toml` untouched, the CUDA wheel's METADATA byte-identical). PyPI has no ROCm classifier, so none is added.

All four of session 19's questions are now ruled: compressed bundles at level 3 for all targets; `--offload-jobs` with a bounded total; `toolkit_version` stays None with `rocm_version` and `hip_version` added; classifiers removed. These go to the porter as the turn after the performance round.
### Item 2: temporary frees on HIP (`dd41f3d2`, pushed, `advance-head` recorded)

`[ROCm] Free temporary device memory without the null stream on HIP` (`warp.cu` +29, HIP-only): in `wp_free_device_async`'s non-graph, no-capture branch, HIP calls a new `free_async_after_device_streams(context, ptr)`: for every other stream in `g_streams` on the same device (`hipStreamGetDevice`), `cuEventRecord_f(cached_event, other)` + `cuStreamWaitEvent_f(current, cached_event)`, then `cudaFreeAsync(ptr, current)`; no current stream -> the old null-stream free. Deferred frees (`free_deferred_allocs`) and the graph-destroyed path still use the null stream (they run at sync points, rarely).

Why this rule (evidence, scripts below): upstream's comment relies on the CUDA legacy null stream ordering the free after the queued work of every blocking stream; Warp creates its streams with `CU_STREAM_DEFAULT` (blocking), so on CUDA a free can never overtake work on another Warp stream. HIP's null stream gives the same ordering (CLR's null stream waits on the device's blocking streams) and is correct, only slow. Three candidates built side by side at `36612c9c` (gfx950):

| tree | change | `free_race.py` alloc_on_s1 (late writes / reuse) | alloc_on_s2 | same_stream | `mesh.refit()` 10^4 | alloc+free, 0 / 3 / 15 extra streams |
|---|---|---|---|---|---|---|
| base (`9882f005`) | null-stream free | 0/10 (pointer not reused) | 0/10 (reused 10/10) | 0/10 | 132.0 us | 11.9 / 11.1 / 13.4 us |
| fx (plan's +6 scratch) | free on the current stream | **10/10** (reused 10/10) | **10/10** | 0/10 | 48.6 us | 8.5 / 9.0 / 8.1 us |
| amd (`da56e495`'s rule, main path only, +31) | track allocation stream, event wait if the free stream differs | 0/10 | **10/10** | 0/10 | 49.2 us | 6.3 / 6.8 / 8.1 us |
| free (committed, +29) | wait on every other Warp stream of the device, free on the current stream | 0/10 | 0/10 | 0/10 | 47.8 us | 6.2 / 6.8 / 12.2 us |

`free_race.py`: a 10 ms kernel on stream s1 writes 1.0 into a 4 MiB array after a spin; with s2 current the array is freed and a same-size `wp.zeros` is made on s2; late writes show as y.max() == 1. alloc_on_s1 = allocated with s1 current; alloc_on_s2 = allocated with s2 current but written on s1 (the case allocation-stream tracking misses). So stream tracking is not needed; the cheaper correct rule is the null stream's own ordering made explicit with events, at about 0.4 us per extra live stream (zero extra calls with one stream, the common case).

Where HIP's null-stream free costs time (`free_loop2.py`, 500 x {allocate 4 KiB, launch, free}): base alloc 2.7 / launch 7.3 / **free 25.8 us**, committed 2.7 / 7.0 / 3.8 us; per iteration (`free_loop.py`) 36.5 -> 10.9 us, issue time equal to drain time on base. A single free, or two, behind a 10 ms kernel does not block in either tree (`free_block*.py`). Side observation, same in both trees and so not caused by the stream choice: in a batch of {allocate, launch, free} behind a 10 ms kernel, the second free blocks the host until the kernel finishes (`free_block8.py`: free1 10.04 ms base, 9.55 ms committed; the others 0.00-0.03 ms). Probably CLR's pool release logic (`MemoryPool::FreeMemory` releases the free heap when it outgrows the busy heap); not investigated further, not a port issue.

Tests (fork clone built for gfx950 at the commit; `agent_space/runtests.sh`, one `python -m warp.tests -s autodetect -p <file>` per pattern): test_streams 24 OK (3 skipped), test_async 2373 OK, test_mempool 7 OK (2 skipped), test_graph 35 OK (3 skipped), test_copy 14 OK, test_bvh 13 OK, test_grouped_bvh 11 OK, test_mesh* 49 OK (18 skipped). CUDA: `cuda_pp.sh` (nvcc 12.8, sm_90) at `36612c9c` vs `dd41f3d2`: 24 preprocessed units, all identical except warp.cu's two, which differ only in `__LINE__` values (identical after `s/, [0-9]+\)/, L)/`). clang-format 22.1.2 applied.

Question for Jeff (no decision taken): `free_race.py` is a probe in these notes, not a fork test. Options: (a) leave it as a probe (smallest diff); (b) add it to `warp/tests/cuda/test_streams.py` as a HIP-and-CUDA test of cross-stream free ordering (about 40 lines; it passes on CUDA by the null-stream semantics, untested here without an NVIDIA GPU).

### Item 3: GPU LBVH build (`43c348ea`, pushed, `advance-head` recorded)

`[ROCm] Build the GPU LBVH top-down on HIP` (`bvh.cu` +321/-0; everything HIP-only: new kernels under `#if defined(__HIP_PLATFORM_AMD__)`, a HIP `LinearBVHBuilderGPU::build` before upstream's in `#else`, the constructor/destructor allocations under `#if !HIP`, and a HIP-only `depths` parameter in `mark_packed_leaf_nodes`). Credit in the commit: AMD-Ecosystem/warp `release/1.13.0.amd0` `6f0ae752` and `664cc4c5` (rtmadduri).

On this GPU the gap was much larger than on the gfx942 partition: same-host baseline (`res/bvh_s0`, `9882f005` vs AMD 1.13 built here with ROCm 10.1 and its documented `--hipcc-options="-Xarch_device -fno-inline"`): lbvh `bvh_build` 1.57x / 2.71x / 5.07x / 9.28x at 10^4 / 10^5 / 10^6 / 10^7 (10^7: 58.8 vs 6.3 ms). The 1.14 build is ~60 ms at 10^7 on both GPUs, i.e. it does not scale with CU count.

Profile (`rocprofv3 --kernel-trace --stats`, 3 builds of 10^7, `prof.sh`): 1.14 `build_hierarchy` 50.2 ms/build, `compute_total_bounds` 4.6, radix sort 1.7 (24 kernels: 64-bit keys), `mark_packed_leaf_nodes` 0.70; AMD 1.13 `build_karras_topology` 0.67, `compute_node_depths_and_max` 0.93, `refit_nodes_at_depth` 3.56 (35 levels), block bounds 0.19, sort 0.63 (30-bit), mark_packed 0.22.

Ablation (scratch tree `agent_space/wabl` = `dd41f3d2` + AMD's kernels added beside upstream's, builder pieces chosen at run time by a `WP_ABL` bitmask: 1 no `__threadfence` in build_hierarchy (timing only, not correct), 2 Karras topology + depths + depth-level refit with the max depth read to the host, 4 the same with a fixed launch count and device-side early exit, 8 block-bounds reduction, 16 one pooled allocation, 32 sort into `primitive_indices` with end_bit 30, 64 mark_packed walks parents anyway, 128 with 4: launch bound key_bits + index_bits instead of 97). `bvh_build`, ms, first sweep (`res/abl_s1`, `bench_bvh.py`, BVH_CONS=lbvh, median of 5 builds):

| variant | 10^4 | 10^5 | 10^6 | 10^7 |
|---|---|---|---|---|
| AMD 1.13 | 0.257 | 0.331 | 0.809 | 6.48 |
| upstream builder (abl0) | 0.436 | 1.340 | 6.26 | 60.6 |
| no fence (abl1, incorrect) | 0.452 | 0.651 | 1.93 | 18.4 |
| (a) Karras + depth-level refit, host max (abl2) | 0.516 | 0.511 | 1.72 | 11.6 |
| (b) block bounds only (abl8) | 0.459 | 1.278 | 5.81 | 53.4 |
| (c) pool only (abl16) | 0.450 | 1.290 | 6.74 | 58.1 |
| (d) sort only (abl32) | 0.459 | 1.430 | 6.24 | 59.9 |
| a+b (abl10) | 0.468 | 0.518 | 1.105 | 6.69 |
| a+b+c (abl26) | 0.323 | 0.397 | 1.29 | 6.60 |
| a+b+c+d (abl58) | 0.281 | 0.459 | 1.27 | 6.29 |

Interleaved rounds (`rounds.sh`, `bench_build.py`: 3 warm-up then 30 builds per size (10 at 10^7), median of the 5 runs' medians, ratio to AMD 1.13 in the same rounds; `res/build_s3`, `res/build_s4`):

| variant | 10^4 | 10^5 | 10^6 | 10^7 |
|---|---|---|---|---|
| a+b (abl10) | 1.31x | 1.19x | 1.04x | 1.09x |
| a+b, mark_packed walks (abl74) | 1.25x | 1.15x | 1.11x | 1.16x |
| a+b+c (abl26) | 0.99x | 0.96x | 1.00x | 1.11x |
| a+b+d (abl42) | 1.28x | 1.17x | 1.06x | 1.06x |
| a+b+c+d (abl58) | 0.92x | 0.89x | 0.95x | 1.01x |
| a+b+c+d, mark_packed walks (abl122) | 0.98x | 0.91x | 0.95x | 1.11x |
| a+b+c+d capture-safe, 97 launches (abl60) | 1.20x | 1.15x | 1.10x | 1.05x |
| a+b+c+d capture-safe, bound key+index bits (abl188) | 0.97x | 1.00x | 1.02x | 1.04x |

(mesh_build ratios within a few percent of these.) Answers to the hypotheses: the cost of the bottom-up pass is mostly the device-scope `__threadfence()` (60.6 -> 18.4 ms at 10^7 with only the fence removed); the remaining walk with its `atomicAdd` is about 8 ms more than the whole top-down topology + depth-level refit. So neither the integer atomic nor the walk alone: the fence dominates, which is why the cost does not shrink on a bigger GPU. The fused bounds kernels are the second piece at large sizes (4.6 ms at 10^7); the pool is what matters at 10^4-10^5 (about 0.1 ms of allocation calls); the 30-bit sort and the precomputed depths are each about 0.5 ms at 10^7. All four were needed to be within ~10% at every size, so the commit has all of them. (c) in 1.14 is only the pooled allocation: upstream 1.14's builder has no mid-build synchronize.

Capture: AMD's builder reads the max depth with `wp_memcpy_d2h` and ends with `cuCtxSynchronize`; on 1.14 that fails the capture-rebuild tests (abl58 tree: test_bvh FAILED errors=4, test_grouped_bvh failures=1 errors=2, "Warp CUDA error 700: an illegal memory access" from `wp_memcpy_d2h`, `res/t3_abl58.log`). The commit instead launches key_bits + index_bits refit levels (30 + ceil(log2 n), or 64 + ... with groups): along any root-to-leaf path the Karras common-prefix length (64 + clz(i^j) for equal keys) strictly increases and takes at most that many distinct values, so the bound is exact by construction; levels deeper than the device-side max return at once. Cost vs the host read: abl188 vs abl58 within noise (0.97-1.04x of AMD vs 0.94-1.02x).

A bug found and fixed before commit: the first version of the HIP build named its block count `num_blocks`, the name of a local inside the `wp_launch_device` macro, whose arguments are expanded in the macro's scope, so `reduce_block_bounds` received 1 instead of ceil(n/256) and the total bounds covered only the first 256 items. Trees stayed correct (every test passed, query results identical) but the Morton keys degenerated: `mesh_query_point` at 4x10^6 took 27 s and 10^7 ran past a 60 s timeout. Found by the benchmark's query step, not by tests; renamed `num_bounds_blocks` (device printf showed `nb 1` at 10^5). Every number above for the committed code was re-measured after the fix.

Committed code vs AMD 1.13 and the tree before item 3 (`res/build_s7`, 5 interleaved rounds, ms, `wp.Bvh` / `wp.Mesh`): 10^4 0.41/0.45 (before) -> 0.41/0.43 (AMD 0.40/0.41); 10^5 1.16/1.18 -> 0.58/0.53 (0.55/0.58); 10^6 6.41/6.28 -> 1.70/1.50 (1.67/1.45); 10^7 62.6/62.0 -> 6.98/8.00 (8.64/9.11). Spreads in that set were +-15-35% (GPUs 0, 3 and 4 of this board were at 100% from other users during these runs).

Tree quality: query outputs on the new trees are bit-identical to the upstream-built trees at every size (`cmp_bvh.py`: max|d dist| 0, max|d t| 0, 0 AABB count mismatches; AMD 1.13 differs by 1.3e-7 / 2.4e-7, as on gfx942). Query times, from the quiet first sweep where the same topology was built by abl58/abl60 (identical trees: same kernels and sort): `mesh_query_point` 0.88-1.00x, `bvh_query_aabb` 0.95-1.00x of the upstream-built tree; `mesh_query_ray` too noisy there (+-29%) and item 4's subject. Later query rounds on the committed tree (`res/q_s7`, `res/q_s8`) had spreads up to +-500% from the outside load and are not usable for 10% statements; they show nothing outside their noise. Refit: GPU time of `bvh_refit_kernel` per call at 10^5 (`rocprofv3`, 50 refits): 0.407 ms on the upstream-built tree, 0.408 ms on the new tree (AMD 1.13 0.502 ms); refit wall time at 10^4-10^5 is noisy (+-20-40%, `res/refit_s1`).

Tests (fork clone at the commit, gfx950): test_bvh 13 OK, test_grouped_bvh 11 OK (both incl. capture rebuild), test_mesh* 49 OK (18 skipped), test_apic_mesh 12 OK; `small_n.py` (1, 2, 3, 5, 17 items, build and rebuild, AABB hits and closest points vs brute force) 0 mismatches. `n == 1` needed a fix that AMD's kernel lacks: with no internal node nothing wrote `*root`; the topology kernel writes 0 then. CUDA: `cuda_pp.sh` `dd41f3d2` vs `43c348ea`: 22 identical, bvh.cu's two LINE-ONLY.

Observation (not a work item, question for Jeff): `Bvh.refit()`/`Mesh.refit()` use the same fence-bound bottom-up walk (`bvh_refit_kernel`) and take 44-46 ms at 10^7 in every tree, AMD's included; a depth-level refit (depths kept in the BVH) would likely be ~10x faster at large sizes. Options: (a) leave it (equal to AMD 1.13, no gap); (b) a follow-up HIP-only change in this line (more diff: depths stored per BVH, kept across rebuilds).

### Not a work item: `-ffp-contract=on` vs `fast` (recorded only; nothing changed, Jeff has not ruled)

The port passes `-ffp-contract=on` to hipRTC (`warp.cu` `wp_hip_compile_program`, comment "nvcc contracts within an expression only; Clang's 'on' mode is the closest match"); AMD's 1.13 passes `fast`. NVRTC's default is `--fmad=true`, and it corresponds to Clang's **`fast`**, not `on`:
- Measured (`agent_space/fmad_probe.py`, CUDA 12.8 `libnvrtc.so`, compute_90 PTX): a multiply and an add in one statement and in two statements (`float t = a*b; o = t + c;`) both compile to one `fma.rn.f32` with default options and with `--fmad=true`; `--fmad=false` gives `mul` + `add` in both. So NVRTC contracts across statements.
- Measured (`agent_space/hip_contract_probe.py`, ROCm 10.2 hipRTC, gfx950): `-ffp-contract=on` fuses the one-statement form (1 `v_fma`) but not the two-statement form (`v_mul_f32` + `v_add_f32`); `fast` fuses both.
- Sources (fetched 2026-11-20): Clang Users Manual (https://clang.llvm.org/docs/UsersManual.html), `-ffp-contract`: `fast` "enable fusion across statements disregarding pragmas, breaking compliance with the C and C++ standards (default for CUDA)"; `on` "enable C and C++ standard compliant fusion in the same statement unless dictated by pragmas (default for languages other than CUDA/HIP)"; `off` "disable fusion"; `fast-honor-pragmas` "fuse across statements unless dictated by pragmas (default for HIP)". NVRTC documentation (https://docs.nvidia.com/cuda/nvrtc/index.html), `--fmad={true|false}`: "Enables (disables) the contraction of floating-point multiplies and adds/subtracts into floating-point multiply-add operations (FMAD, FFMA, or DFMA)", "Default: true".
- So the port's comment is inaccurate: matching NVRTC's default would be `fast` (or `fast-honor-pragmas`, Clang's HIP default). The gfx942 attribution showed the choice changes `example_dem`'s trajectory (not its speed).

Question for Jeff (no decision taken): (a) keep `on` and correct the comment; (b) switch to `fast-honor-pragmas` (Clang's HIP default, cross-statement like NVRTC, and `#pragma clang fp contract` still applies); (c) switch to `fast` (AMD 1.13's choice). Any change alters floating-point results of every HIP kernel and needs a full-suite run.

### Item 4: speculative `mesh_query_ray` (`fb4803bb`, pushed, `advance-head` recorded)

`[ROCm] Traverse mesh_query_ray speculatively on HIP` (`mesh.h` +157, HIP-only under AMD's own guard `#if defined(__HIPCC__) && defined(__HIP_PLATFORM_AMD__)` with upstream's body in `#else`). Source: the two traversal hunks of `git diff v1.13.0 origin/release/1.13.0.amd0 -- warp/native/mesh.h` (the 527ba15c traversal as updated by f8f90f93 to dispatch cuBQL first); `git show 527ba15c` itself does not apply to 1.14. The third hunk of that diff (`WP_RESTRICT` on `Mesh::lowers/uppers/solid_angle_props`) is shared code and was left out. clang-format 22 reflowed it (whitespace only, checked).

Is it active in JIT kernels? `#error` under `__HIPCC_RTC__` inside the guard fired in a hipRTC compile, so yes (the guard is also true in the library's own HIP compile of `mesh.cu`).

Measurement method: `bench_bvh.py` wall times and the first two `rocprofv3` sweeps (`ray_sweep.sh`, `res/ray_s1`, `res/ray_s2`) were unusable for 10% statements: the same kernel's per-dispatch median varied 4-9x between sweeps minutes apart (e.g. upstream traversal, 10^5 sah: 1.66 vs 6.93 ms) while GPUs 0, 3 and 4 of this board ran other users' work at 100%; GPU 1's clock was not under my control. Several runs also had multi-second stalls (single dispatches up to 16 s, whole runs ~50 s); they occurred with the upstream traversal too (one 53 s run in 15, `res/` notes), so they are environmental. Final method (`ray_ab.sh`, `res/ray_ab1`): per cell three alternating pairs (upstream traversal tree `43c348ea`, then speculative tree), each `ray_one.py` = one Mesh + 20 launches of the bench ray kernel over 10^6 rays under `rocprofv3 --kernel-trace`, per-dispatch median, ratio within the pair:

| N | sah | median | lbvh |
|---|---|---|---|
| 10^4 | 0.63 (0.13/0.63/0.77) | 0.57 (0.56/0.69/0.57) | 0.65 (0.65/0.59/0.83) |
| 10^5 | **1.22** (1.23/1.16/1.22) | 0.54 (0.20/0.54/0.63) | 0.52 (0.52/0.82/0.51) |
| 10^6 | 0.76 (0.91/0.76/0.70) | 0.53 (0.09/0.53/0.69) | 0.70 (0.65/0.70/0.74) |
| 10^7 | 0.79 (0.13/0.79/1.41) | 0.85 (1.16/0.74/0.85) | 0.79 (0.79/0.79/0.79) |

(speculative / upstream, median of the three pairs; per-pair ratios in parentheses.) So the gain reproduces in 11 of 12 cells (15-48% less GPU time); SAH trees at 10^5 triangles are consistently 1.16-1.23x slower (sweep 1 also 1.13x). Not investigated further. Agreement: `ray_one.py` saved all 10^6 hit distances per cell: speculative vs upstream traversal max|dt| 0 and 0 hit mismatches in all 12 cells; AMD 1.13 vs upstream 1.2e-7 to 2.4e-7, 0 mismatches.

Tests (with `usd-core` installed in the job venv, which un-skips 18 mesh tests incl. ray grad/groups/count): test_mesh* 49 OK on both the speculative and the upstream-traversal tree, test_apic_mesh 12 OK, test_bvh 13 OK. CUDA: `cuda_pp.sh` `43c348ea` vs `fb4803bb`: all 24 identical.

Question for Jeff: the 10^5 SAH cell is a reproducible 1.2x loss inside an otherwise 0.5-0.85x change. Options: (a) keep as committed (AMD 1.13's behaviour, net win); (b) investigate a tuning (e.g. skipping the speculative child tests for CPU-built SAH/median trees) at more diff; (c) revert the commit.

### Item 5: tile Cholesky pivot (NOT committed: the gain did not reproduce on gfx950)

Change tried (scratch worktree `agent_space/wchol`, `tile_cholesky.h` +15/-0, guarded `#if defined(__HIP_DEVICE_COMPILE__)` so CUDA without libmathdx and CPU keep upstream's code; idea from `431e0533`, Rishi Madduri and Zhihui Du): in `scalar_cholesky_impl`, thread 0 computes the diagonal and writes `Out(j,j) = sqrt(s)`, `WP_TILE_SYNC()`, then every thread uses `invS = 1.0 / Out(j,j)`. Kept upstream's `1.0 /` (the plan's sketch wrote `T(1.0) /`, which rounds differently from upstream in fp32). CUDA: `cuda_pp.sh` with the change vs `fb4803bb`: all 24 units identical (device passes included).

Measurements (`chol_one.py`: 8192 blocks of `tile_cholesky`, 20 launches, `rocprofv3` per-dispatch median; NumPy check every run, all ok):
- Alternating pairs, 1.14 (`fb4803bb`) then 1.14 + pivot, 3 pairs per cell, full plan grid (`chol_ab.sh`, `res/chol_ab1`): target cells n=16: fp32 bd128 0.97 (0.97/0.89/0.97), fp32 bd256 0.96 (1.20/0.96/0.73), fp64 bd128 1.03 (1.03/0.91/1.10), fp64 bd256 1.17 (1.41/1.17/0.99). Elsewhere mostly 0.95-1.08 with some noisy outliers (fp32 n=8 bd256 1.44, fp64 n=4 bd64 2.04, fp64 n=8 bd256 1.65).
- Same cells with AMD 1.13 in the same sequence (113, 1.14, 1.14+pivot, 3 runs, `res/chol_ab_113`), median us: fp32 n16 bd128 48.5 / 50.2 / 48.8; fp32 n16 bd256 75.8 / 90.8 / 76.5 (1.14/1.13 1.06-1.22, pivot/1.13 0.89-1.03); fp32 n32 bd256 273 / 266 / 280; fp64 n16 bd128 53.7 / 58.1 / 58.6 (1.14/1.13 1.08-1.10, pivot/1.13 1.05-1.13); fp64 n16 bd256 77.7 / 87.7 / 85.6 (1.03-1.13; 0.97-1.10); fp64 n32 bd256 280 / 284 / 273.
- So the 1.14 gap at n=16 does exist on gfx950 (fp32 bd256 and fp64), the pivot closes fp32 bd256 in one measurement and not in the other, and does not close fp64. Not a reproduced gain; not committed, as ruled. Patch kept at `agent_space/chol_pivot.patch` (scratch; content is the diff above).
- In-place: upstream's `tile_cholesky_inplace` passes the same tile as `A` and `Out`, and every thread reads `A(j,j)` while thread 0 writes `Out(j,j)` in the same iteration, a theoretical race on multi-wave blocks. Not observed: 54 configurations (fp32/fp64, n 8/16/32, bd 64/128/256, in place) correct on both codes.

Question for Jeff: (a) leave the Cholesky as upstream's (the measured gap at n=16 bd 128/256 is 1.03-1.22x on this GPU); (b) port the rest of `431e0533`'s `cooperative_cholesky_impl` (its row-update structure differs too) and measure again, more diff in code CUDA also compiles unless guarded; (c) commit the fp32-helping pivot change anyway.

### Final validation at `fb4803bb` (gfx950 source build, nvcc build)

- Full default suite from the source build (fork clone built with `python build_lib.py --rocm-path "$(python -m rocm_sdk path --root)" --hip-arch gfx950 -j 64`, then `agent_space/suite.sh` = `python -m warp.tests -s autodetect` in the clone, `HIP_VISIBLE_DEVICES=1`, private kernel cache, wrapped in `utils/timeit.sh`; `usd-core` now installed in the job venv, which session 19's wheel run lacked): **Ran 7556 tests in 510.4 s, FAILED (failures=2, skipped=81)**. Skips: no suitable devices 24, conditional graph nodes 16, MathDx 8+1, two CUDA devices 7, mempool access 5+1+1, peer access 3+1+1, CPU memory access 2, NVRTC two archs 2, multi-GPU 2, rmm 1, memory-usage flakiness 1, Requires CUDA 1, NVRTC 1, stream priority 1 (the 27 usd-core skips of session 19 now run and pass; ml_dtypes 5 also gone).
- The two failures: `test_async` `test_copy_s2f_d2d_SrcPoolOff_DstPoolOff_NoStream_NoGrad_Graph_AccessSrcDst` and `..._NoGraph_AccessBoth`: after `wp.copy` and a stream sync, the last 124,992 of 1,000,000 destination elements were still 0. The same family failed in session 19's first full run before any of this round's changes (`test_copy_s2f_d2d_SrcPoolOff_DstPoolOff_NoStream_NoGrad_Graph_AccessSrcDst` and an `s2c` case, there the first 125,184 elements 0). Not reproduced in isolation: `test_async.py` alone 3 x 2373 OK; the `s2f_d2d`/`s2c_d2d` subset (192 tests) 15 x OK (2880 executions). Pool off on both arrays, so the arrays are freed with `cudaFree`, not through item 2's path. Signature: almost exactly 1/8 of the array (MI355X has 8 XCDs), at the tail here and the head in session 19, which looks like one XCD's writes not yet visible to the device-to-host read; only under full-suite load (parallel workers on the GPU). Not investigated further; recorded for the validator and as a possible ROCm report (question below).
- CUDA build at `fb4803bb` (worktree `agent_space/wcuda`, `agent_space/cudabuild.sh` = `python build_lib.py --cuda-path /usr/local/cuda --no-use-libmathdx -j 32`, nvcc 12.8, 13 gencode targets, 22 nvcc compiles): "Warp build succeeded", warp.so 199,765,024 bytes (the size session 19 recorded at `9882f005`), no `.rocm` marker; `import warp; warp.init()` -> "CUDA Toolkit 12.8, CUDA driver not available (NVRTC compilation available)", version 1.14.0+rocm10.2.0. Per-commit preprocessed checks are in each item above (items 2-4: only `__LINE__` changes in the changed file; item 1 changes no `.cu` unit).
- Integrity: `git -C projects/warp-1.14/src status --porcelain` shows no modified tracked file (build outputs untracked only).

### Item 4 reverted (`113e1e70`, pushed, `advance-head` recorded): the warmed measurement contradicts the gain

`[ROCm] Revert the speculative mesh_query_ray traversal` (`git revert fb4803bb`, new commit, no force push). `git diff 43c348ea 113e1e70` is empty, so the CUDA preprocessed check of `43c348ea` and the nvcc build (same CUDA translation units, `cuda_pp` c3 == c4) still apply.

Why: the `rocprofv3` pairs above ran each 20-launch process from an idle GPU; the same kernel's absolute time varied 4-18x between processes (upstream traversal, 10^4 sah: 2.0 to 18.9 ms per dispatch in the pairs vs 0.76-1.26 ms warmed), so those ratios compared GPUs at different clocks. The plan's own `bench_bvh.py` (warm-up launch, then timed repetitions) on the final trees, and then a dedicated warmed A/B, both disagree with them:
- `res/final_bvh` (one run each, `9882f005` / `fb4803bb` / AMD 1.13): ray after/before 1.00, 1.01, 1.00 at 10^4; 1.01, 0.97, 0.62 at 10^5; **1.50**, 1.00, 0.98 at 10^6; 1.01, 0.71, 0.98 at 10^7 (sah, median, lbvh). And 1.14 before is already faster than AMD 1.13 in most ray cells on this GPU (e.g. 10^4 sah 0.887 vs 1.064 ms).
- `res/q_final` (`bench_query.py` = `bench_bvh.py` queries only, 15 x 5 launches after a warm-up launch; tree `w43` = `43c348ea` sources with the same `warp.so` vs `fb4803bb`; 3 interleaved rounds; ratio of medians, speculative / upstream): sah 0.92 / 1.01 / 1.19 / 1.03, median 1.21 / 1.54 / 1.02 / 0.99, lbvh 1.23 / 1.21 / 1.00 / 1.02 at 10^4 / 10^5 / 10^6 / 10^7. Same rounds, unchanged code: `mesh_query_point` 0.99-1.03, `bvh_query_aabb` 1.00-1.08 (the noise floor).
So on gfx950 the speculative traversal does not reproduce the gfx942 gain; per the ruling it is not carried. Lesson (moat-porting/cuda-to-rocm candidate, below): time GPU kernels after a warm-up inside one process; per-process cold runs under a profiler compare different clock states.
After the revert: test_mesh* 49 OK, test_apic_mesh 12 OK (usd-core installed).

## Lesson for cuda-to-rocm: compare GPU kernel variants warmed up, interleaved, against a noise floor
Reference file: `references/validation.md` (performance comparisons) or `diagnosis.md`. On a shared multi-GPU board (other jobs at 100% on sibling GPUs), absolute kernel times of the same code varied 4-18x between short processes, and per-process profiler runs that start from an idle GPU compared different clock states: a traversal change looked 15-48% faster in 11 of 12 cells that way and was 0-54% slower when timed after a warm-up launch with repeated launches in one process. What held up: (1) warm up inside the process and time many launches; (2) interleave the trees in rounds (A, B, A, B) and take the median of per-run medians; (3) include a metric whose code is identical in both trees as the noise floor in the same rounds; (4) for functional regressions that only show as speed, include a query/throughput check of the built structures, because correctness tests pass on degenerate-but-valid trees. (warp-1.14)

### Item 5 addendum: warmed GPU-time triplets (verdict unchanged)

`chol_one.py` with 200 launches under `rocprofv3`, median of the last 100 dispatches, AMD 1.13 / 1.14 (`113e1e70`) / 1.14 + pivot interleaved, 3 runs (`res/chol_warm2`): median pivot/1.14 = fp32 n16 bd128 0.99, n16 bd256 0.97, n32 bd256 0.84; fp64 n16 bd128 1.01, n16 bd256 1.04, n32 bd256 0.93. Per-run ratios still swing 0.3-3x (GPU clock under the board's outside load). A wall-clock `bench_cholesky.py`-style A/B in this period (`res/chol_warm`) sat on a ~125 us per-launch host floor in every tree and is unusable. No reproducible gain at the target cells; item 5 stays uncommitted.

### Final benchmarks (plan's four scripts; 1.14 before = `9882f005`, after, AMD 1.13 = `release/1.13.0.amd0` `5035c8ed` built here for gfx950 with ROCm 10.1 and its documented flags; all on this MI355X, so the three columns are same-host)

Two full runs, `drive.sh <script> <out> base src 113`: run 1 (`res/final_*`, after = `fb4803bb`, i.e. with the since-reverted ray traversal) and run 2 (`res/final2_*`, after = `113e1e70`, the final head). **Caveat: both runs were taken while other users loaded GPUs 0, 3 and 4 of this board; cells whose code is identical in two trees differ by up to 21x (run 1, tile matmul fp32 16x16 bd256 0.415 vs 0.043 ms) and `example_dem` gives bit-identical final states (absmean 8.212409503796131 before and after) at 10.3 vs 18.5 s. Single-run cells are not 10%-accurate; the interleaved measurements in items 2-5 are the evidence.** The plan's gfx942 numbers (different GPU, 38-CU partition) are not comparable in absolute terms.

lbvh builds (run 1, the quieter one; ms; after/before; after/AMD):

| op | N | before | after | AMD 1.13 | after/before | after/AMD |
|---|---|---|---|---|---|---|
| mesh_build | 10^4 | 0.670 | 0.457 | 0.456 | 0.68 | 1.00 |
| bvh_build | 10^4 | 0.595 | 0.433 | 0.431 | 0.73 | 1.00 |
| mesh_build | 10^5 | 1.390 | 0.616 | 0.627 | 0.44 | 0.98 |
| bvh_build | 10^5 | 1.465 | 0.585 | 0.575 | 0.40 | 1.02 |
| mesh_build | 10^6 | 7.014 | 1.727 | 1.744 | 0.25 | 0.99 |
| bvh_build | 10^6 | 6.433 | 1.373 | 1.240 | 0.21 | 1.11 |
| mesh_build | 10^7 | 58.63 | 7.979 | 7.976 | 0.14 | 1.00 |
| bvh_build | 10^7 | 59.30 | 7.178 | 7.450 | 0.12 | 0.96 |

(Run 2 lbvh builds after/before: 0.61, 0.22, 2.41 (noise outlier: 4.456 ms for bvh at 10^5), 0.29, 0.60, 0.11, 0.10.) `mesh.refit()` (run 1): sah/median 10^4 0.57x before (0.142 -> 0.081 ms, AMD 0.089), lbvh 10^4 0.66x, median 10^5 0.64x; 10^6-10^7 within noise. Queries: result agreement `cmp_bvh.py` after vs before 0 / 0 / 0 mismatches in both runs (AMD 1.13 1.3e-7 / 2.4e-7); `mesh_query_point` and `bvh_query_aabb` after/before 0.85-1.08 (run 1). `mesh_query_ray`: see item 4 and its revert; at the final head the traversal is upstream's, and AMD 1.13's speculative one was faster in run 2 (after/AMD 1.11-1.84) and slower in run 1 (before/AMD 0.81-0.83 at 10^4), consistent with the cold/warm dependence described in the revert.
Full tables: `res/final_bvh.md`, `res/final2_bvh.md` (scratch; the key rows are above).

Tile Cholesky (run 2, factorization rows at n=16/32, bd 128/256; ms): fp32 n16 bd128 0.059 / 0.109 / 0.058, n16 bd256 0.096 / 0.135 / 0.135, n32 bd128 0.243 / 0.248 / 0.235, n32 bd256 0.319 / 0.343 / 0.326; fp64 n16 bd128 0.114 / 0.112 / 0.075, n16 bd256 0.143 / 0.112 / 0.102, n32 bd128 0.293 / 0.292 / 0.293, n32 bd256 0.347 / 0.334 / 0.337 (before / after / AMD; the Cholesky code is identical before and after, so before/after differences are noise). Correctness vs NumPy: 1.14 before and after pass all 128 cells in both runs; AMD 1.13 fails 21 cells in each run (10 `cholesky_solve`, 11 `upper_solve`, block_dim 128/256), the inherited upstream 1.13 race the gfx942 comparison found, now reproduced on gfx950.

Tile matmul (run 2; ms, before / after / AMD): fp32 16/64 0.015 / 0.071 / 0.069; 16/256 0.022 / 0.073 / ERR (AMD's rocWMMA static assert, as on gfx942); 32/64 0.084 / 0.077 / 0.098; 32/256 0.081 / 0.079 / 0.083; 64/64 0.284 / 0.282 / 0.207; 64/256 0.106 / 0.111 / 0.138; fp16 16/64 0.013 / 0.011 / 0.015; 16/256 0.077 / 0.019 / 0.020; 32/64 0.073 / 0.019 / 0.075; 32/256 0.029 / 0.029 / 0.077; 64/64 0.115 / 0.087 / 0.102; 64/256 0.097 / 0.086 / 0.071. The matmul code is identical before and after; differences are noise. All cells correct.

Examples (run 2; s or ms, before / after / AMD): example_dem 9.38 / 16.05 / 15.92 s (graph capture on in all; final state identical before and after, AMD 8.36 vs 8.21 as on gfx942: different floating-point contraction); example_sph 784 / 696 / 696 ms; navier_stokes 136 / 170 / 73 ms. Noise dominates (run 1: dem 10.34 / 18.51 / 10.89 s).

### Summary of session 20

Fork `release/1.14.0.amd0` `9882f005..113e1e70` (all pushed, `advance-head` after each; stage left at `porting` as directed):
1. `36612c9c` [ROCm] Version the AMD GPU build as 1.14.0+rocm10.2.0
2. `dd41f3d2` [ROCm] Free temporary device memory without the null stream on HIP (+29, HIP-only): `mesh.refit()` 10^4 132 -> 48 us; alloc+launch+free 36.5 -> 10.9 us; correct where the plan's +6 change and AMD's stream tracking free too early.
3. `43c348ea` [ROCm] Build the GPU LBVH top-down on HIP (+321, HIP-only; credit 6f0ae752/664cc4c5 rtmadduri): 10^7 build ~60 -> 7 ms, within ~10% of AMD 1.13 at every size (0.81-1.06x in the last interleaved rounds), capturable, query results bit-identical.
4. `fb4803bb` speculative `mesh_query_ray` (credit 527ba15c deepsek) -> reverted by `113e1e70`: no gain on gfx950 when measured warmed.
5. Cholesky pivot: measured, no reproducible gain on gfx950, not committed.
Validation at `113e1e70`: full default suite from the gfx950 source build 7556 tests OK (skipped=81); nvcc 12.8 CUDA build of the same sources succeeded (warp.so 199,765,024 bytes), every preprocessed CUDA unit identical to `9882f005`'s apart from `__LINE__` values in bvh.cu and warp.cu; `jargon.py --port` clean before each push.

### Questions for Jeff (session 20; no decision taken; session 19's four questions are still open)

1. Item 2 probe: (a) keep `free_race.py` as a probe in the notes; (b) add it as a cross-stream free-ordering test to `test_streams.py` (~40 lines).
2. `Bvh.refit()`/`Mesh.refit()` stay on the fence-bound bottom-up walk (44-46 ms at 10^7 in every tree, AMD's too): (a) leave; (b) a HIP depth-level refit in this line (more diff).
3. `mesh_query_ray`: reverted on the warmed measurement. (a) leave reverted (current); (b) re-apply `fb4803bb` if Jeff weighs the cold/contended-GPU behaviour (15-48% less GPU time there) over the warmed result (0.92-1.54x).
4. Cholesky n=16 (1.14 is 1.03-1.22x of AMD 1.13 here): (a) leave upstream's; (b) port the rest of `431e0533`'s cooperative factorization and measure on a quiet GPU; (c) commit the pivot change anyway.
5. `-ffp-contract`: (a) keep `on` and correct the comment; (b) `fast-honor-pragmas`; (c) `fast` (details in the section above).
6. The full-suite-only `test_async` copy flake with a 1/8-of-the-array signature (seen in sessions 19 and 20, never in isolation): (a) leave as a known flake for the validators; (b) register a deferred ROCm report and build a standalone reproducer first.

### Next step (head `113e1e70`)

- Jeff: the questions above and session 19's four (offload compression, `--offload-jobs`, `toolkit_version` on HIP, classifiers); they may add commits before `ported`.
- Performance re-measurement belongs on a quiet GPU: this session's absolute numbers were taken with three sibling GPUs at 100%. Items 2 and 3 were decided on interleaved in-process measurements (robust to that); item 4's revert and item 5's non-commit should be re-checked on an idle board if Jeff wants either reconsidered.
- Validators: every platform revalidates at `113e1e70`. Device-code changes since `9882f005`: `warp.cu` (HIP frees) and `bvh.cu` (HIP LBVH builder), both in the library; no JIT header changes (`mesh.h` is back to `43c348ea`'s). Windows: the HIP free path and the LBVH builder are not Windows-specific but have never run there; run test_streams, test_async, test_mempool, test_graph, test_bvh, test_grouped_bvh (incl. capture rebuild) and test_mesh* first. wave32 (gfx1100/gfx1201): the LBVH kernels use 256-thread blocks with `cub::BlockReduce` and a shared-memory reduction sized 256, wave-size independent by construction, but check test_bvh/test_grouped_bvh there.
- Then the release items from session 19's next step (release notes text, Windows wheel, C++ examples on other platforms).
- Scripts: `notes-session20-scripts.md`.

## Porting session 21 (2026-11-20, linux-gfx950 MI355X, porter): the four session-19 rulings, release build and wheel, flake and fp-contract evidence

Same job and environment as session 20 (lock with linux-gfx950, `HIP_VISIBLE_DEVICES=1`, `ulimit -c 0`, job venv with `rocm[devel]==10.2.0`). Fork `release/1.14.0.amd0` `113e1e70..7059a340` (one push, `advance-head` recorded); stage left at `porting`.

GPU 1 was not exclusively this job's: `amd-smi process -g 1` repeatedly showed a process from outside the fleet (no other fleet job was active) holding 13-28 GB with changing PIDs, also when none of this job's processes ran. The dispatcher's pin was kept. This matters for the timing noise of session 20 and for the flake below (no run was truly alone on the device).

### Fork commits

1. `17ce1131` [ROCm] Compress the device code bundles of the HIP library (`build_dll.py` +2): `--offload-compress` (clang's default zstd level 3) in the HIP compile options. 25-target Linux build: warp.so 10,877,768 bytes (330.6 MB uncompressed in session 19; slightly larger than session 19's 10,574,664 because of the new HIP LBVH kernels), 11 `CCOB` bundles, 0 uncompressed; 317 s at -j 64 (session 19: 320 s uncompressed). test_bvh 13, test_grouped_bvh 11, test_array_reduce 16, test_utils 62, test_streams 24 (3 skipped), test_hash_grid 23 OK on that build.
2. `a85b2a09` [ROCm] Compile each source's GPU targets in parallel, within -j (`build_dll.py` +8). Split chosen: the library's C++ files are compiled first (a wait on their futures before the HIP compiles are submitted, HIP builds only), then at most `c = min(jobs, number of .cu files)` HIP files run at once, each with `--offload-jobs = max(1, jobs // c)`, so files at once times jobs per file never exceeds the job count the library build has. Note: `build_lib.py` already gives the library half of `-j` because `warp-clang` builds beside it (`parallel_args.jobs = max(1, args.jobs // 2)`). Measured full 25-target compressed build (`agent_space/relbuild.sh`, wall time, 256-core host):

| build | -j 8 (the default here: min(cores, 8)) | -j 64 |
|---|---|---|
| before (`17ce1131`) | 481 s, 479 s (4 workers) | 317 s (32 workers) |
| after (`a85b2a09`) | 484 s, 485 s (4 workers, `--offload-jobs=1`) | 181 s (32 workers, `--offload-jobs=2`) |

   (load average 6-15 during the -j 8 pairs; the first after-run at -j 8 (513 s) and the -j 64 after-run overlapped flake experiments, load up to 118; the -j 8 pairs above were rerun with nothing else of this job's on the CPU except a 20 s build.) So at the default -j the bound gives one job per file and nothing changes: 4 workers for 12 HIP files are already busy, and no split within the bound can use more than 4 threads. Session 19's 39 s used `--offload-jobs=16` with 11 concurrent files at -j 64, i.e. 176 threads, which the ruling's bound forbids.
3. `30df3d97` [ROCm] Report the ROCm and HIP runtime versions on AMD GPU builds (`context.py` +17/-7, `warp.cu` +5/-1, `warp.h` 1 line, `test_devices.py` +15). `runtime.rocm_version` = (major, minor) from the `.rocm` marker's version line (the marker `read_rocm_marker` already parses, written by every HIP build since session 19), set where the library is loaded, None otherwise. `runtime.hip_version` existed (port commit `bf6bd5ba`) but was `HIP_VERSION` of the build's headers; `wp_hip_runtime_version()` now returns `hipRuntimeGetVersion()` (same encoding: 71726434 -> 7.17, checked through ctypes; 0 on failure -> None). Banner on HIP: "ROCm 10.2, HIP runtime 7.17" (was "HIP 7.17"); the no-device HIP branch uses the same text. `toolkit_version`/`driver_version` stay None on HIP. Test `test_runtime_rocm_and_hip_versions` (TestDevices): on HIP, toolkit/driver None, rocm_version two ints, hip_version > (0, 0) when a GPU is present; otherwise both None. HIP: test_devices 14 OK, test_logger 34 OK, test_fixedarray 20 OK. CUDA build (nvcc 12.8, `wcuda` at `7059a340`): banner unchanged ("CUDA Toolkit 12.8, CUDA driver not available (NVRTC compilation available)"), rocm_version None, hip_version None, toolkit_version (12, 8); test_devices 10 OK (3 skipped) incl. the new test.
4. `7059a340` [ROCm] Drop the NVIDIA CUDA classifiers from ROCm wheels (`setup.py` +7): `rocm_wheel` flag set from the marker check; `parse_config_files` filters `self.metadata.classifiers` (prefix "Environment :: GPU :: NVIDIA CUDA") only for a ROCm wheel. ROCm wheel: 12 classifiers, no NVIDIA CUDA ones, `Dynamic: classifier` added by setuptools. CUDA wheel (nvcc build binaries, `python -m build --wheel -C--build-option=-Plinux-x86_64` with the old and the new setup.py): METADATA byte-identical (23,081 bytes, md5 0a3e3feeff8e7c50e63f40599e55ee7b), file list identical (552 entries), both CUDA classifiers present.

CUDA check on the final tree (`cuda_pp.sh` at `7059a340` vs `43c348ea` = `113e1e70`'s tree): 22 units identical, warp.cu's two LINE-ONLY. nvcc build at `7059a340`: "Warp build succeeded", warp.so 199,765,024 bytes (same as sessions 19 and 20). `jargon.py --port` clean before the push.

### Release build, wheel and first full suite on a compressed build (head `7059a340`)

- Release configuration: `agent_space/relbuild.sh <clone> 64` = `python build_lib.py --rocm-path "$(python -m rocm_sdk path --root)" -j 64` (default target list from the ROCm SDK: gfx908, gfx90a, gfx942, gfx950, gfx1010-1012, gfx1030-1036, gfx1100-1103, gfx1150-1153, gfx1200, gfx1201, gfx1250 = 25), wrapped in `utils/timeit.sh`: "Warp build succeeded"; every one of the 14 HIP compile commands carries `--offload-compress` and `--offload-jobs=2`; warp.so 10,881,864 bytes, 11 `CCOB` bundles, 0 uncompressed; `warp.so.rocm` = ROCm root and `10.2.0`.
- Wheel: `python -m build --wheel -C--build-option=-Plinux-x86_64 -o <job>/wheel` in the clone at `7059a340` (no modified tracked file). **Kept at `/var/lib/jenkins/fleet/jobs/mi355-20261119-154230-warp114-perf-porter-mi355-prompt/wheel/warp_lang-1.14.0+rocm10.2.0-py3-none-manylinux_2_28_x86_64.whl`: 37,154,691 bytes, sha256 `2100e06cdf8a29c78c850f417ff8202ee535e5227e41b83fead5b59d54d5ea25`.** Contents: `warp/bin/warp.so` 10,881,864, `warp/bin/warp-clang.so` 67,136,896, `warp/bin/warp.so.rocm` 140; no nvrtc/libmathdx/cudart/nvJitLink file; 14 licence files (no CUDA/NVRTC/libmathdx licence). METADATA: `Version: 1.14.0+rocm10.2.0`, `Requires-Dist: numpy`, `Requires-Dist: rocm<10.3,>=10.2`, no `Environment :: GPU :: NVIDIA CUDA` classifier; WHEEL tag `py3-none-manylinux_2_28_x86_64`. (Session 19's uncompressed wheel: 58,613,145 bytes.)
- Install: a clean venv was not possible: the fleet guard refuses `pip` for any environment other than the job venv ("pip changing the Python environment every job on this host shares"), and I did not work around it. Installed instead into the job venv (`$FLEET_VENV/bin/pip install <wheel> --extra-index-url https://stable.repo.amd.com/rocm/whl-next/`), which had no `warp` before and already had rocm/rocm-sdk-core/rocm-sdk-devel 10.2.0, numpy 1.26.4, usd-core, pillow: "Successfully installed warp-lang-1.14.0+rocm10.2.0". Installed size `du -sb site-packages/warp` = **106,300,189 bytes** (session 19 uncompressed: 425,949,591); installed warp.so sha256 equals the build's. From outside the tree (`<job>/wheelrun`): `warp.__file__` in site-packages; banner "ROCm 10.2, HIP runtime 7.17"; a hipRTC kernel runs; libamdhip64.so.7 and libhiprtc.so.7 mapped from the venv's `_rocm_sdk_core/lib`.
- Full default suite from the installed wheel (`agent_space/suite_wheel.sh`, cwd `<job>/wheelrun`, `HIP_VISIBLE_DEVICES=1`, fresh kernel cache): **Ran 7557 tests in 500.0 s, OK (skipped=81)**. Same 81 skips as session 20's source-build run (no suitable devices 24, conditional graph nodes 16, MathDx 8+1, two CUDA devices 7, mempool access 5+1+1, peer access 3+1+1, CPU memory access 2, NVRTC two archs 2, multi-GPU 2, rmm 1, memory-usage flakiness 1, Requires CUDA 1, NVRTC 1, stream priority 1); the extra test is the new version test. This is the first full suite on a compressed build; it passed (the outside process was on GPU 1 during it).

### Evidence 6 (not a commit): `-ffp-contract=fast-honor-pragmas` in place of `on`

Scratch worktree `agent_space/wff` = `113e1e70` with only `wp_hip_compile_program`'s `stored_options.push_back(fuse_fp ? "-ffp-contract=on" : ...)` changed to `"-ffp-contract=fast-honor-pragmas"`; gfx950 build; `WARP_DEBUG=1` confirms hipRTC receives `-ffp-contract=fast-honor-pragmas` (the fork shows `on`). Full default suite from that tree (`agent_space/suite_ff.sh`, fresh cache): **Ran 7556 tests in 578.5 s, FAILED (failures=1, skipped=81)**, against 7556 OK (81 skipped) for `113e1e70` with `on` (session 20). Same skips. The one changed test: `test_mat.TestMat.test_inverse_float16_cuda_0`, line 702, the adjoint of `wp.inverse` for a 2x2 fp16 matrix: result -31.31 against the reference -31.378423, absolute difference 0.0659 with the test's fp16 tolerance `atol=0.05` (an fp16 ulp at 31 is 0.0156, so about 4 ulps). Deterministic: `-k test_inverse_float16` fails 3/3 with `fast-honor-pragmas` and passes 3/3 with `on`. Nothing else changes outcome. So option (b) of question 5 costs one fp16 tolerance failure in an upstream test (whether CUDA, which also fuses across statements, passes it with margin is not known here; the test's CUDA tolerance would need a `TEST_WITH_ROCM`-gated bump or the option stays `on`).

## Release notes draft (1.14.0+rocm10.2.0)

Draft for Jeff (written by the porter in session 21; nothing published). Wheel sizes for reference: Linux wheel 37.2 MB, 106.3 MB installed. Text between the fences is the proposed GitHub Release body; one line per paragraph as GitHub renders it.

```
# Warp 1.14.0 for AMD GPUs (ROCm 10.2)

This release is NVIDIA Warp 1.14.0 ported to AMD GPUs with HIP. It is built from the release-1.14 branch of NVIDIA/warp with the changes needed to compile and run Warp's GPU code on AMD Instinct and Radeon GPUs; Warp's Python API, kernels and features are those of upstream 1.14.0, apart from the differences listed below. It is not an NVIDIA release; report problems with this build at https://github.com/AMD-Ecosystem/warp/issues.

## Install

Linux x86-64 (Python 3.10 or newer):

    pip install https://github.com/AMD-Ecosystem/warp/releases/download/v1.14.0+rocm10.2.0/warp_lang-1.14.0+rocm10.2.0-py3-none-manylinux_2_28_x86_64.whl --extra-index-url https://stable.repo.amd.com/rocm/whl-next/

Windows x86-64: the same command with the win_amd64 wheel from this release.

The wheel depends on the ROCm 10.2 Python packages (`rocm>=10.2,<10.3`), which pip installs from AMD's package index given by `--extra-index-url`; no system ROCm installation is needed. Check the install with `python -c "import warp; warp.init()"`, which prints the ROCm and HIP runtime versions and the AMD GPUs found.

## Version numbers

`1.14.0` is the upstream Warp version. The local part `rocm10.2.0` names the ROCm release the wheel is built for and requires (10.2), followed by this build's number for that Warp version and ROCm release. The last number is not a ROCm patch version (AMD's PyTorch wheels use the same shape with that other meaning): a rebuild of 1.14.0 for ROCm 10.2 with fixes becomes 1.14.0+rocm10.2.1, an upstream patch release starts again at 1.14.1+rocm10.2.0, and a build for a later ROCm changes the middle, for example 1.14.0+rocm10.3.0.

## Supported GPUs

The library contains code for every GPU target ROCm 10.2 supports on the operating system: 25 targets on Linux (gfx908, gfx90a, gfx942, gfx950, gfx1010, gfx1011, gfx1012, gfx1030 to gfx1036, gfx1100 to gfx1103, gfx1150 to gfx1153, gfx1200, gfx1201, gfx1250) and 22 on Windows. Kernels written in Python are compiled at run time for the GPU present.

Tested on hardware with Warp's test suite: AMD Instinct MI355X (gfx950), MI300X (gfx942) and MI250X (gfx90a), Radeon Pro W7800 (gfx1100) and Radeon AI PRO R9700 (gfx1201) on Linux, and a Ryzen AI Max (gfx1151) on Windows. The other targets are compiled but have not been run.

## Differences from Warp's CUDA build

- `wp.get_runtime().toolkit_version` and `driver_version` (the CUDA Toolkit and CUDA driver versions) are `None`, as in a CPU-only build. The new `rocm_version` is the ROCm release the library was built with, for example `(10, 2)`, and `hip_version` is the version of the HIP runtime loaded in the process, for example `(7, 17)`; both are `None` in CUDA and CPU builds. Code that compares `toolkit_version` with a tuple without checking for `None` fails on AMD GPUs; use the capability functions (for example `wp.is_conditional_graph_supported()`) instead.
- Conditional graph nodes are not available in HIP: `wp.capture_if` and `wp.capture_while` cannot be captured into a graph on AMD GPUs (`wp.is_conditional_graph_supported()` returns False), as with CUDA before 12.4.
- MathDx (cuBLASDx, cuFFTDx, cuSOLVERDx) is NVIDIA-only and not included. Tile operations use Warp's built-in implementations, as in a CUDA build without MathDx; 2D and 3D `tile_fft`/`tile_ifft` of `vec2f`/`vec2d` tiles, which have no built-in implementation, are not available.
- `tile_matmul` uses AMD matrix instructions (MFMA on Instinct GPUs, WMMA on gfx11 and gfx12 for fp16 and bf16) where the tile shapes allow it. They accumulate fp16 and bf16 products in fp32, so results can differ from the scalar path in the last bits (they are closer to the exact product).
- Shared-memory tiles are limited to 64 KiB per block (tiles plus Warp's small static allocation) on MI250X, MI300X and Radeon GPUs and to 160 KiB on MI355X, less than on recent NVIDIA GPUs; Warp warns when a kernel's request does not fit.
- Textures: MI300X and MI355X (and MI250X for 3D textures) have no texture sampling hardware. Textures there are sampled in software on the GPU, with the same results as Warp's CPU textures and slower than hardware sampling; the CUDA interop handles of those textures (`cuda_array`, `cuda_texture`, `cuda_surface`) are not available on them.
- Spin locks contended within one wavefront do not make progress on AMD GPUs (no independent thread scheduling within a wavefront); `warp/examples/core/example_spin_lock.py` reports this instead of hanging.
- In native snippets, `__frsqrt_rn` and `__fsqrt_rn` are within 1 ulp but not correctly rounded on AMD GPUs (wrong in the last bit for about 11 and 15 percent of inputs), unlike CUDA. Warp's own kernels do not use them.
- In native snippets, warp-level intrinsics need 64-bit lane masks: AMD Instinct GPUs have 64-lane wavefronts, so a 32-bit mask such as `0xffffffff` is a compile error. Use `~0ull` or `__activemask()`, which are correct on 32- and 64-lane GPUs, and do not assume `lane = tid & 31`.
- Stream priorities are accepted and reported, but the HIP runtime schedules all streams at normal priority.
- On Windows, a graph launch waits for the graph to finish before returning (it works around a HIP runtime race between successive launches of a graph).
- Floating-point results of the same kernel can differ in the last bits from a CUDA build, as between any two compilers: multiply-adds are fused within an expression only.
- The precompiled library stores its GPU code compressed; the first use of each part of it in a process costs a few tens of milliseconds while the HIP runtime's code-object cache is empty (later about 2 ms).
- Not included: MathDx, NVRTC and the CUDA runtime. The wheel contains no NVIDIA software, and its licence files list only what it contains.
```

Notes for Jeff on the draft (not part of the body):
- Hardware list: the GPUs in the evidence and porting sessions of these notes (gfx950, gfx942, gfx90a, gfx1100, gfx1201 on Linux; gfx1151 on Windows). The release's gate platforms must validate the uploaded wheels at the final commit before this is true of the release itself; at the time of writing only gfx950 has run the final wheel (session 21).
- The fp-contract sentence describes the current `-ffp-contract=on`; it changes if question 5 is ruled (b) or (c).
- Windows wheel name and the 22 Windows targets are from session 13; the Windows wheel has not been built yet.
- Upstream's "requires NVIDIA driver" paragraphs in the docs are untouched; the installation guide's ROCm paragraph already points here.

### Evidence 5 (not a commit): the `test_async` copy flake, a wrong-result defect on gfx950 under concurrent GPU load

Reproduced, with a diagnostic that locates it; no reproducer without Warp yet, so no deferred item registered.

Reproduction rates (tree `agent_space/w43` = `113e1e70` sources; subset = `python -m warp.tests -s autodetect -p test_async.py -k s2f_d2d -k s2c_d2d`, 192 tests, ~10 s per run):

| condition | failing runs | notes |
|---|---|---|
| subset alone (session 20) | 0 / 15 | |
| subset alone, clean (`flake_b.sh ... 0 30`) | 0 / 30 | |
| subset while a full suite runs on the same GPU, run 1 (`flake_a.sh`) | 5 / 51 (6 failing tests of 9,792) | during it, `copy_flake load` processes and -j 8 builds also ran for part of the time; failures at 03:29-03:39 coincided with those |
| the same, run 2 with only the suite beside it (instrumented) | 0 / 225 | suite killed after 38 min |
| subset with 8 `copy_flake load` processes (4 streams each), nothing else of this job's (`flake_b.sh ... 8 30`) | 1 / 30 | |
| the same, instrumented (`flake_b.sh ... 8 40`) | 2 / 40 | diagnostics below |
| full default suite alone | 1 run with 2 failures (session 20, `fb4803bb`), 0 in 3 other full runs (sessions 19 wheel run 2, 20 at `113e1e70`, 21 wheel) | session 19's first full run also had 2 |

Caveat for every row: an outside, non-fleet process used GPU 1 throughout (13-28 GB), so "alone" means without this job's other processes. So it needs at least a second process on the device; with only the subset and nothing else of this job's it never failed (45 runs).

Every failing case so far has the destination allocated with the memory pool off (`hipMalloc`), the default stream, and a strided source copied by Warp's kernel (`s2c` and `s2f`; source pooled or not, graph-captured or not). The instrumented test (scratch edit in `w43/warp/tests/cuda/test_async.py` only: on a mismatch, read `dst` again, then after `wp.synchronize()`, and print the wrong-index range) gave:
- `test_copy_s2f_d2d_SrcPoolOn_DstPoolOff_NoStream_NoGrad_NoGraph_AccessSrcDst`: 125,184 wrong, indices 256..999,935 (blocks 1..3905), first wrong value 1263.0 (data from an earlier allocation at that address, the test's per-test offsets differ); second read 125,184 wrong; after `wp.synchronize()` 125,184 wrong.
- `test_copy_s2c_d2d_SrcPoolOn_DstPoolOff_NoStream_NoGrad_NoGraph_AccessSrcDst`: first read 118,272 wrong, indices 768..998,399 (blocks 3..3899), values 0.0; second read 124,928 wrong; after `wp.synchronize()` 124,928 wrong.

Reading: 125,184 = 489 x 256 and 124,928 = 488 x 256 are exactly the copy kernel's workgroups with block index = 1 (mod 8) (489 of them between 1 and 3905) and = 3 (mod 8) (488 between 3 and 3899). Workgroups are placed round-robin over the 8 XCDs of an MI355X, so in each failure exactly one XCD's share of the copy is missing. The stale data stays wrong in device memory across a second read and a device synchronize, and in one case the wrong count grew between two reads (more of that XCD's elements reverted to zero later). That looks like one XCD's L2 lines for this buffer being lost or overwritten (dirty stale lines from the address's previous use written back late, or the copy kernel's writes not written back), not like an early host read: the read-back is not the cause (it stays wrong after synchronizing). Earlier sessions' "first/last 1/8 is zero" reading came from NumPy's abbreviated printout; the full index sets are interleaved, not contiguous.

Standalone HIP program (`agent_space/flake/copy_flake.hip`, below): the same sequence without Warp (`hipMalloc`ed destination zero-filled from pageable memory, strided source written by a kernel, device sync, strided-to-contiguous kernel of 3907 256-thread blocks on a blocking stream, stream sync, `hipMemcpyAsync` D2H on the null stream into fresh `malloc` memory, immediate check, per-iteration `hipFree`/`hipMalloc` so addresses and stale data recur; variants with a null-stream sync and with the source from `hipMallocAsync`), under 4-8 `copy_flake load` processes with 1-4 streams each and once beside a full Warp suite: **0 wrong in 200 + 2,000 + 3,000 + 4,000 iterations.** Not reproduced without Warp, so no deferred item (as instructed). What Warp does that the program does not: kernels from Warp's library and hipRTC modules, graph capture and graph exec destruction in neighbouring tests, mempool trimming/access settings, more streams and events per process. Needs gfx950? Not decidable on this host (no other arch); MI300X (gfx942) is also an 8-XCD device and would show the same placement pattern if it is an XCD effect; MI250X (one die per device) and Radeon (single XCD) would be the controls.

Question for Jeff (release blocker until understood): options: (a) dispatch the instrumented subset-under-load to the gfx942 host (8 XCDs) and to gfx90a/gfx1100 (one XCD per device) to see whether it follows the multi-XCD hardware; (b) capture an `AMD_LOG_LEVEL=4` (or `rocprofv3 --hip-runtime-trace --kernel-trace --memory-copy-trace`) trace of a failing subset run to extract Warp's exact API sequence for a Warp-free reproducer, then register a `rocm-bug-report` deferred item with it; (c) both. Without a reproducer it cannot be reported to ROCm.

### Questions for Jeff (session 21; no decision taken; session 20's six are still open)

7. The `test_async` flake (Evidence 5): (a) cross-arch dispatch of the instrumented subset under load; (b) API trace of a failing run to build a Warp-free reproducer, then a deferred `rocm-bug-report`; (c) both. Until then it is a known wrong-result risk on gfx950 under concurrent GPU work, not seen without a second process.
8. Build parallelism: `build_lib.py` halves `-j` for the library because `warp-clang` builds alongside, but `warp-clang` finishes in about 5 s while the library's HIP compile takes ~480 s at the default -j 8, so half the budget is idle (4 threads busy on an 8-thread budget). Options: (a) leave it (upstream's logic, smallest diff); (b) give the library's HIP phase the full `-j` once `warp-clang` is done (more diff in upstream's `build_lib.py`). With (b) the default build would run 8 HIP files at once instead of 4 (not measured).
9. Release-notes draft below: wording, the hardware-run list, and whether the fp-contract and compressed-bundle lines belong in it.

### Next step (head `7059a340`)

- Not moved to `ported`, as directed: session 20's six questions and this session's three (7-9) are with Jeff and may add commits; the release notes draft needs his edit.
- The release wheel at `7059a340` is built and passed the full suite on gfx950 from the installed wheel (not a clean venv; see above): `/var/lib/jenkins/fleet/jobs/mi355-20261119-154230-warp114-perf-porter-mi355-prompt/wheel/warp_lang-1.14.0+rocm10.2.0-py3-none-manylinux_2_28_x86_64.whl`, sha256 `2100e06c...d5ea25`. If the head moves, the wheel is rebuilt and revalidated (release goal).
- Owed by the ruling on compression: the full suite on a compressed build on Windows (gfx1151) and the 22-target Windows size; the Windows `--offload-jobs` measurement (with the bound, the default -j 8 gives one job per file there too: 11 HIP files, 4 workers).
- Validators: every platform revalidates at `7059a340` (compressed library, new `rocm_version`/`hip_version`, setup.py classifiers); the gate platforms should validate the wheel file that will be uploaded.
- Flake (question 7) before the release.
- Scripts: `notes-session21-scripts.md`.

## Evidence session (2026-11-20, linux-gfx942, evidence only): test_async copy flake at `7059a340`

Requested by Jeff Daily as an evidence-only fleet job (the porting lock stays with linux-gfx950; no lock taken, no state changed, nothing pushed to the fork). Question: does session 21's Evidence 5 (one XCD's share of the copy kernel's workgroups left unwritten, under a second process on the GPU) occur on an MI300X, which also has 8 XCDs, with the same mod-8 block pattern? **Answer: yes, readily, and with the same pattern.** Every one of the 47 diagnosed failing subset tests (plus the 2 the full suite itself hit) has exactly one mod-8 class of 256-element blocks persistently wrong (in whole blocks where the stale data is zero); it needs a second process on the same GPU (0/30 alone, 0/160 with the same load on another GPU); the standalone `copy_flake` program still does not reproduce it (0 in 27,000 iterations under the same load).

### Host and build

- GPU: AMD Instinct MI300X HF (gfx942:sramecc+:xnack-, 304 CUs, 192 GiB, device id 0x74a9), accelerator partition SPX, memory partition NPS1 (so one device = 8 XCDs, as on the MI355X); VBIOS version 00151101. Dispatcher pin `HIP_VISIBLE_DEVICES=4` kept; every GPU run under `ulimit -c 0`.
- Kernel 6.8.0-110-generic, amdgpu driver 6.16.13 (`/sys/module/amdgpu/version`; amd-smi 27.1.0, whose host ROCm reports 10.1.0, not used by the runs).
- Job venv (`$FLEET_VENV`, Python 3.11.17, numpy 1.26.4) had ROCm 10.1 wheels; `$FLEET_VENV/bin/pip install "rocm[devel]==10.2.*" --extra-index-url https://stable.repo.amd.com/rocm/whl-next/` -> rocm, rocm-sdk-core, rocm-sdk-devel 10.2.0 (`python -m rocm_sdk version` 10.2.0; the venv's 10.1 `rocm-sdk-libraries` and `rocm-sdk-device-*` packages stayed, Warp loads neither). Warp banner: "ROCm 10.2, HIP runtime 7.17"; `runtime.rocm_version` (10, 2), `hip_version` (7, 17).
- Fork clone `git clone --branch release/1.14.0.amd0 https://github.com/AMD-Ecosystem/warp.git w; git -C w checkout 7059a340`; build `python build_lib.py --rocm-path "$(python -m rocm_sdk path --root)" --hip-arch gfx942 -j 112` (112 = physical cores = `MAX_JOBS`): "Warp build succeeded" in 43 s. Only tracked change in the clone: the scratch diagnostic (below), never committed.
- `amd-smi process -g 4`: before the runs "No running processes detected" (unlike session 21's GPU 1 on the MI355X, no outside process shared the device, so "alone" here is truly alone); during the load runs only this job's 8 `copy_flake load` processes (2.2 GB each) and the subset; during the suite runs only the suite's workers and the subset. GPU 0 had another job's process throughout; GPU 5 (used for step 4) was idle before it.

### Diagnostic

Session 21's scratch diagnostic applied to `w/warp/tests/cuda/test_async.py`, extended (still scratch, never committed) to save every failing test's reads to an `.npz` (`dst` first read `x1`, second read `x2`, read after `wp.synchronize()` `x3`, the reference `src.numpy()` and, from load run 49 on, a second `src.numpy()`) and to analyse them against the expected values `arange(offset, offset + n)`. That matters: **in 10 of the 47 diagnosed subset failures (and 1 of the suite's own 2) the wrong array is the reference, not `dst`.** `src` is strided, so `src.numpy()` runs the same strided-to-contiguous copy kernel into a staging buffer and copies that to the host; the zeros are in that staging copy, and a second `src.numpy()` is correct every time (8/8 where it was taken, the suite's `fi2fi` included). So the fault is in the strided copy kernel's output wherever it runs (also `fi2fi` and an `s2s` grad copy in the full suite), not in the `dst` allocation.

Scripts (adapted `flake_b.sh`, `flake_c.sh` = `flake_a.sh` that restarts the suite until N subset runs are done, `cf_load.sh`, the analysis scripts) and the diagnostic diff: `notes-evidence-gfx942-flake-scripts.md`.

### Results (subset = `python -m warp.tests -s autodetect -p test_async.py -k s2f_d2d -k s2c_d2d`, 192 tests)

| condition | command | failing runs | failing tests | mean subset time |
|---|---|---|---|---|
| alone | `flake_b.sh $PWD/w $PWD/logs/alone 0 30` | **0 / 30** | 0 | 8.0 s |
| 8 `copy_flake load` (4 streams each) on the same GPU | `flake_b.sh $PWD/w $PWD/logs/load8 8 160` | **17 / 160** | 18 of 30,720 | 11.6 s |
| beside the full default suite on the same GPU | `flake_c.sh $PWD/w $PWD/logs/suite 60` | **18 / 60** | 29 of 11,520 | 16.5 s |
| 8 `copy_flake load` on GPU 5, subset on GPU 4 | `flake_b.sh $PWD/w $PWD/logs/load8_gpu5 8 160 5` | **0 / 160** | 0 | 10.8 s |
| `copy_flake check` (no Warp) with 8 load processes on the same GPU | `cf_load.sh 2000`, `cf_load.sh 20000`, `cf_load.sh 5000 pool` | 0 wrong of 2,000 + 20,000 + 5,000 iterations | | |

The full suite beside the subset (suite 1 of the suite condition, `HIP_VISIBLE_DEVICES=4`): **Ran 7557 tests in 858.5 s, FAILED (failures=2, skipped=113)**; the two failures are the same defect in the suite's own `test_async`: `test_copy_fi2fi_d2d_SrcPoolOff_DstPoolOff_NoStream_NoGrad_Graph_AccessSrcDst` (reference read-back, 124,928 wrong = class 3 mod 8, 488/488 whole blocks) and `test_copy_s2s_d2d_SrcPoolOn_DstPoolOff_NoStream_Grad_Graph_AccessDstSrc` (the grad comparison, 125,184 = 489 x 256 mismatched; not instrumented). (113 skips against 81 on gfx950 not examined; not this session's question.) The second suite instance was killed when the 60 subset runs were done.

### Every failure

"class r mod 8: a/b blocks" = wrong 256-element blocks with block index = r (mod 8), a of the b blocks in that class; "whole" = all 256 (or the 64 of the tail block 3906) elements wrong. The grid is 3907 blocks of 256 threads for n = 1,000,000. Where the stale data is non-zero (an earlier test's values at the same address), elements whose stale value happens to equal the expected one count as right, so blocks show as partial.

**Load condition (8 `copy_flake load` on the same GPU), 17 failing runs, 18 failing tests:**

| run | test (`test_copy_` omitted) | mismatched | wrong read, persistent mod-8 class, blocks |
|---|---|---|---|
| 6 | `s2f_d2d_SrcPoolOn_DstPoolOff_NoStream_NoGrad_NoGraph_AccessBoth` | 236416 | (no dump yet; from the printed diagnostic) source read-back wrong (the wrong `dst` values equal the expected ones, the reference holds zeros); 489 whole blocks, i.e. one full class of 489 (0, 1 or 2; the class is not printed), plus 913 partial blocks in classes 0-4, 6 |
| 7 | `s2c_d2d_SrcPoolOn_DstPoolOff_Stream0_NoGrad_Graph_AccessNone` | 86064 | (no dump yet) dst 86064 -> 59392 after re-read/sync; first read also partial blocks in classes 0, 7 (cleared on re-read); class 3 mod 8: 232/488 blocks, 232 whole, blocks 2051..3899 |
| 15 | `s2c_d2d_SrcPoolOff_DstPoolOff_Stream0_NoGrad_NoGraph_AccessNone` | 247872 | (no dump yet) dst 247872 -> 124928 after re-read/sync; first read also partial blocks in all other classes (cleared on re-read); class 5 mod 8: 488/488 blocks, 488 whole |
| 42 | `s2f_d2d_SrcPoolOn_DstPoolOff_NoStream_NoGrad_Graph_AccessDstSrc` | 248896 | source read-back wrong (248896) class 6 mod 8: 488/488 blocks, 456 whole, blocks 6..3902; other classes partial: [0, 1, 2, 3, 4, 5, 7] |
| 48 | `s2f_d2d_SrcPoolOff_DstPoolOff_NoStream_NoGrad_NoGraph_AccessSrcDst` | 207104 | source read-back wrong (207104) class 1 mod 8: 489/489 blocks, 489 whole, blocks 1..3905; other classes partial: [0, 2, 3, 4, 6] |
| 67 | `s2f_d2d_SrcPoolOn_DstPoolOff_NoStream_NoGrad_Graph_AccessDstSrc` | 210944 | source read-back wrong (210944); re-read ok class 6 mod 8: 488/488 blocks, 457 whole, blocks 6..3902; other classes partial: [0, 1, 2, 3, 4, 5, 7] |
| 83 | `s2f_d2d_SrcPoolOff_DstPoolOff_NoStream_NoGrad_NoGraph_AccessSrcDst` | 124928 | source read-back wrong (124928); re-read ok class 4 mod 8: 488/488 blocks, 488 whole, blocks 4..3900 |
| 94 | `s2f_d2d_SrcPoolOn_DstPoolOff_NoStream_NoGrad_NoGraph_AccessDstSrc` | 240320 | source read-back wrong (240320); re-read ok class 4 mod 8: 488/488 blocks, 488 whole, blocks 4..3900; other classes partial: [1, 3, 5, 7] |
| 103 | `s2f_d2d_SrcPoolOff_DstPoolOff_NoStream_NoGrad_NoGraph_AccessSrcDst` | 250368 | dst 250368 -> 125184 after re-read/sync; first read also class [1] (cleared on re-read); class 0 mod 8: 489/489 blocks, 489 whole, blocks 0..3904 |
| 103 | `s2f_d2d_SrcPoolOn_DstPoolOff_NoStream_NoGrad_Graph_AccessNone` | 247872 | source read-back wrong (247872); re-read ok class 3 mod 8: 488/488 blocks, 488 whole, blocks 3..3899; other classes partial: [1, 5, 6, 7] |
| 111 | `s2f_d2d_SrcPoolOn_DstPoolOff_NoStream_NoGrad_NoGraph_AccessDstSrc` | 250368 | dst 250368 -> 125184 after re-read/sync; first read also class [1] (cleared on re-read); class 0 mod 8: 489/489 blocks, 489 whole, blocks 0..3904 |
| 116 | `s2f_d2d_SrcPoolOn_DstPoolOff_NoStream_NoGrad_NoGraph_AccessBoth` | 226432 | source read-back wrong (226432); re-read ok class 4 mod 8: 488/488 blocks, 488 whole, blocks 4..3900; other classes partial: [1, 3, 5, 6, 7] |
| 120 | `s2f_d2d_SrcPoolOn_DstPoolOff_Stream0_NoGrad_Graph_AccessNone` | 119040 | dst 119040 -> 59392 after re-read/sync; first read also class [1] (cleared on re-read); class 6 mod 8: 232/488 blocks, 232 whole, blocks 2054..3902 |
| 130 | `s2f_d2d_SrcPoolOn_DstPoolOff_Stream0_NoGrad_Graph_AccessNone` | 124928 | dst 124928 -> 124928 after re-read/sync; class 6 mod 8: 488/488 blocks, 488 whole, blocks 6..3902 |
| 137 | `s2f_d2d_SrcPoolOn_DstPoolOff_NoStream_NoGrad_Graph_AccessDstSrc` | 248896 | source read-back wrong (248896); re-read ok class 6 mod 8: 488/488 blocks, 456 whole, blocks 6..3902; other classes partial: [0, 1, 2, 3, 4, 5, 7] |
| 146 | `s2f_d2d_SrcPoolOn_DstPoolOff_NoStream_NoGrad_NoGraph_AccessBoth` | 249920 | source read-back wrong (249920); re-read ok class 4 mod 8: 488/488 blocks, 488 whole, blocks 4..3900; other classes partial: [1, 3, 5, 6, 7] |
| 149 | `s2c_d2d_SrcPoolOff_DstPoolOff_NoStream_NoGrad_Graph_AccessNone` | 212864 | dst 212864 -> 124928 after re-read/sync; first read also class [1, 5, 6, 7] (cleared on re-read); class 3 mod 8: 488/488 blocks, 488 whole, blocks 3..3899 |
| 160 | `s2c_d2d_SrcPoolOn_DstPoolOff_Stream0_NoGrad_NoGraph_AccessBoth` | 249984 | dst 249984 -> 124992 after re-read/sync; first read also class [1, 3, 4, 5, 7] (cleared on re-read); class 2 mod 8: 489/489 blocks, 489 whole, blocks 2..3906 |

**Beside the full default suite on the same GPU, 18 failing runs, 29 failing tests:**

| run | test (`test_copy_` omitted) | mismatched | wrong read, persistent mod-8 class, blocks |
|---|---|---|---|
| 2 | `s2f_d2d_SrcPoolOn_DstPoolOff_NoStream_NoGrad_Graph_AccessNone` | 125184 | dst 125184 -> 125184 after re-read/sync; class 0 mod 8: 489/489 blocks, 489 whole, blocks 0..3904 |
| 4 | `s2f_d2d_SrcPoolOff_DstPoolOff_NoStream_NoGrad_Graph_AccessDstSrc` | 249856 | dst 249856 -> 124928 after re-read/sync; first read also class [6] (cleared on re-read); class 5 mod 8: 488/488 blocks, 488 whole, blocks 5..3901 |
| 6 | `s2f_d2d_SrcPoolOn_DstPoolOff_Stream0_NoGrad_Graph_AccessBoth` | 250112 | dst 250112 -> 124928 after re-read/sync; first read also class [1] (cleared on re-read); class 6 mod 8: 488/488 blocks, 488 whole, blocks 6..3902 |
| 7 | `s2f_d2d_SrcPoolOff_DstPoolOff_NoStream_NoGrad_NoGraph_AccessBoth` | 250368 | dst 250368 -> 125184 after re-read/sync; first read also class [1] (cleared on re-read); class 0 mod 8: 489/489 blocks, 489 whole, blocks 0..3904 |
| 10 | `s2c_d2d_SrcPoolOn_DstPoolOff_Stream0_NoGrad_NoGraph_AccessSrcDst` | 125184 | dst 125184 -> 125184 after re-read/sync; class 0 mod 8: 489/489 blocks, 489 whole, blocks 0..3904 |
| 10 | `s2f_d2d_SrcPoolOff_DstPoolOff_Stream0_NoGrad_NoGraph_AccessNone` | 250368 | dst 250368 -> 125184 after re-read/sync; first read also class [1] (cleared on re-read); class 0 mod 8: 489/489 blocks, 489 whole, blocks 0..3904 |
| 16 | `s2c_d2d_SrcPoolOn_DstPoolOff_NoStream_NoGrad_NoGraph_AccessSrcDst` | 124992 | dst 124992 -> 124992 after re-read/sync; class 2 mod 8: 489/489 blocks, 489 whole, blocks 2..3906 |
| 17 | `s2c_d2d_SrcPoolOff_DstPoolOff_Stream0_NoGrad_NoGraph_AccessSrcDst` | 124992 | dst 124992 -> 124992 after re-read/sync; class 2 mod 8: 489/489 blocks, 489 whole, blocks 2..3906 |
| 17 | `s2f_d2d_SrcPoolOff_DstPoolOff_Stream0_NoGrad_NoGraph_AccessBoth` | 250368 | dst 250368 -> 125184 after re-read/sync; first read also class [1] (cleared on re-read); class 0 mod 8: 489/489 blocks, 489 whole, blocks 0..3904 |
| 18 | `s2c_d2d_SrcPoolOff_DstPoolOff_Stream0_NoGrad_Graph_AccessDstSrc` | 125184 | dst 125184 -> 125184 after re-read/sync; class 0 mod 8: 489/489 blocks, 489 whole, blocks 0..3904 |
| 18 | `s2c_d2d_SrcPoolOff_DstPoolOff_Stream0_NoGrad_Graph_AccessNone` | 125184 | dst 125184 -> 125184 after re-read/sync; class 0 mod 8: 489/489 blocks, 489 whole, blocks 0..3904 |
| 19 | `s2f_d2d_SrcPoolOn_DstPoolOff_Stream0_NoGrad_NoGraph_AccessDstSrc` | 249920 | dst 249920 -> 124992 after re-read/sync; first read also class [3] (cleared on re-read); class 2 mod 8: 489/489 blocks, 489 whole, blocks 2..3906 |
| 19 | `s2f_d2d_SrcPoolOn_DstPoolOff_Stream0_NoGrad_NoGraph_AccessNone` | 249920 | dst 249920 -> 124992 after re-read/sync; first read also class [3] (cleared on re-read); class 2 mod 8: 489/489 blocks, 489 whole, blocks 2..3906 |
| 22 | `s2c_d2d_SrcPoolOff_DstPoolOff_NoStream_NoGrad_Graph_AccessSrcDst` | 124928 | dst 124928 -> 124928 after re-read/sync; class 3 mod 8: 488/488 blocks, 488 whole, blocks 3..3899 |
| 24 | `s2c_d2d_SrcPoolOff_DstPoolOff_Stream0_NoGrad_Graph_AccessBoth` | 125184 | dst 125184 -> 125184 after re-read/sync; class 1 mod 8: 489/489 blocks, 489 whole, blocks 1..3905 |
| 33 | `s2c_d2d_SrcPoolOff_DstPoolOff_NoStream_NoGrad_Graph_AccessSrcDst` | 1227 | dst 1227 -> 1227 after re-read/sync; class 0 mod 8: 15/489 blocks, 0 whole, blocks 0..344 (gaps) |
| 33 | `s2c_d2d_SrcPoolOff_DstPoolOff_NoStream_NoGrad_NoGraph_AccessNone` | 4876 | dst 4876 -> 4876 after re-read/sync; class 0 mod 8: 36/489 blocks, 8 whole, blocks 0..688 (gaps) |
| 33 | `s2f_d2d_SrcPoolOn_DstPoolOff_NoStream_NoGrad_Graph_AccessNone` | 6656 | dst 6656 -> 6656 after re-read/sync; class 5 mod 8: 26/488 blocks, 26 whole, blocks 5..205 |
| 34 | `s2c_d2d_SrcPoolOff_DstPoolOff_Stream0_NoGrad_Graph_AccessBoth` | 8192 | dst 8192 -> 8192 after re-read/sync; class 0 mod 8: 32/489 blocks, 32 whole, blocks 0..248 |
| 34 | `s2c_d2d_SrcPoolOff_DstPoolOff_Stream0_NoGrad_NoGraph_AccessBoth` | 11008 | dst 11008 -> 11008 after re-read/sync; class 5 mod 8: 43/488 blocks, 43 whole, blocks 5..341 |
| 34 | `s2c_d2d_SrcPoolOff_DstPoolOff_Stream0_NoGrad_NoGraph_AccessNone` | 124992 | dst 124992 -> 124992 after re-read/sync; class 2 mod 8: 489/489 blocks, 489 whole, blocks 2..3906 |
| 34 | `s2f_d2d_SrcPoolOff_DstPoolOff_NoStream_NoGrad_NoGraph_AccessBoth` | 124928 | dst 124928 -> 124928 after re-read/sync; class 5 mod 8: 488/488 blocks, 488 whole, blocks 5..3901 |
| 34 | `s2f_d2d_SrcPoolOn_DstPoolOff_NoStream_NoGrad_NoGraph_AccessNone` | 124928 | dst 124928 -> 124928 after re-read/sync; class 5 mod 8: 488/488 blocks, 488 whole, blocks 5..3901 |
| 35 | `s2f_d2d_SrcPoolOff_DstPoolOff_NoStream_NoGrad_NoGraph_AccessDstSrc` | 249856 | dst 249856 -> 124928 after re-read/sync; first read also class [6] (cleared on re-read); class 5 mod 8: 488/488 blocks, 488 whole, blocks 5..3901 |
| 45 | `s2f_d2d_SrcPoolOff_DstPoolOff_Stream0_NoGrad_NoGraph_AccessNone` | 250368 | dst 250368 -> 125184 after re-read/sync; first read also class [1] (cleared on re-read); class 0 mod 8: 489/489 blocks, 489 whole, blocks 0..3904 |
| 48 | `s2c_d2d_SrcPoolOn_DstPoolOff_NoStream_NoGrad_Graph_AccessBoth` | 124928 | dst 124928 -> 124928 after re-read/sync; class 4 mod 8: 488/488 blocks, 488 whole, blocks 4..3900 |
| 50 | `s2c_d2d_SrcPoolOn_DstPoolOff_NoStream_NoGrad_NoGraph_AccessBoth` | 125184 | dst 125184 -> 125184 after re-read/sync; class 0 mod 8: 489/489 blocks, 489 whole, blocks 0..3904 |
| 57 | `s2c_d2d_SrcPoolOff_DstPoolOff_Stream0_NoGrad_NoGraph_AccessBoth` | 124928 | dst 124928 -> 124928 after re-read/sync; class 5 mod 8: 488/488 blocks, 488 whole, blocks 5..3901 |
| 57 | `s2f_d2d_SrcPoolOn_DstPoolOff_NoStream_NoGrad_NoGraph_AccessBoth` | 250112 | dst 250112 -> 124928 after re-read/sync; first read also class [0] (cleared on re-read); class 7 mod 8: 488/488 blocks, 488 whole, blocks 7..3903 |

### Reading

- **Same defect, same pattern as gfx950.** In all 47 diagnosed failing subset tests exactly one block class mod 8 stays wrong after a re-read and a `wp.synchronize()`. In 40 it is that class's whole share of the copy (488 or 489 blocks, the 125,184 / 124,992 / 124,928 counts of session 21); in the rest it is a contiguous part of the class: blocks 2051..3899 and 2054..3902 (232 blocks each, i.e. exactly the class blocks after block 2048 = 2 MiB into the array), or the class's first 26, 32 or 43 blocks (blocks 5..205, 0..248, 5..341), or class-0 blocks with non-zero stale data (1,227 and 4,876 wrong). Nothing narrower fits: for the persistent reads the printed residues are one value mod 2, 4 and 8, two mod 16 and 38 mod 304 (one block per CU), i.e. exactly one mod-8 class. The class varies (0-7 all seen), as expected if the XCD varies.
- **A second, transient class.** In 17 of the 37 `dst` failures the first read has a second class wrong too (whole blocks, or partial), which is correct on the second read without any synchronization in between: those workgroups' data reached memory late, after the host copy of the first read; the persistent class never arrives. When the reference read-back is the wrong array (read only once), the same shape appears: one class whole plus partial blocks in other classes.
- **Same-device only.** 0/160 with the identical 8-process load on another GPU of the host versus 17/160 on the same GPU (and 0/30 alone), so it is contention on the device, not host load. The suite beside the subset gave the highest rate (18/60), unlike session 21's instrumented suite run on gfx950 (0/225).
- `amd-smi process` snapshots during the loaded runs show non-zero `EVICTED_TIME` for some of the processes on GPU 4 (24 ms to 13 s), i.e. queue evictions happened while the copies ran; whether one coincides with a failure was not measured (the snapshots are every 10 or 25 runs). Together with the "class blocks after the 2 MiB boundary" and "first N class blocks" cases, it points at the per-XCD handling of one dispatch around a preemption, eviction or translation change, rather than at Warp's synchronization: a late or missing write-back of one XCD's L2 or a stale per-XCD translation would both give one XCD's share and keep the stale data across a device synchronize. This is a reading, not a finding.
- **Not gfx950-specific**: it follows the 8-XCD hardware (gfx942 and gfx950). The single-XCD controls (gfx90a: one die per device; gfx1100) are still open, as is a Warp-free reproducer: `copy_flake` under the same load gave 0 in 27,000 iterations, so the difference is in what Warp does around the copy (its library kernel, staging buffers from its allocator, graphs and memory pools in neighbouring tests, more streams and events), not in the bare kernel-then-D2H sequence.

### For Jeff (no decision taken here)

- Option (a) of session 21's question 7 is answered for gfx942: same defect, so it is not an MI355X-only risk; the release-blocking wrong-result risk applies to MI300X as well. A run on gfx90a and gfx1100 would show whether single-XCD devices are immune.
- Option (b) (API trace of a failing run for a Warp-free reproducer) is easier here than on gfx950: about one failing run in nine under 8 load processes, one in three beside the suite, ~10-17 s per run, and the device is otherwise idle.

## Evidence session (2026-11-20, linux-gfx90a, evidence only): test_async copy flake at `7059a340`

Requested by Jeff Daily as an evidence-only fleet job (the porting lock stays with linux-gfx950; no lock taken, no project state changed, nothing pushed to the fork). Question: does session 21's Evidence 5 (one XCD's share of the strided copy kernel's workgroups left unwritten while a second process uses the GPU) occur at all on a device with one die? **Answer: no, not observed.** 0 failing runs of 250 (0 of 48,000 subset tests) across alone, 8 `copy_flake load` processes and a full suite beside it, under the same conditions that fail readily on gfx942 (17/160 and 18/60, previous section) and on gfx950 (session 21).

### Host and build

- GPU: AMD INSTINCT MI250 (MCM) OAM (amd-smi market name "AMD Instinct MI250X / MI250", device id 0x740c, part 113-D65209-073), gfx90a:sramecc+:xnack-. Each device is one GCD: 104 CUs, 64 GiB, one L2 (so an MI250 rather than an MI250X, which has 110 CUs per GCD; the same die either way). 4 devices on the host (2 OAM packages). Dispatcher pin `HIP_VISIBLE_DEVICES=0` kept; every GPU run under `ulimit -c 0`.
- Kernel 6.8.0-101-generic, amdgpu driver 6.16.6 (`/sys/module/amdgpu/version`; amd-smi 27.1.0, whose host ROCm reports 10.1.0, not used by the runs).
- Job venv (`$FLEET_VENV`, Python 3.11.17, numpy 1.26.4) had ROCm 10.1 wheels; `$FLEET_VENV/bin/pip install "rocm[devel]==10.2.*" --extra-index-url https://stable.repo.amd.com/rocm/whl-next/` -> rocm, rocm-sdk-core, rocm-sdk-devel 10.2.0 (`python -m rocm_sdk version` 10.2.0; the venv's 10.1 `rocm-sdk-libraries` and `rocm-sdk-device-*` stayed, Warp loads neither). Warp banner: "Warp 1.14.0+rocm10.2.0 ... ROCm 10.2, HIP runtime 7.17", device "AMD Instinct MI250X / MI250" (64 GiB, gfx90a:sramecc+:xnack-, mempool enabled).
- Fork clone: `git clone --branch release/1.14.0.amd0 https://github.com/AMD-Ecosystem/warp.git warp; git -C warp checkout 7059a340`. Build (in `utils/timeit.sh warp-1.14 build`): `python build_lib.py --rocm-path "$(python -m rocm_sdk path --root)" --hip-arch gfx90a -j 128` (128 = physical cores = `MAX_JOBS`): "Warp build succeeded" in 62 s; warp.so 6,409,032 bytes. Only tracked change in the clone: the scratch diagnostic, never committed (`git status --porcelain`: ` M warp/tests/cuda/test_async.py`).
- `copy_flake.hip` unchanged from session 21, `hipcc -O2 --offload-arch=gfx90a` from the 10.2 SDK; `copy_flake check 50` alone: 0 wrong.
- `amd-smi process`: before the runs "No running processes detected" on all 4 devices (10 MB VRAM used each), so "alone" is truly alone and no outside process shared GPU 0 at any point; during the load condition exactly this job's 8 `copy_flake load` processes (1.1 GB each, at start and at run 80) plus the subset, GFX activity 99%; during the suite condition 9-10 processes (the suite's workers and the subset). No process left on GPU 0 afterwards.

### Diagnostic

Session 21's scratch diagnostic applied to `warp/tests/cuda/test_async.py` (first read, second read, read after `wp.synchronize()`, wrong-index range), extended with a second line that reports, for the wrong 256-element blocks, whether they are whole blocks and which moduli (2-8, 16) give a constant block-index residue, so a single-die pattern other than mod 8 (e.g. per shader engine) would have been named. It never fired: no run had a mismatch. A mismatch in the reference read-back (`src.numpy()`, which runs the same strided copy kernel; the gfx942 session found 10 of 47 failures there) would also have failed the assertion and printed, so neither array was ever wrong. Scripts and the diff: `notes-evidence-gfx90a-flake-scripts.md`.

### Results (subset = `python -m warp.tests -s autodetect -p test_async.py -k s2f_d2d -k s2c_d2d`, 192 tests per run, every run "Ran 192 tests")

| condition | command | failing runs | failing tests | mean subset time |
|---|---|---|---|---|
| alone | `flake_b.sh <job>/warp <job>/flk/c1_alone 0 30` | **0 / 30** | 0 of 5,760 | 13.8 s |
| 8 `copy_flake load` (4 streams each) on the same GPU | `flake_b.sh <job>/warp <job>/flk/c2_load8 8 160` (05:52-07:02 UTC; 8/8 load processes alive at the end) | **0 / 160** | 0 of 30,720 | 25.1 s |
| beside the full default suite on the same GPU | `flake_a.sh <job>/warp <job>/flk/c3_suite 60` (07:02-07:25 UTC) | **0 / 60** | 0 of 11,520 | 20.0 s |
| supplementary, no Warp: `copy_flake check` with 8 `copy_flake load` (4 streams) on the same GPU | `copy_flake check 2000`, `copy_flake check 2000 pool` | 0 wrong of 2,000 + 2,000 iterations | | |

The suite beside the subset (suite 1, GPU 0): **Ran 7559 tests in 1091.5 s, OK (skipped=110)**, its own `test_async` included (on gfx942 the same suite run hit the defect twice). Suite 2 was started when suite 1 ended and killed when the 60 subset runs were done. The 7559 tests and 110 skips against gfx950's 7557 and 81 were not examined (not this session's question).

The load was heavier relative to the device than on gfx942 (subset 25.1 s under load against 13.8 s alone, 1.8x; gfx942 11.6 s against 8.0 s, 1.45x), so the zero is not a lighter load. Against gfx942's rates, 0/160 has probability (1 - 17/160)^160 = 1.6e-8 and 0/60 has (1 - 18/60)^60 = 5e-10 if the rate were the same here.

Step 4 of the request (`copy_flake check 2000` under load and the load on another GPU) applied only if it reproduced; it did not, so no other GPU was used. The `copy_flake check` row above is extra, on GPU 0 only.

### Every failure

None.

### For Jeff (no decision taken here)

- With gfx942 (8 XCDs) failing and gfx90a (one die, one L2) clean under the same subset, load and suite, the defect follows the multi-XCD devices: consistent with session 21's reading of one XCD's L2 lines for the buffer being lost or written back late. It does not rule out a gfx94x/gfx950 code-generation or runtime difference (gfx90a takes other code paths in both); a gfx942 run in CPX mode (one XCD per device) would separate the two on the same chip, if that host can be repartitioned (a host-wide change, not mine to make).
- gfx1100 (one XCD, wave32) was not run here.

## Evidence session 2 (2026-11-20, linux-gfx942): narrowing the copy flake

Requested by Jeff Daily, same evidence-only job as the section above (lock with linux-gfx950, no state change, nothing pushed to the fork; `HIP_VISIBLE_DEVICES=4` kept, `ulimit -c 0` on every GPU run; background processes started and stopped by recorded PID, never by pattern). Same clone `agent_space/w` at `7059a340`, same venv (ROCm 10.2, HIP runtime 7.17), same driver (amdgpu 6.16.13). Host module parameters, for the record: `mtype_local=0` (MTYPE_RW for local VRAM), `cwsr_enable=1`, `noretry=1`, `sched_policy=0` (HWS), `mes=0`.

**Result: a Warp-free reproducer exists.** `copy_flake2.hip` (below), about 120 lines of plain HIP following Warp's per-iteration call sequence, fails 8 times in 40,000 iterations under 8 `copy_flake load` processes, with the same signature: exactly one mod-8 class of 256-thread blocks (all 488 or 489 of them, 124,928 or 125,184 floats) left as zeros for good. It needs both (a) the destination allocated with `hipMalloc` at the address of a buffer that was `hipFree`d just before (0/40,000 when the destination is allocated before the free, so the address is not reused), and (b) other processes working on the same GPU (0/40,000 alone, 0/40,000 with one load process). Registered as deferred item `hip-mi300-xcd-lost-writes-after-free-realloc` (rocm-bug-report, not drafted, not filed).

### Step 1: the call sequence

`AMD_LOG_LEVEL=3 AMD_LOG_MASK=1` on 4 iterations of `loop.py` (the test's `s2c_d2d` body: source pool off, destination pool off, Warp's default stream, no graph), log `agent_space/e2/logs/trace3.log`. The per-iteration sequence (S = Warp's device stream from `hipStreamCreateWithPriority(&s, 0, 0)`, e = `hipEventCreateWithFlags(.., hipEventDisableTiming)`):

```
hipMalloc(4000000)                     -> A = 0x...1fa00000         (wp.array(data=src): contiguous copy of the source)
hipMemcpyAsync(A <- pageable, 4000000, H2D, S)
hipMalloc(8000000)                     -> Y = 0x...1de00000         (wp.zeros with stride 8 B: the strided source)
hipLaunchKernel(fill kernel,  {3907}, {256}, S)                     (zero every 2nd float of Y)
hipLaunchKernel(array_copy_1d_kernel, {3907}, {256}, S)             (A -> Y, dst stride 8)
hipFree(A)                                                          <- no synchronize before it; hipFree synchronizes implicitly
hipMalloc(4000000)                     -> D = 0x...1fa00000 = A     (the destination, pool off)
hipMemcpyAsync(D <- pageable zeros, 4000000, H2D, S)
hipDeviceSynchronize()
hipLaunchKernel(array_copy_1d_kernel, {3907}, {256}, S)             (Y -> D, src stride 8: the copy under test)
hipStreamSynchronize(S)
hipEventRecord(e, S); hipStreamWaitEvent(null stream, e)
hipMemcpyAsync(pageable <- D, 4000000, D2H, null stream)            (dst.numpy(); returns after the copy)
hipFree(D); hipFree(Y)
```

So the destination always gets the address the source's temporary `A` had a moment earlier: `A` was written by an H2D copy, read by every XCD in the `A -> Y` kernel, then freed (with that kernel possibly still in flight; `hipFree` waits). The subset's `copy_template` makes the same calls in the same order (`as_strided_array` builds `A` and `Y`, frees `A`, then `dst_ctor` allocates the destination). Session 21's `copy_flake.hip` allocated its destination before freeing its `a`, so its destination never reused an address, which is why it never failed. A trace of a *failing* subset run was not taken: the loop and the reproducer fail with the same signature, so no further trace was needed.

### Step 2: narrowing inside Warp (`loop.py`, under 8 `copy_flake load` processes, 4 streams each)

`loop.py` is session 21's `s2c_loop.py` with switches that remove one ingredient each, plus a fuller diagnostic. Every iteration hipMallocs and frees as above. Session 21's `s2c_loop.py` itself: 5 of 20,000 wrong with the source pool off, 0 of 20,000 with it on.

| variant (`ablate.sh 40000 <label> <option>`) | wrong iterations |
|---|---|
| baseline (4 runs: 30,000, 30,000, 30,000, 40,000) | 6, 4, 1, 5 = **16 / 130,000** (about 1 in 8,000) |
| `keep`: no free/realloc of `dst` and the strided source between iterations | **0 / 40,000** |
| `pool_dst`: destination from the mempool (`hipMallocAsync`) | **0 / 40,000** |
| `pool_src`: source temporaries from the mempool, destination still `hipMalloc` | **0 / 40,000** (and 0 / 20,000 with `s2c_loop.py`) |
| `kernel_zero`: destination zero-filled by `wp.zeros` (Warp's memset kernel) instead of the pageable H2D copy | 10 / 40,000 |
| `sync_before_copy`: an extra `wp.synchronize_device()` right before the copy | 1 / 40,000 |
| `rtc`: the copy by a hipRTC-compiled Warp kernel instead of the library's `array_copy_1d_kernel` | 5 / 40,000 |

At the baseline rate a variant would show about 5 failures in 40,000, so 0 means a chance of about 0.4 % that the ingredient is not needed (0.02 % for `pool_src` over 60,000). Needed: the free and re-`hipMalloc` of the destination, at the address the source temporaries just used. Not needed: how the destination is zeroed, an extra synchronize, which kernel (library or hipRTC) does the copy.

**What the XCDs see.** On a failure `loop.py` re-reads, synchronizes, and runs device-side checks in which block b reads block b + s (s = 0..7). Since blocks are dealt round robin to the 8 XCDs, each shift puts the reader of class c on a different XCD. Logs `agent_space/e2/logs/abl_base*.log`, `abl_kernel_zero.log`:
- After `hipDeviceSynchronize()` the XCDs disagree about the same memory. In the first failures (`loop_base.log`, iterations 5141, 12450, 19016, 20024, 23530) a reader with shift 0 (the XCD that wrote the class) counted 0 wrong, while every other shift counted that class fully wrong. With shift 1, the writing XCD reading the *next* class also saw part of it wrong (58,944 to 100,352 floats). `copy_flake2` shows the same (for example iteration 6729 of `cf2_load8.log`: shift 2 sees class 4 correct, shifts 0, 1 and 3-7 see it wrong).
- The wrong count can grow after the copy has finished: `abl_base3.log` iteration 6403 read 12,352 wrong, then 56,128 on the second read, then 124,928 (the whole class) after a device synchronize and 1 s. So the correct data first reached memory and was later overwritten with zeros, in exactly that XCD's blocks. Conversely, the second partial class seen in many first reads clears on a re-read.
- By value (a gather kernel), the bad class reads as zeros, never as the previous iteration's values or anything else.

Reading (hypothesis): one XCD's cached copy of the reused range is out of step with memory after the free/re-allocation, either stale lines in that XCD's L2 for the old buffer written back late, or a stale per-XCD translation. Either is below the HIP API. The two session-1 failures that lost only the class blocks after block 2048 (= 2 MiB into a 2 MiB-aligned buffer) fit a per-2 MiB-page effect. In every `copy_flake2` failure the destination is 2 MiB-aligned (4,000,000 B spans two 2 MiB pages) and the whole class was lost across both pages, so 2 MiB alignment is common but no boundary condition was isolated.

### Step 3: the Warp-free reproducer

`copy_flake2.hip` follows the trace exactly: Warp's `array_copy_1d_kernel` body (memcpy of `elem_size` bytes through `size_t` strides), 3907 x 256 launches with `<<<>>>` (= `hipLaunchKernel`, as Warp's library launches its kernels; the hipRTC module path is not needed, see `rtc` above), a fill kernel for the strided source, the same stream and event flags, the same order of frees and allocations, the pageable H2D and D2H copies, `hipDeviceSynchronize` before the copy and the event wait before the read-back.

```
// copy_flake2.hip
// copy_flake2: Warp's per-iteration HIP call sequence for test_async's s2c_d2d copy (from an AMD_LOG_LEVEL=3 trace of
// warp 1.14 at 7059a340), with Warp's array_copy_1d_kernel body, 3907 x 256 launches and its stream setup:
//   A = hipMalloc(n*4); hipMemcpyAsync(A <- pageable src, H2D, S)
//   Y = hipMalloc(2n*4); fill kernel (zero every 2nd float of Y) on S; copy kernel A -> Y (stride 8 B) on S
//   hipFree(A)                                   <- no explicit sync: relies on hipFree's implicit synchronization
//   D = hipMalloc(n*4)  (normally returns A's address); hipMemcpyAsync(D <- pageable zeros, H2D, S)
//   hipDeviceSynchronize(); copy kernel Y -> D on S; hipStreamSynchronize(S)
//   hipEventRecord(e, S); hipStreamWaitEvent(null, e); hipMemcpyAsync(pageable out <- D, D2H, null); check out
//   hipFree(D); hipFree(Y)
// S = hipStreamCreateWithPriority(0, 0), e = hipEventCreateWithFlags(hipEventDisableTiming).
// Usage: copy_flake2 <iterations> [noreuse] [syncfree] [memsetfill]
//   noreuse: allocate D before freeing A, so D never gets A's address; syncfree: hipDeviceSynchronize() before hipFree(A);
//   memsetfill: zero D with hipMemsetAsync on S instead of the pageable H2D copy
// On a mismatch: wrong count per (block % 8) for the first read, a second read, a read after hipDeviceSynchronize,
// and a device-side count by a kernel whose block b reads block b + s (s = 0..7).
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { printf("%s:%d %s -> %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e_)); exit(2); } } while (0)

static __global__ void array_copy_1d_kernel(void* dst, const void* src, size_t dst_stride, size_t src_stride,
                                            const int* dst_indices, const int* src_indices, size_t n, size_t elem_size)
{
    size_t i = size_t(blockIdx.x) * size_t(blockDim.x) + size_t(threadIdx.x);
    if (i < n) {
        size_t src_idx = src_indices ? src_indices[i] : i;
        size_t dst_idx = dst_indices ? dst_indices[i] : i;
        const char* p = (const char*)src + src_idx * src_stride;
        char* q = (char*)dst + dst_idx * dst_stride;
        memcpy(q, p, elem_size);
    }
}

static __global__ void fill_strided(float* dst, size_t stride_elems, size_t n)
{
    size_t i = size_t(blockIdx.x) * size_t(blockDim.x) + size_t(threadIdx.x);
    if (i < n)
        dst[i * stride_elems] = 0.0f;
}

static __global__ void count_wrong(const float* d, float off, int shift, int n, int* wrong)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int j = (i + shift * 256) % n;
        if (d[j] != off + float(j))
            atomicAdd(&wrong[(j / 256) % 8], 1);
    }
}

static void classes(const float* x, const float* ref, int n, int* cnt, int* tot)
{
    memset(cnt, 0, 8 * sizeof(int)); *tot = 0;
    for (int i = 0; i < n; ++i) if (x[i] != ref[i]) { cnt[(i / 256) % 8]++; (*tot)++; }
}

int main(int argc, char** argv)
{
    const int n = 1000000; const int blocks = (n + 255) / 256;
    int iters = argc > 1 ? atoi(argv[1]) : 1000;
    bool noreuse = false, syncfree = false, memsetfill = false;
    for (int a = 2; a < argc; ++a) {
        if (!strcmp(argv[a], "noreuse")) noreuse = true;
        else if (!strcmp(argv[a], "syncfree")) syncfree = true;      // hipDeviceSynchronize() before hipFree(A)
        else if (!strcmp(argv[a], "memsetfill")) memsetfill = true;  // zero D with hipMemsetAsync instead of a pageable H2D copy
    }
    hipStream_t S; CK(hipStreamCreateWithPriority(&S, 0, 0));
    hipEvent_t ev; CK(hipEventCreateWithFlags(&ev, hipEventDisableTiming));
    int* wrong; CK(hipMalloc(&wrong, 8 * sizeof(int)));
    int bad = 0, reused = 0;
    for (int it = 0; it < iters; ++it) {
        float off = float(it * 7 % 1000);
        float* src = (float*)malloc(n * 4); for (int i = 0; i < n; ++i) src[i] = off + float(i);
        float* zeros = (float*)calloc(n, 4);
        float *A, *Y, *D;
        CK(hipMalloc(&A, n * 4));
        CK(hipMemcpyAsync(A, src, n * 4, hipMemcpyHostToDevice, S));
        CK(hipMalloc(&Y, 2 * (size_t)n * 4));
        fill_strided<<<blocks, 256, 0, S>>>(Y, 2, n);
        array_copy_1d_kernel<<<blocks, 256, 0, S>>>(Y, A, 8, 4, nullptr, nullptr, n, 4);
        if (syncfree) CK(hipDeviceSynchronize());
        if (noreuse) { CK(hipMalloc(&D, n * 4)); CK(hipFree(A)); }
        else { CK(hipFree(A)); CK(hipMalloc(&D, n * 4)); }
        if (D == A) ++reused;
        if (memsetfill) CK(hipMemsetAsync(D, 0, n * 4, S));
        else CK(hipMemcpyAsync(D, zeros, n * 4, hipMemcpyHostToDevice, S));
        CK(hipDeviceSynchronize());
        array_copy_1d_kernel<<<blocks, 256, 0, S>>>(D, Y, 4, 8, nullptr, nullptr, n, 4);
        CK(hipStreamSynchronize(S));
        float* out = (float*)malloc(n * 4);
        CK(hipEventRecord(ev, S)); CK(hipStreamWaitEvent(nullptr, ev, 0));
        CK(hipMemcpyAsync(out, D, n * 4, hipMemcpyDeviceToHost, nullptr));
        int c1[8], t1; classes(out, src, n, c1, &t1);
        if (t1) {
            ++bad;
            int c2[8], t2, c3[8], t3, c4[8], t4;
            CK(hipMemcpy(out, D, n * 4, hipMemcpyDeviceToHost)); classes(out, src, n, c2, &t2);
            CK(hipDeviceSynchronize());
            CK(hipMemcpy(out, D, n * 4, hipMemcpyDeviceToHost)); classes(out, src, n, c3, &t3);
            int dv[8][8];
            for (int s = 0; s < 8; ++s) {
                CK(hipMemset(wrong, 0, 8 * sizeof(int)));
                count_wrong<<<blocks, 256>>>(D, off, s, n, wrong);
                CK(hipMemcpy(dv[s], wrong, 8 * sizeof(int), hipMemcpyDeviceToHost));
            }
            usleep(1000000);
            CK(hipMemcpy(out, D, n * 4, hipMemcpyDeviceToHost)); classes(out, src, n, c4, &t4);
            printf("iter %d D=%p A-reuse=%d: read1 %d [%d %d %d %d %d %d %d %d] | read2 %d [%d %d %d %d %d %d %d %d] | after sync %d [%d %d %d %d %d %d %d %d] | after 1 s %d [%d %d %d %d %d %d %d %d]\n",
                   it, (void*)D, D == A, t1, c1[0], c1[1], c1[2], c1[3], c1[4], c1[5], c1[6], c1[7], t2, c2[0], c2[1], c2[2], c2[3], c2[4], c2[5], c2[6], c2[7],
                   t3, c3[0], c3[1], c3[2], c3[3], c3[4], c3[5], c3[6], c3[7], t4, c4[0], c4[1], c4[2], c4[3], c4[4], c4[5], c4[6], c4[7]);
            for (int s = 0; s < 8; ++s)
                printf("    device check, block b reads block b+%d: wrong per class %d %d %d %d %d %d %d %d\n", s, dv[s][0], dv[s][1], dv[s][2], dv[s][3], dv[s][4], dv[s][5], dv[s][6], dv[s][7]);
            fflush(stdout);
        }
        free(out); free(src); free(zeros);
        CK(hipFree(D)); CK(hipFree(Y));
        if ((it + 1) % 5000 == 0) { printf("progress %d iterations, %d wrong\n", it + 1, bad); fflush(stdout); }
    }
    printf("%d of %d iterations wrong%s%s%s (D got A's address in %d)\n", bad, iters, noreuse ? " (noreuse)" : "", syncfree ? " (syncfree)" : "", memsetfill ? " (memsetfill)" : "", reused);
    return bad ? 1 : 0;
}
```
Build and run: `hipcc --offload-arch=gfx942 -O2 -o copy_flake2 copy_flake2.hip`; load: 8 x `copy_flake load 7200 4` (session 21's `copy_flake.hip`, in `notes-session21-scripts.md`) on the same GPU; `copy_flake2 40000`.

### Steps 3 and 4: the reproducer by condition and load kind (HIP_VISIBLE_DEVICES=4; load processes on the same GPU)

Logs `agent_space/e2/logs/cf2_*.log`, `cf2c_*.log`; drivers `cf2_batch.sh`, `cf2_batch3.sh`; load kinds from `loadkind.hip` (`alloc`: `hipMalloc`/`hipMemset`/`hipFree` of 16-64 MiB in a loop; `kernel`: a compute kernel on 4 streams over resident buffers; `copy`: D2D plus D2H `hipMemcpyAsync` on 4 streams, no kernels).

| `copy_flake2` variant | load on GPU 4 | wrong iterations |
|---|---|---|
| default | 8 x `copy_flake load` (kernels + copies, 4 streams) | **8 / 40,000**; again 10 / 40,000 (eviction-sampling build) |
| `noreuse` (destination allocated before the free) | 8 x `copy_flake load` | **0 / 40,000** |
| default | none | **0 / 40,000** |
| default | 1 x `copy_flake load` | **0 / 40,000** |
| default | 8 x `loadkind kernel` | 3 / 40,000 |
| default | 8 x `loadkind copy` | 7 / 40,000 |
| default | 8 x `loadkind alloc` | **32 in about 5,800** (1 in ~180; killed at the 1,500 s timeout, so the summary line was lost; last failing iteration 5792) |
| `syncfree` (`hipDeviceSynchronize()` before `hipFree(A)`) | 8 x `copy_flake load` | 8 / 40,000 |
| `memsetfill` (`hipMemsetAsync` zero fill instead of the pageable H2D) | 8 x `copy_flake load` | 15 / 40,000 |

Every one of the 83 `copy_flake2` failures lost exactly one whole class (125,184 or 124,928 floats), always at a 2 MiB-aligned destination. Within one run the lost class repeats (8 `alloc`: class 1 23 times, class 0 9 times; 8 `copy_flake load`: class 4 five times, 1 twice, 7 once), consistent with particular XCDs.

The load matters by amount and kind: one process of any kind I tried does not trigger it; 8 processes doing only kernels or only copies do; 8 processes that only allocate and free memory raise the rate about 25-fold. Caution for whoever repeats this on a shared host: under 8 `alloc` processes the kernel spent its time clearing VRAM (`amdgpu_fill_buffer` in the load processes' wait channel), `amd-smi` timed out ("amdsmi_init() timed out after 60s. The GPU driver may be unresponsive") and an exiting `copy_flake2e` sat in `D` state in `dma_fence_wait_any_timeout` for minutes. It recovered by itself once those processes were stopped (amd-smi and a 500-iteration `copy_flake2` run normal afterwards). The remaining `alloc` variants were not run for that reason.

### Evictions and the 2 MiB question

- `copy_flake2e` (the reproducer plus per-iteration sampling of this process's `/sys/class/kfd/kfd/proc/<pid>/stats_*/evicted_ms`, summed over GPUs, before the first allocation and after the read-back of each iteration) under 8 `copy_flake load`: **10 failures, none in an iteration where `evicted_ms` moved** (39 -> 39 each time; the 39 ms are from start-up). In the whole run of 40,000 iterations only 1 iteration (a passing one) saw an eviction (22 ms in total). So the failures do not coincide with a queue eviction of the failing process at the 1 ms resolution of that counter. Preemption of its queues by the hardware scheduler without an eviction is not visible in this counter and was not measured. The subset-level `amd-smi process` sampling was superseded by this per-iteration counter.
- 2 MiB: in every `copy_flake2` and `loop.py` failure the destination was 2 MiB-aligned and 4,000,000 bytes (two 2 MiB pages, the second partly used); every failure there lost the class across both pages. Session 1's two partial failures (class blocks from block 2048 on) are the only page-shaped cases. Alignment is therefore always present, but a boundary dependence was not shown.

### Could the port cause this?

No, from this evidence it is below the HIP API (runtime, KFD/driver, firmware or hardware), not in the port:
- `copy_flake2` contains no Warp code: plain HIP calls, the same signature, the same dependence on address reuse and on a second process.
- The call sequence is legal. `hipFree` synchronizes implicitly, and an explicit `hipDeviceSynchronize()` before it (`syncfree`) changes nothing (8 / 40,000). The copy is followed by `hipStreamSynchronize` and, for the read-back, an event wait; a `hipDeviceSynchronize()` before the copy (`sync_before_copy` in Warp) does not help.
- After `hipDeviceSynchronize()` different XCDs read different values from the same device address, and the visible content changes later without any further API call. No ordering of HIP calls can produce that.
- The session 20 free-path change is not involved: at `7059a340` `wp_free_device_default` is a plain `hipFree` outside graph captures, the reproducer uses neither Warp nor that path, and the session 19 failure predates the change.

What the port can do (no decision taken, Jeff's call): Warp's default (mempool enabled on this device) allocates through `hipMallocAsync`, which did not fail (`pool_dst` 0 / 40,000), so the exposure is code that turns the mempool off, as `test_async`'s `DstPoolOff` cases do. A workaround inside Warp, for example never returning a freshly freed address to `hipMalloc`, would mask a runtime defect rather than fix it.

### For Jeff (no decision taken here)

- Deferred item `hip-mi300-xcd-lost-writes-after-free-realloc` (rocm-bug-report) is registered with the reproducer as its reference; nothing drafted or filed.
- Open: whether gfx950 fails `copy_flake2` the same way (expected, not run); whether a single-XCD part (gfx90a, clean in 250 subset runs) is clean under `copy_flake2` and the 8-process load; whether the effect is L2 write-back or translation (a test would be to make the freed range's last reader one XCD only).

## Evidence session (2026-11-20, linux-gfx950): copy_flake2

Requested by Jeff Daily; evidence only (same fleet job as porting sessions 20-21; no state change, nothing pushed to the fork). `copy_flake2.hip` taken verbatim from "Evidence session 2 (2026-11-20, linux-gfx942)" (md5 of the extracted file ab7bc9af7ddb1f35fede7eb8f9bb4a1f), built for this GPU: `clang++ -x hip --rocm-path=<ROCm 10.2 root> --offload-arch=gfx950 -O2 -std=c++17 copy_flake2.hip -o copy_flake2 -lamdhip64` (ROCm 10.2 from the job venv, HIP runtime 7.17). Load: `copy_flake load 7200 4` (session 21's `copy_flake.hip`, the binary built in session 21). Driver script `agent_space/cf2/run_cf2.sh <label> <loads> <iterations> [option]` starts the load processes, keeps their PIDs, runs `copy_flake2`, and kills only those PIDs. `HIP_VISIBLE_DEVICES=1` (dispatcher pin), `ulimit -c 0`. The `alloc` load kind was not run.

| run | load on the same GPU | iterations | wrong | time |
|---|---|---|---|---|
| default, 1 | 8 x `copy_flake load` | 40,000 | **10** | 146 s |
| default, 2 | 8 x `copy_flake load` | 40,000 | **9** | 142 s |
| `noreuse` | 8 x `copy_flake load` | 40,000 | **0** | 151 s |
| default | none | 40,000 | **0** | 191 s |

So gfx950 (MI355X, 8 XCDs) reproduces it at about the gfx942 rate (19 / 80,000 here, 8 and 10 / 40,000 there), with the same two conditions: the destination at the just-freed source's address (0 with `noreuse`) and other processes on the GPU (0 without load). The failures have the gfx942 signature, one whole mod-8 block class zero and staying zero (first four of run 1):

```
iter 4566 D=0x7feecaa00000 A-reuse=1: read1 125184 [0 125184 0 0 0 0 0 0] | read2 125184 [0 125184 0 0 0 0 0 0] | after sync 125184 [0 125184 0 0 0 0 0 0] | after 1 s 125184 [0 125184 0 0 0 0 0 0]
iter 12033 D=0x7feecaa00000 A-reuse=1: read1 125184 [0 125184 0 0 0 0 0 0] | read2 125184 [0 125184 0 0 0 0 0 0] | after sync 125184 [0 125184 0 0 0 0 0 0] | after 1 s 125184 [0 125184 0 0 0 0 0 0]
iter 17278 D=0x7feecaa00000 A-reuse=1: read1 125184 [0 125184 0 0 0 0 0 0] | read2 125184 [0 125184 0 0 0 0 0 0] | after sync 125184 [0 125184 0 0 0 0 0 0] | after 1 s 125184 [0 125184 0 0 0 0 0 0]
iter 19748 D=0x7feecaa00000 A-reuse=1: read1 118464 [0 118464 0 0 0 0 0 0] | read2 118464 [0 118464 0 0 0 0 0 0] | after sync 118464 [0 118464 0 0 0 0 0 0] | after 1 s 125184 [0 125184 0 0 0 0 0 0]
iter 20631 D=0x7feecaa00000 A-reuse=1: read1 124928 [0 0 0 0 124928 0 0 0] | read2 124928 [0 0 0 0 124928 0 0 0] | after sync 124928 [0 0 0 0 124928 0 0 0] | after 1 s 124928 [0 0 0 0 124928 0 0 0]
iter 22023 D=0x7feecaa00000 A-reuse=1: read1 124928 [0 0 0 0 0 0 0 124928] | read2 124928 [0 0 0 0 0 0 0 124928] | after sync 124928 [0 0 0 0 0 0 0 124928] | after 1 s 124928 [0 0 0 0 0 0 0 124928]
iter 27111 D=0x7feecaa00000 A-reuse=1: read1 115712 [0 0 0 0 0 0 115712 0] | read2 115712 [0 0 0 0 0 0 115712 0] | after sync 115712 [0 0 0 0 0 0 115712 0] | after 1 s 124928 [0 0 0 0 0 0 124928 0]
iter 31294 D=0x7feecaa00000 A-reuse=1: read1 125184 [0 125184 0 0 0 0 0 0] | read2 125184 [0 125184 0 0 0 0 0 0] | after sync 125184 [0 125184 0 0 0 0 0 0] | after 1 s 125184 [0 125184 0 0 0 0 0 0]
iter 39476 D=0x7feecaa00000 A-reuse=1: read1 124928 [0 0 0 0 0 0 0 124928] | read2 124928 [0 0 0 0 0 0 0 124928] | after sync 124928 [0 0 0 0 0 0 0 124928] | after 1 s 124928 [0 0 0 0 0 0 0 124928]
iter 244 D=0x7fbca3a00000 A-reuse=1: read1 125184 [0 125184 0 0 0 0 0 0] | read2 125184 [0 125184 0 0 0 0 0 0] | after sync 125184 [0 125184 0 0 0 0 0 0] | after 1 s 125184 [0 125184 0 0 0 0 0 0]
iter 6446 D=0x7fbca3a00000 A-reuse=1: read1 124928 [0 0 0 0 124928 0 0 0] | read2 124928 [0 0 0 0 124928 0 0 0] | after sync 124928 [0 0 0 0 124928 0 0 0] | after 1 s 124928 [0 0 0 0 124928 0 0 0]
iter 15410 D=0x7fbca3a00000 A-reuse=1: read1 118272 [0 0 0 0 0 0 118272 0] | read2 118272 [0 0 0 0 0 0 118272 0] | after sync 118272 [0 0 0 0 0 0 118272 0] | after 1 s 124928 [0 0 0 0 0 0 124928 0]
iter 24430 D=0x7fbca3a00000 A-reuse=1: read1 125184 [0 125184 0 0 0 0 0 0] | read2 125184 [0 125184 0 0 0 0 0 0] | after sync 125184 [0 125184 0 0 0 0 0 0] | after 1 s 125184 [0 125184 0 0 0 0 0 0]
iter 24873 D=0x7fbca3a00000 A-reuse=1: read1 118272 [0 0 0 0 0 0 118272 0] | read2 118272 [0 0 0 0 0 0 118272 0] | after sync 118272 [0 0 0 0 0 0 118272 0] | after 1 s 124928 [0 0 0 0 0 0 124928 0]
iter 26111 D=0x7fbca3a00000 A-reuse=1: read1 124928 [0 0 0 0 124928 0 0 0] | read2 124928 [0 0 0 0 124928 0 0 0] | after sync 124928 [0 0 0 0 124928 0 0 0] | after 1 s 124928 [0 0 0 0 124928 0 0 0]
iter 26820 D=0x7fbca3a00000 A-reuse=1: read1 124928 [0 0 0 0 124928 0 0 0] | read2 124928 [0 0 0 0 124928 0 0 0] | after sync 124928 [0 0 0 0 124928 0 0 0] | after 1 s 124928 [0 0 0 0 124928 0 0 0]
iter 30993 D=0x7fbca3a00000 A-reuse=1: read1 124928 [0 0 0 0 0 0 0 124928] | read2 124928 [0 0 0 0 0 0 0 124928] | after sync 124928 [0 0 0 0 0 0 0 124928] | after 1 s 124928 [0 0 0 0 0 0 0 124928]
iter 31478 D=0x7fbca3a00000 A-reuse=1: read1 124928 [0 0 0 0 124928 0 0 0] | read2 124928 [0 0 0 0 124928 0 0 0] | after sync 124928 [0 0 0 0 124928 0 0 0] | after 1 s 124928 [0 0 0 0 124928 0 0 0]
iter 37592 D=0x7fbca3a00000 A-reuse=1: read1 125184 [0 125184 0 0 0 0 0 0] | read2 125184 [0 125184 0 0 0 0 0 0] | after sync 125184 [0 125184 0 0 0 0 0 0] | after 1 s 125184 [0 125184 0 0 0 0 0 0]
```

(the third shows the wrong count growing after the copy, 118,272 -> 124,928 a second later, as on gfx942).

**What else was on the GPU, and a correction to sessions 20-22.** amd-smi does not number GPUs in HIP's order: under `HIP_VISIBLE_DEVICES=1` HIP's device is PCI `0000:05:00.0` (`hipDeviceGetPCIBusId`), which is **amd-smi GPU 0**; amd-smi GPU 1 is `0000:15:00.0`, a different GPU. Every `amd-smi ... -g 1` check in porting sessions 20-22 (the GPU monitor, the drive.sh idle checks, and the "outside process on GPU 1, 13-28 GB" of session 21) looked at that other GPU, not at this job's. The four runs above also recorded `amd-smi process -g 1` (before, 3 s in, 120 s in, after: "No running processes detected" every time), i.e. the wrong GPU. Checked on the right index right after the runs: `amd-smi process -g 0` idle = "No running processes detected", GFX_ACTIVITY 0 %; with 8 `copy_flake load` processes started (PIDs 3073110-3073117, killed after) it listed exactly 8 processes of 2.2 GB each and GFX_ACTIVITY 100 %; after killing them, again none. So nothing outside this job was on its GPU before or after these runs; during the runs it was not observed on the right index. Session 21's statement that "no run was truly alone on the device" is withdrawn: the outside process was on amd-smi GPU 1, another GPU. (Sessions 20-21's timing noise therefore cannot be blamed on a second process on this GPU; board-level load from other GPUs remains possible.)

## Evidence session (2026-11-20, linux-gfx90a): copy_flake2

Requested by Jeff Daily, same evidence-only job as the gfx90a section above (lock with linux-gfx950, no state change, nothing pushed to the fork, no GitHub writes; `HIP_VISIBLE_DEVICES=0` kept, `ulimit -c 0` on every GPU run; load processes started by the driver script and stopped by their recorded PIDs only; the `alloc` load kind was not run). Same host and stack: AMD INSTINCT MI250 (gfx90a:sramecc+:xnack-, one GCD per device, 104 CUs), kernel 6.8.0-101-generic, amdgpu 6.16.6, job venv ROCm 10.2 (`rocm-sdk-devel` 10.2.0, HIP runtime 7.17).

**Result: `copy_flake2` does not fail on gfx90a. 0 wrong in 160,000 iterations**, including 0 in 80,000 under the 8-process load in which gfx942 fails 8-10 times in 40,000. The destination got the freed source's address in every iteration of the default runs, so the address-reuse condition was met throughout.

- Source: `copy_flake2.hip` copied verbatim from "Evidence session 2 (2026-11-20, linux-gfx942)" above (123 lines). Build: `$(python -m rocm_sdk path --root)/bin/hipcc --offload-arch=gfx90a -O2 -o copy_flake2 copy_flake2.hip`; a 500-iteration smoke run alone: 0 wrong, D got A's address in 500.
- Load: session 21's `copy_flake load 7200 4` (the gfx90a build from the section above), 8 processes on GPU 0.
- Driver `cf2_run.sh <label> <nload> <iterations> [options]`: starts `<nload>` load processes on GPU 0 and records their PIDs, waits 3 s, runs `amd-smi process -g 0`, starts `copy_flake2 <iterations> [options]` (cwd `/tmp`), runs `amd-smi process -g 0` again 60 s in, waits for it, checks the load PIDs are still alive, then kills and reaps exactly those PIDs.

```
cf2_run.sh load8_a 8 40000
cf2_run.sh load8_b 8 40000
cf2_run.sh load8_noreuse 8 40000 noreuse
cf2_run.sh alone 0 40000
```

| `copy_flake2` variant | load on GPU 0 | wrong iterations | D got A's address | wall time |
|---|---|---|---|---|
| default, run 1 | 8 x `copy_flake load` (4 streams each); 8/8 alive at the end | **0 / 40,000** | 40,000 | 350 s |
| default, run 2 | 8 x `copy_flake load`; 8/8 alive at the end | **0 / 40,000** | 40,000 | 326 s |
| `noreuse` | 8 x `copy_flake load`; 8/8 alive at the end | **0 / 40,000** | 0 | 319 s |
| default | none | **0 / 40,000** | 39,999 | 214 s |

`amd-smi process`: before the runs no process on GPU 0, 1 or 3 (10 MB VRAM, 0 % activity); GPU 2 had another job's process (329 MB, 99 % activity), not touched and not part of this measurement. In each load run, at start 8 processes at 1.1 GB each (the load), and 60 s in those 8 plus one at 397-401 MB (`copy_flake2`); in the run with no load, only `copy_flake2` (458 MB). amd-smi shows host-namespace PIDs. They differ from the container's PIDs by a constant offset (the 8 consecutive load PIDs map 1:1), so the counts and sizes identify the processes. Nothing was left on GPU 0 afterwards.

At gfx942's rate (about 9 in 40,000 under this load) 0 in 80,000 would have probability about e^-18 = 1.5e-8. Under load the run took 1.5-1.6x as long as without load (gfx942's load factor is not recorded in its section), so the load was real. With the 250 clean subset runs above, both the Warp test and the Warp-free reproducer are clean on this single-die device and fail on the 8-XCD gfx942: consistent with a per-XCD effect, and evidence for the deferred item `hip-mi300-xcd-lost-writes-after-free-realloc` that gfx90a is not affected (still not a proof that the cause is the XCD split rather than a gfx94x-specific runtime or firmware path; a gfx942 run in CPX mode would separate the two).

## Evidence session 3 (2026-11-20, linux-gfx942): copy_flake2 on other ROCm versions and the issue draft

Requested by Jeff Daily, same evidence-only job (no lock, no state change, no fork push, nothing posted; `HIP_VISIBLE_DEVICES=4` kept, `ulimit -c 0`, every background process started and stopped by recorded PID). Host: MI300X HF (gfx942, SPX, NPS1), kernel 6.8.0-110-generic, amdgpu 6.16.13 for every run below.

**Result: not fixed on the newest nightly.** ROCm 10.1.0 (the image's own), 10.2.0 and nightly 10.3.0a20261120 all reproduce it at similar rates. A one-file reproducer (`lost_writes.hip`, inline in the draft) starts its own load, prints one line per failure and exits 1; its controls are clean. No existing upstream issue matches; a new-issue draft for `ROCm/rocm-systems` is below, with placeholder rows for the gfx950 and gfx90a runs.

### Versions

| ROCm | `hipconfig --version` | `copy_flake2` x2 (8 `copy_flake load`) | `lost_writes` v1 (kernels fill the source) | `lost_writes` final (host fills the source) |
|---|---|---|---|---|
| 10.1.0 (image, `/opt/python-3.11-venv`) | 7.16.26385-0000000 | 11, 7 / 40,000 | 5 / 40,000 | 9, 5 / 40,000 |
| 10.2.0 (job venv) | 7.17.26434-0000000 | 8, 10 / 40,000 (session 2) | 2 and 0 / 40,000 | 10, 11 / 40,000 |
| 10.3.0a20261120 (nightly) | 7.18.26463-0000000 | 8, 3 / 40,000 | 5 / 40,000 | 7, 19 / 40,000 |

Controls with the final program on 10.2.0: `noreuse` 0 / 40,000; no load processes 0 / 40,000. The first no-load run overlapped a debug run of mine that started its own 8 load processes (my mistake), so it was repeated alone: 0 / 40,000 both times. With v1 on 10.2.0, `noreuse` and no load were also 0 / 40,000 each. Every failure of the final program on all three versions lost exactly one whole class (1, 4 or 7). Most failures also show a second zero region that is not aligned to workgroups: one contiguous 1 KiB run per 8 KiB, starting 240 bytes into a block (from a dump of one failure, `lw_dbg.hip`). Session 2's "two classes lost" reading of v1's first output was this region counted by class, and is withdrawn.

Nightly install: the fleet guard refuses `pip` for any environment except `$FLEET_VENV` (a second venv under the job directory was refused: "pip changing the Python environment every job on this host shares"), so the nightly went into `$FLEET_VENV` temporarily (`pip install --pre "rocm[devel]==10.3.0a20261120" --extra-index-url https://nightly.repo.amd.com/rocm/whl-next/`), and 10.2.0 was restored afterwards (`pip install "rocm[devel]==10.2.0" "rocm-sdk-core==10.2.0" "rocm-sdk-devel==10.2.0" --extra-index-url https://stable.repo.amd.com/rocm/whl-next/`; `python -m rocm_sdk version` = 10.2.0 at the end). The empty second venv was deleted.

### Experiments (nightly 10.3.0a20261120; `exp.hip` = `copy_flake2`'s sequence with a size and a "last reader" knob, 8 `copy_flake load` processes)

| variant | wrong iterations | note |
|---|---|---|
| 1 MiB buffers | 0 / 40,000 | destination got the freed address every time |
| exactly 2 MiB | 11 / 40,000 | destination never got the freed address |
| 4,000,000 B (baseline) | 2 / 40,000 | |
| 4 MiB | 11 / 40,000 | |
| 16 MiB | 1 / 10,000 | |
| freed buffer read by one workgroup only | 0 / 40,000 | that kernel runs much longer, so timing changes too |
| only its first 1/8 read by a kernel | 0 / 40,000 | |
| never read by a kernel (written by H2D only, strided source filled from the host) | **93 / 40,000** | became the final `lost_writes` sequence |

Reading: the condition is a fresh allocation of 2 MiB or more right after one was freed, not the virtual address as such (at 1 MiB the address is reused without failures, probably suballocated without unmapping; at 2 MiB a different address still fails). Stale cached copies from the freed buffer's last kernel reader are not the cause: the rate is highest when no kernel touches it. The one-workgroup and part-read zeros are suggestive at best given the baseline's spread (2 to 11).

Environment variables (one run each of `lost_writes` v1, 40,000 iterations, variable inherited by the 8 load processes): `HSA_XNACK=1` 0; `HSA_ENABLE_SDMA=0` 10 (no change); `GPU_MAX_HW_QUEUES=1` 0. Not repeated; at the v1 baseline (0 to 5) the two zeros are weak evidence.

### Existing reports (read-only search, `ROCm/rocm-systems`, `ROCm/ROCm` (now `ROCm/legacy-rocm-build`, archived) and `ROCm/TheRock`)

Queries on stale data, lost writes, XCD, multi-process, hipFree/hipMalloc reuse, MI300 corruption; the search API rate limit was hit once and the queries were repeated. No match. Closest:
- rocm-systems #5815 (open): MI300X, 16 processes, `hipHostMalloc`/`hipHostFree` staging reused, staging buffer reads back stale zeros in a contiguous range, a second D2H is correct. Same hardware, multi-process and allocation-reuse conditions, but host-pinned memory, contiguous, transient; here device memory, interleaved by XCD, persistent. Named as possibly related in the draft, not commented on.
- rocm-systems #15007 and TheRock #8744 (gfx1201, VMM remap at a reused virtual address uses a stale mapping): single-XCD consumer part, VMM API; not the same.

### Component and labels

`project: clr` plus `bug` (from `gh label list -R ROCm/rocm-systems`). Why clr rather than the kernel driver: the failing calls are clr's `hipFree`/`hipMalloc`, and the size threshold suggests its suballocator decides when a buffer is really unmapped. rocm-systems has no label for the amdgpu kernel driver, so clr maintainers are the right first owners and can route it on. `project: rocr-runtime` would be the alternative if Jeff prefers.

### Draft (for Jeff to review and file; nothing posted)

Repository: `ROCm/rocm-systems`. Labels: `bug`, `project: clr`.
Title: `[clr] MI300X: one XCD's kernel writes lost after hipFree and hipMalloc` (70 characters)

====
My agent came up with this analysis, but I've read and approved it.

### Problem Description

On an MI300X, a kernel that writes a buffer just obtained from `hipMalloc` sometimes leaves one eighth of its output unwritten: every 256-thread workgroup whose index is k mod 8 (for one k per failure), which is the share of the workgroups one XCD runs. Every API call returns `hipSuccess`, and the wrong data stays wrong in device memory after `hipDeviceSynchronize()`. It needs two things: the destination allocation is made right after another allocation of 2 MiB or more was freed with `hipFree`, and other processes are using the same GPU at the same time. The practical consequence is silently wrong results with no error reported.

Expected: after the copy kernel and `hipStreamSynchronize`, the destination holds the copied values (here `it % 1000 + i` at index i), on every iteration.

Actual: in roughly 1 iteration in 2,000 to 8,000 under load, all 488 or 489 workgroups of one class (block index mod 8 = k) left their part of the destination with the zeros written before the kernel ran (124,928 or 125,184 of 1,000,000 floats). Re-reading after `hipDeviceSynchronize()` returns the same wrong data. In most failures of this program there is also a second, address-shaped region of zeros that is not aligned to workgroups: in every 8 KiB of the destination one contiguous 1 KiB run, starting 240 bytes past a block boundary (the "partial" blocks in the output).

What is known about the conditions:

- Allocation reuse after `hipFree`: if the destination is allocated before the previous buffer is freed (`noreuse` below), there are no failures (0 of 40,000). With 1 MiB buffers there are no failures (0 of 40,000) even though the destination gets the freed address every time; with exactly 2 MiB buffers it fails (11 of 40,000) even though the destination did not get the freed address. So the condition seems to be a freshly made allocation of 2 MiB or more replacing a freed one, not the virtual address as such.
- A second process on the same device: 0 of 40,000 with no load processes and 0 of 40,000 with one; it fails with 8. The same 8 load processes on a different GPU of the same host gave 0 failures in a larger application test. Load processes that only run kernels, or only run copies, are each enough. A load made only of processes that allocate and free memory in a loop raised the rate further, but it also left the driver unresponsive to `amd-smi` for over a minute, so the reproducer does not use it.
- One XCD's share: the lost elements are always whole 256-element blocks of one class (block index mod 8), which matches round-robin placement of workgroups across the 8 XCDs.
- The XCDs disagree about the data: after `hipDeviceSynchronize()`, a check kernel whose workgroups run on the XCD that did the writing can read the correct values, while workgroups on the other XCDs and host copies read the stale zeros. The wrong count can also grow after the copy has finished (for example 12,352 wrong on the first read, 124,928 a second later), so the correct values can reach memory and then be overwritten by the stale ones.
- Not involved: whether the freed buffer was last read by a kernel (the rate is highest, 93 of 40,000, when no kernel touches it at all), how the destination is zeroed (`hipMemsetAsync`, a kernel or a pageable H2D copy), an extra `hipDeviceSynchronize()` before the `hipFree` or before the copy, and `HSA_ENABLE_SDMA=0` (10 of 40,000).
- Allocations from the stream-ordered pool (`hipMallocAsync`) did not fail in the application where this was found (0 of 40,000).
- With `HSA_XNACK=1` or `GPU_MAX_HW_QUEUES=1` (set for all processes) one run each gave 0 of 40,000, against 0 to 11 per 40,000 without; not repeated.
- The failing process's queues were not evicted during the failing iterations (its `evicted_ms` counter in sysfs did not change in any of 10 failing iterations).
- Seen only on 8-XCD devices so far: an MI250X (one GCD per device) was clean in an application-level test; the MI355X and MI250X rows of the table come from separate runs of this reproducer.

The cause is not known. The evidence points below the HIP API (the call sequence is legal and the XCDs end up with different views of the same memory after a device synchronize), in the handling of a newly mapped allocation by one XCD while other processes are active, but whether that is in the runtime's allocator, the kernel driver, or firmware is not established.

Possibly related: #5815 (MI300X, multiple processes, reused staging allocations read back as zeros). That report's wrong data is in host-pinned staging memory, contiguous and correct on a second read; here it is device memory, interleaved by XCD, and stays wrong.

Results (40,000 iterations per run, 8 load processes started by the program):

| GPU | ROCm (HIP runtime) | wrong iterations per run |
|---|---|---|
| MI300X (gfx942) | 10.1.0 (7.16.26385) | 9, 5 |
| MI300X (gfx942) | 10.2.0 (7.17.26434) | 10, 11 |
| MI300X (gfx942) | 10.3.0a20261120 nightly (7.18.26463) | 7, 19 |
| MI300X (gfx942) | 10.2.0, `noreuse` (control) | 0 |
| MI300X (gfx942) | 10.2.0, no load processes (control) | 0 |
| MI355X (gfx950) | [PLACEHOLDER: gfx950 run pending] | [PLACEHOLDER] |
| MI250X (gfx90a) | [PLACEHOLDER: gfx90a run pending] | [PLACEHOLDER] |

Every failure in the MI300X runs above lost exactly one whole class (block index mod 8 = 1, 4 or 7).

Earlier, longer versions of the same call sequence (with kernels instead of host copies filling the source buffers) gave 11, 7 and 5 (10.1.0), 8, 10, 2 and 0 (10.2.0) and 8, 3 and 5 (10.3.0a20261120) per 40,000 iterations on the same machine.

Sample output (ROCm 10.2.0):

```
iteration  wrong        after sync   whole blocks wrong after sync, class (block % 8):count
147        249920       249920       4:488 (+1041 partial)
1629       249920       249920       4:488 (+1041 partial)
5861       250192       250192       1:489 (+1042 partial)
15053      249920       249920       4:488 (+1041 partial)
...
10 of 40000 iterations read back wrong data (8 load processes, destination reusing the freed address)
```

<details>
<summary>Reproducer: lost_writes.hip</summary>

```cpp
// lost_writes.hip: a kernel's writes from one XCD are lost when its destination reuses a just-freed address
// while other processes use the same GPU.
// Build: hipcc --offload-arch=gfx942 -O2 -o lost_writes lost_writes.hip
// Run:   ./lost_writes [iterations=40000] [loads=8] [noreuse]
//   Starts <loads> copies of itself as background load (kernels and copies on 4 streams), then per iteration:
//   A = hipMalloc(4 MB), H2D copy into A; Y = hipMalloc(8 MB), H2D copy of the source into every second float of Y;
//   hipFree(A); D = hipMalloc(4 MB) (gets A's address; "noreuse" allocates D before freeing A); H2D zeros into D;
//   hipDeviceSynchronize; kernel copies Y -> D (3907 blocks of 256 threads); hipStreamSynchronize; D2H and compare.
//   Exit status 1 if any iteration read back wrong data.
#include <hip/hip_runtime.h>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/wait.h>
#include <unistd.h>
#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e_)); exit(2); } } while (0)

__global__ void copy_strided(float* dst, const float* src, size_t dst_stride, size_t src_stride, size_t n)
{
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) dst[i * dst_stride] = src[i * src_stride];
}

__global__ void burn(float* x, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float v = x[i]; for (int k = 0; k < 100; ++k) v = v * 0.999f + 0.5f; x[i] = v; }
}

static int load()  // background load: kernels and copies on 4 streams until the parent exits
{
    const size_t m = 16 << 20;
    hipStream_t s[4]; float *a[4], *b[4]; float* h = (float*)calloc(m / 16, 4);
    for (int k = 0; k < 4; ++k) { CK(hipStreamCreate(&s[k])); CK(hipMalloc(&a[k], m * 4)); CK(hipMalloc(&b[k], m * 4)); }
    pid_t parent = getppid();
    while (getppid() == parent) {
        for (int k = 0; k < 4; ++k) {
            burn<<<(m + 255) / 256, 256, 0, s[k]>>>(a[k], (int)m);
            CK(hipMemcpyAsync(b[k], a[k], m * 4, hipMemcpyDeviceToDevice, s[k]));
            CK(hipMemcpyAsync(h, b[k], m / 16 * 4, hipMemcpyDeviceToHost, s[k]));
        }
        for (int k = 0; k < 4; ++k) CK(hipStreamSynchronize(s[k]));
    }
    return 0;
}

static int count_wrong(const float* x, const float* ref, int n)
{
    int w = 0;
    for (int i = 0; i < n; ++i) w += x[i] != ref[i];
    return w;
}

// "class:blocks" for each block class (block index % 8) with whole 256-element blocks wrong, plus partial blocks
static void lost_blocks(const float* x, const float* ref, int n, char* text, size_t len)
{
    int whole[8] = {0}, partial = 0;
    for (int b = 0; b * 256 < n; ++b) {
        int m = n - b * 256 < 256 ? n - b * 256 : 256, w = 0;
        for (int i = 0; i < m; ++i) w += x[b * 256 + i] != ref[b * 256 + i];
        if (w == m) ++whole[b % 8]; else if (w) ++partial;
    }
    text[0] = 0;
    for (int k = 0; k < 8; ++k) if (whole[k]) snprintf(text + strlen(text), len - strlen(text), "%d:%d ", k, whole[k]);
    snprintf(text + strlen(text), len - strlen(text), "(+%d partial)", partial);
}

int main(int argc, char** argv)
{
    if (argc > 1 && !strcmp(argv[1], "load")) return load();
    int iters = argc > 1 ? atoi(argv[1]) : 40000, loads = argc > 2 ? atoi(argv[2]) : 8;
    bool noreuse = argc > 3 && !strcmp(argv[3], "noreuse");
    pid_t pids[64];
    for (int k = 0; k < loads && k < 64; ++k)
        if ((pids[k] = fork()) == 0) { execl("/proc/self/exe", argv[0], "load", (char*)nullptr); _exit(127); }
    sleep(2);

    const int n = 1000000, blocks = (n + 255) / 256;
    hipStream_t S; CK(hipStreamCreate(&S));
    float* src = (float*)malloc(n * 4); float* zeros = (float*)calloc(n, 4); float* out = (float*)malloc(n * 4);
    float* strided = (float*)calloc(2 * (size_t)n, 4);
    int bad = 0;
    printf("%-10s %-12s %-12s %s\n", "iteration", "wrong", "after sync", "whole blocks wrong after sync, class (block % 8):count");
    for (int it = 0; it < iters; ++it) {
        for (int i = 0; i < n; ++i) strided[2 * i] = src[i] = float(it % 1000 + i);
        float *A, *Y, *D;
        CK(hipMalloc(&A, n * 4));
        CK(hipMemcpyAsync(A, src, n * 4, hipMemcpyHostToDevice, S));
        CK(hipMalloc(&Y, 2 * (size_t)n * 4));
        CK(hipMemcpyAsync(Y, strided, 2 * (size_t)n * 4, hipMemcpyHostToDevice, S));
        if (noreuse) { CK(hipMalloc(&D, n * 4)); CK(hipFree(A)); }
        else { CK(hipFree(A)); CK(hipMalloc(&D, n * 4)); }
        CK(hipMemcpyAsync(D, zeros, n * 4, hipMemcpyHostToDevice, S));
        CK(hipDeviceSynchronize());
        copy_strided<<<blocks, 256, 0, S>>>(D, Y, 1, 2, n);
        CK(hipStreamSynchronize(S));
        CK(hipMemcpy(out, D, n * 4, hipMemcpyDeviceToHost));
        int w = count_wrong(out, src, n);
        if (w) {
            ++bad;
            CK(hipDeviceSynchronize());
            CK(hipMemcpy(out, D, n * 4, hipMemcpyDeviceToHost));
            char lost[128]; lost_blocks(out, src, n, lost, sizeof(lost));
            printf("%-10d %-12d %-12d %s\n", it, w, count_wrong(out, src, n), lost);
            fflush(stdout);
        }
        CK(hipFree(D)); CK(hipFree(Y));
    }
    for (int k = 0; k < loads && k < 64; ++k) { kill(pids[k], SIGTERM); waitpid(pids[k], nullptr, 0); }
    printf("%d of %d iterations read back wrong data (%d load processes, destination %s)\n", bad, iters, loads,
           noreuse ? "allocated before the free" : "reusing the freed address");
    return bad ? 1 : 0;
}
```

</details>

### Operating System

Ubuntu 24.04.5 LTS (container), host kernel 6.8.0-110-generic

### CPU

2x Intel Xeon Platinum 8480C

### GPU

AMD Instinct MI300X HF (gfx942), SPX compute partition, NPS1 memory partition; amdgpu driver 6.16.13

### ROCm Version

ROCm 10.1.0, 10.2.0 and nightly 10.3.0a20261120, all as Python wheels (`rocm[devel]`)

### ROCm Component

clr (HIP runtime)

### Steps to Reproduce

```
hipcc --offload-arch=gfx942 -O2 -o lost_writes lost_writes.hip
./lost_writes 40000        # exits 1 and prints one line per failing iteration when the defect appears
./lost_writes 40000 8 noreuse   # control: destination allocated before the free, expected 0 failures
./lost_writes 40000 0      # control: no load processes, expected 0 failures
```

The program starts 8 copies of itself as load on the same GPU (kernels and copies on 4 streams each) and stops them at the end. Use one GPU (for example `HIP_VISIBLE_DEVICES=0`) that nothing else is using. Failures are rare and bursty; run 40,000 iterations (a few minutes) more than once.
====

Placeholders to fill before filing: the MI355X (gfx950) and MI250X (gfx90a) rows of the table, from the parallel sessions. Scripts: `notes-evidence3-gfx942-flake-scripts.md`.

## Evidence session (2026-11-20, linux-gfx950): lost_writes

Requested by Jeff Daily; evidence only (no state change, nothing pushed to the fork, the draft untouched). `lost_writes.hip` extracted verbatim from the issue draft in "Evidence session 3 (2026-11-20, linux-gfx942)" (md5 e7f987429d1e500ab2ca84594ef274b9), built with ROCm 10.2.0 from the job venv (`hipconfig --version` 7.17.26434-0000000): `hipcc --offload-arch=gfx950 -O2 -o lost_writes lost_writes.hip` (links the venv's `libamdhip64.so.7`). Run with `HIP_VISIBLE_DEVICES=1` (dispatcher pin) and `ulimit -c 0`; the program forks and stops its own load children; nothing else was started or killed.

Environment: HIP device under the pin = PCI `0000:05:00.0` (`hipDeviceGetPCIBusId`) = amd-smi GPU 0, `MARKET_NAME: AMD Instinct MI355X` (device id 0x75a3, `TARGET_GRAPHICS_VERSION: gfx950`). Kernel `6.8.0-101-generic` (`uname -r`, the host's). amdgpu driver `6.19.14.31400100` (`/sys/module/amdgpu/version`; amd-smi 27.1.0 reports the same). `amd-smi process -g 0` before and after every run: "No running processes detected"; 30 s into each loaded run: the reproducer (~0.85 GB) and its 8 load children (2.2 GB each), nothing else; in the no-load run the reproducer alone.

| run | wrong iterations | lost class (block % 8): whole blocks, per failure |
|---|---|---|
| `./lost_writes 40000` (1) | **2 / 40,000** | 1:489 (+1 partial block, 125,200 floats); 1:489 (125,184) |
| `./lost_writes 40000` (2) | **5 / 40,000** | 1:489 (+1 partial) x3; 4:488 (124,928) x2 |
| `./lost_writes 40000 8 noreuse` | **0 / 40,000** | |
| `./lost_writes 40000 0` | **0 / 40,000** | |

Every failure was still wrong after a synchronize (wrong count after sync equal to the first read). Times: 125, 124, 129 and 49 s. So gfx950 reproduces the final reproducer (7 / 80,000 under its own 8-process load), and both controls are clean, as on gfx942.

## Evidence session (2026-11-20, linux-gfx90a): lost_writes

Requested by Jeff Daily, same evidence-only job (no lock, no state change, no fork push, no GitHub writes; `HIP_VISIBLE_DEVICES=0` kept, `ulimit -c 0`; no `alloc` load; only processes this job started were stopped, and `lost_writes` stops its own load children itself). The issue draft in "Evidence session 3 (2026-11-20, linux-gfx942)" was not edited.

**Result: `lost_writes` does not fail on gfx90a. 0 wrong iterations in every run (no lost class to report)**, including 80,000 iterations under its own 8 load processes, where gfx942 loses one class 5 to 19 times per 40,000.

- GPU, exactly as amd-smi names it: market name `AMD Instinct MI250X / MI250`, product name `AMD INSTINCT MI250 (MCM) OAM AC MBA` (device id 0x740c, 104 CUs per device, gfx90a:sramecc+:xnack-; one GCD per HIP device).
- Kernel `6.8.0-101-generic`; amdgpu driver `6.16.6` (`/sys/module/amdgpu/version`, the same in `amd-smi static`).
- ROCm 10.2.0 (`python -m rocm_sdk version` in the job venv), `hipconfig --version` `7.17.26434-0000000`.
- Device mapping: under `HIP_VISIBLE_DEVICES=0`, `hipDeviceGetPCIBusId` gives HIP device 0 = `0000:29:00.0`, which `amd-smi list` shows as GPU 0 (all four devices map 1:1 in order with no mask set). So `amd-smi process -g 0` is the pinned device, here and in this job's earlier gfx90a sections.
- Source: `lost_writes.hip` copied verbatim from the draft's `<details>` block (114 lines). Build: `$(python -m rocm_sdk path --root)/bin/hipcc --offload-arch=gfx90a -O2 -o lost_writes lost_writes.hip`.
- Driver `lw_run.sh <label> <args>`: `amd-smi process -g 0` before; `./lost_writes <args>` in the background; 60 s in, `amd-smi process -g 0` and the program's child list (`ps --ppid`); wait; count children left afterwards (0 in every run).

| run | command | wrong iterations | lost classes | wall time | GPU 0 during (amd-smi, 60 s in) |
|---|---|---|---|---|---|
| 1 | `./lost_writes 40000` | **0 / 40,000** | none | 351 s | 8 x 1.1 GB (the program's 8 `load` children) + 1 x 395 MB |
| 2 | `./lost_writes 40000` | **0 / 40,000** | none | 360 s | 8 x 1.1 GB + 1 x 383 MB |
| control | `./lost_writes 40000 8 noreuse` | **0 / 40,000** | none | 361 s | 8 x 1.1 GB + 1 x 395 MB (see note) |
| control | `./lost_writes 40000 0` | **0 / 40,000** | none | 113 s | 1 x 456 MB only |

Before every run `amd-smi process -g 0` showed no process. Note on the controls: my first `./lost_writes 40000 0` was started while the `noreuse` run was still going (an earlier command had moved to the background; my mistake). It also gave 0 / 40,000, but its GPU shared the `noreuse` run's 9 processes, so it is not a no-load control and is not in the table. It was repeated alone (the row above, nothing else on GPU 0 before or during). The `noreuse` run had that extra `lost_writes` process beside it for its last ~2 minutes (11:30-11:32); for a control that came out 0 that only adds load. Runs 1 and 2 did not overlap anything.

The load was real: with 8 load processes a run took 3.1-3.2x as long as alone (351-361 s against 113 s). For the draft's results table, the gfx90a row reads: MI250X / MI250 (gfx90a), 10.2.0 (7.17.26434), 0, 0 per 40,000 (controls `noreuse` 0, no load 0). With the 250 Warp subset runs and 160,000 `copy_flake2` iterations above, this single-die device is clean at every level tested.

## Coordinator note (2026-11-20): lost-writes issue draft finalized, awaiting Jeff

The Evidence session 3 draft was completed with the `lost_writes` rows from gfx950 (2 and 5 per 40,000, controls 0) and gfx90a (0, 0), the MI250 naming corrected, and the title generalized to `[clr] One XCD's kernel writes lost after hipFree and hipMalloc`. Body and filing script are staged for Jeff (labels `bug`, `project: clr`); nothing filed. Deferred item `hip-mi300-xcd-lost-writes-after-free-realloc` still needs Jeff's ruling. Correction carried from the gfx950 job: the "outside process on GPU 1" of sessions 20-22 was a different GPU (amd-smi index differs from the HIP index); the pinned GPU was idle.

## Ruling on `-ffp-contract` (jeff, 2026-11-20)

Jeff: "go with (b) fast-honor-pragmas if you can confirm CUDA behavior first."

CUDA confirmation (coordinator, V100, CUDA 12.9.86, driver 575.57.08; sources in `agent_space/p6ev/fmad/`: `kern.cu`, `drv.cpp`, `inv16.py`):
- Three kernels: `r = a*b + c`; `t = a*b; r = t + c`; `t = mul(a,b); r = add(t,c)` through inline device functions. NVRTC with no option and with `--fmad=true`: one `fma.rn.f32` in each kernel's PTX, and each returns 5.96e-08 for a = b = 1+2^-12, c = -(1+2^-11) (0 would mean the product was rounded first). `--fmad=false`: no fma, result 0. `nvcc -ptx` gives the same counts.
- ROCm clang 23 (`-x hip --offload-arch=gfx942 -O2 -emit-llvm`): `-ffp-contract=on` contracts only the one-expression kernel; `fast` and `fast-honor-pragmas` contract all three; `off` none. Warp's generated code is the third shape, so `on` leaves generated kernels with no fusion.
- fp16 differs: upstream's `warp-lang==1.14.0` CUDA wheel passes `test_inverse_float16_cuda_0` with margin (53 checks, worst difference 0.0312 against tolerance 0.05; identical values with module option `fuse_fp` True and False). CUDA's `float_to_half` is inline asm `cvt.rn.f16.f32`, so a half product is always rounded before the add. On HIP the conversions are `_Float16` casts the optimizer can see through; hypothesis: that lets cross-statement contraction form a half fma, which is the one failure of Evidence 6.

Consequence: the porter was asked (session 22) to switch to `fast-honor-pragmas` and keep fp16 arithmetic unfused on HIP with a guarded pragma instead of changing the test tolerance; the ROCm-gated tolerance bump jeff approved is the fallback only.

## Porting session 22 (2026-11-20, linux-gfx950 MI355X, porter): -ffp-contract ruling

Ruling (Jeff Daily, 2026-11-20, recorded above): hipRTC to `-ffp-contract=fast-honor-pragmas`, on the condition that it matches CUDA. Fork `release/1.14.0.amd0` `7059a340..e92f2611` (pushed, `advance-head` recorded); stage left at `porting`. Same job and environment as sessions 20-21 (`HIP_VISIBLE_DEVICES=1` = PCI 0000:05:00.0 = amd-smi GPU 0).

### The fp16 difference: hypothesis confirmed

`agent_space/f16/half_fma.py` (below): kernels `t = a[i] * b[i]; r[i] = t + c[i]` in float16, bfloat16 and float32 through Warp's operators, compiled with a private kernel cache, disassembled (`llvm-objdump -d --mcpu=gfx950`); fp16 results compared with "product rounded to half, then add" (CUDA's behaviour: `float_to_half` is `cvt.rn.f16.f32` inline asm) and with one fused fma, 100,000 inputs in [1,2) x [1,2) + [-4,4) (the two differ in 36,963):

| tree | hipRTC option | fp16 kernel | fp16 results equal to product-rounded-then-added | bf16 kernel | fp32 kernel |
|---|---|---|---|---|---|
| `7059a340` (fork before) | `on` | `v_mul_f16` + `v_add_f16` | 100,000 | `v_mul_f32`, `v_add_f32` (software rounding) | `v_mul_f32` + `v_add_f32` (no fusion) |
| `wff` scratch (`113e1e70` + option only) | `fast-honor-pragmas` | **`v_fma_f16`** | 63,037 (all equal to the fused fma) | unchanged | `v_fmac_f32` |
| `e92f2611` (committed) | `fast-honor-pragmas` + builtin.h pragma | `v_mul_f16` + `v_add_f16` | **100,000** | unchanged | `v_fmac_f32` |

So on HIP, LLVM narrows `float_to_half(half_to_float(a) * half_to_float(b))` (the `_Float16` casts of builtin.h's `__clang__` branch) to a native half multiply, which rounds exactly like CUDA's convert-after-multiply; with cross-statement contraction it then fuses that multiply and the following add into a half fma, which CUDA never does. That was Evidence 6's one failure. bfloat16 rounds with integer operations (`wp_float_to_bfloat16_bits_sw`) and has nothing to fuse.

### The change: `e92f2611` [ROCm] Contract multiply-adds across statements in HIP kernels, as NVRTC (`builtin.h` +10, `warp.cu` +2/-2)

- `wp_hip_compile_program`: `fuse_fp ? "-ffp-contract=fast-honor-pragmas" : "-ffp-contract=off"`, comment "NVRTC's default --fmad=true contracts across statements; Clang's HIP default does too and honors pragmas".
- `builtin.h`: under `#if defined(__HIP_DEVICE_COMPILE__)`, `#pragma STDC FP_CONTRACT OFF` before `struct half;` and `#pragma STDC FP_CONTRACT DEFAULT` after the free half operators (`operator/(half, half)`); the region holds the half type and its compound operators, the bfloat16 type, the conversion functions and the free half operators (no fp multiply-add pairs other than the half operators'). A comment says why. Scoping tried and rejected, with evidence (`agent_space/f16/p_*.hip`, `q.hip`, device-only `-S -emit-llvm`, ROCm 10.2 clang): `#pragma float_control(push)/(pop)` is ignored on AMDGPU ("'#pragma float_control' is not supported on this target - ignored"), so the first version's `clang fp contract(off)` leaked to the end of the file and the fp32 kernel lost its fusion (caught by the disassembly table: fp32 back to `v_mul_f32` + `v_add_f32`); restoring with `#pragma clang fp contract(fast)` would re-enable fusion when the compile was given `-ffp-contract=off` (Warp's `fuse_fp=False`). `STDC FP_CONTRACT DEFAULT` restores whatever the command line said: measured, after `contract(off)`, functions after `DEFAULT` get `contract` flags under `-ffp-contract=fast-honor-pragmas` and none under `off` or `on`.
- The precompiled library: its HIP compile commands carry no `-ffp-contract` (and no `-ffast-math`); a `-x hip` device compile with no option marks cross-statement fmul/fadd `contract` and honors `#pragma clang fp contract(off)` (`agent_space/f16/d.hip`), i.e. the library already uses `fast-honor-pragmas`; left as is. The builtin.h region applies to the library's device compiles too, so half arithmetic in precompiled kernels stops fusing as well (matching CUDA).
- CUDA: `cuda_pp.sh` (nvcc 12.8) with the change vs `7059a340`: all 24 preprocessed units identical (device passes included, so NVRTC sees the same header text). CPU (warp-clang) never defines `__HIP_DEVICE_COMPILE__`: unchanged.

### Tests (fork clone built for gfx950 at the change)

- `test_mat -k test_inverse_float16`: OK 3/3 at upstream's tolerance, no test change (Evidence 6: failed 3/3 with the option alone).
- test_mat 134 OK, test_fp16 6 OK, test_bf16 48 OK, test_math 16 OK, test_vec_scalar_ops 402 OK.
- Full default suite from a fresh kernel cache (`agent_space/suite.sh`, cache directory deleted first): **Ran 7557 tests in 534.2 s, OK (skipped=81)**, the same counts as session 21 at `7059a340`; no test changed outcome; the `test_async` lost-writes failures did not appear.
- `jargon.py --port` clean before the push.

### Next step (head `e92f2611`)

- The release wheel validated in session 21 (`<job>/wheel/warp_lang-1.14.0+rocm10.2.0-py3-none-manylinux_2_28_x86_64.whl`, sha256 `2100e06c...d5ea25`, built at `7059a340`) is **stale**: the head moved and the kernel compile options changed. It is rebuilt and revalidated after Jeff's remaining rulings.
- Not moved to `ported`: more rulings from Jeff are pending (questions 1-4, 6, 8, 9 of sessions 20-21; question 5 is ruled here, 7 is the lost-writes runtime defect).
- Validators revalidate at `e92f2611` (every HIP kernel's floating-point results change slightly where multiply-adds now fuse).
- Release-notes draft: the fp-contract line ("multiply-adds are fused within an expression only") becomes "multiply-adds are fused across statements, as with NVRTC's default; fp16 arithmetic is rounded to half after each operation, as on CUDA".

### Script from this session

`agent_space/f16/half_fma.py`:
```
# usage: half_fma.py <cache dir>; one kernel computing a*b + c on wp.float16 through Warp's operators (wp::mul, wp::add),
# and the same on wp.bfloat16 and float32 (kernels for the disassembly); prints the results for inputs where fusing changes the rounding
import sys
import numpy as np
import warp as wp
wp.config.quiet = True
wp.config.kernel_cache_dir = sys.argv[1]
wp.init()

@wp.kernel
def half_mad(a: wp.array(dtype=wp.float16), b: wp.array(dtype=wp.float16), c: wp.array(dtype=wp.float16), r: wp.array(dtype=wp.float16)):
    i = wp.tid()
    t = a[i] * b[i]
    r[i] = t + c[i]

@wp.kernel
def bf16_mad(a: wp.array(dtype=wp.bfloat16), b: wp.array(dtype=wp.bfloat16), c: wp.array(dtype=wp.bfloat16), r: wp.array(dtype=wp.bfloat16)):
    i = wp.tid()
    t = a[i] * b[i]
    r[i] = t + c[i]

@wp.kernel
def f32_mad(a: wp.array(dtype=float), b: wp.array(dtype=float), c: wp.array(dtype=float), r: wp.array(dtype=float)):
    i = wp.tid()
    t = a[i] * b[i]
    r[i] = t + c[i]

rng = np.random.default_rng(0)
n = 100000
A = rng.uniform(1, 2, n).astype(np.float16); B = rng.uniform(1, 2, n).astype(np.float16); C = rng.uniform(-4, 4, n).astype(np.float16)
r = wp.zeros(n, dtype=wp.float16)
wp.launch(half_mad, dim=n, inputs=[wp.array(A), wp.array(B), wp.array(C), r])
unfused = ((A.astype(np.float32) * B.astype(np.float32)).astype(np.float16).astype(np.float32) + C.astype(np.float32)).astype(np.float16)
fused = (A.astype(np.float64) * B.astype(np.float64) + C.astype(np.float64)).astype(np.float16)
x = r.numpy()
rf = wp.zeros(n, dtype=float)
wp.launch(f32_mad, dim=n, inputs=[wp.array(A.astype(np.float32)), wp.array(B.astype(np.float32)), wp.array(C.astype(np.float32)), rf])
rb = wp.zeros(n, dtype=wp.bfloat16)
wp.launch(bf16_mad, dim=n, inputs=[wp.array(A.astype(np.float32), dtype=wp.bfloat16), wp.array(B.astype(np.float32), dtype=wp.bfloat16), wp.array(C.astype(np.float32), dtype=wp.bfloat16), rb])
print(f"half: equal to unfused (product rounded to half, then add) {np.sum(x == unfused)} / {n}; equal to a fused fma {np.sum(x == fused)} / {n}; unfused != fused in {np.sum(unfused != fused)}")
```

## Porting session 23 (2026-11-20, linux-gfx950 MI355X, porter): cross-stream free test

Ruling (Jeff Daily, 2026-11-20): add session 20's `free_race.py` probe to `warp/tests/cuda/test_streams.py` as a test, a visible record of the hazard. Fork `release/1.14.0.amd0` `e92f2611..e88cf857` (pushed, `advance-head` recorded); stage left at `porting`. `HIP_VISIBLE_DEVICES=1` (PCI 0000:05:00.0 = amd-smi GPU 0), `ulimit -c 0`; no process was stopped by pattern.

### The commit: `e88cf857` Test that a free waits for work on other streams (`test_streams.py` +35, Python only)

- Module-level kernel `write_late(a, iters)` (spins `iters` times, then writes 1.0) and `test_stream_free_waits_for_other_streams`: under `wp.ScopedMempool(device, True)`, streams `writer` and `freer`; for each case (array allocated with `writer` current, or with `freer` current) 3 trials of: `wp.empty(1M floats)`, `write_late` launched on `writer` (20,000 iterations, ~10 ms on MI355X), then with `freer` current `del a` and `b = wp.zeros(1M)`, `wp.synchronize_device()`, assert `np.count_nonzero(b.numpy()) == 0`. Data-only assertion, no timing; registered on `get_selected_cuda_test_devices_with_mempool()` like the mempool tests elsewhere. No `[ROCm]` prefix: the test is shared and names no HIP.
- Pooled allocations on purpose: the lost-writes runtime defect (rocm-systems 16355) needs a `hipMalloc` (pool-off) destination at a just-freed address under multi-process load; with the pool on it was 0/40,000 in the gfx942 narrowing.
- Teeth: scratch worktree `agent_space/wnowait` = `e92f2611` plus one line in `free_async_after_device_streams` (`continue;` at the top of the stream loop: free on the current stream, no wait; nothing committed), gfx950 build: the new test FAILED 3/3 (`AssertionError: 1022464 != 0`, `1024000 != 0`, `1023680 != 0` of 1,048,576 elements: the late writes landed in the new array). Committed rule: OK 3/3, then **20/20 OK** in 20 separate runs (0.63 s per run).
- test_streams.py on the pinned GPU: 25 OK (3 skipped). Full default suite from a fresh kernel cache (`agent_space/suite.sh`): **Ran 7558 tests in 2315.7 s, FAILED (failures=2, skipped=81)**. The two failures are the known upstream isolation failures recorded since session 2 (`test_fem_quadrature.test_gimp_quadrature`, `test_fem_shape.test_cube_shape_functions`: "Unexpected output" whose first line is Warp's init banner, captured by the first `check_output` test in its worker); this run's scheduling put a fem file first in a worker, and each file run alone fails the same way (1 failure each). Everything else passed, including the new test; the 7557 of session 22 plus the new test is 7558. The run took 2316 s instead of the usual ~530 s (scheduling or host load; not investigated).
- Python-only change: no native file changes, so no nvcc check (the CUDA build compiles nothing new); `jargon.py --port` clean before the push.

### Evidence (not a commit): `free_race.py` against AMD's shipped 1.13 build

Tree `agent_space/w113` = AMD-Ecosystem/warp `release/1.13.0.amd0` at **`5035c8ed`** ("Merge pull request #27 from AMD-Ecosystem/docs/26.10"), no modified tracked file, built in session 20 with its documented command for gfx950 on the image's ROCm 10.1 (`ROCM_PATH=<10.1 root> python build_lib.py --hip-arch=gfx950 --hipcc-options="-Xarch_device -fno-inline"`), run with ROCm 10.1; mempool enabled on the device (`wp.is_mempool_enabled` True). Session 20's original `free_race.py`, unmodified, twice:

| case | run 1 | run 2 |
|---|---|---|
| same_stream (control) | 0 / 10 late writes (pointer reused 10/10) | 0 / 10 (10/10) |
| alloc_on_s1 (allocated on the writing stream, freed with s2 current) | 0 / 10 (10/10) | 0 / 10 (10/10) |
| alloc_on_s2 (allocated with s2 current, written on s1, freed with s2 current) | **10 / 10** (10/10) | **10 / 10** (10/10) |

So AMD's shipped 1.13 code does show late writes, in the alloc_on_s2 case, every time; the same result as session 20's reimplementation of `da56e495`'s rule. Nothing in 1.13 makes that case unreachable: its streams are created exactly like 1.14's (`cuStreamCreateWithPriority_f(&stream, CU_STREAM_DEFAULT, priority)`, blocking), pooled arrays are freed through the same `wp_free_device_async` from `Allocator.deallocate`, and its free waits only for the stream recorded at allocation (`g_alloc_streams`), which in this case is the freeing stream itself; the probe uses only public API (`wp.ScopedStream`, `wp.empty`, `wp.launch(..., stream=)`, `del`, `wp.zeros`).

### Next step (head `e88cf857`)

- Not moved to `ported`; more rulings from Jeff are pending. The release wheel validated at `7059a340` stays stale (head now `e88cf857`); rebuild and revalidate after the remaining rulings.
- Validators: the new test runs in test_streams on every platform with a mempool (data-only, about 0.6 s on MI355X).

## Rulings on the free-ordering test and refit (jeff, 2026-11-20)

- Free-ordering test: "I'm fine with (a) since it's a publicly visible record of this hazard." Implemented as `e88cf857` (session 23). CUDA reference (coordinator, V100, upstream `warp-lang==1.14.0`, `free_race.py 20000`, two runs): every case reused 10/10, late writes 0/10.
- Depth-level refit on HIP: "let's do (b) this time." Dispatched to the porter as session 24 (prompt `agent_space/p6ev/refit-porter.txt`).
- Open with jeff: AMD 1.13 (`5035c8ed`) shows late writes in the alloc_on_s2 case 10/10 (session 23); whether to tell its authors is his decision.

## Rulings on the ray traversal and Cholesky (jeff, 2026-11-20)

- `mesh_query_ray`: "(a) Leave it reverted". The fork keeps upstream's traversal.
- Tile Cholesky n=16: "(a) Leave upstream's code". No pivot change, no cooperative port.

## Ruling on the library build's use of -j (jeff, 2026-11-20)

"(a) Leave it". `build_lib.py` keeps upstream's half-of-`-j` split. All six session 20/21 questions are now ruled: fp-contract (b, `e92f2611`), free-ordering test (a, `e88cf857`), refit (b, session 24 in progress), ray traversal (a), Cholesky (a), build -j (a).

## Ruling on a Warp workaround for the lost-writes defect (jeff, 2026-11-20)

"(a)": no workaround in Warp; a known-issue entry in the release notes: "On MI300X and MI355X, arrays allocated with the memory pool disabled can lose kernel writes when other processes share the GPU (ROCm/rocm-systems#16355). Keep the memory pool enabled, which is the default." Validators on gfx942/gfx950 that see the `test_async` pool-off copy failures in a full run rerun those tests alone and record both results; no skip.

## Ruling on the AMD 1.13 defects (jeff, 2026-11-20)

Defer: no issue on `AMD-Ecosystem/warp` for the 1.13 cross-stream free defect or the 1.13 tile-solve failures until after the 1.14 release is out. Registered as deferred items `amd-warp-113-cross-stream-free-late-writes` and `amd-warp-113-tile-solve-race`. Until then nothing GitHub-visible on that repository beyond the port branch itself, and the release notes do not call out 1.13's defects unless jeff asks for it.

## Ruling on the two fem isolation failures (jeff, 2026-11-20)

"(b)": fix the isolation in the fork so `test_gimp_quadrature` and `test_cube_shape_functions` pass when theirs is the first Warp initialization in a worker (smallest change; not ROCm-specific, so no `[ROCm]` prefix). Queued for the porter after session 24 (prompt `agent_space/p6ev/fem-porter.txt`).
## Porting session 24 (2026-11-20, linux-gfx950 MI355X, porter): depth-level refit

Ruling (Jeff Daily, 2026-11-20) on session 20's refit observation: (b), a HIP depth-level refit in this line. Fork `release/1.14.0.amd0` `e88cf857..f6ada527` (pushed, `advance-head` recorded); stage left at `porting`. `HIP_VISIBLE_DEVICES=1` (PCI 0000:05:00.0 = amd-smi GPU 0), `ulimit -c 0`; no process stopped by pattern.

### The commit: `f6ada527` [ROCm] Refit large BVHs one depth level at a time on HIP (`bvh.cu` +71/-3, all HIP-only or inside the existing HIP builder)

- New kernel `refit_tree_at_depth` (one thread per node, nodes of one depth): a node with `b` set (leaf or packed leaf) takes its primitives' bounds (`bounds3::add_bounds` in primitive order) unless its parent is packed (`b` set); an internal node takes `min/max` of its children in the walk's operand order; only `x, y, z` written, as the walk does. This reproduces `bvh_refit_kernel`'s rules exactly, including the muted nodes below packed leaves (which the walk also updates). It could not reuse the builder's `refit_nodes_at_depth`: after packing, packed nodes have lost their child indices (the walk climbs by parent pointers instead), and leaves need their primitives' bounds.
- Depth storage: the BVH's existing `node_counts` array (`max_nodes` ints), which only the bottom-up walks use, as atomic counters. **Memory added per BVH: 0 bytes**; the LBVH builder's scratch pool shrinks by 4 x max_nodes bytes because it now writes its depths into `node_counts` (`int* depths = bvh.node_counts`). Host-built trees (sah, median, also grouped): at creation, `node_counts` zeroed and `compute_node_depths` run over `num_nodes` (nodes past `num_nodes` keep depth 0 and are never touched; their parent entries are uninitialized host memory). `compute_node_depths` now accepts a null max pointer (1 line).
- Topology changes: LBVH build and in-place rebuild recompute the depths (the builder); host-built trees get theirs at creation; an in-place rebuild of a host-built GPU tree runs the LBVH builder, which rewrites the depths. Capture: the builder is capture-safe (session 20); the refit launches a fixed number of levels and reads nothing back, so it is legal inside a graph.
- Level count: `max(bvh.max_depth, key_bits + index_bits) + 1` (key_bits 30, or 64 with groups). `max_depth` is the host builder's depth for host-built trees (0-based, so +1) and, newly, the Karras bound `key_bits + index_bits` for LBVH trees (set by the HIP builder; the descriptor field was left uninitialized for device LBVH trees before). The max covers a host-built tree that an in-place rebuild turned into an LBVH tree (`wp_bvh_rebuild_device` does not update the stored descriptor); +1 for the leaves. Levels below the deepest node do nothing (each thread reads its depth and returns).
- Threshold: `num_leaf_nodes >= 32768` uses the levels, smaller trees keep the walk (below). A tree never switches path (num_leaf_nodes fixed per tree), so the walk's use of `node_counts` as counters never meets depths it needs. Meshes with winding-number support (`solid_angle_props`) keep their own walk (`bvh_refit_with_solid_angle_kernel` in mesh.cu, a separate kernel with solid-angle data); not converted (question below).

### Correctness

Scratch worktree `agent_space/wrf` with the same change and an environment switch (`WP_REFIT_DEPTH_MIN`) forcing either path; `agent_space/bench/refit_check.py` (inline below) reads the BVH descriptor at `bvh.id` from device memory, saves the node arrays as raw uint32 words after each step, and the two runs are compared bit for bit. Trees: random boxes, 10^3, 10^4, 10^5, 10^6, 10^7 items, lbvh (all sizes) and sah/median (to 10^6), without and with 7 groups, leaf_size 1 and 4 (packed leaves and muted nodes); steps: three refits with moved boxes, `rebuild()` then a refit, then a refit captured in a graph and relaunched three times with moved boxes. **520 arrays compared: 0 differ.** Every node changes between consecutive refits (so the comparison bites). The fork tree with the threshold vs the forced walk at 10^5 and 10^6: 240 arrays, 0 differ. Queries (`refit_query.py`, below): meshes (lbvh/sah/median, 10^5 and 10^6 triangles) with points moved and refit three times, 10^6 point and ray queries, plus Bvh AABB queries: 18 result arrays identical between the fork tree and the forced walk.

### Performance (`refit_perf.py`, 3 interleaved rounds of `rfwalk` / `rfdepth` = the same build forced onto each path; per call, 10 warm-up then 7 x N back-to-back calls and a sync; noise floor = a mesh_query_point launch on the same mesh, unchanged code)

| op | constructor | N | rfwalk us | rfdepth us | rfdepth/rfwalk | per-round ratios | noise floor rfdepth/rfwalk |
|---|---|---|---|---|---|---|---|
| bvh_refit | lbvh | 1000 | 31.9 | 72.2 | 2.26 | 4.84 / 1.97 / 2.26 | 1.00 |
| bvh_refit | lbvh | 3000 | 39.0 | 152.1 | 3.90 | 3.90 / 2.36 / 4.12 | 0.99 |
| bvh_refit | lbvh | 10000 | 58.8 | 123.8 | 2.11 | 2.77 / 1.52 / 2.00 | 1.00 |
| bvh_refit | lbvh | 30000 | 153.8 | 106.6 | 0.69 | 0.86 / 0.59 / 0.74 | 1.00 |
| bvh_refit | lbvh | 100000 | 430.9 | 107.2 | 0.25 | 0.35 / 0.24 / 0.25 | 1.00 |
| bvh_refit | lbvh | 1000000 | 4250.4 | 307.4 | 0.07 | 0.07 / 0.07 / 0.07 | 1.00 |
| bvh_refit | lbvh | 10000000 | 43622.9 | 3200.0 | 0.07 | 0.06 / 0.07 / 0.09 | 1.00 |
| bvh_refit | median | 1000 | 22.0 | 68.9 | 3.12 | 7.27 / 3.12 / 2.78 | 1.00 |
| bvh_refit | median | 3000 | 34.0 | 138.1 | 4.06 | 5.27 / 2.06 / 3.24 | 0.99 |
| bvh_refit | median | 10000 | 45.8 | 97.4 | 2.13 | 3.49 / 1.65 / 1.69 | 1.00 |
| bvh_refit | median | 30000 | 111.0 | 114.6 | 1.03 | 1.56 / 0.71 / 1.10 | 0.99 |
| bvh_refit | median | 100000 | 328.2 | 117.0 | 0.36 | 0.42 / 0.26 / 0.36 | 0.99 |
| bvh_refit | median | 1000000 | 3096.8 | 243.1 | 0.08 | 0.08 / 0.07 / 0.08 | 0.97 |
| bvh_refit | sah | 1000 | 23.2 | 65.9 | 2.84 | 2.88 / 2.88 / 2.52 | 1.00 |
| bvh_refit | sah | 3000 | 31.2 | 116.1 | 3.71 | 3.65 / 2.29 / 4.72 | 1.00 |
| bvh_refit | sah | 10000 | 47.1 | 113.9 | 2.42 | 3.56 / 1.65 / 2.42 | 1.00 |
| bvh_refit | sah | 30000 | 115.5 | 104.7 | 0.91 | 1.40 / 0.70 / 0.92 | 1.00 |
| bvh_refit | sah | 100000 | 336.0 | 106.7 | 0.32 | 0.47 / 0.27 / 0.30 | 1.00 |
| bvh_refit | sah | 1000000 | 3729.8 | 280.5 | 0.08 | 0.08 / 0.06 / 0.08 | 1.00 |
| mesh_refit | lbvh | 1000 | 50.5 | 95.5 | 1.89 | 2.23 / 1.59 / 1.89 | 1.00 |
| mesh_refit | lbvh | 3000 | 60.9 | 140.8 | 2.31 | 2.31 / 1.80 / 2.54 | 0.99 |
| mesh_refit | lbvh | 10000 | 83.7 | 139.0 | 1.66 | 2.07 / 1.49 / 1.55 | 1.00 |
| mesh_refit | lbvh | 30000 | 174.0 | 138.6 | 0.80 | 1.01 / 0.75 / 0.84 | 1.00 |
| mesh_refit | lbvh | 100000 | 463.2 | 150.9 | 0.33 | 0.37 / 0.34 / 0.33 | 1.00 |
| mesh_refit | lbvh | 1000000 | 4268.6 | 363.1 | 0.09 | 0.08 / 0.09 / 0.09 | 1.00 |
| mesh_refit | lbvh | 10000000 | 44663.1 | 3049.7 | 0.07 | 0.06 / 0.07 / 0.07 | 1.00 |
| mesh_refit | median | 1000 | 34.4 | 77.1 | 2.24 | 4.60 / 2.21 / 1.91 | 1.00 |
| mesh_refit | median | 3000 | 43.8 | 158.5 | 3.62 | 4.61 / 1.85 / 2.27 | 0.99 |
| mesh_refit | median | 10000 | 49.8 | 123.0 | 2.47 | 3.53 / 1.97 / 1.88 | 1.00 |
| mesh_refit | median | 30000 | 58.0 | 124.3 | 2.15 | 2.90 / 1.63 / 2.22 | 0.99 |
| mesh_refit | median | 100000 | 120.3 | 130.6 | 1.09 | 1.42 / 0.78 / 1.09 | 0.99 |
| mesh_refit | median | 1000000 | 799.2 | 213.7 | 0.27 | 0.27 / 0.27 / 0.26 | 0.97 |
| mesh_refit | sah | 1000 | 33.8 | 81.8 | 2.42 | 4.31 / 2.40 / 2.27 | 1.00 |
| mesh_refit | sah | 3000 | 41.4 | 133.6 | 3.23 | 3.23 / 2.16 / 3.62 | 1.00 |
| mesh_refit | sah | 10000 | 48.9 | 123.1 | 2.52 | 3.32 / 2.04 / 2.45 | 1.00 |
| mesh_refit | sah | 30000 | 61.2 | 125.5 | 2.05 | 2.88 / 1.69 / 2.05 | 1.00 |
| mesh_refit | sah | 100000 | 122.2 | 126.4 | 1.03 | 1.48 / 0.85 / 1.06 | 1.00 |
| mesh_refit | sah | 1000000 | 1278.1 | 240.9 | 0.19 | 0.19 / 0.17 / 0.19 | 1.00 |

Large trees: 13-15x faster at 10^6 and 10^7 (Bvh lbvh 10^7 43.6 -> 3.2 ms); the noise floor stayed at 0.97-1.00. Small trees: the levels cost about 0.1-0.15 ms (about 50 launches) and lose to the walk below about 20-30k leaves for leaf-size-1 trees; meshes built on the host break even near 10^5 triangles (their default leaf size gives about 25-30k leaves there). Threshold chosen on leaves, 32768: every measured cell is then the walk's time or better.

### Tests

test_bvh 13 OK, test_grouped_bvh 11 OK, test_mesh* 49 OK, test_graph 35 OK (3 skipped), test_capture_mode 2 OK, test_conditional_captures 32 OK (16 skipped), test_array_fill_capture 3 OK, test_apic_mesh 12 OK. Full default suite from a fresh kernel cache: **Ran 7558 tests in 509.9 s, OK (skipped=81)** (the fem banner isolation failures did not occur this run). CUDA: `cuda_pp.sh` with the change vs `e92f2611`: 22 identical, bvh.cu's two LINE-ONLY; nvcc 12.8 build of the change (worktree `agent_space/wcuda_s24`): "Warp build succeeded", warp.so 199,765,024 bytes. clang-format applied; `jargon.py --port` clean.

### Question for Jeff

10. Meshes created with winding-number support (`support_winding_number=True`) refit through `bvh_refit_with_solid_angle_kernel`, the same fence-bound walk with solid-angle data. Options: (a) leave them on the walk (smallest diff; only those meshes); (b) a depth-level version of that kernel too (about 40 more lines, HIP-only).

### Next step (head `f6ada527`)

- Not moved to `ported`; more rulings pending. The wheel validated at `7059a340` stays stale.
- Validators revalidate at `f6ada527` (BVH refit on HIP changed for trees with 32768+ leaves).

### Scripts

`agent_space/bench/refit_check.py`, `refit_query.py`, `refit_perf.py`, `rf_table.py`:

`refit_check.py`:
```
# usage: refit_check.py <out.npz> [sizes...]; deterministic Bvh trees (lbvh/sah/median, with and without groups, leaf_size 1 and 4) refit
# three times with moved boxes, then rebuilt and refit, then refit inside a captured graph relaunched three times;
# after each step the BVH node arrays (lowers, uppers as raw uint32 words) are saved for comparison across builds of
# Warp. The BVH descriptor is read from device memory at bvh.id: node_lowers, node_uppers, node_parents, node_counts,
# primitive_indices (pointers), then max_depth, max_nodes, num_nodes, num_leaf_nodes (int32).
import sys
import numpy as np
import warp as wp
wp.config.quiet = True
wp.init()
OUT = sys.argv[1]
SIZES = [int(float(s)) for s in sys.argv[2:]] or [10**3, 10**4, 10**5, 10**6, 10**7]
dev = wp.get_device("cuda:0")

@wp.kernel
def move(lo0: wp.array(dtype=wp.vec3), hi0: wp.array(dtype=wp.vec3), step: float, lo: wp.array(dtype=wp.vec3), hi: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    d = wp.vec3(wp.sin(float(i) * 0.37 + step), wp.cos(float(i) * 0.11 + 2.0 * step), wp.sin(float(i) * 0.05 - step)) * 0.01
    lo[i] = lo0[i] + d
    hi[i] = hi0[i] + d

def nodes(bvh):
    desc = wp.array(ptr=bvh.id, dtype=wp.uint64, shape=(5,), device=dev, copy=False).numpy()
    ints = wp.array(ptr=bvh.id + 40, dtype=wp.int32, shape=(4,), device=dev, copy=False).numpy()
    max_depth, max_nodes, num_nodes, num_leaf = (int(v) for v in ints)
    lo = wp.array(ptr=int(desc[0]), dtype=wp.uint32, shape=(max_nodes, 4), device=dev, copy=False).numpy().copy()
    hi = wp.array(ptr=int(desc[1]), dtype=wp.uint32, shape=(max_nodes, 4), device=dev, copy=False).numpy().copy()
    return lo, hi, num_nodes, max_nodes

rng = np.random.default_rng(0)
saved = {}
for N in SIZES:
    c = rng.uniform(0, 1, (N, 3)).astype(np.float32)
    h = rng.uniform(0.0005, 0.002, (N, 3)).astype(np.float32)
    lo0 = wp.array(c - h, dtype=wp.vec3); hi0 = wp.array(c + h, dtype=wp.vec3)
    groups = wp.array((np.arange(N) * 7 // N).astype(np.int32), dtype=int)
    for con in ("lbvh", "sah", "median"):
        if con != "lbvh" and N > 10**6:
            continue  # host builds of 10^7 boxes take minutes with groups
        for g, leaf in ((None, 1), (groups, 1), (None, 4), (groups, 4)):
            key = f"{N}_{con}_{'g' if g is not None else 'n'}_leaf{leaf}"
            lo = wp.clone(lo0); hi = wp.clone(hi0)
            if g is not None:
                bvh = wp.Bvh(lo, hi, constructor=con, groups=g, leaf_size=leaf)
            else:
                bvh = wp.Bvh(lo, hi, constructor=con, leaf_size=leaf)
            for step in range(3):
                wp.launch(move, dim=N, inputs=[lo0, hi0, float(step + 1), lo, hi])
                bvh.refit()
                a, b, nn, mx = nodes(bvh)
                k = nn if con != "lbvh" else mx
                saved[f"{key}_refit{step}_lo"] = a[:k]; saved[f"{key}_refit{step}_hi"] = b[:k]
            bvh.rebuild()
            wp.launch(move, dim=N, inputs=[lo0, hi0, 7.0, lo, hi])
            bvh.refit()
            a, b, nn, mx = nodes(bvh)
            saved[f"{key}_rebuild_refit_lo"] = a; saved[f"{key}_rebuild_refit_hi"] = b
            with wp.ScopedCapture(device=dev, force_module_load=False) as cap:
                bvh.refit()
            for step in range(3):
                wp.launch(move, dim=N, inputs=[lo0, hi0, 10.0 + step, lo, hi])
                wp.capture_launch(cap.graph)
            a, b, nn, mx = nodes(bvh)
            saved[f"{key}_graph_lo"] = a; saved[f"{key}_graph_hi"] = b
            del bvh
    print("done", N, flush=True)
np.savez_compressed(OUT, **saved)
```

`refit_query.py`:
```
# usage: refit_query.py <out.npz>; meshes (lbvh/sah/median, 10^5 and 10^6 triangles) and Bvhs whose points or boxes are
# moved and refit three times, then 10^6 point, ray and AABB queries; results saved for comparison across builds
import sys
import numpy as np
import warp as wp
from bench_bvh_geom import grid_mesh
wp.config.quiet = True
wp.init()

@wp.kernel
def move_pts(p0: wp.array(dtype=wp.vec3), step: float, p: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    p[i] = p0[i] + wp.vec3(0.0, 0.0, 0.01 * wp.sin(float(i) * 0.01 + step))

@wp.kernel
def q(mesh: wp.uint64, qp: wp.array(dtype=wp.vec3), ro: wp.array(dtype=wp.vec3), rd: wp.array(dtype=wp.vec3), d: wp.array(dtype=float), t: wp.array(dtype=float)):
    i = wp.tid()
    a = wp.mesh_query_point_no_sign(mesh, qp[i], 10.0)
    d[i] = wp.where(a.result, a.u + 2.0 * a.v + float(a.face), -1.0)
    b = wp.mesh_query_ray(mesh, ro[i], rd[i], 10.0)
    t[i] = wp.where(b.result, b.t, -1.0)

@wp.kernel
def qa(bvh: wp.uint64, lo: wp.array(dtype=wp.vec3), hi: wp.array(dtype=wp.vec3), c: wp.array(dtype=int)):
    i = wp.tid()
    it = wp.bvh_query_aabb(bvh, lo[i], hi[i])
    k = int(0)
    n = int(0)
    while wp.bvh_query_next(it, k):
        n += 1 + k % 3
    c[i] = n

rng = np.random.default_rng(0)
NQ = 10**6
qp = wp.array(rng.uniform(0, 1, (NQ, 3)).astype(np.float32) * [1, 1, 0.2], dtype=wp.vec3)
ro = wp.array(np.concatenate([rng.uniform(0, 1, (NQ, 2)), np.ones((NQ, 1))], 1).astype(np.float32), dtype=wp.vec3)
rd = wp.array(np.tile(np.array([[0.05, 0.02, -1.0]], np.float32), (NQ, 1)), dtype=wp.vec3)
d = wp.zeros(NQ, dtype=float); t = wp.zeros(NQ, dtype=float); cnt = wp.zeros(NQ, dtype=int)
saved = {}
for N in (10**5, 10**6):
    pts_np, ind_np, r = grid_mesh(N, rng)
    p0 = wp.array(pts_np, dtype=wp.vec3)
    for con in ("lbvh", "sah", "median"):
        p = wp.clone(p0)
        mesh = wp.Mesh(points=p, indices=wp.array(ind_np, dtype=int), bvh_constructor=con)
        for step in range(3):
            wp.launch(move_pts, dim=len(pts_np), inputs=[p0, float(step), p])
            mesh.refit()
        wp.launch(q, dim=NQ, inputs=[mesh.id, qp, ro, rd, d, t])
        saved[f"mesh_{N}_{con}_d"] = d.numpy().copy(); saved[f"mesh_{N}_{con}_t"] = t.numpy().copy()
        tri = mesh.points.numpy()[ind_np.reshape(-1, 3)]
        lo = wp.array(tri.min(axis=1), dtype=wp.vec3); hi = wp.array(tri.max(axis=1), dtype=wp.vec3)
        bvh = wp.Bvh(lo, hi, constructor=con)
        bvh.refit()
        c = rng.uniform(0, 1, (NQ, 2)).astype(np.float32)
        qlo = wp.array(np.concatenate([c - 2.0 / r, np.full((NQ, 1), -1.0, np.float32)], 1), dtype=wp.vec3)
        qhi = wp.array(np.concatenate([c + 2.0 / r, np.full((NQ, 1), 1.0, np.float32)], 1), dtype=wp.vec3)
        wp.launch(qa, dim=NQ, inputs=[bvh.id, qlo, qhi, cnt])
        saved[f"bvh_{N}_{con}_aabb"] = cnt.numpy().copy()
np.savez_compressed(sys.argv[1], **saved)
print("saved", len(saved))
```

`refit_perf.py`:
```
# usage: refit_perf.py [sizes...]; per size and constructor, wall time per wp.Bvh.refit() and wp.Mesh.refit() (lbvh, sah,
# median; 10 warm-up calls, then 7 x <reps> back-to-back calls and a sync, median), plus a noise-floor metric whose
# code is the same in every variant: one 10^6-point mesh_query_point launch on the same mesh. Records via common.record.
import statistics, sys, time
import numpy as np
import warp as wp
from bench_bvh_geom import grid_mesh
from common import record
wp.config.quiet = True
wp.init()

@wp.kernel
def qpoint(mesh: wp.uint64, qp: wp.array(dtype=wp.vec3), dist: wp.array(dtype=float)):
    t = wp.tid()
    q = wp.mesh_query_point_no_sign(mesh, qp[t], 10.0)
    dist[t] = wp.where(q.result, q.u, -1.0)

def per_call(fn, reps):
    for _ in range(10):
        fn()
    wp.synchronize_device()
    res = []
    for _ in range(7):
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        wp.synchronize_device()
        res.append((time.perf_counter() - t0) / reps)
    return statistics.median(res), min(res), max(res)

rng = np.random.default_rng(0)
NQ = 10**6
qp = wp.array(rng.uniform(0, 1, (NQ, 3)).astype(np.float32), dtype=wp.vec3); dist = wp.zeros(NQ, dtype=float)
for N in [int(float(s)) for s in sys.argv[1:]] or [10**3, 3000, 10**4, 30000, 10**5, 10**6, 10**7]:
    pts_np, ind_np, r = grid_mesh(N, rng)
    pts = wp.array(pts_np, dtype=wp.vec3); ind = wp.array(ind_np, dtype=int)
    tri = pts_np[ind_np.reshape(-1, 3)]
    lo = wp.array(tri.min(axis=1), dtype=wp.vec3); hi = wp.array(tri.max(axis=1), dtype=wp.vec3)
    reps = 200 if N <= 10**5 else (30 if N <= 10**6 else 5)
    for con in ("lbvh", "sah", "median"):
        if con != "lbvh" and N > 10**6:
            continue
        bvh = wp.Bvh(lo, hi, constructor=con)
        m, mn, mx = per_call(bvh.refit, reps)
        record(area="refit", op="bvh_refit", N=N, constructor=con, med=m, min=mn, max=mx, reps=7)
        del bvh
        mesh = wp.Mesh(points=pts, indices=ind, bvh_constructor=con)
        m, mn, mx = per_call(mesh.refit, reps)
        record(area="refit", op="mesh_refit", N=N, constructor=con, med=m, min=mn, max=mx, reps=7)
        m, mn, mx = per_call(lambda: wp.launch(qpoint, dim=NQ, inputs=[mesh.id, qp, dist]), 5)
        record(area="refit", op="noise_query_point", N=N, constructor=con, med=m, min=mn, max=mx, reps=7)
        del mesh
```

`rf_table.py`:
```
# usage: rf_table.py <jsonl> [treeA treeB]; per op/constructor/N: median over rounds of each run's median for A and B (us),
# B/A, the spread of B/A over the paired rounds, and the noise-floor ratio (noise_query_point B/A) for the same cell
import json, statistics, sys
from collections import defaultdict
rows = [json.loads(l) for l in open(sys.argv[1])]
A, B = (sys.argv[2:4] + ["rfwalk", "rfdepth"])[:2] if len(sys.argv) > 3 else ("rfwalk", "rfdepth")
d = defaultdict(list)
for r in rows:
    d[(r["op"], r["constructor"], r["N"], r["tree"])].append(r["med"])
keys = sorted({(r["op"], r["constructor"], r["N"]) for r in rows if r["op"] != "noise_query_point"}, key=lambda k: (k[0], k[1], k[2]))
print(f"| op | constructor | N | {A} us | {B} us | {B}/{A} | per-round ratios | noise floor {B}/{A} |")
print("|---|---|---|---|---|---|---|---|")
for op, con, N in keys:
    a = d[(op, con, N, A)]; b = d[(op, con, N, B)]
    na = d[("noise_query_point", con, N, A)]; nb = d[("noise_query_point", con, N, B)]
    if not a or not b: continue
    per = " / ".join(f"{y / x:.2f}" for x, y in zip(a, b))
    nr = statistics.median(nb) / statistics.median(na) if na and nb else float("nan")
    print(f"| {op} | {con} | {N} | {statistics.median(a)*1e6:.1f} | {statistics.median(b)*1e6:.1f} | {statistics.median(b)/statistics.median(a):.2f} | {per} | {nr:.2f} |")
```

## Ruling on session 24's question 10 (jeff, 2026-11-20)

"(b) Convert that kernel too": winding-number meshes get the depth-level refit as well. Queued for the porter after the fem isolation fix, followed by the final build (prompt `agent_space/p6ev/winding-final-porter.txt`).

## Porting session 25 (2026-11-20, linux-gfx950 MI355X, porter): fem test isolation

Ruling "fix the fem test isolation in the fork" implemented as fork commit `bc1ae44b` "Initialize Warp before capturing a test's output" (warp/tests/unittest_utils.py +3; no [ROCm] prefix, shared test code). head_sha advanced to it; stage stays `porting`.

Mechanism: test_fem_quadrature.py and test_fem_shape.py register every test with `add_function_test(...)` and no devices, so importing them never calls `wp.init()`. unittest orders tests alphabetically, so `test_gimp_quadrature` and `test_cube_shape_functions` run first in their classes, and their first Warp call initializes Warp inside `CheckOutput`'s `StdOutCapture`; the init greeting is then "Unexpected output". The failure appears whenever that file is the first Warp user in a process (alone, or first in a parallel worker), which is why it depended on scheduling. Fix: `CheckOutput.__enter__` calls `wp.init()` before `capture.begin()` (a no-op once initialized; the output check is unchanged).

Same-hazard scan (modules that do not initialize Warp on import): interop/test_jax, test_assert, test_builtins_resolution, test_coloring, test_compilation, test_composite_component_adjoint, test_context, test_cpu_precompiled_headers, test_diagnostics, test_implicit_init, test_kernel_cache, test_logger, test_module_parallel_load, test_version. Most have no `add_function_test`; test_implicit_init uses `check_output=False` throughout (its subject is implicit init, untouched by the fix); test_builtins_resolution passes alone. Only the two fem files could fail this way; the fix in the shared utility covers any future case too.

Evidence (HIP_VISIBLE_DEVICES=1, source build in the fork clone, `python -m warp.tests -s autodetect -p <file>`):

| run | before | after |
|---|---|---|
| test_fem_quadrature.py, GPU | FAILED (1) | 5 OK |
| test_fem_quadrature.py, CPU only (HIP_VISIBLE_DEVICES=-1) | FAILED (1) | 5 OK |
| test_fem_shape.py, GPU | FAILED (1) | 4 OK |
| test_fem_shape.py, CPU only | FAILED (1) | 4 OK |

- `-p "test_fem*.py"`: Ran 107 tests in 952 s, 1 error: `test_fem.example_kelvin_helmholtz_cuda_0` subprocess TimeoutExpired at 600 s while the host load average was about 28 with other processes on the GPU. The same test passed in the full run below, so this was load, not the change.
- Full default suite from a fresh kernel cache (`agent_space/suite.sh`, cache removed first): Ran 7558 tests in 2321.7 s (wall 2326 s), OK (skipped=81). test_gimp_quadrature, test_cube_shape_functions and kelvin_helmholtz all ok.
- jargon: clean. nvcc check not needed (Python-only change).

Next step: the release wheel is stale (built before f6ada527 and bc1ae44b). Question 10 has since been ruled (b), winding-number meshes get the depth-level refit; that porter round and then the final build are next.


## Porting session 26 (2026-11-20, linux-gfx950 MI355X, porter): winding-number refit stopped before committing; release notes draft

Ruling on question 10 (b): convert `bvh_refit_with_solid_angle_kernel` (mesh.cu) to the depth-level scheme, HIP-only, with the condition "if bit-identity is impossible because the walk's own result depends on thread order ..., stop and report with the evidence instead of committing". **Stopped: no fork commit, head stays `bc1ae44b`, stage stays `porting`.** Because the head is not final, the final step (nvcc check, release build, wheel, suite from the wheel, `ported`) was not run. The release notes draft was updated (below), since it does not depend on the head. `HIP_VISIBLE_DEVICES=1` (amd-smi GPU 0), `ulimit -c 0`; no process was stopped by pattern; the fork clone is clean at `bc1ae44b` (the change is kept as a patch below).

### What was built

New HIP-only kernel `refit_tree_with_solid_angle_at_depth` (mesh.cu), structured like f6ada527's `refit_tree_at_depth`: one thread per node of one depth; a node with `b` set computes its triangles' `SolidAngleProps` (the walk's loop, verbatim) unless its parent is packed, and writes the box to the node bounds; an internal node takes min/max of its children and `combine_precomputed_solid_angle_props(props[node], left, right-or-NULL)`. Depths come from `node_counts`, the same as for `bvh_refit_device`; the level count is factored out of bvh.cu into `bvh_refit_max_depth(const BVH&)` (declared in bvh.h, HIP-only), so both refits launch the same fixed number of levels and stay capturable. The patch is at the end of this section.

### Correctness: node bounds and queries identical; solid angle data not bit-identical

`sa_check.py` (below): closed noisy torus meshes with `support_winding_number=True`, about 10^3, 10^4, 10^5, 10^6, 10^7 triangles; lbvh at every size, sah/median to 10^6; without and with 7 groups; leaf sizes 1 and 4; steps: creation (which refits), three refits with moved points, `mesh.points = <new array>` (refits; a Mesh has no `rebuild()`), a new mesh on the moved points, and a refit captured in a graph and relaunched three times. After each step the node arrays and the solid angle array are read from device memory and hashed. Then 10^5 `mesh_query_point_sign_winding_number` (the only query that reads the solid angle data), `mesh_query_point_no_sign` and `mesh_query_ray` results are hashed. Every solid angle row changes between consecutive refits; 25-27% of the query points are inside. The same build (scratch worktree `agent_space/wrf2`, with a scratch environment switch `WP_SA_REFIT_MIN` forcing either path) is run on the walk and on the levels: 52 configurations, 1248 arrays.

- Node lowers/uppers: **0 of 728 differ.** Query results (winding-number sign, no-sign, ray): **0 of 156 differ.**
- Solid angle arrays: **196 of 364 differ**: every leaf-size-4 configuration at every size (leaves holding 2 to 4 triangles, combined inside the leaf), and at 10^7 also the leaf-size-1 lbvh trees (which have some packed leaves at that size). Single-triangle leaves are identical. The differences are in the last bits (relative 7e-8 to 3e-7) of the accumulated moment fields (`sum_permute_n_xyz`, `two_n_*`, `n_xy`, `n_yx`, `n_zy`, `n_xz`), and they propagate to the ancestors.
- Excluded from every comparison, because it is indeterminate in every build: `max_p_dist_sq` of single-triangle leaves. `precompute_triangle_solid_angle_props` never sets it, so the walk stores whatever was in the register (walk against walk, two runs: it differs in 75 of 75 single-triangle leaves at 10^3; 308 of 364 snapshots differ in every pairing). Nothing reads it (`evaluate_node_solid_angle` is called only on internal nodes). This is an upstream quirk and harmless. Entries the refit never writes (leaves below packed leaves, nodes past `num_nodes` of host-built trees) are also excluded.

### Cause: compiler contraction, not thread order

- The library's device code is compiled with clang's HIP default, which contracts multiply-adds across statements (`-ffp-contract=fast-honor-pragmas`; session 22). Which products get fused depends on each kernel's code after inlining.
- Test: the same scratch tree rebuilt with `-ffp-contract=off` for the library (scratch worktree `agent_space/wrf3`, build-option hack): walk and levels then match bit for bit (lbvh n 4, sah n 4, median g 4 at 10^3; create and refit steps). The device IR of the normal build carries `contract` on the fmul/fadd/fdiv in both kernels. A first attempt with a file-scope `#pragma clang fp contract(off)` in mesh.cu was ineffective: the builtin.h region from e92f2611 ends with `STDC FP_CONTRACT DEFAULT`, which restores the command-line mode.
- **The walk is deterministic** (walk against walk: identical apart from the indeterminate field). It has no thread-order dependence. In an LBVH tree the leaf threads cover indices 0..n-1 and packed leaves are internal indices (992..1978 in the 10^3 tree), so no node is computed by both of the walk's code paths.
- **But the walk is not self-consistent.** It holds two inlined copies of the leaf arithmetic: the leaf-thread branch, and the climb branch that recomputes packed leaves. The compiler fused the two copies differently. Evidence: a variant (`agent_space/wrf4`) that moves the leaf arithmetic and the internal combine into `__noinline__` device functions called by both kernels reproduces today's walk bit for bit on every sah and median tree (leaf-thread branch) and on lbvh leaf-size-1 trees to 10^6, but differs from it on lbvh packed leaves (climb branch): 84 of 1248 arrays, solid angle only, queries identical. Today's walk therefore gives a leaf in the leaf array and a packed leaf with the same triangles different last bits. A new kernel can match the walk only by reproducing the compiler's fusion choices for both copies, which nothing guarantees. Bit-identity with today's walk is impossible by construction.
- With the shared helpers (wrf4), **walk and levels are bit-identical: 0 of 1248 arrays differ** (all 52 configurations, 10^3 to 10^7, every step, every query).

### Performance (`sa_perf.py`, 3 interleaved rounds of `sawalk` / `salevels` (wrf2, the inlined kernel forced onto each path) and `sa4walk` / `sa4levels` (wrf4, the shared helpers); `wp.Mesh.refit()` of a winding-number torus mesh with the default leaf size (4), per call, 10 warm-up then 7 x N back-to-back calls and a sync; noise floor = a 10^6-point `mesh_query_point_no_sign` launch on the same mesh, unchanged code; GPU idle before each run)

| constructor | N | leaves | walk us | levels us | levels/walk | per-round levels/walk | shared-helper walk us | shared-helper levels us | helper walk / walk | helper levels / levels | noise floor max/min |
|---|---|---|---|---|---|---|---|---|---|---|---|
| lbvh | 1000 | 990 | 158.4 | 292.1 | 1.84 | 1.84 / 1.86 / 1.87 | 171.9 | 306.4 | 1.09 | 1.05 | 1.01 |
| lbvh | 3000 | 3002 | 173.0 | 306.5 | 1.77 | 1.77 / 1.77 / 1.80 | 187.5 | 316.8 | 1.08 | 1.03 | 1.01 |
| lbvh | 10000 | 10010 | 203.3 | 384.2 | 1.89 | 1.90 / 1.87 / 1.91 | 220.1 | 398.0 | 1.08 | 1.04 | 1.02 |
| lbvh | 30000 | 30012 | 300.5 | 462.8 | 1.54 | 1.56 / 1.54 / 1.54 | 324.4 | 481.6 | 1.08 | 1.04 | 1.01 |
| lbvh | 45000 | 45000 | 378.6 | 481.5 | 1.27 | 1.26 / 1.27 / 1.27 | 408.5 | 506.0 | 1.08 | 1.05 | 1.01 |
| lbvh | 65000 | 64980 | 498.9 | 496.3 | 0.99 | 0.99 / 0.99 / 1.00 | 538.5 | 535.8 | 1.08 | 1.08 | 1.01 |
| lbvh | 100000 | 99904 | 763.8 | 570.0 | 0.75 | 0.74 / 0.75 / 0.75 | 805.2 | 598.9 | 1.05 | 1.05 | 1.01 |
| lbvh | 150000 | 149878 | 1126.1 | 709.3 | 0.63 | 0.63 / 0.63 / 0.63 | 1186.2 | 699.2 | 1.05 | 0.99 | 1.00 |
| lbvh | 200000 | 200028 | 1444.5 | 806.5 | 0.56 | 0.56 / 0.56 / 0.56 | 1519.7 | 768.0 | 1.05 | 0.95 | 1.01 |
| lbvh | 300000 | 299924 | 2170.4 | 994.1 | 0.46 | 0.46 / 0.46 / 0.46 | 2297.8 | 902.7 | 1.06 | 0.91 | 1.01 |
| lbvh | 1000000 | 999696 | 7332.1 | 2309.9 | 0.32 | 0.32 / 0.31 / 0.32 | 7922.2 | 1931.9 | 1.08 | 0.84 | 1.01 |
| lbvh | 10000000 | 9999392 | 79379.1 | 23838.9 | 0.30 | 0.30 / 0.30 / 0.30 | 81855.6 | 18110.8 | 1.03 | 0.76 | 1.01 |
| sah | 1000 | 308 | 94.7 | 192.7 | 2.03 | 2.05 / 2.03 / 1.94 | 99.8 | 206.8 | 1.05 | 1.07 | 1.01 |
| sah | 3000 | 920 | 107.1 | 207.9 | 1.94 | 1.94 / 2.39 / 1.95 | 111.5 | 222.0 | 1.04 | 1.07 | 1.01 |
| sah | 10000 | 3032 | 129.1 | 236.0 | 1.83 | 1.86 / 1.83 / 1.80 | 134.6 | 251.9 | 1.04 | 1.07 | 1.02 |
| sah | 30000 | 9157 | 159.8 | 248.9 | 1.56 | 1.56 / 1.56 / 1.54 | 164.4 | 263.2 | 1.03 | 1.06 | 1.01 |
| sah | 45000 | 13669 | 180.5 | 259.5 | 1.44 | 1.41 / 1.44 / 1.44 | 185.6 | 269.5 | 1.03 | 1.04 | 1.01 |
| sah | 65000 | 19680 | 210.1 | 259.1 | 1.23 | 1.23 / 1.22 / 1.23 | 213.5 | 270.9 | 1.02 | 1.05 | 1.02 |
| sah | 100000 | 30377 | 267.4 | 289.4 | 1.08 | 1.09 / 1.08 / 1.08 | 271.1 | 296.5 | 1.01 | 1.02 | 1.01 |
| sah | 150000 | 45515 | 353.4 | 298.8 | 0.85 | 0.84 / 0.85 / 0.85 | 361.0 | 318.2 | 1.02 | 1.07 | 1.01 |
| sah | 200000 | 60886 | 441.8 | 321.8 | 0.73 | 0.73 / 0.73 / 0.72 | 439.3 | 339.5 | 0.99 | 1.05 | 1.01 |
| sah | 300000 | 90813 | 663.6 | 361.4 | 0.54 | 0.55 / 0.54 / 0.54 | 662.7 | 385.0 | 1.00 | 1.07 | 1.01 |
| sah | 1000000 | 303618 | 2069.5 | 708.7 | 0.34 | 0.34 / 0.34 / 0.34 | 2065.2 | 667.3 | 1.00 | 0.94 | 1.00 |
| median | 1000 | 256 | 83.3 | 143.2 | 1.72 | 1.73 / 1.72 / 1.68 | 87.3 | 148.4 | 1.05 | 1.04 | 1.01 |
| median | 3000 | 1024 | 92.6 | 156.8 | 1.69 | 1.69 / 1.66 / 1.70 | 98.1 | 159.2 | 1.06 | 1.02 | 1.01 |
| median | 10000 | 3866 | 125.9 | 194.3 | 1.54 | 1.57 / 1.54 / 1.52 | 130.8 | 200.6 | 1.04 | 1.03 | 1.02 |
| median | 30000 | 8192 | 138.1 | 184.2 | 1.33 | 1.34 / 1.32 / 1.33 | 142.8 | 186.6 | 1.03 | 1.01 | 1.01 |
| median | 45000 | 16384 | 169.2 | 189.5 | 1.12 | 1.12 / 1.12 / 1.12 | 172.8 | 190.8 | 1.02 | 1.01 | 1.01 |
| median | 65000 | 16384 | 173.4 | 193.5 | 1.12 | 1.11 / 1.13 / 1.12 | 176.2 | 194.9 | 1.02 | 1.01 | 1.01 |
| median | 100000 | 32768 | 251.4 | 215.9 | 0.86 | 0.86 / 0.86 / 0.86 | 255.8 | 208.2 | 1.02 | 0.96 | 1.01 |
| median | 150000 | 51574 | 365.2 | 244.3 | 0.67 | 0.68 / 0.67 / 0.67 | 367.8 | 246.7 | 1.01 | 1.01 | 1.00 |
| median | 200000 | 65536 | 418.7 | 245.5 | 0.59 | 0.59 / 0.59 / 0.59 | 421.4 | 237.5 | 1.01 | 0.97 | 1.00 |
| median | 300000 | 103316 | 696.3 | 302.3 | 0.43 | 0.43 / 0.43 / 0.44 | 701.6 | 308.0 | 1.01 | 1.02 | 1.01 |
| median | 1000000 | 262144 | 1570.1 | 588.7 | 0.37 | 0.37 / 0.38 / 0.37 | 1551.0 | 488.0 | 0.99 | 0.83 | 1.01 |

- Large meshes: the levels take 0.30x the walk's time at 10^7 (79.4 -> 23.8 ms, lbvh) and 0.32-0.37x at 10^6. The gain is smaller than for plain refits (0.07x) because the solid angle arithmetic, not the fence, is a larger share here. Small meshes: the levels cost 0.1-0.15 ms more (about 50 launches plus heavier nodes).
- Break-even, in leaves: lbvh about 65k (0.99 at 64,980), sah 30k-45k (1.08 at 30,377, 0.85 at 45,515), median 16k-33k (1.12 at 16,384, 0.86 at 32,768). An LBVH tree's `num_leaf_nodes` equals its item count, packed or not, so leaves measure lbvh trees differently from host-built ones. In items, break-even is about 65k for lbvh, 80k for median and 120k for sah.
- Threshold choices against the measured cells: `num_leaf_nodes >= 32768` (f6ada527's rule) puts lbvh 45,000 on the levels at 1.27x; `num_leaf_nodes >= 65536` leaves sah 45-60k and median 33-52k leaves on the walk at up to 1.5x its levels time; **`num_items >= 65536` keeps every measured cell within 8% of the faster path** (worst: sah 99,904 items on the levels at 1.08x).
- Shared helpers: the walk costs 3-9% more on lbvh and up to 6% on sah/median (non-inlined calls). Their levels are 1-8% slower than the inlined levels up to 300k triangles (except lbvh from 150k) and faster at 10^6 and above (0.76x at 10^7 lbvh).

### Question for Jeff (no decision taken; not committed)

11. The ruling's condition holds in substance. The walk's results are deterministic, but they depend on the compiler's per-kernel contraction choices, and the walk's two copies of the leaf arithmetic already disagree with each other. So no new kernel can be bit-identical to it by construction. Options:
   (a) Shared `__noinline__` helpers for the leaf arithmetic and the internal combine, called by both the walk and the level kernel on HIP (CUDA keeps today's inlined code under `#else`, so that nvcc's fusion is unchanged). Walk and levels are then bit-identical in all 1248 arrays. Against today's HIP walk: sah/median trees and lbvh leaf-size-1 trees to 10^6 are unchanged; lbvh packed leaves change in the last bits; queries were unchanged in every test. Costs the walk 3-9%; threshold `num_items >= 65536`. The diff is larger than (b): the walk is restructured on HIP and keeps its original form for CUDA.
   (b) Commit the inlined level kernel as built. Node bounds and all query results are identical to the walk's; solid angle moments of multi-triangle leaves differ in the last bits (relative 1e-7), so crossing the threshold changes those bits. Same threshold.
   (c) Leave winding-number meshes on the walk (the session 24 state).
   If (a) or (b): one commit as ruled (correctness rerun on the committed build, the tests listed in the ruling, nvcc check), then the final step.

### Release notes draft (1.14.0+rocm10.2.0), updated

Updated per the ruling: fp contraction now matches CUDA (fp16 unfused as on CUDA), replacing the sentence that described the old `on` setting. The note to Jeff that depended on `on` was removed, and the known issue was added verbatim. No mention of the 1.13 port's defects (`grep 1.13`: none). Body checked with `utils/prose.py`: clean. This supersedes the session 21 draft above; the copy is at `agent_space/release_notes_draft.md`.

Draft for Jeff (written by the porter in session 21, updated in session 26 for the fp-contract ruling and the known issue; nothing published). Text between the fences is the proposed GitHub Release body; one line per paragraph as GitHub renders it.

```
# Warp 1.14.0 for AMD GPUs (ROCm 10.2)

This release is NVIDIA Warp 1.14.0 ported to AMD GPUs with HIP. It is built from the release-1.14 branch of NVIDIA/warp with the changes needed to compile and run Warp's GPU code on AMD Instinct and Radeon GPUs; Warp's Python API, kernels and features are those of upstream 1.14.0, apart from the differences listed below. It is not an NVIDIA release; report problems with this build at https://github.com/AMD-Ecosystem/warp/issues.

## Install

Linux x86-64 (Python 3.10 or newer):

    pip install https://github.com/AMD-Ecosystem/warp/releases/download/v1.14.0+rocm10.2.0/warp_lang-1.14.0+rocm10.2.0-py3-none-manylinux_2_28_x86_64.whl --extra-index-url https://stable.repo.amd.com/rocm/whl-next/

Windows x86-64: the same command with the win_amd64 wheel from this release.

The wheel depends on the ROCm 10.2 Python packages (`rocm>=10.2,<10.3`), which pip installs from AMD's package index given by `--extra-index-url`; no system ROCm installation is needed. Check the install with `python -c "import warp; warp.init()"`, which prints the ROCm and HIP runtime versions and the AMD GPUs found.

## Version numbers

`1.14.0` is the upstream Warp version. The local part `rocm10.2.0` names the ROCm release the wheel is built for and requires (10.2), followed by this build's number for that Warp version and ROCm release. The last number is not a ROCm patch version (AMD's PyTorch wheels use the same shape with that other meaning): a rebuild of 1.14.0 for ROCm 10.2 with fixes becomes 1.14.0+rocm10.2.1, an upstream patch release starts again at 1.14.1+rocm10.2.0, and a build for a later ROCm changes the middle, for example 1.14.0+rocm10.3.0.

## Supported GPUs

The library contains code for every GPU target ROCm 10.2 supports on the operating system: 25 targets on Linux (gfx908, gfx90a, gfx942, gfx950, gfx1010, gfx1011, gfx1012, gfx1030 to gfx1036, gfx1100 to gfx1103, gfx1150 to gfx1153, gfx1200, gfx1201, gfx1250) and 22 on Windows. Kernels written in Python are compiled at run time for the GPU present.

Tested on hardware with Warp's test suite: AMD Instinct MI355X (gfx950), MI300X (gfx942) and MI250X (gfx90a), Radeon Pro W7800 (gfx1100) and Radeon AI PRO R9700 (gfx1201) on Linux, and a Ryzen AI Max (gfx1151) on Windows. The other targets are compiled but have not been run.

## Differences from Warp's CUDA build

- `wp.get_runtime().toolkit_version` and `driver_version` (the CUDA Toolkit and CUDA driver versions) are `None`, as in a CPU-only build. The new `rocm_version` is the ROCm release the library was built with, for example `(10, 2)`, and `hip_version` is the version of the HIP runtime loaded in the process, for example `(7, 17)`; both are `None` in CUDA and CPU builds. Code that compares `toolkit_version` with a tuple without checking for `None` fails on AMD GPUs; use the capability functions (for example `wp.is_conditional_graph_supported()`) instead.
- Conditional graph nodes are not available in HIP: `wp.capture_if` and `wp.capture_while` cannot be captured into a graph on AMD GPUs (`wp.is_conditional_graph_supported()` returns False), as with CUDA before 12.4.
- MathDx (cuBLASDx, cuFFTDx, cuSOLVERDx) is NVIDIA-only and not included. Tile operations use Warp's built-in implementations, as in a CUDA build without MathDx; 2D and 3D `tile_fft`/`tile_ifft` of `vec2f`/`vec2d` tiles, which have no built-in implementation, are not available.
- `tile_matmul` uses AMD matrix instructions (MFMA on Instinct GPUs, WMMA on gfx11 and gfx12 for fp16 and bf16) where the tile shapes allow it. They accumulate fp16 and bf16 products in fp32, so results can differ from the scalar path in the last bits (they are closer to the exact product).
- Shared-memory tiles are limited to 64 KiB per block (tiles plus Warp's small static allocation) on MI250X, MI300X and Radeon GPUs and to 160 KiB on MI355X, less than on recent NVIDIA GPUs; Warp warns when a kernel's request does not fit.
- Textures: MI300X and MI355X have no texture sampling hardware. Textures there are sampled in software on the GPU, with the same results as Warp's CPU textures and slower than hardware sampling. On MI250X, 3D textures are also sampled in software, because HIP does not support linear filtering of 3D textures on that GPU. The CUDA interop handles (`cuda_array`, `cuda_texture`, `cuda_surface`) are not available for textures sampled in software.
- Spin locks contended within one wavefront do not make progress on AMD GPUs (no independent thread scheduling within a wavefront); `warp/examples/core/example_spin_lock.py` reports this instead of hanging.
- In native snippets, `__frsqrt_rn` and `__fsqrt_rn` are within 1 ulp but not correctly rounded on AMD GPUs (wrong in the last bit for about 11 and 15 percent of inputs), unlike CUDA. Warp's own kernels do not use them.
- In native snippets, warp-level intrinsics need 64-bit lane masks: AMD Instinct GPUs have 64-lane wavefronts, so a 32-bit mask such as `0xffffffff` is a compile error. Use `~0ull` or `__activemask()`, which are correct on 32- and 64-lane GPUs, and do not assume `lane = tid & 31`.
- Stream priorities are accepted and reported, but the HIP runtime schedules all streams at normal priority.
- On Windows, a graph launch waits for the graph to finish before returning (it works around a HIP runtime race between successive launches of a graph).
- Floating-point contraction matches CUDA: multiply-adds are fused across statements as with NVRTC's default, and half-precision (fp16) arithmetic is not fused, as on CUDA. Results of the same kernel can still differ from a CUDA build in the last bits, as between any two compilers.
- The precompiled library stores its GPU code compressed; the first use of each part of it in a process costs a few tens of milliseconds while the HIP runtime's code-object cache is empty (later about 2 ms).
- Not included: MathDx, NVRTC and the CUDA runtime. The wheel contains no NVIDIA software, and its licence files list only what it contains.

## Known issues

- On MI300X and MI355X, arrays allocated with the memory pool disabled can lose kernel writes when other processes share the GPU (ROCm/rocm-systems#16355). Keep the memory pool enabled, which is the default.
```

Notes for Jeff on the draft (not part of the body):
- Hardware list: the GPUs in the evidence and porting sessions of these notes (gfx950, gfx942, gfx90a, gfx1100, gfx1201 on Linux; gfx1151 on Windows). The release's gate platforms must validate the uploaded wheels at the final commit before this is true of the release itself; at the time of writing only gfx950 has run the final wheel (session 21).
- Windows wheel name and the 22 Windows targets are from session 13; the Windows wheel has not been built yet.
- Upstream's "requires NVIDIA driver" paragraphs in the docs are untouched; the installation guide's ROCm paragraph already points here.

### Next step (head `bc1ae44b`)

- Jeff rules question 11. Then the porter makes the commit, or skips it under (c). After that comes the final step of the round: nvcc check if a native file changed, release build (25 targets, compressed, `-j 64`), wheel, install into the job venv, full default suite from outside the tree, wheel size/installed size/sha256, and `ported` if every suite result is clean.
- The release wheel validated at `7059a340` is stale.

### Patch (the inlined level kernel, against `bc1ae44b`; `SA_THRESHOLD` was the placeholder for the measured threshold)

```diff
diff --git a/warp/native/bvh.cu b/warp/native/bvh.cu
index 05b91ef1..e4dddf62 100644
--- a/warp/native/bvh.cu
+++ b/warp/native/bvh.cu
@@ -1132,6 +1132,18 @@ void bvh_destroy_device(BVH& bvh)
     bvh.root = NULL;
 }
 
+#if defined(__HIP_PLATFORM_AMD__)
+int bvh_refit_max_depth(const BVH& bvh)
+{
+    int index_bits = 0;
+    while ((1ll << index_bits) < bvh.num_items)
+        index_bits++;
+    // the depth of a host-built tree, or the bound of the tree an in-place rebuild may have made of it; leaves are one
+    // level deeper than the internal nodes
+    return std::max(bvh.max_depth, (bvh.item_groups ? 64 : 30) + index_bits) + 1;
+}
+#endif
+
 void bvh_refit_device(BVH& bvh)
 {
     ContextGuard guard(bvh.context);
@@ -1141,13 +1153,7 @@ void bvh_refit_device(BVH& bvh)
     // time there instead. Its launch count (fixed, so the refit stays capturable) costs more than the walk on small
     // trees.
     if (bvh.num_leaf_nodes >= 32768) {
-        int index_bits = 0;
-        while ((1ll << index_bits) < bvh.num_items)
-            index_bits++;
-        // the depth of a host-built tree, or the bound of the tree an in-place rebuild may have made of it; leaves are
-        // one level deeper than the internal nodes
-        const int max_depth = std::max(bvh.max_depth, (bvh.item_groups ? 64 : 30) + index_bits) + 1;
-        for (int depth = max_depth; depth >= 1; depth--) {
+        for (int depth = bvh_refit_max_depth(bvh); depth >= 1; depth--) {
             wp_launch_device(
                 WP_CURRENT_CONTEXT, refit_tree_at_depth, bvh.max_nodes,
                 (bvh.max_nodes, depth, bvh.node_counts, bvh.node_parents, bvh.primitive_indices, bvh.item_lowers,
diff --git a/warp/native/bvh.h b/warp/native/bvh.h
index 0e122f60..2871bb1b 100644
--- a/warp/native/bvh.h
+++ b/warp/native/bvh.h
@@ -619,6 +619,10 @@ void bvh_create_device(
 );
 void bvh_destroy_device(BVH& bvh);
 void bvh_refit_device(BVH& bvh);
+#if defined(__HIP_PLATFORM_AMD__)
+// the deepest level a depth-level refit of the device tree launches (depths stored in node_counts)
+int bvh_refit_max_depth(const BVH& bvh);
+#endif
 void cubql_bvh_create_device(
     void* context, vec3* lowers, vec3* uppers, int num_items, int leaf_size, CuBQLBVH& bvh_device_on_host
 );
diff --git a/warp/native/mesh.cu b/warp/native/mesh.cu
index edb1ce67..a731b8c9 100644
--- a/warp/native/mesh.cu
+++ b/warp/native/mesh.cu
@@ -219,11 +219,88 @@ __global__ void bvh_refit_with_solid_angle_kernel(
     }
 }
 
+#if defined(__HIP_PLATFORM_AMD__)
+// Refit of the nodes at one depth, deepest level first, giving bvh_refit_with_solid_angle_kernel's results: a leaf or
+// packed leaf takes its triangles' solid angle data unless its parent is a packed leaf; an internal node combines its
+// children's.
+__global__ void refit_tree_with_solid_angle_at_depth(
+    int num_nodes,
+    int depth,
+    const int* __restrict__ depths,
+    const int* __restrict__ parents,
+    BVHPackedNodeHalf* __restrict__ node_lowers,
+    BVHPackedNodeHalf* __restrict__ node_uppers,
+    const vec3* points,
+    const int* indices,
+    const int* primitive_indices,
+    SolidAngleProps* solid_angle_props
+)
+{
+    const int index = blockDim.x * blockIdx.x + threadIdx.x;
+    if (index >= num_nodes || depths[index] != depth)
+        return;
+
+    BVHPackedNodeHalf& lower = node_lowers[index];
+    BVHPackedNodeHalf& upper = node_uppers[index];
+    if (lower.b) {
+        const int parent = parents[index];
+        if (parent != -1 && node_lowers[parent].b)
+            return;
+
+        SolidAngleProps node_solid_angle_props;
+        const int start = lower.i;
+        const int end = upper.i;
+        for (int primitive_counter = start; primitive_counter < end; primitive_counter++) {
+            int primitive_index = primitive_indices[primitive_counter];
+            if (primitive_counter == start) {
+                precompute_triangle_solid_angle_props(
+                    points[indices[primitive_index * 3 + 0]], points[indices[primitive_index * 3 + 1]],
+                    points[indices[primitive_index * 3 + 2]], node_solid_angle_props
+                );
+            } else {
+                SolidAngleProps triangle_solid_angle_props;
+                precompute_triangle_solid_angle_props(
+                    points[indices[primitive_index * 3 + 0]], points[indices[primitive_index * 3 + 1]],
+                    points[indices[primitive_index * 3 + 2]], triangle_solid_angle_props
+                );
+                node_solid_angle_props
+                    = combine_precomputed_solid_angle_props(&node_solid_angle_props, &triangle_solid_angle_props);
+            }
+        }
+
+        (vec3&)lower = node_solid_angle_props.box.lower;
+        (vec3&)upper = node_solid_angle_props.box.upper;
+        solid_angle_props[index] = node_solid_angle_props;
+    } else {
+        const int left_child = lower.i;
+        const int right_child = upper.i;
+        (vec3&)lower = min((vec3&)node_lowers[left_child], (vec3&)node_lowers[right_child]);
+        (vec3&)upper = max((vec3&)node_uppers[left_child], (vec3&)node_uppers[right_child]);
+
+        SolidAngleProps* left_child_data = &solid_angle_props[left_child];
+        SolidAngleProps* right_child_data = (left_child != right_child) ? &solid_angle_props[right_child] : NULL;
+        combine_precomputed_solid_angle_props(solid_angle_props[index], left_child_data, right_child_data);
+    }
+}
+#endif
 
 void bvh_refit_with_solid_angle_device(BVH& bvh, Mesh& mesh)
 {
     ContextGuard guard(bvh.context);
 
+#if defined(__HIP_PLATFORM_AMD__)
+    // as in bvh_refit_device: one depth level at a time on large trees, where the walk's __threadfence() dominates
+    if (bvh.num_leaf_nodes >= SA_THRESHOLD) {
+        for (int depth = bvh_refit_max_depth(bvh); depth >= 1; depth--) {
+            wp_launch_device(
+                WP_CURRENT_CONTEXT, refit_tree_with_solid_angle_at_depth, bvh.max_nodes,
+                (bvh.max_nodes, depth, bvh.node_counts, bvh.node_parents, bvh.node_lowers, bvh.node_uppers,
+                 mesh.points, mesh.indices, bvh.primitive_indices, mesh.solid_angle_props)
+            );
+        }
+        return;
+    }
+#endif
     // clear child counters
     wp_memset_device(WP_CURRENT_CONTEXT, bvh.node_counts, 0, sizeof(int) * bvh.max_nodes);
     wp_launch_device(
```

### The shared-helper variant (option a), as a diff of mesh.cu against the patch above (scratch: no `#else` CUDA branch yet, and the environment switch)

```diff
--- wrf2/warp/native/mesh.cu	2026-11-20 22:11:00.126934287 +0000
+++ wrf4/warp/native/mesh.cu	2026-11-20 22:24:21.770463652 +0000
@@ -59,6 +59,38 @@
     m->average_edge_length = sum_edge_lengths[n - 1] / (3 * n);
 }
 
+#define WP_SA_NOINLINE __noinline__  // SCRATCH variant: one copy of the arithmetic for every refit kernel
+
+__device__ WP_SA_NOINLINE void leaf_solid_angle_props(
+    int start, int end, const vec3* points, const int* indices, const int* primitive_indices, SolidAngleProps& out
+)
+{
+    SolidAngleProps node_solid_angle_props;
+    for (int primitive_counter = start; primitive_counter < end; primitive_counter++) {
+        int primitive_index = primitive_indices[primitive_counter];
+        if (primitive_counter == start) {
+            precompute_triangle_solid_angle_props(
+                points[indices[primitive_index * 3 + 0]], points[indices[primitive_index * 3 + 1]],
+                points[indices[primitive_index * 3 + 2]], node_solid_angle_props
+            );
+        } else {
+            SolidAngleProps triangle_solid_angle_props;
+            precompute_triangle_solid_angle_props(
+                points[indices[primitive_index * 3 + 0]], points[indices[primitive_index * 3 + 1]],
+                points[indices[primitive_index * 3 + 2]], triangle_solid_angle_props
+            );
+            node_solid_angle_props
+                = combine_precomputed_solid_angle_props(&node_solid_angle_props, &triangle_solid_angle_props);
+        }
+    }
+    out = node_solid_angle_props;
+}
+
+__device__ WP_SA_NOINLINE void internal_solid_angle_props(SolidAngleProps* props, int node, int left, int right)
+{
+    combine_precomputed_solid_angle_props(props[node], &props[left], (left != right) ? &props[right] : NULL);
+}
+
 __global__ void bvh_refit_with_solid_angle_kernel(
     int n,
     const int* __restrict__ parents,
@@ -90,29 +122,7 @@
 
             if (true_leaf) {
                 SolidAngleProps node_solid_angle_props;
-
-                const int start = lower.i;
-                const int end = upper.i;
-
-                // loops through primitives in the leaf
-                for (int primitive_counter = start; primitive_counter < end; primitive_counter++) {
-                    int primitive_index = primitive_indices[primitive_counter];
-                    if (primitive_counter == start) {
-                        precompute_triangle_solid_angle_props(
-                            points[indices[primitive_index * 3 + 0]], points[indices[primitive_index * 3 + 1]],
-                            points[indices[primitive_index * 3 + 2]], node_solid_angle_props
-                        );
-                    } else {
-                        SolidAngleProps triangle_solid_angle_props;
-                        precompute_triangle_solid_angle_props(
-                            points[indices[primitive_index * 3 + 0]], points[indices[primitive_index * 3 + 1]],
-                            points[indices[primitive_index * 3 + 2]], triangle_solid_angle_props
-                        );
-                        node_solid_angle_props = combine_precomputed_solid_angle_props(
-                            &node_solid_angle_props, &triangle_solid_angle_props
-                        );
-                    }
-                }
+                leaf_solid_angle_props(lower.i, upper.i, points, indices, primitive_indices, node_solid_angle_props);
 
                 (vec3&)lower = node_solid_angle_props.box.lower;
                 (vec3&)upper = node_solid_angle_props.box.upper;
@@ -158,27 +168,7 @@
 
                     if (true_leaf) {
                         SolidAngleProps node_solid_angle_props;
-                        const int start = parent_lower.i;
-                        const int end = parent_upper.i;
-                        // loops through primitives in the leaf
-                        for (int primitive_counter = start; primitive_counter < end; primitive_counter++) {
-                            int primitive_index = primitive_indices[primitive_counter];
-                            if (primitive_counter == start) {
-                                precompute_triangle_solid_angle_props(
-                                    points[indices[primitive_index * 3 + 0]], points[indices[primitive_index * 3 + 1]],
-                                    points[indices[primitive_index * 3 + 2]], node_solid_angle_props
-                                );
-                            } else {
-                                SolidAngleProps triangle_solid_angle_props;
-                                precompute_triangle_solid_angle_props(
-                                    points[indices[primitive_index * 3 + 0]], points[indices[primitive_index * 3 + 1]],
-                                    points[indices[primitive_index * 3 + 2]], triangle_solid_angle_props
-                                );
-                                node_solid_angle_props = combine_precomputed_solid_angle_props(
-                                    &node_solid_angle_props, &triangle_solid_angle_props
-                                );
-                            }
-                        }
+                        leaf_solid_angle_props(parent_lower.i, parent_upper.i, points, indices, primitive_indices, node_solid_angle_props);
 
                         (vec3&)parent_lower = node_solid_angle_props.box.lower;
                         (vec3&)parent_upper = node_solid_angle_props.box.upper;
@@ -203,11 +193,7 @@
                     (vec3&)parent_upper = upper;
 
                     // combine
-                    SolidAngleProps* left_child_data = &solid_angle_props[left_child];
-                    SolidAngleProps* right_child_data
-                        = (left_child != right_child) ? &solid_angle_props[right_child] : NULL;
-
-                    combine_precomputed_solid_angle_props(solid_angle_props[parent], left_child_data, right_child_data);
+                    internal_solid_angle_props(solid_angle_props, parent, left_child, right_child);
                 }
                 // move onto processing the parent
                 index = parent;
@@ -248,25 +234,7 @@
             return;
 
         SolidAngleProps node_solid_angle_props;
-        const int start = lower.i;
-        const int end = upper.i;
-        for (int primitive_counter = start; primitive_counter < end; primitive_counter++) {
-            int primitive_index = primitive_indices[primitive_counter];
-            if (primitive_counter == start) {
-                precompute_triangle_solid_angle_props(
-                    points[indices[primitive_index * 3 + 0]], points[indices[primitive_index * 3 + 1]],
-                    points[indices[primitive_index * 3 + 2]], node_solid_angle_props
-                );
-            } else {
-                SolidAngleProps triangle_solid_angle_props;
-                precompute_triangle_solid_angle_props(
-                    points[indices[primitive_index * 3 + 0]], points[indices[primitive_index * 3 + 1]],
-                    points[indices[primitive_index * 3 + 2]], triangle_solid_angle_props
-                );
-                node_solid_angle_props
-                    = combine_precomputed_solid_angle_props(&node_solid_angle_props, &triangle_solid_angle_props);
-            }
-        }
+        leaf_solid_angle_props(lower.i, upper.i, points, indices, primitive_indices, node_solid_angle_props);
 
         (vec3&)lower = node_solid_angle_props.box.lower;
         (vec3&)upper = node_solid_angle_props.box.upper;
@@ -277,9 +245,7 @@
         (vec3&)lower = min((vec3&)node_lowers[left_child], (vec3&)node_lowers[right_child]);
         (vec3&)upper = max((vec3&)node_uppers[left_child], (vec3&)node_uppers[right_child]);
 
-        SolidAngleProps* left_child_data = &solid_angle_props[left_child];
-        SolidAngleProps* right_child_data = (left_child != right_child) ? &solid_angle_props[right_child] : NULL;
-        combine_precomputed_solid_angle_props(solid_angle_props[index], left_child_data, right_child_data);
+        internal_solid_angle_props(solid_angle_props, index, left_child, right_child);
     }
 }
 #endif
```

### Scripts (`agent_space/bench/`; `torus_mesh` is also appended to `bench_bvh_geom.py`; the trees `sawalk`, `salevels`, `sa4walk` and `sa4levels` are registered in `run.sh`; `drive.sh` now probes amd-smi GPU 0, which is HIP device 1 here)

`sa_check.py`:
```python
# usage: sa_check.py <out.json> [sizes...]; winding-number meshes (support_winding_number=True; a closed noisy torus of
# about N triangles) built with lbvh/sah/median, without and with 7 groups, leaf sizes 1 and 4, then: refit at creation,
# three refits with moved points, a points replacement (mesh.points = new array, which refits), a new mesh built on the
# moved points, and a refit captured in a graph relaunched three times with moved points. After each step the node
# arrays and the solid angle array are read from device memory and hashed (sha256 of the raw words); the hashes are
# compared across builds or path switches. Solid angle entries are hashed over the nodes the refit writes (nodes of the
# tree that are not below a packed leaf); max_p_dist_sq of single-triangle leaves (left unset by the code that computes
# them, and read by nothing) is hashed separately. Also recorded: the fraction of nodes whose words changed between
# consecutive refits, and hashes of query results (sign_winding_number, no_sign, ray) after the final step.
import hashlib, json, sys
import numpy as np
import warp as wp

wp.config.quiet = True
wp.init()
OUT = sys.argv[1]
SIZES = [int(float(s)) for s in sys.argv[2:]] or [10**3, 10**4, 10**5, 10**6, 10**7]
dev = wp.get_device("cuda:0")
SA_WORDS = 36  # sizeof(SolidAngleProps) / 4; max_p_dist_sq is the last word


def torus_mesh(ntri, rng):
    # (u, v) periodic grid: 2*nu*nv triangles, closed and watertight
    nv = max(3, int(round((ntri / 8) ** 0.5)))
    nu = max(3, int(round(ntri / (2 * nv))))
    u, v = np.meshgrid(np.arange(nu) * 2 * np.pi / nu, np.arange(nv) * 2 * np.pi / nv, indexing="ij")
    rr = 0.25 + 0.01 * np.sin(5 * u) * np.cos(3 * v)
    x = (0.5 + rr * np.cos(v)) * np.cos(u)
    y = (0.5 + rr * np.cos(v)) * np.sin(u)
    z = rr * np.sin(v)
    pts = np.stack([x, y, z], -1).reshape(-1, 3)
    pts += 1e-4 * rng.standard_normal(pts.shape)
    i = np.arange(nu)[:, None]
    j = np.arange(nv)[None, :]
    a = i * nv + j
    b = ((i + 1) % nu) * nv + j
    c = i * nv + (j + 1) % nv
    d = ((i + 1) % nu) * nv + (j + 1) % nv
    tris = np.concatenate([np.stack([a, b, c], -1).reshape(-1, 3), np.stack([b, d, c], -1).reshape(-1, 3)])
    return pts.astype(np.float32), tris.astype(np.int32).reshape(-1)


@wp.kernel
def move(p0: wp.array(dtype=wp.vec3), step: float, p: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    d = wp.vec3(wp.sin(float(i) * 0.37 + step), wp.cos(float(i) * 0.11 + 2.0 * step), wp.sin(float(i) * 0.05 - step))
    p[i] = p0[i] + d * 0.002


@wp.kernel
def queries(
    mesh: wp.uint64,
    qp: wp.array(dtype=wp.vec3),
    wn: wp.array(dtype=wp.vec4),
    ns: wp.array(dtype=wp.vec4),
    ry: wp.array(dtype=wp.vec4),
):
    t = wp.tid()
    p = qp[t]
    q = wp.mesh_query_point_sign_winding_number(mesh, p, 10.0)
    wn[t] = wp.vec4(wp.where(q.result, 1.0, 0.0), q.sign, float(q.face), q.u + q.v * 3.0)
    q2 = wp.mesh_query_point_no_sign(mesh, p, 10.0)
    ns[t] = wp.vec4(wp.where(q2.result, 1.0, 0.0), float(q2.face), q2.u, q2.v)
    d = wp.normalize(wp.vec3(wp.sin(float(t)), wp.cos(float(t) * 1.3), wp.sin(float(t) * 0.7 + 1.0)))
    r = wp.mesh_query_ray(mesh, p, d, 10.0)
    ry[t] = wp.vec4(wp.where(r.result, 1.0, 0.0), r.t, float(r.face), r.u)


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def view(ptr, shape, dtype=wp.uint32):
    return wp.array(ptr=int(ptr), dtype=dtype, shape=shape, device=dev, copy=False).numpy().copy()


def snapshot(mesh, host_built):
    m = int(mesh.id)
    sa_ptr = view(m + 184, (1,), wp.uint64)[0]
    num_tris = int(view(m + 196, (1,), wp.int32)[0])
    bvh_ptrs = view(m + 200, (5,), wp.uint64)
    max_depth, max_nodes, num_nodes, num_leaf = (int(x) for x in view(m + 240, (4,), wp.int32))
    k = num_nodes if host_built else max_nodes
    lo = view(bvh_ptrs[0], (k, 4))
    hi = view(bvh_ptrs[1], (k, 4))
    parents = view(bvh_ptrs[2], (k,), wp.int32)
    sa = view(sa_ptr, (k, SA_WORDS))
    is_leaf = (lo[:, 3] >> 31) == 1
    par_leaf = np.zeros(k, bool)
    ok = parents >= 0
    par_leaf[ok] = is_leaf[parents[ok]]
    written = ~(is_leaf & par_leaf)
    single = is_leaf & ~par_leaf & (((hi[:, 3] & 0x7FFFFFFF).astype(np.int64) - (lo[:, 3] & 0x7FFFFFFF)) == 1)
    sa_masked = sa[written].copy()
    sa_masked[single[written], SA_WORDS - 1] = 0
    return (
        {
            "lo": sha(lo),
            "hi": sha(hi),
            "sa": sha(sa_masked),
            "sa_leaf_maxp": sha(sa[single, SA_WORDS - 1]),
            "num_tris": num_tris,
            "num_nodes": num_nodes,
            "num_leaf": num_leaf,
            "written": int(written.sum()),
            "single": int(single.sum()),
        },
        (lo, hi, sa_masked),
    )


rng = np.random.default_rng(0)
res = {}
for N in SIZES:
    pts_np, ind_np = torus_mesh(N, rng)
    p0 = wp.array(pts_np, dtype=wp.vec3)
    ind = wp.array(ind_np, dtype=int)
    ntri = len(ind_np) // 3
    groups = wp.array((np.arange(ntri) * 7 // ntri).astype(np.int32), dtype=int)
    NQ = 10**5
    qp = wp.array(rng.uniform(-0.9, 0.9, (NQ, 3)).astype(np.float32) * np.array([1, 1, 0.4], np.float32), dtype=wp.vec3)
    for con in ("lbvh", "sah", "median"):
        if con != "lbvh" and N > 10**6:
            continue
        for g, leaf in ((None, 1), (groups, 1), (None, 4), (groups, 4)):
            key = f"{N}_{con}_{'g' if g is not None else 'n'}_leaf{leaf}"
            host = con != "lbvh"
            p = wp.clone(p0)
            mesh = wp.Mesh(points=p, indices=ind, support_winding_number=True, bvh_constructor=con, bvh_leaf_size=leaf, groups=g)
            r = {}
            r["create"], prev = snapshot(mesh, host)
            changed = []
            for step in range(3):
                wp.launch(move, dim=len(pts_np), inputs=[p0, float(step + 1), p])
                mesh.refit()
                r[f"refit{step}"], cur = snapshot(mesh, host)
                changed.append(float(np.mean(np.any(cur[2] != prev[2], axis=1))))
                prev = cur
            p2 = wp.empty_like(p)
            wp.launch(move, dim=len(pts_np), inputs=[p0, 5.0, p2])
            mesh.points = p2
            r["set_points"], _ = snapshot(mesh, host)
            del mesh
            mesh = wp.Mesh(points=p2, indices=ind, support_winding_number=True, bvh_constructor=con, bvh_leaf_size=leaf, groups=g)
            r["recreate"], _ = snapshot(mesh, host)
            with wp.ScopedCapture(device=dev, force_module_load=False) as cap:
                mesh.refit()
            for step in range(3):
                wp.launch(move, dim=len(pts_np), inputs=[p0, 10.0 + step, p2])
                wp.capture_launch(cap.graph)
            r["graph"], _ = snapshot(mesh, host)
            wn = wp.zeros(NQ, dtype=wp.vec4)
            ns = wp.zeros(NQ, dtype=wp.vec4)
            ry = wp.zeros(NQ, dtype=wp.vec4)
            wp.launch(queries, dim=NQ, inputs=[mesh.id, qp, wn, ns, ry])
            wnn = wn.numpy()
            r["q_wn"] = sha(wnn)
            r["q_ns"] = sha(ns.numpy())
            r["q_ray"] = sha(ry.numpy())
            r["inside_frac"] = float(np.mean(wnn[:, 1] < 0))
            r["changed_sa_rows"] = changed
            res[key] = r
            del mesh, cap
    print("done", N, flush=True)
json.dump(res, open(OUT, "w"), indent=1)
```

`sa_cmp.py`:
```python
# usage: sa_cmp.py <a.json> <b.json>; compares sa_check.py hashes field by field
import json, sys
a = json.load(open(sys.argv[1])); b = json.load(open(sys.argv[2]))
n = diff = 0; maxp = [0, 0]; diffs = []
for key in sorted(set(a) & set(b)):
    for step, va in a[key].items():
        vb = b[key][step]
        if isinstance(va, dict):
            for f in ("lo", "hi", "sa"):
                n += 1
                if va[f] != vb[f]:
                    diff += 1; diffs.append(f"{key} {step} {f}")
            maxp[0] += 1; maxp[1] += va["sa_leaf_maxp"] != vb["sa_leaf_maxp"]
        elif step.startswith("q_"):
            n += 1
            if va != vb:
                diff += 1; diffs.append(f"{key} {step}")
print(f"configs {len(set(a) & set(b))} (only a: {len(set(a) - set(b))}, only b: {len(set(b) - set(a))}); arrays compared {n}, differ {diff}")
print(f"single-triangle leaf max_p_dist_sq: {maxp[1]} of {maxp[0]} snapshots differ")
for d in diffs[:20]:
    print("  ", d)
ch = [min(v["changed_sa_rows"]) for v in a.values()]
print(f"min fraction of solid angle rows changed between consecutive refits: {min(ch):.4f}; inside fraction range {min(v['inside_frac'] for v in a.values()):.3f}..{max(v['inside_frac'] for v in a.values()):.3f}")
```

`sa_perf.py` (run with `./rounds.sh sa_perf.py <out> 3 sawalk salevels sa4walk sa4levels --`):
```python
# usage: sa_perf.py [sizes...]; wall time per wp.Mesh.refit() of a winding-number mesh (support_winding_number=True,
# closed torus of about N triangles; lbvh, sah, median, default leaf size; 10 warm-up calls, then 7 x <reps>
# back-to-back calls and a sync, median), plus a noise-floor metric whose code is the same in every variant: one
# 10^6-point mesh_query_point launch on the same mesh. Records via common.record.
import statistics, sys, time
import numpy as np
import warp as wp
from bench_bvh_geom import torus_mesh
from common import record
wp.config.quiet = True
wp.init()

@wp.kernel
def qpoint(mesh: wp.uint64, qp: wp.array(dtype=wp.vec3), dist: wp.array(dtype=float)):
    t = wp.tid()
    q = wp.mesh_query_point_no_sign(mesh, qp[t], 10.0)
    dist[t] = wp.where(q.result, q.u, -1.0)

def per_call(fn, reps):
    for _ in range(10):
        fn()
    wp.synchronize_device()
    res = []
    for _ in range(7):
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        wp.synchronize_device()
        res.append((time.perf_counter() - t0) / reps)
    return statistics.median(res), min(res), max(res)

rng = np.random.default_rng(0)
NQ = 10**6
qp = wp.array(rng.uniform(-0.9, 0.9, (NQ, 3)).astype(np.float32), dtype=wp.vec3); dist = wp.zeros(NQ, dtype=float)
for N in [int(float(s)) for s in sys.argv[1:]] or [10**3, 3000, 10**4, 30000, 10**5, 300000, 10**6, 10**7]:
    pts_np, ind_np = torus_mesh(N, rng)
    pts = wp.array(pts_np, dtype=wp.vec3); ind = wp.array(ind_np, dtype=int)
    reps = 200 if N <= 10**5 else (30 if N <= 10**6 else 5)
    for con in ("lbvh", "sah", "median"):
        if con != "lbvh" and N > 10**6:
            continue
        mesh = wp.Mesh(points=pts, indices=ind, bvh_constructor=con, support_winding_number=True)
        m, mn, mx = per_call(mesh.refit, reps)
        record(area="refit", op="wn_mesh_refit", N=N, constructor=con, med=m, min=mn, max=mx, reps=7)
        m, mn, mx = per_call(lambda: wp.launch(qpoint, dim=NQ, inputs=[mesh.id, qp, dist]), 5)
        record(area="refit", op="noise_query_point", N=N, constructor=con, med=m, min=mn, max=mx, reps=7)
        del mesh
```

## Ruling on session 26's question 11 (jeff, 2026-11-23)

(b): commit the inlined depth-level kernel for winding-number meshes as built, HIP-only, threshold `num_items >= 65536`. Node bounds and every query result match the walk; the solid angle moment data of multi-triangle leaves differs from the walk in the last bits (relative about 1e-7) because the compiler contracts each kernel differently, which is the same size as the difference between the walk's own two leaf copies. The bit-identity condition in the earlier prompt was the coordinator's, not jeff's, and is withdrawn for the solid angle arrays; it still holds for node bounds and query results. The shared-helper variant (a) is not taken.


## Porting session 27 (2026-11-23, linux-gfx950 MI355X, porter): winding-number refit committed; final build, wheel and suite; ported

Ruling on question 11: (b). Fork `release/1.14.0.amd0` `bc1ae44b..26be6725` was pushed and `advance-head` recorded. The final step of the round ran on head `26be6725`, and every suite result is clean, so the stage moves to `ported` and the lock is released. `HIP_VISIBLE_DEVICES=1` (amd-smi GPU 0), `ulimit -c 0`; no process was stopped by pattern.

### The commit: `26be6725` [ROCm] Refit large winding-number meshes one depth level at a time (`mesh.cu` +78, `bvh.cu` +13/-7, `bvh.h` +4)

Session 26's patch as built. The `SA_THRESHOLD` placeholder is replaced by `bvh.num_items >= 65536`, with a comment saying why it counts triangles: an LBVH tree counts every item as a leaf. No environment switch or scratch code; clang-format 22.1.2 (the pre-commit version) applied; jargon and prose clean. A winding-number mesh never changes path (`num_items` is fixed), so the walk's use of `node_counts` as counters never meets depths the levels need.

### Correctness on the final code (fork build at the change, against the walk: scratch tree `wrf2` forced onto the walk, whose walk kernel is the same source)

Same configurations and steps as session 26: closed torus meshes with `support_winding_number=True`; lbvh at every size, sah and median to 10^6; with and without groups; leaf sizes 1 and 4. Steps: creation, three moved-point refits, a points replacement, a rebuilt mesh, and a captured graph relaunched three times. Queries checked: winding-number sign, no-sign and ray. Sizes: 10^3, 10^4, **64,980 triangles (just below the threshold) and 66,066 (just above)**, 10^5, 10^6, 10^7. Arrays were kept rather than hashed, so the moments are compared numerically (`sa_check2.py`, described below).

| N (triangles) | constructor | node arrays equal | query result sets equal | solid angle arrays differing | max relative difference | max difference / field scale |
|---|---|---|---|---|---|---|
| 990 | lbvh | 28/28 | 4/4 | 0/28 | 0.00e+00 | 0.00e+00 |
| 990 | median | 28/28 | 4/4 | 0/28 | 0.00e+00 | 0.00e+00 |
| 990 | sah | 28/28 | 4/4 | 0/28 | 0.00e+00 | 0.00e+00 |
| 10010 | lbvh | 28/28 | 4/4 | 0/28 | 0.00e+00 | 0.00e+00 |
| 10010 | median | 28/28 | 4/4 | 0/28 | 0.00e+00 | 0.00e+00 |
| 10010 | sah | 28/28 | 4/4 | 0/28 | 0.00e+00 | 0.00e+00 |
| 64980 | lbvh | 28/28 | 4/4 | 0/28 | 0.00e+00 | 0.00e+00 |
| 64980 | median | 28/28 | 4/4 | 0/28 | 0.00e+00 | 0.00e+00 |
| 64980 | sah | 28/28 | 4/4 | 0/28 | 0.00e+00 | 0.00e+00 |
| 66066 | lbvh | 28/28 | 4/4 | 14/28 | 6.71e-05 | 2.86e-07 |
| 66066 | median | 28/28 | 4/4 | 14/28 | 1.01e-07 | 6.12e-10 |
| 66066 | sah | 28/28 | 4/4 | 14/28 | 7.58e-08 | 9.61e-11 |
| 99904 | lbvh | 28/28 | 4/4 | 14/28 | 7.50e-05 | 2.91e-07 |
| 99904 | median | 28/28 | 4/4 | 14/28 | 1.01e-07 | 3.06e-10 |
| 99904 | sah | 28/28 | 4/4 | 14/28 | 0.00e+00 | 7.67e-11 |
| 999696 | lbvh | 28/28 | 4/4 | 14/28 | 6.66e-06 | 6.95e-08 |
| 999696 | median | 28/28 | 4/4 | 14/28 | 0.00e+00 | 3.91e-11 |
| 999696 | sah | 28/28 | 4/4 | 14/28 | 0.00e+00 | 1.95e-11 |
| 9999392 | lbvh | 28/28 | 4/4 | 28/28 | 4.02e-05 | 5.68e-07 |

total: node snapshots equal 532/532 (x2 arrays), query sets equal 76/76 (x3 queries), solid angle arrays differing 154/532, max relative 7.50e-05, non-finite 0

- Node bounds: bit-identical in every snapshot (532 x 2 arrays). Query results: bit-identical in every set (76 x 3). Below the threshold (up to 64,980) everything is identical, solid angle included, because the walk runs.
- Above the threshold, the solid angle arrays differ only in leaf-size-4 trees (plus every lbvh tree at 10^7). On sah and median trees the largest relative difference is 1.0e-7. On lbvh the largest difference is **5.7e-7 of the field's largest value**, but per element the relative difference reaches **7.5e-5** in 42 of the 532 snapshots. Every such element is a small result of cancellation. The worst case examined (`n_zy`, an internal node at depth 10) has a value of -1.56e-5 against a field scale of 7.7e-3; it sums a child of -3.3e-6, a child of 1.4e-7 that itself differs by 5e-5 relative, and displacement products. The absolute error is at the rounding level of the operands, and no leaf element exceeds 1e-6 relative. Read per element, the ruling's "about 1e-6 relative" is exceeded by those elements; read against the field's magnitude, it holds everywhere.
- The ruling's own yardstick, "the same size as the difference between the walk's own two leaf copies", measured with the same tool: today's walk against the walk with shared non-inlined helpers (`wrf4`, session 26), whose leaf code is one copy.

| N (triangles) | constructor | node arrays equal | query result sets equal | solid angle arrays differing | max relative difference | max difference / field scale |
|---|---|---|---|---|---|---|
| 990 | lbvh | 28/28 | 4/4 | 14/28 | 3.56e-05 | 1.66e-06 |
| 990 | median | 28/28 | 4/4 | 0/28 | 0.00e+00 | 0.00e+00 |
| 990 | sah | 28/28 | 4/4 | 0/28 | 0.00e+00 | 0.00e+00 |
| 10010 | lbvh | 28/28 | 4/4 | 14/28 | 2.19e-04 | 7.52e-07 |
| 10010 | median | 28/28 | 4/4 | 0/28 | 0.00e+00 | 0.00e+00 |
| 10010 | sah | 28/28 | 4/4 | 0/28 | 0.00e+00 | 0.00e+00 |
| 64980 | lbvh | 28/28 | 4/4 | 14/28 | 2.66e-05 | 1.40e-07 |
| 64980 | median | 28/28 | 4/4 | 0/28 | 0.00e+00 | 0.00e+00 |
| 64980 | sah | 28/28 | 4/4 | 0/28 | 0.00e+00 | 0.00e+00 |
| 66066 | lbvh | 28/28 | 4/4 | 14/28 | 6.71e-05 | 2.86e-07 |
| 66066 | median | 28/28 | 4/4 | 0/28 | 0.00e+00 | 0.00e+00 |
| 66066 | sah | 28/28 | 4/4 | 0/28 | 0.00e+00 | 0.00e+00 |
| 99904 | lbvh | 28/28 | 4/4 | 14/28 | 7.50e-05 | 2.91e-07 |
| 99904 | median | 28/28 | 4/4 | 0/28 | 0.00e+00 | 0.00e+00 |
| 99904 | sah | 28/28 | 4/4 | 0/28 | 0.00e+00 | 0.00e+00 |
| 999696 | lbvh | 28/28 | 4/4 | 14/28 | 6.66e-06 | 6.95e-08 |
| 999696 | median | 28/28 | 4/4 | 0/28 | 0.00e+00 | 0.00e+00 |
| 999696 | sah | 28/28 | 4/4 | 0/28 | 0.00e+00 | 0.00e+00 |
| 9999392 | lbvh | 28/28 | 4/4 | 28/28 | 4.02e-05 | 5.68e-07 |

total: node snapshots equal 532/532 (x2 arrays), query sets equal 76/76 (x3 queries), solid angle arrays differing 112/532, max relative 2.19e-04, non-finite 0

  The walk's two copies differ by up to 2.2e-4 per element and 1.7e-6 of field scale, more than the new kernel differs from the walk (7.5e-5, 5.7e-7). Above the threshold on lbvh the two tables carry the same figures: the new kernel's differences there come from the packed leaves, which the walk computes in its other copy. Determinism: two runs of the walk give 1824 of 1824 arrays identical (hashes).

### Tests (fork build at `26be6725`, gfx950)

test_mesh* 49 OK, test_bvh 13 OK, test_grouped_bvh 11 OK, test_graph 35 OK (3 skipped), test_capture_mode 2 OK, test_conditional_captures 32 OK (16 skipped), test_array_fill_capture 3 OK, test_apic_mesh 12 OK.

### nvcc check (as in session 19)

`cuda_pp.sh` (nvcc 12.8, sm_90) of `bc1ae44b` against the change: 20 translation units identical, and `bvh` and `mesh` (device and host) LINE-ONLY. CUDA build of the change (worktree `agent_space/wcuda_s24` at `bc1ae44b` plus the three files; `build_lib.py --cuda-path /usr/local/cuda --no-use-libmathdx -j 32`, every arch sm_52 to sm_120): "Warp build succeeded", warp.so 199,752,736 bytes. Done before the commit, on the same content (`git stash create` object `53f36bcd`, no ref written).

### Release build, wheel and suite (head `26be6725`, clone clean)

- Build: `agent_space/relbuild.sh <clone> 64` = `python build_lib.py --rocm-path "$(python -m rocm_sdk path --root)" -j 64` (the committed defaults: the 25 Linux targets ROCm 10.2 supports, `--offload-compress`). Wall 181 s; 25 `--offload-arch` per HIP compile; 12 of 12 HIP units compressed. `warp.so` 10,947,400 bytes, `warp.so.rocm` records ROCm 10.2.0.
- Wheel: `python -m build --wheel -C--build-option=-Plinux-x86_64 -o <job>/wheel`. **Kept at `/var/lib/jenkins/fleet/jobs/mi355-20261119-154230-warp114-perf-porter-mi355-prompt/wheel/warp_lang-1.14.0+rocm10.2.0-py3-none-manylinux_2_28_x86_64.whl`: 37,218,821 bytes, sha256 `d083e104a64cee825b70d3ae2edf2d84009731ade960effee27980c34e23fdf9`.** Contents: warp.so 10,947,400, warp-clang.so 67,136,896, warp.so.rocm 140; 548 entries; no nvrtc/libmathdx/cudart/nvJitLink binary. METADATA has `Version: 1.14.0+rocm10.2.0`, `Requires-Dist: rocm<10.3,>=10.2` and no NVIDIA CUDA classifier; tag `py3-none-manylinux_2_28_x86_64`. Session 21's wheel (built at `7059a340`) was moved to `<job>/wheel/old_7059a340/`.
- Install: `$FLEET_VENV/bin/pip install --force-reinstall --no-deps <wheel>` into the job venv (the guard allows only that venv; deps were already present; `pip check` clean): "Successfully installed warp-lang-1.14.0+rocm10.2.0". **Installed size (`du -sb site-packages/warp`) 106,377,434 bytes**; the installed warp.so sha256 equals the build's.
- Full default suite from the installed wheel (`agent_space/suite_wheel.sh`: cwd `<job>/wheelrun`, `warp.__file__` in site-packages, `HIP_VISIBLE_DEVICES=1`, fresh kernel cache): banner "Warp 1.14.0+rocm10.2.0 ... ROCm 10.2, HIP runtime 7.17"; **Ran 7558 tests in 576.3 s, OK (skipped=81)**, wall 581 s. Host load average 40 at the start (other users), no other process on the GPU. No failures, so nothing needed a rerun alone.

### Release notes draft

Session 26's draft stands unchanged. The winding-number refit needs no user-visible sentence: it changes refit speed on large meshes, and its last-bit differences sit in internal solid angle data that no query result showed. Nothing about the 1.13 port is in the draft.

### Stage

Every result on head `26be6725` is clean, so the stage moves to `ported` (`set-state warp-1.14 linux-gfx950 ported`, which releases the lock) for the reviewer. Validators revalidate at `26be6725`.

### Scripts

`sa_check2.py` is `sa_check.py` (session 26, inline above) with a `keep()` step after every snapshot: `save` writes each snapshot's node words, solid angle words (written rows; `max_p_dist_sq` of single-triangle leaves zeroed) and query arrays to a directory, and `cmp` compares the running build's against them and deletes them (node words and query words exactly; solid angle values as the largest relative difference over elements of at least 1e-3 of the field's largest value, and the largest difference divided by the field's largest value). Run: `sa_run2.sh <tree> <WP_SA_REFIT_MIN or -> save|cmp <dir> <out.json> <sizes>`; summary `sa_sum2.py`. `sa_dump2.py` reproduces one configuration's creation on moved points, for element-level inspection.

Lock note (session 27): the first `set-state ... ported` was refused because the work lock still named the holder that took it on 2026-11-19T22:13:57Z. Its digest (`06f1c783051d`) matches the earlier porter job of this round on this host (`...-140431-warp114-phase7-porter-...`), which had stopped: no process had its job directory as working directory. This job (sessions 20 to 27) had never re-stamped the lock. Resumed per the rule for a stopped holder: `set-state warp-1.14 linux-gfx950 porting` (re-stamped with the warning), then `set-state warp-1.14 linux-gfx950 ported`. Stage is now `ported` at `26be6725` and the lock is released.

Surface note (session 27): the push of the `ported` transition was refused by the pre-push `surface` gate. It judges a project once it claims success, and `surface.json` had never been accounted (its `covered` and `scoped_out` lists were empty since planning). Accounting now done in `surface.json`:
- 51 components covered, each with its evidence (session, commit, or the session 27 wheel suite).
- 2 scoped out by Jeff's rulings: `feature:conditional-graph-nodes` (Q4, deferred `warp-hip-conditional-graph-nodes`) and `feature:clang-cuda-jit` (Q7, deferred `warp-hip-clang-jit`).
- 3 owed, each with a reason: `packaging:wheel-windows` (Windows wheel not built yet); `test:wheel-install` (Linux suite from the wheel passed, but in the job venv, because the fleet guard refuses pip elsewhere; Windows not run); `feature:gl-interop` (mapped to `hipGraphicsGLRegister*` but never exercised: no display on the fleet hosts).
- The two cuBQL CMake library targets removed from the floor: cuBQL's own CMakeLists is not part of Warp's build, and the sources compile into `bvh.cu` (`feature:cubql-backend`).
`utils/surface.py check warp-1.14`: fully accounted. The owed items do not block `ported`, but `pr_ready` refuses while any remain; they are for Jeff and the Windows round.

## Review 2026-11-23 (linux-gfx950 MI355X, reviewer): `release/1.14.0.amd0` at `26be6725` vs `release-1.14` (`3601b2d0`)

Scope: the whole fork diff (71 commits, 115 files, +4282/-517), with the pr-review skill: seven area reviews (build and packaging; Python runtime and driver layer; `warp.cu`; device headers and tiles; matrix-core `tile_matmul`; BVH, mesh refit and cuBQL; textures, tests and commit hygiene), then every finding re-read against the code. The ones marked "measured" were reproduced on gfx950 (`HIP_VISIBLE_DEVICES=1`, `ulimit -c 0`) with session 27's installed wheel at `26be6725` (sha256 `d083e104...`), or compiled with ROCm 10.2 clang or nvcc 12.8. Jeff's rulings were taken as given. The findings below concern how they were implemented, or code outside them.

### Summary

The port is HIP for Warp 1.14's native library, runtime compiler, tiles, textures, BVH and packaging. Verdict: **Request Changes**. Five defects affect users of the shipped wheel: a ~900 ms `block_dim=1` launch, NaN/Inf checks folded away under `fast_math`, silent no-op fills above 2^32 elements, fp64/bf16 `tile_matmul` not compiling on gfx908, and debug-mode tile collectives trapping on partial wavefronts.

### Port correctness

1. **`warp/native/warp.cu:5517`: `block_dim == 1` launches 2^32 - 1 blocks (measured).** `int(0xFFFFFFFFu / unsigned(block_dim))` is -1 for `block_dim` 1. `max_blocks` becomes -1, `grid_dim > max_blocks` is always true, and `grid_dim = -1` reaches `cuLaunchKernel_f` as gridDimX 0xFFFFFFFF. HIP accepts that, because 0xFFFFFFFF x 1 thread is within its limit. Measured: a warm `wp.launch(k, dim=1, block_dim=1)` takes 902 ms per launch, against 0.02 ms at `block_dim=2`. The results are correct only because the grid-stride loop lets the extra blocks exit. Upstream tests launch this way (`test_tile_cholesky_no_mathdx.py:266`, `test_tile_solve_no_mathdx.py`), so this probably costs suite time too. Fix: do the arithmetic unsigned and clamp to INT_MAX before narrowing, e.g. `max_blocks = int(std::min<unsigned>(unsigned(max_blocks), std::min(0xFFFFFFFFu / unsigned(block_dim), 0x7FFFFFFFu)));`.
2. **`warp/native/warp.cu:4579`: hipRTC `-ffast-math` also enables `-ffinite-math-only`, which NVRTC's `--use_fast_math` does not (measured).** A kernel computing `int(wp.isnan(x)) + 2 * int(wp.isinf(x))` on `[nan, inf, 1.0]` returns `[1, 2, 0]` without fast math and `[0, 0, 0]` with module option `fast_math=True`. nvcc 12.8 `-use_fast_math` keeps both tests (`setp.gtu.ftz.f32` and `setp.eq.ftz.f32` against inf). The NaN handling in `builtin.h`'s min/max and the `isfinite` guards in its division are folded the same way. Fix: push `-fno-finite-math-only` after `-ffast-math`. Better still, map NVRTC's meaning (ftz, approximate div/sqrt; fmad is already covered by the contraction option) explicitly. Then add a `fast_math` NaN/Inf case to `test_fast_math.py`.
3. **`warp/native/cuda_util.h:23-42` (`wp_launch_device`), and the Windows captured memset at `warp/native/warp.cu:1331,1333`: one thread per element with no cap, so launches above HIP's 2^32 - 1 threads per dimension fail (measured).** Commit `63fa49bf` capped only `wp_cuda_launch_kernel`, although the port's own lesson (`## Lesson ... 2^32 - 1 threads per dimension`) says one-thread-per-element kernels must split the launch. Measured: `a = wp.zeros((65536, 65544), dtype=wp.int16); a.fill_(7)` (4,295,491,584 elements, 8.6 GB) leaves the array all zeros, with no error, because release builds do not check `wp_launch_device`. On CUDA the same launch is 16.8M blocks, within the limit. Every built-in `wp_launch_device` kernel has the same exposure (`memtile_value_kernel` in `wp_memtile_device`, and the others in `warp.cu`/`bvh.cu`/`mesh.cu` that take an element count). Fix: on HIP, make `wp_launch_device` cap `num_blocks` at `0xFFFFFFFF / 256` and give the affected kernels a grid-stride loop, or split the launch; do the same for the two Windows memset kernels. Then add a test that fills a contiguous array of more than 2^32 elements (skipped on devices without the memory).
4. **`warp/native/tile_matmul_mma.h:23,42-53,71-87`: on gfx908, fp64 and bf16 `tile_matmul` kernels fail to compile (compiled).** The MFMA family is keyed on `__has_builtin(__builtin_amdgcn_mfma_f32_16x16x4f32)`, which is true on gfx908. But `op<float64>` (`__builtin_amdgcn_mfma_f64_16x16x4f64`) and `op<bfloat16>` (`__builtin_amdgcn_mfma_f32_16x16x16bf16_1k`) need `gfx90a-insts`. Compiling the header with a 16x16x16, `block_dim` 64 `tile_mma::matmul` instantiation gives, for gfx908: `error: '__builtin_amdgcn_mfma_f64_16x16x4f64' needs target feature gfx90a-insts` (float64), the same error for `mfma_f32_16x16x16bf16_1k` (bf16 inputs, float32 output), and float32 compiles. gfx90a compiles all three. gfx908 is one of the 25 targets the wheel ships (notes, release build) and hipRTC compiles for the GPU present, so on an MI100 any such module fails to build, including the new `test_tile_matmul_fragment_shapes_float64`. Session 9's statement that "the MFMA builtins are present for gfx90a/gfx942/gfx950 only" is wrong for gfx908. Fix: guard `op<float64>` with `#if __has_builtin(__builtin_amdgcn_mfma_f64_16x16x4f64)` and `op<bfloat16>` with `__has_builtin(__builtin_amdgcn_mfma_f32_16x16x16bf16_1k)`, so gfx908 keeps the scalar path for those dtypes. Also record that gfx908 now takes MFMA for fp32 and fp16 without a hardware run (same CDNA 16x16 layouts, unmeasured).

### Fault classes

5. **Wavefront masks in debug builds: `WP_TILE_FULL_MASK` (`~0ull` on gfx9) passed to HIP's `*_sync` intrinsics traps whenever lanes are inactive (measured).** HIP's `__hip_check_mask` asserts `mask == __ballot(true)` and traps unless `NDEBUG` is defined, and Warp's debug hipRTC compile passes `-D_DEBUG` without `NDEBUG` (`warp.cu:4566-4571`). Measured with `wp.config.mode = "debug"`: `tile_sum` over 128 floats is correct at `block_dim` 64, but at 32 and at 96 the launch ends in `Warp CUDA error 700: an illegal memory access` (the trap). Release mode at 32 is correct. Sites: `tile_reduce.h:181,241,338,392,463,711` (`__ballot_sync`), `tile_scan.h:61,80`, and `tile_radix_sort.h:154-155,482,540,621,642`. These include the radix sort's `if (warp_id == 0)` scan at `:567`, where `warp_id` counts 32-thread groups, so lanes 32-63 are inactive on every block size, and the bitonic `if (thread_id < 32)` paths at `:289-291,:388-390`. The notes record no debug-mode run on HIP. Fix: on HIP, pass the mask of executing lanes (`__activemask()` / `__ballot(1)`) to the sync intrinsics. Keep the constant only for the `mask == WP_TILE_FULL_MASK` fast-path tests in `warp_reduce`/`warp_reduce_tracked` (`tile_reduce.h:115,143`), which must stay false for a partial wavefront. The CUDA `0xFFFFFFFF` stays. Then run the tile tests in debug mode (`python -m warp.tests --warp-debug -p 'test_tile_*.py'` or the suite's debug option) on a wave64 GPU and record it.
6. **`warp/native/texture_software.cpp:148-151`: the result of the descriptor upload is ignored.** `wp_texture_software_object_create` tracks and returns `desc_device` even when `wp_memcpy_h2d` fails (for example after an earlier sticky error). Python then holds a handle to an uninitialized `wp::Texture`, and the next sampling kernel dereferences a garbage `data` pointer, instead of `Texture.__init__` raising its "Failed to create" error. Fix: on failure, free `desc_device` and return 0 without tracking it.

### Minimal footprint

7. **`warp/native/tile.h:54`: operator precedence changes upstream's condition.** `#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__) && !defined(__INTELLISENSE__)` parses as `__CUDA_ARCH__ || (HIP && !INTELLISENSE)`. Upstream (`tile.h:40` at the base) is `defined(__CUDA_ARCH__) && !defined(__INTELLISENSE__)`. So with `__INTELLISENSE__` defined, CUDA translation units now get the unroll branch upstream deliberately hid. nvcc and NVRTC output are unaffected. Fix: `#if (defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)) && !defined(__INTELLISENSE__)`.
8. **`wp_cuda_device_get_warp_size` is an unused new export** (`warp/native/warp.h:423`, `warp.cpp:1017`, `warp.cu:2527-2535`, and the `DeviceInfo::warp_size` field set at `warp.cu:324`; from `3f731039`). Nothing binds or calls it (no ctypes declaration in `context.py`, no native caller, no test). Every later release line would carry it. Remove it and the field; `arch_name` stays.
9. **`warp/native/tile_matmul_mma.h:91-92,169`: stale comments.** They say the gfx12 bf16 path "assumes the gfx12 fp16 layout", but the gfx1201 evidence session at `fda2f89a` measured it. Say measured. At `:248`, `fragments / (WP_TILE_BLOCK_DIM / kWave)` divides by zero for `block_dim < 32` on wave32. The branch is discarded at compile time, but clang still writes a `-Wdivision-by-zero` warning into the hipRTC log. Guard it with `WP_TILE_BLOCK_DIM >= kWave &&`.

### Build system and packaging

10. **`setup.py:226-233` (`get_rocm_requirement`): nothing ties the wheel's hard-coded `+rocm10.2.0` label to the ROCm it was built against.** The pin and `runtime.rocm_version` come from the auto-detected ROCm's `.info/version` via the marker. A build that picks up another ROCm (a venv with the 10.1 SDK, a `/opt/rocm` or `ROCM_PATH` install of another release) produces `warp_lang-1.14.0+rocm10.2.0-...whl` with `Requires-Dist: rocm<10.2,>=10.1`. Fix: when `rocm_wheel` is set, parse `rocmX.Y` from the version's local segment and raise if it differs from the marker's major.minor.
11. **The release wheel carries the build machine's absolute paths (verified in session 27's wheel).** `warp/bin/warp.so.rocm`'s first line is `<build venv>/lib/python3.11/site-packages/_rocm_sdk_devel` (written by `warp/_src/build_dll.py:1065-1066` and copied unchanged by `setup.py:217-220`). `warp.so`'s RUNPATH is `$ORIGIN:<build venv>/.../_rocm_sdk_devel/lib` (`build_dll.py:931`). For a public release asset this exposes the builder's directory layout (job names) and does nothing on users' machines: the wheel's `rocm` dependency means `load_hip_runtime` preloads through `rocm_sdk` and never reads the root. Fix: write the wheel's copy of the marker with an empty root line (the loader already copes with a missing root), and drop or relativize the ROCm rpath for wheel builds against the `rocm_sdk` packages.
12. **`setup.py:216-220`: a stale copied marker can survive in `warp/bin/`.** When the library comes from a platform subdirectory, the marker is copied to `warp/bin/<lib>.rocm`, and nothing removes it when a later CUDA library is copied over `warp/bin/<lib>`. Running from the tree or an editable install then sends `context.py` into `load_hip_runtime` for a CUDA library. Fix: in the `src != dst` branch, delete `dst + ".rocm"` when `src + ".rocm"` does not exist.

### Testing

13. **The committed matrix-core tests do not cover what the Q3 ruling asks (`warp/tests/tile/test_tile_matmul.py:385-408,456-462`).** The ruling asks for "cases that hit the fragment boundaries (... each dtype)" for "every `block_dim`", including blocks that are not a whole number of wavefronts or have fewer threads than a fragment needs. The new test covers fp16/fp32/fp64 only, with `fragments_n` 1 in every shape and `block_dim` 64 and 96. That leaves bf16 inputs untested on MFMA and WMMA (`6761183c`), and the `group` 2/4 A-sharing paths of `930d0098` untested. On wave32 both block sizes are whole wavefronts, so the comment "one that is not" holds only on wave64. No block is smaller than a wavefront, and the gfx11 `amortizes_operand_loads` boundary is never reached. The `mma_check*.py` scripts that covered these live in `agent_space` and are not durable. Fix: add a bf16 case (float32 output), shapes with N in {32, 64}, and `block_dim` values {16, 32, 48, 64, 96}, still compared against the CPU.

### Ruling implementation, for jeff

14. **gfx1250 fp32 WMMA ships off** (`tile_matmul_mma.h:89-93,191-208`: the family turns on only for the gfx11/gfx12 f16 builtins or `WP_TILE_WMMA`, which nothing defines). The Q3 ruling says "fp32 also where the target has it, gfx1250". The notes give the reason (layout from documentation, no hardware to measure it), but no ruling accepts the omission and `deferred.json` has no entry for it. Owed: a deferral registered with `utils/deferred.py add ... --project warp-1.14` for jeff to rule, or a gfx1250 layout measurement.

### Records

15. **The lesson "Windows HIP graph relaunch races its own memory-free nodes" recommends the superseded workaround.** It says to `hipStreamSynchronize` before each `hipGraphLaunch`. Session 15 replaced that with launch-then-wait (`6b639456`), because waiting first leaves the first-launch allocation failure open (`graph_alloc_first_launch 5 256 d2h`: 5 of 5 failures with wait-before, 0 of 5 with wait-after). Jeff's ruling kept launch-then-wait. Fix the lesson before promotion: wait after each launch when the stream is not capturing, and say why waiting before is not enough.
16. **Release notes draft (session 26), Textures bullet: "MI300X and MI355X (and MI250X for 3D textures) have no texture sampling hardware".** That is wrong for MI250X. `c9595bce` says gfx90a has texture hardware but HIP refuses 3D texture objects with linear filtering there, and the port sends all 3D textures on gfx90a to the software sampler. Suggested wording: "On MI250X, 3D textures are also sampled in software, because HIP does not support linear filtering of 3D textures on that GPU."
17. **For jeff: commit `dd41f3d2`'s body (second paragraph) names commit `da56e495` of `release/1.13.0.amd0` and its author.** Under the 2026-11-20 ruling that defers statements about that branch until after the 1.14 release, jeff decides whether this sentence stays in the history the release tag will carry. Changing it means rewriting published history, which only he can authorize. The other references to that branch in the history (`43c348ea`, `fb4803bb`) are credit lines.

### Not findings (checked)

- The `jargon.py --port warp-1.14` run needs a local `release-1.14` branch in the clone (`git branch release-1.14 origin/release-1.14`); with it: `jargon: clean`.
- Commit hygiene across the 71 commits holds: titles at most 72 characters, `[ROCm]` only on ROCm-specific commits, AI disclosure and Test Plans, no Co-Authored-By trailer, no noreply address, no non-ASCII.

## Coordinator routing of the 2026-11-23 review (2026-11-23)

Findings 1 to 13, 15 and 16 go to the porter as a fix round. Finding 14 is registered as deferred item `warp-hip-gfx1250-fp32-wmma` for jeff to rule. Finding 17 (the `dd41f3d2` commit body) is with jeff; the porter does not touch history. For finding 2 the porter adds `-fno-finite-math-only` now and reports what else clang's `-ffast-math` changes that NVRTC's `--use_fast_math` does not, as a question, without implementing a wider remap.

## Ruling on review finding 17 (jeff, 2026-11-23)

(a): leave commit `dd41f3d2`'s body as it is. No history rewrite.

## Porting session 28 (2026-11-23, linux-gfx950 MI355X, porter): fix round for the 2026-11-23 review

Took the lock with `set-state warp-1.14 linux-gfx950 porting` (from `changes-requested`). Fork `release/1.14.0.amd0` `26be6725..3c6714b8`: 14 commits, pushed; `advance-head` recorded. `HIP_VISIBLE_DEVICES=1` (amd-smi GPU 0), `ulimit -c 0`. "Before" measurements used session 27's installed wheel (`26be6725`, the reviewed head) from `<job>/wheelrun`; "after" used the fork tree built for gfx950 (`PYTHONPATH`). Stopped processes: only PIDs 175972 and 176053, the two processes of my own hung debug-mode run (identified by their command line and start time, 0% CPU for 10 minutes after a trap); no pattern kill.

### The commits (in order)

| commit | finding | change |
|---|---|---|
| `c0d91109` | 1 | `warp.cu`: HIP grid cap computed unsigned and clamped to INT_MAX (`block_dim` 1 gave max_blocks -1, gridDim.x 0xFFFFFFFF) |
| `4d230f8a` | 2 | `warp.cu`: `-fno-finite-math-only` after `-ffast-math` in the hipRTC options; `test_fast_math.py`: new `test_fast_math_nan_inf`; `test_fast_math_cuda` expects 4.0 on HIP (see below) |
| `2a08c179` | 3 | `cuda_util.h`: `wp_launch_grid`/`WP_LAUNCH_GRID`/`WP_GRID_THREAD_INDEX` (HIP: overflow into gridDim.y; CUDA: the old grid and index text); `warp.cu`: the 64-bit-count kernels' index lines and the Windows captured-memset launches; `test_array.py`: new `test_fill_large` |
| `15c07dd6` | 4 | `tile_matmul_mma.h`: `op<float64>` and `op<bfloat16>` guarded by their own builtins |
| `1de40317` | 5 | `tile_reduce.h`, `tile_scan.h`, `tile_radix_sort.h`: `WP_TILE_SYNC_MASK` (`__activemask()` on HIP, 0xFFFFFFFF on CUDA) at the sync-intrinsic calls; the two fast-path comparisons keep `WP_TILE_FULL_MASK` |
| `60051d5e` | new (found by the finding 5 debug run) | `tile.h`: on HIP, `dynamic_smem_base` declared 16-byte aligned |
| `cf188478` | 6 | `texture_software.cpp`: free the descriptor and return 0 when its upload fails |
| `92e4a2bf` | 7 | `tile.h`: `(__CUDA_ARCH__ \|\| __HIP_DEVICE_COMPILE__) && !__INTELLISENSE__` |
| `363daea4` | 8 | `warp.h`/`warp.cpp`/`warp.cu`: `wp_cuda_device_get_warp_size` and `DeviceInfo::warp_size` removed |
| `52ab1a75` | 9 | `tile_matmul_mma.h`: comments say the gfx12 bf16 layout is measured; `amortizes_operand_loads` term rewritten without a division |
| `75d5b66f` | 10 | `setup.py`: raise when the version's `+rocmX.Y` differs from the marker's ROCm (or is missing) |
| `90d2ea90` | 11 | `setup.py`: `build_py` hook writes the wheel's marker with an empty root line; `build_dll.py`: no ROCm rpath for a build against the ROCm Python packages; `-fmacro-prefix-map` for ROCm builds (HIP compiles and Linux host C++) |
| `57dff8bd` | 12 | `setup.py`: delete `warp/bin/<lib>.rocm` when a platform-subdirectory library without a marker is copied over it |
| `3c6714b8` | 13 | `test_tile_matmul.py`: fragment shapes add N 32 and 64, block_dim {16, 32, 48, 64, 96}, new bf16 case (out tile, `enable_backward=False`) |

The first attempt at the commit series built each commit's file contents from `HEAD`, which advanced after each commit, so the finding 2 hunk landed in the finding 3 commit. Those 14 commits were local and never pushed; I reset the branch to `26be6725` (working tree kept) and recreated them from a fixed base, so each commit holds exactly its finding. No published commit was rewritten.

### Evidence per finding (gfx950, ROCm 10.2)

1. Warm launch of a one-element kernel: `block_dim=1` 1739 ms before, 0.030 ms after (`block_dim=2`: 0.125 / 0.034 ms). Suite time: the two upstream files with `block_dim=1` launches, warm cache, one GPU: `test_tile_cholesky_no_mathdx` 7.47 -> 6.29 s, `test_tile_solve_no_mathdx` 7.00 -> 4.33 s (about 1.3 s per such launch, 3.9 s per run). Timing check recorded here rather than as a test (Warp's guidelines advise against timing assertions).
2. `int(isnan(x)) + 2 * int(isinf(x))` on `[nan, inf, 1.0]` with `fast_math`: `[0, 0, 0]` before, `[1, 2, 0]` after. The new test's kernel (adds `4 * isfinite`): `[4, 4, 4, 4]` before, `[1, 2, 2, 4]` after. Side effect: upstream's `test_fast_math_cuda` (expects `pow(-2, 2)` = NaN under fast math, NVRTC's approximate pow) had passed on HIP only because its `isnan` check was folded; HIP's fast pow gives 4.0, so the test now expects 4.0 on HIP devices and NaN elsewhere. test_fast_math 14 OK (1 skipped).
3. `wp.zeros((65536, 65544), dtype=wp.int16).fill_(7)` (4,295,491,584 elements): all 0 before, all 7 after. New test (65536 x 65537 int16, first and last row; the last row crosses flat index 2^32; skipped below 12 GB free) passes. Kernels exposed (64-bit element count, one thread per element, correct on CUDA above 2^32 - 1 and not on HIP): `memtile_value_kernel`, `memtile_kernel`, `memtile_kernel_by_value<N>`, `array_copy_{1,2,3,4}d_kernel`, the seven fabric copy kernels, `array_fill_{1,2,3,4}d_kernel` and `_by_value<N>`, `array_fill_fabric(_indexed)_kernel` and `_by_value<N>`, and the Windows captured-memset `memset_kernel`/`memtile_value_kernel<uint8_t>` launches. Not exposed (int counts, at most 2^31 - 1 elements): every kernel in `bvh.cu`, `mesh.cu`, `hashgrid.cu`, `sparse.cu`, `volume.cu` and the remaining `warp.cu` launches. Chosen form: the y-dimension overflow, one line per kernel; a capped grid plus grid-stride loop would re-indent every kernel body, and split launches would need an offset argument in every kernel.
4. hipRTC compile of a 16x16x16 `tile_matmul` (block_dim 64) for gfx908 / gfx90a / gfx950 (`agent_space/s28/mfma_targets.py`, `warp._src.build.build_cuda(..., hip_target=...)`): before, fp64 and bf16 (float32 out) fail on gfx908 (HIP kernel build error 6), all others compile; after, all four types compile on all three targets. `v_mfma` count in the code objects after: gfx908 fp32 16, fp16 4, fp64 0, bf16 0; gfx90a fp32 16, fp64 16, fp16 4, bf16 1. **gfx908's fp32 and fp16 MFMA paths have no hardware run** (same CDNA 16x16 layouts as gfx90a, unmeasured).
5. Debug mode `tile_sum` over 128 floats: block_dim 32 -> error 700 (the trap) before; 64, 32, 96 all 8128 after. All tile tests in debug mode (`python -m warp.tests -s autodetect --warp-debug -p "test_tile*.py"`): before (wheel), 416 OK and 120 errors, then the run hung in a trapped worker (stopped after 10 minutes at 0% CPU). After the mask change alone: 599 run, 147 errors, all from a second cause (next item). After both: **599 tests OK (9 skipped)**, no device assertion, no error 700. Release mode: the 21 tile kernels of `tile_perf.py` (tile_sum, tile_max, tile_argmax, tile_reduce, tile_scan_inclusive/exclusive, tile_sort at block_dim 32, 64, 256) plus a control kernel disassemble to identical instructions and operands before and after the mask change (22 of 22).
   New finding, fixed in `60051d5e`: AMDGPU places `extern __shared__ char dynamic_smem_base[]` right after the kernel's static shared variables at the array's alignment (1 byte), while CUDA starts dynamic shared memory 16-byte aligned; shared tiles are copied with 16-byte accesses. In debug mode `copy_from_global`'s assertion "shared tile pointer not 16-byte aligned" fired in test_tile_reduce, test_tile_func_arg, test_tile_cholesky_no_mathdx and test_tile_solve_no_mathdx (release-mode kernels of those four files at `26be6725`: 38 of 92 have a static shared size that is not a multiple of 16). With `__attribute__((aligned(16)))` on HIP the four files pass in debug mode. Performance, release mode, mask and alignment changes together against `26be6725`, 65536 tiles of 256 floats, 7 interleaved rounds, minimum: every cell 0.998x to 1.011x, noise floor 1.000x (table below). Release tile tests 599 OK (9 skipped).
6. Code change only (the failure needs an earlier sticky error); `Texture.__init__` already raises on a 0 handle. test_texture 118 OK (7 skipped).
7. Precedence restored; compiler output unchanged.
8. Export and field removed; nothing referenced them. test_devices 14 OK, test_context 1 OK.
9. Comments corrected. The `&& WP_TILE_BLOCK_DIM >= kWave` guard the reviewer proposed still drew clang's warning (it evaluates the constant division), so the term is now `WP_TILE_BLOCK_DIM >= kWave && fragments >= 4 * (WP_TILE_BLOCK_DIM / kWave)`, equal where the old one was defined. fp16 64x64x64 `tile_matmul` at block_dim 16 through hipRTC for gfx1100: 4 `-Wdivision-by-zero` warnings before, 0 after.
10. `get_rocm_requirement` on scratch files (`agent_space/s28/setupcheck/check.py`), VERSION 1.14.0+rocm10.2.0: before, a 10.1.0 marker gave `rocm>=10.1,<10.2`; after, it raises; a 10.2.0 marker gives `rocm>=10.2,<10.3` both times.
11. Session 27's wheel had the build venv's ROCm root as the marker's first line, the same root's `lib` in `warp.so`'s RUNPATH, and 22 `__FILE__` strings with the job path in `warp.so` (`check_cuda`/`check_cu` and nanovdb/cuBQL error paths). The new release wheel: marker `'\n10.2.0\n'`, RUNPATH `$ORIGIN`, no file in the wheel contains the job path (warp.so and warp-clang.so checked with `strings`, every wheel member with a byte search). Remaining, not changed: the host C++ of a Windows build goes through MSVC, which has no documented `-fmacro-prefix-map` equivalent, so a future Windows wheel would still carry `__FILE__` paths of its `.cpp` files (its `.cu` files go through clang and are covered).
12. `get_warp_libraries` on scratch files (a CUDA library in `bin/linux-x86_64`, a stale `warp/bin/warp.so.rocm`): the stale marker survived before, removed after.
13. test_tile_matmul 26 OK in release and in debug mode (fp16, fp32, fp64 and bf16 fragment tests, 6 shapes x 5 block sizes each).
15. The lesson "Windows HIP graph relaunch races its own memory-free nodes" now recommends waiting after each launch and says why waiting before is not enough (edited in place above, before promotion).
16. Release notes draft (session 26 copy above and `agent_space/release_notes_draft.md`): the Textures bullet now reads "MI300X and MI355X have no texture sampling hardware. ... On MI250X, 3D textures are also sampled in software, because HIP does not support linear filtering of 3D textures on that GPU. The CUDA interop handles (...) are not available for textures sampled in software." `utils/prose.py`: clean; no mention of the 1.13 port.

Finding 14 (gfx1250 fp32 WMMA) and finding 17: not touched, as routed. Jeff's later rulings in this branch (finding 17: leave the body; `warp-hip-gfx1250-fp32-wmma` decided "now", to be measured on the gfx1250 host) arrived during the round; the gfx1250 measurement needs that host and is not done here.

Tile performance, release mode, `26be6725` (before) against the fork tree with the mask and alignment commits (after), GPU time per launch over 65536 tiles of 256 floats, 7 interleaved rounds (`agent_space/s28/tile_perf.py`, `perf_rounds.sh`):

| op | block_dim | before us (min / median) | after us (min / median) | after/before (min) | after/before (median) |
|---|---|---|---|---|---|
| argmax | 32 | 24.9 / 25.1 | 24.9 / 25.0 | 1.000 | 0.998 |
| argmax | 64 | 19.0 / 19.1 | 19.0 / 19.1 | 1.004 | 1.001 |
| argmax | 256 | 52.7 / 53.0 | 52.8 / 53.0 | 1.002 | 1.000 |
| max | 32 | 20.4 / 20.5 | 20.4 / 20.6 | 1.002 | 1.005 |
| max | 64 | 17.8 / 17.9 | 17.8 / 17.8 | 1.000 | 0.999 |
| max | 256 | 47.5 / 47.6 | 47.5 / 47.7 | 0.999 | 1.001 |
| reduce | 32 | 19.0 / 19.2 | 19.0 / 19.1 | 1.001 | 0.997 |
| reduce | 64 | 17.5 / 17.8 | 17.6 / 17.7 | 1.002 | 0.997 |
| reduce | 256 | 44.4 / 44.5 | 44.4 / 44.6 | 1.001 | 1.002 |
| scan_exc | 32 | 87.7 / 88.0 | 87.7 / 87.9 | 1.000 | 0.999 |
| scan_exc | 64 | 48.9 / 49.0 | 48.8 / 49.0 | 0.998 | 1.000 |
| scan_exc | 256 | 60.9 / 61.0 | 60.9 / 61.0 | 1.000 | 1.001 |
| scan_inc | 32 | 82.7 / 82.9 | 82.6 / 82.9 | 0.999 | 1.001 |
| scan_inc | 64 | 46.3 / 46.6 | 46.4 / 46.4 | 1.002 | 0.996 |
| scan_inc | 256 | 60.5 / 60.6 | 60.4 / 60.7 | 1.000 | 1.001 |
| sort | 32 | 498.1 / 500.2 | 498.3 / 499.9 | 1.000 | 0.999 |
| sort | 64 | 229.4 / 229.7 | 229.5 / 229.9 | 1.000 | 1.001 |
| sort | 256 | 313.5 / 314.4 | 313.4 / 314.2 | 1.000 | 0.999 |
| sum | 32 | 19.8 / 20.0 | 19.9 / 20.0 | 1.003 | 1.000 |
| sum | 64 | 18.2 / 18.5 | 18.5 / 18.5 | 1.011 | 1.002 |
| sum | 256 | 44.4 / 44.5 | 44.5 / 44.6 | 1.002 | 1.002 |
| noise_floor | 0 | 32.6 / 32.7 | 32.6 / 32.7 | 1.000 | 1.000 |

### Question for Jeff (finding 2, no decision taken)

12. Besides `-ffinite-math-only` (now undone), Clang's `-ffast-math` on HIP differs from NVRTC's `--use_fast_math` in ways a `fast_math` module can observe. Measured on gfx950 after this round's fix (`agent_space/s28/fastmath_probe.py`):
   - Reassociation (`-fassociative-math`): `(x + 1e8) - 1e8` with x = 1 gives 1.0 with fast math (0.0 without); NVRTC's option does not reassociate.
   - No signed zeros (`-fno-signed-zeros`): `1 / (z + 0.0)` with z = -0.0 gives -inf with fast math (+inf without); NVRTC keeps IEEE signed zeros.
   - Denormals: HIP keeps them with fast math (`1e-40 * 1` stays 1e-40); NVRTC's `--use_fast_math` implies `--ftz=true` and flushes them.
   - Alike: division is approximate on both (HIP: 300,713 of 1,048,576 quotients differ from the correctly rounded value, at most 2 ulp; NVRTC uses approximate division with `--prec-div=false`), and sin/exp/sqrt use approximate forms on both (HIP: up to 8 ulp from the non-fast result).
   Dependence: Warp's configuration guide defines the option as compiling with "the `--use_fast_math` compiler option" (NVRTC's); the only test of its semantics is `test_fast_math.py` (negative-base pow, approximate division and inverse, and now NaN/Inf), and none depends on reassociation, signed zeros or denormal flushing. Options: (a) keep `-ffast-math -fno-finite-math-only` (this round); (b) replace `-ffast-math` on HIP with the parts that match NVRTC's option: approximate functions and reciprocal division (`-fapprox-func -freciprocal-math`) and denormal flushing (`-fgpu-flush-denormals-to-zero`), leaving out reassociation and signed-zero folding; (c) (a) plus a sentence in the release notes. Not implemented, as instructed.

### Final step (head `3c6714b8`)

- nvcc check (as in session 19): `cuda_pp.sh` (nvcc 12.8, sm_90) of `26be6725` against `3c6714b8`, line numbers normalized: 2 of 24 translation units identical; the others differ only by the removed `wp_cuda_device_get_warp_size` declaration (finding 8; the port had added it), and `warp.cu` also by the copy/fill kernels' index expressions (`size_t(blockIdx.x) * size_t(blockDim.x) + size_t(threadIdx.x)`, parenthesized, and for the memtile kernels in that operand order instead of blockDim * blockIdx: the same value). `WP_LAUNCH_GRID` expands to the old grid text. CUDA build of `3c6714b8` (worktree `agent_space/wcuda_s24`, `build_lib.py --cuda-path /usr/local/cuda --no-use-libmathdx -j 32`, sm_52 to sm_120): "Warp build succeeded", warp.so 199,760,928 bytes.
- Release build: `agent_space/relbuild.sh <clone> 64` (the 25 Linux targets, `--offload-compress`, `-fmacro-prefix-map`), wall 254 s; warp.so 10,922,824 bytes, no build-path string, RUNPATH `$ORIGIN`.
- Wheel: `python -m build --wheel -C--build-option=-Plinux-x86_64 -o <job>/wheel` in the clean clone at `3c6714b8`. **Kept at `/var/lib/jenkins/fleet/jobs/mi355-20261119-154230-warp114-perf-porter-mi355-prompt/wheel/warp_lang-1.14.0+rocm10.2.0-py3-none-manylinux_2_28_x86_64.whl`: 37,193,073 bytes, sha256 `e757280392167d5c59c0f2d45b10121666781cad257e2869190f750051f3bb00`.** Contents: warp.so 10,922,824, warp-clang.so 67,136,896, warp.so.rocm 8 (`'\n10.2.0\n'`); 548 entries; no NVIDIA binary; no member contains the job path. METADATA: `Version: 1.14.0+rocm10.2.0`, `Requires-Dist: rocm<10.3,>=10.2`, no NVIDIA CUDA classifier. Session 27's wheel moved to `<job>/wheel/old_26be6725/`.
- Install: `$FLEET_VENV/bin/pip install --force-reinstall --no-deps <wheel>` (job venv; `pip check` clean). **Installed size (`du -sb site-packages/warp`) 106,364,306 bytes**; installed warp.so sha256 equals the build's.
- Full default suite from the installed wheel (`agent_space/suite_wheel.sh`, cwd `<job>/wheelrun`, `warp.__file__` in site-packages, fresh kernel cache, `HIP_VISIBLE_DEVICES=1`): **Ran 7562 tests in 612.8 s, OK (skipped=81)**, wall 619 s; host load average 232 at the start (other users' jobs, 8 of their processes on the GPU). 7562 = session 27's 7558 + test_fill_large, test_fast_math_nan_inf (CPU and GPU) and the bf16 fragment test.
- Tile tests in debug mode from the installed wheel (`python -m warp.tests -s autodetect --warp-debug -p "test_tile*.py"`, cwd `<job>/wheelrun`): **Ran 599 tests in 162.5 s, OK (skipped=9)**, no device assertion, no error 700.
- Every result clean: stage moved to `ported` (lock released) for the reviewer. Validators revalidate at `3c6714b8`. The gfx1250 fp32 WMMA measurement ruled "now" is outstanding on the gfx1250 host; if it changes the head, the wheel is rebuilt after it.

### Scripts (`agent_space/s28/`)

`repro.py` (findings 1, 2, 3, 5), `fm_test_kernel.py`, `mfma_targets.py`, `wmma_warn.py`, `fastmath_probe.py`, `tile_perf.py` with `perf_rounds.sh`, `lds_load.py` with `lds_scan.sh`, `setupcheck/check.py`, `tests_after.sh`, `split.py` (the commit split). Recipes are in the evidence lines above; the scripts are scratch.

## Lesson for cuda-to-rocm: dynamic shared memory is not 16-byte aligned on AMDGPU unless declared so
Reference file: `references/diagnosis.md` (or the fault-class reference). CUDA starts a kernel's dynamic shared memory (`extern __shared__ char buf[];`) 16-byte aligned, and code that carves vector-typed tiles out of it relies on that. AMDGPU places the dynamic window directly after the kernel's static `__shared__` variables, at the alignment declared on the `extern` array (1 byte for `char`), so where it starts depends on how much static shared memory the kernel has. Release builds usually run anyway (the hardware tolerates the misaligned LDS vector access), which hides it; an alignment assertion in a debug build, or a kernel whose static shared size is not a multiple of 16 (read `.group_segment_fixed_size` with `llvm-readelf --notes` on the code object), shows it. Fix on HIP: `extern __shared__ char buf[] __attribute__((aligned(16)));` (an `alignas` after `__shared__` is a syntax error). (warp-1.14)

## Lesson for cuda-to-rocm: HIP's `*_sync` warp functions check the mask in debug builds
Reference file: the wavefront reference. HIP's `__ballot_sync`, `__shfl_*_sync` and `__syncwarp(mask)` assert, in any compile without `NDEBUG`, that the mask names exactly the lanes executing the call, and trap otherwise (the launch ends in "illegal memory access", error 700). Code that passes a constant full mask (`~0ull`, or CUDA's `0xFFFFFFFF` widened) works in release builds, where the mask is not checked and a ballot already covers only the executing lanes, but traps in debug builds wherever part of the wavefront is inactive: a block narrower than 64 threads on a 64-wide GPU, the last wavefront of a block that is not a whole number of wavefronts, or a branch on a 32-thread group. Pass `__activemask()` on HIP; keep a constant only where the code compares a mask against "all lanes" to choose a fast path. Release machine code does not change. (warp-1.14)

Archive note (session 28): the push of this session's notes was refused by the pre-push `blobs` gate (`notes.md` 1003 KB, limit 1000 KB). Per the moat-porting procedure, porting sessions 4 to 9 and the evidence sessions between them (13 sections, 210 KB, all dated 2026-11-19 and superseded as platform baselines) were moved verbatim to `notes-archive.md` in commit `d647a77b`, with a pointer under the title; every `## Lesson for` section stayed here.
