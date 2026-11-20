# oidn port plan

**Status: waiting on Jeff Daily's rulings.** The questions are at the top of the planner section in `notes.md` (2026-11-20). The plan below is written for the recommended answers (Q1 option A, Q2 no gfx908, Q3 base branch `master`, Q4 local submodule override, Q5 training scoped out). A porter must not start until Q1 is ruled, because the other answers change the branch base and the scope.

## Project

- Name: oidn (Intel Open Image Denoise)
- Upstream: https://github.com/RenderKit/oidn, default branch `master` (development happens on `devel`, which is 3 commits ahead of `master` and touches no HIP or GPU file, checked 2026-11-20)
- Fork: https://github.com/AMD-Ecosystem/oidn, GitHub default branch `master` at 6602ee2ca38a (= upstream `master`, v2.5.1 docs update of 2026-08-14). The MOAT record says `fork_default_branch: main`, which does not exist on the fork (Q3).
- Licence: Apache-2.0 (intake).
- Planned against fork `master` 6602ee2ca38a1a2a02135beed8f6e68eed630180 and upstream PR #302 head 75bc1d936dfd77fb860acf7cfab2270c150280c0.

## Existing AMD support

Mature, upstream, authoritative: `devices/hip/` is an Intel-maintained HIP backend that uses Composable Kernel (CK) convolutions. It ships in the release binaries for Linux and Windows. It covers RDNA2 (`ck_conv_dl.cpp`, CK DL path) and RDNA3/3.5/4 (`ck_conv_wmma.cpp`, CK WMMA path). AMD already contributes to it (PR #267, merged as 5a962dc; PR #288).

New since intake (2026-11-05): **https://github.com/RenderKit/oidn/pull/302**, "HIP: add CDNA4 gfx950 (AMD Instinct MI350) support", opened 2026-11-06 by `wensongc`, whose GitHub profile names AMD as employer. State on 2026-11-20: open, mergeable, no reviews or comments, one commit on top of `master` 6602ee2. Judgement: **authoritative AMD work in progress, but incomplete.** It enables gfx950 only. The author reports testing it only on MI350X (`oidnTest --device hip`, 2403 assertions passed; MAE 1.8e-4 against the CPU device). RDNA was only compiled, and nothing was built on Windows. It adds:
- `devices/hip/ck_conv_mfma.cpp`: CK `DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle` with one tile (256 threads, 128x64x64, AK1=BK1=16, 32x32 XDL). It does `#define warpSize 64` before including CK so that `ck::get_warp_size()` is constexpr.
- `devices/hip/CMakeLists.txt`: a gfx950 compiler probe, the `OIDN_HIP_GFX950` (default ON) and `OIDN_HIP_TARGETS` options, and per-source `--no-offload-arch` lists so that the WMMA/DL files never build for gfx950 and the MFMA file builds only for it. `DL_KERNELS;CK_USE_WMMA` move to per-source definitions on the DL/WMMA files, and CK's `config.h` is now generated in the build tree.
- `hip_device.{h,cpp}`: a new `HIPArch::MFMA` for gfx950, with `tensorBlockC = 32`. `hip_conv.cpp` adds the dispatch case.

Decision (pending Q1): do not re-port and do not open a competing PR (the lesson MOAT learned on gsplat). The differentiator we can add is (1) the rest of CDNA, gfx90a (MI200) and gfx942 (MI300), which needs the same path generalised from one target to a list, plus an XDL tile valid where the double-rate f16 MFMA does not exist; (2) the first real RDNA (wave32) and Windows runs of the PR's restructured build; (3) fixes to whatever those runs break.

Port versus rewrite: not applicable. CK is already the AMD-native target and nothing is translated from CUDA. The CUDA device (`devices/cuda/`, CUTLASS) is a separate backend and stays untouched.

Delivery vehicle: not a new upstream PR while #302 is open. Coordinate with the PR's author: our commits go onto his branch, or into a follow-up PR after #302 merges. This needs a person's decision (Q1). MOAT's normal `--publish` step would open a competing PR.

## Build classification

`cmake`, recorded with `set-ext-type`. Evidence: top-level `CMakeLists.txt` options `OIDN_DEVICE_HIP` (line 33) and the others. `devices/CMakeLists.txt:91-185` builds the HIP device as an `ExternalProject` with ROCm's `clang++` as `CMAKE_CXX_COMPILER` (Ninja or Make only). `devices/hip/CMakeLists.txt` uses `find_package(hip)` with `hip::device`, and `GPU_TARGETS` is a hard-coded `CACHE INTERNAL` list. There is no torch dependency in the library. `training/` is pure PyTorch scripts and is not part of the CMake build.

## Port strategy

Neither A nor B. The HIP backend is written natively against HIP: it has no compat header, runs no hipify, and contains no CUDA spellings. The work is (option A) extending a native HIP backend to more targets, plus validation. The CUDA path is untouched by construction: every planned edit is under `devices/hip/`, except README/CHANGELOG lines and, if #302's top-level option survives, one `option()` line and its pass-through in `devices/CMakeLists.txt`.

## CUDA surface inventory

The `surface.py` census found 12 files. None of them is port surface in the CUDA-to-HIP sense:
- `devices/cuda/*` (cuda_conv.cu, cuda_engine.{cu,h}, cuda_device.cpp, curtn.cpp driver-API shim (26 driver-API hits), cutlass_conv*.{h,cu} (48 CUTLASS hits)): the NVIDIA backend, scoped out of the port. It stays byte-identical and only the nvcc no-regression compile touches it.
- `common/platform.h`, `core/kernel.h`: shared kernel abstraction. HIP already has its own branches here (`OIDN_COMPILE_HIP`): `__shfl(x, id)` with the default width `warpSize`, `getSubgroupSize()` = `warpSize`, and `getSubgroupLocalID()` = `% warpSize`. All of these are width-agnostic.
- `devices/hip/hip_device.cpp`, `hip_engine.h`: these are the HIP backend itself (the census matched `__global__` and `<<<`).

Wavefront-relevant shared kernels, `devices/gpu/` (compiled into the HIP device):
- `gpu_input_process.h`: the HIP path uses the runtime subgroup size and requires `subgroupSize % dstPaddedC == 0`. With wave64 and `tensorBlockC` 8 or 32, `dstPaddedC` is at most 32 (input channels <= 9), so 64 % 32 == 0 holds and the group is `{1, 64}`. The transpose loop `i += subgroupSize / dstPaddedC` is width-generic. Wave64-safe by reading, and #302's gfx950 run is evidence for it.
- `gpu_autoexposure.h`: shared-memory tree reductions with a `groupBarrier()` per step and 1024-thread groups. Width-independent.
- `gpu_pool.h`, `gpu_upsample.h`, `gpu_output_process.h`, `gpu_image_copy.h`: element-wise kernels with no subgroup operations.
- `HIPEngine::suggestWorkGroupSize`: 1024 / 32x32 / 1x32x32. Valid on both widths (1024 threads is HIP's default launch bound).

Libraries: Composable Kernel only (submodule `external/composable_kernel` pinned at 3e6d21adeb33, 2025-04-09). That pin already carries `device_grouped_conv_fwd_multiple_abd_xdl_cshuffle{,_v3}.hpp`, and its `ck.hpp` defines `__gfx9__` for gfx908/gfx90a/gfx942/gfx950, so the CDNA path needs no CK bump. No BLAS, FFT, RNG, Thrust/CUB, textures, or graph capture. Memory: `hipMalloc`, `hipMallocManaged` (managed buffers in tests), and `hipHostMalloc`-style host storage through USM. External memory: OpaqueFD (Linux) and D3D11/D3D12/Win32 handles (Windows). External semaphores: Windows only (Linux reports `None`).

## Risk list

1. **Wavefront width across translation units.** `ck_conv_wmma.cpp` is compiled `-mcumode -mno-wavefrontsize64` (only that file; intake's note said the whole backend, which is wrong). A CDNA target must never reach that file, and #302 handles this with per-source `--no-offload-arch`. Generalising to gfx90a/gfx942 must extend those exclusion lists, and also exclude CDNA from `ck_conv_dl.cpp`. Verify with `--save-temps` or `clang-offload-bundler --list` on the built objects, not by reading CMake.
2. **`#define warpSize 64` in `ck_conv_mfma.cpp`.** This is safe only while that file builds for wave64 targets alone. Any change that lets an RDNA target into it silently runs wave32 hardware with 64-lane CK tiling. Keep the exclusion and the define together, and add a `static_assert` or `#error` under `__gfx10__/__gfx11__/__gfx12__` in that TU if the review wants a guard.
3. **CK configuration split across TUs.** #302 moves `DL_KERNELS;CK_USE_WMMA` from target-wide to per-source. `hip_conv.cpp` and the MFMA file now include CK headers under a different macro set than the DL/WMMA files, so inline CK code can differ by TU (an ODR hazard). Check that `hip_conv.cpp` uses only declarations that do not depend on those macros. The RDNA run is the evidence.
4. **XDL tile validity on gfx90a/gfx942.** #302's tile uses AK1=BK1=16 with 32x32 XDL, chosen for gfx950's double-rate `v_mfma_f32_32x32x16_f16`. gfx90a/gfx942 have `32x32x8` f16. CK may accept K1=16 by packing, or `IsSupportedArgument` may refuse it. `newHIPConv` selects by cost alone and never calls `isSupported()`, so a refused tile fails at run time, not at selection. Take a K1=8 tile from CK's own instance list at the pinned sha (`library/include/ck/library/tensor_operation_instance/gpu/grouped_conv_fwd/device_grouped_conv_fwd_xdl_instance.hpp`, the NHWGC/GKYXC fp16 entries), and make every tile pass `isSupported()` for every network layer on every CDNA target.
5. **Channel padding.** `tensorBlockC` must be a multiple of the tile's vector widths (#302 sets 32 for MFMA). `HIPEngine::newInputProcess` supports only 8 and 32. Any other value needs a new instantiation there.
6. **CK pin against new ROCm.** The CK pin is from April 2025. Upstream already needed `-Wno-invalid-constexpr` for ROCm 7. Newer ROCm (10.x) clang can break old CK templates. Treat such a break as a toolchain issue: guard the workaround in `devices/hip/CMakeLists.txt` to the compiler version, and draft the upstream bug for Jeff. Do not bump the submodule unless that is the only fix (a bump is a large upstream-visible delta).
7. **Build time.** The HIP device compiles about 19 RDNA targets plus CDNA, with heavy CK templates. Cap `MAX_JOBS` and `HIP_CLANG_NUM_PARALLEL_JOBS` on shared hosts. Upstream's default `HIP_CLANG_NUM_PARALLEL_JOBS` is every logical core *per source file*, multiplied by Ninja's jobs. `OIDN_HIP_TARGETS` (from #302) allows a fast single-arch inner loop, but validation uses the default target list, because that is what ships.
8. **Weights submodule.** `.gitmodules` uses the relative URL `../oidn-weights.git`. On the fork that resolves to `AMD-Ecosystem/oidn-weights`, which does not exist, so `git submodule update --init` fails. The weights repository also stores `*.tza` in Git LFS, so `git-lfs` must be installed or the build embeds 132-byte LFS pointers. Workaround (Q4): `git config submodule.weights.url https://github.com/RenderKit/oidn-weights.git` before `git submodule update --init --recursive`, with `git lfs install` first. Do not edit `.gitmodules` on `moat-port`, because upstream's relative URL is correct for upstream.
9. **Windows build.** `devices/CMakeLists.txt` looks for `bin/hipconfig{,.bat,.exe,.pl}` under `HIP_PATH`/`ROCM_PATH` or `%PROGRAMFILES%/AMD/ROCm/*`, and uses `clang++` from `ROCM_PATH/bin` or `llvm/bin`. With a TheRock SDK the layout may differ. #302's gfx950 probe and per-source `--no-offload-arch` were never run on Windows. Fix any break in the project's CMake, guarded to `WIN32` (no README-only workaround).
10. **Uninitialised device memory.** `hipMalloc` memory is not zeroed. Tensors are written before they are read in the normal pipeline, but the padding channels (from `round_up(C, tensorBlockC)`) must be written. `gpu_input_process.h` writes zeros for channels >= 9 (`dstBlock = 0`), so this looks covered. The fill-pattern differential is the cheap check if a CDNA result differs from CPU only near channel padding.
11. **Unsupported-arch behaviour is part of the contract.** On a GPU outside the list, `getPhysicalDevices` must hide it and `newDevice(HIP)` must report `UnsupportedHardware`. Under option B, gfx90a/gfx942 must keep doing exactly that.

## File-by-file change list (option A)

Branch `moat-port` on the fork from #302's head `75bc1d936dfd77fb860acf7cfab2270c150280c0`, fetched with `git fetch https://github.com/RenderKit/oidn pull/302/head`. Its parent is fork `master` 6602ee2, so this is a fast-forward of the mirror and keeps wensongc's authorship. Add our commits on top, and never rewrite his.

- `devices/hip/CMakeLists.txt`: replace the single gfx950 probe with a CDNA list (`gfx90a;gfx942;gfx950`, and gfx908 only if Q2 says so). Probe each target, append the accepted ones to `OIDN_HIP_TARGETS`, and build the MFMA source for exactly the accepted CDNA targets. Every CDNA target gets `--no-offload-arch` on the DL and WMMA sources. Keep `OIDN_HIP_TARGETS` as the full override. Option naming (`OIDN_HIP_GFX950` vs a CDNA-wide name) is coordinated with the PR author. The minimal default keeps his name for gfx950 and adds nothing new unless the generalisation requires it.
- `devices/hip/ck_conv_mfma.cpp`: add the K1=8 tile(s) for gfx90a/gfx942 (risk 4), selected by architecture (below). Keep the `warpSize` define under the wave64-only TU rule (risk 2).
- `devices/hip/hip_device.h/.cpp`: map `gfx90a`, `gfx942` (and gfx950 as #302 does) in `getArch`. If gfx950 and gfx90a/942 need different tile lists, use a second enum value, or pass the arch name into `getCKConvInstances`. The simplest form that keeps `newHIPConv` choosing only supported tiles wins. `tensorBlockC` follows the tiles (risk 5). The `#if defined(OIDN_HIP_GFX950)` guards widen to whatever macro marks "CDNA kernels built".
- `devices/hip/hip_conv.cpp`: the dispatch case(s) for the CDNA arch value(s).
- `README.md`, `doc/overview.md`, `CHANGELOG.md`: extend #302's lines from "CDNA 4 (gfx950)" to the CDNA generations actually enabled and validated.
- Fixes found by the RDNA and Windows runs go wherever the fault is, minimal, guarded to the platform when platform-specific.
- No edit to `devices/cuda/`, `devices/gpu/`, `core/` or `common/` is planned. If a wave64 fault is found in `devices/gpu/`, the fix must be correct on both widths (cuda-to-rocm portability rule) and must leave CUDA and SYCL untouched.

Option B: no planned source change. The fork's `moat-port` is #302's head, and edits are only the fixes that validation forces.

## Build commands

Prerequisites: CMake >= 3.21, Ninja, Python 3, ROCm (>= 6.4.2 per README), `git-lfs`. For the CPU device (the reference for the image differential and the non-GPU regression set), also ISPC >= 1.21 and oneTBB dev. Build missing ones from source or a release tarball inside the job directory, not into shared paths.

```
git clone https://github.com/AMD-Ecosystem/oidn src && cd src
git checkout moat-port
git lfs install
git config submodule.weights.url https://github.com/RenderKit/oidn-weights.git
git submodule update --init --recursive external/composable_kernel weights
cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Release \
      -DOIDN_DEVICE_HIP=ON -DOIDN_DEVICE_CPU=ON -DROCM_PATH=/opt/rocm \
      -DHIP_CLANG_NUM_PARALLEL_JOBS=<cap> \
      [-DTBB_ROOT=<prefix>] [-DISPC_EXECUTABLE=<path>/ispc]
cmake --build build -j <MAX_JOBS>
```

(The `cutlass` submodule is needed only for `OIDN_DEVICE_CUDA` and can be skipped on AMD hosts.) Without ISPC/TBB, `-DOIDN_DEVICE_CPU=OFF` still builds the HIP device and the apps, but loses the CPU reference. Record that as a gap if it happens. Windows: the same configure from a shell with TheRock's ROCm, with `ROCM_PATH`/`HIP_PATH` pointing at it, Ninja generator, and the host compiler per the cuda-to-rocm `windows.md` guidance.

Confirm the CDNA kernels are present: `roc-obj-ls` / `clang-offload-bundler --list` on `build/hip/build/libOpenImageDenoise_device_hip.so*` shows `gfx90a`, `gfx942` and `gfx950` code objects (option A).

## Test plan

GPU evidence on every platform (wave64: gfx90a, gfx942 and gfx950 under option A, gfx950 alone under B; wave32: an RDNA2/3/4 card; Windows: the gfx1151 laptop or another RDNA Windows host):

1. `build/oidnDenoise --ld` shows the GPU as a HIP device. Then `build/oidnTest --device hip`: all 20 test cases (physical device, device creation/params, error callback, buffers incl. managed and async, single/multiple filters, single/multiple devices, shared image, in-place filter, filter update, async filter, image size, sanitization, progress monitor, user weights). Record the pass/fail/assertion counts; #302 reports 2403 assertions on gfx950. Count a GPU dispatch at least once (`AMD_LOG_LEVEL=3` or `rocprofv3 --kernel-trace` on one filter run) so a green run cannot be a silent CPU fallback. `--device hip` fails outright when there is no HIP device, which also guards against fallback.
2. Image differential against the CPU device (upstream's `scripts/test.py` regression needs Intel's private baseline repository `OIDN_VALIDATION_REPO`, so it is unusable here). Generate deterministic noisy HDR inputs with an albedo and a normal (numpy-written PFM, sizes including 1920x1080 and an odd size like 1023x577). Then for each of filter `RT` with `--hdr` (+`--alb` +`--nrm`), `--ldr --srgb`, `--clean_aux`, filter `RTLightmap` (`--hdr`, `--dir`), and each `-q high|balanced|fast`, `-t float|half`, `--maxmem 256` (tiled path), `--inplace`, `--buffer device|managed`:
   `oidnDenoise -d cpu <inputs> -o ref.pfm` then `oidnDenoise -d hip <inputs> -r ref.pfm` (default `--maxerror` 0.003, or 0.05 for normal/directional). Record numErrors/avgError per case. #302's own comparison gave MAE 1.8e-4 against CPU on gfx950.
3. `build/oidnBenchmark -d hip` once per platform: record the numbers as information, not a gate.
4. Non-GPU regression set: `build/oidnTest --device cpu`, all test cases pass, on at least one Linux host.
5. CUDA no-regression compile (on the host with nvcc): configure with `-DOIDN_DEVICE_CUDA=ON -DOIDN_DEVICE_CPU=OFF -DOIDN_DEVICE_HIP=OFF`, init the `cutlass` submodule, and build `OpenImageDenoise_device_cuda`. Run it once at the final head. The planned edits do not touch `devices/cuda/`, but the top-level option and pass-through do sit in shared CMake.
6. Under option B, on gfx90a/gfx942: `oidnTest --device hip` must report no HIP device / `UnsupportedHardware` cleanly (risk 11). That is the expected result, and the platform is then `set-blocked` with "unsupported by upstream design (CDNA2/3 not enabled)", not failed.

## Open questions

See the top of the planner section in `notes.md` (Q1 to Q5). Summary: Q1 scope and delivery relative to upstream PR #302 (A extend it to gfx90a/gfx942 and validate RDNA/Windows; B validate #302 as-is; C independent port, not recommended). Q2 gfx908. Q3 `fork_default_branch` is `main` in the record but `master` on the fork. Q4 weights submodule URL. Q5 training scripts scoped out.
