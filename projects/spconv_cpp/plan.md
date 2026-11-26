# spconv_cpp port plan

Planner: linux-gfx90a fleet job, 2026-11-26. Read-only analysis of upstream `aa1021e68` (2026-04-14 head, full history fetched blob-less), spconv `263d6b474` (= tag v2.3.8, master head), cumm `d971d02a0` (and tags v0.5.3 to v0.8.2) and pccm `860790f` (PyPI 0.4.16). Nothing was built and the fork was not touched. Built on cumm's plan (`port/cumm` at `1614d936`, `projects/cumm/plan.md`); spconv's planning run was in progress on another job when this was written (`port/spconv` at `2edafec1`, lock taken, no plan yet), so the interface this port needs from spconv is stated below as requirements, not as a reading of its plan.

## Project

- Name: spconv_cpp. Upstream https://github.com/autowarefoundation/spconv_cpp, default branch `main`. Fork https://github.com/AMD-Ecosystem/spconv_cpp, port branch `moat-port`.
- What it is: a distribution snapshot, not a source project. `spconv/` is the output of `python -m spconv.gencode --inference_only=True` (spconv 2.3.8) plus CMake/CPack files that build `libspconv.so` and a Debian package; `cumm/` is a header-only CMake package (`cumm::cumm`, `cumm::cumm_cpu`) wrapping a copy of cumm's `tensorview` headers. Consumers (Autoware's `autoware_tensorrt_plugins` and `autoware_bevfusion`) do `find_package(cumm)` + `find_package(spconv)`, link `spconv::spconv`, and call `SpconvOps::get_indice_pairs{,_implicit_gemm}`, `ConvGemmOps::implicit_gemm`, `ConvGemmOps::indice_conv` (native), `Point2Voxel::point_to_voxel_hash_static`, with fp16 and fp32 activations, on a caller-provided (TensorRT, not zeroed) workspace through `StaticAllocator`.
- Rulings that bind this plan: cumm's licence condition (the four GPL-3.0-or-later WITH GCC-exception-3.1 headers are never modified; this port copies the ported cumm's headers verbatim, so the condition holds by construction); cumm's and spconv's scope ruling "Option B" (MFMA and WMMA families for fp16/bf16), which this snapshot inherits by regeneration; ccimport and pccm consumed unmodified from PyPI; no `cumm-rocm`/`spconv-rocm` package names (per-ROCm-major names such as `spconv-rocm7`).

## What upstream actually contains (pinned)

- History: 10 commits. `e99acba9` (2025-01-23) is the generated tree (22,437 files). Every later commit hand-edits generated or copied files: `969fe3e0`, `e45eba5a`, `111ffdf6` rewrote all 1,548 generated per-target `CMakeLists.txt` (gencode lists, a `CMAKE_CUDA_COMPILER_VERSION` switch for sm_101/sm_110); `aa1021e6` rewrote `SpconvOps_sort_1d_by_key{,_split}_allocator_v2.cu` from Thrust to `cub::DeviceRadixSort` and added (11,0) to `CompileInfo_arch_is_compiled_gemm.cc`; `5ce0f623` made `SpconvOps_reverse_bits.cu`'s `reverse(uint64_t)` throw (wrong return type, 32-bit shift UB); `1b23b70d` edited `tensorview/cuda/arch_defs.h`. So the CUDA tree is no longer pure generator output, and a regeneration cannot reproduce it.
- Versions: the `tensorview` copy is cumm **v0.5.3** (`2bb4166`, 2024-06-12) plus two edits (`core/printf2.h` adds `<cstdint>`, `cuda/arch_defs.h` adds SM entries); 75 header files differ from v0.8.2. spconv v2.3.8's `setup.py` pins `cumm>=0.7.11,<0.8.0`, while `HOW_TO_UPDATE.md` says cumm v0.8.2 and the README says 0.5.3. The generator cumm version is therefore unpinned upstream; this plan does not depend on it, because the CUDA tree stays as it is.
- Generated structure: 1,548 `CMakeLists.txt` (one OBJECT library per generated class, emitted by `pccm.builder.pybind.gen_cmake`), 858 `.cu`, 154 `.cc`, 19,759 headers. Kernel libraries: GEMM main SIMT 30, Volta 10, Turing 16; conv main SIMT 16, Volta 8, Turing 110, Ampere 201 (each with a `cp*` params library); 52 non-kernel libraries (ops, tuners, allocators, CPU references, `CompileInfo`, the two `*MainUnitTest` dispatchers). Link: `cudart cublasLt`.
- No NVRTC in libspconv: `{Conv,Gemm}TunerSimple::cached_get_nvrtc_params` throw "must be overrided in python"; gencode drops `is_nvrtc` kernels (it keeps int8 ones). No tests, no CI, no examples. Linux only (Debian packages, ARM64 cross toolchain for Jetson/Thor).

## Existing AMD support

None at this level: no HIP in upstream, no AMD fork with a regenerated snapshot (intake). AMD work exists only for spconv/cumm (`ZJLi2013/spconv_rocm` branch `rocm`, `HYGON-AI/spconv-das`), ruled not to be inherited. Not authoritative, not used.

Delivery vehicle: upstream is a packaging repository maintained by TIER IV engineers that takes outside PRs (7 of 9 merged). It does not link platform forks. A ROCm snapshot is a large generated addition; whether it goes up as one PR is a later moat-checkup and maintainer question (notes, question 3), not a planning blocker.

Mechanical versus AMD-native: decided upstream of this port. The kernels in the HIP snapshot are whatever the ported cumm generates: SIMT (mechanical) plus the AMD-native MFMA/WMMA families. This port writes no kernel code.

## Build classification

**CMake** (`set-ext-type spconv_cpp cmake`). Evidence: `spconv/CMakeLists.txt:3` `project(spconv LANGUAGES CXX CUDA VERSION 2.3.8)`, `add_subdirectory(src)`; `spconv/src/CMakeLists.txt` (generated) adds 1,547 OBJECT libraries and `add_library(spconv SHARED $<TARGET_OBJECTS:...>)`; per-target flags sit inside `$<$<COMPILE_LANGUAGE:CUDA>:...>` (nvcc) and `$<$<COMPILE_LANGUAGE:CXX>:...>` (g++); `cumm/CMakeLists.txt` uses `find_package(CUDAToolkit)` and `CUDA::cudart`. No torch, no Python at build time. But the sources are generated by a pccm/ccimport Python build (cumm's plan, "Build classification"), so the port has two halves: a regeneration step (Python, run by a person or porter when the snapshot is updated, output committed) and a CMake HIP build of the result.

## Port strategy

Strategy-A-shaped CMake HIP build over a **second, regenerated tree**, leaving the CUDA tree byte-identical.

Why a second tree, and not a HIP branch through the existing files:
- The kernel sets differ by backend. The CUDA tree's 664 Volta/Turing/Ampere kernel libraries are NVIDIA tensor-op PTX with no HIP spelling (2,832 files with `cp.async`, 654 with `ldmatrix`, 345 with `mma.sync`); the HIP set is SIMT plus the AMD families, generated by a different cumm with different descriptor lists, `CompileInfo` and tuner code. No single generated tree serves both.
- The CUDA tree carries hand edits that regeneration would lose (above), and its `tensorview` is cumm v0.5.3 while the ported cumm is v0.8.2+ with HIP branches. Replacing either changes the CUDA build.
- Hand-hipifying the existing 22k generated files is the wrong port (generated code belongs to its generator) and would lack the AMD families.
- Running gencode at CMake time on HIP would add a Python + pccm + cumm + spconv build dependency to a repository whose purpose is to remove it.

Layout (mirrors upstream's two projects, so consumers see the same package and target names):

```
rocm/
  cumm/    CMakeLists.txt  cmake/{cummConfig.cmake.in,Packing.cmake}  include/tensorview/...  version.txt
  spconv/  CMakeLists.txt  cmake/{spconvConfig.cmake.in,Packing.cmake,hip_flags.cmake}  include/spconvlib/...  src/...  version.txt
  tools/   regenerate.py (or .sh)  patches/*.patch
  test/    CMakeLists.txt  test_*.cc  consumer/
```

1. **Regeneration recipe** (`rocm/tools/`), the ROCm counterpart of `HOW_TO_UPDATE.md`, made repeatable: install the ported cumm and spconv at pinned commits (fork `moat-port` SHAs recorded in `rocm/*/version.txt` or a `rocm/SOURCES` file), set the HIP build environment cumm defines (`CUMM_HIP_ARCH_LIST=all` or its final spelling), run `python -m spconv.gencode --include=rocm/spconv/include --src=rocm/spconv/src --inference_only=True`, copy the ported cumm's `include/tensorview` verbatim to `rocm/cumm/include/`, then normalise and patch (next items). Running it twice on the same inputs must give a byte-identical tree; the porter proves that once.
2. **Normalisation of the generated CMake.** pccm's `gen_cmake` (`pccm/core/__init__.py:2386-2489`) emits only the `nvcc` flag bucket (under `COMPILE_LANGUAGE:CUDA`) and the `g++`/`cl` bucket (chosen by the *generating* OS, under `COMPILE_LANGUAGE:CXX`), writes the generating host's absolute include paths, a `find_package(CUDAToolkit)`, `CUDA_STANDARD`, and bare library names. cumm's HIP route puts its flags in the `clang++` bucket, which `gen_cmake` never writes, and pccm is not ours to change. So the normaliser (a deterministic text transform of pccm's fixed template, in the regeneration script) rewrites each per-target file into HIP form: `set_source_files_properties(<srcs> PROPERTIES LANGUAGE HIP)`, `HIP_STANDARD`, the defines and flags from one generated `cmake/hip_flags.cmake` (written at generation time from cumm's own HIP flag set, the `get_hip_build_kwargs()`-style helper cumm's plan exports, never hand-copied), no absolute include paths (includes come from `hip::host` and the ROCm library targets), and OS-neutral host flags (`-fPIC`/`-O3` only off Windows; clang for an msvc target rejects `-fPIC`). The top-level `rocm/spconv/src/CMakeLists.txt` links `hip::host` (amdhip64) and `roc::hipblaslt` (whatever `SimpleExternalSpconvMatmul` uses after the spconv port) instead of bare names. Fallback if text normalisation proves brittle: keep pccm's output and have `rocm/spconv/CMakeLists.txt` walk the targets after `add_subdirectory` and set `LANGUAGE HIP` with `TARGET_DIRECTORY` (CMake 3.21 for the HIP language anyway); the porter picks one and records why.
3. **Post-generation patches** (`rocm/tools/patches/`, applied by the script, each with a one-line reason): the `reverse(uint64_t)` guard from `5ce0f623` (a host-side upstream bug, the same in the regenerated code; spconv `csrc/sparse/all.py:1228`). Not carried: `printf2.h <cstdint>` (cumm main already has it) and the `arch_defs.h` SM list (NVIDIA-only enum, reachable only through `cuda/all.h`). The `aa1021e6` Thrust-to-CUB sort rewrite was a CUDA 13 CCCL fix; on HIP the regenerated code uses whatever the spconv port emits (rocThrust or hipCUB); the porter carries it only if that code fails, and records the decision.
4. **CMake projects** `rocm/cumm` and `rocm/spconv`: copies of upstream's two top-level `CMakeLists.txt`/`cmake/*` with the CUDA pieces replaced (`project(... LANGUAGES CXX HIP)`, `find_package(hip)`, `hip::host` on the INTERFACE of `cumm::cumm` with `TV_CUDA` and the HIP platform defines that cumm's headers need in a host TU, `find_dependency(hip)` in the config files, `CMAKE_HIP_ARCHITECTURES` defaulting to the generated arch list, version from `version.txt`). Package and target names stay `cumm`/`spconv`, `cumm::cumm`, `spconv::spconv`, so Autoware's `find_package` lines work unchanged; the Debian package names follow the per-ROCm-major rule (`cumm-rocm7`, `spconv-rocm7`, with `Conflicts:` on the CUDA packages), notes question 2.
5. **Arch list consistency.** The generated `CompileInfo` records the gfx list at generation time; a build for a different list would make `arch_is_compiled_gemm` lie and fail with "invalid device function". The HIP CMake accepts a subset of the generated list (narrowing for a faster local build) and refuses (FATAL_ERROR) any target outside it.
6. **README and `HOW_TO_UPDATE.md`**: a ROCm build section (documented commands, no device-selection variables) and a ROCm regeneration section. No other shared file changes; `cumm/`, `spconv/` and `extras/` stay byte-identical.

## Interface this port needs from cumm and spconv

These are requirements on the generators, written here because spconv's plan was not yet written; the spconv_cpp porter checks each against the pinned commits before regenerating, and a missing one goes back to that project rather than being patched into the generated tree.

1. cumm: a generation mode that works with no GPU on the generating host and an explicit `CUMM_HIP_ARCH_LIST` (generation must not need the target hardware).
2. cumm: AMD-family kernel bodies guarded per device pass, so one translation unit compiles for CDNA and RDNA targets together (MFMA builtins under a gfx11/gfx12 pass and WMMA under gfx9 must not be emitted). spconv's own multi-arch wheels need the same.
3. cumm: its HIP flag and define set exposed as data (the helper in cumm's plan), so the normaliser can write `hip_flags.cmake` from it.
4. cumm: generated `CompileInfo` keyed by gfx name; on HIP never routes to the NVRTC fallback (libspconv has none: `cached_get_nvrtc_params` throws). A device whose gfx is not compiled must fail with a clear message.
5. spconv: generated C++ tuners (`ConvTunerSimple::get_all_available`, `GemmTunerSimple::get_all_available`) select by family on HIP, not by CC tuple. Today `arch >= (7,0) && is_fp16` drops every SIMT fp16 kernel "if we have tensor core", so on gfx90a (CC 9.0) with no matching AMD family no algorithm survives: the fp16 wall jeff's study hit, here in C++. On HIP SIMT fp16 may be skipped only when an AMD-family kernel exists for that device.
6. spconv: `inference_only=True` keeps working with the HIP descriptor lists (it filters to `kForward` and drops `ShuffleAB`), and the int8-inference rule ("keep all int8 kernels in libspconv") does not reference NVIDIA families that are absent on HIP.
7. spconv: the CUDA-library substitutions in generated code (`cub::` / `thrust::cuda::par` / cublasLt) done in the generator, so the regenerated tree needs no hand edits beyond the patch list.
8. spconv/cumm versions: spconv 2.3.8 pins `cumm<0.8.0` and the cumm port is on 0.8.2+. Whatever spconv's port does about that pin decides which cumm the regeneration uses; this port only records the two SHAs.

## CUDA surface inventory

All of it is generated (or copied from cumm) and owned by cumm or spconv; this port changes none of it. Census at `aa1021e68` (`surface.json`, `cuda` section): 13,362 files; 945 kernels, 989 warp-intrinsic sites, 408 runtime-PTX sites (inline `asm`), 1 launch, 1 driver-API site; libraries thrust (79 files), cub (8), cublas/cublasLt (50), curand (11, tensorview `tensor.h`), cutlass (3). Hand counts: `__CUDA_ARCH__` 4,893 files, `__shfl` 782, `cudaFuncSetAttribute` 16 (the two `*MainUnitTest` dispatchers, smem opt-in).

| area | where | HIP treatment (by whom) |
|---|---|---|
| NVIDIA tensor-op kernels | 664 Volta/Turing/Ampere kernel libraries | not in the HIP tree; AMD MFMA/WMMA families generated instead (cumm) |
| SIMT kernels | 62 libraries (GEMM 30, conv 16 + 16 params) | regenerated with the HIP backend: logical 32-lane warps, body gates, PTX replacements (cumm) |
| Ops kernels | `SpconvOps`, `Point2Voxel` 1-4D, `SparseConvIndicesKernel` 1-4D, `HashTable`, `IndiceMaxPool`, `BoxOps`, `LaunchUtils` | regenerated (spconv: hash and pair kernels, Thrust/CUB to rocThrust/hipCUB) |
| GEMM library | `SimpleExternalSpconvMatmul` (cublasLt) | hipBLASLt (spconv); link `roc::hipblaslt` |
| Tuners and dispatch | `ConvTunerSimple`, `GemmTunerSimple`, `ConvGemmOps`, `*MainUnitTest` (smem opt-in) | family-keyed selection (spconv), 64 KiB smem check (cumm) |
| Arch identity | `CompileInfo` (CC tuples, PTX, NVRTC table; hand edits) | gfx names, no PTX, no NVRTC (cumm) |
| tensorview runtime | `cumm/include/tensorview` (v0.5.3) | replaced in the HIP tree by the ported cumm's copy |
| Build | 1,548 generated CMake files, 2 top-level, 2 config templates, CPack | new `rocm/` projects + normaliser (this port) |

## Risk list

- **Generated CMake loses HIP flags silently** (highest): a `.cu` with no `LANGUAGE HIP` in a CUDA-less project is ignored by CMake, not an error, so a library can link with kernels missing. Mitigation: the normaliser's output is checked by a count (number of HIP objects built = number of generated `.cu`), and the harness asserts the expected kernel family ran.
- **Absolute paths from the generating host** in generated CMake (pccm writes `get_global_includes()`, e.g. `/usr/local/cuda/include` in the CUDA tree). Normalise them away; a grep for `/opt/rocm`, `site-packages` and the job directory over `rocm/` must be empty.
- **Arch-list drift** between `CompileInfo` and the build (item 5 above).
- **CC collision** (gfx90a 9.0, gfx942 9.4, gfx11 11.x): consumers pass `ConvGemmOps::get_compute_capability()` as `arch`; selection must not depend on it (spconv requirement 5).
- **Fresh device memory is not zero on ROCm**: Autoware hands libspconv an un-zeroed TensorRT workspace through `StaticAllocator`; the harness poisons every workspace (0xFF) before each call.
- **Wavefront width**: inherited from cumm (32-lane logical SIMT warps on wave64, wave64 MFMA warps); validate both widths.
- **Shared memory**: generated configurations above 64 KiB must not be in the HIP tree (cumm drops them) or must fail clearly; the `*MainUnitTest` opt-in path is the place to check.
- **Windows**: host `.cc` compile with clang (one driver for CXX and HIP); `-fPIC` rejected for an msvc target; MSVC/clang strictness in code built only with nvcc + g++; CPack DEB is Linux-only (build and test only).
- **Build cost**: a HIP tree of SIMT plus two families times nine gfx targets; cap parallelism at physical cores; a per-host build narrows `CMAKE_HIP_ARCHITECTURES` to its own gfx.
- **Repository size**: the HIP tree adds a few thousand generated files (estimate 3k to 8k files, 30 to 80 MB, dominated by per-kernel headers; the CUDA tree is 22k files, 196 MB). Recorded so the maintainers' view (question 3) is informed.
- **Licence boundary**: `git diff` of the four GPL headers between the ported cumm and `rocm/cumm/include` must be empty in every round.
- **`arch_defs.h`**: has a duplicate enumerator (`kSm80 = 80` and `kSm80 = 90`) in every version; harmless only while nothing includes `tensorview/cuda/all.h`. Note, do not fix.

## File-by-file change list

All new except the two docs. Nothing under `cumm/`, `spconv/` or `extras/` changes.
- `rocm/tools/regenerate.py` (or `.sh`), `rocm/tools/patches/0001-reverse-uint64-guard.patch`, `rocm/SOURCES` (pinned cumm and spconv commits and the gfx list).
- `rocm/cumm/CMakeLists.txt`, `rocm/cumm/cmake/cummConfig.cmake.in`, `rocm/cumm/cmake/Packing.cmake`, `rocm/cumm/version.txt`, `rocm/cumm/LICENSE` (cumm's, as upstream's `cumm/LICENSE`), `rocm/cumm/include/tensorview/**` (copied).
- `rocm/spconv/CMakeLists.txt`, `rocm/spconv/cmake/{spconvConfig.cmake.in,Packing.cmake,hip_flags.cmake}`, `rocm/spconv/version.txt`, `rocm/spconv/LICENSE`, `rocm/spconv/include/**` and `rocm/spconv/src/**` (generated, normalised).
- `rocm/test/CMakeLists.txt` (option `SPCONV_BUILD_TESTS`, CTest), `rocm/test/test_point2voxel.cc`, `test_indice_pairs.cc`, `test_conv.cc`, `test_maxpool.cc` (names the porter's), `rocm/test/consumer/{CMakeLists.txt,main.cc}`.
- `README.md` (ROCm section), `HOW_TO_UPDATE.md` (ROCm regeneration section).

Expected size: hand-written 15 to 25 files, 1,000 to 2,000 lines (tests about half); generated as above. Porter rounds: 3 to 5, and the first cannot start until cumm and spconv reach a validated head on Linux (the selector enforces `depends_on`). Order: (1) regeneration script and normaliser against the ported generators, reproducibility proof; (2) CMake projects, build on one wave64 host, consumer test; (3) harness, both widths; (4) Windows.

## Build commands

Regeneration (Linux, any host with the ported cumm and spconv installed in a venv; no GPU needed if cumm requirement 1 holds):

```
python -m venv .venv && . .venv/bin/activate
pip install pccm ccimport pybind11 numpy fire
pip install <cumm port at the pinned commit> <spconv port at the pinned commit>
python rocm/tools/regenerate.py            # writes rocm/cumm/include and rocm/spconv/{include,src}
git status --porcelain rocm/               # second run on the same inputs: empty
```

Build and package (Linux, ROCm from `/opt/rocm` or the pip SDK on `CMAKE_PREFIX_PATH`; `<gfx>` is the executing GPU):

```
cmake -S rocm/cumm -B rocm/cumm/build -DCMAKE_PREFIX_PATH=<rocm root>
cmake --build rocm/cumm/build && cmake --install rocm/cumm/build --prefix <prefix>
(cd rocm/cumm/build && cpack -G DEB)
cmake -S rocm/spconv -B rocm/spconv/build -DCMAKE_PREFIX_PATH="<rocm root>;<prefix>" -DCMAKE_HIP_ARCHITECTURES=<gfx> -DSPCONV_BUILD_TESTS=ON
cmake --build rocm/spconv/build -j <physical cores>
ctest --test-dir rocm/spconv/build --output-on-failure
cmake --install rocm/spconv/build --prefix <prefix> && (cd rocm/spconv/build && cpack -G DEB)
```

Wrap with `utils/timeit.sh spconv_cpp compile|test -- ...`. Windows: the same configure/build/ctest from a shell with the ROCm SDK (TheRock) on PATH, `-G Ninja`, the SDK's clang for both `CMAKE_CXX_COMPILER` and `CMAKE_HIP_COMPILER`, `<gfx>` = gfx1151; no CPack.

## Test plan

Upstream has no tests. The port ships a CTest harness under `rocm/test` (GPU tests skip with a clear message when no device is present), modelled on spconv's `example/libspconv/main.cu` flow and Autoware's plugin calls, with synthetic, seeded point clouds (no data file copied), comparing GPU to the CPU reference implementations that libspconv itself contains (`Point2VoxelCPU`, `SparseConvIndicesCPU`, `GatherCPU`, `IndiceMaxPoolCPU`) or to a plain C++ reference:

1. Point2Voxel: `point_to_voxel_hash_static` 3D vs `Point2VoxelCPU` (voxel set, counts and per-voxel point sets compared as sets; GPU order is not CPU order); 1D/2D/4D smoke.
2. Indice pairs: `get_indice_pairs` (kNative) and `get_indice_pairs_implicit_gemm` (MaskImplicitGemm and MaskSplitImplicitGemm, subm and regular, kernel 3 and stride 2) vs `SparseConvIndicesCPU`, with every workspace poisoned to 0xFF first, as a TensorRT workspace would be.
3. Convolution: `indice_conv` (native, through `SimpleExternalSpconvMatmul` / hipBLASLt) and `implicit_gemm` for fp32 and fp16 activations, C in {16, 32, 64, 128}, against a C++ gather-GEMM-scatter reference in double; tolerance per dtype (fp32 relative 1e-5 against the double reference; fp16 scaled by K). Assert which algorithm family the tuner picked for fp16: the device's MFMA family on CDNA, WMMA on RDNA, and run fp32 through SIMT, so a green run cannot be a silent SIMT fallback or an empty kernel (also check the output is not all zeros and count dispatches once with the HIP runtime log).
4. Max pool: `IndiceMaxPool` forward vs `IndiceMaxPoolCPU`.
5. Consumer: configure and build `rocm/test/consumer` against the installed packages with `find_package(cumm)` + `find_package(spconv)`, link `spconv::spconv`, run one implicit-GEMM convolution (catches export, RUNPATH and INTERFACE-define faults).
6. Packaging (Linux): `cpack -G DEB` produces both packages; `dpkg-deb -c` lists the headers, `libspconv.so` and the CMake config files.

Non-GPU regression set: the CUDA tree is byte-identical to upstream (`git diff --stat <upstream base> moat-port -- cumm spconv extras` empty); the CPU-only `cumm::cumm_cpu` target still configures without ROCm; the regeneration is reproducible (second run leaves `git status` clean).

Per gate:
- **wave64**: gfx90a or gfx942 (both wanted: the CC collision differs, 9.0 vs 9.4). Items 1-6; item 3 asserts the MFMA family.
- **wave32**: gfx1100 and gfx1201. The two WMMA layouts are different code, so the gfx12 run is owed even after a gfx11 pass satisfies the gate. Items 1-6; item 3 asserts the WMMA family.
- **windows**: windows-gfx1151, last, after both Linux widths pass at the same head: items 1-5 (no CPack). Upstream and its consumer are Linux-only; whether to validate or to ask for a waiver is notes question 1. Planned as validate.
- **CUDA no-regression** (mi355 host, real nvcc, before any round goes to jeff): the byte-identity diff above, plus `cmake -S spconv -B build-cuda` configure and a build of a sample of targets (`spconvlib_cumm_common_CompileInfo`, one SIMT GEMM, one Ampere conv, `spconvlib_spconv_csrc_sparse_all_SpconvOps`) against the upstream `cumm` package, since the README build itself is unchanged.

Downstream evidence, not a gate: Autoware's `autoware_tensorrt_plugins` reference tests on ROCm, when autoware_universe stage 4 starts and adds this project to its `depends_on`.

## Open questions

None blocks the plan or the porter; the porter cannot start before cumm and spconv are validated anyway. Recorded in `notes.md` for jeff:
1. Windows gate: validate (planned) or suggest a waiver (Linux-only packaging repo; its only consumer, autoware_universe, has an approved windows waiver).
2. Debian package names for the ROCm build (`cumm-rocm7` / `spconv-rocm7` proposed, CMake names unchanged).
3. Upstream vehicle for a large generated addition (one PR with `rocm/`, or a separate snapshot repository or branch); a question for the maintainers, through moat-checkup, after validation.
4. int8 inference on HIP follows cumm question 1 (int8 stays on whatever the spconv/cumm HIP lists provide); Autoware's spconv plugins assert fp16/fp32 only.
