# walberla notes

## Recommendation (intake, 2026-11-20): decline as `already-supported`; do the real work in the espresso project

waLBerla already supports HIP upstream. Upstream added the `WALBERLA_BUILD_WITH_HIP` option in !508 (merged 2023-05-17), first released it in v7.1 (2025-04-11), and has run it on LUMI-G with up to 4096 MI250X. On gfx942 with ROCm 10.1, upstream master builds 663/663 once a one-line CMake version check is fixed. The gpu-module tests run 7 of 8 passed on the GPU, and the one failure tests undefined behaviour. A MOAT port of waLBerla would therefore port nothing. The gaps are small upstream fixes (below) plus the HIP CI job that upstream already plans in #347.

The lattice-Boltzmann and electrokinetics GPU code that ESPResSo users miss on AMD lives in ESPResSo, not in waLBerla. It is ESPResSo's `src/walberla_bridge`: its hand-written bridge and its 56 checked-in generated kernels, which ESPResSo's own scripts and templates produce. On this host 42 of the 56 kernels already compile as HIP. The remaining 14 compile after three small fixes, two in ESPResSo's templates and one in the vendored pystencils philox header. Separately, a ~12-line CMake change in the espresso port builds ESPResSo with HIP core GPU code and waLBerla's CPU solvers kept. It configured, built 377/377, and passed 131/131 C++ unit tests, including every waLBerla LB/EK suite. That change alone removes most of the user-facing cost.

Suggested `SKIP_REASON`: `already-supported`. This is a recommendation, not a decision. Declining, or keeping waLBerla open for the small upstream fixes, is jeff's call.

What jeff may want to decide (nothing here has been acted on):
1. Whether the espresso port gets a round that keeps waLBerla's CPU LB/EK on in the HIP build (question 3b). It is outside jeff's 2026-11-19 "nothing else" ruling for that round, so it needs a new ruling.
2. Whether a later espresso round (or a new release of the deferral `espresso-walberla-lb-ek-gpu`) does the full LB/EK GPU on HIP inside ESPResSo (question 3a). This round would be medium-sized and live in espresso, not in a waLBerla project.
3. Whether to report the two waLBerla upstream defects found here (the HIP version check and `TestGpuRAII`) on gitos.rrze.fau.de. Both are drafted below. Nothing has been posted.

## Licence

- waLBerla: GPL-3.0-or-later. `COPYING.txt` is the GPLv3 text, and source headers say "either version 3 of the License, or (at your option) any later version". Recorded with `set-license-spdx walberla GPL-3.0-or-later` (tier 3; needs a person's approval before anything is offered upstream). There is no CLA: CONTRIBUTING.md says "By contributing to waLBerla, you automatically agree to the terms of this license."
- Vendored code: `extern/lodepng` (zlib-style LodePNG licence) and `extern/sqlite3` (public domain). There are no git submodules (`.gitmodules` is empty).
- `python3 utils/licenses.py scan-nvidia agent_space/walberla-screen`: no NVIDIA proprietary licence text found.
- Code generators: pystencils and lbmpy are GPL-3.0-or-later from 1.4 on (AGPL-3.0 up to 1.3.7; "Change license to GPLv3", 2025-10-17). pystencils-sfg is GPL-3.0-or-later in every version. ESPResSo's generated kernels carry the waLBerla GPLv3 header.

## Home and destination (question 4)

- The canonical repository moved in Oct 2026 (CHANGELOG 7.4) from i10git.cs.fau.de, which still returns 502 or times out, to https://gitos.rrze.fau.de/cs10/walberla/walberla. https://github.com/lssfau/walberla is a read-only mirror. Its description says "Please open pull requests on GitLab", issues are disabled, and it was last pushed 2026-11-13. `status.json.upstream_url` points at the GitHub mirror only because scaffold derives it from the candidate. Contributions go to gitos.
- Process (CONTRIBUTING.md): merge requests on gitos only. Academics sign in through DFN-AAI/eduGAIN. Non-academics are "onboarded via our waLBerla mirror on GitHub", so an AMD account needs a manual onboarding step. There is no CLA. The repository has contribution and AI guidelines (!815, !822, !845).
- A GitHub PR would not be looked at. The mirror has had 3 PRs ever, all closed unmerged. On #1 (2017) the maintainer asked for a resubmission on GitLab, and #2 (2020) and #3 (2025) were closed without comment.
- Activity is healthy: 95 master commits in 2026, spread over every month; 1 to 9 merged MRs per month; 18 open MRs; releases v7.3 (2026-07-21) and v7.4 (2026-10-27). pystencils and lbmpy had commits on 2026-11-17/18.
- MOAT tooling has no GitLab destination path (`upstream.py` publishes GitHub PRs). Any upstream change here would be a hand-carried gitos MR, which jeff would approve word for word.

## Q1. Existing AMD support

- **waLBerla: in-tree HIP, merged and released.**
  - Added by !508 "Support HIP for AMD GPUs" (opened 2022-01-06, merged 2023-05-17). The CHANGELOG lists the HIP bullets under 6.1, but the v6.1 tree has no HIP option; v7.1 is the first release with it.
  - Follow-ups: !681, !729 "Generate `.hip` files instead of `.cpp` files" (2025-04), !734, !793 "Add HIP instructions", !816, and !862 "Require minimum versions for MPI, OpenMP, CUDA and HIP" (2026-11-02).
  - Mechanism: `src/gpu/GPUWrapper.h` and `DeviceWrapper.h` alias `gpuMalloc`/`gpuStream_t`/... to CUDA or HIP. Generated code is written with the `.hip` suffix (`CODEGEN_FILE_SUFFIX`). libhipcxx is fetched for the experimental v8 core.
  - Hardware evidence: LUMI-G MI250X weak scaling to 4096 GPUs (SC23 poster https://sc23.supercomputing.org/proceedings/tech_poster/poster_files/rpost132s3-file3.pdf; arXiv 2408.06880).
- **Not in CI.**
  - `.gitlab-ci.yml` runs only CUDA and CPU jobs. The `.hip` fragment in `utilities/workflows/cmake-fragments.json` is unused.
  - !862 says HIP is "not covered by CI yet (no HIP image, cs10/ci/images#9)".
  - Open issue #347 "Add a HIP CI job to the nightly pipeline" (2026-10-27) plans to register "our AMD GPU machine" as a nightly runner.
- **Known HIP gaps upstream.**
  - Open #314 "ShiftedPeriodicity fails for AMD GPUs" (2026-03-17). In my HIP build `src/gpu/ShiftedPeriodicity.cu` is silently dropped: it appears 0 times in `build.ninja`, because a `.cu` source has no enabled language under HIP. Its header includes the CUDA-only `<device_launch_parameters.h>`, and it calls `fa.set(blockIdx, threadIdx)`, which HIP's builtin index types do not convert to `uint3`. ESPResSo does not use ShiftedPeriodicity (grep).
  - Open #250 "Preprocessor-aware code generation", where an ESPResSo developer says it "will help us add support for AMD GPUs".
  - The open MR !852 moves CUDA/HIP setup into the core CMake.
- **Generators.** All live under https://gitos.rrze.fau.de/cs10/pycodegen/.
  - pystencils 2.x has `Target.HIP` and `backend/platforms/hip.py` (!458, merged 2025-03-20), plus hipCUB reductions (!527, 2026-06).
  - pystencils 1.x has no HIP enum: `Target.GPU` with `Backend.CUDA` emits CUDA-dialect C, which runs on AMD through hipcc or hiprtc (!333, !335, 2023).
  - lbmpy follows pystencils. pystencils-sfg emits `.hip` and HIP kernel launches (!13, !24).
  - waLBerla's legacy generators (`python/pystencils_walberla`, `lbmpy_walberla`) use pystencils 1.x `Target.GPU`, and the output compiles through the GPUWrapper aliases. SweepGen uses pystencils 2.1 / sfg 0.3 / lbmpy 2.1.1. No pycodegen CI job mentions HIP.
- **ESPResSo.**
  - Had ROCm support from 2018 to 2020 and removed it in espressomd/espresso#3966 (2020-10-22). Reasons given: unstable ROCm, flaky CI, no known AMD users.
  - No issue or PR since then about HIP for the waLBerla LB/EK. #4766 (open) lists "AMD-equipped runners"; #5307 mentions LUMI CrayClang+ROCm.
- **Forks.** AMD-Ecosystem/walberla and ROCm/walberla do not exist (404). `gh search repos "walberla hip"` returns nothing. The only AMD effort is MOAT's own `espresso` port.

### Probe A: upstream master with HIP on this host (2026-11-20)

Host: gfx942 (MI300), pip ROCm SDK 10.1.0 (HIP 7.16.26385, AMD clang 24.0.0), GCC 13.3.0, CMake 3.31.6, OpenMPI 4.1.6. waLBerla master e2383d29 (2026-11-02).

```
cmake -S walberla -B wb-hip -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -DCMAKE_HIP_COMPILER=$ROCM_SDK/llvm/bin/clang++ -DCMAKE_PREFIX_PATH=$ROCM_SDK \
  -DCMAKE_HIP_ARCHITECTURES=gfx942 -DWALBERLA_BUILD_WITH_HIP=ON -DWALBERLA_BUILD_WITH_MPI=ON \
  -DWALBERLA_BUILD_WITH_OPENMP=OFF -DWALBERLA_BUILD_TESTS=ON -DWALBERLA_BUILD_WITH_CODEGEN=OFF \
  -DWALBERLA_BUILD_WITH_PYTHON=OFF -DWALBERLA_BUILD_BENCHMARKS=OFF -DWALBERLA_BUILD_TUTORIALS=OFF \
  -DWALBERLA_BUILD_SHOWCASES=OFF -DWALBERLA_BUILD_EXAMPLES=OFF
ninja -C wb-hip -k 0 -j 56
HIP_VISIBLE_DEVICES=6 ctest --test-dir wb-hip/tests/gpu --timeout 300 -j 4
```

- **Configure fails on any ROCm 7+.** `find_package (hip ${WALBERLA_HIP_MIN_VERSION} REQUIRED)` (CMakeLists.txt:738, added by !862 / 25bcdd25 on 2026-11-02) asks for 6.1. hip's `hip-config-version.cmake` is SameMajorVersion ("only if the requested major version is the same as the current one"), so HIP 7.16 is rejected: "Could not find a configuration file for package "hip" that is compatible ... version: 7.16.26385". A version range fails the same way, because the file requires both endpoints in the same major. Fix: `find_package(hip REQUIRED)` followed by `if(hip_VERSION VERSION_LESS ${WALBERLA_HIP_MIN_VERSION}) message(FATAL_ERROR ...)`. ESPResSo's pin 623a60c1 predates this and uses a plain `find_package(hip REQUIRED)`.
- **Build:** with that check patched locally, 663/663 Ninja steps, 0 FAILED, 0 warnings. Only 3 TUs are HIP-compiled without codegen; the gpu-module `.cpp` files go through the host compiler with `hip::host`. Codegen-driven GPU apps and tests were not built, because that needs Python 3.12 + pystencils/lbmpy.
- **GPU tests** (GPU 6, idle): 7/8 passed (AlignmentTest, FieldTransferTest, GPUBlockSelectorCommunicationTest, GPUPackInfoCommunicationTest, GPUPackInfoTest, TestGenericGpuPackInfos_1/2/4). `TestGpuRAII` failed at `tests/gpu/TestGpuRAII.hpp:62`: `WALBERLA_CHECK(gpuStreamSynchronize(underlyingStream) != gpuSuccess)` under `#if defined(WALBERLA_BUILD_WITH_HIP)`. The test expects an error when synchronizing an already-destroyed stream. That is undefined behaviour, and HIP 7.16 returns success. This is a test defect, not a GPU fault.

## Q2. Where the CUDA is

- **Hand-written** (`src/gpu`): 7 `.cpp`, 1 `.cu` (ShiftedPeriodicity), and 47 headers in the top directory, plus `communication/`, `lbm/`, `sweeps/`, `timeloop/` and `timing/`. About 7.9k lines in the top directory. It is written against the `gpu*` aliases, not raw CUDA; the only `__global__` kernels by hand are in `ShiftedPeriodicity` and the PSM codegen wrappers in `lbm_mesapd_coupling`. The whole tree has only 11 `.cu` files (4 in tests/v8, 3 in tests/gpu, 2 in lbm_mesapd_coupling, 1 in src/gpu, 1 tutorial).
- **Generated:** every LB/EK GPU sweep, pack info and boundary kernel. They are emitted at build time by waLBerla's legacy generators (`pystencils_walberla`/`lbmpy_walberla`, pystencils/lbmpy 1.4) or by SweepGen (pystencils 2.1 + pystencils-sfg). Both already emit HIP: a `.hip` suffix with CUDA-dialect bodies over the gpu aliases (1.x), or a native HIP target (2.x).
- **Projects a port would change.** For waLBerla itself, only waLBerla, and only the small fixes above; no generator change is required. For ESPResSo's LB/EK GPU, mainly ESPResSo (question 3), plus optionally a one-line fix to pystencils 1.x's `philox_rand.h`. pystencils, lbmpy and pystencils-sfg are not MOAT candidates of their own: 2.x already targets HIP, and the 1.x fix is a one-liner that ESPResSo can carry in its vendored copy.

## Q3. The ESPResSo side

### 3a. How ESPResSo's generated kernels are made, and what HIP needs

- 62 `.cu` and 7 `.cuh` files in `src/walberla_bridge`. 56 of the `.cu` files are generated kernels (LB, EK and reactions, single and double precision).
- **How they are made.** `maintainer/walberla_kernels/generate_lb_kernels.py` and `generate_ek_kernels.py`, with `--gpu` (= `ps.Target.GPU`), using ESPResSo's own jinja templates (`templates/FieldAccessors.tmpl.cu`, `EK_FieldAccessors.tmpl.cu`, `preprocessor.tmpl.cuh`, ...) and post-generation patching (`patch_packinfo_kernel`, ...). The Readme pins lbmpy@e9efe343 and pystencils from a contributor fork, `cs10/contributors/jean-noel-grad/pystencils@e851f4e3`. `requirements.txt` pins `pystencils==1.4.0`, `lbmpy==1.4.0`, `sympy==1.12.1`. The file headers read "pystencils v1.4+1.ge851f4e, lbmpy v1.4+1.ge9efe34, sympy v1.12.1, lbmpy_walberla/pystencils_walberla from waLBerla commit 007e77e0".
- **The output is already nearly HIP-neutral.** Identifier tally across the generated files: `blockIdx`/`threadIdx`/`blockDim` about 2.9k uses, `gpuStream_t` 136, `__launch_bounds__` 124. There are no `cuda*` runtime calls, warp intrinsics, `warpSize` or textures. The thrust includes resolve to rocThrust. The `__CUDA_ARCH__` blocks only switch warning pragmas.
- **Probe:** `clang++ -x hip --offload-arch=gfx942 -std=c++20 -fsyntax-only -D__HIP_PLATFORM_AMD__ -DWALBERLA_BUILD_WITH_HIP`, against waLBerla 623a60c1 configured with HIP (`waLBerlaDefinitions.h`). **42/56 pass unchanged.** The 14 failures have exactly three causes:
  1. `RESTRICT` is undefined under clang-HIP. ESPResSo's `preprocessor.tmpl.cuh` prelude defines it only for `__NVCC__` or `__clang__ && __CUDA__`. Affects PackInfoPdf/Vec, StreamCollideSweepThermalized and FieldAccessors.
  2. `philox_rand.h` (vendored from pystencils): `QUALIFIERS` adds `__device__` only for `__CUDA_ARCH__ || __HIP_DEVICE_COMPILE__ || (__clang__ && __CUDA__)`, so in the HIP host pass the philox functions are host-only and `__global__` callers fail. Released pystencils 1.4's copy has the same condition without the clang clause. Adding `|| defined(__HIP__)` fixes it. Affects the thermalized LB and EK kernels.
  3. ESPResSo's FieldAccessors templates pass HIP's builtin `blockIdx`/`threadIdx` where waLBerla's `gpu::FieldAccessor::set(uint3, uint3)` and a local `getLinearIndex(uint3, ...)` expect `uint3`, and return `make_uint3` as `dim3`. Constructing `uint3{blockIdx.x, ...}` explicitly fixes it; waLBerla's own `lbm_mesapd_coupling` code uses that idiom (`blockIdx_uint3`).

  With those three fixes applied in scratch copies, **56/56 pass** (syntax only, host and device passes; no codegen, link or run). So the answer is "regenerate or hand-edit the templates and one header", not a compatibility header over CUDA. The fixes belong in ESPResSo's templates and its vendored philox header. Regenerating needs the pinned contributor-fork pystencils, and hand-editing the 14 files plus the templates gives the same result.
- **The rest of a full HIP LB/EK** (hand-written bridge, not probed to completion):
  - `__CUDACC__` guards select the GPU field types and paths (51 sites in 13 files, e.g. `LBWalberlaImpl.hpp`, `PoissonSolverFFT.hpp`, `BoundaryHandling.hpp`). clang-HIP does not define `__CUDACC__` (it defines `__HIPCC__`/`__HIP__`; checked with `-dM -E`), so each needs `__HIPCC__` or the port's `cuda_to_hip.h` mapping.
  - cuFFT types (`greens_function.hpp`, `PoissonSolverFFT.hpp`) need hipFFT, and `heffte::backend::cufft` needs heFFTe's rocfft backend.
  - 3 of the 4 bridge `.cu` files stop on `cuda_runtime.h`.
  - CMake must turn on `WALBERLA_BUILD_WITH_HIP`, set the bridge and generated `.cu` files to LANGUAGE HIP, and link `hip::host`/hipFFT instead of `CUDA::cudart`/`CUDA::cuda_driver`.
  - Runtime correctness on wave64 and wave32 is untested. Risk looks low (plain index arithmetic, no warp-level code), but the LB/EK statistical and flow tests would be the gate.

### 3b. HIP core GPU with waLBerla's CPU solvers (decoupling)

What couples them today, in the espresso port at bc8d73dd9:
- The HIP branch at `CMakeLists.txt:279-284` stops at configure with "ESPRESSO_BUILD_WITH_HIP requires ESPRESSO_BUILD_WITH_WALBERLA=OFF".
- The FetchContent block sets `WALBERLA_BUILD_WITH_CUDA on` whenever `ESPRESSO_BUILD_WITH_CUDA` is on (`CMakeLists.txt:1143`).
- `src/walberla_bridge/CMakeLists.txt` links `OpenMP::OpenMP_CUDA` on `ESPRESSO_BUILD_WITH_CUDA`. That target does not exist in a HIP build.
- The LB/EK GPU init sources (`lb_walberla_init.cu`, `ek_walberla_init.cu`, `ek_poisson_none_init_cuda.cu`, `ek_poisson_fft_init_cuda.cu`) are added on `ESPRESSO_BUILD_WITH_CUDA` and would be dropped or compiled as CUDA.

The bridge already has the fallback this needs. Each `new_*_gpu` factory is `#if not defined(WALBERLA_BUILD_WITH_CUDA) throw std::runtime_error("waLBerla was compiled without CUDA support")`. Compiling those files as C++ when waLBerla has no GPU backend keeps the symbols the script interface references under `ESPRESSO_CUDA`.

Probe (scratch branch in my own clone; push URL disabled; the fork is untouched). The diff is 5 files, +12/-10:
- delete the FATAL_ERROR block;
- `if(ESPRESSO_BUILD_WITH_CUDA AND NOT ESPRESSO_BUILD_WITH_HIP)` around `set(WALBERLA_BUILD_WITH_CUDA on)`;
- `OpenMP::OpenMP_CUDA` genex keyed on `WALBERLA_BUILD_WITH_CUDA` (2 places in `src/walberla_bridge/CMakeLists.txt`, 1 in its `tests/CMakeLists.txt`);
- `if(ESPRESSO_BUILD_WITH_CUDA AND NOT WALBERLA_BUILD_WITH_CUDA) set_source_files_properties(<the four .cu> TARGET_DIRECTORY espresso_walberla PROPERTIES LANGUAGE CXX)` in the LB and EK CMakeLists.

```
cmake -S espresso -B es-dec -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -DESPRESSO_BUILD_WITH_CUDA=ON -DESPRESSO_BUILD_WITH_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx942 \
  -DCMAKE_HIP_COMPILER=$ROCM_SDK/llvm/bin/clang++ -DCMAKE_PREFIX_PATH=$ROCM_SDK \
  -DPython_EXECUTABLE=/usr/bin/python3.12 -DESPRESSO_BUILD_WITH_WALBERLA=ON \
  -DESPRESSO_BUILD_WITH_FFTW=ON -DESPRESSO_BUILD_WITH_PYTHON=OFF -DESPRESSO_BUILD_TESTS=ON \
  -DESPRESSO_BUILD_WITH_HDF5=OFF -DESPRESSO_BUILD_WITH_SCAFACOS=OFF \
  -DESPRESSO_BUILD_WITH_STOKESIAN_DYNAMICS=OFF -DESPRESSO_BUILD_WITH_GSL=OFF \
  -DESPRESSO_BUILD_WITH_NLOPT=OFF -DESPRESSO_BUILD_WITH_CALIPER=OFF
ninja -C es-dec -k 0 -j 56
HIP_VISIBLE_DEVICES=6 cmake --build es-dec -j 48 --target check_unit_tests
```

- Configure: OK. Cache has `WALBERLA_BUILD_WITH_CUDA=OFF` and `WALBERLA_BUILD_WITH_HIP=OFF`.
- Build: **377/377, 0 FAILED**. 16 HIP TUs in `src/core`, and the four bridge init files built with the C++ compiler. The 21 warnings are 20 of the known ROCm `nodiscard hipError_t` `-Wunused-value` and 1 `-Wunused-macros`; none fails the build, as the espresso notes already say.
- Unit tests: **131/131 passed**, including lb_particle_coupling_test, ek_interface_test, lb/ek_kernels_unit_tests, LatticeWalberla, LBWalberlaImpl (unit, bspline, flow, lees_edwards, statistical, field_accessors) and EKinWalberlaImpl.
- **What is still open** for that round (not done here):
  - With Python on, about 37 LB/EK Python test files mark their GPU classes `@utx.skipIfMissingGPU()` only. `gpu_available()` would be true, so those cases would hit the "compiled without CUDA support" runtime error instead of skipping. The round needs a way to report "waLBerla has no GPU backend" (for example a `code_info` entry) and a skip keyed on it, or the GPU LB/EK request should fail feature checks.
  - The user-guide sentence would change from "waLBerla must be disabled" to "the LB/EK solvers are CPU-only on AMD GPUs".
  - The CUDA path is unchanged by construction (every change is behind HIP, or is keyed on `WALBERLA_BUILD_WITH_CUDA`, which equals `ESPRESSO_BUILD_WITH_CUDA` for a FetchContent nvcc build). A preinstalled waLBerla (`find_package(waLBerla 7.2)`) built without CUDA would now take the CPU path instead of failing, which is arguably better. The nvcc no-regression compile is still owed.

## Q5. Size class

- **waLBerla as a MOAT port: small, and not a port.** One CMake line (the HIP version check), the `TestGpuRAII` UB assertion, and optionally ShiftedPeriodicity (#314, which upstream already tracks). CI is upstream's own plan (#347). There is nothing for a planner to study.
- **ESPResSo LB/EK GPU on HIP: medium, in the espresso project.**
  - Already known: three template/header fixes for the 56 generated kernels.
  - Not yet attempted: guard and cuFFT/heFFTe work in about 13 hand-written bridge headers, CMake language and linkage, the Python test-suite skips, and full LB/EK GPU validation on wave64, wave32 and Windows.
  - It is not "a generator backend": both generator generations already emit HIP-clean bodies.
- **ESPResSo HIP plus CPU LB/EK (3b): small,** as an espresso round. The CMake part is proven above; the Python test skips remain.

## Draft upstream reports (not posted; jeff approves every word)

These were not filed, and nobody was contacted. If jeff wants them, they go to https://gitos.rrze.fau.de/cs10/walberla/walberla as issues or MRs.

1. CMake: `find_package(hip 6.1 REQUIRED)` rejects ROCm 7.x. hip's config version file is SameMajorVersion, so any HIP 7.x fails at configure since !862. Proposed fix: find without a version, then compare `hip_VERSION`.
2. tests/gpu/TestGpuRAII: the HIP-only check that `gpuStreamSynchronize` on a destroyed stream returns an error tests undefined behaviour; HIP 7.16 returns `hipSuccess`. Suggest removing the check.

## Telemetry and scratch

The scratch clones, build trees and logs are under `agent_space/` (container-local): `walberla-screen` (GitHub mirror clone), `walberla-623a` (worktree at ESPResSo's pin), `espresso-ro` (read-only fork clone, branch `scratch-decouple`, push URL set to an invalid value), and `logs/`.
