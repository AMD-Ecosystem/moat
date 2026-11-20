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

## Wave32 check (linux-gfx1201)

Measurement job requested by jeff on 2026-11-20 (not a pipeline role; no fork, no state change). Question: does waLBerla's HIP backend need anything from us for 32-wide Radeon GPUs?

**Answer: no 32-wide-specific fault was found.** Every GPU result on gfx1201 either matches the MI300X intake, agrees with the CPU at round-off, or passes upstream's own analytic checks. What a HIP user does hit are HIP-build gaps that are not wave-size related. Most cause no warning: GPU tests and sources that CMake drops or never builds under HIP. Candidate upstream fixes are ranked at the end of this section. All of them are arch-independent, so the MI300X re-run listed below is for completeness, not suspicion.

### Host and versions

- GPU: AMD Radeon AI PRO R9700 (gfx1201, wave32), pinned with `HIP_VISIBLE_DEVICES=1`; every GPU run was under `ulimit -c 0`.
- ROCm: pip ROCm SDK 10.1.0 (HIP 7.16.26385, AMD clang 24.0.0), the same SDK as the intake. GCC 13.3.0, CMake 3.31.6.
- MPI: OpenMPI 5.0.11 from the `openmpi` pip wheel in the job venv (the intake used system OpenMPI 4.1.6; this host has no system MPI).
- waLBerla: master **e2383d29** (2026-11-02), the intake's commit. **Master has not moved**: `git ls-remote` on gitos.rrze.fau.de and the GitHub mirror both return e2383d29.
- Legacy codegen: `lbmpy==1.4.1` (pulls pystencils 1.4, sympy 1.12.1), `jinja2 lxml setuptools py-cpuinfo islpy`. These are the pins in upstream's CI (`utilities/workflows/ci-common.yaml`, `.testsuite-common`). They went into the job venv, which is Python 3.11. waLBerla requires 3.12 (`WALBERLA_PYTHON_MIN_VERSION`), and the fleet guard allows pip only into the job venv, so the codegen build carries a local one-line change of that minimum to 3.11 (environment only).
- SweepGen/v8: waLBerla's own managed venv (`WALBERLA_SWEEPGEN_MANAGED_VENV`, created by CMake under the build tree from `/usr/bin/python3.12`): pystencils 2.1, pystencilssfg 0.3, lbmpy 2.1.1, sympy 1.14.0.

Local changes, all on a scratch clone, never pushed:
1. The intake's HIP version-check fix (`find_package(hip REQUIRED)` + `if(hip_VERSION VERSION_LESS ...) FATAL_ERROR`), in every tree.
2. Legacy-codegen tree only: Python minimum 3.11 (above). The UniformGridCPU/GPU velocity VTK writers changed from float32 to float64, and UniformGridGPU's VTK writer moved from after the time step to before it, as UniformGridCPU does (output only; see 3c).
3. Prototype tree only: the candidate fixes listed at the end.

### 1. Probe A build (same options as the intake, `CMAKE_HIP_ARCHITECTURES=gfx1201`)

```
cmake -S walberla -B wb-hip -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -DCMAKE_HIP_COMPILER=$ROCM_SDK/llvm/bin/clang++ -DCMAKE_PREFIX_PATH=$ROCM_SDK \
  -DCMAKE_HIP_ARCHITECTURES=gfx1201 -DWALBERLA_BUILD_WITH_HIP=ON -DWALBERLA_BUILD_WITH_MPI=ON \
  -DWALBERLA_BUILD_WITH_OPENMP=OFF -DWALBERLA_BUILD_TESTS=ON -DWALBERLA_BUILD_WITH_CODEGEN=OFF \
  -DWALBERLA_BUILD_WITH_PYTHON=OFF -DWALBERLA_BUILD_BENCHMARKS=OFF -DWALBERLA_BUILD_TUTORIALS=OFF \
  -DWALBERLA_BUILD_SHOWCASES=OFF -DWALBERLA_BUILD_EXAMPLES=OFF
ninja -C wb-hip -k 0 -j 12
```

- Unpatched e2383d29: configure fails exactly as on MI300X ("Could not find a configuration file for package "hip" that is compatible with requested version "6.1"").
- Patched: **663/663, 0 FAILED, 0 warnings** (2 min, `-j 12`). Same as the intake.

### 2. gpu-module tests: 7/8, same single failure

`ctest --test-dir wb-hip/tests/gpu --timeout 300 -j 4`: 7/8 passed. GPUPackInfoTest, GPUPackInfoCommunicationTest, GPUBlockSelectorCommunicationTest, TestGenericGpuPackInfos_1/2/4 and FieldTransferTest pass. **TestGpuRAII fails at `tests/gpu/TestGpuRAII.hpp:62`**, the same line and expression as on MI300X (`hipStreamSynchronize(underlyingStream) != hipSuccess` on a destroyed stream; UB, HIP returns success). Line 79 has the same check and would fail next.

Correction to the intake's list: `AlignmentTest` is built but **not registered** (`tests/gpu/CMakeLists.txt` has no `waLBerla_execute_test` for it), so it was not among the 8. The intake's 8 names include it and omit TestGpuRAII. Run by hand here, `./AlignmentTest` exits 0.

### 3. Lattice-Boltzmann GPU paths beyond the unit tests

Three more trees were built: legacy codegen (`-DWALBERLA_BUILD_WITH_CODEGEN=ON`, then `-DWALBERLA_BUILD_WITH_PYTHON=ON`, benchmarks, tutorials and showcases on), v8 + SweepGen (`-DWALBERLA_ENABLE_V8CORE=ON -DWALBERLA_ENABLE_SWEEPGEN=ON -DWALBERLA_BUILD_EXAMPLES=ON -DWALBERLA_BUILD_LEGACY_LBM=OFF`), and the prototype.

**3a. Legacy-codegen build** (the generator generation ESPResSo uses): 1777 steps, **0 warnings**, 3 failures, all HIP-build gaps (items B, C and D below):
- `apps/tutorials/gpu/01_GameOfLife_cuda`: link error, `undefined reference to walberla::GameOfLifeSweepCUDA::operator()`. Its `01_GameOfLife_kernels.cu` has no enabled language under HIP, so CMake drops it without a warning.
- `tests/gpu/TestShiftedPeriodicityGPU.cpp.o`: `ShiftedPeriodicity.h:44: fatal error: device_launch_parameters.h: No such file or directory` (CUDA-only include). It builds only with Python on, which the intake's build did not have.
- `apps/benchmarks/PhaseFieldAllenCahn` codegen: "Files only specified in CMake [...hip]" / "Unexpected generated files [...cpp]". `multiphase_codegen.py` branches on `ctx.cuda` (lines 152 and 180) instead of `ctx.gpu`, so under HIP it emits CPU kernels while CMake expects `.hip`. The other `ctx.cuda` hits in the tree are harmless: the `src/lbm_generated/*_generation_script.py` maintainer scripts hard-code `cuda=False`, and `tests/lbm_generated/*.py` have it commented out.

GPU tests in that tree (`ctest -R 'gpu|Gpu|GPU'`): GeneratedFieldPackInfoTestGPU and GeneratedFieldNonUniformPackInfoTestGPU pass. They first failed to load: `libamdhip64.so.7: cannot open shared object file`. Executables that the HIP linker links get `libamdhip64` from the compiler's implicit link directories, which CMake leaves out of RUNPATH, and the pip SDK's `lib/` is not on the loader path. Classified **environment** (a system `/opt/rocm` with an `ld.so.conf` entry would not see it). Every run from then on used `LD_LIBRARY_PATH=$ROCM_SDK/lib`.

**3b. More than 8 MPI ranks on one gfx1201 hang in SDMA copies: environment/ROCm, not waLBerla.** `LBM_MESAPD_COUPLING_PSM_ParticleAndVolumeFractionMapping_CPU_GPU` (27 ranks, 48^3 cells, 100 steps) timed out at 600 s with no output. TestGenericGpuPackInfos passes at 1, 4 and 8 ranks and hangs at 9, 10, 11, 12, 16 and 27. Stacks: 8 ranks spin in `rocr::core::BusyWaitSignal::WaitRelaxed` under `hipStreamSynchronize`, after `hipMemcpy3DAsync` and a small D2H `hipMemcpyAsync` on each of 4 streams (`AMD_LOG_LEVEL=3`); the other ranks wait in MPI for them. **`HSA_ENABLE_SDMA=0` fixes it**: 12 ranks pass in 5 s, 27 ranks of TestGenericGpuPackInfos in 12 s, and the 27-rank PSM mapping test in 2 s. Things that did not fix it: `GPU_MAX_HW_QUEUES=1`, `--bind-to none`, excluding UCX/OFI. Plain HIP programs did not reproduce it: 27 concurrent processes doing H2D/kernel/D2H, 12 processes with high-priority streams, and 4-stream `hipMemcpy3DAsync` D2D + D2H with and without `MPI_Init` all completed. It needs more than 8 GPU processes on one Radeon, which no real deployment of a single-GPU host does, and the trigger is in ROCm's SDMA path. A ROCm issue would need a standalone reproducer first (not attempted further: stop discipline). **To run on MI300X next:** the 27-rank PSM mapping test with SDMA enabled, to see whether the hang is Radeon-specific.

**3c. CPU vs GPU, legacy generated kernels, lid-driven cavity.** Upstream's `apps/benchmarks/UniformGridCPU` and `UniformGridGPU` come from the same generator (`generate_lbm_package`), read the same `.prm`, and seed the same initial shear flow (`InitShearVelocity.h`, identical files, `seedRandomGenerator(0)`). Run here on one parameter file: 32^3 cells, one block, all six walls (`UBB` lid on N at 0.05, `NoSlip` elsewhere), omega 1.6, `timeStepStrategy noOverlap`, 2001 steps, `gpuBlockSize <64,1,1>`. Velocity written in float64 at steps 0, 1000 and 2000 and compared cell by cell over all 32^3 x 3 components (scratch script `cmpvtk.py`, reading the `.vti` binary arrays).

A first comparison disagreed by 3.6x at step 0 and 6.7e-4 at step 1000. The cause was the benchmarks themselves: UniformGridCPU writes VTK *before* each time step (`addFuncBeforeTimeStep`), UniformGridGPU *after* (`addFuncAfterTimeStep`), so they were one step apart. With the GPU hook aligned:

| config | max abs du at step 2000 | rel to max abs u (0.0452) | GPU / CPU MLUPS |
|---|---|---|---|
| d3q19_pull_srt | 5.6e-17 | 1.2e-15 | 426 / 49.5 |
| d3q19_pull_trt | 2.8e-17 | 6.1e-16 | 398 / 43.0 |
| d3q19_pull_mrt | 1.0e-16 | 2.2e-15 | 403 / 47.4 |
| d3q19_aa_srt (in-place) | 5.6e-17 | 1.2e-15 | 437 / 45.9 |
| d3q19_esotwist_trt (in-place) | 2.1e-17 | 4.6e-16 | 458 / 41.0 |
| d3q27_pull_srt | 4.2e-17 | 9.2e-16 | 381 / 31.7 |
| d3q27_pull_mrt | 9.0e-17 | 2.0e-15 | 402 / 30.7 |
| d3q27_pull_cumulant | 3.3e-8 | 7.4e-7 | 341 / 19.9 |

Step 1000 agrees to the same orders. Everything except cumulant agrees to double-precision round-off, including both in-place streaming patterns and the NoSlip/UBB GPU boundary kernels. Cumulant differs **by design**: `UniformGridGPU.py` applies `insert_fast_divisions`/`insert_fast_sqrts` (`optimize = True`), and the generated cumulant kernel has 4 single-precision `__fdividef`; the CPU script does not. MLUPS are from a 32^3 domain and are not a benchmark. Not run on MI300X by the intake: **to run there next** with the same `.prm`.

**3d. v8 + SweepGen build.** The first build failed in nearly every host TU: `include/hipccl/cuda/std/__cccl/assert.h:99: error: '__host__' does not name a type`, via `src/core/Stdlib.hpp:56` `#include <hip/std/cstddef>`. `cmake/walberla_v8core.cmake` declares libhipcxx with `GIT_TAG release/2.7.x` but also `FIND_PACKAGE_ARGS CONFIG`, so it accepts any installed libhipcxx. ROCm 10.1's SDK ships one (`lib/cmake/libhipcxx`, `CCCL_VERSION 3000002`) whose headers do not compile under plain g++. This is a **HIP-build gap, not wave-size related**; the MI300X host has the same SDK and would fail the same way. With `-DCMAKE_DISABLE_FIND_PACKAGE_libhipcxx=ON` (forcing the pinned fetch, libhipcxx 9ac637b66): **555/555, 0 warnings**. `ctest --test-dir wb-v8 --timeout 600 -j 6`: **290/291 passed**, the only failure TestGpuRAII. These include:
- `ParallelPlates - Couette` and `- Poiseuille` (`apps/examples/ParallelPlates`, compiled as HIP, generated with SweepGen, 32^3, 1000 steps, checked against the analytic profile): Lmax x-velocity error **1.68e-16** (threshold 1e-15) and **3.79e-8** (threshold 1e-6).
- `TestLbmScenarios_CXX` and `_HIP` (FullyPeriodic, MirroredHalfChannel, FreeSlipPipe, each at 1, 2 and 4 ranks): all 18 pass, so the CPU and GPU builds of the same generated scenarios pass the same built-in checks.
- `TestGPUSweeper` (3), `TestSweepGeneration` (7), and the v8 memory and halo-exchange tests.

### 4. GPU tests that a HIP build skips or cannot build (gap list)

| what | where | why under HIP | prototype result on gfx1201 |
|---|---|---|---|
| SimpleKernelTest, FieldIndexing3DTest | `tests/gpu/CMakeLists.txt` `if( WALBERLA_BUILD_WITH_CUDA)` ("won't compile with HIP") | `.cu` sources have no language; `FieldAccessor*::set(uint3, uint3)` rejects HIP's builtin `blockIdx`/`threadIdx`; `FieldAccessor3D.h` uses `dim3`/`uint3`/`__forceinline__` without including `gpu/GPUWrapper.h` (nvcc pre-includes `cuda_runtime.h`, clang-HIP does not) | both pass |
| CodegenJacobiGPU, CodegenPoissonGPU | same block | `OUT_FILES` hard-code `.cu` instead of `${CODEGEN_FILE_SUFFIX}` | both pass |
| CommTest, CudaMPI | same block | need GPU-aware MPI; not executed even for CUDA (commented out) | left CUDA-only |
| TestShiftedPeriodicityGPU, and `src/gpu/ShiftedPeriodicity.cu` (open upstream #314) | `src/gpu/CMakeLists.txt`, `ShiftedPeriodicity.h` | `.cu` dropped from `walberla_gpu` without a warning; CUDA-only `<device_launch_parameters.h>`; `fa.set(blockIdx, threadIdx)` and `getLinearIndex(blockIdx, threadIdx, gridDim, blockDim)` | passes |
| 28 PSM CPU_GPU tests (14 FuncTests: DragForceSphere, TorqueSphere x12, SettlingSphere; plus their 14 long runs), FluidizedBed and Percolation benchmarks, the PSM codegen library | `tests/lbm_mesapd_coupling/CMakeLists.txt:95`, `src/lbm_mesapd_coupling/partially_saturated_cells_method/codegen/CMakeLists.txt:4`, `apps/benchmarks/CMakeLists.txt:17` | gated on `WALBERLA_BUILD_WITH_DOUBLE_ACCURACY`, which nothing defines (the option is `WALBERLA_DOUBLE_ACCURACY`); introduced by 7a0c6023 (2026-06-10, "Reduce the workload on the CUDA Git Runner"). **Dead in every build, CUDA and CPU included**, not HIP-specific | with `-DWALBERLA_BUILD_WITH_DOUBLE_ACCURACY=ON`, all 14 FuncTests pass on gfx1201 |
| 01_GameOfLife_cuda tutorial | `apps/tutorials/gpu/CMakeLists.txt` | `.cu` dropped, link fails | builds and runs |
| PhaseFieldAllenCahn GPU benchmark | `apps/benchmarks/PhaseFieldAllenCahn/multiphase_codegen.py` | `ctx.cuda` instead of `ctx.gpu` | not fixed in prototype |
| PhaseFieldAllenCahn GPU showcase | `apps/showcases/PhaseFieldAllenCahn/CMakeLists.txt:3` | `if ( WALBERLA_BUILD_WITH_CUDA)` | not tried |
| AlignmentTest, MicroBenchmarkGpuLbm | `tests/gpu/CMakeLists.txt` | built, never registered (CUDA too) | AlignmentTest exits 0 by hand |

Prototype tree (scratch branch on top of the intake fix; `-DWALBERLA_BUILD_WITH_CODEGEN=ON -DWALBERLA_BUILD_WITH_PYTHON=ON -DWALBERLA_BUILD_TUTORIALS=ON -DWALBERLA_BUILD_WITH_DOUBLE_ACCURACY=ON`): **1434/1434 build steps, 0 warnings**. `ctest -R 'gpu|Gpu|GPU' -LE 'longrun|verylongrun'` with `HSA_ENABLE_SDMA=0`: **26/27 passed** (TestGpuRAII the only failure). SimpleKernelTest, FieldIndexing3DTest and FieldTransferTest: 3/3 passed. The prototype diff (+50/-4, 8 files):
- `src/gpu/FieldAccessor.h`, `FieldAccessorXYZ.h`, `FieldAccessor3D.h`: a template overload `set(const BlockIdx_T&, const ThreadIdx_T&)` forwarding to `set(make_uint3(...), make_uint3(...))`, and the same for `getLinearIndex`. With CUDA, `blockIdx` is a `uint3` and the existing non-template overload still wins, so CUDA is unchanged in principle; not compiled with nvcc here (owed before any upstream MR).
- `FieldAccessor3D.h`: `#include "gpu/GPUWrapper.h"`.
- `src/gpu/ShiftedPeriodicity.h`: `<device_launch_parameters.h>` under `#if defined(WALBERLA_BUILD_WITH_CUDA)`.
- `src/gpu/CMakeLists.txt`, `apps/tutorials/gpu/CMakeLists.txt`, `tests/gpu/CMakeLists.txt`: `set_source_files_properties(<.cu> PROPERTIES LANGUAGE HIP)` under HIP; the HIP-able tests are moved out of the CUDA-only block; `${CODEGEN_FILE_SUFFIX}` in the two codegen `OUT_FILES`.

The same `uint3` fault is cause 3 of ESPResSo's 14 kernel failures (Q3a above). A waLBerla-side overload would fix that cause for ESPResSo's templates too, if ESPResSo moved to a waLBerla that has it.

### 5. Classification of every failure seen

| failure | class | MI300X comparison |
|---|---|---|
| configure: `find_package(hip 6.1)` rejects HIP 7.16 | HIP-build gap (ours) | same on MI300X (intake) |
| TestGpuRAII `.hpp:62` | test bug (UB) | same line on MI300X |
| GameOfLife tutorial link, ShiftedPeriodicity / TestShiftedPeriodicityGPU, CUDA-only tests/gpu block | HIP-build gap (ours) | intake did not build these (Python and tutorials off); arch-independent build errors |
| PhaseFieldAllenCahn benchmark `ctx.cuda` | HIP-build gap (ours) | not built by intake; arch-independent |
| v8 core picks up ROCm 10.1's libhipcxx 3.x, g++ TUs fail | HIP-build gap (ours) | not built by intake; same SDK, so expected identical |
| PSM tests dead via `WALBERLA_BUILD_WITH_DOUBLE_ACCURACY` | build bug, all backends (upstream's own) | n/a |
| `libamdhip64.so.7` not found for HIP-linked executables | environment (pip SDK not on loader path) | n/a |
| >8 ranks on one GPU hang in SDMA copies | environment/ROCm (not waLBerla; `HSA_ENABLE_SDMA=0` fixes) | not run by intake; run there next |
| UniformGrid CPU/GPU 1-step offset | benchmark inconsistency (VTK hook before vs after the step), not a numerical fault | n/a |
| cumulant 7e-7 | by design (`__fdividef` fast division in the GPU script) | n/a |

No failure is 32-wide-specific. **To run on MI300X next** (tests the intake did not run, for completeness): the UniformGrid CPU/GPU cavity comparison (same `.prm`), the v8/SweepGen ctest (ParallelPlates, TestLbmScenarios_HIP), the prototype's GPU ctest, and the 27-rank PSM mapping test with SDMA on.

### 6. Conclusion

waLBerla needs nothing from us that is specific to 32-wide AMD GPUs. Its generated lattice-Boltzmann kernels (legacy and SweepGen) run on gfx1201 and agree with the CPU to round-off, or meet their analytic checks. What a HIP user meets are build-system and header gaps, the same on any AMD GPU. Candidate upstream fixes, most valuable first (1 and 2 are the intake's):

1. **CMake: `find_package(hip ${WALBERLA_HIP_MIN_VERSION})` rejects every ROCm 7.x** (intake draft 1). Configure fails, so nothing builds. One-line fix.
2. **Accept HIP's builtin index types in `gpu::FieldAccessor*` and compile `.cu` sources as HIP** (closes #314 and re-enables 5 GPU tests plus the GPU tutorial under HIP). Today ShiftedPeriodicity is dropped from the HIP build without a warning. Prototype above: +50/-4, all affected tests pass on gfx1201. Needs an nvcc compile before it goes upstream.
3. **v8 core: pin the libhipcxx found by `find_package`** (a version in `FIND_PACKAGE_ARGS`, or include `<hip/std/...>` only in HIP TUs). With ROCm 10.1's bundled libhipcxx 3.x, the whole v8/SweepGen HIP build fails; workaround `-DCMAKE_DISABLE_FIND_PACKAGE_libhipcxx=ON`. This is the path upstream is moving to (SweepGen, !852).
4. **`WALBERLA_BUILD_WITH_DOUBLE_ACCURACY` should be `WALBERLA_DOUBLE_ACCURACY`** in 3 CMake files. Since 2026-06-10 it disables the PSM GPU tests and two benchmarks on every backend. Not HIP-specific, and it may need a word with the author of 7a0c6023 in case the disabling was intended.
5. **`multiphase_codegen.py`: `ctx.cuda` should be `ctx.gpu`** (2 lines); the PhaseFieldAllenCahn GPU benchmark fails to generate under HIP. The showcase is CUDA-gated in CMake as well.
6. **TestGpuRAII: drop the two HIP-only destroyed-stream checks** (intake draft 2).
7. Small: register `AlignmentTest`; make UniformGridGPU write VTK before the step, as UniformGridCPU does.

All of this is upstream-hosted on gitos and arch-independent. A MOAT port is still not warranted. Whether any of 1 to 7 goes to gitos (account onboarding needed) is jeff's call; nothing was posted.

Scratch (container-local, under the job directory `ws/`): clone `walberla` (branch `wave32-check`), worktrees `walberla-cg` (`wave32-codegen`), `walberla-proto` (`wave32-proto`) and `walberla-pristine`; build trees `wb-hip`, `wb-cg`, `wb-v8`, `wb-proto`; `scripts/ldc.prm`, `scripts/cmpvtk.py`, `scripts/run_ug.sh`, the plain-HIP reproducers `scripts/multi*.hip`; logs in `logs/` (`proto.diff`, `ug-compare.log`, `ctest-*.log`).
