# spconv notes

Earlier screens (seven, 2026-06 to 2026-08, ending in the 2026-08-20 decline) are in
history: `git show 2cc3927f97db:projects/spconv/notes.md`.

## Decisions

- 2026-08-20, Jeff Daily: declined as `already-supported` (spconv-triton). Recorded in
  `data/dispositions.json`, since cleared.
- 2026-11-20, Jeff Daily: "Decision: adopt spconv". Disposition cleared with
  `triage.py unskip` (`data/decisions.jsonl`; commit "dispositions: clear the spconv
  decline").
- 2026-11-23, Jeff Daily: "port/spconv: Option A: delete the old branch." The retired
  `origin/port/spconv` (`683d9b93`, no project folder; its only unique content was the
  cleared skip entry) was deleted from the remote by jeff's session, and this record was
  scaffolded fresh from main without `--force`.
- 2026-11-23, Jeff Daily: "Decision: the existing AMD-side spconv work -- Option A". Proceed with this port independently of `ZJLi2013/spconv_rocm` and `ZJLi2013/cumm-rocm`; do not build on that rewrite, and do not publish under their package names (`cumm-rocm`, `spconv-rocm`): use names per ROCm major such as `cumm-rocm7` and `spconv-rocm7`. The planner does not wait on any contact with them; a heads-up, if any, is jeff's own.
- 2026-11-23, Jeff Daily: cumm licence cleared with a condition (recorded on `port/cumm`): the port must not modify the four GPL-headed nvrtc files; a hipRTC path goes in new files. ccimport declined as already supported (PyPI, no fork).
- 2026-11-23, Jeff Daily: scope ruling "Option B". The port is not complete until an AMD matrix-instruction kernel family (MFMA and WMMA, for fp16 and bf16) exists beside the plain kernels, as with Quest prefill. No pull request, review page or wheels before then; the planner plans both families.

## Intake -- 2026-11-20 to 2026-11-23, linux-gfx942 (fleet job, headless)

Recommendation: **FORK** (adoption already ruled by jeff). Upstream:
https://github.com/traveller59/spconv, default branch `master`, screened at
`263d6b474` (2024-12-15; still the head on 2026-11-23). Hard build dependencies
cumm (fork recommended) and ccimport (no fork recommended; consumed unmodified), both
screened the same day on their own `port/` branches.

### Why the record was late

On 2026-11-20 `scaffold traveller59/spconv` refused (`spconv is already claimed:
port/spconv exists on the remote`), `retire-branch spconv` refused (`no record for
spconv can be read`), and merging main into the old branch conflicted in
`data/dispositions.json` with the old skip entry on the branch side. The screen was kept
in cumm's notes until jeff's 2026-11-23 ruling above; it is moved here verbatim apart
from headings, tense and a
fuller pointer to the study.

### Licence -- Apache-2.0 (with one BSD-3-Clause file), tier 1

    python3 utils/licenses.py check traveller59/spconv             # Apache-2.0, tier 1
    python3 utils/licenses.py scan-nvidia agent_space/spconv-screen  # clean

At `263d6b474` (2024-12-15), 160 files, no submodules, `LICENSE` the only licence file;
headers are `Copyright 2019-2022 Yan Yan` + Apache-2.0. Recorded
`Apache-2.0 AND BSD-3-Clause` with evidence
https://github.com/traveller59/spconv/blob/263d6b474/LICENSE. Exceptions:

| path | licence | compiled/shipped? | port modifies? |
|---|---|---|---|
| `spconv/pytorch/identity.py` (10 lines) | "Copyright 2016-present, Facebook, Inc." with a pointer to "the LICENSE file in the root" -- copied from facebookresearch/SparseConvNet, whose LICENSE is BSD-3-Clause | shipped (Python) | no |
| `spconv/csrc/sparse/all.py:1159` | Stack Overflow popcount snippet (CC BY-SA by origin) | compiled | no |

Binary files: `docs/spconv2_algo.pdf`, `example/libspconv/benchmark-pc.jarr`,
`test/data/benchmark-pc.npz`, `test/data/test_spconv.pkl` (test data and a paper; no
separate terms; the `.pkl` is a pickle the tests load). Nothing needs a person's
clearance.

### Duplicate effort and upstream state

- Upstream: no PR ever mentions HIP/ROCm/AMD; branches `2.1.x`, `develop`,
  `feature/ampere`, `feature/bf16`, `master`, `v1.1`, `v1.2.1`; docs grep: no hits.
- Issue #780 (2026-06-09, "Request: Windows HIP/ROCm build path for spconv + cumm"),
  closed by its author, who said two days later it was closed by accident (it stays
  closed). The only answer is from a
  third party, the spconv-triton author (L-Reichardt, 2026-07-09), pointing at
  spconv-triton. **No maintainer reply.** #774 (Ascend NPU) likewise unanswered.
- Activity: last commit 2024-12-15. 9 PRs ever merged; the last outside one 2020-07-07
  (#177); the last of any kind 2022-12-27 (#547, the author's). 10 open PRs, the newest
  #762 (2025-09-01, Python 3.12/3.13). The latest author comment found on a spconv issue is
  2023-06-02 (#563); none is among the repository's 100 most recent issue comments. Dormant for outside contributions; the same standing rule as cumm applies
  (send a clean, complete port as a PR; fork and wheels are the deliverable).
- Existing AMD work:
  - `L-Reichardt/spconv-triton` (Apache-2.0, PyPI 1.0.0 2026-07-20, 6 stars, no commit
    since): a pure-Triton drop-in replacement verified on MI300X; no Windows (no Triton
    there), autotune/compile on first use. Was the 2026-08-20 decline reason; jeff
    reversed that on 2026-11-20.
  - `JeffreyXiang/FlexGEMM` (MIT, Triton sparse-conv backend) and its HIP downstreams.
  - **`ZJLi2013/spconv_rocm`, branch `rocm`** (`36172b6ac`, 28 commits ahead of
    upstream, last push 2026-06-18, 2 stars, Apache-2.0 fork of spconv; default branch `master` is an unmodified mirror).
    **This corrects the moat-porting skill, which says that repository is empty.** It
    replaces pccm/cumm codegen with about 1.5k lines of Python plus self-contained HIP
    kernels (own Murmur3 hash for index pairs, `at::mm` loop for GEMM, optional
    cumm-rocm FlyDSL implicit GEMM for inference), JIT-built with
    `torch.utils.cpp_extension.load()`; Point2Voxel is stubbed; "27 tests passing on
    MI308X (gfx942)". Not on PyPI.
  - `HYGON-AI/spconv-das` (Apache-2.0, active 2026-11-12): Hygon DCU (HCU/DTK)
    downstream of `ZJLi2013/spconv_rocm`, packaged as `spconv-rocm` built with
    `CXX=hipcc`, no cumm.
  - `jiaqiwang969/spconv-rocm`: README only, not a port.
  None is a port of spconv as it stands (all drop the codegen and the prebuilt
  implicit-GEMM kernels, and none targets Windows), and none is authoritative. They are
  a coordination question (ZJLi2013), not an `already-supported` case.

### Viability, dependencies, deliverable

- Viable: yes, per jeff's study (`~/.claude/plans/moat-cumm-hip-backend-study.md`, summarised in `projects/cumm/notes.md` on `port/cumm`, Viability). fp32 SubMConv3d and SparseConv3d (Native, MaskImplicitGemm, MaskSplitImplicitGemm, fwd/dgrad/wgrad) match the CPU on gfx942 after the cumm-side fixes; the fp16 wall is spconv's own descriptor selection. spconv-side work: thrust
  `thrust::cuda::par` -> `thrust::hip::par` (8 sites), the fp16 tensor-op descriptor
  gating in `spconv/algo.py`, arch lists, wheel scripts; 10 to 14 files.
- `depends_on: [cumm, ccimport]` (recorded at scaffold): spconv's `setup.py` requires
  `pccm>=0.4.16`, `ccimport>=0.4.4` and a `cumm-cuXXX` build; its CUDA sources are
  generated by cumm at build time.
- Releases: PyPI `spconv-cu124`/`cu126` 2.3.8, manylinux_2_28 x86_64 and win_amd64,
  built by its own GitHub Actions with cumm's build system. A ROCm deliverable is the
  fork's `moat-port` plus `spconv-rocm7`-style wheels built against the ported cumm.
- Recommendation: `set-intake spconv fork`, `--viable yes`; duplicate effort names
  spconv-triton, FlexGEMM and ZJLi2013/spconv_rocm. Adoption itself is jeff's ruling
  (below); the fork is an admin's act.

### Recorded

    python3 utils/moatlib.py scaffold traveller59/spconv --branch master --deps cumm ccimport
    python3 utils/moatlib.py set-license-spdx spconv "Apache-2.0 AND BSD-3-Clause" --evidence https://github.com/traveller59/spconv/blob/263d6b474/LICENSE
    python3 utils/moatlib.py set-intake spconv fork ...
    python3 utils/moatlib.py set-state spconv linux-gfx942 awaiting-fork --agent intake

The intake queue issue was not published from this job (a GitHub write outside its
limits). Open questions jeff is ruling on separately (recorded in cumm's and ccimport's
notes): cumm licence clearance, whether ccimport gets a fork, the ZJLi2013 effort, and
whether a SIMT-only port is complete.

## Planning -- 2026-11-26, linux-gfx90a (fleet job, headless)

Plan written to `plan.md`; surface accounting in `surface.json` (`surface.py check spconv` clean; the regex census sees one `.cu` file because all device code is generated from Python, so 33 components were added by hand and 4 scoped out with reasons). `set-ext-type spconv pccm`. Read-only analysis of upstream `263d6b474`, built on cumm's plan (`port/cumm` `1614d936`) and read against cumm `d971d02a0`, pccm `860790f`, ccimport `f5af43d`, cloned into the job's `agent_space`; nothing built, fork untouched.

Decisions in the plan:
- Same HIP build route as cumm, taken from cumm: ccimport unmodified, `.cu` mapped to logical `clang++` via `build_pybind(suffix_to_compiler=...)`, `clang++` bound to `amdclang++` via `CCIMPORT_COMPILER_LINKER_MAP`, HIP flags in cumm's `clang++` bucket; `setup.py` gets a `build_ext` on HIP because `PCCMExtension` cannot forward `suffix_to_compiler`.
- Option B in spconv: HIP descriptor lists for cumm's MFMA (CDNA) and WMMA (RDNA) families, f16 in / f32 accumulate, mirroring the Turing f16 coverage; NVIDIA tensor-op lists not generated on HIP; `ALL_NATIVE_PARAMS`/`ALL_IMPGEMM_PARAMS` become backend-selected and are used by `setup.py`, `build.py` and `gencode` (CUDA output unchanged).
- Selection by device family, not CC tuple, at the four filter sites (Python `SimpleGemm`/`SimpleConv`, C++ `GemmTunerSimple`/`ConvTunerSimple`, the C++ one being the default path). Today fp16 on gfx9 skips SIMT at "CC >= 7.0" and picks Turing descriptors: the study's fp16 wall.
- int8 kept, on cumm's SimtDP4A family: upstream's int8-inference descriptors are tensor-op only, and upstream's own `test/test_all_algo.py` runs only int8 and asserts descriptors exist, so leaving int8 without kernels would fail upstream's test and cut a feature.
- rocThrust has no `thrust::cuda::par` (checked in the ROCm 10.1 SDK headers on this host: `thrust/system/hip/detail/par.h` defines `thrust::hip::par`; `thrust/system/cuda` is the CUDA-only CCCL copy). Seven live sites branch to `thrust::hip::par`. This also settles cumm's open rocThrust question.
- libspconv: pccm's `gen_cmake` emits CUDA-language CMake only (`pccm/core/__init__.py:2386-2490`); on HIP `spconv.gencode` adds a CMake fragment setting `LANGUAGE HIP` on the generated `.cu` sources and cumm's HIP options, and the example CMake gets a HIP branch. spconv_cpp's planner should build on this route.
- cumm version: spconv pins `cumm<0.8.0`, but cumm's 0.8.0 bump was a CI gcc/ABI change ("change version to 0.8.0 to avoid break spconv", cumm `e6a95bd`), not an API change; the HIP release pins `cumm-rocm<major>` at the cumm port's version. CUDA pins unchanged.

Dependencies: `depends_on` stays `[cumm, ccimport]` (ccimport dispositioned already supported). No new hard dependency; pccm is a PyPI build requirement and ROCm PyTorch a runtime environment requirement, neither a MOAT project. ROCm components: amdhip64, hipRTC, rocThrust (via cumm), hipBLASLt (`SimpleExternalSpconvMatmul`, JIT build and libspconv only). Optional, not hard: Boost (`BOOST_ROOT`, `BoxOps` rotated-box ops), tqdm and pytest (tests), torchvision (mnist examples), TensorRT (`custom_fx2trt.py`, scoped out). The selector currently reports spconv dep-blocked on cumm (adopted, no arch started); that clears when cumm completes on a platform of the host's OS.

### Interface spconv needs from the cumm port

For the cumm porter, so the names are exposed once (spconv's porter consumes them; neither project's plan is changed by this list):
1. One helper returning the HIP `build_pybind` kwargs and environment (cumm plan's `get_hip_build_kwargs()`), and preferably the HIP `build_ext` class, importable from `setup.py` at build time.
2. A Python HIP-build constant beside `CUMM_CPU_ONLY_BUILD`, and the ROCm version used for `cumm-rocm<major>` naming.
3. Device family availability keyed by gcnArchName, callable from Python and from generated C++ (spconv's C++ tuners run inside `core_cc` and libspconv), plus a device-arch-name binding.
4. The AMD `GemmAlgo` values and exported tuned parameter sets for gemm shuffle (gather A/C, scatter) and conv (fwd, dgrad, wgrad; mask_sparse, increment_k_first, access_per_vector 0/1, dynamic mask, split-K).
5. SimtDP4A implicit-GEMM conv with the `int8_inference` epilogue working on HIP (cumm's own SimtDP4A conv lists are commented out upstream, so this combination is untested there).
6. hipRTC for `ConstantMemory` mode with `CummNVRTCModule`, and `CompileInfo.arch_is_compiled_gemm`/`arch_is_compatible` semantics on HIP (spconv calls them with the CC tuple).
7. The HIP compile options as a list a CMake consumer can read (for spconv's gencode fragment).

### Questions for jeff (none blocks the porter; none is a scope cut)

1. int8 on HIP. Planned: cumm's SimtDP4A kernels (correct, slower), which keeps upstream's int8 quantized inference and its test working. Alternative: extend cumm's AMD families to int8 (MFMA i8, WMMA iu8; cumm question 1) and use AMD int8 descriptors instead.
2. bf16. Upstream spconv has no bf16 GEMM or conv descriptors on any backend (bf16 only in gather/pooling/inference). Planned: upstream parity, no HIP-only bf16 descriptors. Your Option B ruling names fp16 and bf16 for the matrix families; say if spconv should gain bf16 conv on HIP (a feature CUDA users do not have).
3. Wheel name, venue and gfx list for `spconv-rocm7`, after the port is complete (same decision as cumm's).
