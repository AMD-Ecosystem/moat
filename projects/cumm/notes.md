# cumm notes

## Intake -- 2026-11-20, linux-gfx942 (fleet job, headless)

Recommendation: **FORK.** cumm is where most of a spconv HIP port lands (the code
generator, tensorview headers and build discovery), so spconv cannot be ported without
changing it. jeff ruled spconv adopted on 2026-11-20 (`data/decisions.jsonl`, unskip of
the 2026-08-20 decline); cumm is its hard build dependency. The fork is a person's act;
nothing here created one.

Upstream: https://github.com/FindDefinition/cumm, default branch `main`, screened at
`d971d02a0` (2026-03-21). 268 tracked files, no submodules, no binary files.

### Licence -- top level Apache-2.0; recorded as a compound expression, tier 3 by the gate

    python3 utils/licenses.py check FindDefinition/cumm            # Apache-2.0, tier 1 (top level only)
    git clone https://github.com/FindDefinition/cumm agent_space/cumm-screen
    python3 utils/licenses.py scan-nvidia agent_space/cumm-screen  # clean: no NVIDIA proprietary licence text

Recorded with `set-license-spdx`:
`Apache-2.0 AND BSD-3-Clause AND MIT AND BSL-1.0 AND Unlicense AND GPL-3.0-or-later WITH GCC-exception-3.1`
(evidence: `LICENSE` at `d971d02a0`). `tier_of` tiers the WITH part by its base, so the
gate reports **tier 3, needs approval before anything is offered upstream**. That is
deliberate: the four GPL-headed files below are real and are shipped in every wheel, and
recording only `Apache-2.0` would have hidden them from the gate.

Per-file scan (every tracked file read for copyright, SPDX and licence text; provenance
comments grepped for "taken/adapted/copied from" and GitHub blob links). `LICENSE` is the
only licence file in the tree; `third_party/` holds only a README (CI clones NVIDIA CCCL
into it at wheel-build time, see below). 159 files carry `Copyright 2024 Yan Yan` +
Apache-2.0 headers.

Components whose terms differ from the top-level Apache-2.0:

| path | licence | origin | compiled/shipped? | would the port modify it? |
|---|---|---|---|---|
| `include/tensorview/core/nvrtc_std.h`, `core/nvrtc/core.h`, `core/nvrtc/limits.h`, `core/nvrtc/type_traits.h` | **GPL-3.0-or-later WITH GCC-exception-3.1** (libstdc++ headers, "Copyright (C) 2008-2021 Free Software Foundation") | copied from GCC's libstdc++ for NVRTC, which has no std library | Shipped as headers in every cumm wheel (`setup.py` copies `include/` into the package); included by `tensorview.h`, `dtypes.h`, `half.h`, `bfloat16.h`, `float8.h`, `tf32.h`, `parallel/ops.h`, `mp_helper.h`, so compiled into every device TU when built by NVRTC (pre-12.6 CUDA) | Possibly. `nvrtc_std.h` is in fact cumm's own 52-line dispatcher (CUDA >= 12.6 uses libcu++ instead) wearing the GPL header; a hipRTC path would most naturally add a branch there. `type_traits.h:492` also carries a snippet from NVIDIA CCCL libcudacxx (Apache-2.0 WITH LLVM-exception) |
| `include/tensorview/gemm/arch/{memory.h,memory_sm75.h,semaphore.h,transpose.h}`, `gemm/dtypes/{bfloat16,complex,half,subint,tf32,uint128}.h`, `gemm/math/{functional,numeric_convert}.h`, `math/fastmath.h` | BSD-3-Clause, "Copyright (c) 2017-2021, NVIDIA CORPORATION" | CUTLASS | Compiled into the generated kernels | **Yes**: `memory.h` and `semaphore.h` were changed in the HIP study (inline PTX); others hold PTX behind arch guards. Keep the notice; tier 1 |
| `include/tensorview/gemm/dtypes/float8.h` | BSD-3-Clause (SPDX header), NVIDIA CORPORATION & AFFILIATES 2017-2022 | CUTLASS | compiled | maybe (PTX behind guards) |
| `include/tensorview/cuda/driver.h` | Apache-2.0, NVIDIA 2021 | TensorRT `cudaDriverWrapper` | compiled (NVRTC/driver path) | likely, for a hipModule path; same licence as top level |
| `include/tensorview/core/cc17.h`, a block in `include/tensorview/tensor.h` | BSD-3-Clause (PyTorch/Caffe2 licence text: Facebook, Idiap, Deepmind, NEC, NYU, Google) | PyTorch c10 helpers | compiled (host) | `tensor.h` likely (allocation/streams); notice stays |
| `include/tensorview/core/mp_helper.h` (part) | BSL-1.0, Peter Dimov 2015-2017 | Boost.Mp11 fragment | compiled | unlikely |
| `include/tensorview/prettyprint.h` | BSL-1.0, Louis Delacroix 2010-2014 | cxx-prettyprint | compiled (host) | no |
| `include/tensorview/thirdparty/nlohmann/json.hpp` | MIT (SPDX), Niels Lohmann 2013-2023 | nlohmann/json | compiled (host) | no |
| `include/tensorview/thirdparty/tsl/robin_{growth_policy,hash,map,set}.h` | MIT, Thibaut Goetghebuer-Planchon 2017 | Tessil/robin-map | compiled (host) | no |
| `include/tensorview/core/arrayops/linalg.h` | Unlicense (no header in file; comment "from https://github.com/sgorsten/linalg", which is Unlicense) | sgorsten/linalg | compiled | no |
| `include/tensorview/profile/cuda_profiler.h` (about 20 lines) | PSF-2.0 by origin, no header ("adapted from cpython pytime.c") | CPython | compiled (host, `TV_USE_LIBRT`) | no |
| Stack Overflow snippets (`core/array.h:525`, `utility/tuplehash.h`, `pybind.h:254`, `cumm/nvrtc/__init__.py`) | CC BY-SA by origin, a few lines each | Stack Overflow | compiled | no |

Not in the tree but put into upstream's CUDA wheels by its build: NVIDIA CCCL
`libcudacxx/include` (cloned by `.github/workflows/build.yaml` and
`tools/build-wheels-dev.sh` into `third_party/cccl`, copied to `cumm/libcudacxx_include`,
Apache-2.0 WITH LLVM-exception) and, on Linux, `/usr/local/cuda/lib64/libcudadevrt.a`
(NVIDIA CUDA EULA redistributable). A ROCm wheel would ship neither; the port's wheel
script must not copy them.

Metadata inconsistency to know about: `setup.py` says `license='MIT'` and the MIT trove
classifier, so PyPI shows cumm-cu12x as MIT, while `LICENSE` and every header say
Apache-2.0. The `LICENSE` file governs; nothing to do unless a PR touches `setup.py`.

**What needs a person's clearance:** one item. The tier-3 component (the four
GPL-3.0-or-later WITH GCC-exception-3.1 headers) means `license-gate cumm` stays closed
until someone runs `record-license-clearance`. Two things for that person to decide:
(1) contributing to an Apache-2.0 project that already ships these headers is the
question the tier asks about; (2) if the port has to edit `nvrtc_std.h` (likely for a
hipRTC branch), that edit lands in a GPL-headed file. The planner can avoid (2) by adding
the hipRTC std dispatch in a new Apache-2.0 header or in `common.py`'s include list
rather than editing the GPL-headed files. Everything else is tier 1 (BSD-3-Clause, MIT,
BSL-1.0, Unlicense, PSF-2.0, Apache-2.0 WITH LLVM-exception) and needs only its notices
kept. Scope: this is about contributing; shipping ROCm wheels of cumm that include the
GPL-with-exception headers is the same distribution upstream already does on PyPI, but a
use/ship question is separate from the tier.

### Duplicate effort and upstream state

- Upstream PRs, issues and branches mentioning HIP/ROCm/AMD: **none** (`gh pr list`
  and `gh issue list --search "hip OR rocm OR amd" --state all`; branches are `0.2.x`,
  `feature/ampere`, `feature/debug`, `main`). Docs grep `amd|rocm|hip|gfx`: no hits.
- AMD-Ecosystem and ROCm orgs: no cumm, spconv, ccimport or pccm repository.
- **`ZJLi2013/cumm-rocm`** (fork of cumm, Apache-2.0, 2 stars): the work is on the
  non-default branch `rocm` (`31e1a2f1c`, 110 commits ahead of upstream, last push 2026-05-09; `main` is an unmodified
  mirror at `d971d02a0`). It is a **rewrite, not a port**: its README says it "replaces
  the entire codegen + nvcc pipeline" with ROCm FlyDSL (MLIR) kernels, a dense
  `FlyDSLGemmTuner` plus fused implicit-GEMM MFMA kernels for spconv inference; large
  channels and training fall back to `at::mm`; experimental kernels have documented
  correctness issues. Not on PyPI (`cumm-rocm` does not resolve). It is consumed by
  `ZJLi2013/spconv_rocm` (branch `rocm`), which in turn is the base of
  `HYGON-AI/spconv-das` (Hygon DCU downstream, active 2026-11-12) -- but the Hygon tree
  says "cumm and FlyDSL are not build or runtime dependencies of the current HCU path".
  Single author (profile shows only a name and a city), FlyDSL and MI308X in the
  README suggest an AMD-adjacent engineer. **Coordination question for jeff:** is
  ZJLi2013 an AMD colleague, and does a cumm codegen HIP backend compete with or
  complement that FlyDSL rewrite? It is dormant six months, not authoritative, and
  covers a different design point (inference speed on gfx942 via MFMA, no CUDA codegen
  path), so it does not make this an `already-supported` case.
- The Triton replacements (`L-Reichardt/spconv-triton`, `JeffreyXiang/FlexGEMM`) replace
  spconv, not cumm; they are why spconv was declined on 2026-08-20 and are covered in
  the spconv section below.
- Activity: last commit 2026-03-21 ("fix some nvrtc bug in cuda 13.0"); 11 commits since
  2025, all by the author, mostly new-CUDA-version enablement (v0.8.2, CUDA 12.8,
  2025-05-28). 3 PRs ever merged, the last 2022-12-22; only 1 from outside (#10, a README
  fix, 2022). Open outside PRs: #34 (2025-08-21, "Fix hardcoded Linux CUDA path in
  common.py" -- the same discovery code a HIP port changes) and #36 (2026-04-22, int64
  gather indices), both unanswered. The author's last comment on a cumm issue was
  2023-04-06; he is active elsewhere on GitHub (triton, torchtitan, accelerate in
  2026-11), so the account is alive but this repository gets only his own maintenance.
- Destination: upstream is effectively dormant for outside contributions. Under the
  standing rule a clean, complete port is still sent as a PR (after approval), with no
  expectation of a merge; the fork's `moat-port` branch and its ROCm wheels are the
  real deliverable. If the maintainer declines or never answers, the person ruling it
  records `set-maintained`. A partial port (for example SIMT-only with fp16 tensor-op
  algorithms just switched off) is a scope cut that needs jeff's answer first.

### Viability -- yes (from jeff's study, not redone)

Source: `~/.claude/plans/moat-cumm-hip-backend-study.md`, Results and "Follow-up: the
index-pair bug" (gfx942, cumm v0.7.13, spconv `263d6b4`).

- A throwaway shim (fake nvcc calling hipcc after hipify, fake CUDA root) built
  spconv's whole SIMT kernel set (265 generated `.cu`, 168 host `.cc`) for gfx942. All
  45 gather/scatter GEMMs were correct in f32/f16 after one wave64 fix (the kernel
  prologue `warp_idx` broadcast needs `__shfl_sync(..., 0, 32)`).
- The one wrong-result bug was an uninitialised local `warp_res` in tensorview's
  `atomicAggInc` (`tensorview/cuda/device_ops.h:84`, `tensorview/parallel/ops.h:149`):
  undefined behaviour nvcc tolerates and clang optimises on (counters reached c^2). A
  two-line fix made pair generation and fp32 SubMConv3d / SparseConv3d (Native,
  MaskImplicitGemm, MaskSplitImplicitGemm, fwd/dgrad/wgrad) match the CPU.
- cumm-side walls, all in cumm: nvcc discovery and `/usr/local/cuda` assumptions
  (`common.py`), inline PTX (10 `asm_block` sites in the generator plus
  `memory.h`/`semaphore.h`), `__CUDA_ARCH__` kernel-body gates (HIP never defines it),
  `TV_ASSERT_*` in host-device code (clang strictness), nvcc extended-lambda trait,
  CC collision (gfx942 reports 9.4; `arch_is_compiled` exact match), NVRTC path ->
  hipRTC/hipModule (50 of 100 SIMT implicit-GEMM configs are NVRTC-only).
- Second known wall: fp16 selects NVIDIA tensor-op kernel descriptors (gfx942's (9,4)
  passes Turing's `min_arch`); fix is to admit only prebuilt `tensorop == 0`
  descriptors on HIP (a few lines in spconv `algo.py`).
- Size, all three repos: 38 to 48 files, 900 to 1500 lines, 4 to 6 porter rounds;
  cumm is about 25 to 30 of the files. Doublers left: the hipRTC path, upstream-quality
  abstraction instead of `__HIP_PLATFORM_AMD__` branches, Windows (MSVC host + clang
  device). An MFMA tensor-op family (fp16/bf16 speed) is separate, 3,000 to 6,000 new
  lines, and not needed for correctness.
- NVIDIA libraries: thrust (spconv), curand (tensor.h); rocThrust and hipRAND cover
  them. No CUTLASS library dependency (cumm reimplements it); cuBLASLt only in a debug
  class.

### Dependencies and release

- Build chain: cumm -> `pccm` (FindDefinition/pccm, MIT, pure-Python code generator)
  -> `ccimport` (ninja-file writer). cumm's `setup.py` requires `pccm>=0.4.15` and
  `pybind11`; ccimport arrives through pccm. `depends_on: [ccimport]` is recorded (at
  scaffold) for the edge jeff named; it clears when ccimport is completed or carries an
  `already-supported` disposition, which is the ccimport recommendation (see
  `projects/ccimport/notes.md` on `port/ccimport`): ccimport already routes the `nvcc`
  compiler slot to whatever `CUDACXX` names, and its nvcc rule is a plain
  gcc-style command line hipcc accepts.
- `pccm` is not a MOAT project and not recorded as a dependency. The study changed no
  pccm file; pccm's PTX DSL (`pccm/targets/cuda_ptx.py`) only emits the `asm` text, and
  the probe replaced those sites in cumm's generator. If the planner wants a C++ lowering
  in the DSL itself instead, pccm needs its own intake request first.
- spconv depends on cumm (`cumm>=0.7.11,<0.8.0` on spconv master; cumm head is 0.8.2;
  spconv PR #749 about the pin is open and unmerged).
- Releases: cumm publishes per-CUDA wheels to PyPI (`cumm-cu124`, `cumm-cu126`, 0.8.2,
  manylinux_2_28 x86_64 and win_amd64, 5 Python versions each) from its own GitHub
  Actions (`.github/workflows/build.yaml`), using pccm/ccimport to build the prebuilt
  `core_cc` module. A ROCm deliverable would be the fork's `moat-port` branch plus
  wheels named in the same style per ROCm major (for example `cumm-rocm7`), prebuilt for
  a fixed gfx list, with a documented source build as fallback. Where such wheels are
  published (fork releases, an AMD index, PyPI) is a person's decision. Note
  `ZJLi2013` already uses the names `cumm-rocm`/`spconv-rocm` (not on PyPI).

### Recorded

    python3 utils/moatlib.py scaffold FindDefinition/cumm --deps ccimport
    python3 utils/moatlib.py set-license-spdx cumm "Apache-2.0 AND BSD-3-Clause AND MIT AND BSL-1.0 AND Unlicense AND GPL-3.0-or-later WITH GCC-exception-3.1" --evidence https://github.com/FindDefinition/cumm/blob/d971d02a0/LICENSE
    python3 utils/moatlib.py set-intake cumm fork ...
    python3 utils/moatlib.py set-state cumm linux-gfx942 awaiting-fork --agent intake

The intake queue issue was not published (`intake_queue.py publish --apply` is a GitHub
write outside this job's limits); run it from a session that may.

### Questions for jeff

1. Licence clearance for the GPL-3.0-or-later WITH GCC-exception-3.1 headers
   (`record-license-clearance cumm`), or a ruling that the port must not edit them.
2. Is `ZJLi2013` (cumm-rocm / spconv_rocm, FlyDSL rewrite) an AMD colleague to
   coordinate with?
3. Scope: is a Tier-1 port (SIMT kernels on HIP, tensor-op descriptors excluded, MFMA
   family deferred) the complete port for PR purposes, or a scope cut?

## spconv screen

Moved to `projects/spconv/notes.md` (on `port/spconv`) on 2026-11-23, after jeff's ruling to delete the old branch let the record be scaffolded.

## Lesson for moat-porting: spconv_rocm is not empty; its work is on branch `rocm`

The Dispositions section says the GitHub repository `spconv_rocm` is empty, as checked
2026-11 by the Pointcept port. `ZJLi2013/spconv_rocm` is a fork of spconv whose default
branch `master` is an unmodified upstream mirror, so it looks empty of ROCm work; the
port is on the non-default branch `rocm` (28 commits ahead of `master`, 165 files
changed, last 2026-06-18), and its cumm counterpart `ZJLi2013/cumm-rocm` likewise on
`rocm` (110 commits ahead, 296 files). Correct the line, and note that
it is a FlyDSL rewrite usable as environment, not a ROCm build of spconv as published.

## Lesson for cuda-to-rocm: list every branch of a fork before calling it empty

For assess-existing-support.md. A fork whose default branch is an untouched upstream
mirror can hold a complete port on another branch (`rocm`, `hip`, `amd`), and GitHub's
repository page, description and search show only the default branch. When a candidate
port repository looks empty or identical to upstream, list its branches
(`gh api repos/<owner>/<repo>/branches`) and compare each tip to upstream before
concluding there is no prior work. Seen on a spconv/cumm FlyDSL rewrite (2026-11).
