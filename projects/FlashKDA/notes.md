# FlashKDA notes

## Intake screen (2026-08-13, linux-gfx1100)

Upstream: https://github.com/MoonshotAI/FlashKDA — "FlashKDA: Flash Kimi Delta
Attention", high-performance KDA attention kernels built on CUTLASS/CuTe. Screened
from a shallow clone at `agent_space/FlashKDA-screen`; no fork exists and none was
requested.

**Recommendation: decline, reason `cant-port`.** The argument is below. It is a
recommendation, not a decision — a person answers it through the intake queue.

### Licence

`license_spdx = MIT` (tier 1, cleared to contribute). Confirmed by reading `LICENSE`
in the tree, not just GitHub's field: verbatim MIT, "Copyright (c) 2026 MoonshotAI".
GitHub's API agrees (`MIT`).

`scan-nvidia` over the main tree is clean.

**One finding a person must rule on if this is ever adopted.** The repo carries a
submodule, `cutlass` -> https://github.com/NVIDIA/cutlass.git, pinned at
`5c149f52a436782210263fb2f19b354443a61c6a`. That tree is dual-licensed per-part:

- `LICENSE.txt` is BSD-3-Clause and covers the C++ headers, and it says explicitly
  that "the files located in the `python/CuTeDSL` directory are licensed under the
  NVIDIA End User License Agreement (EULA)".
- `EULA.txt` is that agreement. `scan-nvidia` over the submodule flags ~40 files, all
  of them under `python/CuTeDSL/`, and none anywhere else.

The EULA restricts the Software to "systems with NVIDIA GPUs" (1.1) and clause 2.11
forbids reverse-engineering output artifacts "for the purpose of translating such
output artifacts to target a non-NVIDIA platform" — which names the activity a
CUDA-to-ROCm port performs. It applies only to the CuTeDSL Python package.

FlashKDA does not use that package. Its build pulls in only the BSD-3-Clause C++
headers: `setup.py` adds `cutlass/include`, `cutlass/examples/common`, and
`cutlass/tools/util/include` to the include path, and every source include is a
`cute/...` or `cutlass/...` C++ header. Nothing imports CuTeDSL, and the port would
neither modify nor redistribute it.

So the finding is almost certainly benign — but per the intake role an NVIDIA
proprietary licence on any file needs a person's decision before proceeding, and I do
not clear it myself. If the decline is upheld the question is moot; if someone
overrides to fork, this must be answered first and not skipped.

### Duplicate effort

No AMD or ROCm effort on FlashKDA itself:

- No `FlashKDA` repository in AMD-Ecosystem (checked the org listing; the near-name
  matches are `flashinfer`, `flashinfer-bench`, `ffpa-attn` — different projects).
- Nothing matching in the ROCm org.
- GitHub repo search for `flashkda` returns upstream plus `vllm-project/FlashKDA`
  (12 stars, branches `master`/`dev` plus two feature branches, no ROCm branch),
  two empty personal forks, and `popfido/FlashKDA-mlx` — an Apple MLX port, so there
  is porting interest, but none toward AMD.
- `grep -rniE 'amd|rocm|hip|gfx[0-9]' README* docs/ BENCHMARK*.md` yields one hit,
  and it is the substring in "on-chip". No notable-forks section, no platform ports.
- No MOAT disposition, no opt-out, no other `port/` branch for it or for a related
  project.

**The capability, though, already reaches AMD by another route.** FlashKDA is not a
standalone library — it is an optional backend for `flash-linear-attention`'s
`chunk_kda`, auto-dispatched when installed and disabled with `FLA_FLASH_KDA=0`, at
which point FLA falls back to its own Triton kernels. Those Triton kernels
(`fla/ops/kda/`) run on ROCm, and fla-org tests them on AMD in CI —
`.github/workflows/amd-mi300.yml`. So an AMD user wanting Kimi Delta Attention has a
supported, upstream-CI-tested path today.
[**CORRECTED 2026-08-14 by the gfx942 screen: the CI half of this is wrong.** That
workflow is disabled (`if: false`) and has never run — 0 runs. The Triton path is
installable on ROCm but upstream-unvalidated on AMD hardware. See the gfx942 section.]
This is not `already-supported` for
FlashKDA, which has no AMD port at all, but it changes what a FlashKDA port would
buy: speed on hardware-specific code, not access to a missing capability.

### Viability

Genuine CUDA, and unusually concentrated: a torch extension of ~2,300 lines across
six files, all of the GPU work in `csrc/smxx/`.

The problem is what it is built on.

- **Every kernel is CuTe/CUTLASS.** `csrc/smxx/utils.cuh` alone pulls 19 CUTLASS
  headers, including `cute/arch/copy_sm90_tma.hpp`, `cute/arch/cluster_sm90.hpp`,
  `cutlass/cluster_launch.hpp`, `cutlass/arch/barrier.h`, and
  `cutlass/pipeline/sm90_pipeline.hpp`. Layouts, tensors, and the GEMMs are all CuTe
  types. NVIDIA CUTLASS has no ROCm/HIP backend; AMD's analogue is Composable Kernel,
  a different library with a different API. Porting therefore is not translating a
  CUDA path — it is reimplementing the kernels against a different tile library.
- **TMA is pervasive, not incidental.** 25 `SM90_TMA` references and 24 `make_tma`
  call sites, plus `CUTE_GRID_CONSTANT` TMA descriptor parameters threaded through
  every kernel entry point (`tma_load_q`, `tma_load_k`, `tma_store_ws_*`, ...), and
  `ClusterTransactionBarrier` in shared memory. The most recent upstream commit is
  literally "fix missing proxy fences around TMA accesses". Neither CDNA3 nor RDNA3
  has a TMA equivalent; the data movement would have to be rewritten, and with it the
  pipeline and barrier structure that is the design.
- **Upstream targets Hopper and Blackwell only.** `SUPPORTED_CUDA_ARCHS =
  ["90a", "100a", "103a", "120a"]`, README requirement "SM90 and above", benchmark
  files for GB200 and H20.
- **This host's platform is the worst fit of the family.** gfx1100 is RDNA3: wave32,
  no MFMA, no async-copy analogue of any of this. The `wave64` gate would need CDNA
  and would still need the same full rewrite; there is no cheap gate to satisfy.
- Three inline PTX asm uses (`ex2.approx.ftz.f32`, `tanh.approx.f32`,
  `cvt.f32.bf16`) — trivial next to the above, noted only for completeness.

That combination puts the work far outside MOAT's standing rule to build the smallest
complete port preserving upstream structure and the CUDA path. A from-scratch
Composable-Kernel or raw-HIP reimplementation of a Hopper-tuned attention kernel is a
new kernel project, and it would arrive upstream as a large unsolicited parallel
backend in a repo whose reason for existing is Hopper/Blackwell tuning — a poor
prospect for acceptance even if it were built and validated.

**Dependencies.** None on any MOAT project, so `depends_on` stays empty. External
build dependencies are PyTorch (>= 2.4), CUDA 12.9+, and the vendored CUTLASS
submodule. The `flash-linear-attention` relationship is integration, not a build
dependency.

**Upstream health.** Healthy, and not a factor in the decline: not archived, 1208
stars, 114 forks, last push 2026-07-30, active commits through late July 2026. If the
recommendation is overridden, upstream is at least alive enough to receive a PR.

### If a person overrides to fork

Two things must happen before any porting work, in this order: rule on the CuTeDSL
EULA finding above, and accept that the first real task is a kernel reimplementation
rather than a translation — so the planner should scope it as such, and on a CDNA
host, not on gfx1100.

## Intake re-screen (2026-08-14, linux-gfx90a)

Independent second-host confirmation of the 2026-08-13 screen. **The decline
recommendation stands, unchanged: `cant-port`.** Verified against a fresh shallow
clone at upstream head `1ce47ea` ("optimize kda prepare cu_seqlens scan with
prefix-sum and binary search (#13)"), one commit ahead of the first screen. Nothing
material changed.

Re-verified, each independently rather than taken on trust:

- Licence MIT, tier 1 (`licenses.py check` → `license=MIT tier=1`). `scan-nvidia`
  over the main tree: clean.
- The `cutlass` submodule is still pinned at `5c149f52`, and that pin's `LICENSE.txt`
  still carries the per-part clause ("The files located in the `python/CuTeDSL`
  directory are licensed under the NVIDIA End User License Agreement"), with
  `EULA.txt` alongside it. FlashKDA still uses none of it: no `cutedsl`,
  `cutlass.cute`, or `import cutlass` anywhere in `csrc/`, `flash_kda/`, `setup.py`,
  `tests/`, or `benchmarks/`, and `setup.py` adds only the BSD-3-Clause C++ include
  paths. **Still a person's ruling, still not cleared here.**
- No AMD-Ecosystem or ROCm FlashKDA repo (both 404). Repo search adds nothing new
  toward AMD. Note for whoever re-runs this: a fork-list regex for `amd` matches
  `Shamdon/FlashKDA` on the substring in the owner's name — a false positive, not an
  AMD fork.
- `grep -rniE 'amd|rocm|hip|gfx[0-9]' README* docs/ BENCHMARK*.md` still yields
  exactly one hit, still the substring in "on-chip".
- The AMD route to the capability is real and still live: `fla-org/flash-linear-attention`
  has `fla/ops/kda/` (Triton, with a `backends/` subpackage) and
  `.github/workflows/amd-mi300.yml`. [**CORRECTED by the gfx942 screen:** confirming
  the workflow file exists is not confirming CI coverage. It is disabled and has
  never run.]
- `SUPPORTED_CUDA_ARCHS = ["90a", "100a", "103a", "120a"]`; README requirements
  "SM90 and above", CUDA 12.9+, PyTorch 2.4+. Source is 2,346 lines across six files.
- CuTe/TMA density confirmed: 24 `SM90_TMA` references, 24 `make_tma` sites, 8
  cluster/sm90-pipeline references. The first screen said 25 TMA sites; the correct
  count is 24, so the queue summary is corrected. The difference changes nothing.

**One refinement this host adds, and it is the reason the re-screen was worth
running.** The first screen was performed on gfx1100 and noted that RDNA3 is "the
worst fit of the family" — wave32, no MFMA. That invites the objection that the
decline is an artifact of screening on unsuitable hardware. It is not. This host is
gfx90a: CDNA2, wave64, MFMA present — the `wave64` gate architecture, and a far
better-suited target. The decline is unchanged from here, because it never rested on
the host's wavefront or matrix-core support. It rests on two things neither CDNA2 nor
CDNA3 changes:

1. NVIDIA CUTLASS/CuTe has no ROCm backend, and every layout, tensor, and GEMM in
   this codebase is a CuTe type. AMD's Composable Kernel is a different library with
   a different API, not a drop-in.
2. TMA has no AMD equivalent on any current architecture, and TMA is the design here,
   not an optimization layered on top — descriptors are threaded through every kernel
   entry point as `CUTE_GRID_CONSTANT` parameters and the pipeline and barrier
   structure is built around them.

So a CDNA host would face the same from-scratch kernel reimplementation that gfx1100
would. There is no platform in the fleet on which this becomes a translation rather
than a rewrite, which is precisely what `cant-port` means.

**Dependencies re-checked:** `depends_on` stays empty. No MOAT project provides
CUTLASS or flash-linear-attention (checked against `moatlib.py projects` and
`DEPENDENCIES.md`), so there is no unknown hard dependency needing an intake request.
PyTorch, CUDA 12.9+, and the vendored CUTLASS submodule are external build
dependencies only.

**Upstream health:** not archived, not disabled, 1211 stars, 115 forks, last push
2026-07-30. A PR would have a live destination if the recommendation were overridden.

**Unrelated data defect noticed while screening (first noted 2026-08-14, gfx90a).**
`data/candidates.json` carries two entries named `FlashKDA`:
`MoonshotAI/FlashKDA` (the one screened here) and `0xwilliamortiz/FlashKDA`
(195 stars, MIT, "memory-efficient KDA kernels for training and decode"). The second
is a **404 on the GitHub API** — deleted or never public. Beyond being stale, it is a
name collision: adopting it would scaffold to the same `projects/FlashKDA` and the
same `port/FlashKDA` claim as this project. Not fixed here, since candidate curation
is not the intake role's to edit and the decline makes it non-urgent.

## Intake re-screen (2026-08-14, linux-gfx942)

Third-host screen, on CDNA3/MI300 — the architecture the previous two screens
pointed at as the best-case target and the one the claimed AMD fallback route names.
**The decline recommendation stands: `cant-port`.** But this screen corrects a
supporting fact that both earlier screens recorded as true, and that correction is
the reason this run was worth doing.

Verified against a fresh shallow clone at upstream head `1ce47ea` — the same commit
the gfx90a screen saw, so the code has not moved.

### Correction: the "CI-tested on MI300" claim is false

Both prior screens, and the recorded `intake.duplicate_effort` and `summary` fields,
stated that `fla-org/flash-linear-attention` tests its Triton KDA kernels on AMD in
CI, citing `.github/workflows/amd-mi300.yml`. The file exists. It has never run.

- The job is guarded `if: false`, and the file's own header comment says: "Disabled
  by default: fla does not yet operate an amd-mi300 runner. Flip the `if:` guard
  below (or replace with a real condition) once a runner is wired in. Workflow lives
  here so the install / sanity-check pattern is documented."
- Its trigger is `workflow_dispatch` only — no push or PR trigger.
- Run count via the Actions API: **0**. For contrast, `nvidia-h100.yml` has 3028.

The existence of a workflow file is not evidence of CI coverage. Checking the run
count is the cheap disambiguation, and it should be the habit whenever an
existing-support claim rests on a CI file.

**What is actually true about the AMD route**, stated at the strength the evidence
supports: fla ships KDA as Triton kernels (`fla/ops/kda/`, the default path) and
offers a documented ROCm install extra (`pip install -e ".[rocm]"`, which
deliberately does not pin a Triton flavor so the ROCm wheel index supplies
`pytorch-triton-rocm`). So the AMD path is intended and plausible — Triton targets
ROCm — but it is **unvalidated by upstream on AMD hardware**. Nobody should record
that AMD users have a tested KDA path today.

This weakens the "capability already reaches AMD" argument rather than the decline.
It also points at the tractable work, which is worth saying plainly in the queue: if
someone wants Kimi Delta Attention working on MI300, the cheap, high-value task is
validating fla's existing Triton `fla/ops/kda` on gfx942 — a different project, and
one whose upstream has already built the scaffolding and is visibly waiting for a
runner. Rewriting FlashKDA's CuTe kernels is the expensive way to the same capability.

### Re-verified independently on this host

- Licence MIT, tier 1 (`licenses.py check` → `license=MIT tier=1`); `LICENSE` read
  directly, verbatim MIT, "Copyright (c) 2026 MoonshotAI". `scan-nvidia` over the
  main tree: clean.
- `cutlass` submodule still pinned at `5c149f52a436782210263fb2f19b354443a61c6a`
  (`git ls-tree HEAD` gitlink), url `https://github.com/NVIDIA/cutlass.git`. The
  per-part CuTeDSL EULA finding from the first screen is unchanged and **still not
  cleared here — it remains a person's ruling if this is ever adopted.** FlashKDA
  uses none of it: a case-insensitive search for `cutedsl`, `cutlass.cute`,
  `import cutlass`, `from cutlass` across the whole tree returns nothing, and
  `setup.py` adds only the BSD-3-Clause C++ include paths (`cutlass/include`,
  `cutlass/examples/common`, `cutlass/tools/util/include`).
- No `FlashKDA` repo in AMD-Ecosystem or ROCm (both 404), nor `AMD-Ecosystem/flash-kda`
  or `ROCm/flash-linear-attention`. Repo search returns upstream, `vllm-project/FlashKDA`
  (12 stars), two empty personal repos, and `popfido/FlashKDA-mlx`. Nothing toward AMD.
- No MOAT disposition for FlashKDA in `data/dispositions.json` (282 entries), no
  opt-out (`optout.py list` → nobody), no other `port/` branch.
- `SUPPORTED_CUDA_ARCHS = ["90a", "100a", "103a", "120a"]`; 2346 lines across six
  source files; three inline PTX uses.
- CuTe/TMA density: 24 `SM90_TMA` uses, 24 `make_tma` sites, 22 `CUTE_GRID_CONSTANT`
  parameters, 5 `ClusterTransactionBarrier`, plus `cutlass/pipeline/sm90_pipeline.hpp`
  and `cutlass/cluster_launch.hpp`.

**The 24-vs-25 TMA discrepancy between the first two screens is settled**, since it
cost a correction once already: case-sensitive `SM90_TMA` in `csrc/` is 24;
case-insensitive is 25, the extra match being the include filename
`cute/arch/copy_sm90_tma.hpp` on line 14 of `utils.cuh`. Both counts were right about
different questions. 24 is the number of code uses.

### Why CDNA3 does not rescue it

gfx942 is the strongest case AMD can make here — CDNA3, wave64, MFMA, and the
`wave64` gate architecture. The decline is unchanged from this host for the reason
the gfx90a screen already gave, which this host confirms rather than repeats: the
blocker is not wavefront width or matrix-core availability. It is that every layout,
tensor, and GEMM is a CuTe type and NVIDIA CUTLASS has no ROCm backend, and that TMA
is the design rather than an optimization — descriptors threaded through every kernel
entry point as `CUTE_GRID_CONSTANT` parameters, with the pipeline and barrier
structure built around them. CDNA3 has no TMA equivalent. Three hosts spanning RDNA3,
CDNA2 and CDNA3 now agree, which is as much as screening can establish: there is no
platform in the fleet on which this becomes a translation rather than a from-scratch
kernel reimplementation. That is what `cant-port` means.

**Dependencies:** `depends_on` stays empty; no MOAT project provides CUTLASS or
flash-linear-attention. PyTorch, CUDA 12.9+, and the vendored CUTLASS submodule are
external build dependencies only.

**Upstream health:** not archived, not disabled, 1211 stars, 115 forks, last push
2026-07-30, default branch `master`. A PR would have a live destination if the
recommendation were overridden.

**Minor data observation, not load-bearing (gfx942 screen).** `data/retired_stats.jsonl` line 83
carries a single token record for project `FlashKDA` dated 2026-06-04 with source
`porter` — two months before this project's `adopted_at` (2026-08-07) and before
upstream's own recorded activity. There is no matching disposition and no other
trace of a prior lifecycle. Flagged for whoever maintains telemetry; it does not
affect the recommendation and I did not edit it.

## Intake delta-check (2026-08-19, linux-gfx1100) — fourth dispatch, NOT a fourth screen

This host was dispatched FlashKDA at `stage: unclaimed` for the fourth time. Three
screens (2026-08-13 gfx1100, 2026-08-14 gfx90a, 2026-08-14 gfx942) already agree on
`decline` / `cant-port`, and that recommendation is already recorded in
`status.json.intake`. So this run deliberately did **not** re-derive the screen. It
checked only what could have changed since 2026-08-14, and stopped.

**Nothing changed. The recommendation stands, unchanged: `decline`, `cant-port`.**

Delta checks, all read-only, no clone:

- **Upstream head is still `1ce47ea`** ("optimize kda prepare cu_seqlens scan with
  prefix-sum and binary search (#13)", authored 2026-07-29), the same commit the
  gfx90a and gfx942 screens verified. `master` is the only branch. The tree is
  therefore byte-identical to what was screened three times, so every code-level
  finding — 24 `SM90_TMA` uses, 24 `make_tma` sites, 22 `CUTE_GRID_CONSTANT`
  descriptors, `SUPPORTED_CUDA_ARCHS = ["90a","100a","103a","120a"]`, 2346 lines
  across six files, the `cutlass` submodule pinned at `5c149f52` — holds by
  construction. Re-cloning to recount them would produce the same numbers from the
  same bytes, which is why it was skipped rather than repeated.
- **Licence unchanged: MIT, tier 1.** GitHub API `license.spdx_id = MIT`; the
  `LICENSE` file was read directly by all three prior screens at this same SHA.
  `license_spdx` was already recorded and stays `MIT`.
- **The CuTeDSL EULA finding is unchanged and still uncleared.** Same submodule pin,
  so same per-part licensing. It remains a person's ruling if this is ever adopted,
  and this host does not clear it either.
- **No AMD effort appeared.** `AMD-Ecosystem/FlashKDA`, `ROCm/FlashKDA`, and
  `AMD-Ecosystem/flash-kda` are all still 404. A fresh repo search surfaces one name
  the earlier screens did not see, `Unitflexmed1821/FlashKDA` (0 stars, not a fork,
  pushed 2026-08-19, head "Update README.md", description advertising CUTLASS
  kernels). Its README has zero matches for `amd|rocm|hip|gfx[0-9]` — an unrelated
  re-upload, not an AMD port. Also new: `atomicmilkshake/godzilla-llama.cpp`, an
  MSVC+CUDA Windows llama.cpp fork carrying KDA, likewise nothing toward AMD.
- **Upstream still healthy**, still a live PR destination if the recommendation is
  overridden: not archived, not disabled, 1218 stars (was 1211), 117 forks (was 115),
  last push 2026-07-30, default branch `master`.
- `depends_on` stays empty.

### Why this project keeps coming back, which is the finding worth acting on

The screen is not stuck — the *decision* is. FlashKDA is row 1 of the open intake
queue, issue AMD-Ecosystem/moat#8, opened 2026-08-07 and carrying **zero comments
after 12 days**. Its recommendation has been correct and complete since 2026-08-13.

An agent may not record a decline, so `stage` correctly stays `unclaimed` — and the
selector offers `unclaimed` projects to whatever host asks next. A decline
recommendation awaiting a person is therefore indistinguishable, to the selector,
from a project nobody has looked at. That is the whole mechanism, and it is the same
one that had spconv screened six times on four platforms; spconv is row 2 of the same
unanswered issue. Four sessions have now been spent on FlashKDA to reach the answer
the first one reached.

Two things would stop it, and both belong to a person, not to this role:

1. **Answer issue #8.** One reply decides this batch, and `intake_queue.py apply`
   records it. This is the real fix; everything else is a workaround.
2. **`moatlib.py set-hold FlashKDA`** if the answer will be a while. AGENTS.md
   reserves `set-hold` to a person, so this host did not set it — but a held project
   is skipped by the selector on every platform with no state touched, which is
   exactly the behaviour wanted for a screened project queued behind a pending
   decision.

Neither was done here. The only write this run made was refreshing the queue summary
so the row itself reports that four hosts agree and the screen is not the bottleneck.
