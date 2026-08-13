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
supported, upstream-CI-tested path today. This is not `already-supported` for
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
