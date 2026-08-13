# diff-surfel-tracing notes

A dependency fork: the tracer EnvGS's reflection path renders through.

## Why the platform row is empty

This has no test suite of its own. The code is exercised only through the project
that consumes it, so a GPU run against this repository alone would prove nothing.
The empty row is accurate, not a gap in the record.

The validation lives with **EnvGS**, `completed` on linux-gfx1100, linux-gfx90a, windows-gfx1101, windows-gfx1201.

## Port state

The `moat-port` branch predates this project being tracked here, so the port exists
but its provenance was not recorded: no plan, no dated validation entry, no note of
which commit was tested. Treat it as real work of unverified state rather than as a
validated port.

## Intake screen (2026-08-13, linux-gfx1100)

Recommendation: **fork** -- i.e. take it up. The fork already exists, so the state
went to `screened` rather than `awaiting-fork`. Recorded with `set-intake`.

This is an unusual intake: it is not a fresh candidate. The port is already written
and GPU-validated, but it was done and recorded entirely under EnvGS's project
record. What is missing is this project's own record and its own route upstream --
`xbillowy/diff-surfel-tracing` is a different repository from `zju3dv/EnvGS`, so
the work can only reach its maintainer as a separate PR. That is what taking this
up buys; it is not a request to port anything from scratch.

### Licence: MIT (tier 1), verified by reading the file

`licenses.py check` returns UNPARSED -- GitHub reports NOASSERTION / "Other" for
this repo. That is a misparse, not a restriction. `LICENSE` at upstream HEAD is the
verbatim MIT text, copyright 2024 3D Vision Group, State Key Lab of CAD&CG,
Zhejiang University. `license_spdx: "MIT"` in status.json was already correct and
is confirmed independently, not inherited on trust.

Worth knowing: upstream had NO licence at all until commit `ef6f24be`
("add: license", 2025-10-14). The `moat-port` branch is cut from a base BEFORE that
commit, so the branch's own tree contains no LICENSE file. Contribution today is
judged against upstream today, which is cleanly MIT.

Scope reminder: MIT clears CONTRIBUTING upstream. It says nothing about the
vendored third-party content below.

### NVIDIA proprietary headers in the OptiX submodule -- needs a person's decision

`.gitmodules` points `third_party/optix` at `NVIDIA/optix-dev`. Of its 14 headers,
**9 carry NVIDIA proprietary licence text** and 5 are BSD-3-Clause;
`third_party/optix/LICENSE.txt` is the NVIDIA DesignWorks SDK EULA. Per the intake
role, NVIDIA-proprietary files need a decision before proceeding, so this is
flagged rather than resolved here.

Mitigating fact for whoever rules on it: the port REPLACES the OptiX path with
HIPRT and does not modify, redistribute, or depend on those headers. They are
upstream's pre-existing submodule, not something the port adds.

**Tooling gap found while checking this.** `licenses.py scan-nvidia` reported the
tree clean and it is not. The markers in `config/licenses.toml` are matched with
`grep -rlF` (case-SENSITIVE, utils/licenses.py:78). The marker reads
`NVIDIA CORPORATION and its licensors retain all intellectual property`; the OptiX
headers read `NVIDIA Corporation ...`. Adding `-i` turns 0 hits into 9. This is a
control-plane bug affecting every screen run to date, so it was NOT fixed from this
port branch -- it needs its own change on the trunk and a re-run of past screens.

### Duplicate effort: none outside MOAT

- No `diff-surfel*` repo in AMD-Ecosystem or ROCm other than our own fork.
- Upstream README has zero matches for amd/rocm/hip/gfx/radeon -- no notable-forks
  link, no existing platform port.
- Upstream forks are `wasahaiah`, `rhombus19`, `piotrmwojcik`, and ours; none are
  AMD ports.
- No entry in `data/candidates.json`, no disposition, no opt-out.
- Upstream has one closed PR (#5, unrelated: setup.py compilation) and no open PRs.

The only existing AMD work is MOAT's own, via EnvGS Stage 2.

### Viability: yes

Genuinely GPU code: `optix_tracer/{forward.cu,backward.cu}` (~1500 lines) plus a
torch `CUDAExtension` (`setup.py`), so `ext_type` is a torch-extension rather than
the recorded `unknown` (intake has no command to set that field).

It is an OptiX ray-tracing pipeline, not ordinary CUDA: `optixAccelBuild` over
`OPTIX_BUILD_INPUT_TYPE_TRIANGLES`, `optixTrace`, module/pipeline/SBT plumbing, and
`__raygen__`/`__anyhit__` programs. OptiX has no ROCm equivalent, so this was a
rewrite onto HIPRT rather than a hipify -- and that rewrite is already done and
validated (EnvGS notes, "Stage 2 port: OptiX -> HIPRT", gfx90a PASS, including a
genuine uninitialized-`cutoff` bug and two return-type UB bugs latent in the
upstream OptiX sources).

Upstream is alive but quiet: not archived, not disabled, 58 stars, 5 forks, last
push 2025-10-14 (~10 months). A PR has a real destination.

`depends_on` stays empty. It is standalone -- the README references
diff-surfel-rasterization only for API similarity, and nothing imports it. The
relationship runs the other way: EnvGS consumes this. The sibling candidate
`diff-surfel-rasterizations` is being screened concurrently by another agent; its
records were not touched here.

### What a planner inherits (not intake's call, but it decides the effort)

1. **Vendored HIPRT/Orochi is the real obstacle to a PR.** `moat-port` adds 110
   files under `third_party/hiprt/`, including a **1.9 MB prebuilt
   `hiprtc0604.dll`** and two more win64 DLLs committed as real binaries. The only
   licence file anywhere under that tree is cuew's; HIPRT's and Orochi's own
   licence files were not vendored with the code. Asking an upstream maintainer to
   take a vendored SDK plus binary blobs is a hard sell, and the missing licence
   files must be resolved regardless.
2. **The branch is 3 upstream commits stale**, including `e0016a27`
   ("update: latest version"), `3b97d5d3`, and the `ef6f24be` licence commit. It
   needs rebasing onto current upstream before any PR.
3. No plan.md, no surface.json, and no validation recorded against this project's
   own `head_sha` (still null). The existing note above explains the empty platform
   row: there is no standalone test suite, so evidence arrives through EnvGS.
