# TRELLIS.2 notes

Upstream: https://github.com/microsoft/TRELLIS.2
Intake screen: 2026-08-27, linux-gfx1100, read-only on a shallow scratch clone
(`agent_space/TRELLIS.2-screen`, `git clone --depth 1 --recurse-submodules`).

Microsoft's 4B-parameter image-to-3D generative model (10.9k stars, created
2025-11-26, last push 2026-07-10, not archived). Almost none of its CUDA lives in
this repository: the compute surface is in extension packages (flash-attn, CuMesh,
FlexGEMM, nvdiffrast, nvdiffrec), plus one vendored extension, `o-voxel/`.

## Licence

**MIT, tier 1.** `python3 utils/licenses.py check microsoft/TRELLIS.2` -> `license=MIT
tier=1`. Verified by reading the file rather than trusting the GitHub field: `LICENSE`
is the verbatim MIT text, "Copyright (c) Microsoft Corporation." (indented four
spaces, and the trailing newline is missing, which is cosmetic).

Per-file and vendored checks, all clean:

- `python3 utils/licenses.py scan-nvidia agent_space/TRELLIS.2-screen` ->
  "no NVIDIA proprietary licence text". No NVIDIA-licensed file ships in the tree.
- The only submodule is `o-voxel/third_party/eigen` (gitlab.com/libeigen/eigen).
  Eigen is MPL-2.0 per its `COPYING.README`, with some BSD/MINPACK/Apache parts.
  MPL-2.0 is tier 2 in `config/licenses.toml`; header-only build dependency.
- `o-voxel/` has **no LICENSE file of its own** -- the EnvGS-shaped trap -- but it is
  not ambiguous here. Every licensed header under `o-voxel/src/` carries
  `Copyright (C) 2025, Jianfeng XIANG <belljig@outlook.com>` /
  `Licensed under The MIT License [see LICENSE for details]`
  (`src/hash/api.h`, `src/rasterize/api.h`, `src/serialize/api.h`,
  `src/convert/api.h`, `src/io/api.h`). "[see LICENSE for details]" resolves to the
  repository's MIT LICENSE. Different copyright holder, same licence; consistent,
  so no unresolved-licensing stop-and-ask.
- Grepped `o-voxel/src` and `trellis2/` for third-party provenance markers
  (`inria|graphdeco|max-planck|mpii|Gaussian.Splatting|nvidia|derived from|adapted
  from|copyright`): nothing but two false hits on the word "based on" in ordinary
  comments. `o-voxel/src/rasterize/` looks superficially like the 3DGS rasterizer
  family (`BLOCK_X`/`BLOCK_Y`, `auxiliary.h`, cooperative groups) but carries no
  Inria/graphdeco attribution and is authored MIT.
- `assets/hdri/license.txt`: HDRIs are CC0 (Poly Haven).

### The README's "separate license terms" carve-out

README lines 316-320 name exactly two: **nvdiffrast** and **nvdiffrec**. Both are
*external* packages installed by `setup.sh` from their own repositories -- neither
ships any code in this tree. So they place no licence obligation on a TRELLIS.2 fork
and nothing about **contributing** to TRELLIS.2 is encumbered.

They do bear on **using** the result, which is the distinction intake is required to
draw. `NVlabs/nvdiffrast` already has a recorded MOAT disposition:
`skip / license-blocked`, "Wholly under the NVIDIA Source Code License, proprietary
non-commercial." nvdiffrec is being screened as its own project; note that
`setup.sh` installs *JeffreyXiang*'s fork at branch `renderutils`, not
`NVlabs/nvdiffrec` directly, so its licence has to be established on the fork.

Recorded `license_spdx = "MIT"`.

## Duplicate effort -- the decisive finding

**`microsoft/TRELLIS.2#155, "[ROCm] Add AMD GPU support for TRELLIS.2", is open.**
Author `andyluo7`, created 2026-04-27, **not a draft**, `MERGEABLE`, 7 files,
+505/-1. `updatedAt == createdAt`; zero comments, zero reviews -- four months with no
maintainer engagement at all. This is the FLAMEGPU2 case the role definition warns
about, and it was found only by listing open PRs: it is invisible to a docs grep
(README/docs mention neither AMD nor ROCm) and to a fork search.

What #155 actually contains, read from `gh pr diff 155`:

- `o-voxel/src/{hash/hash.cu, rasterize/rasterize.cu, serialize/hilbert.cu,
  serialize/z_order.cu}`: nothing but `#ifdef __HIP_PLATFORM_AMD__` /
  `#include <hip/hip_runtime.h>` include guards. About 20 lines total.
- `o-voxel/setup.py`: changes the HIP default arch from `native` to `gfx942`.
  (A regression for everyone not on MI300X; `native` was the better default.)
- `trellis2/renderers/nvdiffrast_rocm_adapter.py` (403 new lines) and
  `rocm_compat.py`: a pure-PyTorch stand-in for `dr.rasterize/interpolate/texture/
  antialias/DepthPeeler`, monkey-patched over `import nvdiffrast.torch`.

Its own "Known Limitations" are candid: no antialiasing and slower than the CUDA
rasterizer, **nvdiffrec PBR lighting stubbed, not ported**, flash-attention falling
back to SDPA, gfx942 only.

**Its companion is dead.** #155's body points at `JeffreyXiang/CuMesh#31` ("[ROCm] Add
HIP support for AMD Instinct GPUs") as the other half. That PR is **CLOSED** --
closed by `andyluo7` himself on 2026-06-19T19:29:18Z, one hour nineteen minutes after
Jeff Daily opened MOAT's `JeffreyXiang/CuMesh#36` at 2026-06-19T18:10:31Z (timeline
API confirms actor). He stood down on CuMesh in MOAT's favour, silently, and left the
TRELLIS.2 PR open. So #155 as written cannot work end to end: it needs a HIP CuMesh
that upstream CuMesh does not have and that MOAT is currently porting.

This is a **coordination question, not a race.** `andyluo7`'s GitHub profile lists no
company; affiliation unknown, but he has MI300X/ROCm 7.0.2 access and has already
deferred to MOAT once. Somebody should talk to him before a second parallel port of
the same four files exists. Per the role definition, the correct posture is to build
on #155 and credit it, not to compete with it.

Other duplicate-effort checks, all negative:

- `AMD-Ecosystem/TRELLIS.2` -- 404, no fork. Also 404 for `TRELLIS`, `o-voxel`,
  `O-Voxel`, `FlexGEMM`, `nvdiffrec`. `AMD-Ecosystem/CuMesh` exists (MOAT's).
- Upstream branches: `gh api repos/microsoft/TRELLIS.2/branches` has nothing matching
  `hip|rocm|amd`.
- README/docs grep for `amd|rocm|hip|gfx[0-9]`: no hits, no "notable forks" section.
- Forks owned by an org matching amd/rocm: only `OldProgramDeveloper/TRELLIS.2`
  (false positive on the substring), nothing real.
- Code-wide, upstream *already ships* ROCm awareness (see below), which is a
  different thing from an existing port.

## Upstream is ROCm-aware already

`setup.sh` detects the platform via `nvidia-smi` / `rocminfo` and branches on
`PLATFORM=hip`: it installs ROCm PyTorch (`--index-url .../rocm6.2.4`) and builds
`ROCm/flash-attention` at `v2.7.3-cktile` with `GPU_ARCHS=gfx942`. It prints
"Unsupported platform" for `--nvdiffrast` and `--nvdiffrec` on HIP -- upstream itself
already treats those two as the CUDA-only pieces.

`o-voxel/setup.py` reads `IS_HIP_EXTENSION` from `torch.utils.cpp_extension`, honours
`BUILD_TARGET=rocm`, and emits `--offload-arch=$GPU_ARCHS` (default `native`). The
build system needs no porting work at all.

## Viability

### The vendored o-voxel CUDA surface

`o-voxel/src` is 4580 lines total. The device code is small and unusually clean:

| file | lines | notes |
|---|---|---|
| `hash/hash.cu` + `hash.cuh` | 532 | Murmur3 hash map; 2 atomic call sites |
| `rasterize/rasterize.cu` + `auxiliary.h` + `config.h` | 685 | tile rasterizer, `BLOCK_X=BLOCK_Y=8` |
| `serialize/{api,hilbert,z_order}.cu` | 507 | Z-order / Hilbert curve encoding |
| `convert/{flexible_dual_grid,volumetic_attr}.cpp` | 1647 | **host-only**, Eigen + `unordered_map`, no device code |
| `io/*.cpp`, headers | ~700 | host-only |

So roughly **1350 lines of real `.cu`**, and the rest is CPU C++ that hipify will not
touch.

Fault classes visible on a read -- notable mostly for what is absent:

- **No warp intrinsics at all.** Grepping `__shfl|warpSize|__ballot|__activemask|
  __syncwarp|__any|__all_sync|0xffffffff` over `o-voxel/src` returns exactly one hit,
  and it is the comment `// 32 bit Murmur3 hash`. No wave64/wave32 hazard.
- **No cub, no thrust, no cuBLAS/cuSPARSE/cuRAND, no `mma.h`, no `cuda_fp16.h`.**
- `cooperative_groups.h` is included in three `.cu` files, but every use is
  `cg::this_grid().thread_rank()` or `cg::this_thread_block()` (8 call sites) --
  the trivially HIP-supported subset.
- Host API surface: one `cudaDeviceSynchronize()` in `rasterize/auxiliary.h:276`, one
  `__launch_bounds__(BLOCK_X * BLOCK_Y)` = 64 threads. No streams, no graphs, no raw
  `cudaMalloc`/`cudaMemcpy`; allocation goes through torch.
- `rasterize/config.h` hard-codes a 8x8=64-thread tile. 64 is a full wave64 wavefront
  and two wave32 wavefronts -- worth the porter's attention for occupancy tuning, but
  not a correctness fault.

This matches what #155 demonstrates empirically: header guards alone got it compiling
under hipcc. Genuinely small.

### CUDA outside o-voxel

None. `find . -name '*.cu' -o -name '*.cuh'` outside `o-voxel/` returns nothing. All
remaining GPU work is in pip/git dependencies.

### Dependencies

Hard, unconditional top-level imports on the **core** image-to-3D path
(`Trellis2ImageTo3DPipeline` -> `representations/mesh/base.py`, lines 4-5):

- **`cumesh`** -- MOAT project `CuMesh`, fork `AMD-Ecosystem/CuMesh`, stage `porting`,
  upstream PR JeffreyXiang/CuMesh#36 open. Also used for `cuBVH`
  (`o_voxel/postprocess.py:122`), and MOAT's cubvh work is already merged upstream.
- **`flex_gemm`** -- MOAT project `FlexGEMM`, scaffolded and being screened in
  parallel. It is also the default sparse-conv backend
  (`trellis2/modules/sparse/config.py: CONV = 'flex_gemm'`). Triton-based, so ROCm is
  plausible; #155 claims "compiles on ROCm (no changes needed)".

Recorded as `depends_on = ["CuMesh", "FlexGEMM"]`.

Soft / substitutable:

- **flash-attn** -- `attention/config.py: BACKEND = 'flash_attn'`, but `ATTN_BACKEND`
  accepts `sdpa` and `naive`, and `setup.sh` already builds `ROCm/flash-attention`
  for HIP. Not a blocker.
- **spconv / torchsparse** -- alternative `SPARSE_CONV_BACKEND` values; both are
  themselves CUDA extensions, so they are not an escape hatch from FlexGEMM.

### nvdiffrast: deliberately NOT put in depends_on

This is the one judgement in the screen a person may want to overturn, so it is
spelled out rather than encoded.

As the code stands nvdiffrast is a **hard import** of the core pipeline, transitively:
`o_voxel/__init__.py` imports `postprocess` unconditionally, and
`o_voxel/postprocess.py:10` does `import nvdiffrast.torch as dr` at module top; the
core VAE (`trellis2/models/sc_vaes/fdg_vae.py:20`) imports from `o_voxel.convert`.
So today, `import o_voxel` fails without nvdiffrast installed.

But that specific hardness is a one-line laziness fix, and functionally nvdiffrast is
confined to rasterisation for (a) preview/video rendering
(`renderers/mesh_renderer.py`, `pbr_mesh_renderer.py` -- all already lazy, inside
functions), (b) the texturing pipeline (`pipelines/trellis2_texturing.py:13`), and
(c) UV texture baking in `o_voxel.postprocess.to_glb`, which is only three call sites
(`postprocess.py:230,237,249`: `dr.RasterizeCudaContext()`, `dr.rasterize`,
`dr.interpolate`).

That third one is the substantive loss. `to_glb` is step 5 of `example.py` -- the
headline deliverable is a textured `.glb`, and without a rasteriser there is no baked
texture atlas. So a MOAT port bounded by the nvdiffrast disposition delivers
"generation runs and produces geometry" but not the product, unless somebody writes a
rasteriser replacement. #155 wrote one (403 lines of pure PyTorch) and calls it
slower and non-antialiased.

`nvdiffrast` is **not** listed in `depends_on` because it is a runtime Python import,
not a build/link dependency, and because `nvlabs/nvdiffrast` carries a
`license-blocked` disposition, which is in `DEP_DOOMED_BY_DISPOSITION`. Adding it
would mark TRELLIS.2 dep-doomed and remove it from the selector -- which is a decision
about the project, and decisions are a person's. If the reviewer judges a textured
`.glb` to be the whole point, adding it is the right move and this note is the case
for doing so.

### Upstream will almost certainly not merge anything

`microsoft/TRELLIS.2` has merged **two PRs in its entire history**, and neither was
from a human: #166 by `app/copilot-swe-agent` and #2 by
`app/microsoft-github-policy-service`. Meanwhile 23 PRs are open (several trivial:
"Fix installation command for o-voxel", "fix windows build", DINOv3/transformers-5.x
compatibility fixes duplicated across four separate PRs) and 147 issues. Last push
2026-07-10.

Per the role definition this is a note, not a block -- the same logic as an archived
upstream. The realistic destination for this port is an `AMD-Ecosystem/TRELLIS.2`
fork that people find and build from, not a merged upstream PR. **Nobody should wait
for a PR here to land.** It also cuts against #155: "contribute to the existing PR
instead" is not obviously a better outcome when the existing PR has sat unread for
four months.

## Recommendation: fork (a person decides)

Recorded via `moatlib.py set-intake TRELLIS.2 fork --viable yes`.

The case for taking it up:

- Tier-1 MIT, clean top-level, clean vendored, clean per-file, no NVIDIA text.
- The in-repo porting work is genuinely small: ~1350 lines of `.cu` with no warp
  intrinsics, no cub/thrust, and a build system that already understands ROCm. #155
  proves header guards suffice to compile.
- MOAT is uniquely placed to finish it. The two hard dependencies are MOAT projects:
  CuMesh is ours and in flight, FlexGEMM is in screening. #155's CuMesh half is dead
  precisely because its author yielded CuMesh to MOAT.
- It is the end-to-end integration test for the CuMesh and FlexGEMM ports -- a
  10.9k-star Microsoft flagship actually running on AMD is worth more as
  demonstration than the four files are as code.

The case a reviewer might prefer for declining, stated fairly:

- `duplicate` -- #155 is open, non-draft, and covers the same four files. If the
  answer is "let the existing contributor finish", that is legitimate and this screen
  will not have wasted anything.
- Not `already-supported`: #155 is a stale proof of concept with a dead companion
  PR, a stubbed nvdiffrec, an admitted degraded rasteriser, and gfx942-only testing.
  Nothing here is already supported.

Two conditions that should be attached to a fork decision:

1. **Reach out to `andyluo7` on #155 before porting** (a human's GitHub write). Credit
   the existing work and build on it. A second parallel port of the same include
   guards would be the FLAMEGPU2 mistake repeated with the evidence already in hand.
2. **Scope the deliverable explicitly around nvdiffrast.** Either accept
   "geometry generation on AMD, no textured `.glb` export", or budget separately for a
   rasteriser replacement. Do not let a porter discover this at build time.

Also worth doing regardless: keep `GPU_ARCHS` defaulting to `native` rather than
copying #155's `gfx942`, so the build is not silently MI300X-only.
