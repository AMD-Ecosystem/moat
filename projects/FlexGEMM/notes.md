# FlexGEMM notes

## Intake screen (2026-08-27)

Upstream: https://github.com/JeffreyXiang/FlexGEMM — 148 stars, active
(`pushed_at` 2026-06-25), not archived. Author Jianfeng Xiang, the TRELLIS.2
author; MOAT already ported CuMesh from the same author.

**Recommendation: decline, reason `already-supported`.** Upstream already runs on
ROCm, by the author's design and by a merged upstream PR. Details below.

### Licence — MIT, tier 1

`utils/licenses.py check JeffreyXiang/FlexGEMM` -> `license=MIT tier=1`. Verified
by reading the file, not just GitHub's field: `LICENSE` is the verbatim MIT text,
"Copyright (c) 2025 Jianfeng Xiang (belljig@outlook.com)". `pyproject.toml` carries
`license = { file = "LICENSE" }`. Recorded as `status.json.license_spdx = "MIT"`.

Per-file checks, both clean:

- `utils/licenses.py scan-nvidia agent_space/FlexGEMM-screen` -> "no NVIDIA
  proprietary licence text". No `LICENSE_NVIDIA`-style file anywhere.
- No submodules and no vendored third-party directory. `.gitmodules` does not
  exist; the clone was taken with `--recurse-submodules` and produced nothing
  extra. The whole tree is 70-odd first-party files plus `assets/` and
  `autotune_cache.json`. The EnvGS failure mode (permissive top level over an
  unlicensed submodule) does not apply here.

Nothing about the licence is unresolved and nothing needs a clearance decision.

### The viability crux: NOT CUTLASS-bound, NOT tensor-core-shaped

The screening brief flagged a risk that FlexGEMM leans on CUTLASS or other
NVIDIA-specific GEMM infrastructure. It does not. The opposite is true, and it is
the central fact of this screen.

FlexGEMM's GEMM is **Triton**, not CUDA. The README calls it a "Triton-powered
GEMM backend" and "Triton-First Architecture"; `pyproject.toml` describes the
package as "A Cross-Platform Backend for High-Performance Sparse Convolutions" and
depends on `triton>=3.2.0` (`triton-windows` on Windows). All three algorithm
variants — Explicit, Implicit, Masked Implicit, each with a Split-K form — live in
`flex_gemm/kernels/triton/spconv/*.py`, about 3.3k lines of Triton.

A grep of the entire C++/CUDA tree for `cutlass|cublas|cusparse|cusolver|curand|
mma\.|wmma|__ldg|ptx|asm volatile|__shfl|cooperative_groups|cudnn|thrust|cub::|
__nv_|__half` returns **zero hits**. There is no NVIDIA library dependency, no
inline PTX, no warp-shuffle, no tensor-core intrinsic anywhere in the compiled
extension.

The CUDA extension is small and plain — 3,333 lines across
`flex_gemm/kernels/cuda/`, and it is not GEMM at all. It is index-and-bookkeeping
work: an open-addressing hashmap (`hash/`, 584 lines, `atomicCAS` on 32- and
64-bit keys), Z-order and Hilbert serialization (`serialize/`, 739 lines, pure
integer bit-twiddling), grid sample (`grid_sample/`, 357 lines), and sparse
neighbor-map construction (`spconv/`, ~1,400 lines). The only warp-level code in
the whole extension is one reduction in
`flex_gemm/kernels/cuda/spconv/migemm_neighmap_pp.cu`, and it already uses
`warpSize` as a runtime value rather than a hardcoded 32.

Build system is `setup.py` with `torch.utils.cpp_extension.CUDAExtension` — the
standard PyTorch path, which hipifies. There is no CMake, no bazel, no vendored
build.

So on portability grounds this project is easy. That is not the reason to decline;
the reason is that the work is already done.

### Duplicate effort — the port already exists upstream

Four independent, mutually confirming signals:

**1. A merged upstream ROCm PR.** `JeffreyXiang/FlexGEMM#18`, "Add AMD ROCm/HIP
support (2-line fix)" by ZJLi2013 (zhengjia), opened 2026-04-08 and **merged
2026-04-20** as `92ea4372`. Its two commits are `778e1dbd` ("Fix __syncwarp for
ROCm/HIP: AMD wavefront lockstep via wave_barrier") and `c8939d6a` ("Disable tf32
input_precision on AMD/ROCm (only ieee supported)"). Total diff: 8 added lines, 1
deleted, across two files. Both changes are present at current HEAD:

    flex_gemm/kernels/cuda/spconv/migemm_neighmap_pp.cu:9
      #if defined(__HIP_PLATFORM_AMD__)
        #if !defined(__syncwarp)
          #define __syncwarp(...) __builtin_amdgcn_wave_barrier()

    flex_gemm/kernels/triton/spconv/config.py:6
      allow_tf32 = not torch.version.hip

That is the entire delta the CUDA extension needed. An earlier identical PR (#17)
was closed and resubmitted as #18.

**2. `setup.py` has a first-class ROCm build path**, independent of PR #18:

    from torch.utils.cpp_extension import CUDAExtension, BuildExtension, IS_HIP_EXTENSION
    ...
    BUILD_TARGET = os.environ.get("BUILD_TARGET", "auto")   # "auto" | "cuda" | "rocm"
    if IS_HIP: archs = os.getenv("GPU_ARCHS", "native").split(";")
               cc_flag = [f"--offload-arch={arch}" for arch in archs]

`GPU_ARCHS` / `--offload-arch` is exactly the knob a ROCm build needs, and it is
already there with an explicit `BUILD_TARGET=rocm` override.

**3. The author designed for AMD from day one.** `flex_gemm/kernels/triton/
spconv/config.py` carries a dedicated `'hip'` platform config list *and* a
`'MI300X'` device config list, and both are tuned with ROCm-Triton-specific MFMA
knobs — `waves_per_eu`, `kpack`, `matrix_instr_nonkdim` — that have no CUDA
meaning. Checking the blob at every commit that ever touched that file shows the
`'hip'` block and the `MI300X` block present in the **first commit**
(`3061da44`, 2025-08-25), not retrofitted. `flex_gemm/kernels/triton/utils.py`
dispatches on `torch.version.hip` to return platform `'hip'`.

**4. Real MI300X autotuning is shipped in the repo.** `autotune_cache.json`
(2.7 MB, installed to `~/.flex_gemm/` by `setup.py`) has three top-level GPU keys:
`NVIDIA A100-SXM4-40GB`, `NVIDIA A100 80GB PCIe`, and **`AMD Instinct MI300X VF`**.
The MI300X entry holds roughly 18,500 tuned configurations across 24 kernels —
e.g. 5,452 entries for `sparse_submanifold_conv_bwd_weight_masked_implicit_gemm_
splitk_kernel`, 4,501 for the matching input kernel, 4,074 for the fwd splitk
kernel. Somebody ran the full autotuner on real MI300X hardware and committed the
results. That is beyond a compile-fix; it is performance work.

Searches that came up empty or non-authoritative:

- No `AMD-Ecosystem/FlexGEMM` and no `ROCm/FlexGEMM`. A global
  `FlexGEMM in:name` search returns no AMD-org repo. No other AMD team owns this.
- `grep -rniE 'amd|rocm|hip|gfx[0-9]' README* docs/` — the README has no "notable
  forks" section and no AMD mention; the ROCm support is in the code, not the docs.
  A docs-only grep would have missed all of this.
- Open upstream PRs: #31 (Windows/MSVC build, rwfsmith) and #29 (draft, CPU
  fallbacks, RisingRedstone). Neither is ROCm work, so there is no in-flight port
  to contribute to either.
- Upstream branches: `adaptation_for_ocnn`, `dev/all_triton`, `dev/serialize`,
  `dev/sparse_conv`, `dev/tilelang`, `main`, `uint64_hash`. None ROCm-related.
- `ATLAS-0321/FlexGEMM-rocm`, branch `rocm-port`, 1 commit ahead of upstream
  (2026-05-02): a single commit "Add ROCm/HIP kernel port for grid_sample, hash,
  serialize, spconv" that adds a **duplicated** `flex_gemm/kernels/hip/` tree —
  hipify-style `.hip` copies of every `.cu`, ~3,300 added lines, with the original
  `cuda/` tree left in place. It was never offered upstream (no PR), it is not by
  an AMD team, and its approach — forking the source tree rather than the two-line
  guard upstream took — is the one MOAT would not recommend. Not authoritative; it
  is superseded by the merged #18. Noting it only so nobody rediscovers it and
  mistakes it for prior art worth building on.
- `Cardboard-box-a/FlexGEMM-rocm` is not a GitHub fork (`parent` is null) and is
  not comparable against upstream; it appears to be a re-upload. No signal.

### Dependencies

Runtime imports across `flex_gemm/` are `torch`, `triton`, `filelock`, and stdlib
only. No MOAT project is a dependency, so `depends_on` stays empty. FlexGEMM is
itself a dependency of TRELLIS.2 (screened in parallel) — see below.

### Consequence for TRELLIS.2

TRELLIS.2 lists FlexGEMM in its dependency stack. Because upstream FlexGEMM
already supports ROCm and installs via `pip install .` with the ROCm path
auto-detected (`IS_HIP_EXTENSION`), **FlexGEMM is a satisfied dependency for
TRELLIS.2, not a blocker.** `moatlib`'s `DEP_SATISFIED_BY_DISPOSITION` treats
`already-supported` as satisfying a hard dependency, so if a person records this
decline the TRELLIS.2 selector will not wait on it. TRELLIS.2 should install
upstream FlexGEMM as-is.

The one caveat to hand to whoever ports TRELLIS.2: the MI300X autotune cache
covers submanifold-conv kernels, and the `'hip'` fallback config list is short (5
configs versus 8 for CUDA and 16-18 for the tuned A100/H100/MI300X device lists).
On a non-MI300X AMD part, or for the `sparse_conv` (non-submanifold) kernels,
Triton will autotune from that 5-config list at first run — correct, but a cold
start and possibly not peak. That is a tuning observation for the consumer, not a
port.

### Residual technical observation (not a reason to adopt)

`__builtin_amdgcn_wave_barrier()` is a scheduling barrier, not an execution or
memory barrier. In the `migemm_neighmap_pp.cu` reduction it stands in for
`__syncwarp()` between LDS read-modify-writes where lanes consume values written
by other lanes:

    if (iters_warpwise > 0 && threadIdx.x < warpSize) {
        for (int i = 0; i < iters_warpwise; i++) {
            int cur_len = warpSize >> i;
            buf[threadIdx.x] |= buf[threadIdx.x + cur_len];
            __syncwarp();
        }
    }

On CDNA this works in practice because a wavefront executes in lockstep, and the
`warpSize`-relative indexing is already wave64-correct. It is nonetheless the kind
of construct that can break under a future compiler that reorders LDS accesses
across the barrier; `__threadfence_block()` would be the stronger form. This is a
small upstream hardening opportunity, not a port, and not sufficient reason to
adopt the project. If someone wants it, it is a one-line drive-by PR, not a MOAT
pipeline run.

### Recommendation

Decline, `SKIP_REASON = already-supported`. Upstream supports ROCm today: the
enabling change is merged (#18), the build system has a ROCm target, the Triton
tuning tables include MI300X-specific MFMA configs from the first commit, and
~18,500 real MI300X autotune results ship in the repo. There is no port left to
write. Adopting it would produce a fork whose diff against upstream is, at most,
one defensive barrier change.

Recorded via `moatlib.py set-intake FlexGEMM decline --reason already-supported`.
Per the autonomy boundary this is a recommendation only — no `triage.py skip` was
run and `dispositions.json` was not touched. A person decides.

### Commands run

    python3 utils/licenses.py check JeffreyXiang/FlexGEMM
    git clone --depth 1 --recurse-submodules https://github.com/JeffreyXiang/FlexGEMM \
        agent_space/FlexGEMM-screen
    python3 utils/licenses.py scan-nvidia agent_space/FlexGEMM-screen
    gh pr list --repo JeffreyXiang/FlexGEMM --state all
    gh api repos/JeffreyXiang/FlexGEMM/branches --paginate --jq '.[].name'
    gh api repos/JeffreyXiang/FlexGEMM/forks
    gh api repos/JeffreyXiang/FlexGEMM/compare/main...ATLAS-0321:rocm-port

Read-only throughout; no fork created, no upstream write.

## Intake re-screen for the windows gate (2026-11-03, windows-gfx1151)

Why a re-screen: the 2026-08-27 decline (`already-supported`) was never ruled, and since then jeff ruled TRELLIS.2 `fork` (2026-10-29). TRELLIS.2 records `depends_on = ["CuMesh", "FlexGEMM"]` and owes all three required gates, `windows` included, so the question is no longer "does FlexGEMM run on ROCm" but "does it run on ROCm on every gate TRELLIS.2 must pass". `moatlib.py dep-doc-gaps` also lists FlexGEMM as a provider with no `## Install as a dependency` section.

**Recommendation: fork (reversing the August decline), scoped to the Windows ROCm build path plus validation on all three gates.** The Linux finding stands unchanged: upstream supports ROCm on Linux. Windows is not covered, and the dependency recipe cannot be written from a build nobody has run.

### Licence -- unchanged, MIT tier 1

`python3 utils/licenses.py check JeffreyXiang/FlexGEMM` -> `license=MIT tier=1, cleared to contribute`. Fresh clone at upstream `main` 6dd94a8 (2026-04-22): still no `.gitmodules`, no vendored third-party tree; LICENSE is the verbatim MIT text (Copyright (c) 2025 Jianfeng Xiang). `python3 utils/licenses.py scan-nvidia agent_space/FlexGEMM-screen` -> no NVIDIA proprietary licence text. Nothing unresolved.

### Upstream since August

- Default branch `main` has not moved since 6dd94a8 (2026-04-22); both PR #18 changes are still there. Repo not archived, 158 stars, last push 2026-06-25.
- `dev/all_triton` received "FlexGEMM 2.0" (PR #28, +20574/-66192, merged 2026-06-25 into `dev/all_triton`, not `main`). It still ships the same `flex_gemm/kernels/cuda/*.cu` extension and the same Windows dependency split. TRELLIS.2's `setup.sh` clones the default branch, so `main` is what a dependent installs.
- Open PRs: #31 (rwfsmith, 2026-08-21) "Fix Windows/MSVC build: template disambiguator and C++20 flags" edits the exact Windows `extra_compile_args` block in `setup.py` a Windows ROCm fix would touch -- coordinate, do not collide; #34 (explicit-GEMM blocking, no-Triton path); #29 (draft CPU fallbacks). None is ROCm or Windows-ROCm work.
- No `AMD-Ecosystem/FlexGEMM`, no ROCm-org repo.

### Why upstream does not cover the windows gate

1. **Build flags.** `setup.py`'s `platform.system() == "Windows"` branch gives `nvcc` the flags `-Xcompiler=/std:c++17 -Xcompiler=/EHsc -Xcompiler=/permissive- -Xcompiler=/Zc:__cplusplus`, unconditionally; the HIP branch only appends its `--offload-arch` list to them. On a ROCm torch, `torch.utils.cpp_extension` routes `extra_compile_args['nvcc']` straight to hipcc on Windows (`win_hip_flags` = `COMMON_HIPCC_FLAGS + COMMON_HIP_FLAGS + cflags + arch flags`, no filtering; read in the TheRock torch 2.12/ROCm 7.14 wheel here). `-Xcompiler=` is an nvcc option the clang driver does not accept; this is the fault class already recorded for gsplat/pytorch3d on Windows ("guard win32 device flags on `torch.version.hip`"). The host side additionally defaults to `cl.exe`, which cannot parse the HIP/c10-hip headers; MOAT's Windows recipe is `CC=CXX=clang-cl`. Expected delta: a `IS_HIP`-guarded Windows flag set in `setup.py`, a documented build env, and `GPU_ARCHS` instead of `native`. Not compiled here (intake does no builds), so the planner should confirm.
2. **Triton on Windows AMD.** All the GEMM work is Triton. `pyproject.toml` pulls `triton-windows` on Windows. TheRock publishes Windows torch wheels but zero Windows Triton wheels (counted on `whl-staging-multi-arch`, `v2/gfx1151`, `whl-multi-arch`: 282, 768, 2464 triton wheels, none `win`). `triton-windows` gained an AMD/TheRock HIP backend in woct0rdho/triton-windows#179 (merged 2025-12-30; README still calls AMD support "beginning"). MOAT precedent: ffpa-attn validated on windows-gfx1101 and windows-gfx1201 with `triton-windows 3.7.0.post26`, including a Triton-Windows autotuner picking a config over the 64 KB LDS limit. So viable, but unproven for FlexGEMM, and the shipped autotune cache has no entry for any RDNA part (A100 x2 and MI300X only); the `'hip'` fallback list has 5 configs tuned for CDNA MFMA knobs (`matrix_instr_nonkdim`, `kpack`), which the planner should check against RDNA3.5/wave32.
3. **No evidence on wave32 at all.** Upstream's AMD evidence is MI300X (CDNA, wave64). The deferred `flexgemm-wave-barrier-syncwarp` (jeff: defer, "revisit if FlexGEMM is adopted as a TRELLIS.2 dependency and a wave32 card shows a real fault") is exactly what wave32 validation would answer. The `migemm_neighmap_pp.cu` reduction uses `warpSize`, so it is wave-size-agnostic in indexing.

### Why fork rather than record `already-supported`

- `already-supported` would satisfy TRELLIS.2's dependency edge by disposition (`DEP_SATISFIED_BY_DISPOSITION`) and remove the record, which also silences `dep-doc-gaps`. TRELLIS.2's Windows validator would then meet the build break above with no FlexGEMM project to put the fix in, and the fix (a `setup.py` guard in an upstream repo) belongs on a FlexGEMM fork, not in TRELLIS.2.
- The `## Install as a dependency` section must be written from the provider's own validated build (DEPENDENCIES.md). Only a validator run produces that; on the fork path the validators write it for each gate.
- Upstream is active and merged #18 twelve days after it opened, so a small Windows-ROCm PR has a live destination. The Linux-side delta should be zero; the port is the Windows build plus wave32/windows evidence.

The fair case for the August answer: on Linux there is nothing to port, and if jeff is content for TRELLIS.2's windows gate to be waived or to carry a local FlexGEMM build recipe, `already-supported` is the cheaper ruling. A third option is to record `already-supported` now and let TRELLIS.2's Windows round raise a FlexGEMM intake again if the build fails; that defers the same work.

### Dependencies

Unchanged: runtime imports are torch, triton (`triton-windows` on Windows), filelock. No MOAT dependency; `depends_on` stays empty.

### Install as a dependency

Not written: no validated build exists yet (DEPENDENCIES.md forbids a guessed recipe). The first validator on each gate writes it. For Linux the expected shape is upstream's own `pip install <src> --no-build-isolation` with `GPU_ARCHS=<gfx>` against the ROCm torch.

### Commands run (re-screen)

    python3 utils/moatlib.py branch-sync --apply
    python3 utils/licenses.py check JeffreyXiang/FlexGEMM
    python3 utils/licenses.py scan-nvidia agent_space/FlexGEMM-screen
    git clone --recurse-submodules https://github.com/JeffreyXiang/FlexGEMM agent_space/FlexGEMM-screen
    git fetch --depth 50 origin dev/all_triton
    gh pr list --repo JeffreyXiang/FlexGEMM --state all
    gh pr view {20,21,28,31,34} --repo JeffreyXiang/FlexGEMM --json baseRefName,...
    gh pr diff 31 --repo JeffreyXiang/FlexGEMM
    gh pr view 179 --repo woct0rdho/triton-windows
    curl https://rocm.nightlies.amd.com/{whl-staging-multi-arch,v2/gfx1151,whl-multi-arch}/triton/
    python3 utils/moatlib.py dep-doc-gaps

Read-only; no fork created, no upstream write, no build or GPU work.
