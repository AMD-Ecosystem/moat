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

## Planning (2026-11-20, linux-gfx942 fleet job)

**Decisions for Jeff: none needed before the porter starts.** The plan ports the whole project: the CUDA extension, every Triton kernel family, the Python ops, every test script, both examples, and the FlexGEMM columns of the benchmark scripts. Nothing is deferred and no app, GUI or library layer is cut. Two points for your information, neither needing a ruling:

- The deferred `flexgemm-wave-barrier-syncwarp` (you ruled defer on 2026-10-09: revisit if FlexGEMM becomes a TRELLIS.2 dependency AND a wave32 card shows a real fault) now meets only the first half of that condition. The plan keeps your ruling and makes no change. The validators exercise the kernel on wave32 and wave64, and a fault there brings the item back to you. New evidence for that discussion: torch builds extensions with `-DHIP_ENABLE_WARP_SYNC_BUILTINS=1`, so HIP's own `__syncwarp()` (wavefront release fence, `wave_barrier`, acquire fence) is available. Upstream's `#define __syncwarp(...) __builtin_amdgcn_wave_barrier()` replaces it with the weaker form.
- The test scripts' third-party comparison columns (traveller59 spconv, torchsparse, fvdb, warpconvnet; all CUDA-only) are dropped by a validator-side driver, or spconv is satisfied by `spconv-triton`. Those columns are not FlexGEMM code, and the correctness reference is FlexGEMM's own torch-matmul `EXPLICIT_GEMM` plus a dense `torch.nn.Conv3d`. I do not count this as a scope cut. If you do, say so before the validator runs.

### Findings

- **Upstream `main` does not build against current torch on any platform.** `setup.py` forces `-std=c++17`, and torch >= 2.14 headers `#error` below C++20 (pytorch#178150, first in v2.14.0). I confirmed it here against the image's torch 2.16.0a0 (ROCm 7.16) with a preprocessor-only check, no GPU and no build: `echo '#include <torch/extension.h>' | g++ -std=c++17 -E -x c++ -I<torch>/include -I<torch>/include/torch/csrc/api/include -I<python include> -` gives `all.h:9:2: error: #error C++20 or later compatible compiler is required to use PyTorch.`, and `-std=c++20` is clean. `BuildExtension` appends the right standard itself when none is given, so the fix is to drop the explicit flags (a shared fix, no `[ROCm]` prefix). PR #18's MI300X evidence predates this (torch 2.6).
- The Windows+HIP breakage the intake predicted holds after reading the torch source. `win_hip_flags` forwards `extra_compile_args['nvcc']` unfiltered to hipcc, so the `-Xcompiler=/...` flags reach clang. `GPU_ARCHS` defaults to `native`, which fails on Windows hipcc and also suppresses `PYTORCH_ROCM_ARCH`.
- The CUDA extension needs no device-code change. Dynamic LDS is at most 32 KB (the 32-bit gray code caps V at 32), `atomicCAS` is 32/64-bit only, there are no libraries or intrinsics beyond `__ffs`/`__popc`/`warpSize`, and every `torch::empty` output is fully written.
- The Triton risk on RDNA is the fp32 `'hip'` config `B1=256,B2=256,BK=16,num_stages=2` (64 KB of operand tiles, the RDNA LDS limit). FlexGEMM's autotuner does not catch `OutOfResources`. fp16 is 32 KB.
- Upstream activity since the re-screen: PR #35 (2026-11-12, Triton 3.9 fixes on `dev/all_triton`, not `main`); issue #19 (Windows nvcc `data_ptr<T>` template disambiguator, also fixed by open PR #31; clang does not need it).

Full analysis: `plan.md`. Machine-checked surface: `surface.json` (24 components; `surface.py check` passes). `ext_type` set to `torch-extension` via `moatlib.py set-ext-type`.

### Commands run (planning)

    bash utils/orient.sh                       # selected FlexGEMM planner, linux-gfx942
    python3 utils/moatlib.py set-state FlexGEMM linux-gfx942 planning --agent planner
    git merge origin/main                      # branch-sync refused on a data/deferred.json conflict;
                                               # resolved by keeping main's file plus this branch's
                                               # moatlib-set-license-subcommand entry
    git clone https://github.com/AMD-Ecosystem/FlexGEMM projects/FlexGEMM/src   # + upstream remote, read-only
    gh pr view 18|31|35 --repo JeffreyXiang/FlexGEMM; gh pr diff 31|35 ...; gh issue list/view 14,19 ...
    git -C /var/lib/jenkins/pytorch log -S'C++20 or later compatible compiler' -- torch/csrc/api/include/torch/all.h
    g++ -std=c++17|c++20 -E (torch/extension.h)  # preprocessor only
    python3 utils/surface.py generate FlexGEMM && python3 utils/surface.py check FlexGEMM
    python3 utils/moatlib.py set-ext-type FlexGEMM torch-extension

## Porting (2026-11-20, linux-gfx942 fleet job, AMD Instinct MI300X)

**Questions for Jeff (none blocks the reviewer; the port proceeds without them):**

1. **`examples/spconv.py` is broken upstream on every platform.** It unpacks `out_feats, neignbor_cache = sparse_conv3d(...)`, but `sparse_conv3d` returns three values `(feats, coords, neighbor_cache)` since the API change, so it raises `ValueError: too many values to unpack (expected 2)` before any GPU work. This is pure Python and not ROCm-specific, so the port leaves it alone (smallest port). With the unpack corrected (agent-side copy, no fork edit) it runs on MI300X with finite output and gradients. Options: (a) leave it (current); (b) add a separate unprefixed commit fixing the example in this port; (c) mention it to upstream separately. (b) or (c) is your call, since it widens the upstream-visible change.
2. **The HIP launch-size fix is a config prune, not a kernel change.** Commit bcf7c6f drops, on HIP only, the grid_sample backward configs whose launch would reach 2^32 work-items (details below). The alternative is to change `indice_weighed_sum_bwd_input` to a 2D grid or a persistent loop, which also covers inputs where every config is too large (roughly M*V*C at or above 2^34, i.e. tens of GB of indices and gradients), but edits a shared Triton kernel that upstream tunes. I chose the prune as the smaller change. No ruling is needed unless you prefer the kernel change.

### What was done

Fork `AMD-Ecosystem/FlexGEMM`, branch `moat-port` cut from `main` 6dd94a8 (upstream head). Four commits, head **bcf7c6f5eee9f780a0ecdff4de7cecca8f73c728**, pushed:

| Commit | Title | Change |
|---|---|---|
| 06873dd | setup.py: let torch choose the C++ standard | drop `-std=c++17`, `/std:c++17`, `-Xcompiler=/std:c++17` (shared fix, plan item 1) |
| 078c4ab | [ROCm] setup.py: Windows HIP flags and default target | Windows+HIP: hipcc gets `["-O3"] + cc_flag` only; `GPU_ARCHS` default is empty on Windows so torch uses `PYTORCH_ROCM_ARCH`; `.gitignore` gains `flex_gemm/kernels/hip/` (hipify output written in-tree by every ROCm build; upstream left it untracked) |
| d45c5e2 | [ROCm] README: describe the AMD GPU build | short "AMD GPUs (ROCm)" paragraph under Installation (GPU_ARCHS, Windows triton-windows + VS prompt). No device-selection env vars. |
| bcf7c6f | [ROCm] grid_sample: skip bwd configs past the HIP launch size limit | `early_config_prune` on `indice_weighed_sum_bwd_input_kernel`, HIP only |

`/openmp` in the Windows `cxx` list was kept (plan risk 4: drop only if the Windows link proves it must go). No `.cu`/`.cuh`, spconv Triton kernel, op, test or autotune-cache change. The deferred `flexgemm-wave-barrier-syncwarp` is untouched per your ruling; every neighbour-cache test (which runs `reduce_code_kernel`) passes bit-exact on wave64.

Flag-list diff, upstream vs port, from `agent_space/flags.py` (monkeypatched `setuptools.setup`, `platform.system` patched), GPU_ARCHS unset:

```
BUILD_TARGET=cuda Linux:   cxx/nvcc lose only -std=c++17
BUILD_TARGET=cuda Windows: cxx loses /std:c++17; nvcc loses -std=c++17 and -Xcompiler=/std:c++17 (all else identical)
BUILD_TARGET=rocm Linux:   cxx/nvcc lose only -std=c++17 (--offload-arch=native kept)
BUILD_TARGET=rocm Windows: nvcc ['-O3', '-std=c++17', '-Xcompiler=/std:c++17', '-Xcompiler=/EHsc', '-Xcompiler=/permissive-', '-Xcompiler=/Zc:__cplusplus', '--offload-arch=native'] -> ['-O3']
```

### Environment (job-local, nothing in the image's shared paths)

- TheRock ROCm **10.2.0a20261022** wheels from `https://nightly.repo.amd.com/rocm/whl-next/` installed into `$FLEET_VENV` (Python 3.11): `torch==2.16.0a0+rocm10.2.0a20261022`, `amd-torch-device-gfx942`, `triton==3.9.0+gitfc8d31a5.rocm10.2.0a20261022`, `rocm`, `rocm-sdk-core`, `rocm-sdk-devel`, `rocm-sdk-libraries`, `rocm-sdk-device-gfx942` all at that version; `python -m rocm_sdk init`, `python -m rocm_sdk test` = 27 OK. 10.2.0a20261022 is the newest 10.2 build on that index (later nightlies are 10.3). The job venv's base-env symlinks to ROCm 10.1 / Triton were removed from the job's own `venv-base-links` so only 10.2 is importable.
- Build env: `PATH=$FLEET_VENV/bin:$PATH`, `ROCM_HOME=ROCM_PATH=HIP_PATH=$(python -m rocm_sdk path --root)`, `GPU_ARCHS=gfx942`, `HOME` pointed at a job-local directory (setup.py and the autotuner write `~/.flex_gemm/autotune_cache.json`; an accidental write to the shared home during the first baseline build was removed), `TRITON_CACHE_DIR` job-local, `HIP_VISIBLE_DEVICES=2` (dispatcher pin), `ulimit -c 0`.
- Extra test deps in the venv: `tqdm`, `matplotlib`.

### Build

```
# baseline, upstream main 6dd94a8:
pip install <src> --no-build-isolation --no-deps
  -> torch/csrc/api/include/torch/all.h:9:2: error: C++20 or later compatible compiler is required to use PyTorch.
# port:
utils/timeit.sh FlexGEMM compile -- $FLEET_VENV/bin/python -m pip install <src> --no-build-isolation --no-deps
  -> Successfully installed flex_gemm-1.0.0, 0 warnings; hipcc gets -std=c++20 --offload-arch=gfx942 -DHIP_ENABLE_WARP_SYNC_BUILTINS=1 -O3
```

### CUDA no-regression compile (this host has no nvcc; job-local pip toolchain)

`pip install --target <job>/cudapkgs --index-url https://download.pytorch.org/whl/cu130 torch==2.14.1+cu130 nvidia-cuda-nvcc==13.0.* nvidia-cuda-cccl==13.0.* nvidia-cuda-crt==13.0.* nvidia-nvvm==13.0.*`; `CUDA_HOME=<cudapkgs>/nvidia/cu13` (job-local `lib64 -> lib` and `libcudart.so -> libcudart.so.13` symlinks added for the link), `PYTHONPATH=<cudapkgs>`, `BUILD_TARGET=cuda TORCH_CUDA_ARCH_LIST="8.0;9.0" python setup.py build_ext` on `git archive` exports:

- upstream `main`: **fails**, same `#error C++20 or later compatible compiler is required` (torch 2.14.1).
- port bcf7c6f: **builds**, nvcc 13.0.88, 6 `.cu` with `-std=c++20`, `sm_80` + `sm_90`, `cuda.cpython-311-x86_64-linux-gnu.so` linked. Warnings only from torch headers (`#3189-D "module" is parsed as an identifier`, 126x) and 2x `#177-D variable "b" declared but never referenced` in upstream code.
- Not covered: an older CUDA torch (< 2.14, where BuildExtension appends c++17) and Windows nvcc. The behaviour follows from torch's `append_std17_if_no_std_present`; the mi355 host can confirm with real nvcc before presenting.

### GPU tests (MI300X gfx942, GPU 2, head bcf7c6f content)

Asserting scripts, run unchanged from `tests/`:

| Script | Result |
|---|---|
| `hashmap.py` | pass (RES 1..1024, 4 dtype pairs) |
| `serialize.py` | pass (z_order + hilbert, cpu + cuda, RES up to 2048) |
| `spconv/out_coords.py` | pass (HASHMAP + EXPAND_UNIQUE vs torch, RES up to 2048) |
| `spconv/neighbor_cache.py` | pass (`torch.equal` for egemm/igemm/migemm caches incl. `reduce_code_kernel`, RES up to 512) |
| `submconv/neighbor_cache.py` | pass (egemm/igemm/igemmk/migemm/migemmk, RES up to 512) |
| `grid_sample.py` | **failed at 078c4ab** in "Trilinear Bwd" RES=2048,C=8 (`Triton Error [HIP]: Code: 1, Messsage: invalid argument`); **passes at bcf7c6f**: all four tables err_max 0 per mille, err_mean 0 |

Conv correctness tables (fp16), run through the plan's B.2 driver (`agent_space/run_noext.py`, text below; it comments out the `import spconv...`/torchsparse/fvdb/warpconvnet lines and their `kernel_functions` entries, no fork edit). Pass criterion from the plan: each Triton algorithm's err_max <= 2x dense, err_mean <= 5 per mille.

- `spconv/fwd.py` (14 rows, RES 8..512): igemm/igemmk/migemm/migemmk err_max 2 per mille on every row (dense column 2 per mille; one row 1), err_mean 0 everywhere. Dense is N/A from RES=256 by the script's design. **Pass.** 483 s cold.
- `submconv/fwd.py` (7 rows, RES 8..512): all four algorithms err_max 2, err_mean 0. **Pass.** 68 s.
- `spconv/bwd.py` (14 rows): igemm/igemmk/migemm/migemmk err_mean 0 on every row; err_max 1-4 per mille up to RES=64, then 10-16 (RES=128 K=3), 9-23 (RES=256 K=3), 8-44 (RES=512 K=3; igemm 44, migemm 26, the split-K variants 8). The largest values are on the biggest K=3 weight-gradient reductions and are smallest for split-K, which fits fp16 accumulation-order effects rather than a fault; there is no CUDA table to compare. The torch dense `Conv3d` column, against the same `EXPLICIT_GEMM` reference, reads 175-4164 per mille err_max (err_mean up to 11), far worse than every Triton algorithm. The plan's criterion (err_mean <= 5 everywhere; the 2x-dense err_max rule applies to fwd) is met. **Pass.** 1359 s cold (autotuning dominates).
- `submconv/bwd.py` (7 rows, RES 8..512): egemm err_max 0; igemm/igemmk/migemm/migemmk err_max 1-13 (largest igemm 13, migemm 12 at RES=512; split-K 4), err_mean 0 everywhere. **Pass.** 382 s.
- `examples/submconv.py`: runs to completion (rc 0, 99 s cold).
- `submconv/inf_ablation.py` (rc 0, 315 s) and `submconv/train_ablation.py` (rc 0, 631 s), FlexGEMM-only, run unchanged: all rows RES 8..512 complete. Inference at RES=512,C=128: egemm 29.0 ms, igemm 4.3, igemm_splitk 4.8, migemm 3.0, migemm_splitk 2.9. Training at RES=256,C=256: egemm 36.9 ms, igemm 14.2, migemm_splitk 7.0. These are the RDNA-tuning baseline the plan asks for, not a gate.
- `submconv/inf_benchmark.py` and `train_benchmark.py` through the driver (torchsparse/fvdb/warpconvnet/spconv columns removed): flex_gemm column completes (rc 0; 5-6 s because the autotune cache was already warm). Inference RES=512,C=128 3.77 ms; training RES=256,C=256 7.35 ms.
- The ablation scripts write `conv_*.png` plots into `tests/submconv/`; moved out of the fork tree after the run (untracked test output, not a source change).

`tests/utils.py get_device_max_flops` has no MI300X entry, so the Tflops tables read N/A (cosmetic, not a port issue).

Driver `agent_space/run_noext.py`:

```
import os, re, sys
path = os.path.abspath(sys.argv[1])
third = ("spconv", "torchsparse", "fvdb", "warpconvnet")
out = []
for line in open(path).read().splitlines():
    s = line.strip()
    if re.match(r"(import|from)\s+(%s)\b" % "|".join(third), s) or re.match(r"'(%s)'\s*:\s*\(" % "|".join(third), s):
        out.append("# [removed] " + line); continue
    out.append(line)
sys.argv = [path] + sys.argv[2:]
sys.path.insert(0, os.path.dirname(path))
exec(compile("\n".join(out) + "\n", path, "exec"), {"__name__": "__main__", "__file__": path})
```

### The HIP launch-size fault (commit bcf7c6f)

`indice_weighed_sum_bwd_input_kernel` uses a 1D grid of `cdiv(C,BK) * cdiv(M,BM) * V` programs. At RES=2048 trilinear (M=4,194,304, C=8, V=8) the config `BM=1, BK=64, num_warps=2` gives 33,554,432 programs x 128 threads = exactly 2^32 work-items. Instrumented run (`agent_space/repro_gs.py`) listed every config's size; only that one reaches 2^32 and the autotuner's benchmark of it raised the error. Standalone HIP check on this GPU (`agent_space/lim/lim.hip`):

```
hipLaunchKernel       grid=33554431 block=128 total=4294967168 -> hipSuccess, last block ran
hipLaunchKernel       grid=33554432 block=128 total=4294967296 -> hipErrorInvalidConfiguration, nothing ran
hipModuleLaunchKernel grid=33554431 block=128                  -> hipSuccess
hipModuleLaunchKernel grid=33554432 block=128                  -> hipErrorInvalidValue (code 1, Triton's message)
(same at block=256: 16777215 ok, 16777216 rejected)
```

CUDA allows 2^31-1 blocks in x regardless of block size, so the config is legal there and the prune returns the full list off HIP. Warp size comes from `triton.runtime.driver.active.get_current_target().warp_size`, so wave32 parts keep twice the headroom. `examples/submconv.py` passes; `examples/spconv.py` is the upstream bug in question 1.

### Not done in this round (validator / later work, per plan)

- wave32 and Windows validation (other hosts); the Windows half of commits 078c4ab/d45c5e2 has not been compiled anywhere yet.
- `## Install as a dependency` is written by each validator from its validated build (plan E).
- RDNA autotune performance (not a gate).

## Lesson for cuda-to-rocm: HIP rejects a launch of 2^32 or more work-items per grid dimension

Promoted to cuda-to-rocm/references/platform-limits.md in 0bd85094.

Belongs in `platform-limits.md`. (Rewritten in porting round 2 after review finding 4; the first version advised folding the grid into 2D without the CUDA caveat.)

Symptom: a kernel that runs on CUDA fails to launch on AMD only at large sizes; Triton reports `Triton Error [HIP]: Code: 1, Messsage: invalid argument` (often from inside autotuning, sometimes wrapped as `SystemError: <built-in method __reversed__ of list object ...> returned a result with an exception set`); `hipLaunchKernel` returns `hipErrorInvalidConfiguration`, `hipModuleLaunchKernel` `hipErrorInvalidValue`, and nothing runs.

Cause: HIP counts a launch in work-items per dimension: gridDim.x * blockDim.x must stay below 2^32, and likewise y and z. CUDA limits block counts instead: 2^31-1 blocks in x, but only 65,535 in y and z. Measurements, ROCm 10.2:

- x, MI300X (gfx942): grid 33,554,431 x 128 runs, 33,554,432 x 128 is rejected (also 16,777,215 / 16,777,216 at block 256), with both launch APIs.
- y and z, MI355X (gfx950): `grid=(1,4194303) x block=(1,1024)` runs, `(1,4194304) x (1,1024)` is rejected; `grid=(1,1,67108864) x (1,1,64)` is rejected. `hipDeviceProp.maxGridSize` reports `(2147483647, 65535, 65535)`, but y is not capped at 65,535 blocks: `grid=(1,65536) x (1,128)` and `grid=(1,33554432) x (128,1)` both run. The y and z limit is also a work-item count.
- A launch whose total over all dimensions exceeds 2^32 is fine: MI300X ran `grid=(4129777, 65) x 128` (about 2^35 work-items) correctly.

Flat 1D grids with small tiles and an extra multiplier (a neighbour count, a batch) reach the limit first.

Fix: keep the large program count on axis 0 and give it no more than (2^32 - 1) / blockDim.x programs there; put the overflow on a second axis whose block dimension is 1 (HIP allows nearly 2^32 there), and compute the program index as `pid0 + pid1 * num_programs0`, masking the programs that overrun the last row. Do this only on HIP, or keep the axis-1 count at or below 65,535: moving the large count itself to axis 1 or 2 trades the HIP failure for a CUDA one. A capped grid with a grid-stride (persistent) loop is the other option. In Triton the grid callable receives only the kernel arguments, not `num_warps`, so size axis 0 for the largest work-group (1024 work-items) unless the launch knows its config. Pruning autotune configs is not a fix on its own: past a size every config is too large and the list goes empty. In HIP C++, cap the grid and grid-stride. (FlexGEMM)

`surface.json` after this round: every component is covered (evidence per entry) except three owed items: `build:windows-rocm` (Windows compile/validation pending), `example:spconv` (question 1 above), and `doc:install-as-dependency` (validators). `pr_ready` refuses while anything is owed, so question 1 has to be answered before an upstream PR.

## Review 2026-11-20 (linux-gfx950 fleet job, AMD Instinct MI355X; reviewed head bcf7c6f5eee9f780a0ecdff4de7cecca8f73c728)

Reviewed `git diff 6dd94a8...bcf7c6f` (4 commits) with the pr-review skill. Every finding below was re-checked against the code, and the launch-size ones were reproduced on the GPU. **Verdict: changes requested.** Findings 1 and 2 are the launch-size fix. 3 to 5 are wording and record fixes.

### Findings

1. **The prune returns an empty config list for legal inputs of a few GiB, and the autotuner then crashes** (`flex_gemm/kernels/triton/grid_sample/indice_weighed_sum_bwd.py:11-20`, bcf7c6f). The configs that launch the fewest work-items are `BM=32,BK=8` / `BM=16,BK=16` (both `num_warps=2`), at about `M*C*V/2` work-items on wave64 and `M*C*V/4` on wave32. So the list is empty from `M*C*V ~ 2^33` (wave64) or `2^34` (wave32). For trilinear (V=8) that means `M*C >= 2^30`, which is 2 GiB of fp16 / 4 GiB of fp32 `grad_output` on wave64. With an empty list, `TritonPersistentCacheAutotuner.run` (`flex_gemm/utils/autotuner.py:63-66`) runs `builtins.min({})`. Reproduced on MI355X (Triton 3.9.0, warp_size 64), calling `indice_weighed_sum_bwd_input` directly (`cudacheck/repro_launch.py` in the job directory; extension stubbed, Triton kernels from the fork tree):
   ```
   bwd M=4194304 C=8    V=8 -> OK (best BM 32, BK 8)          # the tests/grid_sample.py RES=2048 case
   bwd M=1040000 C=1024 V=8 -> OK
   bwd M=1048576 C=1024 V=8 -> ValueError: min() arg is an empty sequence
   upstream 6dd94a8, same M=1048576 C=1024 V=8 -> SystemError: <built-in method __reversed__ of list object ...> returned a result with an exception set
   ```
   This is not a regression: upstream on HIP fails the same input. But the fix is incomplete. The commit body says "The remaining configs (larger BM) cover the same shapes", which is false in this range. The notes' question 2 says the gap needs "roughly M*V*C at or above 2^34, i.e. tens of GB", which is wrong for wave64 (2^33, 2 to 4 GiB) and also understates wave32 (4 to 8 GiB). Fix: change the launch so that no legal input is left without a config (see the view on question 2 below). If a prune remains as well, it must never return `[]`.
2. **The forward kernel of the same op has the same fault and no guard** (`flex_gemm/kernels/triton/grid_sample/indice_weighed_sum_fwd.py:80`). It uses the same `config.autotune_config` and the same 1D grid without the V factor: `cdiv(C,BK)*cdiv(M,BM)`. `BM=1,BK=64,num_warps=2` reaches exactly 2^32 work-items at `M*cdiv(C,64) = 2^25`. Reproduced on MI355X:
   ```
   fwd M=1048576 C=1024 V=8 -> OK (best BM 4, BK 64)
   fwd M=2097152 C=1024 V=8 -> SystemError: <built-in method __reversed__ of list object ...> returned a result with an exception set   (port and upstream alike)
   ```
   That is 2M query points with 1024 channels, a 4 GiB fp16 output. Whatever fixes finding 1 has to cover this kernel too. I checked the other Triton launch grids, in `flex_gemm/kernels/triton/spconv/*.py`: their worst config packs 16 output elements per work-item (`B1=128,B2=32,num_warps=4`), so they reach 2^32 only at `N*Co >= 2^36` (about 128 GiB of fp16 output). They are not a practical concern and are not part of this finding.
3. **The 078c4ab commit body states a false reason.** It says "torch adds its own MSVC flags to the hipcc command". In the default ninja path (`BuildExtension.win_wrap_ninja_compile`; torch 2.16 ROCm `cpp_extension.py:1246` and `1272`), a HIP build extends `common_cflags` with `COMMON_HIP_FLAGS` only, and that is what hipcc gets behind `-Xcompiler`. `COMMON_MSVC_FLAGS` reach hipcc only in the non-ninja `win_wrap_single_compile` fallback (`:1184`). The change itself is fine, because clang enables C++ exceptions by default and sets `__cplusplus` correctly, so hipcc needs none of `/EHsc`, `/permissive-` or `/Zc:__cplusplus`. Reword the body to give that reason instead.
4. **The lesson's "fold the grid into 2D" advice would break CUDA without a caveat** (`## Lesson for cuda-to-rocm: HIP rejects a launch ...`). CUDA caps gridDim.y and gridDim.z at 65,535 blocks, so moving the large tile count onto axis 1 or 2 trades the HIP failure for a CUDA one. The lesson must say that the large count stays on axis 0, or that the program loops. I also measured the per-dimension claim on MI355X (gfx950, ROCm 10.2, `cudacheck/lim/lim.hip`), and it holds: `grid=(1,4194303)x(1,1024)` runs, `(1,4194304)x(1,1024)` gives `hipErrorInvalidConfiguration`; `grid=(1,1,67108864)x(1,1,64)` is rejected. HIP's y/z limit counts work-items, not blocks: `hipDeviceProp.maxGridSize` reports `(2147483647,65535,65535)`, but `grid=(1,65536)x(1,128)` and `grid=(1,33554432)x(128,1)` both run. Add these y/z measurements and the CUDA caveat before promotion.
5. **The 06873dd Test Plan has no CUDA build, although the commit changes the CUDA flags.** This is the shared, unprefixed commit, and a CUDA maintainer will read it. Add the CUDA build from the compile check below, for example:
   ```
   BUILD_TARGET=cuda TORCH_CUDA_ARCH_LIST="8.0;9.0;12.0" python setup.py build_ext --inplace   # torch 2.14.1+cu126 and 2.11.0+cu128, nvcc 12.8
   ```

### Checked with no finding (for the porter's and Jeff's information, not findings)

- **06873dd on CUDA Linux:** built with real nvcc (below). The nvcc and g++ flag lists are identical to upstream's except for the standard, which torch now supplies: c++17 on torch 2.11, c++20 on torch 2.14.
- **06873dd on CUDA Windows**, checked by reading torch's code (no Windows nvcc host). In both torch 2.11 and 2.14, BuildExtension always gives Windows nvcc its own `-std=` (2.11 `cpp_extension.py:1014/1101` c++17; 2.14 `:1122/1211` c++20). On torch < 2.14, upstream sent nvcc that flag plus its own `-std=c++17`; the port sends torch's alone. On torch >= 2.14, upstream sent `-std=c++20` (torch), `-std=c++17` and `-Xcompiler=/std:c++17`; the port sends only `-std=c++20`, and nvcc forwards the matching `/std` to cl. For `cxx`, `append_std17_if_no_std_present` now adds `/std:c++20` (2.14) or `/std:c++17` (older), where upstream's explicit `/std:c++17` suppressed it. The port is correct there; a Windows CUDA build remains unexercised.
- **bcf7c6f arithmetic:** the prune's expression, `cdiv(C,BK)*cdiv(M,BM)*V*num_warps*warp_size < 2**32`, matches the kernel's grid lambda (`:93`) times Triton's block (`num_warps*warpSize`). The strict `<` matches the measured boundary. `warp_size` comes from Triton's AMD driver `get_current_target()`, which reads the device's `warpSize` (32 on RDNA), so wave32 keeps twice the headroom. It is unreachable on CUDA: `get_platform_name()` returns `'cuda'` when `torch.version.hip` is None, and the function returns before touching the driver. `nargs` holds `M`, `C` and `V`, because they are positional kernel arguments. The prune runs only on a cache miss with more than one config. The shipped `autotune_cache.json` has no MI300X entry for this kernel (only an empty A100 one), so no cached config bypasses it.
- **078c4ab Windows HIP flags**, checked by reading torch's code (no Windows host yet). With `GPU_ARCHS` unset on Windows, `cc_flag` is empty, so `_get_rocm_arch_flags` takes `PYTORCH_ROCM_ARCH`, or else the arch of every visible GPU. That matches both the README and the setup.py comment. With no GPU visible and no env var, torch itself would emit an empty `--offload-arch=`, which is torch's edge case and the reason the README says to set one. The `if arch` filter also fixes an empty or trailing-`;` `GPU_ARCHS` on Linux. `/openmp` stays in `cxx` (cl.exe), as the plan says. The Windows compile remains owed (`build:windows-rocm`).
- Commit hygiene: titles are 43-67 characters, `[ROCm]` is on exactly the three ROCm-specific commits, there are no trailers, `utils/jargon.py --port FlexGEMM` reports clean, and the diff adds no non-ASCII.

### Input on the porter's two questions (left open for Jeff)

1. *examples/spconv.py*: my view is (a) for this PR: the bug is pure Python, has nothing to do with ROCm, and widens a ROCm PR. Option (c), a separate one-line upstream PR or issue, is the better home if it should be fixed. Either way, `surface.json`'s `example:spconv` item needs your ruling before `pr_ready` can pass.
2. *prune vs kernel change*: given findings 1 and 2, I would prefer the kernel change. The prune leaves 2 to 4 GiB inputs failing on wave64, not "tens of GB", and it does not cover the forward kernel. One shape for the change: keep the M tiles on axis 0, move `(k, v)` to axis 1 (at most `cdiv(C,8)*V`, well under CUDA's 65,535), and keep a prune only for `BM` values whose axis-0 count reaches 2^32. Alternatively, a persistent grid-stride loop over a capped 1D grid removes the limit entirely. Both edit shared Triton kernels that upstream tunes, and both keep CUDA behaviour.

## CUDA compile check (2026-11-20, mi355 fleet host: real nvcc, no NVIDIA GPU)

Toolkit `/usr/local/cuda-12.8` (nvcc V12.8.93), host g++ 13.3.0, Python 3.11. Two job-local environments: the fleet guard allows pip only into the job's own venv, so they live at `$FLEET_VENV/cuda-new` and `$FLEET_VENV/cuda-old`. Neither touches the image's Python:

```
uv venv -p /opt/python-3.11-venv/bin/python3.11 $FLEET_VENV/cuda-new
uv pip install -p $FLEET_VENV/cuda-new/bin/python --index-strategy unsafe-best-match --index-url https://download.pytorch.org/whl/cu126 torch==2.14.1+cu126 --extra-index-url https://pypi.org/simple setuptools wheel ninja numpy
uv venv -p /opt/python-3.11-venv/bin/python3.11 $FLEET_VENV/cuda-old
uv pip install -p $FLEET_VENV/cuda-old/bin/python --index-strategy unsafe-best-match --index-url https://download.pytorch.org/whl/cu128 torch==2.11.0+cu128 --extra-index-url https://pypi.org/simple setuptools wheel ninja numpy
```

(cu128 wheels stop at torch 2.11, so torch >= 2.14 comes from the cu126 build. torch warns about a minor version mismatch, 12.8 vs 12.6, and continues.) Each tree is a `git archive <rev>` export, built exactly as setup.py does:

```
export CUDA_HOME=/usr/local/cuda-12.8 PATH=$VENV/bin:$CUDA_HOME/bin:/usr/bin:/bin HOME=<job-local> BUILD_TARGET=cuda TORCH_CUDA_ARCH_LIST="8.0;9.0;12.0" MAX_JOBS=32
python setup.py build_ext --inplace
```

| Tree | torch | Result |
|---|---|---|
| upstream 6dd94a8 | 2.14.1+cu126 | **fails**: `torch/csrc/api/include/torch/all.h:5:2: error: #error C++20 or later compatible compiler is required to use PyTorch.` (and the ATen.h twin), from the forced `-std=c++17`. This confirms the porter's reason. |
| port bcf7c6f | 2.14.1+cu126 | **builds**: all 6 `.cu` through nvcc with `-O3 --use_fast_math -allow-unsupported-compiler -std=c++20` and `sm_80/sm_90/sm_120`, `ext.cpp` through g++ with `-std=c++20`, the `.so` links. Warnings come only from torch headers (`#3189-D "module" is parsed as an identifier`, 168x) and upstream's 3x `#177-D variable "b" declared but never referenced`. |
| upstream 6dd94a8 | 2.11.0+cu128 | builds (`-std=c++17` from setup.py) |
| port bcf7c6f | 2.11.0+cu128 | **builds**: torch appends `-std=c++17`, all other flags identical to upstream; same 3 upstream warnings |

Both port builds' `.so` load with their torch (`importlib` on the built `cuda.cpython-311-x86_64-linux-gnu.so`: 18 exported functions). There is no NVIDIA GPU here, so nothing ran on CUDA. **The port causes no CUDA compile failure.**

## Porting round 2 (2026-11-20, linux-gfx942 fleet job, AMD Instinct MI300X)

Answers the 2026-11-20 review (MI355X, head bcf7c6f). New fork head **e7d36f10fcef3efc623656ca7592361db662f04b**, one commit on top of bcf7c6f, pushed plainly to `moat-port`. No history was rewritten (no amend, rebase or force-push); the wording faults in older commit bodies are recorded under "Correction for the final history" below.

**Questions for Jeff (none blocks the reviewer):**

1. Question 1 of round 1 (`examples/spconv.py`, `surface.json` `example:spconv` owed) is still open; this round did not touch it.
2. Question 2 of round 1 (prune vs kernel change) is answered by doing the kernel change the reviewer preferred, so it needs no ruling. Round 1's estimate of where the prune left gaps ("roughly M*V*C at or above 2^34, i.e. tens of GB") was wrong: the gap started at M*C*V ~ 2^33 on wave64 (2 to 4 GiB of trilinear `grad_output`) and 2^34 on wave32, as the reviewer measured.
3. New, upstream and not ROCm-specific: the grid_sample backward kernel returns wrong gradients on the first call for each new shape key, on every platform. Registered as deferred `flexgemm-grid-sample-bwd-autotune-accumulates`; details below. Options: (a) leave it out of this port (current); (b) a separate unprefixed one-line commit in this port; (c) an upstream issue or PR. (b) or (c) widens what goes upstream, so it is your call.

### Finding 1 and 2: the launch change (commit e7d36f1)

Title `[ROCm] grid_sample: split large launches over two grid axes on HIP` (67 characters).

- New helper `flat_grid(num_programs)` in `flex_gemm/kernels/triton/utils.py`, next to `get_platform_name`. Off HIP it returns `(num_programs,)`, exactly the grid both kernels launched before. On HIP it computes `rows = max(1, cdiv(num_programs, (2**32 - 1) // 1024))` and returns `(cdiv(num_programs, rows), rows)`. Axis 0 thus never holds more than 4,194,303 programs, which keeps it under 2^32 work-items for any work-group up to 1024 (HIP's maximum, so it holds for every `num_warps` on wave64 and wave32 alike). Axis 1 has a block dimension of 1 in Triton, and HIP's axis-1 limit is also a work-item count (reviewer's MI355X measurement), so `rows` (at most a few hundred for any input below the 32-bit indexing limit) is far inside it. `rows` stays far below CUDA's 65,535 as well, although the split never runs on CUDA.
- Why the 1024 bound instead of the config's real block size: Triton 3.9 calls the grid lambda with the kernel's bound arguments only (`JITFunction.run`, `grid = grid(bound_args)`), which do not include `num_warps`, and the prune approach needed `get_current_target().warp_size` per launch. The cost is that HIP splits a launch somewhat earlier than strictly needed (at 4.19M programs instead of 33.5M for a 128-wide group), which changes only the grid shape, not the work.
- Both kernels: `block_id = tl.program_id(axis=0) + tl.program_id(axis=1) * tl.num_programs(axis=0)`. The forward kernel's store mask gains `& (block_id_m < tl.cdiv(M, BM))`, because its `offset_m` wraps modulo M, so without it the up-to-`rows - 1` programs that overrun the last row would recompute and rewrite valid rows. The backward kernel needs no new mask: an overrunning program has `offset_m >= M`, so its loads and atomics are already masked off.
- bcf7c6f's `early_config_prune` is removed, so no prune remains and no config list can go empty; the `get_platform_name` import moves to the helper.
- **CUDA is unchanged in behaviour:** `flat_grid` returns the same 1-tuple as before, so the launched grid, `program_id(0)`, the program-to-tile mapping and the autotune configs (`config.py` untouched) are identical; `program_id(1)` is 0 and the new forward mask term is always true there, because the grid covers exactly `cdiv(M,BM)*cdiv(C,BK)` programs. The compiled kernel gains one `program_id(1)` read, a multiply-add and one compare.
- Diff vs bcf7c6f: 3 files, +24 / -20. Diff vs upstream 6dd94a8 in `flex_gemm/`: 3 files, +23 / -5.

Out of scope and shared with CUDA (recorded, not changed): both kernels compute offsets in 32-bit (`offset_m * C`, `neigh_idx * C`), so tensors of 2^31 or more elements overflow on every platform. The fix covers every input below that bound; the largest shapes tested sit at it (M*C = 2^31 in the forward run).

### Reproduction before and after (MI300X gfx942, GPU 2, ROCm 10.2, torch 2.16, Triton 3.9.0)

Driver `agent_space/repro_launch.py <tree> fwd|bwd M C V [dtype]`: imports the Triton kernels of an exported tree (`git archive <rev> flex_gemm`; CUDA extension stubbed, autotune cache off), uses uint32 indices with every 7th row's first neighbour missing (0xffffffff), float32 weights and N = 65,536 rows, autotunes all 12 configs, and compares against a chunked float32 torch reference (`index_add_` for backward), with tolerance 1% of the reference maximum. For backward it checks the second call, because the first call's result is corrupted by question 3. `FORCE_CFG=<i>` replaces the config list with config i alone and prints the launched grid.

Before (launch outcome only):

```
bcf7c6f bwd M=4194304 C=8    V=8 -> launches (tests/grid_sample.py RES=2048 case)
bcf7c6f bwd M=1040000 C=1024 V=8 -> launches
bcf7c6f bwd M=1048576 C=1024 V=8 -> ValueError: min() arg is an empty sequence
bcf7c6f fwd M=1048576 C=1024 V=8 -> OK
bcf7c6f fwd M=2097152 C=1024 V=8 -> SystemError: <built-in method __reversed__ of list object ...> returned a result with an exception set
6dd94a8 bwd: all three shapes -> SystemError (same); fwd: same as bcf7c6f
```

After, e7d36f1 (autotuned):

```
bwd M=4194304 C=8    V=8 float16 -> OK (best BM16 BK16; max_abs_err 0.538 tol 0.616)
bwd M=1040000 C=1024 V=8 float16 -> OK (best BM4 BK64; 0.18 / 0.383)
bwd M=1048576 C=1024 V=8 float16 -> OK (best BM4 BK64; 0.187 / 0.383)
bwd M=2097152 C=1024 V=8 float16 -> OK (best BM4 BK64; 0.337 / 0.54)
bwd M=1048576 C=1024 V=8 float32 -> OK (best BM4 BK64; 1.14e-05 / 0.381)
fwd M=1048576 C=1024 V=8 float16 -> OK (best BM4 BK64; 0.00781 / 0.119)
fwd M=2097152 C=1024 V=8 float16 -> OK (best BM4 BK64; 0.00781 / 0.123)
fwd M=2097152 C=1024 V=8 float32 -> OK (best BM2 BK64; 1.91e-06 / 0.123)
fwd M=4194304 C=8    V=8 float16 -> OK (best BM32 BK8; 0.00391 / 0.108)
fwd/bwd M=1000 C=8 V=8 float16   -> OK (no split)
```

(The float16 backward errors are float16 atomic accumulation, about 512 contributions per row; float32 is exact to 1e-5.)

Every config forced in turn, float32, all 36 runs OK: backward M=2097152 C=1024 V=8 (grids from `(3947581, 17)` for the 64-wide-tile configs to `(4129777, 65)` for BM=1 BK=64; 67M to 268M programs, max_abs_err 1.91e-05), forward M=2097152 C=1024 V=8 (grids `(2796203, 3)` to `(3728271, 9)`, 1.91e-06), backward M=4194304 C=8 V=8 (grids `(1048576, 1)` to `(3728271, 9)`, about 5e-05). Most split grids overrun their program count (by 1, 4, 7, 13 or 49 programs; e.g. `(3728271, 9)` launches 33,554,439 programs for 33,554,432), so the overrun masking is exercised in both kernels; `(2097152, 2)` divides exactly. The `(4129777, 65) x 128` launch is about 2^35 work-items in total, which HIP accepts: the limit is per dimension. Output in `agent_space/repro-forced.txt` (job-local).

Wave32 is covered by the same 1024-work-item bound, which does not depend on wavefront size; it has not been run on a wave32 GPU in this round (the wave32 validator will).

### Project tests at e7d36f1 (MI300X)

```
source agent_space/env.sh      # HOME, TRITON_CACHE_DIR job-local; ROCM_HOME from rocm_sdk; GPU_ARCHS=gfx942; ulimit -c 0
utils/timeit.sh FlexGEMM compile -- $FLEET_VENV/bin/python -m pip install <src> --no-build-isolation --no-deps   # Successfully installed flex_gemm-1.0.0
utils/timeit.sh FlexGEMM test -- bash -c "cd <src>/tests && python grid_sample.py"     # rc 0, 73 s, autotune cache cleared first
utils/timeit.sh FlexGEMM test -- bash -c "cd <src>/examples && python submconv.py"     # rc 0, 97 s
```

- `tests/grid_sample.py`, unchanged: all four tables (Nearest Fwd, Trilinear Fwd, Nearest Bwd, Trilinear Bwd) complete through RES=2048, with flex_gemm err_max 0 per mille and err_mean 0 per mille on every row. Trilinear Bwd times: RES=2048,C=8 105.9 ms (torch 168.7 ms).
- No example uses grid_sample (`grep -rl grid_sample examples/` is empty). `examples/submconv.py` was run as a sanity check and passes; `examples/spconv.py` is question 1 of round 1. The spconv and submconv kernels and tests are untouched by this round (the diff touches only the grid_sample kernels and a new helper), so round 1's results for them stand until the validator reruns them at this head.
- `git -C projects/FlexGEMM/src status --porcelain`: clean.
- `python3 utils/jargon.py --port FlexGEMM`: clean. `python3 utils/prose.py` on the commit message: clean.

### Upstream bug: grid_sample backward is wrong on the first call for each shape (question 3)

`indice_weighed_sum_bwd_input` allocates `grad_input = torch.zeros(...)` and the kernel `tl.atomic_add`s into it, but `@triton_autotune` on the kernel has no `reset_to_zero`. On an autotune cache miss, `TritonPersistentCacheAutotuner.run` benchmarks every config on the same arguments (many runs each), all accumulating into that buffer, and then launches the best config into it once more and returns it. Measured on upstream 6dd94a8 (`agent_space/firstcall.py`, M=4096 C=8 V=8 N=1024 float32): first-call vs second-call max ratio 24,058. Adding `reset_to_zero=['grad_input'],` to the decorator makes the two calls equal (ratio 0.9999999). Every platform is affected, including CUDA; a persisted autotune cache hides it. The forward kernel stores (no accumulation), so it is unaffected. Not changed in this port (not ROCm-specific; smallest port); see question 3.

### Environment (this job)

Fresh job directory, so the round-1 environment was rebuilt the same way: TheRock `torch==2.16.0a0+rocm10.2.0a20261022`, `triton==3.9.0+gitfc8d31a5.rocm10.2.0a20261022`, `rocm[libraries,devel]==10.2.0a20261022`, `rocm-sdk-device-gfx942==10.2.0a20261022` and `amd-torch-device-gfx942==2.16.0a0+rocm10.2.0a20261022` from `https://nightly.repo.amd.com/rocm/whl-next/` into `$FLEET_VENV`; the job's own `venv-base-links` symlinks to the base env's ROCm 10.1 and Triton removed. Two traps: an unpinned `amd-torch-device-gfx942` sends pip backtracking through every 2.15/rocm10.3 nightly (pin it to torch's exact version), and without that wheel every torch GPU kernel fails with `hipErrorInvalidKernelFile` (torch's `device-gfx942` extra names it, but a plain `torch==` install does not pull it). `python -m rocm_sdk test` OK. `HIP_VISIBLE_DEVICES=2` (dispatcher pin), `ulimit -c 0`, `HOME` and `TRITON_CACHE_DIR` job-local.

## Correction for the final history

Jeff's standing rule for fork branches is no amend, no rebase and no force-push, so these commit bodies stay as they are on `moat-port`. Apply the corrected text below when the commits are prepared for the upstream pull request (a later, person-approved step). Titles are unchanged.

### 06873dd `setup.py: let torch choose the C++ standard` (review finding 5)

Replace the Test Plan with the following (rest of the body unchanged):

````
Test Plan:
Against torch 2.16 (ROCm build), the unmodified setup.py fails with the #error above. With this change:

```
pip install . --no-build-isolation --no-deps
cd tests && python hashmap.py && python serialize.py
```

CUDA (nvcc 12.8, no GPU needed): against torch 2.14.1+cu126 the unmodified setup.py fails with the same #error; with this change it builds with -std=c++20 from torch. Against torch 2.11.0+cu128 it builds with -std=c++17 from torch, with every other flag as before.

```
BUILD_TARGET=cuda TORCH_CUDA_ARCH_LIST="8.0;9.0;12.0" python setup.py build_ext --inplace
```
````

### 078c4ab `[ROCm] setup.py: Windows HIP flags and default target` (review finding 3)

Replace the first paragraph with the following (rest of the body unchanged):

```
On Windows with a ROCm build of PyTorch, the "nvcc" flag list goes to hipcc unchanged. The -Xcompiler=/... options there are nvcc syntax that clang does not accept, so give hipcc only -O3 and the target list on Windows; the host flags still reach cl.exe through the "cxx" list. hipcc needs none of /EHsc, /permissive- or /Zc:__cplusplus: clang enables C++ exceptions by default and sets __cplusplus to the language standard in use.
```

### bcf7c6f `[ROCm] grid_sample: skip bwd configs past the HIP launch size limit` and e7d36f1

bcf7c6f's body is false where it says "The remaining configs (larger BM) cover the same shapes" (review finding 1), and e7d36f1 removes everything bcf7c6f added. Recommended: squash the two into one commit with e7d36f1's title, and e7d36f1's body with its second paragraph ("The previous commit (bcf7c6f) ...") dropped and the first sentence of its third paragraph reading "This commit changes the launch shape:" instead of "This commit removes that prune and changes the launch shape instead.". If the two stay separate commits instead, replace bcf7c6f's second paragraph with:

```
Add an early_config_prune that drops, on HIP only, the configs whose launch would reach that limit. This covers the RES=2048 test case but not larger inputs: from about M * C * V = 2^33 on 64-wide wavefronts every config reaches the limit and the list is empty; the next commit replaces the prune with a launch that splits the grid. CUDA keeps every config.
```

### e7d36f1 and 8592d5f `[ROCm] grid_sample: decide the platform once for flat_grid` (round-2 review finding 1)

e7d36f1's fourth paragraph is wrong about CUDA: it says the CUDA path is unchanged, but e7d36f1's `flat_grid()` ran `get_platform_name()` on every launch, so every launch on CUDA gained a `torch.cuda.is_available()` call (0.8 us measured on the AMD host). 8592d5f moves the platform check to import time.

**If bcf7c6f, e7d36f1 and 8592d5f are squashed (recommended), use one commit** with e7d36f1's title and e7d36f1's body with the two edits listed in the subsection above (second paragraph dropped; third paragraph opening "This commit changes the launch shape:"), plus these two changes. Replace the fourth paragraph with:

```
On CUDA the grid, the program-to-tile mapping and the autotune configs are unchanged: flat_grid() returns (num_programs,), program_id(1) is 0, and the added store mask term is always true. flat_grid() decides the platform once at import, as the autotune config selection does, so a launch on CUDA costs one module-level truth test more than before.
```

Then add this sentence to the end of the Test Plan prose paragraph, before the fenced block:

```
Per-call time of the two kernels at M=1000, C=8, V=8 after autotuning is unchanged within noise (indice_weighed_sum_fwd 28.4-28.9 us before, 28.7-29.5 us after).
```

**If e7d36f1 and 8592d5f stay separate commits**, replace e7d36f1's fourth paragraph with:

```
On CUDA the grid, the program-to-tile mapping and the autotune configs are unchanged: flat_grid() returns (num_programs,), program_id(1) is 0, and the added store mask term is always true. The grid callable does call get_platform_name() on every launch; the next commit moves that check to import time.
```

and replace 8592d5f's second paragraph ("The previous commit's message says ...") with:

```
On CUDA the previous commit kept the launched grid, the program-to-tile mapping and the autotune configs, but every launch also paid for a torch.cuda.is_available() call (0.8 us per call measured on the AMD host), host cost a HIP-only change should not add.
```

## Review 2026-11-20 round 2 (linux-gfx942 fleet job, AMD Instinct MI300X; reviewed head e7d36f10fcef3efc623656ca7592361db662f04b)

Reviewed `git diff 6dd94a8...e7d36f1` (5 commits) with the pr-review skill, focusing on e7d36f1 and on the porter's answers to the round-1 review. Every finding was re-checked against the code and measured on the GPU. **Verdict: changes requested, one finding.** The launch change itself is correct on every input I could construct: round-1 findings 1 and 2 are resolved, and findings 3 to 5 are resolved by the "Correction for the final history" section (the corrected text is accurate; it has to be applied when the commits are prepared for the upstream PR).

### Findings

1. **`flat_grid` adds a per-launch host cost on both platforms, and the CUDA path pays it for a HIP-only fix** (`flex_gemm/kernels/triton/utils.py:28-31`, e7d36f1). Triton calls the grid lambda on every launch (`triton/runtime/jit.py:778-779`, `grid = grid(bound_args)`), so every `indice_weighed_sum_fwd`/`_bwd_input` call now runs `get_platform_name()`, which calls `torch.cuda.is_available()` (0.86 us here), and on HIP two more `triton.cdiv` calls (1.3 us each; `triton.cdiv` is a wrapped function, not plain arithmetic). `flat_grid(1000)` measures 4.0 us. End to end, a small forward call (M=1000 C=8 V=8, autotuned, 2000 calls, best of 5) goes from 28.0-28.7 us/call at bcf7c6f to 33.3 us/call at e7d36f1 on MI300X, about 16% slower. On CUDA the cost is the `is_available()` call, about 0.9 us per launch: small, but it contradicts the commit body's "on CUDA ... unchanged" and is the kind of change an upstream CUDA maintainer rejects in a ROCm PR. Upstream evaluates the platform once, at import (`get_autotune_config`, `utils.py:53`). Fix: decide the platform once at import and use integer arithmetic, for example
   ```
   def flat_grid(num_programs):
       ...
       if not _IS_HIP:
           return (num_programs,)
       rows = max(1, -(-num_programs // ((2**32 - 1) // 1024)))
       return (-(-num_programs // rows), rows)

   _IS_HIP = get_platform_name() == 'hip'
   ```
   I measured exactly that variant (`agent_space/review2/fixcheck`, not committed anywhere): 29.2-29.8 us/call, and all 12 configs still pass `bwd M=4194304 C=8 V=8`. The CUDA path then costs one module-global truth test per launch. A new commit on top (no amend), with the e7d36f1 body's correction note in "Correction for the final history" extended if the two are to be squashed.

### Checked with no finding (for the porter's and Jeff's information)

- **Round-1 findings 1 and 2 (launch size):** no prune remains, so no config list can go empty. Axis 0 holds `cdiv(n, rows) <= 4,194,303` programs because `rows >= n / 4,194,303`, so axis 0 stays below 2^32 work-items for any work-group up to 1024 (Triton's maximum: 16 warps of 64, or 32 of 32). Axis 1 has block dimension 1, and `rows <= 512` for any program count Triton's int32 `program_id` can index. The limit is a property of the HIP dispatch (32-bit work-item counts per dimension), not of the wavefront size, and FlexGEMM's configs are all `num_warps=2` (64 work-items on wave32, 128 on wave64), so the same bound holds on wave32 parts with more headroom. Not run on wave32 here (no such GPU); the wave32 validator covers it.
- **Masking of overrun programs:** backward: an overrunning program has `block_id_m >= cdiv(M,BM)`, so `offset_m >= M`, and the `grad_output`, `indices` (`other=0xffffffff`, so `valid_mask` false) and `weight` loads and the atomic are all masked (`indice_weighed_sum_bwd.py:43-56`). Forward: `offset_m` wraps modulo M (`indice_weighed_sum_fwd.py:41`), so loads stay in bounds, and the new `block_id_m < tl.cdiv(M, BM)` term (`:63`) stops the store. Without it the overrun would rewrite wrapped rows with identical values (benign, but racy); the mask is right.
- **CUDA kernel behaviour:** off HIP `flat_grid` returns `(num_programs,)`, the same 1-tuple as upstream; `program_id(1)` is 0 and the new store term is always true because the grid is exactly `cdiv(M,BM)*cdiv(C,BK)`. `config.py` and the autotune keys are untouched. Apart from finding 1, CUDA behaviour is unchanged.
- **32-bit indexing claim in the notes:** with every config having `BM*BK >= 64` and V <= 64 (V is 1 or 8 from the op), any input whose tensors stay below 2^31 elements gives fewer than 2^31 programs, so `block_id` cannot overflow before the pre-existing (shared with CUDA) offset overflow. The notes' bound is right.
- **Lesson `HIP rejects a launch of 2^32 or more work-items ...`:** the CUDA caveat (keep the large count on axis 0, or keep axis 1 at or below 65,535), the y/z measurements and the "pruning is not a fix" point are now correct. "The grid callable receives only the kernel arguments, not num_warps" checks out against Triton 3.9 `jit.py:779`. The MI300X total-size claim is confirmed and extended: `grid=(4129777, 65) x 128` (2^35 work-items) and `grid=(4177984, 257) x 128` (about 2^37) both ran correctly here. Ready to promote.
- **Deferred `flexgemm-grid-sample-bwd-autotune-accumulates`:** reproduced (`agent_space/firstcall.py`, M=4096 C=8 V=8 N=1024 float32): first/second call max ratio 26,971 on 6dd94a8 and 26,339 on e7d36f1. Upstream, not ROCm-specific, unchanged by the port; the ruling stays with Jeff.
- **Porter's driver** (`agent_space/repro_launch.py`) tests what it says, but its tolerance (1% of the reference maximum) is loose enough to hide a missing tile in backward, and in an autotuned forward run the benchmark runs of other configs fill the same output buffer, which would mask a config that skips tiles. So I wrote a stricter driver (below) and checked it with a negative control.
- Commit hygiene: e7d36f1's title is 67 characters with `[ROCm]`, no trailer, AI-agent disclosure present, `python3 utils/jargon.py --port FlexGEMM` clean, no non-ASCII added lines. `git -C projects/FlexGEMM/src status --porcelain` clean. The installed `flex_gemm` in the venv matches the e7d36f1 export (`diff -rq`, ignoring build output).

### GPU runs (MI300X gfx942, GPU 2 via the dispatcher's HIP_VISIBLE_DEVICES=2, TheRock ROCm 10.2.0a20261022, torch 2.16.0a0, Triton 3.9.0)

```
source agent_space/env.sh
git -C projects/FlexGEMM/src archive e7d36f1 flex_gemm | tar -x -C agent_space/review2/tree
```

Previously failing shapes, through the wrappers with autotuning (`agent_space/repro_launch.py <tree> <fwd|bwd> M C V float16`, absolute tree paths):

```
                         e7d36f1                                   bcf7c6f                    6dd94a8 (upstream)
bwd M=1048576 C=1024 V=8 OK (BM4 BK64; err 0.187, tol 0.383)       ValueError: min() arg is   SystemError (__reversed__ ...)
                                                                   an empty sequence
fwd M=2097152 C=1024 V=8 OK (BM4 BK64; err 0.00781, tol 0.123)     SystemError (__reversed__) SystemError
bwd M=4194304 C=8    V=8 OK (BM16 BK8; err 0.478, tol 0.616)       OK (BM16 BK16)             SystemError
float32 at e7d36f1: bwd 1048576/1024/8 err 1.14e-05; fwd 2097152/1024/8 err 1.91e-06; bwd 4194304/8/8 err 6.1e-05 (all OK)
```

Every config forced, strict driver `agent_space/review2/allcfg.py <tree> <fwd|bwd> M C V [cfgs|all]`: launches the raw JIT kernel once per config with the `flat_grid` grid (no autotuner, so no benchmark run touches the output), float32, forward output prefilled with NaN, backward output fresh zeros, float64 torch reference, tolerance 1e-5 x reference max, and it counts never-written outputs. Negative control: a copy of the e7d36f1 tree with `block_id = tl.program_id(axis=0)` restored (`agent_space/review2/sabotage`) fails `fwd M=4194305 C=64 V=8 cfg2` with 134,217,728 NaN outputs; the real tree passes it.

```
SUMMARY bwd M=1048576   C=1024 V=8: 12/12 configs OK   grids up to (4067204, 33), overrun up to 7
SUMMARY fwd M=2097152   C=1024 V=8: 12/12 configs OK   grids up to (3728271, 9)
SUMMARY bwd M=4194304   C=8    V=8: 12/12 configs OK   (1048576, 1) to (3728271, 9)
SUMMARY bwd M=2097152   C=1024 V=8: 12/12 configs OK   up to (4129777, 65), overrun 49
SUMMARY fwd M=270003    C=1000 V=8: 12/12 configs OK   partial last M tile, odd num_k (BK=8)
SUMMARY bwd M=270003    C=1000 V=8: 12/12 configs OK
SUMMARY fwd M=4194303   C=64   V=8: cfg2 OK            (4194303, 1): last unsplit size
SUMMARY fwd M=4194304   C=64   V=8: cfg2 OK            (2097152, 2): first split size
SUMMARY bwd M=134217728 C=8    V=8: cfg2, cfg11 OK     (4177984, 257) x 128, 2^30 programs, overrun 64
SUMMARY fwd M=134217728 C=8    V=8: cfg2, cfg11 OK     (4067204, 33)
```

Max abs errors: 1.9e-06 forward everywhere; backward 2.1e-05 to 6.9e-05 (C=1024 / C=8), 1.7e-03 at M=134217728 (about 16,000 float32 atomic contributions per row, tolerance 3.9e-03). Full log: `agent_space/review2/allcfg.log` (job-local).

`tests/grid_sample.py`, unchanged, against the installed e7d36f1 build with the autotune cache off (`FLEX_GEMM_USE_AUTOTUNE_CACHE=0 FLEX_GEMM_AUTOSAVE_AUTOTUNE_CACHE=0 python grid_sample.py` in `src/tests`): rc 0, all four tables (Nearest/Trilinear Fwd/Bwd) through RES=2048, all 72 err_max/err_mean cells 0 per mille.

Launch overhead (`agent_space/review2/launch_overhead.py <tree>`, finding 1): bcf7c6f 27.8-28.7 us/call, e7d36f1 33.3, e7d36f1 with the import-time platform check 29.2-29.8.

## Porting round 3 (2026-11-20, linux-gfx942 fleet job, AMD Instinct MI300X)

Answers the round-2 review (one finding, `flat_grid` per-launch host cost). New fork head **8592d5fcb68855864c8a82274d3033ae651f8027**, one commit on top of e7d36f1, pushed plainly to `moat-port` (no amend, rebase or force-push). Title `[ROCm] grid_sample: decide the platform once for flat_grid` (58 characters).

**Questions for Jeff:** none new. Round 1's question 1 (`examples/spconv.py`, `surface.json` `example:spconv`) and round 2's question 3 (deferred `flexgemm-grid-sample-bwd-autotune-accumulates`) are unchanged.

### The change

`flex_gemm/kernels/triton/utils.py`: `_IS_HIP = get_platform_name() == 'hip'` at module level, right after `get_platform_name`; `flat_grid` tests `_IS_HIP` and computes `rows = max(1, -(-n // ((2**32 - 1) // 1024)))`, `(-(-n // rows), rows)` instead of two `triton.cdiv` calls. 1 file, +6 / -3. `triton` is still used in the module (`get_autotune_config`), so no import is orphaned. The callers' `triton.cdiv(C, META['BK']) * triton.cdiv(M, META['BM'])` inside the grid lambdas is upstream's own code and stays.

The commit body states what e7d36f1's body got wrong: CUDA's grid, mapping and configs were unchanged, but every CUDA launch paid for a `torch.cuda.is_available()` call that the message did not mention. "Correction for the final history" now covers the squashed and the separate cases.

### Import order (checked, not assumed)

- `_IS_HIP` is assigned at module import, after `get_platform_name` is defined and before `flat_grid` is defined. `flat_grid` is referenced only inside the two grid lambdas (`indice_weighed_sum_fwd.py:81`, `indice_weighed_sum_bwd.py:80`), which Triton evaluates at launch, so no call can run before the module finishes importing. `utils.py` is imported by both `grid_sample/config.py` and `spconv/config.py` during `import flex_gemm`.
- Upstream already evaluates the platform at import: `spconv/config.py` calls `get_autotune_config(platform=...)` at module level, which calls `get_platform_name()`. So the new flag adds no new import-time device query in a process that imports `flex_gemm`.
- On this torch (2.16 ROCm), `torch.cuda.is_available()` is True before any CUDA initialization even though `torch.cuda.device_count()` reads 0 until then, so the import-time value is right: the installed package reports `_IS_HIP True`.
- `flat_grid(n)` equals the e7d36f1 `triton.cdiv` form for n in {1, 4194303, 4194304, 4194305, 33554439, 2^30, 2^31-1} (asserted). `flat_grid(1000) = (1000, 1)`, `flat_grid(4194304) = (2097152, 2)`, `flat_grid(2^30) = (4177984, 257)`.

### Per-call time (finding 1), MI300X GPU 2

Driver `agent_space/fix3/launch_overhead.py <tree>` (job-local): imports the Triton kernels of a `git archive` tree (CUDA extension stubbed, autotune cache off), M=1000 C=8 V=8 N=1024 float32, autotunes, then 2000 calls of each wrapper, best of 5. Three interleaved runs per tree:

```
               fwd us/call       bwd us/call       flat_grid(1000)
bcf7c6f        28.40-28.93       30.98-32.05       (none)
e7d36f1        33.06-33.60       35.65-36.18       4.16-4.67 us
8592d5f        28.70-29.46       31.69-32.97       0.19-0.20 us
```

The non-HIP branch of the new `flat_grid` (flag forced False) costs 0.077 us per call; `torch.cuda.is_available()` costs 0.817 us here, which is what e7d36f1 added per launch on CUDA.

### Strict launch driver, every config, large shapes (round-2 reviewer's method)

`agent_space/fix3/allcfg.py <tree> fwd|bwd M C V [cfgs|all]`, rewritten to the round-2 description: raw JIT kernel launched once per config with the `flat_grid` grid (no autotuner), float32, forward output prefilled with NaN, backward output zeros, float64 torch reference, tolerance 1e-5 x reference max, never-written outputs counted. N=65536, uint32 indices with every 7th row's first neighbour 0xffffffff.

```
SUMMARY bwd M=1048576   C=1024 V=8: 12/12 OK  grids (3728271, 9)..(4067204, 33), overrun <= 13, err 2.1e-05 (tol 4.0e-04)
SUMMARY bwd M=4194304   C=8    V=8: 12/12 OK  grids (2796203, 3)..(3728271, 9), overrun <= 7, err 5.2e-05..6.0e-05
SUMMARY bwd M=2097152   C=1024 V=8: 12/12 OK  up to (4129777, 65), overrun <= 49, err 4.2e-05..4.4e-05
SUMMARY bwd M=270003    C=1000 V=8: 12/12 OK  err 5.9e-06..6.1e-06
SUMMARY fwd M=2097152   C=1024 V=8: 12/12 OK  grids (2796203, 3)..(3728271, 9), err 2.0e-06
SUMMARY fwd M=270003    C=1000 V=8: 12/12 OK  partial last M tile, err 1.6e-06
SUMMARY fwd M=4194303   C=64   V=8: cfg2 OK   (4194303, 1), last unsplit size
SUMMARY fwd M=4194304   C=64   V=8: cfg2 OK   (2097152, 2), first split size
SUMMARY fwd M=4194305   C=64   V=8: cfg2 OK   (2097153, 2), overrun 1
SUMMARY bwd M=134217728 C=8    V=8: cfg2, cfg11 OK  (4177984, 257), overrun 64, err 1.9e-03..2.0e-03 (tol 3.8e-03)
SUMMARY fwd M=134217728 C=8    V=8: cfg2, cfg11 OK  (4067204, 33), err 1.8e-06
```

Negative control: a copy of the 8592d5f tree with the forward kernel's `block_id = tl.program_id(axis=0)` restored fails `fwd M=4194305 C=64 V=8 cfg2` with 134,217,728 NaN outputs (the real tree passes). Logs `agent_space/fix3/allcfg-bwd.log`, `allcfg-fwd.log` (job-local).

### Build and project test

```
source agent_space/env.sh   # HOME, TRITON_CACHE_DIR job-local; ROCM_HOME from rocm_sdk; PYTORCH_ROCM_ARCH=gfx942; ulimit -c 0
utils/timeit.sh FlexGEMM compile -- $FLEET_VENV/bin/python -m pip install ./projects/FlexGEMM/src --no-build-isolation --no-deps   # Successfully installed flex_gemm-1.0.0
utils/timeit.sh FlexGEMM test -- bash -c "cd projects/FlexGEMM/src/tests && FLEX_GEMM_USE_AUTOTUNE_CACHE=0 FLEX_GEMM_AUTOSAVE_AUTOTUNE_CACHE=0 python grid_sample.py"   # rc 0
```

`tests/grid_sample.py`, unchanged: all four tables (Nearest/Trilinear Fwd/Bwd) through RES=2048, all 72 err_max/err_mean cells 0 per mille. `git -C projects/FlexGEMM/src status --porcelain` clean. `python3 utils/jargon.py --port FlexGEMM` clean; `utils/prose.py` on the commit body clean; no non-ASCII added.

CUDA compile check: not rerun. The commit changes only Python (`utils.py`), which nvcc never compiles; the round-1 nvcc check covers every CUDA source at this head.

Environment rebuilt in this fresh job as in round 2 (TheRock `torch==2.16.0a0+rocm10.2.0a20261022`, `triton==3.9.0+gitfc8d31a5.rocm10.2.0a20261022`, `rocm[libraries,devel]`, `rocm-sdk-device-gfx942` and `amd-torch-device-gfx942` pinned to the same nightly, base-env ROCm 10.1 and Triton links removed from `venv-base-links`); `python -m rocm_sdk test` OK. The fleet guard accepts pip only as `$FLEET_VENV/bin/python -m pip` spelled literally.

## Review 2026-11-20 round 3 (linux-gfx942 fleet job, AMD Instinct MI300X; reviewed head 8592d5fcb68855864c8a82274d3033ae651f8027)

Reviewed `git diff 6dd94a8...8592d5f` (6 commits) with the pr-review skill, focusing on 8592d5f, the porter's round-3 claims and the new "Correction for the final history" subsection. **Verdict: review passed, no findings.** Round-2 finding 1 is resolved: `flat_grid` (`flex_gemm/kernels/triton/utils.py:21-34`) tests the import-time `_IS_HIP` (`:18`) and uses integer arithmetic.

### Re-checked (evidence for the porter, Jeff and the validator)

- **Import-time platform check:** `_IS_HIP` is computed once, after `get_platform_name` is defined (`utils.py:10-18`). `flat_grid` is referenced only inside the grid lambdas (`indice_weighed_sum_fwd.py:81`, `indice_weighed_sum_bwd.py:80`), so nothing calls it during import. Upstream already queries the device at import: `spconv/config.py:10` calls `get_autotune_config(platform=..., device=...)` at module level, which calls `torch.cuda.get_device_name()` and possibly `get_platform_name()`, and `flex_gemm/kernels/triton/__init__.py` imports `spconv`. So the flag adds no new kind of import-time device query, on CUDA or on HIP. "As the autotune config selection already does" in the 8592d5f body and in the correction text is accurate.
- **Arithmetic:** a pure-Python check (300,000 random n in [1, 2^31-1] plus the edge values around 4,194,303) confirmed that the new `flat_grid` returns the same grid as e7d36f1's `triton.cdiv` form for every n, that axis 0 stays at or below 4,194,303 programs (so below 2^32 work-items with any work-group up to 1024), and that the overrun stays below the row count. `flat_grid(2^31-1) = (4186128, 513)`.
- **Per-call time** (`agent_space/review3/launch_overhead.py`, the porter's driver copied unchanged; `git archive` trees per commit; M=1000 C=8 V=8 float32, autotuned, 2000 calls, best of 5; 8 interleaved runs for 6dd94a8 and 8592d5f, 3 for e7d36f1):
  ```
                 fwd us/call     bwd us/call     flat_grid(1000)
  6dd94a8        27.65-29.46     30.65-32.35     -
  bcf7c6f        28.18-29.62     30.98-33.06     -
  e7d36f1        32.73-33.29     35.73-36.83     4.19-4.62 us
  8592d5f        28.78-29.96     31.11-33.20     0.18-0.19 us
  ```
  The e7d36f1 regression is gone. The ranges for 8592d5f overlap upstream's, so "unchanged within noise" in the correction holds. The medians sit about 0.5 us higher on HIP, of which the measured grid-callable delta is 0.28 us.
- **Host cost of the grid callable alone** (`agent_space/review3/gridcost.py`, upstream lambda vs flat_grid lambda, 200k calls, best of 5): with `_IS_HIP` False (the CUDA path) the flat_grid form costs 0.057-0.088 us more per launch (one extra Python call and the flag test). With `_IS_HIP` True it costs 0.27-0.29 us more. This matches the commit body's "one module-level truth test more" in magnitude. Before 8592d5f the CUDA path also paid `torch.cuda.is_available()` (0.8 us).
- **Strict per-config driver on the head tree** (`agent_space/review3/allcfg.py`, the porter's driver): `fwd M=4194305 C=64 V=8 cfg2`, grid (2097153, 2) with overrun 1: OK, nan 0. `bwd M=1048576 C=1024 V=8`: 12/12 configs OK.
- **Correction text:** the quoted phrases match the e7d36f1 and 8592d5f bodies verbatim ("This commit removes that prune and changes the launch shape instead.", "The previous commit's message says ..."). In both the squashed and the separate case, the replacement paragraphs are accurate against the code and the measurements above. When the squashed body is assembled, the "fourth paragraph" means e7d36f1's original fourth paragraph, the one beginning "On CUDA the grid, ...". The replacement starts with that same sentence. It does not mean the AI-disclosure line, which becomes the fourth paragraph once the second is dropped.
- **Commit hygiene:** 8592d5f's title is 58 characters with `[ROCm]`, correctly, because it changes only the HIP branch's helper. No `Co-Authored-By` or noreply line in any of the 6 commits, every body carries the AI-agent disclosure and a Test Plan, and the author is the user's own address. `python3 utils/jargon.py --port FlexGEMM` is clean, `utils/prose.py` on the 8592d5f body is clean, and the diff adds no non-ASCII. `git -C projects/FlexGEMM/src status --porcelain` is clean, and `origin/moat-port` is at 8592d5f. The installed `flex_gemm` in the job venv matches the 8592d5f export (`diff -rq`, excluding build output).
- **CUDA compile check:** skipping the nvcc rerun is correct, because 8592d5f changes only Python.

Commands, from the moat checkout, after `source agent_space/env.sh` (HIP_VISIBLE_DEVICES=2, TheRock ROCm 10.2 nightly, torch 2.16.0a0, Triton 3.9.0):

```
for s in 6dd94a8 bcf7c6f e7d36f1 8592d5f; do mkdir -p agent_space/review3/t-$s && git -C projects/FlexGEMM/src archive $s flex_gemm | tar -x -C agent_space/review3/t-$s; done
cd agent_space/review3
for run in 1 2 3 4 5; do for s in 6dd94a8 8592d5f bcf7c6f; do $FLEET_VENV/bin/python launch_overhead.py $PWD/t-$s; done; done
$FLEET_VENV/bin/python gridcost.py $PWD/t-8592d5f
$FLEET_VENV/bin/python allcfg.py $PWD/t-8592d5f fwd 4194305 64 8 2
$FLEET_VENV/bin/python allcfg.py $PWD/t-8592d5f bwd 1048576 1024 8 all
```

## Validation 2026-11-21 (linux-gfx942 fleet job, AMD Instinct MI300X; validated head 8592d5fcb68855864c8a82274d3033ae651f8027)

**Result: pass on linux-gfx942 (wave64).** Every test script and example in the plan's test plan was run at the reviewed head. Every correctness criterion holds. The one example that fails (`examples/spconv.py`) fails before any GPU work because of the known upstream unpack bug (round-1 question 1, still Jeff's call); an agent-side corrected copy passes. No project test fails because of the deferred `flexgemm-grid-sample-bwd-autotune-accumulates`: `tests/utils.py benchmark_kernel` runs 2 warmup plus 20 timed calls and the error is computed on the last call's output, so the first-call accumulation is never compared.

### Environment and build

MI300X gfx942, GPU 2 (dispatcher pin `HIP_VISIBLE_DEVICES=2`), TheRock ROCm 10.2.0a20261022 wheels in the job venv (torch 2.16.0a0+rocm10.2.0a20261022, Triton 3.9.0+gitfc8d31a5.rocm10.2.0a20261022; recipe in "Porting round 3"), `ulimit -c 0`, job-local `HOME` (so `~/.flex_gemm` starts empty: the autotune cache was cold) and `TRITON_CACHE_DIR`. Extra test deps in the venv: `matplotlib` (pip), `spconv-triton 1.0.0` (`pip install --no-deps spconv-triton`).

```
source agent_space/env.sh
rm -rf projects/FlexGEMM/src/build
utils/timeit.sh FlexGEMM compile -- $FLEET_VENV/bin/python -m pip install ./projects/FlexGEMM/src --no-build-isolation --no-deps --force-reinstall   # Successfully installed flex_gemm-1.0.0
```

`projects/FlexGEMM/src` at 8592d5f; `git status --porcelain` clean after the run (the ablation scripts' `tests/submconv/conv_fwd.png` was moved out to `agent_space/`).

Every script ran from its own directory through `utils/timeit.sh FlexGEMM test -- bash -c "cd <dir> && $FLEET_VENV/bin/python <script>"` (runner `agent_space/val/run_all.sh`, logs in `agent_space/val/logs/`, job-local).

### A. Asserting tests (unchanged)

| Script | Result |
|---|---|
| `tests/hashmap.py` | rc 0, RES 1..1024, 4 key/value dtype pairs |
| `tests/serialize.py` | rc 0, z_order + hilbert, cpu + cuda, RES up to 2048 |
| `tests/spconv/out_coords.py` | rc 0, RES up to 2048 |
| `tests/spconv/neighbor_cache.py` | rc 0, RES up to 512, every cache incl. masked |
| `tests/submconv/neighbor_cache.py` | rc 0, RES up to 512 |
| `tests/grid_sample.py` | rc 0, all four tables (Nearest/Trilinear Fwd/Bwd) through RES=2048; all 72 err_max/err_mean cells 0 per mille |

### B. Conv correctness tables (fp16)

Third-party columns: the plan's option 1 (`spconv-triton`) cannot run "unchanged", because the package imports as `spconv_triton` (its README's one-line change). Driver `agent_space/val/run_tp.py` (text below) maps `import spconv.pytorch as spconv` to `import spconv_triton.pytorch as spconv` (and `import spconv as spconv_core` likewise), and removes the torchsparse, fvdb and warpconvnet imports and their `kernel_functions` entries. With `DROP_SPCONV=1` it removes the spconv column too (plan option 2). No fork edit.

- `tests/spconv/fwd.py`: with spconv-triton the script stops in the spconv column, `spconv_triton/pytorch/conv.py:377 AssertionError: only support reuse subm indices` (the script reuses a non-submanifold `indice_dict`; a spconv-triton limit, not FlexGEMM). Rerun with `DROP_SPCONV=1`: rc 0, 14 rows RES 8..512; igemm/igemmk/migemm/migemmk err_max 2 per mille on every row, dense 2 per mille (N/A from RES=256 by design); err_mean 0 everywhere. **Pass** (err_max <= 2x dense, err_mean <= 5).
- `tests/submconv/fwd.py` (spconv-triton column kept): rc 0, 7 rows; spconv and all four algorithms err_max 2, err_mean 0. **Pass.**
- `tests/spconv/bwd.py` (spconv-triton kept): rc 0, 14 rows, 29.4 min cold. err_mean 0 for every algorithm on every row. err_max 1-4 per mille through RES=64; at K=3, RES=128: igemm 11, igemmk 13, migemm 11, migemmk 12 (spconv-triton 15); RES=256: 27/8/15/8 (spconv-triton 10); RES=512: 46/13/27/13 (spconv-triton 11). Dense column 167-4984 per mille. Same pattern as round 1 (igemm 44 at RES=512), and the third-party Triton implementation shows the same magnitude, so this is fp16 accumulation order on the largest weight-gradient reductions, not a fault. **Pass.**
- `tests/submconv/bwd.py` (spconv-triton kept): rc 0, 7 rows; egemm err_max 0; igemm 2-13, igemmk 1-7, migemm 2-10, migemmk 2-5 (spconv-triton 1-7); err_mean 0 everywhere. **Pass.**

### C. Examples and FlexGEMM-only benchmarks

- `examples/submconv.py`: rc 0.
- `examples/spconv.py`: rc 1, `ValueError: too many values to unpack (expected 2)` at line 20, the upstream bug (round-1 question 1). Agent-side copy with only that line corrected (`out_feats, out_coords, neignbor_cache = sparse_conv3d(`; `agent_space/val/example_spconv_fixed.py`, run from `examples/` with `PYTHONPATH=examples`): output (51664, 256), finite, and the backward gradients are finite.
- `tests/submconv/inf_ablation.py` (rc 0) and `train_ablation.py` (rc 0), unchanged, RES 8..512. Inference RES=512,C=128: egemm 29.11 ms, igemm 4.16, igemm_splitk 4.16, migemm 3.11, migemm_splitk 3.11. Training RES=512,C=128: egemm 71.85, igemm 38.17, igemm_splitk 11.84, migemm 40.89, migemm_splitk 8.28; RES=256,C=256: migemm_splitk 6.94 ms.
- `tests/submconv/inf_benchmark.py` and `train_benchmark.py` with the spconv-triton column kept: the timing and memory tables complete, then the plotting code raises `ZeroDivisionError` at `gray = 0.8 - 0.6 * (i / (len(kernels) - 2))` (inf :243, train :246) because only two kernels remain once the CUDA-only columns are removed. That is a driver artifact. Inference times from that run: spconv-triton 5.77-25.15 ms, flex_gemm 1.35-4.04 ms. Rerun with `DROP_SPCONV=1` (one kernel; round 1's setup): both rc 0. flex_gemm inference RES=512,C=128 4.10 ms; training RES=256,C=256 8.89 ms (warm autotune cache).

### Other gates

- CUDA compile gate: the compiled sources (`setup.py`, every `.cu`/`.cuh`, `ext.cpp`) are identical between bcf7c6f and 8592d5f (`git diff --stat bcf7c6f 8592d5f` lists only the three Triton Python files). The real-nvcc check at bcf7c6f (mi355 host, "CUDA compile check" above) therefore covers this head. Not rerun: this host has no CUDA toolkit.
- Documentation: README "AMD GPUs (ROCm)" under Installation (d45c5e2) is present at this head.
- `python3 utils/jargon.py --port FlexGEMM`: clean.

Driver `agent_space/val/run_tp.py`:

```
import os, re, sys
path = os.path.abspath(sys.argv[1])
drop = ["torchsparse", "fvdb", "warpconvnet"]
if os.environ.get("DROP_SPCONV") == "1":
    drop.append("spconv")
alias = {
    "import spconv.pytorch as spconv": "import spconv_triton.pytorch as spconv",
    "import spconv as spconv_core": "import spconv_triton as spconv_core",
}
out = []
for line in open(path).read().splitlines():
    s = line.strip()
    if re.match(r"(import|from)\s+(%s)\b" % "|".join(drop), s) or re.match(r"'(%s)'\s*:\s*\(" % "|".join(drop), s):
        out.append("# [removed] " + line); continue
    if s in alias:
        line = line.replace(s, alias[s])
    out.append(line)
sys.argv = [path] + sys.argv[2:]
sys.path.insert(0, os.path.dirname(path))
exec(compile("\n".join(out) + "\n", path, "exec"), {"__name__": "__main__", "__file__": path})
```

## Install as a dependency

Linux, from the linux-gfx942 validated build at 8592d5f (the Windows and wave32 validators add their own OS's recipe here). FlexGEMM is a PyTorch extension plus Triton kernels. A dependent installs it into the same Python environment as its ROCm PyTorch and Triton:

```
git clone -b moat-port https://github.com/AMD-Ecosystem/FlexGEMM _deps/FlexGEMM/src
export ROCM_HOME=$(python -m rocm_sdk path --root)   # TheRock wheels; omit with a system ROCm
GPU_ARCHS=<gfx> python -m pip install _deps/FlexGEMM/src --no-build-isolation --no-deps
```

- Prerequisites in the environment: a ROCm build of torch (>= 2.14 needs no flag; the C++ standard comes from torch) with its device wheel for `<gfx>`, the matching ROCm Triton, `filelock`, `ninja`, `setuptools`. `GPU_ARCHS` is a `;`-separated list (`gfx942`, `gfx1100;gfx1201`); unset builds for the GPU in the machine.
- Consume with `import flex_gemm` (ops in `flex_gemm.ops.spconv`, `flex_gemm.ops.grid_sample`, `flex_gemm.ops.serialize`). Nothing to link: the HIP extension is `flex_gemm.kernels.cuda`, inside the wheel.
- The autotuner reads and writes `~/.flex_gemm/autotune_cache.json` (override with `FLEX_GEMM_AUTOTUNE_CACHE_PATH`; `FLEX_GEMM_USE_AUTOTUNE_CACHE=0` and `FLEX_GEMM_AUTOSAVE_AUTOTUNE_CACHE=0` turn it off). Point `HOME` or that variable at a job-local path on shared hosts.

## CUDA run on real NVIDIA hardware (2026-11-20, V100, run by the orchestrating session; head 8592d5fc)

Purpose: the port's shared changes (06873dd drops the forced C++17; e7d36f1 and 8592d5f change how the grid_sample launches are shaped) had been compiled with nvcc but never run on a CUDA GPU. Also a CUDA data point for the deferred `flexgemm-grid-sample-bwd-autotune-accumulates`.

Setup: Tesla V100-SXM2-32GB (driver 575.57.08), nvcc 12.9 (V12.9.86), fresh venv with `torch==2.14.1+cu126` (Triton 3.8.0), fork `moat-port` at 8592d5fcb68855864c8a82274d3033ae651f8027, fresh HOME (cold autotune cache) and Triton cache.

```
export PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST="7.0" BUILD_TARGET=cuda
pip install . --no-build-isolation --no-deps      # Successfully installed flex_gemm-1.0.0
cd tests && python hashmap.py && python serialize.py && python grid_sample.py
```

Results:
- Build: succeeds against torch 2.14.1 (the release where upstream main stops building because of its forced C++17).
- `tests/hashmap.py` rc 0 (through res=1024, all key/value dtype pairs); `tests/serialize.py` rc 0 (z_order and hilbert, cpu and cuda, through res=2048).
- `tests/grid_sample.py` rc 0: all four tables (nearest and trilinear, forward and backward) show err_max 0 and err_mean 0 per mille on every row from RES=8,C=2048 to RES=2048,C=8. So the launch helper's CUDA path (a 1-tuple grid, as upstream) gives correct results on CUDA.
- First-call gradient defect, on CUDA: trilinear backward at a shape not in the autotune cache (RES=48, C=8), autograd through `grid_sample_3d` twice against the `grid_sample_3d_torch` reference:

```
max|g1| = 3.415e+04   max|g2| = 7.396   max|ref| = 7.396   ratio g1/g2 = 4617
relerr(first call) = 4.62e+03          relerr(second call) = 1.29e-07
```

  The first call's gradient is about 4600 times too large and the second is correct, so the upstream defect is confirmed on NVIDIA hardware, not only on AMD. `tests/grid_sample.py` does not see it because its benchmark helper calls each kernel several times and compares a later call.

Not run on CUDA: the conv tables (`tests/spconv`, `tests/submconv`) and the examples.

Script (primary host, not in the repo): `agent_space/pn/v100-flexgemm.sh`.

## Validation 2026-11-21 (linux-gfx1100 fleet job, AMD Radeon Pro W7800 48GB, wave32; head 8592d5fcb68855864c8a82274d3033ae651f8027)

**Result: validation-failed on linux-gfx1100, for one port finding.** Every FlexGEMM kernel is correct on wave32. The fault: the four `tests/submconv/{inf,train}_{ablation,benchmark}.py` scripts in plan section C fail before any GPU work on this GPU. They set `flex_gemm.kernels.triton.spconv.config.allow_tf32 = True`, so the spconv kernels call `tl.dot(..., input_precision='tf32')`. Triton 3.9's AMD backend adds `'tf32'` to the allowed precisions only when `arch == 'gfx942'` (`triton/backends/amd/compiler.py:367-371`, "Enable XF32 (TF32) for CDNA3 GPUs"); everywhere else the allowed set is `('ieee', 'bf16x3', 'bf16x6')`. The result is `triton.compiler.errors.CompilationError: ... input_precision must be one of ('ieee', 'bf16x3', 'bf16x6'). Got tf32`, raised from inside autotuning (`Autotuner._bench` catches only `OutOfResources`, `CompileTimeAssertionFailure` and `PTXASError`). It happens with fp16 operands too, because the string is checked before the dtype matters. So these scripts fail on gfx90a, gfx950 and every RDNA part; the gfx942-only evidence hid it. The library default is safe (`allow_tf32 = not torch.version.hip`), but the public knob crashes on every non-gfx942 AMD target. With only that script line flipped to `allow_tf32 = False` (driver switch, no fork edit), all four scripts complete (below), so tf32 is the only obstacle. Porter's choice of fix: for example, pass `'tf32'` only where the target accepts it (the HIP branch of the kernels or `config.py`), or have the scripts follow `config.allow_tf32`. Not fixed here (validator).

### Environment and build

Radeon Pro W7800 48GB, gfx1100, `warp_size` 32, 44.98 GiB visible, GPU 0 (dispatcher pin `HIP_VISIBLE_DEVICES=0`). TheRock ROCm 10.2.0a20261022 wheels in the job venv from `https://nightly.repo.amd.com/rocm/whl-next/`: `torch==2.16.0a0+rocm10.2.0a20261022`, `amd-torch-device-gfx1100==2.16.0a0+rocm10.2.0a20261022`, `triton==3.9.0+gitfc8d31a5.rocm10.2.0a20261022`, `rocm[libraries,devel]==10.2.0a20261022`, `rocm-sdk-device-gfx1100==10.2.0a20261022` (installed with `$FLEET_VENV/bin/python -m pip`). The job's `venv-base-links` entries for the base env's ROCm 10.1 and Triton were moved aside. `python -m rocm_sdk test`: 27 OK. Extra test deps: `matplotlib`, `tqdm`, `spconv-triton 1.0.0` (`--no-deps`). `ulimit -c 0`; job-local `HOME` (the setup.py-installed `~/.flex_gemm/autotune_cache.json` held only the A100 and MI300X entries, so the autotuner started cold for this GPU) and `TRITON_CACHE_DIR`.

```
export ROCM_HOME=$(python -m rocm_sdk path --root) ROCM_PATH=$ROCM_HOME HIP_PATH=$ROCM_HOME GPU_ARCHS=gfx1100 PYTORCH_ROCM_ARCH=gfx1100
utils/timeit.sh FlexGEMM compile -- $FLEET_VENV/bin/python -m pip install -v ./projects/FlexGEMM/src --no-build-isolation --no-deps --force-reinstall   # Successfully installed flex_gemm-1.0.0
```

hipcc got `-std=c++20 --offload-arch=gfx1100 -DHIP_ENABLE_WARP_SYNC_BUILTINS=1`; no compiler warnings. Every script ran from its own directory through `utils/timeit.sh FlexGEMM test -- timeout 5400 bash -c "cd <dir> && $FLEET_VENV/bin/python <script>"`. Third-party columns went through the gfx942 validation's `run_tp.py` driver (spconv to spconv-triton, torchsparse/fvdb/warpconvnet removed, `DROP_SPCONV=1` drops spconv). This run added four switches to it, each of which only comments out or rewrites lines of the test script: `MAX_RES`/`MIN_RES` (drop config rows), `BATCH1` (set `'B'` to 1 on the kept rows), `DROP_EXTRA=dense` (drop the torch dense column) and `TF32_OFF` (the line `allow_tf32 = True` becomes False). `git -C projects/FlexGEMM/src status --porcelain` was clean after the run.

### (1) Deferred `flexgemm-wave-barrier-syncwarp` on wave32: bit-exact, no fault

- **The neighbour-cache tests do not exercise that kernel.** The round-1 note "every neighbour-cache test (which runs `reduce_code_kernel`) passes bit-exact" is wrong. `tests/{spconv,submconv}/neighbor_cache.py` assert `torch.equal` on `neighbor_map` only. `_compute_neighbor_cache` calls `neighbor_map_post_process_for_masked_implicit_gemm_1`/`_1_no_bwd`. `reduce_code_kernel` (the `__syncwarp` loop, `migemm_neighmap_pp.cu:246-297`) runs only in `..._2`, which `compute_kernel_idx` calls lazily from the masked-GEMM conv paths with `block_size = B1`. Both neighbour-cache scripts still pass here (rc 0, RES up to 512; same as wave64).
- **Direct check** (`agent_space/val/reduce_check.py`, job-local). It calls `kernels.cuda.neighbor_map_post_process_for_masked_implicit_gemm_2` and compares with a torch reference: per-block OR of `gray_code[sorted_idx]`, popcount cumsum, and set-bit positions in ascending order. Inputs: 48 sources, namely real `gray_code`/`sorted_idx` from submanifold and strided sparse-conv neighbour caches (fwd and bwd) at RES 8..256, plus random 27-bit, random 32-bit and one-bit-per-element codes at N in {1, 31, 32, 33, 255, 256, 257, 1000, 65537, 2^20}. Block sizes 2..256 (on wave32 every size from 16 up runs the warp-level iterations), 20 repetitions each. **7680/7680 bit-exact.** Negative control: a reference with each block's last element zeroed differs from the kernel in 85 of 155 cases (the rest are dense masks where that element adds no new bit), so the comparison is sensitive.
- **ISA** (gfx1100 code object unbundled from `migemm_neighmap_pp.o`, `llvm-objdump -d`): in the warp-level loop every iteration is `ds_load_b32 buf[tid+cur_len]; ds_load_b32 buf[tid]; s_waitcnt lgkmcnt(0); v_or_b32; ds_store_b32`. The next iteration's loads follow the previous store in program order and are not hoisted above it. LDS operations from one wave complete in order, and on wave32 the branch `threadIdx.x < 32` is exactly one wave. So the weaker `__builtin_amdgcn_wave_barrier()` stand-in is correct with this compiler on wave32, as on wave64.
- The masked GEMMs that consume it (`migemm`, `migemmk`) agree with the other algorithms and with float64 truth in every conv run below. **No wave32 fault; nothing to bring back to Jeff on this item.**

### (2) Triton fp32 configs and the 64 KB LDS limit: nothing fails, nothing near the limit

- `max_shared_mem` reported by Triton for this GPU: 65,536. The spconv list in use is the `'hip'` list (5 configs), because no device key matches "AMD Radeon Pro W7800 48GB".
- Every config forced alone (`agent_space/val/cfg_probe.py`). Each of the 31 spconv `TritonPersistentCacheAutotuner` kernels was given a one-entry config list, so no autotuner could absorb an error, then all four Triton algorithms ran forward plus backward, submanifold and strided (stride 2, padding 1) sparse conv, fp32 and fp16, sphere RES=64, C=256 (N=15,192). Result: **80/80 runs OK**, compared against EXPLICIT_GEMM in float64 (relative error over each output's max, below 1e-3 for fp32 and 2e-2 for fp16).
- Compiled LDS (`CompiledKernel.metadata.shared`, 362 kernels): the largest is **33,792 B**. That is the `B1=256, B2=256, BK=16, num_warps=8, num_stages=2` config at fp32 (32,768-33,792 B; its fp16 build uses 16,384-17,408 B). Triton's AMD backend keeps one LDS stage at `num_stages=2`, so the plan's "64 KB of operand tiles" estimate never materializes.
- What the autotuner would do: `TritonPersistentCacheAutotuner.run` benchmarks through Triton's `Autotuner._bench`, which in Triton 3.9 catches `OutOfResources` (also `CompileTimeAssertionFailure`, `PTXASError`) and scores the config `inf`. So the plan's "FlexGEMM's autotuner does not catch `OutOfResources`" holds only for the single-config path (a one-entry list, or a config loaded from the persisted cache, launches without `_bench`) and if every config fails (`min` then returns an `inf` config, whose launch raises). Neither arises here. Any other compile error (the tf32 finding above) propagates out of autotuning uncaught.
- Autotuned fp32 runs (`agent_space/val/lds_probe.py`, same shapes, normal autotuning): fp32 relative error 3.9e-7 to 3.2e-6 on output, dfeat, dweight and dbias for every algorithm; fp16 3.1e-4 to 5.8e-4.

### (3) Two-dimensional grid_sample launch, strict per-config driver, large shapes

`agent_space/val/allcfg.py` is rebuilt from the round-3 description (the round-3 copy was job-local). It launches the raw JIT kernel (no autotuner) once per config with the `flat_grid` grid. Data: float32, N=65,536, uint32 indices with every 7th row's first neighbour 0xffffffff; forward output prefilled with NaN, backward output zeros; float64 torch reference (chunked), tolerance 1e-5 x reference max, NaN outputs counted. This GPU has 45 GiB, so the round-3 list ran in full, including M=134,217,728:

```
bwd M=1048576   C=1024 V=8: 12/12 OK  grids (3728271, 9)..(4067204, 33), overrun <= 13, err 2.1e-05 (tol 3.7e-04)
bwd M=4194304   C=8    V=8: 12/12 OK  overrun <= 7, err 6.2e-05 (tol 6.4e-04)
bwd M=2097152   C=1024 V=8: 12/12 OK  grids (3947581, 17)..(4129777, 65), overrun <= 49, err 4.5e-05 (tol 5.5e-04)
bwd M=270003    C=1000 V=8: 12/12 OK  overrun <= 8, err 6.9e-06
bwd M=134217728 C=8    V=8: cfg2, cfg11 OK  (4177984, 257) overrun 64, (4067204, 33); err 1.6e-03 (tol 3.6e-03)
fwd M=2097152   C=1024 V=8: 12/12 OK  grids (2796203, 3)..(3728271, 9), overrun <= 7, err 1.7e-06, nan 0
fwd M=270003    C=1000 V=8: 12/12 OK  partial last M tile, err 1.9e-06, nan 0
fwd M=4194303   C=64   V=8: cfg2 OK   (4194303, 1), last unsplit size
fwd M=4194304   C=64   V=8: cfg2 OK   (2097152, 2), first split size
fwd M=4194305   C=64   V=8: cfg2 OK   (2097153, 2), overrun 1
fwd M=134217728 C=8    V=8: cfg2, cfg11 OK  err 1.7e-06, nan 0
```

Negative control: a copy of the installed package with the forward kernel's `block_id = tl.program_id(axis=0)` restored fails `fwd M=4194305 C=64 V=8 cfg2` with nan=134,217,728, the same count as round 3. FlexGEMM's configs are all `num_warps=2`, which is 64 work-items on wave32, so axis 0 (at most 4,194,303 programs) stays far below 2^32 work-items, and the largest launch, (4129777, 65) x 64, is about 2^34 work-items in total.

### A. Asserting tests (unchanged)

| Script | Result |
|---|---|
| `tests/hashmap.py` | rc 0, 8 s |
| `tests/serialize.py` | rc 0, 204 s (z_order + hilbert, cpu + cuda, RES up to 2048) |
| `tests/spconv/out_coords.py` | rc 0, 133 s |
| `tests/spconv/neighbor_cache.py` | rc 0, 25 s |
| `tests/submconv/neighbor_cache.py` | rc 0, 6 s |
| `tests/grid_sample.py` | rc 0, 66 s; all four tables (Nearest/Trilinear Fwd/Bwd) through RES=2048; all 72 err_max/err_mean rows 0 per mille in both columns |

### B. Conv correctness tables (fp16)

On this GPU the err_max column is larger than on gfx942 for every algorithm and for the reference columns alike. It is the distance between two fp16 results of equal accuracy, not a FlexGEMM error. Float64 ground truth (`agent_space/val/refcheck.py`, `bwdcheck.py`; same generators as the tests, batch reduced to fit float64; `calc_err` from `tests/utils.py`, per mille):

```
forward, sparse conv K=3 S=1   egemm(test reference)  igemm   migemmk   truth rounded to fp16
RES=8  C=1024 B=16             10.4                   10.8    8.4       0.5
RES=32 C=1024 B=2              11.1                   11.4    11.4      0.5
RES=64 C=512  B=1               4.3                    4.6    4.6       0.5
submanifold backward (dfeat / dweight / dbias)
RES=8   C=1024 B=16  egemm 10.2/1.3/0.5   igemm=igemmk=migemm=migemmk 11.0/1.2/0.5
RES=64  C=1024 B=1   egemm 10.7/13.5/0.5  igemm..migemmk 10.7/11.2/0.5
RES=128 C=512  B=1   egemm 4.7/79.6/0.5   igemm 4.8/87.0/0.5  igemmk 4.8/87.0  migemm 4.9/87.0  migemmk 4.9/41.2
```

torch's fp16 matmul is accurate here (0.4 per mille relative to max |r|, K up to 27,648). The `min(abs, rel)` metric weighs outputs near magnitude 1, where both fp32-accumulated results carry absolute errors of about 0.01 from cancellation in sums of magnitude about 600. The test reference and every FlexGEMM algorithm sit at the same distance from truth; mean errors are 0.2 per mille.

- `tests/spconv/fwd.py`: with spconv-triton it stops at `spconv_triton/pytorch/conv.py:377 AssertionError: only support reuse subm indices` (as on gfx942). With `DROP_SPCONV=1`: rc 0, 14 rows RES 8..512, 819 s cold. err_max: dense = igemm = igemmk = migemm = migemmk on every row (16-18 per mille at C=1024 K=3, 5-6 at K=2, 8/3 at RES=128, 4/2 at RES=256, 2/2 at RES=512; dense N/A from RES=256 by design); err_mean 0 everywhere. **Pass** (err_max <= 2x dense, err_mean <= 5).
- `tests/submconv/fwd.py` (spconv-triton kept): rc 1 at the last row, RES=512 C=128 B=4. The EXPLICIT_GEMM reference's im2col (N x 27 x 128 fp16, about 40 GB) does not fit this 45 GiB GPU; `benchmark_kernel` returns `None` and the script's ratio raises `TypeError`, which loses the whole table. Rerun with `MAX_RES=256`: rc 0, 6 rows. spconv-triton and all four algorithms equal on every row (17, 16, 18, 15, 8, 4 per mille), err_mean 0. **Pass**; RES=512 not runnable on this card (reference capacity).
- `tests/submconv/bwd.py` (spconv-triton kept): rc 0, 7 rows, 585 s. err_mean 0 for every column on every row. err_max: igemm/igemmk/migemm/migemmk 40-60 per mille through RES=64 (spconv-triton the same, 40-60); RES=128: 186/571/203/571 (spconv-triton 203); RES=256: 324/1591/512/1587 (512); RES=512: 2682/948/2682/971 (1647). The independent third-party Triton implementation shows the same magnitudes, and the float64 check above puts FlexGEMM's gradients level with the test's own reference. **Pass.**
- `tests/spconv/bwd.py`: the full run (rc 1, 1737 s) hit the reference OOM at row 9 (RES=128 K=3 C=512 B=4). The EXPLICIT_GEMM backward alone peaks at 36.8 GiB there (N=256,640, M=856,608, a 22 GiB im2col per pass), and `benchmark_kernel` keeps the previous result alive, so the row cannot fit. Rerun in parts:
  - `MAX_RES=64`: rc 0, 8 rows, 343 s; err_mean 0 for all; err_max igemm 9-65, igemmk 7-65, migemm/migemmk 8-75, spconv-triton 8-75, dense 12-1689 per mille.
  - `MIN_RES=128 MAX_RES=256 BATCH1=1 DROP_EXTRA=dense`: rc 0, 4 rows; err_mean 0. RES=128 K=3: spconv-triton 84, igemm 78, igemmk 121, migemm 84, migemmk 123; K=2: 8-9. RES=256 K=3: 359 / 259 / 1182 / 359 / 1166; K=2: 52 / 29 / 53 / 34 / 52.
  - RES=512 at batch 1 still exceeds 45 GiB in the reference. Not runnable on this card.
  - A `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True` attempt made things worse: mapping OOM, then `hipErrorNoBinaryForGpu` from a later op. Recorded only as what not to try.
  - **Pass** for every row this GPU can hold.

### C. Examples, ablations, benchmarks

- `examples/submconv.py`: rc 0, 12 s.
- `examples/spconv.py`: rc 1, `ValueError: too many values to unpack (expected 2)` at line 20, the upstream bug (round-1 question 1, still Jeff's call). The corrected copy (`agent_space/val/example_spconv_fixed.py`, line 20 only, plus a print) gives output (51664, 256), finite, with finite weight and bias gradients.
- Unchanged, `inf_ablation.py`, `train_ablation.py`, `inf_benchmark.py`, `train_benchmark.py` (the last two with and without the spconv-triton column) all rc 1 in 3-7 s with the tf32 `CompilationError` (the finding above).
- `TF32_OFF=1`:
  - `inf_ablation.py` rc 0 (100 s), RES=512 L=2: egemm 147.63 ms, igemm 37.57, igemm_splitk 37.59, migemm 20.91, migemm_splitk 20.92.
  - `train_ablation.py` rc 0 (147 s), RES=256 C=256: egemm 201.12, igemm 102.55, igemm_splitk 94.41, migemm 58.88, migemm_splitk 53.38; RES=512: migemm_splitk 60.87.
  - `inf/train_benchmark.py` with spconv-triton: tables complete, then the known plotting `ZeroDivisionError` (driver artifact, as on gfx942). Inference RES=256: spconv-triton 32.40 ms, flex_gemm 20.39; RES=512: 40.02 / 23.70. Training RES=256: 94.98 / 53.45.
  - With `DROP_SPCONV=1`: both rc 0; flex_gemm inference RES=512 23.53 ms, training RES=256 53.16 ms.
  - These are the RDNA tuning baseline from the `'hip'` list, not a gate.

### Other gates

- CUDA compile gate: covered at this head by the real-nvcc check at bcf7c6f (the compiled sources are identical, per the gfx942 validation) and by the V100 run at 8592d5fc. This host has no CUDA toolkit.
- Documentation: README "AMD GPUs (ROCm)" is present at this head and names `gfx1100;gfx1201` in its `GPU_ARCHS` example.
- `python3 utils/jargon.py --port FlexGEMM`: clean (needed local `main`/`moat-port` branches in the fork clone).

## Lesson for cuda-to-rocm: Triton on AMD accepts input_precision='tf32' only on gfx942

Promoted to cuda-to-rocm/references/platform-limits.md in 0bd85094.

Belongs where Triton porting faults are listed. Symptom: a Triton kernel that runs on CUDA and on MI300X fails to compile on every other AMD GPU (MI200, MI355, every RDNA part) with `CompilationError: ... input_precision must be one of ('ieee', 'bf16x3', 'bf16x6'). Got tf32`. It is raised from inside autotuning, which does not catch it, and it happens even when the `tl.dot` operands are fp16. Cause: Triton's AMD backend (3.9; `backends/amd/compiler.py`, "Enable XF32 (TF32) for CDNA3 GPUs") adds `'tf32'` to the allowed dot precisions only when the target arch is `gfx942`. A project that exposes an `allow_tf32` switch and passes `input_precision='tf32' if allow_tf32 else 'ieee'` breaks wherever a user, a benchmark script or a default turns the switch on, so validating only on MI300X hides it. Fix: choose the precision from the target, for example `'tf32'` only off HIP or when `triton.runtime.driver.active.get_current_target().arch == 'gfx942'`, and `'ieee'` otherwise; do not rely on the default alone. Better than naming gfx942: ask Triton for the target's accepted set once on the host, `'tf32' in triton.compiler.make_backend(triton.runtime.driver.active.get_current_target()).parse_options({}).allowed_dot_input_precisions`, which follows Triton's own per-target rule (and honours `TRITON_OVERRIDE_ARCH`), and pass the result into the kernels as a module-level `tl.constexpr`. Triton's deprecated `tl.dot(allow_tf32=...)` makes the same choice in-kernel, but `TRITON_F32_DEFAULT` overrides it, which would change the CUDA path. (FlexGEMM)

## Porting round 4 (2026-11-21, linux-gfx1100 fleet job, AMD Radeon Pro W7800 48GB, wave32)

Answers the gfx1100 validation finding (`allow_tf32 = True` crashes on every AMD target except gfx942). New fork head **ab4245582c7ecdbe9ed687577d00c39d14f9de39**, one commit on top of 8592d5f, pushed plainly to `moat-port` (no amend, rebase or force-push). Title `[ROCm] spconv: request tf32 only where Triton accepts it` (56 characters). 13 files, +37 / -20. `jargon.py --port FlexGEMM` and `prose.py` on the body: clean.

### Design

- **Where:** `flex_gemm/kernels/triton/spconv/config.py`, the module that owns `allow_tf32`. At import it computes `TF32_PRECISION = tl.constexpr(_tf32_input_precision())`. The 20 `tl.dot` calls (12 kernel files, every `tl.dot` in the package) change `input_precision='tf32' if allow_tf32 else 'ieee'` to `input_precision=config.TF32_PRECISION if allow_tf32 else 'ieee'`; every one of those files already had `from . import config`. Trailing `# (B1, B2)` comments keep their column. Nothing is evaluated per launch: the constexpr is resolved when Triton compiles a kernel, and the precision is a compile-time constant in the binary.
- **How the target is learned (no arch list copied):** on HIP, `make_backend(triton.runtime.driver.active.get_current_target()).parse_options({}).allowed_dot_input_precisions`, then `'tf32'` if listed, else `'ieee'`. `triton.compiler.make_backend` is in `triton.compiler`'s public names; `parse_options` is exactly what the compiler runs, so the answer matches the compile (it also honours `TRITON_OVERRIDE_ARCH`). Shown on this host, import only: no override -> `ieee`; `TRITON_OVERRIDE_ARCH=gfx942` -> `tf32`; gfx90a, gfx950, gfx1201 -> `ieee`.
- **CUDA provably unchanged:** `_tf32_input_precision()` returns `'tf32'` immediately when `torch.version.hip` is falsy, before any Triton query, so `config.TF32_PRECISION if allow_tf32 else 'ieee'` is the upstream expression with the same value; no new device or driver call on CUDA. The default `allow_tf32 = not torch.version.hip` is untouched (still `'ieee'` by default on every AMD GPU, including gfx942).
- **Semantics:** `allow_tf32` stays a permission (like `torch.backends.cuda.matmul.allow_tf32`): reading it back still gives what the user set. The autotune key still contains `allow_tf32`, so on a non-tf32 AMD target the on and off settings tune separately to the same binaries; harmless.
- **Rejected:** Triton's deprecated `tl.dot(..., allow_tf32=...)` does the same choice in-kernel, but `TRITON_F32_DEFAULT` overrides it, which would change the CUDA path. Editing the 33 wrapper call sites (`allow_tf32=config.allow_tf32 and ...`) touches more lines and still needs the one-time check. A property on the config module would change attribute semantics on CUDA too.
- Test scripts not edited.

### Evidence (gfx1100, GPU 0, TheRock ROCm 10.2.0a20261022 wheels as in the validation section, job-local HOME and TRITON_CACHE_DIR, `ulimit -c 0`)

Fresh job venv: `$FLEET_VENV/bin/python -m pip install --index-url https://nightly.repo.amd.com/rocm/whl-next/ --extra-index-url https://pypi.org/simple torch==2.16.0a0+rocm10.2.0a20261022 amd-torch-device-gfx1100==2.16.0a0+rocm10.2.0a20261022 triton==3.9.0+gitfc8d31a5.rocm10.2.0a20261022 "rocm[libraries,devel]==10.2.0a20261022" rocm-sdk-device-gfx1100==10.2.0a20261022 matplotlib tqdm` (the per-arch `gfx110X-all/` index does not carry these; the flat index does), plus `spconv-triton==1.0.0 --no-deps`; the job's `venv-base-links` ROCm 10.1 / Triton entries moved aside first. Build: `pip install -v ./projects/FlexGEMM/src --no-build-isolation --no-deps --force-reinstall` with `ROCM_HOME=$(python -m rocm_sdk path --root)`, `GPU_ARCHS=gfx1100`: Successfully installed, before and after the change.

- **Reproduced at 8592d5f:** `inf_ablation.py`, `train_ablation.py` (run unmodified from `tests/submconv`) and `inf_benchmark.py`, `train_benchmark.py` (through the third-party driver with `DROP_SPCONV=1`, no TF32 switch) all rc 1 in seconds with `CompilationError ... input_precision must be one of ('ieee', 'bf16x3', 'bf16x6'). Got tf32`.
- **At ab42455, same commands, `allow_tf32 = True` as written:** all four rc 0.
  - `inf_ablation.py` (fp16): RES=512 C=128 L=2 egemm 145.22 ms, igemm 36.47, igemm_splitk 36.17, migemm 19.75, migemm_splitk 20.62 (round-3 validation with TF32_OFF: 147.63 / 37.57 / 37.59 / 20.91 / 20.92).
  - `train_ablation.py`: RES=256 C=256 egemm 199.35, igemm 103.42, igemm_splitk 93.02, migemm 58.31, migemm_splitk 53.57; RES=512 migemm_splitk 59.19.
  - `inf_benchmark.py`: flex_gemm RES=256 20.44 ms, RES=512 23.77 ms. `train_benchmark.py`: RES=256 55.39 ms. Same as the TF32_OFF numbers of the validation, as expected for identical binaries.
- **Conv correctness for the precision actually used (`ieee` on this GPU)**, driver `tf32check.py` (text below), sphere RES=64 C=256 N=15,192, K=3; each Triton algorithm run with `allow_tf32` False then True, forward + backward; submanifold and strided (stride 2, padding 1); reference EXPLICIT_GEMM in float64. Relative error = max |x - ref| / max |ref|, per output (out, dfeat, dweight, dbias):

```
kind    dtype    algo                         on==off  relerr vs float64
subm    float32  implicit_gemm                True     2.5e-06 2.9e-06 4.7e-06 1.7e-07
subm    float32  implicit_gemm_splitk         True     1.1e-06 1.2e-06 9.6e-07 1.7e-07
subm    float32  masked_implicit_gemm         True     2.5e-06 2.6e-06 4.7e-06 1.7e-07
subm    float32  masked_implicit_gemm_splitk  True     1.1e-06 1.1e-06 9.6e-07 1.7e-07
subm    float16  (all four)                   True     4.0e-04..5.5e-04 4.6e-04 4.9e-04 3.9e-04
strided float32  implicit_gemm                True     2.1e-06 2.5e-06 2.3e-06 1.1e-07
strided float32  implicit_gemm_splitk         True     2.1e-06 2.5e-06 1.1e-06 1.1e-07
strided float32  masked_implicit_gemm         True     2.1e-06 2.5e-06 2.3e-06 1.1e-07
strided float32  masked_implicit_gemm_splitk  True     7.2e-07 2.5e-06 7.0e-07 1.1e-07
strided float16  (all four)                   True     4.1e-04..5.8e-04 4.6e-04 4.8e-04..4.9e-04 3.3e-04
FAILURES 0
```

16/16 bitwise equal on vs off; fp32 at ieee accuracy (about 1e-6), fp16 at fp16 accuracy. (A first run of the driver reused the fp32 leaf tensor across runs and accumulated gradients; fixed with `.detach().clone()`, not a library issue.)
- `git -C projects/FlexGEMM/src status --porcelain` clean after the runs (the benchmark's `conv_fwd.png` output removed).
- Not rerun here: CUDA compile (no CUDA toolkit on this host; the CUDA sources are untouched, only Triton Python changed, and the CUDA value of the expression is the upstream one by construction). The real-nvcc and V100 evidence at earlier heads covers the compiled sources; a V100 run of a Triton spconv conv with `allow_tf32 = True` at ab42455 would confirm the Triton side, and is the remaining CUDA check for whoever has the hardware.

Driver `run_tp.py` (round-3 text plus switches `MAX_RES`, `MIN_RES`, `BATCH1`, `DROP_EXTRA`, `TF32_OFF`, each only commenting out or rewriting lines of the script):

```
import os, re, sys
path = os.path.abspath(sys.argv[1])
drop = ["torchsparse", "fvdb", "warpconvnet"]
if os.environ.get("DROP_SPCONV") == "1":
    drop.append("spconv")
extra = [e for e in os.environ.get("DROP_EXTRA", "").split(",") if e]
alias = {
    "import spconv.pytorch as spconv": "import spconv_triton.pytorch as spconv",
    "import spconv as spconv_core": "import spconv_triton as spconv_core",
}
max_res = int(os.environ.get("MAX_RES", "1000000"))
min_res = int(os.environ.get("MIN_RES", "0"))
batch1 = os.environ.get("BATCH1") == "1"
out = []
for line in open(path).read().splitlines():
    s = line.strip()
    if re.match(r"(import|from)\s+(%s)\b" % "|".join(drop), s) or re.match(r"'(%s)'\s*:\s*\(" % "|".join(drop + extra), s):
        out.append("# [removed] " + line); continue
    m = re.match(r"\{'RES':\s*(\d+),", s)
    if m:
        if not (min_res <= int(m.group(1)) <= max_res):
            out.append("# [removed] " + line); continue
        if batch1:
            line = re.sub(r"'B':\s*\d+", "'B': 1", line)
    if s in alias:
        line = line.replace(s, alias[s])
    if os.environ.get("TF32_OFF") == "1" and s == "allow_tf32 = True":
        line = line.replace("True", "False")
    out.append(line)
sys.argv = [path] + sys.argv[2:]
sys.path.insert(0, os.path.dirname(path))
exec(compile("\n".join(out) + "\n", path, "exec"), {"__name__": "__main__", "__file__": path})
```

Driver `tf32check.py <tests dir> <RES>`:

```
import sys, torch
sys.path.insert(0, sys.argv[1])
from utils import sphere_coords, calc_err
from flex_gemm.ops import spconv as ops
from flex_gemm.ops.spconv import sparse_submanifold_conv3d, sparse_conv3d
from flex_gemm.kernels.triton.spconv import config
RES, C = int(sys.argv[2]), 256
A = ops.Algorithm
ALGOS = [A.IMPLICIT_GEMM, A.IMPLICIT_GEMM_SPLITK, A.MASKED_IMPLICIT_GEMM, A.MASKED_IMPLICIT_GEMM_SPLITK]
torch.manual_seed(0)
f32, coords, shape = sphere_coords(RES, C, 1, dtype=torch.float32)
w32 = torch.randn(C, 3, 3, 3, C, device='cuda') * 0.05
b32 = torch.randn(C, device='cuda')
def run(kind, algo, dtype, tf32):
    config.allow_tf32 = tf32
    ops.set_algorithm(algo)
    f = f32.detach().clone().to(dtype).requires_grad_()
    w = w32.detach().clone().to(dtype).requires_grad_()
    b = b32.detach().clone().to(dtype).requires_grad_()
    if kind == 'subm':
        out, _ = sparse_submanifold_conv3d(f, coords, shape, w, b)
    else:
        out, _, _ = sparse_conv3d(f, coords, shape, w, b, stride=(2, 2, 2), padding=(1, 1, 1))
    g = torch.Generator(device='cuda').manual_seed(1)
    out.backward(torch.randn(out.shape, device='cuda', generator=g).to(dtype))
    return [out.detach(), f.grad, w.grad, b.grad]
# for kind in subm, strided: ref = run(kind, EXPLICIT_GEMM, float64, False);
# for dtype in fp32, fp16 and each algo: off/on = run(..., False/True);
# same = all(torch.equal); rel = max|on-ref|/max|ref|; fail if not same or rel >= 1e-4 (fp32) / 2e-2 (fp16)
```

## Review 2026-11-21 round 4 (linux-gfx1100 fleet job, AMD Radeon Pro W7800; reviewed head ab4245582c7ecdbe9ed687577d00c39d14f9de39)

Scope: the delta `git diff 8592d5f..ab42455` (one commit, `[ROCm] spconv: request tf32 only where Triton accepts it`, 13 files, +37/-20) and its message. Verdict: **review-passed**, no findings.

Checked, each against the code (Triton 3.9.0 in the job venv):

- CUDA path: `_tf32_input_precision()` (`flex_gemm/kernels/triton/spconv/config.py:12-21`) returns `'tf32'` before any Triton query when `torch.version.hip` is falsy, so every `tl.dot` requests what upstream requested. The only new CUDA-side work is `from triton.compiler import make_backend` at import, a public name of `triton.compiler` since the 3.0 backend refactor (the project requires `triton>=3.2.0`).
- Coverage: all 20 `tl.dot` calls in the package (12 spconv kernel files; `grid_sample` has none) use `config.TF32_PRECISION if allow_tf32 else 'ieee'`, and each file already imports `config`. Nothing orphaned.
- The query: `HIPBackend.parse_options` (`triton/backends/amd/compiler.py:356-371`) is the function the compiler runs, and it adds `'tf32'` only for `gfx942`. `get_current_target()` (`backends/amd/driver.py:371-376`) honours `TRITON_OVERRIDE_ARCH`. Reproduced here: no override -> `ieee`; overrides gfx942 / gfx90a / gfx950 / gfx1201 -> `tf32` / `ieee` / `ieee` / `ieee`, matching the commit's Test Plan.
- JIT behaviour of a module attribute: `config` is a module global, which `DependenciesFinder.record_reference` skips (`runtime/jit.py:120`), and the attribute's value is not added to `used_global_vals`, so it is not part of the kernel cache key. That is safe: the value is fixed for the process at import and depends only on the target arch, which is already in the compile cache key (and so is an override arch). `CodeGenerator.visit_Attribute` resolves the module attribute to the `tl.constexpr`, and the `if allow_tf32` IfExp picks a branch at compile time.
- Import side effects: the new query touches the device at import on HIP. 8592d5f already did that (`config.py` -> `get_autotune_config` -> `get_gpu_name` -> `torch.cuda.get_device_name`): both heads initialize torch.cuda on `import flex_gemm`, and both fail to import with no visible GPU (checked by importing the 8592d5f Python sources over the installed extension). No new behaviour change. A process that mixes AMD archs gets device 0's answer, as upstream's autotune config already does.
- Knob check rerun on this GPU with the porter's `tf32check.py` at RES=32 (installed `flex_gemm` Python matches the fork tree at ab42455): 16/16 bitwise equal with `allow_tf32` on and off, fp32 rel err <= 3.4e-6 and fp16 <= 7.0e-4 against float64, `FAILURES 0`.
- Commit message: title 56 characters, `[ROCm]` is right (the added lines branch on HIP), AI assistance disclosed, no trailer, Test Plan in fenced blocks; its claims (gfx942-only in 3.9, the error text, the CUDA value unchanged) match the code. `python3 utils/jargon.py --port FlexGEMM`: clean.
- `## Lesson for cuda-to-rocm: Triton on AMD accepts input_precision='tf32' only on gfx942`: accurate for Triton 3.9. The `TRITON_F32_DEFAULT` remark is confirmed by `tl.dot` (`language/core.py:2267-2270`: `knobs.language.fp32_default` takes precedence when `input_precision` is None).

Still open for later stages, not a review finding: no NVIDIA run of a Triton spconv conv with `allow_tf32 = True` at ab42455 yet (the CUDA value is the upstream expression by construction). The gfx1100 platform needs its revalidation at this head.

## Validation 2026-11-21 (linux-gfx1100 fleet job, AMD Radeon Pro W7800 48GB, wave32; validated head ab4245582c7ecdbe9ed687577d00c39d14f9de39)

**Result: completed on linux-gfx1100.** Full wave32 rerun at the reviewed head after porting round 4. The tf32 finding is fixed: all four `tests/submconv/{inf,train}_{ablation,benchmark}.py` scripts complete with `allow_tf32 = True` as written. Every other result matches the 8592d5f wave32 validation above. The three special investigations from that validation (`flexgemm-wave-barrier-syncwarp` on wave32, LDS limits of the Triton fp32 configs, the 2-D grid_sample launch) were not repeated. Their conclusions still hold, because the delta only changes the `input_precision` string the spconv `tl.dot` calls compile with, and on this GPU that string is `'ieee'` whichever way the knob is set, the same value 8592d5f compiled by default.

Environment: GPU 0 (pin `HIP_VISIBLE_DEVICES=0`), the job venv with the TheRock ROCm 10.2.0a20261022 wheels (install command in "Porting round 4"), `spconv-triton 1.0.0`, `ulimit -c 0`, job-local `HOME` and `TRITON_CACHE_DIR`. The autotune cache was warm for the shapes the porter and reviewer runs had already used and cold for the rest. Clean build: `git clean -xdf` in the fork clone, then `utils/timeit.sh FlexGEMM compile -- pip install -v ./projects/FlexGEMM/src --no-build-isolation --no-deps --force-reinstall` (with `ROCM_HOME=$(python -m rocm_sdk path --root)` and `GPU_ARCHS=gfx1100`). It succeeded, with 6 hipcc compiles at `--offload-arch=gfx1100 -std=c++20` and no compiler warnings. `python -m rocm_sdk test`: OK. Each script ran from its own directory through `utils/timeit.sh FlexGEMM test -- timeout 5400 ...`. Third-party columns went through `run_tp.py` (text in "Porting round 4").

### A. Asserting tests

| Script | Result |
|---|---|
| `tests/hashmap.py` | rc 0, 7 s |
| `tests/serialize.py` | rc 0, 204 s |
| `tests/spconv/out_coords.py` | rc 0, 118 s |
| `tests/spconv/neighbor_cache.py` | rc 0, 23 s |
| `tests/submconv/neighbor_cache.py` | rc 0, 5 s |
| `tests/grid_sample.py` | rc 0, 63 s; every err_max and err_mean row 0 per mille in both columns |

### B. Conv correctness tables (fp16, per mille)

- `tests/submconv/fwd.py` (spconv-triton kept, `MAX_RES=256`; RES=512 exceeds this card in the EXPLICIT_GEMM reference, as before): rc 0, 91 s. err_max: spconv-triton = igemm = igemmk = migemm = migemmk on every row (16, 15, 15, 16, 8, 3). err_mean 0. **Pass.**
- `tests/submconv/bwd.py` (spconv-triton kept): rc 0, 310 s, 7 rows. err_mean 0 everywhere. err_max for spconv-triton / igemm / igemmk / migemm / migemmk: RES 8-64 all equal, 44-66. RES=128: 214 / 187 / 655 / 214 / 655. RES=256: 462 / 368 / 1198 / 462 / 1198. RES=512: 1734 / 2742 / 904 / 1778 / 940. Same pattern and magnitude as at 8592d5f, where the float64 check showed this is fp16 distance between equally accurate results. **Pass.**
- `tests/spconv/fwd.py` (`DROP_SPCONV=1`): rc 0, 844 s, 14 rows RES 8..512. err_max for dense = igemm = igemmk = migemm = migemmk on every row (16 at K=3 through RES=64, 6-7 at K=2, 9/3 at RES=128; from RES=256 on, where dense is N/A by design, 3/2 and 2/2). err_mean 0. **Pass.**
- `tests/spconv/bwd.py`:
  - `MAX_RES=64`: rc 0, 1330 s, 8 rows. err_mean 0 for every FlexGEMM column. err_max: igemm 7-68, igemmk 7-68, migemm and migemmk 6-73, spconv-triton 6-73, dense 9-1313.
  - `MIN_RES=128 MAX_RES=256 BATCH1=1 DROP_EXTRA=dense`: rc 0, 286 s. err_mean 0. RES=128 K=3: spconv-triton 72, igemm 67, igemmk 123, migemm 72, migemmk 122. RES=128 K=2: 8-11. RES=256 K=3: 385 / 296 / 1086 / 385 / 1102. RES=256 K=2: 35-51.
  - RES=512 does not fit this card in the reference, as before. **Pass** for every row this GPU can hold.
- Knob on vs off (porter's `tf32check.py`, RES=64, fp32 and fp16, forward and backward, submanifold and strided conv): 16/16 bitwise equal, and the fp32 results are within 1e-6 of a float64 reference ("Porting round 4"). The reviewer reran it at RES=32 with the same result.

### C. Examples, ablations, benchmarks

- `examples/submconv.py`: rc 0.
- `examples/spconv.py`: rc 1, `ValueError: too many values to unpack (expected 2)`. This is the known upstream bug (round-1 question 1, still Jeff's call), unchanged.
- `inf_ablation.py`: rc 0 (25 s), unmodified with `allow_tf32 = True`. RES=512 L=2: egemm 146.03 ms, igemm 36.18, igemm_splitk 36.15, migemm 21.32, migemm_splitk 21.25.
- `train_ablation.py`: rc 0 (64 s), unmodified. RES=256: 200.64 / 102.95 / 94.74 / 58.91 / 54.73 ms. RES=512: migemm_splitk 60.13 ms.
- `inf_benchmark.py` and `train_benchmark.py` with `DROP_SPCONV=1`: both rc 0. flex_gemm inference was 20.36 ms at RES=256 and 23.67 ms at RES=512; training was 54.72 ms at RES=256.
- The same two scripts with the spconv-triton column kept: the tables complete, then the known plotting `ZeroDivisionError` (a driver artifact). Inference at RES=256 was 31.64 ms for spconv-triton and 19.99 ms for flex_gemm; at RES=512, 41.27 / 23.36. Training at RES=256 was 96.07 / 55.68.

### Other gates

- CUDA compile gate: no compiled source changed (`git diff --stat 8592d5f ab42455 -- '*.cu' '*.cuh' '*.cpp' '*.h' setup.py` is empty; the delta is Triton Python only), so the real-nvcc check at bcf7c6f still covers the compiled sources. This host has no CUDA toolkit. On CUDA, `TF32_PRECISION` is `'tf32'` by construction, so the kernels make upstream's request. No NVIDIA run of a Triton spconv conv with `allow_tf32 = True` has been done at this head (`cuda-not-validated` for that runtime path: no NVIDIA GPU here).
- Documentation: README "AMD GPUs (ROCm)" is present and unchanged by the delta.
- `python3 utils/jargon.py --port FlexGEMM`: clean (the porter checked before the push, the reviewer checked at this head).
- `git -C projects/FlexGEMM/src status --porcelain`: clean after the run (benchmark PNGs removed).

## CUDA run of the Triton spconv path with `allow_tf32` on (2026-11-21, head ab4245582c7ecdbe9ed687577d00c39d14f9de39)

Closes the open item from the round-4 review and the linux-gfx1100 validation ("no NVIDIA run of a Triton spconv conv with `allow_tf32 = True` at ab42455"). Run by the dispatching session on a Tesla V100-SXM2-32GB (driver 575.57.08, CUDA toolkit 12.9, torch 2.14.1+cu126, Triton 3.8.0), `BUILD_TARGET=cuda`, `TORCH_CUDA_ARCH_LIST=7.0`, a clean clone of `moat-port` at ab42455 built with `pip install . --no-build-isolation --no-deps`, job-local `HOME` and `TRITON_CACHE_DIR` (cold autotune cache).

- `config.TF32_PRECISION` is `constexpr[tf32]` on CUDA: the kernels make upstream's request.
- A submanifold conv (RES=32, C=64, fp32, forward and backward) with the knob off and on, relative error of out / dfeats / dweight / dbias against a float64 EXPLICIT_GEMM reference:

```
implicit_gemm               tf32 False  1.05e-06 1.08e-06 1.93e-06 1.19e-07
implicit_gemm               tf32 True   1.05e-06 1.08e-06 1.93e-06 1.19e-07
masked_implicit_gemm_splitk tf32 False  2.70e-07 2.58e-07 5.07e-07 1.19e-07
masked_implicit_gemm_splitk tf32 True   2.70e-07 2.75e-07 5.07e-07 1.19e-07
```

  Both settings compile and run. This GPU has no tensor cores, so `tf32` does not lower precision here; the run shows the request is accepted and the result is correct, not what TF32 accuracy looks like on newer NVIDIA hardware.
- `tests/submconv/inf_ablation.py` unmodified (`allow_tf32 = True`): 3 of 7 resolutions finished without error in 22 minutes of cold autotuning (478 s, 284 s, 574 s per resolution), then the 25-minute cap ended the run. Not a failure of the port; a complete run needs about an hour on this GPU and was not repeated.
## Validation 2026-11-21 (linux-gfx942 fleet job, AMD Instinct MI300X, wave64; validated head ab4245582c7ecdbe9ed687577d00c39d14f9de39)

**Result: completed on linux-gfx942.** Full wave64 rerun at the reviewed head after porting round 4. No carry-forward: `classify 8592d5f ab42455` says `mixed`, and gfx942 is the one AMD target where the delta changes the compiled kernels (`config.TF32_PRECISION` resolves to `constexpr[tf32]` here; it is `ieee` on gfx1100). Every test script and example from the 8592d5f gfx942 validation gives the same result. All four `tests/submconv/{inf,train}_{ablation,benchmark}.py` scripts run with `allow_tf32 = True` as written. With the knob on, fp32 conv results are within tf32 accuracy of a float64 reference. The porter's `tf32check.py` pass criterion does not hold on this GPU, which is expected: it asks for knob-on and knob-off to be bitwise equal and for fp32 error below 1e-4, and both conditions only hold where tf32 is unavailable. Numbers below.

### Environment and build

MI300X gfx942, GPU 3 (dispatcher pin `HIP_VISIBLE_DEVICES=3`), `ulimit -c 0`, job-local `HOME` (cold `~/.flex_gemm` autotune cache) and `TRITON_CACHE_DIR`. Fresh job venv: the job's `venv-base-links` ROCm 10.1 / Triton entries moved aside, then `$FLEET_VENV/bin/python -m pip install --index-url https://nightly.repo.amd.com/rocm/whl-next/ --extra-index-url https://pypi.org/simple torch==2.16.0a0+rocm10.2.0a20261022 amd-torch-device-gfx942==2.16.0a0+rocm10.2.0a20261022 triton==3.9.0+gitfc8d31a5.rocm10.2.0a20261022 "rocm[libraries,devel]==10.2.0a20261022" rocm-sdk-device-gfx942==10.2.0a20261022 matplotlib tqdm filelock ninja setuptools`, plus `pip install --no-deps spconv-triton==1.0.0`. `python -m rocm_sdk test`: OK.

```
git clone --recurse-submodules -b moat-port https://github.com/AMD-Ecosystem/FlexGEMM projects/FlexGEMM/src   # HEAD ab42455
export ROCM_HOME=$($FLEET_VENV/bin/python -m rocm_sdk path --root) GPU_ARCHS=gfx942
utils/timeit.sh FlexGEMM compile -- timeout 1800 $FLEET_VENV/bin/python -m pip install -v ./projects/FlexGEMM/src --no-build-isolation --no-deps --force-reinstall   # Successfully installed flex_gemm-1.0.0; 6 hipcc compiles at --offload-arch=gfx942
```

Every script ran from its own directory as `utils/timeit.sh FlexGEMM test -- timeout 5400 env <switches> bash -c "cd <dir> && $FLEET_VENV/bin/python <script>"`. Third-party columns went through `run_tp.py`, using the "Porting round 4" text unchanged.

### A. Asserting tests

| Script | Result |
|---|---|
| `tests/hashmap.py` | rc 0, 5 s |
| `tests/serialize.py` | rc 0, 219 s |
| `tests/spconv/out_coords.py` | rc 0, 31 s |
| `tests/spconv/neighbor_cache.py` | rc 0, 10 s |
| `tests/submconv/neighbor_cache.py` | rc 0, 4 s |
| `tests/grid_sample.py` | rc 0, 74 s; every err_max/err_mean cell 0 per mille through RES=2048 |

### B. Conv correctness tables (fp16, per mille)

These scripts do not set `allow_tf32`, so they run at the library default (`ieee` on HIP), as at 8592d5f.

- `tests/spconv/fwd.py` (`DROP_SPCONV=1`; spconv-triton stops on the known `only support reuse subm indices` limit): rc 0, 478 s, 14 rows RES 8..512. err_max for igemm, igemmk, migemm and migemmk is 1-2 on every row, and dense is 2 (N/A from RES=256 on, by design). err_mean 0 everywhere. **Pass.**
- `tests/spconv/bwd.py` (spconv-triton kept): rc 0, 1548 s, 14 rows. err_mean 0 for every non-dense column. err_max is 1-5 through RES=64. Values for spconv-triton / igemm / igemmk / migemm / migemmk at K=3: RES=128 14 / 12 / 14 / 16 / 14; RES=256 8 / 20 / 8 / 11 / 8; RES=512 9 / 49 / 9 / 27 / 8. K=2 rows are 1-3. The dense column is 150-4438. This is the 8592d5f pattern (igemm 46 at RES=512 then): fp16 accumulation order on the largest weight-gradient reductions. **Pass.**
- `tests/submconv/fwd.py` (spconv-triton kept): rc 0, 61 s, 7 rows. err_max is 2 for spconv-triton and every algorithm (migemmk 1-2). err_mean 0. **Pass.**
- `tests/submconv/bwd.py` (spconv-triton kept): rc 0, 306 s, 7 rows. egemm err_max 0; spconv-triton 1-6, igemm 2-17, igemmk 2-5, migemm 2-21, migemmk 2-4; err_mean 0 everywhere. The largest values (igemm 17, migemm 21 at RES=512) are slightly above 8592d5f's (13, 10). That run used the same default `ieee` path, so the compiled kernels are unchanged. The difference comes from cold-cache autotune picking different tile configurations, which changes the fp16 accumulation order. **Pass.**

### C. Examples, ablations, benchmarks (tf32 knob on)

- `examples/submconv.py`: rc 0.
- `examples/spconv.py`: rc 1, `ValueError: too many values to unpack (expected 2)` at line 20. This is the known upstream bug (round-1 question 1), unchanged. An agent-side copy with only that line corrected, run from `examples/` with `PYTHONPATH=.`: rc 0, forward and backward.
- `inf_ablation.py`: rc 0 (292 s, cold autotune), unmodified with `allow_tf32 = True`. RES=512 C=128 L=2: egemm 28.82 ms, igemm 4.16, igemm_splitk 4.17, migemm 2.88, migemm_splitk 2.89 (8592d5f knob-off: 29.11 / 4.16 / 4.16 / 3.11 / 3.11).
- `train_ablation.py`: rc 0 (647 s), unmodified. RES=256 C=256: 36.71 / 14.18 / 9.89 / 13.76 / 6.89 ms. RES=512 C=128: 71.18 / 38.19 / 11.77 / 40.72 / 8.08 (8592d5f: migemm_splitk 6.94 at RES=256 and 8.28 at RES=512).
- `inf_benchmark.py` and `train_benchmark.py` with `DROP_SPCONV=1`: both rc 0. flex_gemm inference 3.01 ms at RES=256 and 3.78 ms at RES=512; training 7.32 ms at RES=256.
- The same two scripts with the spconv-triton column kept: the timing and memory tables complete, then the plotting code raises the known `ZeroDivisionError` (inf :243, train :246; a driver artifact). Inference was 5.52 / 2.99 ms (spconv-triton / flex_gemm) at RES=256 and 8.40 / 3.76 at RES=512; training was 17.88 / 7.39 at RES=256. The script also sets `SPCONV_ALLOW_TF32` for the spconv-triton column.
- The ablation and benchmark scripts are fp16 (`DTYPE = torch.float16`), so on this GPU the knob changes the requested precision string and the autotune key, but not the arithmetic (see the fp16 rows below).

### D. tf32 knob on vs off (porter's `tf32check.py`, RES=64, N=15,192, C=256, K=3)

Driver: the head and loop are as described in "Porting round 4" (completed text below). The table prints, for each output (out, dfeat, dweight, dbias): the max-normalized relative error against the float64 EXPLICIT_GEMM reference with the knob on and with it off, plus the on-vs-off difference normalized the same way. `TF32_PRECISION constexpr[tf32]` was printed at start. rc 0, 1326 s (cold autotune).

```
kind    dtype    algo                         on==off  relerr_on(out,dfeat,dw,db)          relerr_off(...)                     on-vs-off rel
subm    float32  implicit_gemm                False    8.1e-04 8.0e-04 8.2e-04 1.9e-07   2.8e-06 2.6e-06 3.9e-06 1.9e-07   8.1e-04 8.0e-04 8.2e-04 0.0e+00
subm    float32  implicit_gemm_splitk         False    8.1e-04 8.0e-04 8.2e-04 1.9e-07   9.9e-07 6.8e-07 6.5e-07 1.9e-07   8.1e-04 8.0e-04 8.2e-04 0.0e+00
subm    float32  masked_implicit_gemm         False    8.1e-04 8.0e-04 8.2e-04 1.9e-07   2.9e-06 2.6e-06 4.1e-06 1.9e-07   8.1e-04 8.0e-04 8.2e-04 0.0e+00
subm    float32  masked_implicit_gemm_splitk  False    8.1e-04 8.0e-04 8.2e-04 1.9e-07   2.9e-06 6.3e-07 9.4e-07 1.9e-07   8.1e-04 8.0e-04 8.2e-04 0.0e+00
subm    float16  implicit_gemm                True     6.8e-04 4.8e-04 4.6e-04 3.9e-04   6.8e-04 4.8e-04 4.6e-04 3.9e-04   0.0e+00 0.0e+00 0.0e+00 0.0e+00
subm    float16  implicit_gemm_splitk         False    4.7e-04 4.8e-04 4.6e-04 3.9e-04   4.7e-04 4.8e-04 4.6e-04 3.9e-04   5.2e-04 4.9e-04 4.8e-04 0.0e+00
subm    float16  masked_implicit_gemm         True     6.8e-04 4.8e-04 4.6e-04 3.9e-04   6.8e-04 4.8e-04 4.6e-04 3.9e-04   0.0e+00 0.0e+00 0.0e+00 0.0e+00
subm    float16  masked_implicit_gemm_splitk  False    6.8e-04 4.8e-04 4.6e-04 3.9e-04   6.8e-04 4.8e-04 4.6e-04 3.9e-04   0.0e+00 2.5e-04 0.0e+00 0.0e+00
strided float32  implicit_gemm                False    7.9e-04 8.0e-04 7.3e-04 1.8e-07   2.5e-06 1.7e-06 2.0e-06 1.8e-07   7.9e-04 8.0e-04 7.3e-04 0.0e+00
strided float32  implicit_gemm_splitk         False    7.9e-04 8.0e-04 7.3e-04 1.8e-07   4.2e-07 5.7e-07 2.1e-07 1.8e-07   7.9e-04 8.0e-04 7.3e-04 0.0e+00
strided float32  masked_implicit_gemm         False    7.9e-04 8.0e-04 7.3e-04 1.8e-07   2.6e-06 1.7e-06 1.9e-06 1.8e-07   7.9e-04 8.0e-04 7.3e-04 0.0e+00
strided float32  masked_implicit_gemm_splitk  False    7.9e-04 8.0e-04 7.3e-04 1.8e-07   4.1e-07 5.8e-07 9.7e-07 1.8e-07   7.9e-04 8.0e-04 7.3e-04 0.0e+00
strided float16  implicit_gemm                True     6.5e-04 4.7e-04 4.3e-04 3.0e-04   6.5e-04 4.7e-04 4.3e-04 3.0e-04   0.0e+00 0.0e+00 0.0e+00 0.0e+00
strided float16  implicit_gemm_splitk         True     4.4e-04 4.7e-04 4.3e-04 3.0e-04   4.4e-04 4.7e-04 4.3e-04 3.0e-04   0.0e+00 0.0e+00 0.0e+00 0.0e+00
strided float16  masked_implicit_gemm         True     6.5e-04 4.7e-04 4.3e-04 3.0e-04   6.5e-04 4.7e-04 4.3e-04 3.0e-04   0.0e+00 0.0e+00 0.0e+00 0.0e+00
strided float16  masked_implicit_gemm_splitk  True     4.4e-04 4.7e-04 4.3e-04 3.0e-04   4.4e-04 4.7e-04 4.3e-04 3.0e-04   0.0e+00 0.0e+00 0.0e+00 0.0e+00
FAILURES (original criterion: on==off and relerr_on < 1e-4 fp32 / 2e-2 fp16) 10
FAILURES (accuracy criterion: relerr_on < 2e-2, relerr_off < 1e-4 fp32 / 2e-2 fp16) 0
```

Reading:

- **fp32, knob off (`ieee`):** relerr 4e-7 to 4e-6 on every output, the same ieee accuracy as gfx1100 and as 8592d5f.
- **fp32, knob on (tf32):** relerr 7.3e-4 to 8.2e-4 on out, dfeat and dweight for all 8 fp32 rows. dbias stays at 1.9e-7 because it is a reduction that does not go through `tl.dot`. On-vs-off differs by the same 7-8e-4. This is the accuracy tf32 implies: tf32 keeps a 10-bit mantissa (unit roundoff 2^-11 = 4.9e-4), so rounding both operands of each product bounds the per-product relative error at about 2u = 9.8e-4 before fp32 accumulation. The measured 8e-4 is under that bound and has the same magnitude as the fp16 rows (4-7e-4), which share the 10-bit mantissa. So tf32 really is in use on gfx942 when the knob is on, and it gives tf32-level, not degraded, accuracy. **Within tf32 accuracy: yes, all 8.**
- **fp16:** knob on and off are bitwise equal for 6 of 8 rows. The two that differ are subm `implicit_gemm_splitk` and subm `masked_implicit_gemm_splitk`. On and off are equally accurate there (relerr 4.7e-4 / 6.8e-4 against float64, identical in both settings), and the on-vs-off difference (up to 5.2e-4) is fp16 rounding. Cause, checked in the job's autotune cache: `allow_tf32` is part of the Triton autotune key, and for this shape (`sparse_submanifold_conv_fwd_implicit_gemm_splitk_kernel`, key `(13, 256, 256, 27, 4, <tf32>, float16, ...)`) the tuner picked `B1 256, B2 128, BK 64, num_warps 8` for False and `B1 256, B2 256, BK 64, num_warps 16` for True. The fp16 difference is a tile-shape (accumulation-order) effect. It is not caused by the precision string: the non-split-K fp16 kernels compiled with `'tf32'` are bitwise equal to `'ieee'`.
- **Driver criterion, unchanged:** 10 of 16 rows fail it. The 8 fp32 rows fail both the bitwise comparison and the fp32 1e-4 bound, because that bound was set for ieee and tf32 cannot meet it by design. The 2 fp16 split-K rows fail only the bitwise comparison, for the autotune reason above, and are within the fp16 bound. No row fails an accuracy bound appropriate to the precision in use (the second FAILURES line: 0). I recorded this as expected behaviour on gfx942, not a port fault. The bitwise half of the criterion can only hold where `TF32_PRECISION` is `ieee`, which the porter's and reviewer's gfx1100 runs showed.

```
# tf32check.py <tests dir> <RES>: the round-4 head (imports, RES, C, ALGOS, data, run()) unchanged, then:
def rel(x, ref):
    return ((x.double() - ref).abs().max() / ref.abs().max()).item()
for kind in ('subm', 'strided'):
    ref = run(kind, A.EXPLICIT_GEMM, torch.float64, False)
    for dtype, tol in ((torch.float32, 1e-4), (torch.float16, 2e-2)):
        for algo in ALGOS:
            off = run(kind, algo, dtype, False); on = run(kind, algo, dtype, True)
            same = all(torch.equal(a, b) for a, b in zip(on, off))
            r_on = [rel(x, r) for x, r in zip(on, ref)]; r_off = [rel(x, r) for x, r in zip(off, ref)]
            d = [((a.double() - b.double()).abs().max() / r.abs().max()).item() for a, b, r in zip(on, off, ref)]
            fails += (not same or max(r_on) >= tol)                 # original criterion
            tf32_fails += (max(r_on) >= 2e-2 or max(r_off) >= tol)  # tf32 shares fp16's 10-bit mantissa
```

### Other gates

- CUDA compile gate: no compiled source changed (`git diff --stat 8592d5f ab42455 -- '*.cu' '*.cuh' '*.cpp' '*.h' setup.py` is empty), so the real-nvcc check at bcf7c6f still covers the compiled sources. This host has no CUDA toolkit. No NVIDIA run of a Triton spconv conv with `allow_tf32 = True` at this head (`cuda-not-validated` for that runtime path; CUDA's value of the expression is upstream's by construction).
- Documentation: README "AMD GPUs (ROCm)" under Installation is present and unchanged by the delta.
- `python3 utils/jargon.py --port FlexGEMM`: clean (after `git fetch origin main:main` in the fork clone; without a local `main` it cannot resolve the range).
- `git -C projects/FlexGEMM/src status --porcelain`: clean at ab42455 after the run (the benchmark's `tests/submconv/conv_fwd.png` moved to `agent_space/`).

## Windows attempt 2026-11-20 (windows-gfx1151, Radeon 8060S APU; head ab4245582c7ecdbe9ed687577d00c39d14f9de39): build OK, host froze in the first GPU test

Written by the dispatching session on 2026-11-23 from the validator's leftover files; the validator itself died with the machine and recorded nothing.

- **Build: succeeded.** First Windows build of the port. Venv with `triton-windows 3.9.0.post30` and `spconv-triton 1.0.0`, TheRock ROCm SDK from an existing torch venv (`ROCM_HOME`/`HIP_PATH` at its `_rocm_sdk_devel`, `HIP_DEVICE_LIB_PATH` at the bitcode directory), `CC=CXX=clang-cl`. The second build attempt ended `Successfully built flex_gemm` / `Successfully installed flex_gemm-1.0.0` at 22:41 local. What the first attempt lacked was not recorded; the two command files differ only in environment, and whoever repeats this must check whether the documented command alone works on Windows.
- **Test: `tests/hashmap.py`, unmodified, was the first GPU script started (22:42:20). Windows' last recorded heartbeat is 22:41:58 and the next event is the unclean-shutdown record at the 2026-11-23 reboot.** No bugcheck was logged; the machine stopped responding and stayed that way until it was power-cycled. The log file is empty, so there is no test output and no evidence that the port computed anything wrong.
- Why this script is a hazard on this host: it runs 44 cases (RES 1..1024, four key/value dtype pairs) and repeats each insert and lookup 110 times. The insert kernel is an `atomicCAS` open-addressing loop. Repeated CAS retry loops under contention are a known way to starve or lock this 20-CU APU (see the stdgpu and symforce entries in the laptop's history); the validator prompt forbade stress loops but did not recognize that this project test is one.
- **Not retried.** The platform is recorded blocked so the selector does not hand this run to the laptop again. Whether to retry with a reduced driver (RES <= 64, one repetition, a watchdog) or to ask for a Windows waiver is Jeff's decision. The compiled Windows extension and the venv are still on the laptop.
