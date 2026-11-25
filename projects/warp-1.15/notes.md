# warp-1.15 notes

## Intake (2026-11-24)

Release line of NVIDIA/warp for upstream's `release-1.15`, depending on `warp-1.14`. Jeff Daily ruled on 2026-11-24 (warp-1.14 notes, morning decisions, decision 5 (a)): start the 1.15 planner now; start the 1.15 porter once warp-1.14 passes review and Linux revalidation at its final Linux head, without waiting for the 1.14 release. The take-up and base rulings of the development-line project (`git show origin/port/warp:projects/warp/notes.md`) bind this line as they bound 1.14.

Recorded (`--by jeffdaily`, from that ruling and the 1.14 pattern): `set-fork-url https://github.com/AMD-Ecosystem/warp`, `set-fork-branch release/1.15.0.amd0`, `set-delivery fork-release --tag-format v{version}+rocm10.2.{rev}` (tag `v1.15.0+rocm10.2.0`, as 1.14's `v1.14.0+rocm10.2.0`), `set-license-spdx Apache-2.0` (evidence LICENSE.md on `release-1.15`). `mirror-branch` created `release-1.15` and `release/1.15.0.amd0` in the fork, both at upstream's `ffa038b1adf3`.

For the planner:
- The port branch is currently an exact copy of upstream `release-1.15`. The 1.14 port lives on `release/1.14.0.amd0` (final Linux head `1c94e68b` at this writing; Windows round and a gfx1250 rerun still ahead). Plan how the 1.14 port is carried onto 1.15 (merging `release/1.14.0.amd0` into the 1.15 branch keeps the 1.14 history and lets later 1.14 commits arrive as further merges; say which, and why).
- Not yet done here: `utils/licenses.py scan-nvidia warp-1.15` (needs a fork clone; 1.14's found only licence notices) and the Git LFS hook handling from 1.14's intake notes (move the LFS `pre-push` aside, then `protect-fork warp-1.15`).
- Coordinator's upstream 1.14 -> 1.15 survey (what changed upstream that will need porter work) is in the coordinator's scratch and summarized in warp-1.14's discussion; the planner should do its own survey of `release-1.14..release-1.15` (native `.cu`/`.h` changes, new NVRTC/CUDA-only features, new libmathdx uses, build and test changes) and account for each in `surface.json`.
- Lessons from 1.14 for this line: run every added or changed test on a real NVIDIA GPU with libmathdx (the Slurm V100 nodes), and every fix round on a 64 KiB shared-memory device and a wave32 device, not only gfx950.

## Planning (2026-11-24, linux-gfx942 MI300X host, planner)

Plan written to `plan.md` from `release-1.15` at `ffa038b1adf3` and the warp-1.14 port at `1c94e68b`; no build or GPU run (planning only). Lock taken with `set-state warp-1.15 linux-gfx942 planning`. Stage left at `planning`: six questions for jeff are in the plan (Q1-Q6), and Q1 (how the 1.14 port is carried: two merges, `-s ours release-1.14` then `release/1.14.0.amd0`) blocks the porter's first step. The porter does not start before warp-1.14 passes Linux revalidation at its final head (jeff's ruling), whatever Q1's answer.

Owed items from intake, done here:
- `utils/licenses.py scan-nvidia warp-1.15` on the fork clone `projects/warp-1.15/src`: 4 hits, the same licence notices as 1.14 (`README.md`, `licenses/cuda-LICENSE.txt`, `licenses/libmathdx-LICENSE.txt`, `licenses/nanovdb-LICENSE.txt`), no proprietary source.
- Git LFS hook: this host has no `git-lfs`, so the clone had no LFS `pre-push` hook; `python3 utils/moatlib.py protect-fork warp-1.15` installed the fork guard ("fork pre-push hook installed"). Hosts with `git-lfs` follow 1.14's recipe (move LFS's hook aside first). 1.15 adds no LFS-tracked file.

Trial merge (scratch clone `agent_space/trial`, push URL invalid, never pushed): `git merge --no-commit origin/release/1.14.0.amd0` into `release/1.15.0.amd0` -> 45 conflicting files; merging pristine `release-1.14` alone -> 32 (all upstream-vs-upstream); `git merge -s ours origin/release-1.14` (tree identical to `release-1.15`, `git diff --stat` empty) then the port -> 20 conflicting files, each listed with its nature in the plan.

`surface.json`: `python3 utils/surface.py generate warp-1.15` (43 CUDA files, 19 components; the floor now includes 1.15's new top-level CMake build) extended to 79 components (1.14's ids kept, plus deterministic mode, clusters, lean launches, mipmaps, managed memory, the radix-sort bit range, the CMake HIP question, the release-tag workflows, the NVIDIA-libmathdx and 64 KiB test duties); 77 owed, 2 scoped out by binding 1.14 rulings (conditional graph nodes, Clang-CUDA JIT), the two cuBQL CMake targets removed from the floor with a reason; `surface.py check warp-1.15`: fully accounted for.

Survey method: the 1.14..1.15 diff was read in four parts (host runtime, device headers, AOT kernels, build/packaging/tests); every finding the plan relies on was re-read in source, and HIP API facts were checked against the ROCm 10.1 SDK headers on this host (to be re-checked on 10.2 by the porter).

## Rulings

### Ruling on plan Q1, how the 1.14 port is carried (jeff, 2026-11-24)

(a), two merges. First `git merge -s ours release-1.14` on `release/1.15.0.amd0`, then `git merge release/1.14.0.amd0`; later 1.14 commits arrive as further merges. Recorded in `plan.md` (Q1 marked ruled; the carry section and Phase 0 state the two merges as decided). Q2 to Q6 remain open; they block only the phases the plan names (Q2 Phase 1's cluster work, Q3 Phase 3's mipmap part, Q4 Phase 6, Q5 Phase 2's sort work) and the release (Q6), not the start of porting, so the stage moves to `planned`. The porter still waits for warp-1.14 to pass Linux revalidation at its final head.

## Ruling on plan Q2, thread block clusters (jeff, 2026-11-24)

(a): clusters are unsupported on HIP, decided by device capability, the way upstream treats pre-sm_90 NVIDIA GPUs. `cluster_dim` is reported ignored with upstream's warning, `get_cuda_max_cluster_dim()` returns 1, the native cluster calls are no-ops, and the launch uses `cluster_dim = 1`. `test_cluster_dim.py` selects devices by capability, not by `arch`, so its "ignored" cases run on HIP. Register a deferred item for gfx1250 cluster support.

## Ruling on plan Q3, mipmapped textures (jeff, 2026-11-24)

(a): hardware mipmapped arrays on Linux devices with image support; the GPU software sampler with mip levels on devices without it (gfx942, gfx950, and gfx90a 3D); on Windows, try `hipMipmappedArray` on gfx1151 and route to the software sampler if it fails. The six mipmap tests run on every AMD GPU against Warp's CPU result.

## Ruling on plan Q4, the optional CMake build (jeff, 2026-11-24)

(b): no HIP branch in the CMake build. On a host with ROCm and no CUDA Toolkit, CMake stops at configure with a message that ROCm builds use `build_lib.py`; the install docs say the same; a deferred item records CMake HIP support. Evidence checked before the ruling: through upstream `main` (changelog to 1.19.0, release branches to release-1.20), `build_lib.py` is still the release build and is not deprecated; the installation docs call CMake "the alternate CMake build path" and say to "Use `build_lib.py` for release builds that need Warp's full GPU architecture coverage". Revisit in a later line if upstream makes CMake the release build.
