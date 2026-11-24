# warp-1.15 notes

## Intake (2026-11-24)

Release line of NVIDIA/warp for upstream's `release-1.15`, depending on `warp-1.14`. Jeff Daily ruled on 2026-11-24 (warp-1.14 notes, morning decisions, decision 5 (a)): start the 1.15 planner now; start the 1.15 porter once warp-1.14 passes review and Linux revalidation at its final Linux head, without waiting for the 1.14 release. The take-up and base rulings of the development-line project (`git show origin/port/warp:projects/warp/notes.md`) bind this line as they bound 1.14.

Recorded (`--by jeffdaily`, from that ruling and the 1.14 pattern): `set-fork-url https://github.com/AMD-Ecosystem/warp`, `set-fork-branch release/1.15.0.amd0`, `set-delivery fork-release --tag-format v{version}+rocm10.2.{rev}` (tag `v1.15.0+rocm10.2.0`, as 1.14's `v1.14.0+rocm10.2.0`), `set-license-spdx Apache-2.0` (evidence LICENSE.md on `release-1.15`). `mirror-branch` created `release-1.15` and `release/1.15.0.amd0` in the fork, both at upstream's `ffa038b1adf3`.

For the planner:
- The port branch is currently an exact copy of upstream `release-1.15`. The 1.14 port lives on `release/1.14.0.amd0` (final Linux head `1c94e68b` at this writing; Windows round and a gfx1250 rerun still ahead). Plan how the 1.14 port is carried onto 1.15 (merging `release/1.14.0.amd0` into the 1.15 branch keeps the 1.14 history and lets later 1.14 commits arrive as further merges; say which, and why).
- Not yet done here: `utils/licenses.py scan-nvidia warp-1.15` (needs a fork clone; 1.14's found only licence notices) and the Git LFS hook handling from 1.14's intake notes (move the LFS `pre-push` aside, then `protect-fork warp-1.15`).
- Coordinator's upstream 1.14 -> 1.15 survey (what changed upstream that will need porter work) is in the coordinator's scratch and summarized in warp-1.14's discussion; the planner should do its own survey of `release-1.14..release-1.15` (native `.cu`/`.h` changes, new NVRTC/CUDA-only features, new libmathdx uses, build and test changes) and account for each in `surface.json`.
- Lessons from 1.14 for this line: run every added or changed test on a real NVIDIA GPU with libmathdx (the Slurm V100 nodes), and every fix round on a 64 KiB shared-memory device and a wave32 device, not only gfx950.
