---
name: moat-porting
description: MOAT's workflow layer over the cuda-to-rocm skill. Where porting findings are recorded, how gates, fork commits, carry-forward and revalidation interact, MOAT-specific dispositions and dependencies, traps in MOAT's own tools, and fleet build recipes. Use together with cuda-to-rocm in every MOAT role (intake, planner, porter, reviewer, validator).
---

# MOAT porting practice

`cuda-to-rocm` is the general porting guide and is published outside this repository, so it
names no MOAT tool, record or state. This skill carries everything that only makes sense
inside MOAT. Load both: `cuda-to-rocm` for the engineering, this file for how the result is
recorded and gated here. Section names below mirror the `cuda-to-rocm` file each one
extends.

## Where a lesson goes

- A fact about one port: `projects/<name>/notes.md`. Fault-class entries in `cuda-to-rocm`
  name the project they were learned on; the full incident lives in that project's notes.
- A lesson that would help someone porting a DIFFERENT project, with no MOAT tooling in it:
  `cuda-to-rocm`, in the file its reader would open. Write it for a reader who has never
  heard of MOAT: no `moatlib`, `notes.md`, gates, roles, states or fleet hosts. Say "your
  port plan" or "your porting notes" instead of naming our files.
- A lesson about MOAT's pipeline, records, tools or hosts: this skill.

Both skills are trunk files, one organized copy each, and a port branch may not edit them
(the pre-push `branch-scope` gate). Write the lesson in the project's `notes.md` under
`## Lesson for cuda-to-rocm: <title>` or `## Lesson for moat-porting: <title>`, with the
reference file it belongs in. The reviewer reads it with the port. After review, a trunk
session promotes it: it puts the lesson where its reader would look, merges it with what
is already there instead of appending, and adds `Promoted to <file> in <sha>.` under the
heading. `python3 utils/moatlib.py lessons` lists lessons awaiting promotion.

`python3 utils/check.py skills` enforces the boundary: it fails on MOAT vocabulary inside
`.claude/skills/cuda-to-rocm/`.

## Assessing existing support (extends `assess-existing-support.md`)

- The planner records the authoritativeness call in `plan.md` with the URL of any existing
  AMD effort and a one-line authoritative-vs-community judgment, and records the
  mechanical-port versus AMD-native-rewrite decision for performance-critical kernels.
- Mature ROCm/HIP support upstream, or a mature separate AMD project, is the
  `already-supported` disposition. An authoritative but incomplete AMD port means pointing
  MOAT at it: validate and improve, do not re-port.
- MOAT's history behind the examples: llm.c (a linked fork missed, one full wasted
  pipeline), gsplat (MOAT opened a fourth competing AMD PR, #970), gRASPA (a third-party
  HIP backend merged while our port was in flight, found only at PR prep), colmap#4420
  (opened by a third party, half its commits ours).
- Pushing a branch cut from a maintainer's branch can be refused for lack of `workflow`
  token scope. Granting that scope is a person's decision; never rewrite the branch to
  dodge it.

## Dispositions and dependencies (extends `no-hip-equivalent.md`)

- `cant-port` means "no AMD-native target exists", not "hipify will not do it". When the
  blocker is CUTLASS or CuTe, the intake recommendation names the reimplementation target
  and a rough cost, and a person decides.
- FlashKDA, NATTEN, mirage and spconv are recorded `cant-port` on CUTLASS/CuTe grounds from
  before FlyDSL was available. Revisiting them is a person's call, not an agent's.
- MOAT ported OpenCV's CUDA modules (opencv/opencv_contrib#4147). A project blocked on
  "OpenCV CUDA has no ROCm build" records `depends_on: [opencv_contrib]` rather than a
  decline.
- A dependency that already has a ROCm port (`spconv_rocm`, pyg-rocm-build wheels,
  `amd_gsplat`) is consumed through the `_deps/` mechanism as environment, not adopted as a
  deliverable.

## Gates, fork commits and carry-forward (extends `validation.md`)

- Coverage is the required gates in `config/arches.toml` (currently `wave64`, `wave32`,
  `windows`). A platform is `<os>-<gfx>`; its gates follow from its name. An unknown
  architecture family is refused until its wavefront width is added under `[wave]`.
  Records made against a platform no longer in service still satisfy gates; a platform
  under `[offline]` owes no revalidation while its gates are satisfied.
- No platform leads: whichever host picks the project up ports it, and the others validate
  the same `head_sha` independently and in parallel.
- Never amend a commit any platform has validated. Amending orphans its `validated_sha`,
  `classify` can no longer judge the delta, and every passed platform drops to a full
  revalidation. Add a new commit instead: `advance-head` carries validation forward for
  documentation-only, comment/format-only and CI-config deltas, and a `codeobj_diff`
  binary-equivalence check can carry forward the rest.
- A Windows-labeled delta is not inert elsewhere. When it flips a passed platform to
  `revalidate`, rebuild rather than carrying forward on "it is Windows-gated" reasoning;
  a binary-equivalence carry-forward builds, which is how the aihwkit and MMseqs2
  regressions were caught.
- Fork GitHub Actions stay disabled. A CI-config commit does not force revalidation (the
  regression guard classifies it as inert); the rule exists because CPU-only workflows gate
  nothing.
- A third-party defect found during validation (TheRock's missing `.CRT$XCU` section,
  `therock-windows-lld-link-crt-xcu`; HIP Windows stream serialization,
  `hip-win-stream-serialization`) is registered with `utils/deferred.py` against the project
  that found it; `deferred.py list` finds it across refs. The `/ALTERNATENAME` workaround's
  reproducer sits in the local `findings/` area; the owning project's `deferred.json`
  records whether it was filed. A failure caused by such a defect is never grounds to
  suggest a gate waiver: LichtFeld-Studio got a wrongly-suggested windows waiver that way.

### `classify` needs the fork cloned at `projects/<name>/src`

`moatlib.py classify <name> <old_sha> <new_sha>` diffs the fork's git history, so it reads
`projects/<name>/src` as a checkout. In a fresh worktree (the usual case for a dispatched
revalidation) that path does not exist and the tool fails closed --
`class=unknown arch_independent=False (classification failed -> revalidate)` -- which looks
exactly like a genuine indeterminate verdict. Clone the fork's port branch there first, then
classify; never read `unknown` as evidence the delta is functional before checking the clone
exists. (SCAMP)

## The CUDA no-regression compile on the fleet (extends `validation.md`)

- Run it on a fleet host that has nvcc. On a GPU-less Linux host the dedicated env is
  `/opt/conda/envs/cuda-12.8/bin/nvcc`, created with the `conda create` line in
  `cuda-to-rocm`'s `validation.md`.
- When the check cannot run for reasons outside the port (a ROCm-only PyTorch on the host
  for a torch extension; legacy `FindCUDA` under CMP0146), record
  `cuda-not-validated: <the exact error>` instead of failing the gate. Fleet hosts carry a
  ROCm dev build of PyTorch (`/var/lib/jenkins/pytorch`), which is why torch-extension
  ports hit the header wall described in `cuda-to-rocm`.
- "notes.md says the CUDA path is preserved" is not a compile. Record the nvcc command and
  its result in `notes.md`.

## Upstream-visible text (extends `naming.md` and the write-up section of `SKILL.md`)

Run `python3 utils/jargon.py --port <name>` before pushing; it scans the whole branch.
Terms and their replacements are in `config/jargon.toml`. In-house words that must never
reach an upstream reader include "lead"/"follower", "head_sha", "curated commit",
"validated_sha", "revalidate", and any define or identifier named after MOAT
(`-DTORCH_HIPIFY_V2`, never a moat-named macro). The checker matches a fixed list, so also
grep new comments for the names of other projects MOAT has ported (arrayfire carried five
such references the checker missed).

### `jargon.py --port` alone misses an in-flight fix round

`--port` resolves its range from `obj.get("fork_branch") or PORT_BRANCH`, and nothing in
`moatlib.py` writes `fork_branch`, so it always scans `moat-port`. Once a fix round is
staged on `moat-fix-<pr#>` (the published `moat-port` is frozen while a PR is open), it
silently scans the frozen branch and never sees the round's commits, so it can report clean
while new jargon sits on the staging branch. Scan the real range explicitly:

```
python3 utils/jargon.py -C projects/<name>/src --commits <base>..<fix-branch> --diff <base>..<fix-branch>
```

Two things make this fail open: a shallow single-branch clone of the fork usually has no
local `main`, and `scan_commits`/`scan_diff` (unlike `port_range`'s range-count check)
report zero hits on an unresolvable range instead of erroring. Fetch first
(`git fetch --depth 50 origin main:main`) and confirm the range resolves to a nonzero commit
count before trusting "clean". (mahout)

The opposite false alarm happens after an upstream sync: the scan's base is the clone's
local default branch, and if that branch still points at the old mirror tip, every
upstream commit the merge brought in is scanned as if it were the port's, and upstream's
own wording ("Strategy B", "revalidate", "COLMAP model") is reported. Fast-forward the local
default branch to the fetched upstream (`git branch -f master origin/master`) before
scanning. (LichtFeld-Studio)

### A rate-limited host can still get a usable fork clone: filter the blobs, sparse the tree

Some hosts have GitHub blob traffic throttled hard enough that a full clone of a large repo is not survivable inside a stop-discipline budget (measured at roughly 1-3 MB/min for a ~110 MB repo, so more than half an hour before anything is readable). A lazy promisor checkout of the whole tree is no better -- it re-fetches per file and stalls part way through. But history and trees are cheap: a blob-filtered clone with the working tree narrowed to the files you actually need is seconds, not minutes.

```
git clone --filter=blob:none --no-checkout --branch <port-branch> <fork> <dest>
git -C <dest> sparse-checkout init --no-cone
git -C <dest> sparse-checkout set '/docs/*.md' '/*.md'    # the paths this round edits
git -C <dest> checkout <port-branch>
git -C <dest> branch master origin/master                 # or the fork's default branch
```

Why each line matters:

- `--filter=blob:none` fetches every commit and tree but no file contents, so the diff, log, and `git ls-tree` all work immediately. `git show <ref>:<path>` then pulls single blobs on demand -- enough to read a handful of build files without ever checking them out.
- Files outside the sparse set carry `skip-worktree`, so `git status --porcelain` stays clean and the integrity gate reads correctly rather than reporting the rest of the tree as deleted.
- The last line is not optional for MOAT tooling: `jargon.py --port` resolves `<default>..<port-branch>` and fails with "cannot resolve" on a clone that has only the remote-tracking ref.

This is a docs, diff, and review checkout, not a build checkout -- a compile needs the whole tree and there is no way around paying for it. Use it for a documentation-only porter round, a reviewer reading a delta, or a validator verifying that a delta is confined to another platform's guards. (LichtFeld-Studio)

## Fleet hosts

- Host and toolchain diagnostics specific to MOAT's machines (the gfx1151 laptop's System32
  runtime, TheRock layouts per host) belong in project notes or memory, not in
  `cuda-to-rocm`; the general Windows lessons there are written host-neutral.
- `hip_link_win.py` and `HipCxx23Shim.cmake` have their canonical copies in
  `cuda-to-rocm/assets/`. Fix bugs there first, then re-copy into each fork that vendors
  them.
