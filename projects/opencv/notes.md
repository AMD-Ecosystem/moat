# opencv notes

A dependency fork: the OpenCV core the cv::cuda modules in contrib cannot build without.

## Why the platform rows carry no validation

This has no test suite of its own. The code is exercised only through the project
that consumes it, so a GPU run against this repository alone would prove nothing.

The validation lives with **opencv_contrib**, `completed` on linux-gfx1100,
linux-gfx90a, windows-gfx1101, windows-gfx1201 at contrib `041d5528`. Every one of
those runs built this fork: opencv_contrib's recorded build configures cmake against
`src-core/` (this repo at `moat-port`) with `OPENCV_EXTRA_MODULES_PATH=../src/modules`,
so core and contrib are one build tree, not two.

All four platforms here are therefore `blocked` with that reason rather than left to
derive `port-ready`. That is deliberate: from `review-passed` on, `arch_task` derives
`port-ready` for any arch with no evidence at head, which would send a validator to a
repository that cannot be validated on its own. Refresh the evidence by revalidating
**opencv_contrib**, not this project.

## Record reconciliation 2026-08-12

Until today this project read `stage: unclaimed`, `head_sha: null`, no platforms and
no PR, while the port was in fact written, reviewed and in front of upstream
maintainers. Two things went wrong because of it: `dep_status("opencv")` returned
`waiting`, so a dependent (DynOSAM) would have sat in `dep-blocked` waiting on work
that was already done, and `orient.sh` kept advertising `opencv (unclaimed -> intake)`,
inviting an intake screen -- or a `scaffold` -- over a finished port.

Corrected on a person's decision (jeff, 2026-08-12) to match the facts already
recorded in `projects/opencv_contrib/notes.md`:

- `head_sha` -> `50f05b150687734a0d0d7084f6215de6cd3a5b95`, the tip of
  `AMD-Ecosystem/opencv @ moat-port` and the head of upstream PR #29285.
- `pr_url` / `pr_number` -> https://github.com/opencv/opencv/pull/29285 (OPEN,
  "[ROCm] Add AMD GPU support for cv::cuda via HIP (core)"), so `moat-checkup` tracks
  it like any other open PR instead of it being invisible to the control plane.
- stage walked `unclaimed -> screened -> planning -> planned -> porting -> ported ->
  review-passed`. The intermediate transitions are bookkeeping to reach the stage the
  work actually reached; the work itself is real and is recorded under opencv_contrib.

### What the evidence does and does not cover

The review is genuine and covered this fork explicitly: opencv_contrib's review entry
records "Reviewed both fork branches base..HEAD via /pr-review: core (AMD-Ecosystem/opencv
f90ef85) and contrib", verdict review-passed.

The validation is by consumption, and it is **not** at the current `head_sha`. The four
platform completions were taken at core `adcd50ca`/`0404733`-era commits; core then moved
`0404733 -> 50f05b1` for the PR #29285 follow-ups, under an explicit recorded decision
("jeff override: no revalidations for these", opencv_contrib notes, 2026-06-19). Those
commits are behaviour-preserving on the success path. This section states that plainly
rather than letting `review-passed` imply the head was proved on a GPU.

## Port state

The `moat-port` branch predates this project being tracked here, so the port's
provenance lives in `projects/opencv_contrib/notes.md` rather than here: the plan,
the dated validation entries and the per-commit test results are all recorded there,
including the two core-side fault classes found during the port (`cudev`
`simd_functions.hpp` PTX with non-saturating emulation, and `CUDART_VERSION` undefined
on HIP gating out modern paths).

## Install as a dependency

Consumed through opencv_contrib's two-repo build, not on its own -- see the
`## Install as a dependency` section of `projects/opencv_contrib/notes.md`. A dependent
that needs `cv::cuda` depends on **opencv_contrib**; this fork comes with it.
