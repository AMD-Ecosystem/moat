# Contributing to MOAT

MOAT ports CUDA projects to ROCm/HIP. This repository is the control plane: it holds
per-project state, the porting knowledge, and the tooling. The ports themselves live on
forks under the AMD-Ecosystem organisation.

## Before you push

Install the hooks once per clone:

    python3 utils/install_hooks.py

They run the same gates as CI (`utils/check.py`), so a push that passes locally passes in
CI. To see what will be checked:

    python3 utils/check.py

## Branch model

`main` is protected. Work happens on a per-project branch, `port/<name>`, which is shared:
every host and person working that project pushes state to it. **The branch existing on the
remote is the claim** -- it is how another host knows the project is in flight, so there is
no lock file and no expiry. Writes to the fork are serialised separately, by the
`porting` {arch, since} lock in `status.json`, which the state transition takes and
releases; that one is about who may push code, not about who holds the project.

The branch is deleted when its pull request merges -- that is what retires the claim. A
`port/<name>` left behind reads as live work forever, and the selector has no timeout
that would notice; taking over one that is genuinely stale is a human decision, not an
expiry.

A project does **not** open a pull request per screen. That was the earlier model and it
cost a decision per project; screens collect into one issue instead, so a batch is one
decision. What still holds is the part that mattered: do not silently drop a project. A
project that fails its licence screen or turns out not to be portable reaches a terminal
answer that gets recorded -- as a disposition in `data/dispositions.json` -- and a negative
outcome is still a deliverable.

A port opens its pull request when it has something to review. The branch existing is what
makes the work visible before then.

## Saying yes, and saying no

A project sits in `awaiting-fork` until someone decides.

**Yes is a fork.** Creating `AMD-Ecosystem/<name>` releases the project -- creating one
is a deliberate act by someone who can, so its existence carries the decision.
`orient.sh` runs `upstream.py --forks --apply` before every selection, so the state
advances and the PR gets its comment the next time anyone starts work.

**No is recorded by a person, through the intake queue.** An agent writes the case and
recommends a reason; it never records one. Screens collect into one issue
(`utils/intake_queue.py publish --apply`), a person replies in prose, and an agent
round-trips that reading as a small pull request carrying only the declines. Approving
that is the record. Reasons are the `SKIP_REASONS` in `moatlib`; `declined` exists for a
deliberate no whose reasoning is intentionally not written down, since this repo is
public and a written reason is permanent and quotable.

A decline reaches `data/dispositions.json` on the trunk or the project is simply
proposed again and the work repeats; `scaffold` then refuses it unless forced, so
revisiting stays possible but deliberate.

## What a pull request should look like

Two kinds arrive here and they want opposite readings. A port carries one project and
is judged against that port. A change to MOAT is judged against every project at once,
because it applies to all of them.

The branch says which: `port/<name>` for a port, anything else for a change to MOAT.
The title says the same thing in the place people actually read -- the list of open
pull requests, every notification, and the trunk's history after it merges:

| | branch | title |
|---|---|---|
| a port | `port/<name>` | `<project>: what changed` |
| a change to MOAT | anything else | say what it changes |

Never put a stage or a verdict in a title -- not `intake`, not `declined`, not
`screened`. Each names a moment, and the branch outlives it: one PR carries the screen,
the plan, the port and the validation records. A neutral title also survives a reversal,
so a decline later overturned needs no retitle. Under 72 characters, and never open a
MOAT change with a project name -- it will read
as that project's work everywhere the branch is not shown. CI checks the title and
fails on it, which costs one click to fix. It also reports where the diff disagrees
with the branch: a port reaching into tooling, or touching projects it did not claim.
That part is reported and not enforced, because a fleet-wide sweep legitimately edits
every project at once and a gate that fires on ordinary work is one people learn to
ignore.

Promoting a lesson to the `cuda-to-rocm` skill inside a port is the one crossing that
belongs, and is expected rather than tolerated: the lesson and the evidence for it
should land together.

## Sending it upstream

The port is reviewed on its own PR **inside the fork** (`moat-port` against the fork's
default branch), not here. A PR in this repository contains state files and markdown, so
approving it alone would approve a file that merely *asserts* the port works; the fork PR
is the actual diff, and review comments there attach to the code and persist for whoever
picks the project up next.

Approve it with a review comment containing `/moat approve` on a line by itself. The
Approved button is unavailable: agents open the pull request with the maintainer's
credentials, so it is self-authored, and GitHub does not let an author approve. A review
comment records the commit it was written against, which is what the staleness checks
rest on.

That fork PR carries the title and body the upstream PR will use, so **one approval covers
the code, the title and the body together** -- write them for the external maintainer from
the start, not as a draft to be rewritten. Approving it is the decision to submit; the
publishing step below opens the upstream PR with exactly that content, records it, and
closes the review PR.

An approval covers what was on screen when it was given. A commit pushed afterwards, or an
edited title or body, voids it. `upstream.py --approvals` reports those, and
`--approvals --apply` dismisses the stale approval and asks for a fresh look rather than
publishing something nobody read. Nothing surfaces them on its own, so the report is a step
of the `moat-checkup` skill. Editing the record does not revive an approval, and only a
person can approve.

Everything after that first post -- replies to maintainers, follow-up comments, a re-request
for review -- is its own act and needs its own yes.

## Who opens the upstream pull request

A person, from a session on their own machine, running one command:

    python3 utils/upstream.py --publish --apply

It carries the approved title and body verbatim, re-checks the approval and every gate at
that moment, opens the pull request, records it, and closes the review PR. `orient.sh`
names any project waiting on it, so an agent picking up work sees it without being told.

This is deliberately not automated. Opening a pull request on someone else's repository
needs write access to public repositories generally -- GitHub offers no narrower way to
reach a third-party upstream -- and that is more than an unattended job should hold.
Keeping the credential in a human session costs one command and removes a standing key.

**Nothing here runs on a schedule.** Record maintenance -- polling upstream pull requests,
releasing projects whose fork has appeared, reporting overtaken approvals -- happens when
someone runs the `moat-checkup` skill or `orient.sh`, not on a timer. That is a real
trade: drift accumulates silently between sweeps, and on one measured sweep 6 of 74
records disagreed with GitHub, two of them merges nobody had noticed. `orient.sh` reports
how long it has been since the last reconciliation so the gap stays visible. The only
workflow left is `ci.yml`, which gates pull requests and holds `contents: read`.

Two mechanisms that look like they would close the gap do not. A **GitHub App** acts only
where it is installed, and we cannot install one in someone else's repository. A
**fine-grained token** is issued against a single resource owner, and every upstream we
contribute to belongs to neither our account nor the organisation. Both handle the
fork-side work and neither reaches the part that matters.

## What the gates check

| gate | why |
|---|---|
| `code` | the Python resolves: no undefined names, no dead imports or locals (pyflakes). Two NameErrors shipped on success paths before this existed, one of them in the approval gate |
| `schema` | `status.json` validates against a schema generated from `moatlib`, so the two cannot disagree |
| `readme` | the generated project table matches the data it describes. The table renders across refs, so it can only be judged where those refs are visible: enforced in a full clone, which is what the pre-push hook runs in, and skipped -- loudly -- in a CI checkout, which fetches one branch and would call every branch-only row stale (`python3 utils/gen_readme.py`) |
| `licenses` | tier lists are well-formed and no identifier sits in two tiers, which would silently disable the review gate |
| `blobs` | nothing tracked that looks like build output (`.a`, `.so`, `.o`, archives, wheels, model weights) and nothing over 1 MB without an entry in the allowlist saying why it is data rather than spill |
| `states` | every state is one `moatlib` knows, every platform is a well-formed `<os>-<gfx>` with a known wavefront width, and no waiver lacks a maintainer's approval |
| `jargon` | the in-house-vocabulary config loads and its patterns compile |
| `surface` | every component the port enumerated is covered or explicitly scoped out with a reason -- accounting, not coverage, so the failure it prevents is the silent omission |
| `forks` | no fork carries uncommitted source edits (local only -- needs the clones) |

## Adding a project

Via PR, carrying the `projects/<name>/` folder, the regenerated README row, and any porting
knowledge the work produced. `utils/moatlib.py scaffold <owner/repo>` creates the skeleton.

Licence is established at intake, before any porting effort, as a FACT rather than a
judgement: read what upstream actually publishes and record the SPDX identifier in
`status.json.license_spdx`. Do not trust GitHub's licence field alone -- it fails to parse a fifth of
repos -- and do not leave it blank. An unrecorded licence blocks the route upstream exactly
as a restrictive one does, and it is the worse of the two failures, because a restrictive
licence announces itself while an empty field looks like nothing is wrong.

Tiers 1 and 2 are cleared to contribute. **Tier 3 and tier 4 always wait for a person**,
one project at a time (`moatlib.py record-license-clearance <name> --by <who>`); a
clearance sets no precedent for its tier, and an agent may carry someone's decision into
the record but never make it. Check with `moatlib.py license-gate <name>`; the tier lists
are in `config/licenses.toml`.

## Writing for upstream

Commit messages, PR bodies and code comments on a fork reach maintainers who do not know
our vocabulary. Check before pushing:

    python3 utils/jargon.py --port <name>

Terms and their replacements are in `config/jargon.toml`. If you find in-house vocabulary
the checker missed, add it in the same change.

## Porting knowledge

Capture per-project findings in `projects/<name>/notes.md`. If a lesson would help someone
porting a **different** project, promote it into the `cuda-to-rocm` skill's `references/`
at the moment you learn it, naming the source project. A later sweep is what failed before:
an append-only file nobody was told to open went unused from the day it was created.

File it by who needs it, not by where you found it. The activity that surfaces a lesson is
rarely the question its reader will be asking, so a rule about torch's hipify generations
belongs with the pytorch-extension strategy even if a Windows build is what exposed it.
