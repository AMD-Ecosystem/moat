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
no separate lock file.

A project opens a **draft PR** into the trunk as soon as it is under consideration, not when
it is finished. That makes work visible, runs CI continuously rather than once at the end,
and lets a reviewer object to the approach on day one. The PR is marked ready when there is
a terminal answer -- **including a negative one**. A project that fails its licence screen or
turns out not to be portable still merges its PR: that merge is what records the
disposition. Do not silently drop a project.

## Saying yes, and saying no

A project's draft PR sits in `awaiting-fork` until someone decides.

**Yes is a fork.** Creating `AMD-Ecosystem/<name>` releases the project -- creating one
is a deliberate act by someone who can, so its existence carries the decision. A
scheduled job advances the state and comments on the PR; nobody has to notice by hand.

**No is a label on the draft PR:**

| label | recorded as |
|---|---|
| `declined:license` | `license-blocked` |
| `declined:already-supported` | `already-supported` |
| `declined` | `declined` -- a decision was made, the reasoning deliberately not recorded |

The bare label exists on purpose. This repo is public, so a written reason is
permanent and quotable, and a project can be reconsidered later without anything to
walk back.

A decline **merges** its PR rather than closing it. The disposition has to reach
`data/dispositions.json` on the trunk, or the project is simply proposed again and the
work repeats; `scaffold` then refuses it unless forced, so revisiting stays possible
but deliberate.

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

Under 72 characters, and never open a MOAT change with a project name -- it will read
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

That fork PR carries the title and body the upstream PR will use, so **one approval covers
the code, the title and the body together** -- write them for the external maintainer from
the start, not as a draft to be rewritten. Approving it is the decision to submit: a
scheduled job opens the upstream PR with exactly that content, records it, and closes the
review PR.

An approval covers what was on screen when it was given. A commit pushed afterwards, or an
edited title or body, voids it: the job dismisses the stale approval and asks for a fresh
look rather than publishing something nobody read. Editing the record does not revive it,
and only a person can approve.

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
reach a third-party upstream -- and that is more than a scheduled job should hold
unattended. Keeping the credential in a human session costs one command and removes a
standing key.

What the scheduled jobs can and cannot do follows from one fact: the token a workflow gets
is scoped to this repository. So they read public state anywhere and write only here --
they poll upstream pull requests, release projects whose fork has appeared, and report
approvals that a later push or edit has overtaken. Acting on any of that touches a fork or
an upstream, so it waits for a session.

Two mechanisms that look like they would close the gap do not. A **GitHub App** acts only
where it is installed, and we cannot install one in someone else's repository. A
**fine-grained token** is issued against a single resource owner, and every upstream we
contribute to belongs to neither our account nor the organisation. Both handle the
fork-side work and neither reaches the part that matters.

## What the gates check

| gate | why |
|---|---|
| `schema` | `status.json` validates against a schema generated from `moatlib`, so the two cannot disagree |
| `readme` | the generated project table matches the data it describes |
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
`upstream.json`. Do not trust GitHub's licence field alone -- it fails to parse a fifth of
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

    python3 utils/jargon.py --commits <base>..HEAD -C projects/<name>/src

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
