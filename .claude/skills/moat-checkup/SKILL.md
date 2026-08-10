---
name: moat-checkup
description: Check what MOAT needs from a person -- ports approved and ready to submit upstream, maintainers waiting on a reply, upstream PRs that merged or closed, and landed work going stale. Use at the start of a session, after approving a port, when a maintainer comments, or when the user asks what needs attention.
---

# MOAT checkup

Everything after a port is approved: getting the contribution in front of maintainers,
keeping it alive through review, and noticing when landed work goes stale.

Run this when you want to know what MOAT needs from you -- at the start of a session, after
approving a port, or when a maintainer has been in touch. It is a checkup, not a pipeline
stage: none of it is selected by `next-task`, because it is event-driven (a maintainer
comments, a PR merges, a landed port goes stale) rather than state-driven.

It is a skill rather than a dispatched agent because every step ends at a person. A
subagent could read the threads and draft the replies, but it could not post them, so it
would hand everything back anyway -- and the round trip costs more than doing it here.

## The checkup, in order

    gh issue list --repo AMD-Ecosystem/moat --label opt-out --state open   # anyone who asked us to stop
    bash utils/orient.sh                         # approved ports, fork releases, next work
    python3 utils/upstream.py --review           # finished ports with no review PR open
    python3 utils/upstream.py --attention        # who is waiting on us
    python3 utils/upstream.py --approvals        # approvals overtaken by a push or a body edit
    python3 utils/upstream.py --dry-run          # where our record disagrees with GitHub
    python3 utils/moatlib.py waivers             # gate waivers waiting on a maintainer
    python3 utils/deferred.py pending            # deferrals nobody has ruled on

The first is section 0 and comes before everything else: a maintainer who asked us to
stop is the one item here where continuing to work is worse than doing nothing. The
second names any port whose approval is standing and whose gates are met. The third
is where work piles up: a port cannot be approved until its review PR exists, and
nothing opens one automatically, so ports sit finished and unreviewable -- the report
names them all; do not trust any remembered count.
`--review --apply --name <p> --title '<t>' --body-file <f>` opens one. The fourth lists open PRs where a maintainer asked for something, had the last word,
or has gone quiet. The fifth catches a review GitHub still shows as green over content
nobody approved. The sixth is bookkeeping. The seventh is section 6: a waiver nobody has
answered is a finished port that cannot be submitted.

Nothing runs on a schedule. This checkup IS the sweep, so the record only reconciles when
someone runs it -- which is why `orient.sh` reports how long it has been.

## The gate is the approval, and it covers exactly what was approved

Approving the review PR on our own fork means approving the code, the title and the body.
That is the whole content of the upstream PR, so opening it publishes nothing that was not
read and agreed to. Opening it is therefore MECHANICAL -- you do not ask again. It still
happens here rather than on a schedule, because the credentials to open a PR on someone
else's repository belong in a session and not in a standing secret.

The approval happens in ONE place -- that PR page -- and the maintainer does one thing
there: leave a comment containing `/moat approve` on a line by itself, in either box
GitHub offers -- a review comment or an ordinary conversation comment. Not the
Approved button: they authored the pull request, because agents open it with their
credentials, and GitHub greys the button out for an author. The review form is the
better box because it carries the commit it was written against, so the gate can tell
an approval of this code from an approval of an earlier push; a conversation comment
counts too, judged by its time against the branch tip. Do not ask them to also read a
copy of the body somewhere in this repo.

Your job is to snapshot what they approved:

    python3 utils/upstream.py --publish            # what is approved and ready
    python3 utils/upstream.py --publish --apply    # open it, close the review PR

That second command is the whole submission: it re-checks the approval, the gates, the
licence, the fork's cleanliness (only where this host has the fork clone -- a host
without one has nothing to judge and the report says so, since the check already ran
on the hosts that validated), and the title and body -- for in-house vocabulary and
for hand-wrapping, since a maintainer may have asked for an edit and an edit is where
both creep back in. It then opens the upstream PR with the approved title and body
verbatim, records it, and closes the review PR. It runs where a human's credentials are --
your session -- because opening a pull request on someone else's repository needs access
no scheduled job can safely hold. `orient.sh` names any project waiting on it.

The BRANCH's vocabulary is re-scanned here too, read from the pull request itself
(commit messages and added lines), so no local clone is needed. The open-time scan
covered the whole branch, but a commit that lands between the review PR opening and the
approval is seen by neither that scan nor the staleness check -- the approval is given
against the new tip -- so publish is the one gate that can catch it.

Under the hood it snapshots the approval first:

    python3 utils/moatlib.py record-pr-approval <name>

reads the review PR and records the approver, the exact commit the review was attached to,
and a hash of the title and body. Before opening upstream, `moatlib.py pr-approval <name>`
re-reads that PR and refuses if the approval was withdrawn or someone has changes
requested, if any commit landed after the approval, or if the title or body was edited
after it. An unreachable PR refuses too, and says which it is -- an outage is not a
withdrawn approval.

Those checks compare GitHub against GitHub -- the commit the review was given against
versus the branch tip now, the body's last-edit time versus the approval's -- so they hold
even though `status.json` is a file agents can write. **Editing the record does not make a
stale approval valid**, and an agent may neither grant one nor repair one. When it fails,
the answer is a fresh approval.

`upstream.py --approvals` REPORTS overtaken approvals; `--approvals --apply` dismisses
them and re-requests review. Nothing surfaces these on its own, so running the report is
a step of this checkup rather than something you react to. Dismissing is a write on the
fork and needs your credentials. GitHub goes on showing "Approved"
through both a later push and a body edit, so the page otherwise claims someone signed off
on content they never saw. Leave alone an approval already withdrawn -- no nagging -- and
any case where our record merely disagrees with GitHub, which is a human's to sort out and
never a reason to dismiss someone's review.

Everything past that first post is a different matter. Replies to maintainers, follow-up
comments, an edited body, a re-request for review: none of these were read by anyone, each
is its own act, and each needs its own explicit yes. Draft it, show it, wait. **You never
speak for the project unprompted**, and an approval given for one post never carries to the
next.

## 0. Anyone who asked us to stop

    gh issue list --repo AMD-Ecosystem/moat --label opt-out --state open
    python3 utils/optout.py record <owner|owner/repo> --source <issue or comment URL>

A request can arrive as an issue here or as a comment on any of our pull requests, so
`--attention` (section 2) is the other place it turns up. Read it as an opt-out on the
plainest reading and do not ask for a reason -- nobody owes us one, and the cost of
honouring an ambiguous no is one project.

Recording it is the one decision of this kind an agent may make alone: it is the
maintainer's decision, carried into the record, and it can only ever cause less work.
It retires the projects it covers and blocks discovery, adoption and both routes
upstream. What is left over -- closing the open pull request, deleting the fork -- is
visible on their repository and stays with a person; the command prints exactly what
to run.

Then reply once on the thread, saying it is done and that nothing further will arrive.
That is an upstream-visible post like any other and needs its own approval.

## 1. Opening the PR

Preconditions, all of them:
- A standing approval: `moatlib.py pr-approval <name>` must pass.
- Every required gate satisfied (`moatlib.py pr-ready <name>`), or a maintainer-approved
  waiver. An agent-suggested waiver satisfies nothing.
- No skip disposition recorded in `data/dispositions.json`. A skip means the project
  was settled some other way -- delivered as a validation record, already supported
  upstream, or set aside -- and is not a PR candidate. (A `verify` flag from
  `triage.py verify` settles nothing and does not block.)
- The fork's working tree is clean. A validation built against uncommitted edits produces
  an unbuildable PR; this stranded baspacho and arrayfire.

The title and body you put on the review PR ARE the upstream title and body -- that is what
makes one approval enough. Write them for the external maintainer from the start, not as a
draft to be rewritten later; rewriting them after approval voids it.

The body describes the change on its own terms. **No MOAT vocabulary** -- no "lead", "follower",
"Strategy A/B", "head_sha", "validated_sha", "moat-port" as jargon. External maintainers
have no idea what those mean. Keep the technical rationale, drop the in-house labels. State
which GPUs it was tested on and what passed.

Check it mechanically before showing it for approval -- `python3 utils/jargon.py -` on the
drafted body, and `--port <name>` on the branch. Never a hand-typed `--commits` range:
that is how the check once got scoped to the newest commit and stayed that way through
a full review, and a bare range also skips the added-lines scan the review gate runs.
This rule was written down and still kept reaching PRs, which is why it is now a
command rather than a reminder.

Then record it: `moatlib.py set-pr-open <name> <pr_url> <pr_number>`.

## 2. Maintainer rounds

This is the bulk of the work and where the value is.

- Read the whole thread before responding, including review comments on specific lines
  (`gh pr view <n> --repo <upstream> --json comments,reviews`).
- Distinguish a request for a code change (route to the porter: set state `porting`,
  which takes the fork-write lock -- `changes-requested` is not reachable from
  `review-passed`, the stage an open upstream PR sits at) from a question you can
  answer.
- When a fix lands, the fork HEAD moves, which flips validated platforms to `revalidate`.
  That is correct and expected -- do not suppress it.
- Reply tone: plain and short. No "happy to...", no employer name-dropping, nothing
  lawyerly. Answer the question asked.

If a maintainer signals they will not take the contribution, stop and record it rather
than pushing. Record it with `moatlib.py set-pr-closed <name> --note "<why>"`, and if the
project is settled for good ask for a disposition; a declined PR is a real result and
belongs in the record.

## 3. Merge and after

On merge: `moatlib.py set-pr-merged <name>`.

Landed work still needs tending. Upstream moves, and a port that worked six months ago can
stop building. Periodically re-check merged projects: does the current upstream still build
and pass on AMD hardware? This is the case the org fork ownership exists to support -- the
fork is the place a fix gets prepared.

## 4. Keeping the record straight

A protected trunk creates a bookkeeping gap: once a project's PR has merged into the
control plane, later events -- the upstream PR merging, closing, or a maintainer requesting
changes -- still need recording, and the trunk cannot be written directly.

Poll the recorded `pr_url`s, update state, regenerate the README table, and open a small
automated PR with the result. Without this, every upstream merge needs a hand-made PR and
the table silently rots.

Also reactivate `awaiting-upstream` projects: those are parked on an external event (a
third party's PR landing, say), and when it happens the project becomes workable again.
No project is in that state at the moment, so there is nothing to check here until one
is parked -- the state exists for the case, not the other way round.

## 5. Fork requests

    python3 utils/upstream.py --forks

Projects screened and waiting for someone to create the fork. Agents cannot create
one, so this is a list for a person, and `orient.sh` prints it on every run.

`--forks --apply` releases them, wherever the record lives. A project whose folder is on
its own `port/<name>` branch is advanced there directly, through git plumbing that needs
no checkout, so you do not have to be standing on a branch to release the project it
holds -- and every project waiting on a fork is branch-resident, so reporting those
rather than advancing them once meant the one command for the job released nothing.

Safe here for the reason it would not be for the selector: this advances a RECORD.
Handing an agent a project whose plan and notes are absent from its tree is the thing
that must not happen, and releasing a fork does not do that.

Declines do not happen here. They are recorded through the intake queue --
`intake_queue.py apply --decline`, carrying a person's answer -- and the labels that
older documents describe record nothing.

## 6. Gate waivers awaiting a maintainer

    python3 utils/moatlib.py waivers

A port whose obstacle is the PLATFORM rather than the GPU -- a host runtime written to
POSIX, a Windows toolchain that will not load the runtime library -- can still go
upstream, but only behind a waiver on the `windows` gate, the one gate
`config/arches.toml` marks waivable. The porter that hit the obstacle records the case
(`moatlib.py suggest-waiver <name> windows --reason '<what stops it>'`); it satisfies
nothing and BLOCKS `pr-ready` until a maintainer answers, so suggesting one can never
let a port out early.

Approving is a person's act and never an agent's:
`moatlib.py approve-waiver <name> windows --by <who>`. Show them the reason and the
platform records behind it, and wait.

This list exists because the two ends are far apart in time. The obstacle is found
mid-port, often by an unattended run with nobody to ask; the answer comes from a person
who was not there. Before this had a writer, the determination was made once and then
hand-copied onto the second Windows arch in prose -- "carried from windows-gfx1101
determination" -- which is what a finding with nowhere to go looks like.

Refusing is an answer too, and the queue has to empty both ways or an unanswered
suggestion is reprinted forever and the only escape is approving it:
`moatlib.py refuse-waiver <name> <gate> --by <who> --note '<what to investigate>'`. The
note is required, because a refusal without one leaves the next agent where the last one
was and it will suggest the same waiver again. The gate stays unsatisfied either way; the
refusal just makes the block a known quantity.

Waiving is not the only answer, and often not the right one. A gate that no arch can
satisfy because the CODEBASE cannot be ported is `set-not-portable`; a gate failing on
one card because of a toolchain or library defect is a per-arch `blocked` flag with the
report registered against that project (`deferred.py add --project <name>`, see section
7), and it gates nothing as long as a sibling arch carrying the same attribute passes.

## 7. Deferrals nobody has ruled on

    python3 utils/deferred.py pending

Work a port set aside, with no person having decided it should stay set aside. A
deferral is cheap to record and easy to forget, and it fails silently: "we will get to
it" becomes nobody ever looked, and the port ships covering less than anyone realises.

`deferred.py decide <id> --choice defer|now --by <who>` records the ruling. `--by` is
required and never defaulted, the same rule as a licence clearance or a gate waiver: an
agent may surface a deferral and may not rule on one, because deciding a scoped-out
feature stays scoped out is a judgement about what MOAT is delivering.

A deferral lives in `projects/<name>/deferred.json` on that project's branch, with the
notes and the plan it came out of, so it is reviewed with the port that produced it.
`data/deferred.json` keeps only what is genuinely not project-scoped: a bug isolated
against a ROCm component with no port attached, and the record of work deferred by a
project that has since been removed.

You do not have to check anything out to rule on one. A ruling is a record rather than
a working file -- `pending` already gave you the entry, and nothing in `decide` opens a
plan or a note -- so a deferral whose folder is on a port branch is written straight
there and pushed, and `decide` prints which branch it landed on. Working the whole list
from wherever you happen to be is the intended use. The one refusal is a trunk-resident
project while you are on a port branch: `main` is protected, so that ruling has to
arrive by pull request like any other trunk change.

## Stop and ask

- Any upstream-visible post: draft, show, wait.
- A maintainer raising a licensing or provenance question: stop. Do not improvise an
  answer about AMD's position.
- A PR that has sat without response for weeks: report it rather than nudging. Finding the
  right reviewer is a human's job, and a second ping from us rarely helps.
