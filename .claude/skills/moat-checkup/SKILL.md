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

    bash utils/orient.sh                         # approved ports, fork releases, next work
    python3 utils/upstream.py --review           # finished ports with no review PR open
    python3 utils/upstream.py --attention        # who is waiting on us
    python3 utils/upstream.py --approvals        # approvals overtaken by a push or a body edit
    python3 utils/upstream.py --dry-run          # where our record disagrees with GitHub

The first names any port whose approval is standing and whose gates are met. The second
is where work piles up: a port cannot be approved until its review PR exists, and
nothing opens one automatically, so ports sit finished and unreviewable -- 28 of them
when this was written. `--review --apply --name <p> --title '<t>' --body-file <f>` opens
one. The third lists open PRs where a maintainer asked for something, had the last word,
or has gone quiet. The fourth catches a review GitHub still shows as green over content
nobody approved. The fifth is bookkeeping.

Nothing runs on a schedule. This checkup IS the sweep, so the record only reconciles when
someone runs it -- which is why `orient.sh` reports how long it has been.

## The gate is the approval, and it covers exactly what was approved

Approving the review PR on our own fork means approving the code, the title and the body.
That is the whole content of the upstream PR, so opening it publishes nothing that was not
read and agreed to. Opening it is therefore MECHANICAL -- you do not ask again. It still
happens here rather than on a schedule, because the credentials to open a PR on someone
else's repository belong in a session and not in a standing secret.

The approval happens in ONE place -- that PR page -- and the maintainer does one thing
there: leave a review comment containing `/moat approve` on a line by itself. Not the
Approved button: they authored the pull request, because agents open it with their
credentials, and GitHub greys the button out for an author. A review comment is allowed
and carries the commit it was written against, so the gate can still tell an approval of
this code from an approval of an earlier push. Do not ask them to also read a copy of the
body somewhere in this repo.

Your job is to snapshot what they approved:

    python3 utils/upstream.py --publish            # what is approved and ready
    python3 utils/upstream.py --publish --apply    # open it, close the review PR

That second command is the whole submission: it re-checks the approval, the gates, the
licence and the vocabulary, opens the upstream PR with the approved title and body
verbatim, records it, and closes the review PR. It runs where a human's credentials are --
your session -- because opening a pull request on someone else's repository needs access
no scheduled job can safely hold. `orient.sh` names any project waiting on it.

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

## 1. Opening the PR

Preconditions, all of them:
- A standing approval: `moatlib.py pr-approval <name>` must pass.
- Every required gate satisfied (`moatlib.py pr-ready <name>`), or a maintainer-approved
  waiver. An agent-suggested waiver satisfies nothing.
- No disposition recorded in `data/dispositions.json`. A project with one has been
  settled some other way -- delivered as a validation record, already supported
  upstream, or set aside -- and is not a PR candidate.
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
drafted body, and `--commits <base>..HEAD` on the branch. This rule was written down and
still kept reaching PRs, which is why it is now a command rather than a reminder.

Then record it: `moatlib.py set-pr-open <name> <pr_url> <pr_number>`.

## 2. Maintainer rounds

This is the bulk of the work and where the value is.

- Read the whole thread before responding, including review comments on specific lines
  (`gh pr view <n> --repo <upstream> --json comments,reviews`).
- Distinguish a request for a code change (route to the porter, state
  `changes-requested`) from a question you can answer.
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
colmap is the standing example.

## 5. Fork requests

    python3 utils/upstream.py --forks

Projects screened and waiting for someone to create the fork. Agents cannot create
one, so this is a list for a person, and `orient.sh` prints it on every run.

A project whose folder lives on its own `port/<name>` branch is REPORTED rather than
advanced: releasing writes to its record and the record is not in this checkout. Check
that branch out to release it, or let the next session on it do so.

Declines do not happen here. They are recorded through the intake queue --
`intake_queue.py apply --decline`, carrying a person's answer -- and the labels that
older documents describe record nothing.

## Stop and ask

- Any upstream-visible post: draft, show, wait.
- A maintainer raising a licensing or provenance question: stop. Do not improvise an
  answer about AMD's position.
- A PR that has sat without response for weeks: report it rather than nudging. Finding the
  right reviewer is a human's job, and a second ping from us rarely helps.
