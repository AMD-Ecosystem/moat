---
name: intake-queue
description: Publish the intake queue and carry a person's decision back into state. Use after a batch of intake screens finishes, when someone replies to the queue issue, or when asked what is waiting on a fork-or-decline decision.
---

# The intake queue

N screened projects, one issue, one decision. The 2026-08-06 dry run screened four
projects and produced four pull requests, each wanting its own clicks, for what is a
single question: which of these do we fork?

Two commands and one judgement call in the middle.

## 1. Publish

    python3 utils/intake_queue.py build            # see it first
    python3 utils/intake_queue.py publish --apply  # open or update THE issue

One issue, edited in place, never duplicated -- a second one splits the queue and
neither is the real one. Rows are recommendations, declines first, each linking its
`notes.md` rather than pasting it, because a pasted copy drifts the moment the next
agent appends to the source.

Re-run it any time. A project drops off when its decision exists in the world: the
fork appeared (`upstream.py --forks --apply` turns that into `screened`) or a
disposition is recorded. So a batch part-executed -- 36 forks created of 38 -- just
comes back as the two stragglers.

## 2. Read the reply

Someone answers in prose, usually a diff against the recommendations: *"accept all
but X and Y -- X because the licence is wrong, Y because upstream already does it."*
That is the point of the recommendation column; their effort scales with disagreement
rather than with the size of the batch.

**Accepts need no PR.** The fork appearing IS the decision. `apply --accept
<owner/repo>` writes the intake decision to the project's branch so the queue stops
re-proposing it while the fork lags the decision, prints the prefilled
`gh repo fork` block that carries it out (the commands ride with the decision, not
on the queue issue), and closes the answered issue so the remainder comes back
fresh on the next publish.

## 3. Round-trip the declines

Never act on your reading of prose directly. Reply in-thread with what you understood
-- the list of repos and the reason for each -- and open the PR that carries exactly
that:

    python3 utils/intake_queue.py apply \
      --decline uos/rmagine:cant-port --note "<their words on rmagine, quoted>" \
      --decline foo/bar:already-supported --note "<their words on foo/bar, quoted>" \
      --apply

Notes pair with declines positionally -- one `--note` per `--decline`, each carrying
what the person said about THAT project. A single shared note puts every project's
reasoning on every record, and the note is the only thing a person sees when the
project resurfaces years later.

That branches from the trunk, writes those dispositions (plus the regenerated board,
which moves with them), and opens one small PR. The person MERGES it -- one click, on
a diff they can check at a glance -- and the merge is the act of record: approving is
impossible on a self-authored PR, and everything here is self-authored because agents
run on the maintainer's credentials. Merging records the declines and closes the
issue.

The round trip is the safeguard, not ceremony. Parsing an approval out of free prose
puts a model inside a trust boundary, and MOAT is emphatic elsewhere that **an agent
may neither grant an approval nor repair one**. Proposing an interpretation and
having a person confirm it against a diff keeps that intact: the decision stays a
real GitHub act with an actor and a timestamp.

## What you must not do

- **Do not record a disposition you inferred.** `apply` writes only what you pass, and
  you pass only what a person said. There is deliberately no "apply the
  recommendations" mode -- that would let the queue decide.
- **Do not treat silence as agreement.** An unanswered queue is an unanswered queue;
  it reappears next run.
- **Do not open a per-project PR for a screen.** That is the thing this replaces.
- **Do not create forks.** Agents cannot, and that is the point: the fork is the
  decision and only an admin makes it.

## Related

- `.claude/agents/intake.md` -- the screen itself, and `moatlib.py set-intake`, which
  puts a recommendation in the queue.
- `utils/port_request.py` -- the other way a repo enters intake: a community
  suggestion, or a port discovering it needs an unported dependency.
