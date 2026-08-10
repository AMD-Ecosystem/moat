## What is this for

<!-- Delete the one that does not apply. CI derives the same thing from the diff and
     the branch name, so a mismatch is worth explaining rather than hiding. -->

- **A port** -- branch `port/<name>`. Carries one project's folder and what follows
  from it. Reviewed against that port.
- **A change to MOAT** -- any other branch. Tooling, agents, the porting knowledge,
  the gates. Reviewed against every project at once, because it applies to all of them.

Title it accordingly, because the title is what appears in the list of open pull
requests, in every notification, and in the trunk's history afterwards:

| | title |
|---|---|
| a port | `<project>: what changed` -- e.g. `colmap: re-derive GPU SIFT onto current main` |
| a change to MOAT | say what it changes -- e.g. `moat: derive PR intent from the branch` |

Keep it under 72 characters, and do not open a MOAT change with a project name: it
will read as that project's work everywhere the branch is not shown.

## What this changes

<!-- One or two sentences. For a project PR, say which project and what state it reached. -->

## For a project PR

- [ ] Licence recorded and the gate passes (`moatlib.py license-gate <name>`). Tier 3 and 4 need a clearance recorded against that project; an unrecorded licence blocks too, and the fix is to read it, not to approve it
- [ ] The port diff was reviewed on the fork's review PR, not just this one -- link it, and record it with `moatlib.py set-review-pr <name> <url>`:
- [ ] That PR's title and body are the ones the upstream PR will carry, written for the external maintainer. Approving it approves all three of code, title and body at once, so a placeholder title means the real one is never read
- [ ] Coverage gates satisfied, or a waiver approved by a maintainer (`moatlib.py pr-ready <name>`)
- [ ] Fork working tree clean; nothing validated against uncommitted edits
- [ ] No in-house vocabulary anywhere on the fork branch (`utils/jargon.py --port <name>` -- the whole branch, not the newest commit)
- [ ] Generalizable lessons promoted to the `cuda-to-rocm` skill, in the file their reader would open, with the source project named

## For a negative outcome

A project we will not port still merges -- merging is what records the decision, so it is
not proposed again.

- [ ] Disposition and a concrete reason in `data/dispositions.json` (`utils/triage.py skip <owner/repo> --reason <reason> --by <who> --note "..."`)
- [ ] If this records a decline, it came from `intake_queue.py apply --decline` carrying a person's answer on the queue issue. Labels record nothing.

## Notes for the reviewer

<!-- What you want a second pair of eyes on. If a gate is waived, say why here. -->
