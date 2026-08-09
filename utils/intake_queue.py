#!/usr/bin/env python3
"""The intake queue: one issue, N screened projects, one decision from a person.

The 2026-08-06 dry run screened four projects and produced four pull requests,
each needing its own clicks, for what is one question: which of these do we fork?
This collects finished screens into a single issue instead.

    python3 utils/intake_queue.py build              # what is waiting, as a table
    python3 utils/intake_queue.py publish --apply    # open or update the queue issue
    python3 utils/intake_queue.py apply --decline <owner/repo>:<reason> ... --apply

Three properties matter more than the mechanics.

ONLY WHAT IS STILL OPEN. A row leaves the queue when it has been answered, not when
the answer has been carried out -- an approved project awaiting its fork is a job to
do, not a question to ask, and listing it re-raises a decision already taken.

ONE OPEN ISSUE AT A TIME, and each one is a SNAPSHOT. `publish` edits the open
queue while it is still undecided, so re-running it before anyone answers does not
litter. But the moment a decision lands, that issue closes and the remainder comes
back as a new one -- because the body is mutable and the comment thread is not.
Regenerating under an answered comment turned "decline the other two" into a
sentence pointing at a table with no two to decline.

THE RECOMMENDATION IS THE DEFAULT. Every row carries what intake would choose, so
a reviewer's reply is a diff against it -- "accept all but X and Y" -- rather than
N separate answers. Effort scales with disagreement, not with the size of the
batch. Rows are ordered declines-first so the ones needing attention are on top.

THE AGENT NEVER DECIDES. A `decline` verdict in the table is a recommendation;
`apply` writes dispositions ONLY from arguments a person's reply supplied, and
puts them on a branch for review rather than on the trunk. Accepts need no
argument at all: creating the fork is the decision, and `upstream.py --forks`
already turns that into state.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import moatlib  # noqa: E402
import prose  # noqa: E402

REPO = "AMD-Ecosystem/moat"
LABEL = "intake-queue"
TITLE = "Intake queue: projects screened and awaiting a fork-or-decline decision"
BRANCH = "intake-decisions"


def gh(args, check=False):
    r = subprocess.run(["gh", *args], capture_output=True, text=True, timeout=90)
    if check and r.returncode:
        raise RuntimeError((r.stderr or r.stdout).strip())
    return r


def gh_json(args):
    r = gh(args)
    if r.returncode:
        return None
    try:
        return json.loads(r.stdout)
    except json.JSONDecodeError:
        return None


def queue():
    """Projects whose screen is finished and whose decision has not been made.

    A row leaves when it has been ANSWERED, not when the answer has been executed.
    Fork existence used to stand in for that, which put "APPROVED, awaiting the fork"
    rows on a queue whose whole job is to ask questions -- resurfacing a decision
    already made. The decision is recorded now (intake.decided for a fork, a
    disposition for a decline), so the queue can ask only what is still open. What
    happens next -- the fork actually being created -- is execution, and
    `upstream.py --forks` is what reports it."""
    rows = []
    for name in sorted(moatlib.all_projects()):
        obj, _where = moatlib.project_record(name)
        if obj is None:
            continue
        d = moatlib.PROJECTS / name
        rec = obj.get("intake")
        if not rec:
            continue                       # not screened yet
        full = moatlib.upstream_full_name(d.name)
        disp = moatlib.get_disposition(full or d.name) or {}
        if disp.get("disposition") == "skip":
            continue                       # already declined by a person
        # A `verify` is the opposite of a decision -- "somebody look at this" -- so it
        # must stay ON the queue and say so. Testing the disposition for truthiness
        # dropped these silently, which hid exactly the rows a person had flagged as
        # needing a second look.
        flagged = disp.get("disposition") == "verify"
        if (rec.get("decided") or {}).get("choice"):
            continue                       # a person already answered this one
        if moatlib.project_stage(obj) not in ("awaiting-fork", "unclaimed", None):
            continue                       # already being worked
        rows.append({
            "name": d.name, "full_name": full, "priority": obj.get("priority", 0),
            "license": obj.get("license_spdx") or "UNRECORDED",
            "tier": moatlib.license_tier(d.name),
            "flagged": flagged, "flag_note": disp.get("note") if flagged else None,
            **rec,
        })
    # Declines first, then by priority: the rows a reviewer must actually think
    # about sit at the top, and "accept the rest" covers everything below them.
    rows.sort(key=lambda r: (r["verdict"] != "decline", -float(r["priority"] or 0)))
    return rows


def render(rows):
    if not rows:
        return ("_Nothing is waiting on a decision._\n\n"
                "Re-run `python3 utils/intake_queue.py publish --apply` after the next "
                "batch of screens.\n")
    out = [
        f"{len(rows)} project(s) screened and waiting. Each row is a "
        "**recommendation**; reply in one comment saying what you want, e.g. "
        "*\"accept all but X and Y -- X because ..., Y because ...\"*. Rows marked "
        "**on hold** are asking something else and are listed under Held back "
        "below.\n",
        "| # | project | licence | duplicate effort | viable | recommend |",
        "|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(rows, 1):
        viable = {True: "yes", False: "no", None: "?"}[r.get("viable")]
        # A held row keeps its number -- it IS a decision waiting on you, just not a
        # fork-or-decline one -- but carries no verdict. Composing "ON HOLD" onto
        # "**fork**" recommended forking the very thing we were saying needs an answer
        # first, and the paragraph below then called it excluded from the decision.
        rec = ("**on hold**" if r.get("flagged") else
               "**decline** (" + (r.get("reason") or "?") + ")"
               if r["verdict"] == "decline" else "**fork**")
        out.append(f"| {i} | [{r['name']}](https://github.com/{r['full_name']}) "
                   f"| {r['license']} (t{r['tier']}) | {r.get('duplicate_effort') or '-'} "
                   f"| {viable} | {rec} |")
    out.append("")
    for r in rows:
        # A nested list item, not a two-space hard break onto an indented line. The
        # latter is indistinguishable from hand-wrapped prose -- utils/prose.py flagged
        # every body this renderer has ever produced, and the publish path never ran it.
        out.append(f"- **{r['name']}** -- {r['summary']}\n"
                   f"  - full write-up: `projects/{r['name']}/notes.md`")
    held = [r for r in rows if r.get("flagged")]
    if held:
        out += ["", "### Held back", "",
                "Flagged with `triage.py verify` because something needs settling "
                "before fork-or-decline is even the right question. They are in the "
                "table above and they do need an answer -- just an answer to what is "
                "written here. Clearing the flag (`triage.py unskip <owner/repo>`) "
                "returns one to an ordinary row.", ""]
        out += [f"- `{r['full_name']}` -- {r.get('flag_note') or 'no note'}" for r in held]
    out += ["", "---", "_This table is a snapshot. It is regenerated in place only "
            "while nobody has answered; once a decision is recorded this issue closes "
            "and anything still undecided returns as a new one, so a reply always "
            "refers to the table above it._"]
    return "\n".join(out) + "\n"


def find_issue():
    rows = gh_json(["issue", "list", "--repo", REPO, "--label", LABEL,
                    "--state", "open", "--limit", "5",
                    "--json", "number,url,title"]) or []
    return rows[0] if rows else None


def publish(apply=False):
    body = render(queue())
    # The review-PR and publish paths in upstream.py check this; the queue never did,
    # and it posts to GitHub exactly as they do. A project summary is free text an
    # intake agent wrote, so the hard-wrapping this catches arrives from outside this
    # file and no amount of care in the renderer prevents it.
    wrapped = prose.check(body, "issue body")
    if wrapped:
        raise SystemExit("intake_queue: " + "; ".join(wrapped) +
                         "\n  write each paragraph as one line; GitHub reflows it")
    existing = find_issue()
    if not apply:
        return ("would-" + ("update" if existing else "open"),
                (existing or {}).get("url", "") + "\n\n" + body)
    if existing:
        gh(["issue", "edit", str(existing["number"]), "--repo", REPO, "--body", body],
           check=True)
        return ("updated", existing["url"])
    prev = gh_json(["issue", "list", "--repo", REPO, "--label", LABEL, "--state",
                    "closed", "--limit", "1", "--json", "number"]) or []
    if prev:
        body += f"\nContinues from #{prev[0]['number']}, which was closed when its batch was decided.\n"
    r = gh(["issue", "create", "--repo", REPO, "--label", LABEL,
            "--title", TITLE, "--body", body], check=True)
    return ("opened", r.stdout.strip())


def record_accepts(accepts, by, apply=False):
    """Record that a person chose to fork, on each project's own branch.

    The fork appearing is still what releases a project. This records the answer for
    the window before that, which is unbounded -- an admin question can sit for days --
    and during which the queue otherwise re-proposes intake's recommendation. That is
    worst on an override, where the recommendation argues against the decision that
    was actually made."""
    out = []
    for full in accepts:
        name = full.split("/")[-1]
        obj, where = moatlib.project_record(name)
        if obj is None:
            out.append((full, "no record found")); continue
        if where != "branch":
            out.append((full, f"record is on the {where}, not a port branch")); continue
        rec = dict(obj.get("intake") or {})
        if not rec:
            out.append((full, "not screened, so there is no recommendation to answer"))
            continue
        rec["decided"] = {"choice": "fork", "by": by, "at": moatlib.now_iso()}
        obj["intake"] = rec
        # A project intake recommended DECLINING was never going to be forked, so
        # nothing was waiting on one. Overriding that to fork has to say so, or
        # `--forks` never looks at it and its fork never releases it. One field: this
        # used to seed an arch record per platform to express a fact about the project.
        obj["stage"] = "awaiting-fork"
        if not apply:
            out.append((full, f"would record fork on port/{name}")); continue
        sha = moatlib.commit_to_branch(
            f"port/{name}", {f"projects/{name}/status.json": json.dumps(obj, indent=2) + "\n"},
            f"{name}: {by} chose to fork, answering intake's recommendation")
        out.append((full, f"recorded on port/{name} at {sha[:9]}"))
    return out


def fork_commands(accepts):
    """The commands that carry out an accept. They belong with the decision, not on
    the queue: the queue asks what is undecided, and this is what to do once it is."""
    return "\n".join(f"gh repo fork {f} --org AMD-Ecosystem --clone=false"
                      for f in accepts)


def apply_decisions(declines, note, by, apply=False, accepts=()):
    """Record the declines a person asked for, on a branch, for one approval.

    Only what is passed in is written. There is deliberately no "apply the
    recommendations" mode: that would let the queue decide, and the whole point of
    the round trip is that a person's words become a diff they can check.

    The diff is confirmed by MERGING, not by approving. Approve is unavailable on a
    self-authored pull request and everything here is self-authored, since agents run
    on the maintainer's credentials. Merging is not blocked that way and carries the
    same actor and timestamp, so it is the signal. For a handful of declines, running
    `triage.py skip` directly is simpler still and skips this entirely -- the round
    trip earns its cost when the batch is large enough that mis-reading the prose is
    a real risk."""
    # One note per decline, positionally. A single shared note put every project's
    # reasoning on every record, so HAMi-core's disposition explained pegainfer too --
    # tolerable for two, useless for thirty, and the note is the only thing a person
    # sees when the project resurfaces years later.
    parsed = []
    for i, spec in enumerate(declines):
        full, _, reason = spec.partition(":")
        if reason not in moatlib.SKIP_REASONS:
            raise ValueError(f"{full}: reason must be one of {moatlib.SKIP_REASONS}")
        parsed.append((full, reason, note[i] if i < len(note) else ""))
    if not apply:
        return ("would-record", "; ".join(f"{f} -> {r}: {n[:60]}" for f, r, n in parsed))

    moatlib._git("fetch", "-q", "origin", "main", check=False)
    moatlib._git("checkout", "-q", "-B", BRANCH, "origin/main", check=True)
    for full, reason, why in parsed:
        moatlib.set_disposition(full, "skip", reason, why, by=by)
    # A disposition changes the project's Outcome cell, so the generated table moves
    # with it. Committing only dispositions.json left this tool opening a pull request
    # that failed the repository's own README gate.
    subprocess.run([sys.executable, "utils/gen_readme.py"], cwd=str(moatlib.REPO_ROOT),
                   capture_output=True, text=True)
    moatlib._git("add", "--", "data/dispositions.json", "README.md")
    if not moatlib._git("diff", "--cached", "--name-only", check=False).stdout.strip():
        return ("nothing", "these dispositions are already recorded")
    msg = ("intake: record declines from the queue\n\n" +
           "\n".join(f"{f} -> {r}\n  {w}" for f, r, w in parsed))
    moatlib._git("commit", "-q", "-m", msg)
    moatlib._git("push", "-q", "--force-with-lease", "-u", "origin", BRANCH, check=True)
    issue = find_issue()
    body = ("Recorded from the intake queue"
            + (f" ({issue['url']})" if issue else "") + ":\n\n"
            + "\n".join(f"- `{f}` -- `{r}`  \n  {w}" for f, r, w in parsed)
            + (("\n\nApproved in the same batch, still to be forked:\n\n```bash\n"
                + fork_commands(accepts) + "\n```\n") if accepts else "")
            + "\n\n**Merge this to record them.** No approving review is needed or "
              "possible: agents open pull requests with the maintainer's credentials, so "
              "this is self-authored and GitHub greys out Approve for an author -- but it "
              "never blocks merging your own pull request, and the merge is the act of "
              "record.\n\nThat holds under branch protection too, with one exception. "
              "Requiring a pull request and requiring status checks are both fine: "
              "neither needs a second person. Requiring APPROVING REVIEWS is the one "
              "that bites, because an author cannot approve their own -- though a "
              "repository admin can still merge via the bypass GitHub offers by "
              "default. It only becomes a hard stop if approvals are required AND "
              "bypassing is disallowed, which for a single-maintainer repository means "
              "nothing can ever merge.\n\n"
              "Accepts are not here: a fork appearing is what records those.\n"
            + (f"\nCloses #{issue['number']}. Whatever is still undecided comes back "
               f"as a fresh queue, because a decision comment refers to the table it "
               f"was written under.\n" if issue else ""))
    r = gh(["pr", "create", "--repo", REPO, "--head", BRANCH, "--base", "main",
            "--title", "intake: record declines from the queue", "--body", body])
    if r.returncode and "already exists" in (r.stderr + r.stdout):
        return ("updated-pr", "branch updated; the open PR now carries it")
    if r.returncode:
        raise RuntimeError((r.stderr or r.stdout).strip())
    return ("pr", r.stdout.strip())


def main(argv=None):
    ap = argparse.ArgumentParser(prog="intake_queue")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build", help="print the queue as it would be rendered")
    p = sub.add_parser("publish", help="open or update the single queue issue")
    p.add_argument("--apply", action="store_true")
    a = sub.add_parser("apply", help="record the declines a person asked for")
    a.add_argument("--decline", action="append", default=[],
                   metavar="owner/repo:reason", help="repeatable")
    a.add_argument("--note", action="append", default=[],
                   help="the reviewer's words for the matching --decline; repeatable "
                        "and paired positionally")
    a.add_argument("--accept", action="append", default=[], metavar="owner/repo",
                   help="record that a person chose to fork; repeatable")
    a.add_argument("--by", help="who decided; defaults to the authenticated account")
    a.add_argument("--apply", action="store_true")
    args = ap.parse_args(argv)

    if args.cmd == "build":
        print(render(queue()))
        return 0
    if args.cmd == "publish":
        action, detail = publish(apply=args.apply)
        print(f"intake-queue: {action}\n{detail}")
        return 0
    by = args.by or (gh(["api", "user", "--jq", ".login"]).stdout.strip() or "unknown")
    if args.accept:
        for full, detail in record_accepts(args.accept, by, apply=args.apply):
            print(f"  accept {full}: {detail}")
        print("\nCreate the forks -- that is what releases these projects:\n")
        print(fork_commands(args.accept))
    if not args.decline:
        return 0 if args.accept else 1
    action, detail = apply_decisions(args.decline, args.note, by, apply=args.apply,
                                     accepts=args.accept)
    print(f"intake-queue: {action} -- {detail}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
