#!/usr/bin/env python3
"""The intake queue: one issue, N screened projects, one decision from a person.

The 2026-08-06 dry run screened four projects and produced four pull requests,
each needing its own clicks, for what is one question: which of these do we fork?
This collects finished screens into a single issue instead.

    python3 utils/intake_queue.py build              # what is waiting, as a table
    python3 utils/intake_queue.py publish --apply    # open or update the queue issue
    python3 utils/intake_queue.py apply --decline <owner/repo>:<reason> ... --apply

Three properties matter more than the mechanics.

ONE ISSUE, regenerated in place. A second issue would split the queue and neither
would be the real one. `publish` finds the open `intake-queue` issue and edits it.

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

    A decision shows up as one of two facts in the world, not as a field someone
    remembered to set. YES is the fork existing, which `upstream.py --forks --apply`
    turns into `screened` on the next orient run; anything past `awaiting-fork` is
    therefore decided. NO is a disposition. Either drops the row, which is what makes
    re-running safe -- a partly executed batch just produces a shorter list, and a
    fork created for 36 of 38 leaves exactly the two stragglers behind."""
    rows = []
    for d in sorted(moatlib.PROJECTS.iterdir()):
        if not (d / "status.json").exists():
            continue
        try:
            obj = moatlib.load_status(d.name)
        except (ValueError, json.JSONDecodeError):
            continue
        rec = obj.get("intake")
        if not rec:
            continue                       # not screened yet
        full = moatlib.upstream_full_name(d.name)
        if moatlib.get_disposition(full or d.name):
            continue                       # already declined by a person
        states = {b.get("state") for b in moatlib.validations(obj).values()}
        if states and not states & {"awaiting-fork", "unclaimed"}:
            continue                       # released (a fork appeared) or already worked
        rows.append({
            "name": d.name, "full_name": full, "priority": obj.get("priority", 0),
            "license": obj.get("license_spdx") or "UNRECORDED",
            "tier": moatlib.license_tier(d.name),
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
        f"{len(rows)} project(s) screened and waiting. Each row is a **recommendation**; "
        "reply in one comment saying what you want, e.g. *\"accept all but X and Y -- "
        "X because ..., Y because ...\"*.\n",
        "| # | project | licence | duplicate effort | viable | recommend |",
        "|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(rows, 1):
        viable = {True: "yes", False: "no", None: "?"}[r.get("viable")]
        rec = ("**decline** (" + (r.get("reason") or "?") + ")"
               if r["verdict"] == "decline" else "**fork**")
        out.append(f"| {i} | [{r['name']}](https://github.com/{r['full_name']}) "
                   f"| {r['license']} (t{r['tier']}) | {r.get('duplicate_effort') or '-'} "
                   f"| {viable} | {rec} |")
    out.append("")
    for r in rows:
        out.append(f"- **{r['name']}** -- {r['summary']}  \n"
                   f"  full write-up: `projects/{r['name']}/notes.md`")
    forks = [r for r in rows if r["verdict"] == "fork"]
    if forks:
        out += ["", "### Accepting", "",
                "Delete any line you are rejecting, then run it. The forks appearing "
                "IS the decision -- `orient.sh` turns that into state on its next run, "
                "so nothing here needs recording by hand.", "", "```bash"]
        out += [f"gh repo fork {r['full_name']} --org AMD-Ecosystem --clone=false"
                for r in forks]
        out += ["```"]
    declines = [r for r in rows if r["verdict"] == "decline"]
    if declines:
        out += ["", "### Declining", "",
                "Say so in a comment. An agent will read it back to you and open a "
                "small PR recording exactly those dispositions -- one approval, and "
                "the merge closes this issue. Nothing is recorded without that.", ""]
        out += [f"- `{r['full_name']}` -- suggested reason `{r.get('reason')}`"
                for r in declines]
    out += ["", "---", "_Regenerated in place by "
            "`python3 utils/intake_queue.py publish --apply`; this issue is never "
            "duplicated._"]
    return "\n".join(out) + "\n"


def find_issue():
    rows = gh_json(["issue", "list", "--repo", REPO, "--label", LABEL,
                    "--state", "open", "--limit", "5",
                    "--json", "number,url,title"]) or []
    return rows[0] if rows else None


def publish(apply=False):
    body = render(queue())
    existing = find_issue()
    if not apply:
        return ("would-" + ("update" if existing else "open"),
                (existing or {}).get("url", "") + "\n\n" + body)
    if existing:
        gh(["issue", "edit", str(existing["number"]), "--repo", REPO, "--body", body],
           check=True)
        return ("updated", existing["url"])
    r = gh(["issue", "create", "--repo", REPO, "--label", LABEL,
            "--title", TITLE, "--body", body], check=True)
    return ("opened", r.stdout.strip())


def apply_decisions(declines, note, apply=False):
    """Record the declines a person asked for, on a branch, for one approval.

    Only what is passed in is written. There is deliberately no "apply the
    recommendations" mode: that would let the queue decide, and the whole point of
    the round trip is that a person's words become a diff they can check."""
    parsed = []
    for spec in declines:
        full, _, reason = spec.partition(":")
        if reason not in moatlib.SKIP_REASONS:
            raise ValueError(f"{full}: reason must be one of {moatlib.SKIP_REASONS}")
        parsed.append((full, reason))
    if not apply:
        return ("would-record", "; ".join(f"{f} -> {r}" for f, r in parsed))

    moatlib._git("fetch", "-q", "origin", "main", check=False)
    moatlib._git("checkout", "-q", "-B", BRANCH, "origin/main", check=True)
    for full, reason in parsed:
        moatlib.set_disposition(full, "skip", reason, note or "")
    moatlib._git("add", "--", "data/dispositions.json")
    if not moatlib._git("diff", "--cached", "--name-only", check=False).stdout.strip():
        return ("nothing", "these dispositions are already recorded")
    msg = ("intake: record declines from the queue\n\n" +
           "\n".join(f"{f} -> {r}" for f, r in parsed) +
           ("\n\n" + note if note else ""))
    moatlib._git("commit", "-q", "-m", msg)
    moatlib._git("push", "-q", "--force-with-lease", "-u", "origin", BRANCH, check=True)
    issue = find_issue()
    body = ("Recorded from the intake queue"
            + (f" ({issue['url']})" if issue else "") + ":\n\n"
            + "\n".join(f"- `{f}` -- `{r}`" for f, r in parsed)
            + ("\n\n" + note if note else "")
            + "\n\nApproving this merges the dispositions. Accepts are not here: a fork "
              "appearing is what records those.\n"
            + (f"\nCloses #{issue['number']}\n" if issue else ""))
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
    a.add_argument("--note", help="the reviewer's words, recorded with the disposition")
    a.add_argument("--apply", action="store_true")
    args = ap.parse_args(argv)

    if args.cmd == "build":
        print(render(queue()))
        return 0
    if args.cmd == "publish":
        action, detail = publish(apply=args.apply)
        print(f"intake-queue: {action}\n{detail}")
        return 0
    if not args.decline:
        print("intake-queue: nothing to record -- pass --decline owner/repo:reason")
        return 1
    action, detail = apply_decisions(args.decline, args.note, apply=args.apply)
    print(f"intake-queue: {action} -- {detail}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
