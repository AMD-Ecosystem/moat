#!/usr/bin/env python3
"""Say what a pull request is FOR, from what it changes.

Two kinds of change arrive here and they want different review. A PROJECT pull
request carries one project's folder and the record that follows from it -- a
reviewer reads it against that port. An INFRA pull request changes MOAT itself,
and a reviewer reads it against every project at once.

Intent is derived rather than declared, because a declaration is a claim about a
diff and the diff is right there. The template still asks, so a mismatch between
what someone says and what they changed is itself worth seeing.

Intent comes from the BRANCH, because the branch already declares it: work on a
port happens on `port/<name>`, and everything else is a change to MOAT. This tool
checks that claim against the diff and says where the two disagree.

Deriving it from the files instead does not work, and the attempt is instructive.
Counting projects fails because a fleet-wide sweep touches every one. Counting
which projects have an authored file changed fails for the same reason -- a sweep
that rewords notes across the fleet looks exactly like eighty ports at once. What
separates them is why the change was made, and the branch is where that was said.

Knowledge counts as project scope on purpose. The porter is told to promote a
reusable lesson into the cuda-to-rocm skill IN THE SAME CHANGE as the port that
taught it, so requiring a second pull request would either split the evidence
from the lesson or, more likely, lose the lesson.

The title is CHECKED, not merely reported. A port's title names its project, so a
list of open pull requests reads as a list of what is being worked on; anything
else says what it changes to MOAT. Unlike the shape of a diff, a title has no
legitimate exception -- it costs one click to fix -- so this part fails.

    python3 utils/pr_intent.py                  # against origin/main
    python3 utils/pr_intent.py --base <ref> --title "colmap: ..." --branch port/colmap
"""

import argparse
import pathlib
import re
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]

PROJECT = re.compile(r"^projects/([^/]+)/(.*)$")
# Written by a person about one port. Everything else under a project folder is
# generated or mechanical, and a sweep touches it across the fleet.
AUTHORED = ("notes.md", "plan.md", "surface.json")
# Files that follow from a project change and belong with it.
RECORD = ("README.md", "data/dispositions.json", "data/deferred.json")
KNOWLEDGE = ".claude/skills/cuda-to-rocm/"


def changed(base):
    r = subprocess.run(["git", "diff", "--name-only", f"{base}...HEAD"],
                       cwd=str(REPO), capture_output=True, text=True)
    if r.returncode:
        r = subprocess.run(["git", "diff", "--name-only", base],
                           cwd=str(REPO), capture_output=True, text=True)
    return [p for p in r.stdout.split("\n") if p]


def classify(paths):
    projects, authored, record, knowledge, infra = set(), set(), [], [], []
    for p in paths:
        m = PROJECT.match(p)
        if m:
            projects.add(m.group(1))
            if m.group(2) in AUTHORED:
                authored.add(m.group(1))
        elif p in RECORD:
            record.append(p)
        elif p.startswith(KNOWLEDGE):
            knowledge.append(p)
        else:
            infra.append(p)
    return projects, authored, record, knowledge, infra


VAGUE = re.compile(r"^(update|updates|fix|fixes|wip|changes?|misc|cleanup|tweaks?|"
                   r"improvements?|stuff|various)\b[.! ]*$", re.I)
TITLE_MAX = 72


def check_title(title, claimed, known_projects):
    """Problems with a pull-request title, most important first.

    A port names its project because the list view is the index of what is in
    flight. A change to MOAT must not open with a project name, or it reads as
    that project's work in every place a title appears without its branch."""
    if not title:
        return []
    bad = []
    if len(title) > TITLE_MAX:
        bad.append(f"{len(title)} characters; keep it to {TITLE_MAX} so it is not "
                   f"truncated in a list")
    if VAGUE.match(title.strip()):
        bad.append(f"{title.strip()!r} says nothing -- a title is read far more often "
                   f"than the diff")
    if claimed:
        if not title.startswith(f"{claimed}:"):
            bad.append(f"a port on port/{claimed} should be titled "
                       f"'{claimed}: <what changed>', so the list of open pull requests "
                       f"reads as a list of projects")
    else:
        first = title.split(":", 1)[0].strip()
        if first in known_projects:
            bad.append(f"opens with the project name {first!r} but is not on a "
                       f"port/{first} branch, so it will read as that project's work "
                       f"wherever the branch is not shown")
    return bad


def report_title(title, problems):
    if not title:
        print("  (no title given; pass --title or set PR_TITLE to check it)")
        return 0
    if not problems:
        print(f"  title: ok -- {title!r}")
        return 0
    print(f"\n  TITLE -- {title!r}")
    for b in problems:
        print(f"    {b}")
    return 1


def branch_name(explicit):
    if explicit:
        return explicit
    for env in ("GITHUB_HEAD_REF", "GITHUB_REF_NAME"):
        v = subprocess.os.environ.get(env)
        if v:
            return v
    return subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                          cwd=str(REPO), capture_output=True, text=True).stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="origin/main")
    ap.add_argument("--branch", default=None)
    ap.add_argument("--title", default=subprocess.os.environ.get("PR_TITLE", ""))
    ap.add_argument("--check-title", action="store_true",
                    help="check the title alone and exit; needs no diff, so it can run "
                         "before the pull request is opened")
    a = ap.parse_args()

    if a.check_title:
        branch = branch_name(a.branch)
        claimed = branch.split("/", 1)[1] if branch.startswith("port/") else None
        known = {d.name for d in (REPO / "projects").iterdir() if d.is_dir()}
        problems = check_title(a.title, claimed, known)
        if not a.title:
            print("pr-intent: no title given")
            return 2
        if problems:
            print(f"pr-intent: title rejected -- {a.title!r}")
            for b in problems:
                print(f"    {b}")
            want = f"{claimed}: <what changed>" if claimed else "<what it changes to MOAT>"
            print(f"  expected: {want}")
            return 1
        print(f"pr-intent: title ok -- {a.title!r}")
        return 0

    paths = changed(a.base)
    if not paths:
        print("pr-intent: no changes against " + a.base)
        return 0
    projects, authored, record, knowledge, infra = classify(paths)
    branch = branch_name(a.branch)
    claimed = branch.split("/", 1)[1] if branch.startswith("port/") else None
    known = {d.name for d in (REPO / "projects").iterdir() if d.is_dir()}
    title_problems = check_title(a.title, claimed, known)

    if claimed:
        print(f"pr-intent: PORT -- {claimed} (from the branch name)")
        stray = sorted(projects - {claimed})
        if stray:
            print(f"  but it also changes {len(stray)} other project(s): "
                  f"{', '.join(stray[:6])}")
            print("  A reviewer reading this against one port will not look at those.")
        if knowledge:
            print("  + a lesson promoted to the cuda-to-rocm skill, which belongs with "
                  "the port that taught it")
        if infra:
            print(f"  + {len(infra)} file(s) changing MOAT itself:")
            for q in infra[:6]:
                print(f"      {q}")
            print("  Say why they are here, or split them: a change to MOAT reviewed as "
                  "part of a port is a change nobody reviewed.")
        return report_title(a.title, title_problems)

    kind = "MAINTENANCE" if projects else "INFRA"
    print(f"pr-intent: {kind} -- changes MOAT itself, not one port (branch {branch!r})")
    if projects:
        print(f"  touches records across {len(projects)} project(s); read it as a sweep, "
              f"where the question is whether the rule is right")
    if len(authored) == 1:
        only = next(iter(authored))
        print(f"  NOTE: it changes an authored record for {only} and nothing else's. "
              f"If this is that port's work, it belongs on port/{only}.")
    for q in (infra + knowledge)[:8]:
        print(f"    {q}")
    return report_title(a.title, title_problems)


if __name__ == "__main__":
    sys.exit(main())
