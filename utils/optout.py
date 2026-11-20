#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Maintainers who asked not to receive pull requests from this effort.

Anyone can say no, and saying it once has to be enough. This records the request and
the three places it binds: discovery stops offering the repos (triage.py review),
adoption refuses them (moatlib scaffold), and the route upstream refuses them
(moatlib pr_ready, which both the review PR and the publish step pass through). The
last one is the point -- an opt-out normally arrives BECAUSE a pull request showed up,
so it has to bind work that is already finished, not only work not yet started.

    python3 utils/optout.py record <owner|owner/repo> --source <url> [--note "..."]
    python3 utils/optout.py list
    python3 utils/optout.py check <owner/repo>
    python3 utils/optout.py remove <owner|owner/repo>

Scope is an OWNER by default, because that is what people mean. Record `owner/repo`
only when the request was explicitly about that one repository.

Unlike a disposition or a licence clearance, an agent may write this one: it carries
somebody else's decision into the record and the only thing it can do is less work.
Removing one is the opposite -- it resumes contact with someone who asked us to stop,
so it takes a person and a reason from that person.

Recording does not undo what is already out there. Open pull requests and forks are
listed after a record so someone can close and delete them; those are visible acts on
other people's repositories and stay with a person.

A record binds every host at once. data/optout.json reaches the trunk through a pull
request, and until that merges only this checkout has it -- so `record` also pushes
the registry to its own branch, `optout-pending`, immediately, and the two writes to
an upstream repository (`upstream.py --publish` and `--merge-fix`) read that branch
before they act. The pull request is still how the record reaches the trunk; the
branch is what makes "stop" mean stop in the meantime.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import identity  # noqa: E402
import moatlib  # noqa: E402

PENDING_BRANCH = "optout-pending"
PENDING_REF = f"refs/remotes/origin/{PENDING_BRANCH}"
REGISTRY = "data/optout.json"


def _git(repo, *args, env=None, stdin=None):
    return subprocess.run(["git", *args], cwd=str(repo), capture_output=True,
                          text=True, encoding="utf-8", errors="replace", env=env,
                          input=stdin)


def _pending(repo):
    """(branch exists, entries, problem) for the pending branch, freshly fetched."""
    f = _git(repo, "fetch", "--quiet", "origin",
             f"+refs/heads/{PENDING_BRANCH}:{PENDING_REF}")
    if f.returncode:
        if "couldn't find remote ref" in (f.stderr or ""):
            return False, {}, None
        why = (f.stderr or "").strip().splitlines()
        return False, None, (f"could not fetch the {PENDING_BRANCH} branch "
                             f"({why[-1][:120] if why else 'fetch failed'})")
    show = _git(repo, "show", f"{PENDING_REF}:{REGISTRY}")
    if show.returncode:
        return True, None, (f"the {PENDING_BRANCH} branch carries no readable "
                            f"{REGISTRY}")
    try:
        d = json.loads(show.stdout)
    except ValueError:
        return True, None, (f"{REGISTRY} on the {PENDING_BRANCH} branch is not "
                            f"valid JSON")
    if not isinstance(d, dict):
        return True, None, f"{REGISTRY} on the {PENDING_BRANCH} branch is not an object"
    return True, d, None


def read_pending(repo=None):
    """(entries, problem): the opt-outs on the pending branch, freshly fetched.

    ({}, None) when the branch does not exist. (None, why) when it could not be
    read -- which is not the same answer: a request to stop that could not be read
    is not a request that was never made, and callers refuse on it."""
    _exists, entries, problem = _pending(repo or moatlib.REPO_ROOT)
    return entries, problem


def pending_for(full_name, repo=None):
    """(entry or None, problem or None) for `owner/repo` on the pending branch, with
    the same owner-or-repo matching as moatlib.optout_for."""
    d, problem = read_pending(repo)
    if d is None:
        return None, problem
    key = (full_name or "").lower()
    return (d.get(key) or d.get(key.split("/")[0])) if key else None, None


def publish_pending(add=None, remove=None, repo=None):
    """Put one change to the registry on the pending branch and push it.

    Builds the commit with plumbing, on top of the branch (or of origin/main when the
    branch does not exist yet), without touching this checkout's branch, index or
    working tree. Returns (ok, detail)."""
    repo = repo or moatlib.REPO_ROOT
    exists, current, problem = _pending(repo)
    if current is None:
        return False, problem
    parent = PENDING_REF
    if not exists:
        # First record: the branch starts from the trunk and its registry.
        parent = "origin/main"
        base = _git(repo, "show", f"{parent}:{REGISTRY}")
        try:
            current = json.loads(base.stdout) if not base.returncode else {}
        except ValueError:
            return False, f"{REGISTRY} on origin/main is not valid JSON"
    updated = dict(current)
    if add:
        updated[add["who"].lower()] = add
    if remove:
        updated.pop(remove.lower(), None)
    if updated == current and exists:
        return True, f"the {PENDING_BRANCH} branch already says so"
    if updated == current:
        return True, "the trunk already says so"
    text = json.dumps(updated, indent=2, sort_keys=True) + "\n"
    blob = _git(repo, "hash-object", "-w", "--stdin", stdin=text)
    if blob.returncode:
        return False, f"could not write the registry blob ({blob.stderr.strip()[:120]})"
    with tempfile.TemporaryDirectory() as tmp:
        env = {**os.environ, "GIT_INDEX_FILE": os.path.join(tmp, "index")}
        steps = (("read-tree", parent),
                 ("update-index", "--add", "--cacheinfo",
                  f"100644,{blob.stdout.strip()},{REGISTRY}"))
        for step in steps:
            r = _git(repo, *step, env=env)
            if r.returncode:
                return False, f"git {step[0]} failed ({r.stderr.strip()[:120]})"
        tree = _git(repo, "write-tree", env=env)
    if tree.returncode:
        return False, f"git write-tree failed ({tree.stderr.strip()[:120]})"
    what = (f"{add['who']} asked not to receive pull requests" if add
            else f"opt-out for {remove} withdrawn")
    commit = _git(repo, "commit-tree", tree.stdout.strip(), "-p", parent, "-m",
                  f"optout: {what}\n\nBinds every host now; the trunk copy follows "
                  f"through its own pull request.")
    if commit.returncode:
        return False, f"git commit-tree failed ({commit.stderr.strip()[:120]})"
    push = _git(repo, "push", "--quiet", "origin",
                f"{commit.stdout.strip()}:refs/heads/{PENDING_BRANCH}")
    if push.returncode:
        return False, (f"could not push the {PENDING_BRANCH} branch "
                       f"({(push.stderr or push.stdout).strip()[-200:]})")
    return True, f"pushed to the {PENDING_BRANCH} branch"


def stamp_optout(who):
    """Add the authenticated account to an opt-out just written (identity.py)."""
    d = moatlib.load_optouts()
    entry = d.get(who.lower())
    if entry is not None and identity.recorded_as():
        identity.stamp(entry)
        moatlib.save_optouts(d)
    return entry


def affected(who):
    """Projects this opt-out covers, with whatever of ours is still visible on them."""
    who = who.lower()
    out = []
    for name in sorted(moatlib.all_projects()):
        d, _where = moatlib.project_record(name)
        if not d:
            continue
        full = (d.get("upstream_url") or "").split("github.com/", 1)[-1]
        if not full:
            continue
        if full.lower() != who and full.lower().split("/")[0] != who:
            continue
        out.append({"name": name, "full_name": full, "stage": d.get("stage"),
                    "maintained": bool(d.get("maintained")),
                    "fork": d.get("fork_url"), "pr": d.get("pr_url"),
                    "pr_state": d.get("pr_state"), "review_pr": d.get("review_pr")})
    return out


def cmd_record(args):
    rec = moatlib.record_optout(args.who, args.source, args.note or "")
    rec = stamp_optout(args.who) or rec
    print(f"recorded: {rec['who']} ({rec['scope']}) -- {rec['source']}")
    # First, before anything else can fail: make it bind the hosts that are not this
    # one. Another session may be minutes from `--publish` for this owner.
    bound, detail = publish_pending(add=rec)
    print(("binding on every host now: " if bound else
           "NOT YET BINDING ANYWHERE ELSE -- ") + detail)
    rows = affected(rec["who"])
    if not rows:
        print("nothing of ours is on their repositories.")
    else:
        # Retiring the adopted projects is part of recording, not a follow-up someone
        # might forget. An opted-out project left in the pipeline keeps being selected,
        # and the next agent to pick it up spends an attempt on work that can never be
        # submitted. This is the one disposition an agent may write, for the same reason
        # the opt-out itself is: the decision was made by the maintainer.
        print(f"\n{len(rows)} adopted project(s) covered by this:")
        # A fork a person ruled we keep working (`set-maintained`) is not retired:
        # the request is not to receive pull requests, and that ruling already
        # means none is ever offered. Its branch stays on our fork and keeps
        # absorbing upstream through sync rounds.
        for r in (r for r in rows if r["maintained"]):
            print(f"  {r['name']} ({r['full_name']}) is a maintained fork -- kept: "
                  f"no pull request is offered, and its branch stays on our fork")
        rows = [r for r in rows if not r["maintained"]]
        for r in rows:
            moatlib.set_disposition(r["full_name"], "skip", "opted-out",
                                    f"maintainer asked us to stop: {rec['source']}",
                                    by=rec["who"])
            print(f"  {r['name']} ({r['full_name']}) was {r['stage']} -- retired "
                  f"in data/dispositions.json")
        if identity.recorded_as():
            d = moatlib.load_dispositions()
            for r in rows:
                if r["full_name"].lower() in d:
                    identity.stamp(d[r["full_name"].lower()])
            moatlib.save_dispositions(d)
        print("\nWhat is left is visible on their repositories, so it stays with a person:")
        for r in rows:
            if r["review_pr"]:
                print(f"  close our review PR: gh pr close {r['review_pr']} --comment \"...\"")
            if r["pr"] and r["pr_state"] not in ("merged", "closed"):
                print(f"  close {r['pr']}, then: {moatlib.PY} utils/moatlib.py "
                      f"set-pr-closed {r['name']} --note \"maintainer opted out\"")
            if r["fork"]:
                print("  delete the fork: gh repo delete "
                      f"{r['fork'].split('github.com/', 1)[-1]} --yes")
    # Both files are working-tree writes. The pending branch (above) already stops
    # every host from opening or moving an upstream pull request for this owner, but
    # discovery, adoption and the README still read the trunk, so the record is not
    # finished until it merges. Said at the end because it is the last thing to do.
    if bound:
        print("\nUpstream writes for this owner are refused on every host from now. "
              "data/optout.json (and any disposition above) are still working-tree "
              "writes:\ncommit them and open the pull request -- until it merges, "
              "another host can still adopt and port for this owner.")
    else:
        print("\nNot binding anywhere else yet: the pending branch could not be "
              "pushed, and data/optout.json (and any disposition above) are "
              "working-tree writes.\nCommit them and open the pull request NOW -- "
              "until it merges, another host can still adopt, port and submit for "
              "this owner.")
    return 0 if bound else 1


def cmd_list(args):
    d = moatlib.load_optouts()
    if not d:
        print("(nobody has opted out)")
        return 0
    for k in sorted(d):
        v = d[k]
        note = f" -- {v['note']}" if v.get("note") else ""
        print(f"{v['scope']:>5}  {v['who']:<40}  {v.get('requested_at', '')}  "
              f"{v.get('source', '')}{note}")
    return 0


def cmd_check(args):
    v = moatlib.optout_for(args.repo)
    if not v:
        print(f"{args.repo}: no opt-out recorded")
        return 0
    print(f"{args.repo}: covered by the {v['scope']} opt-out for {v['who']} "
          f"({v.get('source', '')})")
    return 1


def cmd_remove(args):
    if not args.by:
        print("removing an opt-out resumes contact with someone who asked us to stop; "
              "--by <who authorised it> is required", file=sys.stderr)
        return 2
    was = moatlib.load_optouts().get(args.who.lower())
    if moatlib.clear_optout(args.who, args.by):
        # The entry is deleted, so who authorised that and what it said go to the
        # decision log; the pending branch is told too, or it would go on refusing.
        identity.log_removal(moatlib.REPO_ROOT, "optout-remove", args.who, args.by, was)
        ok, detail = publish_pending(remove=args.who)
        print(f"removed the opt-out for {args.who}, authorised by {args.by} "
              f"(recorded in data/decisions.jsonl; {detail})")
        return 0 if ok else 1
    print("no opt-out recorded for that owner or repo")
    return 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("record", help="record a request not to receive our PRs")
    r.add_argument("who", help="owner, or owner/repo for a single repository")
    r.add_argument("--source", required=True,
                   help="where the request was made (issue, PR comment, email thread)")
    r.add_argument("--note", help="anything worth keeping from the request")
    r.set_defaults(fn=cmd_record)

    li = sub.add_parser("list", help="everyone who has opted out")
    li.set_defaults(fn=cmd_list)

    c = sub.add_parser("check", help="is this repo covered by an opt-out?")
    c.add_argument("repo", help="owner/repo")
    c.set_defaults(fn=cmd_check)

    rm = sub.add_parser("remove", help="withdraw an opt-out (needs a person)")
    rm.add_argument("who")
    rm.add_argument("--by", help="who authorised resuming contact")
    rm.set_defaults(fn=cmd_remove)

    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    import trunk  # noqa: E402 - run the trunk's copy of this tool, see trunk.py
    trunk.reexec(__file__)
    sys.exit(main())
