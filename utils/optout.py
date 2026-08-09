#!/usr/bin/env python3
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
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import moatlib  # noqa: E402


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
                    "fork": d.get("fork_url"), "pr": d.get("pr_url"),
                    "pr_state": d.get("pr_state"), "review_pr": d.get("review_pr")})
    return out


def cmd_record(args):
    rec = moatlib.record_optout(args.who, args.source, args.note or "")
    print(f"recorded: {rec['who']} ({rec['scope']}) -- {rec['source']}")
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
        for r in rows:
            moatlib.set_disposition(r["full_name"], "skip", "opted-out",
                                    f"maintainer asked us to stop: {rec['source']}")
            print(f"  {r['name']} ({r['full_name']}) was {r['stage']} -- retired "
                  f"in data/dispositions.json")
        print("\nWhat is left is visible on their repositories, so it stays with a person:")
        for r in rows:
            if r["review_pr"]:
                print(f"  close our review PR: gh pr close {r['review_pr']} --comment \"...\"")
            if r["pr"] and r["pr_state"] not in ("merged", "closed"):
                print(f"  close {r['pr']}, then: python3 utils/moatlib.py "
                      f"set-pr-closed {r['name']} --note \"maintainer opted out\"")
            if r["fork"]:
                print("  delete the fork: gh repo delete "
                      f"{r['fork'].split('github.com/', 1)[-1]} --yes")
    # Both files are working-tree writes, so this binds THIS checkout and nothing else
    # until it reaches the trunk. Every other host still has no opt-out recorded, and
    # `--publish` runs from whichever session someone happens to be in -- so the window
    # between recording and merging is a window where the request is not yet honoured
    # anywhere but here. Said at the end because it is the last thing left to do.
    print("\nNot binding anywhere else yet: data/optout.json (and any disposition above) "
          "are working-tree writes.\nCommit them and open the pull request now -- until "
          "it merges, another host can still adopt, port and submit for this owner.")
    return 0


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
    if moatlib.clear_optout(args.who, args.by):
        print(f"removed the opt-out for {args.who}, authorised by {args.by}")
        return 0
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
    sys.exit(main())
