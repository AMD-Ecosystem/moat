#!/usr/bin/env python3
"""Triage discovery candidates: mark projects you will NOT port (already ported,
already supported, can't port, not a target) with a reason, and review the
remaining actionable candidates. Decisions persist in data/dispositions.json and
block accidental adoption (moatlib scaffold refuses a skipped project).

Usage:
  python3 utils/triage.py review [--top N] [--all]
  python3 utils/triage.py skip <owner/repo> --reason <r> [--note "..."]
  python3 utils/triage.py unskip <owner/repo>
  python3 utils/triage.py skipped
  python3 utils/triage.py backfill-ids

Reasons: already-ported, already-supported, cant-port, not-a-target, duplicate, other
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import moatlib  # noqa: E402

CANDIDATES = moatlib.REPO_ROOT / "data" / "candidates.json"


def load_candidates():
    if not CANDIDATES.exists():
        return []
    try:
        return json.loads(CANDIDATES.read_text())
    except json.JSONDecodeError:
        return []


def cmd_review(args):
    disp = moatlib.load_dispositions()
    skips = {k for k, v in disp.items() if v.get("disposition") == "skip"}
    # Also by GitHub repo id, because owner/repo is not stable. Two candidates in the
    # 2026-08-07 rerun were decided months earlier under their old names -- sonar is
    # dphnAI/aphrodite-engine and lucebox is Luce-Org/lucebox-hub -- and `scaffold`
    # refused both on the id while this listing still offered them, so two agents were
    # dispatched to re-screen settled projects. candidates.json has no id, so the map
    # is built from the dispositions that do.
    skip_ids = {v["repo_id"] for v in disp.values()
                if v.get("disposition") == "skip" and v.get("repo_id")}
    id_cache = {}

    def decided_by_id(full_name):
        if not skip_ids:
            return False
        if full_name not in id_cache:
            id_cache[full_name] = moatlib.github_repo_id(full_name)
        return id_cache[full_name] in skip_ids
    # Keyed on owner/repo, not on the directory name. A project folder is named after
    # the repo's basename, so matching on that alone makes foo/bar and baz/bar the same
    # project -- a different upstream would read as already adopted and vanish from the
    # queue. Dispositions have always keyed on the full name; this now agrees with them.
    adopted = set()
    for name in moatlib.all_projects():
        full = moatlib.upstream_full_name(name)
        if full:
            adopted.add(full.lower())
    cands = load_candidates()

    def is_adopted(fn):
        return fn.lower() in adopted

    pending = [c for c in cands
               if c["full_name"].lower() not in skips and not is_adopted(c["full_name"])]
    # The id check costs an API call per surviving row, so it runs only on what is left.
    renamed = [c for c in pending if decided_by_id(c["full_name"])]
    pending = [c for c in pending if c not in renamed]
    n_skip = sum(1 for c in cands if c["full_name"].lower() in skips) + len(renamed)
    n_adopt = sum(1 for c in cands if is_adopted(c["full_name"]))
    shown = pending if args.all else pending[:args.top]
    print(f"# {len(pending)} pending ({n_skip} skipped, {n_adopt} adopted) of {len(cands)}; showing {len(shown)}")
    for c in renamed:
        d = moatlib.get_disposition(c["full_name"], id_cache.get(c["full_name"])) or {}
        print(f"  renamed since a decision: {c['full_name']} is {d.get('full_name')} "
              f"({d.get('reason')}) -- not offered")
    print(f"{'#':>3}  {'prio':>5}  {'stars':>7}  project -- description")
    for i, c in enumerate(shown, 1):
        tag = " [verify]" if c["full_name"].lower() in disp else ""
        desc = (c.get("description") or "")[:64]
        print(f"{i:>3}  {c['priority']:>5}  {c['stars']:>7}  {c['full_name']}{tag} -- {desc}")
    print(f"\nskip:   python3 utils/triage.py skip <owner/repo> --reason <{'|'.join(moatlib.SKIP_REASONS)}> --note \"...\"")
    print("verify: python3 utils/triage.py verify <owner/repo> --note \"...\"")
    return 0


def cmd_skip(args):
    d = moatlib.set_disposition(args.repo, "skip", args.reason, args.note or "")
    print(f"skip {d['full_name']} ({d['reason']}) {d['note']}".rstrip())
    return 0


def cmd_verify(args):
    d = moatlib.set_disposition(args.repo, "verify", "verify", args.note or "")
    print(f"verify {d['full_name']}" + (f" -- {d['note']}" if d["note"] else ""))
    return 0


def cmd_unskip(args):
    print("unskipped" if moatlib.clear_disposition(args.repo) else "no disposition for that repo")
    return 0


def cmd_skipped(args):
    disp = moatlib.load_dispositions()
    if not disp:
        print("(no dispositions yet)")
        return 0
    for v in sorted(disp.values(), key=lambda x: (x["reason"], x["full_name"].lower())):
        note = f" -- {v['note']}" if v.get("note") else ""
        print(f"{v['disposition']:>5}  {v['reason']:<18}  {v['full_name']}{note}")
    return 0


def cmd_backfill_ids(args):
    """Record the GitHub repo id on dispositions written before ids were kept.

    Without it a rename slips a decided project back into discovery, which is what
    happened to lucebox. One pass, then new dispositions carry the id themselves."""
    d = moatlib.load_dispositions()
    todo = [k for k, v in d.items() if v.get("repo_id") is None]
    print(f"{len(todo)} disposition(s) without a repo id")
    filled = gone = 0
    for k in todo:
        rid = moatlib.github_repo_id(d[k].get("full_name") or k)
        if rid is None:
            gone += 1
            continue
        d[k]["repo_id"] = rid
        filled += 1
        if filled % 25 == 0:
            moatlib.save_dispositions(d)
            print(f"  ...{filled} filled")
    moatlib.save_dispositions(d)
    print(f"filled {filled}; {gone} unreachable (deleted, private or renamed away) "
          f"-- those stay name-keyed")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="triage")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("review", help="show undecided candidates")
    r.add_argument("--top", type=int, default=40)
    r.add_argument("--all", action="store_true")
    r.set_defaults(fn=cmd_review)
    s = sub.add_parser("skip", help="mark a project not-to-port")
    s.add_argument("repo")
    s.add_argument("--reason", required=True, choices=moatlib.SKIP_REASONS)
    s.add_argument("--note")
    s.set_defaults(fn=cmd_skip)
    v = sub.add_parser("verify", help="flag a project to investigate (not a skip)")
    v.add_argument("repo")
    v.add_argument("--note")
    v.set_defaults(fn=cmd_verify)
    u = sub.add_parser("unskip", help="remove a disposition")
    u.add_argument("repo")
    u.set_defaults(fn=cmd_unskip)
    k = sub.add_parser("skipped", help="list all dispositions")
    k.set_defaults(fn=cmd_skipped)
    b = sub.add_parser("backfill-ids", help="record GitHub repo ids on old dispositions")
    b.set_defaults(fn=cmd_backfill_ids)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
