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
    adopted = set()
    if moatlib.PROJECTS.exists():
        adopted = {p.name for p in moatlib.PROJECTS.iterdir() if (p / "status.json").exists()}
    cands = load_candidates()

    def is_adopted(fn):
        return fn.split("/")[-1] in adopted

    pending = [c for c in cands if c["full_name"].lower() not in skips and not is_adopted(c["full_name"])]
    n_skip = sum(1 for c in cands if c["full_name"].lower() in skips)
    n_adopt = sum(1 for c in cands if is_adopted(c["full_name"]))
    shown = pending if args.all else pending[:args.top]
    print(f"# {len(pending)} pending ({n_skip} skipped, {n_adopt} adopted) of {len(cands)}; showing {len(shown)}")
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
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
