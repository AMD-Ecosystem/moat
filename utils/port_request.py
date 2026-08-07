#!/usr/bin/env python3
"""File and check port-request issues -- the one intake queue.

Two things feed it and they are the same thing. A community member suggests a CUDA
project via the issue form; an in-progress port discovers it needs a dependency that
has no ROCm support anywhere. Both are "somebody should look at this repo", both end
at the same human decision (fork it or decline it), so both are issues carrying the
`port-request` label rather than two mechanisms that drift apart.

An agent files rather than adopts. Adopting is a fork, and only an admin makes one.

    python3 utils/port_request.py check <owner/repo>
    python3 utils/port_request.py list
    python3 utils/port_request.py file <owner/repo> --blocks <project> --why "..."

Checking before filing matters more than it looks. `owner/repo` is not a stable key:
lucebox was skipped in May as luce-org/lucebox-hub, was renamed, and came back through
discovery under the new name because nothing matched. GitHub resolves an old name to
the new one but never the reverse, so `check` tests both spellings AND the numeric
repo id, which a rename does not change.
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import moatlib  # noqa: E402

REPO = "AMD-Ecosystem/moat"
LABEL = "port-request"


def gh_json(args):
    r = subprocess.run(["gh", *args], capture_output=True, text=True, timeout=90)
    if r.returncode:
        return None
    try:
        return json.loads(r.stdout)
    except json.JSONDecodeError:
        return None


def canonical(full_name):
    """(current owner/repo, numeric id), following any rename. (None, None) if
    unreachable -- which is NOT the same as "does not exist", so callers must not
    read it as a clean bill of health."""
    d = gh_json(["api", f"repos/{full_name}", "--jq", "{full_name:.full_name,id:.id}"])
    return ((d or {}).get("full_name"), (d or {}).get("id"))


def open_requests():
    """Open port-request issues, with the repo each one names."""
    rows = gh_json(["issue", "list", "--repo", REPO, "--label", LABEL,
                    "--state", "open", "--limit", "200",
                    "--json", "number,title,url,body"]) or []
    for r in rows:
        m = re.search(r"([\w.-]+/[\w.-]+)", r.get("title", ""))
        r["repo"] = m.group(1) if m else None
    return rows


def known(full_name):
    """Why this repo needs no new request, or None if it does.

    Checks every place a decision could already live: adopted as a project,
    dispositioned, or sitting in an open request."""
    canon, repo_id = canonical(full_name)
    names = {full_name} | ({canon} if canon else set())

    for n in names:
        short = n.split("/")[-1]
        if (moatlib.PROJECTS / short / "status.json").exists():
            return f"already adopted as projects/{short}"
    # By id as well as by name: a rename changes the name and not the id, and GitHub
    # resolves old->new but never new->old.
    disp = next((moatlib.get_disposition(n, repo_id) for n in names
                 if moatlib.get_disposition(n, repo_id)), None)
    if disp:
        return (f"already dispositioned: {disp.get('disposition')} "
                f"({disp.get('reason')}) -- {disp.get('note', '')}".strip())

    for r in open_requests():
        if r.get("repo") and r["repo"].lower() in {n.lower() for n in names}:
            return f"already requested in {r['url']}"

    if canon and canon != full_name:
        return None  # renamed but otherwise unknown; caller files under the canonical name
    return None


def file_request(full_name, why, blocks=None, apply=False):
    canon = canonical(full_name)[0] or full_name
    reason = known(full_name)
    if reason:
        return ("known", reason)

    blocks_line = (f"\n**Blocks:** `{blocks}` -- its build needs this and there is no "
                   f"ROCm/HIP support for it upstream or in MOAT.\n" if blocks else "")
    body = (
        f"**Repository:** https://github.com/{canon}\n"
        f"{blocks_line}\n"
        f"**Why:** {why}\n\n"
        "Filed by an agent during a port, not adopted: taking a project up is a fork, "
        "and only an admin creates one. This is a request for the same intake screen "
        "a community suggestion gets -- licence, duplicate effort, viability.\n"
    )
    title = f"[port] {canon}"
    if not apply:
        return ("would-file", f"{title}\n{body}")
    out = subprocess.run(
        ["gh", "issue", "create", "--repo", REPO, "--label", LABEL,
         "--title", title, "--body", body],
        capture_output=True, text=True, timeout=90)
    if out.returncode:
        return ("error", (out.stderr or out.stdout).strip())
    return ("filed", out.stdout.strip())


def main(argv=None):
    ap = argparse.ArgumentParser(prog="port_request")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="open port-request issues")
    c = sub.add_parser("check", help="is this repo already adopted, dispositioned or requested?")
    c.add_argument("full_name")
    f = sub.add_parser("file", help="open a port-request issue for a blocking dependency")
    f.add_argument("full_name")
    f.add_argument("--why", required=True)
    f.add_argument("--blocks", help="the MOAT project whose build needs it")
    f.add_argument("--apply", action="store_true", help="actually open the issue")
    a = ap.parse_args(argv)

    if a.cmd == "list":
        rows = open_requests()
        for r in rows:
            print(f"  #{r['number']:<5} {str(r.get('repo')):<40} {r['url']}")
        print(f"{len(rows)} open port request(s)")
        return 0

    if a.cmd == "check":
        reason = known(a.full_name)
        canon = canonical(a.full_name)[0]
        if canon is None:
            print(f"{a.full_name}: UNREACHABLE on GitHub -- cannot confirm; do not "
                  f"treat this as 'not known'")
            return 2
        if canon != a.full_name:
            print(f"note: {a.full_name} is now {canon} (renamed)")
        print(f"{canon}: {reason or 'not known here -- a request would be new'}")
        return 0

    action, detail = file_request(a.full_name, a.why, a.blocks, apply=a.apply)
    print(f"port-request: {action} -- {detail}")
    return 1 if action == "error" else 0


if __name__ == "__main__":
    sys.exit(main())
