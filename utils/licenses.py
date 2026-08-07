#!/usr/bin/env python3
"""License tiering.

The tier governs one thing: whether the port may be OFFERED UPSTREAM. It does not
decide whether a project is taken up -- a fork appearing in the org is that decision,
and only an admin can make it, so tier 1 confers no head start. Recording the licence
early still costs nothing and means the answer is known before the work is done.

    python3 utils/licenses.py tier MIT              # -> 1
    python3 utils/licenses.py check <owner/repo>    # classify a GitHub repo
    python3 utils/licenses.py audit                 # re-tier every adopted project
    python3 utils/licenses.py --check-config        # CI: validate licenses.toml
"""

import argparse
import json
import pathlib
import re
import subprocess
import sys
import tomllib

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "config" / "licenses.toml"
SPDX_RE = re.compile(r"^[A-Za-z0-9.+-]+$")


def load():
    with open(CONFIG, "rb") as f:
        return tomllib.load(f)


def tier_of(spdx, cfg=None):
    """Tier for an SPDX identifier. Unknown/absent identifiers are tier 4: not
    'probably fine', because tier 4 is whatever falls through."""
    cfg = cfg or load()
    if not spdx or spdx in ("NOASSERTION", "NONE", "null"):
        return 4
    for n in (1, 2, 3):
        if spdx in cfg[f"tier{n}"]["spdx"]:
            return n
    return 4


def route(tier):
    """What unblocks a project at this tier."""
    return {
        1: "cleared to contribute",
        2: "cleared to contribute to a third-party project; including it in our own "
           "software is a separate question needing its own consult",
        3: "strong copyleft -- contributing needs a person's approval, recorded per project",
        4: "not open source -- contributing needs a person's approval, and the licensor's "
           "if the licence grants no contribution rights",
    }[tier]


def scan_nvidia(repo_dir, cfg=None):
    """Files carrying an NVIDIA proprietary licence, which force tier 3 regardless
    of the top-level licence. Matches licence TEXT: grepping for "NVIDIA" alone
    flags every CUDA project, and an NVIDIA copyright under Apache-2.0 is clean."""
    cfg = cfg or load()
    markers = cfg["tier3"]["nvidia_proprietary"]["text_markers"]
    hits = []
    for m in markers:
        r = subprocess.run(["grep", "-rlF", "--", m, str(repo_dir)],
                           capture_output=True, text=True)
        hits.extend(line for line in r.stdout.splitlines() if line)
    return sorted(set(hits))


def check_config(cfg=None):
    """CI gate on config/licenses.toml. The overlap check is the important one:
    an identifier in two tiers makes its classification meaningless."""
    cfg = cfg or load()
    problems = []
    seen = {}
    for n in (1, 2, 3):
        ids = cfg[f"tier{n}"]["spdx"]
        for i in ids:
            if not SPDX_RE.match(i):
                problems.append(f"tier{n}: {i!r} is not a valid SPDX identifier")
            if i in seen:
                problems.append(f"{i} appears in BOTH tier{seen[i]} and tier{n}")
            seen[i] = n
        if ids != sorted(ids):
            problems.append(f"tier{n}: not sorted")
        if len(ids) != len(set(ids)):
            dupes = sorted({x for x in ids if ids.count(x) > 1})
            problems.append(f"tier{n}: duplicates {dupes}")
    return problems


def repo_license(full_name):
    r = subprocess.run(["gh", "api", f"repos/{full_name}",
                        "--jq", '.license.spdx_id // "NONE"'],
                       capture_output=True, text=True, timeout=60)
    return r.stdout.strip() or "NONE" if r.returncode == 0 else None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", nargs="?", choices=["tier", "check", "audit"])
    ap.add_argument("arg", nargs="?")
    ap.add_argument("--check-config", action="store_true")
    a = ap.parse_args()
    cfg = load()

    if a.check_config:
        problems = check_config(cfg)
        for p in problems:
            print(f"licenses: {p}", file=sys.stderr)
        if not problems:
            print(f"licenses: config ok "
                  f"({len(cfg['tier1']['spdx'])}/{len(cfg['tier2']['spdx'])}/"
                  f"{len(cfg['tier3']['spdx'])} in tiers 1/2/3)")
        return 1 if problems else 0

    if a.cmd == "tier":
        t = tier_of(a.arg, cfg)
        print(f"{a.arg}: tier {t} -- {route(t)}")
        return 0

    if a.cmd == "check":
        lic = repo_license(a.arg)
        # NOASSERTION/NONE is GitHub failing to PARSE, not a restrictive licence, and
        # roughly a fifth of repos hit it -- an SPDX header inside a markdown comment,
        # a prose COPYING file. Reporting that as "tier 4" reads as a verdict and
        # invites an agent to record a restriction that is not there. colmap is the
        # example: plainly BSD-3-Clause in COPYING.txt, reported NOASSERTION tier=4.
        # So say UNPARSED and refuse to name a tier; the caller must read the file.
        if lic in ("NOASSERTION", "NONE", None):
            print(f"{a.arg}: license=UNPARSED (GitHub returned {lic})")
            print("  NOT a tier -- GitHub could not classify it, which is not the same "
                  "as restrictive.")
            print("  Read the licence file yourself and record the SPDX id in "
                  "status.json.license_spdx.")
            return 3
        t = tier_of(lic, cfg)
        print(f"{a.arg}: license={lic} tier={t}\n  {route(t)}")
        return 0

    if a.cmd == "audit":
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
        import moatlib
        rows = []
        for sp in sorted((REPO_ROOT / "projects").glob("*/status.json")):
            n = sp.parent.name
            disp = moatlib.get_disposition(moatlib.upstream_full_name(n) or "") or {}
            rows.append((n, disp.get("reason")))
        print(f"{len(rows)} adopted projects; run `check <owner/repo>` per project "
              f"for live tiering (this listing is offline).")
        blocked = [n for n, o in rows if o == "license-blocked"]
        print(f"  recorded license-blocked: {len(blocked)}")
        for n in blocked:
            print(f"    {n}")
        return 0

    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
