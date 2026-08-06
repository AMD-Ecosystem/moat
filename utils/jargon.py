#!/usr/bin/env python3
"""Scan upstream-visible text for MOAT in-house vocabulary.

Everything MOAT publishes to an upstream project -- commit messages, PR titles and
bodies, code comments, docs -- is read by maintainers who have no idea what our
internal terms mean. "The lead platform validated at head_sha" is meaningless
outside this repo, and it caused repeated review churn.

The rule was written down and still kept reaching PRs, because a rule nobody runs
is a rule nobody follows. This makes it checkable.

    python3 utils/jargon.py <file>...            # scan files
    python3 utils/jargon.py --commits <range>    # scan commit messages in a fork
    python3 utils/jargon.py --diff <range>       # scan ADDED lines only
    echo "text" | python3 utils/jargon.py -      # scan stdin (a drafted PR body)

Exit 1 if anything is found.
"""

import argparse
import re
import pathlib
import subprocess
import sys
import tomllib

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "config" / "jargon.toml"


def load():
    """Terms live in config/jargon.toml, not here: adding one is a config edit
    reviewed as a PR, the same shape as config/licenses.toml. The procedure for
    when to add or remove a term is documented in that file."""
    with open(CONFIG, "rb") as f:
        cfg = tomllib.load(f)
    allow = [re.compile(p, re.I) for p in cfg.get("allow", {}).get("patterns", [])]
    return cfg["terms"], allow


def scan_text(text, label, terms, allow):
    """Scan prose. Fenced code blocks and shell commands are skipped: a Test Plan
    that documents `grep -niE '\bmoat\b'` as its own jargon check is not itself
    jargon, and flagging it trains people to ignore the tool."""
    hits = []
    in_fence = False
    for i, line in enumerate(text.splitlines(), 1):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        stripped = line.strip()
        # an indented command block, or a line that is plainly a shell invocation
        if line.startswith(("    ", "\t")) and re.match(r"^\s*[\w./-]+\s", line):
            if re.match(r"^\s*(git|grep|python3?|bash|sh|gh|cmake|make|\./)\b", line):
                continue
        if any(a.search(line) for a in allow):
            continue
        if stripped.startswith(("#", "//")) and "jargon" in stripped.lower():
            continue
        for pat, fix in terms.items():
            m = re.search(pat, line, re.I)
            if m:
                hits.append((label, i, m.group(0), fix, line.strip()[:90]))
    return hits


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--commits", metavar="RANGE",
                    help="scan commit messages in a git range")
    ap.add_argument("--diff", metavar="RANGE",
                    help="scan lines ADDED in a git range")
    ap.add_argument("-C", dest="repo", default=".", help="git repo")
    a = ap.parse_args()
    terms, allow = load()
    hits = []

    if a.commits:
        r = subprocess.run(["git", "-C", a.repo, "log", "--format=%H%n%B%n--END--", a.commits],
                           capture_output=True, text=True)
        for block in r.stdout.split("--END--"):
            if block.strip():
                sha = block.strip().split("\n")[0][:9]
                hits += scan_text(block, f"commit {sha}", terms, allow)

    if a.diff:
        r = subprocess.run(["git", "-C", a.repo, "diff", "--unified=0", a.diff],
                           capture_output=True, text=True)
        added = "\n".join(l[1:] for l in r.stdout.splitlines()
                          if l.startswith("+") and not l.startswith("+++"))
        hits += scan_text(added, "added lines", terms, allow)

    for p in a.paths:
        if p == "-":
            hits += scan_text(sys.stdin.read(), "stdin", terms, allow)
        else:
            try:
                hits += scan_text(open(p, encoding="utf-8", errors="replace").read(), p, terms, allow)
            except OSError as e:
                print(f"jargon: {e}", file=sys.stderr)

    for label, line, term, fix, ctx in hits:
        print(f"{label}:{line}: '{term}' -- {fix}\n    {ctx}")
    if hits:
        print(f"\njargon: {len(hits)} instance(s) of in-house vocabulary in "
              f"upstream-visible text", file=sys.stderr)
        return 1
    print("jargon: clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
