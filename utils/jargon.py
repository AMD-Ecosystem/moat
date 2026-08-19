#!/usr/bin/env python3
"""Scan upstream-visible text for MOAT in-house vocabulary.

Everything MOAT publishes to an upstream project -- commit messages, PR titles and
bodies, code comments, docs -- is read by maintainers who have no idea what our
internal terms mean. "The lead platform validated at head_sha" is meaningless
outside this repo, and it caused repeated review churn.

The rule was written down and still kept reaching PRs, because a rule nobody runs
is a rule nobody follows. This makes it checkable.

    python3 utils/jargon.py --port <name>        # a port's WHOLE branch (use this)
    python3 utils/jargon.py <file>...            # scan files
    python3 utils/jargon.py --commits <range>    # scan commit messages in a fork
    python3 utils/jargon.py --diff <range>       # scan ADDED lines only
    echo "text" | python3 utils/jargon.py -      # scan stdin (a drafted PR body)

Prefer `--port`. Passing a range by hand is how the check got scoped to the newest
commit and stayed that way through a full review: faster-gaussian-splatting shipped
"Strategy B (torch hipify)" in the commit its branch starts from, because every round
checked only what that round added. Everything on the branch goes upstream, whichever
round wrote it, so the range is the fork's default branch to `moat-port` -- which
`--port` works out from the project's own record rather than asking anyone to type it.

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


# git writes UTF-8 regardless of the platform's locale. `text=True` alone decodes
# with the locale codec, so on Windows (cp1252) a single non-Latin-1 byte anywhere
# in the history raises UnicodeDecodeError -- and because that happens inside
# subprocess, `r.stdout` comes back None and the scan dies with a misleading
# `AttributeError: 'NoneType' object has no attribute 'split'`. A jargon check that
# crashes instead of answering is the dangerous shape here: it gates publication.
_GIT_TEXT = {"encoding": "utf-8", "errors": "replace"}


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


def port_range(name):
    """(repo, commits_range, diff_range) for a project's whole port branch.

    Read from the project's own record so nobody types a range: the base is
    `fork_default_branch`, never the previous head. Raises if the project or its fork
    clone is not here, because a check that silently scans nothing passes."""
    sys.path.insert(0, str(REPO_ROOT / "utils"))
    import moatlib
    obj, _where = moatlib.project_record(name)
    if obj is None:
        raise ValueError(f"{name}: no record found in this checkout or on the refs")
    repo = REPO_ROOT / "projects" / name / "src"
    if not (repo / ".git").exists():
        raise ValueError(f"{name}: no fork clone at {repo} -- nothing to scan")
    base = obj.get("fork_default_branch") or "main"
    branch = obj.get("fork_branch") or moatlib.PORT_BRANCH
    rng = f"{base}..{branch}"
    n = subprocess.run(["git", "-C", str(repo), "rev-list", "--count", rng],
                       capture_output=True, text=True, **_GIT_TEXT)
    if n.returncode or not n.stdout.strip().isdigit():
        raise ValueError(f"{name}: cannot resolve {rng} in {repo} -- "
                         f"is the fork clone fetched?")
    if int(n.stdout.strip()) == 0:
        raise ValueError(f"{name}: {rng} contains no commits -- a range that scans "
                         f"nothing reports clean, which is worse than not running")
    return (str(repo), rng, f"{base}...{branch}")


def scan_commits(repo, rng, terms, allow):
    """Commit messages in a range. Extracted from main so a gate can call it."""
    r = subprocess.run(["git", "-C", repo, "log", "--format=%H%n%B%n--END--", rng],
                       capture_output=True, text=True, **_GIT_TEXT)
    hits = []
    for block in r.stdout.split("--END--"):
        if block.strip():
            sha = block.strip().split("\n")[0][:9]
            hits += scan_text(block, f"commit {sha}", terms, allow)
    return hits


def scan_diff(repo, rng, terms, allow):
    """Lines ADDED in a range -- code comments and docs the port introduces."""
    r = subprocess.run(["git", "-C", repo, "diff", "--unified=0", rng],
                       capture_output=True, text=True, **_GIT_TEXT)
    added = "\n".join(l[1:] for l in r.stdout.splitlines()
                      if l.startswith("+") and not l.startswith("+++"))
    return scan_text(added, "added lines", terms, allow)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--commits", metavar="RANGE",
                    help="scan commit messages in a git range")
    ap.add_argument("--diff", metavar="RANGE",
                    help="scan lines ADDED in a git range")
    ap.add_argument("-C", dest="repo", default=".", help="git repo")
    ap.add_argument("--port", metavar="NAME",
                    help="scan a project's whole port branch, range worked out for you")
    a = ap.parse_args()
    terms, allow = load()
    hits = []

    if a.port:
        try:
            a.repo, a.commits, a.diff = port_range(a.port)
        except ValueError as e:
            print(f"jargon: {e}", file=sys.stderr)
            return 1

    if a.commits:
        hits += scan_commits(a.repo, a.commits, terms, allow)
    if a.diff:
        hits += scan_diff(a.repo, a.diff, terms, allow)

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
