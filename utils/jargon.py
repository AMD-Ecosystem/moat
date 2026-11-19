#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
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

import os
import argparse
import re
import pathlib
import subprocess
import sys
import tomllib

# The checkout this acts on. MOAT_REPO is set when the trunk's copy of the tools runs
# against another checkout (see trunk.py); config and schema come from TOOLS_ROOT.
TOOLS_ROOT = pathlib.Path(__file__).resolve().parents[1]
REPO_ROOT = pathlib.Path(os.environ["MOAT_REPO"]).resolve() if os.environ.get("MOAT_REPO") else TOOLS_ROOT
CONFIG = TOOLS_ROOT / "config" / "jargon.toml"


class Terms(dict):
    """The prose terms, as the mapping callers have always been handed, carrying the
    smaller table that is also searched inside code blocks and command lines."""
    fenced = {}


def load():
    """Terms live in config/jargon.toml, not here: adding one is a config edit
    reviewed as a PR, the same shape as config/licenses.toml. The procedure for
    when to add or remove a term is documented in that file."""
    with open(CONFIG, "rb") as f:
        cfg = tomllib.load(f)
    allow = [re.compile(p, re.I) for p in cfg.get("allow", {}).get("patterns", [])]
    terms = Terms(cfg["terms"])
    terms.fenced = dict(cfg.get("fenced", {}))
    return terms, allow


# git writes UTF-8 regardless of the platform's locale. `text=True` alone decodes
# with the locale codec, so on Windows (cp1252) a single non-Latin-1 byte anywhere
# in the history raises UnicodeDecodeError -- and because that happens inside
# subprocess, `r.stdout` comes back None and the scan dies with a misleading
# `AttributeError: 'NoneType' object has no attribute 'split'`. A jargon check that
# crashes instead of answering is the dangerous shape here: it gates publication.
_GIT_TEXT = {"encoding": "utf-8", "errors": "replace"}


_COMMAND_LINE = re.compile(r"^\s*(git|grep|rg|python3?|bash|sh|gh|cmake|make|\./)\b")
# The checker's own invocation, quoted in a Test Plan: `grep -niE '\bmoat\b'` and
# `jargon.py --port x` name the vocabulary because looking for it is their job.
_OWN_CHECK = re.compile(r"^\s*(?:\$\s*)?(?:grep|rg)\b|jargon", re.I)


# An embedded image or key is not prose. A long enough run of base64 spells every
# short word eventually, and a logo checked into a port flagged two terms that way.
_DATA_RUN = re.compile(r"[A-Za-z0-9+/=_-]{120,}")


def _fence_states(lines):
    """Per line: "fence" for a delimiter, "code" inside a block, else "prose".

    A fence that is never closed does not swallow the rest of the text. One stray
    backtick line in a commit message used to turn everything after it -- in that
    message and, in a concatenated scan, in every later one -- into "code" that was
    never read; the unpaired opener is treated as an ordinary line instead."""
    marks = [i for i, line in enumerate(lines) if line.lstrip().startswith("```")]
    if len(marks) % 2:
        marks.pop()
    states, inside, marks = [], False, set(marks)
    for i in range(len(lines)):
        if i in marks:
            inside = not inside
            states.append("fence")
        else:
            states.append("code" if inside else "prose")
    return states


def scan_text(text, label, terms, allow):
    """Scan one piece of text: a title, a body, one commit message, one file's
    added lines. Pass each separately (scan_units) rather than joined, so code
    blocks pair up within the piece they belong to.

    Prose is searched for every term. A match of an `allow` pattern -- a URL -- is
    masked and the rest of its line still read: skipping the whole line hid
    everything beside a link, and a GitHub body is one line per paragraph.

    Code blocks and command lines are searched for the `fenced` table only: the
    names that are wrong wherever they appear (our tools, our records, a path on
    one of our machines), not the prose terms a command may legitimately spell. A
    Test Plan that quotes the jargon check itself is left alone; flagging it
    trains people to ignore the tool."""
    hits = []
    fenced = getattr(terms, "fenced", {})
    lines = text.splitlines()
    for i, (line, state) in enumerate(zip(lines, _fence_states(lines)), 1):
        if state == "fence":
            continue
        stripped = line.strip()
        command = state == "code" or (line.startswith(("    ", "\t"))
                                      and bool(_COMMAND_LINE.match(line)))
        if command:
            if _OWN_CHECK.search(line):
                continue
            table = fenced
        else:
            if stripped.startswith(("#", "//")) and "jargon" in stripped.lower():
                continue
            table = {**terms, **fenced}
        masked = _DATA_RUN.sub(lambda m: " " * len(m.group(0)), line)
        for a in allow:
            masked = a.sub(lambda m: " " * len(m.group(0)), masked)
        for pat, fix in table.items():
            m = re.search(pat, masked, re.I)
            if m:
                hits.append((label, i, m.group(0), fix, stripped[:90]))
    return hits


def scan_units(units, label, terms, allow):
    """scan_text over [(name, text)], each on its own."""
    hits = []
    for name, text in units:
        hits += scan_text(text, f"{label} {name}".strip(), terms, allow)
    return hits


def added_by_file(diff):
    """[(path, added lines)] from a unified diff, one entry per file."""
    units, path, lines = [], None, []
    for line in diff.splitlines():
        if line.startswith("+++ "):
            if path is not None:
                units.append((path, "\n".join(lines)))
            path, lines = line[4:].removeprefix("b/"), []
        elif line.startswith("+") and path is not None:
            lines.append(line[1:])
    if path is not None:
        units.append((path, "\n".join(lines)))
    return units


def port_range(name):
    """(repo, commits_range, diff_range) for a project's whole port branch.

    Read from the project's own record so nobody types a range: the base is
    `fork_default_branch`, never the previous head. Raises if the project or its fork
    clone is not here, because a check that silently scans nothing passes."""
    sys.path.insert(0, str(TOOLS_ROOT / "utils"))
    import moatlib
    obj, _where = moatlib.project_record(name)
    if obj is None:
        raise ValueError(f"{name}: no record found in this checkout or on the refs")
    repo = REPO_ROOT / "projects" / name / "src"
    if not (repo / ".git").exists():
        raise ValueError(f"{name}: no fork clone at {repo} -- nothing to scan")
    base = obj.get("fork_default_branch") or "main"
    branch = moatlib.upstream_visible_branch(obj)
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
    return scan_units(added_by_file(r.stdout), "added lines", terms, allow)


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
    import trunk  # noqa: E402 - run the trunk's copy of this tool, see trunk.py
    trunk.reexec(__file__)
    sys.exit(main())
