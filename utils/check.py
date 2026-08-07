#!/usr/bin/env python3
"""Every gate MOAT enforces, in one place.

CI and the pre-push hook both call this, so they cannot drift. Adding a gate here
adds it to both; implementing it twice is how the two ended up disagreeing before.

    python3 utils/check.py              # everything
    python3 utils/check.py --fast       # skip anything that shells out to git/gh
    python3 utils/check.py schema readme

Exit 1 if any gate fails. Each gate prints one line per problem, prefixed with its
own name, so a CI log says which gate to look at.
"""

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "utils"))

# Anything that has no business in a control-plane repo. The usual source is
# of these -- .a/.so/.o from validators building in the repo root, and DEM CSV output
# -- which is why the migration had to squash rather than filter.
BLOB_SUFFIXES = {".a", ".so", ".o", ".dylib", ".dll", ".lib", ".exe", ".whl",
                 ".tar", ".gz", ".zip", ".7z", ".bin", ".pt", ".pth", ".onnx"}
BLOB_MAX_BYTES = 1_000_000

# Large files that are genuinely data, not build spill. Each needs a reason: the
# point of the size gate is that someone justifies the next big file rather than
# quietly raising the limit until it stops firing.
BLOB_ALLOW = {
    "projects/popsift/reference/": (
        "canonical SIFT descriptor output (gfx90a/MI250X) used as the cross-arch "
        "comparison oracle; referenced from popsift/notes.md. Real validation data, "
        "and the reason MOAT can prove popsift matches across architectures."),
}


def _run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO), **kw)


def gate_schema():
    """status.json schema is generated from moatlib, and every file validates."""
    problems = []
    r = _run([sys.executable, "utils/gen_schema.py", "--check"])
    if r.returncode:
        problems.append("schema/status.schema.json is stale (run utils/gen_schema.py)")
    try:
        import jsonschema
    except ImportError:
        problems.append("jsonschema not installed; cannot validate status files")
        return problems
    schema = json.loads((REPO / "schema" / "status.schema.json").read_text())
    v = jsonschema.Draft202012Validator(schema)
    for sp in sorted((REPO / "projects").glob("*/status.json")):
        for e in sorted(v.iter_errors(json.loads(sp.read_text())), key=lambda e: list(e.path))[:1]:
            problems.append(f"{sp.relative_to(REPO)}: {'.'.join(map(str, e.path))}: {e.message[:120]}")
    return problems


def local_port_branch():
    """True only when this is a local run on a port branch -- not CI.

    CI sets GITHUB_* ; a pull request there builds the MERGE commit, so the tree is
    exactly what the trunk will become and every gate should see it."""
    if os.environ.get("GITHUB_ACTIONS") or os.environ.get("GITHUB_HEAD_REF"):
        return False
    r = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    return r.returncode == 0 and r.stdout.strip().startswith("port/")


def gate_readme():
    """The generated project table matches the data it claims to describe.

    Skipped for a LOCAL push to a port branch, and only there. Mid-port the tree is
    the trunk plus one project, so the table differs for a row that belongs on the
    trunk once the port lands; making every push regenerate it is friction with no
    payoff. Three of the four agents on the 2026-08-06 dry run regenerated the README
    purely to get a push through.

    It is NOT skipped in CI. A pull request builds the merge commit, so `gen_readme`
    there describes the trunk as it will be, and the gate fails on the pull request
    that has to fix it. An earlier version of this skipped on the branch name alone,
    which let the offending PR go green and dropped the failure on whoever next
    pushed to the trunk -- a stale table is one command to fix, but it should not be
    a stranger's command."""
    if local_port_branch():
        return []
    r = _run([sys.executable, "utils/gen_readme.py", "--check"])
    return [] if r.returncode == 0 else ["README table is stale (run utils/gen_readme.py)"]


def gate_licenses():
    """Tier lists are well-formed, and no identifier sits in two tiers -- which
    would silently disable the review gate for that licence."""
    r = _run([sys.executable, "utils/licenses.py", "--check-config"])
    return [] if r.returncode == 0 else [l for l in (r.stdout + r.stderr).splitlines() if l.strip()]


def gate_blobs():
    """No build artifacts or large binaries. Checks the working tree, not history."""
    problems = []
    r = _run(["git", "ls-files", "-z"])
    for path in r.stdout.split("\0"):
        if not path:
            continue
        p = REPO / path
        if not p.is_file():
            continue
        if p.suffix.lower() in BLOB_SUFFIXES:
            problems.append(f"{path}: build artifact ({p.suffix}) must not be tracked")
        elif any(path.startswith(a) for a in BLOB_ALLOW):
            continue
        elif p.stat().st_size > BLOB_MAX_BYTES:
            problems.append(f"{path}: {p.stat().st_size // 1000} KB exceeds "
                            f"{BLOB_MAX_BYTES // 1000} KB -- is this a build output?")
    return problems


def gate_states():
    """Every recorded state and arch is one moatlib knows. Catches a status.json
    written by an older checkout after a state rename."""
    import moatlib as m
    problems = []
    for sp in sorted((REPO / "projects").glob("*/status.json")):
        d = json.loads(sp.read_text())
        for arch, blk in (d.get("platforms") or {}).items():
            if m.platform_problem(arch):
                problems.append(f"{sp.parent.name}: unknown arch {arch!r} "
                                f"(add it to config/arches.toml)")
            if blk.get("state") not in m.STATES:
                problems.append(f"{sp.parent.name}/{arch}: unknown state {blk.get('state')!r}")
        w = d.get("waivers") or {}
        for gate, rec in w.items():
            if gate not in m.WAIVABLE_GATES:
                problems.append(f"{sp.parent.name}: gate {gate!r} is not waivable")
            elif not rec.get("approved_by"):
                problems.append(f"{sp.parent.name}: waiver on {gate!r} has no approved_by "
                                f"-- an agent cannot self-certify past a gate")
    return problems


def gate_jargon():
    """The jargon config is loadable and its patterns compile. The jargon SCAN
    itself runs against a fork's commits, not this repo, so it belongs to the
    porter and the reviewer -- this only catches a malformed config that would
    make those checks silently pass."""
    import re as _re
    try:
        import tomllib
        cfg = tomllib.load(open(REPO / "config" / "jargon.toml", "rb"))
    except Exception as e:
        return [f"config/jargon.toml will not load: {e}"]
    problems = []
    for pat in list(cfg.get("terms", {})) + cfg.get("allow", {}).get("patterns", []):
        try:
            _re.compile(pat)
        except _re.error as e:
            problems.append(f"pattern {pat!r} does not compile: {e}")
    if not cfg.get("terms"):
        problems.append("no terms defined -- the checker would pass everything")
    return problems


def gate_surface():
    """Where a project has a surface.json, every component is either covered or
    explicitly scoped out with a reason. The gate is ACCOUNTING, not coverage: a
    scoped-out component is a deliberate, reviewable decision, and the failure being
    eliminated is the SILENT omission -- a port that claimed success while covering a
    subset. Projects with no surface.json yet are not failed; the planner generates
    one, and requiring it retroactively for 164 existing projects would just be noise."""
    r = _run([sys.executable, "utils/surface.py", "check", "--all"])
    if r.returncode == 0:
        return []
    return [l.replace("surface: ", "") for l in r.stdout.splitlines() if l.strip()][:20]


def gate_forks():
    """No fork carries uncommitted source edits. A validation built against local
    changes that are not in the branch produces an unbuildable PR."""
    r = _run([sys.executable, "utils/moatlib.py", "audit-clean"])
    if r.returncode == 0:
        return []
    return [l for l in r.stdout.splitlines() if l.strip()][:10]


GATES = {
    "schema": (gate_schema, False),
    "readme": (gate_readme, False),
    "licenses": (gate_licenses, False),
    "blobs": (gate_blobs, False),
    "states": (gate_states, False),
    "jargon": (gate_jargon, False),
    "surface": (gate_surface, False),
    "forks": (gate_forks, True),      # slow: shells out per fork clone
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gates", nargs="*", choices=list(GATES) + [], default=None)
    ap.add_argument("--fast", action="store_true", help="skip slow gates")
    a = ap.parse_args()

    selected = a.gates or [g for g, (_, slow) in GATES.items() if not (a.fast and slow)]
    failed = 0
    for name in selected:
        fn, _ = GATES[name]
        try:
            problems = fn()
        except Exception as e:                       # a broken gate is a failed gate
            problems = [f"gate raised {type(e).__name__}: {e}"]
        if problems:
            failed += 1
            for p in problems:
                print(f"{name}: {p}")
        else:
            print(f"{name}: ok")
    if failed:
        print(f"\n{failed} gate(s) failed", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
