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
import importlib.util
import json
import pathlib
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


def gate_code():
    """The Python actually resolves: no undefined names, no dead imports or locals.

    check.py validated data thoroughly and code not at all, and py_compile -- the only
    code check any Test Plan ran -- cannot see an undefined name. Two shipped that way,
    both on SUCCESS paths, which is the shape only production traffic finds: the
    approval gate raised NameError precisely when it meant to authorize, and the
    record-sync PR raised when opening a new one rather than updating.

    Fails when pyflakes is absent rather than skipping, on the same principle as the
    schema gate: a check that quietly does not run is worse than one that is missing."""
    if importlib.util.find_spec("pyflakes") is None:
        return ["pyflakes not installed (pip install pyflakes); cannot check code"]
    files = sorted(str(p) for p in (REPO / "utils").glob("*.py"))
    r = _run([sys.executable, "-m", "pyflakes", *files])
    out = (r.stdout + r.stderr).strip()
    return [ln.replace(str(REPO) + "/", "") for ln in out.splitlines() if ln.strip()]


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
    import moatlib as m
    schema = json.loads((REPO / "schema" / "status.schema.json").read_text())
    v = jsonschema.Draft202012Validator(schema)
    # Across refs: a record on its own branch still has to validate, and a checkout
    # that cannot see it must not report the file as fine by not looking.
    for name in sorted(m.all_projects()):
        obj, where = m.project_record(name)
        if obj is None:
            continue
        for e in sorted(v.iter_errors(obj), key=lambda e: list(e.path))[:1]:
            problems.append(f"projects/{name}/status.json ({where}): "
                            f"{'.'.join(map(str, e.path))}: {e.message[:120]}")
    return problems


def gate_readme():
    """The generated project table matches the data it claims to describe.

    Only where the port branches are visible. The table now renders across refs, so a
    project in flight appears on the board -- which means the table cannot be
    reproduced without the refs it was generated from. A CI checkout fetches one
    branch, so it sees neither the branch-only rows nor the current state of a project
    that exists on both the trunk and its own branch, and calls both differences
    staleness. Eight runs failed that way, then four more after the first fix, every
    one a false alarm.

    Fetching every ref in CI would trade those for something worse: with the branches
    visible, any push to any port branch stales the trunk's table and fails the trunk's
    next push, for whoever happens to make it.

    So this is judged where it can be judged -- a full clone, which is what the
    pre-push hook runs in -- and skipped, loudly, where it cannot."""
    if not _run(["git", "for-each-ref", "--format=%(refname)",
                 "refs/remotes/origin/port/"]).stdout.strip():
        print("readme: skipped -- no port branches in this checkout, so the table "
              "cannot be verified here", file=sys.stderr)
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
    for name in sorted(m.all_projects()):
        d, _where = m.project_record(name)
        if d is None:
            continue
        for arch, blk in (d.get("platforms") or {}).items():
            if m.platform_problem(arch):
                problems.append(f"{name}: unknown arch {arch!r} "
                                f"(add it to config/arches.toml)")
            if blk.get("state") not in m.STATES:
                problems.append(f"{name}/{arch}: unknown state {blk.get('state')!r}")
        w = d.get("waivers") or {}
        for gate, rec in w.items():
            if gate not in m.WAIVABLE_GATES:
                problems.append(f"{name}: gate {gate!r} is not waivable")
            elif not rec.get("approved_by"):
                problems.append(f"{name}: waiver on {gate!r} has no approved_by "
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
    subset.

    It has never judged anything. Zero projects have a surface.json, because nothing
    writes one: `utils/surface.py` is named in no agent instruction, and planner.md
    describes the idea while telling the planner to write prose into plan.md, which
    this never reads. So the gate passes vacuously and reports a count of zero as if
    it were a clean bill.

    Left in place and made honest rather than deleted: the accounting it describes is
    worth having, and the missing piece is one instruction in planner.md plus a
    `surface.py generate` anyone can run. Saying "0 accounted for" out loud is what
    stops it reading as nine gates passing when it is eight."""
    r = _run([sys.executable, "utils/surface.py", "check", "--all"])
    if r.returncode == 0:
        if not list((REPO / "projects").glob("*/surface.json")):
            print("surface: VACUOUS -- no project has a surface.json, so this gate "
                  "judged nothing (see planner.md; `surface.py generate <name>`)",
                  file=sys.stderr)
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
    "code": (gate_code, False),
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
