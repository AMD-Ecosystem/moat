#!/usr/bin/env python3
"""Every gate MOAT enforces, in one place.

CI and the pre-push hook both call this, so they cannot drift. Adding a gate here
adds it to both; implementing it twice is how the two ended up disagreeing before.

    python3 utils/check.py              # everything
    python3 utils/check.py --fast       # skip the forks gate (walks every fork clone)
    python3 utils/check.py schema readme

Exit 1 if any gate fails. Each gate prints one line per problem, prefixed with its
own name, so a CI log says which gate to look at.
"""

import argparse
import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import tomllib

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "utils"))

# Anything that has no business in a control-plane repo. The usual sources of
# these are .a/.so/.o from validators building in the repo root, and DEM CSV output
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
    # The board is a TRUNK artifact. A port branch carries whatever copy it inherited,
    # regenerating it there produces a commit that is discarded when the branch merges,
    # and judging it there is what serialises concurrent work: every record any host
    # pushes stales every other checkout's copy, so three agents pushing three
    # unrelated projects each fail on a table none of them touched.
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]).stdout.strip()
    if branch.startswith("port/"):
        return []
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
        if "stage" in d and d["stage"] not in m.STAGE_STATES:
            problems.append(f"{name}: unknown stage {d['stage']!r}")
        for arch, blk in (d.get("platforms") or {}).items():
            if m.platform_problem(arch):
                problems.append(f"{name}: unknown arch {arch!r} "
                                f"(add it to config/arches.toml)")
            if blk.get("state") is not None and blk["state"] not in m.STATES:
                problems.append(f"{name}/{arch}: unknown state {blk['state']!r}")
        w = d.get("waivers") or {}
        for gate, rec in w.items():
            if gate not in m.WAIVABLE_GATES:
                problems.append(f"{name}: gate {gate!r} is not waivable")
            # A missing approved_by is NOT a defect: it is what a suggestion is, and
            # suggesting one is how the obstacle reaches a person at all. What stops an
            # agent certifying its own way past a gate is that such a waiver satisfies
            # nothing and blocks pr_ready -- enforced there, where it bites, rather than
            # by failing the repo's own checks over a decision someone has yet to make.
            # Checked here instead: the record says what is being waived.
            elif not (rec.get("reason") or "").strip():
                problems.append(f"{name}: waiver on {gate!r} has no reason -- nobody can "
                                f"approve or refuse a case that is not stated")
    return problems


def gate_jargon():
    """The jargon config is loadable and its patterns compile. The jargon SCAN
    itself runs against a fork's commits, not this repo, so it belongs to the
    porter and the reviewer -- this only catches a malformed config that would
    make those checks silently pass."""
    import re as _re
    try:
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


def gate_optout():
    """The opt-out record is well-formed, and nothing it covers is still adopted.

    A malformed entry here fails open -- the filters would stop matching and the
    repos would quietly come back into the queue -- so the shape is checked rather
    than assumed. The second half catches the case that matters: an opt-out recorded
    while a project from that owner is still in the pipeline. pr_ready blocks the
    route upstream on its own, but the project sitting there adopted reads as live
    work, and someone has to retire it."""
    import moatlib as m
    problems = []
    d = m.load_optouts()
    if not isinstance(d, dict):
        return ["data/optout.json is not an object"]
    for key, v in d.items():
        if not isinstance(v, dict):
            problems.append(f"{key}: not an object")
            continue
        if key != (v.get("who") or "").lower():
            problems.append(f"{key}: key does not match who={v.get('who')!r}")
        if v.get("scope") not in ("owner", "repo"):
            problems.append(f"{key}: scope must be owner or repo, got {v.get('scope')!r}")
        if not v.get("source"):
            problems.append(f"{key}: no source -- an opt-out must say where it was asked for")
    for name in sorted(m.all_projects()):
        rec, _where = m.project_record(name)
        if not rec:
            continue
        full = (rec.get("upstream_url") or "").split("github.com/", 1)[-1]
        if not full or not m.optout_for(full):
            continue
        disp = m.get_disposition(full) or {}
        if disp.get("disposition") != "skip":
            problems.append(f"{name} ({full}): its owner opted out but the project is "
                            f"still live at stage {rec.get('stage')} -- "
                            f"python3 utils/optout.py record retires it")
    return problems


def gate_surface():
    """Where a project has a surface.json, every component is either covered or
    explicitly scoped out with a reason. The gate is ACCOUNTING, not coverage: a
    scoped-out component is a deliberate, reviewable decision, and the failure being
    eliminated is the SILENT omission -- a port that claimed success while covering a
    subset.

    Few projects carry one: planner.md now instructs `surface.py generate`, most
    existing projects predate that instruction, and nothing backfills one (colmap,
    on its own branch, was the first). The check globs THIS working tree only --
    a surface.json on another project's branch is invisible here -- so on most
    checkouts the gate passes vacuously and says so out loud rather than reading
    as a clean bill."""
    r = _run([sys.executable, "utils/surface.py", "check", "--all"])
    if r.returncode == 0:
        if not list((REPO / "projects").glob("*/surface.json")):
            print("surface: VACUOUS -- no project in this checkout has a "
                  "surface.json, so this gate judged nothing here (see planner.md; "
                  "`surface.py generate <name>`)",
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


def gate_published():
    """No local clone carries commits on an open PR's branch that the PR does not.

    A local branch is not evidence about a remote one -- it is only as fresh as the
    last fetch -- so this asks the one question a clone CAN answer: is its copy of
    the port branch AHEAD of the published tip? That is the precursor to an
    unapproved push, and the fix flow's answer to it is `moat-fix-<pr#>`. Behind is
    just a stale fetch and passes silently, which also keeps `--merge-fix` from
    failing this gate on the very host that performed the approved merge: that push
    goes to the fork by sha and leaves the local branch where it was.

    Diverged (neither one an ancestor) is reported too -- the branch was rebased or
    rewritten under an open PR. Where no clone exists the gate says it judged
    nothing; the reconciler (`upstream.py --dry-run`) covers the remote side."""
    sys.path.insert(0, str(REPO / "utils"))
    import moatlib
    problems, judged = [], 0
    for src in sorted((REPO / "projects").glob("*/src/.git")):
        name = src.parent.parent.name
        try:
            obj, _where = moatlib.project_record(name)
        except Exception:
            continue
        if not obj or obj.get("pr_state") != "open":
            continue
        judged += 1
        pub = obj.get("published_sha")
        if not pub:
            # Records from before this field existed are stamped by the
            # reconciliation sweep once it has verified the live PR head agrees
            # with the record. fix-branch also backfills, but it opens a fix
            # round; the sweep stamps without starting one.
            print(f"published: note -- {name} has an open PR but no published_sha "
                  f"recorded (predates the fix flow; `utils/upstream.py --apply` "
                  f"backfills it after verifying against the live PR)",
                  file=sys.stderr)
            continue
        branch = obj.get("fork_branch") or moatlib.PORT_BRANCH
        clone = str(src.parent)
        tip = _run(["git", "-C", clone, "rev-parse", "--verify", "--quiet",
                    f"refs/heads/{branch}"]).stdout.strip()
        if not tip or moatlib.same_commit(tip, pub):
            continue                      # no local copy of the branch, or in step
        if _run(["git", "-C", clone, "cat-file", "-e", f"{pub}^{{commit}}"]).returncode:
            continue                      # the published commit was never fetched here
        if not _run(["git", "-C", clone, "merge-base", "--is-ancestor",
                     tip, pub]).returncode:
            continue                      # behind: a stale fetch is not a problem
        ahead = not _run(["git", "-C", clone, "merge-base", "--is-ancestor",
                          pub, tip]).returncode
        count = _run(["git", "-C", clone, "rev-list", "--count",
                      f"{pub}..{tip}"]).stdout.strip() or "?"
        problems.append(
            f"{name}: local {branch} carries {count} commit(s) the open upstream PR "
            f"({pub[:12]}) does not"
            + ("" if ahead else ", and is not descended from it -- the branch was "
                                "rewritten under an open PR")
            + f". A fix goes on moat-fix-<pr#> (`moatlib.py fix-branch {name}`); "
              f"only `upstream.py --merge-fix --apply` moves {branch}")
    if not problems and judged == 0:
        print("published: VACUOUS -- no project in this checkout has both an open "
              "PR and a fork clone, so this gate judged nothing here",
              file=sys.stderr)
    return problems


def gate_gh_guard():
    """The gh guard classifies correctly, and on a working host it is actually wired in.

    This gate is what makes the `gh api` allowance in .claude/settings.json safe. That
    rule exists because read-only queries should not prompt, and no prefix or wildcard
    rule can separate a query from a write: the deciding flag comes after arbitrary
    text and can be spelled -f, -F, --input, or -X POST in any order. So the harness
    permits `gh api` broadly and gh_guard is the thing that actually refuses the
    writes. If the guard stops working, that allowance silently becomes the hole it was
    written to close, which is a gate failure rather than a nuisance.

    The classifier cases run everywhere, including CI, because they are pure logic. The
    installation half is skipped under CI, where no agent runs and ~/.local/bin is not
    expected on PATH.
    """
    import shutil
    import gh_guard
    import install_hooks

    problems = [f"gh_guard misclassifies `gh {' '.join(argv)}`"
                for argv, want in gh_guard.CASES
                if gh_guard.classify(argv).allow != want]
    problems += install_hooks.launcher_case_problems()

    if os.environ.get("CI") or gh_guard.real_gh() is None:
        return problems

    shim = install_hooks.gh_launcher_path()
    problem = install_hooks.gh_launcher_problem(shim)
    if problem:
        problems.append(problem)
    else:
        resolved = shutil.which("gh")
        if not resolved or pathlib.Path(resolved).resolve() != shim.resolve():
            problems.append(
                f"gh guard is installed but inert: `gh` resolves to "
                f"{resolved or 'nothing'}, not {shim}. Put {shim.parent} first in PATH.")
    return problems


def _skill_metadata(path):
    """The two discovery fields in a skill's small YAML frontmatter."""
    lines = path.read_text().splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError("missing opening ---")
    try:
        end = lines.index("---", 1)
    except ValueError as e:
        raise ValueError("missing closing ---") from e
    fields = {}
    for line in lines[1:end]:
        if ":" in line:
            key, value = line.split(":", 1)
            fields[key.strip()] = value.strip()
    return fields.get("name"), fields.get("description")


def gate_harness_adapters():
    """Claude Code and Codex both discover the same rules, skills, and roles.

    Shared behavior has one canonical copy. Claude Code imports AGENTS.md and reads the
    canonical `.claude` skills and roles; Codex discovers checked-in `.agents` and
    `.codex` adapters that point back to those canonical files. This gate makes adding
    only one side impossible to miss and hands the Codex config to a real Codex binary
    when one is installed.
    """
    import shutil

    problems = []
    agents_md = REPO / "AGENTS.md"
    claude_md = REPO / "CLAUDE.md"
    cfg_path = REPO / ".codex" / "config.toml"

    if not agents_md.exists():
        problems.append("AGENTS.md is missing")
        agents_size = 0
    else:
        agents_size = agents_md.stat().st_size
        # Codex's documented default combined project-instruction limit is 32 KiB.
        if agents_size > 32768:
            problems.append(f"AGENTS.md is {agents_size} bytes; Codex would truncate "
                            "it at the 32768-byte project instruction limit")

    if not claude_md.exists():
        problems.append("CLAUDE.md is missing")
    else:
        first = next((line.strip() for line in claude_md.read_text().splitlines()
                      if line.strip()), "")
        if first != "@AGENTS.md":
            problems.append("CLAUDE.md must import the canonical rules with @AGENTS.md "
                            "as its first non-empty line")

    config = None
    if not cfg_path.exists():
        problems.append(".codex/config.toml is missing")
    else:
        try:
            config = tomllib.loads(cfg_path.read_text())
        except (OSError, tomllib.TOMLDecodeError) as e:
            problems.append(f".codex/config.toml will not load: {e}")
        if config is not None:
            unknown = set(config) - {"project_doc_max_bytes"}
            if unknown:
                problems.append(".codex/config.toml has unsupported project keys: "
                                + ", ".join(sorted(unknown)))
            configured_limit = config.get("project_doc_max_bytes", 32768)
            if not isinstance(configured_limit, int) or configured_limit < agents_size:
                problems.append(".codex/config.toml project_doc_max_bytes is smaller "
                                "than AGENTS.md")

    canonical_skills = {
        p.parent.name: p for p in (REPO / ".claude" / "skills").glob("*/SKILL.md")
    }
    codex_skills = {
        p.parent.name: p for p in (REPO / ".agents" / "skills").glob("*/SKILL.md")
    }
    for name in sorted(canonical_skills.keys() - codex_skills.keys()):
        problems.append(f".agents/skills/{name}/SKILL.md adapter is missing")
    for name in sorted(codex_skills.keys() - canonical_skills.keys()):
        problems.append(f".agents/skills/{name}/SKILL.md has no canonical skill")
    for name in sorted(canonical_skills.keys() & codex_skills.keys()):
        canonical = canonical_skills[name]
        adapter = codex_skills[name]
        try:
            canonical_meta = _skill_metadata(canonical)
            adapter_meta = _skill_metadata(adapter)
        except (OSError, ValueError) as e:
            problems.append(f"skill {name}: invalid frontmatter: {e}")
            continue
        if canonical_meta != adapter_meta:
            problems.append(f"skill {name}: adapter name/description differs from "
                            "the canonical skill")
        canonical_ref = f".claude/skills/{name}/SKILL.md"
        if canonical_ref not in adapter.read_text():
            problems.append(f"skill {name}: adapter does not load {canonical_ref}")

    canonical_roles = {
        p.stem: p for p in (REPO / ".claude" / "agents").glob("*.md")
    }
    codex_roles = {
        p.stem: p for p in (REPO / ".codex" / "agents").glob("*.toml")
    }
    for name in sorted(canonical_roles.keys() - codex_roles.keys()):
        problems.append(f".codex/agents/{name}.toml adapter is missing")
    for name in sorted(codex_roles.keys() - canonical_roles.keys()):
        problems.append(f".codex/agents/{name}.toml has no canonical role")
    for name in sorted(canonical_roles.keys() & codex_roles.keys()):
        try:
            role = tomllib.loads(codex_roles[name].read_text())
        except (OSError, tomllib.TOMLDecodeError) as e:
            problems.append(f"role {name}: adapter will not load: {e}")
            continue
        if role.get("name") != name:
            problems.append(f"role {name}: adapter name must be {name!r}")
        if not role.get("description"):
            problems.append(f"role {name}: adapter has no description")
        if role.get("sandbox_mode") != "workspace-write":
            problems.append(f"role {name}: adapter sandbox_mode must be workspace-write")
        canonical_ref = f".claude/agents/{name}.md"
        if canonical_ref not in role.get("developer_instructions", ""):
            problems.append(f"role {name}: adapter does not load {canonical_ref}")

    codex = shutil.which("codex")
    if config is not None and codex:
        # This parses the complete configuration stack without calling a model.
        # `features list` loads project config without contacting a model. Codex 0.147
        # exposes --strict-config globally but rejects it for this read-only command,
        # so the gate validates our intentionally tiny key set above and uses the CLI
        # to catch implementation-level parsing or config-stack failures.
        r = subprocess.run([codex, "features", "list"],
                           cwd=str(REPO), capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            detail = (r.stderr or r.stdout).strip().splitlines()
            problems.append("Codex refuses this checkout's config: "
                            + " / ".join(detail[:3])[:300])
    return problems


GATES = {
    "gh-guard": (gate_gh_guard, False),
    "harness-adapters": (gate_harness_adapters, False),
    "code": (gate_code, False),
    "schema": (gate_schema, False),
    "readme": (gate_readme, False),
    "licenses": (gate_licenses, False),
    "blobs": (gate_blobs, False),
    "states": (gate_states, False),
    "jargon": (gate_jargon, False),
    "optout": (gate_optout, False),
    "surface": (gate_surface, False),
    "published": (gate_published, False),
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
