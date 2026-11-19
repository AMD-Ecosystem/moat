#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Every gate MOAT enforces, in one place.

CI and the pre-push hook both call this, so they cannot drift. Adding a gate here
adds it to both; implementing it twice is how the two ended up disagreeing before.

Several gates scope themselves to what the push in progress updates, which they read
from MOAT_PUSH_UPDATES and MOAT_PUSH_REFS. The hook exports both from the refs git
hands it; CI exports the same two from the push event (see .github/workflows/ci.yml).
A gate must not ask "am I in CI" to decide its scope: when only the hook set those
variables, CI judged a port branch by the trunk-owned files it had inherited and
failed 28 pushes of one branch over a skill file the branch could not fix. (readme
is the one gate that consults CI, and not for scope: see its docstring.)

    python3 utils/check.py              # everything
    python3 utils/check.py --fast       # skip the forks and commits gates (per fork clone)
    python3 utils/check.py schema readme

Exit 1 if any gate fails. Each gate prints one line per problem, prefixed with its
own name, so a CI log says which gate to look at.
"""

import argparse
import importlib.metadata
import importlib.util
import json
import os
import pathlib
import re
import subprocess
import sys
import tomllib

# The checkout this acts on. MOAT_REPO is set when the trunk's copy of the tools runs
# against another checkout (see trunk.py); config and schema come from TOOLS_ROOT.
TOOLS_ROOT = pathlib.Path(__file__).resolve().parents[1]
REPO = pathlib.Path(os.environ["MOAT_REPO"]).resolve() if os.environ.get("MOAT_REPO") else TOOLS_ROOT
sys.path.insert(0, str(TOOLS_ROOT / "utils"))

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
    # UTF-8 explicitly: the locale code page on Windows raises on a non-ASCII
    # commit subject or path in a tool's output.
    kw.setdefault("encoding", "utf-8")
    kw.setdefault("errors", "replace")
    return subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO), **kw)


ZERO = "0" * 40


def push_updates():
    """(local sha, remote ref, remote sha) for each ref the push in progress updates.
    The pre-push hook exports it from git's own list and CI from the push event, so
    both judge a push the same way. None outside a push (a manual run, a pull request)."""
    raw = os.environ.get("MOAT_PUSH_UPDATES")
    if raw is None:
        return None
    out = []
    for item in raw.split():
        lsha, _, rest = item.partition(":")
        rref, _, rsha = rest.rpartition(":")
        if lsha and rref:
            out.append((lsha, rref, rsha))
    return out


def port_only_push():
    """True when the push in progress updates only port/<name> branches.

    A port branch may change nothing outside its own project folder (gate_branch_scope),
    so the gates that judge trunk-owned files -- the tooling's own lint, the generated
    schema, the skills, the harness adapters -- have nothing of that branch's to judge.
    Running them there judged whatever copy of those files the branch inherited, and
    failed pushes over files the push never touched."""
    ups = push_updates()
    return bool(ups) and all(r.startswith("refs/heads/port/") for _, r, _ in ups)


def gate_branch_scope():
    """A push to `port/<name>` adds no commit that changes anything outside
    `projects/<name>/`.

    The trunk owns the tooling, the skills and the shared data. A port branch that edits
    them carries its own version of trunk files, which is what made every trunk merge
    into a branch, and every branch merge into the trunk, a conflict: thirty-seven
    branches held edits to the same few skill and data files. Tooling fixes and lessons
    go to the trunk; a port records a lesson in its notes until it is promoted there.

    Only commits this push adds are judged, and merge commits are not: syncing the trunk
    into a branch is a merge, and an old branch's earlier edits stay until it lands.
    README.md is allowed because branch tooling regenerates it and the trunk's copy wins
    every merge."""
    ups = push_updates()
    if not ups:
        return []
    problems = []
    for lsha, rref, rsha in ups:
        if not rref.startswith("refs/heads/port/") or lsha == ZERO:
            continue
        name = rref[len("refs/heads/port/"):]
        own = f"projects/{name.lower()}/"
        rng = [lsha, "^origin/main"]
        if rsha and rsha != ZERO and _run(["git", "cat-file", "-e", rsha]).returncode == 0:
            rng.append(f"^{rsha}")
        commits = _run(["git", "rev-list", "--no-merges", *rng]).stdout.split()
        for c in commits:
            paths = _run(["git", "diff-tree", "--no-commit-id", "--name-only", "-r",
                          c]).stdout.split("\n")
            foreign = [x for x in (q.strip() for q in paths) if x
                       and not x.lower().startswith(own) and x != "README.md"]
            if foreign:
                subj = _run(["git", "log", "-1", "--format=%s", c]).stdout.strip()
                problems.append(
                    f"port/{name}: {c[:9]} ({subj[:50]}) changes {', '.join(foreign[:3])}"
                    f"{' ...' if len(foreign) > 3 else ''} outside projects/{name}/ -- "
                    f"tooling, skills and shared data change on main; record a lesson "
                    f"in the project's notes for promotion")
    return problems


# The pyflakes CI installs (.github/workflows/ci.yml; tests/test_ci_scope.py holds the
# two together). pyflakes gains checks between releases, so a host on another version
# can pass what CI fails.
PYFLAKES_PIN = "4.0.3"


def gate_code():
    """The Python actually resolves: no undefined names, no dead imports or locals.

    check.py validated data thoroughly and code not at all, and py_compile -- the only
    code check any Test Plan ran -- cannot see an undefined name. Two shipped that way,
    both on SUCCESS paths, which is the shape only production traffic finds: the
    approval gate raised NameError precisely when it meant to authorize, and the
    record-sync PR raised when opening a new one rather than updating.

    Fails when pyflakes is absent rather than skipping, on the same principle as the
    schema gate: a check that quietly does not run is worse than one that is missing."""
    if port_only_push():
        return []
    if importlib.util.find_spec("pyflakes") is None:
        return ["pyflakes not installed (pip install pyflakes); cannot check code"]
    have = importlib.metadata.version("pyflakes")
    if have != PYFLAKES_PIN:
        # A warning, not a failure: the result below still stands for this host, but
        # CI may disagree with it in either direction.
        print(f"check: code: this host has pyflakes {have} and CI runs {PYFLAKES_PIN}, "
              f"so the two can disagree (pip install pyflakes=={PYFLAKES_PIN})",
              file=sys.stderr)
    files = sorted(str(p) for p in (REPO / "utils").glob("*.py"))
    r = _run([sys.executable, "-m", "pyflakes", *files])
    out = (r.stdout + r.stderr).strip()
    return [ln.replace(str(REPO) + "/", "") for ln in out.splitlines() if ln.strip()]


def gate_schema():
    """status.json schema is generated from moatlib, and every file validates."""
    problems = []
    # The schema file is a trunk artifact; on a port-only push the branch's copy is not
    # in play, and the records are validated against the trunk's schema below.
    if not port_only_push():
        r = _run([sys.executable, str(TOOLS_ROOT / "utils" / "gen_schema.py"), "--check"])
        if r.returncode:
            problems.append("schema/status.schema.json is stale (run utils/gen_schema.py)")
    try:
        import jsonschema
    except ImportError:
        problems.append("jsonschema not installed; cannot validate status files")
        return problems
    import moatlib as m
    schema = json.loads((TOOLS_ROOT / "schema" / "status.schema.json").read_text())
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

    Judged only in a pre-push hook that is updating the trunk, and in a manual run.
    The table renders across refs, so whether it is current depends on the state of
    every port branch at the moment of the push. The pusher can act on that: the hook
    refuses and they regenerate. A hosted CI run cannot: it starts after the push, and
    any port branch that recorded a state in between makes the trunk's table read
    stale for a commit that was correct when pushed (775f8a2e failed that way twenty
    seconds after an unrelated port/jitllm push). So CI skips it, loudly, whatever
    refs its checkout happens to hold."""
    if os.environ.get("GITHUB_ACTIONS") == "true":
        print("readme: skipped in CI -- the table depends on every port branch at "
              "push time; the pre-push hook judges it", file=sys.stderr)
        return []
    # The board is a TRUNK artifact. A port branch carries whatever copy it inherited,
    # regenerating it there produces a commit that is discarded when the branch merges,
    # and judging it there is what serialises concurrent work: every record any host
    # pushes stales every other checkout's copy, so three agents pushing three
    # unrelated projects each fail on a table none of them touched.
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]).stdout.strip()
    if branch.startswith("port/"):
        return []
    # The checkout's branch is the wrong question for a push: recording a fact writes a
    # commit straight to `port/<name>` from whatever is checked out (the publisher's
    # approval snapshot does), and judging the trunk's table there refused the push
    # because some other project's state had moved. The pre-push hook passes the refs
    # being updated; unless one of them is the trunk, the table is not in play.
    pushed = os.environ.get("MOAT_PUSH_REFS")
    if pushed is not None and "refs/heads/main" not in pushed.split():
        return []
    if not _run(["git", "for-each-ref", "--format=%(refname)",
                 "refs/remotes/origin/port/"]).stdout.strip():
        print("readme: skipped -- no port branches in this checkout, so the table "
              "cannot be verified here", file=sys.stderr)
        return []
    r = _run([sys.executable, str(TOOLS_ROOT / "utils" / "gen_readme.py"), "--check"])
    return [] if r.returncode == 0 else ["README table is stale (run utils/gen_readme.py)"]


def gate_licenses():
    """Tier lists are well-formed, and no identifier sits in two tiers -- which
    would silently disable the review gate for that licence."""
    if port_only_push():
        return []
    r = _run([sys.executable, str(TOOLS_ROOT / "utils" / "licenses.py"), "--check-config"])
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
        cfg = tomllib.load(open(TOOLS_ROOT / "config" / "jargon.toml", "rb"))
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
        # A fork a person ruled we keep working is not retired by an opt-out: it
        # offers no pull request, and its branch stays on our fork.
        if rec.get("maintained"):
            continue
        disp = m.get_disposition(full) or {}
        if disp.get("disposition") != "skip":
            problems.append(f"{name} ({full}): its owner opted out but the project is "
                            f"still live at stage {rec.get('stage')} -- "
                            f"{m.PY} utils/optout.py record retires it")
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
    r = _run([sys.executable, str(TOOLS_ROOT / "utils" / "surface.py"), "check", "--all"])
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
    r = _run([sys.executable, str(TOOLS_ROOT / "utils" / "moatlib.py"), "audit-clean"])
    if r.returncode == 0:
        return []
    return [l for l in r.stdout.splitlines() if l.strip()][:10]


def gate_commits():
    """Fork commits obey the message rules AGENTS.md states.

    Those rules were written down and never executed. jargon.py walks the same range but
    only for in-house vocabulary; pr_intent.py checks the PULL REQUEST title, which is a
    different string. So a body missing its Test Plan could reach a maintainer unremarked,
    and publication sends the branch as it stands -- `upstream.py --publish` opens the pull
    request with `--head <fork>:<branch>` and squashing first is supported but not
    enforced.

    Slow for the same reason as the fork gate: it shells out per clone.
    """
    r = _run([sys.executable, str(TOOLS_ROOT / "utils" / "moatlib.py"), "audit-commits"])
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
    sys.path.insert(0, str(TOOLS_ROOT / "utils"))
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
    if port_only_push():
        return []
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


# Spelled by construction so this file cannot trip its own gate.
CONFLICT_START = "<" * 7 + " "
CONFLICT_MID = "=" * 7
CONFLICT_END = ">" * 7 + " "


def gate_conflict_markers():
    """No tracked file still carries an unresolved merge conflict.

    `git add` on a conflicted file marks it RESOLVED, so committing the markers
    themselves needs no force and produces no warning. A trunk-merge sweep across the
    port branches does that resolution in bulk, which is where one slip becomes many
    bad branches. Caught exactly that way on 2026-08-19: a resolution reached
    port/colmap with the markers still in a skill reference, and every other gate
    reported ok.

    A conflict is the full start/middle/end triple in order, so a markdown setext
    underline or a row of equals signs on its own is not one. An unterminated start is
    reported too: prose about conflicts does not begin a line with the marker.
    """
    problems = []
    r = _run(["git", "ls-files", "-z"])
    for path in r.stdout.split(chr(0)):
        if not path:
            continue
        p = REPO / path
        if not p.is_file():
            continue
        try:
            lines = p.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            continue                     # binary or unreadable: not this gate's business
        start = mid = None
        for n, line in enumerate(lines, 1):
            if line.startswith(CONFLICT_START):
                start, mid = n, None
            elif start and mid is None and line.rstrip() == CONFLICT_MID:
                mid = n
            elif start and mid and line.startswith(CONFLICT_END):
                problems.append(f"{path}:{start}: unresolved merge conflict committed "
                                f"(`git add` alone only marks it resolved)")
                start = mid = None
        if start:
            problems.append(f"{path}:{start}: merge conflict marker with no end marker")
    return problems


# cuda-to-rocm is published outside MOAT as an AMD skill, so the files an agent loads
# from it must stand alone. MOAT-only practice lives in the moat-porting skill. The
# skill card and evals are catalog metadata, not guidance, and are not scanned.
STANDALONE_SKILL = REPO / ".claude" / "skills" / "cuda-to-rocm"
# One group of alternatives per class of leak. Each is narrow on purpose: "validation",
# "validated", a documentation URL with /projects/ in it and the rocm-libraries
# monorepo's own projects/<library> paths are ordinary technical text and must pass
# (tests/test_skill_vocabulary.py holds both kinds of line).
MOAT_VOCABULARY = re.compile(
    # the repository, its tools and its records
    r"\bMOAT\b|moatlib|(?<![\w/.])utils/|\bnotes\.md\b|\bplan\.md\b|\bstatus\.json\b"
    r"|\bsurface\.json\b|\bdeferred\.(?:py|json)\b|\bfleet\b"
    r"|\b(?:head|validated|published)_sha\b|cuda-not-validated|codeobj_diff"
    r"|advance[_-]head|config/(?:arches|jargon)\.toml|\bdepends_on\b"
    r"|(?<![\w/])_deps/|(?<![\w/])findings/|/var/lib/jenkins|\bupstream-issues\b"
    # a project folder: projects/<name>, or a real one such as projects/TornadoVM
    r"|(?<![\w/.-])projects/(?!(?:hipcub|rocprim|rocthrust)\b)[\w<]"
    # the fork organization, except the one upstream project that lives there
    r"|\bAMD-Ecosystem/(?!hipMM\b)"
    # branch and state names, dispositions
    r"|\bmoat-(?:port|fix|sync)\b|\bport/<|\brevalidat|\bcarry-forward\b"
    r"|\b(?:validation-failed|port-ready|review-passed|changes-requested|delta-ported)\b"
    r"|\b(?:cant-port|not-portable|already-supported)\b"
    # deferral ids and rulings
    r"|\bdeferrals?\b"
    # role names and the workflow around them
    r"|\bporters?\b|\bvalidators?\b|\breviewer role\b|\bintake\b"
    r"|\bintegrity gate\b|\bretry budget\b|\bstop[- ]discipline\b|\bfork commit\b"
    # one host described as a fixture
    r"|\bthe gfx\d\w* host\b",
    re.IGNORECASE)


def gate_skills():
    """The cuda-to-rocm skill names no MOAT tool, record, or state, and its generated
    entry index and contents blocks match the reference files."""
    if port_only_push():
        return []
    import gen_skill_index
    problems = [f"{rel}: generated index or contents is stale (run utils/gen_skill_index.py)"
                for rel in gen_skill_index.stale(REPO)]
    paths = [STANDALONE_SKILL / "SKILL.md"]
    for sub in ("references", "assets"):
        paths += sorted(p for p in (STANDALONE_SKILL / sub).rglob("*") if p.is_file())
    for path in paths:
        rel = path.relative_to(REPO).as_posix()
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            m = MOAT_VOCABULARY.search(line)
            if m:
                problems.append(f"{rel}:{n}: MOAT vocabulary {m.group(0)!r}; move it to "
                                f".claude/skills/moat-porting/ or say it generically")
    return problems


GATES = {
    "gh-guard": (gate_gh_guard, False),
    "branch-scope": (gate_branch_scope, False),
    "skills": (gate_skills, False),
    "harness-adapters": (gate_harness_adapters, False),
    "code": (gate_code, False),
    "schema": (gate_schema, False),
    "readme": (gate_readme, False),
    "licenses": (gate_licenses, False),
    "blobs": (gate_blobs, False),
    "conflicts": (gate_conflict_markers, False),
    "states": (gate_states, False),
    "jargon": (gate_jargon, False),
    "optout": (gate_optout, False),
    "surface": (gate_surface, False),
    "published": (gate_published, False),
    "forks": (gate_forks, True),
    "commits": (gate_commits, True),   # slow: shells out per fork clone
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # No `choices` here: Python 3.11 argparse checks the empty default of a
    # nargs="*" positional against them and rejects a bare `check.py --fast`.
    ap.add_argument("gates", nargs="*", metavar="gate",
                    help="gates to run (default: all): " + ", ".join(GATES))
    ap.add_argument("--fast", action="store_true", help="skip slow gates")
    a = ap.parse_args()
    unknown = [g for g in a.gates if g not in GATES]
    if unknown:
        ap.error(f"unknown gate: {', '.join(unknown)} (choose from {', '.join(GATES)})")

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
    import trunk  # noqa: E402 - run the trunk's copy of this tool, see trunk.py
    trunk.reexec(__file__)
    sys.exit(main())
