#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Install MOAT's git hooks and the gh guard into this clone.

Two gates, both of the same kind: a rule you cannot skip under time pressure beats one
you are told to follow. The pre-push hook runs check.py before a push exists; the gh
guard (utils/gh_guard.py, installed as a PATH shim) refuses adoption and upstream
GitHub writes while the working directory is inside a MOAT checkout.

CI catches a bad push at PR time. The hook catches it before it exists, which is the
difference between a rule you are told to follow and one you cannot skip under time
pressure. Both call utils/check.py; CI additionally checks the pull request's title
(utils/pr_intent.py), which does not exist at push time, so a locally-clean push can
still fail CI on its title -- and only on that.

Both run the TRUNK's code. The hook goes in the clone's shared hooks directory and runs
`origin/main`'s check.py against whatever checkout is pushing (see trunk.py); the guard
launcher runs a host copy of the trunk's gh_guard.py. A port branch's own utils/ is
never what gates it.

    python3 utils/install_hooks.py              # install
    python3 utils/install_hooks.py --check      # is it installed and current?
    python3 utils/install_hooks.py --uninstall

Run once per clone. Hooks are not tracked by git, so a fresh clone has none --
orient.sh installs them, the same way it registers the status.json merge driver.

Everything installed here runs under the interpreter that installed it. The name
`python3` is not that interpreter on every host: on one it is an unrelated virtual
environment with none of the gates' dependencies, and a push then failed closed with
gate errors that named the wrong cause. So each script carries the installing
`sys.executable`, and probes for a usable Python only if that one has gone.

Bypassing the gates (`git push --no-verify`) is a person's call, never an agent's.
If a person finds themselves reaching for it routinely, the gate is wrong and should
be fixed or removed, not routed around.
"""

import argparse
import os
import pathlib
import re
import shutil
import stat
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import gh_guard                                            # noqa: E402 - path first

# The checkout this acts on. MOAT_REPO is set when the trunk's copy of the tools runs
# against another checkout (see trunk.py); config and schema come from TOOLS_ROOT.
TOOLS_ROOT = pathlib.Path(__file__).resolve().parents[1]
REPO = pathlib.Path(os.environ["MOAT_REPO"]).resolve() if os.environ.get("MOAT_REPO") else TOOLS_ROOT

MARKER = "# moat-hook v2"
# Installers from before v2 refuse to overwrite a hook without their own stamp, so a
# stale branch's orient cannot put its hook back over this one. The old stamp is still
# recognised here, as ours to upgrade.
OLD_MARKERS = ("# moat-hook v1",)

PYTHON_SLOT = "@MOAT_PYTHON@"


def python_path():
    """The running interpreter, spelled so both bash and cmd accept it."""
    return pathlib.Path(sys.executable).as_posix()


PRE_PUSH_TEMPLATE = f"""#!/usr/bin/env bash
{MARKER}
# Runs MOAT's gates before a push leaves this machine. Same check.py gates as CI;
# CI additionally checks the PR title (utils/pr_intent.py), which does not exist
# at push time, so that one check can still fail there.
#
# The gates are the TRUNK's, run against this checkout: a port branch's own copy of
# utils/ can be weeks old, and gating a push with it let stale rules through.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)" || exit 0

# The refs this push updates, for gates that judge only some destinations (the README
# table is a trunk artifact; a port branch may only change its own project). git hands
# them to pre-push on stdin, one per line.
MOAT_PUSH_REFS=""
MOAT_PUSH_UPDATES=""
only_deletes=1
while read -r _lref lsha rref rsha; do
  MOAT_PUSH_REFS="$MOAT_PUSH_REFS $rref"
  MOAT_PUSH_UPDATES="$MOAT_PUSH_UPDATES $lsha:$rref:$rsha"
  case "$lsha" in *[!0]*) only_deletes=0 ;; esac
done
export MOAT_PUSH_REFS MOAT_PUSH_UPDATES

# A push that only deletes branches sends no content, so no gate has anything to
# judge; running them all held a merged branch's deletion for minutes.
[ "$only_deletes" = 1 ] && exit 0

# The interpreter that installed this hook, because the gates need its packages. If it
# has gone (a rebuilt virtual environment), find one that can run them; a Python that
# cannot import them would fail every gate and blame the push.
MOAT_PY="{PYTHON_SLOT}"
if ! "$MOAT_PY" -c 'import jsonschema, tomllib' >/dev/null 2>&1; then
  MOAT_PY=""
  for c in python3 python; do
    if "$c" -c 'import jsonschema, tomllib' >/dev/null 2>&1; then MOAT_PY=$c; break; fi
  done
  if [ -z "$MOAT_PY" ]; then
    echo "moat: no Python that can run the pre-push gates (needs 3.11+ and jsonschema)." >&2
    echo "moat: tried {PYTHON_SLOT}, python3 and python; re-run utils/install_hooks.py" >&2
    echo "moat: under the interpreter MOAT uses on this host. The push is refused." >&2
    exit 1
  fi
fi

# --fast skips the two gates that walk every fork clone, forks and commits: they
# would add tens of seconds to every push. pr_ready enforces fork cleanliness where
# it matters; the commit rules are judged by a bare check.py, or by
# `moatlib.py audit-commits`, on a host that has the clones.
if git cat-file -e origin/main:utils/trunk.py 2>/dev/null; then
  "$MOAT_PY" -c "$(git show origin/main:utils/trunk.py)" run check.py --fast
else
  "$MOAT_PY" utils/check.py --fast
fi
if [ $? -ne 0 ]; then
  echo >&2
  echo "moat: pre-push gates failed -- fix what they report and push again." >&2
  echo "moat: (bypassing them with --no-verify is a person's call, never an agent's)" >&2
  exit 1
fi
"""


def pre_push_text(python=None):
    """The pre-push hook as this interpreter would install it."""
    return PRE_PUSH_TEMPLATE.replace(PYTHON_SLOT, python or python_path())


def _same_script(installed, template):
    """Is `installed` this template, whichever interpreter it was installed under?
    Two checkouts that run different Pythons must not reinstall over each other on
    every orient; an interpreter that no longer exists is another matter."""
    head, _, tail = template.partition(PYTHON_SLOT)
    if not installed.startswith(head):
        return False
    rest = installed[len(head):]
    baked = rest.split('"', 1)[0].splitlines()[0] if rest else ""
    if not baked or template.replace(PYTHON_SLOT, baked) != installed:
        return False
    return baked == python_path() or os.path.exists(baked)


# The launcher is the only platform-specific piece, and it is one line of dispatch into
# gh_guard.py. Everything it would otherwise have to decide -- is the working directory
# inside a checkout, is this a write, where is the real gh -- is Python in that file,
# so Windows and Linux cannot drift apart in what they permit.
#
# The guard itself needs nothing but the standard library, so the fallback when the
# installing interpreter has gone is any Python at all.
_SH = """#!/usr/bin/env sh
# {marker}
py="@MOAT_PYTHON@"
[ -x "$py" ] || {{ py=python3; command -v "$py" >/dev/null 2>&1 || py=python; }}
set -- "{guard}" {mode} "$@"
exec "$py" "$@"
"""
_CMD = """@echo off
rem {marker}
setlocal
set "MOAT_PY=@MOAT_PYTHON@"
if not exist "%MOAT_PY%" set "MOAT_PY=python"
set MOAT_GUARD_ARGS="{guard}" {mode}
"%MOAT_PY%" %MOAT_GUARD_ARGS% %*
"""
GH_LAUNCHERS = {
    "posix": ("gh", _SH.replace("{mode}", "--shim --")),
    "nt": ("gh.cmd", _CMD.replace("{mode}", "--shim --")),
}
# What a harness PreToolUse hook runs: the same guard, reading the tool event on
# standard input and judging the command text (gh_guard.py --hook). A launcher of
# its own so the settings file names one stable path instead of an interpreter.
HOOK_LAUNCHERS = {
    "posix": ("moat-gh-hook", _SH.replace("{mode}", "--hook").replace(' "$@"\nexec',
                                                                    '\nexec')),
    "nt": ("moat-gh-hook.cmd", _CMD.replace("{mode}", "--hook")),
}


def _render(body):
    return body.format(marker=gh_guard.SHIM_MARKER, guard=guard_copy_path().as_posix())


def gh_launcher_path():
    name, _ = GH_LAUNCHERS["nt" if os.name == "nt" else "posix"]
    return pathlib.Path.home() / ".local" / "bin" / name


def gh_launchers():
    """(path, text) for every launcher this host needs. Windows needs both: cmd and
    PowerShell find `gh.cmd` through PATHEXT, but Git Bash does not use PATHEXT, so
    with only `gh.cmd` installed `gh` in bash ran the real binary, unguarded."""
    return _launchers(GH_LAUNCHERS)


def hook_launchers():
    """(path, text) for the PreToolUse hook launcher(s), on the same rule."""
    return _launchers(HOOK_LAUNCHERS)


def _launchers(table):
    kinds = ["nt", "posix"] if os.name == "nt" else ["posix"]
    out = []
    for k in kinds:
        name, body = table[k]
        out.append((pathlib.Path.home() / ".local" / "bin" / name,
                    _render(body).replace(PYTHON_SLOT, python_path())))
    return out


def _launcher_current(path, table_text):
    """Is the launcher at `path` the one this code would write (see _same_script)?"""
    try:
        return _same_script(path.read_text(), table_text)
    except OSError:
        return False


# The guard the launcher runs is a HOST copy of the trunk's gh_guard.py, not a file in
# some checkout. The launcher is one per host while checkouts are many and come and go
# (worktrees, fleet job clones). Pointing it into one of them meant every install from
# another checkout re-pointed it, a deleted checkout left gh broken, a stale branch
# could install its stale guard, and the guard bound only inside the checkout it was
# installed from. The copy records the commit it came from, so a checkout that fetched
# later can tell an older copy from a newer one.
def guard_copy_path():
    return pathlib.Path.home() / ".local" / "share" / "moat" / "gh_guard.py"


def guard_source_path():
    return guard_copy_path().with_suffix(".source")


GUARD_IN_LAUNCHER = re.compile(r'"([^"\n]+?gh_guard\.py)"\s+--shim\s')


def gh_launcher_guard(text):
    """The gh_guard.py an installed launcher runs, or None if it runs no guard."""
    if gh_guard.SHIM_STAMP not in text:
        return None
    m = GUARD_IN_LAUNCHER.search(text)
    return pathlib.Path(m.group(1)) if m else None


def launcher_case_problems():
    """Self-check for the launcher parse, in the style of gh_guard.CASES.

    That regex decides which guard the gate inspects, so breaking it would quietly pass
    a launcher that runs something else. Building the cases from the same templates the
    installer writes keeps the two from drifting.
    """
    problems = []
    want = "/x/share/moat/gh_guard.py"
    for name, body in GH_LAUNCHERS.values():
        text = body.format(marker=gh_guard.SHIM_MARKER, guard=want).replace(
            PYTHON_SLOT, "C:/Program Files/Python312/python.exe")
        got = gh_launcher_guard(text)
        if got != pathlib.Path(want):
            problems.append(f"gh_launcher_guard misreads the {name} launcher: {got}")
        if "moat-gh-guard v1" in text:
            problems.append(f"the {name} launcher carries the v1 stamp, so installers "
                            f"from before v2 would overwrite it")

        # Installers from before the interpreter was carried read the launcher with
        # a looser pattern; they must still find the guard, or they would call this
        # launcher foreign and put their own back over it.
        old = re.search(r'"(.+?gh_guard\.py)"\s+--shim\s', text)
        if not old or old.group(1) != want:
            problems.append(f"an older installer would misread the {name} launcher")

    _, posix = GH_LAUNCHERS["posix"]
    if gh_launcher_guard(posix.format(marker="", guard=want)) is not None:
        problems.append("gh_launcher_guard accepts a launcher that is not ours")
    for name, body in HOOK_LAUNCHERS.values():
        if "--hook" not in body or "--shim" in body:
            problems.append(f"the {name} launcher does not run the guard's hook mode")

    nl = chr(10)
    inert = "# " + gh_guard.SHIM_MARKER + nl + "exec gh $@" + nl
    if gh_launcher_guard(inert) is not None:
        problems.append("gh_launcher_guard accepts a launcher that skips the guard")
    return problems


def _git_out(*args):
    r = subprocess.run(["git", *args], cwd=str(REPO), capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else ""


def tools_commit():
    """The commit this copy of the tools came from: the trunk commit when the trunk's
    export runs, else this checkout's HEAD."""
    if os.environ.get("MOAT_TRUNK_TOOLS", "local") != "local":
        return _git_out("rev-parse", "origin/main")
    return _git_out("rev-parse", "HEAD")


def _is_ancestor(old, new):
    return bool(old and new) and subprocess.run(
        ["git", "merge-base", "--is-ancestor", old, new], cwd=str(REPO),
        capture_output=True).returncode == 0


def gh_launcher_problem(shim=None):
    """Why the installed gh launcher is not a working, current guard, or None if it is.

    Current means the host copy is this code's gh_guard.py, or a copy from a commit this
    checkout cannot show is older (another checkout on the host fetched a newer trunk
    first). Only a copy provably older than these tools is stale, so two checkouts that
    fetched at different times do not reinstall over each other.
    """
    shim = shim or gh_launcher_path()
    hint = "(run utils/install_hooks.py)"
    if not shim.exists():
        return f"gh guard is not installed at {shim} {hint}"
    text = shim.read_text()
    if gh_guard.SHIM_STAMP not in text:
        return f"{shim} exists and is not the moat gh guard"
    guard = gh_launcher_guard(text)
    if guard is None:
        return f"gh guard at {shim} does not dispatch to a gh_guard.py --shim {hint}"
    if guard != guard_copy_path():
        return (f"gh guard at {shim} runs {guard}, a copy inside one checkout, instead "
                f"of the host copy {guard_copy_path()} {hint}")
    if not guard.exists():
        return f"gh guard at {shim} runs {guard}, which does not exist {hint}"
    for path, _ in gh_launchers():
        if not path.exists() or gh_launcher_guard(path.read_text()) != guard_copy_path():
            return f"gh launcher {path} is missing or does not run {guard_copy_path()} {hint}"
    kinds = ["nt", "posix"] if os.name == "nt" else ["posix"]
    for table in (GH_LAUNCHERS, HOOK_LAUNCHERS):
        for k in kinds:
            name, body = table[k]
            path = shim.parent / name
            if not _launcher_current(path, _render(body)):
                return (f"launcher {path} is missing, or predates the installing "
                        f"interpreter being carried in it {hint}")
    if guard.read_text() == (TOOLS_ROOT / "utils" / "gh_guard.py").read_text():
        return None
    src = guard_source_path()
    installed_from = src.read_text().strip() if src.exists() else ""
    if installed_from and not _is_ancestor(installed_from, tools_commit()):
        return None
    return f"gh guard copy {guard} is older than these tools {hint}"


def install_gh_guard(check=False, uninstall=False):
    shim = gh_launcher_path()
    if uninstall:
        removed = False
        for path, _ in gh_launchers() + hook_launchers():
            if path.exists() and gh_guard.SHIM_STAMP in path.read_text(errors="ignore"):
                path.unlink()
                removed = True
        print("install_hooks: gh guard removed" if removed
              else "install_hooks: no moat gh guard to remove")
        return 0

    for path, _ in gh_launchers() + hook_launchers():
        if path.exists() and gh_guard.SHIM_STAMP not in path.read_text(errors="ignore"):
            print(f"install_hooks: {path} exists and is not ours -- refusing to overwrite",
                  file=sys.stderr)
            return 1

    if gh_guard.real_gh() is None:
        # No gh at all is fine: a validation-only host never calls it. A launcher that
        # shadows a binary which is not there is not.
        print("install_hooks: gh not found; skipping the gh guard")
        return 0

    problem = gh_launcher_problem(shim)
    if check:
        if problem is None:
            print("install_hooks: gh guard installed and current")
            return path_problem(shim)
        print(f"install_hooks: {problem}", file=sys.stderr)
        return 1
    if problem is None:
        # A newer copy from another checkout stays; replacing it would be a downgrade.
        print("install_hooks: gh guard installed and current")
        return path_problem(shim)

    copy = guard_copy_path()
    copy.parent.mkdir(parents=True, exist_ok=True)
    tmp = copy.with_suffix(".tmp")
    tmp.write_text((TOOLS_ROOT / "utils" / "gh_guard.py").read_text())
    os.replace(str(tmp), str(copy))
    guard_source_path().write_text(tools_commit() + "\n")
    for path, text in gh_launchers() + hook_launchers():
        path.parent.mkdir(parents=True, exist_ok=True)
        # LF only: the sh launcher must not get CRLF line endings on Windows.
        with open(path, "w", newline="\n") as f:
            f.write(text)
        path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        print(f"install_hooks: gh guard installed at {path} (runs {copy})")

    return path_problem(shim)


def path_problem(shim=None):
    """0 when `gh` on PATH is the launcher, else say so and return 1.

    A launcher PATH never reaches is worse than none, because it reads as installed.
    This used to be a warning printed once, at install time, into output orient
    discarded; the guard then sat inert behind the real binary and only a push-time
    gate ever said so."""
    shim = shim or gh_launcher_path()
    resolved = shutil.which("gh")
    if resolved and (gh_guard.is_shim(resolved)
                     or pathlib.Path(resolved).resolve() == shim.resolve()):
        return 0
    print(f"install_hooks: `gh` resolves to {resolved or 'nothing'}, not the guard "
          f"launcher, so the guard is INERT in this shell. Put its directory first:\n"
          f"    export PATH=\"{shim.parent.as_posix()}:$PATH\"", file=sys.stderr)
    return 1


def hook_path(name="pre-push"):
    """Where git looks for the hook: the clone's shared hooks directory, which every
    worktree uses. `--git-dir` named a worktree's private directory, where git never
    looks, so an install from a worktree wrote a hook nothing ran."""
    r = subprocess.run(["git", "rev-parse", "--git-path", f"hooks/{name}"], cwd=str(REPO),
                       capture_output=True, text=True)
    if r.returncode:
        return None
    p = pathlib.Path(r.stdout.strip())
    return (p if p.is_absolute() else REPO / p).resolve()


def _ours(text):
    return MARKER in text or any(m in text for m in OLD_MARKERS)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--uninstall", action="store_true")
    a = ap.parse_args()

    p = hook_path()
    if p is None:
        print("install_hooks: not a git repo", file=sys.stderr)
        return 1

    if a.uninstall:
        if p.exists() and _ours(p.read_text()):
            p.unlink()
            print("install_hooks: pre-push removed")
        else:
            print("install_hooks: no moat pre-push hook to remove")
        return install_gh_guard(uninstall=True)

    current = p.exists() and _same_script(p.read_text(), PRE_PUSH_TEMPLATE)
    if a.check:
        rc = 0
        if current:
            print("install_hooks: pre-push installed and current")
        elif p.exists() and not _ours(p.read_text()):
            print("install_hooks: a NON-moat pre-push hook is installed; not touching it",
                  file=sys.stderr)
            rc = 1
        else:
            print("install_hooks: pre-push missing or stale (run utils/install_hooks.py)",
                  file=sys.stderr)
            rc = 1
        return install_gh_guard(check=True) or rc

    if p.exists() and not _ours(p.read_text()):
        print(f"install_hooks: {p} exists and is not ours -- refusing to overwrite",
              file=sys.stderr)
        return 1
    p.parent.mkdir(parents=True, exist_ok=True)
    # newline="\n": the default translates to CRLF on Windows, and the hook is a
    # shell script.
    p.write_text(pre_push_text(), newline="\n")
    p.chmod(p.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    print(f"install_hooks: pre-push installed at {p}")
    return install_gh_guard()


if __name__ == "__main__":
    import trunk  # noqa: E402 - run the trunk's copy of this tool, see trunk.py
    trunk.reexec(__file__)
    sys.exit(main())
