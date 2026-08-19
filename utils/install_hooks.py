#!/usr/bin/env python3
"""Install MOAT's git hooks and the gh guard into this clone.

Two gates, both of the same kind: a rule you cannot skip under time pressure beats one
you are told to follow. The pre-push hook runs check.py before a push exists; the gh
guard (utils/gh_guard.py, installed as a PATH shim) refuses adoption and upstream
GitHub writes while the working directory is inside this checkout.

CI catches a bad push at PR time. The hook catches it before it exists, which is the
difference between a rule you are told to follow and one you cannot skip under time
pressure. Both call utils/check.py; CI additionally checks the pull request's title
(utils/pr_intent.py), which does not exist at push time, so a locally-clean push can
still fail CI on its title -- and only on that.

    python3 utils/install_hooks.py              # install
    python3 utils/install_hooks.py --check      # is it installed and current?
    python3 utils/install_hooks.py --uninstall

Run once per clone. Hooks are not tracked by git, so a fresh clone has none --
orient.sh installs them, the same way it registers the status.json merge driver.

To bypass in a genuine emergency: `git push --no-verify`. If you find yourself
reaching for that routinely, the gate is wrong and should be fixed or removed, not
routed around.
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

REPO = pathlib.Path(__file__).resolve().parents[1]
MARKER = "# moat-hook v1"

PRE_PUSH = f"""#!/usr/bin/env bash
{MARKER}
# Runs MOAT's gates before a push leaves this machine. Same check.py gates as CI;
# CI additionally checks the PR title (utils/pr_intent.py), which does not exist
# at push time, so that one check can still fail there.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)" || exit 0

# --fast skips the fork-cleanliness gate: it walks every fork clone and would add
# tens of seconds to every push. pr_ready enforces it where it matters.
if ! python3 utils/check.py --fast; then
  echo >&2
  echo "moat: pre-push gates failed -- fix, or 'git push --no-verify' if genuinely urgent" >&2
  exit 1
fi
"""


# The launcher is the only platform-specific piece, and it is one line of dispatch into
# gh_guard.py. Everything it would otherwise have to decide -- is the working directory
# inside the checkout, is this a write, where is the real gh -- is Python in that file,
# so Windows and Linux cannot drift apart in what they permit.
GH_LAUNCHERS = {
    "posix": ("gh", """#!/usr/bin/env sh
# {marker}
exec python3 "{guard}" --shim -- "$@"
"""),
    "nt": ("gh.cmd", """@echo off
rem {marker}
python "{guard}" --shim -- %*
"""),
}


def gh_launcher_path():
    name, _ = GH_LAUNCHERS["nt" if os.name == "nt" else "posix"]
    return pathlib.Path.home() / ".local" / "bin" / name


def gh_launcher_text():
    _, body = GH_LAUNCHERS["nt" if os.name == "nt" else "posix"]
    return body.format(marker=gh_guard.SHIM_MARKER,
                       guard=str(REPO / "utils" / "gh_guard.py"))


# The one absolute path in the launcher is the guard it dispatches to, and every
# worktree of this repository generates a different one. Matching it out again is what
# lets the gate ask whether the guard works rather than whether it was last installed
# from this exact directory.
GUARD_IN_LAUNCHER = re.compile(r'"(.+?gh_guard\.py)"\s+--shim\s')


def gh_launcher_guard(text):
    """The gh_guard.py an installed launcher runs, or None if it runs no guard."""
    if gh_guard.SHIM_MARKER not in text:
        return None
    m = GUARD_IN_LAUNCHER.search(text)
    return pathlib.Path(m.group(1)) if m else None


def launcher_case_problems():
    """Self-check for the launcher parse, in the style of gh_guard.CASES.

    That regex is the whole of the path-insensitivity, so breaking it would quietly
    restore the ping-pong rather than fail loudly. Building the cases from the same
    templates the installer writes keeps the two from drifting.
    """
    problems = []
    want = "/x/utils/gh_guard.py"
    for name, body in GH_LAUNCHERS.values():
        got = gh_launcher_guard(body.format(marker=gh_guard.SHIM_MARKER, guard=want))
        if got != pathlib.Path(want):
            problems.append(f"gh_launcher_guard misreads the {name} launcher: {got}")

    _, posix = GH_LAUNCHERS["posix"]
    if gh_launcher_guard(posix.format(marker="", guard=want)) is not None:
        problems.append("gh_launcher_guard accepts a launcher that is not ours")

    nl = chr(10)
    inert = "# " + gh_guard.SHIM_MARKER + nl + "exec gh $@" + nl
    if gh_launcher_guard(inert) is not None:
        problems.append("gh_launcher_guard accepts a launcher that skips the guard")
    return problems


def gh_launcher_problem(shim=None):
    """Why the installed gh launcher is not a working guard, or None if it is.

    The property worth gating on is that `gh` runs this checkout's guard logic. It is
    not that the launcher names this particular directory: a worktree and the clone it
    was cut from share utils/gh_guard.py but embed different absolute paths, so
    comparing the launcher text verbatim reports "stale" from wherever it was not last
    installed. That made the gate a reinstall ping-pong -- installing from a worktree
    broke the main checkout and back -- and left a sweep that runs in a worktree by
    construction unable to push at all. So compare the guard the launcher reaches
    instead, which a worktree and its parent agree on.
    """
    shim = shim or gh_launcher_path()
    hint = "(python3 utils/install_hooks.py)"
    if not shim.exists():
        return f"gh guard is not installed at {shim} {hint}"
    text = shim.read_text()
    if gh_guard.SHIM_MARKER not in text:
        return f"{shim} exists and is not the moat gh guard"
    guard = gh_launcher_guard(text)
    if guard is None:
        return f"gh guard at {shim} does not dispatch to a gh_guard.py --shim {hint}"
    if not guard.exists():
        return f"gh guard at {shim} runs {guard}, which does not exist {hint}"
    if guard.read_text() != (REPO / "utils" / "gh_guard.py").read_text():
        return (f"gh guard at {shim} runs {guard}, which differs from this checkout's "
                f"utils/gh_guard.py {hint}")
    return None


def install_gh_guard(check=False, uninstall=False):
    shim = gh_launcher_path()
    if uninstall:
        if shim.exists() and gh_guard.SHIM_MARKER in shim.read_text():
            shim.unlink()
            print("install_hooks: gh guard removed")
        else:
            print("install_hooks: no moat gh guard to remove")
        return 0

    if shim.exists() and gh_guard.SHIM_MARKER not in shim.read_text():
        print(f"install_hooks: {shim} exists and is not ours -- refusing to overwrite",
              file=sys.stderr)
        return 1

    if gh_guard.real_gh() is None:
        # No gh at all is fine: a validation-only host never calls it. A launcher that
        # shadows a binary which is not there is not.
        print("install_hooks: gh not found; skipping the gh guard")
        return 0

    if check:
        problem = gh_launcher_problem(shim)
        if problem is None:
            print("install_hooks: gh guard installed and current")
            return 0
        print(f"install_hooks: {problem}", file=sys.stderr)
        return 1

    shim.parent.mkdir(parents=True, exist_ok=True)
    shim.write_text(gh_launcher_text())
    if os.name != "nt":
        shim.chmod(shim.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    print(f"install_hooks: gh guard installed at {shim}")

    # A launcher PATH never reaches is worse than none, because it reads as installed.
    resolved = shutil.which("gh")
    if not resolved or pathlib.Path(resolved).resolve() != shim.resolve():
        print(f"install_hooks: WARNING -- `gh` resolves to {resolved or 'nothing'}, not "
              f"the guard. Put {shim.parent} ahead of it in PATH, or the guard is inert.",
              file=sys.stderr)
    return 0


def hook_path(name="pre-push"):
    r = subprocess.run(["git", "rev-parse", "--git-dir"], cwd=str(REPO),
                       capture_output=True, text=True)
    if r.returncode:
        return None
    return (REPO / r.stdout.strip() / "hooks" / name).resolve()


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
        if p.exists() and MARKER in p.read_text():
            p.unlink()
            print("install_hooks: pre-push removed")
        else:
            print("install_hooks: no moat pre-push hook to remove")
        return install_gh_guard(uninstall=True)

    current = p.exists() and p.read_text() == PRE_PUSH
    if a.check:
        rc = 0
        if current:
            print("install_hooks: pre-push installed and current")
        elif p.exists() and MARKER not in p.read_text():
            print("install_hooks: a NON-moat pre-push hook is installed; not touching it",
                  file=sys.stderr)
            rc = 1
        else:
            print("install_hooks: pre-push missing or stale (run utils/install_hooks.py)",
                  file=sys.stderr)
            rc = 1
        return install_gh_guard(check=True) or rc

    if p.exists() and MARKER not in p.read_text():
        print(f"install_hooks: {p} exists and is not ours -- refusing to overwrite",
              file=sys.stderr)
        return 1
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(PRE_PUSH)
    p.chmod(p.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    print(f"install_hooks: pre-push installed at {p}")
    return install_gh_guard()


if __name__ == "__main__":
    sys.exit(main())
