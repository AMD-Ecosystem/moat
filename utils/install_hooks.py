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

    want = gh_launcher_text()
    if check:
        if shim.exists() and shim.read_text() == want:
            print("install_hooks: gh guard installed and current")
            return 0
        print("install_hooks: gh guard missing or stale (run utils/install_hooks.py)",
              file=sys.stderr)
        return 1

    shim.parent.mkdir(parents=True, exist_ok=True)
    shim.write_text(want)
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
