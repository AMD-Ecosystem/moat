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

# The checkout this acts on. MOAT_REPO is set when the trunk's copy of the tools runs
# against another checkout (see trunk.py); config and schema come from TOOLS_ROOT.
TOOLS_ROOT = pathlib.Path(__file__).resolve().parents[1]
REPO = pathlib.Path(os.environ["MOAT_REPO"]).resolve() if os.environ.get("MOAT_REPO") else TOOLS_ROOT

MARKER = "# moat-hook v2"
# Installers from before v2 refuse to overwrite a hook without their own stamp, so a
# stale branch's orient cannot put its hook back over this one. The old stamp is still
# recognised here, as ours to upgrade.
OLD_MARKERS = ("# moat-hook v1",)

PRE_PUSH = f"""#!/usr/bin/env bash
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
while read -r _lref lsha rref rsha; do
  MOAT_PUSH_REFS="$MOAT_PUSH_REFS $rref"
  MOAT_PUSH_UPDATES="$MOAT_PUSH_UPDATES $lsha:$rref:$rsha"
done
export MOAT_PUSH_REFS MOAT_PUSH_UPDATES

# --fast skips the fork-cleanliness gate: it walks every fork clone and would add
# tens of seconds to every push. pr_ready enforces it where it matters.
if git cat-file -e origin/main:utils/trunk.py 2>/dev/null; then
  python3 -c "$(git show origin/main:utils/trunk.py)" run check.py --fast
else
  python3 utils/check.py --fast
fi
if [ $? -ne 0 ]; then
  echo >&2
  echo "moat: pre-push gates failed -- fix, or 'git push --no-verify' if genuinely urgent" >&2
  exit 1
fi
"""


# The launcher is the only platform-specific piece, and it is one line of dispatch into
# gh_guard.py. Everything it would otherwise have to decide -- is the working directory
# inside a checkout, is this a write, where is the real gh -- is Python in that file,
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


def gh_launcher_text():
    _, body = GH_LAUNCHERS["nt" if os.name == "nt" else "posix"]
    return body.format(marker=gh_guard.SHIM_MARKER, guard=str(guard_copy_path()))


GUARD_IN_LAUNCHER = re.compile(r'"(.+?gh_guard\.py)"\s+--shim\s')


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
        text = body.format(marker=gh_guard.SHIM_MARKER, guard=want)
        got = gh_launcher_guard(text)
        if got != pathlib.Path(want):
            problems.append(f"gh_launcher_guard misreads the {name} launcher: {got}")
        if "moat-gh-guard v1" in text:
            problems.append(f"the {name} launcher carries the v1 stamp, so installers "
                            f"from before v2 would overwrite it")

    _, posix = GH_LAUNCHERS["posix"]
    if gh_launcher_guard(posix.format(marker="", guard=want)) is not None:
        problems.append("gh_launcher_guard accepts a launcher that is not ours")

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
    hint = "(python3 utils/install_hooks.py)"
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
        if shim.exists() and gh_guard.SHIM_STAMP in shim.read_text():
            shim.unlink()
            print("install_hooks: gh guard removed")
        else:
            print("install_hooks: no moat gh guard to remove")
        return 0

    if shim.exists() and gh_guard.SHIM_STAMP not in shim.read_text():
        print(f"install_hooks: {shim} exists and is not ours -- refusing to overwrite",
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
            return 0
        print(f"install_hooks: {problem}", file=sys.stderr)
        return 1
    if problem is None:
        # A newer copy from another checkout stays; replacing it would be a downgrade.
        print("install_hooks: gh guard installed and current")
        return 0

    copy = guard_copy_path()
    copy.parent.mkdir(parents=True, exist_ok=True)
    tmp = copy.with_suffix(".tmp")
    tmp.write_text((TOOLS_ROOT / "utils" / "gh_guard.py").read_text())
    os.replace(str(tmp), str(copy))
    guard_source_path().write_text(tools_commit() + "\n")
    shim.parent.mkdir(parents=True, exist_ok=True)
    shim.write_text(gh_launcher_text())
    if os.name != "nt":
        shim.chmod(shim.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    print(f"install_hooks: gh guard installed at {shim} (runs {copy})")

    # A launcher PATH never reaches is worse than none, because it reads as installed.
    resolved = shutil.which("gh")
    if not resolved or pathlib.Path(resolved).resolve() != shim.resolve():
        print(f"install_hooks: WARNING -- `gh` resolves to {resolved or 'nothing'}, not "
              f"the guard. Put {shim.parent} ahead of it in PATH, or the guard is inert.",
              file=sys.stderr)
    return 0


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

    current = p.exists() and p.read_text() == PRE_PUSH
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
    p.write_text(PRE_PUSH)
    p.chmod(p.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    print(f"install_hooks: pre-push installed at {p}")
    return install_gh_guard()


if __name__ == "__main__":
    import trunk  # noqa: E402 - run the trunk's copy of this tool, see trunk.py
    trunk.reexec(__file__)
    sys.exit(main())
