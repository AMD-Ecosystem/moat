#!/usr/bin/env python3
"""Install MOAT's git hooks into this clone.

CI catches a bad push at PR time. The hook catches it before it exists, which is the
difference between a rule you are told to follow and one you cannot skip under time
pressure. Both call utils/check.py, so they enforce the same thing.

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
import pathlib
import stat
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
MARKER = "# moat-hook v1"

PRE_PUSH = f"""#!/usr/bin/env bash
{MARKER}
# Runs MOAT's gates before a push leaves this machine. Same gates as CI
# (utils/check.py), so passing here means passing there.
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
        return 0

    current = p.exists() and p.read_text() == PRE_PUSH
    if a.check:
        if current:
            print("install_hooks: pre-push installed and current")
            return 0
        if p.exists() and MARKER not in p.read_text():
            print("install_hooks: a NON-moat pre-push hook is installed; not touching it",
                  file=sys.stderr)
            return 1
        print("install_hooks: pre-push missing or stale (run utils/install_hooks.py)",
              file=sys.stderr)
        return 1

    if p.exists() and MARKER not in p.read_text():
        print(f"install_hooks: {p} exists and is not ours -- refusing to overwrite",
              file=sys.stderr)
        return 1
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(PRE_PUSH)
    p.chmod(p.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    print(f"install_hooks: pre-push installed at {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
